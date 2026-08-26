# Materials and Sources

## Data-free Earth–ionosphere material

`EarthIonosphereMaterial` is the default model. It provides a homogeneous
lithosphere and exponential atmosphere/ionosphere without external datasets:

- lithosphere conductivity: $10^{-3}\ \mathrm{S/m}$;
- lithosphere relative permittivity: 10;
- atmosphere relative permittivity: 1;
- exponential ionospheric conductivity controlled by reference height, scale
  height, and prefactor.

All values are configurable constructor arguments. Custom material objects may
be supplied if they implement

```python
sample(directions, altitudes_m, earth_radius_m) -> (sigma, epsilon_r)
```

and return finite arrays of shape `(directions, altitudes)`.

## Layered geographic material

`LayeredEarthIonosphereMaterial` supports land/ocean classification or sampled
surface elevation, seawater, upper crust, asthenosphere, deep rock, and an
exponential ionosphere. Supply exactly one of `land_classifier` or
`surface_elevation_sampler`.

Use `tangential_interface_mode="fractional"` when a tangential cell straddling
a radial material interface should receive thickness-weighted properties.

## Spatial profiles and gridded observations

`SpatialEarthIonosphereMaterial` accepts direction samplers for ionosphere
reference height and scale height, plus an optional direction-by-altitude crust
conductivity sampler. This supports measured day/night or geographic profile
maps without coupling the solver to a particular data product.

`GriddedMaterial.from_npz(path)` imports a complete three-dimensional material
volume. The archive must contain:

| Array | Shape |
|---|---|
| `latitudes_deg` | `(n_latitude,)`, increasing |
| `longitudes_deg` | `(n_longitude,)`, increasing, periodic |
| `altitudes_m` | `(n_altitude,)`, increasing |
| `conductivity_s_m` | `(n_latitude, n_longitude, n_altitude)` |
| `relative_permittivity` | `(n_latitude, n_longitude, n_altitude)` |

Sampling is trilinear with periodic longitude. Requested altitudes must lie
inside the data volume; extrapolation is deliberately rejected. Convert source
datasets to SI units and this canonical schema in a provenance-preserving
preprocessing step rather than embedding study-specific file formats in the
runtime solver.

## Provenance and mesh-native material artifacts

Use `MeshMaterialArtifact` after the source grids have been converted and a
production mesh has been selected. It freezes the four material arrays at the
actual staggered solver supports: radial conductivity/permittivity on vertices
and radial nodes, and tangential conductivity/permittivity on edges and radial
cell centers. A later run therefore performs no global-grid interpolation and
does not need to retain the source volume in accelerator memory.

Every artifact requires one or more `DatasetProvenance` records. Each record
contains a stable dataset identifier, title and version, download URL,
citation, license, timezone-qualified retrieval timestamp, exact source-file
SHA-256, coordinate reference system, and per-variable source/canonical units
with the applied conversion. The artifact separately records the interpolation
policy and an ordered list of processing steps.

```python
from ionosphere_fdtd import (
    DatasetProvenance,
    MeshMaterialArtifact,
    VariableProvenance,
)

source = DatasetProvenance.from_file(
    "downloads/source.nc",
    dataset_id="provider.product.release",
    title="Provider product title",
    version="release",
    source_url="https://provider.example/product",
    citation="Provider citation for this release.",
    license="documented source-data license",
    retrieved_at="2026-08-20T10:00:00Z",
    coordinate_reference_system="WGS 84 latitude/longitude; altitude above MSL",
    variables=(
        VariableProvenance(
            "conductivity", "mS/m", "S/m", "multiply by 1e-3"
        ),
    ),
)
artifact = MeshMaterialArtifact.from_simulation(
    preprocessing_simulation,
    provenance=(source,),
    interpolation="periodic lon, linear lat/alt, no extrapolation",
    processing_steps=("convert to SI", "sample declared solver supports"),
)
artifact.save("artifacts/materials/production-mesh.npz")
```

Loading with `MeshMaterialArtifact.load()` verifies every embedded array
checksum. Solver construction also verifies the mesh vertex and face hashes,
Earth radius, complete radial grid, radial/tangential support rules, anomaly
policy, and entity counts. Regridding, changing support rules, or optimizing
mesh coordinates requires generating a new artifact; silently reusing an
almost-matching file is prohibited. `content_sha256` is the stable identity to
record in run metadata.

For a mesh-native ground map, pass its edge-associated shallow conductivity to
`ConductiveHalfSpaceSurface`. The surface ADE then replaces the explicit
underground volume; it does not consume the artifact's vertex-by-altitude
arrays. Preserve the conductivity map's provenance and the resulting surface
model parameters together with the run configuration.

## Magnetized ionosphere tensor and ADE

`MeshPlasmaModel` represents the ionosphere as one or more charged fluids at
mesh-face/radial-cell centers. Each `ColdPlasmaSpecies` stores number density,
collision frequency, signed charge, and mass; the model stores the geocentric
Cartesian magnetic field. For angular frequency $\omega$, each species
contributes

$$
\begin{aligned}
\sigma_\parallel &= \frac{nq^2/m}{\nu+i\omega},\\
\sigma_P &= \frac{(nq^2/m)(\nu+i\omega)}{(\nu+i\omega)^2+\Omega^2},\\
\sigma_H &= \frac{(nq^2/m)\Omega}{(\nu+i\omega)^2+\Omega^2},
\qquad \Omega=\frac{q|B|}{m}.
\end{aligned}
$$

The Hall coefficient multiplies $E\times\hat b$, so its sign follows the
species charge. In time domain, an exact constant-$E$ ADE advances

$$
\frac{dJ_s}{dt}=\frac{n_sq_s^2}{m_s}E-\nu_sJ_s
                  +\Omega_s J_s\times\hat b.
$$

The solver reconstructs a three-component electric vector from the three
oriented edge samples and the face-centered radial interpolation, advances all
fluid currents, then projects the result back to the staggered `Er` and `Et`
supports. This is the required path for Hall coupling; a scalar `sigma` cannot
represent it.

Prepare source data outside the solver and bind it to the exact mesh with
`MeshPlasmaModel.from_mesh()`. A production provenance set should normally
identify electron/ion densities from IRI, collision inputs derived from a
declared neutral-atmosphere product, and the IGRF magnetic-field release. The
pickle-free NPZ produced by `save()` records those provenance entries and
checksums every density, collision, magnetic-field, and radial-grid array.

```python
from ionosphere_fdtd import ColdPlasmaSpecies, MeshPlasmaModel

electrons = ColdPlasmaSpecies(
    name="electron",
    charge_c=-1.602176634e-19,
    mass_kg=9.1093837139e-31,
    number_density_m3=electron_density_on_faces,
    collision_frequency_hz=electron_neutral_collision_on_faces,
)
plasma = MeshPlasmaModel.from_mesh(
    mesh,
    radial_midpoint_altitudes_m,
    magnetic_field_t,
    (electrons,),
    provenance=(iri_source, neutral_source, igrf_source),
    interpolation="declared geographic/time interpolation policy",
)
simulation = GeodesicFDTD(config, mesh=mesh, plasma=plasma)
```

The explicit electric/current coupling adds a plasma-frequency stability limit
$\Delta t\,\omega_p\leq2$ in addition to the electromagnetic CFL limit. The
current solver enforces the tighter value. Checkpoint version 4 embeds the
plasma artifact and every species-current ADE state. Distributed vector-current
halos are not yet implemented; that path is rejected rather than silently
dropping Hall coupling.

## Spherical anomalies

```python
from ionosphere_fdtd import EarthIonosphereMaterial, SphericalAnomaly

anomaly = SphericalAnomaly(
    latitude_deg=69.0,
    longitude_deg=-156.0,
    radius_m=40_000.0,
    altitude_min_m=-2_000.0,
    altitude_max_m=-500.0,
    conductivity_factor=0.1,
)
material = EarthIonosphereMaterial(anomalies=(anomaly,))
```

An anomaly only affects electric samples intersecting its horizontal and radial
support. With `horizontal_anomaly_mode="conservative-nearest"`, the solver
preserves the requested spherical area by filling the nearest vertex dual cells
and edge diamonds, with at most one fractional support in each association. It
does not calculate exact intersections with the nominal circular boundary.

With `conservative-nearest`, vertical treatment follows the staggered field
layout: `Et` material receives the exact overlap fraction between an anomaly
interval and each radial cell, while `Er` material is selected by whether a
radial node lies inside that interval. In the default `point` mode, both
components instead use their staggered sample altitudes. The current anomaly
mixer applies a linear conductivity fraction and does not perform
component-specific thin-layer homogenization, such as a harmonic
complex-admittance average for the normal component. A layer only one cell
thick is therefore a discrete compatibility model, not independent evidence of
radial convergence. Use multiple radial resolutions before making a
quantitative claim about a shallow layer. The CLI warns when no selected
electric sample intersects an anomaly.

## Vertical Gaussian current

`GaussianCurrent` launches a localized radial electric source. Its main
parameters are:

| Parameter | Meaning |
|---|---|
| `latitude_deg`, `longitude_deg`, `altitude_m` | Exact geographic location |
| `peak_current_a` | Peak current |
| `vertical_element_length_m` | Current-element length |
| `center_time_s` | Gaussian center |
| `one_over_e_half_width_s` | Gaussian $1/e$ half-width |
| `carrier_frequency_hz` | Optional cosine carrier |

The source is distributed barycentrically across the containing surface
triangle and linearly across adjacent staggered radial planes. This preserves
the configured current moment under refinement.

When a carrier is present and no width is supplied, the half-width defaults to
$0.5/f$. The solver rejects a carrier at or above the time-step Nyquist limit.

### Differentiable source currents

With the PyTorch backend, `step()`, `record_er_observations()`, and
`record_h_observations()` accept a one-dimensional `currents` tensor with one
value per requested step. A tensor already on the simulation device and dtype
is used directly, so its autograd history is preserved. Observation methods
return tensors on that same device and do not detach them.

`GaussianCurrent.current_tensor_a()` evaluates the built-in waveform on the
tensor device. Its `peak_current_a` override is the differentiable waveform
parameter:

```python
import torch

peak = torch.tensor(
    1.0e6,
    device=simulation.er.device,
    dtype=simulation.er.dtype,
    requires_grad=True,
)
times = (
    torch.arange(steps, device=peak.device, dtype=peak.dtype) + 0.5
) * simulation.time_step_s
currents = source.current_tensor_a(
    times,
    simulation.time_step_s,
    peak_current_a=peak,
)
traces = simulation.record_er_observations(
    vertex_indices,
    radial_layers,
    weights,
    steps,
    currents=currents,
)
traces.square().mean().backward()
```

The stored scalar source attributes, geographic location, mesh search,
interpolation topology, and distribution weights are static in this API.
Constructing `currents` through other tensor operations also makes any
upstream tensor parameters differentiable. Use `simulation.to_numpy(traces)`
only after gradient-based work is complete. Device barriers are opt-in through
`synchronize_every`; terminal export performs the final synchronization and
host copy.

## Tangential Gaussian current

`TangentialGaussianCurrent` projects one or more geographic azimuths onto the
three oriented edges of the containing triangle:

```python
from ionosphere_fdtd import TangentialGaussianCurrent

source = TangentialGaussianCurrent(
    latitude_deg=46.5,
    longitude_deg=-90.9,
    carrier_frequency_hz=20.0,
    azimuths_deg=(0.0, 90.0),
    line_lengths_m=(22_500.0, 22_500.0),
)
```

Azimuth is measured clockwise from geographic north. `edge_assignment` accepts
`projected` for a vector reconstruction or `nearest` for compatibility studies.
