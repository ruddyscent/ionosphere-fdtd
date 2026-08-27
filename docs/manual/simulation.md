# Simulation Configuration

## Geometry

`SimulationConfig` controls the surface mesh, radial grid, time step, material
sampling, boundary condition, and geometry compatibility mode.

| Option | Default | Meaning |
|---|---:|---|
| `subdivision` | 2 | Recursive icosahedral surface refinement |
| `radial_cells` | 24 | Number of radial intervals |
| `minimum_altitude_m` | −100,000 | Lower radial boundary |
| `maximum_altitude_m` | 100,000 | Upper radial boundary |
| `earth_radius_m` | 6,371,000 | Reference Earth radius |
| `courant_factor` | 0.35 | Fraction of the conservative CFL limit |
| `time_step_s` | `None` | Explicit step, validated against the CFL limit |

The default radial nodes are uniform. Supply a strictly increasing tuple through
`radial_altitudes_m` for a custom grid; its length must be
`radial_cells + 1` and it must include both configured altitude bounds.

Smoothly graded nodes use `radial_grid_policy="smooth"`. Abrupt subgridding is
first-order at the transition and requires an explicit
`radial_grid_policy="allow-abrupt"` selection.

For deterministic static h-refinement, use `build_refined_radial_grid()` and
`radial_grid_policy="balanced-2to1"`. The builder bisects intersecting cells to
meet each `RadialRefinementRegion.maximum_step_m` and then enforces at most one
level difference between neighbors:

```python
from ionosphere_fdtd import (
    RadialRefinementRegion,
    SimulationConfig,
    build_refined_radial_grid,
)

altitudes = build_refined_radial_grid(
    0.0,
    100_000.0,
    10_000.0,
    (RadialRefinementRegion(60_000.0, 90_000.0, 1_250.0),),
)
config = SimulationConfig(
    radial_cells=len(altitudes) - 1,
    minimum_altitude_m=altitudes[0],
    maximum_altitude_m=altitudes[-1],
    radial_altitudes_m=altitudes,
    radial_grid_policy="balanced-2to1",
)
```

This grid is static and shared by every surface column. It reduces radial
storage but is not a moving or geographically local 3-D AMR hierarchy. See the
[refinement decision](../benchmarks/refinement-strategy.md) for the measured
memory result and the reasons local subcycling and dynamic AMR are deferred.

## Static surface refinement

`build_adaptive_geodesic_mesh()` creates one closed, conforming surface mesh
with fine cores and graded transition rings. It introduces no hanging nodes or
overset boundaries, keeps neighboring face levels 2:1 balanced, and uses the
same global leapfrog time step as a uniform mesh.

```python
from ionosphere_fdtd import (
    GeodesicFDTD,
    SimulationConfig,
    SphericalRefinementRegion,
    build_adaptive_geodesic_mesh,
)

mesh = build_adaptive_geodesic_mesh(
    4,
    (
        SphericalRefinementRegion(
            latitude_deg=69.0,
            longitude_deg=-156.0,
            radius_deg=2.0,
            target_subdivision=6,
            transition_width_deg=2.0,
            label="receiver",
        ),
    ),
)
simulation = GeodesicFDTD(
    SimulationConfig(subdivision=4, radial_cells=24),
    mesh=mesh,
)
```

Refinement regions are fixed before stepping. They reduce surface storage
relative to a globally uniform fine grid but do not provide local time
subcycling or a geographically local radial hierarchy. Validate production
regions against source, receiver, coastline, and material-feature support, and
record the generated mesh metadata with the run.

## Maxwell layout

For an oriented primal surface edge, the solver advances

```text
Ht += dt / mu0 * (d_surface Er - d_radial Et)
Hr -= dt / mu0 * curl_surface Et
Er  = Ca * Er + Cb * (curl_surface Ht - Jr)
Et  = Ca * Et + Cb * (d_dual Hr - d_radial Ht)
```

`geometry_mode="full-spherical"` uses

$$
\frac{1}{r}\frac{\partial(rE_t)}{\partial r}
\quad\text{and}\quad
\frac{1}{r}\frac{\partial(rH_t)}{\partial r}.
$$

`geometry_mode="thin-shell"` retains radius-independent radial differences for
paper compatibility. New physical simulations should normally use
`full-spherical`.

## Conductive integration

The default `loss_integration="exponential"` uses

$$
q=\frac{\sigma\Delta t}{\epsilon},\qquad
C_a=e^{-q},\qquad
C_b=\frac{\Delta t}{\epsilon}\frac{1-e^{-q}}{q},
$$

with the continuous $q=0$ limit. Use `trapezoidal` only when compatibility with
a legacy discretization is required.

## Boundaries and stability

The default `radial_boundary_condition="pec"` uses odd tangential-electric
ghost cells to place the electric trace at zero on both radial boundaries.

For an atmosphere-only physical model, set the minimum altitude to zero,
select `radial_boundary_condition="surface-impedance"`, and pass a
`ConductiveHalfSpaceSurface` to `GeodesicFDTD`. The upper boundary remains PEC;
the lower boundary applies

$$
E_t(s)=-Z_s(s)H_t(s),\qquad
Z_s(s)\simeq\sqrt{\frac{\mu s}{\sigma}}.
$$

The default 16-term diffusive approximation is fitted over 5–45 Hz. Positive
poles and residues make it causal and passive, and its trapezoidal ADE is
coupled implicitly to the lower magnetic update. Conductivity may be scalar or
contain one value per surface edge, so a preprocessed land/ocean/crust map can
drive a global boundary without an explicit underground volume.

```python
from ionosphere_fdtd import (
    ConductiveHalfSpaceSurface,
    GeodesicFDTD,
    SimulationConfig,
)

config = SimulationConfig(
    minimum_altitude_m=0.0,
    maximum_altitude_m=100_000.0,
    radial_boundary_condition="surface-impedance",
)
surface = ConductiveHalfSpaceSurface(
    conductivity_s_m=1.0 / 50.0,
)
simulation = GeodesicFDTD(config, surface_impedance=surface)
```

The array length must match the chosen mesh's edge count. The approximation is
the conductive half-space limit; displacement current and explicit layered
resonances are not represented. Generate and validate a different passive
rational model before using it where those effects are material.

An ADE stores `terms * n_edges` scalar states. On a closed triangular mesh this
cost is recovered after eliminating roughly six explicit underground radial
cells; a skin-depth-resolving seawater or crust volume would require many more.
Checkpoint version 3 preserves the ADE memory so a resumed run has no boundary
transient. Single-backend compilation and the two-rank NCCL CUDA Graph path
both update the state in place.

The solver computes a geometry- and material-aware CFL limit and rejects an
explicit `time_step_s` above `courant_factor * cfl_time_step_limit_s`.

## Material support controls

| Option | Choices | Purpose |
|---|---|---|
| `radial_material_support` | `point`, `dual-cell` | Sample the radial E component at a vertex or average horizontally over its surface dual cell |
| `tangential_material_support` | `point`, `edge-diamond` | Sample the tangential E component at an edge midpoint or average horizontally over its edge diamond |
| `horizontal_anomaly_mode` | `point`, `conservative-nearest` | Point-select supports or preserve the declared anomaly area using nearest supports |

The averaging modes are valuable at discontinuous coastlines and anomaly
boundaries but cost more during setup. Here, “radial material” names the radial
electric-field component; `dual-cell` does not average along altitude.
`conservative-nearest` preserves total spherical area but does not compute exact
geometric intersections with a circular anomaly boundary. See
[Materials and Sources](materials-and-sources.md#spherical-anomalies) for the
vertical support rule and thin-layer limitation.

## Mesh controls

`mesh_orientation` accepts `polar` or `native`. `mesh_relaxations` and
`mesh_optimization_steps` alter coordinates while preserving topology. Do not
combine these controls with an explicitly supplied `GeodesicMesh`. Use
`build_adaptive_geodesic_mesh()` when selected geographic regions need higher
surface resolution; pass the resulting mesh explicitly to `GeodesicFDTD`.

## Diagnostics and memory

```python
values = simulation.diagnostics()
print(values["cfl_time_step_limit_s"])
print(values["field_memory_bytes"])
print(simulation.persistent_runtime_bytes)
```

`memory_bytes` counts the four evolving fields. `persistent_runtime_bytes`
also includes material coefficients, geometry, topology, and any surface ADE
state resident on the selected backend.
