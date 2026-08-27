import numpy as np
import pytest

from ionosphere_fdtd.constants import MU_0
from ionosphere_fdtd.materials import (
    EarthIonosphereMaterial,
    LayeredEarthIonosphereMaterial,
    SphericalAnomaly,
)
from ionosphere_fdtd.mesh import (
    build_geodesic_mesh,
    build_geodesic_mesh_from_topology,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import (
    GaussianCurrent,
    TangentialGaussianCurrent,
    geographic_direction,
    geographic_distribution,
    geographic_face_index,
)


def small_config(**changes: object) -> SimulationConfig:
    values = dict(subdivision=1, radial_cells=6, courant_factor=0.25)
    values.update(changes)
    return SimulationConfig(**values)


def test_simulation_config_rejects_negative_mesh_steps() -> None:
    with pytest.raises(ValueError, match="mesh_relaxations"):
        small_config(mesh_relaxations=-1)
    with pytest.raises(ValueError, match="mesh_optimization_steps"):
        small_config(mesh_optimization_steps=-1)


def test_simulation_config_rejects_nonfinite_and_inconsistent_geometry() -> None:
    with pytest.raises(ValueError, match="finite"):
        small_config(time_step_s=np.nan)
    with pytest.raises(ValueError, match="finite"):
        small_config(earth_radius_m=np.nan)
    with pytest.raises(ValueError, match="radial_cells"):
        small_config(radial_altitudes_m=(-100_000.0, 0.0, 100_000.0))


def test_zero_fields_are_stationary() -> None:
    simulation = GeodesicFDTD(config=small_config())
    simulation.step(3)
    assert not np.any(simulation.to_numpy(simulation.er))
    assert not np.any(simulation.to_numpy(simulation.et))
    assert not np.any(simulation.to_numpy(simulation.hr))
    assert not np.any(simulation.to_numpy(simulation.ht))


def test_solver_accepts_custom_closed_topology() -> None:
    uniform = build_geodesic_mesh(1)
    custom = build_geodesic_mesh_from_topology(
        uniform.vertices,
        np.roll(uniform.faces, 1, axis=0),
    )
    simulation = GeodesicFDTD(config=small_config(), mesh=custom)

    simulation.step(3)

    assert simulation.mesh.topology_kind == "custom"
    assert not np.any(simulation.to_numpy(simulation.er))
    assert not np.any(simulation.to_numpy(simulation.et))
    assert not np.any(simulation.to_numpy(simulation.hr))
    assert not np.any(simulation.to_numpy(simulation.ht))


def test_memory_diagnostics_distinguish_fields_from_persistent_arrays() -> None:
    simulation = GeodesicFDTD(config=small_config())
    diagnostics = simulation.diagnostics()

    assert diagnostics["field_memory_bytes"] == simulation.memory_bytes
    assert diagnostics["persistent_runtime_bytes"] == (
        simulation.persistent_runtime_bytes
    )
    assert simulation.persistent_runtime_bytes > simulation.memory_bytes


def test_gaussian_source_launches_finite_fields() -> None:
    simulation = GeodesicFDTD(
        config=small_config(), source=GaussianCurrent(peak_current_a=1.0e6)
    )
    simulation.step(80)
    assert np.isfinite(simulation.to_numpy(simulation.er)).all()
    assert np.isfinite(simulation.to_numpy(simulation.ht)).all()
    assert np.max(np.abs(simulation.to_numpy(simulation.er))) > 0.0
    assert np.max(np.abs(simulation.to_numpy(simulation.ht))) > 0.0
    assert simulation.time_s == pytest.approx(80 * simulation.time_step_s)


def test_default_source_is_located_in_gwangju() -> None:
    source = GaussianCurrent()
    assert source.latitude_deg == pytest.approx(35.1595)
    assert source.longitude_deg == pytest.approx(126.8526)


def test_source_distribution_preserves_exact_direction() -> None:
    source = GaussianCurrent()
    simulation = GeodesicFDTD(config=small_config(), source=source)
    vertices, _, weights = source.distribution(simulation)
    represented = weights @ simulation.mesh.vertices[vertices]
    represented /= np.linalg.norm(represented)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(weights >= 0.0)
    assert represented @ source.direction() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("latitude_deg", "longitude_deg"),
    (
        (0.0, -137.0),
        (0.0, -92.0),
        (0.0, -47.0),
        (0.0, -2.0),
        (0.0, 43.0),
        (46.5, -90.9),
        (69.0, -156.0),
    ),
)
def test_geographic_distribution_rejects_antipodal_face(
    latitude_deg: float,
    longitude_deg: float,
) -> None:
    simulation = GeodesicFDTD(config=small_config(subdivision=3))
    vertices, _, weights = geographic_distribution(
        simulation, latitude_deg, longitude_deg, 0.0
    )
    represented = weights @ simulation.mesh.vertices[vertices]
    represented /= np.linalg.norm(represented)

    np.testing.assert_allclose(
        represented,
        geographic_direction(latitude_deg, longitude_deg),
        atol=1.0e-12,
    )


def test_antipodal_observations_use_distinct_faces() -> None:
    simulation = GeodesicFDTD(config=small_config(subdivision=3))

    assert geographic_face_index(simulation, 0.0, 43.0) != geographic_face_index(
        simulation, 0.0, -137.0
    )


def test_source_distribution_preserves_exact_staggered_altitude() -> None:
    source = GaussianCurrent(altitude_m=2_500.0)
    simulation = GeodesicFDTD(config=small_config(), source=source)
    vertices, layers, weights = source.staggered_distribution(simulation)
    represented_altitude = weights @ simulation.altitudes_m[layers]
    horizontal_weights = np.asarray(
        [weights[vertices == vertex].sum() for vertex in np.unique(vertices)]
    )

    assert represented_altitude == pytest.approx(source.altitude_m)
    assert weights.sum() == pytest.approx(1.0)
    assert horizontal_weights.sum() == pytest.approx(1.0)
    assert len(np.unique(layers)) == 2


def test_staggered_source_update_preserves_current_moment() -> None:
    source = GaussianCurrent(altitude_m=2_500.0)
    simulation = GeodesicFDTD(config=small_config(), source=source)
    vertices, layers, expected_weights = source.staggered_distribution(simulation)

    simulation._update_electric_fields(1.0)
    dual_areas = (
        simulation.mesh.dual_cell_solid_angles[vertices]
        * simulation.radii_m[layers] ** 2
    )
    represented_current_density = (
        -simulation.er[vertices, layers]
        / simulation._cb_er[vertices, layers]
    )
    represented_moments = (
        represented_current_density
        * dual_areas
        * simulation.radial_node_control_lengths_m[layers]
    )

    np.testing.assert_allclose(
        represented_moments,
        source.vertical_element_length_m * expected_weights,
    )
    assert represented_moments.sum() == pytest.approx(
        source.vertical_element_length_m
    )


@pytest.mark.parametrize("radial_cells", (24, 40, 80))
def test_vertical_source_current_moment_is_radial_grid_independent(
    radial_cells: int,
) -> None:
    source = GaussianCurrent(
        altitude_m=2_500.0,
        peak_current_a=3.0,
        vertical_element_length_m=7_500.0,
    )
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=radial_cells,
            courant_factor=0.2,
        ),
        source=source,
    )
    vertices, layers, _ = source.staggered_distribution(simulation)

    simulation._update_electric_fields(source.peak_current_a)
    dual_areas = (
        simulation.mesh.dual_cell_solid_angles[vertices]
        * simulation.radii_m[layers] ** 2
    )
    current_density = (
        -simulation.er[vertices, layers]
        / simulation._cb_er[vertices, layers]
    )
    represented_moment = np.sum(
        simulation.to_numpy(current_density)
        * dual_areas
        * simulation.radial_node_control_lengths_m[layers]
    )

    assert represented_moment == pytest.approx(
        source.peak_current_a * source.vertical_element_length_m
    )


def test_vertical_source_current_moment_is_preserved_on_nonuniform_grid() -> None:
    altitudes = (-10_000.0, -6_000.0, -2_000.0, -1_000.0, 0.0, 4_000.0)
    source = GaussianCurrent(
        altitude_m=-500.0,
        peak_current_a=3.0,
        vertical_element_length_m=7_500.0,
    )
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            radial_altitudes_m=altitudes,
            radial_grid_policy="allow-abrupt",
            courant_factor=0.2,
        ),
        source=source,
    )
    vertices, layers, _ = source.staggered_distribution(simulation)

    simulation._update_electric_fields(source.peak_current_a)
    dual_areas = (
        simulation.mesh.dual_cell_solid_angles[vertices]
        * simulation.radii_m[layers] ** 2
    )
    current_density = (
        -simulation.er[vertices, layers]
        / simulation._cb_er[vertices, layers]
    )
    represented_moment = np.sum(
        simulation.to_numpy(current_density)
        * dual_areas
        * simulation.radial_node_control_lengths_m[layers]
    )

    assert represented_moment == pytest.approx(
        source.peak_current_a * source.vertical_element_length_m
    )


def test_five_kilometre_source_retains_uniform_grid_deposition() -> None:
    source = GaussianCurrent(
        altitude_m=2_500.0,
        peak_current_a=1.0,
        vertical_element_length_m=5_000.0,
    )
    simulation = GeodesicFDTD(
        SimulationConfig(subdivision=0, radial_cells=40, courant_factor=0.2),
        source=source,
    )
    vertices, layers, weights = source.staggered_distribution(simulation)

    simulation._update_electric_fields(1.0)
    dual_areas = (
        simulation.mesh.dual_cell_solid_angles[vertices]
        * simulation.radii_m[layers] ** 2
    )
    represented_currents = (
        -simulation.er[vertices, layers]
        * dual_areas
        / simulation._cb_er[vertices, layers]
    )

    np.testing.assert_allclose(represented_currents, weights, rtol=5.0e-16)


def test_tangential_source_update_uses_dual_face_current_density() -> None:
    source = TangentialGaussianCurrent(
        altitude_m=0.0,
        peak_current_a=1.0,
        azimuths_deg=(0.0, 90.0),
        line_lengths_m=(22_500.0, 22_500.0),
    )
    simulation = GeodesicFDTD(config=small_config(), source=source)
    edges, layers, expected_weights = source.edge_distribution(simulation)

    simulation._update_electric_fields(1.0)
    dual_face_areas = (
        simulation.mesh.dual_edge_angles[edges]
        * simulation.radial_midpoints_m[layers]
        * simulation.radial_steps_m[layers]
    )
    represented_currents = (
        -simulation.et[edges, layers]
        * dual_face_areas
        / simulation._cb_et[edges, layers]
    )

    np.testing.assert_allclose(represented_currents, expected_weights)


def test_tangential_surface_source_preserves_exact_staggered_altitude() -> None:
    source = TangentialGaussianCurrent(
        altitude_m=0.0,
        azimuths_deg=(0.0,),
        line_lengths_m=(22_500.0,),
    )
    simulation = GeodesicFDTD(
        config=small_config(
            radial_altitudes_m=(
                -5_000.0,
                -3_750.0,
                -2_500.0,
                -1_250.0,
                0.0,
                5_000.0,
                10_000.0,
            ),
                radial_cells=6,
                minimum_altitude_m=-5_000.0,
                maximum_altitude_m=10_000.0,
                radial_grid_policy="allow-abrupt",
            ),
        source=source,
    )

    edges, layers, weights = source.edge_distribution(simulation)

    assert set(simulation.radial_midpoint_altitudes_m[layers]) == {
        -625.0,
        2_500.0,
    }
    for edge in np.unique(edges):
        selected = edges == edge
        edge_weights = weights[selected]
        if np.any(edge_weights):
            assert edge_weights.sum() != pytest.approx(0.0)
            represented_altitude = (
                edge_weights
                @ simulation.radial_midpoint_altitudes_m[layers[selected]]
                / edge_weights.sum()
            )
            assert represented_altitude == pytest.approx(source.altitude_m)


def test_tangential_source_rejects_mismatched_ground_lines() -> None:
    with pytest.raises(ValueError, match="line_lengths_m must match"):
        TangentialGaussianCurrent(
            azimuths_deg=(0.0, 90.0),
            line_lengths_m=(22_500.0,),
        )


def test_sources_reject_nonfinite_or_invalid_waveforms() -> None:
    with pytest.raises(ValueError, match="finite"):
        GaussianCurrent(one_over_e_half_width_s=np.nan)
    with pytest.raises(ValueError, match="positive"):
        GaussianCurrent(one_over_e_half_width_s=0.0)
    with pytest.raises(ValueError, match="latitude"):
        TangentialGaussianCurrent(latitude_deg=91.0)
    with pytest.raises(ValueError, match="finite"):
        TangentialGaussianCurrent(line_lengths_m=(np.inf,))
    with pytest.raises(ValueError, match="finite"):
        GaussianCurrent(vertical_element_length_m=np.inf)
    with pytest.raises(ValueError, match="positive"):
        GaussianCurrent(vertical_element_length_m=0.0)


def test_solver_rejects_temporally_aliased_source_carrier() -> None:
    baseline = GeodesicFDTD(config=small_config())
    nyquist = 0.5 / baseline.time_step_s

    with pytest.raises(ValueError, match="Nyquist"):
        GeodesicFDTD(
            config=small_config(),
            source=GaussianCurrent(carrier_frequency_hz=nyquist),
        )


def test_nearest_edge_source_uses_at_most_one_edge_per_ground_line() -> None:
    source = TangentialGaussianCurrent(
        azimuths_deg=(0.0, 90.0),
        line_lengths_m=(22_500.0, 22_500.0),
        edge_assignment="nearest",
    )
    simulation = GeodesicFDTD(config=small_config(), source=source)
    edges, _, weights = source.edge_distribution(simulation)

    assert len(np.unique(edges[weights != 0.0])) <= 2


def test_requested_unstable_time_step_is_rejected() -> None:
    baseline = GeodesicFDTD(config=small_config())
    with pytest.raises(ValueError, match="exceeds conservative limit"):
        GeodesicFDTD(
            config=small_config(time_step_s=2.0 * baseline.maximum_stable_time_step_s)
        )


def test_simulation_config_rejects_unknown_mesh_orientation() -> None:
    with pytest.raises(ValueError, match="mesh_orientation"):
        small_config(mesh_orientation="sideways")


def test_simulation_config_rejects_unknown_material_support() -> None:
    with pytest.raises(ValueError, match="tangential_material_support"):
        small_config(tangential_material_support="unknown")

    with pytest.raises(ValueError, match="radial_material_support"):
        small_config(radial_material_support="unknown")


def test_simulation_config_rejects_unknown_radial_boundary() -> None:
    with pytest.raises(ValueError, match="radial_boundary_condition"):
        small_config(radial_boundary_condition="absorbing")


def test_simulation_config_rejects_unknown_loss_integration() -> None:
    with pytest.raises(ValueError, match="loss_integration"):
        small_config(loss_integration="forward-euler")


def test_courant_factor_scales_the_unfactored_cfl_limit() -> None:
    simulation = GeodesicFDTD(config=small_config(courant_factor=0.25))

    assert simulation.maximum_stable_time_step_s == pytest.approx(
        0.25 * simulation.cfl_time_step_limit_s
    )


def test_cfl_limit_tracks_fastest_custom_material_wave_speed() -> None:
    class UniformPermittivity(EarthIonosphereMaterial):
        def sample(
            self,
            directions: np.ndarray,
            altitudes_m: np.ndarray,
            earth_radius_m: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            del earth_radius_m
            shape = (len(directions), len(altitudes_m))
            return np.zeros(shape), np.full(shape, 0.25)

    vacuum = GeodesicFDTD(config=small_config(courant_factor=1.0))
    fast = GeodesicFDTD(
        config=small_config(courant_factor=1.0),
        material=UniformPermittivity(),
    )

    assert fast.cfl_time_step_limit_s == pytest.approx(
        0.5 * vacuum.cfl_time_step_limit_s
    )
    assert fast.time_step_s == pytest.approx(fast.cfl_time_step_limit_s)


def test_time_step_unsafe_for_custom_permittivity_is_rejected() -> None:
    class UniformPermittivity(EarthIonosphereMaterial):
        def sample(
            self,
            directions: np.ndarray,
            altitudes_m: np.ndarray,
            earth_radius_m: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            del earth_radius_m
            shape = (len(directions), len(altitudes_m))
            return np.zeros(shape), np.full(shape, 0.25)

    vacuum = GeodesicFDTD(config=small_config(courant_factor=1.0))
    with pytest.raises(ValueError, match="exceeds conservative limit"):
        GeodesicFDTD(
            config=small_config(
                courant_factor=1.0,
                time_step_s=vacuum.cfl_time_step_limit_s,
            ),
            material=UniformPermittivity(),
        )


def test_nonuniform_radial_grid_advances() -> None:
    altitudes = (
        -100_000.0,
        -60_000.0,
        -20_000.0,
        -5_000.0,
        -1_250.0,
        0.0,
        1_250.0,
        5_000.0,
        20_000.0,
        60_000.0,
        100_000.0,
    )
    simulation = GeodesicFDTD(
        config=small_config(
            radial_altitudes_m=altitudes,
            radial_cells=len(altitudes) - 1,
            radial_grid_policy="allow-abrupt",
        ),
        source=GaussianCurrent(),
    )
    simulation.step(5)
    assert np.allclose(simulation.altitudes_m, altitudes)
    assert np.isfinite(simulation.to_numpy(simulation.er)).all()


def test_custom_radial_grid_rejects_unsafe_spacing_jump() -> None:
    with pytest.raises(ValueError, match="smoothly graded"):
        small_config(
            radial_cells=4,
            radial_altitudes_m=(-100_000.0, -5_000.0, -1_250.0, 0.0, 100_000.0),
        )


def test_simulation_config_rejects_unknown_radial_grid_policy() -> None:
    with pytest.raises(ValueError, match="radial_grid_policy"):
        small_config(radial_grid_policy="unchecked")


def test_simulation_config_rejects_unknown_geometry_mode() -> None:
    with pytest.raises(ValueError, match="geometry_mode"):
        small_config(geometry_mode="cylindrical")


def test_smooth_nonuniform_radial_derivative_converges_at_second_order() -> None:
    errors = []
    for radial_cells in (20, 40):
        coordinate = np.linspace(0.0, 1.0, radial_cells + 1)
        mapped = coordinate + 0.2 * coordinate * (1.0 - coordinate)
        altitudes = -10_000.0 + 14_000.0 * mapped
        simulation = GeodesicFDTD(
            config=small_config(
                radial_cells=radial_cells,
                minimum_altitude_m=float(altitudes[0]),
                maximum_altitude_m=float(altitudes[-1]),
                radial_altitudes_m=tuple(altitudes),
                geometry_mode="thin-shell",
            )
        )
        midpoints = simulation.radial_midpoint_altitudes_m
        simulation.et[0].copy_(simulation._runtime.as_tensor(midpoints**2))
        derivative = simulation.to_numpy(simulation._radial_derivative_et())[0]
        exact = 2.0 * altitudes[1:-1]
        errors.append(float(np.max(np.abs(derivative[1:-1] - exact))))

    assert errors[0] / errors[1] == pytest.approx(4.0, rel=2.0e-10)


def test_solver_rejects_incompatible_provided_mesh_configuration() -> None:
    native = build_geodesic_mesh(1, orientation="native")
    with pytest.raises(ValueError, match="orientation"):
        GeodesicFDTD(config=small_config(), mesh=native)
    polar = build_geodesic_mesh(1, orientation="polar")
    with pytest.raises(ValueError, match="cannot accompany"):
        GeodesicFDTD(
            config=small_config(mesh_optimization_steps=1),
            mesh=polar,
        )


@pytest.mark.parametrize(
    ("kind", "pattern"),
    (("shape", "shape"), ("nan", "finite"), ("gain", "negative")),
)
def test_solver_validates_custom_material_outputs(kind: str, pattern: str) -> None:
    class InvalidMaterial(EarthIonosphereMaterial):
        def sample(
            self,
            directions: np.ndarray,
            altitudes_m: np.ndarray,
            earth_radius_m: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            del earth_radius_m
            shape = (len(directions), len(altitudes_m))
            sigma = np.ones(shape)
            epsilon_r = np.ones(shape)
            if kind == "shape":
                sigma = sigma[:1]
            elif kind == "nan":
                epsilon_r[0, 0] = np.nan
            else:
                sigma[0, 0] = -1.0
            return sigma, epsilon_r

    with pytest.raises(ValueError, match=pattern):
        GeodesicFDTD(config=small_config(), material=InvalidMaterial())


def test_solver_uses_fractional_tangential_material_cells() -> None:
    material = LayeredEarthIonosphereMaterial(
        surface_elevation_sampler=lambda directions: np.full(
            len(directions), -207.0
        ),
        tangential_interface_mode="fractional",
    )
    simulation = GeodesicFDTD(
        config=small_config(
            radial_altitudes_m=(-5_000.0, 0.0, 5_000.0),
            radial_cells=2,
            minimum_altitude_m=-5_000.0,
            maximum_altitude_m=5_000.0,
        ),
        material=material,
    )
    water_fraction = 207.0 / 5_000.0
    expected = (
        (1.0 - water_fraction) / material.upper_crust_resistivity_ohm_m
        + water_fraction / material.sea_water_resistivity_ohm_m
    )

    np.testing.assert_allclose(
        simulation.to_numpy(simulation.sigma_et)[:, 0], expected
    )


def test_edge_diamond_support_averages_tangential_material() -> None:
    class DirectionMaterial(EarthIonosphereMaterial):
        def sample(
            self,
            directions: np.ndarray,
            altitudes_m: np.ndarray,
            earth_radius_m: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            del earth_radius_m
            sigma = 1.0e-3 * (2.0 + directions[:, 0, None])
            sigma = np.broadcast_to(
                sigma, (len(directions), len(altitudes_m))
            ).copy()
            return sigma, np.ones_like(sigma)

    config = small_config(tangential_material_support="edge-diamond")
    simulation = GeodesicFDTD(config=config, material=DirectionMaterial())
    mesh = simulation.mesh
    midpoint = mesh.edge_midpoints()
    endpoints = mesh.vertices[mesh.edges]
    left = mesh.face_centers[mesh.edge_left_faces]
    right = mesh.face_centers[mesh.edge_right_faces]
    supports = (
        midpoint + endpoints[:, 0] + left,
        midpoint + left + endpoints[:, 1],
        midpoint + endpoints[:, 1] + right,
        midpoint + right + endpoints[:, 0],
    )
    quadrant_areas = mesh.edge_diamond_quadrant_solid_angles()
    quadrant_weights = quadrant_areas / quadrant_areas.sum(axis=1, keepdims=True)
    expected_direction_x = np.zeros(mesh.n_edges)
    for quadrant, directions in enumerate(supports):
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        expected_direction_x += quadrant_weights[:, quadrant] * directions[:, 0]
    expected_sigma = 1.0e-3 * (2.0 + expected_direction_x)

    np.testing.assert_allclose(
        simulation.to_numpy(simulation.sigma_et)[:, 0], expected_sigma
    )


def test_dual_cell_support_area_averages_radial_material() -> None:
    class DirectionMaterial(EarthIonosphereMaterial):
        def sample(
            self,
            directions: np.ndarray,
            altitudes_m: np.ndarray,
            earth_radius_m: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            del earth_radius_m
            sigma = 1.0e-3 * (2.0 + directions[:, 0, None])
            sigma = np.broadcast_to(
                sigma, (len(directions), len(altitudes_m))
            ).copy()
            return sigma, np.ones_like(sigma)

    simulation = GeodesicFDTD(
        config=small_config(radial_material_support="dual-cell"),
        material=DirectionMaterial(),
    )
    mesh = simulation.mesh
    vertices = mesh.edges.ravel()
    edges = np.repeat(np.arange(mesh.n_edges), 2)
    directions, areas = mesh.dual_cell_wedge_quadrature(vertices, edges)
    weighted_x = np.bincount(
        vertices,
        weights=areas * directions[:, 0],
        minlength=mesh.n_vertices,
    ) / mesh.dual_cell_solid_angles
    expected = 1.0e-3 * (2.0 + weighted_x)

    np.testing.assert_allclose(
        simulation.to_numpy(simulation.sigma_er)[:, 0], expected
    )


def test_dual_cell_support_preserves_uniform_radial_material() -> None:
    point = GeodesicFDTD(config=small_config())
    averaged = GeodesicFDTD(
        config=small_config(radial_material_support="dual-cell")
    )

    np.testing.assert_allclose(averaged.sigma_er, point.sigma_er)
    np.testing.assert_allclose(averaged.epsilon_r_er, point.epsilon_r_er)


def test_uniform_material_coefficients_can_be_broadcast_from_one_row() -> None:
    regular = GeodesicFDTD(config=small_config(), dtype="float64")
    compressed = GeodesicFDTD(
        config=small_config(compress_uniform_material_coefficients=True),
        dtype="float64",
    )
    assert compressed._ca_er.shape[0] == 1
    assert compressed._cb_et.shape[0] == 1
    regular.step(3)
    compressed.step(3)
    np.testing.assert_allclose(compressed.er, regular.er)
    np.testing.assert_allclose(compressed.et, regular.et)


def test_modulated_source_uses_frequency_scaled_default_envelope() -> None:
    source = GaussianCurrent(carrier_frequency_hz=20.0, peak_current_a=1.0)
    assert source.current_a(0.1, 1.0e-6) == pytest.approx(1.0)


def test_loss_coefficient_damps_uncoupled_radial_field() -> None:
    material = EarthIonosphereMaterial(lithosphere_conductivity_s_m=1.0e-2)
    simulation = GeodesicFDTD(config=small_config(), material=material)
    simulation.er[:, 0] = 1.0
    expected = simulation.to_numpy(simulation._ca_er[:, 0]).copy()
    simulation.step()
    assert np.allclose(simulation.to_numpy(simulation.er[:, 0]), expected)


def test_radial_pec_ghost_cells_give_the_expected_one_sided_curl() -> None:
    simulation = GeodesicFDTD(config=small_config(geometry_mode="thin-shell"))
    profile = np.arange(1, simulation.et.shape[1] + 1, dtype=np.float64)
    simulation.et[0].copy_(simulation._runtime.as_tensor(profile))

    expected_derivative = np.empty(simulation.ht.shape[1])
    expected_derivative[0] = 2.0 * profile[0] / simulation.radial_steps_m[0]
    expected_derivative[-1] = -2.0 * profile[-1] / simulation.radial_steps_m[-1]
    expected_derivative[1:-1] = np.diff(profile) / np.diff(
        simulation.radial_midpoints_m
    )
    simulation._update_magnetic_fields()

    np.testing.assert_allclose(
        simulation.to_numpy(simulation.ht[0]),
        -simulation.time_step_s / MU_0 * expected_derivative,
        rtol=2.0e-16,
        atol=0.0,
    )


def test_backend_native_observation_recording_includes_initial_state() -> None:
    simulation = GeodesicFDTD(
        config=small_config(), source=GaussianCurrent(peak_current_a=1.0e6)
    )
    traces = simulation.record_er_observations(
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray((3,), dtype=np.int64),
        np.asarray(((0.2, 0.3, 0.5),)),
        5,
        synchronize_every=2,
    )

    assert traces.shape == (6, 1)
    assert traces[0, 0] == 0.0
    assert simulation.steps == 5
    assert np.isfinite(traces).all()


def test_backend_native_h_recording_includes_initial_state() -> None:
    simulation = GeodesicFDTD(
        config=small_config(), source=GaussianCurrent(peak_current_a=1.0e6)
    )
    hr, ht = simulation.record_h_observations(
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((2,),), dtype=np.int64),
        np.asarray(((1.0,),)),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray(((2, 2, 2),), dtype=np.int64),
        np.asarray(((0.2, -0.3, 0.5),)),
        5,
        synchronize_every=2,
    )

    assert hr.shape == (6, 1)
    assert ht.shape == (6, 1)
    assert hr[0, 0] == 0.0
    assert ht[0, 0] == 0.0
    assert simulation.steps == 5
    assert np.isfinite(hr).all()
    assert np.isfinite(ht).all()


def test_h_observation_decimation_preserves_selected_samples_and_final_step() -> None:
    arguments = (
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((2,),), dtype=np.int64),
        np.asarray(((1.0,),)),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray(((2, 2, 2),), dtype=np.int64),
        np.asarray(((0.2, -0.3, 0.5),)),
    )
    full = GeodesicFDTD(
        config=small_config(), source=GaussianCurrent(peak_current_a=1.0e6)
    )
    sampled = GeodesicFDTD(
        config=small_config(), source=GaussianCurrent(peak_current_a=1.0e6)
    )

    full_hr, full_ht = full.record_h_observations(*arguments, 8)
    sampled_hr, sampled_ht = sampled.record_h_observations(
        *arguments, 8, sample_every=3
    )

    np.testing.assert_allclose(sampled_hr, full_hr[[0, 3, 6, 8]])
    np.testing.assert_allclose(sampled_ht, full_ht[[0, 3, 6, 8]])
    assert sampled.steps == 8


def test_electric_and_magnetic_clocks_follow_leapfrog_staggering() -> None:
    simulation = GeodesicFDTD(config=small_config())

    assert simulation.electric_time_s == 0.0
    assert simulation.magnetic_time_s == pytest.approx(
        -0.5 * simulation.time_step_s
    )
    simulation.step(3)
    diagnostics = simulation.diagnostics()

    assert simulation.electric_time_s == pytest.approx(3.0 * simulation.time_step_s)
    assert simulation.magnetic_time_s == pytest.approx(2.5 * simulation.time_step_s)
    assert diagnostics["electric_time_s"] == pytest.approx(
        simulation.electric_time_s
    )
    assert diagnostics["magnetic_time_s"] == pytest.approx(
        simulation.magnetic_time_s
    )


def test_conservative_anomalies_support_default_material() -> None:
    anomaly = SphericalAnomaly(
        latitude_deg=0.0,
        longitude_deg=0.0,
        radius_m=1_000_000.0,
        altitude_min_m=-100_000.0,
        altitude_max_m=-1.0,
        conductivity_factor=0.5,
        target_area_m2=1.0e12,
    )
    simulation = GeodesicFDTD(
        config=small_config(horizontal_anomaly_mode="conservative-nearest"),
        material=EarthIonosphereMaterial(anomalies=(anomaly,)),
    )

    assert hasattr(simulation, "anomaly_horizontal_fractions_er")
    assert hasattr(simulation, "anomaly_horizontal_fractions_et")


def test_conservative_anomaly_mode_rejects_unsupported_material() -> None:
    class MaterialWithoutAnomalies:
        def sample(
            self,
            directions: np.ndarray,
            altitudes_m: np.ndarray,
            earth_radius_m: float,
        ) -> tuple[np.ndarray, np.ndarray]:
            del earth_radius_m
            shape = (len(directions), len(altitudes_m))
            return np.zeros(shape), np.ones(shape)

    with pytest.raises(ValueError, match="anomalies collection"):
        GeodesicFDTD(
            config=small_config(horizontal_anomaly_mode="conservative-nearest"),
            material=MaterialWithoutAnomalies(),  # type: ignore[arg-type]
        )


def test_step_and_observation_controls_require_integers() -> None:
    simulation = GeodesicFDTD(config=small_config())

    with pytest.raises(ValueError, match="integer"):
        simulation.step(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="integer"):
        simulation.step(True)
    with pytest.raises(ValueError, match="integers"):
        simulation.record_er_observations(
            np.asarray(((0.9,),)),
            np.asarray((0,), dtype=np.int64),
            np.asarray(((1.0,),)),
            0,
        )
    with pytest.raises(ValueError, match="integer"):
        simulation.record_er_observations(
            np.asarray(((0,),), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray(((1.0,),)),
            0,
            synchronize_every=1.5,  # type: ignore[arg-type]
        )


def test_observation_recording_rejects_nonfinite_weights() -> None:
    simulation = GeodesicFDTD(config=small_config())

    with pytest.raises(ValueError, match="finite"):
        simulation.record_er_observations(
            np.asarray(((0,),), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray(((np.nan,),)),
            0,
        )
    with pytest.raises(ValueError, match="finite"):
        simulation.record_h_observations(
            np.asarray(((0,),), dtype=np.int64),
            np.asarray(((0,),), dtype=np.int64),
            np.asarray(((np.nan,),)),
            np.asarray(((0,),), dtype=np.int64),
            np.asarray(((0,),), dtype=np.int64),
            np.asarray(((1.0,),)),
            0,
        )
