import numpy as np
import pytest
import torch

from ionosphere_fdtd.constants import C_0, EPSILON_0
from ionosphere_fdtd.mesh import build_geodesic_mesh
from ionosphere_fdtd.mesh_quality import scalar_laplacian
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import (
    TangentialGaussianCurrent,
    geographic_tangent_basis,
)


class VacuumMaterial:
    def sample(
        self,
        directions: np.ndarray,
        altitudes_m: np.ndarray,
        earth_radius_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del earth_radius_m
        shape = (len(directions), len(altitudes_m))
        return np.zeros(shape), np.ones(shape)


class UniformConductiveMaterial(VacuumMaterial):
    def __init__(self, conductivity_s_m: float) -> None:
        self.conductivity_s_m = conductivity_s_m

    def sample(
        self,
        directions: np.ndarray,
        altitudes_m: np.ndarray,
        earth_radius_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma, epsilon_r = super().sample(
            directions, altitudes_m, earth_radius_m
        )
        sigma.fill(self.conductivity_s_m)
        return sigma, epsilon_r


class UniformPermittivityMaterial(VacuumMaterial):
    def __init__(self, relative_permittivity: float) -> None:
        self.relative_permittivity = relative_permittivity

    def sample(
        self,
        directions: np.ndarray,
        altitudes_m: np.ndarray,
        earth_radius_m: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        sigma, epsilon_r = super().sample(
            directions, altitudes_m, earth_radius_m
        )
        epsilon_r.fill(self.relative_permittivity)
        return sigma, epsilon_r


def _surface_mode_config() -> SimulationConfig:
    return SimulationConfig(
        subdivision=1,
        radial_cells=2,
        minimum_altitude_m=-1.0e9,
        maximum_altitude_m=1.0e9,
        earth_radius_m=1.0e10,
        courant_factor=0.8,
    )


def _operator_matrix(size: int, apply: object) -> np.ndarray:
    identity = np.eye(size)
    return np.column_stack([apply(identity[:, index]) for index in range(size)])


def test_surface_incidence_pairs_satisfy_discrete_adjoint_identities() -> None:
    mesh = build_geodesic_mesh(2)
    generator = np.random.default_rng(20260805)
    edge_values = generator.standard_normal(mesh.n_edges)
    face_values = generator.standard_normal(mesh.n_faces)
    vertex_values = generator.standard_normal(mesh.n_vertices)

    primal_left = face_values @ mesh.face_circulation(edge_values)
    primal_right = mesh.dual_edge_difference(face_values) @ edge_values
    dual_left = vertex_values @ mesh.dual_cell_circulation(edge_values)
    dual_right = -mesh.edge_difference(vertex_values) @ edge_values

    assert primal_left == pytest.approx(primal_right, abs=2.0e-13)
    assert dual_left == pytest.approx(dual_right, abs=2.0e-13)


def test_tm_r_surface_eigenmode_matches_leapfrog_dispersion() -> None:
    simulation = GeodesicFDTD(
        _surface_mode_config(), material=VacuumMaterial(), dtype="float64"
    )
    mesh = simulation.mesh
    operator = _operator_matrix(
        mesh.n_vertices, lambda values: scalar_laplacian(mesh, values)
    )
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    selected = int(np.argmin(np.abs(eigenvalues.real + 2.0)))
    eigenvalue = -float(eigenvalues[selected].real)
    mode = eigenvectors[:, selected].real
    layer = 1
    simulation.er[:, layer].copy_(torch.as_tensor(mode))
    amplitudes = [1.0]

    for _ in range(100):
        simulation._update_magnetic_fields()
        simulation._update_electric_fields()
        simulation.et.zero_()
        simulation.hr.zero_()
        amplitudes.append(float(simulation.to_numpy(simulation.er[:, layer]) @ mode / (mode @ mode)))

    radius = simulation.radii_m[layer]
    q = (C_0 * simulation.time_step_s) ** 2 * eigenvalue / radius**2
    phase = np.arccos(1.0 - 0.5 * q)
    steps = np.arange(len(amplitudes))
    expected = np.cos((steps + 0.5) * phase) / np.cos(0.5 * phase)

    np.testing.assert_allclose(amplitudes, expected, rtol=0.0, atol=8.0e-14)
    assert not simulation.hr.any().item()


def test_te_r_surface_eigenmode_matches_leapfrog_dispersion() -> None:
    simulation = GeodesicFDTD(
        _surface_mode_config(), material=VacuumMaterial(), dtype="float64"
    )
    mesh = simulation.mesh

    def te_operator(values: np.ndarray) -> np.ndarray:
        edge_gradient = (
            mesh.primal_edge_angles
            / mesh.dual_edge_angles
            * mesh.dual_edge_difference(values)
        )
        return mesh.face_circulation(edge_gradient) / mesh.face_solid_angles

    operator = _operator_matrix(mesh.n_faces, te_operator)
    eigenvalues, eigenvectors = np.linalg.eig(operator)
    selected = int(np.argmin(np.abs(eigenvalues.real - 2.0)))
    eigenvalue = float(eigenvalues[selected].real)
    mode = eigenvectors[:, selected].real
    layer = 1
    simulation.hr[:, layer].copy_(torch.as_tensor(mode))
    amplitudes = [1.0]

    for _ in range(100):
        simulation._update_magnetic_fields()
        simulation.ht.zero_()
        simulation._update_electric_fields()
        simulation.er.zero_()
        amplitudes.append(float(simulation.to_numpy(simulation.hr[:, layer]) @ mode / (mode @ mode)))

    radius = simulation.radial_midpoints_m[layer]
    q = (C_0 * simulation.time_step_s) ** 2 * eigenvalue / radius**2
    phase = np.arccos(1.0 - 0.5 * q)
    steps = np.arange(len(amplitudes))
    expected = np.cos((steps - 0.5) * phase) / np.cos(0.5 * phase)

    np.testing.assert_allclose(amplitudes, expected, rtol=0.0, atol=8.0e-14)
    assert not simulation.er.any().item()


@pytest.mark.parametrize("orientation", ("polar", "native"))
@pytest.mark.parametrize("subdivision", (1, 3))
def test_tangential_source_covariant_samples_reconstruct_requested_moment(
    subdivision: int,
    orientation: str,
) -> None:
    line_length = 22_500.0
    source = TangentialGaussianCurrent(
        latitude_deg=46.5,
        longitude_deg=-90.9,
        altitude_m=0.0,
        azimuths_deg=(0.0, 90.0),
        line_lengths_m=(line_length, line_length),
    )
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=subdivision,
            radial_cells=2,
            minimum_altitude_m=-5_000.0,
            maximum_altitude_m=5_000.0,
            mesh_orientation=orientation,
        ),
        source=source,
        dtype="float64",
    )
    edges, _, weights = source.edge_distribution(simulation)
    unique_edges = np.unique(edges)
    endpoints = simulation.mesh.vertices[simulation.mesh.edges[unique_edges]]
    directions = endpoints[:, 1] - endpoints[:, 0]
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    lengths = (
        simulation.mesh.primal_edge_angles[unique_edges]
        * simulation.config.earth_radius_m
    )
    covariant_samples = np.asarray(
        [
            weights[edges == edge].sum() * length
            for edge, length in zip(unique_edges, lengths, strict=True)
        ]
    )
    east, north = geographic_tangent_basis(
        source.latitude_deg, source.longitude_deg
    )
    analysis = np.column_stack((directions @ east, directions @ north))
    reconstructed = np.linalg.pinv(analysis) @ covariant_samples

    np.testing.assert_allclose(
        reconstructed,
        (line_length, line_length),
        rtol=0.0,
        atol=2.0e-11,
    )


def test_conductive_update_is_passive_even_in_the_stiff_limit() -> None:
    simulation = GeodesicFDTD(
        SimulationConfig(subdivision=0, radial_cells=2, courant_factor=0.2),
        material=UniformConductiveMaterial(1.0),
        dtype="float64",
    )
    simulation.er.fill_(1.0)
    norms = [float(np.linalg.norm(simulation.er))]

    for _ in range(20):
        simulation.step()
        norms.append(float(np.linalg.norm(simulation.er)))

    assert np.all(np.diff(norms) <= 0.0)
    assert torch.max(simulation._ca_er).item() < 1.0
    assert torch.min(simulation._ca_er).item() >= 0.0


def test_exponential_conductive_forcing_converges_at_second_order() -> None:
    base = GeodesicFDTD(
        SimulationConfig(subdivision=0, radial_cells=2, courant_factor=1.0),
        material=VacuumMaterial(),
        dtype="float64",
    )
    coarse_dt = 0.4 * base.cfl_time_step_limit_s
    coarse_steps = 20
    duration = coarse_steps * coarse_dt
    conductivity = 0.5 * EPSILON_0 / duration
    errors = []

    for refinement in (1, 2):
        time_step = coarse_dt / refinement
        simulation = GeodesicFDTD(
            SimulationConfig(
                subdivision=0,
                radial_cells=2,
                courant_factor=1.0,
                time_step_s=time_step,
            ),
            material=UniformConductiveMaterial(conductivity),
            dtype="float64",
        )
        coefficient_a = float(simulation._ca_er[0, 0])
        coefficient_b = float(simulation._cb_er[0, 0])
        field = 0.0
        for step in range(coarse_steps * refinement):
            midpoint_time = (step + 0.5) * time_step
            field = coefficient_a * field + coefficient_b * midpoint_time
        exact = (
            duration / conductivity
            - EPSILON_0
            / conductivity**2
            * (1.0 - np.exp(-conductivity * duration / EPSILON_0))
        )
        errors.append(abs(field - exact))

    assert errors[0] / errors[1] == pytest.approx(4.0, rel=3.0e-3)


def test_exponential_loss_matches_exact_stiff_decay_without_sign_flip() -> None:
    conductivity = 1.0
    simulation = GeodesicFDTD(
        SimulationConfig(subdivision=0, radial_cells=2, courant_factor=0.2),
        material=UniformConductiveMaterial(conductivity),
        dtype="float64",
    )
    simulation.er.fill_(1.0)
    exact = np.exp(-conductivity * simulation.time_step_s / EPSILON_0)

    simulation.step()

    np.testing.assert_allclose(simulation.er, exact, rtol=0.0, atol=1.0e-300)
    assert torch.all(simulation.er >= 0.0).item()


def test_trapezoidal_loss_mode_retains_legacy_coefficients() -> None:
    conductivity = 1.0
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=2,
            courant_factor=0.2,
            loss_integration="trapezoidal",
        ),
        material=UniformConductiveMaterial(conductivity),
        dtype="float64",
    )
    loss = conductivity * simulation.time_step_s / (2.0 * EPSILON_0)

    np.testing.assert_allclose(
        simulation._ca_er,
        (1.0 - loss) / (1.0 + loss),
        rtol=0.0,
        atol=0.0,
    )
    assert torch.all(simulation._ca_er < 0.0).item()


def test_material_aware_cfl_keeps_subvacuum_permittivity_bounded() -> None:
    simulation = GeodesicFDTD(
        SimulationConfig(subdivision=0, radial_cells=2, courant_factor=1.0),
        material=UniformPermittivityMaterial(0.01),
        dtype="float64",
    )
    generator = np.random.default_rng(20260805)
    simulation.er.copy_(torch.as_tensor(generator.standard_normal(simulation.er.shape)))
    simulation.et.copy_(torch.as_tensor(generator.standard_normal(simulation.et.shape)))
    initial_maximum = max(
        float(np.max(np.abs(simulation.to_numpy(simulation.er)))),
        float(np.max(np.abs(simulation.to_numpy(simulation.et)))),
    )
    maximum = initial_maximum

    for _ in range(400):
        simulation.step()
        maximum = max(
            maximum,
            float(np.max(np.abs(simulation.to_numpy(simulation.er)))),
            float(np.max(np.abs(simulation.to_numpy(simulation.et)))),
        )

    assert np.isfinite(maximum)
    assert maximum < 2.0 * initial_maximum


def test_nonuniform_radial_stencils_satisfy_weighted_adjoint_identity() -> None:
    altitudes = (-10_000.0, -6_000.0, -2_000.0, -1_000.0, 0.0, 4_000.0)
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            radial_altitudes_m=altitudes,
            radial_grid_policy="allow-abrupt",
            geometry_mode="thin-shell",
        ),
        material=VacuumMaterial(),
        dtype="float64",
    )
    generator = np.random.default_rng(20260805)
    simulation.et.copy_(torch.as_tensor(generator.standard_normal(simulation.et.shape)))
    simulation.ht.copy_(torch.as_tensor(generator.standard_normal(simulation.ht.shape)))
    derivative_et = simulation.to_numpy(simulation._radial_derivative_et())
    derivative_ht = simulation.to_numpy(simulation._radial_derivative_ht())
    ht_weights = np.empty(len(altitudes))
    ht_weights[0] = 0.5 * simulation.radial_steps_m[0]
    ht_weights[-1] = 0.5 * simulation.radial_steps_m[-1]
    ht_weights[1:-1] = np.diff(simulation.radial_midpoints_m)

    left = np.sum(simulation.to_numpy(simulation.ht) * derivative_et * ht_weights[None, :])
    right = np.sum(
        simulation.to_numpy(simulation.et) * derivative_ht * simulation.radial_steps_m[None, :]
    )

    assert left + right == pytest.approx(0.0, abs=2.0e-11)


def test_nonuniform_radial_derivative_annihilates_constant_ht() -> None:
    altitudes = (-10_000.0, -6_000.0, -2_000.0, -1_000.0, 0.0, 4_000.0)
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            radial_altitudes_m=altitudes,
            radial_grid_policy="allow-abrupt",
            geometry_mode="thin-shell",
        ),
        material=VacuumMaterial(),
        dtype="float64",
    )
    simulation.ht.fill_(1.0)

    derivative = simulation.to_numpy(simulation._radial_derivative_ht())

    np.testing.assert_array_equal(derivative, 0.0)


def test_full_spherical_radial_stencils_satisfy_physical_adjoint_identity() -> None:
    altitudes = (-10_000.0, -6_000.0, -2_000.0, -1_000.0, 0.0, 4_000.0)
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            radial_altitudes_m=altitudes,
            radial_grid_policy="allow-abrupt",
            geometry_mode="full-spherical",
        ),
        material=VacuumMaterial(),
        dtype="float64",
    )
    generator = np.random.default_rng(20260805)
    simulation.et.copy_(torch.as_tensor(generator.standard_normal(simulation.et.shape)))
    simulation.ht.copy_(torch.as_tensor(generator.standard_normal(simulation.ht.shape)))
    derivative_et = simulation.to_numpy(simulation._radial_derivative_et())
    derivative_ht = simulation.to_numpy(simulation._radial_derivative_ht())
    ht_weights = (
        simulation.radial_node_control_lengths_m * simulation.radii_m**2
    )
    et_weights = (
        simulation.radial_steps_m * simulation.radial_midpoints_m**2
    )

    left = np.sum(simulation.to_numpy(simulation.ht) * derivative_et * ht_weights[None, :])
    right = np.sum(simulation.to_numpy(simulation.et) * derivative_ht * et_weights[None, :])

    assert abs(left + right) <= 2.0e-15 * max(abs(left), abs(right))


def test_full_spherical_metric_annihilates_inverse_radius_profiles() -> None:
    base = dict(
        subdivision=0,
        radial_cells=8,
        minimum_altitude_m=-100_000.0,
        maximum_altitude_m=100_000.0,
    )
    simulation = GeodesicFDTD(
        SimulationConfig(**base, geometry_mode="full-spherical"),
        material=VacuumMaterial(),
        dtype="float64",
    )
    simulation.et.copy_(torch.as_tensor(1.0 / simulation.radial_midpoints_m[None, :]))
    simulation.ht.copy_(torch.as_tensor(1.0 / simulation.radii_m[None, :]))

    derivative_et = simulation.to_numpy(simulation._radial_derivative_et())
    derivative_ht = simulation.to_numpy(simulation._radial_derivative_ht())

    np.testing.assert_allclose(derivative_et[:, 1:-1], 0.0, atol=1.0e-27)
    np.testing.assert_allclose(derivative_ht, 0.0, atol=1.0e-27)

    thin_shell = GeodesicFDTD(
        SimulationConfig(**base, geometry_mode="thin-shell"),
        material=VacuumMaterial(),
        dtype="float64",
    )
    thin_shell.et.copy_(torch.as_tensor(1.0 / thin_shell.radial_midpoints_m[None, :]))
    thin_derivative = thin_shell.to_numpy(thin_shell._radial_derivative_et())
    assert np.max(np.abs(thin_derivative[:, 1:-1])) > 2.0e-14


def test_graded_nonuniform_grid_remains_bounded_at_cfl_limit() -> None:
    altitudes = (-10_000.0, -6_000.0, -2_000.0, -1_000.0, 0.0, 4_000.0)
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            radial_altitudes_m=altitudes,
            radial_grid_policy="allow-abrupt",
            courant_factor=1.0,
        ),
        material=VacuumMaterial(),
        dtype="float64",
    )
    generator = np.random.default_rng(20260805)
    simulation.er.copy_(torch.as_tensor(generator.standard_normal(simulation.er.shape)))
    simulation.et.copy_(torch.as_tensor(generator.standard_normal(simulation.et.shape)))
    initial_maximum = max(
        float(np.max(np.abs(simulation.to_numpy(simulation.er)))),
        float(np.max(np.abs(simulation.to_numpy(simulation.et)))),
    )
    maximum = initial_maximum

    for _ in range(1_000):
        simulation.step()
        maximum = max(
            maximum,
            float(np.max(np.abs(simulation.to_numpy(simulation.er)))),
            float(np.max(np.abs(simulation.to_numpy(simulation.et)))),
        )

    assert np.isfinite(maximum)
    assert maximum < 2.0 * initial_maximum


@pytest.mark.parametrize("orientation", ("native", "polar"))
def test_surface_spectra_remain_inside_the_cfl_limit(orientation: str) -> None:
    config = _surface_mode_config()
    config = SimulationConfig(
        subdivision=config.subdivision,
        radial_cells=config.radial_cells,
        minimum_altitude_m=config.minimum_altitude_m,
        maximum_altitude_m=config.maximum_altitude_m,
        earth_radius_m=config.earth_radius_m,
        courant_factor=1.0,
        mesh_orientation=orientation,
    )
    simulation = GeodesicFDTD(config, material=VacuumMaterial(), dtype="float64")
    mesh = simulation.mesh
    tm_operator = _operator_matrix(
        mesh.n_vertices, lambda values: scalar_laplacian(mesh, values)
    )

    def apply_te(values: np.ndarray) -> np.ndarray:
        gradient = (
            mesh.primal_edge_angles
            / mesh.dual_edge_angles
            * mesh.dual_edge_difference(values)
        )
        return mesh.face_circulation(gradient) / mesh.face_solid_angles

    te_operator = _operator_matrix(mesh.n_faces, apply_te)
    tm_eigenvalue = -float(np.linalg.eigvals(tm_operator).real.min())
    te_eigenvalue = float(np.linalg.eigvals(te_operator).real.max())
    tm_q = (
        C_0 * simulation.time_step_s / float(simulation.radii_m.min())
    ) ** 2 * tm_eigenvalue
    te_q = (
        C_0 * simulation.time_step_s / float(simulation.radial_midpoints_m.min())
    ) ** 2 * te_eigenvalue

    assert 0.0 < tm_q < 4.0
    assert 0.0 < te_q < 4.0


def test_surface_laplacian_spectrum_is_rotation_invariant() -> None:
    spectra = []
    for orientation in ("native", "polar"):
        mesh = build_geodesic_mesh(1, orientation=orientation)
        operator = _operator_matrix(
            mesh.n_vertices, lambda values: scalar_laplacian(mesh, values)
        )
        spectra.append(np.sort(np.linalg.eigvals(operator).real))

    np.testing.assert_allclose(spectra[0], spectra[1], rtol=0.0, atol=2.0e-13)
