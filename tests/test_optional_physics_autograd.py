import inspect

import numpy as np
import pytest

import torch

from ionosphere_fdtd import (  # noqa: E402
    ConductiveHalfSpaceSurface,
    DatasetProvenance,
    GeodesicFDTD,
    PlasmaCoefficientTensors,
    PlasmaSpeciesCoefficientTensors,
    SimulationConfig,
    SurfaceImpedanceCoefficientTensors,
    VariableProvenance,
    build_geodesic_mesh,
)
from ionosphere_fdtd._torch_step import (  # noqa: E402
    advance_optional_physics,
    advance_optional_physics_chunk,
    advance_plasma,
    advance_surface_impedance,
)
from ionosphere_fdtd.plasma import (  # noqa: E402
    ColdPlasmaSpecies,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    MeshPlasmaModel,
)


def _surface_config() -> SimulationConfig:
    return SimulationConfig(
        subdivision=0,
        radial_cells=4,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
        radial_boundary_condition="surface-impedance",
    )


def _plasma_config() -> SimulationConfig:
    return SimulationConfig(
        subdivision=0,
        radial_cells=4,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
    )


def _plasma_model(mesh) -> MeshPlasmaModel:
    config = _plasma_config()
    altitudes = np.linspace(
        config.minimum_altitude_m,
        config.maximum_altitude_m,
        config.radial_cells + 1,
    )
    midpoints = 0.5 * (altitudes[:-1] + altitudes[1:])
    shape = (mesh.n_faces, len(midpoints))
    magnetic = np.zeros((*shape, 3))
    magnetic[..., 2] = 50.0e-6
    species = ColdPlasmaSpecies(
        "electron",
        -ELEMENTARY_CHARGE_C,
        ELECTRON_MASS_KG,
        np.full(shape, 1.0e3),
        np.full(shape, 200.0),
    )
    provenance = DatasetProvenance(
        dataset_id="test.optional-physics.plasma.v1",
        title="Synthetic optional-physics autograd fixture",
        version="1",
        source_url="https://example.invalid/optional-physics",
        citation="Synthetic fixture.",
        license="CC0-1.0",
        retrieved_at="2026-08-27T00:00:00Z",
        source_sha256="0" * 64,
        coordinate_reference_system="geocentric Cartesian",
        variables=(
            VariableProvenance("B", "T", "T", "identity"),
            VariableProvenance("number_density", "m^-3", "m^-3", "identity"),
            VariableProvenance("collision_frequency", "Hz", "Hz", "identity"),
        ),
    )
    return MeshPlasmaModel.from_mesh(
        mesh,
        midpoints,
        magnetic,
        (species,),
        provenance=(provenance,),
        interpolation="synthetic constants at face/cell centers",
    )


def _seed_fields(simulation: GeodesicFDTD) -> None:
    generator = np.random.default_rng(20260827)
    for name in ("er", "et", "hr", "ht"):
        values = (
            generator.standard_normal(getattr(simulation, name).shape) * 1.0e-9
        )
        getattr(simulation, name).copy_(simulation._runtime.as_tensor(values))


def _surface_coefficients(
    template: GeodesicFDTD,
) -> tuple[SurfaceImpedanceCoefficientTensors, torch.Tensor]:
    ade = template._surface_impedance_ade
    scale = ade._scale.detach().clone().requires_grad_()
    return (
        SurfaceImpedanceCoefficientTensors(
            ade._decay.detach().clone(),
            ade._drive.detach().clone(),
            ade._history_weights.detach().clone(),
            scale,
        ),
        scale,
    )


def _plasma_coefficients(
    template: GeodesicFDTD,
) -> tuple[PlasmaCoefficientTensors, torch.Tensor]:
    ade = template._plasma_coupler.ade
    species = []
    target = None
    for group in ade._coefficients:
        values = [value.detach().clone() for value in group]
        values[0].requires_grad_()
        target = values[0]
        species.append(PlasmaSpeciesCoefficientTensors(*values))
    assert target is not None
    return (
        PlasmaCoefficientTensors(
            ade._magnetic_direction.detach().clone(), tuple(species)
        ),
        target,
    )


def _surface_run(compile_step: bool):
    model = ConductiveHalfSpaceSurface(0.02)
    template = GeodesicFDTD(
        _surface_config(),
        surface_impedance=model,
        device="cpu",
        dtype="float64",
    )
    coefficients, target = _surface_coefficients(template)
    simulation = GeodesicFDTD(
        _surface_config(),
        surface_impedance=model,
        device="cpu",
        dtype="float64",
        compile_step=compile_step,
        compile_chunk_size=2,
        surface_impedance_tensors=coefficients,
    )
    _seed_fields(simulation)
    simulation.step(4)
    loss = 1.0e18 * (
        sum(
            field.square().sum()
            for field in (
                simulation.er,
                simulation.et,
                simulation.hr,
                simulation.ht,
            )
        )
        + simulation._surface_impedance_ade.memory.square().sum()
    )
    return simulation, target, loss


def _plasma_run(compile_step: bool):
    mesh = build_geodesic_mesh(0)
    model = _plasma_model(mesh)
    template = GeodesicFDTD(
        _plasma_config(),
        mesh=mesh,
        plasma=model,
        device="cpu",
        dtype="float64",
    )
    coefficients, target = _plasma_coefficients(template)
    simulation = GeodesicFDTD(
        _plasma_config(),
        mesh=mesh,
        plasma=model,
        device="cpu",
        dtype="float64",
        compile_step=compile_step,
        compile_chunk_size=2,
        plasma_tensors=coefficients,
    )
    _seed_fields(simulation)
    simulation.step(4)
    loss = 1.0e18 * (
        sum(
            field.square().sum()
            for field in (
                simulation.er,
                simulation.et,
                simulation.hr,
                simulation.ht,
            )
        )
        + sum(
            current.square().sum()
            for current in simulation._plasma_coupler.ade.current_density
        )
    )
    return simulation, target, loss


def _assert_forward_close(
    eager: GeodesicFDTD, compiled: GeodesicFDTD
) -> None:
    for eager_field, compiled_field in zip(
        (eager.er, eager.et, eager.hr, eager.ht),
        (compiled.er, compiled.et, compiled.hr, compiled.ht),
        strict=True,
    ):
        torch.testing.assert_close(
            compiled_field, eager_field, rtol=3.0e-13, atol=1.0e-22
        )


def test_surface_scale_has_finite_eager_and_compiled_gradients() -> None:
    eager, eager_target, eager_loss = _surface_run(False)
    compiled, compiled_target, compiled_loss = _surface_run(True)
    eager_gradient = torch.autograd.grad(eager_loss, eager_target)[0]
    compiled_gradient = torch.autograd.grad(compiled_loss, compiled_target)[0]

    _assert_forward_close(eager, compiled)
    torch.testing.assert_close(
        compiled._surface_impedance_ade.memory,
        eager._surface_impedance_ade.memory,
        rtol=3.0e-13,
        atol=1.0e-22,
    )
    for gradient in (eager_gradient, compiled_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    torch.testing.assert_close(
        compiled_gradient, eager_gradient, rtol=1.0e-10, atol=1.0e-10
    )


def test_plasma_decay_has_finite_eager_and_compiled_gradients() -> None:
    eager, eager_target, eager_loss = _plasma_run(False)
    compiled, compiled_target, compiled_loss = _plasma_run(True)
    eager_gradient = torch.autograd.grad(eager_loss, eager_target)[0]
    compiled_gradient = torch.autograd.grad(compiled_loss, compiled_target)[0]

    _assert_forward_close(eager, compiled)
    for eager_current, compiled_current in zip(
        eager._plasma_coupler.ade.current_density,
        compiled._plasma_coupler.ade.current_density,
        strict=True,
    ):
        torch.testing.assert_close(
            compiled_current, eager_current, rtol=3.0e-13, atol=1.0e-22
        )
    for gradient in (eager_gradient, compiled_gradient):
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    torch.testing.assert_close(
        compiled_gradient, eager_gradient, rtol=2.0e-6, atol=3.0e-6
    )


@pytest.mark.parametrize("physics", ("surface", "plasma"))
def test_optional_physics_step_does_not_mutate_input_state(physics: str) -> None:
    simulation, _, _ = (
        _surface_run(False) if physics == "surface" else _plasma_run(False)
    )
    old_state = simulation._torch_optional_physics_state()
    snapshots = tuple(value.detach().clone() for value in old_state.fields)
    surface_snapshot = (
        old_state.surface_memory.detach().clone()
        if old_state.surface_memory is not None
        else None
    )
    plasma_snapshots = tuple(
        value.detach().clone() for value in old_state.plasma_current_density
    )

    new_state = advance_optional_physics(
        old_state,
        simulation._optional_physics_step_parameters,
        torch.tensor(0.0, dtype=torch.float64),
    )

    for old, snapshot, new in zip(
        old_state.fields, snapshots, new_state.fields, strict=True
    ):
        torch.testing.assert_close(old, snapshot, rtol=0.0, atol=0.0)
        assert new is not old
    if surface_snapshot is not None:
        torch.testing.assert_close(
            old_state.surface_memory, surface_snapshot, rtol=0.0, atol=0.0
        )
        assert new_state.surface_memory is not old_state.surface_memory
    for old, snapshot, new in zip(
        old_state.plasma_current_density,
        plasma_snapshots,
        new_state.plasma_current_density,
        strict=True,
    ):
        torch.testing.assert_close(old, snapshot, rtol=0.0, atol=0.0)
        assert new is not old


@pytest.mark.parametrize("physics", ("surface", "plasma"))
def test_optional_physics_step_and_chunk_have_no_graph_breaks(
    physics: str,
) -> None:
    simulation, _, _ = (
        _surface_run(False) if physics == "surface" else _plasma_run(False)
    )
    state = simulation._torch_optional_physics_state()
    parameters = simulation._optional_physics_step_parameters
    current = torch.tensor(0.0, dtype=torch.float64)
    currents = torch.zeros(2, dtype=torch.float64)

    step = torch._dynamo.explain(advance_optional_physics)(
        state, parameters, current
    )
    chunk = torch._dynamo.explain(advance_optional_physics_chunk)(
        state, parameters, currents
    )

    assert step.graph_count == 1
    assert step.graph_break_count == 0
    assert chunk.graph_count == 1
    assert chunk.graph_break_count == 0


def test_differentiable_optional_core_has_no_host_or_detach_boundary() -> None:
    source = "\n".join(
        inspect.getsource(function)
        for function in (
            advance_surface_impedance,
            advance_plasma,
            advance_optional_physics,
            advance_optional_physics_chunk,
        )
    )

    for forbidden in (".detach(", ".cpu(", ".numpy(", ".item(", "np."):
        assert forbidden not in source


@pytest.mark.parametrize("physics", ("surface", "plasma"))
def test_checkpoint_detaches_but_preserves_functional_ade_state(
    physics: str, tmp_path
) -> None:
    simulation, _, _ = (
        _surface_run(False) if physics == "surface" else _plasma_run(False)
    )
    restored = GeodesicFDTD.load_checkpoint(
        simulation.save_checkpoint(tmp_path / f"{physics}.npz"),
        device="cpu",
        dtype="float64",
    )

    if physics == "surface":
        expected = simulation._surface_impedance_ade.memory
        actual = restored._surface_impedance_ade.memory
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        assert not actual.requires_grad
    else:
        for expected, actual in zip(
            simulation._plasma_coupler.ade.current_density,
            restored._plasma_coupler.ade.current_density,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
            assert not actual.requires_grad


def test_optional_tensor_coefficients_validate_model_backend_and_shape() -> None:
    surface = ConductiveHalfSpaceSurface(0.02)
    template = GeodesicFDTD(
        _surface_config(),
        surface_impedance=surface,
        device="cpu",
        dtype="float64",
    )
    coefficients, _ = _surface_coefficients(template)

    simulation = GeodesicFDTD(
        _surface_config(),
        surface_impedance=surface,
        surface_impedance_tensors=coefficients,
    )
    assert simulation.runtime == "torch"
    with pytest.raises(ValueError, match="requires a surface impedance model"):
        GeodesicFDTD(
            SimulationConfig(subdivision=0, radial_cells=4),
            device="cpu",
            dtype="float64",
            surface_impedance_tensors=coefficients,
        )
    invalid = SurfaceImpedanceCoefficientTensors(
        coefficients.decay,
        coefficients.drive,
        coefficients.history_weights,
        torch.ones(1, dtype=torch.float64),
    )
    with pytest.raises(ValueError, match="scale must have shape"):
        GeodesicFDTD(
            _surface_config(),
            surface_impedance=surface,
            device="cpu",
            dtype="float64",
            surface_impedance_tensors=invalid,
        )


def test_optional_tensor_dtype_conversion_preserves_graph() -> None:
    surface = ConductiveHalfSpaceSurface(0.02)
    template = GeodesicFDTD(
        _surface_config(),
        surface_impedance=surface,
        device="cpu",
        dtype="float64",
    )
    ade = template._surface_impedance_ade
    scale = ade._scale.to(torch.float32).detach().requires_grad_()
    coefficients = SurfaceImpedanceCoefficientTensors(
        ade._decay.to(torch.float32),
        ade._drive.to(torch.float32),
        ade._history_weights.to(torch.float32),
        scale,
    )
    simulation = GeodesicFDTD(
        _surface_config(),
        surface_impedance=surface,
        device="cpu",
        dtype="float64",
        surface_impedance_tensors=coefficients,
    )

    gradient = torch.autograd.grad(
        simulation._surface_impedance_ade._scale.sum(), scale
    )[0]

    assert simulation.surface_impedance_tensors.scale.dtype == torch.float64
    assert gradient.dtype == torch.float32
    assert torch.isfinite(gradient).all()
