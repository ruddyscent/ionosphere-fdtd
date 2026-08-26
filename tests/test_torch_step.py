import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ionosphere_fdtd._torch_step import (  # noqa: E402
    FieldState,
    advance,
    advance_chunk,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig  # noqa: E402
from ionosphere_fdtd.sources import GaussianCurrent  # noqa: E402


def _simulation() -> GeodesicFDTD:
    return GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=2,
            courant_factor=0.2,
        ),
        source=GaussianCurrent(peak_current_a=1.0e6),
        backend="torch",
        device="cpu",
        dtype="float64",
    )


def _trainable_case():
    simulation = _simulation()
    generator = np.random.default_rng(20260826)
    state = FieldState(
        *(
            torch.asarray(
                1.0e-9 * generator.standard_normal(field.shape),
                dtype=torch.float64,
            ).requires_grad_()
            for field in (
                simulation.er,
                simulation.et,
                simulation.hr,
                simulation.ht,
            )
        )
    )
    parameters = simulation._field_step_parameters._replace(
        ca_er=simulation._ca_er.detach().clone().requires_grad_(),
        cb_er=simulation._cb_er.detach().clone().requires_grad_(),
        ca_et=simulation._ca_et.detach().clone().requires_grad_(),
        cb_et=simulation._cb_et.detach().clone().requires_grad_(),
    )
    currents = torch.asarray(
        [1.0e6, -5.0e5], dtype=torch.float64
    ).requires_grad_()
    targets = tuple(state) + (
        currents,
        parameters.ca_er,
        parameters.cb_er,
        parameters.ca_et,
        parameters.cb_et,
    )
    return state, parameters, currents, targets


def _loss(state: FieldState):
    return sum(field.square().mean() for field in state)


def test_advance_does_not_mutate_input_state() -> None:
    state, parameters, currents, _ = _trainable_case()
    snapshots = tuple(field.detach().clone() for field in state)

    result = advance(state, parameters, currents[0])

    for field, snapshot in zip(state, snapshots, strict=True):
        torch.testing.assert_close(field, snapshot, rtol=0.0, atol=0.0)
    for old, new in zip(state, result, strict=True):
        assert old is not new


def test_two_step_loss_has_finite_core_gradients() -> None:
    state, parameters, currents, targets = _trainable_case()

    result = advance_chunk(state, parameters, currents)
    gradients = torch.autograd.grad(_loss(result), targets)

    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradients[4]) == len(currents)
    for gradient in gradients[5:]:
        assert torch.count_nonzero(gradient) > 0


def test_compiled_single_and_chunk_have_zero_graph_breaks() -> None:
    state, parameters, currents, _ = _trainable_case()
    single_explanation = torch._dynamo.explain(advance)(
        state, parameters, currents[0]
    )
    chunk_explanation = torch._dynamo.explain(advance_chunk)(
        state, parameters, currents
    )

    assert single_explanation.graph_count == 1
    assert single_explanation.graph_break_count == 0
    assert chunk_explanation.graph_count == 1
    assert chunk_explanation.graph_break_count == 0

    compiled_single = torch.compile(advance, fullgraph=True, dynamic=False)
    compiled_chunk = torch.compile(
        advance_chunk, fullgraph=True, dynamic=False
    )
    expected_single = advance(state, parameters, currents[0])
    actual_single = compiled_single(state, parameters, currents[0])
    expected_chunk = advance_chunk(state, parameters, currents)
    actual_chunk = compiled_chunk(state, parameters, currents)
    for expected, actual in zip(
        expected_single, actual_single, strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=1.0e-18)
    for expected, actual in zip(expected_chunk, actual_chunk, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2.0e-12, atol=1.0e-18)


def test_compiled_two_step_gradients_match_eager() -> None:
    state, parameters, currents, targets = _trainable_case()
    eager_result = advance_chunk(state, parameters, currents)
    eager_gradients = torch.autograd.grad(_loss(eager_result), targets)

    compiled_chunk = torch.compile(
        advance_chunk, fullgraph=True, dynamic=False
    )
    compiled_result = compiled_chunk(state, parameters, currents)
    compiled_gradients = torch.autograd.grad(_loss(compiled_result), targets)

    for eager, compiled in zip(
        eager_result, compiled_result, strict=True
    ):
        torch.testing.assert_close(compiled, eager, rtol=2.0e-12, atol=1.0e-18)
    for eager, compiled in zip(
        eager_gradients, compiled_gradients, strict=True
    ):
        torch.testing.assert_close(compiled, eager, rtol=1.0e-7, atol=1.0e-14)
