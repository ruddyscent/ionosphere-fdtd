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


def _er_receiver(simulation: GeodesicFDTD):
    assert simulation.source is not None
    vertices, layers, _ = simulation.source.staggered_distribution(
        simulation
    )
    return (
        vertices[:1, None],
        layers[:1],
        np.ones((1, 1), dtype=np.float64),
    )


@pytest.mark.parametrize("compile_step", (False, True))
def test_receiver_loss_reaches_current_and_waveform_amplitude(
    compile_step: bool,
) -> None:
    source = GaussianCurrent(
        center_time_s=0.0,
        one_over_e_half_width_s=1.0,
    )
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=2,
            courant_factor=0.2,
        ),
        source=source,
        backend="torch",
        device="cpu",
        dtype="float64",
        compile_step=compile_step,
        compile_chunk_size=2,
    )
    steps = 4
    peak_current_a = torch.tensor(
        1.0e6, dtype=torch.float64, requires_grad=True
    )
    times_s = (
        torch.arange(steps, dtype=torch.float64) + 0.5
    ) * simulation.time_step_s
    currents = source.current_tensor_a(
        times_s,
        simulation.time_step_s,
        peak_current_a=peak_current_a,
    )

    assert simulation._source_currents(
        steps, currents=currents
    ) is currents
    traces = simulation.record_er_observations(
        *_er_receiver(simulation),
        steps,
        currents=currents,
    )
    current_gradient, amplitude_gradient = torch.autograd.grad(
        traces.square().sum(), (currents, peak_current_a)
    )

    assert torch.is_tensor(traces)
    assert traces.device == simulation.er.device
    assert traces.dtype == simulation.er.dtype
    assert traces.requires_grad
    assert torch.isfinite(current_gradient).all()
    assert torch.count_nonzero(current_gradient) == steps
    assert torch.isfinite(amplitude_gradient)
    assert torch.count_nonzero(amplitude_gradient) == 1


def test_h_observations_preserve_the_current_graph() -> None:
    simulation = _simulation()
    currents = torch.linspace(
        0.5e6,
        1.0e6,
        4,
        dtype=torch.float64,
        requires_grad=True,
    )
    radial_traces, tangential_traces = simulation.record_h_observations(
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((1.0,),)),
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((1.0,),)),
        4,
        currents=currents,
    )
    gradient = torch.autograd.grad(
        tangential_traces.square().sum(), currents
    )[0]

    for traces in (radial_traces, tangential_traces):
        assert torch.is_tensor(traces)
        assert traces.device == simulation.er.device
        assert traces.dtype == simulation.er.dtype
    assert tangential_traces.requires_grad
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_observation_export_has_one_explicit_host_copy(monkeypatch) -> None:
    simulation = _simulation()
    calls = 0
    export_numpy = simulation.backend._runtime.export_numpy

    def counted_export(values):
        nonlocal calls
        calls += 1
        return export_numpy(values)

    monkeypatch.setattr(
        simulation.backend._runtime, "export_numpy", counted_export
    )
    traces = simulation.record_er_observations(
        *_er_receiver(simulation), 2
    )

    assert calls == 0
    exported = simulation.to_numpy(traces)
    assert calls == 1
    assert isinstance(exported, np.ndarray)


def test_external_currents_require_a_source_and_one_value_per_step() -> None:
    simulation = _simulation()
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        simulation.step(
            2, currents=torch.ones(3, dtype=torch.float64)
        )

    source_free = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=2,
            courant_factor=0.2,
        ),
        backend="torch",
        device="cpu",
        dtype="float64",
    )
    with pytest.raises(ValueError, match="configured source"):
        source_free.step(
            2, currents=torch.ones(2, dtype=torch.float64)
        )


def test_tensor_waveform_matches_scalar_waveform() -> None:
    source = GaussianCurrent(
        peak_current_a=3.0,
        center_time_s=0.2,
        one_over_e_half_width_s=0.04,
        carrier_frequency_hz=5.0,
    )
    times = torch.linspace(0.1, 0.3, 11, dtype=torch.float64)
    expected = torch.tensor(
        [source.current_a(float(time), 1.0e-4) for time in times],
        dtype=torch.float64,
    )

    actual = source.current_tensor_a(times, 1.0e-4)

    torch.testing.assert_close(actual, expected, rtol=2.0e-15, atol=0.0)


def test_torch_source_generation_does_not_use_numpy_fromiter(
    monkeypatch,
) -> None:
    def reject_fromiter(*args, **kwargs):
        del args, kwargs
        raise AssertionError("Torch source generation used NumPy")

    monkeypatch.setattr(np, "fromiter", reject_fromiter)
    simulation = _simulation()

    simulation.step(3)

    assert torch.isfinite(simulation.er).all()


def test_observation_synchronization_is_opt_in(monkeypatch) -> None:
    simulation = _simulation()
    calls = 0

    def counted_synchronize():
        nonlocal calls
        calls += 1

    monkeypatch.setattr(simulation.backend, "synchronize", counted_synchronize)

    simulation.record_er_observations(*_er_receiver(simulation), 2)
    assert calls == 0

    simulation.record_er_observations(
        *_er_receiver(simulation), 1, synchronize_every=1
    )
    assert calls == 2
