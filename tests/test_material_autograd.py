from dataclasses import replace

import numpy as np
import pytest

import torch

from ionosphere_fdtd import (  # noqa: E402
    GaussianCurrent,
    GeodesicFDTD,
    MaterialUpdateCoefficientTensors,
    SampledMaterialTensors,
    SimulationConfig,
)
import ionosphere_fdtd.solver as solver_module  # noqa: E402


def _config(**kwargs) -> SimulationConfig:
    return SimulationConfig(
        subdivision=0,
        radial_cells=2,
        courant_factor=0.2,
        **kwargs,
    )


def _material_shapes(
    config: SimulationConfig | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    simulation = GeodesicFDTD(config or _config())
    if simulation.config.compress_uniform_material_coefficients:
        return (1, len(simulation.radii_m)), (
            1,
            len(simulation.radial_midpoints_m),
        )
    return simulation.sigma_er.shape, simulation.sigma_et.shape


def _sampled_material(
    sigma,
    epsilon_r,
    *,
    config: SimulationConfig | None = None,
) -> SampledMaterialTensors:
    er_shape, et_shape = _material_shapes(config)
    return SampledMaterialTensors(
        torch.ones(er_shape, dtype=torch.float64) * sigma,
        torch.ones(er_shape, dtype=torch.float64) * epsilon_r,
        torch.ones(et_shape, dtype=torch.float64) * sigma,
        torch.ones(et_shape, dtype=torch.float64) * epsilon_r,
    )


def _receiver_loss(
    sigma_value: float,
    epsilon_r_value: float,
    *,
    compile_step: bool = False,
    requires_grad: bool = False,
):
    sigma = torch.tensor(
        sigma_value, dtype=torch.float64, requires_grad=requires_grad
    )
    epsilon_r = torch.tensor(
        epsilon_r_value, dtype=torch.float64, requires_grad=requires_grad
    )
    source = GaussianCurrent(peak_current_a=1.0e6)
    simulation = GeodesicFDTD(
        _config(),
        source=source,
        device="cpu",
        dtype="float64",
        compile_step=compile_step,
        compile_chunk_size=2,
        material_tensors=_sampled_material(sigma, epsilon_r),
    )
    vertices, layers, _ = source.staggered_distribution(simulation)
    traces = simulation.record_er_observations(
        vertices[:1, None],
        layers[:1],
        np.ones((1, 1), dtype=np.float64),
        4,
    )
    return traces.square().sum() * 1.0e18, sigma, epsilon_r, traces


@pytest.mark.parametrize("loss_integration", ("exponential", "trapezoidal"))
def test_sampled_material_loss_has_finite_nonzero_gradients(
    loss_integration: str,
) -> None:
    config = _config(loss_integration=loss_integration)
    sigma = torch.tensor(1.0e-8, dtype=torch.float64, requires_grad=True)
    epsilon_r = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    material = _sampled_material(sigma, epsilon_r, config=config)
    source = GaussianCurrent(peak_current_a=1.0e6)
    simulation = GeodesicFDTD(
        config,
        source=source,
        device="cpu",
        dtype="float64",
        material_tensors=material,
    )
    vertices, layers, _ = source.staggered_distribution(simulation)
    traces = simulation.record_er_observations(
        vertices[:1, None], layers[:1], [[1.0]], 4
    )
    gradients = torch.autograd.grad(
        traces.square().sum() * 1.0e18, (sigma, epsilon_r)
    )

    for gradient in gradients:
        assert gradient is not None
        assert torch.isfinite(gradient)
        assert torch.count_nonzero(gradient) == 1


def test_sampled_material_gradients_match_central_finite_difference() -> None:
    loss, sigma, epsilon_r, _ = _receiver_loss(
        1.0e-8, 2.0, requires_grad=True
    )
    sigma_gradient, epsilon_gradient = torch.autograd.grad(
        loss, (sigma, epsilon_r)
    )
    sigma_step = 1.0e-11
    epsilon_step = 1.0e-4
    sigma_difference = (
        _receiver_loss(1.0e-8 + sigma_step, 2.0)[0]
        - _receiver_loss(1.0e-8 - sigma_step, 2.0)[0]
    ) / (2.0 * sigma_step)
    epsilon_difference = (
        _receiver_loss(1.0e-8, 2.0 + epsilon_step)[0]
        - _receiver_loss(1.0e-8, 2.0 - epsilon_step)[0]
    ) / (2.0 * epsilon_step)

    torch.testing.assert_close(
        sigma_gradient, sigma_difference, rtol=2.0e-8, atol=0.0
    )
    torch.testing.assert_close(
        epsilon_gradient, epsilon_difference, rtol=2.0e-8, atol=0.0
    )


@pytest.mark.parametrize("loss_integration", ("exponential", "trapezoidal"))
def test_sampled_tensor_forward_matches_static_material(
    loss_integration: str,
) -> None:
    config = _config(loss_integration=loss_integration)
    source = GaussianCurrent(peak_current_a=1.0e6)
    static = GeodesicFDTD(
        config,
        source=source,
        device="cpu",
        dtype="float64",
    )
    tensor = GeodesicFDTD(
        config,
        source=source,
        device="cpu",
        dtype="float64",
        material_tensors=SampledMaterialTensors(
            torch.asarray(static.sigma_er),
            torch.asarray(static.epsilon_r_er),
            torch.asarray(static.sigma_et),
            torch.asarray(static.epsilon_r_et),
        ),
    )

    for static_coefficient, tensor_coefficient in zip(
        (static._ca_er, static._cb_er, static._ca_et, static._cb_et),
        (tensor._ca_er, tensor._cb_er, tensor._ca_et, tensor._cb_et),
        strict=True,
    ):
        torch.testing.assert_close(
            tensor_coefficient,
            static_coefficient,
            rtol=2.0e-15,
            atol=0.0,
        )
    static.step(4)
    tensor.step(4)
    for static_field, tensor_field in zip(
        (static.er, static.et, static.hr, static.ht),
        (tensor.er, tensor.et, tensor.hr, tensor.ht),
        strict=True,
    ):
        torch.testing.assert_close(
            tensor_field, static_field, rtol=2.0e-15, atol=1.0e-24
        )


def test_zero_conductivity_has_a_finite_exponential_gradient() -> None:
    loss, sigma, _, _ = _receiver_loss(0.0, 2.0, requires_grad=True)

    gradient = torch.autograd.grad(loss, sigma)[0]

    assert torch.isfinite(gradient)
    assert torch.count_nonzero(gradient) == 1


def test_dtype_normalization_preserves_the_input_graph() -> None:
    sigma = torch.tensor(1.0e-8, dtype=torch.float32, requires_grad=True)
    epsilon_r = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    material = _sampled_material(sigma, epsilon_r)
    simulation = GeodesicFDTD(
        _config(),
        device="cpu",
        dtype="float64",
        material_tensors=material,
    )

    loss = simulation._ca_er.sum() + simulation._cb_et.sum()
    gradients = torch.autograd.grad(loss, (sigma, epsilon_r))

    assert simulation.material_tensors.sigma_er.dtype == torch.float64
    for gradient in gradients:
        assert gradient.dtype == torch.float32
        assert torch.isfinite(gradient)
        assert torch.count_nonzero(gradient) == 1


def test_sampled_material_eager_and_compiled_gradients_match() -> None:
    eager_loss, eager_sigma, eager_epsilon, eager_traces = _receiver_loss(
        1.0e-8, 2.0, requires_grad=True
    )
    compiled_loss, compiled_sigma, compiled_epsilon, compiled_traces = (
        _receiver_loss(1.0e-8, 2.0, compile_step=True, requires_grad=True)
    )
    eager_gradients = torch.autograd.grad(
        eager_loss, (eager_sigma, eager_epsilon)
    )
    compiled_gradients = torch.autograd.grad(
        compiled_loss, (compiled_sigma, compiled_epsilon)
    )

    torch.testing.assert_close(
        compiled_traces, eager_traces, rtol=2.0e-12, atol=1.0e-18
    )
    for eager, compiled in zip(
        eager_gradients, compiled_gradients, strict=True
    ):
        torch.testing.assert_close(compiled, eager, rtol=1.0e-7, atol=1.0e-12)


def test_direct_update_coefficients_retain_gradients() -> None:
    template = GeodesicFDTD(
        _config(), device="cpu", dtype="float64"
    )
    inputs = tuple(
        value.detach().clone().requires_grad_()
        for value in (
            template._ca_er,
            template._cb_er,
            template._ca_et,
            template._cb_et,
        )
    )
    material = MaterialUpdateCoefficientTensors(*inputs)
    source = GaussianCurrent(peak_current_a=1.0e6)
    simulation = GeodesicFDTD(
        _config(),
        source=source,
        device="cpu",
        dtype="float64",
        material_tensors=material,
    )
    vertices, layers, _ = source.staggered_distribution(simulation)
    traces = simulation.record_er_observations(
        vertices[:1, None], layers[:1], [[1.0]], 4
    )
    gradients = torch.autograd.grad(traces.square().sum(), inputs)

    for supplied, retained, gradient in zip(
        inputs,
        (
            simulation._ca_er,
            simulation._cb_er,
            simulation._ca_et,
            simulation._cb_et,
        ),
        gradients,
        strict=True,
    ):
        assert retained is supplied
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_sampled_material_tensors_never_enter_numpy(monkeypatch) -> None:
    material = _sampled_material(1.0e-8, 2.0)
    original_asarray = solver_module.np.asarray

    def reject_tensor(values, *args, **kwargs):
        if torch.is_tensor(values):
            raise AssertionError("material tensor entered NumPy")
        return original_asarray(values, *args, **kwargs)

    monkeypatch.setattr(solver_module.np, "asarray", reject_tensor)
    simulation = GeodesicFDTD(
        _config(),
        device="cpu",
        dtype="float64",
        material_tensors=material,
    )

    assert simulation.material_tensors.sigma_er is material.sigma_er
    assert simulation.material_tensors.epsilon_r_er is material.epsilon_r_er
    assert simulation.material_tensors.sigma_et is material.sigma_et
    assert simulation.material_tensors.epsilon_r_et is material.epsilon_r_et


def test_uniform_compression_requires_broadcast_row_shapes() -> None:
    config = _config(compress_uniform_material_coefficients=True)
    material = _sampled_material(1.0e-8, 2.0, config=config)

    simulation = GeodesicFDTD(
        config,
        device="cpu",
        dtype="float64",
        material_tensors=material,
    )

    assert simulation._ca_er.shape == material.sigma_er.shape
    assert simulation._ca_et.shape == material.sigma_et.shape


@pytest.mark.parametrize(
    ("replacement", "error", "message"),
    (
        ({"sigma_er": np.zeros((12, 3))}, TypeError, "PyTorch tensor"),
        ({"sigma_er": torch.zeros((1, 1))}, ValueError, "shape"),
        (
            {"sigma_er": torch.zeros((12, 3), dtype=torch.int64)},
            TypeError,
            "floating-point dtype",
        ),
        ({"sigma_er": torch.full((12, 3), torch.nan)}, ValueError, "finite"),
        ({"sigma_er": torch.full((12, 3), -1.0)}, ValueError, "at least"),
        (
            {"epsilon_r_er": torch.zeros((12, 3))},
            ValueError,
            "greater than",
        ),
    ),
)
def test_invalid_sampled_material_tensors_fail_clearly(
    replacement: dict, error: type[Exception], message: str
) -> None:
    material = replace(_sampled_material(1.0e-8, 2.0), **replacement)

    with pytest.raises(error, match=message):
        GeodesicFDTD(
            _config(),
            device="cpu",
            dtype="float64",
            material_tensors=material,
        )


def test_material_tensors_use_default_runtime_and_validate_container() -> None:
    material = _sampled_material(1.0e-8, 2.0)
    simulation = GeodesicFDTD(_config(), material_tensors=material)
    assert simulation.er.dtype == torch.float64
    with pytest.raises(TypeError, match="SampledMaterialTensors"):
        GeodesicFDTD(
            _config(),
            device="cpu",
            dtype="float64",
            material_tensors=object(),
        )


def test_static_material_sampler_rejects_torch_outputs() -> None:
    class TensorSampler:
        def sample(self, directions, altitudes_m, earth_radius_m):
            del earth_radius_m
            shape = (len(directions), len(altitudes_m))
            return torch.zeros(shape), torch.ones(shape)

    with pytest.raises(TypeError, match="material_tensors"):
        GeodesicFDTD(_config(), material=TensorSampler())
