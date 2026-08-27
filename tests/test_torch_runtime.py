import warnings

import numpy as np
import pytest
import torch

from ionosphere_fdtd._torch_runtime import _TorchRuntime
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig


def test_existing_tensor_normalization_preserves_graph() -> None:
    runtime = _TorchRuntime(device="cpu", dtype="float64")
    leaf = torch.asarray([1.0, 2.0], dtype=torch.float64).requires_grad_()
    derived = leaf.square()

    normalized = runtime.as_tensor(derived)
    normalized.sum().backward()

    assert normalized is derived
    assert normalized.grad_fn is derived.grad_fn
    torch.testing.assert_close(leaf.grad, 2.0 * leaf.detach())


def test_existing_tensor_cast_preserves_gradient_history() -> None:
    runtime = _TorchRuntime(device="cpu", dtype="float64")
    leaf = torch.asarray([1.0, 2.0], dtype=torch.float32).requires_grad_()

    normalized = runtime.as_tensor(3.0 * leaf)
    normalized.sum().backward()

    assert normalized.dtype is torch.float64
    assert normalized.grad_fn is not None
    torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 3.0))


def test_host_and_index_inputs_inherit_the_runtime_context() -> None:
    runtime = _TorchRuntime(device="cpu", dtype="float64")

    values = runtime.as_tensor(np.asarray([1.0, 2.0], dtype=np.float32))
    existing_indices = torch.asarray([0, 1], dtype=torch.long)
    indices = runtime.index_tensor(existing_indices)

    assert values.device == runtime.device
    assert values.dtype is runtime.dtype
    assert indices is existing_indices
    assert indices.device == runtime.device
    assert indices.dtype is torch.long


def test_read_only_numpy_inputs_are_normalized_without_alias_warnings() -> None:
    runtime = _TorchRuntime(device="cpu", dtype="float64")
    values = np.asarray([1.0, 2.0], dtype=np.float64)
    indices = np.asarray([0, 1], dtype=np.int64)
    values.setflags(write=False)
    indices.setflags(write=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        normalized_values = runtime.as_tensor(values)
        normalized_indices = runtime.index_tensor(indices)

    torch.testing.assert_close(
        normalized_values, torch.asarray([1.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(normalized_indices, torch.asarray([0, 1]))


def test_numpy_export_is_an_explicit_terminal_detach_boundary() -> None:
    runtime = _TorchRuntime(device="cpu", dtype="float64")
    leaf = torch.asarray([1.0, 2.0], dtype=torch.float64).requires_grad_()
    derived = 4.0 * leaf

    exported = runtime.export_numpy(derived)

    assert isinstance(exported, np.ndarray)
    np.testing.assert_array_equal(exported, np.asarray([4.0, 8.0]))
    assert derived.grad_fn is not None


def test_simulation_exposes_canonical_torch_context() -> None:
    simulation = GeodesicFDTD(
        config=SimulationConfig(
            subdivision=0, radial_cells=4, courant_factor=0.2
        ),
        device="cpu",
        dtype="float64",
    )

    assert simulation.device == torch.device("cpu")
    assert simulation.dtype is torch.float64
    assert simulation.threads == torch.get_num_threads()
    for values in (
        simulation.er,
        simulation.et,
        simulation.hr,
        simulation.ht,
        simulation._runtime.face_edge_signs,
    ):
        assert values.device == simulation.device
        assert values.dtype is simulation.dtype
    for indices in (
        simulation._runtime.edges,
        simulation._runtime.face_edges,
        simulation._runtime.edge_left_faces,
        simulation._runtime.edge_right_faces,
    ):
        assert indices.device == simulation.device
        assert indices.dtype is torch.long


def test_simulation_auto_device_matches_allocated_fields() -> None:
    simulation = GeodesicFDTD(
        config=SimulationConfig(
            subdivision=0, radial_cells=4, courant_factor=0.2
        ),
        device="auto",
    )

    assert simulation.device == simulation.er.device
    assert simulation.dtype is simulation.er.dtype
    assert simulation.device.type in {"cpu", "cuda", "mps"}
