"""PyTorch implementation of the FDTD array backend."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .._torch_runtime import _TorchRuntime
from .base import Array, ArrayBackend


class TorchBackend(ArrayBackend):
    """Execute the FDTD update with PyTorch tensors on CPU, MPS, or CUDA."""

    name = "torch"

    def __init__(
        self,
        mesh: Any,
        *,
        device: str = "auto",
        dtype: str = "auto",
        threads: int | None = None,
    ):
        self._runtime = _TorchRuntime(
            device=device, dtype=dtype, threads=threads
        )
        self.torch = self._runtime.torch
        self.torch_device = self._runtime.device
        self.device = str(self.torch_device)
        self.dtype = self._runtime.dtype
        self.dtype_name = self._runtime.dtype_name
        self.edges = self.index_array(mesh.edges)
        self.face_edges = self.index_array(mesh.face_edges)
        self.face_edge_signs = self.asarray(mesh.face_edge_signs)
        self.edge_left_faces = self.index_array(mesh.edge_left_faces)
        self.edge_right_faces = self.index_array(mesh.edge_right_faces)
        self.n_vertices = mesh.n_vertices
        self.vertex_edges, self.vertex_edge_signs = self._vertex_incidence(mesh)

    def _vertex_incidence(self, mesh: Any) -> tuple[Any, Any]:
        """Build a deterministic padded degree-six dual incidence table."""

        edge_indices = np.arange(len(mesh.edges), dtype=np.int64)
        vertices = np.concatenate((mesh.edges[:, 0], mesh.edges[:, 1]))
        incident_edges = np.concatenate((edge_indices, edge_indices))
        incident_signs = np.concatenate(
            (np.ones(len(mesh.edges)), -np.ones(len(mesh.edges)))
        )
        order = np.argsort(vertices, kind="stable")
        vertices = vertices[order]
        incident_edges = incident_edges[order]
        incident_signs = incident_signs[order]
        counts = np.bincount(vertices, minlength=self.n_vertices)
        if not np.array_equal(counts, mesh.vertex_degree):
            raise RuntimeError("mesh vertex degree does not match edge incidence")
        offsets = np.cumsum(np.concatenate(([0], counts[:-1])))
        slots = np.arange(len(vertices)) - np.repeat(offsets, counts)
        maximum_degree = int(counts.max())
        vertex_edges = np.zeros((self.n_vertices, maximum_degree), dtype=np.int64)
        vertex_signs = np.zeros((self.n_vertices, maximum_degree))
        vertex_edges[vertices, slots] = incident_edges
        vertex_signs[vertices, slots] = incident_signs
        return self.index_array(vertex_edges), self.asarray(vertex_signs)

    def compile_step(
        self, step: Callable[[Array], None]
    ) -> Callable[[Array], None]:
        """Compile a static-shape field step with TorchInductor."""

        return self._runtime.compile(step)

    def synchronize(self) -> None:
        self._runtime.synchronize()

    def asarray(self, values: Any) -> Any:
        return self._runtime.as_tensor(values)

    def index_array(self, values: Any) -> Any:
        return self._runtime.index_tensor(values)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        return self._runtime.zeros(shape)

    def empty_like(self, values: Any) -> Any:
        return self._runtime.empty_like(values)

    def diff(self, values: Any, axis: int) -> Any:
        return self.torch.diff(values, dim=axis)

    def edge_difference(self, vertex_values: Any) -> Any:
        return vertex_values[self.edges[:, 1]] - vertex_values[self.edges[:, 0]]

    def dual_edge_difference(self, face_values: Any) -> Any:
        return face_values[self.edge_left_faces] - face_values[self.edge_right_faces]

    def face_circulation(self, edge_values: Any) -> Any:
        sign_shape = (self.face_edges.shape[0],) + (1,) * (
            edge_values.ndim - 1
        )
        result = edge_values[self.face_edges[:, 0]]
        result.mul_(self.face_edge_signs[:, 0].reshape(sign_shape))
        for corner in (1, 2):
            term = edge_values[self.face_edges[:, corner]]
            term.mul_(self.face_edge_signs[:, corner].reshape(sign_shape))
            result.add_(term)
        return result

    def dual_cell_circulation(self, edge_values: Any) -> Any:
        sign_shape = (self.n_vertices,) + (1,) * (edge_values.ndim - 1)
        result = edge_values[self.vertex_edges[:, 0]]
        result.mul_(self.vertex_edge_signs[:, 0].reshape(sign_shape))
        for slot in range(1, self.vertex_edges.shape[1]):
            term = edge_values[self.vertex_edges[:, slot]]
            term.mul_(self.vertex_edge_signs[:, slot].reshape(sign_shape))
            result.add_(term)
        return result

    @property
    def threads(self) -> int | None:
        """Return the current process-wide CPU thread count."""

        return self._runtime.threads

    def to_numpy(self, values: Array) -> np.ndarray:
        return self._runtime.export_numpy(values)

    def scalar(self, value: Array) -> float:
        return self._runtime.export_scalar(value)

    def max_abs(self, values: Any) -> float:
        return self._runtime.export_max_abs(values)

    def nbytes(self, values: Any) -> int:
        return self._runtime.nbytes(values)
