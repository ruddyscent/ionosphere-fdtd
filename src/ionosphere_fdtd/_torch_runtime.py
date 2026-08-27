"""Private PyTorch device, dtype, and host-boundary policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


class BackendUnavailableError(RuntimeError):
    """Raised when a requested PyTorch device cannot be used."""


class _TorchRuntime:
    """Own the canonical tensor context without exposing a backend API."""

    name = "torch"

    def __init__(
        self,
        mesh: Any | None = None,
        *,
        device: str = "cpu",
        dtype: str = "float64",
        threads: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise BackendUnavailableError(
                "install project dependencies with: uv sync"
            ) from error

        self.torch = torch
        self.device = self._resolve_device(device)
        if dtype == "auto":
            dtype = "float32"
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'auto', 'float32', or 'float64'")
        if self.device.type == "mps" and dtype == "float64":
            raise BackendUnavailableError(
                "the MPS runtime does not support float64; use dtype='float32'"
            )
        if threads is not None:
            if (
                isinstance(threads, bool)
                or not isinstance(threads, (int, np.integer))
                or threads < 1
            ):
                raise ValueError("torch_threads must be a positive integer")
            if self.device.type != "cpu":
                raise BackendUnavailableError(
                    "torch_threads is only valid for the PyTorch CPU runtime"
                )
            torch.set_num_threads(int(threads))
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.dtype_name = dtype
        if mesh is not None:
            self._prepare_mesh(mesh)

    def _prepare_mesh(self, mesh: Any) -> None:
        """Move immutable mesh stencils into the canonical tensor context."""

        self.edges = self.index_tensor(mesh.edges)
        self.face_edges = self.index_tensor(mesh.face_edges)
        self.face_edge_signs = self.as_tensor(mesh.face_edge_signs)
        self.edge_left_faces = self.index_tensor(mesh.edge_left_faces)
        self.edge_right_faces = self.index_tensor(mesh.edge_right_faces)
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
        return self.index_tensor(vertex_edges), self.as_tensor(vertex_signs)

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        requested = requested.lower()
        if requested == "gpu":
            requested = "cuda"
        if requested == "auto":
            if torch.cuda.is_available():
                requested = "cuda"
            elif torch.backends.mps.is_available():
                requested = "mps"
            else:
                requested = "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            reason = (
                "PyTorch was not built with MPS support"
                if not torch.backends.mps.is_built()
                else "MPS is unavailable on this macOS device"
            )
            raise BackendUnavailableError(reason)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise BackendUnavailableError("CUDA is unavailable in this PyTorch runtime")
        try:
            device = torch.device(requested)
        except (RuntimeError, ValueError) as error:
            raise BackendUnavailableError(
                f"unsupported PyTorch device: {requested}"
            ) from error
        if device.type not in {"cpu", "mps", "cuda"}:
            raise BackendUnavailableError(
                "PyTorch device must be cpu, mps, cuda, cuda:N, or gpu"
            )
        if (
            device.type == "cuda"
            and device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise BackendUnavailableError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} device(s)"
            )
        if device.type == "cuda" and device.index is None:
            # An indexless CUDA device follows process-global current-device
            # changes. Resolve it once so every tensor remains on one device.
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    @property
    def threads(self) -> int | None:
        """Return the current process-wide CPU thread count."""

        return self.torch.get_num_threads() if self.device.type == "cpu" else None

    def as_tensor(self, values: Any) -> Any:
        """Normalize floating values without severing an existing tensor graph."""

        if self.torch.is_tensor(values):
            return values.to(device=self.device, dtype=self.dtype)
        values = self._writable_host_input(values)
        return self.torch.as_tensor(
            values, dtype=self.dtype, device=self.device
        )

    def index_tensor(self, values: Any) -> Any:
        """Normalize indices without recreating an existing long tensor."""

        if self.torch.is_tensor(values):
            return values.to(device=self.device, dtype=self.torch.long)
        values = self._writable_host_input(values)
        return self.torch.as_tensor(
            values, dtype=self.torch.long, device=self.device
        )

    @staticmethod
    def _writable_host_input(values: Any) -> Any:
        """Copy only read-only NumPy inputs that PyTorch cannot safely alias."""

        if isinstance(values, np.ndarray) and not values.flags.writeable:
            return values.copy()
        return values

    def zeros(self, shape: tuple[int, ...]) -> Any:
        return self.torch.zeros(shape, dtype=self.dtype, device=self.device)

    def empty_like(self, values: Any) -> Any:
        return self.torch.empty_like(values)

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

    def compile(
        self,
        function: Callable[..., Any],
        *,
        backend: str | None = None,
    ) -> Callable[..., Any]:
        """Compile a static-shape tensor function with the selected backend."""

        if backend is None:
            return self.torch.compile(
                function, fullgraph=True, dynamic=False
            )
        return self.torch.compile(
            function,
            backend=backend,
            fullgraph=True,
            dynamic=False,
        )

    def synchronize(self) -> None:
        """Wait for queued work on asynchronous accelerator devices."""

        if self.device.type == "mps":
            self.torch.mps.synchronize()
        elif self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def export_numpy(self, values: Any) -> np.ndarray:
        """Detach values and export them to a host NumPy terminal boundary."""

        if not self.torch.is_tensor(values):
            return np.asarray(values)
        return values.detach().cpu().numpy()

    def export_scalar(self, value: Any) -> float:
        """Detach one tensor value at a scalar diagnostics boundary."""

        return float(value.detach().item())

    def export_max_abs(self, values: Any) -> float:
        """Detach a maximum reduction at a scalar diagnostics boundary."""

        return float(self.torch.max(self.torch.abs(values)).detach().item())

    @staticmethod
    def nbytes(values: Any) -> int:
        return int(values.numel() * values.element_size())
