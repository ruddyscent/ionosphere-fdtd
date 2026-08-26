"""Private PyTorch device, dtype, and host-boundary policy."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .backends.base import BackendUnavailableError


class _TorchRuntime:
    """Own the canonical tensor context without exposing a backend API."""

    def __init__(
        self,
        *,
        device: str = "auto",
        dtype: str = "auto",
        threads: int | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as error:
            raise BackendUnavailableError(
                "install the PyTorch backend with: uv sync --extra pytorch"
            ) from error

        self.torch = torch
        self.device = self._resolve_device(device)
        if dtype == "auto":
            dtype = "float32"
        if dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'auto', 'float32', or 'float64'")
        if self.device.type == "mps" and dtype == "float64":
            raise BackendUnavailableError(
                "the MPS backend does not support float64; use dtype='float32'"
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
                    "torch_threads is only valid for the PyTorch CPU backend"
                )
            torch.set_num_threads(int(threads))
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.dtype_name = dtype

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

    def compile(self, function: Callable[..., Any]) -> Callable[..., Any]:
        """Compile a static-shape tensor function with TorchInductor."""

        return self.torch.compile(function, fullgraph=True, dynamic=False)

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
