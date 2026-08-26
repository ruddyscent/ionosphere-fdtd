"""Passive diffusive surface-impedance models and ADE state updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .constants import MU_0


@dataclass(frozen=True, slots=True)
class SurfaceImpedanceCoefficientTensors:
    """Tensor-native coefficients for the lower-boundary ADE."""

    decay: Any
    drive: Any
    history_weights: Any
    scale: Any


@dataclass(frozen=True, slots=True)
class ConductiveHalfSpaceSurface:
    r"""Positive-real approximation of a conductive half-space impedance.

    The target is :math:`Z(s)=\sqrt{\mu s/\sigma}`. The diffusive identity

    .. math::

       \sqrt{s}=\frac{1}{\pi}\int_0^\infty
       \frac{s}{s+p}p^{-1/2}\,dp

    is integrated on a logarithmic pole grid. Every residue and pole is
    positive, so truncation preserves causality and passivity.
    """

    conductivity_s_m: float | NDArray[np.float64]
    minimum_frequency_hz: float = 5.0
    maximum_frequency_hz: float = 45.0
    terms: int = 16
    pole_span_decades: float = 5.0
    permeability_h_m: float = MU_0

    def __post_init__(self) -> None:
        conductivity = np.asarray(self.conductivity_s_m, dtype=np.float64)
        if conductivity.ndim > 1 or conductivity.size == 0:
            raise ValueError("surface conductivity must be scalar or one-dimensional")
        conductivity = conductivity.reshape(-1)
        if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
            raise ValueError("surface conductivity must be finite and positive")
        values = (
            self.minimum_frequency_hz,
            self.maximum_frequency_hz,
            self.pole_span_decades,
            self.permeability_h_m,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("surface impedance controls must be finite")
        if not 0.0 < self.minimum_frequency_hz < self.maximum_frequency_hz:
            raise ValueError("surface impedance frequency band is invalid")
        if (
            isinstance(self.terms, bool)
            or not isinstance(self.terms, (int, np.integer))
            or self.terms < 1
        ):
            raise ValueError("surface impedance terms must be a positive integer")
        if self.pole_span_decades <= 0.0 or self.permeability_h_m <= 0.0:
            raise ValueError("surface impedance span and permeability must be positive")
        conductivity = np.array(conductivity, copy=True)
        conductivity.setflags(write=False)
        object.__setattr__(self, "conductivity_s_m", conductivity)

    @property
    def pole_rates_s_inv(self) -> NDArray[np.float64]:
        """Return logarithmically spaced positive ADE pole rates."""

        lower = np.log(2.0 * np.pi * self.minimum_frequency_hz)
        upper = np.log(2.0 * np.pi * self.maximum_frequency_hz)
        extension = self.pole_span_decades * np.log(10.0)
        lower -= extension
        upper += extension
        spacing = (upper - lower) / self.terms
        return np.exp(lower + (np.arange(self.terms) + 0.5) * spacing)

    @property
    def diffusive_weights_sqrt_s_inv(self) -> NDArray[np.float64]:
        """Return positive quadrature weights for the square-root kernel."""

        poles = self.pole_rates_s_inv
        logarithmic_spacing = (
            np.log(poles[-1] / poles[0]) / (self.terms - 1)
            if self.terms > 1
            else (
                np.log(self.maximum_frequency_hz / self.minimum_frequency_hz)
                + 2.0 * self.pole_span_decades * np.log(10.0)
            )
        )
        return logarithmic_spacing * np.sqrt(poles) / np.pi

    def conductivity_for_edges(
        self,
        edge_count: int,
        edge_indices: NDArray[np.int64] | None = None,
    ) -> NDArray[np.float64]:
        """Broadcast or select conductivity for the requested edge ownership."""

        conductivity = np.asarray(self.conductivity_s_m)
        if edge_indices is not None:
            indices = np.asarray(edge_indices, dtype=np.int64)
            if conductivity.size == 1:
                return np.full(len(indices), conductivity[0])
            if conductivity.size != edge_count:
                raise ValueError("surface conductivity does not match mesh edges")
            return conductivity[indices]
        if conductivity.size == 1:
            return np.full(edge_count, conductivity[0])
        if conductivity.size != edge_count:
            raise ValueError("surface conductivity does not match mesh edges")
        return np.array(conductivity, copy=True)

    def impedance_ohm(
        self, frequency_hz: float | NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Evaluate the fitted positive-real impedance at real frequencies."""

        frequency = np.asarray(frequency_hz, dtype=np.float64)
        if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError("surface impedance frequencies must be positive")
        s = 2j * np.pi * frequency.reshape(-1)
        poles = self.pole_rates_s_inv[:, None]
        weights = self.diffusive_weights_sqrt_s_inv[:, None]
        root = np.sum(weights * s[None, :] / (s[None, :] + poles), axis=0)
        scale = np.sqrt(
            self.permeability_h_m / np.asarray(self.conductivity_s_m)
        )
        return scale[:, None] * root[None, :]

    def exact_impedance_ohm(
        self, frequency_hz: float | NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        """Evaluate the conductive half-space target used by the fit."""

        frequency = np.asarray(frequency_hz, dtype=np.float64)
        if not np.all(np.isfinite(frequency)) or np.any(frequency <= 0.0):
            raise ValueError("surface impedance frequencies must be positive")
        s = 2j * np.pi * frequency.reshape(-1)
        scale = np.sqrt(
            self.permeability_h_m / np.asarray(self.conductivity_s_m)
        )
        return scale[:, None] * np.sqrt(s)[None, :]

    def to_metadata(self) -> dict[str, Any]:
        """Return portable constructor metadata."""

        return {
            "type": "conductive-half-space-diffusive",
            "conductivity_s_m": np.asarray(self.conductivity_s_m).tolist(),
            "minimum_frequency_hz": self.minimum_frequency_hz,
            "maximum_frequency_hz": self.maximum_frequency_hz,
            "terms": int(self.terms),
            "pole_span_decades": self.pole_span_decades,
            "permeability_h_m": self.permeability_h_m,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> ConductiveHalfSpaceSurface:
        """Restore a model from :meth:`to_metadata` output."""

        if metadata.get("type") != "conductive-half-space-diffusive":
            raise ValueError("unsupported surface impedance model")
        return cls(
            conductivity_s_m=np.asarray(metadata["conductivity_s_m"]),
            minimum_frequency_hz=metadata["minimum_frequency_hz"],
            maximum_frequency_hz=metadata["maximum_frequency_hz"],
            terms=metadata["terms"],
            pole_span_decades=metadata["pole_span_decades"],
            permeability_h_m=metadata["permeability_h_m"],
        )


class SurfaceImpedanceADE:
    """Trapezoidal ADE state for a lower radial impedance boundary."""

    def __init__(
        self,
        model: ConductiveHalfSpaceSurface,
        *,
        edge_count: int,
        time_step_s: float,
        backend: Any,
        global_edge_count: int | None = None,
        global_edge_indices: NDArray[np.int64] | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        total_edges = edge_count if global_edge_count is None else global_edge_count
        conductivity = model.conductivity_for_edges(
            total_edges, global_edge_indices
        )
        if len(conductivity) != edge_count:
            raise ValueError("surface impedance local edge count is invalid")
        poles = model.pole_rates_s_inv
        weights = model.diffusive_weights_sqrt_s_inv
        alpha = 0.5 * time_step_s * poles
        self._decay = backend.asarray((1.0 - alpha) / (1.0 + alpha))
        self._drive = backend.asarray(alpha / (1.0 + alpha))
        self._history_weights = backend.asarray(weights / (1.0 + alpha))
        self._scale = backend.asarray(
            np.sqrt(model.permeability_h_m / conductivity)
        )
        self.memory = backend.zeros((edge_count, model.terms))

    def advance_prescribed_magnetic(
        self,
        h_old: Any,
        h_new: Any,
        *,
        rows: Any | None = None,
    ) -> Any:
        """Return boundary electric field and advance state for prescribed H."""

        memory = self.memory if rows is None else self.memory[rows]
        scale = self._scale if rows is None else self._scale[rows]
        h_average = 0.5 * (h_new + h_old)
        history = scale * (
            memory * self._history_weights[None, :]
        ).sum(axis=1)
        impedance_gain = scale * self._history_weights.sum()
        electric = -impedance_gain * h_average + history
        self._update_memory(memory, h_old, h_new, rows=rows)
        return electric

    def advance_lower_boundary(
        self,
        h_old: Any,
        electric_first_cell: Any,
        surface_gradient_er: Any,
        radial_step_m: Any,
        time_step_s: float,
        *,
        rows: Any | None = None,
    ) -> Any:
        """Implicitly couple the FDTD curl to the passive impedance state."""

        memory = self.memory if rows is None else self.memory[rows]
        scale = self._scale if rows is None else self._scale[rows]
        weighted = self._history_weights[None, :]
        history = scale * (memory * weighted).sum(axis=1)
        impedance_gain = scale * self._history_weights.sum()
        curl_scale = 2.0 * time_step_s / (
            MU_0 * radial_step_m
        )
        surface_forcing = time_step_s / MU_0 * surface_gradient_er
        coupling = curl_scale * impedance_gain
        h_new = (
            (1.0 - 0.5 * coupling) * h_old
            + surface_forcing
            - curl_scale * electric_first_cell
            + curl_scale * history
        ) / (1.0 + 0.5 * coupling)
        self._update_memory(memory, h_old, h_new, rows=rows)
        return h_new

    def _update_memory(
        self,
        memory: Any,
        h_old: Any,
        h_new: Any,
        *,
        rows: Any | None,
    ) -> None:
        updated_memory = memory * self._decay[None, :] + (
            h_new + h_old
        )[:, None] * self._drive[None, :]
        if rows is None:
            self.memory = updated_memory
        elif hasattr(self.memory, "index_copy"):
            self.memory = self.memory.index_copy(0, rows, updated_memory)
        else:
            next_memory = self.memory.copy()
            next_memory[rows] = updated_memory
            self.memory = next_memory

    @property
    def state_bytes(self) -> int:
        """Return persistent ADE state storage in bytes."""

        return self.backend.nbytes(self.memory)

    @property
    def persistent_bytes(self) -> int:
        """Return ADE state and coefficient storage in bytes."""

        arrays = (
            self.memory,
            self._decay,
            self._drive,
            self._history_weights,
            self._scale,
        )
        return sum(self.backend.nbytes(values) for values in arrays)
