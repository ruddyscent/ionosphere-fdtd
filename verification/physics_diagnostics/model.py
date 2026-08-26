"""Backend-neutral field, energy, and conductive-loss diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ionosphere_fdtd.constants import EPSILON_0, MU_0
from ionosphere_fdtd.solver import GeodesicFDTD

from ..common.archive import save_npz_atomic

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HorizontalRegion:
    """Per-field horizontal support weights for one diagnostic region."""

    er_weights: FloatArray
    edge_weights: FloatArray
    hr_weights: FloatArray


@dataclass(frozen=True, slots=True)
class PhysicsSnapshot:
    """One read-only diagnostic sample at the current leapfrog state."""

    step: int
    electric_time_s: float
    magnetic_time_s: float
    scalars: dict[str, float]
    receiver_values: dict[str, float]
    radial_energy_er_j: FloatArray
    radial_energy_et_j: FloatArray
    radial_energy_hr_j: FloatArray
    radial_energy_ht_j: FloatArray
    radial_loss_er_w: FloatArray
    radial_loss_et_w: FloatArray


class PhysicsRecorder(Protocol):
    """Consumer used by chunked observation recording."""

    def record(
        self,
        receiver_values: Mapping[str, float],
        *,
        steps_per_second: float | None = None,
    ) -> PhysicsSnapshot:
        """Record the simulation's current state."""


class PhysicsDiagnosticSampler:
    """Compute discrete volume-weighted diagnostics on the active backend.

    Material arrays are copied to the backend once so checkpoint sampling does
    not transfer complete fields or coefficients to the host. Only scalars and
    radial profiles leave the accelerator.
    """

    def __init__(
        self,
        simulation: GeodesicFDTD,
        *,
        horizontal_regions: Mapping[str, HorizontalRegion] | None = None,
    ) -> None:
        self.simulation = simulation
        backend = simulation.backend
        mesh = simulation.mesh
        self._horizontal_er = backend.asarray(mesh.dual_cell_solid_angles[:, None])
        self._horizontal_hr = backend.asarray(mesh.face_solid_angles[:, None])
        self._horizontal_edge = backend.asarray(
            (mesh.primal_edge_angles * mesh.dual_edge_angles)[:, None]
        )
        self._radial_nodes = backend.asarray(
            simulation.radii_m**2 * simulation.radial_node_control_lengths_m
        )
        self._radial_cells = backend.asarray(
            simulation.radial_midpoints_m**2 * simulation.radial_steps_m
        )
        self._epsilon_er = backend.asarray(EPSILON_0 * simulation.epsilon_r_er)
        self._epsilon_et = backend.asarray(EPSILON_0 * simulation.epsilon_r_et)
        self._sigma_er = backend.asarray(simulation.sigma_er)
        self._sigma_et = backend.asarray(simulation.sigma_et)
        self._horizontal_regions: dict[str, tuple[Any, Any, Any]] = {}
        for name, region in (horizontal_regions or {}).items():
            if not name or "/" in name:
                raise ValueError("horizontal region names must be nonempty tags")
            er_weights = self._validated_region_weights(
                region.er_weights, mesh.n_vertices, f"{name} Er"
            )
            edge_weights = self._validated_region_weights(
                region.edge_weights, mesh.n_edges, f"{name} edge"
            )
            hr_weights = self._validated_region_weights(
                region.hr_weights, mesh.n_faces, f"{name} Hr"
            )
            self._horizontal_regions[name] = (
                self._horizontal_er * backend.asarray(er_weights[:, None]),
                self._horizontal_edge * backend.asarray(edge_weights[:, None]),
                self._horizontal_hr * backend.asarray(hr_weights[:, None]),
            )
        diagnostic_arrays = [
            self._horizontal_er,
            self._horizontal_hr,
            self._horizontal_edge,
            self._radial_nodes,
            self._radial_cells,
            self._epsilon_er,
            self._epsilon_et,
            self._sigma_er,
            self._sigma_et,
        ]
        diagnostic_arrays.extend(
            array
            for region in self._horizontal_regions.values()
            for array in region
        )
        self.diagnostic_backend_bytes = sum(
            backend.nbytes(values)
            for values in diagnostic_arrays
        )
        reference_height = getattr(
            simulation.material, "ionosphere_reference_height_m", 70_000.0
        )
        self.reference_height_m = float(reference_height)

    def sample(
        self,
        receiver_values: Mapping[str, float] | None = None,
        *,
        steps_per_second: float | None = None,
    ) -> PhysicsSnapshot:
        """Reduce the current fields without modifying the simulation state."""

        simulation = self.simulation
        er_energy = self._electric_energy_profile(
            simulation.er,
            self._epsilon_er,
            self._horizontal_er,
            self._radial_nodes,
        )
        et_energy = self._electric_energy_profile(
            simulation.et,
            self._epsilon_et,
            self._horizontal_edge,
            self._radial_cells,
        )
        hr_energy = self._magnetic_energy_profile(
            simulation.hr,
            self._horizontal_hr,
            self._radial_cells,
        )
        ht_energy = self._magnetic_energy_profile(
            simulation.ht,
            self._horizontal_edge,
            self._radial_nodes,
        )
        er_loss = self._conductive_loss_profile(
            simulation.er,
            self._sigma_er,
            self._horizontal_er,
            self._radial_nodes,
        )
        et_loss = self._conductive_loss_profile(
            simulation.et,
            self._sigma_et,
            self._horizontal_edge,
            self._radial_cells,
        )
        profiles = {
            "er": er_energy,
            "et": et_energy,
            "hr": hr_energy,
            "ht": ht_energy,
        }
        loss_profiles = {"er": er_loss, "et": et_loss}
        scalars: dict[str, float] = {}
        for name, field in (
            ("er", simulation.er),
            ("et", simulation.et),
            ("hr", simulation.hr),
            ("ht", simulation.ht),
        ):
            scalars[f"field_rms/{name}"] = self._rms(field)
            scalars[f"field_max_abs/{name}"] = simulation.backend.max_abs(field)
            scalars[f"field_finite/{name}"] = self._all_finite(field)
        for name, profile in profiles.items():
            scalars[f"energy/{name}_j"] = float(np.sum(profile))
        scalars["energy/electric_j"] = scalars["energy/er_j"] + scalars[
            "energy/et_j"
        ]
        scalars["energy/magnetic_j"] = scalars["energy/hr_j"] + scalars[
            "energy/ht_j"
        ]
        scalars["energy/total_staggered_j"] = (
            scalars["energy/electric_j"] + scalars["energy/magnetic_j"]
        )
        for name, profile in loss_profiles.items():
            scalars[f"conductive_loss/{name}_w"] = float(np.sum(profile))
        scalars["conductive_loss/total_w"] = (
            scalars["conductive_loss/er_w"]
            + scalars["conductive_loss/et_w"]
        )
        self._add_region_scalars(scalars, profiles, loss_profiles)
        self._add_horizontal_region_scalars(scalars)
        scalars.update(self._source_scalars())
        scalars["memory/diagnostic_backend_bytes"] = float(
            self.diagnostic_backend_bytes
        )
        scalars["time/electric_s"] = simulation.electric_time_s
        scalars["time/magnetic_s"] = simulation.magnetic_time_s
        scalars["performance/steps_per_second"] = float(
            steps_per_second if steps_per_second is not None else 0.0
        )
        scalars.update(self._accelerator_scalars())
        return PhysicsSnapshot(
            step=simulation.steps,
            electric_time_s=simulation.electric_time_s,
            magnetic_time_s=simulation.magnetic_time_s,
            scalars=scalars,
            receiver_values={
                str(label): float(value)
                for label, value in (receiver_values or {}).items()
            },
            radial_energy_er_j=er_energy,
            radial_energy_et_j=et_energy,
            radial_energy_hr_j=hr_energy,
            radial_energy_ht_j=ht_energy,
            radial_loss_er_w=er_loss,
            radial_loss_et_w=et_loss,
        )

    def _electric_energy_profile(
        self, field: Any, epsilon: Any, horizontal: Any, radial: Any
    ) -> FloatArray:
        values = 0.5 * (epsilon * field * field * horizontal).sum(axis=0)
        values *= radial
        return np.asarray(
            self.simulation.backend.to_numpy(values), dtype=np.float64
        )

    def _magnetic_energy_profile(
        self, field: Any, horizontal: Any, radial: Any
    ) -> FloatArray:
        values = 0.5 * MU_0 * (field * field * horizontal).sum(axis=0)
        values *= radial
        return np.asarray(
            self.simulation.backend.to_numpy(values), dtype=np.float64
        )

    def _conductive_loss_profile(
        self, field: Any, sigma: Any, horizontal: Any, radial: Any
    ) -> FloatArray:
        values = (sigma * field * field * horizontal).sum(axis=0)
        values *= radial
        return np.asarray(
            self.simulation.backend.to_numpy(values), dtype=np.float64
        )

    def _rms(self, field: Any) -> float:
        mean_square = (field * field).mean()
        return float(np.sqrt(self.simulation.backend.scalar(mean_square)))

    def _all_finite(self, field: Any) -> float:
        backend = self.simulation.backend
        if backend.name == "torch":
            finite = backend.torch.isfinite(field).all()
        else:
            finite = np.isfinite(field).all()
        return float(bool(backend.scalar(finite)))

    @staticmethod
    def _validated_region_weights(
        values: FloatArray, expected_count: int, label: str
    ) -> FloatArray:
        weights = np.asarray(values, dtype=np.float64)
        if weights.shape != (expected_count,):
            raise ValueError(
                f"{label} region weights must have shape ({expected_count},)"
            )
        if not np.all(np.isfinite(weights)) or np.any(
            (weights < 0.0) | (weights > 1.0)
        ):
            raise ValueError(f"{label} region weights must be finite in [0, 1]")
        return weights

    def _add_horizontal_region_scalars(
        self, scalars: dict[str, float]
    ) -> None:
        simulation = self.simulation
        node_altitudes = simulation.altitudes_m
        cell_altitudes = simulation.radial_midpoint_altitudes_m
        vertical_regions = {
            "earth": (node_altitudes < 0.0, cell_altitudes < 0.0),
            "atmosphere": (
                (node_altitudes >= 0.0)
                & (node_altitudes < self.reference_height_m),
                (cell_altitudes >= 0.0)
                & (cell_altitudes < self.reference_height_m),
            ),
            "ionosphere": (
                node_altitudes >= self.reference_height_m,
                cell_altitudes >= self.reference_height_m,
            ),
        }
        for name, (horizontal_er, horizontal_edge, horizontal_hr) in (
            self._horizontal_regions.items()
        ):
            energy = {
                "er": self._electric_energy_profile(
                    simulation.er,
                    self._epsilon_er,
                    horizontal_er,
                    self._radial_nodes,
                ),
                "et": self._electric_energy_profile(
                    simulation.et,
                    self._epsilon_et,
                    horizontal_edge,
                    self._radial_cells,
                ),
                "hr": self._magnetic_energy_profile(
                    simulation.hr,
                    horizontal_hr,
                    self._radial_cells,
                ),
                "ht": self._magnetic_energy_profile(
                    simulation.ht,
                    horizontal_edge,
                    self._radial_nodes,
                ),
            }
            loss = {
                "er": self._conductive_loss_profile(
                    simulation.er,
                    self._sigma_er,
                    horizontal_er,
                    self._radial_nodes,
                ),
                "et": self._conductive_loss_profile(
                    simulation.et,
                    self._sigma_et,
                    horizontal_edge,
                    self._radial_cells,
                ),
            }
            scalars[f"energy_horizontal_region/{name}_j"] = float(
                sum(np.sum(values) for values in energy.values())
            )
            scalars[f"conductive_loss_horizontal_region/{name}_w"] = float(
                np.sum(loss["er"]) + np.sum(loss["et"])
            )
            for vertical, (node_mask, cell_mask) in vertical_regions.items():
                scalars[
                    f"conductive_loss_horizontal_region/{name}/{vertical}_w"
                ] = float(
                    np.sum(loss["er"][node_mask])
                    + np.sum(loss["et"][cell_mask])
                )

    def _add_region_scalars(
        self,
        scalars: dict[str, float],
        energy: Mapping[str, FloatArray],
        loss: Mapping[str, FloatArray],
    ) -> None:
        simulation = self.simulation
        node_altitudes = simulation.altitudes_m
        cell_altitudes = simulation.radial_midpoint_altitudes_m
        regions = {
            "earth": (
                node_altitudes < 0.0,
                cell_altitudes < 0.0,
            ),
            "atmosphere": (
                (node_altitudes >= 0.0)
                & (node_altitudes < self.reference_height_m),
                (cell_altitudes >= 0.0)
                & (cell_altitudes < self.reference_height_m),
            ),
            "ionosphere": (
                node_altitudes >= self.reference_height_m,
                cell_altitudes >= self.reference_height_m,
            ),
        }
        for region, (node_mask, cell_mask) in regions.items():
            scalars[f"energy_region/{region}_j"] = float(
                np.sum(energy["er"][node_mask])
                + np.sum(energy["ht"][node_mask])
                + np.sum(energy["et"][cell_mask])
                + np.sum(energy["hr"][cell_mask])
            )
            scalars[f"conductive_loss_region/{region}_w"] = float(
                np.sum(loss["er"][node_mask])
                + np.sum(loss["et"][cell_mask])
            )

    def _source_scalars(self) -> dict[str, float]:
        simulation = self.simulation
        source = simulation.source
        if source is None:
            return {
                "source/current_previous_a": 0.0,
                "source/current_next_a": 0.0,
            }
        previous = (
            source.current_a(
                (simulation.steps - 0.5) * simulation.time_step_s,
                simulation.time_step_s,
            )
            if simulation.steps
            else 0.0
        )
        following = source.current_a(
            (simulation.steps + 0.5) * simulation.time_step_s,
            simulation.time_step_s,
        )
        result = {
            "source/current_previous_a": float(previous),
            "source/current_next_a": float(following),
        }
        length = getattr(source, "vertical_element_length_m", None)
        if length is not None:
            result["source/current_moment_previous_a_m"] = float(
                previous * length
            )
            result["source/current_moment_next_a_m"] = float(following * length)
        return result

    def _accelerator_scalars(self) -> dict[str, float]:
        backend = self.simulation.backend
        if backend.name != "torch" or backend.torch_device.type != "cuda":
            return {}
        torch = backend.torch
        device = backend.torch_device
        return {
            "memory/cuda_allocated_bytes": float(torch.cuda.memory_allocated(device)),
            "memory/cuda_reserved_bytes": float(torch.cuda.memory_reserved(device)),
            "memory/cuda_max_allocated_bytes": float(
                torch.cuda.max_memory_allocated(device)
            ),
        }


def record_er_observations_with_diagnostics(
    simulation: GeodesicFDTD,
    vertex_indices: NDArray[np.int64],
    radial_layers: NDArray[np.int64],
    weights: FloatArray,
    labels: Sequence[str],
    steps: int,
    *,
    diagnostics_every: int,
    recorder: PhysicsRecorder,
    synchronize_every: int = 128,
) -> FloatArray:
    """Record exact receiver traces while sampling diagnostics in chunks."""

    if diagnostics_every < 1:
        raise ValueError("diagnostics_every must be positive")
    if len(labels) != len(vertex_indices):
        raise ValueError("diagnostic labels must match observation rows")
    initial = simulation.to_numpy(
        simulation.record_er_observations(
            vertex_indices,
            radial_layers,
            weights,
            0,
            synchronize_every=synchronize_every,
        )
    ).astype(np.float64, copy=False)
    receiver = dict(zip(labels, initial[0], strict=True))
    recorder.record(receiver)
    blocks = [initial]
    remaining = steps
    while remaining:
        count = min(diagnostics_every, remaining)
        started = time.perf_counter()
        block = simulation.to_numpy(
            simulation.record_er_observations(
                vertex_indices,
                radial_layers,
                weights,
                count,
                synchronize_every=synchronize_every,
            )
        ).astype(np.float64, copy=False)
        elapsed = time.perf_counter() - started
        blocks.append(block[1:])
        receiver = dict(zip(labels, block[-1], strict=True))
        recorder.record(receiver, steps_per_second=count / elapsed)
        remaining -= count
    return np.concatenate(blocks, axis=0)


def save_physics_snapshots(
    path: str | Path,
    snapshots: Sequence[PhysicsSnapshot],
    *,
    node_altitudes_m: FloatArray,
    cell_altitudes_m: FloatArray,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save exact scalar and radial-profile diagnostic samples."""

    if not snapshots:
        raise ValueError("at least one physics snapshot is required")
    scalar_names = tuple(sorted(snapshots[0].scalars))
    receiver_labels = tuple(snapshots[0].receiver_values)
    if any(tuple(sorted(item.scalars)) != scalar_names for item in snapshots):
        raise ValueError("physics snapshots do not share scalar fields")
    if any(tuple(item.receiver_values) != receiver_labels for item in snapshots):
        raise ValueError("physics snapshots do not share receiver labels")
    return save_npz_atomic(
        path,
        steps=np.asarray([item.step for item in snapshots], dtype=np.int64),
        electric_time_s=np.asarray(
            [item.electric_time_s for item in snapshots], dtype=np.float64
        ),
        magnetic_time_s=np.asarray(
            [item.magnetic_time_s for item in snapshots], dtype=np.float64
        ),
        scalar_names=np.asarray(scalar_names),
        scalars=np.asarray(
            [[item.scalars[name] for name in scalar_names] for item in snapshots],
            dtype=np.float64,
        ),
        receiver_labels=np.asarray(receiver_labels),
        receiver_values=np.asarray(
            [
                [item.receiver_values[name] for name in receiver_labels]
                for item in snapshots
            ],
            dtype=np.float64,
        ),
        node_altitudes_m=np.asarray(node_altitudes_m, dtype=np.float64),
        cell_altitudes_m=np.asarray(cell_altitudes_m, dtype=np.float64),
        radial_energy_er_j=np.stack(
            [item.radial_energy_er_j for item in snapshots]
        ),
        radial_energy_et_j=np.stack(
            [item.radial_energy_et_j for item in snapshots]
        ),
        radial_energy_hr_j=np.stack(
            [item.radial_energy_hr_j for item in snapshots]
        ),
        radial_energy_ht_j=np.stack(
            [item.radial_energy_ht_j for item in snapshots]
        ),
        radial_loss_er_w=np.stack([item.radial_loss_er_w for item in snapshots]),
        radial_loss_et_w=np.stack([item.radial_loss_et_w for item in snapshots]),
        metadata=np.asarray(json.dumps(dict(metadata or {}), sort_keys=True)),
    )
