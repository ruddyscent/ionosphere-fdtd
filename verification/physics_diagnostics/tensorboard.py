"""Optional TensorBoard event writer for verification physics snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ionosphere_fdtd.solver import GeodesicFDTD

from .model import HorizontalRegion, PhysicsDiagnosticSampler, PhysicsSnapshot


class TensorBoardPhysicsRecorder:
    """Retain exact snapshots and mirror their scalar values to TensorBoard."""

    def __init__(
        self,
        simulation: GeodesicFDTD,
        log_dir: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
        horizontal_regions: Mapping[str, HorizontalRegion] | None = None,
    ) -> None:
        try:
            from tensorboardX import SummaryWriter
        except ImportError as error:
            raise ImportError(
                "TensorBoard diagnostics require the 'tensorboard' extra: "
                "python -m pip install '.[tensorboard]'"
            ) from error
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sampler = PhysicsDiagnosticSampler(
            simulation, horizontal_regions=horizontal_regions
        )
        self.snapshots: list[PhysicsSnapshot] = []
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.metadata = dict(metadata or {})
        self.metadata.update(
            {
                "runtime": simulation.runtime,
                "device": str(simulation.device),
                "dtype": simulation.dtype_name,
                "compiled": simulation.compiled,
                "subdivision": simulation.config.subdivision,
                "radial_cells": simulation.config.radial_cells,
                "time_step_s": simulation.time_step_s,
                "geometry_mode": simulation.config.geometry_mode,
                "loss_integration": simulation.config.loss_integration,
                "diagnostic_runtime_bytes": self.sampler.diagnostic_runtime_bytes,
            }
        )
        self.writer.add_text(
            "run/metadata",
            f"```json\n{json.dumps(self.metadata, indent=2, sort_keys=True)}\n```",
            0,
        )

    def record(
        self,
        receiver_values: Mapping[str, float],
        *,
        steps_per_second: float | None = None,
    ) -> PhysicsSnapshot:
        snapshot = self.sampler.sample(
            receiver_values, steps_per_second=steps_per_second
        )
        self.snapshots.append(snapshot)
        for tag, value in snapshot.scalars.items():
            self.writer.add_scalar(tag, value, snapshot.step)
        for label, value in snapshot.receiver_values.items():
            self.writer.add_scalar(f"receiver/{label}_v_m", value, snapshot.step)
        self.writer.flush()
        return snapshot

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> TensorBoardPhysicsRecorder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
