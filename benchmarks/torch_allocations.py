"""Inventory eager PyTorch field-step allocations with the native profiler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np

from ionosphere_fdtd.data_artifacts import DatasetProvenance, VariableProvenance
from ionosphere_fdtd.mesh import build_geodesic_mesh
from ionosphere_fdtd.plasma import (
    ColdPlasmaSpecies,
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    MeshPlasmaModel,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.surface_impedance import ConductiveHalfSpaceSurface


def profile_allocations(
    *,
    subdivision: int,
    radial_cells: int,
    dtype: str,
    device: str,
    steps: int,
    warmup_steps: int,
    physics: str = "vacuum",
    trace_path: Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready allocation inventory for one static solver shape."""

    import torch

    simulation = _simulation(
        subdivision=subdivision,
        radial_cells=radial_cells,
        dtype=dtype,
        device=device,
        physics=physics,
    )
    simulation.step(warmup_steps)
    simulation._runtime.synchronize()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if simulation.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(simulation.device)
    before_device_bytes = _device_memory_allocated(simulation)
    with torch.profiler.profile(
        activities=activities,
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        with torch.profiler.record_function("GeodesicFDTD.field_steps"):
            simulation.step(steps)
        simulation._runtime.synchronize()
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))

    operators = []
    for event in profiler.key_averages(group_by_input_shape=True):
        cpu_bytes = int(getattr(event, "self_cpu_memory_usage", 0))
        device_bytes = int(
            getattr(
                event,
                "self_device_memory_usage",
                getattr(event, "self_cuda_memory_usage", 0),
            )
        )
        if cpu_bytes <= 0 and device_bytes <= 0:
            continue
        operators.append(
            {
                "operator": event.key,
                "calls": int(event.count),
                "self_cpu_allocation_bytes": cpu_bytes,
                "self_device_allocation_bytes": device_bytes,
                "input_shapes": event.input_shapes,
            }
        )
    operators.sort(
        key=lambda item: (
            item["self_device_allocation_bytes"],
            item["self_cpu_allocation_bytes"],
        ),
        reverse=True,
    )
    diagnostics = simulation.diagnostics()
    return {
        "system": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(simulation.device),
            "device_name": (
                torch.cuda.get_device_name(simulation.device)
                if simulation.device.type == "cuda"
                else None
            ),
        },
        "configuration": {
            "subdivision": subdivision,
            "radial_cells": radial_cells,
            "dtype": dtype,
            "steps": steps,
            "warmup_steps": warmup_steps,
            "physics": physics,
        },
        "profiled_regions": [
            "GeodesicFDTD._update_magnetic_fields",
            "GeodesicFDTD._update_electric_fields",
            "GeodesicFDTD._radial_derivative_et",
            "GeodesicFDTD._radial_derivative_ht",
            "_TorchRuntime.face_circulation",
            "_TorchRuntime.dual_cell_circulation",
            *(
                ("_torch_step.advance_surface_impedance",)
                if physics == "surface-impedance"
                else ()
            ),
            *(
                ("_torch_step.advance_plasma",)
                if physics == "plasma"
                else ()
            ),
        ],
        "field_memory_bytes": diagnostics["field_memory_bytes"],
        "persistent_runtime_bytes": diagnostics["persistent_runtime_bytes"],
        "device_memory_before_profile_bytes": before_device_bytes,
        "peak_device_memory_bytes": _peak_device_memory(simulation),
        "allocation_operators": operators,
    }


def _simulation(
    *,
    subdivision: int,
    radial_cells: int,
    dtype: str,
    device: str,
    physics: str,
) -> GeodesicFDTD:
    config = SimulationConfig(
        subdivision=subdivision,
        radial_cells=radial_cells,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.35,
        radial_boundary_condition=(
            "surface-impedance" if physics == "surface-impedance" else "pec"
        ),
    )
    mesh = build_geodesic_mesh(subdivision) if physics == "plasma" else None
    surface = (
        ConductiveHalfSpaceSurface(0.02)
        if physics == "surface-impedance"
        else None
    )
    plasma = _synthetic_plasma(mesh, config) if mesh is not None else None
    return GeodesicFDTD(
        config,
        mesh=mesh,
        surface_impedance=surface,
        plasma=plasma,
        device=device,
        dtype=dtype,
    )


def _synthetic_plasma(mesh: Any, config: SimulationConfig) -> MeshPlasmaModel:
    altitudes = np.linspace(
        config.minimum_altitude_m,
        config.maximum_altitude_m,
        config.radial_cells + 1,
    )
    midpoints = 0.5 * (altitudes[:-1] + altitudes[1:])
    shape = (mesh.n_faces, len(midpoints))
    magnetic = np.zeros((*shape, 3))
    magnetic[..., 2] = 50.0e-6
    species = ColdPlasmaSpecies(
        "electron",
        -ELEMENTARY_CHARGE_C,
        ELECTRON_MASS_KG,
        np.full(shape, 1.0e3),
        np.full(shape, 200.0),
    )
    provenance = DatasetProvenance(
        dataset_id="benchmark.synthetic-plasma.v1",
        title="Synthetic allocation-profiler plasma",
        version="1",
        source_url="https://example.invalid/synthetic-plasma",
        citation="Synthetic benchmark fixture.",
        license="CC0-1.0",
        retrieved_at="2026-08-22T00:00:00Z",
        source_sha256="0" * 64,
        coordinate_reference_system="geocentric Cartesian",
        variables=(
            VariableProvenance("B", "T", "T", "identity"),
            VariableProvenance("number_density", "m^-3", "m^-3", "identity"),
            VariableProvenance("collision_frequency", "Hz", "Hz", "identity"),
        ),
    )
    return MeshPlasmaModel.from_mesh(
        mesh,
        midpoints,
        magnetic,
        (species,),
        provenance=(provenance,),
        interpolation="synthetic constants at face/cell centers",
    )


def _device_memory_allocated(simulation: GeodesicFDTD) -> int | None:
    if not simulation.device.type == "cuda":
        return None
    return int(
        simulation._runtime.torch.cuda.memory_allocated(
            simulation.device
        )
    )


def _peak_device_memory(simulation: GeodesicFDTD) -> int | None:
    if not simulation.device.type == "cuda":
        return None
    return int(
        simulation._runtime.torch.cuda.max_memory_allocated(
            simulation.device
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subdivision", type=int, default=2)
    parser.add_argument("--radial-cells", type=int, default=16)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument(
        "--physics",
        choices=("vacuum", "surface-impedance", "plasma"),
        default="vacuum",
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup_steps < 0:
        parser.error("steps must be positive and warm-up non-negative")
    payload = profile_allocations(
        subdivision=args.subdivision,
        radial_cells=args.radial_cells,
        dtype=args.dtype,
        device=args.device,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        physics=args.physics,
        trace_path=args.trace,
    )
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
