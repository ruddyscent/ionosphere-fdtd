"""Portable, versioned NPZ checkpoints for FDTD simulations."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .materials import EarthIonosphereMaterial, SphericalAnomaly
from .mesh import (
    build_geodesic_mesh_from_topology,
    build_geodesic_mesh_from_vertices,
)
from .plasma import MeshPlasmaModel
from .sources import GaussianCurrent, TangentialGaussianCurrent
from .surface_impedance import ConductiveHalfSpaceSurface

CHECKPOINT_FORMAT = "ionosphere-fdtd-checkpoint"
CHECKPOINT_VERSION = 4
SUPPORTED_CHECKPOINT_VERSIONS = (1, 2, 3, CHECKPOINT_VERSION)


class CheckpointError(ValueError):
    """Raised when a checkpoint is unsupported, corrupt, or inconsistent."""


def save_checkpoint(simulation: Any, path: str | Path) -> Path:
    """Atomically save a portable checkpoint and return its path.

    The NPZ file contains JSON metadata plus host NumPy copies of the mesh and
    four evolving fields and any surface-impedance ADE memory. It never stores
    pickled Python objects.
    """

    destination = Path(path)
    metadata = _metadata(simulation)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "metadata": np.asarray(
            json.dumps(metadata, default=_json_default, sort_keys=True)
        ),
        "mesh_vertices": np.asarray(simulation.mesh.vertices, dtype=np.float64),
        "mesh_faces": np.asarray(simulation.mesh.faces, dtype=np.int64),
        "mesh_face_levels": (
            np.asarray(simulation.mesh.face_levels, dtype=np.int64)
            if simulation.mesh.face_levels is not None
            else np.empty(0, dtype=np.int64)
        ),
        "er": simulation.to_numpy(simulation.er),
        "et": simulation.to_numpy(simulation.et),
        "hr": simulation.to_numpy(simulation.hr),
        "ht": simulation.to_numpy(simulation.ht),
        "surface_impedance_memory": (
            simulation.to_numpy(simulation._surface_impedance_ade.memory)
            if simulation._surface_impedance_ade is not None
            else np.empty((0, 0), dtype=np.float64)
        ),
    }
    if simulation.plasma is not None:
        _, plasma_arrays = simulation.plasma.archive_payload()
        arrays.update(
            {f"plasma_model_{name}": values for name, values in plasma_arrays.items()}
        )
        arrays.update(
            {
                f"plasma_current_{index}": simulation.to_numpy(values)
                for index, values in enumerate(
                    simulation._plasma_coupler.ade.current_density
                )
            }
        )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez_compressed(temporary, **arrays)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    dtype: str | None = None,
    compile_step: bool = False,
    compile_chunk_size: int = 8,
    torch_threads: int | None = None,
) -> Any:
    """Restore portable arrays on the selected PyTorch device and dtype."""

    from .solver import GeodesicFDTD, SimulationConfig

    source_path = Path(path)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            if "metadata" not in archive.files:
                raise CheckpointError("checkpoint is missing arrays: metadata")
            metadata = _read_metadata(archive["metadata"])
            required = {"metadata", "mesh_vertices", "er", "et", "hr", "ht"}
            if metadata["version"] >= 2:
                required.update(("mesh_faces", "mesh_face_levels"))
            if metadata["version"] >= 3:
                required.add("surface_impedance_memory")
            plasma_metadata = (
                metadata.get("plasma") if metadata["version"] >= 4 else None
            )
            if plasma_metadata is not None:
                required.update(
                    f"plasma_model_{name}"
                    for name in plasma_metadata["arrays"]
                )
                required.update(
                    f"plasma_current_{index}"
                    for index in range(len(plasma_metadata["species"]))
                )
            missing = required.difference(archive.files)
            if missing:
                raise CheckpointError(
                    f"checkpoint is missing arrays: {', '.join(sorted(missing))}"
                )
            vertices = np.array(archive["mesh_vertices"], dtype=np.float64, copy=True)
            faces = (
                np.array(archive["mesh_faces"], copy=True)
                if metadata["version"] >= 2
                else None
            )
            face_levels = (
                np.array(archive["mesh_face_levels"], copy=True)
                if metadata["version"] >= 2
                else None
            )
            fields = {
                name: np.array(archive[name], copy=True)
                for name in ("er", "et", "hr", "ht")
            }
            surface_impedance_memory = (
                np.array(archive["surface_impedance_memory"], copy=True)
                if metadata["version"] >= 3
                else np.empty((0, 0), dtype=np.float64)
            )
            plasma_arrays = (
                {
                    name: np.array(archive[f"plasma_model_{name}"], copy=True)
                    for name in plasma_metadata["arrays"]
                }
                if plasma_metadata is not None
                else {}
            )
            plasma_currents = (
                tuple(
                    np.array(archive[f"plasma_current_{index}"], copy=True)
                    for index in range(len(plasma_metadata["species"]))
                )
                if plasma_metadata is not None
                else ()
            )
    except CheckpointError:
        raise
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise CheckpointError(
            f"cannot read checkpoint {source_path}: {error}"
        ) from error

    try:
        config_values = dict(metadata["simulation_config"])
        if config_values["radial_altitudes_m"] is not None:
            config_values["radial_altitudes_m"] = tuple(
                config_values["radial_altitudes_m"]
            )
        config = SimulationConfig(**config_values)
        if metadata["version"] == 1:
            mesh = build_geodesic_mesh_from_vertices(
                config.subdivision,
                vertices,
                orientation=config.mesh_orientation,
                normalize_vertices=False,
            )
        else:
            mesh_metadata = metadata["mesh"]
            if not isinstance(mesh_metadata, dict):
                raise CheckpointError("checkpoint mesh metadata must be an object")
            has_face_levels = mesh_metadata["has_face_levels"]
            if not isinstance(has_face_levels, bool):
                raise CheckpointError("checkpoint has_face_levels must be boolean")
            assert faces is not None
            assert face_levels is not None
            if has_face_levels:
                restored_face_levels = face_levels
            else:
                if face_levels.shape != (0,):
                    raise CheckpointError(
                        "checkpoint has unexpected face-level values"
                    )
                restored_face_levels = None
            mesh = build_geodesic_mesh_from_topology(
                vertices,
                faces,
                subdivision=mesh_metadata["subdivision"],
                face_levels=restored_face_levels,
                refinement_spec=mesh_metadata["refinement_spec"],
                topology_kind=mesh_metadata["topology_kind"],
                normalize_vertices=False,
                require_well_centered=False,
            )
        material = _deserialize_material(metadata["material"])
        source = _deserialize_source(metadata["source"])
        surface_impedance = (
            _deserialize_surface_impedance(metadata["surface_impedance"])
            if metadata["version"] >= 3
            else None
        )
        plasma = (
            MeshPlasmaModel.from_archive_payload(
                plasma_metadata, plasma_arrays
            )
            if plasma_metadata is not None
            else None
        )
    except CheckpointError:
        raise
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise CheckpointError(f"invalid checkpoint model metadata: {error}") from error
    selected_dtype = dtype or str(metadata["runtime"]["dtype"])
    construction_config = replace(
        config, mesh_relaxations=0, mesh_optimization_steps=0
    )
    simulation = GeodesicFDTD(
        construction_config,
        material=material,
        source=source,
        surface_impedance=surface_impedance,
        plasma=plasma,
        mesh=mesh,
        device=device,
        dtype=selected_dtype,
        compile_step=compile_step,
        compile_chunk_size=compile_chunk_size,
        torch_threads=torch_threads,
    )
    simulation.config = config
    expected_shapes = {
        name: tuple(getattr(simulation, name).shape)
        for name in ("er", "et", "hr", "ht")
    }
    for name, values in fields.items():
        if values.shape != expected_shapes[name]:
            raise CheckpointError(
                f"checkpoint field {name} has shape {values.shape}, "
                f"expected {expected_shapes[name]}"
            )
        if not np.issubdtype(values.dtype, np.floating):
            raise CheckpointError(f"checkpoint field {name} must be floating point")
        if not np.all(np.isfinite(values)):
            raise CheckpointError(f"checkpoint field {name} contains non-finite values")
        setattr(simulation, name, simulation._runtime.as_tensor(values))
    if simulation._surface_impedance_ade is None:
        if surface_impedance_memory.shape != (0, 0):
            raise CheckpointError(
                "checkpoint has ADE state without a surface impedance model"
            )
    else:
        expected_ade_shape = tuple(simulation._surface_impedance_ade.memory.shape)
        if surface_impedance_memory.shape != expected_ade_shape:
            raise CheckpointError(
                "checkpoint surface impedance state has shape "
                f"{surface_impedance_memory.shape}, expected {expected_ade_shape}"
            )
        if (
            not np.issubdtype(surface_impedance_memory.dtype, np.floating)
            or not np.all(np.isfinite(surface_impedance_memory))
        ):
            raise CheckpointError("checkpoint surface impedance state is invalid")
        simulation._surface_impedance_ade.memory = simulation._runtime.as_tensor(
            surface_impedance_memory
        )
    if simulation._plasma_coupler is not None:
        expected_currents = simulation._plasma_coupler.ade.current_density
        if len(plasma_currents) != len(expected_currents):
            raise CheckpointError("checkpoint plasma species state count is invalid")
        for index, (saved, expected) in enumerate(
            zip(plasma_currents, expected_currents, strict=True)
        ):
            if saved.shape != tuple(expected.shape):
                raise CheckpointError(
                    f"checkpoint plasma current {index} has shape {saved.shape}, "
                    f"expected {tuple(expected.shape)}"
                )
            if not np.issubdtype(saved.dtype, np.floating) or not np.all(
                np.isfinite(saved)
            ):
                raise CheckpointError(
                    f"checkpoint plasma current {index} is invalid"
                )
            expected_currents[index] = simulation._runtime.as_tensor(saved)

    steps = metadata["state"]["steps"]
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise CheckpointError("checkpoint step count must be a non-negative integer")
    saved_time_step = float(metadata["state"]["time_step_s"])
    if not np.isclose(
        saved_time_step, simulation.time_step_s, rtol=0.0, atol=1.0e-15
    ):
        raise CheckpointError(
            "checkpoint time step is inconsistent with the reconstructed model"
        )
    saved_time = float(metadata["state"]["time_s"])
    expected_time = steps * saved_time_step
    if not np.isfinite(saved_time) or not np.isclose(
        saved_time, expected_time, rtol=1.0e-12, atol=1.0e-15
    ):
        raise CheckpointError("checkpoint time is inconsistent with its step count")
    simulation.steps = steps
    simulation.time_s = expected_time
    return simulation


def _metadata(simulation: Any) -> dict[str, Any]:
    if not isinstance(simulation.material, EarthIonosphereMaterial):
        raise CheckpointError(
            "checkpoints currently support EarthIonosphereMaterial only"
        )
    if simulation.source is not None and not isinstance(
        simulation.source, (GaussianCurrent, TangentialGaussianCurrent)
    ):
        raise CheckpointError(
            "checkpoints currently support GaussianCurrent and "
            "TangentialGaussianCurrent sources only"
        )
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "simulation_config": asdict(simulation.config),
        "mesh": {
            "topology_kind": simulation.mesh.topology_kind,
            "subdivision": simulation.mesh.subdivision,
            "has_face_levels": simulation.mesh.face_levels is not None,
            "refinement_spec": simulation.mesh.refinement_spec,
        },
        "material": _serialize_material(simulation.material),
        "source": _serialize_source(simulation.source),
        "surface_impedance": (
            simulation.surface_impedance.to_metadata()
            if simulation.surface_impedance is not None
            else None
        ),
        "plasma": (
            simulation.plasma.archive_payload()[0]
            if simulation.plasma is not None
            else None
        ),
        "state": {
            "steps": simulation.steps,
            "time_s": simulation.time_s,
            "time_step_s": simulation.time_step_s,
        },
        "runtime": {
            "kind": simulation.runtime,
            "device": str(simulation.device),
            "dtype": simulation.dtype_name,
            "compiled": simulation.compiled,
            "compile_chunk_size": simulation.compile_chunk_size,
        },
    }


def _read_metadata(value: np.ndarray) -> dict[str, Any]:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise CheckpointError("checkpoint metadata must be a scalar string")
    metadata = json.loads(str(value.item()))
    if not isinstance(metadata, dict):
        raise CheckpointError("checkpoint metadata must be a JSON object")
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError("file is not an ionosphere-fdtd checkpoint")
    if metadata.get("version") not in SUPPORTED_CHECKPOINT_VERSIONS:
        raise CheckpointError(
            f"unsupported checkpoint version {metadata.get('version')!r}; "
            f"supported versions are {SUPPORTED_CHECKPOINT_VERSIONS}"
        )
    required = {"simulation_config", "material", "source", "state", "runtime"}
    if metadata["version"] >= 2:
        required.add("mesh")
    if metadata["version"] >= 3:
        required.add("surface_impedance")
    if metadata["version"] >= 4:
        required.add("plasma")
    missing = required.difference(metadata)
    if missing:
        raise CheckpointError(
            f"checkpoint metadata is missing: {', '.join(sorted(missing))}"
        )
    return metadata


def _serialize_material(material: EarthIonosphereMaterial) -> dict[str, Any]:
    values = asdict(material)
    values["anomalies"] = [asdict(anomaly) for anomaly in material.anomalies]
    return {"type": "EarthIonosphereMaterial", "parameters": values}


def _deserialize_material(data: Any) -> EarthIonosphereMaterial:
    if not isinstance(data, dict) or data.get("type") != "EarthIonosphereMaterial":
        raise CheckpointError("checkpoint material type is unsupported")
    values = dict(data.get("parameters", {}))
    try:
        values["anomalies"] = tuple(
            SphericalAnomaly(**item) for item in values.get("anomalies", ())
        )
        return EarthIonosphereMaterial(**values)
    except (TypeError, ValueError) as error:
        raise CheckpointError(f"invalid checkpoint material: {error}") from error


def _serialize_source(source: Any) -> dict[str, Any] | None:
    if source is None:
        return None
    return {"type": type(source).__name__, "parameters": asdict(source)}


def _deserialize_source(data: Any) -> GaussianCurrent | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise CheckpointError("checkpoint source metadata must be an object")
    source_types = {
        "GaussianCurrent": GaussianCurrent,
        "TangentialGaussianCurrent": TangentialGaussianCurrent,
    }
    try:
        source_type = source_types[data["type"]]
        values = dict(data.get("parameters", {}))
        for name in ("azimuths_deg", "line_lengths_m"):
            if name in values and values[name] is not None:
                values[name] = tuple(values[name])
        return source_type(**values)
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(f"invalid checkpoint source: {error}") from error


def _deserialize_surface_impedance(
    data: Any,
) -> ConductiveHalfSpaceSurface | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise CheckpointError(
            "checkpoint surface impedance metadata must be an object"
        )
    try:
        return ConductiveHalfSpaceSurface.from_metadata(data)
    except (KeyError, TypeError, ValueError) as error:
        raise CheckpointError(
            f"invalid checkpoint surface impedance: {error}"
        ) from error


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot encode {type(value).__name__} in checkpoint metadata")
