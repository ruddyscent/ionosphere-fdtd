"""Mesh-native magnetized cold-plasma tensors and current ADE coupling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .data_artifacts import (
    DatasetProvenance,
    VariableProvenance,
    array_sha256,
    mesh_faces_sha256,
    mesh_vertices_sha256,
)
from .mesh import GeodesicMesh


ELEMENTARY_CHARGE_C = 1.602_176_634e-19
ELECTRON_MASS_KG = 9.109_383_713_9e-31
MESH_PLASMA_FORMAT = "ionosphere-fdtd-mesh-plasma"
MESH_PLASMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PlasmaSpeciesCoefficientTensors:
    """Tensor-native exact-update coefficients for one plasma species."""

    decay: Any
    cosine: Any
    sine: Any
    drive_parallel: Any
    drive_real: Any
    drive_imag: Any


@dataclass(frozen=True, slots=True)
class PlasmaCoefficientTensors:
    """Tensor-native magnetic direction and per-species ADE coefficients."""

    magnetic_direction: Any
    species: tuple[PlasmaSpeciesCoefficientTensors, ...]


@dataclass(frozen=True, slots=True)
class ColdPlasmaSpecies:
    """One charged fluid sampled at mesh-face/radial-cell centers."""

    name: str
    charge_c: float
    mass_kg: float
    number_density_m3: NDArray[np.float64]
    collision_frequency_hz: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("plasma species name must be nonempty")
        if not np.isfinite(self.charge_c) or self.charge_c == 0.0:
            raise ValueError("plasma species charge must be finite and nonzero")
        if not np.isfinite(self.mass_kg) or self.mass_kg <= 0.0:
            raise ValueError("plasma species mass must be finite and positive")
        density = np.asarray(self.number_density_m3, dtype=np.float64)
        collision = np.asarray(self.collision_frequency_hz, dtype=np.float64)
        if density.ndim != 2 or collision.shape != density.shape:
            raise ValueError("plasma density and collision grids must have one shape")
        if (
            not np.all(np.isfinite(density))
            or np.any(density < 0.0)
            or not np.all(np.isfinite(collision))
            or np.any(collision <= 0.0)
        ):
            raise ValueError("plasma density/collision values are invalid")
        object.__setattr__(self, "number_density_m3", _readonly(density))
        object.__setattr__(self, "collision_frequency_hz", _readonly(collision))


@dataclass(frozen=True, slots=True)
class MeshPlasmaModel:
    """Magnetic field and charged fluids on one exact solver mesh/grid."""

    mesh_vertices_sha256: str
    mesh_faces_sha256: str
    radial_midpoint_altitudes_m: NDArray[np.float64]
    magnetic_field_t: NDArray[np.float64]
    species: tuple[ColdPlasmaSpecies, ...]
    provenance: tuple[DatasetProvenance, ...]
    interpolation: str

    def __post_init__(self) -> None:
        altitudes = np.asarray(
            self.radial_midpoint_altitudes_m, dtype=np.float64
        )
        magnetic = np.asarray(self.magnetic_field_t, dtype=np.float64)
        species = tuple(self.species)
        if (
            altitudes.ndim != 1
            or len(altitudes) < 1
            or not np.all(np.isfinite(altitudes))
            or not np.all(np.diff(altitudes) > 0.0)
        ):
            raise ValueError("plasma radial midpoints must be finite and increasing")
        if magnetic.ndim != 3 or magnetic.shape[1:] != (len(altitudes), 3):
            raise ValueError("magnetic field must have shape (face, radial, 3)")
        if not np.all(np.isfinite(magnetic)):
            raise ValueError("magnetic field must be finite")
        if not species or not all(
            isinstance(value, ColdPlasmaSpecies) for value in species
        ):
            raise ValueError("plasma model requires charged species")
        expected = magnetic.shape[:2]
        if any(value.number_density_m3.shape != expected for value in species):
            raise ValueError("plasma species grids must match the magnetic grid")
        names = [value.name for value in species]
        if len(set(names)) != len(names):
            raise ValueError("plasma species names must be unique")
        provenance = tuple(self.provenance)
        if not provenance or not all(
            isinstance(value, DatasetProvenance) for value in provenance
        ):
            raise ValueError("plasma model requires dataset provenance")
        if not isinstance(self.interpolation, str) or not self.interpolation.strip():
            raise ValueError("plasma interpolation policy must be nonempty")
        object.__setattr__(self, "radial_midpoint_altitudes_m", _readonly(altitudes))
        object.__setattr__(self, "magnetic_field_t", _readonly(magnetic))
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "provenance", provenance)

    @classmethod
    def from_mesh(
        cls,
        mesh: GeodesicMesh,
        radial_midpoint_altitudes_m: NDArray[np.float64],
        magnetic_field_t: NDArray[np.float64],
        species: tuple[ColdPlasmaSpecies, ...],
        *,
        provenance: tuple[DatasetProvenance, ...],
        interpolation: str,
    ) -> MeshPlasmaModel:
        """Bind already sampled IRI/neutral/IGRF inputs to one mesh."""

        return cls(
            mesh_vertices_sha256=mesh_vertices_sha256(mesh),
            mesh_faces_sha256=mesh_faces_sha256(mesh),
            radial_midpoint_altitudes_m=radial_midpoint_altitudes_m,
            magnetic_field_t=magnetic_field_t,
            species=species,
            provenance=provenance,
            interpolation=interpolation,
        )

    def validate_grid(
        self, mesh: GeodesicMesh, radial_midpoint_altitudes_m: NDArray[np.float64]
    ) -> None:
        """Reject use on any different topology, coordinates, or radial grid."""

        if mesh_vertices_sha256(mesh) != self.mesh_vertices_sha256:
            raise ValueError("plasma model mesh vertices do not match")
        if mesh_faces_sha256(mesh) != self.mesh_faces_sha256:
            raise ValueError("plasma model mesh faces do not match")
        if not np.array_equal(
            radial_midpoint_altitudes_m, self.radial_midpoint_altitudes_m
        ):
            raise ValueError("plasma model radial grid does not match")
        if self.magnetic_field_t.shape[0] != mesh.n_faces:
            raise ValueError("plasma model face count does not match")

    def conductivity_components(
        self, frequency_hz: float
    ) -> tuple[
        NDArray[np.complex128],
        NDArray[np.complex128],
        NDArray[np.complex128],
    ]:
        r"""Return summed :math:`\sigma_\parallel,\sigma_P,\sigma_H` arrays.

        The Hall coefficient multiplies :math:`E\times\hat b`; its sign follows
        each species charge.
        """

        if not np.isfinite(frequency_hz) or frequency_hz < 0.0:
            raise ValueError("plasma response frequency must be nonnegative")
        shape = self.magnetic_field_t.shape[:2]
        parallel = np.zeros(shape, dtype=np.complex128)
        pedersen = np.zeros_like(parallel)
        hall = np.zeros_like(parallel)
        magnetic_magnitude = np.linalg.norm(self.magnetic_field_t, axis=2)
        angular_frequency = 2.0 * np.pi * frequency_hz
        for species in self.species:
            collision = species.collision_frequency_hz
            # Frequency-domain values use the exp(+i omega t) convention.
            a = collision + 1j * angular_frequency
            gyro = species.charge_c * magnetic_magnitude / species.mass_kg
            plasma = (
                species.number_density_m3
                * species.charge_c**2
                / species.mass_kg
            )
            denominator = a**2 + gyro**2
            parallel += plasma / a
            pedersen += plasma * a / denominator
            hall += plasma * gyro / denominator
        return parallel, pedersen, hall

    @property
    def content_sha256(self) -> str:
        """Return a stable identity for all model inputs and provenance."""

        payload = {
            "mesh_vertices_sha256": self.mesh_vertices_sha256,
            "mesh_faces_sha256": self.mesh_faces_sha256,
            "radial_midpoints_sha256": array_sha256(
                self.radial_midpoint_altitudes_m
            ),
            "magnetic_field_sha256": array_sha256(self.magnetic_field_t),
            "species": [
                {
                    "name": value.name,
                    "charge_c": value.charge_c,
                    "mass_kg": value.mass_kg,
                    "density_sha256": array_sha256(value.number_density_m3),
                    "collision_sha256": array_sha256(
                        value.collision_frequency_hz
                    ),
                }
                for value in self.species
            ],
            "provenance": [
                (value.dataset_id, value.source_sha256)
                for value in self.provenance
            ],
            "interpolation": self.interpolation,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()

    def save(self, path: str | Path) -> Path:
        """Atomically save a pickle-free, checksum-verified plasma artifact."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata, arrays = self.archive_payload()
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
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

    @classmethod
    def load(cls, path: str | Path) -> MeshPlasmaModel:
        """Load and verify every array in a mesh-native plasma artifact."""

        try:
            with np.load(path, allow_pickle=False) as archive:
                if "metadata" not in archive.files:
                    raise ValueError("plasma artifact is missing metadata")
                metadata = json.loads(str(archive["metadata"].item()))
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "metadata"
                }
        except (OSError, KeyError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read plasma artifact {path}: {error}") from error
        return cls.from_archive_payload(metadata, arrays)

    def archive_payload(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return metadata and numeric arrays for embedding in a checkpoint."""

        arrays: dict[str, Any] = {
            "radial_midpoint_altitudes_m": self.radial_midpoint_altitudes_m,
            "magnetic_field_t": self.magnetic_field_t,
        }
        for index, species in enumerate(self.species):
            arrays[f"species_{index}_number_density_m3"] = (
                species.number_density_m3
            )
            arrays[f"species_{index}_collision_frequency_hz"] = (
                species.collision_frequency_hz
            )
        return self._metadata(arrays), arrays

    @classmethod
    def from_archive_payload(
        cls, metadata: dict[str, Any], arrays: dict[str, Any]
    ) -> MeshPlasmaModel:
        """Verify and restore a payload returned by :meth:`archive_payload`."""

        if metadata.get("format") != MESH_PLASMA_FORMAT:
            raise ValueError("unsupported plasma artifact format")
        if metadata.get("version") != MESH_PLASMA_VERSION:
            raise ValueError("unsupported plasma artifact version")
        records = metadata.get("arrays")
        if not isinstance(records, dict):
            raise ValueError("plasma artifact array metadata is invalid")
        for name, record in records.items():
            if name not in arrays:
                raise ValueError(f"plasma artifact is missing {name}")
            if record.get("shape") != list(arrays[name].shape):
                raise ValueError(f"plasma artifact {name} shape mismatch")
            if record.get("sha256") != array_sha256(arrays[name]):
                raise ValueError(f"plasma artifact {name} checksum mismatch")
        try:
            species = tuple(
                ColdPlasmaSpecies(
                    name=record["name"],
                    charge_c=record["charge_c"],
                    mass_kg=record["mass_kg"],
                    number_density_m3=arrays[
                        f"species_{index}_number_density_m3"
                    ],
                    collision_frequency_hz=arrays[
                        f"species_{index}_collision_frequency_hz"
                    ],
                )
                for index, record in enumerate(metadata["species"])
            )
            provenance = tuple(
                DatasetProvenance(
                    **{
                        **record,
                        "variables": tuple(
                            VariableProvenance(**value)
                            for value in record["variables"]
                        ),
                    }
                )
                for record in metadata["provenance"]
            )
            return cls(
                mesh_vertices_sha256=metadata["mesh"]["vertices_sha256"],
                mesh_faces_sha256=metadata["mesh"]["faces_sha256"],
                radial_midpoint_altitudes_m=arrays[
                    "radial_midpoint_altitudes_m"
                ],
                magnetic_field_t=arrays["magnetic_field_t"],
                species=species,
                provenance=provenance,
                interpolation=metadata["interpolation"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid plasma artifact metadata: {error}") from error

    def _metadata(self, arrays: dict[str, Any]) -> dict[str, Any]:
        return {
            "format": MESH_PLASMA_FORMAT,
            "version": MESH_PLASMA_VERSION,
            "mesh": {
                "vertices_sha256": self.mesh_vertices_sha256,
                "faces_sha256": self.mesh_faces_sha256,
            },
            "species": [
                {
                    "name": value.name,
                    "charge_c": value.charge_c,
                    "mass_kg": value.mass_kg,
                }
                for value in self.species
            ],
            "provenance": [asdict(value) for value in self.provenance],
            "interpolation": self.interpolation,
            "arrays": {
                name: {
                    "shape": list(np.asarray(values).shape),
                    "sha256": array_sha256(values),
                }
                for name, values in arrays.items()
            },
        }


class MagnetizedPlasmaADE:
    """Exact constant-E charged-fluid update at collocated vector samples."""

    def __init__(self, model: MeshPlasmaModel, time_step_s: float, backend: Any):
        self.model = model
        self.backend = backend
        magnetic = model.magnetic_field_t
        magnitude = np.linalg.norm(magnetic, axis=2)
        direction = np.zeros_like(magnetic)
        nonzero = magnitude > 0.0
        direction[nonzero] = magnetic[nonzero] / magnitude[nonzero, None]
        self._magnetic_direction = backend.asarray(direction)
        self._coefficients = []
        self.current_density = []
        for species in model.species:
            collision = species.collision_frequency_hz
            gyro = species.charge_c * magnitude / species.mass_kg
            decay = np.exp(-collision * time_step_s)
            angle = gyro * time_step_s
            eigenvalue = -collision + 1j * gyro
            integral = np.expm1(eigenvalue * time_step_s) / eigenvalue
            parallel_integral = -np.expm1(-collision * time_step_s) / collision
            drive = (
                species.number_density_m3
                * species.charge_c**2
                / species.mass_kg
            )
            self._coefficients.append(
                tuple(
                    backend.asarray(value)
                    for value in (
                        decay,
                        np.cos(angle),
                        np.sin(angle),
                        drive * parallel_integral,
                        drive * integral.real,
                        drive * integral.imag,
                    )
                )
            )
            self.current_density.append(backend.zeros((*magnitude.shape, 3)))

    def advance(self, electric_v_m: Any) -> Any:
        """Advance every species current and return their summed new value."""

        total = self.backend.zeros(tuple(electric_v_m.shape))
        b = self._magnetic_direction
        electric_parallel = (electric_v_m * b).sum(axis=2)[..., None] * b
        electric_perpendicular = electric_v_m - electric_parallel
        electric_cross = _cross_with_b(electric_perpendicular, b, self.backend)
        next_current_density = []
        for index, coefficients in enumerate(self._coefficients):
            decay, cosine, sine, drive_parallel, drive_real, drive_imag = (
                coefficients
            )
            current = self.current_density[index]
            current_parallel = (current * b).sum(axis=2)[..., None] * b
            current_perpendicular = current - current_parallel
            current_cross = _cross_with_b(current_perpendicular, b, self.backend)
            updated = (
                decay[..., None] * current_parallel
                + drive_parallel[..., None] * electric_parallel
                + decay[..., None]
                * (
                    cosine[..., None] * current_perpendicular
                    + sine[..., None] * current_cross
                )
                + drive_real[..., None] * electric_perpendicular
                + drive_imag[..., None] * electric_cross
            )
            next_current_density.append(updated)
            total = total + updated
        self.current_density = next_current_density
        return total

    @property
    def state_bytes(self) -> int:
        return sum(self.backend.nbytes(values) for values in self.current_density)

    @property
    def persistent_bytes(self) -> int:
        arrays = [self._magnetic_direction, *self.current_density]
        arrays.extend(value for group in self._coefficients for value in group)
        return sum(self.backend.nbytes(values) for values in arrays)


class GeodesicPlasmaCoupler:
    """Reconstruct face vectors and scatter ADE current to Er/Et supports."""

    def __init__(
        self,
        model: MeshPlasmaModel,
        mesh: GeodesicMesh,
        radial_midpoint_altitudes_m: NDArray[np.float64],
        time_step_s: float,
        backend: Any,
    ) -> None:
        model.validate_grid(mesh, radial_midpoint_altitudes_m)
        self.mesh = mesh
        self.backend = backend
        reconstruction, face_tangents = _face_reconstruction(mesh)
        self._face_edges = backend.index_array(mesh.face_edges)
        self._faces = backend.index_array(mesh.faces)
        self._reconstruction = backend.asarray(reconstruction)
        self._face_centers = backend.asarray(mesh.face_centers)
        left_slots = _edge_face_slots(mesh, mesh.edge_left_faces)
        right_slots = _edge_face_slots(mesh, mesh.edge_right_faces)
        self._left_faces = backend.index_array(mesh.edge_left_faces)
        self._right_faces = backend.index_array(mesh.edge_right_faces)
        self._left_tangents = backend.asarray(
            face_tangents[mesh.edge_left_faces, left_slots]
        )
        self._right_tangents = backend.asarray(
            face_tangents[mesh.edge_right_faces, right_slots]
        )
        vertex_faces, vertex_weights = _vertex_face_average(mesh)
        self._vertex_faces = backend.index_array(vertex_faces)
        self._vertex_face_weights = backend.asarray(vertex_weights)
        self.ade = MagnetizedPlasmaADE(model, time_step_s, backend)

    def advance(self, er: Any, et: Any) -> tuple[Any, Any]:
        """Advance collocated plasma current and return Jr/Jt density arrays."""

        radial_nodes_at_faces = er[self._faces].sum(axis=1) / 3.0
        radial = 0.5 * (
            radial_nodes_at_faces[:, :-1] + radial_nodes_at_faces[:, 1:]
        )
        edge_values = et[self._face_edges]
        tangential = (
            edge_values[..., None] * self._reconstruction[:, :, None, :]
        ).sum(axis=1)
        electric = tangential + radial[..., None] * self._face_centers[:, None, :]
        current = self.ade.advance(electric)

        left = (
            current[self._left_faces] * self._left_tangents[:, None, :]
        ).sum(axis=2)
        right = (
            current[self._right_faces] * self._right_tangents[:, None, :]
        ).sum(axis=2)
        tangential_current = 0.5 * (left + right)

        radial_at_faces = (
            current * self._face_centers[:, None, :]
        ).sum(axis=2)
        radial_at_vertices = (
            radial_at_faces[self._vertex_faces]
            * self._vertex_face_weights[:, :, None]
        ).sum(axis=1)
        radial_current = self.backend.zeros(
            (radial_at_vertices.shape[0], radial_at_vertices.shape[1] + 1)
        )
        radial_current[:, 0] = radial_at_vertices[:, 0]
        radial_current[:, -1] = radial_at_vertices[:, -1]
        if radial_at_vertices.shape[1] > 1:
            radial_current[:, 1:-1] = 0.5 * (
                radial_at_vertices[:, :-1] + radial_at_vertices[:, 1:]
            )
        return radial_current, tangential_current

    @property
    def persistent_bytes(self) -> int:
        arrays = (
            self._face_edges,
            self._faces,
            self._reconstruction,
            self._face_centers,
            self._left_faces,
            self._right_faces,
            self._left_tangents,
            self._right_tangents,
            self._vertex_faces,
            self._vertex_face_weights,
        )
        return self.ade.persistent_bytes + sum(
            self.backend.nbytes(values) for values in arrays
        )


def _face_reconstruction(
    mesh: GeodesicMesh,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    normals = np.cross(mesh.vertices[mesh.edges[:, 0]], mesh.vertices[mesh.edges[:, 1]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    face_normals = normals[mesh.face_edges]
    tangents = np.cross(face_normals, mesh.face_centers[:, None, :])
    tangents /= np.linalg.norm(tangents, axis=2, keepdims=True)
    reconstruction = np.empty_like(tangents)
    for face in range(mesh.n_faces):
        reconstruction[face] = np.linalg.pinv(tangents[face]).T
    return reconstruction, tangents


def _edge_face_slots(
    mesh: GeodesicMesh, faces: NDArray[np.int64]
) -> NDArray[np.int64]:
    edge_indices = np.arange(mesh.n_edges)[:, None]
    matches = mesh.face_edges[faces] == edge_indices
    if not np.all(np.sum(matches, axis=1) == 1):
        raise RuntimeError("edge is absent from an adjacent face")
    return np.argmax(matches, axis=1)


def _vertex_face_average(
    mesh: GeodesicMesh,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    incident = [[] for _ in range(mesh.n_vertices)]
    for face, vertices in enumerate(mesh.faces):
        for vertex in vertices:
            incident[int(vertex)].append(face)
    maximum = max(len(values) for values in incident)
    faces = np.zeros((mesh.n_vertices, maximum), dtype=np.int64)
    weights = np.zeros((mesh.n_vertices, maximum), dtype=np.float64)
    for vertex, values in enumerate(incident):
        face_indices = np.asarray(values, dtype=np.int64)
        face_weights = mesh.face_solid_angles[face_indices]
        face_weights /= np.sum(face_weights)
        faces[vertex, : len(values)] = face_indices
        weights[vertex, : len(values)] = face_weights
    return faces, weights


def _cross_with_b(values: Any, b: Any, backend: Any) -> Any:
    result = backend.empty_like(values)
    result[..., 0] = values[..., 1] * b[..., 2] - values[..., 2] * b[..., 1]
    result[..., 1] = values[..., 2] * b[..., 0] - values[..., 0] * b[..., 2]
    result[..., 2] = values[..., 0] * b[..., 1] - values[..., 1] * b[..., 0]
    return result


def _readonly(values: NDArray[np.float64]) -> NDArray[np.float64]:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    result.setflags(write=False)
    return result
