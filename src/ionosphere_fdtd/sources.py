"""Localized current sources for radial and tangential electric fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .solver import GeodesicFDTD

GWANGJU_LATITUDE_DEG = 35.1595
GWANGJU_LONGITUDE_DEG = 126.8526


def geographic_direction(
    latitude_deg: float, longitude_deg: float
) -> NDArray[np.float64]:
    """Return a unit vector for a geographic latitude and longitude."""

    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    return np.asarray(
        (
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ),
        dtype=np.float64,
    )


def geographic_tangent_basis(
    latitude_deg: float, longitude_deg: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return local unit vectors pointing east and north."""

    latitude = np.deg2rad(latitude_deg)
    longitude = np.deg2rad(longitude_deg)
    east = np.asarray((-np.sin(longitude), np.cos(longitude), 0.0))
    north = np.asarray(
        (
            -np.sin(latitude) * np.cos(longitude),
            -np.sin(latitude) * np.sin(longitude),
            np.cos(latitude),
        )
    )
    return east, north


def geographic_face_index(
    simulation: GeodesicFDTD,
    latitude_deg: float,
    longitude_deg: float,
) -> int:
    """Return the primal face containing a geographic direction."""

    direction = geographic_direction(latitude_deg, longitude_deg)
    faces = simulation.mesh.faces
    triangles = simulation.mesh.vertices[faces]
    signs = np.column_stack(
        tuple(
            np.einsum(
                "ij,j->i",
                np.cross(triangles[:, edge], triangles[:, (edge + 1) % 3]),
                direction,
            )
            for edge in range(3)
        )
    )
    inside = np.all(signs >= -1.0e-12, axis=1) | np.all(
        signs <= 1.0e-12, axis=1
    )
    candidates = np.flatnonzero(inside)
    return (
        int(
            candidates[
                np.argmax(simulation.mesh.face_centers[candidates] @ direction)
            ]
        )
        if len(candidates)
        else int(np.argmax(simulation.mesh.face_centers @ direction))
    )


def geographic_distribution(
    simulation: GeodesicFDTD,
    latitude_deg: float,
    longitude_deg: float,
    altitude_m: float,
) -> tuple[NDArray[np.int64], int, NDArray[np.float64]]:
    """Return triangle vertices, radial layer, and barycentric weights."""

    direction = geographic_direction(latitude_deg, longitude_deg)
    faces = simulation.mesh.faces
    face_index = geographic_face_index(
        simulation, latitude_deg, longitude_deg
    )
    vertices = faces[face_index]
    point_a, point_b, point_c = simulation.mesh.vertices[vertices]
    normal = np.cross(point_b - point_a, point_c - point_a)
    intersection = direction * float((normal @ point_a) / (normal @ direction))
    edge_ab = point_b - point_a
    edge_ac = point_c - point_a
    point_offset = intersection - point_a
    dot_ab_ab = float(edge_ab @ edge_ab)
    dot_ab_ac = float(edge_ab @ edge_ac)
    dot_ac_ac = float(edge_ac @ edge_ac)
    dot_offset_ab = float(point_offset @ edge_ab)
    dot_offset_ac = float(point_offset @ edge_ac)
    denominator = dot_ab_ab * dot_ac_ac - dot_ab_ac**2
    weight_b = (
        dot_ac_ac * dot_offset_ab - dot_ab_ac * dot_offset_ac
    ) / denominator
    weight_c = (
        dot_ab_ab * dot_offset_ac - dot_ab_ac * dot_offset_ab
    ) / denominator
    weights = np.asarray((1.0 - weight_b - weight_c, weight_b, weight_c))
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()
    layer = int(np.argmin(np.abs(simulation.altitudes_m - altitude_m)))
    return vertices.copy(), layer, weights


def radial_linear_distribution(
    radial_altitudes_m: NDArray[np.float64],
    altitude_m: float,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Represent an altitude exactly on adjacent staggered radial planes."""

    altitudes = np.asarray(radial_altitudes_m, dtype=np.float64)
    if altitudes.ndim != 1 or len(altitudes) < 1:
        raise ValueError("radial_altitudes_m must be a nonempty 1-D array")
    if not np.all(np.diff(altitudes) > 0.0):
        raise ValueError("radial_altitudes_m must be strictly increasing")
    if altitude_m < altitudes[0] or altitude_m > altitudes[-1]:
        raise ValueError("source altitude is outside the radial grid")

    upper = int(np.searchsorted(altitudes, altitude_m, side="left"))
    if upper < len(altitudes) and altitudes[upper] == altitude_m:
        return (
            np.asarray((upper,), dtype=np.int64),
            np.asarray((1.0,), dtype=np.float64),
        )
    lower = upper - 1
    upper_weight = (altitude_m - altitudes[lower]) / (
        altitudes[upper] - altitudes[lower]
    )
    return (
        np.asarray((lower, upper), dtype=np.int64),
        np.asarray((1.0 - upper_weight, upper_weight), dtype=np.float64),
    )


@dataclass(frozen=True, slots=True)
class GaussianCurrent:
    """Localized vertical current with a Gaussian (optionally modulated) pulse."""

    latitude_deg: float = GWANGJU_LATITUDE_DEG
    longitude_deg: float = GWANGJU_LONGITUDE_DEG
    altitude_m: float = 2_500.0
    peak_current_a: float = 1.0e6
    vertical_element_length_m: float = 5_000.0
    center_time_s: float | None = None
    one_over_e_half_width_s: float | None = None
    carrier_frequency_hz: float = 0.0

    def __post_init__(self) -> None:
        finite = (
            self.latitude_deg,
            self.longitude_deg,
            self.altitude_m,
            self.peak_current_a,
            self.vertical_element_length_m,
            self.carrier_frequency_hz,
        )
        if not all(np.isfinite(value) for value in finite):
            raise ValueError("source coordinates and waveform values must be finite")
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise ValueError("source latitude must be in [-90, 90]")
        if self.vertical_element_length_m <= 0.0:
            raise ValueError("vertical source element length must be positive")
        if self.center_time_s is not None and not np.isfinite(self.center_time_s):
            raise ValueError("source center time must be finite")
        if self.one_over_e_half_width_s is not None and (
            not np.isfinite(self.one_over_e_half_width_s)
            or self.one_over_e_half_width_s <= 0.0
        ):
            raise ValueError("source half width must be finite and positive")
        if self.carrier_frequency_hz < 0.0:
            raise ValueError("source carrier frequency cannot be negative")

    def direction(self) -> NDArray[np.float64]:
        """Return the exact geographic source direction."""

        return geographic_direction(self.latitude_deg, self.longitude_deg)

    def location(self, simulation: GeodesicFDTD) -> tuple[int, int]:
        """Return the nearest surface vertex and radial layer."""

        direction = self.direction()
        vertex = int(np.argmax(simulation.mesh.vertices @ direction))
        layer = int(np.argmin(np.abs(simulation.altitudes_m - self.altitude_m)))
        return vertex, layer

    def distribution(
        self, simulation: GeodesicFDTD
    ) -> tuple[NDArray[np.int64], int, NDArray[np.float64]]:
        """Distribute current over the triangle containing the exact location."""

        return geographic_distribution(
            simulation,
            self.latitude_deg,
            self.longitude_deg,
            self.altitude_m,
        )

    def staggered_distribution(
        self, simulation: GeodesicFDTD
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Distribute current exactly in both surface and radial coordinates."""

        vertices, _, horizontal_weights = self.distribution(simulation)
        layers, radial_weights = radial_linear_distribution(
            simulation.altitudes_m, self.altitude_m
        )
        count = len(layers)
        combined_vertices = np.repeat(vertices, count)
        combined_layers = np.tile(layers, len(vertices))
        combined_weights = np.repeat(horizontal_weights, count) * np.tile(
            radial_weights, len(vertices)
        )
        return combined_vertices, combined_layers, combined_weights

    def current_a(self, time_s: float, dt_s: float) -> float:
        half_width, center = self._waveform_times(dt_s)
        envelope = np.exp(-((time_s - center) / half_width) ** 2)
        if self.carrier_frequency_hz:
            envelope *= np.cos(
                2.0 * np.pi * self.carrier_frequency_hz * (time_s - center)
            )
        return float(self.peak_current_a * envelope)

    def current_tensor_a(
        self,
        time_s: Any,
        dt_s: float,
        *,
        peak_current_a: Any | None = None,
    ) -> Any:
        """Evaluate the waveform without leaving a PyTorch tensor graph.

        ``time_s`` must be a floating PyTorch tensor. Passing a tensor as
        ``peak_current_a`` makes the waveform amplitude differentiable; the
        source geometry and stored scalar metadata remain static simulation
        inputs.
        """

        try:
            import torch
        except ImportError as error:
            raise TypeError("time_s must be a PyTorch tensor") from error
        if not torch.is_tensor(time_s) or not time_s.is_floating_point():
            raise TypeError("time_s must be a floating PyTorch tensor")
        half_width, center = self._waveform_times(dt_s)
        envelope = torch.exp(-((time_s - center) / half_width) ** 2)
        if self.carrier_frequency_hz:
            envelope = envelope * torch.cos(
                2.0
                * torch.pi
                * self.carrier_frequency_hz
                * (time_s - center)
            )
        amplitude = self.peak_current_a if peak_current_a is None else peak_current_a
        if torch.is_tensor(amplitude):
            amplitude = amplitude.to(device=time_s.device, dtype=time_s.dtype)
        return amplitude * envelope

    def _waveform_times(self, dt_s: float) -> tuple[float, float]:
        if self.one_over_e_half_width_s is not None:
            half_width = self.one_over_e_half_width_s
        elif self.carrier_frequency_hz:
            # At 20 Hz this is 25 ms, close to the 42.5 ms FWHM Gaussian
            # envelope used for the radar example in Simpson et al.
            half_width = 0.5 / self.carrier_frequency_hz
        else:
            half_width = 12.0 * dt_s
        center = (
            self.center_time_s
            if self.center_time_s is not None
            else max(4.0 * half_width, 36.0 * dt_s)
        )
        return half_width, center


@dataclass(frozen=True, slots=True)
class TangentialGaussianCurrent(GaussianCurrent):
    """Gaussian current impressed along one or more horizontal ground lines.

    Azimuths are degrees clockwise from geographic north.  Each requested
    direction is projected onto the three oriented primal edges of the face
    containing the source.  This provides the tangential-current degrees of
    freedom used by the TE-r update without snapping the polarization to a
    single geodesic edge.
    """

    altitude_m: float = 0.0
    azimuths_deg: tuple[float, ...] = (0.0,)
    line_lengths_m: tuple[float, ...] | None = None
    edge_assignment: str = "projected"

    def __post_init__(self) -> None:
        GaussianCurrent.__post_init__(self)
        if not self.azimuths_deg:
            raise ValueError("azimuths_deg must contain at least one direction")
        if not all(np.isfinite(value) for value in self.azimuths_deg):
            raise ValueError("source azimuths must be finite")
        if self.line_lengths_m is not None:
            if len(self.line_lengths_m) != len(self.azimuths_deg):
                raise ValueError("line_lengths_m must match azimuths_deg")
            if not all(
                np.isfinite(value) and value > 0.0
                for value in self.line_lengths_m
            ):
                raise ValueError("ground-line lengths must be finite and positive")
        if self.edge_assignment not in {"projected", "nearest"}:
            raise ValueError("edge_assignment must be 'projected' or 'nearest'")

    def edge_distribution(
        self, simulation: GeodesicFDTD
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
        """Project ground-line currents onto horizontal and radial samples."""

        face = geographic_face_index(
            simulation, self.latitude_deg, self.longitude_deg
        )
        edges = simulation.mesh.face_edges[face].copy()
        endpoints = simulation.mesh.vertices[simulation.mesh.edges[edges]]
        edge_directions = endpoints[:, 1] - endpoints[:, 0]
        edge_directions /= np.linalg.norm(edge_directions, axis=1, keepdims=True)
        east, north = geographic_tangent_basis(
            self.latitude_deg, self.longitude_deg
        )
        edge_lengths = (
            simulation.mesh.primal_edge_angles[edges]
            * (simulation.config.earth_radius_m + self.altitude_m)
        )
        weights = np.zeros(len(edges), dtype=np.float64)
        line_lengths = self.line_lengths_m or tuple(
            1.0 for _ in self.azimuths_deg
        )
        for azimuth_deg, line_length_m in zip(
            self.azimuths_deg, line_lengths, strict=True
        ):
            azimuth = np.deg2rad(azimuth_deg)
            requested = np.cos(azimuth) * north + np.sin(azimuth) * east
            projections = edge_directions @ requested
            if self.edge_assignment == "projected":
                weights += line_length_m * projections / edge_lengths
            else:
                selected = int(np.argmax(np.abs(projections)))
                weights[selected] += (
                    line_length_m * projections[selected] / edge_lengths[selected]
                )
        radial_layers, radial_weights = radial_linear_distribution(
            simulation.radial_midpoint_altitudes_m,
            self.altitude_m,
        )
        support_edges = np.repeat(edges, len(radial_layers))
        support_layers = np.tile(radial_layers, len(edges))
        support_weights = np.repeat(weights, len(radial_layers)) * np.tile(
            radial_weights, len(edges)
        )
        return support_edges, support_layers, support_weights
