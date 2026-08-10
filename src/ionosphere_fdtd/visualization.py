"""Scientific visualization helpers for geodesic FDTD fields.

Matplotlib and Cartopy provide quantitative maps, sections, and traces.
PyVista provides interactive 3-D topology inspection and animations.  These
libraries are optional so the numerical solver remains lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from .mesh import GeodesicMesh
from .solver import GeodesicFDTD

FloatArray = NDArray[np.float64]


class VisualizationDependencyError(ImportError):
    """Raised when an optional visualization dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class RadialSection:
    """Radial electric field sampled along a great-circle path."""

    distance_m: FloatArray
    altitudes_m: FloatArray
    values: FloatArray
    directions: FloatArray


@dataclass(frozen=True, slots=True)
class Receiver:
    """A radial electric-field observation location."""

    latitude_deg: float
    longitude_deg: float
    altitude_m: float = 0.0
    label: str | None = None


@dataclass(frozen=True, slots=True)
class ReceiverTraces:
    """Time samples recorded at one or more receivers."""

    time_s: FloatArray
    er_v_m: FloatArray
    labels: tuple[str, ...]


def plot_surface_field(
    simulation: GeodesicFDTD,
    component: str = "er",
    *,
    altitude_m: float = 0.0,
    projection: str = "mollweide",
    color_limit: float | None = None,
    scale: str = "linear",
    symlog_linthresh: float | None = None,
    cmap: str = "RdBu_r",
    coastlines: bool = False,
    resolution_deg: float = 2.0,
    neighbors: int = 3,
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Plot ``Er`` or ``Hr`` on a projected global geodesic grid.

    The unstructured field is sampled on a display-only longitude/latitude
    grid before projection, which avoids false triangles across the projection
    seam.  A fixed symmetric color normalization keeps field polarity and
    attenuation comparable between frames.
    """

    if not 0.0 < resolution_deg <= 30.0:
        raise ValueError("resolution_deg must be in (0, 30]")
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    plt, colors, ccrs = _matplotlib_modules()
    values, field_altitude, association = _surface_values(
        simulation, component, altitude_m
    )
    source_directions = (
        simulation.mesh.vertices
        if association == "point"
        else simulation.mesh.face_centers
    )
    longitude = np.arange(-180.0, 180.0 + resolution_deg, resolution_deg)
    latitude = np.arange(-90.0, 90.0 + resolution_deg, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    longitude_radians = np.deg2rad(lon_grid.ravel())
    latitude_radians = np.deg2rad(lat_grid.ravel())
    cos_latitude = np.cos(latitude_radians)
    display_directions = np.column_stack(
        (
            cos_latitude * np.cos(longitude_radians),
            cos_latitude * np.sin(longitude_radians),
            np.sin(latitude_radians),
        )
    )
    display_values = _inverse_distance_sample_chunked(
        display_directions,
        source_directions,
        values[:, None],
        min(neighbors, len(source_directions)),
    )[:, 0].reshape(lon_grid.shape)

    if ax is None:
        figure = plt.figure(figsize=(10.0, 5.4), constrained_layout=True)
        ax = figure.add_subplot(1, 1, 1, projection=_projection(ccrs, projection))
    else:
        figure = ax.figure

    norm = _symmetric_norm(colors, values, color_limit, scale, symlog_linthresh)
    artist = ax.pcolormesh(
        lon_grid,
        lat_grid,
        display_values,
        shading="auto",
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
    )
    ax.set_global()
    ax.gridlines(linewidth=0.45, alpha=0.45)
    if coastlines:
        ax.coastlines(linewidth=0.55)
    label = "Er (V/m)" if component.lower() == "er" else "Hr (A/m)"
    figure.colorbar(artist, ax=ax, orientation="horizontal", pad=0.06, label=label)
    ax.set_title(
        title
        or f"{component.upper()} at {field_altitude / 1_000.0:.1f} km, "
        f"t = {_field_time_s(simulation, component):.6g} s"
    )
    return figure, ax, artist


def sample_radial_section(
    simulation: GeodesicFDTD,
    start_latitude_deg: float,
    start_longitude_deg: float,
    end_latitude_deg: float,
    end_longitude_deg: float,
    *,
    samples: int = 241,
    neighbors: int = 3,
) -> RadialSection:
    """Interpolate ``Er`` along a great-circle distance-height section."""

    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not 1 <= neighbors <= simulation.mesh.n_vertices:
        raise ValueError("neighbors is outside the mesh size")
    start = geographic_direction(start_latitude_deg, start_longitude_deg)
    end = geographic_direction(end_latitude_deg, end_longitude_deg)
    directions, arc = _great_circle_directions(start, end, samples)
    values = _inverse_distance_sample(
        directions,
        simulation.mesh.vertices,
        simulation.to_numpy(simulation.er),
        neighbors,
    )
    return RadialSection(
        distance_m=np.linspace(0.0, arc * simulation.config.earth_radius_m, samples),
        altitudes_m=simulation.altitudes_m.copy(),
        values=values.T,
        directions=directions,
    )


def plot_radial_section(
    section: RadialSection,
    *,
    color_limit: float | None = None,
    scale: str = "linear",
    symlog_linthresh: float | None = None,
    cmap: str = "RdBu_r",
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Plot a great-circle ``Er`` distance-height section."""

    plt, colors, _ = _matplotlib_modules()
    if ax is None:
        figure, ax = plt.subplots(figsize=(10.0, 4.6), constrained_layout=True)
    else:
        figure = ax.figure
    norm = _symmetric_norm(
        colors, section.values, color_limit, scale, symlog_linthresh
    )
    artist = ax.pcolormesh(
        section.distance_m / 1.0e6,
        section.altitudes_m / 1.0e3,
        section.values,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xlabel("Great-circle distance (Mm)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title(title or "Radial electric field along great-circle path")
    figure.colorbar(artist, ax=ax, label="Er (V/m)")
    return figure, ax, artist


def record_receiver_traces(
    simulation: GeodesicFDTD,
    receivers: Sequence[Receiver],
    steps: int,
    *,
    sample_every: int = 1,
) -> ReceiverTraces:
    """Advance a simulation while recording ``Er`` at fixed receivers."""

    if not receivers:
        raise ValueError("at least one receiver is required")
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")

    locations = [_receiver_location(simulation, receiver) for receiver in receivers]
    labels = tuple(
        receiver.label
        or f"{receiver.latitude_deg:g}°, {receiver.longitude_deg:g}°"
        for receiver in receivers
    )
    times = [simulation.time_s]
    rows = [
        [simulation.field_value("er", vertex, layer) for vertex, layer in locations]
    ]
    remaining = steps
    while remaining:
        advance = min(sample_every, remaining)
        simulation.step(advance)
        remaining -= advance
        times.append(simulation.time_s)
        rows.append(
            [
                simulation.field_value("er", vertex, layer)
                for vertex, layer in locations
            ]
        )
    return ReceiverTraces(
        time_s=np.asarray(times), er_v_m=np.asarray(rows), labels=labels
    )


def plot_receiver_traces(
    traces: ReceiverTraces,
    *,
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[Any, Any]:
    """Plot one or more recorded radial electric-field time series."""

    plt, _, _ = _matplotlib_modules()
    if ax is None:
        figure, ax = plt.subplots(figsize=(9.0, 4.4), constrained_layout=True)
    else:
        figure = ax.figure
    for index, label in enumerate(traces.labels):
        ax.plot(traces.time_s, traces.er_v_m[:, index], label=label)
    ax.axhline(0.0, color="0.25", linewidth=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Er (V/m)")
    ax.set_title(title or "Receiver radial electric-field traces")
    if len(traces.labels) > 1:
        ax.legend()
    return figure, ax


def make_pyvista_surface(
    simulation: GeodesicFDTD,
    component: str = "er",
    *,
    altitude_m: float = 0.0,
    radial_exaggeration: float = 1.0,
) -> tuple[Any, str, FloatArray]:
    """Create a PyVista surface carrying ``Er``, ``Hr``, or topology data."""

    if radial_exaggeration <= 0.0:
        raise ValueError("radial_exaggeration must be positive")
    component = component.lower()
    displayed_radius_km = (
        simulation.config.earth_radius_m + radial_exaggeration * altitude_m
    ) / 1.0e3
    if displayed_radius_km <= 0.0:
        raise ValueError("displayed radius must be positive")
    pv = _pyvista_module()

    if component in {"er", "topology"}:
        dataset = _dual_polydata(pv, simulation.mesh, displayed_radius_km)
        if component == "er":
            values, _, _ = _surface_values(simulation, "er", altitude_m)
            name = "Er (V/m)"
        else:
            values = simulation.mesh.vertex_degree.astype(np.float64)
            name = "Dual cell sides"
        dataset.cell_data[name] = values
    elif component == "hr":
        vtk_faces = np.column_stack(
            (np.full(simulation.mesh.n_faces, 3), simulation.mesh.faces)
        ).ravel()
        dataset = pv.PolyData(
            displayed_radius_km * simulation.mesh.vertices, vtk_faces
        )
        values, _, _ = _surface_values(simulation, "hr", altitude_m)
        name = "Hr (A/m)"
        dataset.cell_data[name] = values
    else:
        raise ValueError("3-D surface component must be 'er', 'hr', or 'topology'")
    return dataset, name, values


def plot_mesh_3d(
    simulation: GeodesicFDTD,
    component: str = "topology",
    *,
    altitude_m: float = 0.0,
    radial_exaggeration: float = 1.0,
    color_limit: float | None = None,
    cmap: str = "RdBu_r",
    show_edges: bool = True,
    earth_texture: bool = True,
    field_opacity: float = 0.82,
    screenshot: str | Path | None = None,
    off_screen: bool | None = None,
    show: bool = True,
) -> tuple[Any, Any]:
    """Render the dual topology or a radial field on the spherical mesh."""

    pv = _pyvista_module()
    if screenshot is not None:
        off_screen = True
    plotter = pv.Plotter(off_screen=off_screen)
    dataset, name, values = make_pyvista_surface(
        simulation,
        component,
        altitude_m=altitude_m,
        radial_exaggeration=radial_exaggeration,
    )
    if not 0.0 < field_opacity <= 1.0:
        raise ValueError("field_opacity must be in (0, 1]")
    if earth_texture:
        _add_earth_underlay(plotter, simulation, altitude_m, radial_exaggeration)
    if component.lower() == "topology":
        if earth_texture:
            plotter.add_mesh(
                dataset,
                style="wireframe",
                color="black",
                line_width=1.0,
                name="geodesic-grid",
            )
            pentagons = (
                dataset.extract_cells(values == 5.0)
                .extract_surface(algorithm="dataset_surface")
                .triangulate()
                .subdivide(2)
            )
            radii = np.linalg.norm(pentagons.points, axis=1, keepdims=True)
            surface_radius = float(np.linalg.norm(dataset.points[0]))
            pentagons.points *= (1.0005 * surface_radius) / radii
            plotter.add_mesh(
                pentagons,
                color="purple",
                opacity=0.55,
                lighting=False,
                name="pentagonal-cells",
            )
        else:
            plotter.add_mesh(
                dataset,
                scalars=name,
                categories=True,
                cmap="viridis",
                show_edges=show_edges,
                scalar_bar_args={"title": name, "n_labels": 2},
            )
    else:
        limit = _color_limit(values, color_limit)
        if earth_texture and show_edges:
            plotter.add_mesh(
                dataset,
                style="wireframe",
                color="black",
                opacity=0.5,
                line_width=1.0,
                name="geodesic-grid",
            )
        plotter.add_mesh(
            dataset,
            scalars=name,
            clim=(-limit, limit),
            cmap=cmap,
            show_edges=show_edges and not earth_texture,
            lighting=not earth_texture,
            opacity=(
                _field_opacity_transfer(field_opacity)
                if earth_texture
                else 1.0
            ),
            scalar_bar_args={"title": name},
            name="field-overlay",
        )
    if earth_texture:
        _add_source_marker(plotter, pv, simulation, altitude_m, radial_exaggeration)
    plotter.add_axes()
    _set_source_camera(plotter, simulation, altitude_m, radial_exaggeration)
    if screenshot is not None:
        plotter.show(screenshot=str(screenshot), auto_close=True)
    elif show:
        plotter.show()
    return plotter, dataset


def run_live_surface(
    simulation: GeodesicFDTD,
    component: str = "er",
    *,
    altitude_m: float = 0.0,
    steps_per_frame: int = 10,
    frames_per_second: int = 20,
    max_frames: int | None = None,
    color_limit: float | None = None,
    cmap: str = "RdBu_r",
    radial_exaggeration: float = 1.0,
    show_edges: bool = True,
    earth_texture: bool = True,
    field_opacity: float = 0.82,
) -> int:
    """Advance the simulation from a responsive, interactive PyVista window.

    The VTK event loop owns the main thread.  A timer callback advances a small
    batch of FDTD steps, replaces the surface scalars, and renders the next
    frame.  With no explicit ``color_limit``, the symmetric color range follows
    the current field amplitude so a pulse is visible from startup.

    Close the window or press ``q`` to stop.  If ``max_frames`` is supplied,
    calculation stops at that frame while the final field remains interactive.
    """

    component = component.lower()
    if component not in {"er", "hr"}:
        raise ValueError("live surface component must be 'er' or 'hr'")
    if steps_per_frame < 1:
        raise ValueError("steps_per_frame must be positive")
    if frames_per_second < 1:
        raise ValueError("frames_per_second must be positive")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided")
    if not 0.0 < field_opacity <= 1.0:
        raise ValueError("field_opacity must be in (0, 1]")

    pv = _pyvista_module()
    dataset, name, values = make_pyvista_surface(
        simulation,
        component,
        altitude_m=altitude_m,
        radial_exaggeration=radial_exaggeration,
    )
    limit = _color_limit(values, color_limit)
    plotter = pv.Plotter()
    if earth_texture:
        _add_earth_underlay(plotter, simulation, altitude_m, radial_exaggeration)
        if show_edges:
            plotter.add_mesh(
                dataset,
                style="wireframe",
                color="black",
                opacity=0.5,
                line_width=1.0,
                name="geodesic-grid",
            )
    actor = plotter.add_mesh(
        dataset,
        scalars=name,
        clim=(-limit, limit),
        cmap=cmap,
        show_edges=show_edges and not earth_texture,
        lighting=not earth_texture,
        opacity=(
            _field_opacity_transfer(field_opacity) if earth_texture else 1.0
        ),
        scalar_bar_args={"title": name},
        name="field-overlay",
    )
    if earth_texture:
        _add_source_marker(plotter, pv, simulation, altitude_m, radial_exaggeration)
    plotter.add_axes()
    _set_source_camera(plotter, simulation, altitude_m, radial_exaggeration)
    completed_frames = 0

    def advance(_timer_step: int) -> None:
        nonlocal completed_frames, limit
        simulation.step(steps_per_frame)
        updated, _, _ = _surface_values(simulation, component, altitude_m)
        dataset.cell_data[name] = updated
        dataset.Modified()
        if color_limit is None:
            limit = _color_limit(updated, None)
            actor.mapper.scalar_range = (-limit, limit)
        completed_frames += 1
        plotter.add_text(
            _live_status(
                simulation, component, updated, limit, color_limit is None
            ),
            font_size=10,
            name="simulation-status",
        )

    plotter.add_text(
        _live_status(simulation, component, values, limit, color_limit is None),
        font_size=10,
        name="simulation-status",
    )
    plotter.add_timer_event(
        max_steps=max_frames or 2_147_483_647,
        duration=max(1, round(1_000 / frames_per_second)),
        callback=advance,
    )
    try:
        plotter.show(title="Ionosphere FDTD live field", auto_close=True)
    except KeyboardInterrupt:
        # PyVista closes the render window before re-raising.  Treat Ctrl+C as
        # a normal user stop so the CLI exits without an exception traceback.
        pass
    return completed_frames


def animate_surface_field(
    simulation: GeodesicFDTD,
    output: str | Path,
    *,
    component: str = "er",
    altitude_m: float = 0.0,
    frames: int = 120,
    steps_per_frame: int = 10,
    frames_per_second: int = 24,
    color_limit: float | None = None,
    cmap: str = "RdBu_r",
    radial_exaggeration: float = 1.0,
    show_edges: bool = True,
    earth_texture: bool = True,
    field_opacity: float = 0.82,
) -> Path:
    """Advance the simulation and write a fixed-scale GIF or MP4 animation."""

    if frames < 1:
        raise ValueError("frames must be positive")
    if steps_per_frame < 1:
        raise ValueError("steps_per_frame must be positive")
    if frames_per_second < 1:
        raise ValueError("frames_per_second must be positive")
    if not 0.0 < field_opacity <= 1.0:
        raise ValueError("field_opacity must be in (0, 1]")
    output_path = Path(output)
    if output_path.suffix.lower() not in {".gif", ".mp4"}:
        raise ValueError("animation output must end in .gif or .mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pv = _pyvista_module()
    dataset, name, values = make_pyvista_surface(
        simulation,
        component,
        altitude_m=altitude_m,
        radial_exaggeration=radial_exaggeration,
    )
    limit = _color_limit(values, color_limit)
    plotter = pv.Plotter(off_screen=True)
    if earth_texture:
        _add_earth_underlay(plotter, simulation, altitude_m, radial_exaggeration)
        if show_edges:
            plotter.add_mesh(
                dataset,
                style="wireframe",
                color="black",
                opacity=0.5,
                line_width=1.0,
                name="geodesic-grid",
            )
    plotter.add_mesh(
        dataset,
        scalars=name,
        clim=(-limit, limit),
        cmap=cmap,
        show_edges=show_edges and not earth_texture,
        lighting=not earth_texture,
        opacity=(
            _field_opacity_transfer(field_opacity) if earth_texture else 1.0
        ),
        scalar_bar_args={"title": name},
        name="field-overlay",
    )
    if earth_texture:
        _add_source_marker(plotter, pv, simulation, altitude_m, radial_exaggeration)
    plotter.add_text(
        f"t = {_field_time_s(simulation, component):.6g} s",
        font_size=10,
        name="simulation-time",
    )
    _set_source_camera(plotter, simulation, altitude_m, radial_exaggeration)
    if output_path.suffix.lower() == ".gif":
        plotter.open_gif(str(output_path), fps=frames_per_second)
    else:
        plotter.open_movie(str(output_path), framerate=frames_per_second)

    try:
        for frame in range(frames):
            if frame:
                simulation.step(steps_per_frame)
                updated, _, _ = _surface_values(
                    simulation, component, altitude_m
                )
                dataset.cell_data[name] = updated
                dataset.Modified()
                plotter.add_text(
                    f"t = {_field_time_s(simulation, component):.6g} s",
                    font_size=10,
                    name="simulation-time",
                )
            plotter.write_frame()
    finally:
        plotter.close()
    return output_path


def geographic_direction(latitude_deg: float, longitude_deg: float) -> FloatArray:
    """Convert geographic latitude/longitude to a Cartesian unit vector."""

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


def directions_to_lon_lat(directions: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Convert Cartesian unit vectors to longitude and latitude in degrees."""

    longitude = np.rad2deg(np.arctan2(directions[:, 1], directions[:, 0]))
    latitude = np.rad2deg(
        np.arctan2(
            directions[:, 2],
            np.hypot(directions[:, 0], directions[:, 1]),
        )
    )
    return longitude, latitude


def _surface_values(
    simulation: GeodesicFDTD, component: str, altitude_m: float
) -> tuple[FloatArray, float, str]:
    component = component.lower()
    if component == "er":
        index = int(np.argmin(np.abs(simulation.altitudes_m - altitude_m)))
        return (
            simulation.to_numpy(simulation.er[:, index]),
            float(simulation.altitudes_m[index]),
            "point",
        )
    if component == "hr":
        index = int(
            np.argmin(np.abs(simulation.radial_midpoint_altitudes_m - altitude_m))
        )
        return (
            simulation.to_numpy(simulation.hr[:, index]),
            float(simulation.radial_midpoint_altitudes_m[index]),
            "cell",
        )
    raise ValueError("surface component must be 'er' or 'hr'")


def _field_time_s(simulation: GeodesicFDTD, component: str) -> float:
    """Return the staggered time associated with an electric or magnetic field."""

    return (
        simulation.magnetic_time_s
        if component.lower().startswith("h")
        else simulation.electric_time_s
    )


def _great_circle_directions(
    start: FloatArray, end: FloatArray, samples: int
) -> tuple[FloatArray, float]:
    dot = float(np.clip(start @ end, -1.0, 1.0))
    arc = float(np.arccos(dot))
    if arc < 1.0e-14:
        return np.repeat(start[None, :], samples, axis=0), 0.0
    tangent = end - dot * start
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm < 1.0e-12:
        reference = np.asarray((1.0, 0.0, 0.0))
        if abs(float(start @ reference)) > 0.9:
            reference = np.asarray((0.0, 1.0, 0.0))
        tangent = reference - (reference @ start) * start
        tangent_norm = float(np.linalg.norm(tangent))
    tangent /= tangent_norm
    angles = np.linspace(0.0, arc, samples)
    directions = (
        np.cos(angles)[:, None] * start
        + np.sin(angles)[:, None] * tangent
    )
    return directions, arc


def _inverse_distance_sample(
    targets: FloatArray,
    source_directions: FloatArray,
    source_values: FloatArray,
    neighbors: int,
) -> FloatArray:
    distances = np.arccos(
        np.clip(targets @ source_directions.T, -1.0, 1.0)
    )
    indices = np.argpartition(distances, neighbors - 1, axis=1)[:, :neighbors]
    selected_distances = np.take_along_axis(distances, indices, axis=1)
    weights = 1.0 / np.maximum(selected_distances, 1.0e-12) ** 2
    exact = selected_distances < 1.0e-12
    for row in np.flatnonzero(np.any(exact, axis=1)):
        weights[row] = exact[row].astype(np.float64)
    weights /= weights.sum(axis=1, keepdims=True)
    selected_values = source_values[indices]
    return np.einsum("sk,skr->sr", weights, selected_values)


def _inverse_distance_sample_chunked(
    targets: FloatArray,
    source_directions: FloatArray,
    source_values: FloatArray,
    neighbors: int,
    *,
    chunk_size: int = 2_048,
) -> FloatArray:
    chunks = [
        _inverse_distance_sample(
            targets[start : start + chunk_size],
            source_directions,
            source_values,
            neighbors,
        )
        for start in range(0, len(targets), chunk_size)
    ]
    return np.concatenate(chunks, axis=0)


def _receiver_location(
    simulation: GeodesicFDTD, receiver: Receiver
) -> tuple[int, int]:
    direction = geographic_direction(receiver.latitude_deg, receiver.longitude_deg)
    vertex = int(np.argmax(simulation.mesh.vertices @ direction))
    layer = int(np.argmin(np.abs(simulation.altitudes_m - receiver.altitude_m)))
    return vertex, layer


def _ordered_incident_faces(mesh: GeodesicMesh) -> list[NDArray[np.int64]]:
    incident: list[list[int]] = [[] for _ in range(mesh.n_vertices)]
    for face_index, face in enumerate(mesh.faces):
        for vertex_index in face:
            incident[int(vertex_index)].append(face_index)
    result: list[NDArray[np.int64]] = []
    for vertex, face_indices in zip(mesh.vertices, incident, strict=True):
        reference = np.asarray((1.0, 0.0, 0.0))
        if abs(float(vertex @ reference)) > 0.9:
            reference = np.asarray((0.0, 1.0, 0.0))
        tangent_x = reference - (reference @ vertex) * vertex
        tangent_x /= np.linalg.norm(tangent_x)
        tangent_y = np.cross(vertex, tangent_x)
        centers = mesh.face_centers[face_indices]
        tangent = centers - (centers @ vertex)[:, None] * vertex
        angles = np.arctan2(tangent @ tangent_y, tangent @ tangent_x)
        result.append(np.asarray(face_indices, dtype=np.int64)[np.argsort(angles)])
    return result


def _dual_polydata(pv: Any, mesh: GeodesicMesh, radius_km: float) -> Any:
    polygons = _ordered_incident_faces(mesh)
    vtk_faces = np.concatenate(
        [np.concatenate(([len(face)], face)) for face in polygons]
    ).astype(np.int64)
    return pv.PolyData(radius_km * mesh.face_centers, vtk_faces)


def _add_earth_underlay(
    plotter: Any,
    simulation: GeodesicFDTD,
    altitude_m: float,
    radial_exaggeration: float,
) -> Any:
    """Add PyVista's bundled day-map just inside the field surface."""

    from pyvista import examples

    displayed_radius_km = (
        simulation.config.earth_radius_m + radial_exaggeration * altitude_m
    ) / 1.0e3
    physical_radius_km = simulation.config.earth_radius_m / 1.0e3
    target_radius_km = 0.998 * min(displayed_radius_km, physical_radius_km)
    globe = examples.load_globe().copy(deep=True)
    source_radius = float(np.linalg.norm(globe.points[0]))
    globe.points *= target_radius_km / source_radius
    texture = examples.load_globe_texture()
    plotter.add_mesh(
        globe,
        texture=texture,
        smooth_shading=True,
        ambient=0.35,
        diffuse=0.65,
        specular=0.05,
        name="earth-texture",
    )
    return globe


def _set_source_camera(
    plotter: Any,
    simulation: GeodesicFDTD,
    altitude_m: float,
    radial_exaggeration: float,
) -> None:
    """Start with the current source location facing the viewer."""

    if simulation.source is None:
        plotter.camera_position = "iso"
        return
    direction = simulation.source.direction()
    radius_km = (
        simulation.config.earth_radius_m + radial_exaggeration * altitude_m
    ) / 1.0e3
    view_up = np.asarray((0.0, 0.0, 1.0))
    if abs(float(direction @ view_up)) > 0.95:
        view_up = np.asarray((0.0, 1.0, 0.0))
    plotter.camera_position = (
        4.2 * radius_km * direction,
        (0.0, 0.0, 0.0),
        view_up,
    )


def _add_source_marker(
    plotter: Any,
    pv: Any,
    simulation: GeodesicFDTD,
    altitude_m: float,
    radial_exaggeration: float,
) -> Any | None:
    """Mark the exact geographic source location above the field layer."""

    if simulation.source is None:
        return None
    radius_km = (
        simulation.config.earth_radius_m + radial_exaggeration * altitude_m
    ) / 1.0e3
    center = 1.006 * radius_km * simulation.source.direction()
    marker = pv.Sphere(
        radius=0.012 * radius_km,
        center=center,
        theta_resolution=24,
        phi_resolution=16,
    )
    plotter.add_mesh(
        marker,
        color="yellow",
        lighting=False,
        name="source-location",
    )
    return marker


def _field_opacity_transfer(maximum_opacity: float) -> NDArray[np.uint8]:
    """Return a symmetric opacity ramp that reveals the globe near zero."""

    normalized_field = np.linspace(-1.0, 1.0, 256)
    opacity = maximum_opacity * np.abs(normalized_field) ** 0.55
    opacity[np.abs(normalized_field) < 0.01] = 0.0
    return np.rint(255.0 * opacity).astype(np.uint8)


def _color_limit(values: FloatArray, requested: float | None) -> float:
    if requested is not None:
        if requested <= 0.0:
            raise ValueError("color_limit must be positive")
        return float(requested)
    maximum = float(np.nanmax(np.abs(values))) if values.size else 0.0
    return maximum if maximum > 0.0 else 1.0


def _live_status(
    simulation: GeodesicFDTD,
    component: str,
    values: FloatArray,
    color_limit: float,
    automatic_scale: bool,
) -> str:
    maximum = float(np.nanmax(np.abs(values))) if values.size else 0.0
    scale_label = "auto" if automatic_scale else "fixed"
    return (
        f"step {simulation.steps:,}   "
        f"t = {_field_time_s(simulation, component):.6g} s\n"
        f"max |field| = {maximum:.4g}   {scale_label} scale = ±{color_limit:.4g}\n"
        "Drag: rotate   Wheel: zoom   q: stop"
    )


def _symmetric_norm(
    colors: Any,
    values: FloatArray,
    color_limit: float | None,
    scale: str,
    symlog_linthresh: float | None,
) -> Any:
    limit = _color_limit(values, color_limit)
    if scale == "linear":
        return colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    if scale == "symlog":
        threshold = symlog_linthresh or limit * 1.0e-3
        if threshold <= 0.0:
            raise ValueError("symlog_linthresh must be positive")
        return colors.SymLogNorm(
            linthresh=threshold, vmin=-limit, vmax=limit, base=10
        )
    raise ValueError("scale must be 'linear' or 'symlog'")


def _projection(ccrs: Any, name: str) -> Any:
    projections = {
        "mollweide": ccrs.Mollweide,
        "robinson": ccrs.Robinson,
        "platecarree": ccrs.PlateCarree,
        "orthographic": ccrs.Orthographic,
    }
    try:
        factory = projections[name.lower()]
    except KeyError as error:
        raise ValueError(
            "projection must be mollweide, robinson, platecarree, or orthographic"
        ) from error
    return factory()


def _matplotlib_modules() -> tuple[Any, Any, Any]:
    try:
        import cartopy.crs as ccrs
        import matplotlib.colors as colors
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise VisualizationDependencyError(
            "install the visualization extra: uv sync --extra visualization"
        ) from error
    return plt, colors, ccrs


def _pyvista_module() -> Any:
    try:
        import pyvista as pv
    except ImportError as error:
        raise VisualizationDependencyError(
            "install the visualization extra: uv sync --extra visualization"
        ) from error
    return pv
