import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ionosphere-matplotlib")
)
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")
matplotlib.use("Agg")

import numpy as np

from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent
from ionosphere_fdtd.visualization import (
    Receiver,
    animate_surface_field,
    directions_to_lon_lat,
    geographic_direction,
    make_pyvista_surface,
    plot_radial_section,
    plot_receiver_traces,
    plot_surface_field,
    record_receiver_traces,
    run_live_surface,
    sample_radial_section,
)
from ionosphere_fdtd.viz_cli import _parse_args as parse_visualization_args
from ionosphere_fdtd.viz_cli import main as visualization_main


@pytest.fixture
def simulation() -> GeodesicFDTD:
    result = GeodesicFDTD(
        SimulationConfig(subdivision=1, radial_cells=6, courant_factor=0.2),
        source=GaussianCurrent(peak_current_a=1.0e6),
    )
    result.step(50)
    return result


def test_coordinate_round_trip() -> None:
    direction = geographic_direction(37.5, -122.4)
    longitude, latitude = directions_to_lon_lat(direction[None, :])
    assert latitude[0] == pytest.approx(37.5)
    assert longitude[0] == pytest.approx(-122.4)


def test_surface_field_renders_headlessly(simulation: GeodesicFDTD) -> None:
    figure, ax, artist = plot_surface_field(simulation, "er")
    figure.canvas.draw()
    assert ax.get_title().startswith("ER")
    assert artist.get_array().size > simulation.mesh.n_vertices


def test_visualization_cli_renders_saved_checkpoint_without_advancing(
    tmp_path: Path,
    simulation: GeodesicFDTD,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint = simulation.save_checkpoint(tmp_path / "state.npz")
    output = tmp_path / "surface.png"

    args = parse_visualization_args(
        ["--resume", str(checkpoint), "surface", "--output", str(output)]
    )
    assert args.steps == 0
    assert visualization_main(
        ["--resume", str(checkpoint), "surface", "--output", str(output)]
    ) == 0

    text = capsys.readouterr().out
    assert f"checkpoint={checkpoint} loaded_step={simulation.steps}" in text
    assert output.stat().st_size > 0


def test_visualization_new_model_retains_default_warmup(tmp_path: Path) -> None:
    args = parse_visualization_args(
        ["surface", "--output", str(tmp_path / "surface.png")]
    )

    assert args.steps == 100


def test_magnetic_surface_uses_half_step_timestamp(
    simulation: GeodesicFDTD,
) -> None:
    figure, ax, _ = plot_surface_field(simulation, "hr")
    figure.canvas.draw()

    assert f"t = {simulation.magnetic_time_s:.6g} s" in ax.get_title()
    assert f"t = {simulation.electric_time_s:.6g} s" not in ax.get_title()


def test_surface_field_rejects_zero_neighbors(simulation: GeodesicFDTD) -> None:
    with pytest.raises(ValueError, match="neighbors must be positive"):
        plot_surface_field(simulation, neighbors=0)


def test_radial_section_sampling_and_plot(simulation: GeodesicFDTD) -> None:
    section = sample_radial_section(simulation, 0.0, -47.0, 0.0, 43.0, samples=31)
    assert section.values.shape == (len(simulation.altitudes_m), 31)
    assert np.all(np.diff(section.distance_m) > 0.0)
    figure, _, artist = plot_radial_section(section)
    figure.canvas.draw()
    assert artist.get_array().size == section.values.size


def test_receiver_trace_recording_and_plot(simulation: GeodesicFDTD) -> None:
    starting_step = simulation.steps
    traces = record_receiver_traces(
        simulation,
        (Receiver(0.0, -47.0, label="source"), Receiver(0.0, 43.0)),
        10,
        sample_every=4,
    )
    assert simulation.steps == starting_step + 10
    assert traces.er_v_m.shape == (4, 2)
    assert traces.labels[0] == "source"
    figure, ax = plot_receiver_traces(traces)
    figure.canvas.draw()
    assert len(ax.lines) == 3


def test_pyvista_dual_and_primal_surface_associations(
    simulation: GeodesicFDTD,
) -> None:
    pytest.importorskip("pyvista")
    dual, er_name, er_values = make_pyvista_surface(simulation, "er")
    primal, hr_name, hr_values = make_pyvista_surface(simulation, "hr")
    assert dual.n_cells == simulation.mesh.n_vertices
    assert primal.n_cells == simulation.mesh.n_faces
    assert np.allclose(dual.cell_data[er_name], er_values)
    assert np.allclose(primal.cell_data[hr_name], hr_values)


def test_live_surface_advances_and_updates_without_rendering(
    simulation: GeodesicFDTD, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ionosphere_fdtd.visualization as visualization

    class FakeDataset:
        def __init__(self, values: np.ndarray) -> None:
            self.cell_data = {"Er (V/m)": values.copy()}
            self.modified = 0

        def Modified(self) -> None:
            self.modified += 1

    class FakePlotter:
        last_instance: "FakePlotter | None" = None

        def __init__(self) -> None:
            FakePlotter.last_instance = self
            self.actor = SimpleNamespace(
                mapper=SimpleNamespace(scalar_range=None)
            )
            self.callback = None
            self.max_steps = 0
            self.text = ""
            self.camera_position = None

        def add_mesh(self, *_args: object, **_kwargs: object) -> object:
            return self.actor

        def add_axes(self) -> None:
            pass

        def add_text(self, text: str, **_kwargs: object) -> None:
            self.text = text

        def add_timer_event(
            self, max_steps: int, duration: int, callback: object
        ) -> None:
            assert duration == 50
            self.max_steps = max_steps
            self.callback = callback

        def show(self, **_kwargs: object) -> None:
            assert self.callback is not None
            for frame in range(self.max_steps):
                self.callback(frame)

    surface_index = int(np.argmin(np.abs(simulation.altitudes_m)))
    initial = simulation.to_numpy(simulation.er[:, surface_index]).copy()
    dataset = FakeDataset(initial)
    monkeypatch.setattr(
        visualization,
        "_pyvista_module",
        lambda: SimpleNamespace(Plotter=FakePlotter),
    )
    monkeypatch.setattr(
        visualization,
        "make_pyvista_surface",
        lambda *_args, **_kwargs: (dataset, "Er (V/m)", initial),
    )
    starting_step = simulation.steps

    completed = run_live_surface(
        simulation,
        steps_per_frame=3,
        frames_per_second=20,
        max_frames=2,
        earth_texture=False,
    )

    plotter = FakePlotter.last_instance
    assert plotter is not None
    assert completed == 2
    assert simulation.steps == starting_step + 6
    assert dataset.modified == 2
    assert np.allclose(
        dataset.cell_data["Er (V/m)"], simulation.to_numpy(simulation.er[:, surface_index])
    )
    assert plotter.actor.mapper.scalar_range[0] < 0.0
    assert "simulation step" not in plotter.text
    assert f"step {simulation.steps:,}" in plotter.text


def test_live_surface_validates_update_rate(simulation: GeodesicFDTD) -> None:
    with pytest.raises(ValueError, match="steps_per_frame must be positive"):
        run_live_surface(simulation, steps_per_frame=0)


def test_field_opacity_transfer_uses_vtk_alpha_range() -> None:
    from ionosphere_fdtd.visualization import _field_opacity_transfer

    opacity = _field_opacity_transfer(0.8)
    assert opacity.dtype == np.uint8
    assert opacity[0] == pytest.approx(204, abs=1)
    assert opacity[-1] == pytest.approx(204, abs=1)
    assert opacity[127] == 0
    assert opacity[128] == 0


@pytest.mark.skipif(
    os.environ.get("IONOSPHERE_TEST_PYVISTA_RENDER") != "1",
    reason="requires a working OpenGL render context",
)
def test_short_gif_animation(tmp_path, simulation: GeodesicFDTD) -> None:
    pytest.importorskip("pyvista")
    output = animate_surface_field(
        simulation,
        tmp_path / "field.gif",
        frames=2,
        steps_per_frame=1,
        color_limit=10.0,
    )
    assert output.stat().st_size > 0
