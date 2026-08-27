import os
from pathlib import Path
import tempfile

import pytest


os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ionosphere-matplotlib")
)
matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("cartopy")
matplotlib.use("Agg")

from ionosphere_fdtd.cli import main as simulation_main
from ionosphere_fdtd.solver import GeodesicFDTD
from ionosphere_fdtd.viz_cli import main as visualization_main


CONFIG = Path(__file__).parents[1] / "configs" / "ionosphere.example.toml"
ROOT = Path(__file__).parents[1]


def test_documented_starter_creates_checkpoint_and_surface_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)

    assert simulation_main(["--config", str(CONFIG)]) == 0
    checkpoint = tmp_path / "artifacts/runs/demo.npz"
    assert checkpoint.exists()
    assert GeodesicFDTD.load_checkpoint(checkpoint).steps == 200

    assert visualization_main(["--config", str(CONFIG), "surface"]) == 0
    image = tmp_path / "artifacts/figures/demo-surface.png"
    assert image.stat().st_size > 0

    output = capsys.readouterr().out
    assert "runtime=torch device=cpu dtype=float64" in output
    assert f"checkpoint={checkpoint.relative_to(tmp_path)} loaded_step=200" in output


def test_pytorch_only_migration_contract_is_documented() -> None:
    migration = (ROOT / "docs/manual/pytorch-only-migration.md").read_text()
    for text in (
        "Python | 3.11",
        "NumPy | 2.0",
        "PyTorch | 2.5",
        "ArrayBackend",
        "NumPyBackend",
        "TorchBackend",
        "create_backend",
        "--backend",
        "[ionosphere].backend",
        "[visualization].backend",
        "simulation.backend.name",
        "persistent_backend_bytes",
        "record_er_observations()",
        "simulation.to_numpy(values)",
        "Version | Behavior",
    ):
        assert text in migration

    current_guides = "\n".join(
        path.read_text()
        for path in (
            ROOT / "README.md",
            ROOT / "docs/manual/installation.md",
            ROOT / "docs/manual/quickstart.md",
            ROOT / "docs/manual/troubleshooting.md",
        )
    )
    for stale_promise in (
        "Optional PyTorch",
        "default uses NumPy",
        "starter remains on NumPy",
        "prefer NumPy when",
        "minimal NumPy CPU installation",
    ):
        assert stale_promise not in current_guides
