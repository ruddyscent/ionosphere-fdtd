"""Render throughput, setup-time, and memory curves from scaling JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


COLORS = {
    "torch-cpu-eager": "#f58518",
    "cuda-eager": "#54a24b",
    "cuda-compiled": "#e45756",
    "mps-eager": "#4c78a8",
    "mps-compiled": "#b279a2",
}
LABELS = {
    "torch-cpu-eager": "Torch CPU",
    "cuda-eager": "CUDA eager",
    "cuda-compiled": "CUDA compiled (chunk 32)",
    "mps-eager": "MPS eager",
    "mps-compiled": "MPS compiled (chunk 32)",
}


def load_results(paths: tuple[Path, ...]) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        results.extend(json.loads(path.read_text())["results"])
    return [
        result
        for result in results
        if result["status"] == "ok" and result.get("workload", "bare") == "bare"
    ]


def _series_key(result: dict[str, Any]) -> str:
    return f"{result['implementation']}-{result['mode']}"


def _faceted_plot(
    results: list[dict[str, Any]],
    output: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
    value_transform=lambda value: value,
) -> None:
    import matplotlib.pyplot as plt

    dtypes = ("float32", "float64")
    radial_cells = (16, 40, 80)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for row, dtype in enumerate(dtypes):
        for column, radial in enumerate(radial_cells):
            axis = axes[row, column]
            selected = [
                result
                for result in results
                if result["dtype"] == dtype and result["radial_cells"] == radial
            ]
            for key in LABELS:
                series = sorted(
                    (result for result in selected if _series_key(result) == key),
                    key=lambda result: result["subdivision"],
                )
                values = [result.get(metric) for result in series]
                points = [
                    (result["subdivision"], value_transform(value))
                    for result, value in zip(series, values, strict=True)
                    if value is not None
                ]
                if points:
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        linewidth=2,
                        color=COLORS[key],
                        label=LABELS[key],
                    )
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(f"{dtype}, radial cells {radial}")
            axis.set_xticks(range(2, 8))
            if row == 1:
                axis.set_xlabel("Subdivision")
            if column == 0:
                axis.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle(title, y=0.99, fontsize=15)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def _setup_plot(results: list[dict[str, Any]], output: Path) -> None:
    import matplotlib.pyplot as plt

    dtypes = ("float32", "float64")
    radial_cells = (16, 40, 80)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for row, dtype in enumerate(dtypes):
        for column, radial in enumerate(radial_cells):
            axis = axes[row, column]
            selected = [
                result
                for result in results
                if result["dtype"] == dtype and result["radial_cells"] == radial
            ]
            for key in LABELS:
                series = sorted(
                    (result for result in selected if _series_key(result) == key),
                    key=lambda result: result["subdivision"],
                )
                metric = (
                    "cold_compile_seconds"
                    if key.endswith("-compiled")
                    else "initialization_seconds"
                )
                points = [
                    (result["subdivision"], result[metric])
                    for result in series
                    if result.get(metric) is not None
                ]
                if points:
                    label = (
                        f"{LABELS[key]} cold compile"
                        if key.endswith("-compiled")
                        else f"{LABELS[key]} initialization"
                    )
                    axis.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        marker="o",
                        linewidth=2,
                        color=COLORS[key],
                        label=label,
                    )
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.25)
            axis.set_title(f"{dtype}, radial cells {radial}")
            axis.set_xticks(range(2, 8))
            if row == 1:
                axis.set_xlabel("Subdivision")
            if column == 0:
                axis.set_ylabel("Seconds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.suptitle("Initialization and cold-compile time", y=0.99, fontsize=15)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, facecolor="white")
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    results = load_results(tuple(args.inputs))
    _faceted_plot(
        results,
        args.output_directory / "runtime-scaling-throughput.png",
        metric="steps_per_second",
        ylabel="Steady-state steps/s",
        title="PyTorch steady-state throughput by problem size",
    )
    _setup_plot(results, args.output_directory / "runtime-scaling-setup-time.png")
    _faceted_plot(
        results,
        args.output_directory / "runtime-scaling-persistent-memory.png",
        metric="persistent_memory_bytes",
        ylabel="Persistent solver memory (GiB)",
        title="Persistent runtime memory by problem size",
        value_transform=lambda value: value / 2**30,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
