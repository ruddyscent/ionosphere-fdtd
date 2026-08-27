import numpy as np
import pytest
import torch

from ionosphere_fdtd.radial_grid import (
    RadialRefinementRegion,
    build_refined_radial_grid,
    validate_radial_grid,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig


def test_refined_radial_grid_resolves_region_with_2to1_balance() -> None:
    region = RadialRefinementRegion(60_000.0, 90_000.0, 1_000.0)

    altitudes = build_refined_radial_grid(
        0.0, 100_000.0, 10_000.0, (region,)
    )
    validation = validate_radial_grid(altitudes)
    steps = np.diff(altitudes)
    lower = np.asarray(altitudes[:-1])
    upper = np.asarray(altitudes[1:])
    intersects = (upper > region.minimum_altitude_m) & (
        lower < region.maximum_altitude_m
    )

    assert altitudes[0] == 0.0
    assert altitudes[-1] == 100_000.0
    assert validation.cells == 61
    assert validation.maximum_adjacent_step_ratio <= 2.0
    assert np.max(steps[intersects]) <= region.maximum_step_m
    assert validation.cells < 100


def test_overlapping_radial_regions_use_finest_requested_step() -> None:
    regions = (
        RadialRefinementRegion(40_000.0, 80_000.0, 2_000.0),
        RadialRefinementRegion(60_000.0, 70_000.0, 500.0),
    )

    altitudes = build_refined_radial_grid(
        0.0, 100_000.0, 10_000.0, regions
    )
    steps = np.diff(altitudes)
    lower = np.asarray(altitudes[:-1])
    upper = np.asarray(altitudes[1:])

    for region in regions:
        intersects = (upper > region.minimum_altitude_m) & (
            lower < region.maximum_altitude_m
        )
        assert np.max(steps[intersects]) <= region.maximum_step_m
    assert validate_radial_grid(altitudes).maximum_adjacent_step_ratio <= 2.0


def test_balanced_radial_grid_runs_stably_at_cfl_limit() -> None:
    altitudes = build_refined_radial_grid(
        0.0,
        100_000.0,
        10_000.0,
        (RadialRefinementRegion(60_000.0, 90_000.0, 2_000.0),),
    )
    simulation = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=len(altitudes) - 1,
            minimum_altitude_m=altitudes[0],
            maximum_altitude_m=altitudes[-1],
            radial_altitudes_m=altitudes,
            radial_grid_policy="balanced-2to1",
            courant_factor=1.0,
        ),
        dtype="float64",
    )
    generator = np.random.default_rng(20260820)
    simulation.er.copy_(torch.as_tensor(generator.standard_normal(simulation.er.shape)))
    simulation.et.copy_(torch.as_tensor(generator.standard_normal(simulation.et.shape)))
    initial = max(np.max(np.abs(simulation.to_numpy(simulation.er))), np.max(np.abs(simulation.to_numpy(simulation.et))))

    simulation.step(1_000)

    maximum = max(np.max(np.abs(simulation.to_numpy(simulation.er))), np.max(np.abs(simulation.to_numpy(simulation.et))))
    assert np.isfinite(maximum)
    assert maximum < 2.0 * initial


def test_balanced_policy_rejects_larger_spacing_jump() -> None:
    with pytest.raises(ValueError, match="adjacent-step ratio"):
        SimulationConfig(
            subdivision=0,
            radial_cells=3,
            minimum_altitude_m=0.0,
            maximum_altitude_m=7_000.0,
            radial_altitudes_m=(0.0, 1_000.0, 2_000.0, 7_000.0),
            radial_grid_policy="balanced-2to1",
        )


def test_radial_refinement_rejects_region_outside_domain() -> None:
    with pytest.raises(ValueError, match="outside"):
        build_refined_radial_grid(
            0.0,
            100_000.0,
            10_000.0,
            (RadialRefinementRegion(-1.0, 1_000.0, 100.0),),
        )
