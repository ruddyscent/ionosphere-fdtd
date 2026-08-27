import numpy as np
import pytest
import torch

from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.surface_impedance import (
    ConductiveHalfSpaceSurface,
    SurfaceImpedanceADE,
)


class _ArrayRuntime:
    def as_tensor(self, values):
        return np.asarray(values, dtype=np.float64)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def nbytes(self, values):
        return values.nbytes


def _surface_config(**overrides) -> SimulationConfig:
    values = {
        "subdivision": 0,
        "radial_cells": 4,
        "minimum_altitude_m": 0.0,
        "maximum_altitude_m": 100_000.0,
        "courant_factor": 0.2,
        "radial_boundary_condition": "surface-impedance",
    }
    values.update(overrides)
    return SimulationConfig(**values)


def test_diffusive_fit_matches_passive_halfspace_over_target_band() -> None:
    model = ConductiveHalfSpaceSurface(1.0 / 50.0)
    frequency = np.geomspace(5.0, 45.0, 301)

    fitted = model.impedance_ohm(frequency)[0]
    exact = model.exact_impedance_ohm(frequency)[0]

    assert np.max(np.abs(fitted / exact - 1.0)) < 6.0e-3
    assert np.all(fitted.real > 0.0)
    assert np.all(fitted.imag > 0.0)
    assert np.all(model.pole_rates_s_inv > 0.0)
    assert np.all(model.diffusive_weights_sqrt_s_inv > 0.0)


def test_spatial_conductivity_scales_surface_impedance() -> None:
    model = ConductiveHalfSpaceSurface(np.asarray((4.0, 1.0)))
    impedance = model.impedance_ohm(20.0)[:, 0]

    assert impedance[1] / impedance[0] == pytest.approx(2.0)


def test_ade_harmonic_response_is_causal_passive_and_accurate() -> None:
    model = ConductiveHalfSpaceSurface(1.0 / 50.0)
    time_step = 1.0e-4
    frequency = 20.0
    ade = SurfaceImpedanceADE(
        model,
        edge_count=1,
        time_step_s=time_step,
        runtime=_ArrayRuntime(),
    )
    samples = 50_000
    electric = np.empty(samples)
    magnetic = np.empty(samples)
    for step in range(samples):
        time = step * time_step
        h_old = np.asarray(
            (np.cos(2.0 * np.pi * frequency * (time - 0.5 * time_step)),)
        )
        h_new = np.asarray(
            (np.cos(2.0 * np.pi * frequency * (time + 0.5 * time_step)),)
        )
        magnetic[step] = 0.5 * (h_old[0] + h_new[0])
        electric[step] = ade.advance_prescribed_magnetic(h_old, h_new)[0]

    selected = slice(samples - 10_000, None)
    time = np.arange(samples)[selected] * time_step
    basis = np.exp(-2j * np.pi * frequency * time)
    measured = -np.sum(electric[selected] * basis) / np.sum(
        magnetic[selected] * basis
    )
    target = model.impedance_ohm(frequency)[0, 0]
    assert measured == pytest.approx(target, rel=3.0e-4)
    assert np.mean(-electric[selected] * magnetic[selected]) > 0.0


def test_surface_impedance_requires_ground_boundary_and_model() -> None:
    with pytest.raises(ValueError, match="minimum_altitude_m=0"):
        _surface_config(minimum_altitude_m=-1.0)
    with pytest.raises(ValueError, match="requires a surface model"):
        GeodesicFDTD(_surface_config())
    with pytest.raises(ValueError, match="radial_boundary_condition"):
        GeodesicFDTD(
            SimulationConfig(
                subdivision=0,
                radial_cells=4,
                minimum_altitude_m=0.0,
                maximum_altitude_m=100_000.0,
            ),
            surface_impedance=ConductiveHalfSpaceSurface(0.02),
        )


def test_high_conductivity_surface_converges_to_lower_pec_curl() -> None:
    pec = GeodesicFDTD(
        SimulationConfig(
            subdivision=0,
            radial_cells=4,
            minimum_altitude_m=0.0,
            maximum_altitude_m=100_000.0,
            courant_factor=0.2,
        )
    )
    impedance = GeodesicFDTD(
        _surface_config(),
        surface_impedance=ConductiveHalfSpaceSurface(1.0e30),
    )
    generator = np.random.default_rng(20260820)
    er = generator.standard_normal(pec.er.shape)
    et = generator.standard_normal(pec.et.shape)
    ht = generator.standard_normal(pec.ht.shape)
    for simulation in (pec, impedance):
        simulation.er.copy_(torch.as_tensor(er))
        simulation.et.copy_(torch.as_tensor(et))
        simulation.ht.copy_(torch.as_tensor(ht))

    pec._update_magnetic_fields()
    impedance._update_magnetic_fields()

    np.testing.assert_allclose(impedance.ht, pec.ht, rtol=2.0e-14, atol=2.0e-15)


def test_surface_impedance_solver_remains_bounded() -> None:
    simulation = GeodesicFDTD(
        _surface_config(courant_factor=0.35),
        surface_impedance=ConductiveHalfSpaceSurface(1.0 / 50.0),
    )
    generator = np.random.default_rng(20260820)
    for name in ("er", "et"):
        getattr(simulation, name).copy_(
            torch.as_tensor(
                377.0e-9
                * generator.standard_normal(getattr(simulation, name).shape)
            )
        )
    for name in ("hr", "ht"):
        getattr(simulation, name).copy_(
            torch.as_tensor(
                1.0e-9 * generator.standard_normal(getattr(simulation, name).shape)
            )
        )
    initial_e = max(torch.max(torch.abs(simulation.er)).item(), torch.max(torch.abs(simulation.et)).item())
    initial_h = max(torch.max(torch.abs(simulation.hr)).item(), torch.max(torch.abs(simulation.ht)).item())

    simulation.step(500)

    maximum_e = max(torch.max(torch.abs(simulation.er)).item(), torch.max(torch.abs(simulation.et)).item())
    maximum_h = max(torch.max(torch.abs(simulation.hr)).item(), torch.max(torch.abs(simulation.ht)).item())
    assert np.isfinite(maximum_e)
    assert np.isfinite(maximum_h)
    assert maximum_e < 2.0 * initial_e
    assert maximum_h < 2.0 * initial_h
    assert simulation._surface_impedance_ade.state_bytes > 0


def test_torch_compiled_surface_impedance_matches_eager() -> None:
    model = ConductiveHalfSpaceSurface(1.0 / 50.0)
    source = None
    eager = GeodesicFDTD(
        _surface_config(),
        source=source,
        surface_impedance=model,
        device="cpu",
        dtype="float64",
    )
    compiled = GeodesicFDTD(
        _surface_config(),
        source=source,
        surface_impedance=model,
        device="cpu",
        dtype="float64",
        compile_step=True,
        compile_chunk_size=2,
    )
    generator = np.random.default_rng(20260820)
    for name in ("er", "et", "hr", "ht"):
        values = generator.standard_normal(getattr(eager, name).shape) * 1.0e-9
        getattr(eager, name).copy_(eager._runtime.as_tensor(values))
        getattr(compiled, name).copy_(compiled._runtime.as_tensor(values))

    eager.step(4)
    compiled.step(4)

    for name in ("er", "et", "hr", "ht"):
        np.testing.assert_allclose(
            compiled.to_numpy(getattr(compiled, name)),
            eager.to_numpy(getattr(eager, name)),
            rtol=2.0e-13,
            atol=1.0e-22,
        )
    np.testing.assert_allclose(
        compiled.to_numpy(compiled._surface_impedance_ade.memory),
        eager.to_numpy(eager._surface_impedance_ade.memory),
        rtol=2.0e-13,
        atol=1.0e-22,
    )
