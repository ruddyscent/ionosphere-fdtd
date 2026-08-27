import inspect
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from ionosphere_fdtd.distributed import DistributedGeodesicFDTD
from ionosphere_fdtd.mesh import build_geodesic_mesh
from ionosphere_fdtd.partition import partition_surface_mesh
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig
from ionosphere_fdtd.sources import GaussianCurrent
from ionosphere_fdtd.surface_impedance import ConductiveHalfSpaceSurface


def _distributed_worker(
    rank: int,
    rendezvous: str,
    output: str,
    backend: str = "gloo",
    cuda_graph_chunk_size: int = 0,
    use_surface_impedance: bool = False,
) -> None:
    import torch.distributed as distributed

    if backend == "nccl":
        os.environ.setdefault("NCCL_GRAPH_MIXING_SUPPORT", "0")
        torch.cuda.set_device(rank)
        device_id = torch.device("cuda", rank)
    else:
        device_id = None
    distributed.init_process_group(
        backend,
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        device_id=device_id,
    )
    simulation = None
    try:
        mesh = build_geodesic_mesh(0)
        partition = partition_surface_mesh(mesh)
        config = SimulationConfig(
            subdivision=0,
            radial_cells=4,
            minimum_altitude_m=0.0 if use_surface_impedance else -100_000.0,
            maximum_altitude_m=100_000.0,
            courant_factor=0.2,
            radial_boundary_condition=(
                "surface-impedance" if use_surface_impedance else "pec"
            ),
        )
        source = GaussianCurrent(peak_current_a=1.0e6)
        surface = (
            ConductiveHalfSpaceSurface(1.0 / 50.0)
            if use_surface_impedance
            else None
        )
        simulation = DistributedGeodesicFDTD(
            partition,
            config=config,
            mesh=mesh,
            source=source,
            surface_impedance=surface,
            device="cpu" if backend == "gloo" else f"cuda:{rank}",
            dtype="float64",
        )
        float_tensors = (
            "er",
            "et",
            "hr",
            "ht",
            "_face_edge_signs",
            "_face_primal_edge_angles",
            "_primal_edge_angles",
            "_inverse_primal_edge_angles",
            "_dual_edge_angles",
            "_inverse_dual_edge_angles",
            "_inverse_face_solid_angles",
            "_inverse_dual_cell_solid_angles",
            "_radii",
            "_inverse_radii",
            "_radial_midpoints",
            "_inverse_radial_midpoints",
            "_radial_steps",
            "_radial_center_distances",
            "_radial_node_control_lengths",
            "_vertex_edge_metric",
            "_ca_er",
            "_cb_er",
            "_ca_et",
            "_cb_et",
        )
        for name in float_tensors:
            value = getattr(simulation, name)
            assert value.device == simulation.device
            assert value.dtype == simulation.dtype
        index_tensors = (
            "_edge_endpoints",
            "_face_edges",
            "_edge_left_faces",
            "_edge_right_faces",
            "_vertex_edges",
            "_interior_h_edges",
            "_boundary_h_edges",
            "_interior_h_faces",
            "_boundary_h_faces",
            "_interior_e_vertices",
            "_boundary_e_vertices",
            "_interior_e_edges",
            "_boundary_e_edges",
        )
        for name in index_tensors:
            value = getattr(simulation, name)
            assert value.device == simulation.device
            assert value.dtype == torch.long
        if simulation._source_distribution is not None:
            vertices, layers, weights, areas = simulation._source_distribution
            assert vertices.dtype == layers.dtype == torch.long
            assert weights.dtype == areas.dtype == simulation.dtype
            assert all(
                value.device == simulation.device
                for value in (vertices, layers, weights, areas)
            )
        if cuda_graph_chunk_size:
            simulation.enable_cuda_graph(cuda_graph_chunk_size)
            simulation.step(8)
            radial_trace = np.empty((0, 0))
            tangential_trace = np.empty((0, 0))
        else:
            radial_trace, tangential_trace = simulation.record_h_observations(
                np.asarray(((0,),), dtype=np.int64),
                np.asarray(((0,),), dtype=np.int64),
                np.asarray(((1.0,),)),
                np.asarray(((0,),), dtype=np.int64),
                np.asarray(((0,),), dtype=np.int64),
                np.asarray(((1.0,),)),
                8,
                sample_every=3,
            )
        fields = {}
        for name in ("er", "et", "hr", "ht"):
            fields[name] = simulation.global_field(name)
        memory = torch.tensor(
            (simulation.field_memory_bytes,),
            dtype=torch.int64,
            device=simulation.device,
        )
        gathered = [torch.zeros_like(memory) for _ in range(2)]
        distributed.all_gather(gathered, memory)
        if rank == 0:
            np.savez(
                output,
                **fields,
                local_field_memory_bytes=np.asarray(
                    [int(value.item()) for value in gathered]
                ),
                radial_trace=radial_trace,
                tangential_trace=tangential_trace,
            )
    finally:
        if simulation is not None:
            simulation.close()
        distributed.destroy_process_group()


def test_distributed_constructor_does_not_bootstrap_numpy_solver() -> None:
    source = inspect.getsource(DistributedGeodesicFDTD.__init__)

    assert "GeodesicFDTD(" not in source
    assert 'backend="numpy"' not in source
    assert "NumPyBackend" not in source


def test_distributed_solver_requires_initialized_process_group() -> None:
    mesh = build_geodesic_mesh(0)
    partition = partition_surface_mesh(mesh)
    with pytest.raises(RuntimeError, match="initialize torch.distributed"):
        DistributedGeodesicFDTD(
            partition,
            config=SimulationConfig(subdivision=0, radial_cells=4),
            mesh=mesh,
            device="cpu",
        )


def test_two_rank_gloo_matches_single_torch_solver(tmp_path: Path) -> None:
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is unavailable")
    rendezvous = tmp_path / "distributed-init"
    output = tmp_path / "distributed-fields.npz"
    torch.multiprocessing.start_processes(
        _distributed_worker,
        args=(str(rendezvous), str(output)),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    config = SimulationConfig(subdivision=0, radial_cells=4, courant_factor=0.2)
    reference = GeodesicFDTD(
        config,
        source=GaussianCurrent(peak_current_a=1.0e6),
        device="cpu",
        dtype="float64",
    )
    radial_trace, tangential_trace = reference.record_h_observations(
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((1.0,),)),
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((0,),), dtype=np.int64),
        np.asarray(((1.0,),)),
        8,
        sample_every=3,
    )

    with np.load(output) as values:
        for name in ("er", "et", "hr", "ht"):
            np.testing.assert_allclose(
                values[name],
                reference.to_numpy(getattr(reference, name)),
                rtol=2.0e-13,
                atol=1.0e-24,
            )
        assert np.all(values["local_field_memory_bytes"] < reference.memory_bytes)
        np.testing.assert_allclose(
            values["radial_trace"], radial_trace, rtol=2.0e-13, atol=1.0e-24
        )
        np.testing.assert_allclose(
            values["tangential_trace"],
            tangential_trace,
            rtol=2.0e-13,
            atol=1.0e-24,
        )


def test_two_rank_nccl_matches_single_torch_solver(tmp_path: Path) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("two CUDA devices with NCCL are required")
    rendezvous = tmp_path / "nccl-init"
    output = tmp_path / "nccl-fields.npz"
    torch.multiprocessing.start_processes(
        _distributed_worker,
        args=(str(rendezvous), str(output), "nccl"),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    config = SimulationConfig(subdivision=0, radial_cells=4, courant_factor=0.2)
    reference = GeodesicFDTD(
        config,
        source=GaussianCurrent(peak_current_a=1.0e6),
        device="cpu",
        dtype="float64",
    )
    reference.step(8)

    with np.load(output) as values:
        for name in ("er", "et", "hr", "ht"):
            np.testing.assert_allclose(
                values[name],
                reference.to_numpy(getattr(reference, name)),
                rtol=2.0e-13,
                atol=1.0e-24,
            )


def test_two_rank_nccl_cuda_graph_matches_single_solver(tmp_path: Path) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("two CUDA devices with NCCL are required")
    rendezvous = tmp_path / "nccl-graph-init"
    output = tmp_path / "nccl-graph-fields.npz"
    torch.multiprocessing.start_processes(
        _distributed_worker,
        args=(str(rendezvous), str(output), "nccl", 2),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    config = SimulationConfig(subdivision=0, radial_cells=4, courant_factor=0.2)
    reference = GeodesicFDTD(
        config,
        source=GaussianCurrent(peak_current_a=1.0e6),
        device="cpu",
        dtype="float64",
    )
    reference.step(8)

    with np.load(output) as values:
        for name in ("er", "et", "hr", "ht"):
            np.testing.assert_allclose(
                values[name],
                reference.to_numpy(getattr(reference, name)),
                rtol=2.0e-13,
                atol=1.0e-24,
            )


def test_two_rank_surface_impedance_matches_single_solver(tmp_path: Path) -> None:
    rendezvous = tmp_path / "surface-init"
    output = tmp_path / "surface-fields.npz"
    torch.multiprocessing.start_processes(
        _distributed_worker,
        args=(str(rendezvous), str(output), "gloo", 0, True),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    config = SimulationConfig(
        subdivision=0,
        radial_cells=4,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
        radial_boundary_condition="surface-impedance",
    )
    reference = GeodesicFDTD(
        config,
        source=GaussianCurrent(peak_current_a=1.0e6),
        surface_impedance=ConductiveHalfSpaceSurface(1.0 / 50.0),
        device="cpu",
        dtype="float64",
    )
    reference.step(8)

    with np.load(output) as values:
        for name in ("er", "et", "hr", "ht"):
            np.testing.assert_allclose(
                values[name],
                reference.to_numpy(getattr(reference, name)),
                rtol=2.0e-13,
                atol=1.0e-24,
            )


def test_two_rank_nccl_graph_preserves_surface_impedance_state(
    tmp_path: Path,
) -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not torch.distributed.is_nccl_available()
    ):
        pytest.skip("two CUDA devices with NCCL are required")
    rendezvous = tmp_path / "surface-nccl-init"
    output = tmp_path / "surface-nccl-fields.npz"
    torch.multiprocessing.start_processes(
        _distributed_worker,
        args=(str(rendezvous), str(output), "nccl", 2, True),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    config = SimulationConfig(
        subdivision=0,
        radial_cells=4,
        minimum_altitude_m=0.0,
        maximum_altitude_m=100_000.0,
        courant_factor=0.2,
        radial_boundary_condition="surface-impedance",
    )
    reference = GeodesicFDTD(
        config,
        source=GaussianCurrent(peak_current_a=1.0e6),
        surface_impedance=ConductiveHalfSpaceSurface(1.0 / 50.0),
        device="cpu",
        dtype="float64",
    )
    reference.step(8)

    with np.load(output) as values:
        for name in ("er", "et", "hr", "ht"):
            np.testing.assert_allclose(
                values[name],
                reference.to_numpy(getattr(reference, name)),
                rtol=2.0e-13,
                atol=1.0e-24,
            )
