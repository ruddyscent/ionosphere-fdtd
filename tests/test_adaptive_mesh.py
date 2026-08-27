import numpy as np
import pytest

from ionosphere_fdtd.adaptive_mesh import (
    SphericalRefinementRegion,
    _enforce_local_delaunay,
    build_adaptive_geodesic_mesh,
    validate_adaptive_mesh,
)
from ionosphere_fdtd.mesh import (
    build_geodesic_mesh,
    build_geodesic_mesh_from_topology,
)
from ionosphere_fdtd.solver import GeodesicFDTD, SimulationConfig


def test_disabled_adaptation_is_identical_to_uniform_mesh() -> None:
    expected = build_geodesic_mesh(2)
    actual = build_adaptive_geodesic_mesh(2, ())

    assert actual.topology_kind == "uniform"
    np.testing.assert_array_equal(actual.vertices, expected.vertices)
    np.testing.assert_array_equal(actual.faces, expected.faces)
    np.testing.assert_array_equal(actual.primal_edge_angles, expected.primal_edge_angles)
    np.testing.assert_array_equal(actual.dual_edge_angles, expected.dual_edge_angles)


def test_local_refinement_is_conforming_balanced_and_well_centered() -> None:
    mesh = build_adaptive_geodesic_mesh(
        1,
        (
            SphericalRefinementRegion(
                35.0,
                126.0,
                radius_deg=8.0,
                target_subdivision=3,
                transition_width_deg=8.0,
                label="source",
            ),
        ),
    )
    validation = validate_adaptive_mesh(mesh)

    assert mesh.topology_kind == "adaptive"
    assert set(np.unique(mesh.face_levels)) == {1, 2, 3}
    assert mesh.n_faces < build_geodesic_mesh(3).n_faces
    assert validation.maximum_adjacent_level_jump <= 1
    assert validation.minimum_dual_edge_angle > 0.0
    assert validation.minimum_dual_to_primal_ratio > 0.05
    assert validation.primal_area_error < 1.0e-12
    assert validation.dual_area_error < 1.0e-12

    vertex_values = np.arange(mesh.n_vertices, dtype=np.float64)
    np.testing.assert_allclose(
        mesh.face_circulation(mesh.edge_difference(vertex_values)),
        0.0,
        rtol=0.0,
        atol=0.0,
    )

    base = build_geodesic_mesh(1)
    region = SphericalRefinementRegion(35.0, 126.0, 8.0, 3)
    farthest = int(np.argmin(base.vertices @ region.direction()))
    np.testing.assert_array_equal(mesh.vertices[farthest], base.vertices[farthest])


def test_two_refinement_regions_cover_source_and_receiver() -> None:
    regions = (
        SphericalRefinementRegion(46.5, -90.9, 6.0, 2, 6.0, "source"),
        SphericalRefinementRegion(69.0, -156.0, 6.0, 2, 6.0, "oil"),
    )
    mesh = build_adaptive_geodesic_mesh(1, regions)

    for region in regions:
        nearest = int(np.argmax(mesh.face_centers @ region.direction()))
        assert mesh.face_levels[nearest] == region.target_subdivision
    assert mesh.refinement_spec["regions"][0]["label"] == "source"
    assert mesh.refinement_spec["regions"][1]["label"] == "oil"


def test_multilevel_two_region_mesh_has_positive_disjoint_hodge_supports() -> None:
    mesh = build_adaptive_geodesic_mesh(
        1,
        (
            SphericalRefinementRegion(46.5, -90.9, 8.0, 3, 8.0, "source"),
            SphericalRefinementRegion(69.0, -156.0, 8.0, 3, 8.0, "oil"),
        ),
    )

    validation = validate_adaptive_mesh(mesh)

    assert validation.minimum_dual_edge_angle > 0.0
    assert np.all(mesh.dual_cell_solid_angles > 0.0)
    assert np.all(mesh.edge_diamond_solid_angles() > 0.0)
    assert np.sum(mesh.edge_diamond_solid_angles()) == pytest.approx(4.0 * np.pi)


def test_default_relaxation_prevents_transition_dual_edge_collapse() -> None:
    mesh = build_adaptive_geodesic_mesh(
        3,
        (
            SphericalRefinementRegion(46.5, -90.9, 3.0, 5, 3.0, "source"),
            SphericalRefinementRegion(69.0, -156.0, 3.0, 5, 3.0, "oil"),
        ),
    )

    validation = validate_adaptive_mesh(mesh)

    assert validation.minimum_dual_to_primal_ratio > 0.05
    assert mesh.refinement_spec["algorithm"] == "conforming-red-transition-v3"
    assert mesh.refinement_spec["relaxations_per_level"] == 4
    assert mesh.refinement_spec["optimization_steps_per_level"] == 0


def test_lawson_flip_repairs_non_delaunay_local_edge() -> None:
    base = build_geodesic_mesh(1)
    vertices = base.vertices.copy()
    vertices[15] = (0.3779688486851255, 0.0824166650239095, 0.9221426368789035)
    levels = np.ones(base.n_faces, dtype=np.int64)
    invalid = build_geodesic_mesh_from_topology(
        vertices,
        base.faces,
        face_levels=levels,
        require_well_centered=False,
    )
    with pytest.raises(ValueError, match="non-positive"):
        validate_adaptive_mesh(invalid)

    faces, repaired_levels = _enforce_local_delaunay(
        vertices, base.faces, levels
    )
    repaired = build_geodesic_mesh_from_topology(
        vertices,
        faces,
        face_levels=repaired_levels,
        require_well_centered=False,
    )

    assert np.any(faces != base.faces)
    assert validate_adaptive_mesh(repaired).minimum_dual_edge_angle > 0.0


def test_adaptive_mesh_runs_in_solver_without_changing_zero_state() -> None:
    mesh = build_adaptive_geodesic_mesh(
        1,
        (SphericalRefinementRegion(35.0, 126.0, 8.0, 2),),
    )
    simulation = GeodesicFDTD(
        SimulationConfig(subdivision=1, radial_cells=4, courant_factor=0.2),
        mesh=mesh,
    )

    simulation.step(4)

    assert not simulation.er.any().item()
    assert not simulation.et.any().item()
    assert not simulation.hr.any().item()
    assert not simulation.ht.any().item()


def test_refinement_region_and_builder_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="latitude"):
        SphericalRefinementRegion(91.0, 0.0, 5.0, 2)
    with pytest.raises(ValueError, match="radius"):
        SphericalRefinementRegion(0.0, 0.0, 0.0, 2)
    with pytest.raises(ValueError, match="target subdivision"):
        SphericalRefinementRegion(0.0, 0.0, 5.0, -1)
    with pytest.raises(ValueError, match="regions"):
        build_adaptive_geodesic_mesh(1, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="relaxations_per_level"):
        build_adaptive_geodesic_mesh(1, (), relaxations_per_level=0)
    with pytest.raises(ValueError, match="optimization_steps_per_level"):
        build_adaptive_geodesic_mesh(1, (), optimization_steps_per_level=-1)
