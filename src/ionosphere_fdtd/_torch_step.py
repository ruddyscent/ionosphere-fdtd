"""Functional PyTorch field update for the geodesic FDTD solver."""

from __future__ import annotations

from typing import NamedTuple

import torch


class FieldState(NamedTuple):
    """Evolving electromagnetic field tensors for one leapfrog state."""

    er: torch.Tensor
    et: torch.Tensor
    hr: torch.Tensor
    ht: torch.Tensor


class FieldStepParameters(NamedTuple):
    """Immutable tensor buffers and scalar policy for the field recurrence."""

    time_step_over_mu: float
    full_spherical_geometry: bool
    compress_uniform_material_coefficients: bool
    edges: torch.Tensor
    face_edges: torch.Tensor
    face_edge_signs: torch.Tensor
    edge_left_faces: torch.Tensor
    edge_right_faces: torch.Tensor
    vertex_edges: torch.Tensor
    vertex_edge_signs: torch.Tensor
    primal_edge_angles: torch.Tensor
    inverse_primal_edge_angles: torch.Tensor
    dual_edge_angles: torch.Tensor
    inverse_dual_edge_angles: torch.Tensor
    inverse_dual_cell_solid_angles: torch.Tensor
    inverse_face_solid_angles: torch.Tensor
    radii: torch.Tensor
    inverse_radii: torch.Tensor
    radial_midpoints: torch.Tensor
    inverse_radial_midpoints: torch.Tensor
    radial_steps: torch.Tensor
    radial_node_control_lengths: torch.Tensor
    radial_center_distances: torch.Tensor
    ca_er: torch.Tensor
    cb_er: torch.Tensor
    ca_et: torch.Tensor
    cb_et: torch.Tensor
    radial_source_vertices: torch.Tensor | None
    radial_source_layers: torch.Tensor | None
    radial_source_weights: torch.Tensor | None
    radial_source_element_length: float
    tangential_source_edges: torch.Tensor | None
    tangential_source_layers: torch.Tensor | None
    tangential_source_weights: torch.Tensor | None


def edge_difference(
    vertex_values: torch.Tensor, parameters: FieldStepParameters
) -> torch.Tensor:
    """Return head-minus-tail values on primal edges."""

    return (
        vertex_values[parameters.edges[:, 1]]
        - vertex_values[parameters.edges[:, 0]]
    )


def dual_edge_difference(
    face_values: torch.Tensor, parameters: FieldStepParameters
) -> torch.Tensor:
    """Return left-minus-right values across primal edges."""

    return (
        face_values[parameters.edge_left_faces]
        - face_values[parameters.edge_right_faces]
    )


def face_circulation(
    edge_values: torch.Tensor, parameters: FieldStepParameters
) -> torch.Tensor:
    """Sum oriented primal-edge values in the established corner order."""

    sign_shape = (parameters.face_edges.shape[0],) + (1,) * (
        edge_values.ndim - 1
    )
    result = edge_values[parameters.face_edges[:, 0]] * (
        parameters.face_edge_signs[:, 0].reshape(sign_shape)
    )
    for corner in (1, 2):
        result = result + edge_values[parameters.face_edges[:, corner]] * (
            parameters.face_edge_signs[:, corner].reshape(sign_shape)
        )
    return result


def dual_cell_circulation(
    edge_values: torch.Tensor, parameters: FieldStepParameters
) -> torch.Tensor:
    """Sum oriented dual-edge values in deterministic incidence order."""

    vertex_count = parameters.vertex_edges.shape[0]
    sign_shape = (vertex_count,) + (1,) * (edge_values.ndim - 1)
    result = edge_values[parameters.vertex_edges[:, 0]] * (
        parameters.vertex_edge_signs[:, 0].reshape(sign_shape)
    )
    for slot in range(1, parameters.vertex_edges.shape[1]):
        result = result + edge_values[parameters.vertex_edges[:, slot]] * (
            parameters.vertex_edge_signs[:, slot].reshape(sign_shape)
        )
    return result


def radial_derivative_et(
    et: torch.Tensor, parameters: FieldStepParameters
) -> torch.Tensor:
    """Return the boundary-aware radial derivative used by the Ht update."""

    values = et
    if parameters.full_spherical_geometry:
        values = values * parameters.radial_midpoints[None, :]
    lower = 2.0 * values[:, :1] / parameters.radial_steps[:1]
    upper = -2.0 * values[:, -1:] / parameters.radial_steps[-1:]
    if et.shape[1] > 1:
        interior = torch.diff(values, dim=1) / parameters.radial_center_distances
        result = torch.cat((lower, interior, upper), dim=1)
    else:
        result = torch.cat((lower, upper), dim=1)
    if parameters.full_spherical_geometry:
        result = result * parameters.inverse_radii
    return result


def radial_derivative_ht(
    ht: torch.Tensor, parameters: FieldStepParameters
) -> torch.Tensor:
    """Return the radial derivative used by the Et update."""

    values = ht
    if parameters.full_spherical_geometry:
        values = values * parameters.radii[None, :]
    result = torch.diff(values, dim=1) / parameters.radial_steps[None, :]
    if parameters.full_spherical_geometry:
        result = result * parameters.inverse_radial_midpoints
    return result


def lower_surface_gradient_er(
    state: FieldState, parameters: FieldStepParameters
) -> torch.Tensor:
    """Return the lower-boundary Er gradient required by the surface ADE."""

    gradient = edge_difference(state.er, parameters)
    gradient = gradient * parameters.inverse_primal_edge_angles
    gradient = gradient * parameters.inverse_radii
    return gradient[:, 0]


def advance_magnetic(
    state: FieldState,
    parameters: FieldStepParameters,
    lower_boundary_ht: torch.Tensor | None = None,
) -> FieldState:
    """Return state after one functional magnetic half-step."""

    surface_gradient_er = edge_difference(state.er, parameters)
    surface_gradient_er = (
        surface_gradient_er * parameters.inverse_primal_edge_angles
    )
    surface_gradient_er = surface_gradient_er * parameters.inverse_radii
    magnetic_drive = surface_gradient_er - radial_derivative_et(
        state.et, parameters
    )
    ht = state.ht + parameters.time_step_over_mu * magnetic_drive
    if lower_boundary_ht is not None:
        ht = torch.cat((lower_boundary_ht[:, None], ht[:, 1:]), dim=1)

    electric_circulation = face_circulation(
        state.et * parameters.primal_edge_angles, parameters
    )
    electric_circulation = (
        electric_circulation * parameters.inverse_face_solid_angles
    )
    electric_circulation = (
        electric_circulation * parameters.inverse_radial_midpoints
    )
    hr = state.hr - parameters.time_step_over_mu * electric_circulation
    return FieldState(state.er, state.et, hr, ht)


def _radial_source_correction(
    er: torch.Tensor,
    parameters: FieldStepParameters,
    current: torch.Tensor | float,
) -> torch.Tensor:
    vertices = parameters.radial_source_vertices
    layers = parameters.radial_source_layers
    weights = parameters.radial_source_weights
    if vertices is None or layers is None or weights is None:
        return er
    current_density = (
        weights
        * current
        * parameters.radial_source_element_length
        * parameters.inverse_dual_cell_solid_angles[vertices, 0]
        / parameters.radii[layers] ** 2
        / parameters.radial_node_control_lengths[layers]
    )
    coefficient_vertices = (
        0 if parameters.compress_uniform_material_coefficients else vertices
    )
    updated = er[vertices, layers] - (
        parameters.cb_er[coefficient_vertices, layers] * current_density
    )
    return er.index_put((vertices, layers), updated)


def _tangential_source_correction(
    et: torch.Tensor,
    parameters: FieldStepParameters,
    current: torch.Tensor | float,
) -> torch.Tensor:
    edges = parameters.tangential_source_edges
    layers = parameters.tangential_source_layers
    weights = parameters.tangential_source_weights
    if edges is None or layers is None or weights is None:
        return et
    current_density = (
        weights
        * current
        * parameters.inverse_dual_edge_angles[edges, 0]
        / parameters.radial_midpoints[layers]
        / parameters.radial_steps[layers]
    )
    coefficient_edges = (
        0 if parameters.compress_uniform_material_coefficients else edges
    )
    updated = et[edges, layers] - (
        parameters.cb_et[coefficient_edges, layers] * current_density
    )
    return et.index_put((edges, layers), updated)


def advance_electric(
    state: FieldState,
    parameters: FieldStepParameters,
    current: torch.Tensor | float = 0.0,
    radial_plasma_current: torch.Tensor | None = None,
    tangential_plasma_current: torch.Tensor | None = None,
) -> FieldState:
    """Return state after one functional electric half-step."""

    magnetic_circulation = dual_cell_circulation(
        state.ht * parameters.dual_edge_angles, parameters
    )
    magnetic_circulation = (
        magnetic_circulation * parameters.inverse_dual_cell_solid_angles
    )
    magnetic_circulation = magnetic_circulation * parameters.inverse_radii
    er = state.er * parameters.ca_er + magnetic_circulation * parameters.cb_er
    er = _radial_source_correction(er, parameters, current)
    if radial_plasma_current is not None:
        er = er - parameters.cb_er * radial_plasma_current

    surface_gradient_hr = dual_edge_difference(state.hr, parameters)
    surface_gradient_hr = (
        surface_gradient_hr * parameters.inverse_dual_edge_angles
    )
    surface_gradient_hr = (
        surface_gradient_hr * parameters.inverse_radial_midpoints
    )
    electric_drive = surface_gradient_hr - radial_derivative_ht(
        state.ht, parameters
    )
    et = state.et * parameters.ca_et + electric_drive * parameters.cb_et
    et = _tangential_source_correction(et, parameters, current)
    if tangential_plasma_current is not None:
        et = et - parameters.cb_et * tangential_plasma_current
    return FieldState(er, et, state.hr, state.ht)


def advance(
    state: FieldState,
    parameters: FieldStepParameters,
    current: torch.Tensor | float,
    lower_boundary_ht: torch.Tensor | None = None,
    radial_plasma_current: torch.Tensor | None = None,
    tangential_plasma_current: torch.Tensor | None = None,
) -> FieldState:
    """Return the next complete leapfrog state without mutating inputs."""

    magnetic_state = advance_magnetic(
        state, parameters, lower_boundary_ht=lower_boundary_ht
    )
    return advance_electric(
        magnetic_state,
        parameters,
        current,
        radial_plasma_current=radial_plasma_current,
        tangential_plasma_current=tangential_plasma_current,
    )


def advance_chunk(
    state: FieldState,
    parameters: FieldStepParameters,
    currents: torch.Tensor,
) -> FieldState:
    """Advance one static current chunk by repeatedly using ``advance``."""

    for offset in range(currents.shape[0]):
        state = advance(state, parameters, currents[offset])
        if (
            offset + 1 < currents.shape[0]
            and torch.compiler.is_compiling()
            and state.er.device.type == "cpu"
            and (
                state.er.requires_grad
                or state.et.requires_grad
                or state.hr.requires_grad
                or state.ht.requires_grad
                or currents.requires_grad
                or parameters.ca_er.requires_grad
                or parameters.cb_er.requires_grad
                or parameters.ca_et.requires_grad
                or parameters.cb_et.requires_grad
            )
        ):
            # CPU TorchInductor can otherwise fuse adjacent leapfrog steps in
            # a way that drops part of the Et/Hr adjoint. Materializing just
            # this coupled pair preserves eager gradients without changing
            # the recurrence or penalizing eager and accelerator execution.
            et_size = state.et.numel()
            tangential_state = torch.cat(
                (state.et.reshape(-1), state.hr.reshape(-1))
            )
            state = FieldState(
                state.er,
                tangential_state[:et_size].reshape(state.et.shape),
                tangential_state[et_size:].reshape(state.hr.shape),
                state.ht,
            )
    return state
