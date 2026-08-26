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


class SurfaceImpedanceStepParameters(NamedTuple):
    """Immutable tensor coefficients and geometry for the surface ADE."""

    decay: torch.Tensor
    drive: torch.Tensor
    history_weights: torch.Tensor
    scale: torch.Tensor
    boundary_metric: float
    radial_step: torch.Tensor


class PlasmaSpeciesStepParameters(NamedTuple):
    """Exact charged-fluid update coefficients for one species."""

    decay: torch.Tensor
    cosine: torch.Tensor
    sine: torch.Tensor
    drive_parallel: torch.Tensor
    drive_real: torch.Tensor
    drive_imag: torch.Tensor


class PlasmaStepParameters(NamedTuple):
    """Immutable reconstruction, scattering, and plasma ADE tensors."""

    magnetic_direction: torch.Tensor
    species: tuple[PlasmaSpeciesStepParameters, ...]
    face_edges: torch.Tensor
    faces: torch.Tensor
    reconstruction: torch.Tensor
    face_centers: torch.Tensor
    left_faces: torch.Tensor
    right_faces: torch.Tensor
    left_tangents: torch.Tensor
    right_tangents: torch.Tensor
    vertex_faces: torch.Tensor
    vertex_face_weights: torch.Tensor


class OptionalPhysicsState(NamedTuple):
    """Fields plus explicit surface and plasma ADE state tensors."""

    fields: FieldState
    surface_memory: torch.Tensor | None
    plasma_current_density: tuple[torch.Tensor, ...]


class OptionalPhysicsStepParameters(NamedTuple):
    """Field and optional-physics parameters for one shared recurrence."""

    fields: FieldStepParameters
    surface: SurfaceImpedanceStepParameters | None
    plasma: PlasmaStepParameters | None


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


def advance_surface_impedance(
    memory: torch.Tensor,
    state: FieldState,
    surface: SurfaceImpedanceStepParameters,
    fields: FieldStepParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the lower Ht boundary and next surface memory."""

    h_old = state.ht[:, 0]
    electric_first_cell = surface.boundary_metric * state.et[:, 0]
    gradient = lower_surface_gradient_er(state, fields)
    weighted = surface.history_weights[None, :]
    history = surface.scale * (memory * weighted).sum(dim=1)
    impedance_gain = surface.scale * surface.history_weights.sum()
    curl_scale = 2.0 * fields.time_step_over_mu / surface.radial_step
    coupling = curl_scale * impedance_gain
    h_new = (
        (1.0 - 0.5 * coupling) * h_old
        + fields.time_step_over_mu * gradient
        - curl_scale * electric_first_cell
        + curl_scale * history
    ) / (1.0 + 0.5 * coupling)
    next_memory = memory * surface.decay[None, :] + (
        h_new + h_old
    )[:, None] * surface.drive[None, :]
    return h_new, next_memory


def _cross_with_magnetic_direction(
    values: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    """Return ``values × direction`` without an in-place assembly."""

    return torch.stack(
        (
            values[..., 1] * direction[..., 2]
            - values[..., 2] * direction[..., 1],
            values[..., 2] * direction[..., 0]
            - values[..., 0] * direction[..., 2],
            values[..., 0] * direction[..., 1]
            - values[..., 1] * direction[..., 0],
        ),
        dim=-1,
    )


def advance_plasma(
    current_density: tuple[torch.Tensor, ...],
    state: FieldState,
    plasma: PlasmaStepParameters,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Return scattered plasma currents and every next species state."""

    radial_nodes_at_faces = state.er[plasma.faces].sum(dim=1) / 3.0
    radial = 0.5 * (
        radial_nodes_at_faces[:, :-1] + radial_nodes_at_faces[:, 1:]
    )
    edge_values = state.et[plasma.face_edges]
    tangential = (
        edge_values[..., None] * plasma.reconstruction[:, :, None, :]
    ).sum(dim=1)
    electric = tangential + radial[..., None] * plasma.face_centers[:, None, :]
    direction = plasma.magnetic_direction
    electric_parallel = (electric * direction).sum(dim=2)[..., None] * direction
    electric_perpendicular = electric - electric_parallel
    electric_cross = _cross_with_magnetic_direction(
        electric_perpendicular, direction
    )
    total = torch.zeros_like(electric)
    next_currents = []
    for current, coefficients in zip(
        current_density, plasma.species, strict=True
    ):
        current_parallel = (
            (current * direction).sum(dim=2)[..., None] * direction
        )
        current_perpendicular = current - current_parallel
        current_cross = _cross_with_magnetic_direction(
            current_perpendicular, direction
        )
        updated = (
            coefficients.decay[..., None] * current_parallel
            + coefficients.drive_parallel[..., None] * electric_parallel
            + coefficients.decay[..., None]
            * (
                coefficients.cosine[..., None] * current_perpendicular
                + coefficients.sine[..., None] * current_cross
            )
            + coefficients.drive_real[..., None] * electric_perpendicular
            + coefficients.drive_imag[..., None] * electric_cross
        )
        next_currents.append(updated)
        total = total + updated

    left = (
        total[plasma.left_faces] * plasma.left_tangents[:, None, :]
    ).sum(dim=2)
    right = (
        total[plasma.right_faces] * plasma.right_tangents[:, None, :]
    ).sum(dim=2)
    tangential_current = 0.5 * (left + right)
    radial_at_faces = (total * plasma.face_centers[:, None, :]).sum(dim=2)
    radial_at_vertices = (
        radial_at_faces[plasma.vertex_faces]
        * plasma.vertex_face_weights[:, :, None]
    ).sum(dim=1)
    interior = 0.5 * (
        radial_at_vertices[:, :-1] + radial_at_vertices[:, 1:]
    )
    radial_current = torch.cat(
        (
            radial_at_vertices[:, :1],
            interior,
            radial_at_vertices[:, -1:],
        ),
        dim=1,
    )
    return radial_current, tangential_current, tuple(next_currents)


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


def advance_optional_physics(
    state: OptionalPhysicsState,
    parameters: OptionalPhysicsStepParameters,
    current: torch.Tensor | float,
) -> OptionalPhysicsState:
    """Advance fields and optional ADE state in one functional transition."""

    lower_boundary_ht = None
    surface_memory = state.surface_memory
    if parameters.surface is not None:
        assert surface_memory is not None
        lower_boundary_ht, surface_memory = advance_surface_impedance(
            surface_memory,
            state.fields,
            parameters.surface,
            parameters.fields,
        )

    radial_plasma_current = None
    tangential_plasma_current = None
    plasma_current_density = state.plasma_current_density
    if parameters.plasma is not None:
        (
            radial_plasma_current,
            tangential_plasma_current,
            plasma_current_density,
        ) = advance_plasma(
            plasma_current_density,
            state.fields,
            parameters.plasma,
        )

    fields = advance(
        state.fields,
        parameters.fields,
        current,
        lower_boundary_ht=lower_boundary_ht,
        radial_plasma_current=radial_plasma_current,
        tangential_plasma_current=tangential_plasma_current,
    )
    return OptionalPhysicsState(
        fields, surface_memory, plasma_current_density
    )


def _optional_physics_requires_grad(
    state: OptionalPhysicsState,
    parameters: OptionalPhysicsStepParameters,
    currents: torch.Tensor | float,
) -> bool:
    values = [*state.fields, currents]
    values.extend(
        (
            parameters.fields.ca_er,
            parameters.fields.cb_er,
            parameters.fields.ca_et,
            parameters.fields.cb_et,
        )
    )
    if state.surface_memory is not None:
        values.append(state.surface_memory)
    values.extend(state.plasma_current_density)
    if parameters.surface is not None:
        values.extend(parameters.surface[:4])
    if parameters.plasma is not None:
        values.append(parameters.plasma.magnetic_direction)
        for species in parameters.plasma.species:
            values.extend(species)
    return any(getattr(value, "requires_grad", False) for value in values)


def _materialize_optional_physics_state(
    state: OptionalPhysicsState,
) -> OptionalPhysicsState:
    """Keep every coupled recurrent state visible to the CPU adjoint."""

    tensors = [*state.fields]
    if state.surface_memory is not None:
        tensors.append(state.surface_memory)
    tensors.extend(state.plasma_current_density)
    combined = torch.cat(tuple(value.reshape(-1) for value in tensors))
    offset = 0
    restored = []
    for value in tensors:
        size = value.numel()
        restored.append(combined[offset : offset + size].reshape(value.shape))
        offset += size
    fields = FieldState(*restored[:4])
    index = 4
    surface_memory = None
    if state.surface_memory is not None:
        surface_memory = restored[index]
        index += 1
    return OptionalPhysicsState(
        fields,
        surface_memory,
        tuple(restored[index:]),
    )


def advance_optional_physics_chunk(
    state: OptionalPhysicsState,
    parameters: OptionalPhysicsStepParameters,
    currents: torch.Tensor,
) -> OptionalPhysicsState:
    """Advance a static optional-physics chunk inside one compiled graph."""

    preserve_adjoint = (
        torch.compiler.is_compiling()
        and state.fields.er.device.type == "cpu"
        and _optional_physics_requires_grad(state, parameters, currents)
    )
    for offset in range(currents.shape[0]):
        state = advance_optional_physics(
            state, parameters, currents[offset]
        )
        if preserve_adjoint:
            state = _materialize_optional_physics_state(state)
    return state
