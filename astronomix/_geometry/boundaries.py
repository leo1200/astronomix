"""
Ghost-cell boundary handling for the primitive state.

Fills the padded ghost-cell regions according to the configured per-axis
boundary conditions: open (zero-gradient), periodic, reflective (normal velocity
negated), fixed (Dirichlet, optionally open in the normal momentum) and the
specialised 2D MHD jet-injection boundary. The top-level ``_boundary_handler``
dispatches per spatial axis; everything below it is a single traced branch on
static configuration values, so no Python loops survive into the trace.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSERVATIVE_GAS_STATE,
    FIXED_BOUNDARY,
    FIXED_BOUNDARY_OPEN_MOMENTUM,
    ISOTHERMAL,
    PRIMITIVE_GAS_STATE,
    MAGNETIC_FIELD_ONLY,
    MHD_JET_BOUNDARY,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    STATE_TYPE,
    VELOCITY_ONLY,
    XAXIS,
    YAXIS,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equations import conserved_state_from_primitive

 
# -----------------------------------------------------------------------------
# Indexing helper
# -----------------------------------------------------------------------------
 
def _axis_slice(axis: int, start, stop, ndim: int) -> tuple:
    """Build an indexing tuple of the form ``[:, ..., slice(start, stop), ..., :]``
    that slices only the requested ``axis``. ``axis`` and ``ndim`` are static at
    trace time, so this constructs a plain Python tuple of slices."""
    return (
        (slice(None),) * axis
        + (slice(start, stop),)
        + (slice(None),) * (ndim - axis - 1)
    )
 
 
# -----------------------------------------------------------------------------
# Open boundaries — broadcast the first/last interior cell into all ghost cells
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _open_left_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """All left ghost cells ← first interior cell (via length-1 broadcast)."""
    ndim = primitive_state.ndim
    src = _axis_slice(axis, num_ghost_cells, num_ghost_cells + 1, ndim)
    dst = _axis_slice(axis, 0, num_ghost_cells, ndim)
    return primitive_state.at[dst].set(primitive_state[src])
 
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _open_right_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """All right ghost cells ← last interior cell (via length-1 broadcast)."""
    ndim = primitive_state.ndim
    src = _axis_slice(axis, -num_ghost_cells - 1, -num_ghost_cells, ndim)
    dst = _axis_slice(axis, -num_ghost_cells, None, ndim)
    return primitive_state.at[dst].set(primitive_state[src])
 
 
# -----------------------------------------------------------------------------
# Periodic boundaries — wrap interior cells to the opposite ghost region
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _periodic_boundaries(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """Wrap both ghost regions with a single scatter per side."""
    ndim = primitive_state.ndim
    ng = num_ghost_cells
 
    # Left ghosts ← last ``ng`` interior cells  (state[..., :ng] = state[..., -2*ng:-ng])
    left_dst = _axis_slice(axis, 0, ng, ndim)
    left_src = _axis_slice(axis, -2 * ng, -ng, ndim)
    primitive_state = primitive_state.at[left_dst].set(primitive_state[left_src])
 
    # Right ghosts ← first ``ng`` interior cells  (state[..., -ng:] = state[..., ng:2*ng])
    right_dst = _axis_slice(axis, -ng, None, ndim)
    right_src = _axis_slice(axis, ng, 2 * ng, ndim)
    primitive_state = primitive_state.at[right_dst].set(primitive_state[right_src])
 
    return primitive_state
 
 
# -----------------------------------------------------------------------------
# Reflective boundaries — mirror interior block, negate normal velocity
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _reflective_left_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """Mirror the first ``num_ghost_cells`` interior cells into the left ghost
    region and negate the velocity component normal to the boundary
    (``var_index == axis`` by convention)."""
    ndim = primitive_state.ndim
    ng = num_ghost_cells
 
    # Read the interior block and reverse it along the spatial axis.
    src = _axis_slice(axis, ng, 2 * ng, ndim)
    block = jnp.flip(primitive_state[src], axis=axis)
 
    # Negate the normal-velocity component on the (small) mirrored block before
    # scattering — cheaper than a second scatter on the full array.
    block = block.at[axis].multiply(-1.0)
 
    dst = _axis_slice(axis, 0, ng, ndim)
    return primitive_state.at[dst].set(block)
 
 
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _reflective_right_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    """Mirror image of ``_reflective_left_boundary`` for the right side."""
    ndim = primitive_state.ndim
    ng = num_ghost_cells
 
    src = _axis_slice(axis, -2 * ng, -ng, ndim)
    block = jnp.flip(primitive_state[src], axis=axis)
    block = block.at[axis].multiply(-1.0)
 
    dst = _axis_slice(axis, -ng, None, ndim)
    return primitive_state.at[dst].set(block)
 
 
# -----------------------------------------------------------------------------
# MHD jet injection boundary (2D, y-left only)
# -----------------------------------------------------------------------------
 
@partial(
    jax.jit,
    static_argnames=[
        "axis",
        "num_ghost_cells",
        "grid_spacing",
        "num_cells",
        "type_handled",
    ],
)
def _jet_left_boundary(
    primitive_state: STATE_TYPE,
    num_ghost_cells: int,
    axis: int,
    grid_spacing: float,
    num_cells: int,
    type_handled: int,
) -> STATE_TYPE:
    """Inject a magnetised jet through a y-left boundary patch (2D MHD).

    The bulk of the boundary is treated as open; the central injection patch
    (a band of width ``2 * half_inj_width`` around the domain mid-line) is then
    overwritten with the prescribed jet gas state, velocity or magnetic field,
    selected by ``type_handled``.

    Args:
        primitive_state: The primitive state array.
        num_ghost_cells: The number of ghost cells per side.
        axis: The (injection) spatial axis.
        grid_spacing: The (uniform) grid spacing.
        num_cells: The number of interior cells along the transverse axis.
        type_handled: Which quantity is being written (gas state, velocity or
            magnetic field).

    Returns:
        The primitive state with the jet patch injected.
    """
    # Start from an open boundary (single broadcast; no loop).
    primitive_state = _open_left_boundary(primitive_state, num_ghost_cells, axis)
 
    half_inj_width = 0.025
    half_inj_cell_num = int(half_inj_width / grid_spacing)
 
    B0 = 200**0.5
    to_set_gas_state = jnp.array([5 / 3, 800.0, 0.0, 1.0])
    to_set_velocity = jnp.array([800.0, 0.0, 0.0])
    to_set_magnetic_field = jnp.array([B0, 0.0, 0.0])
 
    jet_lo = num_cells // 2 - half_inj_cell_num
    jet_hi = num_cells // 2 + half_inj_cell_num
 
    if type_handled == PRIMITIVE_GAS_STATE:
        primitive_state = primitive_state.at[
            :, 0:num_ghost_cells, jet_lo:jet_hi
        ].set(to_set_gas_state[:, None, None])
    elif type_handled == VELOCITY_ONLY:
        primitive_state = primitive_state.at[
            :, 0:num_ghost_cells, jet_lo:jet_hi
        ].set(to_set_velocity[:, None, None])
    elif type_handled == MAGNETIC_FIELD_ONLY:
        primitive_state = primitive_state.at[
            :, 0:num_ghost_cells, jet_lo:jet_hi
        ].set(to_set_magnetic_field[:, None, None])
 
    return primitive_state
 
 
# -----------------------------------------------------------------------------
# Per-axis dispatch helpers (branches on static config values only — no loops)
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["num_ghost_cells", "bs", "axis", "type_handled", "registered_variables", "config"])
def _apply_axis_bcs(
    primitive_state: STATE_TYPE, params: SimulationParams, config: SimulationConfig, registered_variables: RegisteredVariables, num_ghost_cells: int, bs, axis: int, type_handled: int = PRIMITIVE_GAS_STATE
) -> STATE_TYPE:
    """Apply left/right/periodic BCs along a single spatial axis. All branches
    are on static (``SimulationConfig``) fields and resolve at trace time."""
    if bs.left_boundary == OPEN_BOUNDARY:
        primitive_state = _open_left_boundary(primitive_state, num_ghost_cells, axis=axis)
    elif bs.left_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_left_boundary(primitive_state, num_ghost_cells, axis=axis)
 
    if bs.right_boundary == OPEN_BOUNDARY:
        primitive_state = _open_right_boundary(primitive_state, num_ghost_cells, axis=axis)
    elif bs.right_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_right_boundary(primitive_state, num_ghost_cells, axis=axis)
 
    if (
        bs.left_boundary == PERIODIC_BOUNDARY
        and bs.right_boundary == PERIODIC_BOUNDARY
    ):
        primitive_state = _periodic_boundaries(primitive_state, num_ghost_cells, axis=axis)

    if (
        bs.left_boundary == FIXED_BOUNDARY or bs.left_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM
    ):
        dst = _axis_slice(axis, 0, num_ghost_cells, primitive_state.ndim) # destination slice
        left_state = (
            params.fixed_boundary_state.x.left_state if axis == XAXIS
            else params.fixed_boundary_state.y.left_state if axis == YAXIS
            else params.fixed_boundary_state.z.left_state
        )

        # The fixed boundary state is supplied as a 1D array of primitive
        # variables. When the boundary handler runs on the conservative state we
        # must convert it first so the ghost cells are written in the same
        # representation as the rest of the array.
        if type_handled == CONSERVATIVE_GAS_STATE:

            if config.mhd or config.equation_of_state == ISOTHERMAL:
                raise NotImplementedError("FIXED_BOUNDARY with conservative state is not implemented for MHD or isothermal EOS yet")

            # convert the fixed primitive state to conservative form
            left_state = conserved_state_from_primitive(left_state, params.gamma, config, registered_variables)

        left_state = left_state.reshape((left_state.shape[0],) + (1,) * (primitive_state.ndim - 1))
        primitive_state = primitive_state.at[dst].set(left_state)

        if bs.left_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM:
            normal_vel_idx = axis
            src = _axis_slice(axis, num_ghost_cells, num_ghost_cells + 1, primitive_state.ndim)
            primitive_state = primitive_state.at[(normal_vel_idx,) + dst[1:]].set(
                primitive_state[(normal_vel_idx,) + src[1:]]
            )

    if (
        bs.right_boundary == FIXED_BOUNDARY or bs.right_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM 
    ):
        dst = _axis_slice(axis, -num_ghost_cells, None, primitive_state.ndim) # destination slice
        right_state = (
            params.fixed_boundary_state.x.right_state if axis == XAXIS
            else params.fixed_boundary_state.y.right_state if axis == YAXIS
            else params.fixed_boundary_state.z.right_state
        )

        # the fixed state is given as a 1D array of primitive variables, if
        # the boundary handler is applied on the conservative state we need to convert it first
        if type_handled == CONSERVATIVE_GAS_STATE:
            
            if config.mhd or config.equation_of_state == ISOTHERMAL:
                raise NotImplementedError("FIXED_BOUNDARY with conservative state is not implemented for MHD or isothermal EOS yet")
            
            # convert the fixed primitive state to conservative form
            right_state = conserved_state_from_primitive(right_state, params.gamma, config, registered_variables)

        right_state = right_state.reshape((right_state.shape[0],) + (1,) * (primitive_state.ndim - 1))
        primitive_state = primitive_state.at[dst].set(right_state)
 
        if bs.right_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM:
            normal_vel_idx = axis
            src = _axis_slice(axis, -num_ghost_cells - 1, -num_ghost_cells, primitive_state.ndim)
            # Broadcast the last-interior-cell's normal velocity into all right ghost cells
            primitive_state = primitive_state.at[(normal_vel_idx,) + dst[1:]].set(
                primitive_state[(normal_vel_idx,) + src[1:]]
            )

    return primitive_state

@partial(jax.jit, static_argnames=["num_ghost_cells", "bs"])
def _apply_axis_bcs_1d(
    primitive_state: STATE_TYPE, params: SimulationParams, num_ghost_cells: int, bs
) -> STATE_TYPE:
    """1D has a flat ``boundary_settings`` (no x/y/z split). The general
    reflective functions handle ``axis=1`` correctly — in 1D the normal-velocity
    variable index happens to equal ``axis``, which is exactly the convention
    those helpers already use."""
    if bs.left_boundary == OPEN_BOUNDARY:
        primitive_state = _open_left_boundary(primitive_state, num_ghost_cells, axis=1)
    elif bs.left_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_left_boundary(primitive_state, num_ghost_cells, axis=1)
 
    if bs.right_boundary == OPEN_BOUNDARY:
        primitive_state = _open_right_boundary(primitive_state, num_ghost_cells, axis=1)
    elif bs.right_boundary == REFLECTIVE_BOUNDARY:
        primitive_state = _reflective_right_boundary(primitive_state, num_ghost_cells, axis=1)
 
    if (
        bs.left_boundary == PERIODIC_BOUNDARY
        and bs.right_boundary == PERIODIC_BOUNDARY
    ):
        primitive_state = _periodic_boundaries(primitive_state, num_ghost_cells, axis=1)

    if (
        bs.left_boundary == FIXED_BOUNDARY or bs.left_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM
    ):
        dst = _axis_slice(1, 0, num_ghost_cells, primitive_state.ndim) # destination slice
        left_state = params.fixed_boundary_state.x.left_state
        left_state = left_state.reshape((left_state.shape[0],) + (1,) * (primitive_state.ndim - 1))
        primitive_state = primitive_state.at[dst].set(left_state)

        if bs.left_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM:
            raise NotImplementedError("FIXED_BOUNDARY_OPEN_MOMENTUM is not implemented for 1D yet")
    
    if (
        bs.right_boundary == FIXED_BOUNDARY
    ):
        dst = _axis_slice(1, -num_ghost_cells, None, primitive_state.ndim) # destination slice
        right_state = params.fixed_boundary_state.x.right_state
        right_state = right_state.reshape((right_state.shape[0],) + (1,) * (primitive_state.ndim - 1))
        primitive_state = primitive_state.at[dst].set(right_state)

        if bs.right_boundary == FIXED_BOUNDARY_OPEN_MOMENTUM:
            raise NotImplementedError("FIXED_BOUNDARY_OPEN_MOMENTUM is not implemented for 1D yet")
 
    return primitive_state
 
 
# -----------------------------------------------------------------------------
# Top-level boundary handler
# -----------------------------------------------------------------------------
 
@partial(jax.jit, static_argnames=["config", "type_handled", "registered_variables"])
def _boundary_handler(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    type_handled: int = PRIMITIVE_GAS_STATE,
) -> STATE_TYPE:
    """Apply all boundary conditions to the primitive state."""
    ng = config.num_ghost_cells
 
    if config.dimensionality == 1:
        return _apply_axis_bcs_1d(primitive_state, params, ng, config.boundary_settings)
 
    # 2D / 3D: dispatch per axis. Each call is a single traced branch.
    primitive_state = _apply_axis_bcs(
        primitive_state, params, config, registered_variables, ng, config.boundary_settings.x, axis=1, type_handled=type_handled
    )
    primitive_state = _apply_axis_bcs(
        primitive_state, params, config, registered_variables, ng, config.boundary_settings.y, axis=2, type_handled=type_handled
    )
    if config.dimensionality == 3:
        primitive_state = _apply_axis_bcs(
            primitive_state, params, config, registered_variables, ng, config.boundary_settings.z, axis=3, type_handled=type_handled
        )
 
    # MHD jet injection (2D only, y-left).
    if (
        config.dimensionality == 2
        and config.boundary_settings.y.left_boundary == MHD_JET_BOUNDARY
    ):
        primitive_state = _jet_left_boundary(
            primitive_state,
            ng,
            axis=2,
            grid_spacing=config.grid_spacing,
            num_cells=config.num_cells.y,
            type_handled=type_handled,
        )
 
    return primitive_state