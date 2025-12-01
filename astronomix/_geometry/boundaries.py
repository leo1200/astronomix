from typing import NamedTuple
import jax.numpy as jnp
from functools import partial
import jax

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from astronomix.option_classes.simulation_config import (
    GAS_STATE,
    MAGNETIC_FIELD_ONLY,
    MHD_JET_BOUNDARY,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    STATE_TYPE,
    VELOCITY_ONLY,
    SimulationConfig,
)

# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["axis", "set_index", "get_index"])
def _set_along_axis(
    primitive_state: STATE_TYPE, axis: int, set_index: int, get_index: int
) -> STATE_TYPE:
    s_set = (
        (slice(None),) * axis
        + (set_index,)
        + (slice(None),) * (primitive_state.ndim - axis - 1)
    )
    s_get = (
        (slice(None),) * axis
        + (get_index,)
        + (slice(None),) * (primitive_state.ndim - axis - 1)
    )

    primitive_state = primitive_state.at[s_set].set(primitive_state[s_get])

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["axis", "set_index", "get_index", "var_index", "factor"]
)
def _set_specific_var_along_axis(
    primitive_state: STATE_TYPE,
    axis: int,
    set_index: int,
    get_index: int,
    var_index: int,
    factor: float,
) -> STATE_TYPE:
    s_set = (
        (var_index,)
        + (slice(None),) * (axis - 1)
        + (set_index,)
        + (slice(None),) * (primitive_state.ndim - axis - 1)
    )
    s_get = (
        (var_index,)
        + (slice(None),) * (axis - 1)
        + (get_index,)
        + (slice(None),) * (primitive_state.ndim - axis - 1)
    )

    primitive_state = primitive_state.at[s_set].set(factor * primitive_state[s_get])

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "type_handled"])
def _boundary_handler(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    type_handled: int = GAS_STATE,
) -> STATE_TYPE:
    """Apply the boundary conditions to the primitive states.

    Args:
        primitive_state: The primitive state array.
        left_boundary: The left boundary condition.
        right_boundary: The right boundary condition.

    Returns:
        The primitive state array with the boundary conditions applied.
    """

    if config.dimensionality == 1:
        if config.boundary_settings.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )
        elif config.boundary_settings.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary1d(primitive_state)

        if config.boundary_settings.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )
        elif config.boundary_settings.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary1d(primitive_state)

        if (
            config.boundary_settings.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.right_boundary == PERIODIC_BOUNDARY
        ):
            primitive_state = _periodic_boundaries(
                primitive_state, config.num_ghost_cells, axis=1
            )

    if config.dimensionality == 2:
        if config.boundary_settings.x.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.x.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.y.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.y.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.x.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.x.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.y.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.y.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY
        ):
            primitive_state = _periodic_boundaries(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if (
            config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
        ):
            primitive_state = _periodic_boundaries(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.y.left_boundary == MHD_JET_BOUNDARY:
            primitive_state = _jet_left_boundary(
                primitive_state,
                config.num_ghost_cells,
                axis=2,
                grid_spacing=config.grid_spacing,
                num_cells=config.num_cells,
                type_handled=type_handled,
            )

    if config.dimensionality == 3:
        if config.boundary_settings.x.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.x.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.y.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.y.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.z.left_boundary == OPEN_BOUNDARY:
            primitive_state = _open_left_boundary(
                primitive_state, config.num_ghost_cells, axis=3
            )

        if config.boundary_settings.z.right_boundary == OPEN_BOUNDARY:
            primitive_state = _open_right_boundary(
                primitive_state, config.num_ghost_cells, axis=3
            )

        if config.boundary_settings.x.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.x.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if config.boundary_settings.y.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.y.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if config.boundary_settings.z.left_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_left_boundary(
                primitive_state, config.num_ghost_cells, axis=3
            )

        if config.boundary_settings.z.right_boundary == REFLECTIVE_BOUNDARY:
            primitive_state = _reflective_right_boundary(
                primitive_state, config.num_ghost_cells, axis=3
            )

        if (
            config.boundary_settings.x.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.x.right_boundary == PERIODIC_BOUNDARY
        ):
            primitive_state = _periodic_boundaries(
                primitive_state, config.num_ghost_cells, axis=1
            )

        if (
            config.boundary_settings.y.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.y.right_boundary == PERIODIC_BOUNDARY
        ):
            primitive_state = _periodic_boundaries(
                primitive_state, config.num_ghost_cells, axis=2
            )

        if (
            config.boundary_settings.z.left_boundary == PERIODIC_BOUNDARY
            and config.boundary_settings.z.right_boundary == PERIODIC_BOUNDARY
        ):
            primitive_state = _periodic_boundaries(
                primitive_state, config.num_ghost_cells, axis=3
            )

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _open_right_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:

    get_index = -num_ghost_cells - 1

    for set_index in range(-num_ghost_cells, 0):
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _open_left_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:

    get_index = num_ghost_cells

    for set_index in range(num_ghost_cells):
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _periodic_boundaries(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:

    # primitive_state = _set_along_axis(primitive_state, axis, 0, -4)
    # primitive_state = _set_along_axis(primitive_state, axis, 1, -3)

    # primitive_state = _set_along_axis(primitive_state, axis, -2, 2)
    # primitive_state = _set_along_axis(primitive_state, axis, -1, 3)

    for i in range(num_ghost_cells):
        # left boundary
        set_index = i
        get_index = i - 2 * num_ghost_cells
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

        # right boundary
        set_index = -i - 1
        get_index = 2 * num_ghost_cells - i - 1
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _reflective_left_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    # reflect the velocity component along the axis
    for i in range(num_ghost_cells):
        get_index = i + num_ghost_cells
        set_index = num_ghost_cells - i - 1

        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)
        primitive_state = _set_specific_var_along_axis(
            primitive_state, axis, set_index, get_index, axis, -1.0
        )

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["axis", "num_ghost_cells"])
def _reflective_right_boundary(
    primitive_state: STATE_TYPE, num_ghost_cells: int, axis: int
) -> STATE_TYPE:
    # reflect the velocity component along the axis
    for i in range(num_ghost_cells):
        get_index = -i - num_ghost_cells - 1
        set_index = -num_ghost_cells + i

        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)
        primitive_state = _set_specific_var_along_axis(
            primitive_state, axis, set_index, get_index, axis, -1.0
        )

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_left_boundary1d(
    primitive_state: Float[Array, "num_vars num_cells"],
) -> Float[Array, "num_vars num_cells"]:

    primitive_state = primitive_state.at[:, 0].set(primitive_state[:, 2])
    primitive_state = primitive_state.at[1, 0].set(-primitive_state[1, 2])

    primitive_state = primitive_state.at[:, 1].set(primitive_state[:, 2])
    primitive_state = primitive_state.at[1, 1].set(-primitive_state[1, 2])

    return primitive_state


# @jaxtyped(typechecker=typechecker)
@jax.jit
def _reflective_right_boundary1d(
    primitive_state: Float[Array, "num_vars num_cells"],
) -> Float[Array, "num_vars num_cells"]:

    primitive_state = primitive_state.at[:, -1].set(primitive_state[:, -3])
    primitive_state = primitive_state.at[1, -1].set(-primitive_state[1, -3])

    primitive_state = primitive_state.at[:, -2].set(primitive_state[:, -3])
    primitive_state = primitive_state.at[1, -2].set(-primitive_state[1, -3])

    return primitive_state


# @jaxtyped(typechecker=typechecker)
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
    get_index = num_ghost_cells

    for set_index in range(num_ghost_cells):
        primitive_state = _set_along_axis(primitive_state, axis, set_index, get_index)

    half_inj_width = 0.025
    half_inj_cell_num = int(half_inj_width / grid_spacing)

    B0 = 200**0.5
    # to_set = jnp.array([5/3, 0.0, 800.0, 0.0, 1.0, 0.0, B0, 0.0])
    #                     rho, v_x, v_y,   v_z, p,   B_x, B_y, B_z
    to_set_gas_state = jnp.array([5 / 3, 800.0, 0.0, 1.0])
    to_set_velocity = jnp.array([800.0, 0.0, 0.0])
    to_set_magnetic_field = jnp.array([B0, 0.0, 0.0])

    if type_handled == GAS_STATE:
        primitive_state = primitive_state.at[
            :,
            0:num_ghost_cells,
            (num_cells // 2 - half_inj_cell_num) : (num_cells // 2 + half_inj_cell_num),
        ].set(to_set_gas_state[:, None, None])
    elif type_handled == VELOCITY_ONLY:
        primitive_state = primitive_state.at[
            :,
            0:num_ghost_cells,
            (num_cells // 2 - half_inj_cell_num) : (num_cells // 2 + half_inj_cell_num),
        ].set(to_set_velocity[:, None, None])
    elif type_handled == MAGNETIC_FIELD_ONLY:
        primitive_state = primitive_state.at[
            :,
            0:num_ghost_cells,
            (num_cells // 2 - half_inj_cell_num) : (num_cells // 2 + half_inj_cell_num),
        ].set(to_set_magnetic_field[:, None, None])

    return primitive_state
