"""
Interface reconstruction for the finite-volume scheme.

Provides the MUSCL split reconstruction (with the predictor half-step) and the
unsplit reconstruction that extrapolate the cell-centred primitive state to the
left/right states at each interface.
"""

# general
from functools import partial

# typing
from typing import Union
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    MUSCL,
    STATE_TYPE,
    STATE_TYPE_ALTERED,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix._fluid_equations._equations import speed_of_sound
from astronomix._finite_volume._state_evolution.limited_gradients import _calculate_limited_gradients


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables", "axis"])
def _reconstruct_at_interface_split(
    primitive_state: STATE_TYPE,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    axis: int,
) -> tuple[STATE_TYPE_ALTERED, STATE_TYPE_ALTERED]:

    # get fluid variables for convenience
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    u = primitive_state[axis]

    # get the limited gradients on the cells
    limited_gradients = _calculate_limited_gradients(
        primitive_state, config, helper_data, axis=axis
    )

    if config.time_integrator == MUSCL:
        # calculate the sound speed
        c = speed_of_sound(rho, p, gamma)

        # ================ construct A_W, the "primitive Jacabian" (not an actual Jacabian) ================
        # see https://diglib.uibk.ac.at/download/pdf/4422963.pdf, 2.11

        # calculate the vectors making up A_W
        A_W = jnp.zeros((registered_variables.num_vars,) + primitive_state.shape)

        # set u diagonal, this way all quantities are automatically advected
        A_W = A_W.at[
            jnp.arange(registered_variables.num_vars),
            jnp.arange(registered_variables.num_vars),
        ].set(u)

        # set rest
        A_W = A_W.at[registered_variables.density_index, axis].set(rho)
        A_W = A_W.at[registered_variables.pressure_index, 1].set(rho * c**2)
        A_W = A_W.at[axis, registered_variables.pressure_index].set(1 / rho)

        # ====================================================================================================

        # project the gradients
        if config.dimensionality == 1:
            projected_gradients = jnp.einsum("bax, ax -> bx", A_W, limited_gradients)
        elif config.dimensionality == 2:
            projected_gradients = jnp.einsum("baxy, axy -> bxy", A_W, limited_gradients)
        elif config.dimensionality == 3:
            projected_gradients = jnp.einsum(
                "baxyz, axyz -> bxyz", A_W, limited_gradients
            )

        # predictor step
        predictors = primitive_state - dt / 2 * projected_gradients
    else:
        raise ValueError(
            f"Time integrator {config.time_integrator} not supported for split reconstruction. Only MUSCL is supported."
        )

    # compute primitives at the interfaces
    if config.geometry == CARTESIAN:
        distances_to_left_interfaces = (
            config.grid_spacing / 2
        )  # distances r_i - r_{i-1/2}
        distances_to_right_interfaces = (
            config.grid_spacing / 2
        )  # distances r_{i+1/2} - r_i
    else:
        r = helper_data.geometric_centers
        rv = helper_data.volumetric_centers

        distances_to_left_interfaces = rv - (r - config.grid_spacing / 2)
        distances_to_right_interfaces = (r + config.grid_spacing / 2) - rv

    primitives_left = predictors - distances_to_left_interfaces * limited_gradients
    primitives_right = predictors + distances_to_right_interfaces * limited_gradients

    # primitives left at i is the left state at the interface
    # between i-1 and i so the right extrapolation from the cell i-1
    p_left_interface = jnp.roll(primitives_right, shift=1, axis=axis)

    # primitives right at i is the right state at the interface
    # between i-1 and i so the left extrapolation from the cell i
    p_right_interface = primitives_left

    return p_left_interface, p_right_interface


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "axis"])
def _reconstruct_at_interface_unsplit_single(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    helper_data: HelperData,
    axis: int,
):
    """
    Unsplit reconstruction.
    """

    limited_gradients = _calculate_limited_gradients(
        primitive_state, config, helper_data, axis=axis
    )
    differences = limited_gradients * config.grid_spacing / 2

    # i-1/2R, ...
    primitives_left_center = (
        primitive_state - differences
    )  # left of the cell center but the right of the interface
    # i+1/2L, ...
    primitives_right_center = (
        primitive_state + differences
    )  # right of the cell center but the left of the interface

    # primitives left at i is the left state at the interface
    # between i-1 and i so the right extrapolation from the cell i-1
    p_left_interface = jnp.roll(primitives_right_center, shift=1, axis=axis)

    # primitives right at i is the right state at the interface
    # between i-1 and i so the left extrapolation from the cell i
    p_right_interface = primitives_left_center

    return p_left_interface, p_right_interface
