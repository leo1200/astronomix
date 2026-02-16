from typing import Union
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equations import conserved_state_from_primitive, speed_of_sound
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig


import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped


from functools import partial


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _lax_friedrichs_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:

    # the flux such that the array at position i stores the interface
    # from from i-1 to i.

    # primitives left at i is the left state at the interface
    # between i-1 and i so the right extrapolation from the cell i-1

    # primitives right at i is the right state at the interface
    # between i-1 and i so the left extrapolation from the cell i


    rho_L = primitives_left[registered_variables.density_index]
    u_L = primitives_left[flux_direction_index]

    rho_R = primitives_right[registered_variables.density_index]
    u_R = primitives_right[flux_direction_index]

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    conserved_left = conserved_state_from_primitive(
        primitives_left, gamma, config, registered_variables
    )
    conserved_right = conserved_state_from_primitive(
        primitives_right, gamma, config, registered_variables
    )

    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # alpha = jnp.max(jnp.maximum(jnp.abs(u_L) + c_L, jnp.abs(u_R) + c_R))
    u = primitive_state[flux_direction_index]
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    c = speed_of_sound(rho, p, gamma)
    # THE COMMON ALPHA PARAMETER MAKES NO SENSE (?) - LOOK INTO PAPER AGAIN???
    alpha = jnp.max(jnp.abs(u) + c)

    fluxes_left = _euler_flux(
        primitives_left, gamma, config, registered_variables, flux_direction_index
    )
    fluxes_right = _euler_flux(
        primitives_right, gamma, config, registered_variables, flux_direction_index
    )
    fluxes = 0.5 * (fluxes_left + fluxes_right) - 0.5 * alpha * (
        conserved_right - conserved_left
    )

    return fluxes
