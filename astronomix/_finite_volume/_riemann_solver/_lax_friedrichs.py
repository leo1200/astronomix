"""
Local Lax-Friedrichs (Rusanov) Riemann solver for the finite-volume scheme.

Returns the conservative interface fluxes from the reconstructed left/right
primitive states, using a single global dissipation coefficient ``alpha`` from
the cell-centred state.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equations import conserved_state_from_primitive, speed_of_sound


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
    """
    Local Lax-Friedrichs (Rusanov) solver returning the conservative interface
    fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        primitive_state: The full cell-centred primitive state (used to set the
            global dissipation coefficient ``alpha``).
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        flux_direction_index: The state index of the velocity normal to the
            interface (the flux direction).

    Returns:
        The conservative fluxes at the interfaces.
    """

    # The flux array is laid out so that position i stores the interface flux
    # from cell i-1 to cell i. ``primitives_left`` at i is the right
    # extrapolation from cell i-1 (the left state of that interface) and
    # ``primitives_right`` at i is the left extrapolation from cell i (the
    # right state of that interface).

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

    u = primitive_state[flux_direction_index]
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    c = speed_of_sound(rho, p, gamma)
    # A single global dissipation coefficient is used for every interface,
    # taken as the maximum signal speed over the whole cell-centred state.
    # TODO: revisit whether a per-interface (local) alpha is preferable here.
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
