"""
HLL-family Riemann solvers for the finite-volume scheme.

Provides the HLL and HLLC solvers that return the conservative interface fluxes
from the reconstructed left/right primitive states.
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
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    speed_of_sound,
)
from astronomix._fluid_equations._fluxes import _euler_flux


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _hll_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """
    Returns the conservative fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.

    Returns:
        The conservative fluxes at the interfaces.

    """

    rho_L = primitives_left[registered_variables.density_index]

    u_L = primitives_left[flux_direction_index]

    rho_R = primitives_right[registered_variables.density_index]

    u_R = primitives_right[flux_direction_index]

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # get the left and right states and fluxes
    fluxes_left = _euler_flux(
        primitives_left, gamma, config, registered_variables, flux_direction_index
    )
    fluxes_right = _euler_flux(
        primitives_right, gamma, config, registered_variables, flux_direction_index
    )

    # Estimate the fastest right- and left-running signal speeds and clamp them
    # to be one-sided (>= 0 and <= 0 respectively), so the HLL average reduces
    # to the upwind flux when the whole fan moves in one direction.
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # get the left and right conserved variables
    conserved_left = conserved_state_from_primitive(
        primitives_left, gamma, config, registered_variables
    )
    conserved_right = conserved_state_from_primitive(
        primitives_right, gamma, config, registered_variables
    )

    # calculate the interface HLL fluxes
    # F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)
    fluxes = (
        wave_speeds_right_plus * fluxes_left
        - wave_speeds_left_minus * fluxes_right
        + wave_speeds_left_minus
        * wave_speeds_right_plus
        * (conserved_right - conserved_left)
    ) / (wave_speeds_right_plus - wave_speeds_left_minus)

    return fluxes


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _hllc_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """
    HLLC Riemann solver returning the conservative interface fluxes.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        flux_direction_index: The state index of the velocity normal to the
            interface (the flux direction).

    Returns:
        The conservative fluxes at the interfaces.
    """

    # NOTE: this scheme shows problems for the 1D radial case, presumably
    # because the radially varying cell surfaces (which make the density-based
    # averaging questionable there) are not accounted for. That regime falls
    # back to HLL, which works fine; revisiting HLLC for radial geometry is
    # left for future work.

    rho_L = primitives_left[registered_variables.density_index]
    u_L = primitives_left[flux_direction_index]

    rho_R = primitives_right[registered_variables.density_index]
    u_R = primitives_right[flux_direction_index]

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    c_L = speed_of_sound(rho_L, p_L, gamma)
    c_R = speed_of_sound(rho_R, p_R, gamma)

    # get the left and right states and fluxes
    F_L = _euler_flux(
        primitives_left, gamma, config, registered_variables, flux_direction_index
    )
    F_R = _euler_flux(
        primitives_right, gamma, config, registered_variables, flux_direction_index
    )

    # Roe average of the velocity
    u_hat = (jnp.sqrt(rho_L) * u_L + jnp.sqrt(rho_R) * u_R) / (
        jnp.sqrt(rho_L) + jnp.sqrt(rho_R)
    )

    # Roe average of the sound speed
    c_hat_squared = (c_L**2 * jnp.sqrt(rho_L) + c_R**2 * jnp.sqrt(rho_R)) / (
        jnp.sqrt(rho_L) + jnp.sqrt(rho_R)
    ) + 0.5 * (
        jnp.sqrt(rho_L) * jnp.sqrt(rho_R) / (jnp.sqrt(rho_L) + jnp.sqrt(rho_R)) ** 2
    ) * (u_R - u_L) ** 2
    c_hat = jnp.sqrt(c_hat_squared)

    # Einfeldt estimates of maximum left and right signal speeds
    S_L = jnp.minimum(u_L - c_L, u_hat - c_hat)
    S_R = jnp.maximum(u_R + c_R, u_hat + c_hat)

    # contact wave signal speed
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
        rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
    )

    # intermediate states
    U_L = conserved_state_from_primitive(
        primitives_left, gamma, config, registered_variables
    )
    U_R = conserved_state_from_primitive(
        primitives_right, gamma, config, registered_variables
    )

    U_star_L = U_L.at[flux_direction_index].set(rho_L * S_star)
    U_star_L = U_star_L.at[registered_variables.pressure_index].add(
        (S_star - u_L) * (rho_L * S_star + p_L / (S_L - u_L))
    )
    U_star_L = U_star_L * (S_L - u_L) / (S_L - S_star)

    U_star_R = U_R.at[flux_direction_index].set(rho_R * S_star)
    U_star_R = U_star_R.at[registered_variables.pressure_index].add(
        (S_star - u_R) * (rho_R * S_star + p_R / (S_R - u_R))
    )
    U_star_R = U_star_R * (S_R - u_R) / (S_R - S_star)

    bulk_flux_star = 0.5 * (F_L + F_R)
    dissipation_term_star = 0.5 * (
        S_L * (U_star_L - U_L)
        + jnp.abs(S_star) * (U_star_L - U_star_R)
        + S_R * (U_star_R - U_R)
    )

    F_star = bulk_flux_star + dissipation_term_star

    fluxes = jnp.where(S_L >= 0, F_L, F_star)
    fluxes = jnp.where(S_R <= 0, F_R, fluxes)

    return fluxes
