# general imports
from functools import partial
import jax.numpy as jnp
import jax

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union

# general astronomix
from astronomix._physics_modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix.variable_registry.registered_variables import RegisteredVariables

# fluid stuff
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    get_absolute_velocity,
    speed_of_sound,
)
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix.option_classes.simulation_config import (
    AM_HLLC,
    HLLC_LM,
    HYBRID_HLLC,
    STATE_TYPE,
    SimulationConfig,
)


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
    if not config.cosmic_ray_config.cosmic_rays:
        c_L = speed_of_sound(rho_L, p_L, gamma)
        c_R = speed_of_sound(rho_R, p_R, gamma)
    else:
        c_L = speed_of_sound_crs(primitives_left, registered_variables)
        c_R = speed_of_sound_crs(primitives_right, registered_variables)

    # get the left and right states and fluxes
    fluxes_left = _euler_flux(
        primitives_left, gamma, config, registered_variables, flux_direction_index
    )
    fluxes_right = _euler_flux(
        primitives_right, gamma, config, registered_variables, flux_direction_index
    )

    # very simple approach for the wave velocities
    wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    # wave_speeds_right_plus = jnp.maximum(u_L - c_L, 0)
    # wave_speeds_left_minus = jnp.minimum(u_R + c_R, 0)

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
    jax.jit,
    static_argnames=[
        "config",
        "registered_variables",
        "flux_direction_index",
        "hllc_lm",
        "low_mach_dissipation_control",
    ],
)
def _hllc_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
    hllc_lm: bool = False,
    low_mach_dissipation_control: bool = False,
) -> STATE_TYPE:

    # There seem to be problems for 1d radial, maybe because the same stuff that has to be
    # taken into consideration in the general scheme (averaging based on density might be
    # problematic, as the surface increases radially, ...) is not taken into account here.
    # Maybe interesting for future research, for now HLL works fine.

    rho_L = primitives_left[registered_variables.density_index]
    u_L = primitives_left[flux_direction_index]

    rho_R = primitives_right[registered_variables.density_index]
    u_R = primitives_right[flux_direction_index]

    p_L = primitives_left[registered_variables.pressure_index]
    p_R = primitives_right[registered_variables.pressure_index]

    # calculate the sound speeds
    if not config.cosmic_ray_config.cosmic_rays:
        c_L = speed_of_sound(rho_L, p_L, gamma)
        c_R = speed_of_sound(rho_R, p_R, gamma)
    else:
        c_L = speed_of_sound_crs(primitives_left, registered_variables)
        c_R = speed_of_sound_crs(primitives_right, registered_variables)

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

    # HLLC-LM adaptation
    # following
    # https://doi.org/10.1016/j.jcp.2020.109762
    if config.riemann_solver == HLLC_LM or hllc_lm:
        Ma_limit = 0.1
        Ma_local = jnp.maximum(jnp.abs(u_L / c_L), jnp.abs(u_R / c_R))
        phi = jnp.sin(jnp.minimum(1, Ma_local / Ma_limit) * jnp.pi / 2)
        S_Llm = S_L * phi
        S_Rlm = S_R * phi

    if config.riemann_solver == HLLC_LM or hllc_lm:
        S_Lstar = S_Llm
        S_Rstar = S_Rlm
    else:
        S_Lstar = S_L
        S_Rstar = S_R

    bulk_flux_star = 0.5 * (F_L + F_R)
    dissipation_term_star = 0.5 * (
        S_Lstar * (U_star_L - U_L)
        + jnp.abs(S_star) * (U_star_L - U_star_R)
        + S_Rstar * (U_star_R - U_R)
    )

    if low_mach_dissipation_control:
        absolute_velocity_L = get_absolute_velocity(
            primitives_left, config, registered_variables
        )
        absolute_velocity_R = get_absolute_velocity(
            primitives_right, config, registered_variables
        )
        Ma_tilde = jnp.maximum(
            jnp.abs(absolute_velocity_L / c_L), jnp.abs(absolute_velocity_R / c_R)
        )
        f = jnp.minimum(1, Ma_tilde)

        if config.dimensionality == 1:
            velocity_start_index = registered_variables.velocity_index
        else:
            velocity_start_index = registered_variables.velocity_index.x

        dissipation_term_star = dissipation_term_star.at[
            velocity_start_index : velocity_start_index + config.dimensionality
        ].set(
            f
            * dissipation_term_star[
                velocity_start_index : velocity_start_index + config.dimensionality
            ]
        )

    F_star = bulk_flux_star + dissipation_term_star

    fluxes = jnp.where(S_L >= 0, F_L, F_star)
    fluxes = jnp.where(S_R <= 0, F_R, fluxes)

    return fluxes


# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["config", "registered_variables", "flux_direction_index"]
)
def _am_hllc_solver(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    flux_direction_index: int,
) -> STATE_TYPE:
    """
    https://www.sciencedirect.com/science/article/pii/S1007570425005891
    """

    d = config.grid_spacing
    C_th = 0.05
    a = speed_of_sound(
        primitive_state[registered_variables.density_index],
        primitive_state[registered_variables.pressure_index],
        gamma,
    )
    # not optimal, sum over all dimensions
    # is carried out once per dimension
    div_v = sum(
        _stencil_add(
            primitive_state[i + 1], indices=(1, -1), factors=(1.0, -1.0), axis=i
        )
        / (2 * d)
        for i in range(config.dimensionality)
    )
    g = jnp.where(div_v < -C_th * a / d, 1, 0)

    if config.riemann_solver == AM_HLLC:
        low_mach_dissipation_control = True
    elif config.riemann_solver == HYBRID_HLLC:
        low_mach_dissipation_control = False
    else:
        raise ValueError("Riemann solver not supported for AM-HLLC.")

    fluxes_hllc = _hllc_solver(
        primitives_left,
        primitives_right,
        gamma,
        config,
        registered_variables,
        flux_direction_index,
        low_mach_dissipation_control=low_mach_dissipation_control,
    )
    fluxes_hllc_lm = _hllc_solver(
        primitives_left,
        primitives_right,
        gamma,
        config,
        registered_variables,
        flux_direction_index,
        hllc_lm=True,
        low_mach_dissipation_control=low_mach_dissipation_control,
    )

    return g * fluxes_hllc_lm + (1 - g) * fluxes_hllc
