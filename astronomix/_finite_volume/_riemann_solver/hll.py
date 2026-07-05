"""
HLL-family Riemann solvers for the finite-volume scheme.

Provides the HLL, HLLC (with optional low-Mach HLLC-LM correction) and the
adaptive/hybrid AM-HLLC solvers that return the conservative interface fluxes
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
from astronomix.option_classes.simulation_config import (
    AM_HLLC,
    HLLC_LM,
    HYBRID_HLLC,
    STATE_TYPE,
)

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._modules._cosmic_rays.cr_fluid_equations import speed_of_sound_crs
from astronomix._stencil_operations._stencil_operations import _stencil_add
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    get_absolute_velocity,
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
        hllc_lm: Force the low-Mach HLLC-LM wave-speed reduction even when the
            configured solver is plain HLLC.
        low_mach_dissipation_control: Scale the velocity-component dissipation
            by the local Mach number to suppress excess dissipation in the
            low-Mach regime.

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
    Adaptive/hybrid HLLC solver that blends two HLLC fluxes per interface using
    a local compression indicator.

    A compression indicator ``g`` is set to 1 where the velocity divergence is
    sufficiently negative (strongly compressive, i.e. near a shock) and 0
    elsewhere. The returned flux is the per-interface blend
    ``g * F_hllc_lm + (1 - g) * F_hllc``: compressive interfaces take the
    HLLC-LM flux (``hllc_lm=True``) and the remaining interfaces take the
    standard HLLC flux. Both HLLC fluxes are computed with
    ``low_mach_dissipation_control`` enabled for the ``AM_HLLC`` solver and
    disabled for ``HYBRID_HLLC``. See
    https://www.sciencedirect.com/science/article/pii/S1007570425005891.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        primitive_state: The full cell-centred primitive state (used for the
            divergence-based shock indicator).
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        flux_direction_index: The state index of the velocity normal to the
            interface (the flux direction).

    Returns:
        The conservative fluxes at the interfaces.
    """

    d = config.grid_spacing
    C_th = 0.05
    a = speed_of_sound(
        primitive_state[registered_variables.density_index],
        primitive_state[registered_variables.pressure_index],
        gamma,
    )
    # NOTE: not optimal — the per-dimension central difference is summed up
    # here once per dimension rather than being computed in a single fused pass.
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
