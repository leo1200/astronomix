# general
import jax.numpy as jnp
import jax
from functools import partial

# type checking
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Union
from astronomix._physics_modules._stellar_wind.stellar_wind import _wind_injection
from astronomix.option_classes.simulation_config import STATE_TYPE, UNSPLIT

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equations import speed_of_sound
from astronomix._physics_modules._cosmic_rays.cr_fluid_equations import (
    gas_pressure_from_primitives_with_crs,
    speed_of_sound_crs,
)
from astronomix._physics_modules.run_physics_modules import _run_physics_modules

# better use same as in the riemann solver??
# now these wave speeds are calculated without
# the reconstruction - for the purely spatial
# reconstruction, we do not need to know
# the time step a priori


# TODO: merge duplicate code in this and hll.py
# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["registered_variables", "config", "flux_direction_index"]
)
def get_wave_speeds(
    primitives_left: STATE_TYPE,
    primitives_right: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    registered_variables: RegisteredVariables,
    config: SimulationConfig,
    flux_direction_index: int,
) -> Union[float, Float[Array, ""]]:
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

    # very simple approach for the wave velocities
    # wave_speeds_right_plus = jnp.maximum(jnp.maximum(u_L + c_L, u_R + c_R), 0)
    # wave_speeds_left_minus = jnp.minimum(jnp.minimum(u_L - c_L, u_R - c_R), 0)

    wave_speeds_right_plus = jnp.abs(u_L) + c_L
    wave_speeds_left_minus = jnp.abs(u_R) + c_R

    max_wave_speed = jnp.maximum(
        jnp.max(jnp.abs(wave_speeds_right_plus)),
        jnp.max(jnp.abs(wave_speeds_left_minus)),
    )

    return max_wave_speed


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step(
    primitive_state: STATE_TYPE,
    grid_spacing: Union[float, Float[Array, ""]],
    dt_max: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    C_CFL: Union[float, Float[Array, ""]] = 0.8,
) -> Float[Array, ""]:
    """Calculate the time step based on the CFL condition.

    Args:
        primitive_state: The primitive state array.
        grid_spacing: The cell width.
        dt_max: The maximum time step.
        gamma: The adiabatic index.
        C_CFL: The CFL number.

    Returns:
        The time step.

    """

    if config.split == UNSPLIT:
        rho = primitive_state[registered_variables.density_index]
        p = primitive_state[registered_variables.pressure_index]
        c = speed_of_sound(rho, p, gamma)
        alpha_lax = jnp.zeros((config.dimensionality,))
        for axis in range(1, config.dimensionality + 1):
            u = primitive_state[axis]
            alpha_lax_i = jnp.max(jnp.abs(u) + c)
            alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

        return C_CFL * 1 / jnp.sum(alpha_lax / grid_spacing)

    else:
        if config.dimensionality == 3:
            # wave speeds in x direction
            primitive_state_left = primitive_state[:, :-1, :, :]
            primitive_state_right = primitive_state[:, 1:, :, :]
            max_wave_speed_x = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                registered_variables.velocity_index.x,
            )

            # wave speeds in y direction
            primitive_state_left = primitive_state[:, :, :-1, :]
            primitive_state_right = primitive_state[:, :, 1:, :]
            max_wave_speed_y = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                registered_variables.velocity_index.y,
            )

            # wave speeds in z direction
            primitive_state_left = primitive_state[:, :, :, :-1]
            primitive_state_right = primitive_state[:, :, :, 1:]
            max_wave_speed_z = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                registered_variables.velocity_index.z,
            )

            # get the maximum wave speed
            max_wave_speed = jnp.maximum(
                jnp.maximum(max_wave_speed_x, max_wave_speed_y), max_wave_speed_z
            )
        elif config.dimensionality == 2:
            # wave speeds in x direction
            primitive_state_left = primitive_state[:, :-1, :]
            primitive_state_right = primitive_state[:, 1:, :]
            max_wave_speed_x = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                registered_variables.velocity_index.x,
            )

            # wave speeds in y direction
            primitive_state_left = primitive_state[:, :, :-1]
            primitive_state_right = primitive_state[:, :, 1:]
            max_wave_speed_y = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                registered_variables.velocity_index.y,
            )

            # get the maximum wave speed
            max_wave_speed = jnp.maximum(max_wave_speed_x, max_wave_speed_y)
        else:
            # wave speeds in x direction
            primitive_state_left = primitive_state[:, :-1]
            primitive_state_right = primitive_state[:, 1:]
            max_wave_speed = get_wave_speeds(
                primitive_state_left,
                primitive_state_right,
                gamma,
                registered_variables,
                config,
                registered_variables.velocity_index,
            )

        # calculate the time step
        dt = C_CFL * grid_spacing / max_wave_speed

        if config.use_max_adaptive_timestep:
            dt = jnp.minimum(dt, dt_max)
        
        return dt


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _source_term_aware_time_step(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
) -> Float[Array, ""]:
    """
    Calculate the time step based on the CFL condition and the source terms. What timestep
    would be chosen if the source terms were added under the current CFL time step?

    Args:
        state: The state array.
        config: The configuration.
        params: The parameters.
        helper_data: The helper data.

    Returns:
        The time step.
    """

    # == experimental: correct the CFL time step based on the physical sources ==

    # calculate the time step based on the CFL condition
    dt = _cfl_time_step(
        primitive_state,
        config.grid_spacing,
        params.dt_max,
        params.gamma,
        config,
        registered_variables,
        params.C_cfl,
    )

    # independent of config.use_max_adaptive_timestep
    dt = jnp.minimum(dt, params.dt_max)

    # ONLY ADD THE STELLAR WIND SOURCE TERM
    # DO NOT RUN ALL PHYSICS MODULES HERE

    # TODO: HOW TO HANDLE FURTHER SOURCE TERMS?

    hypothetical_new_state = _wind_injection(
        primitive_state, dt, config, params, helper_data, registered_variables
    )

    # hypothetical_new_state = _run_physics_modules(
    #     primitive_state,
    #     dt,
    #     config,
    #     params,
    #     helper_data,
    #     registered_variables,
    #     current_time,
    # )

    dt = _cfl_time_step(
        hypothetical_new_state,
        config.grid_spacing,
        params.dt_max,
        params.gamma,
        config,
        registered_variables,
        params.C_cfl,
    )

    # ===========================================================================

    return dt
