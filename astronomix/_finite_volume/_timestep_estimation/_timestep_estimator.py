"""
Finite-volume time-step estimation.

Provides the maximum wave-speed estimate, the CFL time step (split and unsplit,
including the viscous constraint) and the experimental source-term-aware time
step that accounts for the stellar-wind injection.
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
    DYNAMIC_VISCOSITY,
    GHOST_CELLS,
    HLL5R,
    KINEMATIC_VISCOSITY,
    MUSCL_HANCOCK,
    RK2_SSP,
    STATE_TYPE,
    VL2,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix._fluid_equations._fluxes import _euler_flux
from astronomix._fluid_equations._equation_of_state import (
    is_barotropic,
    sound_speed_squared,
)
from astronomix._fluid_equations._equations import speed_of_sound
from astronomix._finite_volume._interface_frame import (
    _barotropic_law,
    _interface_frame_indices,
    _mhd_face_state,
)
from astronomix._finite_volume._riemann._face_state import (
    adiabatic_sound_speed,
    fast_magnetosonic_speed,
)
from astronomix._finite_volume._riemann._hll5r import hll5r_signal_speed
from astronomix.time_stepping._utils import _unpad


# NOTE: these wave speeds are computed without the reconstruction. For the
# purely spatial reconstruction we do not need to know the time step a priori,
# so a coarser estimate than the one used in the Riemann solver is acceptable
# here.
# TODO: merge the duplicate wave-speed code shared with hll.py.
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
    Return the maximum signal (wave) speed across all interfaces along an axis.

    Args:
        primitives_left: States left of the interfaces.
        primitives_right: States right of the interfaces.
        gamma: The adiabatic index.
        registered_variables: The registered variables.
        config: The simulation configuration.
        flux_direction_index: The state index of the velocity normal to the
            interface (the flux direction).

    Returns:
        The maximum wave speed over all interfaces along the given axis.
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

    # A simple symmetric estimate of the maximum signal speed on either side of
    # the interface; the |u| + c form is sufficient for the time-step bound.
    wave_speeds_right_plus = jnp.abs(u_L) + c_L
    wave_speeds_left_minus = jnp.abs(u_R) + c_R

    max_wave_speed = jnp.maximum(
        jnp.max(jnp.abs(wave_speeds_right_plus)),
        jnp.max(jnp.abs(wave_speeds_left_minus)),
    )

    return max_wave_speed


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _maximum_signal_speed_vl2(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """
    The largest hyperbolic signal speed ``max over cells and axes of
    |v_d| + c_d`` of the VL2 scheme, with ``c_d`` the fast magnetosonic speed
    along axis ``d`` (MHD) or the sound speed (hydrodynamics).

    It sets both the CFL time step and, as in AthenaPK, the GLM divergence
    cleaning speed ``c_h``. Ghost cells are excluded.

    Args:
        primitive_state: The primitive state (padded when ghost cells are used).
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The maximum signal speed.
    """
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _unpad(primitive_state, config)

    density = primitive_state[registered_variables.density_index]
    if is_barotropic(config):
        # With an adiabatic index of one the speeds below read the sound speed of the law from rho a^2.
        adiabatic_index = 1.0
        pressure = density * sound_speed_squared(density, None, params, config)
    else:
        adiabatic_index = params.gamma
        pressure = primitive_state[registered_variables.pressure_index]

    # The 1D hydro layout stores a single velocity component as a plain index.
    if isinstance(registered_variables.velocity_index, int):
        velocity_indices = (registered_variables.velocity_index,)
    else:
        velocity_indices = tuple(registered_variables.velocity_index)[: config.dimensionality]

    if config.mhd:
        magnetic_indices = tuple(registered_variables.magnetic_index)

    maximum_speed = None
    for axis in range(config.dimensionality):
        if config.mhd:
            # The fast speed along an axis treats that axis' field component
            # as the normal one and the other two as transverse.
            signal_speed = fast_magnetosonic_speed(
                adiabatic_index,
                density,
                pressure,
                primitive_state[magnetic_indices[axis]],
                primitive_state[magnetic_indices[(axis + 1) % 3]],
                primitive_state[magnetic_indices[(axis + 2) % 3]],
            )
        else:
            signal_speed = adiabatic_sound_speed(adiabatic_index, density, pressure)
        axis_maximum = jnp.max(jnp.abs(primitive_state[velocity_indices[axis]]) + signal_speed)
        if maximum_speed is None:
            maximum_speed = axis_maximum
        else:
            maximum_speed = jnp.maximum(maximum_speed, axis_maximum)

    return maximum_speed


def _maximum_hll5r_speed(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """
    The largest signal speed of the HLL5R solver over the interfaces of all axes, with the cell values as face
    states. The speeds of the solver exceed the fast speed of the cells (by up to 9.9 per cent for constant
    data, by more where neighbouring cells approach each other), so a time step from ``|v| + c_fast`` does not
    bound them (Waagan 2009, eq. 3.37, takes the speeds of the solver as well).

    The speeds are those of the cell values at the start of the step. Waagan, Federrath & Klingenberg (2011) take
    the largest wave speed of the previous step instead; the edge values inside the sweeps exceed the estimate
    here by about ten per cent in Mach 4 turbulence (Courant number 0.85 to 0.88 for ``C_CFL = 0.8``).

    Args:
        primitive_state: The primitive state (padded when ghost cells are used).
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The maximum signal speed of the solver.
    """
    barotropic_law = _barotropic_law(params, config)
    exponent = params.gamma if barotropic_law is None else barotropic_law.exponent
    sound_speed_squared_of = None if barotropic_law is None else barotropic_law.sound_speed_squared_of

    maximum_speed = 0.0
    for axis in range(1, config.dimensionality + 1):
        if config.boundary_handling == GHOST_CELLS:
            # all interfaces between the cells of the padded state
            left_states = jax.lax.slice_in_dim(primitive_state, 0, -1, axis=axis)
            right_states = jax.lax.slice_in_dim(primitive_state, 1, None, axis=axis)
        else:
            left_states, right_states = jnp.roll(primitive_state, 1, axis=axis), primitive_state
        frame_indices = _interface_frame_indices(axis, config, registered_variables)
        signal_speed = hll5r_signal_speed(
            _mhd_face_state(left_states, frame_indices, barotropic_law),
            _mhd_face_state(right_states, frame_indices, barotropic_law),
            exponent,
            sound_speed_squared_of,
        )
        maximum_speed = jnp.maximum(maximum_speed, jnp.max(signal_speed))
    return maximum_speed


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _cfl_time_step(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """Calculate the time step based on the CFL condition.

    Args:
        primitive_state: The primitive state array.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The time step.

    """

    C_CFL = params.C_cfl
    grid_spacing = config.grid_spacing
    dt_max = params.dt_max
    gamma = params.gamma

    if config.solver_config.riemann_solver == HLL5R:
        maximum_speed = _maximum_hll5r_speed(primitive_state, config, params, registered_variables)
        return jnp.minimum(C_CFL * grid_spacing / maximum_speed, dt_max)

    # A barotropic gas under the split scheme takes the cell criterion as well: the interface estimate below is
    # the same maximum of |v_d| + c_d, read from the pressure slot.
    is_barotropic_split = is_barotropic(config) and config.solver_config.time_integrator == MUSCL_HANCOCK
    if config.solver_config.time_integrator == VL2 or is_barotropic_split:
        maximum_speed = _maximum_signal_speed_vl2(
            primitive_state,
            config,
            params,
            registered_variables,
        )
        dt = C_CFL * grid_spacing / maximum_speed
        dt = jnp.minimum(dt, dt_max)
        return dt

    if config.solver_config.time_integrator == RK2_SSP:
        rho = primitive_state[registered_variables.density_index]
        if is_barotropic(config):
            c = jnp.sqrt(sound_speed_squared(rho, None, params, config))
        else:
            p = primitive_state[registered_variables.pressure_index]
            c = speed_of_sound(rho, p, gamma)
        alpha_lax = jnp.zeros((config.dimensionality,))
        for axis in range(1, config.dimensionality + 1):
            u = primitive_state[axis]
            alpha_lax_i = jnp.max(jnp.abs(u) + c)
            alpha_lax = alpha_lax.at[axis - 1].set(alpha_lax_i)

        dt = C_CFL * 1 / jnp.sum(alpha_lax / grid_spacing)

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

        dt = jnp.minimum(dt, dt_max)

    # viscous time step constraint
    if config.diffusion_config.viscosity:
        
        if config.positivity_config.clamp_in_estimates:
            rho_min = jnp.maximum(
                jnp.min(primitive_state[registered_variables.density_index]),
                params.minimum_density,
            )
        else:
            rho_min = jnp.min(primitive_state[registered_variables.density_index])
        
        if config.diffusion_config.viscosity_type == DYNAMIC_VISCOSITY:
            nu_max = params.viscosity / rho_min
        elif config.diffusion_config.viscosity_type == KINEMATIC_VISCOSITY:
            nu_max = params.viscosity

        dt_visc = C_CFL * grid_spacing**2 / (2.0 * config.dimensionality * nu_max)
        dt = jnp.minimum(dt, dt_visc)

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
        config,
        params,
        registered_variables,
    )

    dt = jnp.minimum(dt, params.dt_max)

    # ONLY ADD THE STELLAR WIND SOURCE TERM
    # DO NOT RUN ALL PHYSICS MODULES HERE

    # TODO: HOW TO HANDLE FURTHER SOURCE TERMS?

    hypothetical_new_state = _wind_injection(
        primitive_state, dt, config, params, helper_data, registered_variables
    )

    dt = _cfl_time_step(
        hypothetical_new_state,
        config,
        params,
        registered_variables,
    )

    # ===========================================================================

    return dt
