"""
Iteration-level updates applied once before each hydro step.

These are the physics modules that run as a discrete update on the primitive
state at the start of every time step — stellar wind, cooling, the
neural-net / CNN correctors, viscosity, turbulent forcing, frame tracking and
the per-step positivity floor. Their counterpart is
``_time_integrator_sources``, which instead enters the hydro integrator as a
right-hand-side source term.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    IDEAL_GAS,
    POSITIVITY_HARD_FLOOR,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector import _cnn_mhd_corrector
from astronomix._modules._cooling._cooling import update_pressure_by_cooling
from astronomix._modules._frame_tracking._frame_tracking import _frame_tracking
from astronomix._modules._neural_net_force._neural_net_force import _neural_net_force
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection
from astronomix._modules._turbulent_forcing._turbulent_forcing import (
    _apply_forcing,
    _apply_ou_forcing,
)
from astronomix._modules._viscosity._viscosity import fv_viscosity_update


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_density_floor(
    primitive_state: STATE_TYPE,
    minimum_density: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    maximum_alfven_speed: Union[float, Float[Array, ""]] = float("inf"),
) -> STATE_TYPE:
    """
    AthenaPK's density floor (``ConsToPrim`` in ``eos/adiabatic_glmmhd.hpp``):
    rho <- max(rho, minimum_density) with the momentum and the total energy
    unchanged. The velocity of a floored cell therefore drops to
    v rho_old / rho_new, and for an ideal gas the kinetic energy this removes
    goes into the pressure (and the dual-energy internal energy). Cells above
    the floor are left exactly as they are.

    With a finite ``maximum_alfven_speed`` (MHD) the floor also includes
    AthenaPK's Alfven-speed ceiling (``vAceil`` in
    ``pgen/cluster/cluster_clips.cpp``): rho >= |B|^2 / maximum_alfven_speed^2,
    which bounds the Alfven speed of near-vacuum, strongly magnetised cells, and
    so the time step, however strong their field grows. AthenaPK raises the
    density to sqrt(B^2 / vAceil^2) there, which is not a density; here it is
    B^2 / vAceil^2. With the default ``inf`` the ceiling adds nothing, and the
    result is that of the plain floor bit for bit.

    Args:
        primitive_state: The primitive state.
        minimum_density: The density floor.
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        maximum_alfven_speed: The Alfven-speed ceiling (``inf``: off).

    Returns:
        The floored primitive state.
    """
    density_index = registered_variables.density_index
    velocity_index = registered_variables.velocity_index
    if isinstance(velocity_index, int):
        velocity_indices = (velocity_index,)
    else:
        velocity_indices = tuple(
            index for index in (velocity_index.x, velocity_index.y, velocity_index.z) if index >= 0
        )

    density = primitive_state[density_index]
    density_floor = minimum_density
    if config.mhd:
        magnetic_index = registered_variables.magnetic_index
        magnetic_squared = sum(
            primitive_state[index] ** 2
            for index in (magnetic_index.x, magnetic_index.y, magnetic_index.z)
        )
        density_floor = jnp.maximum(density_floor, magnetic_squared / maximum_alfven_speed ** 2)
    floored_density = jnp.maximum(density, density_floor)
    # Exactly 1 in every cell above the floor, so those cells keep their
    # velocity bit for bit.
    momentum_ratio = density / floored_density

    if config.equation_of_state == IDEAL_GAS:
        velocity_squared = sum(primitive_state[index] ** 2 for index in velocity_indices)
        released_kinetic_energy = 0.5 * density * velocity_squared * (1.0 - momentum_ratio)
        primitive_state = primitive_state.at[registered_variables.pressure_index].add(
            (gamma - 1.0) * released_kinetic_energy
        )
        if registered_variables.internal_energy_index >= 0:
            primitive_state = primitive_state.at[registered_variables.internal_energy_index].add(
                released_kinetic_energy
            )

    primitive_state = primitive_state.at[density_index].set(floored_density)
    for index in velocity_indices:
        primitive_state = primitive_state.at[index].multiply(momentum_ratio)
    return primitive_state


def _iteration_level_updates(
    primitive_state: STATE_TYPE,
    key,
    forcing,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """
    Apply the updates that run once before each hydro iteration.

    The counterpart of this are the ``_time_integrator_sources``, which are
    handled as right-hand-side source terms inside the hydro integrator.

    Args:
        primitive_state: The primitive state array.
        key: The PRNG key.
        forcing: The persistent Ornstein-Uhlenbeck forcing field (or ``None``
            when OU forcing is inactive). Carried across steps in ``LoopState``.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.
        current_time: The current simulation time.

    Returns:
        ``(key, forcing, primitive_state)`` with the iteration-level updates
        applied.
    """

    # Stellar wind.
    # In the finite-difference case this is instead handled as a source term
    # inside the hydro integrator.
    if config.wind_config.stellar_wind and config.solver_mode == FINITE_VOLUME:
        primitive_state = _wind_injection(
            primitive_state,
            dt,
            config,
            params,
            helper_data,
            registered_variables,
        )

    # Cooling.
    # In the finite-difference case this is instead handled as a source term
    # inside the hydro integrator. The grid spacing enables the
    # cooling-resolution limiter, as on the finite-difference path.
    if config.cooling_config.cooling and config.solver_mode == FINITE_VOLUME:
        primitive_state = update_pressure_by_cooling(
            primitive_state,
            registered_variables,
            config.cooling_config,
            params,
            dt,
            grid_spacing=config.grid_spacing,
        )

    # Neural-network body force.
    if config.neural_net_force_config.neural_net_force:
        primitive_state = _neural_net_force(
            primitive_state,
            config,
            registered_variables,
            params,
            helper_data,
            dt,
            current_time,
        )

    # CNN-based MHD corrector.
    if config.cnn_mhd_corrector_config.cnn_mhd_corrector:
        primitive_state = _cnn_mhd_corrector(
            primitive_state,
            config,
            registered_variables,
            params,
            dt,
        )

    # Viscosity.
    # In the finite-difference case this is instead handled as a source term
    # inside the hydro integrator.
    if config.diffusion and config.solver_mode == FINITE_VOLUME:
        primitive_state = fv_viscosity_update(
            primitive_state,
            params,
            config,
            registered_variables,
            dt,
        )

    # Turbulent forcing. The PRNG key and (for OU forcing) the persistent
    # forcing field are threaded through the loop in ``LoopState`` (see
    # astronomix/time_stepping/time_integration.py).
    if config.turbulent_forcing_config.turbulent_forcing:
        if config.turbulent_forcing_config.ou_forcing:
            # OU forcing carries a persistent solenoidal field ``forcing``;
            # bundle it with the key into the (key, field) state the OU update
            # expects and unpack the advanced state back out.
            (key, forcing), primitive_state = _apply_ou_forcing(
                (key, forcing),
                primitive_state,
                dt,
                params.turbulent_forcing_params,
                config,
                registered_variables,
            )
        else:
            key, primitive_state = _apply_forcing(
                key,
                primitive_state,
                dt,
                params.turbulent_forcing_params,
                config,
                registered_variables,
            )

    # Frame tracking. Preliminary and currently very specialized; only 3D is
    # supported at the moment.
    if config.frame_tracking:
        primitive_state = _frame_tracking(
            primitive_state,
            config,
            params,
            registered_variables,
            helper_data,
        )

    # Per-step state floor (HARD_FLOOR): AthenaPK's density floor, momentum and
    # total energy kept (see _apply_density_floor), then the pressure floor.
    # The finite-difference scheme stays positive through the
    # positivity-preserving WENO; there the density floor bounds the Alfven
    # speed of near-vacuum, strongly magnetised cells, which would otherwise
    # set a vanishing time step. It also serves the finite-volume solver and,
    # with ``per_step_specific_floor``, acts as the temperature floor of
    # radiatively cooled runs. The time loop applies the density floor once
    # more right after the state update, so the time step and the snapshots
    # see the floored state; here it then only acts on density that a module
    # above has changed.
    if config.positivity_config.per_step_mode == POSITIVITY_HARD_FLOOR:
        primitive_state = _apply_density_floor(
            primitive_state,
            params.minimum_density,
            params.gamma,
            config,
            registered_variables,
            params.maximum_alfven_speed,
        )
        if config.equation_of_state == IDEAL_GAS:
            pressure_floor = params.minimum_pressure
            if config.positivity_config.per_step_specific_floor:
                # With the density-scaled temperature floor
                # p >= rho * minimum_specific_pressure, radiatively cooled
                # dense layers keep isothermal pressure support instead of
                # falling onto the comparatively negligible constant floor.
                pressure_floor = jnp.maximum(
                    pressure_floor,
                    primitive_state[registered_variables.density_index]
                    * params.minimum_specific_pressure,
                )
            primitive_state = primitive_state.at[registered_variables.pressure_index].set(
                jnp.maximum(
                    primitive_state[registered_variables.pressure_index],
                    pressure_floor,
                )
            )

    return key, forcing, primitive_state
