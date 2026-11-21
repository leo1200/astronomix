"""
Hook 2 of the physics modules: the operator-split updates, once per time step.

``per_step_updates`` runs between two solver steps on the primitive state: the
spherical 1D stellar-wind injection, the user-supplied
``config.per_step_update``, the relaxation zone and the per-step positivity
floor. Two further
operator-split updates sit inside the step of every scheme, because they need
the state at its start or the driving pattern at its end: the advection of the
passive scalars (``_fluid_equations/_passive_scalars.py``) and the kick of the
turbulence driving (``apply_turbulence_driving``). Its counterpart inside the
stages is ``_explicit_sources.py``.
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
from astronomix._modules._explicit_sources import wind_source_active
from astronomix._modules._relaxation_zone._relaxation_zone import apply_relaxation_zone
from astronomix._modules._stellar_wind.stellar_wind import _wind_injection


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


def per_step_updates(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """
    Apply the operator-split updates that run between two solver steps.

    Args:
        primitive_state: The primitive state array.
        dt: The time step that follows.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.
        current_time: The time at the end of the step that follows.

    Returns:
        The primitive state with the updates applied.
    """

    # The spherical 1D stellar wind: injected once per step (the 3D wind is a
    # source of the stages).
    if config.wind_config.stellar_wind and not wind_source_active(config):
        primitive_state = _wind_injection(
            primitive_state,
            dt,
            config,
            params,
            helper_data,
            registered_variables,
        )

    # User-supplied per-step update (config.per_step_update).
    if config.per_step_update is not None:
        primitive_state = config.per_step_update(
            primitive_state,
            dt,
            current_time,
            config,
            params,
            helper_data,
            registered_variables,
        )

    # Relaxation zone: after the user update, so a reservoir or absorbing layer
    # has the last word on the state that the next solver step starts from.
    if config.relaxation_zone_config.relaxation_zone:
        primitive_state = apply_relaxation_zone(primitive_state, dt, config, params)

    # Per-step state floor (HARD_FLOOR): AthenaPK's density floor, momentum and
    # total energy kept (see _apply_density_floor), then the pressure floor.
    # The finite-difference scheme stays positive through the
    # positivity-preserving WENO; there the density floor bounds the Alfven
    # speed of near-vacuum, strongly magnetised cells, which would otherwise
    # set a vanishing time step. It also serves the finite-volume solver. The
    # time loop applies the density floor once more right after the state
    # update, so the time step and the snapshots see the floored state; here
    # it then only acts on density that an update above has changed.
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
            primitive_state = primitive_state.at[registered_variables.pressure_index].set(
                jnp.maximum(
                    primitive_state[registered_variables.pressure_index],
                    params.minimum_pressure,
                )
            )

    return primitive_state
