from typing import Union
import jax
from functools import partial
import jax.numpy as jnp

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector import (
    _cnn_mhd_corrector,
)
from astronomix._physics_modules._cooling._cooling import first_order_pressure_update, update_pressure_by_cooling
from astronomix._physics_modules._cosmic_rays.cr_injection import (
    inject_crs_at_strongest_shock,
)
from astronomix._physics_modules._neural_net_force._neural_net_force import (
    _neural_net_force,
)
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix._geometry.boundaries import _boundary_handler
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import (
    SPHERICAL,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._physics_modules._stellar_wind.stellar_wind import _wind_injection
from astronomix.shock_finder.shock_finder import shock_criteria


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _run_physics_modules(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    current_time: Union[float, Float[Array, ""]],
) -> STATE_TYPE:
    """Run all the physics modules. The physics modules are switched on/off and
    configured in the simulation configuration. Parameters for the physics modules
    (with respect to which the simulation can be differentiated) are stored in the
    simulation parameters.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        The primitive state array with the physics modules applied.
    """

    # stellar wind
    if config.wind_config.stellar_wind:
        primitive_state = _wind_injection(
            primitive_state, dt, config, params, helper_data, registered_variables
        )

        # we might want to run the boundary handler after all physics modules have completed
        # primitive_state = _boundary_handler(primitive_state, config.left_boundary, config.right_boundary)

    if config.cosmic_ray_config.diffusive_shock_acceleration:
        shock_crit = shock_criteria(
            primitive_state, config, registered_variables, helper_data
        )

        # injecting cosmic rays only after a certain amount of time
        # is an ad-hoc fix to problems that come about when a shock
        # has not yet properly formed
        primitive_state = jax.lax.cond(
            jnp.logical_and(
                current_time
                >= params.cosmic_ray_params.diffusive_shock_acceleration_start_time,
                jnp.any(shock_crit),
            ),
            lambda primitive_state: inject_crs_at_strongest_shock(
                primitive_state,
                params.gamma,
                helper_data,
                params.cosmic_ray_params,
                config,
                registered_variables,
                dt,
            ),
            lambda primitive_state: primitive_state,
            primitive_state,
        )

    if config.cooling_config.cooling:
        primitive_state = update_pressure_by_cooling(
            primitive_state,
            registered_variables,
            config.cooling_config,
            params,
            dt,
        )
        # primitive_state = first_order_pressure_update(
        #     primitive_state,
        #     registered_variables,
        #     config,
        #     helper_data,
        #     params,
        #     dt
        # )

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

    if config.cnn_mhd_corrector_config.cnn_mhd_corrector:
        primitive_state = _cnn_mhd_corrector(
            primitive_state, config, registered_variables, params, dt
        )

    return primitive_state
