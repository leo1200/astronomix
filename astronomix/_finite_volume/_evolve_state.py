"""
One finite-volume time step: the choice of the time integrator
(``_integrators.py``) and, for the schemes that take gravity as a source around
the step, that source (``_gravity_coupling.py``).
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import (
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
from astronomix._finite_volume._gravity_coupling import (
    _apply_gravity_source,
    _gravity_source_presolve,
)
from astronomix._finite_volume._integrators import (
    muscl_hancock_step,
    rk2_ssp_step,
    vl2_step,
)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_state_fv(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    step_index=0,
) -> STATE_TYPE:
    """
    Advance the primitive state by one finite-volume time step.

    Args:
        primitive_state: The primitive state ``W^n``.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.
        step_index: The number of the step in the run (the split MHD scheme alternates its sweep order).

    Returns:
        The primitive state ``W^{n+1}``.
    """
    step_arguments = (dt, gamma, config, params, helper_data, registered_variables)

    # VL2 has its gravity coupling inside the stages.
    if config.solver_config.time_integrator == VL2:
        return vl2_step(primitive_state, *step_arguments)

    # The other two schemes take gravity (hydrodynamics only) first order in
    # time: the source is built from the state before the step and added after it.
    if config.gravity_config.gravity:
        gravity_source = _gravity_source_presolve(primitive_state, *step_arguments)

    if config.solver_config.time_integrator == RK2_SSP:
        primitive_state = rk2_ssp_step(primitive_state, *step_arguments)
    else:
        primitive_state = muscl_hancock_step(primitive_state, *step_arguments, step_index)

    if config.gravity_config.gravity:
        primitive_state = _apply_gravity_source(
            primitive_state, gravity_source, gamma, config, params, registered_variables
        )
    return primitive_state
