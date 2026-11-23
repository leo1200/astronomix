"""
One finite-volume time step: the choice of the time integrator
(``_integrators.py``) and the operator-split updates inside the step: the
advection of the passive scalars and the kick of the turbulence driving after
the step.
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
from astronomix._finite_volume._integrators import (
    muscl_hancock_step,
    rk2_ssp_step,
    vl2_step,
)
from astronomix._fluid_equations._passive_scalars import (
    reattach_passive_scalars,
    split_off_passive_scalars,
)
from astronomix._modules._turbulence_driving import apply_turbulence_driving


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
    driving_state=None,
    sink_potential=None,
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
        driving_state: The state of the turbulence driving at the end of the
            step, or ``None`` without driving.
        sink_potential: The gravitational potential of the sink particles,
            or ``None`` without sink particles.

    Returns:
        The primitive state ``W^{n+1}``.
    """
    # The passive scalars are split off the end of the state and advected with the flow at the start of the
    # step; the solver sees the layout without them.
    passive_scalars_active = registered_variables.passive_scalars_active
    if passive_scalars_active:
        primitive_state, passive_scalars, registered_variables = split_off_passive_scalars(
            primitive_state, dt, config, registered_variables
        )

    step_arguments = (dt, gamma, config, params, helper_data, registered_variables)

    if config.solver_config.time_integrator == VL2:
        primitive_state = vl2_step(primitive_state, *step_arguments, sink_potential=sink_potential)
    elif config.solver_config.time_integrator == RK2_SSP:
        primitive_state = rk2_ssp_step(primitive_state, *step_arguments, sink_potential=sink_potential)
    else:
        primitive_state = muscl_hancock_step(
            primitive_state, *step_arguments, step_index, sink_potential=sink_potential
        )

    # The kick of the turbulence driving, operator-split after the update; the
    # ghost cells are filled at the start of the next step.
    if driving_state is not None:
        primitive_state = apply_turbulence_driving(
            primitive_state, driving_state, dt, config, params, registered_variables
        )

    if passive_scalars_active:
        primitive_state = reattach_passive_scalars(primitive_state, passive_scalars, config)
    return primitive_state
