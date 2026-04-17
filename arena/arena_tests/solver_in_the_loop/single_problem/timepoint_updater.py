"""Timepoint updater logic"""

from typing import Tuple
import jax

import jax.numpy as jnp
from astronomix.time_stepping import time_integration

from arena.arena_tests.solver_in_the_loop.model_manager import (
    TrainingConfig,
)

from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix import get_registered_variables
from jaxtyping import Array

import logging

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
)

logger = logging.Logger(__name__)

FRONT_TO_BACK = 0
BACK_TO_FRONT = 1


def timepoint_context(
    i: int,
    training_config: TrainingConfig,
    config: SimulationConfig,
    params: SimulationParams,
    target_states: Array,
    initial_state: STATE_TYPE,
    direction: int = BACK_TO_FRONT,
) -> Tuple[float, SimulationConfig, SimulationParams, Array, Array, Array]:
    "Returns the end_time, configuration, params, target and key to use during training"
    times = training_config.snapshot_timepoints_train
    if direction == BACK_TO_FRONT:
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1)), (
            "The times array is not ordered back to front (ascending order)"
        )
        current_end_time = training_config.snapshot_timepoints_train[i]
        current_config = config._replace(num_snapshots=1)

        # NOTE: we are taking into account already the start_correction_time
        # in the initial state we use
        #
        # if not training_config.correct_from_beggining:
        #     current_params_sim = params._replace(
        #         t_end=current_end_time - training_config.start_correction_time,
        #         snapshot_timepoints=jnp.array(
        #             [current_end_time - training_config.start_correction_time]
        #         ),
        #     )
        # else:
        #     current_params_sim = params._replace(
        #         t_end=current_end_time,
        #         snapshot_timepoints=jnp.array([current_end_time]),
        #     )
        current_params_sim = params._replace(
            t_end=current_end_time,
            snapshot_timepoints=jnp.array([current_end_time]),
        )
        current_initial_state = initial_state
        current_target = target_states[i]

    elif direction == FRONT_TO_BACK:
        assert all(times[i] >= times[i + 1] for i in range(len(times) - 1)), (
            "The times array is not ordered front to back (descending order)"
        )
        absolute_start_time = training_config.snapshot_timepoints_train[i]
        integrate_initial_state = True
        if training_config.correct_from_beggining:
            if absolute_start_time == 0.0:
                integrate_initial_state = False

            current_start_time = absolute_start_time
            current_end_time = training_config.t_end - current_start_time
        else:
            if absolute_start_time == 0.0:
                print(
                    f"As the start correction time is ahead of 0.0, we start training from {training_config.start_correction_time}"
                )
                integrate_initial_state = False
                current_start_time = 0.0
                current_end_time = (
                    training_config.t_end - training_config.start_correction_time
                )
            else:
                current_start_time = (
                    absolute_start_time - training_config.start_correction_time
                )
                current_end_time = training_config.t_end - absolute_start_time
        print(
            f"current_start_time: {current_start_time} | current_end_time = {current_end_time} | end_time = {training_config.t_end}"
        )
        current_config = config._replace(num_snapshots=1)
        current_params_sim = params._replace(
            t_end=current_end_time, snapshot_timepoints=jnp.array([current_end_time])
        )

        registered_variables = get_registered_variables(config)

        logger.debug(f"integrate_initial_state {integrate_initial_state}")
        if integrate_initial_state:
            logger.debug(f"current_start_time {current_start_time}")
            current_initial_state = time_integration(
                primitive_state=initial_state,
                config=config._replace(
                    return_snapshots=False,
                    progress_bar=True,
                    exact_end_time=True,
                    cnn_mhd_corrector_config=CNNMHDconfig(cnn_mhd_corrector=False),
                ),
                params=params._replace(t_end=current_start_time),
                registered_variables=registered_variables,
            )
        else:
            current_initial_state = initial_state

        current_target = target_states[-1]

    else:
        raise ValueError("Direction of training wasn't especified")

    key = jax.random.PRNGKey(112 + i)
    assert isinstance(current_end_time, float)
    assert isinstance(current_initial_state, STATE_TYPE)
    return (
        current_end_time,
        current_config,
        current_params_sim,
        current_initial_state,
        current_target,
        key,
    )
