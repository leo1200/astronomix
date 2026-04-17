from jaxtyping import PyTree
from astronomix import time_integration

from arena.arena_tests.solver_in_the_loop.utils import (
    get_initial_state_training,
    downaverage,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
import jax.numpy as jnp


def eval_model(
    network_static: PyTree,
    network_params: PyTree,
    times_eval: jnp.ndarray,
    num_cells_high_res: int,
    downaverage_factor: int,
    start_correction_time: float,
):
    sim_bundle_hr, sim_bundle_lr = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=times_eval,
    )
    result_high_res = time_integration(*sim_bundle_hr)
    states_target = downaverage(result_high_res.states, downaverage_factor)

    (
        initial_state_low_res,
        config_low_res,
        params,
        registered_variables,
    ) = sim_bundle_lr

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=network_static,
        start_correction_time=start_correction_time,
        correct_from_beggining=False,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=network_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    states_low_res = time_integration(
        initial_state_low_res,
        config_low_res,
        params_low_res,
        registered_variables,
    ).states

    l2_errors = jnp.mean((states_low_res - states_target) ** 2)
    return l2_errors
