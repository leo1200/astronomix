from autocvd import autocvd

autocvd(num_gpus=1)


from jf1uids.data_classes.simulation_helper_data import HelperData

import os
from typing import Tuple
from functools import partial
import jax
import equinox as eqx
import jax.numpy as jnp
import optax
from optax import Schedule
from jaxtyping import PyTree, Array
import math
from jf1uids import (
    SimulationConfig,
    SimulationParams,
)
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from jf1uids.time_stepping import time_integration
from jf1uids.variable_registry.registered_variables import RegisteredVariables
import numpy as np
from arena.arena_tests.solver_in_the_loop.fd_blast_sol import (
    get_initial_state_training,
    downaverage,
    perturb_state,
)
import optuna


def preparing_optuna_study(
    num_cells_high_res: int,
    downaverage_factor: int,
    lr_scheduler: Schedule,
    snapshot_timepoints_train: jnp.ndarray,
    epochs_per_time: list[int],
):
    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=snapshot_timepoints_train,
    )
    result_high_res = time_integration(*simulation_bundle_high_res)
    states_high_res_downsampled = downaverage(
        result_high_res.states, downaverage_factor=downaverage_factor
    )

    experiment_folder = os.path.abspath(
        "/export/home/jalegria/Thesis/jf1uids/arena/data"
    )
    study = optuna.create_study(
        study_name="fd_blast_sol",
        storage=f"sqlite:///{os.path.join(experiment_folder, 'fd_blast_sol.db')}",
        load_if_exists=True,
        directions=["minimize"],
    )
    #     search_space = {
    #         "hidden_channels": optuna.distributions.IntDistribution(6, 38),
    #         "scale": optuna.distributions.FloatDistribution(0.001, 0.05),
    #         "correction_time": optuna.distributions.FloatDistribution(0.0, 0.1),
    #         "noise": optuna.distributions.FloatDistribution(0.0, 0.08),
    #         "hidden_layers": optuna.distributions.IntDistribution(1, 4),
    #     }
    #     study.add_trial(optuna.trial.create_trial(params = {
    #         "hidden_channels": 16,
    #         "hidden_layers": 1,
    #         "scale": 0.005,
    #         "correction_time": 0.05,
    #         "noise": 0.03
    #     }, distributions=search_space, value=(1e8- 20 * 1e6)),
    #  )
    study.optimize(
        partial(
            objective,
            high_res_target=states_high_res_downsampled,
            sim_bundle_lr=simulation_bundle_low_res,
            lr_scheduler=lr_scheduler,
            epochs_per_time=epochs_per_time,
            snapshot_timepoints_train=snapshot_timepoints_train,
        ),
        show_progress_bar=True,
        n_trials=70,
        gc_after_trial=True,
    )


def objective(
    trial: optuna.trial.Trial,
    high_res_target: jnp.ndarray,
    sim_bundle_lr: Tuple,
    lr_scheduler: Schedule,
    epochs_per_time: list[int],
    snapshot_timepoints_train: jnp.ndarray,
):
    hidden_channels = trial.suggest_int("hidden_channels", 6, 38)
    model_initialization_scale = trial.suggest_float("scale", 0.001, 0.05)
    start_correction_time = trial.suggest_float("correction_time", 0.0, 0.1)
    noise_level = trial.suggest_float("noise", 0.0, 0.08)
    hidden_layers = trial.suggest_int("hidden_layers", 1, 4)
    print(
        f"hidden_channels {hidden_channels}, start_correction_time {start_correction_time}"
        f"scale {model_initialization_scale}, noise {noise_level}"
        f"hidden layers {hidden_layers}"
    )
    (
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    ) = sim_bundle_lr
    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=hidden_channels,
        hidden_layers=hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=model_initialization_scale,
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=False,
        start_correction_time=start_correction_time,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    # Set up the optimizer using optax

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_scheduler),
    )

    opt_state = optimizer.init(neural_net_params)

    losses = []

    # This variable will hold the trained parameters and be updated in the loop
    trained_params = neural_net_params

    best_loss = float("inf")

    key = jax.random.PRNGKey(112)

    @eqx.filter_jit
    def train_step(
        network_params_arrays: PyTree,
        opt_state: optax.OptState,
        target_state_arg: Array,
        initial_state_arg: Array,
        config_arg: SimulationConfig,
        params_arg: SimulationParams,
        key_arg: Array,
        helper_data_arg: HelperData,
        registered_variables_arg: RegisteredVariables,
    ):
        noisy_initial_state = perturb_state(
            key_arg, initial_state_arg, noise_level=noise_level
        )

        def loss_fn(network_params_arrays):
            """Calculates the difference between the final state and the target."""
            results_low_res = time_integration(
                noisy_initial_state,
                config_arg,
                params_arg._replace(
                    cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                        network_params=network_params_arrays
                    )
                ),
                helper_data_arg,
                registered_variables_arg,
            )

            # Calculate the L2 loss between the final state and the target state
            loss = jnp.mean((results_low_res.states - target_state_arg) ** 2)
            return loss

        """Performs one step of gradient descent."""
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        ).astype(float)
        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
        return network_params_arrays, opt_state, loss_value, gradients_modulus

    global_step = 0
    for i, epochs in enumerate(epochs_per_time):
        # Update outer loop variables
        current_end_time = snapshot_timepoints_train[i]

        # Update the config/params objects for this specific timeframe
        current_config = config_low_res._replace(num_snapshots=1)
        current_params_sim = params_low_res._replace(
            t_end=current_end_time,
            snapshot_timepoints=jnp.array([current_end_time]),
        )

        current_target = high_res_target[i]

        # Reset key for this epoch block if needed, or keep evolving it
        key = jax.random.PRNGKey(16 + i)

        for _ in range(epochs):
            global_step += 1
            if global_step % 10 == 0:
                print(global_step, end="\r")
            key, subkey = jax.random.split(key)

            # 3. CALL THE FUNCTION
            # Notice we pass EVERYTHING that defines the physics here
            trained_params, opt_state, loss, gradients_mod = train_step(
                trained_params,
                opt_state,
                current_target,
                initial_state_low_res,
                current_config,
                current_params_sim,
                subkey,
                helper_data_low_res,
                registered_variables,
            )
            if math.isnan(gradients_mod):
                trial.set_user_attr("diverged", True)
                trial.set_user_attr("diverge_step", int(epochs))
                bad_loss = 1e8 - 1e6 * float(global_step)
                return bad_loss
            losses.append(loss)
            if loss < best_loss:
                best_loss = loss
    return best_loss


def main():
    num_cells_high_res = 128
    downaverage_scale = 2
    epochs_per_time = [100]
    snapshot_timepoints_train = jnp.array([0.2])
    assert len(epochs_per_time) == len(snapshot_timepoints_train), (
        "len of epochs per time and times to train at"
    )
    total_epochs = np.sum(np.array(epochs_per_time))
    total_epochs = 100
    learning_rate = 1e-4
    peak_lr = 1e-3
    end_lr = 5e-6
    warmup_steps_fraction = 0.4
    warmup_steps = int(total_epochs * warmup_steps_fraction)
    decay_steps = total_epochs - warmup_steps
    learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate,
        peak_value=peak_lr,
        end_value=end_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )
    preparing_optuna_study(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_scale,
        lr_scheduler=learning_rate_scheduler,
        snapshot_timepoints_train=snapshot_timepoints_train,
        epochs_per_time=epochs_per_time,
    )


if __name__ == "__main__":
    main()
