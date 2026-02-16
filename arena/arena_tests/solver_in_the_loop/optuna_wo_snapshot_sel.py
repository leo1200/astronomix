"""Optuna process to find the best model
Uses one last snapshot timepoint
"""

# from autocvd import autocvd

# autocvd(num_gpus=1)
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


from astronomix.data_classes.simulation_helper_data import HelperData

from typing import Tuple
from functools import partial
import jax
import equinox as eqx
import jax.numpy as jnp
import optax
from jaxtyping import PyTree, Array
import math
from astronomix import (
    SimulationConfig,
    SimulationParams,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import RegisteredVariables
import numpy as np
from arena.arena_tests.solver_in_the_loop.fd_blast_sol import (
    get_initial_state_training,
    downaverage,
    perturb_state,
)
import optuna
from typing import Optional


def normalized_weighted_loss(
    pred_state: Array,
    target_state: Array,
    channel_normalizers: Optional[Array] = None,
    physics_weights: Optional[Array] = None,
    verbose: bool = False,
    use_interface: bool = False,
) -> Array:
    """
    Normalized and weighted loss with physics priorities.

    Args:
        pred_state: Predicted state array
        target_state: Target state array
        config: Simulation configuration
        registered_variables: Variable registry
        channel_normalizers: Per-channel standard deviations for normalization
        physics_weights: Per-channel importance weights
        verbose: If True, prints detailed per-channel loss breakdown
    """
    # Default physics weights (emphasize critical variables)
    if physics_weights is None:
        physics_weights = jnp.array(
            [
                1.0,  # 0: density (critical)
                1.0,  # 1: vx
                1.0,  # 2: vy
                1.0,  # 3: vz
                1.0,  # 4: pressure (critical)
                1.0,  # 5: Bx
                1.0,  # 6: By
                1.0,  # 7: Bz
                1.0,  # 8: Bx interface
                1.0,  # 9: By interface
                1.0,  # 10: Bz interface
            ]
        )

    # Normalize by channel statistics
    if channel_normalizers is not None:
        normalizers = channel_normalizers[:, None, None, None]
        normalized_error = (pred_state - target_state) / normalizers
    else:
        normalized_error = pred_state - target_state

    # Apply physics weights
    physics_weights_broadcast = physics_weights[:, None, None, None]
    weighted_error = physics_weights_broadcast * (normalized_error**2)

    if use_interface is False:
        total_loss = jnp.mean(weighted_error[:8])
    else:
        total_loss = jnp.mean(weighted_error)

    if verbose:
        # Per-channel loss contributions
        per_channel_loss = jnp.mean(weighted_error, axis=(0, 2, 3, 4))

        jax.debug.print("========== Per-Channel Loss Breakdown ==========")
        jax.debug.print(
            "density      | Weight: {w} | Loss: {l}",
            w=physics_weights[0],
            l=per_channel_loss[0],
        )
        jax.debug.print(
            "vx           | Weight: {w} | Loss: {l}",
            w=physics_weights[1],
            l=per_channel_loss[1],
        )
        jax.debug.print(
            "vy           | Weight: {w} | Loss: {l}",
            w=physics_weights[2],
            l=per_channel_loss[2],
        )
        jax.debug.print(
            "vz           | Weight: {w} | Loss: {l}",
            w=physics_weights[3],
            l=per_channel_loss[3],
        )
        jax.debug.print(
            "pressure     | Weight: {w} | Loss: {l}",
            w=physics_weights[4],
            l=per_channel_loss[4],
        )
        jax.debug.print(
            "Bx           | Weight: {w} | Loss: {l}",
            w=physics_weights[5],
            l=per_channel_loss[5],
        )
        jax.debug.print(
            "By           | Weight: {w} | Loss: {l}",
            w=physics_weights[6],
            l=per_channel_loss[6],
        )
        jax.debug.print(
            "Bz           | Weight: {w} | Loss: {l}",
            w=physics_weights[7],
            l=per_channel_loss[7],
        )
        jax.debug.print(
            "Bx_int       | Weight: {w} | Loss: {l}",
            w=physics_weights[8],
            l=per_channel_loss[8],
        )
        jax.debug.print(
            "By_int       | Weight: {w} | Loss: {l}",
            w=physics_weights[9],
            l=per_channel_loss[9],
        )
        jax.debug.print(
            "Bz_int       | Weight: {w} | Loss: {l}",
            w=physics_weights[10],
            l=per_channel_loss[10],
        )
        jax.debug.print("------------------------------------------------")
        jax.debug.print("Total Loss: {total}", total=total_loss)
        jax.debug.print("================================================")
    return total_loss


def simple_mse_loss(
    pred_state: Array,
    target_state: Array,
) -> Array:
    """Simple MSE loss between predicted and target states."""
    return jnp.mean((pred_state - target_state) ** 2)


def preparing_optuna_study(
    num_cells_high_res: int,
    downaverage_factor: int,
    end_time: float,
    max_epochs: int,
):
    simulation_bundle_high_res, sim_bundle_lr = get_initial_state_training(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        snapshot_timepoints_train=jnp.array([0.2]),
    )
    result_high_res = time_integration(*simulation_bundle_high_res)
    states_high_res_downsampled = downaverage(
        result_high_res.states, downaverage_factor=downaverage_factor
    )
    print(states_high_res_downsampled.shape)
    experiment_folder = os.path.abspath(
        "/export/home/jalegria/Thesis/astronomix/arena/data"
    )
    study = optuna.create_study(
        study_name="fd_blast_sol",
        storage=f"sqlite:///{os.path.join(experiment_folder, 'sol_optuna_64.db')}",
        load_if_exists=True,
        directions=["minimize"],
    )
    study.set_user_attr("end_time", end_time)
    study.set_user_attr("epochs", max_epochs)
    study.optimize(
        partial(
            objective,
            high_res_target=states_high_res_downsampled,
            sim_bundle_lr=sim_bundle_lr,
            end_time=end_time,
            epochs=max_epochs,
        ),
        show_progress_bar=True,
        n_trials=70,
        gc_after_trial=True,
    )


def fmt(x, *, float_fmt="{:.3g}", int_fmt="{:,}"):
    if isinstance(x, (int, float)):
        # choose float or int formatting
        if isinstance(x, int):
            return int_fmt.format(x)
        return float_fmt.format(x)
    return str(x)  # fallback for lists, arrays, tuples, etc.


def objective(
    trial: optuna.trial.Trial,
    high_res_target: jnp.ndarray,
    sim_bundle_lr: Tuple,
    end_time: float,
    epochs: int,
):
    hidden_channels = trial.suggest_int("hidden_channels", 3, 12)
    model_initialization_scale = trial.suggest_float("scale", 0.001, 0.2)
    start_correction_time = trial.suggest_float("correction_time", 0.0, 0.05)
    noise_level = trial.suggest_float("noise", 0.0, 0.2)
    hidden_layers = trial.suggest_int("hidden_layers", 1, 4)
    base_lr = trial.suggest_float("base_lr", 1e-5, 1e-3, log=True)
    warmup_fraction = trial.suggest_float("warmup_fraction", 0.1, 0.5)
    peak_lr = trial.suggest_float("peak_lr", base_lr, 1e-2, log=True)
    end_lr = trial.suggest_float("end_lr", 1e-6, base_lr, log=True)
    gradient_clip = trial.suggest_float("gradient_clip", 0.5, 1.5)
    c_cfl = trial.suggest_float("c_cfl", 0.5, 1.8)

    loss_to_use = trial.suggest_categorical(
        "loss_to_use", choices=("mse", "norm_mse", "norm_mse_wo_interface")
    )
    losses = {
        "mse": {"loss_fn_factory": simple_mse_loss, "loss_fn_kwargs": {}},
        "norm_mse": {
            "loss_fn_factory": normalized_weighted_loss,
            "loss_fn_kwargs": {
                "channel_normalizers": "auto",  # Will be computed from target data
                "physics_weights": None,  # Will use defaults
                "verbose": False,
                "use_interface": True,
            },
        },
        "norm_mse_wo_interface": {
            "loss_fn_factory": normalized_weighted_loss,
            "loss_fn_kwargs": {
                "channel_normalizers": "auto",
                "physics_weights": None,
                "verbose": False,
                "use_interface": False,
            },
        },
    }
    loss_fn_kwargs = losses[loss_to_use]["loss_fn_kwargs"]
    loss_fn_factory = losses[loss_to_use]["loss_fn_factory"]
    if loss_fn_kwargs is None:
        loss_fn_kwargs = {}
    if (
        "channel_normalizers" in loss_fn_kwargs
        and loss_fn_kwargs["channel_normalizers"] == "auto"
    ):
        if high_res_target.ndim == 5:
            channel_normalizers = jnp.std(high_res_target, axis=(0, 2, 3, 4))
        else:
            channel_normalizers = jnp.std(high_res_target, axis=(1, 2, 3))
        channel_normalizers = jnp.maximum(channel_normalizers, 1e-8)
        loss_fn_kwargs["channel_normalizers"] = channel_normalizers
    print(
        "\n=== Training Configuration ===\n"
        f"{'Hidden channels:':25} {fmt(hidden_channels)}\n"
        f"{'Hidden layers:':25} {fmt(hidden_layers)}\n"
        f"{'Init scale:':25} {fmt(model_initialization_scale)}\n"
        f"{'Noise level:':25} {fmt(noise_level)}\n"
        f"{'Start correction time:':25} {fmt(start_correction_time)}\n"
        f"{'Loss used:':25} {loss_to_use}\n"
        f"{'Base LR:':25} {fmt(base_lr)}\n"
        f"{'Peak LR:':25} {fmt(peak_lr)}\n"
        f"{'End LR:':25} {fmt(end_lr)}\n"
        f"{'Warmup fraction:':25} {warmup_fraction:.2%}\n"
        f"{'Cfl:':25} {fmt(c_cfl)}\n"
        "==============================\n"
    )
    (
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    ) = sim_bundle_lr

    initial_state_low_res = time_integration(
        primitive_state=initial_state_low_res,
        config=config_low_res._replace(
            return_snapshots=False, num_snapshots=1, progress_bar=False
        ),
        params=params._replace(t_end=start_correction_time),
        registered_variables=registered_variables,
        helper_data=helper_data_low_res,
    )
    sim_bundle_lr = (
        initial_state_low_res,
        config_low_res,
        params,
        helper_data_low_res,
        registered_variables,
    )
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
        correct_from_beggining=True,
    )

    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    # Set up the optimizer using optax
    warmup_steps = int(epochs * warmup_fraction)
    decay_steps = epochs - warmup_steps
    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=base_lr,
        peak_value=peak_lr,
        end_value=end_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.adamw(learning_rate=lr_scheduler),
    )

    opt_state = optimizer.init(neural_net_params)

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
            loss = loss_fn_factory(
                pred_state=results_low_res.states,
                target_state=target_state_arg,
                **loss_fn_kwargs,
            )
            return loss

        """Performs one step of gradient descent."""
        loss_value, grads = eqx.filter_value_and_grad(loss_fn)(network_params_arrays)
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        ).astype(float)
        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
        return network_params_arrays, opt_state, loss_value, gradients_modulus

    max_patience = 20
    patience = 0
    params_low_res = params_low_res._replace(C_cfl=c_cfl)

    # Update the config/params objects for this specific timeframe
    config_low_res = config_low_res._replace(num_snapshots=1)
    params_sim_lr = params_low_res._replace(
        t_end=end_time - start_correction_time,
        snapshot_timepoints=jnp.array([end_time - start_correction_time]),
    )

    target = high_res_target

    for current_epoch in range(epochs):
        if current_epoch % 10 == 0:
            print(current_epoch, end="\r")
        key, subkey = jax.random.split(key)

        trained_params, opt_state, loss, gradients_mod = train_step(
            trained_params,
            opt_state,
            target,
            initial_state_low_res,
            config_low_res,
            params_sim_lr,
            subkey,
            helper_data_low_res,
            registered_variables,
        )
        if math.isnan(gradients_mod):
            trial.set_user_attr("diverged", True)
            trial.set_user_attr("diverge_step", int(epochs))
            bad_loss = 10 * (epochs - current_epoch)
            return bad_loss
        if loss < best_loss:
            best_loss = loss
            patience = 0
        else:
            patience += 1
            if patience == max_patience:
                trial.set_user_attr("early_stopped_step", int(epochs))
                break

    eval_loss = eval_model(
        network_static=neural_net_static,
        network_params=neural_net_params,
        times_eval=jnp.linspace(0.0, end_time, num=30),
        num_cells_high_res=high_res_target.shape[-1],
        downaverage_factor=2,
        start_correction_time=start_correction_time,
    )
    return eval_loss


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
        helper_data_low_res,
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
        helper_data_low_res,
        registered_variables,
    ).states

    l2_errors = jnp.mean((states_low_res - states_target) ** 2)
    return l2_errors


def main():
    num_cells_high_res = 64
    downaverage_scale = 2
    total_epochs = 200
    preparing_optuna_study(
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_scale,
        end_time=0.2,
        max_epochs=total_epochs,
    )


if __name__ == "__main__":
    main()
