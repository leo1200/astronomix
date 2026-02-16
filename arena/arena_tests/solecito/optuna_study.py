import os

# from autocvd import autocvd
# autocvd(num_gpus=1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.46"

import logging
import math
from pathlib import Path
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import equinox as eqx

import optuna
from optuna.samplers import TPESampler

from typing import Union

# Reuse everything from your training script
from arena.arena_tests.solecito.training_model import (
    ScalarFieldCNNConfig,
    VectorFieldCNNConfig,
    initialize_training_data,
    initialize_target_data,
    initialize_model,
    initialize_optimizer,
    initialize_loss_fn,
    create_train_step,
    plot_training,
    EarlyStopper,
)

import json
import csv

logger = logging.getLogger(__name__)


def run_single_experiment(
    training_config: Union[ScalarFieldCNNConfig, VectorFieldCNNConfig],
) -> dict:
    """
    Run a single training experiment and return a results dict.
    Returns a dict with keys: model_name, channel_index, hidden_layers, c_cfl,
                               best_loss, final_loss, epochs_run, status, error
    """
    result = {
        "model_name": training_config.model_name,
        "channel_index": training_config.channel_index,
        "hidden_layers": training_config.hidden_layers,
        "c_cfl": training_config.c_cfl,
        "best_loss": float("inf"),
        "final_loss": float("inf"),
        "epochs_run": 0,
        "status": "pending",
        "error": "",
    }

    try:
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting experiment: {training_config.model_name}")
        logger.info(
            f"  channel_index={training_config.channel_index}, "
            f"hidden_layers={training_config.hidden_layers}, "
            f"c_cfl={training_config.c_cfl}, "
            f"warmup_steps={training_config.warmup_steps_fraction}, "
            f"scale={training_config.model_initialization_scale}, "
            f"noise_level={training_config.noise_level}"
        )
        logger.info(f"{'=' * 60}")

        model_folder = (
            Path("arena/arena_tests/solecito/models/optuna")
            / training_config.model_name
        )
        model_folder.mkdir(parents=True, exist_ok=True)

        # Initialize data
        (
            (
                initial_state_high_res,
                simulation_config_high_res,
                simulation_params,
                helper_data_high_res,
                registered_variables,
            ),
            (
                initial_state_low_res,
                simulation_config_low_res,
                simulation_params,
                helper_data_low_res,
                registered_variables,
            ),
        ) = initialize_training_data(training_config=training_config)

        # Initialize model
        cnn_mhd_corrector_params, cnn_mhd_corrector_config = initialize_model(
            registered_variables=registered_variables, training_config=training_config
        )

        # Initialize target data
        target_state = initialize_target_data(
            initial_state=initial_state_high_res,
            simulation_config=simulation_config_high_res,
            simulation_params=simulation_params,
            helper_data=helper_data_high_res,
            registered_variables=registered_variables,
            training_config=training_config,
        )

        # Initialize optimizer
        optimizer, opt_state = initialize_optimizer(
            training_config=training_config,
            neural_net_params=cnn_mhd_corrector_params.network_params,
        )

        simulation_config_low_res_sol = simulation_config_low_res._replace(
            cnn_mhd_corrector_config=cnn_mhd_corrector_config
        )
        simulation_params_low_res_sol = simulation_params._replace(
            cnn_mhd_corrector_params=cnn_mhd_corrector_params
        )

        # Initialize loss and train step
        loss_fn = initialize_loss_fn(target_state=target_state)
        train_step = create_train_step(
            loss_fn=loss_fn,
            optimizer=optimizer,
            simulation_config=simulation_config_low_res_sol,
            initial_state=initial_state_low_res,
            target_state=target_state,
            helper_data=helper_data_low_res,
            registered_variables=registered_variables,
            simulation_params=simulation_params_low_res_sol,
            noise_level=training_config.noise_level,
        )

        # Training loop
        losses = []
        best_loss = float("inf")
        trained_params = cnn_mhd_corrector_params.network_params
        best_params = cnn_mhd_corrector_params.network_params
        key = jax.random.PRNGKey(100)
        early_stopper = EarlyStopper(
            max_patience=training_config.patience,
            use_early_stopper=training_config.use_early_stopper,
        )

        experiment_start = timer()
        for step in range(training_config.epochs):
            start_time_epoch = timer()
            key, subkey = jax.random.split(key)
            trained_params, opt_state, loss, gradients_mod = train_step(
                network_params_arrays=trained_params, opt_state=opt_state, key=subkey
            )

            if math.isnan(loss):
                result["status"] = "nan_loss"
                result["epochs_run"] = step + 1
                result["error"] = f"NaN loss at step {step + 1}"
                logger.warning(
                    f"NaN loss at step {step + 1} for {training_config.model_name}"
                )
                break

            if early_stopper.new_epoch(loss):
                logger.info(f"Early stopped at step {step + 1}")
                result["epochs_run"] = step + 1
                break

            losses.append(float(loss))

            if loss < best_loss:
                best_loss = float(loss)
                best_params = trained_params

            logger.info(
                f"  [{training_config.model_name}] Step {step + 1}/{training_config.epochs} | "
                f"Loss: {loss:.6f} | Time: {(timer() - start_time_epoch):.3f}s | "
                f"Grads: {gradients_mod:.3f}"
            )
        else:
            # Loop completed without break
            result["epochs_run"] = training_config.epochs

        total_time = timer() - experiment_start
        result["best_loss"] = best_loss
        result["final_loss"] = float(losses[-1]) if losses else float("inf")
        result["total_time_s"] = round(total_time, 2)

        if result["status"] == "pending":
            result["status"] = "success"

        # Save the best model params
        eqx.tree_serialise_leaves(str(model_folder / "best_params.eqx"), best_params)

        # Save loss history
        jnp.save(str(model_folder / "losses.npy"), jnp.array(losses))

        # Generate the summary plot for successful runs
        if result["status"] == "success":
            try:
                plot_training(
                    neural_net_params=best_params,
                    times_eval=jnp.linspace(0.0, 0.3, 30, endpoint=True),
                    training_config=training_config,
                    losses=losses,
                    image_folder=model_folder,
                )
                logger.info(f"Plot saved to {model_folder / 'summary.png'}")
            except Exception as e:
                logger.warning(
                    f"Failed to generate plot for {training_config.model_name}: {e}"
                )
                result["error"] = f"Plot failed: {e}"

        logger.info(
            f"Experiment {training_config.model_name} finished: "
            f"best_loss={best_loss:.6f}, time={total_time:.1f}s"
        )

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        logger.error(f"Experiment {training_config.model_name} failed: {e}")

    # TODO: change optuna goal to average improvement
    return result


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    model_types = [
        "scalar_p",
        "scalar_d",
        "vector_v",
        "vector_m",
    ]  # keep your previous selection

    # Base output folder
    optuna_base = "arena/arena_tests/solecito/models/optuna"
    Path(optuna_base).mkdir(parents=True, exist_ok=True)

    # ---- Optuna study config ----
    study_db = Path(optuna_base) / "optuna_study.db"
    study_name = "solecito_optuna"

    sampler = TPESampler(
        multivariate=True,
        constant_liar=True,
    )

    study = optuna.create_study(
        study_name=study_name,
        sampler=sampler,
        direction="minimize",
        storage=f"sqlite:///{study_db}",
        load_if_exists=True,
    )

    results_file = Path(optuna_base) / "optuna_results.json"

    def objective(trial: optuna.Trial):
        model_type = trial.suggest_categorical("model_type", model_types)
        hidden_layers = trial.suggest_int("hidden_layers", 1, 5)
        hidden_channels = trial.suggest_int("hidden_channels", 8, 32, step=8)

        # learning rate schedule
        starting_lr = trial.suggest_float(
            "starting_learning_rate", 1e-5, 5e-3, log=True
        )
        peak_lr = trial.suggest_float("peak_lr", starting_lr, 2e-2, log=True)
        end_lr = trial.suggest_float("end_lr", 1e-6, 5e-4, log=True)

        warmup_steps = trial.suggest_float("warmup_steps", 0.01, 0.49)

        model_initialization_scale = trial.suggest_float(
            "model_initialization_scale", 1e-3, 5e-1, log=True
        )

        noise_level = trial.suggest_float("noise_level", 0.0, 0.10)

        c_cfl = trial.suggest_float("c_cfl", 0.5, 1.5)

        model_name = (
            f"optuna_t{trial.number}"
            f"_{model_type}"
            f"_hl{hidden_layers}_hc{hidden_channels}"
            f"_cfl_{int(c_cfl * 100)}"
        )

        if model_type == "scalar_p":
            config = ScalarFieldCNNConfig(
                model_name=model_name,
                channel_index=4,
                hidden_layers=hidden_layers,
                hidden_channels=hidden_channels,
                c_cfl=c_cfl,
                learning_rate=starting_lr,
                peak_lr=peak_lr,
                end_lr=end_lr,
                warmup_steps_fraction=warmup_steps,
                model_initialization_scale=model_initialization_scale,
                noise_level=noise_level,
                epochs=300,
            )
        elif model_type == "scalar_d":
            config = ScalarFieldCNNConfig(
                model_name=model_name,
                channel_index=4,
                hidden_layers=hidden_layers,
                hidden_channels=hidden_channels,
                c_cfl=c_cfl,
                learning_rate=starting_lr,
                peak_lr=peak_lr,
                end_lr=end_lr,
                warmup_steps_fraction=warmup_steps,
                model_initialization_scale=model_initialization_scale,
                noise_level=noise_level,
                epochs=300,
            )
        elif model_type == "vector_m":
            config = VectorFieldCNNConfig(
                model_name=model_name,
                vector_field="magnetic",
                hidden_layers=hidden_layers,
                hidden_channels=hidden_channels,
                c_cfl=c_cfl,
                learning_rate=starting_lr,
                peak_lr=peak_lr,
                end_lr=end_lr,
                warmup_steps_fraction=warmup_steps,
                model_initialization_scale=model_initialization_scale,
                noise_level=noise_level,
                epochs=300,
            )
        elif model_type == "vector_v":
            config = VectorFieldCNNConfig(
                model_name=model_name,
                vector_field="velocity",
                hidden_layers=hidden_layers,
                hidden_channels=hidden_channels,
                c_cfl=c_cfl,
                learning_rate=starting_lr,
                peak_lr=peak_lr,
                end_lr=end_lr,
                warmup_steps_fraction=warmup_steps,
                model_initialization_scale=model_initialization_scale,
                noise_level=noise_level,
                epochs=300,
            )
        else:
            raise ValueError

        result = run_single_experiment(config)

        # persist trial info
        trial.set_user_attr("model_name", result["model_name"])
        trial.set_user_attr("status", result["status"])
        trial.set_user_attr("best_loss", result["best_loss"])
        trial.set_user_attr("final_loss", result["final_loss"])
        trial.set_user_attr("epochs_run", result["epochs_run"])
        trial.set_user_attr("error", result["error"])
        trial.set_user_attr("total_time_s", result.get("total_time_s", None))

        # Update JSON results incrementally
        all_results = []
        if results_file.exists():
            try:
                with open(results_file, "r") as f:
                    all_results = json.load(f)
            except Exception:
                all_results = []

        all_results.append(result)
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        if result["status"] != "success":
            return float("inf")

        return result["best_loss"]

    # run an indefinite study; stop with Ctrl+C or set n_trials
    study.optimize(objective, n_trials=None)

    # ---- Final summary ----
    csv_file = Path(optuna_base) / "optuna_results.csv"
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = []

    if all_results:
        fieldnames = all_results[0].keys()
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

    logger.info("\nResults saved to:")
    logger.info(f"  JSON: {results_file}")
    logger.info(f"  CSV:  {csv_file}")
    logger.info(f"  Optuna DB: {study_db}")


if __name__ == "__main__":
    main()
