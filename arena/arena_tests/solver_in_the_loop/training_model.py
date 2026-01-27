from autocvd import autocvd
import os

# autocvd(num_gpus=1)
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig

from timeit import default_timer as timer
import datetime
import jax

jax.config.update("jax_debug_nans", True)
from jax.experimental import checkify as jax_checkify
import equinox as eqx
import jax.numpy as jnp
import optax
import math
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from jf1uids.time_stepping import time_integration
from jf1uids.variable_registry.registered_variables import RegisteredVariables

from arena.arena_tests.solver_in_the_loop.utils import (
    initialize_training_data,
    perturb_state,
)

from arena.arena_tests.solver_in_the_loop.loss import (
    normalized_weighted_loss,
    simple_mse_loss,
    EarlyStopper,
    loss_setup,
)

from arena.arena_tests.solver_in_the_loop.plot_training import plot_training
from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    TrainingConfig,
    SimulationConfigTraining,
    ModelMetadata,
    model_loader,
)
from arena.arena_tests.solver_in_the_loop.timepoint_updater import (
    FRONT_TO_BACK,
    BACK_TO_FRONT,
    timepoint_context,
)
import numpy as np

from jf1uids.option_classes.simulation_config import (
    VAN_ALBADA,
)

try:
    from arena.arena_tests.solver_in_the_loop.eval_model import eval_model
except:
    eval_model = None

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def validate_output(last_t: float, gradients_mod: float) -> None:
    """Validate that the model didnt run into nans or exploding grads"""
    if last_t == 0.0:
        raise ValueError("Nan in forward pass")
    if math.isnan(gradients_mod):
        raise ValueError("Nan in grads")


def create_train_step(
    loss_fn_factory,
    loss_fn_kwargs,
    optimizer,
    training_config: TrainingConfig,
    simulation_config: SimulationConfig,
    initial_state: STATE_TYPE,
    target_state: STATE_TYPE | SnapshotData,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
):
    def train_step_core(
        cnn_mhd_corrector_params,
        network_params_arrays,
        opt_state,
        params,
        key,
    ):
        noisy_initial_state = perturb_state(
            key, initial_state, training_config.noise_level
        )

        def loss_fn(network_params_arrays):
            results_low_res = time_integration(
                noisy_initial_state,
                simulation_config,
                params._replace(
                    cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                        network_params=network_params_arrays
                    )
                ),
                helper_data,
                registered_variables,
            )
            assert isinstance(results_low_res, SnapshotData), (
                "results is not a snapshot data"
            )
            loss = loss_fn_factory(
                results_low_res.states, target_state, **loss_fn_kwargs
            )
            jax.debug.print("loss {x}", x=loss)
            if training_config.use_checkify:
                jax_checkify.check(jnp.isfinite(loss), "Loss became NaN or Inf!")
            return loss, results_low_res.time_points[-1]

        (loss_value, last_timepoint), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(network_params_arrays)
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        )
        if training_config.use_checkify:
            jax_checkify.check(
                jnp.isfinite(gradients_modulus), "Gradients became NaN or Inf!"
            )
        updates, opt_state = optimizer.update(grads, opt_state, network_params_arrays)
        network_params_arrays = eqx.apply_updates(network_params_arrays, updates)
        return (
            network_params_arrays,
            opt_state,
            loss_value,
            gradients_modulus,
            last_timepoint,
        )

    if training_config.use_checkify:
        errors = (
            jax_checkify.user_checks
            | jax_checkify.float_checks
            | jax_checkify.nan_checks
            | jax_checkify.div_checks
        )
        checked_step = jax_checkify.checkify(train_step_core, errors=errors)

        @jax.jit
        def train_step(*args):
            err, out = checked_step(*args)
            return err, out
    else:
        train_step = jax.jit(train_step_core)

    return train_step


def training_model(
    model_manager: ModelManager,
    model_name: str,
    training_config: TrainingConfig,
    sim_config_training: SimulationConfigTraining,
    load_model: bool = False,
    load_model_nan: bool = False,
):
    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))
    warmup_steps = int(total_epochs * training_config.warmup_steps_fraction)

    lr_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=training_config.learning_rate,
        peak_value=training_config.peak_lr,
        end_value=training_config.end_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_epochs - warmup_steps,
    )

    (
        states_high_res_downsampled,
        (
            initial_state_low_res,
            config_low_res,
            params,
            helper_data_low_res,
            registered_variables,
        ),
    ) = initialize_training_data(
        snapshot_timepoints_train=jnp.array(training_config.snapshot_timepoints_train),
        t_end=sim_config_training.t_end,
        direction=training_config.direction,
        num_cells_high_res=sim_config_training.num_cells_high_res,
        downaverage_factor=sim_config_training.downaverage_factor,
        start_correction_time=sim_config_training.start_correction_time,
        correct_from_beggining=sim_config_training.correct_from_beggining,
        c_cfl=sim_config_training.c_cfl_target,
        limiter=sim_config_training.limiter,
    )

    # using different c_cfl for the final target and the simulation due to a mistake in the optuna optimization
    params = params._replace(C_cfl=sim_config_training.c_cfl)

    # Setup loss
    loss_fn_kwargs, loss_fn_factory = loss_setup(
        training_config, target_states=states_high_res_downsampled
    )

    # Initialize model
    model = CorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    neural_net_params = model_loader(
        model_manager,
        neural_net_params,
        load_model=load_model,
        load_model_nan=load_model_nan,
    )

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=sim_config_training.correct_from_beggining,
        # we made sure the initial state was at t_correct_begging
        start_correction_time=True,
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)
    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.gradient_clip),
        optax.adamw(learning_rate=lr_scheduler),
    )
    opt_state = optimizer.init(neural_net_params)

    losses = []
    trained_params = neural_net_params
    best_loss = float("inf")
    best_params = trained_params
    logger.info("Starting training...")
    total_step = 0
    succesful_training = True
    early_stopper = EarlyStopper(max_patience=training_config.patience)
    try:
        for i, epochs in enumerate(training_config.epochs_per_time):
            (
                current_end_time,
                current_config,
                current_params_sim,
                current_target,
                current_initial_state,
                key,
            ) = timepoint_context(
                i,
                sim_config_training=sim_config_training,
                training_config=training_config,
                config=config_low_res,
                params=params_low_res,
                target_states=states_high_res_downsampled,
                initial_state=initial_state_low_res,
                direction=training_config.direction,
            )
            early_stopper.reset_patience()

            train_step = create_train_step(
                loss_fn_factory=loss_fn_factory,
                loss_fn_kwargs=loss_fn_kwargs,
                optimizer=optimizer,
                training_config=training_config,
                simulation_config=current_config,
                initial_state=current_initial_state,
                target_state=current_target,
                helper_data=helper_data_low_res,
                registered_variables=registered_variables,
            )

            for step in range(epochs):
                key, subkey = jax.random.split(key)
                start_time_epoch = timer()
                if training_config.use_checkify:
                    (
                        err,
                        (
                            trained_params,
                            opt_state,
                            loss,
                            gradients_mod,
                            last_timepoint,
                        ),
                    ) = train_step(
                        cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                        network_params_arrays=trained_params,
                        opt_state=opt_state,
                        params=current_params_sim,
                        key=subkey,
                    )
                    err.throw()
                else:
                    trained_params, opt_state, loss, gradients_mod, last_timepoint = (
                        train_step(
                            cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                            network_params_arrays=trained_params,
                            opt_state=opt_state,
                            params=current_params_sim,
                            key=subkey,
                        )
                    )

                # print(f"last timepoint : {last_timepoint}")
                validate_output(last_t=last_timepoint, gradients_mod=gradients_mod)

                losses.append(float(loss))
                if early_stopper.new_epoch(loss):
                    logger.info("Finished training due to early stopper")
                    break

                # TODO: this logic doesnt really make sense with several times (at least in the back to front approach)
                if loss < best_loss:
                    best_loss, best_params = float(loss), trained_params

                if (step + 1) % 50 == 0:
                    model_manager.save_checkpoint(
                        trained_params, epoch=step + 1, loss=float(loss)
                    )

                logger.info(
                    f"t={current_end_time:.3f} | Step {step + 1}/{epochs} | Loss: {loss:.6f} | "
                    f"Time: {(timer() - start_time_epoch):.3f}s | Grads: {gradients_mod:.3f} | Last t {last_timepoint:.3f}"
                )
            succesful_training = True
            total_step += 1
    except ValueError as e:
        print(e)
        model_manager.save_model_params(trained_params, "model_params_NAN.pkl")
        succesful_training = False
    training_time = timer() - start_time

    if succesful_training:
        logger.info(
            f"\nTraining Complete! Time: {training_time:.2f}s | Best Loss: {best_loss:.6f}"
        )
        model_manager.save_model_params(best_params)

    model_manager.save_losses(losses)

    if training_config.early_stopper:
        early_stopped = early_stopper.early_stopped
    else:
        early_stopped = None

    metadata = ModelMetadata(
        model_name=model_name,
        created_at=datetime.datetime.now().isoformat(),
        total_epochs=total_epochs,
        final_loss=float(losses[-1]),
        best_loss=best_loss,
        training_time_seconds=training_time,
        succesful_training=succesful_training,
        early_stopped=early_stopped,
        final_epoch=total_step,
    )

    if eval_model and succesful_training:
        performance = eval_model(
            neural_net_static,
            best_params,
            jnp.linspace(0.0, 0.2, 35, endpoint=True),
            sim_config_training.num_cells_high_res,
            sim_config_training.downaverage_factor,
            sim_config_training.start_correction_time,
        )
        metadata.performance_metric = float(performance)
        logger.info(f"Performance: {performance:.6e}")

    model_manager.save_metadata(metadata)
    return best_params, neural_net_static


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def train_new_model(
    model_name=None, load_existing=False, load_existing_nan=False, **overrides
):
    manager = ModelManager(model_name=model_name)

    if load_existing and model_name:
        logger.info(f"Loading configs for model {model_name}")
        training_config = manager.load_training_config()
        sim_config_training = manager.load_simulation_config()
    else:
        model_name = manager.create_model_directory()
        logger.info(f"Created: {model_name}")

        training_config = TrainingConfig(
            epochs_per_time=[], snapshot_timepoints_train=[]
        )
        sim_config_training = SimulationConfigTraining()

        # Apply overrides
        for key, value in overrides.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
            elif hasattr(sim_config_training, key):
                setattr(sim_config_training, key, value)

        manager.save_training_config(training_config)
        manager.save_simulation_config(sim_config_training)

    logger.info("\n" + "=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(
        f"Epochs: {training_config.epochs_per_time} at times {training_config.snapshot_timepoints_train} | Direction {training_config.direction}"
    )
    logger.info(
        f"Architecture: {training_config.hidden_channels}x{training_config.hidden_layers}"
    )
    logger.info(
        f"Resolution: {sim_config_training.num_cells_high_res}, downsample: {sim_config_training.downaverage_factor}x"
    )
    logger.info(
        f"Loss: {training_config.loss_type}, LR: {training_config.learning_rate:.2e} → {training_config.peak_lr:.2e}"
    )
    logger.info(
        f"C_CFL: {sim_config_training.c_cfl}, C_CFL_TARGET: {sim_config_training.c_cfl_target}, limiter: {sim_config_training.limiter}"
    )
    logger.info("=" * 70 + "\n")

    trained_params, neural_net_static = training_model(
        manager,
        model_name,
        training_config,
        sim_config_training,
        load_model=load_existing,
        load_model_nan=load_existing_nan,
    )

    manager.print_model_info()
    return trained_params, neural_net_static


# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(format="->{message}", style="{", level=logging.INFO)
    # Train a new model
    model_name = "optuna_params"
    epochs_per_time = [100, 100, 100]
    c_cfl = 1.5
    limiter = VAN_ALBADA
    start_correction_time = 0.03
    num_cells_high_res = 64
    snapshot_timepoints_train = [0.15, 0.10, 0.05]
    t_end = 0.2
    direction = 1

    # TODO: fix the behavior of not putting a t_end, right now the default on the sim_config_training is 0.2
    model_params = {
        "limiter": VAN_ALBADA,
        "hidden_channels": 5,
        "scale": 0.04,
        "correction_time": 0.0,
        "noise": 0.07,
        "hidden_layers": 3,
        "base_lr": 1.5e-05,
        "warmup_fraction": 0.4,
        "peak_lr": 0.8e-04,
        "end_lr": 3.0e-05,
        "gradient_clip": 0.95,
        "c_cfl": 1.0,
        "c_cfl_target": 1.0,
        "loss_to_use": "norm_mse",
        "direction": FRONT_TO_BACK,
        "epochs_per_time": [5, 5, 5],
        "snapshot_timepoints_train": [0.10, 0.05, 0.0],
        "correct_from_beggining": True,
        "t_end": 0.2,
    }

    optuna_params = {
        "limiter": VAN_ALBADA,
        "hidden_channels": 5,
        "scale": 0.041574555074364715,
        "correction_time": 0.0001352420222864028,
        "noise": 0.06971301853761513,
        "hidden_layers": 3,
        "base_lr": 1.7231054411611903e-05,
        "warmup_fraction": 0.3983952086077301,
        "peak_lr": 2.343594314368986e-05,
        "end_lr": 3.1872436911245206e-06,
        "gradient_clip": 0.9555056246814991,
        "c_cfl": 0.6558539756738294,
        "c_cfl_target": 0.8,
        "loss_to_use": "norm_mse",
        "direction": BACK_TO_FRONT,
        "epochs_per_time": [300],
        "snapshot_timepoints_train": [0.2],
        "correct_from_beggining": True,
        "t_end": 0.2,
        # we change the correct_from_beggining to true as the small time was causing problems
    }

    training_params = model_params

    params, static = train_new_model(
        model_name=model_name,
        t_end=training_params["t_end"],
        direction=training_params["direction"],
        load_existing=False,
        load_existing_nan=False,
        correct_from_beggining=True,
        use_checkify=False,
        limiter=training_params["limiter"],
        epochs_per_time=training_params["epochs_per_time"],
        snapshot_timepoints_train=training_params["snapshot_timepoints_train"],
        hidden_channels=training_params["hidden_channels"],
        hidden_layers=training_params["hidden_layers"],
        c_cfl=training_params["c_cfl"],
        c_cfl_target=training_params["c_cfl_target"],
        gradient_clip=training_params["gradient_clip"],
        start_correction_time=training_params["correction_time"],
        learning_rate=training_params["base_lr"],
        peak_lr=training_params["peak_lr"],
        end_lr=training_params["end_lr"],
        warmup_steps_fraction=training_params["warmup_fraction"],
        model_initialization_scale=training_params["scale"],
        noise_level=training_params["noise"],
        loss_type=training_params["loss_to_use"],
        patience=20,
    )

    model_manager = ModelManager(model_name=model_name)
    model_metadata = model_manager.load_metadata()

    if model_metadata.succesful_training:
        plot_training(
            neural_net_params=params,
            neural_net_static=static,
            times_eval=jnp.linspace(0.0, 0.3, 30),
            num_cells_high_res=num_cells_high_res,
            downaverage_factor=2,
            snapshot_timepoints_train=snapshot_timepoints_train,
            start_correction_time=start_correction_time,
            epochs_per_time=epochs_per_time,
            model_name=model_name,
            cfl=c_cfl,
            limiter=limiter,
        )
    # Continue training an existing model
    # train_new_model(model_name="my_experiment", load_existing=True, epochs_per_time=[100])

    # List all models
    # manager = ModelManager()
    # print(manager.list_models())
