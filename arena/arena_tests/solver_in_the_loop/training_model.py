from autocvd import autocvd
import os

autocvd(num_gpus=1)
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig

from timeit import default_timer as timer
import datetime
import jax

# jax.config.update("jax_debug_nans", True)
from jax.experimental import checkify as jax_checkify
import equinox as eqx
import jax.numpy as jnp
import optax
import math
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import RegisteredVariables

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
from arena.arena_tests.solver_in_the_loop.plot_states_comparison import plot_states
import numpy as np

from astronomix.option_classes.simulation_config import (
    VAN_ALBADA,
)
from functools import partial

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
    perturb_state_partial = partial(
        perturb_state,
        state=initial_state,
        noise_level=training_config.noise_level,
    )

    def train_step_core(
        cnn_mhd_corrector_params,
        network_params_arrays,
        opt_state,
        params,
        key,
    ):
        # jax.debug.callback(
        #     plot_states,
        #     [initial_state],
        #     [16],
        #     training_config.model_name,
        #     "initial_state_before_noisy",
        #     ["initial_state"],
        # )
        noisy_initial_state = perturb_state_partial(key)
        # jax.debug.callback(
        #     plot_states,
        #     [noisy_initial_state],
        #     [16],
        #     training_config.model_name,
        #     "noisy_initial_state",
        #     ["initial_state"],
        # )

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
                results_low_res.states[-1], target_state, **loss_fn_kwargs
            )

            if training_config.use_checkify:
                jax_checkify.check(jnp.isfinite(loss), "Loss became NaN or Inf!")
            #
            # jax.debug.callback(
            #     plot_states,
            #     [results_low_res.states[-1], target_state],
            #     [16, 16],
            #     training_config.model_name,
            #     "final_states",
            #     ["sol", "target"],
            # )
            #
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
    description: str = "",
):
    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))

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

    logger.debug(
        f"states high res downsampled shape {states_high_res_downsampled.shape}"
    )

    # using different c_cfl for the final target and the simulation
    # this is due to a mistake in the optuna optimization
    params = params._replace(C_cfl=sim_config_training.c_cfl)

    # Setup loss
    # NOTE: take a look at the states inputed should this change per target?
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
        correct_from_beggining=True,
        # we made sure the initial state was at t_correct_begging
        start_correction_time=0.0,
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)
    config_low_res = config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    params_low_res = params._replace(cnn_mhd_corrector_params=cnn_mhd_corrector_params)

    # reseting the lr_scheduler after changing the training time
    cumulative_boundaries = list(np.cumsum(training_config.epochs_per_time)[:-1])
    lr_scheduler = optax.schedules.join_schedules(
        schedules=[
            optax.warmup_cosine_decay_schedule(
                init_value=training_config.learning_rate,
                peak_value=training_config.peak_lr,
                end_value=training_config.end_lr,
                warmup_steps=int(epochs * training_config.warmup_steps_fraction),
                decay_steps=epochs
                - int(epochs * training_config.warmup_steps_fraction),
            )
            for epochs in training_config.epochs_per_time
        ],
        boundaries=cumulative_boundaries,
    )
    # warmup_steps = int(total_epochs * training_config.warmup_steps_fraction)
    # base_scheduler = optax.warmup_cosine_decay_schedule(
    #     init_value=training_config.learning_rate,
    #     peak_value=training_config.peak_lr,
    #     end_value=training_config.end_lr,
    #     warmup_steps=warmup_steps,
    #     decay_steps=total_epochs - warmup_steps,
    # )

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
    early_stopper = EarlyStopper(
        max_patience=training_config.patience,
        use_early_stopper=training_config.use_early_stopper,
    )
    try:
        for i, epochs in enumerate(training_config.epochs_per_time):
            (
                current_end_time,
                current_config,
                current_params_sim,
                current_initial_state,
                current_target,
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

            # plot_states(
            #     [current_target],
            #     [16],
            #     training_config.model_name,
            #     f"current_target_{i}",
            #     ["current_target"],
            # )
            # plot_states(
            #     [current_initial_state],
            #     [16],
            #     training_config.model_name,
            #     f"current_initial_state_{i}",
            #     ["initial_state"],
            # )

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
                    model_manager.save_checkpoint(trained_params, epoch=step + 1)

                logger.info(
                    f"t={current_end_time:.3f} | Step {step + 1}/{epochs} | Loss: {loss:.6f} | "
                    f"Time: {(timer() - start_time_epoch):.3f}s | Grads: {gradients_mod:.3f} | Last t {last_timepoint:.3f}"
                )
                total_step += 1
            succesful_training = True
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

    if training_config.use_early_stopper:
        early_stopped = early_stopper.early_stopped
    else:
        early_stopped = None

    if len(losses) == 0:
        losses.append(math.inf)

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
        notes=description,
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
    model_name=None,
    load_existing=False,
    load_existing_nan=False,
    **overrides,
):
    description = overrides.pop("description", "")
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
            else:
                logger.info(f"key {key} not found in the sim nor training config")

        training_config.model_name = model_name

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
        description=description,
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
    num_cells_high_res = 64

    # TODO: fix the behavior of not putting a t_end, right now the default on the sim_config_training is 0.2
    model_params = {
        "model_name": "ftb_3",
        "num_cells_high_res": 64,
        "downaverage_factor": 2,
        "limiter": VAN_ALBADA,
        "hidden_channels": 5,
        "model_initialization_scale": 0.07,
        "start_correction_time": 0.0,
        "noise_level": 0.04,
        "hidden_layers": 4,
        "learning_rate": 8.5e-05,
        "warmup_steps_fraction": 0.4,
        "peak_lr": 1.5e-04,
        "end_lr": 5.0e-05,
        "gradient_clip": 1.0,
        "c_cfl": 0.8,
        "c_cfl_target": 0.8,
        "loss_type": "norm_mse",
        "direction": FRONT_TO_BACK,
        "epochs_per_time": [150, 150, 200],
        "snapshot_timepoints_train": [0.10, 0.05, 0.0],
        "correct_from_beggining": True,
        "t_end": 0.2,
        "use_early_stopper": False,
        "patience": 35,
        "description": "Testing lr_scheduler per times batch",
    }

    optuna_params = {
        "model_name": "optuna_params_2",
        "num_cells_high_res": 64,
        "downaverage_factor": 2,
        "limiter": VAN_ALBADA,
        "hidden_channels": 5,
        "model_initialization_scale": 0.041574555074364715,
        "start_correction_time": 0.0001352420222864028,
        "noise_level": 0.0,
        "hidden_layers": 3,
        "learning_rate": 1.7231054411611903e-05,
        "warmup_steps_fraction": 0.3983952086077301,
        # "peak_lr": 2.343594314368986e-05,
        "peak_lr": 8.0e-5,
        "end_lr": 3.1872436911245206e-06,
        "gradient_clip": 0.9555056246814991,
        # "c_cfl": 0.6558539756738294,
        "c_cfl": 0.8,
        "c_cfl_target": 0.8,
        "loss_type": "norm_mse",
        "direction": BACK_TO_FRONT,
        "epochs_per_time": [300],
        "snapshot_timepoints_train": [0.2],
        "correct_from_beggining": True,
        "t_end": 0.2,
        "use_early_stopper": True,
        "patience": 30,
        "description": "optuna params but now with the right loss (before there was a bug)",
        # we change the correct_from_beggining to true as the small time was causing problems
    }

    training_params = optuna_params

    params, static = train_new_model(
        load_existing=False,
        load_existing_nan=False,
        use_checkify=False,
        **training_params,
    )

    model_manager = ModelManager(model_name=training_params["model_name"])
    model_metadata = model_manager.load_metadata()
    training_config = model_manager.load_training_config()
    sim_training_config = model_manager.load_simulation_config()

    if model_metadata.succesful_training:
        plot_training(
            neural_net_params=params,
            neural_net_static=static,
            times_eval=jnp.linspace(0.0, 0.3, 30),
            num_cells_high_res=training_params["num_cells_high_res"],
            downaverage_factor=training_params["downaverage_factor"],
            start_correction_time=sim_training_config.start_correction_time,
            correct_from_beggining=sim_training_config.correct_from_beggining,
            model_name=training_config.model_name,
            cfl=sim_training_config.c_cfl,
            cfl_target=sim_training_config.c_cfl_target,
            limiter=sim_training_config.limiter,
        )
    # Continue training an existing model
    # train_new_model(model_name="my_experiment", load_existing=True, epochs_per_time=[100])

    # List all models
    # manager = ModelManager()
    # print(manager.list_modelsinitial_state())
