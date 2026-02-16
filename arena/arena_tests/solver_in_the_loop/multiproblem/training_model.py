from dataclasses import dataclass, field
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
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
    ProblemManager,
    PROBLEM_CATALOG,
)
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


@dataclass
class MultiproblemTrainingConfig(TrainingConfig):
    problems: list[str] = field(default_factory=lambda: ["mhd_blast"])


def validate_output(last_t: float, gradients_mod: float) -> None:
    """Validate that the model didnt run into nans or exploding grads"""
    if last_t == 0.0:
        raise ValueError("Nan in forward pass")
    if math.isnan(gradients_mod):
        raise ValueError("Nan in grads")


def create_grad_fn(
    loss_fn_factory,
    loss_fn_kwargs,
    training_config: MultiproblemTrainingConfig,
    simulation_config: SimulationConfig,
    initial_state: STATE_TYPE,
    target_state: STATE_TYPE | SnapshotData,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
):
    """Create a function that computes loss and gradients for a single problem
    (no optimizer update — that happens after averaging across problems)."""
    perturb_state_partial = partial(
        perturb_state,
        state=initial_state,
        noise_level=training_config.noise_level,
    )

    def grad_fn_core(
        cnn_mhd_corrector_params,
        network_params_arrays,
        params,
        key,
    ):
        noisy_initial_state = perturb_state_partial(key)

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
        return loss_value, grads, gradients_modulus, last_timepoint

    if training_config.use_checkify:
        errors = (
            jax_checkify.user_checks
            | jax_checkify.float_checks
            | jax_checkify.nan_checks
            | jax_checkify.div_checks
        )
        checked_fn = jax_checkify.checkify(grad_fn_core, errors=errors)

        @jax.jit
        def grad_fn(*args):
            err, out = checked_fn(*args)
            return err, out
    else:
        grad_fn = jax.jit(grad_fn_core)

    return grad_fn


def training_model(
    model_manager: ModelManager,
    model_name: str,
    training_config: MultiproblemTrainingConfig,
    sim_config_training: SimulationConfigTraining,
    load_model: bool = False,
    load_model_nan: bool = False,
    description: str = "",
):
    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))

    # Build training pairs for all problems via ProblemManager
    problem_manager = ProblemManager(
        problem_names=training_config.problems,
        sim_config_training=sim_config_training,
        training_config=training_config,
    )
    training_pairs = problem_manager.get_training_pairs()

    # Use the first problem's LR bundle to determine model architecture
    first_pair = training_pairs[0]
    registered_variables = first_pair.lr_bundle.reg_vars

    # TODO: i dont think that the loss should average based on the first problem target states
    # Setup loss using the first problem's targets for normalization
    loss_fn_kwargs, loss_fn_factory = loss_setup(
        training_config, target_states=first_pair.target_states
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
        start_correction_time=0.0,
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    # Attach corrector config/params and override c_cfl on each LR bundle
    for pair in training_pairs:
        pair.lr_bundle.override_solver_in_the_loop(
            corrector_config=cnn_mhd_corrector_config,
            corrector_params=cnn_mhd_corrector_params,
        )
        pair.lr_bundle.params = pair.lr_bundle.params._replace(
            C_cfl=sim_config_training.c_cfl
        )

    # Learning rate scheduler
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

    optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.gradient_clip),
        optax.adamw(learning_rate=lr_scheduler),
    )
    opt_state = optimizer.init(neural_net_params)

    # Per-problem and averaged loss tracking
    num_problems = len(training_pairs)
    problem_names = [p.problem_name for p in training_pairs]
    losses_avg = []
    losses_per_problem = {name: [] for name in problem_names}

    trained_params = neural_net_params
    best_loss = float("inf")
    best_params = trained_params
    logger.info(f"Starting training with {num_problems} problem(s): {problem_names}")
    total_step = 0
    succesful_training = True
    early_stopper = EarlyStopper(
        max_patience=training_config.patience,
        use_early_stopper=training_config.use_early_stopper,
    )
    try:
        for i, epochs in enumerate(training_config.epochs_per_time):
            # Build timepoint context and grad function for each problem
            grad_fns = []
            problem_contexts = []
            for pair in training_pairs:
                ctx = timepoint_context(
                    i,
                    sim_config_training=sim_config_training,
                    training_config=training_config,
                    config=pair.lr_bundle.config,
                    params=pair.lr_bundle.params,
                    target_states=pair.target_states,
                    initial_state=pair.lr_bundle.initial_state,
                    direction=training_config.direction,
                )
                problem_contexts.append(ctx)
                (
                    current_end_time,
                    current_config,
                    _,  # params
                    current_initial_state,
                    current_target,
                    _,  # key
                ) = ctx
                grad_fns.append(
                    create_grad_fn(
                        loss_fn_factory=loss_fn_factory,
                        loss_fn_kwargs=loss_fn_kwargs,
                        training_config=training_config,
                        simulation_config=current_config,
                        initial_state=current_initial_state,
                        target_state=current_target,
                        helper_data=pair.lr_bundle.helper,
                        registered_variables=pair.lr_bundle.reg_vars,
                    )
                )

            early_stopper.reset_patience()
            current_end_time = problem_contexts[0][0]
            key = problem_contexts[0][5]

            for step in range(epochs):
                key, subkey = jax.random.split(key)
                start_time_epoch = timer()

                # Compute gradients for each problem
                all_grads = []
                step_losses = {}
                total_grad_mod = 0.0

                for p_idx, (grad_fn, ctx, pair) in enumerate(
                    zip(grad_fns, problem_contexts, training_pairs)
                ):
                    _, _, p_params_sim, _, _, _ = ctx
                    if training_config.use_checkify:
                        err, (p_loss, p_grads, p_grad_mod, p_last_t) = grad_fn(
                            cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                            network_params_arrays=trained_params,
                            params=p_params_sim,
                            key=subkey,
                        )
                        err.throw()
                    else:
                        p_loss, p_grads, p_grad_mod, p_last_t = grad_fn(
                            cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                            network_params_arrays=trained_params,
                            params=p_params_sim,
                            key=subkey,
                        )

                    validate_output(last_t=p_last_t, gradients_mod=p_grad_mod)
                    all_grads.append(p_grads)
                    step_losses[pair.problem_name] = float(p_loss)
                    total_grad_mod += float(p_grad_mod)

                # Average gradients across problems, single optimizer update
                avg_grads = jax.tree.map(lambda *gs: sum(gs) / num_problems, *all_grads)
                updates, opt_state = optimizer.update(
                    avg_grads, opt_state, trained_params
                )
                trained_params = eqx.apply_updates(trained_params, updates)

                avg_loss = sum(step_losses.values()) / num_problems
                avg_grad_mod = total_grad_mod / num_problems

                # Record losses
                losses_avg.append(avg_loss)
                for name in problem_names:
                    losses_per_problem[name].append(step_losses[name])

                if early_stopper.new_epoch(avg_loss):
                    logger.info("Finished training due to early stopper")
                    break

                if avg_loss < best_loss:
                    best_loss, best_params = avg_loss, trained_params

                if (step + 1) % 50 == 0:
                    model_manager.save_checkpoint(trained_params, epoch=step + 1)

                per_problem_str = " | ".join(
                    f"{name}: {step_losses[name]:.6f}" for name in problem_names
                )
                logger.info(
                    f"t={current_end_time:.3f} | Step {step + 1}/{epochs} | "
                    f"Avg Loss: {avg_loss:.6f} | {per_problem_str} | "
                    f"Time: {(timer() - start_time_epoch):.3f}s | "
                    f"Avg Grads: {avg_grad_mod:.3f}"
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

    # Save averaged losses and per-problem losses
    model_manager.save_losses(losses_avg)
    losses_save = {"avg": np.array(losses_avg)}
    for name in problem_names:
        losses_save[name] = np.array(losses_per_problem[name])
    losses_path = (
        model_manager.base_dir / model_manager.model_name / "losses_per_problem.npz"
    )
    np.savez(losses_path, **losses_save)

    if training_config.use_early_stopper:
        early_stopped = early_stopper.early_stopped
    else:
        early_stopped = None

    if len(losses_avg) == 0:
        losses_avg.append(math.inf)

    metadata = ModelMetadata(
        model_name=model_name,
        created_at=datetime.datetime.now().isoformat(),
        total_epochs=total_epochs,
        final_loss=float(losses_avg[-1]),
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
    manager = ModelManager(
        base_dir="arena/data/models/multiproblem", model_name=model_name
    )

    if load_existing and model_name:
        logger.info(f"Loading configs for model {model_name}")
        training_config = manager.load_training_config()
        sim_config_training = manager.load_simulation_config()
    else:
        model_name = manager.create_model_directory()
        logger.info(f"Created: {model_name}")

        training_config = MultiproblemTrainingConfig(
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
        "problems": ["mhd_blast"],
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
        "problems": ["mhd_blast"],
    }

    training_params = optuna_params

    params, static = train_new_model(
        load_existing=False,
        load_existing_nan=False,
        use_checkify=False,
        **training_params,
    )

    model_manager = ModelManager(
        base_dir="arena/data/models/multiproblem",
        model_name=training_params["model_name"],
    )
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
