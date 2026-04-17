"""
Single-GPU training script for H5-based multi-problem solver-in-the-loop correction.

This script provides:
- H5 lazy loading: problems are loaded one-at-a-time per epoch to minimize memory usage
- Configurable per-problem sampling from H5 datasets (e.g., 30 blasts, 10 turbulence per epoch)
- Per-epoch checkpoint saving
- Best model tracking and early stopping
- Integration with conFIG gradient averaging and noise perturbation

Unlike training_model_multigpu.py, this script runs on a single GPU without threading,
making it simpler to debug and suitable for smaller-scale training.
"""

from autocvd import autocvd

autocvd(num_gpus=1)

import gc
import logging
import datetime
import json
import math
import random
import argparse
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict
from timeit import default_timer as timer
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.option_classes.simulation_config import STATE_TYPE

from arena.arena_tests.solver_in_the_loop.conFIG import conFIG
from arena.arena_tests.solver_in_the_loop.utils import perturb_state
from arena.arena_tests.solver_in_the_loop.loss import (
    EarlyStopper,
    loss_setup,
    normalized_weighted_loss,
    simple_mse_loss,
)
from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    TrainingConfig,
    ModelMetadata,
    model_loader,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.dataset.h5_problem_manager import (
    H5ProblemManager,
)

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix.time_stepping import time_integration
from astronomix import finalize_config, get_registered_variables

# Configure logging
logging.basicConfig(
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Default H5 file paths for different problem types
DEFAULT_H5_FILE_PATHS = {
    "mhd_blast": "/export/data/jalegria/solver_in_the_loop/training_blast.h5",
    "turbulence": "/export/data/jalegria/solver_in_the_loop/training_turbulence.h5",
    "ot_vortex": "/export/data/jalegria/solver_in_the_loop/training_ot_vortex.h5",
}

# Default problem counts per epoch
DEFAULT_PROBLEM_COUNTS = {
    "mhd_blast": 30,
    "turbulence": 10,
    "ot_vortex": 3,
}

# Per-problem noise levels
PROBLEM_NOISE_LEVELS = {
    "ot_vortex": 0.03,
    "mhd_blast": 0.03,
    "turbulence": 0.00,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def validate_output(last_t: float, gradients_mod: float, problem_name: str) -> None:
    """Validate that the model didn't run into NaNs or exploding gradients."""
    if last_t == 0.0:
        raise ValueError(f"NaN in forward pass for problem {problem_name}")
    if math.isnan(gradients_mod):
        raise ValueError(f"NaN in gradients for problem {problem_name}")


def _descriptor_problem_name(problem_descriptor) -> str:
    """Resolve the canonical problem name from descriptor variants."""
    if hasattr(problem_descriptor, "name"):
        return problem_descriptor.name
    if hasattr(problem_descriptor, "problem_name"):
        return problem_descriptor.problem_name
    if hasattr(problem_descriptor, "nickname"):
        return problem_descriptor.nickname
    raise AttributeError(
        f"Unable to resolve problem name from descriptor: {problem_descriptor}"
    )


# ============================================================================
# GRADIENT FUNCTION
# ============================================================================


def compute_channel_normalizers(target_state: jnp.ndarray) -> jnp.ndarray:
    """Compute channel normalizers from target state.

    Args:
        target_state: Target state array with shape (num_vars, x, y, z).

    Returns:
        Channel normalizers array with shape (num_vars,).
    """
    return jnp.maximum(jnp.std(target_state, axis=(1, 2, 3)), 1e-8)


def create_grad_fn(
    training_config: TrainingConfig,
    simulation_config,
    registered_variables,
    cnn_mhd_corrector_config,
    noise_level: float,
):
    """Create a grad function that takes states as arguments for lazy loading.

    Unlike the closure-based version in training_model.py, this function returns
    a grad_fn that accepts initial_state and target_state as arguments. This enables
    caching one grad_fn per problem_name (same config) while passing different states.

    Channel normalizers are computed outside the JIT-compiled function and passed
    as a dynamic parameter to avoid recompilation when states change.
    """
    if noise_level == 0.0:
        perturb_state_fn = lambda state, _: state
    else:

        def perturb_state_fn(state, key):
            return perturb_state(state=state, noise_level=noise_level, key=key)

    # Ensure corrector is enabled in config
    simulation_config = simulation_config._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )

    use_norm_mse = training_config.loss_type == "norm_mse"

    def grad_fn_core(
        initial_state,
        target_state,
        cnn_mhd_corrector_params,
        network_params_arrays,
        params,
        key,
        channel_normalizers,
    ):
        """Compute loss and gradients for a single problem with states as args."""
        noisy_initial_state = perturb_state_fn(initial_state, key)

        if use_norm_mse:
            loss_fn_kwargs = {
                "channel_normalizers": channel_normalizers,
                "physics_weights": training_config.physics_weights,
                "use_interface": training_config.use_interface,
            }
            loss_fn_factory = normalized_weighted_loss
        else:
            loss_fn_kwargs = {}
            loss_fn_factory = simple_mse_loss

        def loss_fn(network_params_arrays):
            results_low_res = time_integration(
                noisy_initial_state,
                simulation_config,
                params._replace(
                    cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
                        network_params=network_params_arrays
                    )
                ),
                registered_variables,
            )
            assert isinstance(results_low_res, SnapshotData)
            loss = loss_fn_factory(
                results_low_res.states[-1], target_state, **loss_fn_kwargs
            )
            return loss, results_low_res.time_points[-1]

        (loss_value, last_timepoint), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(network_params_arrays)
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        )
        return loss_value, grads, gradients_modulus, last_timepoint

    grad_fn = jax.jit(grad_fn_core)
    return grad_fn


# ============================================================================
# PROBLEM SAMPLER
# ============================================================================


class ProblemSampler:
    """Manages per-epoch sampling of problems based on configured counts."""

    def __init__(
        self,
        problem_descriptors: List,
        problem_counts: Dict[str, int],
    ):
        self.problem_descriptors = problem_descriptors
        self.problem_counts = problem_counts

        # Build lookup: problem name -> list of descriptors with that name
        self.problem_pairs_map: Dict[str, List] = {}
        for descriptor in problem_descriptors:
            problem_name = _descriptor_problem_name(descriptor)
            if problem_name not in self.problem_pairs_map:
                self.problem_pairs_map[problem_name] = []
            self.problem_pairs_map[problem_name].append(descriptor)

        logger.info(f"ProblemSampler initialized: {self.problem_counts}")

    def sample_epoch(self) -> List:
        """Sample descriptors for one epoch based on configured problem-count combinations.

        Returns:
            List of sampled descriptors for this epoch.
        """
        sampled_pairs = []

        for problem_name, count in self.problem_counts.items():
            if problem_name not in self.problem_pairs_map:
                raise ValueError(
                    f"Problem '{problem_name}' not found. Available: "
                    f"{list(self.problem_pairs_map.keys())}"
                )

            available_descriptors = self.problem_pairs_map[problem_name]
            # Sample with replacement if count > available descriptors
            sampled = random.choices(available_descriptors, k=count)
            sampled_pairs.extend(sampled)

        return sampled_pairs


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def training_model_h5(
    model_manager: ModelManager,
    h5_problem_manager: H5ProblemManager,
    problem_counts: Dict[str, int],
    training_config: TrainingConfig,
    model_name: str,
    load_model: bool = False,
    load_model_nan: bool = False,
    description: str = "",
    use_config_gradient: bool = True,
    early_stopping_patience: int = 30,
    early_stopping_enabled: bool = True,
):
    """Single-GPU training loop with H5-based lazy problem loading.

    Args:
        model_manager: Manager for saving checkpoints and metadata
        h5_problem_manager: Manager for loading training pairs from H5 datasets
        problem_counts: Dict mapping problem_name -> count per epoch
        training_config: Training configuration
        model_name: Name of the model
        load_model: Whether to load existing model
        load_model_nan: Whether to load model from NaN checkpoint
        description: Description of the training run
        use_config_gradient: Whether to use conFIG for gradient averaging
        early_stopping_patience: Patience for early stopping
        early_stopping_enabled: Whether to enable early stopping

    Returns:
        Tuple of (best_params, neural_net_static)
    """

    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Single-GPU H5 Training: {model_name}")
    logger.info(f"Problem counts: {problem_counts}")
    logger.info(f"Epochs: {total_epochs}")
    logger.info("=" * 80 + "\n")

    # Load lightweight problem descriptors from H5 (lazy, no tensors yet)
    problem_descriptors = h5_problem_manager.get_problem_descriptors()
    if not problem_descriptors:
        raise ValueError("No H5 problem descriptors found for training")

    # Initialize sampler
    sampler = ProblemSampler(
        problem_descriptors=problem_descriptors,
        problem_counts=problem_counts,
    )

    # Materialize one pair to derive architecture
    bootstrap_problem_name = next(iter(problem_counts.keys()))
    bootstrap_descriptor = sampler.problem_pairs_map[bootstrap_problem_name][0]
    first_pair = h5_problem_manager.get_training_pairs_for_descriptors(
        [bootstrap_descriptor]
    )[0]
    registered_variables = first_pair.lr_bundle.reg_vars

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

    # Initialize corrector config/params
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=True,
        start_correction_time=0.0,
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    # Setup learning rate scheduler
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

    # Initialize tracking
    losses_avg = []
    losses_per_problem: Dict[str, List[float]] = {}

    trained_params = neural_net_params
    best_loss = float("inf")
    best_params = trained_params
    best_epoch = 0

    early_stopper = EarlyStopper(
        max_patience=early_stopping_patience,
        use_early_stopper=early_stopping_enabled,
    )

    logger.info(f"Total available descriptors: {len(problem_descriptors)}")
    logger.info(f"Problem counts per epoch: {problem_counts}")

    successful_training = True

    # Cache for grad functions (keyed by problem_name)
    grad_fn_cache: Dict[str, Callable] = {}

    try:
        key = jax.random.PRNGKey(112)
        early_stopper.reset_patience()

        # Main training loop
        for epoch in range(training_config.epochs_per_time[0]):
            key, epoch_key = jax.random.split(key)
            start_time_epoch = timer()

            # Sample descriptors for this epoch (lightweight, no tensors loaded yet)
            sampled_descriptors = sampler.sample_epoch()

            logger.info(
                f"\nEpoch {epoch + 1}/{training_config.epochs_per_time[0]} | "
                f"Sampled {len(sampled_descriptors)} problems"
            )

            # Materialize training pairs from H5 (loads states into memory)
            training_pairs = h5_problem_manager.get_training_pairs_for_descriptors(
                sampled_descriptors
            )

            # Compute gradients for each problem
            all_grads = []
            step_losses: Dict[str, float] = {}
            total_grad_mod = 0.0

            for i, pair in enumerate(training_pairs):
                key, subkey = jax.random.split(key)
                problem_name = _descriptor_problem_name(pair.problem_descriptor)
                noise_level = PROBLEM_NOISE_LEVELS.get(
                    problem_name, training_config.noise_level
                )

                # Get or create cached grad_fn for this problem type
                if problem_name not in grad_fn_cache:
                    # Ensure config has corrector enabled
                    config_lr = pair.lr_bundle.config._replace(
                        cnn_mhd_corrector_config=cnn_mhd_corrector_config,
                    )
                    grad_fn_cache[problem_name] = create_grad_fn(
                        training_config=training_config,
                        simulation_config=config_lr,
                        registered_variables=pair.lr_bundle.reg_vars,
                        cnn_mhd_corrector_config=cnn_mhd_corrector_config,
                        noise_level=noise_level,
                    )

                grad_fn = grad_fn_cache[problem_name]

                # Compute channel normalizers outside JIT to avoid recompilation
                channel_normalizers = compute_channel_normalizers(pair.target_state)

                # Compute gradient
                p_loss, p_grads, p_grad_mod, p_last_t = grad_fn(
                    pair.lr_bundle.initial_state,
                    pair.target_state,
                    cnn_mhd_corrector_params,
                    trained_params,
                    pair.lr_bundle.params._replace(
                        cnn_mhd_corrector_params=cnn_mhd_corrector_params
                    ),
                    subkey,
                    channel_normalizers,
                )

                # Validate output
                validate_output(
                    last_t=float(p_last_t),
                    gradients_mod=float(p_grad_mod),
                    problem_name=problem_name,
                )

                all_grads.append(p_grads)
                loss_val = float(p_loss)
                step_losses[f"{problem_name}_{i}"] = loss_val
                losses_per_problem.setdefault(problem_name, []).append(loss_val)
                total_grad_mod += float(p_grad_mod)

            # Free memory from loaded states
            del training_pairs
            gc.collect()

            # Average gradients
            num_problems = len(all_grads)
            if use_config_gradient:
                start_time_config = timer()
                grads = conFIG(all_grads, use_least_square=True)
                logger.info(
                    f"conFIG gradient averaging took {timer() - start_time_config:.3f}s"
                )
            else:
                grads = jax.tree.map(lambda *gs: sum(gs) / num_problems, *all_grads)

            # Optimizer update
            updates, opt_state = optimizer.update(grads, opt_state, trained_params)
            trained_params = eqx.apply_updates(trained_params, updates)

            # Update corrector params
            cnn_mhd_corrector_params = cnn_mhd_corrector_params._replace(
                network_params=trained_params
            )

            # Track losses
            avg_loss = (
                sum(step_losses.values()) / num_problems if num_problems > 0 else 0.0
            )
            avg_grad_mod = total_grad_mod / num_problems if num_problems > 0 else 0.0
            losses_avg.append(avg_loss)

            # Check early stopping
            if early_stopper.new_epoch(avg_loss):
                logger.info("Early stopping triggered")
                break

            # Track best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = trained_params
                best_epoch = epoch + 1
                logger.info(f"✓ New best loss: {best_loss:.6f}")

            # Save checkpoint every 50 epochs
            if (epoch + 1) % 50 == 0:
                model_manager.save_checkpoint(trained_params, epoch=epoch + 1)

            epoch_time = timer() - start_time_epoch
            logger.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.6f}, "
                f"grad_mod={avg_grad_mod:.3f}, time={epoch_time:.2f}s"
            )

        successful_training = True

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        model_manager.save_model_params(trained_params, "model_params_NAN.eqx")
        successful_training = False

    training_time = timer() - start_time

    if successful_training:
        logger.info(
            f"\nTraining Complete! Time: {training_time:.2f}s | Best Loss: {best_loss:.6f}"
        )
        model_manager.save_model_params(best_params)

    # Save losses
    model_manager.save_losses(losses_avg)
    losses_save = {"avg": np.array(losses_avg)}
    for name, losses in losses_per_problem.items():
        losses_save[name] = np.array(losses)
    losses_path = (
        model_manager.base_dir / model_manager.model_name / "losses_per_problem.npz"
    )
    np.savez(losses_path, **losses_save, allow_pickle=True)

    if early_stopping_enabled:
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
        succesful_training=successful_training,
        early_stopped=early_stopped,
        final_epoch=epoch + 1 if successful_training else 0,
        notes=description,
    )
    model_manager.save_metadata(metadata)

    return best_params, neural_net_static


# ============================================================================
# ENTRY POINT
# ============================================================================


def train_new_model_h5(
    h5_file_paths: Dict[str, str],
    h5_problem_ranges: Dict[str, Tuple[int, int]],
    problem_counts: Dict[str, int],
    model_name: Optional[str] = None,
    load_existing: bool = False,
    load_existing_nan: bool = False,
    description: str = "",
    use_config_gradient: bool = True,
    early_stopping_patience: int = 30,
    early_stopping_enabled: bool = True,
    **training_config_overrides,
):
    """Main entry point for H5-based single-GPU training.

    Args:
        h5_file_paths: Dict mapping problem names to h5 file paths
        h5_problem_ranges: Dict mapping problem names to (start, end) index ranges
        problem_counts: Dict mapping problem names to count per epoch
        model_name: Name of the model (auto-generated if None)
        load_existing: Whether to load existing model
        load_existing_nan: Whether to load from NaN checkpoint
        description: Description of the training run
        use_config_gradient: Whether to use conFIG for gradient averaging
        early_stopping_patience: Patience for early stopping
        early_stopping_enabled: Whether to enable early stopping
        **training_config_overrides: Overrides for TrainingConfig fields

    Returns:
        Tuple of (best_params, neural_net_static)
    """

    # Setup model manager
    model_manager = ModelManager(
        base_dir="arena/data/models/multiproblem_w_dataset",
        model_name=model_name,
    )

    if load_existing and model_name:
        logger.info(f"Loading configs for model {model_name}")
        training_config = model_manager.load_training_config()
    else:
        model_name = model_manager.create_model_directory()
        logger.info(f"Created model directory: {model_name}")

        training_config = TrainingConfig(
            epochs_per_time=[100],
            snapshot_timepoints_train=[0.2],
            model_initialization_scale=0.02,
            learning_rate=8.5e-5,
            peak_lr=0.00015,
            end_lr=5e-5,
            noise_level=0.04,
        )

        # Apply overrides
        for key, value in training_config_overrides.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
            else:
                logger.warning(f"Unknown training config field: {key}")

        training_config.model_name = model_name
        model_manager.save_training_config(training_config)

    # Validate h5 configuration
    if not h5_file_paths:
        raise ValueError("h5_file_paths cannot be empty")

    logger.info(f"H5 File Paths: {h5_file_paths}")
    logger.info(f"H5 Problem Ranges: {h5_problem_ranges}")
    logger.info(f"Problem Counts: {problem_counts}")

    # Create H5 problem manager
    h5_problem_manager = H5ProblemManager(
        h5_file_paths=h5_file_paths,
        training_config=training_config,
        problem_ranges=h5_problem_ranges,
    )

    # Validate that all problems in counts have H5 data
    for problem_name in problem_counts.keys():
        if problem_name not in h5_problem_manager.h5_loaders:
            raise ValueError(
                f"Problem '{problem_name}' in counts not found in H5 loaders. "
                f"Available: {list(h5_problem_manager.h5_loaders.keys())}"
            )

    logger.info("\n" + "=" * 80)
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Training Config: epochs={training_config.epochs_per_time}")
    logger.info(f"Problem Counts: {problem_counts}")
    logger.info("=" * 80 + "\n")

    # Save problem counts config
    problem_counts_path = (
        model_manager.base_dir / model_manager.model_name / "problem_counts.json"
    )
    with open(problem_counts_path, "w") as f:
        json.dump(problem_counts, f, indent=2)

    # Run training
    best_params, neural_net_static = training_model_h5(
        model_manager=model_manager,
        h5_problem_manager=h5_problem_manager,
        problem_counts=problem_counts,
        training_config=training_config,
        model_name=model_name,
        load_model=load_existing,
        load_model_nan=load_existing_nan,
        description=description,
        use_config_gradient=use_config_gradient,
        early_stopping_patience=early_stopping_patience,
        early_stopping_enabled=early_stopping_enabled,
    )

    model_manager.print_model_info()
    return best_params, neural_net_static


# ============================================================================
# CLI USAGE
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H5-based single-GPU training for multi-problem solver correction"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the model (auto-generated if not provided)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--h5-file",
        type=str,
        action="append",
        dest="h5_files",
        help=(
            "H5 file specification (format: 'problem_name:/path/to/file.h5'). "
            "Can be repeated. If omitted, defaults to blast/turbulence/ot_vortex "
            "files under /export/data/jalegria/solver_in_the_loop/."
        ),
    )

    parser.add_argument(
        "--h5-range",
        type=str,
        action="append",
        dest="h5_ranges",
        help="H5 problem range (format: 'problem_name:start:end'). Can be repeated.",
    )
    parser.add_argument(
        "--problem",
        type=str,
        action="append",
        dest="problems",
        help=(
            "Problem and count (format: 'mhd_blast:30'). Can be repeated. "
            "Defaults to mhd_blast:30, turbulence:10, ot_vortex:3 if omitted."
        ),
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Disable early stopping",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Disable conFIG gradient averaging (use simple mean)",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=8.5e-05,
        help="Learning rate",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=5,
        help="Number of hidden channels in CNN",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=4,
        help="Number of hidden layers in CNN",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of the training run",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing model weights",
    )

    args = parser.parse_args()

    # Parse H5 file paths
    h5_file_paths = {}
    h5_file_specs = args.h5_files or [
        f"{problem_name}:{filepath}"
        for problem_name, filepath in DEFAULT_H5_FILE_PATHS.items()
    ]
    if args.h5_files is None:
        logger.info("Using default H5 file mapping for blast/turbulence/ot_vortex")

    for h5_spec in h5_file_specs:
        parts = h5_spec.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid H5 file spec: {h5_spec}. Format: problem_name:/path/to/file.h5"
            )
        problem_name, filepath = parts
        h5_file_paths[problem_name] = filepath
        logger.info(f"H5 file for {problem_name}: {filepath}")

    # Parse H5 problem ranges
    h5_problem_ranges = {}
    if args.h5_ranges:
        for range_spec in args.h5_ranges:
            parts = range_spec.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid H5 range spec: {range_spec}. Format: problem_name:start:end"
                )
            problem_name, start, end = parts
            h5_problem_ranges[problem_name] = (int(start), int(end))
            logger.info(f"H5 range for {problem_name}: [{start}, {end})")

    # Parse problem counts
    problem_counts = {}
    problem_specs = args.problems or [
        f"{problem_name}:{count}"
        for problem_name, count in DEFAULT_PROBLEM_COUNTS.items()
    ]
    if args.problems is None:
        logger.info(
            "Using default problem counts: mhd_blast=30, turbulence=10, ot_vortex=3"
        )

    for problem_spec in problem_specs:
        parts = problem_spec.split(":")
        if len(parts) != 2:
            raise ValueError(
                f"Invalid problem spec: {problem_spec}. Format: problem_name:count"
            )
        name, count = parts
        problem_counts[name] = int(count)
        logger.info(f"Problem count: {name}={count}")

    if not problem_counts:
        raise ValueError("At least one problem must be specified via --problem")

    if not h5_problem_ranges:
        logger.info(
            "No --h5-range provided; using full range for each configured problem"
        )

    # Validate that all problems have h5 files
    for problem_name in problem_counts.keys():
        if problem_name not in h5_file_paths:
            raise ValueError(f"Problem '{problem_name}' requires h5 file via --h5-file")

    logger.info(f"Problem counts: {problem_counts}")

    training_config_overrides = {
        "epochs_per_time": [args.epochs],
        "learning_rate": args.learning_rate,
        "hidden_channels": args.hidden_channels,
        "hidden_layers": args.hidden_layers,
        "num_cells_high_res": 64,
        "downaverage_factor": 2,
        "limiter": 4,
        "model_initialization_scale": 0.03,
        "start_correction_time": 0.0,
        "noise_level": 0.00,
        "warmup_steps_fraction": 0.4,
        "peak_lr": 1.5e-04,
        "end_lr": 5.0e-05,
        "gradient_clip": 1.0,
        "c_cfl": 0.8,
        "c_cfl_target": 0.8,
        "loss_type": "norm_mse",
        "snapshot_timepoints_train": [0.666],
        "correct_from_beggining": True,
        "t_end": 0.2,
        "use_early_stopper": False,
        "patience": 35,
    }

    # Run training
    best_params, neural_net_static = train_new_model_h5(
        h5_file_paths=h5_file_paths,
        h5_problem_ranges=h5_problem_ranges,
        problem_counts=problem_counts,
        model_name=args.model_name,
        load_existing=args.load_existing,
        description=args.description,
        use_config_gradient=not args.no_config,
        early_stopping_enabled=args.early_stopping,
        **training_config_overrides,
    )

    logger.info("✓ Training completed successfully")
