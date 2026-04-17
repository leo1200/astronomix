"""
Multi-GPU training script with cached JIT compilation for solver-in-the-loop correction.

This script provides:
- Pre-compilation: JIT-compiles grad functions for each unique problem.name ONCE before training
- Shared cache: Pre-compiled functions are shared across all GPU workers (read-only, no locking)
- External normalizers: Channel normalizers computed OUTSIDE JIT to avoid recompilation
- Full conFIG: All individual gradients from all GPUs collected for conFIG (not per-GPU averages)

Key differences from training_model_multigpu.py:
1. Pre-compiles one grad_fn per problem.name before distributing work
2. Workers use pre-compiled functions without recompilation
3. Channel normalizers passed as dynamic parameters
4. conFIG applied to ALL individual gradients, not averaged per GPU
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from autocvd import autocvd
import logging
import datetime
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict
from timeit import default_timer as timer
import random
import threading
from queue import Queue
import argparse
import sys
import traceback
import time


def _resolve_startup_num_gpus(default_num_gpus: int = 1) -> int:
    """Resolve --num-gpus before JAX setup so autocvd honors CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-gpus", type=int, default=default_num_gpus)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.num_gpus


STARTUP_NUM_GPUS = _resolve_startup_num_gpus()
autocvd(num_gpus=STARTUP_NUM_GPUS)

import gc
import numpy as np
import jax

# Enable JIT compilation logging for debugging cache behavior
# jax.config.update("jax_log_compiles", True)
# # Try to enable cache miss explanations (JAX 0.4.26+)
# try:
#     jax.config.update("jax_explain_cache_misses", True)
# except Exception:
#     pass  # Not available in older JAX versions

import jax.numpy as jnp
import equinox as eqx
import optax
import math
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from astronomix.data_classes.simulation_snapshot_data import SnapshotData

from arena.arena_tests.solver_in_the_loop.conFIG import conFIG
from arena.arena_tests.solver_in_the_loop.utils import perturb_state
from arena.arena_tests.solver_in_the_loop.loss import (
    EarlyStopper,
    normalized_weighted_loss,
    simple_mse_loss,
)
from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    TrainingConfig,
    ModelMetadata,
    model_loader,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.multigpu_config import (
    MultiGPUTrainingConfig,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.dataset.h5_problem_manager import (
    H5ProblemManager,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_sampler import (
    ProblemSampler,
    _descriptor_problem_name,
)

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
    FiLMCorrectorCNN,
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

DEFAULT_H5_FILE_PATHS = {
    "mhd_blast": "/export/data/jalegria/solver_in_the_loop/training_blast.h5",
    "turbulence": "/export/data/jalegria/solver_in_the_loop/training_turbulence.h5",
    "ot_vortex": "/export/data/jalegria/solver_in_the_loop/training_ot_vortex.h5",
}

DEFAULT_PROBLEM_COUNTS = {
    "mhd_blast": 30,
    "turbulence": 10,
    "ot_vortex": 3,
}

PROBLEM_NOISE_LEVELS = {
    "ot_vortex": 0.03,
    "mhd_blast": 0.03,
    "turbulence": 0.00,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def compute_channel_normalizers(target_state: jnp.ndarray) -> jnp.ndarray:
    """Compute channel normalizers from target state.

    This function computes normalizers OUTSIDE the JIT-compiled gradient function
    to avoid recompilation when target states change.

    Args:
        target_state: Target state array with shape (num_vars, x, y, z).

    Returns:
        Channel normalizers array with shape (num_vars,).
    """
    return jnp.maximum(jnp.std(target_state, axis=(1, 2, 3)), 1e-8)


def validate_output(last_t: float, gradients_mod: float, problem_name: str) -> None:
    """Validate that the model didn't run into NaNs or exploding gradients."""
    if last_t == 0.0:
        raise ValueError(f"NaN in forward pass for problem {problem_name}")
    if math.isnan(gradients_mod):
        raise ValueError(f"NaN in gradients for problem {problem_name}")


def is_probable_oom_error(exc: Exception) -> bool:
    """Best-effort classification for memory exhaustion errors."""
    error_text = f"{type(exc).__name__}: {exc}".lower()
    oom_markers = (
        "out of memory",
        "oom",
        "resource exhausted",
        "resource_exhausted",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "memory allocation",
    )
    return any(marker in error_text for marker in oom_markers)


def classify_worker_error(exc: Exception) -> str:
    """Classify worker failures for accurate downstream reporting."""
    if is_probable_oom_error(exc):
        return "oom"

    if "nan" in str(exc).lower():
        return "nan"

    return "other"


def print_exception_summary(context: str, exc: Exception) -> None:
    """Print exception summary to stderr for immediate terminal visibility."""
    print(
        f"[EXCEPTION] {context}: {type(exc).__name__}: {exc}",
        file=sys.stderr,
        flush=True,
    )


def print_exception_traceback(context: str, traceback_text: str) -> None:
    """Print exception traceback to stderr for immediate terminal visibility."""
    if not traceback_text:
        return
    print(
        f"[TRACEBACK] {context}\n{traceback_text}",
        file=sys.stderr,
        flush=True,
    )


def _flatten_gradient_pytree(grad_pytree: Any) -> jnp.ndarray:
    """Flatten a gradient PyTree into a 1D vector of inexact leaves."""
    flat_leaves = []
    for leaf in jax.tree_util.tree_leaves(grad_pytree):
        if leaf is None:
            continue
        leaf_array = jnp.asarray(leaf)
        if jnp.issubdtype(leaf_array.dtype, jnp.inexact):
            flat_leaves.append(jnp.ravel(leaf_array))

    if not flat_leaves:
        return jnp.array([], dtype=jnp.float32)

    return jnp.concatenate(flat_leaves)


def analyze_gradients_debug(
    all_individual_grads: List,
    epoch: int,
    debug_dir: Path,
) -> None:
    """Analyze gradients and save debug visualizations.

    Computes:
    1. Ratio of norm_of_mean / mean_individual_norm
    2. Cosine similarity matrix of normalized gradients
    3. Mean cosine similarity
    4. Histogram of cosine similarity values

    Args:
        all_individual_grads: List of gradient PyTrees from all problems
        epoch: Current epoch number
        debug_dir: Directory to save debug outputs
    """
    logger.info(f"Running gradient analysis for epoch {epoch}...")

    # Create debug directory if needed
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Convert gradients to flat arrays for analysis
    num_grads = len(all_individual_grads)

    # Flatten each gradient PyTree into a single vector
    flat_grads = []
    for grad in all_individual_grads:
        flat = _flatten_gradient_pytree(grad)
        flat_grads.append(flat)

    # Stack into a matrix: (num_grads, grad_dim)
    grad_matrix = jnp.stack(flat_grads, axis=0)

    # 1. Compute ratio = norm_of_mean / mean_individual_norm
    mean_grad = jnp.mean(grad_matrix, axis=0)
    norm_of_mean = jnp.linalg.norm(mean_grad)

    individual_norms = jnp.linalg.norm(grad_matrix, axis=1)
    mean_individual_norm = jnp.mean(individual_norms)

    ratio = norm_of_mean / (mean_individual_norm + 1e-10)

    logger.info(f"  Gradient ratio (norm_of_mean / mean_individual_norm): {ratio:.6f}")

    # 2. Compute cosine similarity matrix with normalized gradients
    # Normalize each gradient
    norms = jnp.linalg.norm(grad_matrix, axis=1, keepdims=True)
    normalized_grads = grad_matrix / (norms + 1e-10)

    # Cosine similarity matrix: (num_grads, num_grads)
    cos_matrix = jnp.dot(normalized_grads, normalized_grads.T)

    # 3. Compute mean cosine similarity (excluding diagonal)
    N = cos_matrix.shape[0]
    mean_cos = (cos_matrix.sum() - N) / (N * (N - 1)) if N > 1 else 0.0

    logger.info(f"  Mean cosine similarity: {mean_cos:.6f}")

    # 4. Plot histogram of cosine similarity values (excluding diagonal)
    # Extract upper triangular part (excluding diagonal)
    cos_values = cos_matrix[jnp.triu_indices(N, k=1)]
    cos_values_np = np.array(cos_values)

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(cos_values_np, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cosine Similarity", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        f"Gradient Cosine Similarity Distribution (Epoch {epoch})", fontsize=14
    )
    ax.axvline(
        mean_cos,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_cos:.4f}",
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save histogram
    hist_path = debug_dir / f"cosine_similarity_epoch_{epoch:04d}.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saved cosine similarity histogram to {hist_path}")

    # Save numerical statistics to a text file
    stats_path = debug_dir / f"gradient_stats_epoch_{epoch:04d}.txt"
    with open(stats_path, "w") as f:
        f.write(f"Gradient Analysis - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of gradients: {num_grads}\n")
        f.write(f"Gradient dimension: {grad_matrix.shape[1]}\n\n")
        f.write(f"Norm of mean gradient: {norm_of_mean:.6e}\n")
        f.write(f"Mean of individual norms: {mean_individual_norm:.6e}\n")
        f.write(f"Ratio (norm_of_mean / mean_individual_norm): {ratio:.6f}\n\n")
        f.write(f"Mean cosine similarity: {mean_cos:.6f}\n")
        f.write(f"Min cosine similarity: {cos_values_np.min():.6f}\n")
        f.write(f"Max cosine similarity: {cos_values_np.max():.6f}\n")
        f.write(f"Std cosine similarity: {cos_values_np.std():.6f}\n")

    logger.info(f"  Saved gradient statistics to {stats_path}")


# ============================================================================
# GRADIENT NORMALIZATION AND CLIPPING UTILITIES
# ============================================================================


def normalize_gradients_per_problem(
    gradients_per_problem: List[Any], problem_names: List[str]
) -> Tuple[List[Any], Dict[str, float]]:
    """Normalize gradients for each problem by their L2 norm.

    Args:
        gradients_per_problem: List of gradient PyTrees, one per problem
        problem_names: List of problem names corresponding to gradients

    Returns:
        Tuple of (normalized_gradients, norm_stats_dict)
    """
    normalized_grads = []
    norm_stats = {}

    for grads, problem_name in zip(gradients_per_problem, problem_names, strict=True):
        # Flatten gradient to compute L2 norm
        flat_grad = _flatten_gradient_pytree(grads)
        grad_norm = jnp.linalg.norm(flat_grad)

        # Normalize by dividing each leaf by the norm
        norm_stats[problem_name] = float(grad_norm)
        if grad_norm > 1e-10:
            normalized = jax.tree.map(
                lambda x: x / grad_norm if x is not None else None, grads
            )
        else:
            # If norm is near zero, keep gradients as-is
            logger.warning(
                f"Gradient norm near zero for {problem_name} ({float(grad_norm):.2e}), skipping normalization"
            )
            normalized = grads

        normalized_grads.append(normalized)

    logger.info(
        f"Normalized {len(gradients_per_problem)} problem gradients. "
        f"Mean norm: {float(jnp.mean(jnp.array(list(norm_stats.values())))):.3f}"
    )

    return normalized_grads, norm_stats


def clip_gradients_per_problem(
    gradients_per_problem: List[Any],
    problem_names: List[str],
    max_norm: float = 1.0,
) -> Tuple[List[Any], Dict[str, float]]:
    """Clip gradients for each problem to a maximum L2 norm.

    Args:
        gradients_per_problem: List of gradient PyTrees, one per problem
        problem_names: List of problem names corresponding to gradients
        max_norm: Maximum L2 norm to clip to (default: 1.0)

    Returns:
        Tuple of (clipped_gradients, norm_stats_dict)
    """
    clipped_grads = []
    norm_stats = {}

    for grads, problem_name in zip(gradients_per_problem, problem_names, strict=True):
        # Flatten gradient to compute L2 norm
        flat_grad = _flatten_gradient_pytree(grads)
        grad_norm = jnp.linalg.norm(flat_grad)

        norm_stats[problem_name] = float(grad_norm)

        # Clip if norm exceeds max_norm
        if grad_norm > max_norm:
            clipped = jax.tree.map(
                lambda x: x * (max_norm / grad_norm) if x is not None else None,
                grads,
            )
        else:
            clipped = grads

        clipped_grads.append(clipped)

    clipped_count = sum(
        1 for norm in norm_stats.values() if norm > max_norm
    )
    logger.info(
        f"Clipped gradients for {clipped_count}/{len(gradients_per_problem)} problems to max_norm={max_norm}. "
        f"Mean norm: {float(jnp.mean(jnp.array(list(norm_stats.values())))):.3f}"
    )

    return clipped_grads, norm_stats


# ============================================================================
# CACHED GRADIENT FUNCTION
# ============================================================================


def create_cached_grad_fn(
    training_config: TrainingConfig,
    simulation_config,
    registered_variables,
    cnn_mhd_corrector_config,
    base_params,
    noise_level: float,
):
    """Create a grad function with minimal dynamic arguments for stable JIT caching.

    This function is designed to be pre-compiled ONCE per problem.name and then
    shared across all GPU workers. All configuration objects are captured in the
    closure to ensure stable PyTree structure.

    The grad_fn only takes arrays as dynamic arguments:
    - initial_state: array
    - target_state: array
    - network_params: PyTree of arrays (trainable parameters)
    - channel_normalizers: array
    - key: PRNGKey

    Args:
        training_config: Training configuration (captured in closure)
        simulation_config: Simulation configuration for this problem type (captured)
        registered_variables: Registered variables for this problem type (captured)
        cnn_mhd_corrector_config: CNN corrector configuration (captured)
        base_params: Base simulation params without network_params (captured)
        noise_level: Noise level for state perturbation (captured)

    Returns:
        JIT-compiled gradient function with clean signature
    """
    # Capture all config in closure - these are immutable after creation
    if noise_level == 0.0:
        perturb_state_fn = lambda state, _: state
    else:

        def perturb_state_fn(state, key):
            return perturb_state(state=state, noise_level=noise_level, key=key)

    # Pre-configure simulation_config with corrector enabled (closure-captured)
    # Use _replace to create new NamedTuple with stable identity
    final_simulation_config = simulation_config._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )

    # Freeze closure values for JAX - ensure they're immutable and won't change
    # These are captured by reference in the grad_fn closure
    # NamedTuples are immutable by design, but we mark them as final here
    _frozen_simulation_config = final_simulation_config
    _frozen_registered_variables = registered_variables
    _frozen_base_params = base_params

    # Determine loss function type (closure-captured)
    use_norm_mse = training_config.loss_type == "norm_mse"
    # Convert physics_weights to JAX array for stable identity (not Python list!)
    # Python lists are mutable and have unstable object identity → JIT cache misses
    if training_config.physics_weights is not None:
        physics_weights = jnp.array(training_config.physics_weights, dtype=jnp.float32)
    else:
        physics_weights = None
    use_interface = training_config.use_interface

    def grad_fn_core(
        initial_state,
        target_state,
        network_params,
        channel_normalizers,
        key,
    ):
        """Compute loss and gradients for a single problem.

        All arguments are arrays or PyTrees of arrays - no config objects.
        Configuration is captured in closure for stable JIT caching.

        Args:
            initial_state: Initial simulation state array
            target_state: Target state array for loss computation
            network_params: PyTree of trainable network parameters
            channel_normalizers: Per-channel normalization factors array
            key: JAX PRNG key for perturbation

        Returns:
            Tuple of (loss_value, gradients, gradient_modulus, last_timepoint)
        """
        noisy_initial_state = perturb_state_fn(initial_state, key)

        def loss_fn(net_params):
            # Construct params with network_params injected
            # This happens inside JIT but structure is stable
            corrector_params = CNNMHDParams(network_params=net_params)
            full_params = _frozen_base_params._replace(
                cnn_mhd_corrector_params=corrector_params
            )

            results_low_res = time_integration(
                noisy_initial_state,
                _frozen_simulation_config,
                full_params,
                _frozen_registered_variables,
            )
            assert isinstance(results_low_res, SnapshotData)

            pred_final = results_low_res.states[-1]

            # Compute loss - conditional on STATIC closure variable (no kwargs dict)
            # This avoids variable PyTree structure that triggers recompilation
            if use_norm_mse:
                loss = normalized_weighted_loss(
                    pred_final,
                    target_state,
                    channel_normalizers=channel_normalizers,
                    physics_weights=physics_weights,
                    use_interface=use_interface,
                )
            else:
                loss = simple_mse_loss(pred_final, target_state)

            return loss, results_low_res.time_points[-1]

        (loss_value, last_timepoint), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(network_params)
        gradients_modulus = jnp.linalg.norm(_flatten_gradient_pytree(grads))
        return loss_value, grads, gradients_modulus, last_timepoint

    grad_fn = jax.jit(grad_fn_core)
    # Note: No static_argnames needed because:
    # - All arguments (initial_state, target_state, network_params,
    #   channel_normalizers, key) are arrays that vary dynamically
    # - All config (training_config, simulation_config, registered_variables,
    #   base_params, noise_level) is captured in closure with stable identity
    # - Closure-captured NamedTuples and JAX arrays have stable object identity

    # Log grad_fn identity for debugging cache issues
    logger.debug(f"Created grad_fn with id={id(grad_fn)}, core_id={id(grad_fn_core)}")

    return grad_fn


def precompile_grad_fns(
    h5_problem_manager: H5ProblemManager,
    training_config: TrainingConfig,
    cnn_mhd_corrector_config,
    noise_levels: Dict[str, float],
    active_problems: Optional[set] = None,
) -> Tuple[Dict[str, Callable], Dict[str, Any]]:
    """Pre-compile gradient functions for each unique problem.name.

    This function loads one representative problem for each problem type to get
    the simulation configuration and registered variables, then pre-compiles
    a JIT gradient function. These compiled functions can then be shared
    across all GPU workers without recompilation.

    Args:
        h5_problem_manager: Manager for loading problems from H5
        training_config: Training configuration
        cnn_mhd_corrector_config: CNN corrector configuration
        noise_levels: Dict mapping problem_name -> noise level
        active_problems: Optional set of problem names to compile. If None, compiles all.

    Returns:
        Tuple of (grad_fn_cache, problem_configs) where:
        - grad_fn_cache: Dict mapping problem_name -> JIT-compiled grad function
        - problem_configs: Dict mapping problem_name -> {config, params, reg_vars}
    """
    grad_fn_cache: Dict[str, Callable] = {}
    problem_configs: Dict[str, Any] = {}

    logger.info("Pre-compiling gradient functions for each problem type...")

    # Determine which problems to compile
    all_problems = set(h5_problem_manager.problem_configs.keys())
    if active_problems is not None:
        problems_to_compile = active_problems & all_problems
        skipped_problems = all_problems - active_problems
        if skipped_problems:
            logger.info(f"  Skipping unused problems: {sorted(skipped_problems)}")
        if active_problems - all_problems:
            logger.warning(
                f"  Requested problems not found in H5: {sorted(active_problems - all_problems)}"
            )
    else:
        problems_to_compile = all_problems

    for problem_name in sorted(problems_to_compile):
        logger.info(f"  Pre-compiling grad_fn for {problem_name}...")

        # Get config and params from h5_problem_manager
        config = h5_problem_manager.problem_configs[problem_name]["config"]
        params = h5_problem_manager.problem_configs[problem_name]["params"]

        # Load one problem to get shape for finalize_config
        h5_loader = h5_problem_manager.h5_loaders[problem_name]
        first_initial, _, _ = h5_loader.get_problem(0)

        # Finalize config with correct resolution
        config_lr = finalize_config(
            config._replace(num_cells=int(first_initial.shape[-1])),
            first_initial.shape,
        )
        reg_vars = get_registered_variables(config_lr)

        # Configure for training
        config_lr = config_lr._replace(
            return_snapshots=True,
            progress_bar=False,
            use_specific_snapshot_timepoints=True,
            num_snapshots=1,
            cnn_mhd_corrector_config=cnn_mhd_corrector_config,
        )
        params_lr = params._replace(snapshot_timepoints=jnp.array([params.t_end]))

        # Store configs for workers
        problem_configs[problem_name] = {
            "config": config_lr,
            "params": params_lr,
            "reg_vars": reg_vars,
        }

        # Create and cache the grad function
        noise_level = noise_levels.get(problem_name, training_config.noise_level)
        grad_fn = create_cached_grad_fn(
            training_config=training_config,
            simulation_config=config_lr,
            registered_variables=reg_vars,
            cnn_mhd_corrector_config=cnn_mhd_corrector_config,
            base_params=params_lr,
            noise_level=noise_level,
        )
        grad_fn_cache[problem_name] = grad_fn

        # Clean up loaded problem
        del first_initial
        gc.collect()

        logger.info(f"    ✓ {problem_name} grad_fn compiled")

    logger.info(f"Pre-compilation complete: {len(grad_fn_cache)} grad functions cached")
    return grad_fn_cache, problem_configs


def _warmup_on_device(
    device: jax.Device,
    grad_fn: Callable,
    initial_state,
    target_state,
    channel_normalizers,
    network_params,
    problem_name: str,
) -> Optional[str]:
    """Warm up a single grad_fn on a single device. Returns error message or None."""
    dummy_key = jax.random.PRNGKey(0)
    try:
        with jax.default_device(device):
            # Place all data on this device
            init_on_device = jax.device_put(initial_state, device)
            target_on_device = jax.device_put(target_state, device)
            normalizers_on_device = jax.device_put(channel_normalizers, device)
            key_on_device = jax.device_put(dummy_key, device)
            network_on_device = jax.device_put(network_params, device)

            # Execute grad_fn with new clean signature
            result = grad_fn(
                init_on_device,
                target_on_device,
                network_on_device,
                normalizers_on_device,
                key_on_device,
            )
            jax.block_until_ready(result)
            return None
    except Exception as e:
        return f"{problem_name} on {device}: {e}"


def warmup_grad_fn_cache(
    grad_fn_cache: Dict[str, Callable],
    problem_configs: Dict[str, Any],
    h5_problem_manager: H5ProblemManager,
    network_params,
    active_problems: Optional[set] = None,
) -> None:
    """Warm up JIT cache by executing grad functions on all devices in parallel.

    JAX JIT cache keys include the device assignment, so a function compiled
    on GPU:0 will recompile when first called on GPU:1. This function executes
    each grad_fn once on each available device to populate the cache upfront.

    Warmup is parallelized across devices using threads.

    Args:
        grad_fn_cache: Dict mapping problem_name -> JIT-compiled grad function
        problem_configs: Dict mapping problem_name -> {config, params, reg_vars}
        h5_problem_manager: Manager for loading problems from H5
        network_params: Neural network parameters (pytree)
        active_problems: Optional set of problem names to warm up. If None, warms all.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    devices = jax.devices()
    if len(devices) == 0:
        logger.info("No devices detected, cannot warm up cache")
        return

    logger.info(f"Warming up JIT cache on {len(devices)} devices (parallel)...")

    # Determine which problems to warm up
    all_problems = set(grad_fn_cache.keys())
    if active_problems is not None:
        problems_to_warmup = active_problems & all_problems
        skipped_problems = all_problems - active_problems
        if skipped_problems:
            logger.info(
                f"  Skipping warmup for unused problems: {sorted(skipped_problems)}"
            )
    else:
        problems_to_warmup = all_problems

    # Validate shape consistency across problems we're warming up (critical for JIT cache)
    expected_normalizer_shape = None
    for problem_name in sorted(problems_to_warmup):
        grad_fn = grad_fn_cache[problem_name]
        h5_loader = h5_problem_manager.h5_loaders[problem_name]
        initial_state, target_state, _ = h5_loader.get_problem(0)
        channel_normalizers = compute_channel_normalizers(target_state)

        if expected_normalizer_shape is None:
            expected_normalizer_shape = channel_normalizers.shape
            logger.info(
                f"  Expected channel_normalizers shape: {expected_normalizer_shape}"
            )
        elif channel_normalizers.shape != expected_normalizer_shape:
            raise ValueError(
                f"Channel normalizer shape mismatch for {problem_name}: "
                f"got {channel_normalizers.shape}, expected {expected_normalizer_shape}. "
                f"All problems must have same number of channels for JIT cache stability."
            )

    for problem_name in sorted(problems_to_warmup):
        grad_fn = grad_fn_cache[problem_name]
        logger.info(f"  Warming up {problem_name} on all devices...")

        # Load one problem to get proper shapes
        h5_loader = h5_problem_manager.h5_loaders[problem_name]
        initial_state, target_state, _ = h5_loader.get_problem(0)
        channel_normalizers = compute_channel_normalizers(target_state)

        # Run warmup on all devices in parallel
        errors = []
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = {
                executor.submit(
                    _warmup_on_device,
                    device,
                    grad_fn,
                    initial_state,
                    target_state,
                    channel_normalizers,
                    network_params,
                    problem_name,
                ): device
                for device in devices
            }
            for future in as_completed(futures):
                error = future.result()
                if error:
                    errors.append(error)
                    logger.warning(f"    ✗ Warmup failed: {error}")

        # Clean up problem data
        del initial_state, target_state, channel_normalizers
        gc.collect()

        if errors:
            logger.warning(f"    ⚠ {problem_name}: {len(errors)} device(s) failed")
        else:
            logger.info(f"    ✓ {problem_name} cache warmed on all devices")

    warmed_count = len(problems_to_warmup)
    total_count = len(grad_fn_cache)
    if warmed_count < total_count:
        logger.info(
            f"Cache warmup complete for {warmed_count}/{total_count} grad functions "
            f"(skipped {total_count - warmed_count} unused)"
        )
    else:
        logger.info(f"Cache warmup complete for {warmed_count} grad functions")


# ============================================================================
# TASK AND WORKER CLASSES
# ============================================================================


class CachedGradientTask:
    """Task for gradient computation using pre-compiled functions.

    This task holds problem indices for lazy loading. The actual states are
    loaded one at a time by the worker to minimize memory usage.

    Only stores the minimal data needed: problem indices and network params.
    """

    def __init__(
        self,
        problem_name: str,
        problem_indices: List[int],
        h5_problem_manager,
        network_params,
        subkeys: List,
    ):
        self.problem_name = problem_name
        self.problem_indices = problem_indices
        self.batch_size = len(problem_indices)
        self.h5_problem_manager = h5_problem_manager
        self.network_params = network_params
        self.subkeys = subkeys
        self.result = None
        self.error = None


def group_descriptors_by_problem_name(
    descriptors: List,
) -> Dict[str, List[int]]:
    """Group problem descriptors by problem_name, returning indices.

    Args:
        descriptors: List of H5ProblemDescriptor objects

    Returns:
        Dict mapping problem_name -> list of problem indices
    """
    indices_by_name: Dict[str, List[int]] = {}
    for descriptor in descriptors:
        problem_name = _descriptor_problem_name(descriptor)
        if problem_name not in indices_by_name:
            indices_by_name[problem_name] = []
        indices_by_name[problem_name].append(descriptor.problem_index)
    return indices_by_name


def assign_cached_tasks_to_devices(
    indices_by_problem: Dict[str, List[int]],
    num_devices: int,
    h5_problem_manager,
    network_params,
    key,
    threads_per_gpu: int = 1,
) -> List[List[CachedGradientTask]]:
    """Assign cached gradient tasks to devices with even load balancing.

    Since all devices share the same pre-compiled grad functions, we simply
    distribute problems evenly across devices using round-robin assignment.
    This ensures balanced load regardless of problem type.

    When threads_per_gpu > 1, tasks are split to enable parallel processing
    by multiple workers on the same device.

    Example: 40 mhd_blast + 2 ot_vortex with 2 GPUs -> 21 problems per GPU

    Args:
        indices_by_problem: Dict mapping problem_name -> list of problem indices
        num_devices: Number of available devices
        h5_problem_manager: Manager for loading problems from H5
        network_params: Neural network parameters (pytree)
        key: JAX PRNG key
        threads_per_gpu: Number of worker threads per GPU. Tasks will be split
                        to provide at least this many tasks per device when possible.

    Returns:
        List of length num_devices, where each element is a list of
        CachedGradientTask for that device.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be > 0, got {num_devices}")

    if not indices_by_problem:
        return [[] for _ in range(num_devices)]
    # Flatten all problems into a single list of (problem_name, problem_index)
    all_problems: List[Tuple[str, int]] = []
    for problem_name, indices in indices_by_problem.items():
        for idx in indices:
            all_problems.append((problem_name, idx))

    # Shuffle to avoid any ordering bias
    random.shuffle(all_problems)

    # Calculate fair distribution
    total_problems = len(all_problems)
    base_count = total_problems // num_devices
    remainder = total_problems % num_devices

    # Distribute problems evenly across devices
    device_problems: List[List[Tuple[str, int]]] = [[] for _ in range(num_devices)]
    idx = 0
    for device_idx in range(num_devices):
        # Give one extra problem to first 'remainder' devices
        count = base_count + (1 if device_idx < remainder else 0)
        device_problems[device_idx] = all_problems[idx : idx + count]
        idx += count

    # Group problems by problem_name within each device for efficient task creation
    device_assignments: List[Dict[str, List[int]]] = [{} for _ in range(num_devices)]
    for device_idx, problems in enumerate(device_problems):
        for problem_name, problem_idx in problems:
            if problem_name not in device_assignments[device_idx]:
                device_assignments[device_idx][problem_name] = []
            device_assignments[device_idx][problem_name].append(problem_idx)

    # Convert assignments to CachedGradientTask objects
    # Split tasks to enable parallel processing when threads_per_gpu > 1
    result: List[List[CachedGradientTask]] = [[] for _ in range(num_devices)]

    for device_idx, assignments in enumerate(device_assignments):
        # Calculate total problems on this device for splitting
        total_on_device = sum(len(indices) for indices in assignments.values())

        # Target: create enough tasks for threads_per_gpu workers
        # Each task should have roughly equal work
        target_tasks = max(threads_per_gpu, len(assignments))
        target_problems_per_task = max(1, total_on_device // target_tasks)

        for problem_name, indices in assignments.items():
            # Split this problem's indices into multiple tasks if beneficial
            # Only split if we have more problems than target and threads_per_gpu > 1
            if threads_per_gpu > 1 and len(indices) > target_problems_per_task:
                # Split into chunks of approximately target_problems_per_task
                num_chunks = min(
                    threads_per_gpu,  # Don't create more tasks than threads
                    max(
                        1,
                        (len(indices) + target_problems_per_task - 1)
                        // target_problems_per_task,
                    ),
                )
                chunk_size = (len(indices) + num_chunks - 1) // num_chunks

                for chunk_idx in range(num_chunks):
                    start = chunk_idx * chunk_size
                    end = min(start + chunk_size, len(indices))
                    chunk_indices = indices[start:end]

                    if not chunk_indices:
                        continue

                    # Split key for this task
                    key, subkey = jax.random.split(key)
                    # Keep as JAX array - do NOT convert to Python list!
                    subkeys = jax.random.split(subkey, len(chunk_indices))

                    task = CachedGradientTask(
                        problem_name=problem_name,
                        problem_indices=chunk_indices,
                        h5_problem_manager=h5_problem_manager,
                        network_params=network_params,
                        subkeys=subkeys,
                    )
                    result[device_idx].append(task)
            else:
                # Keep as single task (original behavior)
                key, subkey = jax.random.split(key)
                # Keep as JAX array - do NOT convert to Python list!
                # Python list creates new objects on indexing → JIT cache instability
                subkeys = jax.random.split(subkey, len(indices))  # Shape: (N, 2)

                task = CachedGradientTask(
                    problem_name=problem_name,
                    problem_indices=indices,
                    h5_problem_manager=h5_problem_manager,
                    network_params=network_params,
                    subkeys=subkeys,
                )
                result[device_idx].append(task)

    # Log distribution
    for device_idx in range(num_devices):
        total = sum(len(t.problem_indices) for t in result[device_idx])
        breakdown = ", ".join(
            f"{t.problem_name}:{len(t.problem_indices)}" for t in result[device_idx]
        )
        logger.debug(f"Device {device_idx}: {total} problems [{breakdown}]")

    return result


# ============================================================================
# WORKER FUNCTION
# ============================================================================


def compute_cached_gradient_worker(
    worker_id: int,
    device: jax.Device,
    task_queue: Queue,
    result_queue: Queue,
    grad_fn_cache: Dict[str, Callable],
    problem_configs: Dict[str, Any],
    worker_monitor: Optional[Dict[str, Any]] = None,
) -> None:
    """Worker using pre-compiled gradient functions with clean signature.

    This worker processes CachedGradientTask objects using pre-compiled grad
    functions. The grad_fn takes only arrays as arguments:
    - initial_state, target_state, network_params, channel_normalizers, key

    Channel normalizers are computed OUTSIDE the JIT function to avoid recompilation.

    Args:
        worker_id: Unique identifier for this worker
        device: JAX device to use for computation
        task_queue: Queue to receive tasks from
        result_queue: Queue to send results to
        grad_fn_cache: Pre-compiled gradient functions (read-only)
        problem_configs: Problem configurations (config, params, reg_vars) - unused now
        worker_monitor: Shared worker activity tracker for concurrency diagnostics
    """
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_start_time = time.perf_counter()
        if worker_monitor is not None:
            with worker_monitor["lock"]:
                worker_monitor["active_workers"] += 1
                worker_monitor["max_active_workers"] = max(
                    worker_monitor["max_active_workers"],
                    worker_monitor["active_workers"],
                )

        try:
            # Get pre-compiled grad_fn (no compilation happens here)
            grad_fn = grad_fn_cache[task.problem_name]
            logger.debug(
                f"Worker {worker_id} using grad_fn id={id(grad_fn)} "
                f"for problem={task.problem_name}"
            )

            # Get H5 loader for this problem type
            h5_loader = task.h5_problem_manager.h5_loaders[task.problem_name]

            # Process problems one at a time to minimize memory usage
            individual_grads = []
            per_problem_results = []

            with jax.default_device(device):
                # Keep model params on device for all problems in this task.
                network_on_device = jax.device_put(task.network_params, device)
                for i, problem_idx in enumerate(task.problem_indices):
                    # Load single problem from H5
                    initial_state, target_state, hyperparams = h5_loader.get_problem(
                        problem_idx
                    )

                    # Compute channel normalizers OUTSIDE JIT
                    channel_normalizers = compute_channel_normalizers(target_state)

                    # Explicitly place data on device (like warmup does)
                    init_on_device = jax.device_put(initial_state, device)
                    target_on_device = jax.device_put(target_state, device)
                    normalizers_on_device = jax.device_put(channel_normalizers, device)
                    key_on_device = jax.device_put(task.subkeys[i], device)

                    # Log shapes for debugging
                    if i == 0:  # Only log first problem in batch
                        logger.debug(
                            f"Worker {worker_id} problem {problem_idx}: "
                            f"initial_shape={initial_state.shape}, "
                            f"target_shape={target_state.shape}, "
                            f"normalizers_shape={channel_normalizers.shape}, "
                            f"network_params_type={type(task.network_params)}"
                        )

                    # Compute gradient using pre-compiled function with clean signature
                    # Only arrays passed: initial_state, target_state, network_params,
                    #                     channel_normalizers, key
                    loss_val, grads, grad_mod, last_t = grad_fn(
                        init_on_device,
                        target_on_device,
                        network_on_device,
                        normalizers_on_device,
                        key_on_device,
                    )
                    # Pull all outputs to host in one synchronization point.
                    loss_val, grads, grad_mod, last_t = jax.device_get(
                        (loss_val, grads, grad_mod, last_t)
                    )

                    # Validate output — skip this problem on NaN, don't crash worker
                    try:
                        validate_output(
                            last_t=float(last_t),
                            gradients_mod=float(grad_mod),
                            problem_name=task.problem_name,
                        )
                    except ValueError as nan_err:
                        logger.warning(
                            "Skipping problem %s (idx=%d) in worker %d: %s",
                            task.problem_name,
                            problem_idx,
                            worker_id,
                            nan_err,
                        )
                        del initial_state, target_state, grads
                        if (i + 1) % 8 == 0:
                            gc.collect()
                        continue

                    # Keep gradients on host to avoid GPU accumulation/OOM.
                    individual_grads.append(grads)
                    per_problem_results.append(
                        {
                            "loss": float(loss_val),
                            "grad_mod": float(grad_mod),
                            "last_t": float(last_t),
                            "problem_name": task.problem_name,
                            "problem_index": problem_idx,
                            "hyperparams": hyperparams,
                        }
                    )

                    # Explicit cleanup to free GPU memory
                    del initial_state, target_state, grads
                    if (i + 1) % 8 == 0:
                        gc.collect()

            if not per_problem_results:
                raise ValueError(
                    f"All problems in worker {worker_id} produced NaN — "
                    f"no valid gradients from this batch."
                )

            # Compute averages for logging
            avg_loss = sum(r["loss"] for r in per_problem_results) / len(
                per_problem_results
            )
            avg_grad_mod = sum(r["grad_mod"] for r in per_problem_results) / len(
                per_problem_results
            )

            result_queue.put(
                (
                    id(task),
                    {
                        "cached": True,
                        "problem_name": task.problem_name,
                        "batch_size": task.batch_size,
                        "individual_grads": individual_grads,
                        "avg_loss": avg_loss,
                        "avg_grad_mod": avg_grad_mod,
                        "per_problem_results": per_problem_results,
                        "worker_id": worker_id,
                        "device": str(device),
                    },
                )
            )
        except Exception as e:
            error_kind = classify_worker_error(e)
            error_traceback = traceback.format_exc()
            print_exception_summary(
                context=(
                    f"worker={worker_id} device={device} problem={task.problem_name} "
                    f"batch_size={task.batch_size} kind={error_kind}"
                ),
                exc=e,
            )
            print_exception_traceback(
                context=(
                    f"worker={worker_id} device={device} problem={task.problem_name} "
                    f"batch_size={task.batch_size} kind={error_kind}"
                ),
                traceback_text=error_traceback,
            )
            if error_kind == "nan":
                logger.warning(
                    "Cached gradient worker %s skipped %s (batch_size=%d): %s",
                    worker_id,
                    task.problem_name,
                    task.batch_size,
                    e,
                )
            elif error_kind == "oom":
                logger.exception(
                    "Cached gradient worker %s hit OOM for %s (batch_size=%d)",
                    worker_id,
                    task.problem_name,
                    task.batch_size,
                )
            else:
                logger.exception(
                    "Cached gradient worker %s failed for %s (batch_size=%d)",
                    worker_id,
                    task.problem_name,
                    task.batch_size,
                )
            result_queue.put(
                (
                    id(task),
                    {
                        "error": str(e),
                        "error_kind": error_kind,
                        "error_type": type(e).__name__,
                        "error_traceback": error_traceback,
                        "cached": True,
                        "worker_id": worker_id,
                        "device": str(device),
                        "problem_name": task.problem_name,
                        "batch_size": task.batch_size,
                    },
                )
            )
        finally:
            if worker_monitor is not None:
                with worker_monitor["lock"]:
                    worker_monitor["active_workers"] -= 1
            logger.debug(
                "Worker %d finished %s (batch_size=%d) in %.3fs",
                worker_id,
                task.problem_name,
                task.batch_size,
                time.perf_counter() - task_start_time,
            )


# ============================================================================
# DISTRIBUTION FUNCTION
# ============================================================================


def _select_training_devices(num_gpus: Optional[int]) -> List[jax.Device]:
    """Resolve devices to use for threaded gradient workers."""
    try:
        gpu_devices = list(jax.devices("gpu"))
    except RuntimeError:
        gpu_devices = []

    all_devices = gpu_devices if gpu_devices else list(jax.devices())
    if not all_devices:
        raise RuntimeError("No JAX devices available for training")

    if num_gpus is None:
        return all_devices

    if num_gpus <= 0:
        raise ValueError(f"num_gpus must be positive, got {num_gpus}")

    if num_gpus > len(all_devices):
        raise ValueError(
            f"Requested {num_gpus} GPUs/devices, but only {len(all_devices)} available"
        )

    return all_devices[:num_gpus]


def distribute_cached_gradient_computation(
    device_tasks: List[List[CachedGradientTask]],
    devices: List[jax.Device],
    threads_per_gpu: int,
    grad_fn_cache: Dict[str, Callable],
    problem_configs: Dict[str, Any],
) -> Tuple[Dict[int, Dict], Dict[str, List[float]], int]:
    """Distribute cached gradient computation across devices with multiple threads per GPU.

    Each device gets a list of CachedGradientTask objects. Multiple worker threads
    per device pull tasks from a shared queue, using pre-compiled gradient functions
    from grad_fn_cache.

    Args:
        device_tasks: List of length num_devices, where each element
                      is a list of CachedGradientTask.
        devices: Resolved devices used for training.
        threads_per_gpu: Number of worker threads per GPU/device
        grad_fn_cache: Pre-compiled gradient functions (read-only)
        problem_configs: Problem configurations (config, params, reg_vars)

    Returns:
        Tuple of (results dict, step_losses dict, total_problem_count)
    """
    if threads_per_gpu <= 0:
        raise ValueError(f"threads_per_gpu must be > 0, got {threads_per_gpu}")

    # Count total tasks and problems
    total_tasks = sum(len(tasks) for tasks in device_tasks)
    total_problems = sum(
        sum(task.batch_size for task in tasks) for tasks in device_tasks
    )

    if total_tasks == 0:
        return {}, {}, 0

    active_device_indices = [idx for idx, tasks in enumerate(device_tasks) if tasks]

    # Calculate total workers
    max_workers = sum(
        min(threads_per_gpu, len(device_tasks[idx])) for idx in active_device_indices
    )

    # Log device assignments
    logger.info(
        "Launching %d cached gradient workers across %d/%d devices "
        "(threads_per_gpu=%d, total: %d tasks, %d problems)",
        max_workers,
        len(active_device_indices),
        len(devices),
        threads_per_gpu,
        total_tasks,
        total_problems,
    )
    for device_idx in active_device_indices:
        tasks = device_tasks[device_idx]
        task_summary = ", ".join(f"{t.problem_name}:{t.batch_size}" for t in tasks)
        problems_on_device = sum(t.batch_size for t in tasks)
        workers_for_device = min(threads_per_gpu, len(tasks))
        if workers_for_device < threads_per_gpu:
            logger.info(
                "  Device %s requested %d worker thread(s), but only %d task(s) "
                "are available. Spawning %d worker(s).",
                devices[device_idx],
                threads_per_gpu,
                len(tasks),
                workers_for_device,
            )
        logger.info(
            "  Device %s: %d worker(s), %d task(s), %d problems [%s]",
            devices[device_idx],
            workers_for_device,
            len(tasks),
            problems_on_device,
            task_summary,
        )

    result_queue: Queue = Queue()
    threads = []
    worker_monitor: Dict[str, Any] = {
        "lock": threading.Lock(),
        "active_workers": 0,
        "max_active_workers": 0,
    }

    # Spawn multiple workers per device based on threads_per_gpu
    for device_idx in active_device_indices:
        device = devices[device_idx]
        tasks = device_tasks[device_idx]

        # Calculate how many workers to spawn for this device
        workers_for_device = min(threads_per_gpu, len(tasks))

        # Create shared task queue for this device
        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)
        # Add sentinel values for each worker
        for _ in range(workers_for_device):
            task_queue.put(None)

        # Spawn workers for this device
        for _ in range(workers_for_device):
            worker_id = len(threads)
            thread = threading.Thread(
                target=compute_cached_gradient_worker,
                args=(
                    worker_id,
                    device,
                    task_queue,
                    result_queue,
                    grad_fn_cache,
                    problem_configs,
                    worker_monitor,
                ),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

    # Collect results
    results: Dict[int, Dict] = {}
    step_losses: Dict[str, List[float]] = {}
    first_error: Optional[Dict[str, Any]] = None

    num_errors = 0
    num_nan_errors = 0
    num_oom_errors = 0
    num_other_errors = 0
    for _ in range(total_tasks):
        task_id, result = result_queue.get()
        if "error" in result:
            error_kind = result.get("error_kind", "other")
            if error_kind == "nan":
                num_nan_errors += 1
            elif error_kind == "oom":
                num_oom_errors += 1
            else:
                num_other_errors += 1

            print(
                "[WORKER-ERROR] "
                f"kind={error_kind} worker={result.get('worker_id', 'unknown')} "
                f"device={result.get('device', 'unknown')} "
                f"problem={result.get('problem_name', 'unknown')} "
                f"batch_size={result.get('batch_size', 'unknown')} "
                f"error={result.get('error')}",
                file=sys.stderr,
                flush=True,
            )
            print_exception_traceback(
                context=(
                    f"worker={result.get('worker_id', 'unknown')} "
                    f"problem={result.get('problem_name', 'unknown')} kind={error_kind}"
                ),
                traceback_text=result.get("error_traceback", ""),
            )
            logger.warning(
                "Skipping failed worker (kind=%s, worker=%s, device=%s, problem=%s, batch_size=%s): %s",
                error_kind,
                result.get("worker_id", "unknown"),
                result.get("device", "unknown"),
                result.get("problem_name", "unknown"),
                result.get("batch_size", "unknown"),
                result.get("error"),
            )
            if first_error is None:
                first_error = result
            num_errors += 1
            continue
        else:
            results[task_id] = result
            # Extract per-problem losses for tracking
            problem_name = result["problem_name"]
            for per_problem in result.get("per_problem_results", []):
                step_losses.setdefault(problem_name, []).append(per_problem["loss"])

    for thread in threads:
        thread.join()
    peak_active_workers = worker_monitor["max_active_workers"]
    logger.info(
        "Observed peak concurrent active workers: %d/%d",
        peak_active_workers,
        max_workers,
    )
    if max_workers > 1 and peak_active_workers <= 1:
        logger.warning(
            "Worker execution on this epoch was effectively sequential "
            "(peak_active_workers=%d).",
            peak_active_workers,
        )

    if not results:
        if first_error is not None:
            print(
                "[FATAL] All gradient workers failed this epoch. "
                f"first_error_kind={first_error.get('error_kind', 'unknown')} "
                f"first_error={first_error.get('error', 'unknown')}",
                file=sys.stderr,
                flush=True,
            )
            print_exception_traceback(
                context=(
                    f"first_failed_worker={first_error.get('worker_id', 'unknown')} "
                    f"problem={first_error.get('problem_name', 'unknown')}"
                ),
                traceback_text=first_error.get("error_traceback", ""),
            )
        raise RuntimeError(
            "All gradient workers failed this epoch — cannot update model. "
            f"First error: {first_error['error'] if first_error else 'unknown'} "
            f"(kind={first_error.get('error_kind', 'unknown') if first_error else 'unknown'})"
        )

    if num_errors > 0:
        if num_nan_errors > 0:
            logger.warning(
                "%d/%d workers had NaN-only batches and were skipped.",
                num_nan_errors,
                total_tasks,
            )
        if num_oom_errors > 0:
            logger.error(
                "%d/%d workers hit OOM and were skipped.",
                num_oom_errors,
                total_tasks,
            )
        if num_other_errors > 0:
            logger.warning(
                "%d/%d workers failed with non-NaN, non-OOM errors and were skipped.",
                num_other_errors,
                total_tasks,
            )
        logger.warning(
            "Proceeding with %d successful workers out of %d total tasks.",
            len(results),
            total_tasks,
        )

    return results, step_losses, total_problems


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def training_multigpu_cached(
    model_manager: ModelManager,
    h5_problem_manager: H5ProblemManager,
    multigpu_config: MultiGPUTrainingConfig,
    training_config: TrainingConfig,
    model_name: str,
    load_model: bool = False,
    load_model_nan: bool = False,
    description: str = "",
    h5_param_filters: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
):
    """Multi-GPU training with pre-compiled gradient functions.

    This function pre-compiles gradient functions for each problem type ONCE
    before training, then shares them across all GPU workers. Channel normalizers
    are computed outside JIT to avoid recompilation. All individual gradients
    from all GPUs are collected for conFIG (not averaged per GPU).

    Args:
        model_manager: Manager for saving checkpoints and metadata
        h5_problem_manager: Manager for loading training pairs from H5 datasets
        multigpu_config: Configuration for multi-GPU training
        training_config: Base training configuration
        model_name: Name of the model
        load_model: Whether to load existing model
        load_model_nan: Whether to load model from NaN checkpoint
        description: Description of the training run
        h5_param_filters: Parameter filters used during dataset loading
                         (for saving to config)

    Returns:
        Tuple of (best_params, neural_net_static)
    """

    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Multi-GPU Training with Cached Compilation: {model_name}")
    logger.info(f"Config: {multigpu_config}")
    logger.info(f"Epochs: {total_epochs} | GPUs: {multigpu_config.num_gpus}")
    logger.info("=" * 80 + "\n")
    if (
        multigpu_config.num_gpus is not None
        and multigpu_config.num_gpus != STARTUP_NUM_GPUS
    ):
        logger.warning(
            "CLI/runtime GPU setup mismatch: startup used %d GPU(s), config requests %d. "
            "Use --num-gpus at launch to guarantee full alignment.",
            STARTUP_NUM_GPUS,
            multigpu_config.num_gpus,
        )
    training_devices = _select_training_devices(multigpu_config.num_gpus)
    logger.info(
        "Resolved %d training device(s): %s",
        len(training_devices),
        ", ".join(str(device) for device in training_devices),
    )

    # Load lightweight problem descriptors from H5 (lazy, no tensors yet)
    problem_descriptors = h5_problem_manager.get_problem_descriptors()
    if not problem_descriptors:
        raise ValueError("No H5 problem descriptors found for training")

    # Initialize sampler
    sampler = ProblemSampler(
        problem_descriptors=problem_descriptors,
        problem_counts=multigpu_config.problem_counts,
        use_fixed_subset=multigpu_config.fixed_problems,
        subset_size=multigpu_config.subset_size,
    )

    # Materialize one pair to derive architecture
    bootstrap_problem_name = next(iter(multigpu_config.problem_counts.keys()))
    bootstrap_descriptor = sampler.problem_pairs_map[bootstrap_problem_name][0]
    first_pair = h5_problem_manager.get_training_pairs_for_descriptors(
        [bootstrap_descriptor]
    )[0]
    registered_variables = first_pair.lr_bundle.reg_vars

    # Initialize model
    _model_kwargs = dict(
        in_channels=registered_variables.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
        normalize_input=training_config.normalize_input,
    )
    if training_config.use_film_corrector:
        model = FiLMCorrectorCNN(**_model_kwargs)
    else:
        model = CorrectorCNN(**_model_kwargs)
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

    # ========================================================================
    # PRE-COMPILATION PHASE
    # ========================================================================
    logger.info("\n" + "-" * 40)
    logger.info("PRE-COMPILATION PHASE")
    logger.info("-" * 40)

    # Only compile and warm up problems that are actually used in training
    active_problem_names = set(multigpu_config.problem_counts.keys())
    logger.info(f"Active problems for this training: {sorted(active_problem_names)}")

    grad_fn_cache, problem_configs = precompile_grad_fns(
        h5_problem_manager=h5_problem_manager,
        training_config=training_config,
        cnn_mhd_corrector_config=cnn_mhd_corrector_config,
        noise_levels=PROBLEM_NOISE_LEVELS,
        active_problems=active_problem_names,
    )

    # Warm up JIT cache on all devices to avoid per-device recompilation
    warmup_grad_fn_cache(
        grad_fn_cache=grad_fn_cache,
        problem_configs=problem_configs,
        h5_problem_manager=h5_problem_manager,
        network_params=neural_net_params,
        active_problems=active_problem_names,
    )

    logger.info("-" * 40 + "\n")
    # ========================================================================

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
        max_patience=multigpu_config.early_stopping_patience,
        use_early_stopper=multigpu_config.early_stopping_enabled,
    )

    logger.info(f"Total available descriptors: {len(problem_descriptors)}")
    logger.info(f"Problem counts: {multigpu_config.problem_counts}")

    successful_training = True

    multigpu_config_path = (
        model_manager.base_dir / model_manager.model_name / "multigpu_config.json"
    )
    with open(multigpu_config_path, "w") as f:
        json.dump(asdict(multigpu_config), f, indent=2)

    try:
        key = jax.random.PRNGKey(112)
        early_stopper.reset_patience()

        # Main training loop
        for epoch in range(training_config.epochs_per_time[0]):
            key, _ = jax.random.split(key)
            start_time_epoch = timer()

            # Sample descriptors for this epoch (lightweight, no tensors loaded yet)
            sampled_descriptors = sampler.sample_epoch()

            logger.info(
                f"\nEpoch {epoch + 1}/{training_config.epochs_per_time[0]} | "
                f"Sampled {len(sampled_descriptors)} problems"
            )

            # Group descriptors by problem_name (no materialization)
            indices_by_problem = group_descriptors_by_problem_name(sampled_descriptors)

            # Split key for task creation
            key, batch_key = jax.random.split(key)

            # Assign tasks to devices with load balancing
            device_tasks = assign_cached_tasks_to_devices(
                indices_by_problem=indices_by_problem,
                num_devices=len(training_devices),
                h5_problem_manager=h5_problem_manager,
                network_params=trained_params,
                key=batch_key,
                threads_per_gpu=multigpu_config.threads_per_gpu,
            )

            # Compute gradients using pre-compiled functions
            results, step_losses, total_problems = (
                distribute_cached_gradient_computation(
                    device_tasks,
                    devices=training_devices,
                    threads_per_gpu=multigpu_config.threads_per_gpu,
                    grad_fn_cache=grad_fn_cache,
                    problem_configs=problem_configs,
                )
            )

            # ================================================================
            # COLLECT ALL INDIVIDUAL GRADIENTS FOR conFIG
            # ================================================================
            all_individual_grads = []
            all_problem_names = []
            total_grad_mod = 0.0
            all_losses = []

            for device_task_list in device_tasks:
                for task in device_task_list:
                    result = results[id(task)]
                    # Collect ALL individual gradients (not averaged) and their problem names
                    all_individual_grads.extend(result["individual_grads"])
                    all_problem_names.extend([result["problem_name"]] * len(result["individual_grads"]))
                    total_grad_mod += sum(
                        pr["grad_mod"] for pr in result["per_problem_results"]
                    )

                    # Log per-task info
                    logger.info(
                        f"  {result['problem_name']} (n={result['batch_size']}): "
                        f"avg_loss={result['avg_loss']:.6f}, avg_grad_mod={result['avg_grad_mod']:.3f}"
                    )

                    # Collect all losses for averaging
                    for per_problem in result.get("per_problem_results", []):
                        all_losses.append(per_problem["loss"])

            # Move CPU-side gradients back to GPU for aggregation.
            # Gradients were offloaded to numpy in the worker to avoid OOM.
            device_grads = [jax.device_put(g) for g in all_individual_grads]

            # Apply per-problem gradient processing if enabled
            if multigpu_config.normalize_per_problem:
                device_grads, norm_stats = normalize_gradients_per_problem(
                    device_grads, all_problem_names
                )
            elif multigpu_config.clip_per_problem:
                device_grads, norm_stats = clip_gradients_per_problem(
                    device_grads, all_problem_names, max_norm=1.0
                )

            # Apply conFIG to ALL individual gradients
            if multigpu_config.use_config_gradient:
                start_time_config = timer()
                grads = conFIG(device_grads, use_least_square=True)
                logger.info(
                    f"conFIG gradient averaging ({len(device_grads)} grads) "
                    f"took {timer() - start_time_config:.3f}s"
                )
            else:
                # Simple averaging of all individual gradients
                num_grads = len(device_grads)
                grads = jax.tree.map(
                    lambda *gs: sum(gs) / num_grads, *device_grads
                )

            # Gradient debug analysis (after first epoch and every 5 epochs)
            if multigpu_config.grads_debug and (
                (epoch + 1) == 1 or (epoch + 1) % 5 == 0
            ):
                debug_dir = model_manager.base_dir / model_manager.model_name / "debug"
                analyze_gradients_debug(
                    all_individual_grads=all_individual_grads,
                    epoch=epoch + 1,
                    debug_dir=debug_dir,
                )

            # Optimizer update
            updates, opt_state = optimizer.update(grads, opt_state, trained_params)
            trained_params = eqx.apply_updates(trained_params, updates)

            # Track losses
            avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
            avg_grad_mod = (
                total_grad_mod / total_problems if total_problems > 0 else 0.0
            )
            losses_avg.append(avg_loss)

            for problem_name, losses in step_losses.items():
                losses_per_problem.setdefault(problem_name, []).extend(losses)

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

            # Save checkpoint every epoch (or per interval)
            if multigpu_config.save_every_epoch:
                checkpoint_name = (
                    "checkpoint_latest"
                    if multigpu_config.checkpoint_override
                    else f"checkpoint_{epoch + 1:04d}"
                )
                model_manager.save_checkpoint(
                    trained_params, epoch=epoch + 1, name=checkpoint_name
                )
            elif (epoch + 1) % multigpu_config.checkpoint_interval == 0:
                model_manager.save_checkpoint(trained_params, epoch=epoch + 1)

            epoch_time = timer() - start_time_epoch
            logger.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.6f}, "
                f"grad_mod={avg_grad_mod:.3f}, time={epoch_time:.2f}s"
            )

        successful_training = True

    except Exception as e:
        print_exception_traceback("training_multigpu_cached", traceback.format_exc())
        print_exception_summary("training_multigpu_cached", e)
        logger.error(f"Training failed with error: {e}")
        failure_checkpoint_name = (
            "model_params_OOM.eqx"
            if is_probable_oom_error(e)
            else "model_params_NAN.eqx"
        )
        model_manager.save_model_params(trained_params, failure_checkpoint_name)
        successful_training = False
        raise

    training_time = timer() - start_time

    if successful_training:
        logger.info(
            f"\n✓ Training Complete!"
            f"\n  Final Loss: {losses_avg[-1]:.6f}"
            f"\n  Best Loss: {best_loss:.6f} (epoch {best_epoch})"
            f"\n  Total Time: {training_time:.2f}s"
        )
        model_manager.save_model_params(best_params)

    # Save losses and metadata
    model_manager.save_losses(losses_avg)

    losses_save = {"avg": np.array(losses_avg)}
    for name, losses in losses_per_problem.items():
        losses_save[name] = np.array(losses)

    losses_path = (
        model_manager.base_dir / model_manager.model_name / "losses_per_problem.npz"
    )
    np.savez(losses_path, **losses_save, allow_pickle=True)
    logger.info(f"Losses saved to {losses_path}")

    # Save best model parameters
    model_manager.save_model_params(best_params, "best_model_params.eqx")

    # Save metadata
    metadata = ModelMetadata(
        model_name=model_name,
        created_at=datetime.datetime.now().isoformat(),
        total_epochs=training_config.epochs_per_time[0],
        final_epoch=len(losses_avg),
        final_loss=float(losses_avg[-1]) if losses_avg else None,
        best_loss=best_loss,
        training_time_seconds=training_time,
        succesful_training=successful_training,
        early_stopped=early_stopper.early_stopped
        if multigpu_config.early_stopping_enabled
        else None,
        notes=description,
    )
    model_manager.save_metadata(metadata)

    # Save multigpu config
    multigpu_config_path = (
        model_manager.base_dir / model_manager.model_name / "multigpu_config.json"
    )
    with open(multigpu_config_path, "w") as f:
        json.dump(asdict(multigpu_config), f, indent=2)
    logger.info(f"MultiGPU config saved to {multigpu_config_path}")

    # Save param filters if they were used
    if h5_param_filters:
        param_filter_path = (
            model_manager.base_dir / model_manager.model_name / "param_filter_config.json"
        )
        param_filter_config = {
            "param_filters": [
                {
                    "problem_name": problem_name,
                    "filters": {
                        param_name: {"min": min_val, "max": max_val}
                        for param_name, (min_val, max_val) in filters.items()
                    }
                }
                for problem_name, filters in h5_param_filters.items()
            ],
            "notes": "Parameter filters applied during H5 dataset loading"
        }
        with open(param_filter_path, "w") as f:
            json.dump(param_filter_config, f, indent=2)
        logger.info(f"Param filter config saved to {param_filter_path}")

    return best_params, neural_net_static


# ============================================================================
# ENTRY POINT
# ============================================================================


def train_with_cached_compilation(
    h5_file_paths: Dict[str, str],
    h5_problem_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
    h5_param_filters: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    multigpu_config: Optional[MultiGPUTrainingConfig] = None,
    model_name: Optional[str] = None,
    load_existing: bool = False,
    load_existing_nan: bool = False,
    description: str = "",
    **training_config_overrides,
):
    """Main entry point for H5-based multi-GPU training with cached compilation.

    Args:
        h5_file_paths: Dict mapping problem names to h5 file paths
        h5_problem_ranges: Dict mapping problem names to (start, end) index ranges.
                          If None, uses full range for all problems.
                          Ignored for any problem covered by h5_param_filters.
        h5_param_filters: Dict mapping problem names to scalar param filters,
                          e.g. {"mhd_blast": {"B0": (5.0, 15.0)}}.
                          When provided for a problem, overrides h5_problem_ranges.
        multigpu_config: Multi-GPU training configuration. If None, uses defaults.
        model_name: Name of the model (auto-generated if None)
        load_existing: Whether to load existing model
        load_existing_nan: Whether to load from NaN checkpoint
        description: Description of the training run
        **training_config_overrides: Overrides for TrainingConfig fields

    Returns:
        Tuple of (best_params, neural_net_static)
    """
    # Default multigpu_config if not provided
    if multigpu_config is None:
        multigpu_config = MultiGPUTrainingConfig(
            num_gpus=STARTUP_NUM_GPUS,
            problem_counts=DEFAULT_PROBLEM_COUNTS,
        )

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

        # Save training config for reproducibility
        model_manager.save_training_config(training_config)

    # Create H5 problem manager
    h5_problem_manager = H5ProblemManager(
        h5_file_paths=h5_file_paths,
        training_config=training_config,
        problem_ranges=h5_problem_ranges,
        param_filters=h5_param_filters,
    )

    return training_multigpu_cached(
        model_manager=model_manager,
        h5_problem_manager=h5_problem_manager,
        multigpu_config=multigpu_config,
        training_config=training_config,
        model_name=model_name,
        load_model=load_existing,
        load_model_nan=load_existing_nan,
        description=description,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-GPU training with cached compilation for solver-in-the-loop correction"
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name of the model (auto-generated if not provided)",
    )

    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")

    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
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
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Early stopping patience",
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
        "--model-scale",
        type=float,
        default=0.03,
        help="Model initialization scale (default: 0.03). Larger values may help with small gradients.",
    )

    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of the training run",
    )

    parser.add_argument(
        "--threads-per-gpu",
        type=int,
        default=1,
        help="Number of worker threads per GPU for parallel gradient computation",
    )

    parser.add_argument(
        "--fixed-problems",
        action="store_true",
        help="Train on fixed equally-spaced subset of problems (excludes extrema) instead of random sampling",
    )

    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help=(
            "Per-epoch sample size when using --fixed-problems. "
            "Randomly samples this many problems from the fixed pool each epoch. "
            "Requires --fixed-problems to be enabled. "
            "Example: --problem mhd_blast:120 --fixed-problems --subset 10 "
            "creates a pool of 120 problems and samples 10 per epoch."
        ),
    )

    parser.add_argument(
        "--load-existing",
        default=False,
        help="Load existing model weights",
    )

    parser.add_argument(
        "--grads-debug",
        action="store_true",
        help="Enable gradient analysis debug mode (analyzes gradients after epoch 1 and every 5 epochs)",
    )

    parser.add_argument(
        "--peak-lr",
        type=float,
        default=1.5e-4,
        help="Peak learning rate for warmup-cosine schedule (default: 1.5e-4)",
    )

    parser.add_argument(
        "--end-lr",
        type=float,
        default=5e-5,
        help="End learning rate for cosine decay (default: 5e-5)",
    )

    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.2,
        help="Fraction of epochs used for LR warmup (default: 0.2)",
    )

    parser.add_argument(
        "--loss-type",
        type=str,
        default="simple_mse_loss",
        choices=["simple_mse_loss", "norm_mse"],
        help="Loss function type (default: simple_mse_loss)",
    )

    parser.add_argument(
        "--normalize-input",
        action="store_true",
        default=False,
        help="Normalize model input by self-consistent Alfvénic scales before each forward pass",
    )

    parser.add_argument(
        "--use-film",
        action="store_true",
        default=False,
        help="Use FiLMCorrectorCNN (adds β and M_A channels + FiLM conditioning per hidden layer)",
    )

    parser.add_argument(
        "--param-filter",
        type=str,
        action="append",
        dest="param_filters",
        help=(
            "Filter problems by scalar hyperparameter value range. "
            "Format: 'problem_name:param_name:min_val:max_val'. "
            "Can be repeated for multiple filters. "
            "Example: --param-filter mhd_blast:B0:5.0:15.0"
        ),
    )

    parser.add_argument(
        "--normalize-per-problem",
        action="store_true",
        default=False,
        help=(
            "Normalize gradients per individual problem before aggregation. "
            "Each problem's gradients are divided by their L2 norm before being "
            "collected for conFIG or averaging."
        ),
    )

    parser.add_argument(
        "--clip-per-problem",
        action="store_true",
        default=False,
        help=(
            "Clip gradients per individual problem to L2 norm of 1.0 before aggregation. "
            "Each problem's gradients are clipped independently to unit norm before being "
            "collected for conFIG or averaging. Incompatible with --normalize-per-problem."
        ),
    )

    args = parser.parse_args()

    # Validate that normalize-per-problem and clip-per-problem are mutually exclusive
    if args.normalize_per_problem and args.clip_per_problem:
        parser.error(
            "--normalize-per-problem and --clip-per-problem are mutually exclusive. "
            "Choose one or neither."
        )

    # Validate subset argument
    if args.subset is not None and not args.fixed_problems:
        parser.error("--subset requires --fixed-problems to be enabled")

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
        h5_problem_ranges = None  # Use full range

    # Parse param filters
    h5_param_filters: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None
    if args.param_filters:
        h5_param_filters = {}
        for filter_spec in args.param_filters:
            parts = filter_spec.split(":")
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid param filter spec: '{filter_spec}'. "
                    "Format: problem_name:param_name:min_val:max_val"
                )
            problem_name, param_name, min_val, max_val = parts
            h5_param_filters.setdefault(problem_name, {})[param_name] = (
                float(min_val),
                float(max_val),
            )
            logger.info(
                f"Param filter: {problem_name}.{param_name} in "
                f"[{float(min_val)}, {float(max_val)}]"
            )

    # Validate that all problems have h5 files
    for problem_name in problem_counts.keys():
        if problem_name not in h5_file_paths:
            raise ValueError(f"Problem '{problem_name}' requires h5 file via --h5-file")

    logger.info(f"Problem counts: {problem_counts}")

    multigpu_config = MultiGPUTrainingConfig(
        num_gpus=args.num_gpus,
        problem_counts=problem_counts,
        threads_per_gpu=args.threads_per_gpu,
        fixed_problems=args.fixed_problems,
        use_config_gradient=not args.no_config,
        early_stopping_enabled=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        grads_debug=args.grads_debug,
        subset_size=args.subset,
        normalize_per_problem=args.normalize_per_problem,
        clip_per_problem=args.clip_per_problem,
    )

    training_config_overrides = {
        "epochs_per_time": [args.epochs],
        "learning_rate": args.learning_rate,
        "hidden_channels": args.hidden_channels,
        "hidden_layers": args.hidden_layers,
        "num_cells_high_res": 64,
        "downaverage_factor": 2,
        "limiter": 4,
        "model_initialization_scale": args.model_scale,
        "start_correction_time": 0.0,
        "noise_level": 0.00,
        "warmup_steps_fraction": args.warmup_fraction,
        "peak_lr": args.peak_lr,
        "end_lr": args.end_lr,
        "gradient_clip": 1.0,
        "c_cfl": 0.8,
        "c_cfl_target": 0.8,
        "loss_type": args.loss_type,
        "snapshot_timepoints_train": [0.666],
        "correct_from_beggining": True,
        "t_end": 0.2,
        "patience": 30,
        "normalize_input": args.normalize_input,
        "use_film_corrector": args.use_film,
    }

    # Run training
    best_params, neural_net_static = train_with_cached_compilation(
        h5_file_paths=h5_file_paths,
        h5_problem_ranges=h5_problem_ranges,
        h5_param_filters=h5_param_filters,
        multigpu_config=multigpu_config,
        model_name=args.model_name,
        load_existing=args.load_existing,
        description=args.description,
        **training_config_overrides,
    )

    logger.info("✓ Training completed successfully")
