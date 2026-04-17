"""
Advanced multi-GPU training script for H5-based multi-problem solver-in-the-loop correction.

This script provides:
- Configurable per-problem sampling from H5 datasets (e.g., 30 blasts, 10 turbulence per epoch)
- Multi-GPU parallelization using autocvds and threading
- Per-epoch checkpoint saving with override
- Best model tracking and early stopping
- Integration with conFIG gradient averaging and noise perturbation
- Hyperparameter extraction and logging from H5 files
"""

from autocvd import autocvd
import logging
import datetime
import json
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import asdict
from timeit import default_timer as timer
from functools import partial
import random
import threading
from queue import Queue
import argparse
import sys


def _resolve_startup_num_gpus(default_num_gpus: int = 2) -> int:
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
import jax.numpy as jnp
import equinox as eqx
import optax
import math

from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.option_classes.simulation_config import STATE_TYPE

from arena.arena_tests.solver_in_the_loop.conFIG import conFIG
from arena.arena_tests.solver_in_the_loop.utils import perturb_state
from arena.arena_tests.solver_in_the_loop.loss import (
    EarlyStopper,
    loss_setup,
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

from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix.time_stepping import time_integration

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
# GRADIENT COMPUTATION & VALIDATION
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
    raise AttributeError(
        f"Problem descriptor has no 'name' or 'problem_name': {problem_descriptor}"
    )


def _descriptor_debug_context(problem_descriptor) -> Dict[str, Any]:
    """Extract robust debug context from descriptor variants."""
    try:
        problem_name = _descriptor_problem_name(problem_descriptor)
    except Exception:
        problem_name = str(problem_descriptor)

    nickname = getattr(problem_descriptor, "nickname", problem_name)
    problem_index = getattr(problem_descriptor, "problem_index", None)
    hyperparams = getattr(problem_descriptor, "hyperparams", None)
    if hyperparams is None:
        hyperparams = getattr(problem_descriptor, "params", None)

    return {
        "problem_name": problem_name,
        "nickname": nickname,
        "problem_index": problem_index,
        "hyperparams": hyperparams,
    }


def create_grad_fn(
    loss_fn_factory,
    loss_fn_kwargs,
    training_config: TrainingConfig,
    simulation_config,
    initial_state: STATE_TYPE,
    target_state: STATE_TYPE | SnapshotData,
    registered_variables,
    noise_level: float,
):
    """Create a function that computes loss and gradients for a single problem."""
    if noise_level == 0.0:
        perturb_state_partial = lambda _: initial_state
    else:
        perturb_state_partial = partial(
            perturb_state,
            state=initial_state,
            noise_level=noise_level,
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


def create_batched_grad_fn(
    loss_fn_factory,
    loss_fn_kwargs,
    training_config: TrainingConfig,
    simulation_config,
    registered_variables,
    noise_level: float,
):
    """Create a vmapped function that computes loss and gradients for a batch of problems.
    
    Unlike create_grad_fn which captures initial_state and target_state in closures,
    this function takes them as arguments, enabling vmap over multiple problems.
    """
    if noise_level == 0.0:
        perturb_state_fn = lambda state, _: state
    else:
        def perturb_state_fn(state, key):
            return perturb_state(state=state, noise_level=noise_level, key=key)

    def single_grad_fn_core(
        initial_state,
        target_state,
        cnn_mhd_corrector_params,
        network_params_arrays,
        params,
        key,
    ):
        """Compute loss and gradients for a single problem."""
        noisy_initial_state = perturb_state_fn(initial_state, key)

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

    # Vmap over batch dimension: initial_state, target_state, and key
    # cnn_mhd_corrector_params, network_params_arrays, and params are shared
    batched_grad_fn_core = jax.vmap(
        single_grad_fn_core,
        in_axes=(0, 0, None, None, None, 0),  # (initial, target, ..shared.., key)
    )

    batched_grad_fn = jax.jit(batched_grad_fn_core)
    return batched_grad_fn


def create_sequential_grad_fn(
    training_config: TrainingConfig,
    simulation_config,
    registered_variables,
    noise_level: float,
):
    """Create a grad function for sequential processing that takes states as arguments.
    
    Unlike create_grad_fn which captures states in closures, this function returns
    a grad_fn that accepts initial_state and target_state as arguments. This enables
    processing problems one at a time without recompilation.
    
    The loss function kwargs (channel_normalizers) are computed inside from the target.
    """
    from arena.arena_tests.solver_in_the_loop.loss import normalized_weighted_loss, simple_mse_loss
    
    if noise_level == 0.0:
        perturb_state_fn = lambda state, _: state
    else:
        def perturb_state_fn(state, key):
            return perturb_state(state=state, noise_level=noise_level, key=key)

    def grad_fn_core(
        initial_state,
        target_state,
        cnn_mhd_corrector_params,
        network_params_arrays,
        params,
        key,
    ):
        """Compute loss and gradients for a single problem with states as args."""
        noisy_initial_state = perturb_state_fn(initial_state, key)
        
        # Compute channel normalizers from this target state
        if training_config.loss_type == "norm_mse":
            # target_state shape: (num_vars, x, y, z)
            channel_normalizers = jnp.maximum(
                jnp.std(target_state, axis=(1, 2, 3)), 1e-8
            )
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


def group_pairs_by_problem_name(
    sampled_pairs: List,
) -> Dict[str, List]:
    """Group training pairs by their problem_name.
    
    Args:
        sampled_pairs: List of TrainingPair objects
        
    Returns:
        Dict mapping problem_name -> list of TrainingPairs with that name
    """
    pairs_by_name: Dict[str, List] = {}
    for pair in sampled_pairs:
        problem_name = _descriptor_problem_name(pair.problem_descriptor)
        if problem_name not in pairs_by_name:
            pairs_by_name[problem_name] = []
        pairs_by_name[problem_name].append(pair)
    return pairs_by_name


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


def create_batched_task(
    problem_name: str,
    training_pairs: List,
    loss_fn_factory: Callable,
    loss_fn_kwargs: Dict[str, Any],
    training_config: TrainingConfig,
    cnn_mhd_corrector_params,
    network_params_arrays,
    noise_level: float,
    key,
) -> "BatchedGradientComputationTask":
    """Create a batched task from a list of training pairs with the same problem_name.
    
    Args:
        problem_name: Shared problem name for all pairs
        training_pairs: List of TrainingPair objects (all same problem_name)
        loss_fn_factory: Factory function for creating loss functions
        loss_fn_kwargs: Keyword arguments for the loss function
        training_config: Training configuration
        cnn_mhd_corrector_params: CNN corrector parameters
        network_params_arrays: Neural network parameters
        noise_level: Noise level for state perturbation
        key: JAX PRNG key for generating subkeys
        
    Returns:
        BatchedGradientComputationTask with stacked states
    """
    batch_size = len(training_pairs)
    
    # Stack initial states: (batch_size, num_vars, ...)
    batched_initial_states = jnp.stack(
        [pair.lr_bundle.initial_state for pair in training_pairs]
    )
    
    # Stack target states: (batch_size, ...) - shape depends on target_state structure
    batched_target_states = jnp.stack(
        [pair.target_state for pair in training_pairs]
    )
    
    # Generate subkeys for each problem in batch
    subkeys = jax.random.split(key, batch_size)
    
    return BatchedGradientComputationTask(
        problem_name=problem_name,
        training_pairs=training_pairs,
        batched_initial_states=batched_initial_states,
        batched_target_states=batched_target_states,
        loss_fn_factory=loss_fn_factory,
        loss_fn_kwargs=loss_fn_kwargs,
        training_config=training_config,
        cnn_mhd_corrector_params=cnn_mhd_corrector_params,
        network_params_arrays=network_params_arrays,
        subkeys=subkeys,
        noise_level=noise_level,
    )


# ============================================================================
# MULTI-GPU & THREADING INFRASTRUCTURE
# ============================================================================


class GradientComputationTask:
    """Task for computing gradients in a separate thread."""

    def __init__(
        self,
        loss_fn_factory: Callable,
        loss_fn_kwargs: Dict[str, Any],
        training_config: TrainingConfig,
        training_pair,
        cnn_mhd_corrector_params,
        network_params_arrays,
        subkey,
        noise_level: float,
    ):
        self.loss_fn_factory = loss_fn_factory
        self.loss_fn_kwargs = loss_fn_kwargs
        self.training_config = training_config
        self.training_pair = training_pair
        self.cnn_mhd_corrector_params = cnn_mhd_corrector_params
        self.network_params_arrays = network_params_arrays
        self.subkey = subkey
        self.noise_level = noise_level
        self.result = None
        self.error = None


class BatchedGradientComputationTask:
    """Task for computing gradients for a batch of problems with the same config.
    
    This enables a single JIT compilation to process multiple problems that share
    the same simulation configuration (identified by problem_name).
    """

    def __init__(
        self,
        problem_name: str,
        training_pairs: List,
        batched_initial_states,  # Shape: (batch_size, num_vars, ...)
        batched_target_states,   # Shape: (batch_size, num_snapshots, num_vars, ...)
        loss_fn_factory: Callable,
        loss_fn_kwargs: Dict[str, Any],
        training_config: TrainingConfig,
        cnn_mhd_corrector_params,
        network_params_arrays,
        subkeys,  # Shape: (batch_size, 2) - array of PRNGKeys
        noise_level: float,
    ):
        self.problem_name = problem_name
        self.training_pairs = training_pairs
        self.batch_size = len(training_pairs)
        self.batched_initial_states = batched_initial_states
        self.batched_target_states = batched_target_states
        self.loss_fn_factory = loss_fn_factory
        self.loss_fn_kwargs = loss_fn_kwargs
        self.training_config = training_config
        self.cnn_mhd_corrector_params = cnn_mhd_corrector_params
        self.network_params_arrays = network_params_arrays
        self.subkeys = subkeys
        self.noise_level = noise_level
        self.result = None
        self.error = None

    @property
    def representative_pair(self):
        """Get the first training pair as representative for shared config."""
        return self.training_pairs[0]


class SequentialGradientTask:
    """Task for sequential gradient computation to minimize memory usage.
    
    Unlike BatchedGradientComputationTask which holds all states in memory,
    this task holds only lightweight descriptors (problem indices). The worker
    loads and processes ONE problem at a time, releasing memory between problems.
    
    This approach trades some parallelism for dramatically reduced memory usage,
    enabling training with many problems that wouldn't fit in GPU memory together.
    """

    def __init__(
        self,
        problem_name: str,
        problem_indices: List[int],
        h5_problem_manager,  # H5ProblemManager reference for lazy loading
        training_config: TrainingConfig,
        cnn_mhd_corrector_config,
        cnn_mhd_corrector_params,
        network_params_arrays,
        subkeys,  # List of PRNGKeys, one per problem
        noise_level: float,
    ):
        self.problem_name = problem_name
        self.problem_indices = problem_indices
        self.batch_size = len(problem_indices)
        self.h5_problem_manager = h5_problem_manager
        self.training_config = training_config
        self.cnn_mhd_corrector_config = cnn_mhd_corrector_config
        self.cnn_mhd_corrector_params = cnn_mhd_corrector_params
        self.network_params_arrays = network_params_arrays
        self.subkeys = subkeys
        self.noise_level = noise_level
        self.result = None
        self.error = None


def _grad_fn_cache_key(task: GradientComputationTask) -> str:
    """Build a stable grad-fn cache key for per-device grad-fn reuse."""
    descriptor = task.training_pair.problem_descriptor
    return getattr(descriptor, "nickname", repr(descriptor))


def _assign_tasks_to_devices(
    tasks: List[GradientComputationTask],
    num_devices: int,
) -> Tuple[List[List[GradientComputationTask]], List[Dict[str, int]]]:
    """Assign tasks to devices while maximizing same-key locality.

    Strategy:
    1. Enforce a base quota of floor(len(tasks) / num_devices) per device.
    2. Fill base quotas by greedily grouping equal cache-keys together.
    3. Assign remainder buckets to devices with the strongest existing key affinity.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be > 0, got {num_devices}")

    assignments: List[List[GradientComputationTask]] = [[] for _ in range(num_devices)]
    device_key_counts: List[Dict[str, int]] = [{} for _ in range(num_devices)]
    if not tasks:
        return assignments, device_key_counts

    tasks_by_key: Dict[str, List[GradientComputationTask]] = {}
    for task in tasks:
        cache_key = _grad_fn_cache_key(task)
        tasks_by_key.setdefault(cache_key, []).append(task)

    sorted_keys = sorted(tasks_by_key.keys(), key=lambda k: (-len(tasks_by_key[k]), k))
    base_quota = len(tasks) // num_devices
    capacities = [base_quota for _ in range(num_devices)]
    overflow_by_key: Dict[str, List[GradientComputationTask]] = {}

    def assign_task(device_idx: int, cache_key: str, task: GradientComputationTask) -> None:
        assignments[device_idx].append(task)
        device_key_counts[device_idx][cache_key] = (
            device_key_counts[device_idx].get(cache_key, 0) + 1
        )

    for cache_key in sorted_keys:
        for task in tasks_by_key[cache_key]:
            candidates = [i for i, capacity in enumerate(capacities) if capacity > 0]
            if not candidates:
                overflow_by_key.setdefault(cache_key, []).append(task)
                continue

            best_device = max(
                candidates,
                key=lambda i: (
                    device_key_counts[i].get(cache_key, 0),
                    capacities[i],
                    -len(device_key_counts[i]),
                    -len(assignments[i]),
                    -i,
                ),
            )
            assign_task(best_device, cache_key, task)
            capacities[best_device] -= 1

    if overflow_by_key:
        overflow_keys = sorted(
            overflow_by_key.keys(),
            key=lambda k: (-len(overflow_by_key[k]), k),
        )
        for cache_key in overflow_keys:
            best_device = max(
                range(num_devices),
                key=lambda i: (
                    device_key_counts[i].get(cache_key, 0),
                    -len(device_key_counts[i]),
                    -len(assignments[i]),
                    -i,
                ),
            )
            for task in overflow_by_key[cache_key]:
                assign_task(best_device, cache_key, task)

    return assignments, device_key_counts


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


def assign_batched_tasks_to_devices(
    pairs_by_problem: Dict[str, List],
    num_devices: int,
    loss_fn_factory: Callable,
    loss_fn_kwargs: Dict[str, Any],
    training_config: TrainingConfig,
    cnn_mhd_corrector_params,
    network_params_arrays,
    cnn_mhd_corrector_config,
    noise_levels: Dict[str, float],
    key,
) -> List[List[BatchedGradientComputationTask]]:
    """Assign batched tasks to devices with load balancing.
    
    Strategy:
    1. Sort problem groups by size (largest first)
    2. Assign groups to devices using greedy "least loaded" approach
    3. Split large groups across multiple devices if beneficial for balance
    
    Args:
        pairs_by_problem: Dict mapping problem_name -> list of TrainingPairs
        num_devices: Number of available devices
        loss_fn_factory: Factory for loss functions
        loss_fn_kwargs: Kwargs for loss function
        training_config: Training configuration
        cnn_mhd_corrector_params: CNN corrector parameters
        network_params_arrays: Neural network parameters
        cnn_mhd_corrector_config: CNN corrector config (for overriding bundles)
        noise_levels: Dict mapping problem_name -> noise level
        key: JAX PRNG key
        
    Returns:
        List of length num_devices, where each element is a list of 
        BatchedGradientComputationTask for that device.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be > 0, got {num_devices}")
    
    if not pairs_by_problem:
        return [[] for _ in range(num_devices)]
    
    # Calculate total problems and fair share per device
    total_problems = sum(len(pairs) for pairs in pairs_by_problem.values())
    fair_share = total_problems / num_devices
    
    # Sort problem names by group size (largest first)
    sorted_names = sorted(
        pairs_by_problem.keys(),
        key=lambda name: -len(pairs_by_problem[name])
    )
    
    # Track device loads
    device_loads = [0] * num_devices
    device_assignments: List[List[Tuple[str, List]]] = [[] for _ in range(num_devices)]
    
    for problem_name in sorted_names:
        pairs = pairs_by_problem[problem_name]
        group_size = len(pairs)
        
        # Check if we should split this group across devices
        # Split if group is larger than fair_share and we have underloaded devices
        min_load = min(device_loads)
        underloaded_devices = [i for i, load in enumerate(device_loads) if load < fair_share]
        
        if group_size > fair_share and len(underloaded_devices) > 1:
            # Split group across underloaded devices
            num_splits = min(len(underloaded_devices), max(1, int(group_size / fair_share)))
            split_size = group_size // num_splits
            remainder = group_size % num_splits
            
            start_idx = 0
            for i in range(num_splits):
                # Distribute remainder across first few splits
                chunk_size = split_size + (1 if i < remainder else 0)
                end_idx = start_idx + chunk_size
                chunk_pairs = pairs[start_idx:end_idx]
                
                # Find least loaded device among underloaded
                target_device = min(underloaded_devices, key=lambda d: device_loads[d])
                device_assignments[target_device].append((problem_name, chunk_pairs))
                device_loads[target_device] += chunk_size
                
                start_idx = end_idx
        else:
            # Assign entire group to least loaded device
            target_device = min(range(num_devices), key=lambda d: device_loads[d])
            device_assignments[target_device].append((problem_name, pairs))
            device_loads[target_device] += group_size
    
    # Convert assignments to BatchedGradientComputationTask objects
    result: List[List[BatchedGradientComputationTask]] = [[] for _ in range(num_devices)]
    
    for device_idx, assignments in enumerate(device_assignments):
        for problem_name, pairs in assignments:
            # Override solver config for each pair
            for pair in pairs:
                pair.lr_bundle.override_solver_in_the_loop(
                    corrector_config=cnn_mhd_corrector_config,
                    corrector_params=cnn_mhd_corrector_params,
                )
            
            # Split key for this batch
            key, subkey = jax.random.split(key)
            
            noise_level = noise_levels.get(problem_name, training_config.noise_level)
            
            batched_task = create_batched_task(
                problem_name=problem_name,
                training_pairs=pairs,
                loss_fn_factory=loss_fn_factory,
                loss_fn_kwargs=loss_fn_kwargs,
                training_config=training_config,
                cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                network_params_arrays=network_params_arrays,
                noise_level=noise_level,
                key=subkey,
            )
            result[device_idx].append(batched_task)
    
    return result


def assign_sequential_tasks_to_devices(
    indices_by_problem: Dict[str, List[int]],
    num_devices: int,
    h5_problem_manager,
    training_config: TrainingConfig,
    cnn_mhd_corrector_config,
    cnn_mhd_corrector_params,
    network_params_arrays,
    noise_levels: Dict[str, float],
    key,
) -> List[List[SequentialGradientTask]]:
    """Assign sequential tasks to devices with load balancing.
    
    Unlike assign_batched_tasks_to_devices, this function works with problem
    indices (not materialized TrainingPairs), enabling lazy loading in workers.
    
    Strategy:
    1. Sort problem groups by size (largest first)
    2. Assign groups to devices using greedy "least loaded" approach
    3. Split large groups across multiple devices if beneficial for balance
    
    Args:
        indices_by_problem: Dict mapping problem_name -> list of problem indices
        num_devices: Number of available devices
        h5_problem_manager: Manager for loading problems from H5
        training_config: Training configuration
        cnn_mhd_corrector_config: CNN corrector config
        cnn_mhd_corrector_params: CNN corrector parameters
        network_params_arrays: Neural network parameters
        noise_levels: Dict mapping problem_name -> noise level
        key: JAX PRNG key
        
    Returns:
        List of length num_devices, where each element is a list of 
        SequentialGradientTask for that device.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be > 0, got {num_devices}")
    
    if not indices_by_problem:
        return [[] for _ in range(num_devices)]
    
    # Calculate total problems and fair share per device
    total_problems = sum(len(indices) for indices in indices_by_problem.values())
    fair_share = total_problems / num_devices
    
    # Sort problem names by group size (largest first)
    sorted_names = sorted(
        indices_by_problem.keys(),
        key=lambda name: -len(indices_by_problem[name])
    )
    
    # Track device loads
    device_loads = [0] * num_devices
    device_assignments: List[List[Tuple[str, List[int]]]] = [[] for _ in range(num_devices)]
    
    for problem_name in sorted_names:
        indices = indices_by_problem[problem_name]
        group_size = len(indices)
        
        # Check if we should split this group across devices
        min_load = min(device_loads)
        underloaded_devices = [i for i, load in enumerate(device_loads) if load < fair_share]
        
        if group_size > fair_share and len(underloaded_devices) > 1:
            # Split group across underloaded devices
            num_splits = min(len(underloaded_devices), max(1, int(group_size / fair_share)))
            split_size = group_size // num_splits
            remainder = group_size % num_splits
            
            start_idx = 0
            for i in range(num_splits):
                chunk_size = split_size + (1 if i < remainder else 0)
                end_idx = start_idx + chunk_size
                chunk_indices = indices[start_idx:end_idx]
                
                target_device = min(underloaded_devices, key=lambda d: device_loads[d])
                device_assignments[target_device].append((problem_name, chunk_indices))
                device_loads[target_device] += chunk_size
                
                start_idx = end_idx
        else:
            # Assign entire group to least loaded device
            target_device = min(range(num_devices), key=lambda d: device_loads[d])
            device_assignments[target_device].append((problem_name, indices))
            device_loads[target_device] += group_size
    
    # Convert assignments to SequentialGradientTask objects
    result: List[List[SequentialGradientTask]] = [[] for _ in range(num_devices)]
    
    for device_idx, assignments in enumerate(device_assignments):
        for problem_name, indices in assignments:
            # Split key for this task
            key, subkey = jax.random.split(key)
            subkeys = list(jax.random.split(subkey, len(indices)))
            
            noise_level = noise_levels.get(problem_name, training_config.noise_level)
            
            task = SequentialGradientTask(
                problem_name=problem_name,
                problem_indices=indices,
                h5_problem_manager=h5_problem_manager,
                training_config=training_config,
                cnn_mhd_corrector_config=cnn_mhd_corrector_config,
                cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                network_params_arrays=network_params_arrays,
                subkeys=subkeys,
                noise_level=noise_level,
            )
            result[device_idx].append(task)
    
    return result


def compute_gradient_threaded_worker(
    worker_id: int,
    device: jax.Device,
    task_queue: Queue,
    result_queue: Queue,
    grad_fn_cache: Dict[str, Callable],
    grad_fn_cache_lock: threading.Lock,
) -> None:
    """Worker loop: build/call grad_fns on a fixed device and report results."""
    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            cache_key = _grad_fn_cache_key(task)
            with grad_fn_cache_lock:
                grad_fn = grad_fn_cache.get(cache_key)
                if grad_fn is None:
                    grad_fn = create_grad_fn(
                        loss_fn_factory=task.loss_fn_factory,
                        loss_fn_kwargs=task.loss_fn_kwargs,
                        training_config=task.training_config,
                        simulation_config=task.training_pair.lr_bundle.config,
                        initial_state=task.training_pair.lr_bundle.initial_state,
                        target_state=task.training_pair.target_state,
                        registered_variables=task.training_pair.lr_bundle.reg_vars,
                        noise_level=task.noise_level,
                    )
                    grad_fn_cache[cache_key] = grad_fn

            with jax.default_device(device):
                p_loss, p_grads, p_grad_mod, p_last_t = grad_fn(
                    cnn_mhd_corrector_params=task.cnn_mhd_corrector_params,
                    network_params_arrays=task.network_params_arrays,
                    params=task.training_pair.lr_bundle.params,
                    key=task.subkey,
                )

            validate_output(
                last_t=float(p_last_t),
                gradients_mod=float(p_grad_mod),
                problem_name=_descriptor_problem_name(
                    task.training_pair.problem_descriptor
                ),
            )

            result_queue.put(
                (
                    id(task),
                    {
                        "loss": float(p_loss),
                        "grads": p_grads,
                        "grad_mod": float(p_grad_mod),
                        "last_t": float(p_last_t),
                        "problem_name": task.training_pair.problem_descriptor.nickname,
                        "worker_id": worker_id,
                        "device": str(device),
                    },
                )
            )
        except Exception as e:
            debug_ctx = _descriptor_debug_context(task.training_pair.problem_descriptor)
            logger.exception(
                "Gradient worker %s failed for %s (nickname=%s, index=%s, hyperparams=%s)",
                worker_id,
                debug_ctx["problem_name"],
                debug_ctx["nickname"],
                debug_ctx["problem_index"],
                debug_ctx["hyperparams"],
            )
            result_queue.put(
                (
                    id(task),
                    {
                        "error": str(e),
                        "worker_id": worker_id,
                        "device": str(device),
                        "problem_name": debug_ctx["problem_name"],
                        "nickname": debug_ctx["nickname"],
                        "problem_index": debug_ctx["problem_index"],
                        "hyperparams": debug_ctx["hyperparams"],
                    },
                )
            )


def compute_batched_gradient_worker(
    worker_id: int,
    device: jax.Device,
    task_queue: Queue,
    result_queue: Queue,
    grad_fn_cache: Dict[str, Callable],
    grad_fn_cache_lock: threading.Lock,
) -> None:
    """Worker loop for batched gradient computation.
    
    Processes BatchedGradientComputationTask objects, computing gradients for
    all problems in the batch with a single JIT-compiled vmapped function.
    """
    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            # Cache key is the problem_name for batched tasks
            cache_key = task.problem_name
            
            with grad_fn_cache_lock:
                batched_grad_fn = grad_fn_cache.get(cache_key)
                if batched_grad_fn is None:
                    rep_pair = task.representative_pair
                    batched_grad_fn = create_batched_grad_fn(
                        loss_fn_factory=task.loss_fn_factory,
                        loss_fn_kwargs=task.loss_fn_kwargs,
                        training_config=task.training_config,
                        simulation_config=rep_pair.lr_bundle.config,
                        registered_variables=rep_pair.lr_bundle.reg_vars,
                        noise_level=task.noise_level,
                    )
                    grad_fn_cache[cache_key] = batched_grad_fn

            # Get shared params from representative pair
            rep_pair = task.representative_pair
            
            with jax.default_device(device):
                # Call batched gradient function
                # Returns: (batch_losses, batch_grads, batch_grad_mods, batch_last_ts)
                batch_losses, batch_grads, batch_grad_mods, batch_last_ts = batched_grad_fn(
                    task.batched_initial_states,
                    task.batched_target_states,
                    task.cnn_mhd_corrector_params,
                    task.network_params_arrays,
                    rep_pair.lr_bundle.params,
                    task.subkeys,
                )

            # Validate all outputs in batch
            for i in range(task.batch_size):
                validate_output(
                    last_t=float(batch_last_ts[i]),
                    gradients_mod=float(batch_grad_mods[i]),
                    problem_name=task.problem_name,
                )

            # Extract individual gradients from the batch for conFIG
            # batch_grads has shape (batch_size, ...) for each leaf
            individual_grads = []
            for i in range(task.batch_size):
                grad_i = jax.tree.map(lambda g: g[i], batch_grads)
                individual_grads.append(grad_i)

            # Build per-problem results for logging and loss tracking
            per_problem_results = []
            for i, pair in enumerate(task.training_pairs):
                per_problem_results.append({
                    "loss": float(batch_losses[i]),
                    "grad_mod": float(batch_grad_mods[i]),
                    "last_t": float(batch_last_ts[i]),
                    "problem_name": task.problem_name,
                    "nickname": pair.problem_descriptor.nickname,
                })

            result_queue.put(
                (
                    id(task),
                    {
                        "batched": True,
                        "problem_name": task.problem_name,
                        "batch_size": task.batch_size,
                        "individual_grads": individual_grads,  # List of grads, one per problem
                        "avg_loss": float(jnp.mean(batch_losses)),
                        "avg_grad_mod": float(jnp.mean(batch_grad_mods)),
                        "per_problem_results": per_problem_results,
                        "worker_id": worker_id,
                        "device": str(device),
                    },
                )
            )
        except Exception as e:
            logger.exception(
                "Batched gradient worker %s failed for %s (batch_size=%d)",
                worker_id,
                task.problem_name,
                task.batch_size,
            )
            result_queue.put(
                (
                    id(task),
                    {
                        "error": str(e),
                        "batched": True,
                        "worker_id": worker_id,
                        "device": str(device),
                        "problem_name": task.problem_name,
                        "batch_size": task.batch_size,
                    },
                )
            )


def compute_sequential_gradient_worker(
    worker_id: int,
    device: jax.Device,
    task_queue: Queue,
    result_queue: Queue,
    grad_fn_cache: Dict[str, Callable],
    grad_fn_cache_lock: threading.Lock,
) -> None:
    """Worker loop for sequential gradient computation with minimal memory usage.
    
    Processes SequentialGradientTask objects by loading and computing gradients
    for ONE problem at a time. This trades some parallelism for dramatically
    reduced memory usage.
    
    Flow for each task:
    1. Get or create cached grad_fn for this problem_type
    2. For each problem index:
       a. Load initial_state and target_state from H5
       b. Create lr_bundle with simulation config
       c. Compute gradient
       d. Accumulate results
       e. Delete loaded states to free memory
    3. Return aggregated results
    """
    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            # Cache key is the problem_name
            cache_key = task.problem_name
            
            # Get problem config from h5_problem_manager
            problem_configs = task.h5_problem_manager.problem_configs
            if task.problem_name not in problem_configs:
                raise ValueError(f"No config found for problem {task.problem_name}")
            
            config = problem_configs[task.problem_name]["config"]
            params = problem_configs[task.problem_name]["params"]
            
            # We need to get registered_variables - load one problem to get shape
            first_initial, first_target, _ = task.h5_problem_manager.h5_loaders[
                task.problem_name
            ].get_problem(task.problem_indices[0])
            
            from astronomix import finalize_config, get_registered_variables
            
            config_lr = finalize_config(
                config._replace(num_cells=int(first_initial.shape[-1])),
                first_initial.shape,
            )
            reg_vars = get_registered_variables(config_lr)
            
            config_lr = config_lr._replace(
                return_snapshots=True,
                progress_bar=False,
                use_specific_snapshot_timepoints=True,
                num_snapshots=1,
                cnn_mhd_corrector_config=task.cnn_mhd_corrector_config,
            )
            params_lr = params._replace(snapshot_timepoints=jnp.array([params.t_end]))
            
            # Clean up first load - we'll reload in the loop
            del first_initial, first_target
            gc.collect()
            
            # Get or create grad_fn
            with grad_fn_cache_lock:
                grad_fn = grad_fn_cache.get(cache_key)
                if grad_fn is None:
                    grad_fn = create_sequential_grad_fn(
                        training_config=task.training_config,
                        simulation_config=config_lr,
                        registered_variables=reg_vars,
                        noise_level=task.noise_level,
                    )
                    grad_fn_cache[cache_key] = grad_fn

            # Process problems one at a time
            individual_grads = []
            per_problem_results = []
            h5_loader = task.h5_problem_manager.h5_loaders[task.problem_name]
            
            with jax.default_device(device):
                for i, problem_idx in enumerate(task.problem_indices):
                    # Load single problem from H5
                    initial_state, target_state, hyperparams = h5_loader.get_problem(
                        problem_idx
                    )
                    
                    # Override corrector config in params
                    params_with_corrector = params_lr._replace(
                        cnn_mhd_corrector_params=task.cnn_mhd_corrector_params
                    )
                    
                    # Compute gradient
                    loss_val, grads, grad_mod, last_t = grad_fn(
                        initial_state,
                        target_state,
                        task.cnn_mhd_corrector_params,
                        task.network_params_arrays,
                        params_with_corrector,
                        task.subkeys[i],
                    )
                    
                    # Validate output
                    validate_output(
                        last_t=float(last_t),
                        gradients_mod=float(grad_mod),
                        problem_name=task.problem_name,
                    )
                    
                    # Accumulate results
                    individual_grads.append(grads)
                    per_problem_results.append({
                        "loss": float(loss_val),
                        "grad_mod": float(grad_mod),
                        "last_t": float(last_t),
                        "problem_name": task.problem_name,
                        "problem_index": problem_idx,
                        "hyperparams": hyperparams,
                    })
                    
                    # Explicit cleanup to free GPU memory
                    del initial_state, target_state, grads
                    gc.collect()

            # Compute averages
            avg_loss = sum(r["loss"] for r in per_problem_results) / len(per_problem_results)
            avg_grad_mod = sum(r["grad_mod"] for r in per_problem_results) / len(per_problem_results)
            
            result_queue.put(
                (
                    id(task),
                    {
                        "sequential": True,
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
            logger.exception(
                "Sequential gradient worker %s failed for %s (batch_size=%d)",
                worker_id,
                task.problem_name,
                task.batch_size,
            )
            result_queue.put(
                (
                    id(task),
                    {
                        "error": str(e),
                        "sequential": True,
                        "worker_id": worker_id,
                        "device": str(device),
                        "problem_name": task.problem_name,
                        "batch_size": task.batch_size,
                    },
                )
            )


def distribute_gradient_computation(
    tasks: List[GradientComputationTask],
    threads_per_gpu: int,
    num_gpus: Optional[int],
) -> Tuple[Dict[int, Dict], Dict[str, List[float]]]:
    """Distribute gradient computation across bounded worker threads.

    Args:
        tasks: List of gradient computation tasks
        threads_per_gpu: Number of worker threads per GPU/device
        num_gpus: Number of GPUs/devices to use (None = all available)

    Returns:
        Tuple of (results list, step_losses dict)
    """
    if not tasks:
        return {}, {}
    if threads_per_gpu <= 0:
        raise ValueError(f"threads_per_gpu must be > 0, got {threads_per_gpu}")

    devices = _select_training_devices(num_gpus)
    assignments, device_key_counts = _assign_tasks_to_devices(tasks, len(devices))
    active_device_indices = [idx for idx, assigned in enumerate(assignments) if assigned]
    max_workers = sum(min(threads_per_gpu, len(assignments[idx])) for idx in active_device_indices)

    logger.info(
        "Launching %d gradient workers across %d/%d devices (threads_per_gpu=%d)",
        max_workers,
        len(active_device_indices),
        len(devices),
        threads_per_gpu,
    )
    for device_idx, assigned_tasks in enumerate(assignments):
        key_counts = device_key_counts[device_idx]
        key_summary = ", ".join(
            f"{key}:{count}"
            for key, count in sorted(key_counts.items(), key=lambda item: (-item[1], item[0]))
        )
        logger.info(
            "  Device %s assigned %d tasks across %d key(s)%s",
            devices[device_idx],
            len(assigned_tasks),
            len(key_counts),
            f" [{key_summary}]" if key_summary else "",
        )

    result_queue: Queue = Queue()

    threads = []
    for device_idx in active_device_indices:
        device = devices[device_idx]
        assigned_tasks = assignments[device_idx]
        workers_for_device = min(threads_per_gpu, len(assigned_tasks))
        task_queue: Queue = Queue()

        for task in assigned_tasks:
            task_queue.put(task)
        for _ in range(workers_for_device):
            task_queue.put(None)

        shared_grad_fn_cache: Dict[str, Callable] = {}
        shared_cache_lock = threading.Lock()
        for _ in range(workers_for_device):
            worker_id = len(threads)
            thread = threading.Thread(
                target=compute_gradient_threaded_worker,
                args=(
                    worker_id,
                    device,
                    task_queue,
                    result_queue,
                    shared_grad_fn_cache,
                    shared_cache_lock,
                ),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

    results: Dict[int, Dict] = {}
    step_losses: Dict[str, List[float]] = {}
    first_error: Optional[Dict[str, Any]] = None

    for _ in range(len(tasks)):
        task_id, result = result_queue.get()
        if "error" in result:
            logger.error(
                "Worker failure (worker=%s, device=%s, problem=%s, nickname=%s, index=%s, hyperparams=%s): %s",
                result.get("worker_id", "unknown"),
                result.get("device", "unknown"),
                result.get("problem_name", "unknown"),
                result.get("nickname", "unknown"),
                result.get("problem_index", "unknown"),
                result.get("hyperparams", "unknown"),
                result.get("error"),
            )
            if first_error is None:
                first_error = result
            continue
        else:
            results[task_id] = result
            step_losses.setdefault(result["problem_name"], []).append(result["loss"])

    for thread in threads:
        thread.join()

    if first_error is not None:
        raise RuntimeError(
            "Gradient computation failed "
            f"(worker={first_error.get('worker_id', 'unknown')}, "
            f"device={first_error.get('device', 'unknown')}, "
            f"problem={first_error.get('problem_name', 'unknown')}, "
            f"nickname={first_error.get('nickname', 'unknown')}, "
            f"index={first_error.get('problem_index', 'unknown')}, "
            f"hyperparams={first_error.get('hyperparams', 'unknown')}): "
            f"{first_error['error']}"
        )

    return results, step_losses


def distribute_batched_gradient_computation(
    device_batched_tasks: List[List[BatchedGradientComputationTask]],
    num_gpus: Optional[int],
) -> Tuple[Dict[int, Dict], Dict[str, List[float]], int]:
    """Distribute batched gradient computation across devices.
    
    Each device gets a list of BatchedGradientComputationTask objects.
    Each task represents a batch of problems with the same problem_name.
    
    Args:
        device_batched_tasks: List of length num_devices, where each element
                              is a list of BatchedGradientComputationTask.
        num_gpus: Number of GPUs/devices to use (None = all available)
        
    Returns:
        Tuple of (results dict, step_losses dict, total_problem_count)
    """
    devices = _select_training_devices(num_gpus)
    
    # Count total tasks and problems
    total_tasks = sum(len(tasks) for tasks in device_batched_tasks)
    total_problems = sum(
        sum(task.batch_size for task in tasks) 
        for tasks in device_batched_tasks
    )
    
    if total_tasks == 0:
        return {}, {}, 0
    
    active_device_indices = [
        idx for idx, tasks in enumerate(device_batched_tasks) if tasks
    ]
    
    # Log device assignments
    logger.info(
        "Launching batched gradient workers across %d/%d devices "
        "(total: %d batched tasks, %d problems)",
        len(active_device_indices),
        len(devices),
        total_tasks,
        total_problems,
    )
    for device_idx in active_device_indices:
        tasks = device_batched_tasks[device_idx]
        task_summary = ", ".join(
            f"{t.problem_name}:{t.batch_size}" for t in tasks
        )
        problems_on_device = sum(t.batch_size for t in tasks)
        logger.info(
            "  Device %s: %d batch(es), %d problems [%s]",
            devices[device_idx],
            len(tasks),
            problems_on_device,
            task_summary,
        )

    result_queue: Queue = Queue()
    threads = []
    
    # One worker per device (batched tasks process entire batch at once)
    for device_idx in active_device_indices:
        device = devices[device_idx]
        tasks = device_batched_tasks[device_idx]
        
        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)
        task_queue.put(None)  # Sentinel to stop worker
        
        shared_grad_fn_cache: Dict[str, Callable] = {}
        shared_cache_lock = threading.Lock()
        
        worker_id = len(threads)
        thread = threading.Thread(
            target=compute_batched_gradient_worker,
            args=(
                worker_id,
                device,
                task_queue,
                result_queue,
                shared_grad_fn_cache,
                shared_cache_lock,
            ),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    # Collect results
    results: Dict[int, Dict] = {}
    step_losses: Dict[str, List[float]] = {}
    first_error: Optional[Dict[str, Any]] = None

    for _ in range(total_tasks):
        task_id, result = result_queue.get()
        if "error" in result:
            logger.error(
                "Batched worker failure (worker=%s, device=%s, problem=%s, batch_size=%s): %s",
                result.get("worker_id", "unknown"),
                result.get("device", "unknown"),
                result.get("problem_name", "unknown"),
                result.get("batch_size", "unknown"),
                result.get("error"),
            )
            if first_error is None:
                first_error = result
            continue
        else:
            results[task_id] = result
            # Extract per-problem losses for tracking
            problem_name = result["problem_name"]
            for per_problem in result.get("per_problem_results", []):
                step_losses.setdefault(problem_name, []).append(per_problem["loss"])

    for thread in threads:
        thread.join()

    if first_error is not None:
        raise RuntimeError(
            "Batched gradient computation failed "
            f"(worker={first_error.get('worker_id', 'unknown')}, "
            f"device={first_error.get('device', 'unknown')}, "
            f"problem={first_error.get('problem_name', 'unknown')}, "
            f"batch_size={first_error.get('batch_size', 'unknown')}): "
            f"{first_error['error']}"
        )

    return results, step_losses, total_problems


def distribute_sequential_gradient_computation(
    device_sequential_tasks: List[List[SequentialGradientTask]],
    num_gpus: Optional[int],
) -> Tuple[Dict[int, Dict], Dict[str, List[float]], int]:
    """Distribute sequential gradient computation across devices.
    
    Each device gets a list of SequentialGradientTask objects.
    Each task represents a group of problem indices to be processed one at a time.
    
    Args:
        device_sequential_tasks: List of length num_devices, where each element
                                 is a list of SequentialGradientTask.
        num_gpus: Number of GPUs/devices to use (None = all available)
        
    Returns:
        Tuple of (results dict, step_losses dict, total_problem_count)
    """
    devices = _select_training_devices(num_gpus)
    
    # Count total tasks and problems
    total_tasks = sum(len(tasks) for tasks in device_sequential_tasks)
    total_problems = sum(
        sum(task.batch_size for task in tasks) 
        for tasks in device_sequential_tasks
    )
    
    if total_tasks == 0:
        return {}, {}, 0
    
    active_device_indices = [
        idx for idx, tasks in enumerate(device_sequential_tasks) if tasks
    ]
    
    # Log device assignments
    logger.info(
        "Launching sequential gradient workers across %d/%d devices "
        "(total: %d sequential tasks, %d problems)",
        len(active_device_indices),
        len(devices),
        total_tasks,
        total_problems,
    )
    for device_idx in active_device_indices:
        tasks = device_sequential_tasks[device_idx]
        task_summary = ", ".join(
            f"{t.problem_name}:{t.batch_size}" for t in tasks
        )
        problems_on_device = sum(t.batch_size for t in tasks)
        logger.info(
            "  Device %s: %d task(s), %d problems [%s]",
            devices[device_idx],
            len(tasks),
            problems_on_device,
            task_summary,
        )

    result_queue: Queue = Queue()
    threads = []
    
    # One worker per device
    for device_idx in active_device_indices:
        device = devices[device_idx]
        tasks = device_sequential_tasks[device_idx]
        
        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)
        task_queue.put(None)  # Sentinel to stop worker
        
        shared_grad_fn_cache: Dict[str, Callable] = {}
        shared_cache_lock = threading.Lock()
        
        worker_id = len(threads)
        thread = threading.Thread(
            target=compute_sequential_gradient_worker,
            args=(
                worker_id,
                device,
                task_queue,
                result_queue,
                shared_grad_fn_cache,
                shared_cache_lock,
            ),
            daemon=False,
        )
        thread.start()
        threads.append(thread)

    # Collect results
    results: Dict[int, Dict] = {}
    step_losses: Dict[str, List[float]] = {}
    first_error: Optional[Dict[str, Any]] = None

    for _ in range(total_tasks):
        task_id, result = result_queue.get()
        if "error" in result:
            logger.error(
                "Sequential worker failure (worker=%s, device=%s, problem=%s, batch_size=%s): %s",
                result.get("worker_id", "unknown"),
                result.get("device", "unknown"),
                result.get("problem_name", "unknown"),
                result.get("batch_size", "unknown"),
                result.get("error"),
            )
            if first_error is None:
                first_error = result
            continue
        else:
            results[task_id] = result
            # Extract per-problem losses for tracking
            problem_name = result["problem_name"]
            for per_problem in result.get("per_problem_results", []):
                step_losses.setdefault(problem_name, []).append(per_problem["loss"])

    for thread in threads:
        thread.join()

    if first_error is not None:
        raise RuntimeError(
            "Sequential gradient computation failed "
            f"(worker={first_error.get('worker_id', 'unknown')}, "
            f"device={first_error.get('device', 'unknown')}, "
            f"problem={first_error.get('problem_name', 'unknown')}, "
            f"batch_size={first_error.get('batch_size', 'unknown')}): "
            f"{first_error['error']}"
        )

    return results, step_losses, total_problems


# ============================================================================
# PROBLEM SAMPLING
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
        """Sample one training pair for each configured problem-count combination.

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


def training_model_multigpu(
    model_manager: ModelManager,
    h5_problem_manager: H5ProblemManager,
    multigpu_config: MultiGPUTrainingConfig,
    training_config: TrainingConfig,
    model_name: str,
    load_model: bool = False,
    load_model_nan: bool = False,
    description: str = "",
):
    """Multi-GPU training loop with H5-based problem sampling.

    Args:
        model_manager: Manager for saving checkpoints and metadata
        h5_problem_manager: Manager for loading training pairs from H5 datasets
        multigpu_config: Configuration for multi-GPU training
        training_config: Base training configuration
        model_name: Name of the model
        load_model: Whether to load existing model
        load_model_nan: Whether to load model from NaN checkpoint
        description: Description of the training run

    Returns:
        Tuple of (best_params, neural_net_static)
    """

    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Multi-GPU Training: {model_name}")
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

    # Load lightweight problem descriptors from H5 (lazy, no tensors yet)
    problem_descriptors = h5_problem_manager.get_problem_descriptors()
    if not problem_descriptors:
        raise ValueError("No H5 problem descriptors found for training")

    # Initialize sampler
    sampler = ProblemSampler(
        problem_descriptors=problem_descriptors,
        problem_counts=multigpu_config.problem_counts,
    )

    # Materialize one pair to derive architecture
    bootstrap_problem_name = next(iter(multigpu_config.problem_counts.keys()))
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
        max_patience=multigpu_config.early_stopping_patience,
        use_early_stopper=multigpu_config.early_stopping_enabled,
    )

    logger.info(f"Total available descriptors: {len(problem_descriptors)}")
    logger.info(f"Problem counts: {multigpu_config.problem_counts}")
    logger.info(f"Threads per GPU: {multigpu_config.threads_per_gpu}")

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

            # Assign sequential tasks to devices with load balancing
            # No states are loaded here - just problem indices
            device_sequential_tasks = assign_sequential_tasks_to_devices(
                indices_by_problem=indices_by_problem,
                num_devices=multigpu_config.num_gpus or len(_select_training_devices(None)),
                h5_problem_manager=h5_problem_manager,
                training_config=training_config,
                cnn_mhd_corrector_config=cnn_mhd_corrector_config,
                cnn_mhd_corrector_params=cnn_mhd_corrector_params,
                network_params_arrays=trained_params,
                noise_levels=PROBLEM_NOISE_LEVELS,
                key=batch_key,
            )

            # Compute gradients with sequential distribution
            # Workers will load ONE problem at a time from H5
            results, step_losses, total_problems = distribute_sequential_gradient_computation(
                device_sequential_tasks,
                num_gpus=multigpu_config.num_gpus,
            )

            # Aggregate gradients from sequential results
            all_grads = []  # Individual gradients from all problems
            total_grad_mod = 0.0
            all_losses = []

            for device_tasks in device_sequential_tasks:
                for task in device_tasks:
                    result = results[id(task)]
                    # Collect individual gradients from each problem
                    all_grads.extend(result["individual_grads"])
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

            # Average gradients
            if multigpu_config.use_config_gradient:
                start_time_config = timer()
                # conFIG receives individual gradient from each problem
                grads = conFIG(all_grads, use_least_square=True)
                logger.info(
                    f"conFIG gradient averaging took {timer() - start_time_config:.3f}s"
                )
            else:
                # Simple averaging of all individual gradients
                num_grads = len(all_grads)
                grads = jax.tree.map(
                    lambda *gs: sum(gs) / num_grads,
                    *all_grads
                )

            # Optimizer update
            updates, opt_state = optimizer.update(grads, opt_state, trained_params)
            trained_params = eqx.apply_updates(trained_params, updates)

            # Update corrector params
            cnn_mhd_corrector_params = cnn_mhd_corrector_params._replace(
                network_params=trained_params
            )

            # Track losses
            avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
            avg_grad_mod = total_grad_mod / total_problems if total_problems > 0 else 0.0
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
        logger.error(f"Training failed with error: {e}")
        model_manager.save_model_params(trained_params, "model_params_NAN.eqx")
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

    return best_params, neural_net_static


# ============================================================================
# ENTRY POINT
# ============================================================================


def train_new_model_multigpu(
    h5_file_paths: Dict[str, str],
    h5_problem_ranges: Dict[str, Tuple[int, int]],
    multigpu_config: MultiGPUTrainingConfig,
    model_name: Optional[str] = None,
    load_existing: bool = False,
    load_existing_nan: bool = False,
    description: str = "",
    **training_config_overrides,
):
    """Main entry point for H5-based multi-GPU training.

    Args:
        h5_file_paths: Dict mapping problem names to h5 file paths
        h5_problem_ranges: Dict mapping problem names to (start, end) index ranges
        multigpu_config: Multi-GPU training configuration
        model_name: Name of the model (auto-generated if None)
        load_existing: Whether to load existing model
        load_existing_nan: Whether to load from NaN checkpoint
        description: Description of the training run
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

    # Create H5 problem manager
    h5_problem_manager = H5ProblemManager(
        h5_file_paths=h5_file_paths,
        training_config=training_config,
        problem_ranges=h5_problem_ranges,
    )
    multigpu_config.validate(available_problems=list(h5_problem_manager.h5_loaders))

    logger.info("\n" + "=" * 80)
    logger.info(f"Model Name: {model_name}")
    logger.info(f"Training Config: epochs={training_config.epochs_per_time}")
    logger.info(f"MultiGPU Config: {multigpu_config}")
    logger.info("=" * 80 + "\n")

    # Run training
    best_params, neural_net_static = training_model_multigpu(
        model_manager=model_manager,
        h5_problem_manager=h5_problem_manager,
        multigpu_config=multigpu_config,
        training_config=training_config,
        model_name=model_name,
        load_model=load_existing,
        load_model_nan=load_existing_nan,
        description=description,
    )

    model_manager.print_model_info()
    return best_params, neural_net_static


# ============================================================================
# CLI USAGE
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="H5-based multi-GPU training for multi-problem solver correction"
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
        "--num-gpus",
        type=int,
        default=2,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--threads-per-gpu",
        type=int,
        default=1,
        help="Number of threads per GPU",
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

    # Create configs
    multigpu_config = MultiGPUTrainingConfig(
        problem_counts=problem_counts,
        num_gpus=args.num_gpus,
        threads_per_gpu=args.threads_per_gpu,
        early_stopping_enabled=args.early_stopping,
    )

    training_config_overrides = {
        "epochs_per_time": [args.epochs],
        "learning_rate": args.learning_rate,
        "hidden_channels": args.hidden_channels,
        "hidden_layers": args.hidden_layers,
    }

    # Run training
    best_params, neural_net_static = train_new_model_multigpu(
        h5_file_paths=h5_file_paths,
        h5_problem_ranges=h5_problem_ranges,
        multigpu_config=multigpu_config,
        model_name=args.model_name,
        load_existing=args.load_existing,
        description=args.description,
        **training_config_overrides,
    )

    logger.info("✓ Training completed successfully")
