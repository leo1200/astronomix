"""
Multi-GPU training script WITHOUT H5 dataset — uses fixed ProblemDescriptors with cached JIT.

This script adapts training_multigpu_cached_compilation.py to work with a small, fixed set
of problems defined via ProblemDescriptors (no H5 files). Target states are computed once
via HR simulation and cached as .npy files by ProblemManager.

Key differences from training_multigpu_cached_compilation.py:
1. Uses ProblemManager (not H5ProblemManager) — problems are fixed, not randomly sampled
2. Pre-compilation uses actual training pair bundles as representative states
3. Workers operate on in-memory (initial_state, target_state) arrays
4. All training pairs are used every epoch (no sampling from a large pool)
5. Supports --base-dir for flexible output directory (e.g. experiment/azimuth_generalized/normal)
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
import threading
from queue import Queue
import argparse
import sys
import gc


def _resolve_startup_args(default_num_gpus: int = 1):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-gpus", type=int, default=default_num_gpus)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Pin this process to a specific physical GPU index.",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.num_gpus, args.cuda_device


STARTUP_NUM_GPUS, STARTUP_CUDA_DEVICE = _resolve_startup_args()
if STARTUP_CUDA_DEVICE is not None:
    # Exclude all GPUs except the target one so autocvd is forced to pick it.
    # Query the actual installed GPU count to avoid ValueError from list.remove().
    from autocvd.nvidia_smi_calls import get_installed_gpus as _get_installed_gpus

    _num_installed = _get_installed_gpus()
    _exclude = [i for i in range(_num_installed) if i != STARTUP_CUDA_DEVICE]
    autocvd(num_gpus=STARTUP_NUM_GPUS, exclude=_exclude)
else:
    autocvd(num_gpus=STARTUP_NUM_GPUS)

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import math

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
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
    ProblemDescriptor,
    ProblemManager,
    TrainingPair,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
    FiLMCorrectorCNN,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)

logging.basicConfig(
    format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PROBLEM_NOISE_LEVELS: Dict[str, float] = {
    "mhd_blast": 0.04,
    "ot_vortex": 0.04,
    "turbulence": 0.00,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def compute_channel_normalizers(target_state: jnp.ndarray) -> jnp.ndarray:
    """Compute per-channel normalization factors OUTSIDE JIT to avoid recompilation."""
    return jnp.maximum(jnp.std(target_state, axis=(1, 2, 3)), 1e-8)


def validate_output(last_t: float, gradients_mod: float, problem_name: str) -> None:
    if last_t == 0.0:
        raise ValueError(f"NaN in forward pass for problem {problem_name}")
    if math.isnan(gradients_mod):
        raise ValueError(f"NaN in gradients for problem {problem_name}")


# ============================================================================
# CACHED GRADIENT FUNCTION (same structure as H5 version)
# ============================================================================


def create_cached_grad_fn(
    training_config: TrainingConfig,
    simulation_config,
    registered_variables,
    cnn_mhd_corrector_config,
    base_params,
    noise_level: float,
) -> Callable:
    """Create a JIT-compiled gradient function with config captured in closure.

    Signature of returned function:
        (initial_state, target_state, network_params, channel_normalizers, key)
        -> (loss, grads, grad_mod, last_t)

    All configuration is captured in the closure for stable JIT caching.
    Channel normalizers are passed as a dynamic array argument (computed outside JIT).
    """
    if noise_level == 0.0:
        perturb_state_fn = lambda state, _: state
    else:

        def perturb_state_fn(state, key):
            return perturb_state(state=state, noise_level=noise_level, key=key)

    final_simulation_config = simulation_config._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    _frozen_simulation_config = final_simulation_config
    _frozen_registered_variables = registered_variables
    _frozen_base_params = base_params

    use_norm_mse = training_config.loss_type == "norm_mse"
    if training_config.physics_weights is not None:
        physics_weights = jnp.array(training_config.physics_weights, dtype=jnp.float32)
    else:
        physics_weights = None
    use_interface = training_config.use_interface

    def grad_fn_core(
        initial_state, target_state, network_params, channel_normalizers, key
    ):
        noisy_initial_state = perturb_state_fn(initial_state, key)

        def loss_fn(net_params):
            from astronomix.time_stepping import time_integration

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
        gradients_modulus = jnp.sqrt(
            sum(jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads))
        )
        return loss_value, grads, gradients_modulus, last_timepoint

    return jax.jit(grad_fn_core)


# ============================================================================
# PRE-COMPILATION FROM TRAINING PAIRS (no H5)
# ============================================================================


def precompile_grad_fns_from_pairs(
    training_pairs: List[TrainingPair],
    training_config: TrainingConfig,
    cnn_mhd_corrector_config,
    noise_levels: Dict[str, float],
) -> Tuple[Dict[str, Callable], Dict[str, Any]]:
    """Pre-compile one grad_fn per unique problem.name from in-memory training pairs.

    Since all problems with the same name share the same simulation config
    (only initial state differs), one compiled function per name suffices.

    Returns:
        grad_fn_cache: problem_name -> JIT grad function
        problem_configs: problem_name -> {config, params, reg_vars}
    """
    grad_fn_cache: Dict[str, Callable] = {}
    problem_configs: Dict[str, Any] = {}

    seen_names = set()
    logger.info("Pre-compiling gradient functions for each problem type...")

    for pair in training_pairs:
        problem_name = pair.problem_descriptor.name
        if problem_name in seen_names:
            continue
        seen_names.add(problem_name)

        logger.info(f"  Pre-compiling grad_fn for {problem_name}...")

        # Use this pair's lr_bundle as the representative configuration
        config_lr = pair.lr_bundle.config._replace(
            cnn_mhd_corrector_config=cnn_mhd_corrector_config
        )
        params_lr = pair.lr_bundle.params
        reg_vars = pair.lr_bundle.reg_vars

        problem_configs[problem_name] = {
            "config": config_lr,
            "params": params_lr,
            "reg_vars": reg_vars,
        }

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
        logger.info(f"    ✓ {problem_name} grad_fn compiled")

    logger.info(
        f"Pre-compilation complete: {len(grad_fn_cache)} grad function(s) cached"
    )
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
    dummy_key = jax.random.PRNGKey(0)
    try:
        with jax.default_device(device):
            result = grad_fn(
                jax.device_put(initial_state, device),
                jax.device_put(target_state, device),
                jax.device_put(network_params, device),
                jax.device_put(channel_normalizers, device),
                jax.device_put(dummy_key, device),
            )
            jax.block_until_ready(result)
            return None
    except Exception as e:
        return f"{problem_name} on {device}: {e}"


def warmup_grad_fn_cache_from_pairs(
    grad_fn_cache: Dict[str, Callable],
    training_pairs: List[TrainingPair],
    network_params,
) -> None:
    """Warm up JIT cache on all devices using actual training pair states."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    devices = jax.devices()
    if len(devices) <= 1:
        logger.info("Single device detected, skipping parallel warmup")
        return

    logger.info(f"Warming up JIT cache on {len(devices)} devices (parallel)...")

    # One representative pair per problem name
    representative: Dict[str, TrainingPair] = {}
    for pair in training_pairs:
        if pair.problem_descriptor.name not in representative:
            representative[pair.problem_descriptor.name] = pair

    for problem_name, pair in representative.items():
        if problem_name not in grad_fn_cache:
            continue
        grad_fn = grad_fn_cache[problem_name]
        initial_state = pair.lr_bundle.initial_state
        target_state = pair.target_state
        channel_normalizers = compute_channel_normalizers(target_state)

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

        if errors:
            logger.warning(
                f"    ⚠ {problem_name}: {len(errors)} device(s) failed warmup"
            )
        else:
            logger.info(f"    ✓ {problem_name} cache warmed on all devices")


# ============================================================================
# TASK AND WORKER (in-memory, no H5)
# ============================================================================


class WoDatasetGradientTask:
    """Task carrying in-memory training pairs for a single device worker."""

    def __init__(
        self,
        problem_name: str,
        training_pairs: List[TrainingPair],
        network_params,
        subkeys,
    ):
        self.problem_name = problem_name
        self.training_pairs = training_pairs
        self.batch_size = len(training_pairs)
        self.network_params = network_params
        self.subkeys = subkeys
        self.result = None
        self.error = None


def assign_tasks_to_devices_from_pairs(
    training_pairs: List[TrainingPair],
    num_devices: int,
    network_params,
    key,
    threads_per_gpu: int = 1,
) -> List[List[WoDatasetGradientTask]]:
    """Distribute training pairs across devices with even load balancing.

    Args:
        training_pairs: All training pairs (fixed set, used every epoch)
        num_devices: Number of available devices
        network_params: Neural network parameters (pytree)
        key: JAX PRNG key
        threads_per_gpu: Number of worker threads per GPU

    Returns:
        List of length num_devices, each containing a list of WoDatasetGradientTask.
    """
    if num_devices <= 0:
        raise ValueError(f"num_devices must be > 0, got {num_devices}")

    if not training_pairs:
        return [[] for _ in range(num_devices)]

    total = len(training_pairs)
    base = total // num_devices
    remainder = total % num_devices

    # Distribute pairs round-robin
    device_pairs: List[List[TrainingPair]] = [[] for _ in range(num_devices)]
    idx = 0
    for dev in range(num_devices):
        count = base + (1 if dev < remainder else 0)
        device_pairs[dev] = training_pairs[idx : idx + count]
        idx += count

    result: List[List[WoDatasetGradientTask]] = [[] for _ in range(num_devices)]

    for dev_idx, pairs in enumerate(device_pairs):
        if not pairs:
            continue

        # Group by problem name within this device for task creation
        by_name: Dict[str, List[TrainingPair]] = {}
        for pair in pairs:
            by_name.setdefault(pair.problem_descriptor.name, []).append(pair)

        for problem_name, name_pairs in by_name.items():
            # Split into multiple tasks if threads_per_gpu > 1 and enough pairs
            if threads_per_gpu > 1 and len(name_pairs) > 1:
                chunk_size = max(
                    1, (len(name_pairs) + threads_per_gpu - 1) // threads_per_gpu
                )
                chunks = [
                    name_pairs[i : i + chunk_size]
                    for i in range(0, len(name_pairs), chunk_size)
                ]
            else:
                chunks = [name_pairs]

            for chunk in chunks:
                key, subkey = jax.random.split(key)
                subkeys = jax.random.split(subkey, len(chunk))
                task = WoDatasetGradientTask(
                    problem_name=problem_name,
                    training_pairs=chunk,
                    network_params=network_params,
                    subkeys=subkeys,
                )
                result[dev_idx].append(task)

    # Log distribution
    for dev_idx in range(num_devices):
        total_dev = sum(t.batch_size for t in result[dev_idx])
        breakdown = ", ".join(
            f"{t.problem_name}:{t.batch_size}" for t in result[dev_idx]
        )
        logger.info(
            f"  Device cuda:{dev_idx}: {len(result[dev_idx])} task(s), "
            f"{total_dev} pairs [{breakdown}]"
        )

    return result


def compute_gradient_worker_from_pairs(
    worker_id: int,
    device: jax.Device,
    task_queue: Queue,
    result_queue: Queue,
    grad_fn_cache: Dict[str, Callable],
    problem_configs: Dict[str, Any],
) -> None:
    """Worker using pre-compiled grad functions on in-memory training pairs."""
    while True:
        task = task_queue.get()
        if task is None:
            return

        try:
            grad_fn = grad_fn_cache[task.problem_name]

            individual_grads = []
            per_problem_results = []

            with jax.default_device(device):
                for i, pair in enumerate(task.training_pairs):
                    initial_state = pair.lr_bundle.initial_state
                    target_state = pair.target_state

                    # Compute channel normalizers OUTSIDE JIT
                    channel_normalizers = compute_channel_normalizers(target_state)

                    loss_val, grads, grad_mod, last_t = grad_fn(
                        initial_state,
                        target_state,
                        task.network_params,
                        channel_normalizers,
                        task.subkeys[i],
                    )

                    try:
                        validate_output(
                            last_t=float(last_t),
                            gradients_mod=float(grad_mod),
                            problem_name=task.problem_name,
                        )
                    except ValueError as nan_err:
                        logger.warning(
                            "Skipping pair %s in worker %d: %s",
                            pair.problem_descriptor.nickname,
                            worker_id,
                            nan_err,
                        )
                        del grads
                        gc.collect()
                        continue

                    # Move grads to CPU immediately to avoid GPU OOM accumulation
                    individual_grads.append(jax.device_get(grads))
                    per_problem_results.append(
                        {
                            "loss": float(loss_val),
                            "grad_mod": float(grad_mod),
                            "last_t": float(last_t),
                            "problem_name": task.problem_name,
                            "nickname": pair.problem_descriptor.nickname,
                        }
                    )
                    del grads
                    gc.collect()

            if not per_problem_results:
                raise ValueError(
                    f"All pairs in worker {worker_id} produced NaN — no valid gradients."
                )

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
                "Worker %d failed for %s (batch_size=%d)",
                worker_id,
                task.problem_name,
                task.batch_size,
            )
            result_queue.put(
                (
                    id(task),
                    {
                        "error": str(e),
                        "worker_id": worker_id,
                        "device": str(device),
                        "problem_name": task.problem_name,
                        "batch_size": task.batch_size,
                    },
                )
            )


# ============================================================================
# DISTRIBUTION FUNCTION
# ============================================================================


def _select_training_devices(num_gpus: Optional[int]) -> List[jax.Device]:
    try:
        gpu_devices = list(jax.devices("gpu"))
    except RuntimeError:
        gpu_devices = []
    all_devices = gpu_devices if gpu_devices else list(jax.devices())
    if not all_devices:
        raise RuntimeError("No JAX devices available")
    if num_gpus is None:
        return all_devices
    if num_gpus > len(all_devices):
        raise ValueError(
            f"Requested {num_gpus} GPUs but only {len(all_devices)} available"
        )
    return all_devices[:num_gpus]


def distribute_gradient_computation_from_pairs(
    device_tasks: List[List[WoDatasetGradientTask]],
    num_gpus: Optional[int],
    threads_per_gpu: int,
    grad_fn_cache: Dict[str, Callable],
    problem_configs: Dict[str, Any],
) -> Tuple[Dict[int, Dict], Dict[str, List[float]], int]:
    """Distribute gradient computation across devices with multiple threads per GPU."""
    devices = _select_training_devices(num_gpus)
    total_tasks = sum(len(tasks) for tasks in device_tasks)
    total_problems = sum(sum(t.batch_size for t in tasks) for tasks in device_tasks)

    if total_tasks == 0:
        return {}, {}, 0

    active_device_indices = [i for i, tasks in enumerate(device_tasks) if tasks]
    max_workers = sum(
        min(threads_per_gpu, len(device_tasks[i])) for i in active_device_indices
    )

    logger.info(
        "Launching %d workers across %d/%d devices (threads_per_gpu=%d, %d tasks, %d pairs)",
        max_workers,
        len(active_device_indices),
        len(devices),
        threads_per_gpu,
        total_tasks,
        total_problems,
    )

    result_queue: Queue = Queue()
    threads = []

    for dev_idx in active_device_indices:
        device = devices[dev_idx]
        tasks = device_tasks[dev_idx]
        workers_for_device = min(threads_per_gpu, len(tasks))

        task_queue: Queue = Queue()
        for task in tasks:
            task_queue.put(task)
        for _ in range(workers_for_device):
            task_queue.put(None)

        for _ in range(workers_for_device):
            worker_id = len(threads)
            thread = threading.Thread(
                target=compute_gradient_worker_from_pairs,
                args=(
                    worker_id,
                    device,
                    task_queue,
                    result_queue,
                    grad_fn_cache,
                    problem_configs,
                ),
                daemon=False,
            )
            thread.start()
            threads.append(thread)

    results: Dict[int, Dict] = {}
    step_losses: Dict[str, List[float]] = {}
    first_error = None
    num_errors = 0

    for _ in range(total_tasks):
        task_id, result = result_queue.get()
        if "error" in result:
            logger.warning(
                "Worker %s failed (problem=%s): %s",
                result.get("worker_id"),
                result.get("problem_name"),
                result.get("error"),
            )
            if first_error is None:
                first_error = result
            num_errors += 1
        else:
            results[task_id] = result
            for pr in result.get("per_problem_results", []):
                step_losses.setdefault(pr["nickname"], []).append(pr["loss"])

    for thread in threads:
        thread.join()

    if not results:
        raise RuntimeError(
            f"All gradient workers failed — cannot update model. "
            f"First error: {first_error['error'] if first_error else 'unknown'}"
        )

    if num_errors > 0:
        logger.warning(
            "%d/%d tasks had errors and were skipped. Proceeding with %d successful.",
            num_errors,
            total_tasks,
            len(results),
        )

    return results, step_losses, total_problems


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================


def training_multigpu_wo_dataset_cached(
    model_manager: ModelManager,
    problem_manager: ProblemManager,
    multigpu_config: MultiGPUTrainingConfig,
    training_config: TrainingConfig,
    model_name: str,
    base_dir: str,
    load_model: bool = False,
    load_model_nan: bool = False,
    description: str = "",
):
    """Multi-GPU training on fixed problems (no H5 dataset) with cached JIT compilation.

    Args:
        model_manager: Handles checkpoint / metadata saving
        problem_manager: Provides training pairs via get_training_pairs()
        multigpu_config: Multi-GPU configuration
        training_config: Training hyperparameters
        model_name: Model name (for logging / saving)
        base_dir: Output base directory
        load_model: Load existing model params
        load_model_nan: Load from NaN checkpoint
        description: Run description for metadata
    """
    start_time = timer()
    total_epochs = int(np.sum(training_config.epochs_per_time))

    logger.info("\n" + "=" * 80)
    logger.info(f"Starting Multi-GPU Training (no dataset): {model_name}")
    logger.info(f"Config: {multigpu_config}")
    logger.info(f"Epochs: {total_epochs} | GPUs: {multigpu_config.num_gpus}")
    logger.info("=" * 80 + "\n")

    # Build training pairs (runs HR simulations and caches .npy files)
    logger.info("Building training pairs (may run HR simulations)...")
    training_pairs = problem_manager.get_training_pairs()
    logger.info(f"  {len(training_pairs)} training pair(s) ready")
    for pair in training_pairs:
        logger.info(f"    · {pair.problem_descriptor.nickname}")

    # Use first pair to determine model architecture
    first_pair = training_pairs[0]
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
        logger.info("Using FiLMCorrectorCNN")
    else:
        model = CorrectorCNN(**_model_kwargs)
        logger.info("Using CorrectorCNN")

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

    # ========================================================================
    # PRE-COMPILATION PHASE
    # ========================================================================
    logger.info("\n" + "-" * 40)
    logger.info("PRE-COMPILATION PHASE")
    logger.info("-" * 40)

    grad_fn_cache, problem_configs = precompile_grad_fns_from_pairs(
        training_pairs=training_pairs,
        training_config=training_config,
        cnn_mhd_corrector_config=cnn_mhd_corrector_config,
        noise_levels=PROBLEM_NOISE_LEVELS,
    )

    warmup_grad_fn_cache_from_pairs(
        grad_fn_cache=grad_fn_cache,
        training_pairs=training_pairs,
        network_params=neural_net_params,
    )

    logger.info("-" * 40 + "\n")
    # ========================================================================

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

    successful_training = True

    # Save multigpu config upfront
    multigpu_config_path = (
        model_manager.base_dir / model_manager.model_name / "multigpu_config.json"
    )
    with open(multigpu_config_path, "w") as f:
        json.dump(asdict(multigpu_config), f, indent=2)

    try:
        key = jax.random.PRNGKey(112)
        early_stopper.reset_patience()

        for epoch in range(training_config.epochs_per_time[0]):
            key, batch_key = jax.random.split(key)
            start_time_epoch = timer()

            logger.info(
                f"\nEpoch {epoch + 1}/{training_config.epochs_per_time[0]} | "
                f"{len(training_pairs)} pair(s)"
            )

            num_devices = multigpu_config.num_gpus or len(
                _select_training_devices(None)
            )

            device_tasks = assign_tasks_to_devices_from_pairs(
                training_pairs=training_pairs,
                num_devices=num_devices,
                network_params=trained_params,
                key=batch_key,
                threads_per_gpu=multigpu_config.threads_per_gpu,
            )

            results, step_losses, total_problems = (
                distribute_gradient_computation_from_pairs(
                    device_tasks=device_tasks,
                    num_gpus=multigpu_config.num_gpus,
                    threads_per_gpu=multigpu_config.threads_per_gpu,
                    grad_fn_cache=grad_fn_cache,
                    problem_configs=problem_configs,
                )
            )

            # Collect ALL individual gradients for conFIG
            all_individual_grads = []
            total_grad_mod = 0.0
            all_losses = []

            for device_task_list in device_tasks:
                for task in device_task_list:
                    task_id = id(task)
                    if task_id not in results:
                        continue
                    result = results[task_id]
                    all_individual_grads.extend(result["individual_grads"])
                    total_grad_mod += sum(
                        pr["grad_mod"] for pr in result["per_problem_results"]
                    )
                    logger.info(
                        f"  {result['problem_name']} (n={result['batch_size']}): "
                        f"avg_loss={result['avg_loss']:.6f}, avg_grad_mod={result['avg_grad_mod']:.3f}"
                    )
                    for pr in result.get("per_problem_results", []):
                        all_losses.append(pr["loss"])

            # Move CPU-offloaded grads back to device for aggregation
            device_grads = [jax.device_put(g) for g in all_individual_grads]

            if multigpu_config.use_config_gradient:
                t0 = timer()
                grads = conFIG(device_grads, use_least_square=True)
                logger.info(
                    f"conFIG gradient averaging ({len(device_grads)} grads) "
                    f"took {timer() - t0:.3f}s"
                )
            else:
                num_grads = len(device_grads)
                grads = jax.tree.map(lambda *gs: sum(gs) / num_grads, *device_grads)

            # Optimizer step
            updates, opt_state = optimizer.update(grads, opt_state, trained_params)
            trained_params = eqx.apply_updates(trained_params, updates)

            avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
            avg_grad_mod = (
                total_grad_mod / total_problems if total_problems > 0 else 0.0
            )
            losses_avg.append(avg_loss)

            for nickname, loss_list in step_losses.items():
                losses_per_problem.setdefault(nickname, []).extend(loss_list)

            if early_stopper.new_epoch(avg_loss):
                logger.info("Early stopping triggered")
                break

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = trained_params
                best_epoch = epoch + 1
                logger.info(f"✓ New best loss: {best_loss:.6f}")

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

            logger.info(
                f"Epoch {epoch + 1}: loss={avg_loss:.6f}, "
                f"grad_mod={avg_grad_mod:.3f}, time={timer() - start_time_epoch:.2f}s"
            )

        successful_training = True

    except Exception as e:
        logger.error(f"Training failed: {e}")
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

    model_manager.save_losses(losses_avg)

    losses_save = {"avg": np.array(losses_avg)}
    for name, loss_list in losses_per_problem.items():
        losses_save[name] = np.array(loss_list)
    losses_path = (
        model_manager.base_dir / model_manager.model_name / "losses_per_problem.npz"
    )
    np.savez(losses_path, **losses_save, allow_pickle=True)
    logger.info(f"Losses saved to {losses_path}")

    model_manager.save_model_params(best_params, "best_model_params.eqx")

    metadata = ModelMetadata(
        model_name=model_name,
        created_at=datetime.datetime.now().isoformat(),
        total_epochs=total_epochs,
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

    # Refresh multigpu config with final state
    with open(multigpu_config_path, "w") as f:
        json.dump(asdict(multigpu_config), f, indent=2)
    logger.info(f"MultiGPU config saved to {multigpu_config_path}")

    return best_params, neural_net_static


# ============================================================================
# ENTRY POINT
# ============================================================================


def train_wo_dataset_multigpu(
    problem_descriptors: List[ProblemDescriptor],
    multigpu_config: Optional[MultiGPUTrainingConfig] = None,
    model_name: Optional[str] = None,
    base_dir: str = "arena/data/models/multiproblem_wo_dataset",
    load_existing: bool = False,
    load_existing_nan: bool = False,
    description: str = "",
    **training_config_overrides,
):
    """Main entry point for no-dataset multi-GPU training with cached JIT compilation.

    Args:
        problem_descriptors: List of ProblemDescriptor defining the fixed training set
        multigpu_config: Multi-GPU configuration (defaults to 1 GPU, 2 threads)
        model_name: Output model name (auto-generated if None)
        base_dir: Output base directory (e.g. 'arena/data/models/experiment/azimuth_generalized/normal')
        load_existing: Resume from existing checkpoint
        load_existing_nan: Resume from NaN checkpoint
        description: Notes for metadata
        **training_config_overrides: Override any TrainingConfig fields
    """
    if multigpu_config is None:
        multigpu_config = MultiGPUTrainingConfig(
            problem_counts={d.name: 1 for d in problem_descriptors},
            num_gpus=STARTUP_NUM_GPUS,
            threads_per_gpu=2,
        )

    model_manager = ModelManager(base_dir=base_dir, model_name=model_name)

    if load_existing and model_name:
        logger.info(f"Loading existing config for model {model_name}")
        training_config = model_manager.load_training_config()
    else:
        model_name = model_manager.create_model_directory()
        logger.info(f"Created model directory: {model_name}")

        training_config = TrainingConfig(
            epochs_per_time=[300],
            snapshot_timepoints_train=[0.666],
        )
        for key, value in training_config_overrides.items():
            if hasattr(training_config, key):
                setattr(training_config, key, value)
            else:
                logger.warning(f"Unknown training config field: {key}")

        training_config.model_name = model_name
        model_manager.save_training_config(training_config)

    problem_manager = ProblemManager(
        problem_descriptors=problem_descriptors,
        training_config=training_config,
    )

    # Save problem descriptors to model directory
    problem_manager.save_problem_descriptors(
        model_name=model_name,
        base_dir=model_manager.base_dir,
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Base dir: {base_dir}")
    logger.info(f"Problems: {[d.nickname for d in problem_descriptors]}")
    logger.info(
        f"Architecture: {training_config.hidden_channels}ch x {training_config.hidden_layers}L | "
        f"normalize_input={training_config.normalize_input} | "
        f"use_film={training_config.use_film_corrector}"
    )
    logger.info(
        f"LR: {training_config.learning_rate:.2e} → {training_config.peak_lr:.2e} → {training_config.end_lr:.2e}"
    )
    logger.info(f"conFIG: {multigpu_config.use_config_gradient}")
    logger.info("=" * 70 + "\n")

    return training_multigpu_wo_dataset_cached(
        model_manager=model_manager,
        problem_manager=problem_manager,
        multigpu_config=multigpu_config,
        training_config=training_config,
        model_name=model_name,
        base_dir=base_dir,
        load_model=load_existing,
        load_model_nan=load_existing_nan,
        description=description,
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-GPU no-dataset training with cached JIT compilation"
    )

    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="Pin this process to a specific physical GPU (handled at startup).",
    )
    parser.add_argument("--threads-per-gpu", type=int, default=2)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument(
        "--base-dir",
        type=str,
        default="arena/data/models/multiproblem_wo_dataset",
        help="Output base directory (e.g. arena/data/models/experiment/azimuth_generalized/normal)",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--peak-lr", type=float, default=8e-5)
    parser.add_argument("--end-lr", type=float, default=5e-6)
    parser.add_argument("--warmup-fraction", type=float, default=0.4)
    parser.add_argument("--hidden-channels", type=int, default=5)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--model-scale", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--loss-type",
        type=str,
        default="norm_mse",
        choices=["simple_mse_loss", "norm_mse"],
    )
    parser.add_argument("--normalize-input", action="store_true", default=False)
    parser.add_argument("--use-film", action="store_true", default=False)
    parser.add_argument(
        "--no-config",
        action="store_true",
        default=False,
        help="Disable conFIG gradient averaging (use simple mean)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=40,
        help="Early stopping patience in epochs",
    )
    parser.add_argument("--no-early-stopping", action="store_true", default=False)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--load-existing", action="store_true", default=False)

    # Problem specification: each --problem is a JSON string with 'name' and 'params'
    parser.add_argument(
        "--problem",
        type=str,
        action="append",
        dest="problems",
        help=(
            "Problem specification as JSON: "
            '\'{"name": "mhd_blast", "params": {"B0": 10.0, "B_direction": [0.717, 0, 0.697]}}\'. '
            "Can be repeated for multiple problems."
        ),
    )

    args = parser.parse_args()

    # Parse problem descriptors from JSON
    if args.problems:
        problem_descriptors = []
        for p_str in args.problems:
            p_dict = json.loads(p_str)
            problem_descriptors.append(
                ProblemDescriptor(
                    name=p_dict["name"],
                    params=p_dict.get("params", {}),
                    config_overrides=p_dict.get("config_overrides", {}),
                )
            )
    else:
        # Default: three azimuth blasts (phi=0, theta=0.8/pi/2/2.3)
        import math as _math

        phi = _math.pi / 4

        def _b_dir(theta, phi=0.0):
            return [
                float(_math.sin(theta) * _math.cos(phi)),
                float(_math.sin(theta) * _math.sin(phi)),
                float(_math.cos(theta)),
            ]

        problem_descriptors = [
            ProblemDescriptor(
                "mhd_blast", params={"B0": 10.0, "B_direction": _b_dir(0.8)}
            ),
            ProblemDescriptor(
                "mhd_blast", params={"B0": 10.0, "B_direction": _b_dir(_math.pi / 2)}
            ),
            ProblemDescriptor(
                "mhd_blast", params={"B0": 10.0, "B_direction": _b_dir(2.3)}
            ),
        ]

    multigpu_config = MultiGPUTrainingConfig(
        problem_counts={d.name: 1 for d in problem_descriptors},
        num_gpus=args.num_gpus,
        threads_per_gpu=args.threads_per_gpu,
        save_every_epoch=True,
        checkpoint_interval=50,
        checkpoint_override=True,
        track_best_model=True,
        early_stopping_enabled=not args.no_early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        noise_perturbation=True,
        use_config_gradient=not args.no_config,
        problems_per_batch=len(problem_descriptors),
    )

    train_wo_dataset_multigpu(
        problem_descriptors=problem_descriptors,
        multigpu_config=multigpu_config,
        model_name=args.model_name,
        base_dir=args.base_dir,
        load_existing=args.load_existing,
        description=args.description,
        # TrainingConfig overrides
        epochs_per_time=[args.epochs],
        snapshot_timepoints_train=[0.666],
        learning_rate=args.learning_rate,
        peak_lr=args.peak_lr,
        end_lr=args.end_lr,
        warmup_steps_fraction=args.warmup_fraction,
        hidden_channels=args.hidden_channels,
        hidden_layers=args.hidden_layers,
        model_initialization_scale=args.model_scale,
        gradient_clip=args.gradient_clip,
        loss_type=args.loss_type,
        normalize_input=args.normalize_input,
        use_film_corrector=args.use_film,
        use_early_stopper=not args.no_early_stopping,
        patience=args.early_stopping_patience,
        conFIG=not args.no_config,
        c_cfl=0.8,
        c_cfl_target=0.8,
        t_end=0.2,
        correct_from_beggining=True,
        num_cells_high_res=64,
        downaverage_factor=2,
    )
