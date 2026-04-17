"""Per-problem analysis plots for solecito models (mhd_blast).

This script mirrors the intent of multiproblem plot_problem_analysis, but for
the single-problem solecito setup. It loads a trained model from:

    arena/arena_tests/solecito/models/optuna/<model_name>/best_params.eqx

and generates:
1) Final-time density slices (target downsampled HR | LR baseline | LR+SOL)
2) Diagonal channel comparison (p, rho, |v|, |B|) with HR/LR/LR+SOL
3) Corrector output log-ratio plot
4) L2 loss vs simulation time (LR vs LR+SOL)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Callable, Union

from autocvd import autocvd

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _resolve_startup_num_gpus(default_num_gpus: int = 1) -> int:
    """Resolve --num-gpus before JAX setup so autocvd honors CLI."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-gpus", type=int, default=default_num_gpus)
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args.num_gpus


autocvd(num_gpus=_resolve_startup_num_gpus())

import numpy as np
import optuna
import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import StaticIntVector
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    VectorField,
    VectorFieldCorrectorCNN,
    update_model_vector_field,
)
from arena.arena_tests.solecito.training_model import (
    FullCNNConfig,
    ScalarFieldCNNConfig,
    TrainingConfig,
    VectorFieldCNNConfig,
    downaverage,
    initialize_loss_fn,
    initialize_training_data,
)

logger = logging.getLogger(__name__)

SOLECITO_MODELS_BASE = Path("arena/arena_tests/solecito/models/optuna")
SOLECITO_STUDY_DB = SOLECITO_MODELS_BASE / "optuna_study.db"
SOLECITO_STUDY_NAME = "solecito_optuna"


def _load_study(study_db: Path):
    if not study_db.exists():
        raise FileNotFoundError(f"Optuna DB not found: {study_db}")
    storage = f"sqlite:///{study_db}"
    return optuna.load_study(study_name=SOLECITO_STUDY_NAME, storage=storage)


def _resolve_trial_for_model(study, model_name: str):
    """Resolve the trial hyperparameters to use for a model directory."""
    if model_name == "best_model":
        return study.best_trial

    # First try exact user_attr match (the training run folder name).
    for trial in reversed(study.trials):
        if trial.user_attrs.get("model_name") == model_name:
            return trial

    # Then try matching by optuna trial number encoded in folder name.
    if model_name.startswith("optuna_t"):
        try:
            trial_number = int(model_name.split("_", maxsplit=2)[1][1:])
            for trial in study.trials:
                if trial.number == trial_number:
                    return trial
        except (IndexError, ValueError):
            pass

    raise ValueError(
        f"Could not find a matching Optuna trial for model '{model_name}'."
    )


def _build_training_config(
    trial,
    model_name: str,
    *,
    num_cells_high_res: int,
    downaverage_factor: int,
    t_end: float,
    epochs: int,
) -> Union[ScalarFieldCNNConfig, VectorFieldCNNConfig, FullCNNConfig]:
    """Build a solecito TrainingConfig from Optuna trial hyperparameters."""
    params = trial.params
    model_type = params["model_type"]

    common_kwargs = dict(
        model_name=model_name,
        epochs=int(epochs),
        hidden_layers=int(params["hidden_layers"]),
        hidden_channels=int(params["hidden_channels"]),
        c_cfl=float(params["c_cfl"]),
        learning_rate=float(params["starting_learning_rate"]),
        peak_lr=float(params["peak_lr"]),
        end_lr=float(params["end_lr"]),
        warmup_steps_fraction=float(params["warmup_steps"]),
        model_initialization_scale=float(params["model_initialization_scale"]),
        noise_level=float(params["noise_level"]),
        num_cells_high_res=int(num_cells_high_res),
        downaverage_factor=int(downaverage_factor),
        t_end=float(t_end),
    )

    if model_type == "scalar_p":
        return ScalarFieldCNNConfig(channel_index=4, **common_kwargs)
    if model_type == "scalar_d":
        return ScalarFieldCNNConfig(channel_index=4, **common_kwargs)
    if model_type == "vector_m":
        return VectorFieldCNNConfig(vector_field="magnetic", **common_kwargs)
    if model_type == "vector_v":
        return VectorFieldCNNConfig(vector_field="velocity", **common_kwargs)

    raise ValueError(f"Unsupported model_type in trial params: {model_type}")


def _snapshot_prefix(
    snapshot_data: SnapshotData,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Trim snapshot series at crash/non-finite tail."""
    times = np.asarray(snapshot_data.time_points)
    states = np.asarray(snapshot_data.states)

    n = min(times.shape[0], states.shape[0])
    times = times[:n]
    states = states[:n]
    crashed = False

    if times.size > 1:
        zero_time_indices = np.where(times[1:] == 0.0)[0]
        if zero_time_indices.size > 0:
            first_zero_idx = int(zero_time_indices[0] + 1)
            times = times[:first_zero_idx]
            states = states[:first_zero_idx]
            crashed = True

    finite_mask = np.isfinite(times)
    invalid_idx = np.where(~finite_mask)[0]
    if invalid_idx.size > 0:
        cutoff = int(invalid_idx[0])
        times = times[:cutoff]
        states = states[:cutoff]
        crashed = True

    if times.size == 0 or states.size == 0:
        raise ValueError("No valid snapshots available after trimming crash tail.")

    return times, states, crashed


def _prepare_loss_curve(
    times: np.ndarray,
    losses: np.ndarray,
    *,
    validation_t_end: float,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Trim invalid tails and extend crashed curves to validation horizon."""
    times_arr = np.asarray(times)
    losses_arr = np.asarray(losses)
    n = min(times_arr.size, losses_arr.size)
    times_arr = times_arr[:n]
    losses_arr = losses_arr[:n]
    if n == 0:
        return np.array([]), np.array([]), None

    finite_pair = np.isfinite(times_arr) & np.isfinite(losses_arr)
    if not np.any(finite_pair):
        return np.array([]), np.array([]), None

    invalid_indices = np.where(~finite_pair)[0]
    prefix_end = int(invalid_indices[0]) if invalid_indices.size > 0 else n
    times_plot = times_arr[:prefix_end]
    losses_plot = losses_arr[:prefix_end]

    if times_plot.size == 0:
        return np.array([]), np.array([]), None

    crash_time = None
    if prefix_end < n:
        crash_time = float(times_plot[-1])
        if validation_t_end > times_plot[-1]:
            times_plot = np.concatenate([times_plot, np.array([validation_t_end])])
            losses_plot = np.concatenate([losses_plot, np.array([losses_plot[-1]])])

    return times_plot, losses_plot, crash_time


def _magnetic_magnitude_mean(state: jnp.ndarray, reg_vars) -> jnp.ndarray:
    """Return scalar mean magnetic magnitude for one state snapshot."""
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)
    magnetic = jnp.sqrt(
        state[reg_vars.magnetic_index.x] ** 2
        + state[reg_vars.magnetic_index.y] ** 2
        + state[reg_vars.magnetic_index.z] ** 2
    )
    return jnp.mean(jnp.abs(magnetic))


def _diagonal_profile(state: np.ndarray, reg_vars, component_name: str) -> np.ndarray:
    """Extract a diagonal slice profile at mid-z for a single state snapshot."""
    assert isinstance(reg_vars.velocity_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)
    num_cells = state.shape[1]
    diag = np.arange(num_cells)
    z_mid = state.shape[-1] // 2

    if component_name == "density":
        field = state[reg_vars.density_index]
    elif component_name == "pressure":
        field = state[reg_vars.pressure_index]
    elif component_name == "velocity":
        field = np.sqrt(
            state[reg_vars.velocity_index.x] ** 2
            + state[reg_vars.velocity_index.y] ** 2
            + state[reg_vars.velocity_index.z] ** 2
        )
    elif component_name == "magnetic":
        field = np.sqrt(
            state[reg_vars.magnetic_index.x] ** 2
            + state[reg_vars.magnetic_index.y] ** 2
            + state[reg_vars.magnetic_index.z] ** 2
        )
    else:
        raise ValueError(f"Unsupported component: {component_name}")

    return field[diag, diag, z_mid]


def _snapshot_callback_factory_vector_magnetic(
    *,
    effective_corrections: list[float],
    states: list[float],
    times: list[float],
    simulation_config,
    registered_variables,
    floor: float,
) -> Callable[..., Any]:
    """Capture callback data for vector_m model output diagnostics."""

    def snapshot_callable(time_step, state, correction):
        states.append(float(_magnetic_magnitude_mean(state, registered_variables)))
        times.append(float(time_step))
        corrected_state = update_model_vector_field(
            primitive_state=state,
            correction=correction,
            time_step=time_step,
            config=simulation_config,
            registered_variables=registered_variables,
            field_type=VectorField.MAGNETIC,
        )
        state_delta = corrected_state - state
        effective_delta = _magnetic_magnitude_mean(state_delta, registered_variables)
        dt_safe = jnp.maximum(jnp.abs(time_step), floor)
        # Store effective correction as a rate so dt * corr corresponds to state delta.
        effective_corrections.append(float(effective_delta / dt_safe))

    return snapshot_callable


def _prepare_corrector_log_ratio(
    *,
    times: list[float],
    effective_corrections: list[float],
    states: list[float],
    validation_t_end: float,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Build crash-safe log-ratio curves for corrector output diagnostics."""
    if not times or not effective_corrections or not states:
        return np.array([]), np.array([]), False

    times_delta = np.asarray(times, dtype=float)
    effective_arr = np.asarray(effective_corrections, dtype=float)
    states_arr = np.asarray(states, dtype=float)
    n = min(times_delta.shape[0], effective_arr.shape[0], states_arr.shape[0])
    times_delta = times_delta[:n]
    effective_arr = effective_arr[:n]
    states_arr = states_arr[:n]

    times_total = np.cumsum(times_delta)
    numerator = np.abs(times_delta * effective_arr)
    denominator = np.abs(states_arr) + floor
    ratio = np.maximum(numerator / denominator, floor)
    log_ratio = np.log10(ratio)

    finite_mask = np.isfinite(times_total)
    finite_mask &= np.isfinite(log_ratio)

    finite_indices = np.where(finite_mask)[0]
    if finite_indices.size == 0:
        return np.array([]), np.array([]), False

    last_finite_idx = int(finite_indices[-1])
    times_plot = times_total[: last_finite_idx + 1]
    log_ratio_plot = log_ratio[: last_finite_idx + 1]
    crashed = finite_indices.size < len(times_total)

    if crashed and validation_t_end > times_plot[-1]:
        times_plot = np.concatenate(
            [times_plot, np.array([validation_t_end], dtype=times_plot.dtype)]
        )
        log_ratio_plot = np.concatenate(
            [log_ratio_plot, np.array([log_ratio_plot[-1]])]
        )

    return times_plot, log_ratio_plot, crashed


def plot_solecito_problem_analysis(
    *,
    model_name: str,
    models_base_dir: Path = SOLECITO_MODELS_BASE,
    num_cells_high_res: int,
    downaverage_factor: int,
    num_snapshots: int,
    validation_time_multiplier: float,
    t_end: float,
    ratio_floor: float,
) -> Path:
    """Generate mhd_blast per-problem analysis plot for a solecito model."""
    model_dir = models_base_dir / model_name
    params_path = model_dir / "best_params.eqx"
    study_db = models_base_dir / "optuna_study.db"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing model params file: {params_path}")

    study = _load_study(study_db)
    trial = _resolve_trial_for_model(study, model_name)
    model_type = str(trial.params["model_type"])
    if model_type != "vector_m":
        raise ValueError(
            f"Model '{model_name}' has model_type='{model_type}'. "
            "This plot currently supports only vector_m and fails fast otherwise."
        )
    epochs = int(trial.user_attrs.get("epochs_run", 300))
    training_config = _build_training_config(
        trial=trial,
        model_name=model_name,
        num_cells_high_res=num_cells_high_res,
        downaverage_factor=downaverage_factor,
        t_end=t_end,
        epochs=epochs,
    )

    logger.info(
        "Using trial %s (model_type=%s, c_cfl=%.6f, hidden_layers=%d, hidden_channels=%d)",
        trial.number,
        model_type,
        training_config.c_cfl,
        training_config.hidden_layers,
        training_config.hidden_channels,
    )

    validation_t_end = float(training_config.t_end * validation_time_multiplier)
    times_eval = jnp.linspace(0.0, validation_t_end, endpoint=True, num=num_snapshots)
    if not jnp.any(times_eval == training_config.t_end):
        times_eval = jnp.sort(
            jnp.concatenate([times_eval, jnp.array([training_config.t_end])])
        )

    trained_t_idx = int(jnp.argmin(jnp.abs(times_eval - training_config.t_end)))

    (
        (
            initial_state_high_res,
            simulation_config_high_res,
            simulation_params,
            _,
            registered_variables,
        ),
        (
            initial_state_low_res,
            simulation_config_low_res,
            simulation_params,
            _,
            registered_variables,
        ),
    ) = initialize_training_data(training_config=training_config)

    simulation_config_high_res = simulation_config_high_res._replace(
        return_snapshots=True,
        num_snapshots=len(times_eval),
        use_specific_snapshot_timepoints=True,
        progress_bar=True,
    )
    simulation_config_low_res = simulation_config_low_res._replace(
        return_snapshots=True,
        num_snapshots=len(times_eval),
        use_specific_snapshot_timepoints=True,
        progress_bar=True,
    )
    simulation_params = simulation_params._replace(
        snapshot_timepoints=times_eval, t_end=validation_t_end
    )

    hr_snapshot_data = time_integration(
        primitive_state=initial_state_high_res,
        config=simulation_config_high_res,
        params=simulation_params,
        registered_variables=registered_variables,
    )
    assert isinstance(hr_snapshot_data, SnapshotData)
    target_states_low_res = downaverage(
        hr_snapshot_data.states, downaverage_factor=training_config.downaverage_factor
    )

    lr_snapshot_data = time_integration(
        primitive_state=initial_state_low_res,
        config=simulation_config_low_res,
        params=simulation_params,
        registered_variables=registered_variables,
    )
    assert isinstance(lr_snapshot_data, SnapshotData)

    effective_corrections: list[float] = []
    callback_states: list[float] = []
    callback_times: list[float] = []
    snapshot_callable = _snapshot_callback_factory_vector_magnetic(
        effective_corrections=effective_corrections,
        states=callback_states,
        times=callback_times,
        simulation_config=simulation_config_low_res,
        registered_variables=registered_variables,
        floor=ratio_floor,
    )

    callback_model = VectorFieldCorrectorCNN(
        in_channels=registered_variables.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        vector_field_output=VectorField.MAGNETIC,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
        snapshot_callable=snapshot_callable,
    )
    callback_params_template, callback_static = eqx.partition(
        callback_model, eqx.is_array
    )
    trained_network_params = eqx.tree_deserialise_leaves(
        str(params_path), callback_params_template
    )
    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=callback_static,
        correct_from_beggining=True,
        start_correction_time=0.0,
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=trained_network_params)

    simulation_config_low_res_sol = simulation_config_low_res._replace(
        cnn_mhd_corrector_config=cnn_mhd_corrector_config
    )
    simulation_params_low_res_sol = simulation_params._replace(
        cnn_mhd_corrector_params=cnn_mhd_corrector_params._replace(
            network_params=trained_network_params
        )
    )

    lr_sol_snapshot_data = time_integration(
        primitive_state=initial_state_low_res,
        config=simulation_config_low_res_sol,
        params=simulation_params_low_res_sol,
        registered_variables=registered_variables,
    )
    assert isinstance(lr_sol_snapshot_data, SnapshotData)

    lr_times, lr_states, _ = _snapshot_prefix(lr_snapshot_data)
    sol_times, sol_states, _ = _snapshot_prefix(lr_sol_snapshot_data)
    hr_times = np.asarray(hr_snapshot_data.time_points)
    hr_states = np.asarray(target_states_low_res)

    loss_fn = initialize_loss_fn(target_state=target_states_low_res[trained_t_idx])
    v_loss = jax.vmap(loss_fn, in_axes=(0, 0))
    lr_losses = np.asarray(v_loss(lr_states, hr_states[: lr_states.shape[0]]))
    sol_losses = np.asarray(v_loss(sol_states, hr_states[: sol_states.shape[0]]))

    lr_times_plot, lr_losses_plot, lr_crash_time = _prepare_loss_curve(
        lr_times, lr_losses, validation_t_end=float(hr_times[-1])
    )
    sol_times_plot, sol_losses_plot, sol_crash_time = _prepare_loss_curve(
        sol_times, sol_losses, validation_t_end=float(hr_times[-1])
    )

    # Continue from LR+SOL state at training time, then evolve with plain LR.
    lr_after_sol_times_plot = np.array([])
    lr_after_sol_losses_plot = np.array([])
    raw_sol_times = np.asarray(lr_sol_snapshot_data.time_points)
    can_run_post_sol = (
        trained_t_idx < raw_sol_times.shape[0]
        and np.isfinite(raw_sol_times[trained_t_idx])
        and raw_sol_times[trained_t_idx] > 0.0
        and abs(float(raw_sol_times[trained_t_idx]) - training_config.t_end) < 1e-6
    )
    if can_run_post_sol:
        post_abs_times = np.asarray(times_eval[trained_t_idx:], dtype=float)
        if post_abs_times.size > 0:
            post_rel_times = post_abs_times - post_abs_times[0]
            continuation_config = simulation_config_low_res._replace(
                return_snapshots=True,
                num_snapshots=len(post_rel_times),
                use_specific_snapshot_timepoints=True,
            )
            continuation_params = simulation_params._replace(
                snapshot_timepoints=jnp.array(post_rel_times),
                t_end=float(post_rel_times[-1]),
            )
            state_at_training_time = lr_sol_snapshot_data.states[trained_t_idx]
            lr_after_sol_snapshot_data = time_integration(
                primitive_state=state_at_training_time,
                config=continuation_config,
                params=continuation_params,
                registered_variables=registered_variables,
            )
            assert isinstance(lr_after_sol_snapshot_data, SnapshotData)
            post_rel_out = np.asarray(lr_after_sol_snapshot_data.time_points)
            post_states_out = np.asarray(lr_after_sol_snapshot_data.states)
            if post_rel_out.size > 0:
                post_abs_out = (
                    post_rel_out - float(post_rel_out[0]) + training_config.t_end
                )
                hr_post_target = hr_states[trained_t_idx:]
                n_post = min(
                    post_abs_out.shape[0],
                    post_states_out.shape[0],
                    hr_post_target.shape[0],
                )
                post_abs_out = post_abs_out[:n_post]
                post_states_out = post_states_out[:n_post]
                hr_post_target = hr_post_target[:n_post]
                post_losses = np.asarray(v_loss(post_states_out, hr_post_target))
                lr_after_sol_times_plot, lr_after_sol_losses_plot, _ = (
                    _prepare_loss_curve(
                        post_abs_out, post_losses, validation_t_end=float(hr_times[-1])
                    )
                )
    else:
        logger.warning(
            "Skipping LR AFTER SOL continuation: LR+SOL has no valid state at training time "
            "(t_train=%.6f, sol_time_at_train=%s).",
            training_config.t_end,
            (
                float(raw_sol_times[trained_t_idx])
                if trained_t_idx < raw_sol_times.shape[0]
                else "out_of_range"
            ),
        )

    lr_plot_idx = int(np.argmin(np.abs(lr_times - training_config.t_end)))
    sol_plot_idx = int(np.argmin(np.abs(sol_times - training_config.t_end)))
    target_final = hr_states[trained_t_idx]
    lr_final = lr_states[lr_plot_idx]
    sol_final = sol_states[sol_plot_idx]

    density_index = registered_variables.density_index
    z_target = target_final.shape[-1] // 2
    z_lr = lr_final.shape[-1] // 2
    z_sol = sol_final.shape[-1] // 2

    images = [
        target_final[density_index, :, :, z_target],
        lr_final[density_index, :, :, z_lr],
        sol_final[density_index, :, :, z_sol],
    ]
    vmin = min(float(np.min(img)) for img in images)
    vmax = max(float(np.max(img)) for img in images)

    corrector_times, corrector_log_ratio, _ = _prepare_corrector_log_ratio(
        times=callback_times,
        effective_corrections=effective_corrections,
        states=callback_states,
        validation_t_end=float(hr_times[-1]),
        floor=ratio_floor,
    )

    logger.info("Corrector callback samples captured: %d", len(callback_times))
    if callback_times and effective_corrections and callback_states:
        times_delta_np = np.asarray(callback_times, dtype=float)
        eff_np = np.asarray(effective_corrections, dtype=float)
        state_np = np.asarray(callback_states, dtype=float)
        n_stats = min(times_delta_np.shape[0], eff_np.shape[0], state_np.shape[0])
        times_delta_np = times_delta_np[:n_stats]
        eff_np = eff_np[:n_stats]
        state_np = state_np[:n_stats]
        scaled_corr = np.abs(times_delta_np * eff_np)
        denom = np.abs(state_np) + ratio_floor
        ratio = np.maximum(scaled_corr / denom, ratio_floor)
        logger.info(
            "Corrector ratio stats [magnetic]: min=%e max=%e median=%e",
            float(np.nanmin(ratio)),
            float(np.nanmax(ratio)),
            float(np.nanmedian(ratio)),
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "problem_analysis_mhd_blast.png"

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(
        4, 1, figure=fig, height_ratios=[1.1, 1.0, 1.0, 1.0], hspace=0.45, wspace=0.2
    )

    # Row 1: image comparison
    gs_images = gs[0].subgridspec(1, 3, wspace=0.3)

    titles = [
        "Target (HR downsampled) density",
        "LR baseline density",
        "LR + SOL density",
    ]
    for i, (img, title) in enumerate(zip(images, titles, strict=True)):
        ax = fig.add_subplot(gs_images[0, i])
        im = ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Row 2: diagonal channel comparisons (HR vs LR vs LR+SOL in same panel)
    gs_diag = gs[1].subgridspec(1, 4, wspace=0.35)
    channel_titles = {
        "pressure": "Pressure diagonal slice",
        "density": "Density diagonal slice",
        "velocity": "Velocity magnitude diagonal slice",
        "magnetic": "Magnetic magnitude diagonal slice",
    }
    comparison_colors = {"HR": "black", "LR": "tab:orange", "LR+SOL": "tab:blue"}
    num_cells_low_res = target_final.shape[1]
    diag_indices = np.arange(num_cells_low_res)
    r_diag = np.sqrt(diag_indices**2 + diag_indices**2) * (
        simulation_config_low_res.box_size / num_cells_low_res
    )
    for ax_idx, component_name in enumerate(
        ("pressure", "density", "velocity", "magnetic")
    ):
        ax_diag = fig.add_subplot(gs_diag[0, ax_idx])
        target_profile = _diagonal_profile(
            target_final, registered_variables, component_name
        )
        lr_profile = _diagonal_profile(lr_final, registered_variables, component_name)
        sol_profile = _diagonal_profile(sol_final, registered_variables, component_name)
        ax_diag.plot(r_diag, target_profile, color=comparison_colors["HR"], label="HR")
        ax_diag.plot(r_diag, lr_profile, color=comparison_colors["LR"], label="LR")
        ax_diag.plot(
            r_diag,
            sol_profile,
            color=comparison_colors["LR+SOL"],
            label="LR+SOL",
        )
        ax_diag.set_title(channel_titles[component_name], fontsize=10)
        ax_diag.set_xlabel("Diagonal position")
        ax_diag.grid(alpha=0.3)
        if component_name in ("pressure", "density"):
            ax_diag.set_ylabel("Value")
        else:
            ax_diag.set_ylabel("Magnitude")
        if ax_idx == 0:
            ax_diag.legend(fontsize=8)

    # Row 3: corrector output (log-ratio)
    ax_corrector = fig.add_subplot(gs[2])
    if corrector_times.size > 0:
        ax_corrector.plot(
            corrector_times,
            corrector_log_ratio,
            color="tab:purple",
            label="magnetic",
        )
        ax_corrector.axhline(
            y=np.log10(ratio_floor),
            color="gray",
            linestyle="--",
            alpha=0.8,
            label=f"log10 floor ({ratio_floor:.1e})",
        )
    else:
        ax_corrector.text(
            0.5,
            0.5,
            "No callback corrector-output data available",
            ha="center",
            va="center",
            transform=ax_corrector.transAxes,
        )
    ax_corrector.axvline(
        x=training_config.t_end,
        color="gray",
        linestyle=":",
        alpha=0.8,
        label=f"trained t={training_config.t_end:.3f}",
    )
    ax_corrector.set_title(
        "Corrector output (magnetic): log10(abs(dt * effective_correction)) - log10(abs(state_mean) + floor)"
    )
    ax_corrector.set_xlabel("Time")
    ax_corrector.set_ylabel("Log ratio")
    ax_corrector.legend(fontsize=8)
    ax_corrector.grid(alpha=0.3)

    # Row 4: simulation L2 loss
    ax_sim = fig.add_subplot(gs[3])
    if lr_times_plot.size > 0:
        ax_sim.plot(lr_times_plot, lr_losses_plot, "--", color="tab:orange", label="LR")
    if sol_times_plot.size > 0:
        ax_sim.plot(sol_times_plot, sol_losses_plot, color="tab:blue", label="LR+SOL")
    if lr_after_sol_times_plot.size > 0:
        ax_sim.plot(
            lr_after_sol_times_plot,
            lr_after_sol_losses_plot,
            "--",
            color="tab:red",
            label="LR AFTER SOL",
        )
    if lr_crash_time is not None and lr_times_plot.size > 0:
        ax_sim.plot(
            [lr_crash_time],
            [lr_losses_plot[-1]],
            marker="x",
            color="tab:orange",
            linestyle="None",
        )
    if sol_crash_time is not None and sol_times_plot.size > 0:
        ax_sim.plot(
            [sol_crash_time],
            [sol_losses_plot[-1]],
            marker="x",
            color="tab:blue",
            linestyle="None",
        )
    ax_sim.axvline(
        x=training_config.t_end,
        color="gray",
        linestyle=":",
        alpha=0.8,
        label=f"trained t={training_config.t_end:.3f}",
    )
    ax_sim.set_title("L2 loss vs simulation time")
    ax_sim.set_xlabel("Time")
    ax_sim.set_ylabel("L2 loss")
    ax_sim.legend()

    fig.suptitle(
        f"Solecito mhd_blast analysis | model={model_name} | c_cfl={training_config.c_cfl:.6f}",
        fontsize=13,
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved per-problem analysis plot to %s", out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-problem mhd_blast analysis plots for solecito models."
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="best_model",
        help="Model name to process (default: best_model)",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name_override",
        type=str,
        default=None,
        help="Optional override for model name (same as positional model_name).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(SOLECITO_MODELS_BASE),
        help="model parent folder",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-cells-high-res", type=int, default=32)
    parser.add_argument("--downaverage-factor", type=int, default=2)
    parser.add_argument("--num-snapshots", type=int, default=70)
    parser.add_argument("--validation-time-multiplier", type=float, default=3.5)
    parser.add_argument("--t-end", type=float, default=0.2)
    parser.add_argument("--ratio-floor", type=float, default=1e-10)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
    )

    model_name = args.model_name_override or args.model_name
    plot_solecito_problem_analysis(
        model_name=model_name,
        models_base_dir=Path(args.model_path),
        num_cells_high_res=args.num_cells_high_res,
        downaverage_factor=args.downaverage_factor,
        num_snapshots=args.num_snapshots,
        validation_time_multiplier=args.validation_time_multiplier,
        t_end=args.t_end,
        ratio_floor=args.ratio_floor,
    )


if __name__ == "__main__":
    main()
