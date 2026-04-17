from autocvd import autocvd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# autocvd(num_gpus=1)
"""
Multiproblem training summary plot.

Per-problem analysis layout (4 rows, plus OT vortex div(B) error row):
  Row 0: density slice images (HR downsampled | LR | LR+SOL)
  Row 1: diagonal channel comparisons (density, pressure, |v|, |B|)
  Row 2: normalized model-output diagnostics with floor reference
  Row 3: L2 loss vs simulation time
  Row 4 (ot_vortex only): mean |div(B) * dx / B| vs simulation time

plot_losses() produces a standalone 2-row figure:
  Row 0: per-problem loss evolution per epoch (one subplot per problem)
  Row 1: average loss evolution per epoch (full width)
"""
import logging
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix import (
    get_registered_variables,
    SimulationConfig,
    time_integration,
    finalize_config,
)
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import StaticIntVector
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from astronomix._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    YAXIS,
    XAXIS,
    ZAXIS,
)
from astronomix._finite_difference._maths._differencing import (
    _interface_field_divergence,
)
from astronomix._finite_difference._maths._interpolate import interp_face_to_center

from arena.arena_tests.solver_in_the_loop.utils import downaverage
from arena.arena_tests.solver_in_the_loop.loss import loss_setup
from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    TrainingConfig,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
    EvaluationResults,
    ProblemManager,
    PROBLEM_CATALOG,
    PROBLEM_DICTIONARY,
    ProblemDescriptor,
)
from arena.arena_tests.solver_in_the_loop.plot_states_comparison import plot_states
from typing import Any, Callable, Optional, List

logger = logging.getLogger(__name__)

MULTIPROBLEM_BASE_DIR = "arena/data/models/multiproblem_w_dataset"

NUM_SNAPSHOTS = 70

SNAPSHOT_TIMEPOINTS_VALIDATION = {
    "turbulence": jnp.linspace(0.0, 1.4 * 0.4, endpoint=True, num=NUM_SNAPSHOTS),
    "ot_vortex": jnp.linspace(0.0, 3.5 * jnp.pi, endpoint=True, num=NUM_SNAPSHOTS),
    "mhd_blast": jnp.linspace(0.0, 3.5 * 0.2, endpoint=True, num=NUM_SNAPSHOTS),
    "turbulent_blast": jnp.linspace(0.0, 1.5 * 0.4, endpoint=True, num=NUM_SNAPSHOTS),
}


def _get_reg_vars():
    """Registered variables for a standard 3D MHD finite-difference simulation."""
    return get_registered_variables(
        SimulationConfig(dimensionality=3, mhd=True, solver_mode=FINITE_DIFFERENCE)
    )


def _compute_sim_losses(
    evaluation_data: list[EvaluationResults],
    training_config: TrainingConfig,
    trained_t_end: dict[str, float],
    snapshot_data_after_lr_sol: Optional[List[SnapshotData]] = None,
) -> list[dict]:
    """Per-snapshot L2 losses for LR and LR+SOL against the downsampled HR target.

    Normalisation in loss_setup uses the downsampled HR state at the trained
    time (problem.t_end), consistent with how the model was trained.
    """
    results = []
    for i, data in enumerate(evaluation_data):
        hr_states = data.hr_snapshot_data.states
        hr_times = np.array(data.hr_snapshot_data.time_points)
        lr_target_states = downaverage(hr_states, training_config.downaverage_factor)

        # Find the snapshot closest to the trained time for loss normalisation
        t_train = trained_t_end[data.problem_descriptor.name]
        train_idx = int(np.argmin(np.abs(hr_times - t_train)))

        loss_fn_kwargs, loss_fn_factory = loss_setup(
            training_config=training_config,
            target_states=lr_target_states[train_idx],
        )
        loss_fn = partial(loss_fn_factory, **loss_fn_kwargs)
        v_loss = jax.vmap(loss_fn, in_axes=(0, 0))

        lr_losses = None
        sol_losses = None
        lr_times = None
        sol_times = None

        if data.lr_snapshot_data is not None:
            lr_states = data.lr_snapshot_data.states
            lr_losses = np.array(v_loss(lr_states, lr_target_states))
            lr_times = np.array(data.lr_snapshot_data.time_points)

        if data.lr_sol_snapshot_data is not None:
            sol_states = data.lr_sol_snapshot_data.states
            sol_losses = np.array(v_loss(sol_states, lr_target_states))
            sol_times = np.array(data.lr_sol_snapshot_data.time_points)
            if sol_times.size > 1:
                # In crash cases, post-crash snapshots are zero-padded in time.
                # Trim LR+SOL losses/times at the first zero after the initial t=0.
                zero_time_indices = np.where(sol_times[1:] == 0)[0]
                if zero_time_indices.size > 0:
                    first_zero_idx = int(zero_time_indices[0] + 1)
                    sol_times = sol_times[:first_zero_idx]
                    sol_losses = sol_losses[:first_zero_idx]

        if snapshot_data_after_lr_sol is not None:
            lr_after_sol_times_raw = np.array(snapshot_data_after_lr_sol[i].time_points)
            if lr_after_sol_times_raw.size > 0:
                lr_after_sol_times = (
                    lr_after_sol_times_raw - float(lr_after_sol_times_raw[0]) + t_train
                )
            else:
                lr_after_sol_times = lr_after_sol_times_raw

            lr_after_sol_states = snapshot_data_after_lr_sol[i].states
            lr_after_sol_losses = np.array(
                v_loss(lr_after_sol_states, lr_target_states[train_idx:])
            )
        else:
            lr_after_sol_times = None
            lr_after_sol_losses = None

        results.append(
            {
                "name": data.problem_descriptor.name,
                "lr_times": lr_times,
                "lr_losses": lr_losses,
                "sol_times": sol_times,
                "sol_losses": sol_losses,
                "lr_after_sol_times": lr_after_sol_times,
                "lr_after_sol_losses": lr_after_sol_losses,
            }
        )
    return results


def _trim_snapshot_prefix(
    snapshot_data: SnapshotData,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim zero-padded/non-finite tails and return aligned (times, states)."""
    times = np.asarray(snapshot_data.time_points)
    states = np.asarray(snapshot_data.states)
    n = min(times.shape[0], states.shape[0])
    times = times[:n]
    states = states[:n]

    if times.size > 1:
        zero_time_indices = np.where(times[1:] == 0.0)[0]
        if zero_time_indices.size > 0:
            first_zero_idx = int(zero_time_indices[0] + 1)
            times = times[:first_zero_idx]
            states = states[:first_zero_idx]

    finite_time_indices = np.where(~np.isfinite(times))[0]
    if finite_time_indices.size > 0:
        cutoff = int(finite_time_indices[0])
        times = times[:cutoff]
        states = states[:cutoff]

    if states.size == 0:
        return times, states

    finite_state_mask = np.isfinite(states).reshape(states.shape[0], -1).all(axis=1)
    invalid_state_indices = np.where(~finite_state_mask)[0]
    if invalid_state_indices.size > 0:
        cutoff = int(invalid_state_indices[0])
        times = times[:cutoff]
        states = states[:cutoff]

    return times, states


def _mean_abs_normalized_magnetic_divergence_error(
    states: np.ndarray, *, reg_vars, grid_spacing: float
) -> np.ndarray:
    assert isinstance(reg_vars.interface_magnetic_field_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)
    mean_abs_normalized_divergence_error: list[float] = []
    eps = 1e-12
    for state in states:
        div_b = _interface_field_divergence(
            state[reg_vars.interface_magnetic_field_index.x],
            state[reg_vars.interface_magnetic_field_index.y],
            state[reg_vars.interface_magnetic_field_index.z],
            grid_spacing,
        )
        b_magnitude = jnp.sqrt(
            state[reg_vars.interface_magnetic_field_index.x] ** 2
            + state[reg_vars.interface_magnetic_field_index.y] ** 2
            + state[reg_vars.interface_magnetic_field_index.z] ** 2
        )
        normalized_divergence_error = (
            jnp.abs(div_b) * grid_spacing / jnp.maximum(jnp.abs(b_magnitude), eps)
        )
        mean_abs_normalized_divergence_error.append(
            float(jnp.mean(normalized_divergence_error))
        )
    return np.asarray(mean_abs_normalized_divergence_error, dtype=float)


def _compute_normalized_magnetic_divergence_error_series(
    snapshot_data: Optional[SnapshotData],
    *,
    reg_vars,
    grid_spacing: float,
    time_offset: Optional[float] = None,
    max_time: Optional[float] = None,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if snapshot_data is None:
        return None, None

    times, states = _trim_snapshot_prefix(snapshot_data)
    if times.size == 0 or states.size == 0:
        return None, None

    if time_offset is not None:
        times = times - float(times[0]) + float(time_offset)

    divergence_values = _mean_abs_normalized_magnetic_divergence_error(
        states, reg_vars=reg_vars, grid_spacing=grid_spacing
    )

    n = min(times.size, divergence_values.size)
    times = times[:n]
    divergence_values = divergence_values[:n]

    finite_pair = np.isfinite(times) & np.isfinite(divergence_values)
    if not np.any(finite_pair):
        return None, None
    invalid_indices = np.where(~finite_pair)[0]
    prefix_end = int(invalid_indices[0]) if invalid_indices.size > 0 else n
    times = times[:prefix_end]
    divergence_values = divergence_values[:prefix_end]
    if times.size == 0:
        return None, None

    if max_time is not None:
        valid_mask = times <= max_time
        if not np.any(valid_mask):
            return None, None
        times = times[valid_mask]
        divergence_values = divergence_values[valid_mask]

    return times, divergence_values


def _diagonal_profile(state: np.ndarray, reg_vars, component_name: str) -> np.ndarray:
    """Extract a diagonal profile at mid-z for one snapshot state."""
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


def _snapshot_callback_factory(
    *,
    corrections: list[Array],
    effective_corrections: list[Array],
    states: list[Array],
    time_deltas: list[float],
    nan_states_before_correction: list[Array],
    nan_states_after_correction: list[Array],
    crash_times: list[float],
    crash_step_indices: list[int],
):
    reg_vars = _get_reg_vars()
    params = SimulationParams()
    step_counter = [0]  # Use list to allow mutation in nested function

    def snapshot_callable(time, state, correction):
        current_step = step_counter[0]
        step_counter[0] += 1

        # Use spatial magnitudes to avoid cancellation of signed corrections
        # (especially for curl-based magnetic interface updates).
        corrections.append(jnp.mean(jnp.abs(correction), axis=[1, 2, 3]))
        states.append(jnp.mean(jnp.abs(state), axis=[1, 2, 3]))
        time_deltas.append(float(time))

        original_state = state
        corrected_state = state
        corrected_state = corrected_state.at[:5].add(correction[:5] * time)
        corrected_state = corrected_state.at[-3:].add(correction[-3:] * time)

        bx_center = interp_face_to_center(corrected_state[-3], XAXIS)
        by_center = interp_face_to_center(corrected_state[-2], YAXIS)
        bz_center = interp_face_to_center(corrected_state[-1], ZAXIS)

        corrected_state = corrected_state.at[reg_vars.magnetic_index.x].set(bx_center)
        corrected_state = corrected_state.at[reg_vars.magnetic_index.y].set(by_center)
        corrected_state = corrected_state.at[reg_vars.magnetic_index.z].set(bz_center)

        # Store pre-clamp state for diagnostics (before jnp.maximum masks issues)
        pre_clamp_state = np.asarray(corrected_state)

        corrected_state = corrected_state.at[reg_vars.pressure_index].set(
            jnp.maximum(
                corrected_state[reg_vars.pressure_index], params.minimum_pressure
            )
        )
        corrected_state = corrected_state.at[reg_vars.density_index].set(
            jnp.maximum(corrected_state[reg_vars.density_index], params.minimum_density)
        )

        effective_corrections.append(
            jnp.mean(jnp.abs(corrected_state - original_state), axis=[1, 2, 3])
        )

        # Detect invalid values BEFORE clamping masks them
        def _has_invalid_values(arr: np.ndarray) -> bool:
            """Check for NaN, Inf, or non-positive density/pressure."""
            return bool(np.any(~np.isfinite(arr)))

        def _has_nonpositive_physics(
            arr: np.ndarray, density_idx: int, pressure_idx: int
        ) -> bool:
            """Check for non-positive density or pressure (physics violation)."""
            density = arr[density_idx]
            pressure = arr[pressure_idx]
            return bool(np.any(density <= 0) or np.any(pressure <= 0))

        original_np = np.asarray(original_state)
        corrected_np = np.asarray(corrected_state)

        # Check both non-finite values AND physics violations (pre-clamp)
        has_invalid = (
            _has_invalid_values(original_np)
            or _has_invalid_values(pre_clamp_state)
            or _has_nonpositive_physics(
                pre_clamp_state, reg_vars.density_index, reg_vars.pressure_index
            )
        )

        if has_invalid:
            # Store pre-clamp state, time, and step index to preserve evidence of issues
            nan_states_before_correction.append(original_np)
            nan_states_after_correction.append(pre_clamp_state)
            crash_times.append(float(time))
            crash_step_indices.append(current_step)

    return snapshot_callable


def _step_time_delta_callback_factory(
    *, time_deltas: list[float]
) -> Callable[..., None]:
    def step_callable(_time, dt):
        time_deltas.append(float(dt))

    return step_callable


def _prepare_time_delta_plot_arrays(
    time_deltas: np.ndarray, validation_t_end: float
) -> tuple[np.ndarray, np.ndarray] | None:
    if time_deltas.size == 0:
        return None

    times_total_arr = np.cumsum(time_deltas)
    finite_mask = np.isfinite(times_total_arr) & np.isfinite(time_deltas)
    finite_indices = np.where(finite_mask)[0]
    if finite_indices.size == 0:
        return None

    last_finite_idx = int(finite_indices[-1])
    times_delta_plot = time_deltas[: last_finite_idx + 1]
    times_total_plot = times_total_arr[: last_finite_idx + 1]

    if validation_t_end > times_total_plot[-1]:
        times_total_plot = np.concatenate(
            [
                times_total_plot,
                np.array([validation_t_end], dtype=times_total_plot.dtype),
            ]
        )
        times_delta_plot = np.concatenate(
            [times_delta_plot, np.array([times_delta_plot[-1]])]
        )

    return times_total_plot, times_delta_plot


def _plot_shifted_mhd_blast_analysis(
    *,
    training_config: TrainingConfig,
    problem_manager: ProblemManager,
    problem_descriptor: ProblemDescriptor,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    model_dir: Path,
    split_loss_line_at_training_time: bool,
    trained_t_end: float,
) -> None:
    shift_values = (0, 8, 16, 24)
    shift_linestyles = {0: "-", 8: "--", 16: "-.", 24: ":"}

    problem = PROBLEM_CATALOG[problem_descriptor.name](**problem_descriptor.params)
    snapshot_timepoints = SNAPSHOT_TIMEPOINTS_VALIDATION[problem_descriptor.name]
    if split_loss_line_at_training_time and not jnp.any(
        snapshot_timepoints == PROBLEM_DICTIONARY[problem_descriptor.name]["t_end"]
    ):
        snapshot_timepoints = jnp.sort(
            jnp.concatenate(
                [
                    snapshot_timepoints,
                    jnp.array([PROBLEM_DICTIONARY[problem_descriptor.name]["t_end"]]),
                ]
            )
        )

    config_overrides = problem.get_config_overrides_evaluation(snapshot_timepoints)
    base_initial_state, base_config, base_params, base_reg_vars = (
        problem.generate_initial_state_with_config_overrides(
            config=problem_manager.hr_config,
            params=problem_manager.hr_params,
            config_overrides=config_overrides,
        )
    )
    downsampled_reg_vars = get_registered_variables(
        finalize_config(
            base_config._replace(
                num_cells=base_config.num_cells // training_config.downaverage_factor
            ),
            downaverage(base_initial_state, training_config.downaverage_factor).shape,
        )
    )

    shifted_eval_data: list[EvaluationResults] = []
    shifted_density_images: dict[int, np.ndarray] = {}
    successful_shifts: list[int] = []

    for shift in shift_values:
        shifted_initial_state = jnp.roll(base_initial_state, shift=shift, axis=1)

        try:
            hr_snapshot_data = time_integration(
                primitive_state=shifted_initial_state,
                config=base_config,
                params=base_params,
                registered_variables=base_reg_vars,
            )
        except Exception:
            logger.warning(
                "Shifted HR simulation failed for mhd_blast (shift=%s).", shift
            )
            continue

        lr_initial_state = downaverage(
            shifted_initial_state, training_config.downaverage_factor
        )
        lr_config = finalize_config(
            base_config._replace(
                num_cells=base_config.num_cells // training_config.downaverage_factor
            ),
            lr_initial_state.shape,
        )
        lr_reg_vars = get_registered_variables(lr_config)

        try:
            lr_snapshot_data = time_integration(
                primitive_state=lr_initial_state,
                config=lr_config._replace(progress_bar=False),
                params=base_params,
                registered_variables=lr_reg_vars,
            )
        except Exception:
            logger.warning(
                "Shifted LR simulation failed for mhd_blast (shift=%s).", shift
            )
            lr_snapshot_data = None

        try:
            lr_sol_snapshot_data = time_integration(
                primitive_state=lr_initial_state,
                config=lr_config._replace(
                    progress_bar=False, cnn_mhd_corrector_config=corrector_config
                ),
                params=base_params._replace(cnn_mhd_corrector_params=corrector_params),
                registered_variables=lr_reg_vars,
            )
        except Exception:
            logger.warning(
                "Shifted LR+SOL simulation failed for mhd_blast (shift=%s).", shift
            )
            lr_sol_snapshot_data = None

        shifted_eval_data.append(
            EvaluationResults(
                problem_descriptor=problem_descriptor,
                hr_snapshot_data=hr_snapshot_data,
                lr_snapshot_data=lr_snapshot_data,
                lr_sol_snapshot_data=lr_sol_snapshot_data,
            )
        )
        successful_shifts.append(shift)

        hr_target_states = downaverage(
            hr_snapshot_data.states, training_config.downaverage_factor
        )
        density_idx = downsampled_reg_vars.density_index
        z_mid = hr_target_states.shape[-1] // 2
        shifted_density_images[shift] = np.asarray(
            hr_target_states[-1][density_idx, :, :, z_mid].T
        )

    if not shifted_eval_data:
        logger.warning(
            "No shifted mhd_blast simulations succeeded; skipping shift plot."
        )
        return

    shifted_losses = _compute_sim_losses(
        shifted_eval_data,
        training_config,
        trained_t_end={problem_descriptor.name: trained_t_end},
    )
    losses_by_shift = {
        shift: sim_loss for shift, sim_loss in zip(successful_shifts, shifted_losses)
    }

    fig_shift = plt.figure(figsize=(20, 10))
    gs_shift = GridSpec(2, 4, height_ratios=[1.0, 1.2], hspace=0.35, wspace=0.25)

    finite_image_values = [
        img[np.isfinite(img)]
        for img in shifted_density_images.values()
        if img is not None
    ]
    if finite_image_values:
        all_values = np.concatenate(finite_image_values)
        vmin = float(np.min(all_values))
        vmax = float(np.max(all_values))
    else:
        vmin, vmax = None, None

    for col_idx, shift in enumerate(shift_values):
        ax_img = fig_shift.add_subplot(gs_shift[0, col_idx])
        img = shifted_density_images.get(shift)
        if img is None:
            ax_img.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax_img.transAxes,
            )
            ax_img.set_title(f"Shift +{shift}")
            ax_img.set_xlabel("x")
            ax_img.set_ylabel("y")
            continue

        im = ax_img.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_img.set_title(f"Shift +{shift} (HR downsampled density)")
        ax_img.set_xlabel("x")
        ax_img.set_ylabel("y")
        fig_shift.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

    ax_loss = fig_shift.add_subplot(gs_shift[1, :])

    def _plot_shifted_curve(
        times: Optional[np.ndarray],
        losses: Optional[np.ndarray],
        *,
        label: str,
        color: str,
        linestyle: str,
    ) -> None:
        if times is None or losses is None:
            return
        times_arr = np.asarray(times)
        losses_arr = np.asarray(losses)
        n = min(times_arr.size, losses_arr.size)
        if n == 0:
            return
        times_arr = times_arr[:n]
        losses_arr = losses_arr[:n]
        finite_mask = np.isfinite(times_arr) & np.isfinite(losses_arr)
        if not np.any(finite_mask):
            return
        invalid_indices = np.where(~finite_mask)[0]
        prefix_end = int(invalid_indices[0]) if invalid_indices.size > 0 else n
        if prefix_end == 0:
            return
        ax_loss.plot(
            times_arr[:prefix_end],
            losses_arr[:prefix_end],
            label=label,
            color=color,
            linestyle=linestyle,
        )

    for shift in shift_values:
        sim_loss = losses_by_shift.get(shift)
        if sim_loss is None:
            continue
        linestyle = shift_linestyles[shift]
        _plot_shifted_curve(
            sim_loss["lr_times"],
            sim_loss["lr_losses"],
            label=f"LR shift +{shift}",
            color="tab:orange",
            linestyle=linestyle,
        )
        _plot_shifted_curve(
            sim_loss["sol_times"],
            sim_loss["sol_losses"],
            label=f"LR+SOL shift +{shift}",
            color="tab:blue",
            linestyle=linestyle,
        )

    ax_loss.axvline(
        x=trained_t_end,
        color="gray",
        linestyle=":",
        alpha=0.7,
        label=f"trained t={trained_t_end:.3f}",
    )
    ax_loss.set_xlabel("Time")
    ax_loss.set_ylabel("L2 Loss")
    ax_loss.set_title("mhd_blast rolled initial state: LR vs LR+SOL")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8, ncol=2)

    problem_nickname_dir = model_dir / problem_descriptor.nickname
    problem_nickname_dir.mkdir(parents=True, exist_ok=True)
    shift_out_path = problem_nickname_dir / "mhd_blast_shifted_roll_comparison.png"
    plt.savefig(shift_out_path, dpi=300, bbox_inches="tight")
    logger.info("Saved shifted mhd_blast comparison plot to %s", shift_out_path)
    plt.close(fig_shift)


def _plot_crash_diagnostics(
    nan_states_before_correction: list[Array],
    nan_states_after_correction: list[Array],
    crash_times: list[float],
    crash_step_indices: list[int],
    plots_dir: Path,
    problem_name: str,
):
    """
    Plot diagnostic information when a crash (NaN/nonpositivity) is detected in lr_sol.

    Note: nan_states_after_correction contains pre-clamp states (before jnp.maximum
    is applied) to preserve evidence of physics violations like negative density/pressure.

    Generates:
    - Middle-slice comparison of states before/after correction
    - Nonpositivity diagnostics for density and pressure
    - NaN/Inf location maps
    - Statistical analysis of problematic regions
    """
    if not nan_states_before_correction:
        logger.info("No crash states detected; skipping crash diagnostics.")
        return

    # Save diagnostics in problem-specific folder
    crash_dir = plots_dir / problem_name / "crash_diagnostics"
    crash_dir.mkdir(parents=True, exist_ok=True)

    reg_vars = _get_reg_vars()
    density_idx = reg_vars.density_index
    pressure_idx = reg_vars.pressure_index

    for crash_idx, (state_before, state_after, crash_time, step_idx) in enumerate(
        zip(
            nan_states_before_correction,
            nan_states_after_correction,
            crash_times,
            crash_step_indices,
        )
    ):
        logger.info(
            f"Generating crash diagnostics for crash event {crash_idx} at t={crash_time:.6e} (step {step_idx})"
        )

        # Find z-slice with most problematic cells for state plots
        slice_z = _find_problematic_slice(
            state_before, state_after, check_nonpositive=True, check_nonfinite=True
        )
        shape = state_before.shape
        mid_x = shape[1] // 2
        mid_y = shape[2] // 2

        # === 1. Plot states sliced at problematic z-level ===
        for slice_axis, slice_level in [("z", slice_z), ("y", mid_y), ("x", mid_x)]:
            plot_states(
                states_list=[state_before, state_after],
                z_levels=[slice_level, slice_level],
                fig_name=f"crash_{crash_idx}_states_{slice_axis}_slice",
                folder=str(crash_dir),
                titles=[
                    f"Before Correction (t={crash_time:.4e}, step {step_idx})",
                    f"After Correction (pre-clamp, t={crash_time:.4e}, step {step_idx})",
                ],
                slice_axis=slice_axis,
            )
            logger.info(
                f"Saved {slice_axis}-slice state comparison for crash {crash_idx}"
            )

        # === 2. Nonpositivity diagnostics (skipped if no nonpositivity) ===
        _plot_nonpositivity_diagnostics(
            state_before,
            state_after,
            crash_idx,
            crash_time,
            step_idx,
            crash_dir,
            reg_vars,
        )

        # === 3. NaN/Inf location maps (skipped if no NaN/Inf) ===
        _plot_nan_inf_locations(
            state_before, state_after, crash_idx, crash_time, step_idx, crash_dir
        )

        # === 4. Statistical summary ===
        _plot_crash_statistics(
            state_before,
            state_after,
            crash_idx,
            crash_time,
            step_idx,
            crash_dir,
            reg_vars,
        )

    logger.info(f"Crash diagnostics saved to {crash_dir}")


def _find_problematic_slice(
    *arrays: np.ndarray,
    check_nonpositive: bool = False,
    check_nonfinite: bool = True,
) -> int:
    """
    Find the z-slice index with the most problematic cells.

    Args:
        *arrays: Arrays to check (shape: [..., x, y, z] or [x, y, z])
        check_nonpositive: If True, count cells <= 0 as problematic
        check_nonfinite: If True, count NaN/Inf cells as problematic

    Returns:
        z-index with the most problematic cells, or middle if none found
    """
    if not arrays:
        return 0

    # Get z dimension from first array
    first_arr = arrays[0]
    z_size = first_arr.shape[-1]

    # Count problematic cells per z-slice
    problems_per_z = np.zeros(z_size, dtype=int)

    for arr in arrays:
        arr = np.asarray(arr)
        for z in range(z_size):
            if arr.ndim == 3:
                slice_data = arr[:, :, z]
            else:
                # Handle higher-dim arrays (e.g., [channels, x, y, z])
                slice_data = arr[..., :, :, z]

            if check_nonfinite:
                problems_per_z[z] += np.sum(~np.isfinite(slice_data))
            if check_nonpositive:
                problems_per_z[z] += np.sum(slice_data <= 0)

    # Return slice with most problems, or middle if no problems found
    if np.max(problems_per_z) > 0:
        return int(np.argmax(problems_per_z))
    return z_size // 2


def _plot_nonpositivity_diagnostics(
    state_before: Array,
    state_after: Array,
    crash_idx: int,
    crash_time: float,
    step_idx: int,
    crash_dir: Path,
    reg_vars,
) -> bool:
    """
    Plot diagnostics for nonpositivity in density and pressure fields.

    Returns:
        True if plot was generated, False if skipped (no nonpositivity found)
    """
    density_idx = reg_vars.density_index
    pressure_idx = reg_vars.pressure_index

    density_before = np.asarray(state_before[density_idx])
    density_after = np.asarray(state_after[density_idx])
    pressure_before = np.asarray(state_before[pressure_idx])
    pressure_after = np.asarray(state_after[pressure_idx])

    # Check if there's any nonpositivity to plot
    total_nonpos = (
        np.sum(density_before <= 0)
        + np.sum(density_after <= 0)
        + np.sum(pressure_before <= 0)
        + np.sum(pressure_after <= 0)
    )
    if total_nonpos == 0:
        logger.info(
            f"Crash {crash_idx}: No nonpositive values found, skipping nonpositivity plot"
        )
        return False

    # Find z-slice with most nonpositive cells
    slice_z = _find_problematic_slice(
        density_before,
        density_after,
        pressure_before,
        pressure_after,
        check_nonpositive=True,
        check_nonfinite=False,
    )

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 0: Density diagnostics
    im0 = axes[0, 0].imshow(
        density_before[:, :, slice_z].T, origin="lower", cmap="viridis"
    )
    axes[0, 0].set_title("Density (Before)")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(
        density_after[:, :, slice_z].T, origin="lower", cmap="viridis"
    )
    axes[0, 1].set_title("Density (After)")
    plt.colorbar(im1, ax=axes[0, 1])

    # Nonpositive density mask (before)
    nonpos_density_before = density_before <= 0
    axes[0, 2].imshow(
        nonpos_density_before[:, :, slice_z].T.astype(float),
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[0, 2].set_title(
        f"Nonpositive Density (Before)\nCount: {np.sum(nonpos_density_before)}"
    )

    # Nonpositive density mask (after)
    nonpos_density_after = density_after <= 0
    axes[0, 3].imshow(
        nonpos_density_after[:, :, slice_z].T.astype(float),
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[0, 3].set_title(
        f"Nonpositive Density (After)\nCount: {np.sum(nonpos_density_after)}"
    )

    # Row 1: Pressure diagnostics
    im2 = axes[1, 0].imshow(
        pressure_before[:, :, slice_z].T, origin="lower", cmap="plasma"
    )
    axes[1, 0].set_title("Pressure (Before)")
    plt.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(
        pressure_after[:, :, slice_z].T, origin="lower", cmap="plasma"
    )
    axes[1, 1].set_title("Pressure (After)")
    plt.colorbar(im3, ax=axes[1, 1])

    # Nonpositive pressure mask (before)
    nonpos_pressure_before = pressure_before <= 0
    axes[1, 2].imshow(
        nonpos_pressure_before[:, :, slice_z].T.astype(float),
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[1, 2].set_title(
        f"Nonpositive Pressure (Before)\nCount: {np.sum(nonpos_pressure_before)}"
    )

    # Nonpositive pressure mask (after)
    nonpos_pressure_after = pressure_after <= 0
    axes[1, 3].imshow(
        nonpos_pressure_after[:, :, slice_z].T.astype(float),
        origin="lower",
        cmap="Reds",
        vmin=0,
        vmax=1,
    )
    axes[1, 3].set_title(
        f"Nonpositive Pressure (After)\nCount: {np.sum(nonpos_pressure_after)}"
    )

    plt.suptitle(
        f"Nonpositivity Diagnostics - Crash {crash_idx} at t={crash_time:.4e}, step {step_idx} (z-slice {slice_z})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(crash_dir / f"crash_{crash_idx}_nonpositivity.png", dpi=300)
    plt.close(fig)
    return True


def _plot_nan_inf_locations(
    state_before: Array,
    state_after: Array,
    crash_idx: int,
    crash_time: float,
    step_idx: int,
    crash_dir: Path,
) -> bool:
    """
    Plot locations of NaN and Inf values in the state arrays.

    Returns:
        True if plot was generated, False if skipped (no NaN/Inf found)
    """
    state_before_np = np.asarray(state_before)
    state_after_np = np.asarray(state_after)

    # Check if there's any NaN/Inf to plot
    total_nonfinite = np.sum(~np.isfinite(state_before_np)) + np.sum(
        ~np.isfinite(state_after_np)
    )
    if total_nonfinite == 0:
        logger.info(
            f"Crash {crash_idx}: No NaN/Inf values found, skipping NaN/Inf plot"
        )
        return False

    n_channels = state_before_np.shape[0]
    base_channel_names = [
        "Density",
        "Vx",
        "Vy",
        "Vz",
        "Pressure",
        "Bx",
        "By",
        "Bz",
        "Bx_interface",
        "By_interface",
        "Bz_interface",
    ]
    # Handle cases where n_channels exceeds predefined names
    if n_channels <= len(base_channel_names):
        channel_names = base_channel_names[:n_channels]
    else:
        channel_names = base_channel_names + [
            f"Ch{i}" for i in range(len(base_channel_names), n_channels)
        ]

    # Find z-slice with most NaN/Inf cells
    slice_z = _find_problematic_slice(
        state_before_np, state_after_np, check_nonpositive=False, check_nonfinite=True
    )

    fig, axes = plt.subplots(2, n_channels, figsize=(3 * n_channels, 11), squeeze=False)

    for ch_idx in range(n_channels):
        field_before = state_before_np[ch_idx]
        field_after = state_after_np[ch_idx]

        # NaN mask before
        nan_before = np.isnan(field_before[:, :, slice_z])
        inf_before = np.isinf(field_before[:, :, slice_z])
        bad_before = nan_before | inf_before

        axes[0, ch_idx].imshow(
            bad_before.T.astype(float), origin="lower", cmap="Reds", vmin=0, vmax=1
        )
        nan_count = np.sum(nan_before)
        inf_count = np.sum(inf_before)
        axes[0, ch_idx].set_title(
            f"{channel_names[ch_idx]}\nNaN:{nan_count} Inf:{inf_count}", fontsize=9
        )
        if ch_idx == 0:
            axes[0, ch_idx].set_ylabel("Before Correction")

        # NaN mask after
        nan_after = np.isnan(field_after[:, :, slice_z])
        inf_after = np.isinf(field_after[:, :, slice_z])
        bad_after = nan_after | inf_after

        axes[1, ch_idx].imshow(
            bad_after.T.astype(float), origin="lower", cmap="Reds", vmin=0, vmax=1
        )
        nan_count_after = np.sum(nan_after)
        inf_count_after = np.sum(inf_after)
        axes[1, ch_idx].set_title(
            f"NaN:{nan_count_after} Inf:{inf_count_after}", fontsize=9
        )
        if ch_idx == 0:
            axes[1, ch_idx].set_ylabel("After Correction")

    plt.suptitle(
        f"NaN/Inf Location Map - Crash {crash_idx} at t={crash_time:.4e}, step {step_idx} (z-slice {slice_z})",
        fontsize=14,
    )
    plt.tight_layout()
    plt.savefig(crash_dir / f"crash_{crash_idx}_nan_inf_map.png", dpi=300)
    plt.close(fig)
    return True


def _plot_crash_statistics(
    state_before: Array,
    state_after: Array,
    crash_idx: int,
    crash_time: float,
    step_idx: int,
    crash_dir: Path,
    reg_vars,
):
    """Plot statistical summary of crash states including histograms and gradients."""
    density_idx = reg_vars.density_index
    pressure_idx = reg_vars.pressure_index

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    density_before = np.asarray(state_before[density_idx])
    density_after = np.asarray(state_after[density_idx])
    pressure_before = np.asarray(state_before[pressure_idx])
    pressure_after = np.asarray(state_after[pressure_idx])

    # Row 0: Histograms
    # Filter out NaN/Inf for histograms
    density_before_finite = density_before[np.isfinite(density_before)]
    density_after_finite = density_after[np.isfinite(density_after)]
    pressure_before_finite = pressure_before[np.isfinite(pressure_before)]
    pressure_after_finite = pressure_after[np.isfinite(pressure_after)]

    if len(density_before_finite) > 0:
        axes[0, 0].hist(
            density_before_finite.flatten(), bins=50, alpha=0.7, label="Before"
        )
        axes[0, 0].axvline(x=0, color="r", linestyle="--", label="Zero")
        axes[0, 0].set_title("Density Histogram (Before)")
        axes[0, 0].set_xlabel("Density")
        axes[0, 0].legend()

    if len(density_after_finite) > 0:
        axes[0, 1].hist(
            density_after_finite.flatten(), bins=50, alpha=0.7, label="After"
        )
        axes[0, 1].axvline(x=0, color="r", linestyle="--", label="Zero")
        axes[0, 1].set_title("Density Histogram (After)")
        axes[0, 1].set_xlabel("Density")
        axes[0, 1].legend()

    if len(pressure_before_finite) > 0:
        axes[0, 2].hist(
            pressure_before_finite.flatten(), bins=50, alpha=0.7, label="Before"
        )
        axes[0, 2].axvline(x=0, color="r", linestyle="--", label="Zero")
        axes[0, 2].set_title("Pressure Histogram (Before)")
        axes[0, 2].set_xlabel("Pressure")
        axes[0, 2].legend()

    if len(pressure_after_finite) > 0:
        axes[0, 3].hist(
            pressure_after_finite.flatten(), bins=50, alpha=0.7, label="After"
        )
        axes[0, 3].axvline(x=0, color="r", linestyle="--", label="Zero")
        axes[0, 3].set_title("Pressure Histogram (After)")
        axes[0, 3].set_xlabel("Pressure")
        axes[0, 3].legend()

    # Row 1: Gradient magnitude (to identify sharp gradients that may cause instability)
    mid_z = density_before.shape[2] // 2

    def compute_gradient_magnitude(field):
        gx = np.gradient(field, axis=0)
        gy = np.gradient(field, axis=1)
        gz = np.gradient(field, axis=2)
        return np.sqrt(gx**2 + gy**2 + gz**2)

    grad_density_before = compute_gradient_magnitude(density_before)
    grad_density_after = compute_gradient_magnitude(density_after)
    grad_pressure_before = compute_gradient_magnitude(pressure_before)
    grad_pressure_after = compute_gradient_magnitude(pressure_after)

    im0 = axes[1, 0].imshow(
        grad_density_before[:, :, mid_z].T, origin="lower", cmap="hot"
    )
    axes[1, 0].set_title("Density Gradient (Before)")
    plt.colorbar(im0, ax=axes[1, 0])

    im1 = axes[1, 1].imshow(
        grad_density_after[:, :, mid_z].T, origin="lower", cmap="hot"
    )
    axes[1, 1].set_title("Density Gradient (After)")
    plt.colorbar(im1, ax=axes[1, 1])

    im2 = axes[1, 2].imshow(
        grad_pressure_before[:, :, mid_z].T, origin="lower", cmap="hot"
    )
    axes[1, 2].set_title("Pressure Gradient (Before)")
    plt.colorbar(im2, ax=axes[1, 2])

    im3 = axes[1, 3].imshow(
        grad_pressure_after[:, :, mid_z].T, origin="lower", cmap="hot"
    )
    axes[1, 3].set_title("Pressure Gradient (After)")
    plt.colorbar(im3, ax=axes[1, 3])

    # Row 2: Correction delta (difference before/after)
    delta_density = density_after - density_before
    delta_pressure = pressure_after - pressure_before

    im4 = axes[2, 0].imshow(
        delta_density[:, :, mid_z].T, origin="lower", cmap="seismic"
    )
    axes[2, 0].set_title("Density Correction Delta")
    plt.colorbar(im4, ax=axes[2, 0])

    im5 = axes[2, 1].imshow(
        np.abs(delta_density[:, :, mid_z]).T, origin="lower", cmap="viridis"
    )
    axes[2, 1].set_title("|Density Correction Delta|")
    plt.colorbar(im5, ax=axes[2, 1])

    im6 = axes[2, 2].imshow(
        delta_pressure[:, :, mid_z].T, origin="lower", cmap="seismic"
    )
    axes[2, 2].set_title("Pressure Correction Delta")
    plt.colorbar(im6, ax=axes[2, 2])

    im7 = axes[2, 3].imshow(
        np.abs(delta_pressure[:, :, mid_z]).T, origin="lower", cmap="viridis"
    )
    axes[2, 3].set_title("|Pressure Correction Delta|")
    plt.colorbar(im7, ax=axes[2, 3])

    # Add text summary
    stats_text = (
        f"Crash Statistics at t={crash_time:.6e}, step {step_idx} (after = pre-clamp state):\n"
        f"Density min/max (before): {np.nanmin(density_before):.4e}/{np.nanmax(density_before):.4e}\n"
        f"Density min/max (after): {np.nanmin(density_after):.4e}/{np.nanmax(density_after):.4e}\n"
        f"Pressure min/max (before): {np.nanmin(pressure_before):.4e}/{np.nanmax(pressure_before):.4e}\n"
        f"Pressure min/max (after): {np.nanmin(pressure_after):.4e}/{np.nanmax(pressure_after):.4e}\n"
        f"NaN count (before): density={np.sum(np.isnan(density_before))}, pressure={np.sum(np.isnan(pressure_before))}\n"
        f"NaN count (after): density={np.sum(np.isnan(density_after))}, pressure={np.sum(np.isnan(pressure_after))}\n"
        f"Inf count (before): density={np.sum(np.isinf(density_before))}, pressure={np.sum(np.isinf(pressure_before))}\n"
        f"Inf count (after): density={np.sum(np.isinf(density_after))}, pressure={np.sum(np.isinf(pressure_after))}\n"
        f"Nonpositive count (before): density={np.sum(density_before <= 0)}, pressure={np.sum(pressure_before <= 0)}\n"
        f"Nonpositive count (after): density={np.sum(density_after <= 0)}, pressure={np.sum(pressure_after <= 0)}"
    )
    fig.text(0.02, 0.02, stats_text, fontsize=9, family="monospace", va="bottom")

    plt.suptitle(
        f"Crash Statistics - Crash {crash_idx} at t={crash_time:.4e}, step {step_idx}",
        fontsize=14,
    )
    plt.tight_layout(rect=[0, 0.12, 1, 0.98])
    plt.savefig(crash_dir / f"crash_{crash_idx}_statistics.png", dpi=300)
    plt.close(fig)

    # Also save a text summary
    with open(crash_dir / f"crash_{crash_idx}_summary.txt", "w") as f:
        f.write(f"Crash Event {crash_idx} at t={crash_time:.6e}, step {step_idx}\n")
        f.write("=" * 50 + "\n\n")
        f.write(stats_text + "\n")


def plot_problem_model_analysis(
    training_config: TrainingConfig,
    problem_manager: ProblemManager,
    model_name: str,
    split_loss_line_at_training_time: bool = False,
    loading_nan_params: bool = False,
    figure_name="multiproblem_summary",
):
    assert len(problem_manager.problem_descriptors) == 1
    model_dir = Path(MULTIPROBLEM_BASE_DIR) / model_name
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    corrections: list[Array] = []
    effective_corrections: list[Array] = []
    states: list[Array] = []
    correction_time_deltas: list[float] = []
    nan_states_before_correction: list[Array] = []
    nan_states_after_correction: list[Array] = []
    crash_times: list[float] = []
    crash_step_indices: list[int] = []
    simulation_time_deltas: dict[str, dict[str, list[float]]] = {}

    snapshot_callable = _snapshot_callback_factory(
        corrections=corrections,
        effective_corrections=effective_corrections,
        states=states,
        time_deltas=correction_time_deltas,
        nan_states_before_correction=nan_states_before_correction,
        nan_states_after_correction=nan_states_after_correction,
        crash_times=crash_times,
        crash_step_indices=crash_step_indices,
    )
    reg_vars = _get_reg_vars()

    callback_model = CorrectorCNN(
        in_channels=reg_vars.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
        snapshot_callable=snapshot_callable,
    )

    neural_net_params_shape, callback_static = eqx.partition(
        callback_model, eqx.is_array
    )
    if loading_nan_params:
        neural_net_params = model_manager.load_model_params_nan(
            like=neural_net_params_shape
        )
    else:
        neural_net_params = model_manager.load_model_params(
            like=neural_net_params_shape, param_type="best"
        )

    corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=callback_static,
        correct_from_beggining=training_config.correct_from_beggining,
        start_correction_time=training_config.start_correction_time,
    )
    corrector_params = CNNMHDParams(network_params=neural_net_params)

    evaluation_data = []
    snapshot_data_after_lr_sol = []
    snapshot_timepoints_trained_index = {}
    magnetic_divergence_grid_spacings: dict[str, dict[str, float]] = {}

    for problem_descriptor in problem_manager.problem_descriptors:
        problem_manager_problem = ProblemManager(
            problem_descriptors=[problem_descriptor], training_config=training_config
        )
        problem = PROBLEM_CATALOG[problem_descriptor.name](**problem_descriptor.params)

        snapshot_timepoints = SNAPSHOT_TIMEPOINTS_VALIDATION[problem_descriptor.name]

        t_end_problem = float(PROBLEM_DICTIONARY[problem_descriptor.name]["t_end"])
        if not jnp.any(jnp.isclose(snapshot_timepoints, t_end_problem)):
            logger.info("added the t end to the snapshot timepoints")
            snapshot_timepoints = jnp.sort(
                jnp.concatenate(
                    [
                        snapshot_timepoints,
                        jnp.array([t_end_problem]),
                    ]
                )
            )

        # Always define the trained-time index; using argmin is robust to FP rounding.
        snapshot_timepoints_trained_index[problem_descriptor.name] = int(
            jnp.argmin(jnp.abs(snapshot_timepoints - t_end_problem))
        )

        config_overrides = problem.get_config_overrides_evaluation(snapshot_timepoints)
        _, config_for_spacing, _, _ = (
            problem.generate_initial_state_with_config_overrides(
                config=problem_manager_problem.hr_config,
                params=problem_manager_problem.hr_params,
                config_overrides=config_overrides,
            )
        )
        hr_grid_spacing = float(config_for_spacing.grid_spacing)
        magnetic_divergence_grid_spacings[problem_descriptor.name] = {
            "hr": hr_grid_spacing,
            "lr": hr_grid_spacing * training_config.downaverage_factor,
        }

        hr_time_deltas: list[float] = []
        lr_time_deltas: list[float] = []
        lr_sol_time_deltas: list[float] = []
        step_callbacks = {
            "hr": _step_time_delta_callback_factory(time_deltas=hr_time_deltas),
            "lr": _step_time_delta_callback_factory(time_deltas=lr_time_deltas),
            "lr_sol": _step_time_delta_callback_factory(time_deltas=lr_sol_time_deltas),
        }

        evaluation_data_problem = problem_manager_problem.get_evaluation_data(
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            snapshot_timepoints=snapshot_timepoints,
            step_callbacks=step_callbacks,
        )

        print("len of corrections ", len(corrections))
        evaluation_data.append(evaluation_data_problem[0])
        simulation_time_deltas[problem_descriptor.name] = {
            "HR": hr_time_deltas,
            "LR": lr_time_deltas,
            "LR+SOL": lr_sol_time_deltas,
        }
        if split_loss_line_at_training_time:
            assert isinstance(
                evaluation_data_problem[0].lr_sol_snapshot_data, SnapshotData
            )
            logger.info(f"snapshot_timepoints {snapshot_timepoints}")
            logger.info(
                f"snapshot_trained_t_index: {snapshot_timepoints_trained_index}",
            )
            logger.info(
                f"len of remaining snapshots {len(evaluation_data_problem[0].hr_snapshot_data.states[snapshot_timepoints_trained_index[problem_descriptor.name] :])}"
            )
            after_sol_timepoints = snapshot_timepoints[
                snapshot_timepoints_trained_index[problem_descriptor.name] :
            ]
            after_sol_timepoints_relative = (
                after_sol_timepoints - after_sol_timepoints[0]
            )
            logger.info(f"after sol timepoints {after_sol_timepoints}")
            logger.info(
                f"after sol relative timepoints {after_sol_timepoints_relative}"
            )

            state_at_training_time = evaluation_data_problem[
                0
            ].lr_sol_snapshot_data.states[
                snapshot_timepoints_trained_index[problem_descriptor.name]
            ]

            _, config, params, registered_variables = (
                problem.generate_initial_state_with_config_overrides(
                    config=problem_manager_problem.hr_config,
                    params=problem_manager_problem.hr_params,
                    config_overrides=problem.get_config_overrides_evaluation(
                        snapshot_timepoints=after_sol_timepoints_relative
                    ),
                )
            )

            config_lr = finalize_config(
                config._replace(
                    num_cells=training_config.num_cells_high_res
                    // training_config.downaverage_factor
                ),
                state_at_training_time.shape,
            )
            reg_vars_lr = get_registered_variables(config_lr)
            sd_after_lr_sol = time_integration(
                primitive_state=state_at_training_time,
                config=config_lr,
                params=params,
                registered_variables=reg_vars_lr,
            )
            snapshot_data_after_lr_sol.append(sd_after_lr_sol)

    # ── Crash diagnostics ────────────────────────────────────────────────────
    # If NaN states were detected during the lr_sol simulation, generate
    # diagnostic plots to understand why the simulation crashed
    if nan_states_before_correction:
        problem_name = problem_manager.problem_descriptors[0].name
        logger.info(
            f"Detected {len(nan_states_before_correction)} crash events in lr_sol "
            f"for problem '{problem_name}'. Generating crash diagnostics..."
        )
        _plot_crash_diagnostics(
            nan_states_before_correction=nan_states_before_correction,
            nan_states_after_correction=nan_states_after_correction,
            crash_times=crash_times,
            crash_step_indices=crash_step_indices,
            plots_dir=plots_dir,
            problem_name=problem_name,
        )

    if not evaluation_data:
        logger.warning("No evaluation data available; skipping plot.")
        return

    # Build trained t_end per problem name (one t_end per problem type, as per training)
    trained_t_end: dict[str, float] = {
        problem: PROBLEM_DICTIONARY[problem]["t_end"] for problem in PROBLEM_DICTIONARY
    }
    for data in evaluation_data:
        p_name = data.problem_descriptor.name
        if p_name not in trained_t_end:
            problem = PROBLEM_CATALOG[p_name](**data.problem_descriptor.params)
            trained_t_end[p_name] = float(problem.t_end)

    if len(snapshot_data_after_lr_sol) == 0:
        snapshot_data_after_lr_sol = None

    sim_losses = _compute_sim_losses(
        evaluation_data,
        training_config,
        trained_t_end,
        snapshot_data_after_lr_sol=snapshot_data_after_lr_sol,
    )

    # ── Figure layout ────────────────────────────────────────────────────────
    include_ot_vortex_divergence_row = any(
        "ot_vortex" in str(data.problem_descriptor.name).lower()
        for data in evaluation_data
    )
    height_ratios = [1.1, 1.0, 1.0, 1.0]
    if include_ot_vortex_divergence_row:
        height_ratios.append(0.9)
    fig = plt.figure(figsize=(18, 19 if include_ot_vortex_divergence_row else 16))
    gs = GridSpec(
        len(height_ratios),
        1,
        height_ratios=height_ratios,
        hspace=0.45,
        wspace=0.2,
    )

    ratio_floor = 1e-10
    # TODO: ASSIGN COLOR AND LINESTYLE TO EACH CHANNEL
    for p_idx, data in enumerate(evaluation_data):
        problem_name = data.problem_descriptor.name
        t_train = trained_t_end[problem_name]
        hr_states = data.hr_snapshot_data.states
        hr_times = np.array(data.hr_snapshot_data.time_points)
        validation_t_end = float(hr_times[-1])
        lr_target_states = downaverage(
            hr_states, downaverage_factor=training_config.downaverage_factor
        )
        target_final = np.asarray(
            lr_target_states[
                snapshot_timepoints_trained_index[data.problem_descriptor.name]
            ]
        )

        lr_final = None
        if data.lr_snapshot_data is not None:
            _, lr_prefix_states = _trim_snapshot_prefix(data.lr_snapshot_data)
            if lr_prefix_states.shape[0] > 0:
                lr_final = np.asarray(
                    lr_prefix_states[
                        snapshot_timepoints_trained_index[data.problem_descriptor.name]
                    ]
                )
            else:
                logger.warning(
                    "LR snapshots are empty after trimming for %s", problem_name
                )

        sol_final = None
        if data.lr_sol_snapshot_data is not None:
            _, sol_prefix_states = _trim_snapshot_prefix(data.lr_sol_snapshot_data)
            if sol_prefix_states.shape[0] > 0:
                sol_final = np.asarray(
                    sol_prefix_states[
                        snapshot_timepoints_trained_index[data.problem_descriptor.name]
                    ]
                )
            else:
                logger.warning(
                    "LR+SOL snapshots are empty after trimming for %s", problem_name
                )

        # ── Row 0: density slice images ───────────────────────────────────────
        density_idx = reg_vars.density_index
        z_mid = target_final.shape[-1] // 2
        target_img = target_final[density_idx, :, :, z_mid].T
        lr_img = None if lr_final is None else lr_final[density_idx, :, :, z_mid].T
        sol_img = None if sol_final is None else sol_final[density_idx, :, :, z_mid].T

        images = [target_img, lr_img, sol_img]
        image_titles = [
            "Target (HR downsampled) density",
            "LR baseline density",
            "LR + SOL density",
        ]

        finite_image_values = [
            img[np.isfinite(img)] for img in images if img is not None
        ]
        if finite_image_values:
            all_values = np.concatenate(finite_image_values)
            vmin = float(np.min(all_values))
            vmax = float(np.max(all_values))
        else:
            vmin, vmax = None, None

        gs_images = gs[0].subgridspec(1, 3, wspace=0.3)
        for i, (img, title) in enumerate(zip(images, image_titles, strict=True)):
            ax_img = fig.add_subplot(gs_images[0, i])
            if img is None:
                ax_img.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax_img.transAxes,
                )
                ax_img.set_title(title)
                ax_img.set_xlabel("x")
                ax_img.set_ylabel("y")
                continue
            im = ax_img.imshow(
                img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
            )
            ax_img.set_title(title)
            ax_img.set_xlabel("x")
            ax_img.set_ylabel("y")
            fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

        # ── Row 1: diagonal comparisons ───────────────────────────────────────
        gs_diag = gs[1].subgridspec(1, 4, wspace=0.35)
        diagonal_components = ("density", "pressure", "velocity", "magnetic")
        diagonal_titles = {
            "pressure": "Pressure diagonal slice",
            "density": "Density diagonal slice",
            "velocity": "Velocity magnitude diagonal slice",
            "magnetic": "Magnetic magnitude diagonal slice",
        }
        comparison_colors = {"HR": "black", "LR": "tab:orange", "LR+SOL": "tab:blue"}
        diagonal_positions = np.arange(target_final.shape[1], dtype=float)

        for ax_idx, component_name in enumerate(diagonal_components):
            ax_diag = fig.add_subplot(gs_diag[0, ax_idx])
            target_profile = _diagonal_profile(target_final, reg_vars, component_name)
            ax_diag.plot(
                diagonal_positions,
                target_profile,
                color=comparison_colors["HR"],
                label="HR",
            )
            if lr_final is not None:
                lr_profile = _diagonal_profile(lr_final, reg_vars, component_name)
                ax_diag.plot(
                    diagonal_positions,
                    lr_profile,
                    color=comparison_colors["LR"],
                    label="LR",
                )
            if sol_final is not None:
                sol_profile = _diagonal_profile(sol_final, reg_vars, component_name)
                ax_diag.plot(
                    diagonal_positions,
                    sol_profile,
                    color=comparison_colors["LR+SOL"],
                    label="LR+SOL",
                )
            ax_diag.set_title(diagonal_titles[component_name], fontsize=10)
            ax_diag.set_xlabel("Diagonal position")
            ax_diag.grid(alpha=0.3)
            if component_name in ("pressure", "density"):
                ax_diag.set_ylabel("Value")
            else:
                ax_diag.set_ylabel("Magnitude")
            if ax_idx == 0:
                ax_diag.legend(fontsize=8)

        def _extract_component(
            values: np.ndarray, component_name: str, *, from_correction: bool = False
        ) -> np.ndarray:
            def _idx(state_idx: int) -> int:
                if not from_correction:
                    return state_idx
                # Correction has num_state_channels - 3 channels: all non-interface
                # channels in order, plus interface-B updates in its last three slots.
                num_state_channels = reg_vars.num_vars
                num_correction_channels = values.shape[1]
                if state_idx >= num_state_channels - 3:
                    return num_correction_channels - (num_state_channels - state_idx)
                return state_idx

            assert isinstance(reg_vars.velocity_index, StaticIntVector)
            if component_name == "density":
                return values[:, _idx(reg_vars.density_index)]
            if component_name == "pressure":
                return values[:, _idx(reg_vars.pressure_index)]
            if component_name == "velocity":
                assert isinstance(reg_vars.velocity_index, StaticIntVector)
                return np.sqrt(
                    values[:, _idx(reg_vars.velocity_index.x)] ** 2
                    + values[:, _idx(reg_vars.velocity_index.y)] ** 2
                    + values[:, _idx(reg_vars.velocity_index.z)] ** 2
                )
            if component_name == "magnetic":
                if from_correction:
                    raise ValueError(
                        "magnetic component is not directly available in correction channels; "
                        "use magnetic_interface for correction extraction."
                    )
                assert isinstance(reg_vars.magnetic_index, StaticIntVector)
                return np.sqrt(
                    values[:, _idx(reg_vars.magnetic_index.x)] ** 2
                    + values[:, _idx(reg_vars.magnetic_index.y)] ** 2
                    + values[:, _idx(reg_vars.magnetic_index.z)] ** 2
                )

            if component_name == "magnetic_interface":
                assert isinstance(
                    reg_vars.interface_magnetic_field_index, StaticIntVector
                )
                return np.sqrt(
                    values[:, _idx(reg_vars.interface_magnetic_field_index.x)] ** 2
                    + values[:, _idx(reg_vars.interface_magnetic_field_index.y)] ** 2
                    + values[:, _idx(reg_vars.interface_magnetic_field_index.z)] ** 2
                )
            raise ValueError(f"Unknown component_name: {component_name}")

        # ── Row 2: model output ───────────────────────────────────────────────
        ax_model_output = fig.add_subplot(gs[2])
        component_names_corrections = (
            "density",
            "pressure",
            "velocity",
            "magnetic_interface",
        )
        component_colors = {
            "density": "g",
            "pressure": "b",
            "velocity": "r",
            "magnetic": "m",
            "magnetic_interface": "m",
        }

        corrections_array = np.array(corrections)
        states_array = np.array(states)
        times_delta = np.array(correction_time_deltas)
        times_total = np.cumsum(times_delta)
        crash_time = None
        if times_delta.size == 0 or corrections_array.ndim < 2 or states_array.ndim < 2:
            logger.warning("No callback entries found; skipping model output.")
            ax_model_output.text(
                0.5,
                0.5,
                "No callback model-output data available",
                ha="center",
                va="center",
                transform=ax_model_output.transAxes,
            )
            components = {}
        else:
            correction_components = {
                name: _extract_component(corrections_array, name, from_correction=True)
                for name in component_names_corrections
            }
            states_components = {
                name: _extract_component(states_array, name)
                for name in component_names_corrections
            }
            components = {
                name: (
                    np.log10(
                        np.abs(times_delta * correction_components[name]) + ratio_floor
                    )
                    - np.log10(states_components[name] + ratio_floor)
                )
                for name in component_names_corrections
            }

            finite_mask = np.isfinite(times_total)
            for name in components:
                finite_mask &= np.isfinite(components[name])
            finite_indices = np.where(finite_mask)[0]
            crashed = finite_indices.size < len(times_total)
            if finite_indices.size == 0:
                logger.warning(
                    "No finite callback entries found; skipping model output."
                )
            else:
                last_finite_idx = int(finite_indices[-1])
                times_total = times_total[: last_finite_idx + 1]
                components = {
                    name: values[: last_finite_idx + 1]
                    for name, values in components.items()
                }
                if crashed:
                    crash_time = float(times_total[-1])
                    if validation_t_end > times_total[-1]:
                        times_total = np.concatenate(
                            [
                                times_total,
                                np.array([validation_t_end], dtype=times_total.dtype),
                            ]
                        )
                        components = {
                            name: np.concatenate([values, np.array([values[-1]])])
                            for name, values in components.items()
                        }

            for name in components:
                ax_model_output.plot(
                    times_total,
                    components[name],
                    label=name,
                    color=component_colors[name],
                )
            if crashed and crash_time is not None:
                crash_idx = int(np.argmin(np.abs(times_total - crash_time)))
                for name in components:
                    ax_model_output.plot(
                        [times_total[crash_idx]],
                        [components[name][crash_idx]],
                        marker="x",
                        color=component_colors[name],
                        linestyle="None",
                        markersize=8,
                    )

        ax_model_output.axhline(
            y=0.0,
            color="gray",
            linestyle="--",
            alpha=0.8,
            label="|dt * correction| = |state|",
        )
        ax_model_output.set_xlabel("Times")
        ax_model_output.set_ylabel(
            "log10(|dt * correction| + floor) - log10(|state| + floor)"
        )
        ax_model_output.set_title("Model output (log-correction contrast)")
        ax_model_output.axvline(
            x=t_train,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"trained t={t_train:.3f}",
        )
        ax_model_output.legend(fontsize=8)
        ax_model_output.grid(alpha=0.3)

        # ── Row 3: loss during simulation ─────────────────────────────────────
        ax_err = fig.add_subplot(gs[3])
        sim = sim_losses[p_idx]

        def _plot_loss_curve(
            times: Optional[np.ndarray],
            losses: Optional[np.ndarray],
            *,
            label: str,
            color: str,
            linestyle: str = "-",
            max_time: Optional[float] = None,
            drop_trailing_zero_losses: bool = False,
            mark_trimmed_end: bool = False,
        ) -> None:
            if times is None or losses is None:
                return
            times_arr = np.asarray(times)
            losses_arr = np.asarray(losses)
            if times_arr.size == 0 or losses_arr.size == 0:
                return

            # Keep only the contiguous prefix up to the first invalid point.
            # This avoids plotting post-crash points when a curve contains NaNs/Infs.
            n = min(times_arr.size, losses_arr.size)
            times_arr = times_arr[:n]
            losses_arr = losses_arr[:n]
            finite_pair = np.isfinite(times_arr) & np.isfinite(losses_arr)
            if not np.any(finite_pair):
                return

            invalid_indices = np.where(~finite_pair)[0]
            prefix_end = int(invalid_indices[0]) if invalid_indices.size > 0 else n
            times_prefix = times_arr[:prefix_end]
            losses_prefix = losses_arr[:prefix_end]
            if times_prefix.size == 0 or losses_prefix.size == 0:
                return

            valid_mask = np.ones_like(times_prefix, dtype=bool)
            if max_time is not None:
                valid_mask &= times_prefix <= max_time
            if not np.any(valid_mask):
                return

            times_plot = times_prefix[valid_mask]
            losses_plot = losses_prefix[valid_mask]
            if drop_trailing_zero_losses:
                eps = 1e-12
                while losses_plot.size > 0 and abs(float(losses_plot[-1])) <= eps:
                    times_plot = times_plot[:-1]
                    losses_plot = losses_plot[:-1]
                if losses_plot.size == 0:
                    return
            ax_err.plot(
                times_plot,
                losses_plot,
                label=label,
                color=color,
                linestyle=linestyle,
            )

            had_invalid_tail = prefix_end < n
            trimmed_by_max_time = max_time is not None and np.any(
                times_prefix > max_time
            )
            if mark_trimmed_end and (had_invalid_tail or trimmed_by_max_time):
                ax_err.plot(
                    [times_plot[-1]],
                    [losses_plot[-1]],
                    marker="x",
                    color=color,
                    linestyle="None",
                    markersize=8,
                )

        _plot_loss_curve(
            sim["lr_times"],
            sim["lr_losses"],
            label="LR",
            color="tab:orange",
            linestyle="--",
            max_time=validation_t_end,
        )
        sol_cutoff_time = validation_t_end
        if crash_time is not None:
            sol_cutoff_time = min(sol_cutoff_time, crash_time)
        _plot_loss_curve(
            sim["sol_times"],
            sim["sol_losses"],
            label="LR+SOL",
            color="tab:blue",
            max_time=sol_cutoff_time,
            drop_trailing_zero_losses=True,
            mark_trimmed_end=True,
        )
        _plot_loss_curve(
            sim["lr_after_sol_times"],
            sim["lr_after_sol_losses"],
            label="LR AFTER SOL",
            color="tab:red",
            linestyle="-.",
            max_time=validation_t_end,
        )
        ax_err.axvline(
            x=t_train,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"trained t={t_train:.3f}",
        )
        ax_err.set_xlabel("Time")
        ax_err.set_ylabel("L2 Loss")
        ax_err.set_title(f"{problem_name}: Loss During Simulation", fontsize=9)
        # ax_err.set_yscale("log")
        ax_err.legend(fontsize=7)

        if include_ot_vortex_divergence_row and (
            "ot_vortex" in str(problem_name).lower()
        ):
            ax_divergence = fig.add_subplot(gs[4])
            spacing_info = magnetic_divergence_grid_spacings.get(problem_name, {})
            hr_grid_spacing = spacing_info.get("hr")
            lr_grid_spacing = spacing_info.get("lr")
            print(hr_grid_spacing, lr_grid_spacing)

            if hr_grid_spacing is None or lr_grid_spacing is None:
                logger.warning(
                    "Missing grid spacing for normalized magnetic divergence plot in %s.",
                    problem_name,
                )
                ax_divergence.text(
                    0.5,
                    0.5,
                    "No normalized magnetic divergence data available",
                    ha="center",
                    va="center",
                    transform=ax_divergence.transAxes,
                )
            else:
                hr_div_times, hr_div_values = (
                    _compute_normalized_magnetic_divergence_error_series(
                        data.hr_snapshot_data,
                        reg_vars=reg_vars,
                        grid_spacing=hr_grid_spacing,
                        max_time=validation_t_end,
                    )
                )
                lr_div_times, lr_div_values = (
                    _compute_normalized_magnetic_divergence_error_series(
                        data.lr_snapshot_data,
                        reg_vars=reg_vars,
                        grid_spacing=lr_grid_spacing,
                        max_time=validation_t_end,
                    )
                )
                sol_div_times, sol_div_values = (
                    _compute_normalized_magnetic_divergence_error_series(
                        data.lr_sol_snapshot_data,
                        reg_vars=reg_vars,
                        grid_spacing=lr_grid_spacing,
                        max_time=validation_t_end,
                    )
                )

                lr_after_sol_snapshot = (
                    snapshot_data_after_lr_sol[p_idx]
                    if snapshot_data_after_lr_sol is not None
                    and p_idx < len(snapshot_data_after_lr_sol)
                    else None
                )
                lr_after_sol_div_times, lr_after_sol_div_values = (
                    _compute_normalized_magnetic_divergence_error_series(
                        lr_after_sol_snapshot,
                        reg_vars=reg_vars,
                        grid_spacing=lr_grid_spacing,
                        time_offset=t_train,
                        max_time=validation_t_end,
                    )
                )

                def _plot_divergence_curve(
                    times: Optional[np.ndarray],
                    values: Optional[np.ndarray],
                    *,
                    label: str,
                    color: str,
                    linestyle: str = "-",
                ) -> None:
                    if times is None or values is None:
                        return
                    if times.size == 0 or values.size == 0:
                        return
                    ax_divergence.plot(
                        times,
                        values,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                    )

                _plot_divergence_curve(
                    hr_div_times,
                    hr_div_values,
                    label="High resolution",
                    color="black",
                )
                _plot_divergence_curve(
                    lr_div_times,
                    lr_div_values,
                    label="LR",
                    color="tab:orange",
                    linestyle="--",
                )
                _plot_divergence_curve(
                    sol_div_times,
                    sol_div_values,
                    label="LR+SOL",
                    color="tab:blue",
                )
                _plot_divergence_curve(
                    lr_after_sol_div_times,
                    lr_after_sol_div_values,
                    label="LR AFTER SOL",
                    color="tab:red",
                    linestyle="-.",
                )

            ax_divergence.axvline(
                x=t_train,
                color="gray",
                linestyle=":",
                alpha=0.7,
                label=f"trained t={t_train:.3f}",
            )
            ax_divergence.set_xlabel("Time")
            ax_divergence.set_ylabel("Mean |div(B) * dx / B|")
            ax_divergence.set_title(
                "ot_vortex: Normalized magnetic divergence error", fontsize=9
            )
            ax_divergence.grid(alpha=0.3)
            ax_divergence.legend(fontsize=7)

    out_path = plots_dir / f"{figure_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved multiproblem summary plot to {out_path}")
    plt.close(fig)

    # ── Time delta evolution plot ─────────────────────────────────────────────
    # Create a separate plot showing how time delta evolves with total time for
    # HR, LR, and LR+SOL simulations.
    for p_idx, data in enumerate(evaluation_data):
        problem_descriptor = data.problem_descriptor
        problem_nickname_dir = model_dir / problem_descriptor.nickname
        problem_nickname_dir.mkdir(parents=True, exist_ok=True)

        hr_times_arr = np.array(data.hr_snapshot_data.time_points)
        validation_t_end_local = float(hr_times_arr[-1])

        fig_delta, ax_delta = plt.subplots(1, 1, figsize=(8, 5))
        delta_series = simulation_time_deltas.get(problem_descriptor.name, {})
        plotted_any_series = False
        series_style = {
            "HR": {"color": "black", "linestyle": "-"},
            "LR": {"color": "tab:orange", "linestyle": "--"},
            "LR+SOL": {"color": "tab:blue", "linestyle": "-"},
        }
        for series_name in ("HR", "LR", "LR+SOL"):
            deltas = np.array(delta_series.get(series_name, []), dtype=float)
            prepared = _prepare_time_delta_plot_arrays(deltas, validation_t_end_local)
            if prepared is None:
                logger.warning(
                    "No finite time delta entries found for %s (%s); skipping this curve.",
                    problem_descriptor.name,
                    series_name,
                )
                continue

            series_times_total, series_times_delta = prepared
            style = series_style[series_name]
            ax_delta.plot(
                series_times_total,
                series_times_delta,
                label=series_name,
                linewidth=1,
                color=style["color"],
                linestyle=style["linestyle"],
            )
            plotted_any_series = True

        if not plotted_any_series:
            logger.warning(
                "No finite time delta entries found for %s; skipping time delta plot.",
                problem_descriptor.name,
            )
            plt.close(fig_delta)
            continue

        ax_delta.set_xlabel("Total Time")
        ax_delta.set_ylabel("Time Delta (dt)")
        ax_delta.set_title(
            f"{problem_descriptor.name}: Time Delta Evolution", fontsize=10
        )
        ax_delta.grid(True, alpha=0.3)

        # Mark training time
        t_train = trained_t_end[problem_descriptor.name]
        ax_delta.axvline(
            x=t_train,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label=f"trained t={t_train:.3f}",
        )
        ax_delta.legend(fontsize=8)

        time_delta_out_path = problem_nickname_dir / "time_delta_evolution.png"
        plt.savefig(time_delta_out_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved time delta evolution plot to {time_delta_out_path}")
        plt.close(fig_delta)

    if problem_manager.problem_descriptors[0].name == "mhd_blast":
        _plot_shifted_mhd_blast_analysis(
            training_config=training_config,
            problem_manager=problem_manager,
            problem_descriptor=problem_manager.problem_descriptors[0],
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            model_dir=model_dir,
            split_loss_line_at_training_time=split_loss_line_at_training_time,
            trained_t_end=trained_t_end["mhd_blast"],
        )

    pass


if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="->{message}", style="{", level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Create per-problem analysis plots for multiproblem models."
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="multiproblem_test_1",
        help="Model name to process (default: multiproblem_test_1)",
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
        default=MULTIPROBLEM_BASE_DIR,
        help="model parent folder",
    )
    args = parser.parse_args()

    model_name = args.model_name_override or args.model_name
    MULTIPROBLEM_BASE_DIR = args.model_path
    loading_nan_params = False
    multigpu_trained = False
    split_loss_line_at_training_time = True

    model_manager = ModelManager(base_dir=MULTIPROBLEM_BASE_DIR, model_name=model_name)
    model_manager.print_model_info()
    training_config = model_manager.load_training_config()

    problem_descriptor_mhd_blast = ProblemDescriptor(name="mhd_blast")
    problem_descriptor_ot_vortex = ProblemDescriptor(name="ot_vortex")
    problem_descriptor_turbulence = ProblemDescriptor(name="turbulence")
    problem_descriptor_turb_blast = ProblemDescriptor(name="turbulent_blast")

    list_of_problems = [
        # problem_descriptor_mhd_blast,
        problem_descriptor_ot_vortex,
        problem_descriptor_turbulence,
        problem_descriptor_turb_blast,
    ]

    for problem in list_of_problems:
        problem_manager = ProblemManager(
            problem_descriptors=[problem], training_config=training_config
        )

        plot_problem_model_analysis(
            training_config=training_config,
            problem_manager=problem_manager,
            model_name=model_name,
            split_loss_line_at_training_time=split_loss_line_at_training_time,
            loading_nan_params=loading_nan_params,
            figure_name=f"problem_analysis_{problem.name}",
        )
