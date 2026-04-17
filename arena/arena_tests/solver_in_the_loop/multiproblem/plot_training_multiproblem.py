from autocvd import autocvd
import os

autocvd(num_gpus=1)
"""
Multiproblem training summary plot.

Layout (N problems, 3 rows × 3N cols):
  Row 0: density slice at final time per problem (HR downsampled | LR | LR+SOL)
  Row 1: training loss evolution per epoch, one subplot per problem
  Row 2: L2 loss vs simulation time, one subplot per problem

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
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    FiLMCorrectorCNN,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    YAXIS,
    XAXIS,
    ZAXIS,
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
    SimulationBundle,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.dataset.h5_problem_manager import (
    H5ProblemManager,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.dataset.h5_problem_descriptor import (
    H5ProblemDescriptor,
)
from arena.arena_tests.solver_in_the_loop.plot_states_comparison import (
    plot_and_animate_states,
    plot_states,
    plot_and_animate_states_with_diagonal,
)
import tqdm
from typing import Any, Callable, Optional, List

logger = logging.getLogger(__name__)

MULTIPROBLEM_BASE_DIR = "arena/data/models/multiproblem_w_dataset"

NUM_SNAPSHOTS = 50

SNAPSHOT_TIMEPOINTS_VALIDATION = {
    "turbulence": jnp.linspace(0.0, 1.4 * 0.4, endpoint=True, num=NUM_SNAPSHOTS),
    "ot_vortex": jnp.linspace(0.0, 1.4 * jnp.pi, endpoint=True, num=NUM_SNAPSHOTS),
    "mhd_blast": jnp.linspace(0.0, 1.4 * 0.2, endpoint=True, num=NUM_SNAPSHOTS),
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

        if snapshot_data_after_lr_sol is not None:
            lr_after_sol_times = np.array(
                snapshot_data_after_lr_sol[i].time_points + t_train
            )

            print("after sol times", lr_after_sol_times)
            print("target times", data.hr_snapshot_data.time_points[train_idx:])

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


def plot_training_multiproblem(
    neural_net_params,
    neural_net_static,
    training_config: TrainingConfig,
    problem_manager: ProblemManager,
    model_name: str,
    split_loss_line_at_training_time: bool = False,
    figure_name="multiproblem_summary",
):
    """
    Produce a 3-row × (3 × N_problems) summary figure and save it to
    <MULTIPROBLEM_BASE_DIR>/<model_name>/plots/multiproblem_summary.png.

    Row 0: density slice at final simulation time
            columns per problem: HR downsampled | LR | LR+SOL
    Row 1: training loss evolution per epoch (one subplot per problem)
    Row 2: L2 loss vs simulation time (one subplot per problem)
    """
    model_dir = Path(MULTIPROBLEM_BASE_DIR) / model_name
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        correct_from_beggining=training_config.correct_from_beggining,
        start_correction_time=training_config.start_correction_time,
    )
    corrector_params = CNNMHDParams(network_params=neural_net_params)

    evaluation_data = []
    snapshot_data_after_lr_sol = []
    snapshot_timepoints_trained_index = {}

    for problem_descriptor in problem_manager.problem_descriptors:
        problem_manager_problem = ProblemManager(
            problem_descriptors=[problem_descriptor], training_config=training_config
        )

        snapshot_timepoints = SNAPSHOT_TIMEPOINTS_VALIDATION[problem_descriptor.name]

        if split_loss_line_at_training_time:
            print(PROBLEM_DICTIONARY[problem_descriptor.name]["t_end"])
            snapshot_timepoints = jnp.sort(
                jnp.concatenate(
                    [
                        snapshot_timepoints,
                        jnp.array(
                            [PROBLEM_DICTIONARY[problem_descriptor.name]["t_end"]]
                        ),
                    ]
                )
            )

        evaluation_data_problem = problem_manager_problem.get_evaluation_data(
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            snapshot_timepoints=SNAPSHOT_TIMEPOINTS_VALIDATION[problem_descriptor.name],
        )
        evaluation_data.append(evaluation_data_problem[0])
        if split_loss_line_at_training_time:
            assert isinstance(
                evaluation_data_problem[0].lr_sol_snapshot_data, SnapshotData
            )
            snapshot_timepoints_trained_index[problem_descriptor.name] = int(
                jnp.argmax(
                    snapshot_timepoints
                    == PROBLEM_DICTIONARY[problem_descriptor.name]["t_end"]
                )
            )
            logger.info(f"snapshot_timepoints {snapshot_timepoints}")
            logger.info(
                f"snapshot_trained_t_index: {snapshot_timepoints_trained_index}",
            )
            logger.info(
                f"len of remaining snapshots {len(evaluation_data_problem[0].hr_snapshot_data.states[snapshot_timepoints_trained_index[problem_descriptor.name] :])}"
            )
            state_at_training_time = evaluation_data_problem[
                0
            ].lr_sol_snapshot_data.states[
                snapshot_timepoints_trained_index[problem_descriptor.name]
            ]
            problem = PROBLEM_CATALOG[problem_descriptor.name](
                **problem_descriptor.params
            )

            _, config, params, registered_variables = (
                problem.generate_initial_state_with_config_overrides(
                    config=problem_manager_problem.hr_config,
                    params=problem_manager_problem.hr_params,
                    config_overrides=problem.get_config_overrides_evaluation(
                        snapshot_timepoints=jnp.linspace(
                            start=0.0,
                            stop=snapshot_timepoints[-1]
                            - snapshot_timepoints[
                                snapshot_timepoints_trained_index[
                                    problem_descriptor.name
                                ]
                            ],
                            num=NUM_SNAPSHOTS
                            - snapshot_timepoints_trained_index[
                                problem_descriptor.name
                            ],
                        )
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

    if not evaluation_data:
        logger.warning("No evaluation data available; skipping plot.")
        return

    num_problems = len(evaluation_data)
    reg_vars = _get_reg_vars()

    # Build trained t_end per problem name (one t_end per problem type, as per training)
    trained_t_end: dict[str, float] = {
        problem: PROBLEM_DICTIONARY[problem]["t_end"] for problem in PROBLEM_DICTIONARY
    }
    for data in evaluation_data:
        p_name = data.problem_descriptor.name
        if p_name not in trained_t_end:
            problem = PROBLEM_CATALOG[p_name](**data.problem_descriptor.params)
            trained_t_end[p_name] = float(problem.t_end)

    # Load epoch losses saved by training_model.py
    losses_path = model_dir / "losses_per_problem.npz"
    if losses_path.exists():
        losses_data = np.load(losses_path)
        losses_avg = losses_data["avg"]
        losses_by_problem = {
            data.problem_descriptor.name: losses_data[data.problem_descriptor.name]
            for data in evaluation_data
            if data.problem_descriptor.name in losses_data.files
        }
    else:
        losses_file = model_dir / "losses.npz"
        losses_avg = np.load(losses_file)["losses"]
        losses_by_problem = {}
    if len(snapshot_data_after_lr_sol) == 0:
        snapshot_data_after_lr_sol = None

    sim_losses = _compute_sim_losses(
        evaluation_data,
        training_config,
        trained_t_end,
        snapshot_data_after_lr_sol=snapshot_data_after_lr_sol,
    )

    # ── Figure layout ────────────────────────────────────────────────────────
    fig_width = max(12, 4.5 * num_problems * 3)
    fig = plt.figure(figsize=(fig_width, 12))
    gs = GridSpec(
        3,
        3 * num_problems,
        height_ratios=[1.5, 1, 1],
        hspace=0.50,
        wspace=0.40,
    )
    colors = plt.cm.tab10.colors

    for p_idx, data in enumerate(evaluation_data):
        problem_name = data.problem_descriptor.name
        col_start = p_idx * 3
        color = colors[p_idx % len(colors)]
        t_train = trained_t_end[problem_name]

        # ── Row 0: density slices at the trained time ─────────────────────────
        hr_states = data.hr_snapshot_data.states
        hr_times = np.array(data.hr_snapshot_data.time_points)
        hr_train_idx = int(np.argmin(np.abs(hr_times - t_train)))
        lr_target_at_train = downaverage(
            hr_states[hr_train_idx], training_config.downaverage_factor
        )

        if data.lr_snapshot_data is not None:
            lr_times = np.array(data.lr_snapshot_data.time_points)
            lr_train_idx = int(np.argmin(np.abs(lr_times - t_train)))
            state_lr_at_train = data.lr_snapshot_data.states[lr_train_idx]
        else:
            state_lr_at_train = None

        if data.lr_sol_snapshot_data is not None:
            sol_times = np.array(data.lr_sol_snapshot_data.time_points)
            sol_train_idx = int(np.argmin(np.abs(sol_times - t_train)))
            state_sol_at_train = data.lr_sol_snapshot_data.states[sol_train_idx]
        else:
            state_sol_at_train = None

        states_trio = [lr_target_at_train, state_lr_at_train, state_sol_at_train]
        col_titles = ["HR downsampled", "LR", "LR + SOL"]
        valid_states = [s for s in states_trio if s is not None]
        vmin = float(min(jnp.min(s[reg_vars.density_index]) for s in valid_states))
        vmax = float(max(jnp.max(s[reg_vars.density_index]) for s in valid_states))
        num_cells_lr = lr_target_at_train.shape[-1]

        for c_idx, (state, col_title) in enumerate(zip(states_trio, col_titles)):
            ax = fig.add_subplot(gs[0, col_start + c_idx])
            if state is not None:
                im = ax.imshow(
                    state[reg_vars.density_index, :, :, num_cells_lr // 2],
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            title = (
                f"{problem_name}  (t={t_train:.3f})\n{col_title}"
                if c_idx == 0
                else col_title
            )
            ax.set_title(title, fontsize=9)

        # ── Row 1: loss per epoch ─────────────────────────────────────────────
        ax_loss = fig.add_subplot(gs[1, col_start : col_start + 3])
        if problem_name in losses_by_problem:
            ax_loss.plot(
                losses_by_problem[problem_name],
                color=color,
                linewidth=1.2,
                label=problem_name,
            )
        ax_loss.plot(
            losses_avg,
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.6,
            label="avg",
        )
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{problem_name}: Loss per Epoch", fontsize=9)
        ax_loss.set_yscale("log")
        ax_loss.legend(fontsize=7)

        # ── Row 2: loss during simulation ─────────────────────────────────────
        ax_err = fig.add_subplot(gs[2, col_start : col_start + 3])
        sim = sim_losses[p_idx]
        if sim["lr_times"] is not None and sim["lr_losses"] is not None:
            ax_err.plot(
                sim["lr_times"],
                sim["lr_losses"],
                label="LR",
                color="tab:orange",
                linestyle="--",
            )
        if sim["sol_times"] is not None and sim["sol_losses"] is not None:
            ax_err.plot(
                sim["sol_times"],
                sim["sol_losses"],
                label="LR+SOL",
                color="tab:blue",
            )
        if snapshot_data_after_lr_sol is not None:
            ax_err.plot(
                sim["lr_after_sol_times"],
                sim["lr_after_sol_losses"],
                label="LR AFTER SOL",
                color="tab:red",
                linestyle="-.",
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
        ax_err.set_yscale("log")
        ax_err.legend(fontsize=7)

    out_path = plots_dir / f"{figure_name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved multiproblem summary plot to {out_path}")
    plt.close(fig)


def plot_losses_multigpu(
    losses_path: str | Path,
    output_path: str | Path,
):
    """Plot loss evolution from a losses_per_problem.npz file.

    Layout (2 rows):
      Row 0: per-problem loss per epoch, one subplot per problem
      Row 1: average loss per epoch (full width)
    """
    losses_data = np.load(losses_path)
    losses_avg = losses_data["avg"]

    fig = plt.figure(figsize=(max(6, 8), 8))
    gs = GridSpec(2, 1, hspace=0.45, wspace=0.35)

    # Row 1: average (full width)
    ax_avg = fig.add_subplot(gs[:])
    ax_avg.plot(losses_avg, color="black", linewidth=1.5)
    ax_avg.set_xlabel("Epoch")
    ax_avg.set_ylabel("Loss")
    ax_avg.set_title("Average Loss per Epoch", fontsize=10)
    ax_avg.set_yscale("log")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved losses plot to {output_path}")
    plt.close(fig)


def plot_losses_multiproblem(
    losses_path: str | Path,
    output_path: str | Path,
):
    """Plot loss evolution from a losses_per_problem.npz file.

    Layout (2 rows):
      Row 0: per-problem loss per epoch, one subplot per problem
      Row 1: average loss per epoch (full width)
    """
    losses_data = np.load(losses_path)
    losses_avg = losses_data["avg"]
    problem_names = [k for k in losses_data.files if k != "avg"]
    num_problems = max(len(problem_names), 1)
    logger.info(f"num of problems: {num_problems}")

    fig = plt.figure(figsize=(max(6 * num_problems, 8), 8))
    gs = GridSpec(2, num_problems, hspace=0.45, wspace=0.35)
    colors = plt.cm.tab10.colors

    # Row 0: per-problem
    for p_idx, name in enumerate(problem_names):
        ax = fig.add_subplot(gs[0, p_idx])
        ax.plot(losses_data[name], color=colors[p_idx % len(colors)], linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{name}: Loss per Epoch", fontsize=9)
        ax.set_yscale("log")

    # Row 1: average (full width)
    ax_avg = fig.add_subplot(gs[1, :])
    ax_avg.plot(losses_avg, color="black", linewidth=1.5)
    ax_avg.set_xlabel("Epoch")
    ax_avg.set_ylabel("Loss")
    ax_avg.set_title("Average Loss per Epoch", fontsize=10)
    ax_avg.set_yscale("log")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved losses plot to {output_path}")
    plt.close(fig)


# def model_output_figures(
#     corrections: list[Array],
#     effective_corrections: list[Array],
#     states: list[Array],
#     times: list[float],
#     output_dir: Path,
# ):
#     reg_vars = _get_reg_vars()
#     corrections_array = np.asarray(corrections)
#     effective_corrections_array = np.asarray(effective_corrections)
#     states_array = np.asarray(states)
#     times_total = np.cumsum(np.asarray(times, dtype=float))
#     print(corrections_array.shape, states_array.shape)
#     if (
#         corrections_array.size == 0
#         or effective_corrections_array.size == 0
#         or states_array.size == 0
#         or times_total.size == 0
#     ):
#         logger.warning("Model output figure skipped: empty callback arrays.")
#         return
#
#     assert isinstance(reg_vars.velocity_index, StaticIntVector)
#     assert isinstance(reg_vars.magnetic_index, StaticIntVector)
#     assert isinstance(reg_vars.interface_magnetic_field_index, StaticIntVector)
#
#     velocity_states = np.sqrt(
#         states_array[:, reg_vars.velocity_index[0]] ** 2
#         + states_array[:, reg_vars.velocity_index[1]] ** 2
#         + states_array[:, reg_vars.velocity_index[2]] ** 2
#     )
#     velocity_corr = np.sqrt(
#         corrections_array[:, reg_vars.velocity_index[0]] ** 2
#         + corrections_array[:, reg_vars.velocity_index[1]] ** 2
#         + corrections_array[:, reg_vars.velocity_index[2]] ** 2
#     )
#     velocity_effective_corr = np.sqrt(
#         effective_corrections_array[:, reg_vars.velocity_index[0]] ** 2
#         + effective_corrections_array[:, reg_vars.velocity_index[1]] ** 2
#         + effective_corrections_array[:, reg_vars.velocity_index[2]] ** 2
#     )
#
#     velocity_corr_norm = velocity_corr / velocity_states
#     velocity_effective_corr_norm = velocity_effective_corr / velocity_states
#
#     magnetic_states = np.sqrt(
#         states_array[:, reg_vars.magnetic_index[0]] ** 2
#         + states_array[:, reg_vars.magnetic_index[1]] ** 2
#         + states_array[:, reg_vars.magnetic_index[2]] ** 2
#     )
#     magnetic_corr = np.sqrt(
#         corrections_array[:, reg_vars.magnetic_index[0]] ** 2
#         + corrections_array[:, reg_vars.magnetic_index[1]] ** 2
#         + corrections_array[:, reg_vars.magnetic_index[2]] ** 2
#     )
#
#     interface_magnetic_states = np.sqrt(
#         states_array[:, reg_vars.interface_magnetic_field_index[0]] ** 2
#         + states_array[:, reg_vars.interface_magnetic_field_index[1]] ** 2
#         + states_array[:, reg_vars.interface_magnetic_field_index[2]] ** 2
#     )
#     magnetic_effective_corr = np.sqrt(
#         effective_corrections_array[:, reg_vars.interface_magnetic_field_index[0]] ** 2
#         + effective_corrections_array[:, reg_vars.interface_magnetic_field_index[1]]
#         ** 2
#         + effective_corrections_array[:, reg_vars.interface_magnetic_field_index[2]]
#         ** 2
#     )
#
#     magnetic_corr_norm = magnetic_corr / magnetic_states
#     magnetic_effective_corr_norm = magnetic_effective_corr / interface_magnetic_states
#
#     density_states = states_array[:, reg_vars.density_index]
#     density_corr = corrections_array[:, reg_vars.density_index]
#     density_effective_corr = effective_corrections_array[:, reg_vars.density_index]
#     density_corr_norm = density_corr / density_states
#     density_effective_corr_norm = density_effective_corr / density_states
#
#     pressure_states = states_array[:, reg_vars.pressure_index]
#     pressure_corr = corrections_array[:, reg_vars.pressure_index]
#     pressure_effective_corr = effective_corrections_array[:, reg_vars.pressure_index]
#     pressure_corr_norm = pressure_corr / pressure_states
#     pressure_effective_corr_norm = pressure_effective_corr / pressure_states
#
#     pressures = [
#         pressure_corr,
#         pressure_corr_norm,
#         pressure_effective_corr,
#         pressure_effective_corr_norm,
#     ]
#     densities = [
#         density_corr,
#         density_corr_norm,
#         density_effective_corr,
#         density_effective_corr_norm,
#     ]
#     magnetics = [
#         magnetic_corr,
#         magnetic_corr_norm,
#         magnetic_effective_corr,
#         magnetic_effective_corr_norm,
#     ]
#     velocities = [
#         velocity_corr,
#         velocity_corr_norm,
#         velocity_effective_corr,
#         velocity_effective_corr_norm,
#     ]
#
#     titles = [
#         "unnormalized",
#         "normalized",
#         "effective_unnormalized",
#         "effective_normalized",
#     ]
#     ylabels = [
#         "Correction",
#         "Correction/state",
#         "Correction",
#         "Correction/state",
#     ]
#
#     for pressure, density, magnetic, velocity, title, ylabel in zip(
#         pressures, densities, magnetics, velocities, titles, ylabels, strict=True
#     ):
#         plot_corrections_figure(
#             density_corr=density,
#             pressure_corr=pressure,
#             velocity_corr_mag=velocity,
#             magnetic_corr_mag=magnetic,
#             times_total=times_total,
#             bins=30,
#             title_suffix=title,
#             ylabel=ylabel,
#             save_path=output_dir / f"model_output_{title}.png",
#         )
#
#     plot_channel_time_evolution_figures(
#         states_array=states_array,
#         model_output_array=corrections_array,
#         times_total=times_total,
#         reg_vars=reg_vars,
#         output_dir=output_dir,
#     )
#


def model_output_figures(
    corrections: List[Array],
    effective_corrections: List[Array],
    states: List[Array],
    times: List[float],
    output_dir: Path,
):
    reg_vars = _get_reg_vars()
    corrections_array = np.array(corrections)
    e_corrections_array = np.array(effective_corrections)
    states_array = np.array(states)
    times_delta = np.array(times)
    times_total = np.cumsum(times_delta)

    assert isinstance(reg_vars.velocity_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)

    def _extract_component(values: np.ndarray, component_name: str) -> np.ndarray:
        if component_name == "density":
            return values[:, reg_vars.density_index]
        if component_name == "pressure":
            return values[:, reg_vars.pressure_index]
        if component_name == "velocity":
            return np.sqrt(
                values[:, reg_vars.velocity_index[0]] ** 2
                + values[:, reg_vars.velocity_index[1]] ** 2
                + values[:, reg_vars.velocity_index[2]] ** 2
            )
        if component_name == "magnetic":
            return np.sqrt(
                values[:, reg_vars.magnetic_index[0]] ** 2
                + values[:, reg_vars.magnetic_index[1]] ** 2
                + values[:, reg_vars.magnetic_index[2]] ** 2
            )
        raise ValueError(f"Unknown component_name: {component_name}")

    component_names = ("density", "pressure", "velocity", "magnetic")
    state_components = {
        name: _extract_component(states_array, name) for name in component_names
    }
    correction_groups = [
        ("", corrections_array),
        ("effective_", e_corrections_array),
    ]
    time_multipliers = [
        ("", times_delta),
        ("_time_derivative", np.ones_like(times_delta)),
    ]

    for suffix, time_multiplier in time_multipliers:
        for group_prefix, correction_array in correction_groups:
            group_components = {
                name: _extract_component(correction_array, name)
                for name in component_names
            }
            normalized_components = {
                name: np.log10(np.abs(time_multiplier * group_components[name]) + 1e-8)
                - np.log10(state_components[name] + 1e-8)
                for name in component_names
            }

            variants = [
                ("unnormalized", group_components, "Correction"),
                ("normalized", normalized_components, "Correction/state"),
            ]
            for norm_name, values, ylabel in variants:
                title = f"{group_prefix}{norm_name}{suffix}"
                plot_corrections_figure(
                    density_corr=values["density"],
                    pressure_corr=values["pressure"],
                    velocity_corr_mag=values["velocity"],
                    magnetic_corr_mag=values["magnetic"],
                    times_total=times_total,
                    bins=100,
                    title_suffix=title,
                    ylabel=ylabel,
                    save_path=output_dir / f"model_output_{title}.png",
                )

    plot_channel_time_evolution_figures(
        states_array=states_array,
        model_output_array=corrections_array,
        times_total=times_total,
        reg_vars=reg_vars,
        output_dir=output_dir,
    )


#

# def plot_corrections_figure(
#     *,
#     density_states: np.ndarray,
#     pressure_states: np.ndarray,
#     velocity_states_mag: np.ndarray,
#     magnetic_states_mag: np.ndarray,
#     density_corr: np.ndarray,
#     pressure_corr: np.ndarray,
#     velocity_corr_mag: np.ndarray,
#     magnetic_corr_mag: np.ndarray,
#     times_total: np.ndarray,
#     bins: int,
#     title_suffix: str,
#     ylabel: str,
#     output_path: Path,
# ):
#     fig, axes = plt.subplots(3, 1, figsize=(9, 15))
#     ax_hist_states, ax_hist_corr, ax_time = axes
#
#     ax_hist_states.hist(
#         density_states, bins=bins, density=False, alpha=0.6, label="Density"
#     )
#     ax_hist_states.hist(
#         pressure_states, bins=bins, density=False, alpha=0.6, label="Pressure"
#     )
#     ax_hist_states.hist(
#         velocity_states_mag, bins=bins, density=False, alpha=0.6, label="Velocity"
#     )
#     ax_hist_states.hist(
#         magnetic_states_mag,
#         bins=bins,
#         density=False,
#         alpha=0.6,
#         label="Magnetic_interface",
#     )
#     ax_hist_states.set_title(f"Distribution of states average ({title_suffix})")
#     ax_hist_states.set_xlabel("Correction value")
#     ax_hist_states.set_ylabel("Probability density")
#     ax_hist_states.legend()
#     ax_hist_states.grid(alpha=0.3)
#
#     ax_hist_corr.hist(
#         density_corr, bins=bins, density=False, alpha=0.6, label="Density"
#     )
#     ax_hist_corr.hist(
#         pressure_corr, bins=bins, density=False, alpha=0.6, label="Pressure"
#     )
#     ax_hist_corr.hist(
#         velocity_corr_mag, bins=bins, density=False, alpha=0.6, label="Velocity"
#     )
#     ax_hist_corr.hist(
#         magnetic_corr_mag,
#         bins=bins,
#         density=False,
#         alpha=0.6,
#         label="Magnetic_interface",
#     )
#     ax_hist_corr.set_title(f"Distribution of corrections average ({title_suffix})")
#     ax_hist_corr.set_xlabel("Correction value")
#     ax_hist_corr.set_ylabel("Probability density")
#     ax_hist_corr.legend()
#     ax_hist_corr.grid(alpha=0.3)
#
#     ax_time.plot(times_total, density_corr, label="Density")
#     ax_time.plot(times_total, pressure_corr, label="Pressure")
#     ax_time.plot(times_total, velocity_corr_mag, label="Velocity")
#     ax_time.plot(times_total, magnetic_corr_mag, label="Magnetic interface")
#     ax_time.set_title(f"Model output ({title_suffix})")
#     ax_time.set_xlabel("Times")
#     ax_time.set_ylabel(ylabel)
#     ax_time.legend()
#
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=400)
#     plt.close(fig)
#     logger.info(f"Saved model output figure to {output_path}")
#


def plot_corrections_figure(
    *,
    density_corr,
    pressure_corr,
    velocity_corr_mag,
    magnetic_corr_mag,
    times_total,
    bins,
    title_suffix,
    ylabel,
    save_path,
):
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    ax_hist, ax_time = axes

    # -----------------------
    # Histogram
    # -----------------------
    ax_hist.hist(density_corr, bins=bins, density=False, alpha=0.6, label="Density")
    ax_hist.hist(pressure_corr, bins=bins, density=False, alpha=0.6, label="Pressure")
    ax_hist.hist(
        velocity_corr_mag, bins=bins, density=False, alpha=0.6, label="Velocity"
    )
    ax_hist.hist(
        magnetic_corr_mag, bins=bins, density=False, alpha=0.6, label="Magnetic"
    )

    ax_hist.set_title(f"Distribution of corrections ({title_suffix})")
    ax_hist.set_xlabel("Correction value")
    ax_hist.set_ylabel("Probability density")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    # -----------------------
    # Time evolution
    # -----------------------
    ax_time.plot(times_total, density_corr, label="Density")
    ax_time.plot(times_total, pressure_corr, label="Pressure")

    ax_time.plot(times_total, velocity_corr_mag, label="Velocity")
    ax_time.plot(times_total, magnetic_corr_mag, label="Magnetic")

    ax_time.set_title(f"Model output ({title_suffix})")
    ax_time.set_xlabel("Times")
    ax_time.set_ylabel(ylabel)
    ax_time.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_channel_time_evolution_figures(
    *,
    states_array: np.ndarray,
    model_output_array: np.ndarray,
    times_total: np.ndarray,
    reg_vars,
    output_dir: Path,
):
    """Plot per-channel mean state and model output (unnormalized) over time."""
    assert isinstance(reg_vars.velocity_index, StaticIntVector)
    assert isinstance(reg_vars.magnetic_index, StaticIntVector)

    _plot_scalar_channel_evolution(
        times_total=times_total,
        state_values=states_array[:, reg_vars.density_index],
        output_values=model_output_array[:, reg_vars.density_index],
        channel_label="density",
        output_path=output_dir / "model_output_time_evolution_density.png",
    )
    _plot_scalar_channel_evolution(
        times_total=times_total,
        state_values=states_array[:, reg_vars.pressure_index],
        output_values=model_output_array[:, reg_vars.pressure_index],
        channel_label="pressure",
        output_path=output_dir / "model_output_time_evolution_pressure.png",
    )
    _plot_vector_channel_group_evolution(
        times_total=times_total,
        states_array=states_array,
        model_output_array=model_output_array,
        component_indices=(
            reg_vars.velocity_index.x,
            reg_vars.velocity_index.y,
            reg_vars.velocity_index.z,
        ),
        component_labels=("vx", "vy", "vz"),
        title="Velocity channels: state vs model output",
        output_path=output_dir / "model_output_time_evolution_velocity.png",
    )
    _plot_vector_channel_group_evolution(
        times_total=times_total,
        states_array=states_array,
        model_output_array=model_output_array,
        component_indices=(
            reg_vars.magnetic_index.x,
            reg_vars.magnetic_index.y,
            reg_vars.magnetic_index.z,
        ),
        component_labels=("Bx", "By", "Bz"),
        title="Magnetic channels: state vs model output",
        output_path=output_dir / "model_output_time_evolution_magnetic.png",
    )


def _plot_scalar_channel_evolution(
    *,
    times_total: np.ndarray,
    state_values: np.ndarray,
    output_values: np.ndarray,
    channel_label: str,
    output_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.plot(times_total, state_values, label=f"State mean ({channel_label})")
    ax.plot(times_total, output_values, label=f"Model output mean ({channel_label})")
    ax.set_title(f"{channel_label.capitalize()}: state and model output over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean value")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    logger.info(f"Saved channel evolution figure to {output_path}")


def _plot_vector_channel_group_evolution(
    *,
    times_total: np.ndarray,
    states_array: np.ndarray,
    model_output_array: np.ndarray,
    component_indices: tuple[int, int, int],
    component_labels: tuple[str, str, str],
    title: str,
    output_path: Path,
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for idx, label in zip(component_indices, component_labels, strict=True):
        ax.plot(times_total, states_array[:, idx], label=f"State mean ({label})")
        ax.plot(
            times_total,
            model_output_array[:, idx],
            linestyle="--",
            label=f"Model output mean ({label})",
        )
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean value")
    ax.legend(ncol=2, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=400)
    plt.close(fig)
    logger.info(f"Saved channel evolution figure to {output_path}")


def _snapshot_callback_factory(
    *,
    corrections: list[Array],
    effective_corrections: list[Array],
    states: list[Array],
    times: list[float],
):
    reg_vars = _get_reg_vars()
    params = SimulationParams()

    def snapshot_callable(time, state, correction):
        corrections.append(jnp.mean(correction, axis=[1, 2, 3]))
        states.append(jnp.mean(state, axis=[1, 2, 3]))
        times.append(float(time))

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

        corrected_state = corrected_state.at[reg_vars.pressure_index].set(
            jnp.maximum(
                corrected_state[reg_vars.pressure_index], params.minimum_pressure
            )
        )
        corrected_state = corrected_state.at[reg_vars.density_index].set(
            jnp.maximum(corrected_state[reg_vars.density_index], params.minimum_density)
        )

        effective_corrections.append(
            jnp.mean(corrected_state - original_state, axis=[1, 2, 3])
        )
        pass

    return snapshot_callable


def plot_model_output_figures_per_problem(
    param_type: str,
    model_manager: ModelManager,
    training_config: TrainingConfig,
    problem_manager: ProblemManager,
    model_name: str,
):
    model_dir = Path(MULTIPROBLEM_BASE_DIR) / model_name
    model_output_root = model_dir / "model output"
    model_output_root.mkdir(parents=True, exist_ok=True)
    reg_vars = _get_reg_vars()

    make_animation = False
    num_snapshots = 50

    # TODO: RATHER HARDCODED SOLUTION
    # there should be a "dictionary of problems"
    # NOTE:
    # obsolete piece of code, gonna leave it here in case its useful in the future
    if make_animation:
        snapshot_timepoints = {
            "turbulence": jnp.linspace(0.0, 0.4, endpoint=True, num=num_snapshots),
            "ot_vortex": jnp.linspace(0.0, jnp.pi, endpoint=True, num=num_snapshots),
            "mhd_blast": jnp.linspace(0.0, 0.2, endpoint=True, num=num_snapshots),
        }
    else:
        snapshot_timepoints = {"turbulence": None, "ot_vortex": None, "mhd_blast": None}

    for problem_descriptor in problem_manager.problem_descriptors:
        corrections: list[Array] = []
        effective_corrections: list[Array] = []
        states: list[Array] = []
        times: list[float] = []

        snapshot_callable = _snapshot_callback_factory(
            corrections=corrections,
            effective_corrections=effective_corrections,
            states=states,
            times=times,
        )

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
        neural_net_params = model_manager.load_model_params(
            like=neural_net_params_shape, param_type=param_type
        )

        model = eqx.combine(neural_net_params, callback_static)

        corrector_config = CNNMHDconfig(
            cnn_mhd_corrector=True,
            network_static=callback_static,
            correct_from_beggining=True,
            start_correction_time=0.0,
        )
        corrector_params = CNNMHDParams(network_params=neural_net_params)

        single_problem_manager = ProblemManager(
            problem_descriptors=[problem_descriptor],
            training_config=training_config,
        )
        evaluation_data = single_problem_manager.get_evaluation_data(
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            snapshot_timepoints=snapshot_timepoints[problem_descriptor.name],
        )
        problem_data = evaluation_data[0]

        assert isinstance(problem_data.lr_snapshot_data, SnapshotData)
        assert isinstance(problem_data.lr_sol_snapshot_data, SnapshotData)

        if make_animation:
            plot_and_animate_states(
                [
                    problem_data.lr_snapshot_data.states,
                    problem_data.lr_sol_snapshot_data.states,
                    problem_data.hr_snapshot_data.states,
                ],
                [16, 16, 32],
                timepoints=problem_data.lr_sol_snapshot_data.time_points,
                save_path=Path(model_output_root, problem_descriptor.nickname),
                titles=["LR", "LR SOL", "HR"],
            )

        if not evaluation_data:
            logger.warning(
                "Skipping model output figures for '%s': no evaluation data.",
                problem_descriptor.nickname,
            )
            continue

        output_dir = model_output_root / problem_descriptor.nickname
        output_dir.mkdir(parents=True, exist_ok=True)
        model_output_figures(
            corrections=corrections,
            effective_corrections=effective_corrections,
            states=states,
            times=times,
            output_dir=output_dir,
        )


def _compute_final_losses(
    evaluation_data: list[EvaluationResults],
    training_config,
    snapshot_timepoints_idx: list[int],
) -> tuple[list[float], list[float], list[jnp.ndarray], list[jnp.ndarray]]:
    final_loss_initial = []
    final_loss_corrected = []
    all_state_loss_initial = []
    all_state_loss_corrected = []

    for data in evaluation_data:
        hr_target_states = data.hr_snapshot_data.states
        lr_target_states = downaverage(
            hr_target_states, downaverage_factor=training_config.downaverage_factor
        )

        # TODO : change this to accept per channel mse loss
        loss_fn_kwargs, loss_fn_factory = loss_setup(
            training_config=training_config,
            target_states=lr_target_states[jnp.array(snapshot_timepoints_idx)],
        )
        loss_fn = partial(loss_fn_factory, **loss_fn_kwargs)
        final_lr_target_state = lr_target_states[snapshot_timepoints_idx[-1]]

        if data.lr_snapshot_data is not None:
            final_lr_state = data.lr_snapshot_data.states[snapshot_timepoints_idx[-1]]
            l2_error_initial = float(loss_fn(final_lr_state, final_lr_target_state))
            final_loss_initial.append(l2_error_initial)
        else:
            final_loss_initial.append(float("nan"))

        if data.lr_sol_snapshot_data is not None:
            final_lr_sol_state = data.lr_sol_snapshot_data.states[
                snapshot_timepoints_idx[-1]
            ]
            l2_error_corrected = float(
                loss_fn(final_lr_sol_state, final_lr_target_state)
            )
            final_loss_corrected.append(l2_error_corrected)
        else:
            final_loss_corrected.append(float("nan"))

        v_loss = jax.vmap(loss_fn, in_axes=(0, 0))
        if data.lr_snapshot_data is not None:
            l2_errors_uncorrected = v_loss(
                data.lr_snapshot_data.states, lr_target_states
            )
        else:
            l2_errors_uncorrected = jnp.full(
                (lr_target_states.shape[0],), jnp.nan, dtype=lr_target_states.dtype
            )
        all_state_loss_initial.append(l2_errors_uncorrected)

        if data.lr_sol_snapshot_data is not None:
            l2_errors_corrected = v_loss(
                data.lr_sol_snapshot_data.states, lr_target_states
            )
            all_state_loss_corrected.append(l2_errors_corrected)
        else:
            all_state_loss_corrected.append(
                jnp.full_like(l2_errors_uncorrected, fill_value=jnp.nan)
            )

    return (
        final_loss_initial,
        final_loss_corrected,
        all_state_loss_initial,
        all_state_loss_corrected,
    )


def _interpolate_nan_for_plot(
    x_values: np.ndarray, values: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    y_values = np.asarray(values, dtype=float)
    nan_mask = np.isnan(y_values)
    if not np.any(nan_mask):
        return y_values, nan_mask

    valid_mask = ~nan_mask
    if not np.any(valid_mask):
        return np.zeros_like(y_values), nan_mask

    y_interp = np.interp(x_values, x_values[valid_mask], y_values[valid_mask])
    y_plot = y_values.copy()
    y_plot[nan_mask] = y_interp[nan_mask]
    return y_plot, nan_mask


def _decode_ot_vortex_axis(axis_value: Any) -> str:
    if isinstance(axis_value, (str, np.str_)):
        return str(axis_value)
    axis_int = int(axis_value)
    axis_map = {0: "x", 1: "y", 2: "z"}
    if axis_int not in axis_map:
        raise ValueError(f"Unsupported vortex axis encoding: {axis_value}")
    return axis_map[axis_int]


def _extract_theta_phi_from_b_direction(b_direction: Any) -> tuple[float, float]:
    vec = np.asarray(b_direction, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"Invalid B_direction with zero norm: {b_direction}")
    unit = vec / norm
    theta = float(np.arccos(np.clip(unit[2], -1.0, 1.0)))
    phi = float(np.mod(np.arctan2(unit[1], unit[0]), 2.0 * np.pi))
    return theta, phi


def _get_mhd_scan_parameter_from_hyperparams(
    scan_parameter_key: str, hyperparams: dict[str, Any]
) -> float:
    if scan_parameter_key == "B0":
        return float(hyperparams["B0"])
    if scan_parameter_key in {"theta", "phi"}:
        theta, phi = _extract_theta_phi_from_b_direction(hyperparams["B_direction"])
        return theta if scan_parameter_key == "theta" else phi
    raise ValueError(f"Unsupported MHD scan parameter key: {scan_parameter_key}")


def _order_mhd_blast_scan_descriptors(
    descriptors: list[H5ProblemDescriptor],
    *,
    parameter_values: list[float],
    scan_parameter_key: str,
    h5_problem_manager: H5ProblemManager,
) -> list[H5ProblemDescriptor]:
    descriptor_values: list[tuple[float, H5ProblemDescriptor]] = []
    for descriptor in descriptors:
        hyperparams = h5_problem_manager.get_problem_hyperparams(
            problem_name=descriptor.problem_name,
            index=descriptor.problem_index,
        )
        scan_value = _get_mhd_scan_parameter_from_hyperparams(
            scan_parameter_key, hyperparams
        )
        descriptor_values.append((scan_value, descriptor))

    remaining = descriptor_values.copy()
    ordered: list[H5ProblemDescriptor] = []
    for target_value in parameter_values:
        if not remaining:
            raise ValueError(
                f"Not enough validation descriptors for {scan_parameter_key} scan."
            )
        diffs = [abs(scan_value - target_value) for scan_value, _ in remaining]
        best_idx = int(np.argmin(np.asarray(diffs, dtype=float)))
        matched_value, matched_descriptor = remaining.pop(best_idx)
        if not np.isclose(matched_value, target_value, atol=1e-6, rtol=1e-5):
            raise ValueError(
                "Validation descriptor parameter mismatch for "
                f"{scan_parameter_key}: expected {target_value}, got {matched_value}"
            )
        ordered.append(matched_descriptor)

    if remaining:
        logger.warning(
            "Validation dataset for %s has %d extra descriptors not used by this scan.",
            scan_parameter_key,
            len(remaining),
        )
    return ordered


def _load_mhd_blast_validation_scan_inputs(
    *,
    training_config,
    validation_data_path: Path,
    parameter_values_by_scan: dict[str, list[float]],
) -> dict[str, tuple[H5ProblemManager, list[H5ProblemDescriptor]]]:
    scan_to_parameter_key = {"b0": "B0", "theta": "theta", "phi": "phi"}
    scan_inputs: dict[str, tuple[H5ProblemManager, list[H5ProblemDescriptor]]] = {}

    for scan_name, scan_values in parameter_values_by_scan.items():
        h5_path = validation_data_path / f"validation_dataset_blast_{scan_name}.h5"
        if not h5_path.exists():
            logger.warning("Validation dataset not found: %s", h5_path)
            continue

        h5_problem_manager = H5ProblemManager(
            h5_file_paths={"mhd_blast": str(h5_path)},
            training_config=training_config,
        )
        descriptors = h5_problem_manager.get_problem_descriptors()
        ordered_descriptors = _order_mhd_blast_scan_descriptors(
            descriptors,
            parameter_values=scan_values,
            scan_parameter_key=scan_to_parameter_key[scan_name],
            h5_problem_manager=h5_problem_manager,
        )
        scan_inputs[scan_name] = (h5_problem_manager, ordered_descriptors)

    return scan_inputs


def _order_ot_vortex_scan_descriptors(
    descriptors: list[H5ProblemDescriptor],
    *,
    parameter_values: list[float],
    vortex_axis: str,
    parity: bool,
    h5_problem_manager: H5ProblemManager,
) -> list[H5ProblemDescriptor]:
    descriptor_values: list[tuple[float, H5ProblemDescriptor]] = []
    for descriptor in descriptors:
        hyperparams = h5_problem_manager.get_problem_hyperparams(
            problem_name=descriptor.problem_name,
            index=descriptor.problem_index,
        )
        hp_axis = _decode_ot_vortex_axis(hyperparams["vortex_axis"])
        hp_parity = bool(hyperparams["parity"])
        if hp_axis != vortex_axis or hp_parity != parity:
            continue
        descriptor_values.append((float(hyperparams["epsilon_p"]), descriptor))

    remaining = descriptor_values.copy()
    ordered: list[H5ProblemDescriptor] = []
    for target_value in parameter_values:
        if not remaining:
            raise ValueError(
                "Not enough validation descriptors for ot_vortex "
                f"(axis={vortex_axis}, parity={parity})."
            )
        diffs = [abs(scan_value - target_value) for scan_value, _ in remaining]
        best_idx = int(np.argmin(np.asarray(diffs, dtype=float)))
        matched_value, matched_descriptor = remaining.pop(best_idx)
        if not np.isclose(matched_value, target_value, atol=1e-6, rtol=1e-5):
            raise ValueError(
                "Validation descriptor parameter mismatch for ot_vortex "
                f"(axis={vortex_axis}, parity={parity}): "
                f"expected {target_value}, got {matched_value}"
            )
        ordered.append(matched_descriptor)

    if remaining:
        logger.warning(
            "Validation dataset for ot_vortex (axis=%s, parity=%s) has %d extra "
            "descriptors not used by this scan.",
            vortex_axis,
            parity,
            len(remaining),
        )
    return ordered


def _load_ot_vortex_validation_scan_inputs(
    *,
    training_config,
    validation_data_path: Path,
    parameter_values: list[float],
) -> dict[tuple[str, bool], tuple[H5ProblemManager, list[H5ProblemDescriptor]]]:
    h5_path = validation_data_path / "validation_dataset_ot_vortex_epsilon.h5"
    if not h5_path.exists():
        logger.warning("Validation dataset not found: %s", h5_path)
        return {}

    h5_problem_manager = H5ProblemManager(
        h5_file_paths={"ot_vortex": str(h5_path)},
        training_config=training_config,
    )
    descriptors = [
        d
        for d in h5_problem_manager.get_problem_descriptors()
        if d.problem_name == "ot_vortex"
    ]
    if not descriptors:
        logger.warning(
            "No ot_vortex descriptors found in validation dataset: %s", h5_path
        )
        return {}

    scan_inputs: dict[
        tuple[str, bool], tuple[H5ProblemManager, list[H5ProblemDescriptor]]
    ] = {}
    for vortex_axis in ["x", "y", "z"]:
        for parity in [False, True]:
            ordered_descriptors = _order_ot_vortex_scan_descriptors(
                descriptors,
                parameter_values=parameter_values,
                vortex_axis=vortex_axis,
                parity=parity,
                h5_problem_manager=h5_problem_manager,
            )
            scan_inputs[(vortex_axis, parity)] = (
                h5_problem_manager,
                ordered_descriptors,
            )

    return scan_inputs


def _build_h5_eval_bundles(
    base_lr_bundle: SimulationBundle,
    *,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
) -> tuple[SimulationBundle, SimulationBundle]:
    """Build deterministic uncorrected/corrected bundles for H5 evaluation."""
    lr_bundle_corrected = base_lr_bundle.copy()
    lr_bundle_corrected.override_solver_in_the_loop(
        corrector_config=corrector_config,
        corrector_params=corrector_params,
    )
    lr_bundle_corrected.config = lr_bundle_corrected.config._replace(
        cnn_mhd_corrector_config=lr_bundle_corrected.config.cnn_mhd_corrector_config._replace(
            cnn_mhd_corrector=True,
            network_static=corrector_config.network_static,
            correct_from_beggining=corrector_config.correct_from_beggining,
            start_correction_time=corrector_config.start_correction_time,
        )
    )
    lr_bundle_corrected.params = lr_bundle_corrected.params._replace(
        cnn_mhd_corrector_params=lr_bundle_corrected.params.cnn_mhd_corrector_params._replace(
            network_params=corrector_params.network_params
        )
    )

    lr_bundle_uncorrected = base_lr_bundle.copy()
    lr_bundle_uncorrected.config = lr_bundle_uncorrected.config._replace(
        cnn_mhd_corrector_config=lr_bundle_uncorrected.config.cnn_mhd_corrector_config._replace(
            cnn_mhd_corrector=False,
            network_static=None,
            correct_from_beggining=True,
            start_correction_time=0.0,
        )
    )
    lr_bundle_uncorrected.params = lr_bundle_uncorrected.params._replace(
        cnn_mhd_corrector_params=lr_bundle_uncorrected.params.cnn_mhd_corrector_params._replace(
            network_params=None
        )
    )
    return lr_bundle_uncorrected, lr_bundle_corrected


def _assert_h5_eval_branch_separation(
    uncorrected_bundle: SimulationBundle, corrected_bundle: SimulationBundle
) -> None:
    """Regression guard to prevent corrected/uncorrected config collapse."""
    if (
        not corrected_bundle.config.cnn_mhd_corrector_config.cnn_mhd_corrector
        or corrected_bundle.params.cnn_mhd_corrector_params.network_params is None
    ):
        raise ValueError("Corrected H5 evaluation bundle is missing corrector setup.")
    if uncorrected_bundle.config.cnn_mhd_corrector_config.cnn_mhd_corrector:
        raise ValueError(
            "Uncorrected H5 evaluation bundle unexpectedly enables corrector."
        )


def _evaluate_h5_descriptors(
    descriptors: list[H5ProblemDescriptor],
    *,
    h5_problem_manager: H5ProblemManager,
    training_config: TrainingConfig,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
) -> tuple[list[float], list[float]]:
    final_loss_initial: list[float] = []
    final_loss_corrected: list[float] = []

    for descriptor in tqdm.tqdm(descriptors):
        pair = h5_problem_manager.get_training_pairs_for_descriptors([descriptor])[0]
        target_state = pair.target_state

        loss_fn_kwargs, loss_fn_factory = loss_setup(
            training_config=training_config,
            target_states=target_state,
        )
        loss_fn = partial(loss_fn_factory, **loss_fn_kwargs)

        lr_bundle_uncorrected, lr_bundle_corrected = _build_h5_eval_bundles(
            pair.lr_bundle,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
        )
        _assert_h5_eval_branch_separation(
            uncorrected_bundle=lr_bundle_uncorrected,
            corrected_bundle=lr_bundle_corrected,
        )

        try:
            lr_sol_state = time_integration(**lr_bundle_corrected.unpack_integrate())
            if hasattr(lr_sol_state, "states"):
                lr_sol_final = lr_sol_state.states[-1]
            else:
                lr_sol_final = lr_sol_state
            final_loss_corrected.append(float(loss_fn(lr_sol_final, target_state)))
        except Exception:
            final_loss_corrected.append(float("nan"))

        try:
            lr_state = time_integration(**lr_bundle_uncorrected.unpack_integrate())
            if hasattr(lr_state, "states"):
                lr_final = lr_state.states[-1]
            else:
                lr_final = lr_state
            final_loss_initial.append(float(loss_fn(lr_final, target_state)))
        except Exception:
            final_loss_initial.append(float("nan"))

    return final_loss_initial, final_loss_corrected


def _plot_h5_envelope_grid(
    *,
    records: list[dict[str, float]],
    fixed_key: str,
    sweep_key: str,
    fixed_values: np.ndarray,
    output_path: Path,
    title_prefix: str,
    log_y_scale: bool = False,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for i, fixed_val in enumerate(fixed_values):
        ax = axes_flat[i]
        panel_records = [
            r for r in records if np.isclose(r[fixed_key], fixed_val, atol=1e-6)
        ]
        if not panel_records:
            ax.set_title(f"{fixed_key}={fixed_val:.3f} (no data)")
            continue

        sweep_points = sorted({float(r[sweep_key]) for r in panel_records})

        unc_min, unc_max = [], []
        cor_min, cor_max = [], []
        for sweep in sweep_points:
            group = [
                r for r in panel_records if np.isclose(r[sweep_key], sweep, atol=1e-6)
            ]
            unc_vals = np.asarray([r["uncorrected"] for r in group], dtype=float)
            cor_vals = np.asarray([r["corrected"] for r in group], dtype=float)
            unc_vals = unc_vals[~np.isnan(unc_vals)]
            cor_vals = cor_vals[~np.isnan(cor_vals)]

            unc_min.append(float(np.min(unc_vals)) if unc_vals.size else np.nan)
            unc_max.append(float(np.max(unc_vals)) if unc_vals.size else np.nan)
            cor_min.append(float(np.min(cor_vals)) if cor_vals.size else np.nan)
            cor_max.append(float(np.max(cor_vals)) if cor_vals.size else np.nan)

        x = np.asarray(sweep_points, dtype=float)
        unc_min_a = np.asarray(unc_min, dtype=float)
        unc_max_a = np.asarray(unc_max, dtype=float)
        cor_min_a = np.asarray(cor_min, dtype=float)
        cor_max_a = np.asarray(cor_max, dtype=float)

        ax.plot(x, unc_min_a, color="tab:orange", label="Uncorrected")
        ax.plot(x, unc_max_a, color="tab:orange", alpha=0.9, label="_nolegend_")
        ax.fill_between(x, unc_min_a, unc_max_a, color="tab:orange", alpha=0.2)

        ax.plot(
            x,
            cor_min_a,
            color="tab:blue",
            label="Corrected",
        )
        ax.plot(
            x,
            cor_max_a,
            color="tab:blue",
            alpha=0.9,
            label="_nolegend_",
        )
        ax.fill_between(x, cor_min_a, cor_max_a, color="tab:blue", alpha=0.2)

        ax.set_title(f"{fixed_key}={fixed_val:.3f}")
        ax.set_xlabel(sweep_key)
        ax.set_ylabel("loss")
        if log_y_scale:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=9,
        )
    fig.suptitle(title_prefix)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved H5 envelope plot to %s", output_path)


def _collect_radial_h5_records(
    *,
    h5_problem_manager: H5ProblemManager,
    problem_name: str,
    training_config: TrainingConfig,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    b0_min: float | None = None,
    b0_max: float | None = None,
) -> list[dict[str, float]]:
    descriptors_all = [
        d
        for d in h5_problem_manager.get_problem_descriptors()
        if d.problem_name == problem_name
    ]
    if not descriptors_all:
        return []

    # Pre-filter descriptors by B0 range to skip expensive evaluations
    descriptors_filtered = descriptors_all
    if b0_min is not None or b0_max is not None:
        b0_min_val = b0_min if b0_min is not None else float("-inf")
        b0_max_val = b0_max if b0_max is not None else float("inf")
        descriptors_filtered = []
        for desc in descriptors_all:
            hp = dict(
                h5_problem_manager.get_problem_hyperparams(
                    desc.problem_name, desc.problem_index
                )
            )
            if "B0" in hp:
                b0_val = float(hp["B0"])
                if b0_min_val <= b0_val <= b0_max_val:
                    descriptors_filtered.append(desc)
        if not descriptors_filtered:
            logger.warning(
                f"No {problem_name} descriptors found in B0 range [{b0_min_val}, {b0_max_val}]"
            )
            return []
        logger.info(
            f"Pre-filtered {len(descriptors_all)} descriptors to {len(descriptors_filtered)} in B0 range [{b0_min_val}, {b0_max_val}]"
        )

    losses_unc, losses_cor = _evaluate_h5_descriptors(
        descriptors_filtered,
        h5_problem_manager=h5_problem_manager,
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
    )

    records: list[dict[str, float]] = []
    for desc, unc, cor in zip(
        descriptors_filtered, losses_unc, losses_cor, strict=True
    ):
        hp = dict(
            h5_problem_manager.get_problem_hyperparams(
                desc.problem_name, desc.problem_index
            )
        )
        if "B0" not in hp or "B_direction" not in hp:
            continue
        theta, phi = _extract_theta_phi_from_b_direction(hp["B_direction"])
        records.append(
            {
                "B0": float(hp["B0"]),
                "theta": theta,
                "phi": phi,
                "uncorrected": float(unc),
                "corrected": float(cor),
            }
        )
    return records


def plot_mhd_blast_parameter_scans_h5(
    training_config: TrainingConfig,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    model_dir: Path,
    h5_file_paths: dict[str, str],
    b0_min: float | None = None,
    b0_max: float | None = None,
):
    h5_problem_manager = H5ProblemManager(
        h5_file_paths=h5_file_paths,
        training_config=training_config,
    )
    records = _collect_radial_h5_records(
        h5_problem_manager=h5_problem_manager,
        problem_name="mhd_blast",
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        b0_min=b0_min,
        b0_max=b0_max,
    )
    if not records:
        logger.warning("No H5 records found for mhd_blast parameter_scans_h5")
        return

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    phi_panels = np.array(
        [
            float(jnp.pi / 4),
            3 * float(jnp.pi / 4),
            5 * float(jnp.pi / 4),
            7 * float(jnp.pi / 4),
        ]
    )
    theta_panels = np.linspace(0.2, np.pi - 0.2, 4)
    available_phi = np.asarray(sorted({r["phi"] for r in records}), dtype=float)
    available_theta = np.asarray(sorted({r["theta"] for r in records}), dtype=float)

    phi_fixed = np.asarray(
        [available_phi[np.argmin(np.abs(available_phi - p))] for p in phi_panels],
        dtype=float,
    )
    theta_fixed = np.asarray(
        [available_theta[np.argmin(np.abs(available_theta - t))] for t in theta_panels],
        dtype=float,
    )

    # Generate B0 range suffix for filenames
    b0_suffix = ""
    b0_range_str = ""
    if b0_min is not None or b0_max is not None:
        b0_min_str = f"{b0_min:.2f}" if b0_min is not None else "inf"
        b0_max_str = f"{b0_max:.2f}" if b0_max is not None else "inf"
        b0_suffix = f"_B0_{b0_min_str}_{b0_max_str}"
        b0_range_str = f" B0∈[{b0_min_str}, {b0_max_str}]"

    _plot_h5_envelope_grid(
        records=records,
        fixed_key="phi",
        sweep_key="B0",
        fixed_values=phi_fixed,
        output_path=plots_dir / f"mhd_parameter_scan_h5_phi_panels{b0_suffix}.png",
        title_prefix=f"MHD blast H5: fixed phi, theta envelope{b0_range_str}",
        log_y_scale=True,
    )
    _plot_h5_envelope_grid(
        records=records,
        fixed_key="theta",
        sweep_key="B0",
        fixed_values=theta_fixed,
        output_path=plots_dir / f"mhd_parameter_scan_h5_theta_panels{b0_suffix}.png",
        title_prefix=f"MHD blast H5: fixed theta, phi envelope{b0_range_str}",
        log_y_scale=True,
    )
    _plot_h5_envelope_grid(
        records=records,
        fixed_key="theta",
        sweep_key="phi",
        fixed_values=theta_fixed,
        output_path=plots_dir
        / f"mhd_parameter_scan_h5_theta_panels_phi_scan{b0_suffix}.png",
        title_prefix=f"MHD blast H5: fixed theta, B0 envelope (phi scan){b0_range_str}",
        log_y_scale=True,
    )
    _plot_h5_envelope_grid(
        records=records,
        fixed_key="phi",
        sweep_key="theta",
        fixed_values=phi_fixed,
        output_path=plots_dir
        / f"mhd_parameter_scan_h5_phi_panels_theta_scan{b0_suffix}.png",
        title_prefix=f"MHD blast H5: fixed phi, B0 envelope (theta scan){b0_range_str}",
        log_y_scale=True,
    )


def plot_turbulence_parameter_scans_h5(
    training_config: TrainingConfig,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    model_dir: Path,
    h5_file_paths: dict[str, str],
    b0_min: float | None = None,
    b0_max: float | None = None,
):
    h5_problem_manager = H5ProblemManager(
        h5_file_paths=h5_file_paths,
        training_config=training_config,
    )
    records = _collect_radial_h5_records(
        h5_problem_manager=h5_problem_manager,
        problem_name="turbulence",
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        b0_min=b0_min,
        b0_max=b0_max,
    )
    if not records:
        logger.warning("No H5 records found for turbulence parameter_scans_h5")
        return

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    phi_panels = np.linspace(0.2, 2.0 * np.pi - 0.2, 4)
    theta_panels = np.linspace(0.2, np.pi - 0.2, 4)
    available_phi = np.asarray(sorted({r["phi"] for r in records}), dtype=float)
    available_theta = np.asarray(sorted({r["theta"] for r in records}), dtype=float)

    phi_fixed = np.asarray(
        [available_phi[np.argmin(np.abs(available_phi - p))] for p in phi_panels],
        dtype=float,
    )
    theta_fixed = np.asarray(
        [available_theta[np.argmin(np.abs(available_theta - t))] for t in theta_panels],
        dtype=float,
    )

    # Generate B0 range suffix for filenames
    b0_suffix = ""
    b0_range_str = ""
    if b0_min is not None or b0_max is not None:
        b0_min_str = f"{b0_min:.2f}" if b0_min is not None else "inf"
        b0_max_str = f"{b0_max:.2f}" if b0_max is not None else "inf"
        b0_suffix = f"_B0_{b0_min_str}_{b0_max_str}"
        b0_range_str = f" B0∈[{b0_min_str}, {b0_max_str}]"

    _plot_h5_envelope_grid(
        records=records,
        fixed_key="phi",
        sweep_key="B0",
        fixed_values=phi_fixed,
        output_path=plots_dir
        / f"turbulence_parameter_scan_h5_phi_panels{b0_suffix}.png",
        title_prefix=f"Turbulence H5: fixed phi, theta envelope{b0_range_str}",
        log_y_scale=True,
    )
    _plot_h5_envelope_grid(
        records=records,
        fixed_key="theta",
        sweep_key="B0",
        fixed_values=theta_fixed,
        output_path=plots_dir
        / f"turbulence_parameter_scan_h5_theta_panels{b0_suffix}.png",
        title_prefix=f"Turbulence H5: fixed theta, phi envelope{b0_range_str}",
        log_y_scale=True,
    )


def plot_ot_vortex_parameter_scans_h5(
    training_config: TrainingConfig,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    model_dir: Path,
    h5_file_paths: dict[str, str],
):
    h5_problem_manager = H5ProblemManager(
        h5_file_paths=h5_file_paths,
        training_config=training_config,
    )
    descriptors_all = [
        d
        for d in h5_problem_manager.get_problem_descriptors()
        if d.problem_name == "ot_vortex"
    ]
    if not descriptors_all:
        logger.warning("No H5 records found for ot_vortex parameter_scans_h5")
        return

    losses_unc, losses_cor = _evaluate_h5_descriptors(
        descriptors_all,
        h5_problem_manager=h5_problem_manager,
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
    )

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vortex_axis in zip(axes, ["x", "y", "z"], strict=True):
        records_pos: list[dict[str, float]] = []
        records_neg: list[dict[str, float]] = []
        for desc, unc, cor in zip(descriptors_all, losses_unc, losses_cor, strict=True):
            hp = dict(
                h5_problem_manager.get_problem_hyperparams(
                    desc.problem_name, desc.problem_index
                )
            )
            hp_axis = _decode_ot_vortex_axis(hp.get("vortex_axis", vortex_axis))
            if hp_axis != vortex_axis:
                continue
            record = {
                "epsilon_p": float(hp["epsilon_p"]),
                "uncorrected": float(unc),
                "corrected": float(cor),
            }
            if bool(hp["parity"]):
                records_neg.append(record)
            else:
                records_pos.append(record)

        def _plot_group(group_records: list[dict[str, float]]):
            if not group_records:
                return
            x = np.asarray([r["epsilon_p"] for r in group_records], dtype=float)
            unc = np.asarray([r["uncorrected"] for r in group_records], dtype=float)
            cor = np.asarray([r["corrected"] for r in group_records], dtype=float)
            order = np.argsort(x)
            x, unc, cor = x[order], unc[order], cor[order]
            ax.plot(
                x,
                unc,
                color="tab:orange",
                label="Uncorrected",
            )
            ax.plot(
                x,
                cor,
                color="tab:blue",
                label="Corrected",
            )

        _plot_group(records_pos)
        _plot_group(records_neg)
        ax.set_xlabel("epsilon_p")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.set_title(f"OT Vortex H5 (axis={vortex_axis})")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = {}
        for h, l in zip(handles, labels, strict=True):
            if l not in by_label:
                by_label[l] = h
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            fontsize=9,
        )

    out_path = plots_dir / "ot_vortex_parameter_scan_h5.png"
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved OT vortex H5 parameter scan to %s", out_path)


def plot_parameter_range_comparison(
    parameter_key: str,
    parameter_values: list[float],
    base_problem_descriptor: ProblemDescriptor,
    training_config,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    snapshot_timepoints: Array,
    trained_on_t: list[float],
    output_path: Optional[Path] = None,
    parameter_to_problem_params: Optional[Callable[[float], dict]] = None,
    ax: Optional[plt.Axes] = None,
    linestyle: str = "-",
    label_suffix: str = "",
    uncorrected_color: Optional[str] = None,
    corrected_color: Optional[str] = None,
    h5_problem_manager: Optional[H5ProblemManager] = None,
    h5_descriptors: Optional[list[H5ProblemDescriptor]] = None,
    log_y_scale: bool = False,
):
    if parameter_to_problem_params is None:
        logger.info("using default parameter to problem params builder")

        def parameter_to_problem_params(parameter_value):
            return {parameter_key: parameter_value}

    # NOTE: Move this to the outisde of this function
    for t in trained_on_t:
        if t not in snapshot_timepoints:
            snapshot_timepoints = jnp.sort(
                jnp.concatenate([snapshot_timepoints, jnp.array([t])])
            )

    # Get the index of the trained times
    trained_on_t_index = []
    for t in trained_on_t:
        trained_on_t_index.append(int(jnp.argmax(snapshot_timepoints == t)))
    if (h5_problem_manager is None) != (h5_descriptors is None):
        raise ValueError(
            "h5_problem_manager and h5_descriptors must be provided together."
        )

    if h5_problem_manager is not None and h5_descriptors is not None:
        final_loss_initial, final_loss_corrected = _evaluate_h5_descriptors(
            descriptors=h5_descriptors,
            h5_problem_manager=h5_problem_manager,
            training_config=training_config,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
        )
        if len(final_loss_initial) != len(parameter_values):
            raise ValueError(
                "Mismatch between parameter values and H5 descriptors for "
                f"{parameter_key}: {len(parameter_values)} vs {len(final_loss_initial)}"
            )
        aligned_loss_initial = final_loss_initial
        aligned_loss_corrected = final_loss_corrected
    else:
        # Create the problem descriptors list going trough the param range
        problem_descriptors = []
        for parameter_value in parameter_values:
            params = dict(base_problem_descriptor.params)
            params.update(parameter_to_problem_params(parameter_value))
            descriptor = base_problem_descriptor._replace(params=params, nickname="")
            problem_descriptors.append(descriptor)

        problem_manager = ProblemManager(
            problem_descriptors=problem_descriptors,
            training_config=training_config,
        )
        evaluation_data = problem_manager.get_evaluation_data(
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            snapshot_timepoints=snapshot_timepoints,
        )

        final_loss_initial, final_loss_corrected, _, _ = _compute_final_losses(
            evaluation_data=evaluation_data,
            training_config=training_config,
            snapshot_timepoints_idx=trained_on_t_index,
        )

        print(final_loss_corrected, final_loss_initial)

        # Map successful results back to the full parameter range so that HR-failed
        # cases are represented as NaN (and interpolated) rather than silently
        # dropped from the x-axis.
        #
        # NOTE: This whole part is a fix to some high res data crashing during the sims
        # the proper thing would be for the high res to not crash but most times life
        # is not as we'd like
        # NOTE: The main idea is to interpolate the loss values and mark them with an x in the plots
        # NOTE: here we use the evaluation_data tuple as it only includes the successful computed sims
        # this only happens for the high res states, as for the other unsuccessful values the values are
        # already nans

        successful_losses: dict[str, tuple[float, float]] = {}

        for data, loss_i, loss_c in zip(
            evaluation_data, final_loss_initial, final_loss_corrected, strict=True
        ):
            sig = data.problem_descriptor.nickname
            successful_losses[sig] = (loss_i, loss_c)

        aligned_loss_initial = []
        aligned_loss_corrected = []
        for descriptor in problem_descriptors:
            sig = descriptor.nickname
            if sig in successful_losses:
                aligned_loss_initial.append(successful_losses[sig][0])
                aligned_loss_corrected.append(successful_losses[sig][1])
            else:
                aligned_loss_initial.append(float("nan"))
                aligned_loss_corrected.append(float("nan"))

    parameter_values_array = np.asarray(parameter_values, dtype=float)
    final_loss_initial_plot, initial_nan_mask = _interpolate_nan_for_plot(
        parameter_values_array, aligned_loss_initial
    )
    final_loss_corrected_plot, corrected_nan_mask = _interpolate_nan_for_plot(
        parameter_values_array, aligned_loss_corrected
    )

    _own_fig = ax is None
    if _own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    else:
        fig = ax.get_figure()
    uncorrected_line = ax.plot(
        parameter_values_array,
        final_loss_initial_plot,
        label=f"Uncorrected{label_suffix}",
        linestyle=linestyle,
        **({} if uncorrected_color is None else {"color": uncorrected_color}),
    )[0]
    corrected_line = ax.plot(
        parameter_values_array,
        final_loss_corrected_plot,
        label=f"Corrected{label_suffix}",
        linestyle=linestyle,
        **({} if corrected_color is None else {"color": corrected_color}),
    )[0]

    if np.any(initial_nan_mask):
        ax.scatter(
            parameter_values_array[initial_nan_mask],
            final_loss_initial_plot[initial_nan_mask],
            marker="x",
            s=50,
            color=uncorrected_line.get_color(),
            label="Uncorrected (interpolated failure)",
        )
    if np.any(corrected_nan_mask):
        ax.scatter(
            parameter_values_array[corrected_nan_mask],
            final_loss_corrected_plot[corrected_nan_mask],
            marker="x",
            s=50,
            color=corrected_line.get_color(),
            label="Corrected (interpolated failure)",
        )

    ax.legend()
    ax.set_xlabel(parameter_key)
    ax.set_ylabel("loss")
    if log_y_scale:
        ax.set_yscale("log")
    if output_path is not None and _own_fig:
        plt.savefig(output_path, dpi=400)
    return fig, ax


def plot_mhd_blast_parameter_scan(
    training_config,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    trained_on_t: list[float],
    model_dir: Path,
    phi_fixed: float,
):
    times_eval_blast = jnp.linspace(0.0, 0.2, 2, endpoint=True)
    base_problem_descriptor = ProblemDescriptor(name="mhd_blast")

    num_tests = 30
    B_0s = jnp.linspace(start=0.5, stop=20.5, num=num_tests).tolist()
    print(B_0s)

    r = 1.0
    thetas = jnp.linspace(start=0.2, stop=jnp.pi - 0.2, num=num_tests).tolist()

    theta_fixed = float(jnp.pi / 2)
    phis = jnp.linspace(start=0.2, stop=2 * jnp.pi - 0.2, num=num_tests).tolist()

    validation_scan_inputs = _load_mhd_blast_validation_scan_inputs(
        training_config=training_config,
        validation_data_path=Path("/export/data/jalegria/solver_in_the_loop"),
        parameter_values_by_scan={
            "b0": B_0s,
            "theta": thetas,
            "phi": phis,
        },
    )
    b0_scan_input = validation_scan_inputs.get("b0")
    theta_scan_input = validation_scan_inputs.get("theta")
    phi_scan_input = validation_scan_inputs.get("phi")
    default_phi_scan = float(jnp.pi / 4.0)
    use_theta_h5 = bool(
        theta_scan_input is not None
        and np.isclose(float(phi_fixed), default_phi_scan, atol=1e-6, rtol=0.0)
    )
    if theta_scan_input is not None and not use_theta_h5:
        logger.info(
            "Theta scan uses on-the-fly simulations because phi_fixed=%.6f differs from "
            "default dataset phi=%.6f.",
            float(phi_fixed),
            default_phi_scan,
        )

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_combined, axes = plt.subplots(1, 3, figsize=(18, 6))

    plot_parameter_range_comparison(
        base_problem_descriptor=base_problem_descriptor,
        snapshot_timepoints=times_eval_blast,
        parameter_key="B0",
        parameter_values=B_0s,
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        trained_on_t=trained_on_t,
        ax=axes[0],
        uncorrected_color="tab:orange",
        corrected_color="tab:blue",
        h5_problem_manager=b0_scan_input[0] if b0_scan_input else None,
        h5_descriptors=b0_scan_input[1] if b0_scan_input else None,
    )

    plot_parameter_range_comparison(
        base_problem_descriptor=base_problem_descriptor,
        snapshot_timepoints=times_eval_blast,
        parameter_key="theta",
        parameter_values=thetas,
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        trained_on_t=trained_on_t,
        parameter_to_problem_params=lambda theta: {
            "B_direction": jnp.array(
                [
                    r * jnp.sin(theta) * jnp.cos(phi_fixed),
                    r * jnp.sin(theta) * jnp.sin(phi_fixed),
                    r * jnp.cos(theta),
                ]
            )
        },
        ax=axes[1],
        uncorrected_color="tab:orange",
        corrected_color="tab:blue",
        h5_problem_manager=theta_scan_input[0] if use_theta_h5 else None,
        h5_descriptors=theta_scan_input[1] if use_theta_h5 else None,
        log_y_scale=False,
    )

    plot_parameter_range_comparison(
        base_problem_descriptor=base_problem_descriptor,
        snapshot_timepoints=times_eval_blast,
        parameter_key="phi",
        parameter_values=phis,
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        trained_on_t=trained_on_t,
        parameter_to_problem_params=lambda phi: {
            "B_direction": jnp.array(
                [
                    r * jnp.sin(theta_fixed) * jnp.cos(phi),
                    r * jnp.sin(theta_fixed) * jnp.sin(phi),
                    0.0,
                ]
            )
        },
        ax=axes[2],
        uncorrected_color="tab:orange",
        corrected_color="tab:blue",
        h5_problem_manager=phi_scan_input[0] if phi_scan_input else None,
        h5_descriptors=phi_scan_input[1] if phi_scan_input else None,
        log_y_scale=True,
    )

    out_path = plots_dir / f"mhd_parameter_scan_phi_{phi_fixed:.6f}.png"
    fig_combined.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig_combined)
    logger.info(f"Saved MHD parameter scan to {out_path}")


def plot_ot_vortex_parameter_scan(
    training_config,
    corrector_config: CNNMHDconfig,
    corrector_params: CNNMHDParams,
    trained_on_t: list[float],
    model_dir: Path,
):
    times_eval_vortex = jnp.linspace(0.0, jnp.pi, 2, endpoint=True)
    base_problem_descriptor = ProblemDescriptor(name="ot_vortex")
    epsilon_p_values = jnp.linspace(0.1, 2.1, 20).tolist()
    validation_scan_inputs = _load_ot_vortex_validation_scan_inputs(
        training_config=training_config,
        validation_data_path=Path("/export/data/jalegria/solver_in_the_loop"),
        parameter_values=epsilon_p_values,
    )

    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_combined, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, vortex_axis in zip(axes, ["x", "y", "z"]):
        positive_scan_input = validation_scan_inputs.get((vortex_axis, False))
        negative_scan_input = validation_scan_inputs.get((vortex_axis, True))
        # positive parity: solid lines
        plot_parameter_range_comparison(
            base_problem_descriptor=base_problem_descriptor,
            snapshot_timepoints=times_eval_vortex,
            parameter_key="epsilon_p",
            parameter_values=epsilon_p_values,
            training_config=training_config,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            trained_on_t=trained_on_t,
            parameter_to_problem_params=lambda eps, _axis=vortex_axis: {
                "epsilon_p": eps,
                "vortex_axis": _axis,
                "parity": False,
            },
            ax=ax,
            linestyle="-",
            label_suffix=" (+parity)",
            uncorrected_color="tab:orange",
            corrected_color="tab:blue",
            h5_problem_manager=positive_scan_input[0] if positive_scan_input else None,
            h5_descriptors=positive_scan_input[1] if positive_scan_input else None,
        )
        # negative parity: dashed lines
        plot_parameter_range_comparison(
            base_problem_descriptor=base_problem_descriptor,
            snapshot_timepoints=times_eval_vortex,
            parameter_key="epsilon_p",
            parameter_values=epsilon_p_values,
            training_config=training_config,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            trained_on_t=trained_on_t,
            parameter_to_problem_params=lambda eps, _axis=vortex_axis: {
                "epsilon_p": eps,
                "vortex_axis": _axis,
                "parity": True,
            },
            ax=ax,
            linestyle="--",
            label_suffix=" (-parity)",
            uncorrected_color="tab:orange",
            corrected_color="tab:blue",
            h5_problem_manager=negative_scan_input[0] if negative_scan_input else None,
            h5_descriptors=negative_scan_input[1] if negative_scan_input else None,
        )
        ax.set_title(f"OT Vortex (axis={vortex_axis}): ε_p scan", fontsize=10)

    out_path = plots_dir / "ot_vortex_parameter_scan.png"
    fig_combined.savefig(out_path, dpi=400, bbox_inches="tight")
    plt.close(fig_combined)
    logger.info(f"Saved OT VORTEX parameter scan to {out_path}")


if __name__ == "__main__":
    import argparse

    logging.basicConfig(format="->{message}", style="{", level=logging.INFO)

    # Parse CLI arguments using argparse
    parser = argparse.ArgumentParser(
        description="Plot training multiproblem results with optional B0 range filtering"
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        default="b0_7_13_best_params",
        help="Model name to process (default: b0_7_13_best_params)",
    )

    parser.add_argument(
        "--multiproblem_wo_dataset",
        action="store_true",
        default=False,
        help="If set to true the model folder is chaned to multiproblem_wo_dataset",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=MULTIPROBLEM_BASE_DIR,
        help="model parent folder",
    )

    parser.add_argument(
        "--b0_min",
        type=float,
        default=None,
        help="Minimum B0 value for filtering H5 parameter scans",
    )

    parser.add_argument(
        "--b0_max",
        type=float,
        default=None,
        help="Maximum B0 value for filtering H5 parameter scans",
    )

    parser.add_argument(
        "--phi_scan",
        type=float,
        default=float(jnp.pi / 4),
        help="value of phi for the parameter scan",
    )

    parser.add_argument("--not_run_h5_scan", action="store_true", default=False)

    args = parser.parse_args()
    model_name = args.model_name
    b0_min = args.b0_min
    b0_max = args.b0_max

    if args.multiproblem_wo_dataset and args.model_path == MULTIPROBLEM_BASE_DIR:
        MULTIPROBLEM_BASE_DIR = "arena/data/models/multiproblem_wo_dataset"
    else:
        MULTIPROBLEM_BASE_DIR = args.model_path

    # param_type option nan, best, normal
    param_type = "best"
    multigpu_trained = True
    split_loss_line_at_training_time = True

    logger.info(f"=== Processing model: {model_name} ===")
    if b0_min is not None or b0_max is not None:
        logger.info(f"B0 range filter: [{b0_min}, {b0_max}]")

    model_manager = ModelManager(base_dir=MULTIPROBLEM_BASE_DIR, model_name=model_name)
    model_manager.print_model_info()
    training_config = model_manager.load_training_config()

    reg_vars = _get_reg_vars()

    model_cls = FiLMCorrectorCNN if training_config.use_film_corrector else CorrectorCNN
    model = model_cls(
        in_channels=reg_vars.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
        normalize_input=bool(training_config.normalize_input),
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)

    neural_net_params = model_manager.load_model_params(
        like=neural_net_params, param_type=param_type
    )

    corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True, network_static=neural_net_static
    )
    corrector_params = CNNMHDParams(network_params=neural_net_params)

    model_dir = Path(MULTIPROBLEM_BASE_DIR) / model_name

    # --- 1. Losses plot ---
    losses_path = Path(MULTIPROBLEM_BASE_DIR) / model_name / "losses_per_problem.npz"
    if losses_path.exists():
        plot_losses_multigpu(
            losses_path=losses_path,
            output_path=model_dir / "plots" / "losses.png",
        )
        plot_losses_multiproblem(
            losses_path=losses_path,
            output_path=model_dir / "plots" / "losses_per_problem.png",
        )

    # --- 2. Simple parameter range scans (no H5) ---
    # Output: mhd_parameter_scan.png

    plot_mhd_blast_parameter_scan(
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        trained_on_t=[0.2],
        model_dir=model_dir,
        phi_fixed=args.phi_scan,
    )

    # Output: ot_vortex_parameter_scan.png
    plot_ot_vortex_parameter_scan(
        training_config=training_config,
        corrector_config=corrector_config,
        corrector_params=corrector_params,
        trained_on_t=[jnp.pi],
        model_dir=model_dir,
    )

    # --- 3. H5-backed parameter scans (last, slow) ---
    # Output: mhd_parameter_scan_h5_phi_panels.png, mhd_parameter_scan_h5_theta_panels.png
    #         turbulence_parameter_scan_h5_phi_panels.png, turbulence_parameter_scan_h5_theta_panels.png
    #         ot_vortex_parameter_scan_h5.png
    if not args.not_run_h5_scan:
        h5_file_paths = {
            "mhd_blast": "/export/data/jalegria/solver_in_the_loop/training_blast.h5",
            "turbulence": "/export/data/jalegria/solver_in_the_loop/training_turbulence.h5",
            "ot_vortex": "/export/data/jalegria/solver_in_the_loop/training_ot_vortex.h5",
        }

        plot_mhd_blast_parameter_scans_h5(
            training_config=training_config,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            model_dir=model_dir,
            h5_file_paths={"mhd_blast": h5_file_paths["mhd_blast"]},
            b0_min=b0_min,
            b0_max=b0_max,
        )
        plot_turbulence_parameter_scans_h5(
            training_config=training_config,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            model_dir=model_dir,
            h5_file_paths={"turbulence": h5_file_paths["turbulence"]},
            b0_min=b0_min,
            b0_max=b0_max,
        )
        plot_ot_vortex_parameter_scans_h5(
            training_config=training_config,
            corrector_config=corrector_config,
            corrector_params=corrector_params,
            model_dir=model_dir,
            h5_file_paths={"ot_vortex": h5_file_paths["ot_vortex"]},
        )
