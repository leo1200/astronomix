from autocvd import autocvd

autocvd(num_gpus=1)
"""
Multiproblem training summary plot.

Layout (N problems):
  - Rows 0..N-1  : one row per problem, 3 columns = [target | uncorrected | corrected] density slices
  - Row N        : per-problem loss curves + average (full width)
  - Row N+1      : L2 error vs snapshot time, one subplot per problem
"""
import os
import logging
from pathlib import Path
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
import equinox as eqx
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from jf1uids.time_stepping import time_integration
from jf1uids.data_classes.simulation_snapshot_data import SnapshotData
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
    CNNMHDParams,
)
from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_finite_element import (
    CorrectorCNN,
)

from arena.arena_tests.solver_in_the_loop.utils import downaverage
from arena.arena_tests.solver_in_the_loop.loss import loss_setup
from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    SimulationConfigTraining,
    model_loader,
)
from arena.arena_tests.solver_in_the_loop.timepoint_updater import (
    FRONT_TO_BACK,
    BACK_TO_FRONT,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.training_model import (
    MultiproblemTrainingConfig,
)
from arena.arena_tests.solver_in_the_loop.multiproblem.problem_manager import (
    PROBLEM_CATALOG,
    SimulationBundle,
    _build_hr_config_and_params,
)

logger = logging.getLogger(__name__)

MULTIPROBLEM_BASE_DIR = "arena/data/models/multiproblem"


def _build_eval_bundle(
    problem_name: str,
    sim_config_training: SimulationConfigTraining,
    times_eval: jnp.ndarray,
    cfl_target: float,
    limiter: int,
):
    """Build HR and LR bundles for evaluation at arbitrary snapshot times."""
    from jf1uids import (
        SimulationConfig,
        SimulationParams,
        finalize_config,
        get_helper_data,
        get_registered_variables,
    )
    from jf1uids.option_classes.simulation_config import (
        BACKWARDS,
        PERIODIC_BOUNDARY,
        BoundarySettings,
        BoundarySettings1D,
        FINITE_DIFFERENCE,
    )

    t_end = float(times_eval[-1])

    simulation_params = SimulationParams(
        C_cfl=cfl_target,
        t_end=t_end,
        snapshot_timepoints=times_eval,
    )

    simulation_config = SimulationConfig(
        num_cells=sim_config_training.num_cells_high_res,
        solver_mode=FINITE_DIFFERENCE,
        differentiation_mode=BACKWARDS,
        mhd=True,
        dimensionality=3,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        use_specific_snapshot_timepoints=True,
        return_snapshots=True,
        num_snapshots=len(times_eval),
        num_checkpoints=100,
        progress_bar=True,
        runtime_debugging=False,
        limiter=limiter,
    )

    ic_fn = PROBLEM_CATALOG[problem_name]
    state, config, params, helper, reg_vars = ic_fn(
        simulation_config, simulation_params
    )
    hr_bundle = SimulationBundle(state, config, params, helper, reg_vars)
    lr_bundle = hr_bundle.convert_to_lr(sim_config_training.downaverage_factor)
    return hr_bundle, lr_bundle


def plot_training_multiproblem(
    neural_net_params,
    neural_net_static,
    times_eval: jnp.ndarray,
    model_name: str,
    sim_config_training: SimulationConfigTraining,
    training_config: MultiproblemTrainingConfig,
):
    model_dir = Path(MULTIPROBLEM_BASE_DIR) / model_name
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    problems = training_config.problems

    # Compute trained-time markers
    if training_config.direction == FRONT_TO_BACK:
        timepoints_train = [
            sim_config_training.t_end - t
            for t in training_config.snapshot_timepoints_train
        ]
    else:
        timepoints_train = training_config.snapshot_timepoints_train

    # Make sure trained times are in times_eval
    for t in timepoints_train:
        if t not in times_eval:
            times_eval = jnp.sort(jnp.concatenate([times_eval, jnp.array([t])]))

    snapshot_timepoints_idx = []
    for t in timepoints_train:
        snapshot_timepoints_idx.append(int(jnp.argmax(times_eval == t)))

    epochs_cumulative = np.cumsum(training_config.epochs_per_time)

    cnn_mhd_corrector_config = CNNMHDconfig(
        cnn_mhd_corrector=True,
        network_static=neural_net_static,
        start_correction_time=sim_config_training.start_correction_time,
        correct_from_beggining=sim_config_training.correct_from_beggining,
    )
    cnn_mhd_corrector_params = CNNMHDParams(network_params=neural_net_params)

    num_problems = len(problems)

    # Collect per-problem data
    problem_data = {}
    for problem_name in problems:
        logger.info(f"Evaluating problem '{problem_name}'...")
        hr_bundle, lr_bundle = _build_eval_bundle(
            problem_name,
            sim_config_training,
            times_eval,
            cfl_target=sim_config_training.c_cfl_target,
            limiter=sim_config_training.limiter,
        )

        # HR simulation → downsample → targets
        result_hr = time_integration(**hr_bundle.unpack_integrate())
        assert isinstance(result_hr, SnapshotData)
        targets_lr = downaverage(
            result_hr.states, sim_config_training.downaverage_factor
        )

        # LR uncorrected
        lr_params_uncorr = lr_bundle.params._replace(C_cfl=sim_config_training.c_cfl)
        states_uncorrected = time_integration(
            lr_bundle.initial_state,
            lr_bundle.config._replace(progress_bar=True),
            lr_params_uncorr,
            lr_bundle.helper,
            lr_bundle.reg_vars,
        ).states

        # LR corrected
        config_corr = lr_bundle.config._replace(
            cnn_mhd_corrector_config=cnn_mhd_corrector_config,
            progress_bar=True,
        )
        params_corr = lr_params_uncorr._replace(
            cnn_mhd_corrector_params=cnn_mhd_corrector_params
        )
        states_corrected = time_integration(
            lr_bundle.initial_state,
            config_corr,
            params_corr,
            lr_bundle.helper,
            lr_bundle.reg_vars,
        ).states

        # Loss function for this problem
        loss_fn_kwargs, loss_fn_factory = loss_setup(
            training_config=training_config,
            target_states=targets_lr[jnp.array(snapshot_timepoints_idx)],
        )
        loss_fn = partial(loss_fn_factory, **loss_fn_kwargs)
        v_loss = vmap(loss_fn, in_axes=(0, 0))

        l2_corrected = v_loss(states_corrected, targets_lr)
        l2_uncorrected = v_loss(states_uncorrected, targets_lr)

        # Pick the final trained snapshot for the image row
        final_idx = snapshot_timepoints_idx[-1]

        problem_data[problem_name] = {
            "targets_lr": targets_lr,
            "states_uncorrected": states_uncorrected,
            "states_corrected": states_corrected,
            "l2_corrected": l2_corrected,
            "l2_uncorrected": l2_uncorrected,
            "final_target": targets_lr[final_idx],
            "final_uncorrected": states_uncorrected[final_idx],
            "final_corrected": states_corrected[final_idx],
            "reg_vars": lr_bundle.reg_vars,
            "config": lr_bundle.config,
        }

    # Load losses
    losses_path = model_dir / "losses_per_problem.npz"
    if losses_path.exists():
        losses_data = np.load(losses_path)
        losses_avg = losses_data["avg"]
        losses_by_problem = {
            name: losses_data[name] for name in problems if name in losses_data
        }
    else:
        losses_file = model_dir / "losses.npz"
        losses_avg = np.load(losses_file)["losses"]
        losses_by_problem = {}

    # ========================
    # Figure layout
    # ========================
    # Row per problem (density slices): num_problems rows × 3 cols
    # 1 row: loss curves (full width)
    # 1 row: L2 error vs time, split into num_problems subplots
    total_rows = num_problems + 2
    height_ratios = [1] * num_problems + [1, 1]
    fig_height = 4 * total_rows
    fig_width = max(12, 4.5 * num_problems)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(
        total_rows,
        3 * num_problems,
        height_ratios=height_ratios,
        hspace=0.35,
        wspace=0.35,
    )

    # --------------------------------------------------
    # Rows 0..N-1: density slices per problem (3 cols each)
    # --------------------------------------------------
    for p_idx, problem_name in enumerate(problems):
        pd = problem_data[problem_name]
        reg_vars = pd["reg_vars"]
        config_lr = pd["config"]
        num_cells_lr = config_lr.num_cells

        states_trio = [
            pd["final_target"],
            pd["final_uncorrected"],
            pd["final_corrected"],
        ]
        col_titles = ["Target", "Uncorrected", "Corrected"]

        vmin = float(min(jnp.min(s[reg_vars.density_index]) for s in states_trio))
        vmax = float(max(jnp.max(s[reg_vars.density_index]) for s in states_trio))

        for c_idx, (state, col_title) in enumerate(zip(states_trio, col_titles)):
            # Each problem gets 3 columns out of the 3*N grid
            ax = fig.add_subplot(gs[p_idx, p_idx * 3 + c_idx])
            im = ax.imshow(
                state[reg_vars.density_index, :, :, num_cells_lr // 2],
                extent=(0, config_lr.box_size, 0, config_lr.box_size),
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            title = f"{problem_name}\n{col_title}" if c_idx == 0 else col_title
            ax.set_title(title, fontsize=9)
            ax.set_aspect("equal", adjustable="box")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

    # --------------------------------------------------
    # Loss evolution row (full width)
    # --------------------------------------------------
    loss_row = num_problems
    ax_loss = fig.add_subplot(gs[loss_row, :])
    ax_loss.plot(losses_avg, label="Average", color="black", linewidth=1.5)
    colors = plt.cm.tab10.colors
    for i, name in enumerate(problems):
        if name in losses_by_problem:
            ax_loss.plot(
                losses_by_problem[name],
                label=name,
                color=colors[i % len(colors)],
                alpha=0.7,
            )
    for t, epoch_end in zip(timepoints_train, epochs_cumulative):
        ax_loss.axvline(
            x=epoch_end, color="gray", linestyle=":", alpha=0.6, label=f"t={t:.3f}"
        )
    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss During Training (per problem + average)")
    ax_loss.legend(fontsize=7)
    ax_loss.set_yscale("log")

    # --------------------------------------------------
    # L2 error vs snapshot time — one subplot per problem
    # --------------------------------------------------
    error_row = num_problems + 1
    for p_idx, problem_name in enumerate(problems):
        pd = problem_data[problem_name]
        col_start = p_idx * 3
        col_end = col_start + 3
        ax_err = fig.add_subplot(gs[error_row, col_start:col_end])

        ax_err.plot(times_eval, pd["l2_corrected"], label="Corrected", color="tab:blue")
        ax_err.plot(
            times_eval,
            pd["l2_uncorrected"],
            label="Uncorrected",
            color="tab:orange",
            linestyle="--",
        )
        for t, epoch_end in zip(timepoints_train, epochs_cumulative):
            ax_err.axvline(x=t, color="gray", linestyle=":", alpha=0.6)
        ax_err.set_xlabel("Time")
        ax_err.set_ylabel("L2 Error")
        ax_err.set_yscale("log")
        ax_err.set_title(f"{problem_name}: Error vs Time", fontsize=9)
        ax_err.legend(fontsize=7)

    plt.savefig(plots_dir / "multiproblem_summary.png", dpi=300, bbox_inches="tight")
    logger.info(
        f"Saved multiproblem summary plot to {plots_dir / 'multiproblem_summary.png'}"
    )
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(format="->{message}", style="{", level=logging.INFO)

    model_name = "optuna_params_2"

    model_manager = ModelManager(base_dir=MULTIPROBLEM_BASE_DIR, model_name=model_name)
    model_manager.print_model_info()
    training_config_raw = model_manager.load_training_config()
    training_config = MultiproblemTrainingConfig(**training_config_raw.to_dict())
    sim_config_training = model_manager.load_simulation_config()

    reg_vars_dummy = None
    from jf1uids import SimulationConfig, get_registered_variables
    from jf1uids.option_classes.simulation_config import FINITE_DIFFERENCE

    reg_vars_dummy = get_registered_variables(
        SimulationConfig(dimensionality=3, mhd=True, solver_mode=FINITE_DIFFERENCE)
    )

    model = CorrectorCNN(
        in_channels=reg_vars_dummy.num_vars,
        hidden_channels=training_config.hidden_channels,
        hidden_layers=training_config.hidden_layers,
        key=jax.random.PRNGKey(100),
        scale=training_config.model_initialization_scale,
    )
    neural_net_params, neural_net_static = eqx.partition(model, eqx.is_array)
    neural_net_params = model_manager.load_model_params(like=neural_net_params)

    plot_training_multiproblem(
        neural_net_params=neural_net_params,
        neural_net_static=neural_net_static,
        times_eval=jnp.linspace(0.0, 0.3, 30),
        model_name=model_name,
        sim_config_training=sim_config_training,
        training_config=training_config,
    )
