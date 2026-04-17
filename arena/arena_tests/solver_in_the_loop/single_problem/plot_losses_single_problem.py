if __name__ == "__main__":
    from autocvd import autocvd

    autocvd(num_gpus=1)

import argparse
import logging
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from astronomix.data_classes.simulation_snapshot_data import SnapshotData
from astronomix.time_stepping import time_integration
from arena.arena_tests.solver_in_the_loop.loss import loss_setup
from arena.arena_tests.solver_in_the_loop.model_manager import (
    ModelManager,
    TrainingConfig,
)
from arena.arena_tests.solver_in_the_loop.single_problem.timepoint_updater import (
    FRONT_TO_BACK,
)
from arena.arena_tests.solver_in_the_loop.utils import (
    downaverage,
    get_initial_state_training,
)

logger = logging.getLogger(__name__)


def _trained_timepoints(training_config: TrainingConfig) -> list[float]:
    if training_config.direction == FRONT_TO_BACK:
        return [
            float(training_config.t_end - t)
            for t in training_config.snapshot_timepoints_train
        ]
    return [float(t) for t in training_config.snapshot_timepoints_train]


def _cumulative_epochs(epochs_per_time: list) -> np.ndarray:
    if len(epochs_per_time) == 0:
        return np.array([], dtype=float)
    return np.cumsum(np.asarray(epochs_per_time, dtype=float))


def _compute_initial_l2_error(
    training_config: TrainingConfig, *, old_version: bool = False
) -> float:
    timepoints_train = _trained_timepoints(training_config)
    if len(timepoints_train) == 0:
        raise ValueError("No training timepoints found in training_config.")

    times_eval = jnp.sort(jnp.array(timepoints_train, dtype=float))
    snapshot_timepoints_idx = [
        int(jnp.argmax(times_eval == t).item()) for t in timepoints_train
    ]

    simulation_bundle_high_res, simulation_bundle_low_res = get_initial_state_training(
        num_cells_high_res=training_config.num_cells_high_res,
        downaverage_factor=training_config.downaverage_factor,
        snapshot_timepoints_train=times_eval,
        c_cfl=training_config.c_cfl_target,
        limiter=training_config.limiter,
        old_version=old_version,
    )

    result_high_res = time_integration(*simulation_bundle_high_res)
    assert isinstance(result_high_res, SnapshotData)
    states_target_low_res = downaverage(
        result_high_res.states, downaverage_factor=training_config.downaverage_factor
    )

    (
        initial_state_low_res,
        config_low_res,
        params,
        registered_variables,
    ) = simulation_bundle_low_res
    states_low_res_uncorrected = time_integration(
        primitive_state=initial_state_low_res,
        config=config_low_res._replace(progress_bar=True),
        params=params._replace(C_cfl=training_config.c_cfl),
        registered_variables=registered_variables,
    ).states

    final_state_target_low_res = states_target_low_res[snapshot_timepoints_idx[-1]]
    final_state_low_res_uncorrected = states_low_res_uncorrected[
        snapshot_timepoints_idx[-1]
    ]

    loss_fn_kwargs, loss_fn_factory = loss_setup(
        training_config=training_config,
        target_states=states_target_low_res[jnp.array(snapshot_timepoints_idx)],
    )
    loss_fn = partial(loss_fn_factory, **loss_fn_kwargs)
    return float(loss_fn(final_state_low_res_uncorrected, final_state_target_low_res))


def plot_losses(
    model_name: str,
    *,
    base_dir: str = "arena/data/models/single_problem",
    output_path: str | None = None,
    old_version: bool = False,
) -> Path:
    model_manager = ModelManager(base_dir=base_dir, model_name=model_name)
    model_dir = Path(model_manager.base_dir) / model_name

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    training_config = model_manager.load_training_config()
    losses_data = np.load(model_dir / "losses.npz")
    if "losses" not in losses_data:
        raise KeyError(f"'losses' key not found in {model_dir / 'losses.npz'}")
    losses = np.asarray(losses_data["losses"], dtype=float)

    l2_error_initial = _compute_initial_l2_error(
        training_config=training_config, old_version=old_version
    )
    timepoints_train = _trained_timepoints(training_config)
    epochs_total = _cumulative_epochs(training_config.epochs_per_time)

    fig, ax_loss = plt.subplots(1, 1, figsize=(10, 5))
    ax_loss.plot(losses, label="Training Loss")
    ax_loss.axhline(
        y=l2_error_initial,
        color="r",
        linestyle="--",
        label="Initial L2 Error (uncorrected)",
    )
    for t, epochs in zip(timepoints_train, epochs_total, strict=False):
        ax_loss.axvline(
            x=epochs,
            color="gray",
            linestyle=":",
            label=f"Training time {t:.3f} / # {int(epochs)}",
        )

    ax_loss.set_xlabel("Training Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss During Training")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(fontsize=8)

    if output_path is None:
        output = model_dir / "plots" / "losses_plot.png"
    else:
        output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=400)
    plt.close(fig)

    logger.info("Saved losses plot to %s", output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Create single-problem training losses plot with training markers."
    )
    parser.add_argument(
        "model_name", type=str, help="Model name inside single_problem."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="arena/data/models/single_problem",
        help="Base directory where model folders are stored.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output file path. Defaults to <model>/plots/losses_plot.png",
    )
    parser.add_argument(
        "--old-version",
        action="store_true",
        help="Use old-version low-res setup (keeps low-res grid spacing behavior).",
    )
    args = parser.parse_args()

    logging.basicConfig(format="->{message}", style="{", level=logging.INFO)
    plot_losses(
        model_name=args.model_name,
        base_dir=args.base_dir,
        output_path=args.output,
        old_version=args.old_version,
    )


if __name__ == "__main__":
    main()
