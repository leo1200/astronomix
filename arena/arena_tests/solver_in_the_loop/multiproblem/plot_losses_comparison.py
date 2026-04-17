"""
Compare training loss evolution for multiproblem_test_1 and
multiproblem_2_problems_conFIG.

Layout (1 row of N+1 subplots):
  First N: per-problem loss per epoch with both runs overlaid
  Last:    average loss per epoch with both runs overlaid
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MULTIPROBLEM_BASE_DIR = Path("arena/data/models/multiproblem_wo_dataset")

RUNS = {
    "test_1": MULTIPROBLEM_BASE_DIR / "multiproblem_test_1" / "losses_per_problem.npz",
    "conFIG": MULTIPROBLEM_BASE_DIR / "multiproblem_2_problems_conFIG" / "losses_per_problem.npz",
}

# Map problem keys between the two runs (same problems, different naming)
PROBLEM_PAIRS = [
    {
        "label": "MHD Blast",
        "test_1": "mhd_blast",
        "conFIG": "mhd_blast_B0_10p0_r0_0p125_B_direction_1x1x0",
    },
    {
        "label": "OT Vortex",
        "test_1": "ot_vortex",
        "conFIG": "ot_vortex_vortex_axis_z_parity_False_epsilon_p_0p2",
    },
]


def plot_losses_comparison(output_path: str | Path = "losses_comparison.png"):
    data = {name: np.load(path) for name, path in RUNS.items()}
    n_problems = len(PROBLEM_PAIRS)
    n_cols = n_problems + 1

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))

    # Per-problem subplots
    for i, pair in enumerate(PROBLEM_PAIRS):
        ax = axes[i]
        for run_name, run_data in data.items():
            ax.plot(run_data[pair[run_name]], label=run_name, linewidth=1.2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(pair["label"])
        ax.set_yscale("log")
        ax.legend()

    # Average loss subplot
    ax_avg = axes[-1]
    for run_name, run_data in data.items():
        ax_avg.plot(run_data["avg"], label=run_name, linewidth=1.2)
    ax_avg.set_xlabel("Epoch")
    ax_avg.set_ylabel("Loss")
    ax_avg.set_title("Average")
    ax_avg.set_yscale("log")
    ax_avg.legend()

    fig.suptitle("Loss per Epoch: test_1 vs conFIG", fontsize=12, y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_losses_comparison()
