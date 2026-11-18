"""Energy budget of Evrard's collapse across the self-gravity couplings.

Figure: ``figures/energy_conservation_comparison_fp64.{svg,pdf,png}`` (with
``--double``; Fig. 14 of the methods paper).

Runs the cold Evrard collapse (``e0 = 0.05``) at ``128^3`` with the
finite-difference baseline (WENO-Z, positivity-preserving WENO, Pallas) and
every self-gravity coupling: the simple source, the flux-based and the
corrected flux-based coupling, and the corrected coupling with flux-corrected
gravitational work, and with the second-order finite-volume scheme (VL2, PLM,
HLLC, C_cfl = 0.3, native backend) with Athena++'s coupling, with and without
the time-centred energy source of Mullen, Hanawa & Gammie (2021). It compares the evolution of the energy budget (total /
internal / kinetic / gravitational) and the relative total-energy error, and
records the smallest pressure of every run.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/energy_conservation_comparison.py --rerun --double

Pass ``--rerun`` to re-run the simulations; otherwise the cached results in
``data/energy_conservation_comparison_fp64.npz`` are used.
``--rerun --double --schemes 3 3_tc`` reruns only the finite-volume scheme.
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import sys

# jax
import jax

# ``--double`` runs in float64, as in the paper. x64 must be set before any
# array is created, hence before jax.numpy is imported (by the helpers below).
DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import PALLAS

# figure helpers
from _collapse import (
    COLD_INITIAL_ENERGY,
    COURANT_NUMBER,
    CollapseScheme,
    relative_energy_error,
    run_collapse,
)
from _common import (
    COLLAPSE_COMPARISON_SCHEMES,
    FIGURE_COLLAPSE_SCHEMES,
    DATA_DIR,
    COLUMN_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    apply_paper_style,
    figure_legend,
    save_figure,
)

SUFFIX = "_fp64" if DOUBLE else ""
PRECISION_LABEL = "float64" if DOUBLE else "float32"

RESOLUTION = 128
T_END = 1.2
NUM_SNAPSHOTS = 121
DATA_FILE = DATA_DIR / f"energy_conservation_comparison{SUFFIX}.npz"
FIGURE_STEM = f"energy_conservation_comparison{SUFFIX}"


def run_and_cache(schemes):
    """Run the collapse for the given couplings and cache the energy diagnostics.

    Writes the per-scheme time series of the total / internal / kinetic /
    gravitational energy, the smallest final pressure and the number of steps
    to ``DATA_FILE``. The cached data of schemes that are not run are kept.

    Args:
        schemes: The schemes to run.
    """
    save_kwargs = {}
    if DATA_FILE.exists():
        save_kwargs.update(np.load(DATA_FILE))
    save_kwargs["resolution"] = np.array(RESOLUTION)
    save_kwargs["courant_number"] = np.array(COURANT_NUMBER)
    for scheme in schemes:
        print(f"  {scheme.label} (N={RESOLUTION}, {PRECISION_LABEL})", flush=True)
        snapshots, _, registered_variables = run_collapse(
            RESOLUTION,
            CollapseScheme(
                self_gravity_version=scheme.self_gravity_version,
                work_flux_correction=scheme.work_flux_correction,
                time_centred_energy_source=scheme.time_centred_energy_source,
            ),
            t_end=T_END,
            initial_energy=COLD_INITIAL_ENERGY,
            backend=PALLAS,
            num_snapshots=NUM_SNAPSHOTS,
        )
        energy_error = np.asarray(relative_energy_error(snapshots))
        minimum_pressure = float(
            snapshots.final_state[registered_variables.pressure_index].min()
        )
        print(
            f"    max |dE|/|E0| = {energy_error.max():.3e}, final {energy_error[-1]:.3e}, "
            f"min p(t_end) = {minimum_pressure:.2e}, steps = {int(snapshots.num_iterations)}",
            flush=True,
        )
        save_kwargs[f"time_{scheme.key}"] = np.asarray(snapshots.time_points)
        save_kwargs[f"total_{scheme.key}"] = np.asarray(snapshots.total_energy)
        save_kwargs[f"internal_{scheme.key}"] = np.asarray(snapshots.internal_energy)
        save_kwargs[f"kinetic_{scheme.key}"] = np.asarray(snapshots.kinetic_energy)
        save_kwargs[f"gravitational_{scheme.key}"] = np.asarray(
            snapshots.gravitational_energy
        )
        save_kwargs[f"minimum_pressure_{scheme.key}"] = np.array(minimum_pressure)
        save_kwargs[f"num_steps_{scheme.key}"] = np.array(int(snapshots.num_iterations))
    np.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    """Draw the two-panel energy figure from the cached diagnostics.

    The top panel shows the energy budget (total / internal / kinetic /
    gravitational) over time, colour-coded by component and line-styled by
    scheme; the bottom panel shows the relative total-energy error per scheme.
    """
    data = np.load(DATA_FILE)
    resolution = int(data["resolution"])

    apply_paper_style()
    # Side by side in the single-column layout, stacked in one column (sharing
    # the time axis) in the two-column layout.
    if TWO_COLUMN:
        fig, (ax_terms, ax_error) = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 6.2), sharex=True, layout="constrained")
    else:
        fig, (ax_terms, ax_error) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.4), layout="constrained")

    # Fixed colours per energy component, none of them a colour of a
    # coupling; line style distinguishes the coupling.
    component_colors = {
        "total": "black",
        "internal": "tab:red",
        "kinetic": "tab:olive",
        "gravitational": "tab:cyan",
    }

    # --------------- ↓ Per-scheme curves ↓ ----------------
    for scheme in FIGURE_COLLAPSE_SCHEMES:
        time = data[f"time_{scheme.key}"]
        for component, color in component_colors.items():
            ax_terms.plot(
                time,
                data[f"{component}_{scheme.key}"],
                color=color,
                linestyle=scheme.linestyle,
                linewidth=1.0,
            )

        total = data[f"total_{scheme.key}"]
        relative_error = np.abs(total - total[0]) / np.abs(total[0])
        # The error is exactly zero at t = 0; leave that point off the log axis.
        ax_error.plot(
            time[1:],
            relative_error[1:],
            color=scheme.color,
            linestyle=scheme.linestyle,
            linewidth=scheme.linewidth,
            label=scheme.label,
        )
    # --------------- ↑ Per-scheme curves ↑ ----------------

    ax_terms.set_ylim(-2.5, 2.5)
    if not TWO_COLUMN:
        ax_terms.set_xlabel("time")
    ax_terms.set_ylabel("energy")
    ax_terms.set_title(f"energy budget ($N = {resolution}^3$)")

    # Colour encodes the energy component (legend in the left panel), line
    # style and colour the coupling (one legend below both panels).
    component_handles = [
        plt.Line2D([], [], color=color, label=component)
        for component, color in component_colors.items()
    ]
    ax_terms.legend(handles=component_handles, loc="upper left")

    ax_error.set_yscale("log")
    ax_error.set_xlabel("time")
    ax_error.set_ylabel("relative total energy error")
    ax_error.set_title("relative total energy error")
    figure_legend(fig, ncols=1 if TWO_COLUMN else 2)

    print(f"Saved -> {save_figure(fig, FIGURE_STEM)}")

    for scheme in COLLAPSE_COMPARISON_SCHEMES:
        total = data[f"total_{scheme.key}"]
        relative_error = np.abs(total - total[0]) / np.abs(total[0])
        print(
            f"  {scheme.label}: max |dE|/|E0| = {relative_error.max():.3e}, "
            f"min p(t_end) = {float(data[f'minimum_pressure_{scheme.key}']):.2e}"
        )


def main():
    """Parse arguments, (re)run or load the cache, then draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="re-run simulations (otherwise use cache)"
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="run in float64 (the figure of the paper)",
    )
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=None,
        help="with --rerun, run only the schemes with these cache keys (e.g. 3 3_tc "
        "for the finite-volume scheme) and keep the cached data of the others",
    )
    args = parser.parse_args()

    schemes = COLLAPSE_COMPARISON_SCHEMES
    if args.schemes is not None:
        schemes = [scheme for scheme in COLLAPSE_COMPARISON_SCHEMES if scheme.key in args.schemes]
    if args.rerun or not DATA_FILE.exists():
        run_and_cache(schemes)
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot()


if __name__ == "__main__":
    main()
