"""Energy conservation vs resolution for the mild Evrard collapse.

Figure: ``figures/evrard_energy_convergence_fp64.{svg,pdf,png}`` (with
``--double``; Fig. 17 of the methods paper).

The warm collapse (initial thermal energy per mass ``e0 = 0.2``) is run to
``t = 1.2`` with the CFL time step at ``32^3 ... 128^3`` for every self-gravity
coupling, with the finite-difference baseline (WENO-Z, positivity-preserving
WENO, Pallas). The final relative total-energy error
``|E(t_end) - E(0)| / |E(0)|`` is plotted against the resolution:

* left: the non-conservative simple source, whose source-term work error is
  spatial and converges only slowly;
* right: the flux-based (conservative) couplings, whose error is the time
  integrator's; under the CFL step ``dt ~ dx`` it falls with the resolution
  until the float64 round-off floor.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/evrard_energy_convergence.py --rerun --double
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
from astronomix.option_classes.simulation_config import SIMPLE_SOURCE

# figure helpers
from _collapse import (
    COURANT_NUMBER,
    MILD_INITIAL_ENERGY,
    CollapseScheme,
    relative_energy_error,
    run_collapse,
)
from _common import (
    COLLAPSE_SCHEMES,
    CORRECTED_SCHEME,
    DATA_DIR,
    SIMPLE_SCHEME,
    COLUMN_WIDTH,
    TEXT_WIDTH,
    TWO_COLUMN,
    apply_paper_style,
    figure_legend,
    save_figure,
)

SUFFIX = "_fp64" if DOUBLE else ""
PRECISION_LABEL = "float64" if DOUBLE else "float32"

RESOLUTIONS = [32, 48, 64, 96, 128]
T_END = 1.2

# Only the final energy is needed. Every snapshot interval also caps the time
# step (dt <= t_end / num_snapshots), which with many snapshots would replace
# the CFL step at the coarse resolutions; three keep the cap above the CFL step.
NUM_SNAPSHOTS = 3

DATA_FILE = DATA_DIR / f"evrard_energy_convergence{SUFFIX}.npz"
FIGURE_STEM = f"evrard_energy_convergence{SUFFIX}"


def run_and_cache():
    """Run the mild collapse over the resolution sweep and cache the errors.

    For every resolution and coupling the final relative total-energy error,
    the smallest final pressure and the number of steps are written to
    ``DATA_FILE``.
    """
    energy_errors = {scheme.key: [] for scheme in COLLAPSE_SCHEMES}
    minimum_pressures = {scheme.key: [] for scheme in COLLAPSE_SCHEMES}
    num_steps = {scheme.key: [] for scheme in COLLAPSE_SCHEMES}

    for num_cells in RESOLUTIONS:
        for scheme in COLLAPSE_SCHEMES:
            print(f"  N={num_cells} {scheme.label} ({PRECISION_LABEL})", flush=True)
            snapshots, _, registered_variables = run_collapse(
                num_cells,
                CollapseScheme(
                    self_gravity_version=scheme.self_gravity_version,
                    work_flux_correction=scheme.work_flux_correction,
                ),
                t_end=T_END,
                initial_energy=MILD_INITIAL_ENERGY,
                backend=PALLAS,
                num_snapshots=NUM_SNAPSHOTS,
            )
            energy_error = float(relative_energy_error(snapshots)[-1])
            minimum_pressure = float(
                snapshots.final_state[registered_variables.pressure_index].min()
            )
            energy_errors[scheme.key].append(energy_error)
            minimum_pressures[scheme.key].append(minimum_pressure)
            num_steps[scheme.key].append(int(snapshots.num_iterations))
            print(
                f"    |dE|/|E0| = {energy_error:.6e}, min p = {minimum_pressure:.2e}, "
                f"steps = {num_steps[scheme.key][-1]}",
                flush=True,
            )

    save_kwargs = {
        "resolutions": np.array(RESOLUTIONS),
        "courant_number": np.array(COURANT_NUMBER),
    }
    for scheme in COLLAPSE_SCHEMES:
        save_kwargs[f"energy_error_{scheme.key}"] = np.array(energy_errors[scheme.key])
        save_kwargs[f"minimum_pressure_{scheme.key}"] = np.array(minimum_pressures[scheme.key])
        save_kwargs[f"num_steps_{scheme.key}"] = np.array(num_steps[scheme.key])
    np.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def _fit_power_law(n, error):
    """Least-squares fit ``error ~ A N**p`` in log-log space.

    Evrard is deterministic, so the trends are clean and every supplied point
    is used (unlike ``_common.fit_power_law``, which masks saturated points).

    Args:
        n: The resolutions (cell counts).
        error: The measured error per resolution.

    Returns:
        ``(p, A)``: the fitted exponent and prefactor.
    """
    n = np.asarray(n, dtype=float)
    error = np.asarray(error, dtype=float)
    p, log_a = np.polyfit(np.log(n), np.log(error), 1)
    return float(p), float(np.exp(log_a))


def plot_convergence():
    """Draw the two-panel resolution-convergence figure from the cache.

    The left panel shows the non-conservative simple source; the right panel
    shows the flux-based conservative couplings. Where the error genuinely
    converges (negative slope) a power-law fit line is overlaid.
    """
    data = np.load(DATA_FILE)
    resolutions = data["resolutions"]

    apply_paper_style()
    conservative_schemes = [
        scheme for scheme in COLLAPSE_SCHEMES if scheme.self_gravity_version != SIMPLE_SOURCE
    ]
    # (subtitle, schemes to draw, scheme to power-law fit, #trailing points to fit)
    # The conservative error is shallow at coarse N, so its fit uses only the
    # last three (well-resolved) points; the simple source is fit throughout.
    panels = [
        ("non-conservative source", [SIMPLE_SCHEME], SIMPLE_SCHEME, None),
        ("flux-based (conservative) sources", conservative_schemes, CORRECTED_SCHEME, 3),
    ]

    # Side by side in the single-column layout, stacked in one column in the
    # two-column layout.
    if TWO_COLUMN:
        fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 6.0), sharex=True, layout="constrained")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 3.2), sharex=True, layout="constrained")
    for ax, (subtitle, schemes, fit_scheme, fit_n_last) in zip(axes, panels):
        for scheme in schemes:
            ax.plot(
                resolutions,
                data[f"energy_error_{scheme.key}"],
                marker=scheme.marker,
                linestyle=scheme.linestyle,
                color=scheme.color,
                linewidth=scheme.linewidth,
                label=scheme.label,
            )
        # Power-law fit, drawn only where the error genuinely converges
        # (negative slope).
        n_fit = np.asarray(resolutions, dtype=float)
        e_fit = np.asarray(data[f"energy_error_{fit_scheme.key}"], dtype=float)
        order = np.argsort(n_fit)
        n_fit, e_fit = n_fit[order], e_fit[order]
        if fit_n_last is not None:
            n_fit, e_fit = n_fit[-fit_n_last:], e_fit[-fit_n_last:]
        p, A = _fit_power_law(n_fit, e_fit)
        print(f"  {fit_scheme.label}: fitted order {p:.2f} over N = {n_fit.astype(int).tolist()}")
        if p < 0:
            n_line = np.array([n_fit.min(), n_fit.max()])
            ax.plot(
                n_line,
                A * n_line**p,
                color="0.25",
                linestyle="-",
                linewidth=0.9,
                zorder=0,
                label=rf"fit: $\propto N^{{{p:.2f}}}$",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(resolutions))
        ax.set_xticklabels([str(int(r)) for r in resolutions])
        ax.minorticks_off()
        if not TWO_COLUMN or ax is axes[-1]:
            ax.set_xlabel("number of cells per dimension $N$")
        ax.set_title(subtitle)
    for ax in axes if TWO_COLUMN else axes[:1]:
        ax.set_ylabel(r"final relative energy error $|\Delta E| / |E_0|$")
    figure_legend(fig, ncols=1 if TWO_COLUMN else 3)
    print(f"Saved -> {save_figure(fig, FIGURE_STEM)}")


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
    args = parser.parse_args()

    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot_convergence()


if __name__ == "__main__":
    main()
