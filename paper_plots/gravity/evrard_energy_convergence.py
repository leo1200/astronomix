"""Energy conservation vs resolution for a *mild* Evrard collapse.

A companion to ``subsonic_turbulence_energy`` but for the canonical Evrard
collapse instead of driven turbulence. The standard cold Evrard test
(initial thermal energy per mass e = 0.05) collapses so violently that the
conservative finite-difference flux schemes NaN below 128^3. Warming the initial
gas to ``INITIAL_ENERGY = 0.20`` gives a milder collapse (peak density ~10-20x,
a bounce rather than a near-singularity) that stays finite for all three schemes
at every resolution 32^3-128^3, while still doing enough gravitational work that
the non-conservative *simple* source accumulates a large energy error.

Figure (two panels, like the turbulence one): final relative total energy error
|E(t_end) - E(0)| / |E(0)| vs resolution.
  * left  -- non-conservative simple source (decreases ~N^-2, but starts at ~12%
             at 32^3): the source-term work error is real and converges only
             slowly.
  * right -- the two flux-based (conservative) schemes, pinned at the round-off
             floor (~1e-5 in float32, ~1e-11 in float64).

As with the Evrard energy-vs-time figure, ``--double`` runs on the native JAX
backend (the Pallas/Triton kernel floors at reduced precision).

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/gravity/evrard_energy_convergence.py
    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/gravity/evrard_energy_convergence.py --rerun --double
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import sys

import jax

DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix.option_classes.simulation_config import (
    SECOND_ORDER_CONSERVATIVE,
    NATIVE_JAX,
    PALLAS,
    SIMPLE_SOURCE,
    FOURTH_ORDER_CONSERVATIVE,
)

from _collapse import run_collapse
from _common import DATA_DIR, FIG_DIR, SCHEMES, apply_paper_style

RUN_BACKEND = NATIVE_JAX if DOUBLE else PALLAS
_SUFFIX = "_fp64" if DOUBLE else ""
_PREC_LABEL = "float64" if DOUBLE else "float32"

RESOLUTIONS = [32, 48, 64, 96, 128]
INITIAL_ENERGY = 0.20   # warm IC -> mild collapse, finite for all schemes at 32^3
T_END = 1.2

DATA_FILE = DATA_DIR / f"evrard_energy_convergence{_SUFFIX}.npz"
FIG_FILE = FIG_DIR / f"evrard_energy_convergence{_SUFFIX}.svg"


def run_and_cache():
    energy_errors = {s.self_gravity_version: [] for s in SCHEMES}

    for num_cells in RESOLUTIONS:
        for scheme in SCHEMES:
            print(f"  N={num_cells} {scheme.label} ({_PREC_LABEL})", flush=True)
            snapshots, _, _ = run_collapse(
                num_cells, scheme.self_gravity_version, t_end=T_END,
                backend=RUN_BACKEND, initial_energy=INITIAL_ENERGY,
            )
            total = snapshots.total_energy
            rel_err = float(jnp.abs(total[-1] - total[0]) / jnp.abs(total[0]))
            energy_errors[scheme.self_gravity_version].append(rel_err)
            print(f"    |dE|/|E0| = {rel_err:.6e}", flush=True)

    save_kwargs = {"resolutions": np.array(RESOLUTIONS)}
    for version, errs in energy_errors.items():
        save_kwargs[f"energy_error_{version}"] = np.array(errs)
    np.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def _fit_power_law(n, error):
    """Least-squares fit ``error ~ A N**p`` in log-log space (Evrard is
    deterministic, so the trends are clean and every point is used)."""
    n = np.asarray(n, dtype=float)
    error = np.asarray(error, dtype=float)
    p, log_a = np.polyfit(np.log(n), np.log(error), 1)
    return float(p), float(np.exp(log_a))


def plot_convergence():
    data = np.load(DATA_FILE)
    resolutions = data["resolutions"]

    apply_paper_style()
    by_version = {s.self_gravity_version: s for s in SCHEMES}
    # (subtitle, schemes to draw, scheme to power-law fit, #trailing points to fit)
    # The conservative error is shallow at coarse N then steepens, so its fit uses
    # only the last three (well-resolved) points; the simple source is a clean
    # power law throughout, so it is fit over all resolutions.
    panels = [
        ("non-conservative source", [SIMPLE_SOURCE], SIMPLE_SOURCE, None),
        (
            "flux-based (conservative) sources",
            [SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE],
            FOURTH_ORDER_CONSERVATIVE,
            3,
        ),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for ax, (subtitle, versions, fit_version, fit_n_last) in zip(axes, panels):
        for v in versions:
            scheme = by_version[v]
            ax.plot(
                resolutions,
                data[f"energy_error_{v}"],
                marker=scheme.marker,
                linestyle=scheme.linestyle,
                color=scheme.color,
                linewidth=2.0,
                label=scheme.label,
            )
        # Power-law fit, drawn only where the error genuinely converges (negative
        # slope). The fp32 conservative panel is round-off-limited and *rises*
        # with N, so no fit is shown there.
        n_fit = np.asarray(resolutions, dtype=float)
        e_fit = np.asarray(data[f"energy_error_{fit_version}"], dtype=float)
        order = np.argsort(n_fit)
        n_fit, e_fit = n_fit[order], e_fit[order]
        if fit_n_last is not None:
            n_fit, e_fit = n_fit[-fit_n_last:], e_fit[-fit_n_last:]
        p, A = _fit_power_law(n_fit, e_fit)
        if p < 0:
            n_line = np.array([n_fit.min(), n_fit.max()])
            ax.plot(
                n_line,
                A * n_line**p,
                color="0.25",
                linestyle="-",
                linewidth=1.5,
                zorder=0,
                label=rf"fit: $\propto N^{{{p:.2f}}}$",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(list(resolutions))
        ax.set_xticklabels([str(int(r)) for r in resolutions])
        ax.minorticks_off()
        ax.set_xlabel("number of cells per dimension $N$")
        ax.set_title(subtitle)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    axes[0].set_ylabel(r"final relative total energy error $|\Delta E| / |E_0|$")

    fig.suptitle(
        f"Mild Evrard collapse ($e_0 = {INITIAL_ENERGY}$, {_PREC_LABEL}): "
        "energy conservation vs resolution"
    )
    fig.tight_layout()
    fig.savefig(FIG_FILE)
    print(f"Saved -> {FIG_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="re-run simulations (otherwise use cache)"
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="run in float64 (native backend) to expose the round-off floor",
    )
    args = parser.parse_args()

    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot_convergence()


if __name__ == "__main__":
    main()
