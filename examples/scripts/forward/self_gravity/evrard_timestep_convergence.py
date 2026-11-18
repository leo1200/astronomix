"""Fixed-time-step convergence of the energy error in Evrard's collapse.

Figure: ``figures/evrard_timestep_convergence_fp64.{svg,pdf,png}`` (Fig. 16 of
the methods paper).

The resolution is fixed at 32^3 and the time step is varied with the
fixed-step driver (``dt = t_end / num_timesteps``), isolating the temporal part
of the energy error. Two collapses share the figure: the standard cold Evrard
collapse (``e0 = 0.05``), where the grid does not resolve the pressure scale
height and the scheme needs positivity-preserving WENO to complete, and the
milder ``e0 = 0.2`` collapse. Both run to ``t = 1.2``, past the formation of the
central accretion shock and its outward propagation through the infalling gas.

The non-conservative simple source has a purely spatial energy error, flat in
``dt``. The flux-based (conservative) couplings conserve total energy exactly in
the semi-discrete limit, so their energy error is the time integrator's and
converges with ``dt`` until the float64 round-off floor. The scheme is the
baseline (WENO-Z, positivity-preserving WENO); the positivity limiter and the
flux-corrected work depend on the state only, never on ``dt``, so the
Runge-Kutta stages stay consistent and the energy error keeps converging. In
the mild collapse it approaches ``dt^4``; in the cold collapse it is produced
in bursts while the central accretion shock forms, where the scheme's
switches (limiter, splitting speeds, weights) make the right-hand side only
Lipschitz, and it falls irregularly, roughly as ``dt^3`` (see
``evrard_order_diagnostics.py``). At the smallest steps the error reaches
the float64 round-off floor.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/evrard_timestep_convergence.py --rerun --double
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

# ``--double`` runs in float64, which the dt^4 convergence needs to be visible
# below 1e-8. x64 must be set before any array is created, hence before
# jax.numpy is imported (by the helpers below).
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
    MILD_INITIAL_ENERGY,
    CollapseScheme,
    relative_energy_error,
    run_collapse,
)
from _common import (
    COLLAPSE_SCHEMES,
    CONSERVATIVE_SCHEMES,
    FIGURE_CONSERVATIVE_SCHEMES,
    SINGLE_PANEL_WIDTH,
    CORRECTED_SCHEME,
    SIMPLE_SCHEME,
    DATA_DIR,
    COLUMN_WIDTH,
    GUIDE_COLOR,
    TEXT_WIDTH,
    TWO_COLUMN,
    apply_paper_style,
    figure_legend,
    observed_orders,
    save_figure,
)

SUFFIX = "_fp64" if DOUBLE else ""
PRECISION_LABEL = "float64" if DOUBLE else "float32"

NUM_CELLS = 32
T_END = 1.2
NUM_TIMESTEPS = [250, 500, 1000, 2000, 4000, 8000, 16000, 32000]
NUM_SNAPSHOTS = 121

#: The reference slopes pass through the corrected coupling at this many
#: steps and are drawn down to the time step where the round-off floor
#: starts to matter.
GUIDE_ANCHOR_STEPS = 1000
GUIDE_SMALLEST_DT = 1.4e-4

#: The level below which the final energy error is float64 round-off (the
#: signed errors of all couplings at the smallest steps scatter below it).
ROUND_OFF_FLOOR = 3e-13

#: The two collapses of the figure: (data key, panel title, e0).
CASES = [
    ("cold", "cold Evrard collapse", COLD_INITIAL_ENERGY),
    ("mild", "mild collapse", MILD_INITIAL_ENERGY),
]

FIGURE_STEM = f"evrard_timestep_convergence{SUFFIX}"


def data_file(case_key):
    """The cache file of one collapse."""
    return DATA_DIR / f"evrard_timestep_convergence{SUFFIX}_{case_key}.npz"


def run_and_cache(case_key, initial_energy):
    """Sweep the fixed time step for every scheme of one collapse and cache it.

    Stores, per scheme, the final relative energy error for every step count
    and the total-energy history (``NUM_SNAPSHOTS`` points) of every run.

    Args:
        case_key: The data key of the collapse (``cold`` or ``mild``).
        initial_energy: The initial thermal energy per unit mass.
    """
    save_kwargs = {
        "num_timesteps": np.array(NUM_TIMESTEPS),
        "dt": T_END / np.array(NUM_TIMESTEPS, dtype=float),
        "initial_energy": np.array(initial_energy),
    }
    for scheme in COLLAPSE_SCHEMES:
        collapse_scheme = CollapseScheme(
            self_gravity_version=scheme.self_gravity_version,
            work_flux_correction=scheme.work_flux_correction,
        )
        final_errors = []
        energy_histories = []
        minimum_pressures = []
        for num_timesteps in NUM_TIMESTEPS:
            snapshots, _, registered_variables = run_collapse(
                NUM_CELLS,
                collapse_scheme,
                T_END,
                initial_energy=initial_energy,
                backend=PALLAS,
                num_snapshots=NUM_SNAPSHOTS,
                fixed_num_timesteps=num_timesteps,
            )
            final_errors.append(float(relative_energy_error(snapshots)[-1]))
            energy_histories.append(np.asarray(snapshots.total_energy))
            minimum_pressures.append(
                float(snapshots.final_state[registered_variables.pressure_index].min())
            )
            print(
                f"  {case_key} nts={num_timesteps:6d} dt={T_END / num_timesteps:.2e} "
                f"{scheme.label}: dE/E = {final_errors[-1]:.4e}, "
                f"min p = {minimum_pressures[-1]:.2e}",
                flush=True,
            )
        print(
            f"  {case_key} {scheme.label}: observed orders "
            + " ".join(f"{order:.2f}" for order in observed_orders(save_kwargs["dt"], final_errors)),
            flush=True,
        )
        save_kwargs[f"energy_error_{scheme.key}"] = np.array(final_errors)
        save_kwargs[f"total_energy_{scheme.key}"] = np.array(energy_histories)
        save_kwargs[f"minimum_pressure_{scheme.key}"] = np.array(minimum_pressures)
        save_kwargs["time"] = np.asarray(snapshots.time_points)
    np.savez(data_file(case_key), **save_kwargs)
    print(f"Saved -> {data_file(case_key)}")


def signed_final_errors(data, scheme):
    """The signed relative energy error ``(E(t_end) - E_0) / |E_0|`` per step count.

    Args:
        data: The cache of one collapse.
        scheme: The coupling.

    Returns:
        The signed error per entry of ``data["num_timesteps"]``.
    """
    total_energy = data[f"total_energy_{scheme.key}"]
    return (total_energy[:, -1] - total_energy[:, 0]) / np.abs(total_energy[:, 0])


def plot():
    """Draw the two-panel (cold, mild) figure from the caches.

    Each panel shows ``|Delta E| / |E_0|`` against the time step for every
    coupling, ``dt^3`` and ``dt^4`` guides through the corrected coupling at
    ``GUIDE_ANCHOR_STEPS`` steps, and the
    float64 round-off floor of the energy error.
    """
    apply_paper_style()
    # The paper shows the cold collapse only; the mild one is quoted in the text.
    figure_cases = CASES[:1]
    fig, ax = plt.subplots(figsize=(SINGLE_PANEL_WIDTH, 4.0), layout="constrained")
    axes = [ax]
    for ax, (case_key, title, initial_energy) in zip(axes, figure_cases):
        data = np.load(data_file(case_key))
        dt = data["dt"]
        # The simple source sits ~8 decades above the conservative couplings
        # and is flat in dt; it is quoted instead of drawn, so the axis can
        # resolve the convergence of the conservative ones.
        simple_errors = data[f"energy_error_{SIMPLE_SCHEME.key}"]
        ax.text(
            0.03,
            0.96,
            f"{SIMPLE_SCHEME.label}: $|\\Delta E|/|E_0| = {simple_errors.mean():.2f}$ at every $\\Delta t$",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color=SIMPLE_SCHEME.color,
        )
        for scheme in FIGURE_CONSERVATIVE_SCHEMES:
            ax.plot(
                dt,
                data[f"energy_error_{scheme.key}"],
                marker=scheme.marker,
                linestyle=scheme.linestyle,
                color=scheme.color,
                linewidth=scheme.linewidth,
                label=scheme.label,
            )

        # Reference slopes through the corrected coupling, drawn over the
        # time steps above the round-off floor.
        anchor_index = list(data["num_timesteps"]).index(GUIDE_ANCHOR_STEPS)
        anchor_dt = dt[anchor_index]
        anchor_error = data[f"energy_error_{CORRECTED_SCHEME.key}"][anchor_index]
        guide_dt = np.sort(dt[dt >= GUIDE_SMALLEST_DT])
        for exponent, linestyle in ((3, (0, (1, 1))), (4, (0, (5, 2)))):
            ax.plot(
                guide_dt,
                anchor_error * 2.0 * (guide_dt / anchor_dt) ** exponent,
                color=GUIDE_COLOR,
                linestyle=linestyle,
                linewidth=1.0,
                label=rf"$\propto \Delta t^{exponent}$",
            )

        ax.axhspan(
            1e-14,
            ROUND_OFF_FLOOR,
            color="0.9",
            zorder=0,
            label="float64 round-off floor",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-14, 1e-7)
        ax.set_xlabel(r"time step $\Delta t$")
        ax.set_title(rf"{title} ($e_{{th,0}} = {initial_energy}$)")
    axes[0].set_ylabel(r"final relative energy error $|\Delta E| / |E_0|$")
    handles, labels = axes[0].get_legend_handles_labels()
    figure_legend(fig, handles, labels, ncols=2)
    figure_path = save_figure(fig, FIGURE_STEM, extensions=("svg", "pdf", "png"))
    print(f"Saved -> {figure_path}")

    # The signed errors and the observed orders between successive time
    # steps, for the text.
    for case_key, _, _ in CASES:
        data = np.load(data_file(case_key))
        for scheme in CONSERVATIVE_SCHEMES:
            signed_errors = signed_final_errors(data, scheme)
            orders = observed_orders(data["dt"], np.abs(signed_errors))
            print(f"  {case_key} {scheme.label}:")
            print("    signed dE/E  " + " ".join(f"{error:+.2e}" for error in signed_errors))
            print("    orders       " + " ".join(f"{order:.2f}" for order in orders))


def main():
    """Parse arguments, (re)run or load the caches, then draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rerun", action="store_true", help="re-run simulations")
    parser.add_argument(
        "--double",
        action="store_true",
        help="run in float64 (the figure of the paper)",
    )
    parser.add_argument(
        "--cases",
        default="cold,mild",
        help="comma-separated collapses to (re)run: cold, mild",
    )
    args = parser.parse_args()
    for case_key, _, initial_energy in CASES:
        if case_key not in args.cases.split(","):
            continue
        if args.rerun or not data_file(case_key).exists():
            run_and_cache(case_key, initial_energy)
        else:
            print(f"Using cached results in {data_file(case_key)} (pass --rerun to recompute).")
    if all(data_file(case_key).exists() for case_key, _, _ in CASES):
        plot()


if __name__ == "__main__":
    main()
