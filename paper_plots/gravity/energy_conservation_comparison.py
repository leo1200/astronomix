"""Energy evolution across the self-gravity source-term schemes.

Figure: ``figures/energy_conservation_comparison.svg``
(the former ``collapse_resolution_study.svg`` -- this is a method comparison at
a single resolution, not a resolution study).

Runs the Evrard collapse at a single resolution with the three finite-difference
self-gravity source-term schemes and compares the evolution of the energy
budget (total / internal / kinetic / gravitational) and the relative total
energy error.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/gravity/energy_conservation_comparison.py

Pass ``--rerun`` to re-run the simulations; otherwise the cached results in
``data/energy_conservation_comparison.npz`` are used.
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import sys

import jax

# ``--double``: rerun Evrard's collapse in float64 to check whether the residual
# energy errors are floating-point round-off. x64 must be set before any array
# is created. Uses the native backend (the Pallas kernel floors at reduced
# precision); see _collapse._backend_kwargs.
DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix.option_classes.simulation_config import (
    NATIVE_JAX,
    PALLAS,
)

from _collapse import run_collapse
from _common import DATA_DIR, FIG_DIR, SCHEMES, apply_paper_style

RUN_BACKEND = NATIVE_JAX if DOUBLE else PALLAS
_SUFFIX = "_fp64" if DOUBLE else ""
_PREC_LABEL = "float64" if DOUBLE else "float32"

# The three shared finite-difference self-gravity source-term schemes.
ENERGY_SCHEMES = list(SCHEMES)


def _scheme_key(scheme):
    """Unique data key per scheme."""
    return f"fd_{scheme.self_gravity_version}"


RESOLUTION = 128
T_END = 1.2
DATA_FILE = DATA_DIR / f"energy_conservation_comparison{_SUFFIX}.npz"
FIG_FILE = FIG_DIR / f"energy_conservation_comparison{_SUFFIX}.svg"


def run_and_cache():
    save_kwargs = {"resolution": np.array(RESOLUTION)}
    for scheme in ENERGY_SCHEMES:
        print(f"  {scheme.label} (N={RESOLUTION}, {_PREC_LABEL})", flush=True)
        snapshots, _, _ = run_collapse(
            RESOLUTION, scheme.self_gravity_version, t_end=T_END,
            backend=RUN_BACKEND, solver_mode=scheme.solver_mode,
        )
        k = _scheme_key(scheme)
        save_kwargs[f"time_{k}"] = np.asarray(snapshots.time_points)
        save_kwargs[f"total_{k}"] = np.asarray(snapshots.total_energy)
        save_kwargs[f"internal_{k}"] = np.asarray(snapshots.internal_energy)
        save_kwargs[f"kinetic_{k}"] = np.asarray(snapshots.kinetic_energy)
        save_kwargs[f"gravitational_{k}"] = np.asarray(snapshots.gravitational_energy)
    jnp.savez(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    data = np.load(DATA_FILE)
    resolution = int(data["resolution"])

    apply_paper_style()
    fig, (ax_terms, ax_err) = plt.subplots(2, 1, figsize=(9, 9))

    # Fixed colours per energy component; line style distinguishes scheme.
    component_colors = {
        "total": "black",
        "internal": "tab:green",
        "kinetic": "tab:red",
        "gravitational": "tab:blue",
    }

    for scheme in ENERGY_SCHEMES:
        k = _scheme_key(scheme)
        time = data[f"time_{k}"]
        for comp, color in component_colors.items():
            ax_terms.plot(
                time,
                data[f"{comp}_{k}"],
                color=color,
                linestyle=scheme.linestyle,
                linewidth=1.8,
            )

        total = data[f"total_{k}"]
        rel_err = np.abs(total - total[0]) / np.abs(total[0])
        ax_err.plot(
            time,
            rel_err,
            color=scheme.color,
            linestyle=scheme.linestyle,
            linewidth=scheme.linewidth,
            label=scheme.label,
        )

    ax_terms.set_ylim(-2.5, 2.5)
    ax_terms.set_xlabel("time")
    ax_terms.set_ylabel("energy")
    ax_terms.set_title(f"Energy budget ($N = {resolution}^3$)")

    # Two compact legends: colour encodes the energy component, line style the
    # source-term scheme.
    component_handles = [
        plt.Line2D([], [], color=color, linewidth=2, label=comp.capitalize())
        for comp, color in component_colors.items()
    ]
    scheme_handles = [
        plt.Line2D(
            [], [], color="0.4", linestyle=scheme.linestyle, linewidth=2,
            label=scheme.label,
        )
        for scheme in ENERGY_SCHEMES
    ]
    legend_components = ax_terms.legend(
        handles=component_handles, loc="upper left", fontsize="small", title="component"
    )
    ax_terms.add_artist(legend_components)
    ax_terms.legend(
        handles=scheme_handles, loc="lower left", fontsize="small", title="scheme"
    )

    ax_err.set_yscale("log")
    ax_err.set_xlabel("time")
    ax_err.set_ylabel("relative total energy error")
    ax_err.set_title("Relative total energy error")
    ax_err.legend(fontsize="small")

    fig.suptitle(
        f"Energy evolution across different source term schemes ({_PREC_LABEL})"
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
        help="run in float64 (native backend) to check for round-off-limited errors",
    )
    args = parser.parse_args()

    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot()


if __name__ == "__main__":
    main()
