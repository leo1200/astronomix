"""Evrard collapse radial profiles across the source-term schemes.

Figure: ``figures/collapse_radial_profiles_comparison.svg``.

Compares the radial profiles (density, radial velocity, entropy P/rho^gamma) of
the collapsed state at ``t = 0.8`` for the three finite-difference self-gravity
source-term schemes, using the same scheme names as the other gravity figures.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/gravity/radial_profiles_comparison.py

Pass ``--rerun`` to re-run the simulations; otherwise the cached results in
``data/radial_profiles_comparison.npz`` are used.
"""

# ==== GPU selection ====
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import argparse
import sys

import jax

# ``--double``: rerun the collapse radial profiles in float64. x64 must be set
# before any array is created. Native backend (the Pallas kernel runs reduced
# precision); see _collapse._backend_kwargs.
DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix.option_classes.simulation_config import NATIVE_JAX, PALLAS

from _collapse import GAMMA, run_collapse
from _common import DATA_DIR, FIG_DIR, SCHEMES, apply_paper_style

RUN_BACKEND = NATIVE_JAX if DOUBLE else PALLAS
_SUFFIX = "_fp64" if DOUBLE else ""
_PREC_LABEL = "float64" if DOUBLE else "float32"

RESOLUTION = 128
T_END = 0.8
DATA_FILE = DATA_DIR / f"radial_profiles_comparison{_SUFFIX}.npz"
FIG_FILE = FIG_DIR / f"collapse_radial_profiles_comparison{_SUFFIX}.svg"


def run_and_cache():
    save_kwargs = {"resolution": np.array(RESOLUTION)}
    r_saved = False
    for scheme in SCHEMES:
        print(f"  {scheme.label} (N={RESOLUTION}, {_PREC_LABEL})", flush=True)
        snapshots, helper_data, regvars = run_collapse(
            RESOLUTION, scheme.self_gravity_version, t_end=T_END, backend=RUN_BACKEND
        )
        final_state = snapshots.final_state

        if not r_saved:
            save_kwargs["r"] = np.asarray(helper_data.r).ravel().astype(np.float32)
            r_saved = True

        density = final_state[regvars.density_index]
        v_r = -jnp.sqrt(
            final_state[regvars.velocity_index.x] ** 2
            + final_state[regvars.velocity_index.y] ** 2
            + final_state[regvars.velocity_index.z] ** 2
        )
        entropy = final_state[regvars.pressure_index] / density**GAMMA

        v = scheme.self_gravity_version
        save_kwargs[f"density_{v}"] = np.asarray(density).ravel().astype(np.float32)
        save_kwargs[f"v_r_{v}"] = np.asarray(v_r).ravel().astype(np.float32)
        save_kwargs[f"entropy_{v}"] = np.asarray(entropy).ravel().astype(np.float32)

    np.savez_compressed(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    data = np.load(DATA_FILE)
    resolution = int(data["resolution"])
    r = data["r"]

    apply_paper_style()
    fig, (ax_rho, ax_v, ax_s) = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax in (ax_rho, ax_v, ax_s):
        ax.grid(False)

    # The point clouds hold ~2M points per scheme; rasterize them so the SVG
    # stays small while axes, ticks and labels remain vector.
    for scheme in SCHEMES:
        v = scheme.self_gravity_version
        ax_rho.scatter(
            r, data[f"density_{v}"], s=1, color=scheme.color, rasterized=True,
            label=scheme.label,
        )
        ax_v.scatter(
            r, data[f"v_r_{v}"], s=1, color=scheme.color, rasterized=True,
            label=scheme.label,
        )
        ax_s.scatter(
            r, data[f"entropy_{v}"], s=1, color=scheme.color, rasterized=True,
            label=scheme.label,
        )

    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_xlim(1e-2, 6e-1)
    ax_rho.set_ylim(1e-2, 1e3)
    ax_rho.set_xlabel("r")
    ax_rho.set_ylabel("density")

    ax_v.set_xscale("log")
    ax_v.set_xlim(1e-2, 6e-1)
    ax_v.set_xlabel("r")
    ax_v.set_ylabel("radial velocity")

    ax_s.set_xscale("log")
    ax_s.set_xlim(1e-2, 6e-1)
    ax_s.set_ylim(0, 0.2)
    ax_s.set_xlabel("r")
    ax_s.set_ylabel(r"P / $\rho^\gamma$")

    # Single shared legend with larger markers than the s=1 scatter points.
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=scheme.color, label=scheme.label
        )
        for scheme in SCHEMES
    ]
    ax_rho.legend(handles=handles, loc="lower left")

    fig.suptitle(
        f"Evrard collapse radial profiles, $t = {T_END}$ "
        f"($N = {resolution}^3$, {_PREC_LABEL})"
    )
    fig.tight_layout()
    fig.savefig(FIG_FILE, dpi=300)  # dpi controls the rasterized scatter layers
    print(f"Saved -> {FIG_FILE}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rerun", action="store_true", help="re-run simulations (otherwise use cache)"
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="run in float64 (native backend)",
    )
    args = parser.parse_args()

    if args.rerun or not DATA_FILE.exists():
        run_and_cache()
    else:
        print(f"Using cached results in {DATA_FILE} (pass --rerun to recompute).")
    plot()


if __name__ == "__main__":
    main()
