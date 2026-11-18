"""Evrard collapse radial profiles across the self-gravity couplings.

Figure: ``figures/collapse_radial_profiles_comparison_fp64.{svg,pdf,png}``
(with ``--double``; Fig. 15 of the methods paper).

Compares the radial profiles (density, radial velocity, entropy P/rho^gamma) of
the cold Evrard collapse (``e0 = 0.05``) at ``t = 0.8`` and ``128^3`` for every
self-gravity coupling, with the finite-difference baseline (WENO-Z,
positivity-preserving WENO, Pallas) and the second-order finite-volume scheme
(VL2 with Athena++'s coupling), and the same scheme names as the other gravity
figures.

Run from the repo root:

    PYTHONPATH=$(git rev-parse --show-toplevel) python examples/scripts/forward/self_gravity/radial_profiles_comparison.py --rerun --double

Pass ``--rerun`` to re-run the simulations; otherwise the cached results in
``data/radial_profiles_comparison_fp64.npz`` are used.
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
# array is created, hence before jax.numpy is imported below.
DOUBLE = "--double" in sys.argv
if DOUBLE:
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# numerics
import numpy as np

# plotting
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# astronomix constants
from astronomix import PALLAS

# figure helpers
from _collapse import (
    COLD_INITIAL_ENERGY,
    GAMMA,
    CollapseScheme,
    run_collapse,
)
from _common import (
    COLLAPSE_COMPARISON_SCHEMES,
    FIGURE_COLLAPSE_SCHEMES,
    DATA_DIR,
    TEXT_WIDTH,
    apply_paper_style,
    figure_legend,
    save_figure,
)

SUFFIX = "_fp64" if DOUBLE else ""
PRECISION_LABEL = "float64" if DOUBLE else "float32"

RESOLUTION = 128
T_END = 0.8
DATA_FILE = DATA_DIR / f"radial_profiles_comparison{SUFFIX}.npz"
FIGURE_STEM = f"collapse_radial_profiles_comparison{SUFFIX}"


def run_and_cache(schemes):
    """Run the given schemes to ``T_END`` and cache the final radial profiles.

    The density, (inward) radial velocity and entropy ``P / rho**gamma`` of the
    collapsed state are flattened to per-cell point clouds against radius and
    written to ``DATA_FILE`` (compressed, float32). The cached profiles of
    schemes that are not run are kept.

    Args:
        schemes: The schemes to run.
    """
    save_kwargs = {}
    if DATA_FILE.exists():
        save_kwargs.update(np.load(DATA_FILE))
    save_kwargs["resolution"] = np.array(RESOLUTION)
    radius_saved = "r" in save_kwargs
    for scheme in schemes:
        print(f"  {scheme.label} (N={RESOLUTION}, {PRECISION_LABEL})", flush=True)
        snapshots, helper_data, registered_variables = run_collapse(
            RESOLUTION,
            CollapseScheme(
                self_gravity_version=scheme.self_gravity_version,
                work_flux_correction=scheme.work_flux_correction,
                time_centred_energy_source=scheme.time_centred_energy_source,
            ),
            t_end=T_END,
            initial_energy=COLD_INITIAL_ENERGY,
            backend=PALLAS,
        )
        final_state = snapshots.final_state
        print(
            f"    min p(t_end) = {float(final_state[registered_variables.pressure_index].min()):.2e}",
            flush=True,
        )

        # The radius is scheme-independent, so it is stored only once.
        if not radius_saved:
            save_kwargs["r"] = np.asarray(helper_data.r).ravel().astype(np.float32)
            radius_saved = True

        density = final_state[registered_variables.density_index]
        v_r = -jnp.sqrt(
            final_state[registered_variables.velocity_index.x] ** 2
            + final_state[registered_variables.velocity_index.y] ** 2
            + final_state[registered_variables.velocity_index.z] ** 2
        )
        entropy = final_state[registered_variables.pressure_index] / density**GAMMA

        save_kwargs[f"density_{scheme.key}"] = (
            np.asarray(density).ravel().astype(np.float32)
        )
        save_kwargs[f"v_r_{scheme.key}"] = np.asarray(v_r).ravel().astype(np.float32)
        save_kwargs[f"entropy_{scheme.key}"] = (
            np.asarray(entropy).ravel().astype(np.float32)
        )

    np.savez_compressed(DATA_FILE, **save_kwargs)
    print(f"Saved -> {DATA_FILE}")


def plot():
    """Draw the three-panel (density / radial velocity / entropy) figure."""
    data = np.load(DATA_FILE)
    resolution = int(data["resolution"])
    r = data["r"]

    apply_paper_style()
    fig, (ax_rho, ax_v, ax_s) = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 3.1), layout="constrained")

    # The point clouds hold ~2M points per scheme; rasterize them so the SVG
    # stays small while axes, ticks and labels remain vector.
    for scheme in FIGURE_COLLAPSE_SCHEMES:
        ax_rho.scatter(
            r,
            data[f"density_{scheme.key}"],
            s=1,
            color=scheme.color,
            rasterized=True,
            label=scheme.label,
        )
        ax_v.scatter(
            r,
            data[f"v_r_{scheme.key}"],
            s=1,
            color=scheme.color,
            rasterized=True,
            label=scheme.label,
        )
        ax_s.scatter(
            r,
            data[f"entropy_{scheme.key}"],
            s=1,
            color=scheme.color,
            rasterized=True,
            label=scheme.label,
        )

    ax_rho.set_xscale("log")
    ax_rho.set_yscale("log")
    ax_rho.set_xlim(1e-2, 6e-1)
    ax_rho.set_ylim(1e-2, 1e3)
    ax_rho.set_xlabel("$r$")
    ax_rho.set_ylabel("density")

    ax_v.set_xscale("log")
    ax_v.set_xlim(1e-2, 6e-1)
    ax_v.set_xlabel("$r$")
    ax_v.set_ylabel("radial velocity")

    ax_s.set_xscale("log")
    ax_s.set_xlim(1e-2, 6e-1)
    ax_s.set_ylim(0, 0.2)
    ax_s.set_xlabel("$r$")
    ax_s.set_ylabel(r"$P / \rho^\gamma$")

    # The radial range spans less than two decades, where matplotlib would
    # label the minor ticks too and the labels collide.
    for ax in (ax_rho, ax_v, ax_s):
        ax.xaxis.set_minor_formatter(NullFormatter())

    # Single shared legend with larger markers than the s=1 scatter points.
    handles = [
        plt.Line2D(
            [], [], marker="o", linestyle="", color=scheme.color, label=scheme.label
        )
        for scheme in FIGURE_COLLAPSE_SCHEMES
    ]
    figure_legend(fig, handles, ncols=2)

    # The dpi sets the resolution of the rasterized scatter layers.
    print(f"Saved -> {save_figure(fig, FIGURE_STEM, dpi=300)}")


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
