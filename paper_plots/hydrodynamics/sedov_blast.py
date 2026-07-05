"""Sedov-Taylor blast wave — paper figure (4x3 layout at 128^3).

Radial profiles (density, |v|, pressure) of the spherical Sedov-Taylor blast
at t = 0.1 compared with the exact self-similar solution, for four solvers,
one per row, all at 128^3 cells:

    row 0:  FV (HLL, minmod)
    row 1:  FV (HLLC, minmod)
    row 2:  FV (AM-HLLC, minmod)
    row 3:  FD (WENO)

Each run is cached separately under ``data/sedov_<key>_<N>.npz`` (binned radial
profiles + a scatter subsample), so the figure can be re-styled without
re-running any simulation.  Pass ``--rerun`` to recompute.

Resolution can be overridden for a quick smoke test:

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/sedov_blast.py --res 32 --rerun

Default run (the paper figure):

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/sedov_blast.py --rerun
"""

import sys

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jr

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    finalize_config,
    time_integration,
)
from astronomix import CARTESIAN
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    FINITE_DIFFERENCE,
    HLL,
    HLLC,
    AM_HLLC,
    HYBRID_HLLC,
    MINMOD,
    PALLAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    PositivityConfig,
)

from exactpack.solvers.sedov.sedov import Sedov

from _common import DATA_DIR, FIG_DIR, rerun_requested
from _common import FV_HLL_COLOR, FV_HLLC_COLOR, FV_AM_HLLC_COLOR, FD_COLOR

# ---- physical setup (matches tests/hydro_tests/sedov3D.py) ----------------
T_END = 0.1
GAMMA = 5.0 / 3.0
E_EXPLOSION = 1.0
RHO_AMBIENT = 1.0
P_AMBIENT = 1e-4
# Injection region: a well-resolved sphere (~8 cells radius at 256^3) with a
# smooth tanh taper of ~2 cells, instead of a single-cell sharp top-hat.  The
# taper removes the grid-imprint noise the sharp injection leaves near the
# origin, and the explosion pressure is renormalised (below) so the total
# deposited thermal energy is exactly E_EXPLOSION regardless of the smoothing.
R_EXPLOSION = 0.03
SMOOTH_CELLS = 2.0
NUM_BINS = 200
NUM_SCATTER = 80_000

# ---- the four solver rows -------------------------------------------------
# (key, label, color, solver_mode, riemann_solver)
RUNS = [
    ("fv_hll", "FV (HLL, minmod)", FV_HLL_COLOR, FINITE_VOLUME, HLL),
    ("fv_hllc", "FV (HLLC, minmod)", FV_HLLC_COLOR, FINITE_VOLUME, HLLC),
    ("fv_amhllc", "FV (AM-HLLC, minmod)", FV_AM_HLLC_COLOR, FINITE_VOLUME, AM_HLLC),
    ("fd", "FD", FD_COLOR, FINITE_DIFFERENCE, HYBRID_HLLC),
]


def _res():
    if "--res" in sys.argv:
        return int(sys.argv[sys.argv.index("--res") + 1])
    return 256


def make_config(solver_mode, riemann_solver, num_cells):
    is_fd = solver_mode == FINITE_DIFFERENCE
    kwargs = dict(
        geometry=CARTESIAN,
        solver_mode=solver_mode,
        riemann_solver=riemann_solver,
        limiter=MINMOD,
        dimensionality=3,
        num_cells=num_cells,
        exact_end_time=True,
        progress_bar=True,
        mhd=is_fd,  # the FD/WENO backend runs in MHD mode (B = 0 here)
    )
    if is_fd:
        # periodic box + positivity protection for the strong point explosion.
        # The FD/WENO backend runs ~10x faster through the Pallas (Triton)
        # backend; results are bit-compatible with native JAX.
        kwargs.update(
            positivity_config=PositivityConfig(default_positivity_protection=True),
            backend=PALLAS,
            pallas_block_shape=(4, 4, 8),
            pallas_use_triton=True,
            pallas_interpret=False,
            boundary_settings=BoundarySettings(
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            ),
        )
    return SimulationConfig(**kwargs)


def simulate(solver_mode, riemann_solver, num_cells):
    config = make_config(solver_mode, riemann_solver, num_cells)
    helper_data = get_helper_data(config)
    rv = get_registered_variables(config)

    shape = (num_cells, num_cells, num_cells)
    rho = jnp.ones(shape) * RHO_AMBIENT
    zeros = jnp.zeros(shape)

    # Smoothly-tapered spherical injection weight in [0, 1] (≈1 inside
    # R_EXPLOSION, tanh taper of SMOOTH_CELLS cells), then renormalise the
    # over-pressure so the deposited thermal energy above ambient,
    #   E = Σ (p - p_amb)/(γ-1) · ΔV ,
    # is exactly E_EXPLOSION (independent of the taper / resolution).
    dx = 1.0 / num_cells  # box_size = 1.0
    smooth_width = SMOOTH_CELLS * dx
    r = helper_data.r
    weight = 0.5 * (1.0 - jnp.tanh((r - R_EXPLOSION) / smooth_width))
    cell_volume = dx**3
    delta_p = E_EXPLOSION * (GAMMA - 1.0) / (jnp.sum(weight) * cell_volume)
    p_gas = P_AMBIENT + delta_p * weight

    fields = dict(
        density=rho, velocity_x=zeros, velocity_y=zeros, velocity_z=zeros,
        gas_pressure=p_gas,
    )
    if config.mhd:
        fields.update(magnetic_field_x=zeros, magnetic_field_y=zeros, magnetic_field_z=zeros)

    initial_state = construct_primitive_state(config=config, registered_variables=rv, **fields)
    config = finalize_config(config, initial_state.shape)

    params = SimulationParams(t_end=T_END, gamma=GAMMA)
    if config.mhd:
        params = params._replace(minimum_density=1e-6, minimum_pressure=1e-6)

    result = time_integration(initial_state, config, params, rv)

    # radial reduction
    r_flat = helper_data.r.flatten()
    rho_flat = result[rv.density_index].flatten()
    p_flat = result[rv.pressure_index].flatten()
    v = rv.velocity_index
    v_abs = jnp.sqrt(
        result[v.x] ** 2 + result[v.y] ** 2 + result[v.z] ** 2
    ).flatten()

    domain_max_r = float(jnp.max(r_flat))
    bins = jnp.linspace(0, domain_max_r, NUM_BINS + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    idx = jnp.clip(jnp.searchsorted(bins, r_flat, side="right") - 1, 0, NUM_BINS - 1)

    counts = jnp.zeros(NUM_BINS).at[idx].add(1.0)
    counts = jnp.where(counts == 0, 1.0, counts)
    mean_rho = jnp.zeros(NUM_BINS).at[idx].add(rho_flat) / counts
    mean_v = jnp.zeros(NUM_BINS).at[idx].add(v_abs) / counts
    mean_p = jnp.zeros(NUM_BINS).at[idx].add(p_flat) / counts

    # scatter subsample (reproducible)
    perm = jr.permutation(jr.PRNGKey(42), r_flat.shape[0])[:NUM_SCATTER]

    return dict(
        bin_centers=np.asarray(bin_centers),
        mean_rho=np.asarray(mean_rho),
        mean_v=np.asarray(mean_v),
        mean_p=np.asarray(mean_p),
        r_scatter=np.asarray(r_flat[perm]),
        rho_scatter=np.asarray(rho_flat[perm]),
        v_scatter=np.asarray(v_abs[perm]),
        p_scatter=np.asarray(p_flat[perm]),
        domain_max_r=np.asarray(domain_max_r),
    )


def cache_path(key, num_cells):
    return DATA_DIR / f"sedov_{key}_{num_cells}.npz"


def get_run(key, solver_mode, riemann_solver, num_cells, rerun):
    path = cache_path(key, num_cells)
    if path.exists() and not rerun:
        return dict(np.load(path))
    print(f"running Sedov {key} at {num_cells}^3 ...")
    data = simulate(solver_mode, riemann_solver, num_cells)
    np.savez(path, **data)
    print(f"  cached -> {path}")
    return data


def exact_solution(domain_max_r):
    solver = Sedov(geometry=3, eblast=E_EXPLOSION / RHO_AMBIENT, gamma=GAMMA, omega=0.0)
    r_exact = np.linspace(0.0, domain_max_r, 500)
    sol = solver(r=r_exact, t=T_END)
    return r_exact, sol


def plot(num_cells, rerun):
    fig, axes = plt.subplots(len(RUNS), 3, figsize=(12, 3.2 * len(RUNS)), sharex=True)

    col_labels = ["density", r"$|v|$", "pressure"]

    for i, (key, label, color, mode, rs) in enumerate(RUNS):
        d = get_run(key, mode, rs, num_cells, rerun)
        domain_max_r = float(d["domain_max_r"])
        r_exact, sol = exact_solution(domain_max_r)

        profiles = [
            ("rho", d["rho_scatter"], d["mean_rho"], sol["density"], False),
            ("v", d["v_scatter"], d["mean_v"], sol["velocity"], False),
            ("p", d["p_scatter"], d["mean_p"], sol["pressure"], True),
        ]
        for j, (_, scatter, mean, exact, logy) in enumerate(profiles):
            ax = axes[i, j]
            ax.scatter(d["r_scatter"], scatter, color="lightgray", alpha=0.5, s=1,
                       rasterized=True, label="simulation (sampled)")
            ax.plot(d["bin_centers"], mean, "-", color=color, lw=1.8, label="binned mean")
            ax.plot(r_exact, exact, "--", color="black", lw=1.5, label="exact solution")
            ax.grid(True, ls=":", alpha=0.6)
            ax.set_xlim(0, domain_max_r)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(bottom=P_AMBIENT / 2)
            else:
                ax.set_ylim(0, None)
            ax.set_ylabel(col_labels[j])
        # row label (solver) on the left, outside the axes — not a title
        axes[i, 0].annotate(
            label, xy=(0, 0.5), xytext=(-axes[i, 0].yaxis.labelpad - 28, 0),
            xycoords=axes[i, 0].yaxis.label, textcoords="offset points",
            ha="right", va="center", rotation=90, fontsize=11,
        )

    for j in range(3):
        axes[-1, j].set_xlabel("radius")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.005))

    plt.tight_layout(rect=[0.04, 0.03, 1, 1])
    out = FIG_DIR / f"sedov_blast_{num_cells}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    num_cells = _res()
    plot(num_cells, rerun_requested())
