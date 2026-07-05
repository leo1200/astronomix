"""3D MHD blast-wave test (Seo & Ryu 2023, Sec. 3.7) — paper figures.

Two figures, both built purely from the cached arena simulation outputs under
``arena/results/`` (no simulation is run here):

1. ``mhd_blast_oscillations_comparison.png`` — resolution (128/256 cells) by
   solver-scheme grid of central density slices, reproducing
   ``arena/fv_oscillations_comparison.png`` as is.  It shows how the
   finite-volume Lax/HLL schemes develop spurious post-shock oscillations
   while the finite-difference WENO scheme stays clean.

2. ``mhd_blast_test1_256cells.png`` — the finite-difference result at 256^3 in
   the first two columns (density / kinetic-energy / magnetic-pressure /
   pressure slices) and, in the third (profile) column, the |B|^2 and pressure
   profiles along the box diagonal with the finite-volume (HLL, minmod) result
   at the same resolution overlaid for comparison.

Run with the repo on PYTHONPATH:

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/mhd_blast.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from astronomix.option_classes.simulation_config import FINITE_VOLUME, FINITE_DIFFERENCE

from _common import (
    ARENA_RESULTS,
    FIG_DIR,
    FD_LABEL,
    FV_HLL_LABEL,
    FD_COLOR,
    FV_HLL_COLOR,
    mhd_registered_variables,
)

BOX_SIZE = 1.0  # periodic box of size 1.0 (Seo & Ryu setup)
B0 = 10.0

# variable registries (index final_state by name, not hard-coded integers)
RV_FV = mhd_registered_variables(FINITE_VOLUME)
RV_FD = mhd_registered_variables(FINITE_DIFFERENCE)


def load_final_state(config_name, num_cells):
    path = ARENA_RESULTS / config_name / "data" / f"mhd_blast_test1_{num_cells}cells.npz"
    return np.load(path)["final_state"]


# ===========================================================================
# Figure 1: resolution x scheme oscillation comparison (reproduces arena
# fv_oscillations_comparison.png)
# ===========================================================================
def make_oscillation_comparison():
    resolutions = [128, 256]
    configs = [
        ("fv_mhd_lax_mid", "FV (Lax)"),
        ("fv_mhd_hll_mid", "FV (HLL)"),
        ("fv_mhd_hll_eul", "FV (HLL, Euler)"),
        ("fd_mhd", "FD"),
    ]

    base = 3
    fig, axs = plt.subplots(
        len(resolutions), len(configs),
        figsize=(base * len(configs), base * len(resolutions)),
    )

    vmin, vmax = 0.0, 1.5

    for i, res in enumerate(resolutions):
        for j, (cfg, title) in enumerate(configs):
            final_state = load_final_state(cfg, res)
            rv = RV_FD if cfg == "fd_mhd" else RV_FV
            density_slice = final_state[rv.density_index, :, :, res // 2]
            im = axs[i, j].imshow(density_slice, vmin=vmin, vmax=vmax, cmap="jet")
            if i == 0:
                axs[i, j].set_title(title)
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])

    axs[0, 0].set_ylabel(f"{resolutions[0]}$^3$ cells", fontsize=12)
    axs[1, 0].set_ylabel(f"{resolutions[1]}$^3$ cells", fontsize=12)

    fig.subplots_adjust(bottom=0.18)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.035])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("density", rotation=0, labelpad=8, fontsize=11)
    cbar.ax.xaxis.set_label_position("bottom")

    out = FIG_DIR / "mhd_blast_oscillations_comparison.png"
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


# ===========================================================================
# Figure 2: FD 256^3 slices (cols 0-1) + diagonal profiles with FV overlay
# (col 2)
# ===========================================================================
def make_fd_with_profiles(num_cells=256):
    fd = load_final_state("fd_mhd", num_cells)
    fv = load_final_state("fv_mhd_hll_mid", num_cells)

    mid = num_cells // 2
    extent = (0, BOX_SIZE, 0, BOX_SIZE)

    def slices(state, rv):
        m = rv.magnetic_index
        v = rv.velocity_index
        density = state[rv.density_index, :, :, mid]
        pressure = state[rv.pressure_index, :, :, mid]
        b_sq = (state[m.x] ** 2 + state[m.y] ** 2 + state[m.z] ** 2)[:, :, mid]
        v_sq = (state[v.x] ** 2 + state[v.y] ** 2 + state[v.z] ** 2)[:, :, mid]
        return density, pressure, b_sq, v_sq

    density, pressure, b_sq, v_sq = slices(fd, RV_FD)

    fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))

    def imshow_panel(ax, field, label):
        im = ax.imshow(field.T, origin="lower", extent=extent, cmap="jet")
        cbar = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
        fig.colorbar(im, cax=cbar, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # --- columns 0-1: FD slices ---
    imshow_panel(axs[0, 0], density, "density")
    imshow_panel(axs[0, 1], v_sq, r"$v^2$")
    imshow_panel(axs[1, 0], b_sq, r"$B^2$")
    imshow_panel(axs[1, 1], pressure, "pressure")

    # --- column 2: diagonal profiles, FD vs FV ---
    diag = np.arange(num_cells)
    r_diag = np.sqrt(2.0) * diag * (BOX_SIZE / num_cells)

    _, fv_pressure_sl, fv_b_sq, _ = slices(fv, RV_FV)

    b_diag_fd = b_sq[diag, diag]
    b_diag_fv = fv_b_sq[diag, diag]
    axs[0, 2].plot(r_diag, b_diag_fd, color=FD_COLOR, label=FD_LABEL)
    axs[0, 2].plot(r_diag, b_diag_fv, color=FV_HLL_COLOR, ls="--", label=FV_HLL_LABEL)
    axs[0, 2].set_ylabel(r"$|B|^2$")
    axs[0, 2].set_xlabel("diagonal")
    axs[0, 2].legend(fontsize=8, loc="lower right")

    p_diag_fd = pressure[diag, diag]
    p_diag_fv = fv_pressure_sl[diag, diag]
    axs[1, 2].plot(r_diag, p_diag_fd, color=FD_COLOR, label=FD_LABEL)
    axs[1, 2].plot(r_diag, p_diag_fv, color=FV_HLL_COLOR, ls="--", label=FV_HLL_LABEL)
    axs[1, 2].set_ylabel("pressure")
    axs[1, 2].set_xlabel("diagonal")
    axs[1, 2].legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / f"mhd_blast_test1_{num_cells}cells.png"
    fig.savefig(out, dpi=400)
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    make_oscillation_comparison()
    make_fd_with_profiles(256)
