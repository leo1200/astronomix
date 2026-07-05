"""Distributions of explicit vs numerical diffusivity in the TRML runs.

For each resolution produced by ``trml_diffusivity.py`` we load the final
primitive state and build, per cell, the temperature ``T = P / rho`` and the
diffusivities:

    explicit (thermal)   chi_phys = (gamma - 1) * kappa / rho
        -- diffusivity acting on T in the conduction term
           (gamma-1)/rho div(kappa grad T) = chi_phys laplacian(T) (constant
           kappa); this is exactly the parabolic-CFL chi.
    numerical (grid)     D_kin = |v| * dx
        -- grid-Peclet (Pe ~ 1) estimate; |v| = sqrt(vx^2+vy^2+vz^2),
           dx = grid spacing.  First-order-upwind-like upper bound.
    numerical (WENO)     D_diss   (from compute_numerical_diffusivity.py)
        -- dissipative-flux estimate D = dx |F_WENO - F_central| / |grad rho|,
           WENO-adaptive; defined on cells with a resolved density gradient.

The constant kinematic viscosity nu = 1e-3 (momentum diffusivity) is drawn as a
reference line; the isobaric thermal relation chi = (gamma-1) kappa T / P0 too.

Output:
    figures/diffusivity_2dhist.png  -- rows = resolution, cols = [explicit,
        numerical |v|dx, numerical dissipative-flux]; 2D histograms in
        (T, diffusivity) space, shared color and axis scales.
    figures/diffusivity_temperature_pdf.png -- temperature PDFs (dV/dT).
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "diffusivity")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# state layout (hydro): [density, vx, vy, vz, pressure]
RHO, VX, VY, VZ, P = 0, 1, 2, 3, 4

# shared plotting ranges
T_LIM = (1e-2, 1.0)          # T_cold .. T_hot
D_LIM = (1e-7, 1e-1)         # diffusivity axis (log)
N_TBINS = 120
N_DBINS = 120

COL_TITLES = [
    "explicit  $\\chi_{\\rm phys}=(\\gamma-1)\\kappa/\\rho$",
    "numerical  $D_{\\rm kin}=|v|\\,\\Delta x$",
    "numerical  $D_{\\rm diss}=\\Delta x\\,|F_{\\rm WENO}-F_{\\rm cent}|/|\\nabla\\rho|$",
]


def load_runs():
    runs = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "trml_N*.npz"))):
        if path.endswith("_numdiff.npz"):
            continue
        d = np.load(path)
        N = int(d["num_cells_x"])
        nd_path = os.path.join(DATA_DIR, f"trml_N{N}_numdiff.npz")
        nd = np.load(nd_path) if os.path.exists(nd_path) else None
        runs[N] = (d, nd)
    return runs


def per_cell_fields(d):
    """Return flat (T, chi_phys, D_kin) for all cells of a run."""
    state = np.asarray(d["final_state"])
    gamma = float(d["gamma"])
    kappa = float(d["thermal_conductivity"])
    dx = float(d["grid_spacing"])

    rho = state[RHO]
    T = state[P] / rho
    speed = np.sqrt(state[VX] ** 2 + state[VY] ** 2 + state[VZ] ** 2)

    chi_phys = (gamma - 1.0) * kappa / rho
    D_kin = speed * dx
    return T, chi_phys, D_kin


def hist2d(T, diff):
    t_edges = np.logspace(np.log10(T_LIM[0]), np.log10(T_LIM[1]), N_TBINS + 1)
    d_edges = np.logspace(np.log10(D_LIM[0]), np.log10(D_LIM[1]), N_DBINS + 1)
    H, _, _ = np.histogram2d(np.asarray(T).ravel(), np.asarray(diff).ravel(),
                             bins=[t_edges, d_edges])
    s = H.sum()
    if s > 0:
        H = H / s  # volume fraction (equal-volume Cartesian cells)
    return H.T, t_edges, d_edges  # transpose so rows=diff, cols=T


def main():
    runs = load_runs()
    if not runs:
        raise SystemExit(f"No runs found in {DATA_DIR}")
    resolutions = sorted(runs)
    print("Resolutions:", resolutions)

    panels = {}   # N -> [H_explicit, H_kin, H_diss(or None)], t_edges, d_edges
    nu = None
    for N in resolutions:
        d, nd = runs[N]
        T, chi, Dkin = per_cell_fields(d)
        nu = float(d["kinematic_viscosity"])
        He, t_edges, d_edges = hist2d(T, chi)
        Hk, _, _ = hist2d(T, Dkin)
        if nd is not None:
            valid = np.asarray(nd["valid"])
            Ddiss = np.asarray(nd["D_num_diss"])
            Hd, _, _ = hist2d(T[valid], Ddiss[valid])
            print(f"  N={N}: dissipative-flux on {valid.mean()*100:.1f}% of cells")
        else:
            Hd = None
            print(f"  N={N}: no dissipative-flux file yet")
        panels[N] = ([He, Hk, Hd], t_edges, d_edges)

    # shared colour scale across every populated panel
    allpos = []
    for hl, _, _ in panels.values():
        for H in hl:
            if H is not None:
                allpos.append(H[H > 0].ravel())
    allpos = np.concatenate(allpos)
    norm = LogNorm(vmin=allpos.min(), vmax=allpos.max())

    # reference: isobaric chi = (gamma-1) kappa T / P0
    d0 = runs[resolutions[0]][0]
    gamma = float(d0["gamma"]); kappa = float(d0["thermal_conductivity"]); P0 = float(d0["P0"])
    t_line = np.logspace(np.log10(T_LIM[0]), np.log10(T_LIM[1]), 200)
    chi_isobaric = (gamma - 1.0) * kappa * t_line / P0

    nrows = len(resolutions)
    ncols = 3
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.6 * nrows), squeeze=False,
        sharex=True, sharey=True,
    )

    im = None
    for r, N in enumerate(resolutions):
        hl, t_edges, d_edges = panels[N]
        for c in range(ncols):
            ax = axes[r, c]
            H = hl[c]
            if H is not None:
                im = ax.pcolormesh(
                    t_edges, d_edges, np.ma.masked_where(H <= 0, H),
                    norm=norm, cmap="viridis", shading="auto",
                )
            else:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(*T_LIM); ax.set_ylim(*D_LIM)
            first = (r == 0 and c == 0)
            ax.axhline(nu, color="orange", ls=":", lw=1.5,
                       label=r"$\nu=10^{-3}$" if first else None)
            ax.plot(t_line, chi_isobaric, color="crimson", ls="--", lw=1.2,
                    label=r"$\chi_{\rm phys}$ (isobaric)" if first else None)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"$N={N}$\n diffusivity")
            if r == nrows - 1:
                ax.set_xlabel("temperature  $T = P/\\rho$")
    axes[0, 0].legend(loc="lower right", fontsize=8, framealpha=0.7)

    fig.subplots_adjust(right=0.90)
    cax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
    fig.colorbar(im, cax=cax, label="volume fraction")
    fig.suptitle("Explicit vs numerical diffusivity (final state)", y=0.997)
    out = os.path.join(FIG_DIR, "diffusivity_2dhist.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)

    # --- temperature PDFs (dV/dT) -----------------------------------------
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    for N in resolutions:
        d = runs[N][0]
        T_cold = float(d["T_cold"]); T_hot = float(d["T_hot"])
        tpdf = np.asarray(d["temperature_pdf"])
        tpts = np.asarray(d["time_points"]); t_sh = float(d["t_sh"])
        mask = tpts >= 10 * t_sh
        if mask.sum() == 0:
            mask = np.ones_like(tpts, dtype=bool)
        avg = tpdf[mask].mean(axis=0)
        n_bins = avg.shape[0]
        logT_edges = np.linspace(np.log10(T_cold), np.log10(T_hot), n_bins + 1)
        T_cen = 10 ** (0.5 * (logT_edges[:-1] + logT_edges[1:]))
        dV_dT = avg / (T_cen * np.log(10))
        V = np.sum(dV_dT * (10 ** logT_edges[1:] - 10 ** logT_edges[:-1]))
        ax2.plot(T_cen, dV_dT / V, label=f"$N={N}$")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("temperature  $T = P/\\rho$")
    ax2.set_ylabel(r"$dV/dT$")
    ax2.set_title(r"Temperature PDF, $t \geq 10\,t_{\rm sh}$")
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(FIG_DIR, "diffusivity_temperature_pdf.png")
    fig2.savefig(out2, dpi=200)
    print("saved", out2)


if __name__ == "__main__":
    main()
