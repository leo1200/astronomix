"""Figure for the MHD-turbulence + self-gravity structure-formation run.

6 panels: (a) density-PDF evolution (volume-weighted), (b) mass-weighted PDF,
(c) rho_max & dense-mass-fraction vs time, (d) rho-B amplification relation,
(e) density mid-plane slice, (f) column density (z-projection).
Usage: python make_fig_structure.py <tag>
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data_structure")
if not os.path.isdir(DATA):
    DATA = os.path.join(os.getcwd(), "data_structure")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "N128_G6"
    d = np.load(os.path.join(DATA, f"structure_{tag}.npz"))
    t = d["t_over_tc"]
    centers = 0.5 * (d["pdf_bins"][1:] + d["pdf_bins"][:-1])

    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (a) volume-weighted density PDF at several times
    idxs = np.linspace(0, len(t) - 1, 6).astype(int)
    cmap = plt.cm.viridis
    for k, i in enumerate(idxs):
        ax[0, 0].plot(centers, d["pdf_vol"][i], color=cmap(k / 5),
                      label=f"t/tc={t[i]:.1f}")
    ax[0, 0].set_yscale("log"); ax[0, 0].set_xlabel(r"$\log_{10}(\rho/\rho_0)$")
    ax[0, 0].set_ylabel("volume PDF"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title("density PDF (volume-weighted)")
    ax[0, 0].set_ylim(1e-4, 5)

    # (b) mass-weighted PDF (highlights the collapsing tail)
    for k, i in enumerate(idxs):
        ax[0, 1].plot(centers, d["pdf_mass"][i], color=cmap(k / 5))
    ax[0, 1].set_yscale("log"); ax[0, 1].set_xlabel(r"$\log_{10}(\rho/\rho_0)$")
    ax[0, 1].set_ylabel("mass PDF"); ax[0, 1].set_title("density PDF (mass-weighted)")
    ax[0, 1].set_ylim(1e-4, 5)

    # (c) rho_max and dense-mass fractions vs time
    axc = ax[0, 2]
    axc.semilogy(t, d["rhomax_t"], "k-", lw=2, label=r"$\rho_{max}$")
    axc.set_xlabel(r"$t/t_{cross}$"); axc.set_ylabel(r"$\rho_{max}/\rho_0$")
    axc2 = axc.twinx()
    thr = d["rho_thresholds"]
    for j in range(len(thr)):
        axc2.plot(t, d["massfrac"][:, j], "--", alpha=0.8,
                  label=fr"$\rho>{thr[j]:.0f}$")
    axc2.set_ylabel("dense-gas mass fraction")
    axc.set_title("collapse: peak density & dense-mass growth")
    axc.legend(loc="upper left", fontsize=8); axc2.legend(loc="lower right", fontsize=7)

    # (d) rho-B relation
    axd = ax[1, 0]
    m = np.isfinite(d["rb_meanB"])
    axd.loglog(d["rb_centers"][m], d["rb_meanB"][m], "o-", ms=3)
    rr = d["rb_centers"][m]
    if len(rr) > 3:
        for kappa, c in [(0.5, "r"), (2.0 / 3.0, "g")]:
            ref = d["rb_meanB"][m][len(rr) // 3] * (rr / rr[len(rr) // 3]) ** kappa
            axd.loglog(rr, ref, c + "--", alpha=0.6, label=fr"$B\propto\rho^{{{kappa:.2f}}}$")
    axd.set_xlabel(r"$\rho/\rho_0$"); axd.set_ylabel(r"$\langle |B| \rangle$")
    axd.set_title("magnetic amplification"); axd.legend(fontsize=8)

    # (e) density slice
    rs = d["rho_slice"]
    im = ax[1, 1].imshow(rs.T, origin="lower", norm=mcolors.LogNorm(
        vmin=max(rs.min(), 1e-2), vmax=rs.max()), cmap="inferno")
    ax[1, 1].set_title(fr"$\rho$ slice (t/tc={float(d['slice_t_over_tc']):.1f})")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)

    # (f) column density (observable)
    cd = d["coldens"]
    im2 = ax[1, 2].imshow(cd.T, origin="lower", norm=mcolors.LogNorm(
        vmin=max(cd.min(), 1e-2), vmax=cd.max()), cmap="bone")
    ax[1, 2].set_title("column density (z-projection)")
    plt.colorbar(im2, ax=ax[1, 2], fraction=0.046)

    fig.suptitle(
        f"MHD turbulence + self-gravity structure formation  "
        f"(N={int(d['N'])}, M_turb~{float(d['mturb']):.0f}, beta={float(d['beta']):.0f}, "
        f"G={float(d['G']):.0f}, lam_J/L={float(d['lam_J']):.2f}, PP-flux limiter)",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG, f"structure_{tag}.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # summary
    print(f"rho_max: {d['rhomax_t'][0]:.1f} -> {np.nanmax(d['rhomax_t']):.0f}")
    print(f"final dense-mass frac rho>10: {d['massfrac'][-1,2]:.3f}, "
          f"rho>100: {d['massfrac'][-1,4]:.3f}")
    print(f"E_B: {d['EB_t'][0]:.4g} -> {d['EB_t'][-1]:.4g} (EB0={float(d['EB0']):.4g})")


if __name__ == "__main__":
    main()
