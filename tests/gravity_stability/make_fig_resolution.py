"""Resolution study: overlay the FINAL-state structure diagnostics across runs.

Compares the converged density PDF (the high-density collapse tail), the
dense-gas mass fraction, and the rho-B amplification relation for the
structure-formation runs at different N. Single-row (final_only) and
time-resolved npz files both work (the last PDF row is used).

Usage: python make_fig_resolution.py N128_G6 N256_G6 N512_G6
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
    tags = sys.argv[1:] or ["N128_G6", "N256_G6", "N512_G6"]
    runs = []
    for tg in tags:
        f = os.path.join(DATA, f"structure_{tg}.npz")
        if os.path.exists(f):
            runs.append((tg, np.load(f)))
        else:
            print(f"(missing {f})")
    if not runs:
        return

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.8, len(runs)))
    print(f"\n{'tag':>12} {'N':>5} {'t/tc':>6} {'rho_max':>9} "
          f"{'f(>10)':>8} {'f(>30)':>8} {'f(>100)':>8} {'EB/EB0':>7}")
    for (tg, d), col in zip(runs, colors):
        N = int(d["N"])
        centers = 0.5 * (d["pdf_bins"][1:] + d["pdf_bins"][:-1])
        ax[0].plot(centers, d["pdf_vol"][-1], color=col, lw=2, label=f"N={N}")
        ax[1].plot(centers, d["pdf_mass"][-1], color=col, lw=2, label=f"N={N}")
        m = np.isfinite(d["rb_meanB"])
        ax[2].loglog(d["rb_centers"][m], d["rb_meanB"][m], "o-", ms=3,
                     color=col, label=f"N={N}")
        ttc = float(d["slice_t_over_tc"]) if "slice_t_over_tc" in d else float("nan")
        mf = d["massfrac"][-1]
        eb = float(d["EB_t"][-1] / d["EB0"]) if "EB0" in d else float("nan")
        print(f"{tg:>12} {N:>5} {ttc:>6.2f} {np.nanmax(d['rhomax_t']):>9.0f} "
              f"{mf[2]:>8.3f} {mf[3]:>8.3f} {mf[4]:>8.3f} {eb:>7.2f}")

    ax[0].set_yscale("log"); ax[0].set_xlabel(r"$\log_{10}(\rho/\rho_0)$")
    ax[0].set_ylabel("volume PDF"); ax[0].set_title("final density PDF (volume)")
    ax[0].set_ylim(1e-4, 5); ax[0].legend()
    ax[1].set_yscale("log"); ax[1].set_xlabel(r"$\log_{10}(\rho/\rho_0)$")
    ax[1].set_ylabel("mass PDF"); ax[1].set_title("final density PDF (mass)")
    ax[1].set_ylim(1e-4, 5); ax[1].legend()
    ax[2].set_xlabel(r"$\rho/\rho_0$"); ax[2].set_ylabel(r"$\langle |B| \rangle$")
    ax[2].set_title(r"$\rho$-B amplification"); ax[2].legend()

    fig.suptitle("MHD turbulence + self-gravity: resolution study (PP-flux limiter)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "structure_resolution.png")
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
