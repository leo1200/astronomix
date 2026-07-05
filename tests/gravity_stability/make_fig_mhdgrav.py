"""Plot the MHD-turbulence + self-gravity stability comparison.

Reads the npz files written by mhd_grav_turb.py (data_mhdgrav/mhdgrav_<tag>.npz)
and produces a 3-panel figure: max density, min pressure, and Mach number vs
time for each (scheme/blend) tag, with a marker at the first non-finite snapshot.
"""

import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
# the runner writes to data_mhdgrav relative to its cwd (the worktree root);
# fall back to that if not found next to this script
DATA = os.path.join(HERE, "data_mhdgrav")
if not os.path.isdir(DATA):
    DATA = os.path.join(os.getcwd(), "data_mhdgrav")
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def main():
    tags = sys.argv[1:]
    files = ([os.path.join(DATA, f"mhdgrav_{t}.npz") for t in tags]
             if tags else sorted(glob.glob(os.path.join(DATA, "mhdgrav_*.npz"))))
    if not files:
        print("no data files found")
        return

    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 4.2))
    for f in files:
        d = np.load(f, allow_pickle=True)
        tag = str(d["tag"])
        t = d["t_over_tc"]
        fb = int(d["first_bad"])
        adapt = float(d["adapt"]) if "adapt" in d else 0.0
        lbl = f"{tag} (G={float(d['G']):g}, adapt={adapt:g})"
        line, = a0.plot(t, d["rhomax_t"], label=lbl)
        a1.plot(t, d["pmin_t"], color=line.get_color())
        a2.plot(t, d["Ms_t"], color=line.get_color())
        if fb >= 0:
            for ax, arr in [(a0, d["rhomax_t"]), (a1, d["pmin_t"]), (a2, d["Ms_t"])]:
                ax.axvline(t[fb], color=line.get_color(), ls=":", alpha=0.6)

    a0.set_yscale("log"); a0.set_ylabel(r"$\rho_{max}$")
    a1.set_ylabel(r"$p_{min}$")
    a2.set_ylabel(r"$M_s$")
    for ax in (a0, a1, a2):
        ax.set_xlabel(r"$t / t_{cross}$"); ax.grid(alpha=0.3)
    a0.legend(fontsize=8, loc="best")
    a1.axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "mhd_selfgravity_stability.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # text summary
    print(f"\n{'tag':>28} {'G':>5} {'adapt':>6} {'first_bad@t/tc':>14} "
          f"{'rho_max':>10} {'p_min':>11}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        fb = int(d["first_bad"])
        t = d["t_over_tc"]
        adapt = float(d["adapt"]) if "adapt" in d else 0.0
        fbt = f"{t[fb]:.2f}" if fb >= 0 else "FINITE"
        print(f"{str(d['tag']):>28} {float(d['G']):>5g} {adapt:>6g} "
              f"{fbt:>14} {np.nanmax(d['rhomax_t']):>10.2f} "
              f"{np.nanmin(d['pmin_t']):>11.3e}")


if __name__ == "__main__":
    main()
