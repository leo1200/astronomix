"""
Aggregate the SS-vs-MS columnar-recovery window sweep into the money plot:
final recovery error and gradient-norm vs window length T (in eddy times),
single shooting (m=1) vs multiple shooting (m=4).  Reads the per-window
rot_ss_vs_ms_T*.npz produced by ss_vs_ms_recovery.py.
"""
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"


def main():
    rows = []
    for f in sorted(DATA.glob("rot_ss_vs_ms_T*.npz")):
        m = re.search(r"_T([0-9.]+)\.npz", f.name)
        if not m:
            continue
        T = float(m.group(1))
        d = np.load(f)
        ms = sorted(int(k[5:]) for k in d.files if k.startswith("err_m"))
        rec = {mm: float(d[f"err_m{mm}"][-1]) for mm in ms}
        # robust SS gradient amplification: max gradient norm over the run
        gmax = {mm: float(np.max(d[f"gnorm_m{mm}"])) for mm in ms}
        rows.append((T, rec, gmax, ms))
        print(f"T={T:>5} t_e:  " + "  ".join(f"m{mm} rec={rec[mm]:.3f}" for mm in ms))
    rows.sort()
    if not rows:
        print("no data yet"); return
    Ts = np.array([r[0] for r in rows])
    all_m = sorted(set().union(*[set(r[3]) for r in rows]))
    tau_L_te = 8.9   # from the long-window mechanism fit

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for mm, c in zip(all_m, ("C3", "C0", "C2", "C1")):
        e = [r[1].get(mm, np.nan) for r in rows]
        g = [r[2].get(mm, np.nan) for r in rows]
        lab = "single shooting (m=1)" if mm == 1 else f"multiple shooting m={mm}"
        ax[0].plot(Ts, e, "-o", color=c, label=lab)
        ax[1].semilogy(Ts, g, "-o", color=c, label=lab)
    for a in ax:
        a.axvline(tau_L_te, ls="--", color="k", alpha=0.5, label=r"$\tau_L\approx 8.9\,t_e$")
        a.set_xlabel(r"window $T$ / eddy time"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    ax[0].set(ylabel="final columnar recovery error", title="Recovery vs window (lower=better)")
    ax[1].set(ylabel="max |grad| over optimisation", title="Gradient amplification (SS blow-up)")
    fig.suptitle("Rotating-turbulence columnar recovery: SS vs MS money plot")
    fig.tight_layout(); fig.savefig(OUT / "rot_money_plot.png", dpi=170)
    print(f"\nFigure -> {OUT / 'rot_money_plot.png'}")


if __name__ == "__main__":
    main()
