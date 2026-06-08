"""
Aggregate the per-horizon single- vs multiple-shooting recovery runs into
the "money plot" of init_optim_theory.md Section 8: large-scale recovery
error vs observation horizon T_obs / t_c, one curve per shooting split m.

Reads all ss_vs_ms_Tobs*.npz produced by ss_vs_ms_recovery.py in DATA_DIR
and plots, for each m, the final (converged) large-scale recovery error
against the horizon. Prediction: single shooting (m=1) cliffs within ~1
turnover of the predictability horizon; multiple shooting (m>1) holds the
large scales further out.
"""

import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "figures"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("ss_vs_ms_Tobs*.npz"))
    if not files:
        print("No ss_vs_ms_Tobs*.npz files found.")
        return

    # collect final recovery error per (m, T_obs)
    curves = {}  # m -> list of (T_obs, final_err)
    for f in files:
        T = float(re.search(r"Tobs([0-9.]+)\.npz", f.name).group(1))
        d = np.load(f)
        for key in d.files:
            mobj = re.match(r"err_m(\d+)$", key)
            if mobj:
                m = int(mobj.group(1))
                final_err = float(d[key][-1])
                curves.setdefault(m, []).append((T, final_err))

    fig, ax = plt.subplots(figsize=(8, 6))
    for m in sorted(curves):
        pts = sorted(curves[m])
        Ts = [p[0] for p in pts]
        errs = [p[1] for p in pts]
        label = "single shooting (m=1)" if m == 1 else f"multiple shooting (m={m})"
        ax.plot(Ts, errs, "-o", label=label)
    ax.set_xlabel("observation horizon  T_obs / t_c")
    ax.set_ylabel("final large-scale recovery error")
    ax.set_title("SS vs MS recovery of the TGV initial condition")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUTPUT_DIR / "money_plot.png"
    fig.savefig(out, dpi=200)
    print(f"Money plot -> {out}")
    for m in sorted(curves):
        print(f"  m={m}: " + ", ".join(f"T={T}:{e:.3f}" for T, e in sorted(curves[m])))


if __name__ == "__main__":
    main()
