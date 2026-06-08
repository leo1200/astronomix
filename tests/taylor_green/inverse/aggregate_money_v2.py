"""
Aggregate the parallel recovery ensemble (data/ensemble/money_T*_m*_s*.npz)
into the rigorous money plot:

  (left)  median large-scale recovery error vs horizon, with inter-quartile
          band over the truth-seed ensemble, per shooting split m;
  (right) success rate (fraction of seeds with recovery error < TAU) vs
          horizon -- the most honest summary for a multimodal problem.

Run: python aggregate_money_v2.py
"""

import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data" / "ensemble"
OUT = Path(__file__).parent / "figures"
TAU = 0.2   # success threshold on the large-scale recovery error
AGG_OPT = os.environ.get("AGG_OPT", "lbfgs").lower()   # which optimizer to aggregate


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("money_T*_m*_s*.npz"))
    if not files:
        print(f"no ensemble npz in {DATA}")
        return

    # (m, T) -> list of final errors over seeds (for the selected optimizer)
    by = defaultdict(list)
    for f in files:
        d = np.load(f)
        opt = str(d["opt"]) if "opt" in d.files else "adam"
        if opt != AGG_OPT:
            continue
        by[(int(d["m"]), float(d["t_obs"]))].append(float(d["final_err"]))
    if not by:
        print(f"no npz with opt={AGG_OPT} (set AGG_OPT=adam to aggregate Adam runs)")
        return

    ms = sorted({k[0] for k in by})
    style = {1: ("C3", "single shooting (m=1)"), 4: ("C0", "multiple shooting (m=4)")}

    fig, (axm, axs) = plt.subplots(1, 2, figsize=(14, 6))
    for m in ms:
        Ts = sorted(T for (mm, T) in by if mm == m)
        med, q25, q75, succ, nseed = [], [], [], [], []
        for T in Ts:
            errs = np.array(by[(m, T)])
            med.append(np.median(errs))
            q25.append(np.percentile(errs, 25))
            q75.append(np.percentile(errs, 75))
            succ.append(np.mean(errs < TAU))
            nseed.append(len(errs))
        c, lab = style.get(m, (f"C{m}", f"m={m}"))
        axm.plot(Ts, med, "o-", color=c, lw=1.8, label=f"{lab}  (n={min(nseed)}-{max(nseed)})")
        axm.fill_between(Ts, q25, q75, color=c, alpha=0.2)
        axs.plot(Ts, succ, "o-", color=c, lw=1.8, label=lab)

    axm.set_xlabel(r"observation horizon  $T_{\rm obs}/t_c$")
    axm.set_ylabel("large-scale recovery error")
    axm.set_title("Median recovery error (IQR band over truth ensemble)")
    axm.grid(alpha=0.3); axm.legend(fontsize=9)

    axs.set_xlabel(r"observation horizon  $T_{\rm obs}/t_c$")
    axs.set_ylabel(f"success rate  (error < {TAU})")
    axs.set_ylim(-0.03, 1.03)
    axs.set_title("Recovery success rate over the truth ensemble")
    axs.grid(alpha=0.3); axs.legend(fontsize=9)

    fig.suptitle(f"TGV IC recovery: single vs multiple shooting (ensemble, {AGG_OPT})",
                 fontsize=13)
    fig.tight_layout()
    out = OUT / f"money_plot_v2_{AGG_OPT}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"-> {out}")
    for m in ms:
        Ts = sorted(T for (mm, T) in by if mm == m)
        print(f"m={m}: " + ", ".join(
            f"T={T}: med={np.median(by[(m,T)]):.3f} (n={len(by[(m,T)])})" for T in Ts))


if __name__ == "__main__":
    main()
