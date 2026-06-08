"""
Aggregate the mu-tuning sweep (data/mu_tuning/rec_*.npz) into a tuning curve:
multiple-shooting recovery error (and data misfit) vs the consistency-defect
penalty mu, with the single-shooting baseline (mu-independent) as a reference.
Picks the mu that minimizes the median recovery error.

Run: python aggregate_mu_tuning.py
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

DATA = Path(__file__).parent / "data" / "mu_tuning"
OUT = Path(__file__).parent / "figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("rec_*.npz"))
    if not files:
        print(f"no rec_*.npz in {DATA}")
        return

    # (m, T, mu) -> list of (err, data) over seeds
    g = defaultdict(list)
    for f in files:
        d = np.load(f)
        g[(int(d["m"]), float(d["t_obs"]), float(d["mu"]))].append(
            (float(d["final_err"]), float(d["final_data"])))

    Ts = sorted({k[1] for k in g})
    for T in Ts:
        # single shooting baseline (m=1, mu-independent)
        ss = [v for k, vs in g.items() if k[0] == 1 and k[1] == T for v in vs]
        ss_err = np.median([e for e, _ in ss]) if ss else None
        # multiple shooting vs mu
        mus = sorted({k[2] for k in g if k[0] != 1 and k[1] == T})
        ms_for_m = defaultdict(lambda: ([], [], []))  # m -> (mus, med_err, med_data)
        ms_ms = sorted({k[0] for k in g if k[0] != 1 and k[1] == T})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        best = {}
        for m in ms_ms:
            xs, me, md = [], [], []
            for mu in mus:
                vs = g.get((m, T, mu), [])
                if not vs:
                    continue
                xs.append(mu)
                me.append(np.median([e for e, _ in vs]))
                md.append(np.median([dd for _, dd in vs]))
            ax1.semilogx(xs, me, "o-", label=f"multiple shooting (m={m})")
            ax2.loglog(xs, md, "o-", label=f"multiple shooting (m={m})")
            if me:
                bi = int(np.argmin(me))
                best[m] = (xs[bi], me[bi])
        if ss_err is not None:
            ax1.axhline(ss_err, color="C3", ls="--", label="single shooting (m=1)")
            ss_data = np.median([dd for _, dd in ss])
            ax2.axhline(ss_data, color="C3", ls="--", label="single shooting (m=1)")
        ax1.set_xlabel(r"consistency penalty  $\mu$")
        ax1.set_ylabel("large-scale recovery error")
        ax1.set_title(f"MS recovery error vs $\\mu$  (T={T})")
        ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=9)
        ax2.set_xlabel(r"consistency penalty  $\mu$")
        ax2.set_ylabel("data misfit at recovered IC")
        ax2.set_title(f"MS data misfit vs $\\mu$  (T={T})")
        ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=9)
        fig.suptitle(f"Multiple-shooting penalty tuning, T_obs={T} t_c", fontsize=13)
        fig.tight_layout()
        out = OUT / f"mu_tuning_T{T}.png"
        fig.savefig(out, dpi=200); plt.close(fig)
        print(f"-> {out}")
        print(f"  T={T}: SS baseline err={ss_err}")
        for m, (mu, e) in best.items():
            print(f"  T={T}: best mu for m={m}: mu={mu} -> err={e:.3f}")


if __name__ == "__main__":
    main()
