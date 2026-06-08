"""
Aggregate a horizon sweep (rec_*.npz in AGG_DIR, default data/horizon_check)
into recovery error and success rate vs observation horizon, single shooting
vs (tuned) multiple shooting. Groups over truth seeds.

Run: AGG_DIR=data/horizon_check python aggregate_horizon.py
"""

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / os.environ.get("AGG_DIR", "data/horizon_check")
OUT = HERE / "figures"
TAU = float(os.environ.get("AGG_TAU", 0.2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA.glob("rec_*.npz"))
    if not files:
        print(f"no rec_*.npz in {DATA}"); return

    by = defaultdict(list)        # (m, T, mu) -> [err...]
    meta = {}
    for f in files:
        d = np.load(f)
        by[(int(d["m"]), float(d["t_obs"]), float(d["mu"]))].append(float(d["final_err"]))
        meta["kcut"] = float(d.get("k_cut", 4.0)); meta["N"] = int(d["n_cells"])

    ms = sorted({k[0] for k in by})
    style = {1: ("C3", "single shooting (m=1)"), 4: ("C0", "multiple shooting (m=4, best mu)")}
    fig, (axm, axs) = plt.subplots(1, 2, figsize=(14, 6))
    best_mu = {}
    for m in ms:
        Ts = sorted({T for (mm, T, _) in by if mm == m})
        med, q25, q75, succ = [], [], [], []
        for T in Ts:
            mus = sorted({mu for (mm, TT, mu) in by if mm == m and TT == T})
            # single shooting is mu-independent; multiple shooting -> pick best mu
            errs_by_mu = {mu: np.array(by[(m, T, mu)]) for mu in mus}
            if m == 1:
                errs = np.concatenate(list(errs_by_mu.values()))
            else:
                bmu = min(mus, key=lambda mu: np.median(errs_by_mu[mu]))
                best_mu[(m, T)] = bmu
                errs = errs_by_mu[bmu]
            med.append(np.median(errs)); q25.append(np.percentile(errs, 25))
            q75.append(np.percentile(errs, 75)); succ.append(np.mean(errs < TAU))
        c, lab = style.get(m, (f"C{m}", f"m={m}"))
        axm.plot(Ts, med, "o-", color=c, lw=1.8, label=lab)
        axm.fill_between(Ts, q25, q75, color=c, alpha=0.2)
        axs.plot(Ts, succ, "o-", color=c, lw=1.8, label=lab)

    sub = f"N={meta.get('N')}, K_cut={meta.get('kcut')}, MS=best-mu"
    axm.set_xlabel(r"observation horizon  $T_{\rm obs}/t_c$")
    axm.set_ylabel("large-scale recovery error")
    axm.set_title(f"Median recovery error (IQR band)\n{sub}")
    axm.grid(alpha=0.3); axm.legend(fontsize=9)
    axs.set_xlabel(r"observation horizon  $T_{\rm obs}/t_c$")
    axs.set_ylabel(f"success rate (err < {TAU})")
    axs.set_ylim(-0.03, 1.03)
    axs.set_title(f"Recovery success rate\n{sub}")
    axs.grid(alpha=0.3); axs.legend(fontsize=9)
    fig.suptitle("TGV IC recovery vs horizon: single vs tuned multiple shooting", fontsize=13)
    fig.tight_layout()
    out = OUT / f"horizon_check_k{meta.get('kcut')}.png"
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f"-> {out}")
    for m in ms:
        Ts = sorted({T for (mm, T, _) in by if mm == m})
        parts = []
        for T in Ts:
            if m == 1:
                e = np.median(np.concatenate([by[k] for k in by if k[0] == 1 and k[1] == T]))
                parts.append(f"T={T}:{e:.3f}")
            else:
                bmu = best_mu[(m, T)]
                parts.append(f"T={T}:{np.median(by[(m,T,bmu)]):.3f}(mu*={bmu:g})")
        print(f"m={m}: " + ", ".join(parts))


if __name__ == "__main__":
    main()
