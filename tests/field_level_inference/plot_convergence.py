#!/usr/bin/env python
"""128³ money plot: terminal logo-matching loss (128³ projection MSE) vs optimisation
wall-time for naive / k-windowing / prolongation-free-multigrid (64³->128³).
Loads the three per-method npz files (parallel runs)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FILES = {
    "naive":       ("data/bench_money128_naive.npz", "naive  (all 6.3M modes, 128³)",      "#888888"),
    "k_windowing": ("data/bench_money128_kwin.npz",  "k-windowing  (8→16→32→64, 128³)","#c0392b"),
    "multigrid":   ("data/bench_money128_mg.npz",    "multigrid + k-windowing  (64³→128³)","#1f8a4c"),
}

sched = None; coarse_cut = 16
fig, ax = plt.subplots(figsize=(8.4, 5.6))
for key, (fn, lab, col) in FILES.items():
    if not os.path.exists(fn):
        print(f"skip missing {fn}"); continue
    d = np.load(fn)
    if key in d and len(d[key]):
        r = d[key]
        ax.plot(r[:, 0] / 60.0, r[:, 1], "-o", color=col, ms=4, lw=2, label=lab)
        if sched is None:
            sched = d["sched"]; coarse_cut = int(d["coarse_cut"])

# multigrid 64³->128³ transition (end of last coarse band)
if sched is not None and os.path.exists(FILES["multigrid"][0]):
    dm = np.load(FILES["multigrid"][0])
    if "multigrid" in dm and len(dm["multigrid"]):
        coarse_steps = sum(int(s) for (kc, s) in sched if kc <= coarse_cut)
        # logged points during coarse stage == coarse_steps (diag-every=1); transition after them
        r = dm["multigrid"]
        if len(r) > coarse_steps and coarse_steps > 0:
            t_tr = r[coarse_steps - 1, 0] / 60.0
            ax.axvline(t_tr, ls="--", lw=1, color="#1f8a4c", alpha=0.6)
            ax.text(t_tr, ax.get_ylim()[1], " 64³→128³", color="#1f8a4c", va="top", fontsize=9)

ax.set_xlabel("cumulative optimisation wall-time  [min]")
ax.set_ylabel("terminal loss   (128³ logo-projection MSE)")
ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
ax.legend(frameon=False, fontsize=10)
ax.set_title("Logo reconstruction at 128³: convergence vs compute", fontsize=12)
fig.tight_layout()
fig.savefig("figures/money128.png", dpi=140)
print("wrote money128.png")
for key, (fn, lab, _) in FILES.items():
    if os.path.exists(fn):
        d = np.load(fn)
        if key in d and len(d[key]):
            r = d[key]
            print(f"  {key:12s} final lossF={r[-1,1]:.3e} @ {r[-1,0]/60:.1f}min ({len(r)} pts)")
