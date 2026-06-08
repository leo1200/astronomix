"""Figure: the OPTIMIZATION frontier. Single-shooting low-k recovery error vs
horizon T (band-limited control KCTRL, SNR=100 on the perturbation). Recovery
works at short T (lowk_err -> 0, early-stop fires at the discrepancy floor) and
collapses past a horizon T* -- the optimization frontier, strictly tighter than
the information frontier. Overlays soft-MS at the transition if present."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def load(pattern):
    rows = []
    for f in sorted(glob.glob(str(D / pattern))):
        m = re.search(r"_T(\d+)_", f)
        if not m:
            continue
        d = np.load(f, allow_pickle=True)
        rows.append((int(m.group(1)), float(d["lowk_err"]), float(d["full_err"]),
                     float(d["best_lowk"]), int(d["stopped_at"]), float(d["runtime"])))
    rows.sort()
    return np.array(rows) if rows else np.empty((0, 6))


ss = load("hz_single_T*_k6.npz")
print("single horizon scan:")
for T, lk, fe, best, stop, rt in ss:
    print(f"  T={T:4.0f} t_g: lowk_err={lk:.3f} full_err={fe:.3f} best={best:.3f} "
          f"stop@{int(stop)} {rt:.0f}s")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
if len(ss):
    ax[0].plot(ss[:, 0], ss[:, 1], "o-", color="C3", label="single: lowk_err (at stop)")
    ax[0].plot(ss[:, 0], ss[:, 3], "s--", color="C3", alpha=0.5, label="single: best lowk (oracle)")
    ax[0].axhline(1.0, color="k", lw=0.5, ls=":")
ax[0].set_xlabel("horizon T / t_g"); ax[0].set_ylabel("low-k recovery error")
ax[0].set_title("optimization frontier: single-shooting recovery vs horizon")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# overlay soft-MS at transition horizon if present
for f in sorted(glob.glob(str(D / "hzms_*_T*_k6.npz"))):
    d = np.load(f, allow_pickle=True)
    m = re.search(r"hzms_(\w+?)_M(\d+)_mu([\d.]+)_T(\d+)", f)
    if not m:
        continue
    ax[0].plot(int(m.group(4)), float(d["lowk_err"]), "D", ms=9,
               label=f"soft M={m.group(2)} mu={m.group(3)} T={m.group(4)}")
ax[0].legend(fontsize=7)

if len(ss):
    ax[1].plot(ss[:, 0], ss[:, 4], "o-", label="stop iteration")
    ax[1].set_xlabel("horizon T / t_g"); ax[1].set_ylabel("early-stop iteration")
    ax[1].set_title("convergence: discrepancy-floor stop (300=never)")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig_horizon_frontier.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_horizon_frontier.png'}")
