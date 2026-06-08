"""Aggregate the 100-run MS-vs-SS campaign: recovery fraction vs M (single, M=4/8/16)
at T=80, with binomial 68% CIs. The robust statistic (individual inits are float-
path-sensitive in the mode-2 regime). Money table + figure."""
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"
THR = 0.1


def load(pat):
    return np.array([float(np.load(f, allow_pickle=True)["ic_err"]) for f in glob.glob(str(D / pat))])


def wilson(k, n):                       # Wilson 68% interval (z=1)
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n; z = 1.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(c - h, 0), min(c + h, 1)


methods = [("single", "camp_single_T80_i*.npz", "single shooting"),
           ("M=4", "camp_msgn_M4_T80_i*.npz", "GN-MS M=4"),
           ("M=8", "camp_msgn_M8_T80_i*.npz", "GN-MS M=8"),
           ("M=16", "camp_msgn_M16_T80_i*.npz", "GN-MS M=16")]
print(f"{'method':8} {'n':>4} {'recovered':>9} {'frac':>6} {'68% CI':>13} {'best':>6} {'median':>7}")
rows = []
for key, pat, lab in methods:
    e = load(pat)
    n = len(e); k = int((e < THR).sum())
    p, lo, hi = wilson(k, n)
    best = e.min() if n else np.nan; med = np.median(e) if n else np.nan
    print(f"{key:8} {n:>4} {k:>9} {p:>6.2f} [{lo:.2f},{hi:.2f}]  {best:>6.3f} {med:>7.3f}")
    rows.append((key, n, k, p, lo, hi, best, med))

fig, ax = plt.subplots(figsize=(7.5, 5))
xs = np.arange(len(rows))
ps = [r[3] for r in rows]
err = np.array([[r[3] - r[4] for r in rows], [r[5] - r[3] for r in rows]])
ax.errorbar(xs, ps, yerr=err, fmt="o", ms=9, capsize=5, color="C0", lw=2)
for i, r in enumerate(rows):
    ax.annotate(f"{r[2]}/{r[1]}", (xs[i], ps[i]), textcoords="offset points",
                xytext=(8, 6), fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel(f"recovery fraction (ic_err < {THR})"); ax.set_ylim(-0.02, 1.0)
ax.set_xlabel("method (single shooting vs constrained GN multiple shooting)")
ax.set_title(f"Recovery fraction at T=80 t_g (genuine mode-2), kx2-6\n"
             f"{rows[0][1]}+ cold inits each, 68% binomial CI")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(OUT / "fig_campaign_fraction.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_campaign_fraction.png'}")
