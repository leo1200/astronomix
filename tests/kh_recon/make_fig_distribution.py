"""Error-distribution figure for the 400-trial T=80 mode-2 campaign.

Shows the full ic_err distribution of all four reconstruction approaches
(single shooting, constrained GN multiple shooting M=4/8/16) on one log-x
axis as a raincloud per method: jittered points (the raw 100 trials) below
a Gaussian-KDE density above the baseline. Medians are marked with vertical
lines; the recovery / success barrier (ic_err < 0.1) is a single vertical
line across all rows. Complements the recovery-fraction figure
(make_fig_campaign.py) by showing the whole distribution, not just the
fraction below threshold."""
from pathlib import Path
import glob
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"
THR = 0.1  # recovery / success barrier (ic_err < THR)

# bottom -> top; "better" methods rise up the figure
METHODS = [
    ("single", "camp_single_T80_i*.npz", "single shooting", "#777777"),
    ("M=4", "camp_msgn_M4_T80_i*.npz", "GN-MS M=4", "#4C72B0"),
    ("M=8", "camp_msgn_M8_T80_i*.npz", "GN-MS M=8", "#55A868"),
    ("M=16", "camp_msgn_M16_T80_i*.npz", "GN-MS M=16", "#C44E52"),
]


def load(pat):
    return np.array([float(np.load(f, allow_pickle=True)["ic_err"])
                     for f in glob.glob(str(D / pat))])


rng = np.random.default_rng(0)
data = [(key, lab, col, load(pat)) for key, pat, lab, col in METHODS]

# common log-x grid for the KDEs
allv = np.concatenate([e for *_, e in data])
lo, hi = np.log10(allv.min()) - 0.15, np.log10(allv.max()) + 0.15
xgrid = np.logspace(lo, hi, 400)
lxgrid = np.log10(xgrid)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
ROW_H = 1.0          # vertical spacing between methods
DENS_H = 0.72        # max height of a KDE bump
JIT = 0.20           # vertical extent of the point strip below the baseline

for i, (key, lab, col, e) in enumerate(data):
    base = i * ROW_H
    n = len(e)
    frac = float((e < THR).mean())
    med = float(np.median(e))

    # KDE in log10 space (errors span orders of magnitude) -> draw above baseline
    kde = gaussian_kde(np.log10(e))
    dens = kde(lxgrid)
    dens = dens / dens.max() * DENS_H
    ax.fill_between(xgrid, base, base + dens, color=col, alpha=0.35, lw=0)
    ax.plot(xgrid, base + dens, color=col, lw=1.5)

    # raw points, jittered below the baseline
    yj = base - rng.uniform(0.04, JIT, size=n)
    ax.scatter(e, yj, s=14, color=col, alpha=0.55, edgecolors="none", zorder=3)

    # median marker (vertical line spanning the row)
    ax.plot([med, med], [base - JIT - 0.03, base + DENS_H], color=col,
            lw=2.2, ls="-", zorder=4)
    ax.annotate(f"median {med:.2f}", (med, base + DENS_H + 0.02),
                color=col, fontsize=8.5, ha="left", va="bottom")

    # row label + recovery fraction
    ax.text(xgrid[0], base + 0.18, f"{lab}", fontsize=11, fontweight="bold",
            color=col, ha="left", va="bottom")
    ax.text(xgrid[0], base - 0.10,
            f"recovered {int(round(frac*n))}/{n}  ({frac:.0%})",
            fontsize=8.5, color=col, ha="left", va="top")

# success barrier across all rows
ax.axvline(THR, color="k", ls="--", lw=1.6, zorder=5)
ax.annotate(f"success barrier\nic_err < {THR}", (THR, len(data) * ROW_H - 0.18),
            xytext=(-6, 0), textcoords="offset points", ha="right", va="top",
            fontsize=9.5, fontweight="bold")

ax.set_xscale("log")
ax.set_xlim(xgrid[0], xgrid[-1])
ax.set_ylim(-JIT - 0.25, (len(data) - 1) * ROW_H + DENS_H + 0.45)
ax.set_yticks([])
ax.set_xlabel("reconstruction error  ic_err  (relative IC error, log scale)")
ax.set_title("KH mode-2 reconstruction error distribution at T=80 $t_g$ (kx2-6)\n"
             "100 cold inits per method; single shooting vs constrained GN multiple shooting")
ax.grid(True, axis="x", which="both", alpha=0.25)
for s in ("left", "right", "top"):
    ax.spines[s].set_visible(False)

fig.tight_layout()
out = OUT / "fig_campaign_distribution.png"
fig.savefig(out, dpi=170)
fig.savefig(OUT / "fig_campaign_distribution.svg")
plt.close(fig)
print(f"-> {out}")
for key, lab, col, e in data:
    print(f"{key:6} n={len(e):3d} median={np.median(e):.3f} "
          f"frac<{THR}={(e<THR).mean():.2f} min={e.min():.4f}")
