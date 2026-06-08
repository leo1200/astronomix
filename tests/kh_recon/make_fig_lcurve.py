"""Tikhonov L-curve: low-k recovery error vs regularization alpha, single shooting
at short horizons. Recovery (lowk_err<<1) at the optimal alpha would be the first
working KH reconstruction; the U-shape (null-space blowup at small alpha, over-
shrink at large alpha) is the textbook ill-posed-inverse signature."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def load_T(T):
    rows = []
    for f in sorted(glob.glob(str(D / f"alpha_T{T}_a*.npz"))):
        m = re.search(r"_a([0-9.eE+-]+)\.npz", f)
        if not m:
            continue
        a = float(m.group(1)); d = np.load(f, allow_pickle=True)
        rows.append((a, float(d["lowk_err"]), float(d["full_err"]),
                     float(d["best_lowk"]), float(d["final_loss"]), int(d["stopped_at"])))
    rows.sort()
    return rows


fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for T, c in [(5, "C0"), (10, "C1")]:
    rows = load_T(T)
    if not rows:
        continue
    a = np.array([r[0] for r in rows]); lk = np.array([r[1] for r in rows])
    best = np.array([r[3] for r in rows])
    ax[0].loglog(a, lk, "o-", color=c, label=f"T={T} t_g (at stop)")
    ax[0].loglog(a, best, "s--", color=c, alpha=0.5, label=f"T={T} best (oracle)")
    print(f"T={T}: " + "  ".join(f"a={r[0]:.0e}:lk={r[1]:.3f}" for r in rows))
ax[0].axhline(1.0, color="k", lw=0.5, ls=":")
ax[0].set_xlabel("Tikhonov alpha"); ax[0].set_ylabel("low-k recovery error")
ax[0].set_title("Tikhonov L-curve: recovery vs regularization")
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

# per-kx error at the best alpha (min lowk_err) for each T
for T, c in [(5, "C0"), (10, "C1")]:
    rows = load_T(T)
    if not rows:
        continue
    abest = min(rows, key=lambda r: r[1])[0]
    f = glob.glob(str(D / f"alpha_T{T}_a*.npz"))
    fb = [x for x in f if abs(float(re.search(r"_a([0-9.eE+-]+)\.npz", x).group(1)) - abest) < 1e-30]
    if not fb:
        continue
    d = np.load(fb[0], allow_pickle=True)
    ax[1].semilogx(d["kx"][1:], d["errk"][1:], "o-", color=c, ms=3,
                   label=f"T={T} (alpha={abest:.0e}, lowk={float(d['lowk_err']):.2f})")
ax[1].axhline(1.0, color="k", lw=0.5)
ax[1].axvline(6, color="gray", ls="--", label="band edge kx=6")
ax[1].set_xlabel("kx"); ax[1].set_ylabel("per-kx recovery error")
ax[1].set_title("per-kx recovery at optimal alpha (truth band kx in [2,6])")
ax[1].set_xlim(1.5, 7); ax[1].set_ylim(0, 1.3); ax[1].legend(fontsize=8)
ax[1].grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig(OUT / "fig_lcurve.png", dpi=160); plt.close(fig)
print(f"-> {OUT/'fig_lcurve.png'}")
