"""Formalize Adam vs Gauss-Newton vs horizon (kx2-6, Re=2000, multi-restart).
(a) ic_err per cold restart vs horizon: Adam fails early (first-order traps),
GN recovers to ~T=60-70, then GENUINE mode-2 onsets (GN scatters into wrong
minima) ~T>=80. (b) conditioning: kappa(J) stays tiny (no mode-1 wall) and
sigma_min stays >> noise (no mode-3) through the onset -> the T>=80 failure is
nonlinear multimodality, not conditioning or information."""
from pathlib import Path
import glob, re
import numpy as np
import matplotlib.pyplot as plt

D = Path(__file__).parent / "data"; OUT = Path(__file__).parent / "figures"


def gn_at(T):
    vals = []
    for f in glob.glob(str(D / f"gnh_T{T}_i*.npz")) + (glob.glob(str(D / "gn_single_i*.npz")) if T == 45 else []):
        vals.append(float(np.load(f, allow_pickle=True)["ic_err"]))
    return np.array(vals)


def adam_at(T):
    if T != 45:
        return np.array([])
    return np.array([float(np.load(f, allow_pickle=True)["lowk_err"])
                     for f in glob.glob(str(D / "basin_single_i*.npz"))])


Ts = [45, 50, 60, 70, 80, 120, 160]
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5))
rng = np.random.default_rng(0)
for T in Ts:
    g = gn_at(T)
    if len(g):
        ax[0].scatter(np.full(len(g), T) + rng.uniform(-1.2, 1.2, len(g)),
                      np.clip(g, 0, 2.0), color="C0", s=30, alpha=0.8,
                      label="Gauss-Newton" if T == Ts[0] else None, zorder=3)
    a = adam_at(T)
    if len(a):
        ax[0].scatter(np.full(len(a), T) + rng.uniform(-1.2, 1.2, len(a)),
                      np.clip(a, 0, 2.0), color="C3", marker="s", s=30, alpha=0.8,
                      label="Adam", zorder=3)
# continuation chain (MS basin-enlargement principle): warm-start GN across T
cont = []
for T in [60, 70, 80, 100, 120]:
    f = D / f"cont_T{T}.npz"
    if f.exists():
        cont.append((T, float(np.load(f, allow_pickle=True)["ic_err"])))
if cont:
    ct = np.array(cont)
    ax[0].plot(ct[:, 0], np.clip(ct[:, 1], 0, 2), "D-", color="C2", ms=7, lw=2,
               label="continuation (MS principle)", zorder=4)
ax[0].axhline(0.1, color="gray", ls=":", label="recovered")
ax[0].axhline(1.0, color="k", lw=0.5)
ax[0].axvspan(75, 165, color="orange", alpha=0.08)
ax[0].text(120, 1.7, "genuine mode-2\n(cold GN fails)", ha="center", fontsize=9, color="C1")
ax[0].set_xlabel("horizon T / t_g"); ax[0].set_ylabel("IC recovery error (clipped at 2)")
ax[0].set_title("(a) Adam vs Gauss-Newton vs horizon (multi-restart)")
ax[0].legend(fontsize=8, loc="center left"); ax[0].grid(alpha=0.3)

cf = D / "gn_conditioning.npz"
if cf.exists():
    c = np.load(cf)
    ax2 = ax[1]; ax2.semilogy(c["T_g"], c["kappa"], "o-", color="C0", label=r"$\kappa(J)$ (cond.)")
    ax2.semilogy(c["T_g"], c["smin"], "s-", color="C2", label=r"$\sigma_{min}$")
    ax2.semilogy(c["T_g"], c["nfloor"], "k--", label="noise level")
    ax2.axhline(1e7, color="C1", ls=":", label="float32 wall")
    ax2.axvspan(75, 165, color="orange", alpha=0.08)
    ax2.set_xlabel("horizon T / t_g"); ax2.set_ylabel("value")
    ax2.set_title("(b) at the mode-2 onset: $\\kappa$ tiny ($\\ll$ precision wall),\n"
                  "$\\sigma_{min}\\gg$ noise -> failure is NONLINEAR (not mode 1 or 3)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")
fig.suptitle("KH recovery: first-order traps dissolve under GN; genuine mode-2 onsets at long T "
             "(well-conditioned & informative)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fig_optimizer_horizon.png", dpi=160); plt.close(fig)
for T in Ts:
    g = gn_at(T)
    if len(g):
        print(f"T={T:4d}: GN recovered {int((g<0.1).sum())}/{len(g)}  median ic_err={np.median(g):.3f}")
print(f"-> {OUT/'fig_optimizer_horizon.png'}")
