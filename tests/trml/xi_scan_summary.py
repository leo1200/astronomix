"""Collate the Damkohler (xi) scan into one figure — with magnitude-INDEPENDENT
reconstruction metrics, so the trend is not faked by the PDF concentrating.

Reads data/fp_result_xi*.npz (written by fokker_planck_test.py). Key point: the
absolute L1 of dM/dlogT shrinks with xi simply because the mixing-range PDF
values shrink (the distribution concentrates into the cold peak), NOT because the
reconstruction is relatively more accurate. We therefore also report:
  * log-shape deviation  <|log10 P_hat - log10 P_eul|>  over the support (dex):
    magnitude-independent; measures the relative/vertical gap on the log plot,
  * KL(P_eul || P_hat): scale-invariant but mass-weighted (cold-peak dominated).
and the diffusive-window diagnostics D(T_mix; dt): the window SHIFTS to smaller
dt as xi grows (faster cooling), eventually below the dt_min we can resolve.

The reconstruction curve saved per xi is the one minimising the *absolute*
support-L1 (the validity-window lag); the relative metrics are evaluated on it.
"""
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

results = [dict(np.load(p)) for p in glob.glob("data/fp_result_xi*.npz")]
results.sort(key=lambda d: float(d["xi"]))
if not results:
    raise SystemExit("no data/fp_result_xi*.npz found — run xi_scan.sh first")
colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))


def metrics(d):
    """Absolute-L1, log-shape (dex) and KL over the support for one xi."""
    T = d["T_centers"]; P = d["eul_mass_pdf"]; Ph = d["recon_dMdlogT"]
    dlogT = np.log10(T[1]) - np.log10(T[0])
    sup = (T >= float(d["T_support_lo"])) & (T <= float(d["T_support_hi"]))
    abs_l1 = 0.5 * np.sum(np.abs(Ph[sup] - P[sup])) * dlogT
    good = sup & (P > 0) & (Ph > 0)
    log_dex = float(np.mean(np.abs(np.log10(Ph[good]) - np.log10(P[good]))))
    kl = float(np.sum(P[good] * np.log(P[good] / Ph[good])) * dlogT)
    return abs_l1, log_dex, kl


xis = np.array([float(d["xi"]) for d in results])
M = np.array([metrics(d) for d in results])
abs_l1, log_dex, kl = M[:, 0], M[:, 1], M[:, 2]

print("  xi    abs_L1   log_shape(dex)   KL")
for x, a, l, k in zip(xis, abs_l1, log_dex, kl):
    print(f"{x:6.0f}   {a:.3f}    {l:.3f}           {k:.4f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.3))

# --- 1) magnitude-dependent vs magnitude-independent error vs xi ---------
ax1.plot(xis, abs_l1, "o-", lw=2, color="tab:blue",
         label=r"absolute $L1$ of $dM/d\log T$  (magnitude-biased)")
ax1.plot(xis, log_dex, "^-", lw=2, color="tab:red",
         label=r"log-shape $\langle|\Delta\log_{10}P|\rangle$ [dex]  (relative)")
ax1.set_xscale("log")
ax1.set_xlabel(r"$\xi = t_{sh}/t_{\rm cool,min}$")
ax1.set_ylabel("reconstruction error")
ax1.set_ylim(0, max(abs_l1.max(), log_dex.max()) * 1.15)
axk = ax1.twinx()
axk.plot(xis, kl, "s--", lw=1.6, color="tab:green", label=r"KL$(P_{\rm eul}\|\widehat P)$")
axk.set_ylabel("KL (mass-weighted)", color="tab:green")
axk.tick_params(axis="y", labelcolor="tab:green")
axk.set_ylim(bottom=0)
lines = ax1.get_lines() + axk.get_lines()
ax1.legend(lines, [ln.get_label() for ln in lines], fontsize=7.5, loc="upper center")
ax1.set_title("relative error is ~flat; absolute/KL fall as the PDF concentrates")

# --- 2) diffusion window shifts (and runs out of resolution) -------------
dt_min = min(float(d["dt_over_tsh"].min()) for d in results)
for d, color in zip(results, colors):
    dt = d["dt_over_tsh"]; D = d["D_vs_lag"]
    fin = np.isfinite(D) & (D > 0)
    norm = np.nanmax(D[fin]) if fin.any() else 1.0
    ax2.plot(dt, D / norm, "o-", ms=3, lw=1.6, color=color,
             label=fr"$\xi={float(d['xi']):.0f}$")
ax2.axvline(dt_min, color="k", ls=":", lw=1, label=r"$\Delta t_{\min}$ (resolution)")
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel(r"$\Delta t / t_{sh}$")
ax2.set_ylabel(r"$D(T_{\rm mix};\Delta t)\,/\,\max_{\Delta t} D$")
ax2.set_title(r"diffusive window shifts to smaller $\Delta t$ with $\xi$")
ax2.set_ylim(1e-3, 2)
ax2.legend(fontsize=8, ncol=2)

# --- 3) reconstruction vs Eulerian per xi (what the eye sees) ------------
for d, color in zip(results, colors):
    T = d["T_centers"]
    ax3.plot(T, d["eul_mass_pdf"], "-", lw=2, color=color)
    ax3.plot(T, d["recon_dMdlogT"], ":", lw=2, color=color)
ax3.set_xscale("log"); ax3.set_yscale("log")
ax3.set_xlim(float(results[0]["T_support_lo"]), float(results[0]["T_support_hi"]))
ax3.set_ylim(bottom=1e-4)
ax3.set_xlabel("T"); ax3.set_ylabel(r"$dM/d\log T$ (normalized)")
ax3.set_title(r"solid: $P_{\rm eul}$   dotted: $\widehat P$   (colour = $\xi$)")
handles = [Line2D([0], [0], color=c, lw=2, label=fr"$\xi={float(d['xi']):.0f}$")
           for d, c in zip(results, colors)]
ax3.legend(handles=handles, fontsize=8, ncol=2)

fig.suptitle("Damkohler (xi) scan — the relative reconstruction quality is "
             "~xi-independent (N=64, M=0.5, chi=100)")
fig.tight_layout()
fig.savefig("figures/fp_xi_scan.png", dpi=150)
fig.savefig("figures/fp_xi_scan.svg")
print("\nwrote figures/fp_xi_scan.png")
