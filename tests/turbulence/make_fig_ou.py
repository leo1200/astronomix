"""
Build paper-comparable figures for the OU-forced isothermal-MHD turbulence
grid (cf. HOW-MHD, https://arxiv.org/pdf/2304.04360):

  fig_ou_spectra.png   - 2x2 kinetic + magnetic power spectra with k^-5/3 guide
  fig_ou_slices.png    - density / |v| / |B| mid-plane slices per case
  fig_ou_timeseries.png- v_rms(t) and M_s(t): statistical-stationarity check

Reads data/ou_Ms{ms}_MA{ma}_N{N}.npz written by ou_turbulence.py.
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

DATA = "data"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

# fixed grid order: rows = M_s (0.5, 2.0), cols = M_A (1.0, 100.0)
MS = [0.5, 2.0]
MA = [1.0, 100.0]
N = int(os.environ.get("OU_N", "64"))


def load(ms, ma):
    pats = glob.glob(os.path.join(DATA, f"ou_Ms{ms}_MA{ma}_N{N}.npz"))
    if not pats:
        return None
    return dict(np.load(pats[0], allow_pickle=True))


def case_label(d):
    return (rf"$M_s={float(d['Ms_stat']):.2f}$ ($\to${float(d['ms_aim']):.1f}), "
            rf"$M_A={float(d['MA_stat']):.1f}$ ($\to${float(d['ma_aim']):.0f})")


# ------------------------------------------------------------------ spectra
fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
for i, ms in enumerate(MS):
    for j, ma in enumerate(MA):
        ax = axes[i, j]
        d = load(ms, ma)
        if d is None:
            ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
            continue
        k = d["k"]
        # mask empty / non-positive bins so loglog doesn't draw vertical drops
        ek = np.where(d["kinetic_spectrum"] > 0, d["kinetic_spectrum"], np.nan)
        em = np.where(d["magnetic_spectrum"] > 0, d["magnetic_spectrum"], np.nan)
        ax.loglog(k, ek, label="kinetic", color="C0")
        ax.loglog(k, em, label="magnetic", color="C1")
        # k^-5/3 guide anchored in the inertial range
        mask = (k > 3 * 2 * np.pi) & (k < 12 * 2 * np.pi)
        if mask.any():
            kref = k[mask]
            anchor = ek[mask][0] * 2.0
            ax.loglog(kref, anchor * (kref / kref[0]) ** (-5 / 3), "k--", lw=1.2)
            ax.text(kref[len(kref) // 2], anchor * (kref[len(kref) // 2] / kref[0]) ** (-5 / 3),
                    r"$k^{-5/3}$", fontsize=11)
        ax.set_title(case_label(d), fontsize=11)
        ax.set_xlim(left=2 * np.pi * 1.5)
        ax.set_ylim(1e-9, 1e-1)
        if i == 1:
            ax.set_xlabel("wavenumber $k$")
        if j == 0:
            ax.set_ylabel("$E(k)$")
        ax.grid(alpha=0.2, which="both")
axes[0, 0].legend(loc="lower left", fontsize=10)
fig.suptitle(f"OU-forced isothermal MHD turbulence, ${N}^3$ — final energy spectra", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ou_spectra.png"), dpi=200)
fig.savefig(os.path.join(FIG, "fig_ou_spectra.svg"))
print("wrote fig_ou_spectra")

# ------------------------------------------------------------------ slices
fig, axes = plt.subplots(4, 3, figsize=(11, 14))
row = 0
for ms in MS:
    for ma in MA:
        d = load(ms, ma)
        titles = ["density", r"$|v|$", r"$|B|$"]
        if d is None:
            for c in range(3):
                axes[row, c].text(0.5, 0.5, "missing", ha="center", transform=axes[row, c].transAxes)
            row += 1
            continue
        slices = [d["dens_slice"], d["vmag_slice"], d["bmag_slice"]]
        for c, (sl, ti) in enumerate(zip(slices, titles)):
            ax = axes[row, c]
            if c == 0:
                im = ax.imshow(sl, origin="lower", cmap="viridis",
                               norm=LogNorm(vmin=max(sl.min(), 1e-3), vmax=sl.max()))
            else:
                im = ax.imshow(sl, origin="lower", cmap="viridis")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(ti, fontsize=12)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(case_label(d), fontsize=9)
        row += 1
fig.suptitle(f"OU-forced isothermal MHD turbulence, ${N}^3$ — final mid-plane slices", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ou_slices.png"), dpi=200)
fig.savefig(os.path.join(FIG, "fig_ou_slices.svg"))
print("wrote fig_ou_slices")

# ------------------------------------------------------------------ stationarity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for ms in MS:
    for ma in MA:
        d = load(ms, ma)
        if d is None:
            continue
        t = d["time_points"]
        lab = rf"$M_s\to{ms}$, $M_A\to{ma:.0f}$"
        ax1.plot(t, d["v_rms_t"], label=lab)
        ax2.plot(t, d["Ms_t"], label=lab)
ax1.set_xlabel("time"); ax1.set_ylabel(r"$v_{\rm rms}$"); ax1.set_title("RMS velocity (stationarity)")
ax1.axhline(1.0, color="k", ls=":", lw=0.8); ax1.grid(alpha=0.2); ax1.legend(fontsize=9)
ax2.set_xlabel("time"); ax2.set_ylabel(r"$M_s$"); ax2.set_title("sonic Mach number")
ax2.grid(alpha=0.2); ax2.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_ou_timeseries.png"), dpi=200)
print("wrote fig_ou_timeseries")
