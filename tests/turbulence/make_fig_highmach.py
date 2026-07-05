#!/usr/bin/env python
"""High-Mach isothermal-MHD turbulence figures.

Plots the precomputed spectra (kinetic E_K, magnetic E_B, density rho)
stored in the run npz files. No simulation / GPU needed -- pure numpy +
matplotlib on the saved arrays.

Runs (data_highmach/):
  paper_isoM50long.npz      N=128, sustained M ~ 55 (stationary)
  paper_isoM100.npz         N=128, sustained M ~ 105 (stationary)
  paper_isoM50_N256_f1e3.npz  N=256 high-res, M ~ 50 (crashed at snap 5)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_highmach")
OUT = os.path.join(HERE, "figures", "highmach")
os.makedirs(OUT, exist_ok=True)


def load(name):
    return np.load(os.path.join(DATA, name), allow_pickle=True)


def valid_snaps(d):
    """Indices of physically valid snapshots (finite spectra, M>0, before crash)."""
    sek = d["spec_EK"]
    Ms = d["Ms_t"]
    fb = int(d["first_bad"])
    idx = []
    for i in range(sek.shape[0]):
        if fb >= 0 and i >= fb:
            continue
        if not np.all(np.isfinite(sek[i])):
            continue
        if not np.isfinite(Ms[i]) or Ms[i] <= 1.0:
            continue
        idx.append(i)
    return idx


def stationary_spectrum(d, key, frac=0.4):
    """Time-average a spectrum over the last `frac` of valid snapshots."""
    idx = valid_snaps(d)
    take = idx[max(0, int(len(idx) * (1 - frac))):]
    spec = d[key][take]
    return np.nanmean(spec, axis=0), take


# ---------------------------------------------------------------------------
runs = {
    "isoM50long": dict(file="paper_isoM50long.npz", label="N=128, M$\\approx$55",
                       color="tab:blue", stationary=True),
    "isoM100": dict(file="paper_isoM100.npz", label="N=128, M$\\approx$105",
                    color="tab:red", stationary=True),
    "isoM50N256": dict(file="paper_isoM50_N256_f1e3.npz", label="N=256, M$\\approx$50 (snap 3)",
                       color="tab:green", stationary=False),
}
for r in runs.values():
    r["d"] = load(r["file"])

# ---------------------------------------------------------------------------
# Figure 1: kinetic + magnetic energy spectra (overlay) with reference slopes
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, key, title in zip(axes, ["spec_EK", "spec_EB"],
                          ["Kinetic energy spectrum  $E_K(k)$",
                           "Magnetic energy spectrum  $E_B(k)$"]):
    for name, r in runs.items():
        d = r["d"]
        k = d["k"]
        if r["stationary"]:
            spec, used = stationary_spectrum(d, key)
            lab = r["label"] + f" (avg {len(used)} snaps)"
        else:
            # last good snapshot (highest Mach before crash)
            i = valid_snaps(d)[-1]
            spec = d[key][i]
            lab = r["label"]
        m = (k > 0) & np.isfinite(spec) & (spec > 0)
        ax.loglog(k[m], spec[m], color=r["color"], lw=1.8, label=lab)
    ax.axvline(8.0, color="0.6", lw=0.8, ls="--", alpha=0.6)  # forcing scale k_f
    ax.set_xlabel("$k$")
    ax.set_ylabel(title.split("  ")[-1])
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, which="both", alpha=0.15)
fig.suptitle("Isothermal supersonic MHD turbulence -- energy spectra", fontsize=13)
fig.tight_layout()
f1 = os.path.join(OUT, "fig_highmach_spectra.png")
fig.savefig(f1, dpi=150)
fig.savefig(f1.replace(".png", ".svg"))
print("wrote", f1)

# ---------------------------------------------------------------------------
# Figure 2: high-res N=256 spectral evolution (snapshots) + Mach time series
# ---------------------------------------------------------------------------
d = runs["isoM50N256"]["d"]
idx = valid_snaps(d)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
k = d["k"]
cmap = plt.cm.viridis(np.linspace(0.15, 0.9, len(idx)))
for c, i in zip(cmap, idx):
    spec = d["spec_EK"][i]
    m = (k > 0) & np.isfinite(spec) & (spec > 0)
    ax.loglog(k[m], spec[m], color=c, lw=1.6,
              label=f"t/t$_c$={d['t_over_tc'][i]:.2f}, M={d['Ms_t'][i]:.0f}")
ax.axvline(8.0, color="0.6", lw=0.8, ls="--", alpha=0.6)  # forcing scale k_f
ax.set_xlabel("$k$"); ax.set_ylabel("$E_K(k)$")
ax.set_title("N=256: kinetic spectrum buildup")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, which="both", alpha=0.15)

ax = axes[1]
for name, r in runs.items():
    d = r["d"]
    vi = valid_snaps(d)
    ax.plot(d["t_over_tc"][vi], d["Ms_t"][vi], "o-", color=r["color"],
            lw=1.6, ms=4, label=r["label"])
ax.set_xlabel("$t / t_c$"); ax.set_ylabel("turbulent Mach number  $M$")
ax.set_title("Mach-number evolution")
ax.legend(fontsize=8, frameon=False)
ax.grid(True, alpha=0.2)
fig.tight_layout()
f2 = os.path.join(OUT, "fig_highmach_N256_evolution.png")
fig.savefig(f2, dpi=150)
fig.savefig(f2.replace(".png", ".svg"))
print("wrote", f2)

# ---------------------------------------------------------------------------
# Figure 3: 2D slices (kinetic / magnetic / density) for each run
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(runs), 3, figsize=(11, 3.4 * len(runs)))
for row, (name, r) in enumerate(runs.items()):
    d = r["d"]
    for col, (key, ttl, cm) in enumerate([
        ("EK_slice", "kinetic energy", "inferno"),
        ("EB_slice", "magnetic energy", "viridis"),
        ("rho_slice", r"density $\rho$", "cividis"),
    ]):
        ax = axes[row, col]
        sl = d[key]
        if key == "rho_slice":
            im = ax.imshow(np.log10(np.maximum(sl, 1e-6)), origin="lower", cmap=cm)
        else:
            im = ax.imshow(sl, origin="lower", cmap=cm)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if row == 0:
            ax.set_title(ttl)
        if col == 0:
            ax.set_ylabel(r["label"], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Mid-plane slices (log density)", fontsize=13)
fig.tight_layout()
f3 = os.path.join(OUT, "fig_highmach_slices.png")
fig.savefig(f3, dpi=150)
print("wrote", f3)
