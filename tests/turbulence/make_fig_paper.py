"""
Paper-style figures for the HOW-MHD turbulence reproduction (OU forcing, Pallas),
analogues of Seo & Ryu (2023) Figs 14-18.

  fig_paper_isothermal_spectra.png  - ISM + ICM density/kinetic/magnetic spectra (Fig 15)
  fig_paper_isothermal_slices.png   - ISM + ICM magnetic-energy slices (Fig 14)
  fig_paper_cmp_timeseries.png      - iso vs adiabatic time evolution (Fig 16)
  fig_paper_cmp_slices.png          - iso vs adiabatic E_K / E_B slices (Fig 17)
  fig_paper_cmp_spectra.png         - iso vs adiabatic spectra (Fig 18)

Spectra are averaged over the paper's saturated windows. Reads data_paper/paper_<tag>.npz.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

DATA = os.environ.get("PAPER_DATA", "data_paper")
FIG = "figures"
os.makedirs(FIG, exist_ok=True)
K0 = 2.0 * np.pi  # fundamental wavenumber for L0 = 1


def safe_lognorm(sl):
    """LogNorm with guaranteed 0 < vmin < vmax from positive slice values."""
    pos = sl[np.isfinite(sl) & (sl > 0)]
    if pos.size == 0:
        return Normalize()
    vmin = np.percentile(pos, 2); vmax = np.percentile(pos, 99.8)
    if not (vmin > 0):
        vmin = pos.min()
    if not (vmax > vmin):
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)

ISM_TAG = os.environ.get("ISM_TAG", "ISM_N128_F3.0")
ICM_TAG = os.environ.get("ICM_TAG", "ICM_N128")
ISO_TAG = os.environ.get("ISO_TAG", "CMPiso_N128")
ADIA_TAG = os.environ.get("ADIA_TAG", "CMPadia_N128")


def load(tag):
    f = os.path.join(DATA, f"paper_{tag}.npz")
    return dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else None


def avg_spec(d, key, tlo, thi):
    """Average a spectrum over snapshots with t/tcross in [tlo, thi], excluding
    collapsed snapshots (v_rms -> 0 after a floor-cascade blow-up)."""
    t = d["t_over_tc"]; S = d[key]; alive = d["vrms_t"] > 0.1
    m = (t >= tlo) & (t <= thi) & np.isfinite(S).all(axis=1) & alive
    if not m.any():  # window collapsed: fall back to the last alive third
        idx = np.where(alive)[0]
        m = np.zeros_like(alive);
        if len(idx):
            m[idx[max(0, len(idx) - max(1, len(idx)//3)):]] = True
    return d["k"], np.nanmean(S[m], axis=0)


def plot_spec(ax, k, P, color, label, kref_lo=2.0, kref_hi=20.0):
    x = k / K0
    good = (P > 0) & np.isfinite(P) & (x > 0)
    ax.plot(np.log10(x[good]), np.log10(P[good]), color=color, label=label, lw=1.4)


def add_kolmogorov(ax, x0, y0, dx=1.0):
    xs = np.array([x0, x0 + dx])
    ax.plot(xs, y0 - (5.0 / 3.0) * (xs - x0), "k-", lw=1.5)
    ax.text(x0 + dx * 0.5, y0 - (5.0 / 3.0) * dx * 0.5 + 0.15, r"$\propto k^{-5/3}$", fontsize=10)


# ============================================================ Fig 15: ISM+ICM spectra
ism = load(ISM_TAG); icm = load(ICM_TAG)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
rows = [("ISM turbulence  ($M_{turb}\\approx10$, $\\beta_p=0.1$)", ism, 2.5, 5.0),
        ("ICM turbulence  ($M_{turb}\\approx0.5$, $\\beta_p=10^6$)", icm, 15.0, 30.0)]
specs = [("spec_rho", r"$\log P_\rho(k)$"), ("spec_EK", r"$\log P_{E_K}(k)$"), ("spec_EB", r"$\log P_{E_B}(k)$")]
for r, (title, d, tlo, thi) in enumerate(rows):
    for c, (key, ylab) in enumerate(specs):
        ax = axes[r, c]
        if d is None:
            ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes); continue
        k, P = avg_spec(d, key, tlo, thi)
        plot_spec(ax, k, P, "C0", None)
        xg = k / K0; Pg = P[(P > 0) & np.isfinite(P)]
        if len(Pg):
            y0 = np.log10(np.nanmax(Pg)) - 0.5
            add_kolmogorov(ax, 0.4, y0)
        ax.set_xlabel(r"$\log(k/k_0)$"); ax.set_ylabel(ylab)
        ax.set_xlim(0, 1.9); ax.grid(alpha=0.2)
        if c == 1:
            mt = float(np.nanmean(d["Ms_t"][(d['t_over_tc']>=tlo)&(d['t_over_tc']<=thi)]))
            ax.set_title(title + f"\n(measured $M_{{turb}}\\approx{mt:.1f}$)", fontsize=10)
fig.suptitle("Isothermal MHD turbulence — power spectra (cf. HOW-MHD Fig. 15)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_paper_isothermal_spectra.png"), dpi=200)
fig.savefig(os.path.join(FIG, "fig_paper_isothermal_spectra.svg")); print("wrote isothermal_spectra")

# ============================================================ Fig 14: ISM+ICM E_B slices
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (title, d) in zip(axes, [("ISM  ($M\\approx10$, $\\beta_p=0.1$)", ism),
                                  ("ICM  ($M\\approx0.5$, $\\beta_p=10^6$)", icm)]):
    if d is None:
        ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes); continue
    EB = d["EB_slice"]; EBm = np.where(EB > 0, EB, np.nan)
    im = ax.imshow(EBm, origin="lower", cmap="magma", norm=safe_lognorm(EB))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$E_B=B^2/2$")
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Magnetic-energy slices at end of run (cf. HOW-MHD Fig. 14)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_paper_isothermal_slices.png"), dpi=200)
print("wrote isothermal_slices")

# ============================================================ Fig 16: iso vs adiabatic time series
iso = load(ISO_TAG); adia = load(ADIA_TAG)
if iso is not None and adia is not None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("Ms_t", r"$M_{turb}$"), ("drho_t", r"$(\rho-\rho_0)_{rms}$"),
              ("EK_t", r"$E_K$"), ("EB_t", r"$E_B - E_{B,0}$")]
    for ax, (key, lab) in zip(axes.ravel(), panels):
        for d, col, nm in [(iso, "C3", "isothermal"), (adia, "C0", "adiabatic")]:
            y = d[key].copy()
            if key == "EB_t":
                y = y - float(d["EB0"])
            m = np.isfinite(d["Ms_t"]) & (d["vrms_t"] > 1e-6)
            ax.plot(d["t_over_tc"][m], y[m], color=col, label=nm, lw=1.6)
        ax.set_xlabel(r"$t/t_{cross}$"); ax.set_ylabel(lab); ax.grid(alpha=0.2)
        ax.axvspan(1.5, 2.5, color="gray", alpha=0.12)
    axes[0, 0].legend()
    fig.suptitle(r"Iso vs adiabatic turbulence ($M_{turb}\approx1$, $\beta_p=1$) — time evolution (cf. Fig. 16)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_paper_cmp_timeseries.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "fig_paper_cmp_timeseries.svg")); print("wrote cmp_timeseries")

    # Fig 18: iso vs adiabatic spectra averaged over 1.5-2.5 tcross
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for c, (key, ylab) in enumerate(specs):
        ax = axes[c]
        for d, col, nm in [(iso, "C3", "isothermal"), (adia, "C0", "adiabatic")]:
            k, P = avg_spec(d, key, 1.5, 2.5)
            plot_spec(ax, k, P, col, nm)
        kk, PP = avg_spec(iso, key, 1.5, 2.5); Pg = PP[(PP > 0) & np.isfinite(PP)]
        if len(Pg):
            add_kolmogorov(ax, 0.4, np.log10(np.nanmax(Pg)) - 0.5)
        ax.set_xlabel(r"$\log(k/k_0)$"); ax.set_ylabel(ylab); ax.set_xlim(0, 1.9); ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(r"Iso vs adiabatic spectra, $1.5\leq t/t_{cross}\leq2.5$ (cf. Fig. 18)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_paper_cmp_spectra.png"), dpi=200)
    fig.savefig(os.path.join(FIG, "fig_paper_cmp_spectra.svg")); print("wrote cmp_spectra")

    # Fig 17: iso vs adiabatic E_K (top) and E_B (bottom) slices
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for col, (d, nm) in enumerate([(iso, "Isothermal"), (adia, "Adiabatic")]):
        for row, key in enumerate(["EK_slice", "EB_slice"]):
            ax = axes[row, col]; sl = d[key]; slm = np.where(sl > 0, sl, np.nan)
            im = ax.imshow(slm, origin="lower", cmap="viridis", norm=safe_lognorm(sl))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(nm)
            if col == 0:
                ax.set_ylabel(r"$E_K$" if row == 0 else r"$E_B$", fontsize=12)
    fig.suptitle(r"Iso vs adiabatic kinetic/magnetic-energy slices (cf. Fig. 17)", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_paper_cmp_slices.png"), dpi=200)
    print("wrote cmp_slices")
