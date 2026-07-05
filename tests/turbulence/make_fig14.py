"""
HOW-MHD (Seo & Ryu 2023) Fig. 14 reproduction -- plain matplotlib style.

2D slices of the magnetic energy log E_B for isothermal-MHD turbulence:

  WENO ISM  (M_turb ~ 10,  beta_p = 0.1)    -- top row
  WENO ICM  (M_turb ~ 0.5, beta_p = 10^6)   -- bottom row

Layout is 2x2: rows are ISM / ICM, columns are a lower- and a higher-resolution
run so the convergence is visible. The high-res column prefers N=512^3 and falls
back to N=448^3 (whatever data is present in data_fig14/), so dropping in the
H200 512^3 npz upgrades the figure automatically.

E_B is run in the cheap v_rms~1 normalisation and converted to the paper's a=1
normalisation by the exact similarity scaling E_B^paper = M_turb^2 * E_B, so the
colorbars match the paper (ISM 0..3, ICM -6..0).
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DATA = "data_fig14"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def load(tag):
    f = os.path.join(DATA, f"paper_{tag}.npz")
    return dict(np.load(f, allow_pickle=True)) if os.path.exists(f) else None


def best_hires(case):
    """Prefer 512^3, fall back to 448^3 for the high-resolution column."""
    for res in ("512", "448"):
        d = load(f"{case}_N{res}")
        if d is not None:
            return d
    return None


def log_EB_paper(d):
    """log10 E_B in the paper's a=1 normalisation (E_B^paper = M_turb^2 E_B)."""
    EB = np.asarray(d["EB_slice"], dtype=np.float64)
    lam2 = float(d["mturb_aim"]) ** 2
    EB = np.where(EB > 0, EB, np.nan) * lam2
    return np.log10(EB)


def _beta_str(b):
    """Plasma beta as a compact LaTeX string (10^6 for big, 0.1 for small)."""
    b = float(b)
    if b >= 100:
        return rf"10^{{{int(round(np.log10(b)))}}}"
    return f"{b:g}"


# rows: (case label, lo-res dict, hi-res dict, colorbar range)
rows = [
    ("WENO ISM", load("ISM_N256"), best_hires("ISM"), (0.0, 3.0)),
    ("WENO ICM", load("ICM_N256"), best_hires("ICM"), (-6.0, 0.0)),
]
extent = [-0.5, 0.5, -0.5, 0.5]

fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6))
row_im = {}  # one representative mappable per row (both columns share vmin/vmax)
for i, (title, d_lo, d_hi, (vmin, vmax)) in enumerate(rows):
    for j, d in enumerate((d_lo, d_hi)):
        ax = axes[i, j]
        if d is None:
            ax.text(0.5, 0.5, "512$^3$ pending\n(run on H200)", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12, color="0.4")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        img = log_EB_paper(d)
        im = ax.imshow(img.T, origin="lower", extent=extent, cmap="Blues",
                       vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        row_im[i] = im
        N = int(d["N"])
        # label with the M_turb of the snapshot the slice was taken from (the
        # last *alive* snapshot = first_bad-1), not Ms_t[-1] which is the NaN->0
        # final snapshot when a run blows up only at late time (e.g. ISM 512^3).
        ms = np.asarray(d["Ms_t"], dtype=float)
        fb = int(d["first_bad"])
        slice_idx = (fb - 1) if fb >= 0 else (len(ms) - 1)
        Mlast = float(ms[slice_idx])
        beta_str = _beta_str(d["beta"])
        ax.set_title(
            rf"$N={N}^3$,  $M_S={Mlast:.1f}$,  $\beta_p={beta_str}$",
            fontsize=13, pad=12,
        )
        ax.set_xticks([]); ax.set_yticks([])  # unit box [0,1]^3 — axes unlabeled

fig.tight_layout()
# Move the two columns close together, then attach ONE shared colorbar per row
# on the far right (both columns share vmin/vmax, so no inter-column colorbar is
# needed). Extra title pad keeps titles clear of the panels above.
fig.subplots_adjust(wspace=0.03, hspace=0.12)
for i in range(len(rows)):
    if i in row_im:
        cb = fig.colorbar(row_im[i], ax=list(axes[i, :]), fraction=0.046, pad=0.02)
        cb.set_label(r"$\log_{10} E_B$")
out = os.path.join(FIG, "fig14_WENO_reproduction")
fig.savefig(out + ".png", dpi=200, bbox_inches="tight")
fig.savefig(out + ".pdf", bbox_inches="tight")
print("wrote", out + ".png/.pdf")

for title, d_lo, d_hi, _ in rows:
    for d in (d_lo, d_hi):
        if d is None:
            continue
        img = log_EB_paper(d)
        print(f"{title} N={int(d['N'])}: M_turb(last)={float(d['Ms_t'][-1]):.2f}, "
              f"log E_B range [{np.nanmin(img):.2f}, {np.nanmax(img):.2f}]")
