"""
Side-by-side comparison of the synthetic Cas A views with real observations.

Left column: real Chandra (deep X-ray) and the 2024 Chandra+Webb+Hubble
composite (downloaded press images in ``real_obs/``). Right column: the same
views synthesized from a ``--save-state`` npz. Also measures the forward- and
reverse-shock radii from the state's radial temperature profile and prints
them against the observed values (FS ~ 2.5 pc, RS ~ 1.6 pc at 3.4 kpc;
Gotthelf et al. 2001).

Usage:
    python compare_real.py casa_n512_jet_dual_x32.npz --los x
"""

# ==== keep jax off the GPUs (imaging is numpy/matplotlib) ====
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# ruff: noqa: E402
# =============================================================

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from _common import (
    FIGURES_DIR,
    snr_code_units,
    temperature_K,
    chandra_deep_figure,
    xray_band_emissivity,
    synchrotron_emissivity,
    ir_dust_emissivity,
    pristine_debris_emissivity,
    _stretch_project,
)

REAL_DIR = Path(__file__).resolve().parent / "real_obs"


def shock_radii_pc(rho, p, cu, box):
    """(reverse, forward) shock radius estimates from the median T(r) profile.

    Moving outward: the cold homologous ejecta (T ~ floor) ends where the
    median temperature first rises above 1e5 K (reverse shock); the shocked
    region ends where it last falls back below 1e5 K (forward shock).
    """
    T = temperature_K(rho, p, cu)
    n = rho.shape[0]
    c = (np.arange(n) + 0.5) * (box / n) - box / 2
    r = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
    nbin = 120
    edges = np.linspace(0.0, box / 2, nbin + 1)
    idx = np.clip(np.searchsorted(edges, r.ravel(), "right") - 1, 0, nbin - 1)
    med = np.full(nbin, np.nan)
    Tr = T.ravel()
    for b in range(nbin):
        sel = idx == b
        if sel.any():
            med[b] = np.median(Tr[sel])
    bc = 0.5 * (edges[:-1] + edges[1:])
    hot = med > 1e5
    if not hot.any():
        return np.nan, np.nan
    return float(bc[np.argmax(hot)]), float(bc[len(hot) - 1 - np.argmax(hot[::-1])])


def synthetic_composite_rgb(rho, p, cu, dx_cm, axis):
    """Full-bleed X-ray+IR composite RGB (same mixing as multiwavelength_figure)."""
    bands, _ = xray_band_emissivity(rho, p, cu)
    xray3d = bands["soft"] + bands["medium"] + bands["hard"]
    X = _stretch_project(xray3d, dx_cm, axis=axis, gamma=0.5)
    S = _stretch_project(synchrotron_emissivity(rho, p, cu), dx_cm, axis=axis, gamma=0.5)
    I = _stretch_project(ir_dust_emissivity(rho, p, cu), dx_cm, axis=axis, gamma=0.5)
    P = _stretch_project(pristine_debris_emissivity(rho, p, cu), dx_cm, axis=axis, gamma=0.5)
    return np.clip(np.stack([
        1.05 * I + 0.90 * P + 0.10 * X + 0.08 * S,
        0.55 * I + 0.22 * P + 0.55 * X + 0.45 * S,
        0.12 * I + 0.10 * P + 1.05 * X + 1.00 * S,
    ], axis=-1), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", help="npz from --save-state")
    ap.add_argument("--los", choices=["x", "y", "z"], default="x",
                    help="synthetic line of sight (default x: near-side dense "
                         "shell in front, like the real Green Monster)")
    ap.add_argument("--prefix", default="cassiopeia_n512")
    args = ap.parse_args()
    axis = {"x": 0, "y": 1, "z": 2}[args.los]

    d = np.load(args.state)
    rho, p = d["rho"], d["press"]
    box = float(d["box"]); age = float(d["age"]); n = int(d["num_cells"])
    cu = snr_code_units()
    dx_cm = (box / n) * float((1.0 * cu.code_length).to(u.cm).value)

    rs_pc, fs_pc = shock_radii_pc(rho, p, cu, box)
    print(f"[compare] shock radii: RS = {rs_pc:.2f} pc, FS = {fs_pc:.2f} pc "
          f"(observed Cas A: RS ~ 1.6 pc, FS ~ 2.5 pc)")

    # synthetic deep-Chandra panel (rendered borderless to a temp png) with the
    # observational forward model on -- comparing against a real exposure
    deep_tmp = FIGURES_DIR / f"_{args.prefix}_deep_tmp.png"
    chandra_deep_figure(rho, p, cu, box, dx_cm, out_path=deep_tmp, axis=axis,
                        observe=True)
    synth_deep = plt.imread(deep_tmp)
    deep_tmp.unlink()

    comp_rgb = synthetic_composite_rgb(rho, p, cu, dx_cm, axis)

    real_deep = plt.imread(REAL_DIR / "casa2024_xray_color.jpg")
    real_comp = plt.imread(REAL_DIR / "casa2024_casa.jpg")

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 12), facecolor="black")
    panels = [
        (real_deep, "REAL: Chandra deep X-ray (NASA/CXC/SAO)"),
        (synth_deep, f"SYNTHETIC: deep X-ray proxy (N={n}, ~{age:.0f} yr, LOS {args.los})"),
        (real_comp, "REAL: Chandra + Webb + Hubble composite (2024)"),
        (comp_rgb, "SYNTHETIC: X-ray + IR composite"),
    ]
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(img if img.ndim == 3 else img, origin="upper"
                  if img is real_deep or img is real_comp or img is synth_deep
                  else "lower")
        ax.set_title(title, color="white", fontsize=11)
        ax.axis("off")
        ax.set_facecolor("black")
    fig.text(0.5, 0.015,
             f"synthetic shock radii: reverse {rs_pc:.1f} pc / forward {fs_pc:.1f} pc"
             f"    |    observed: reverse ~1.6 pc / forward ~2.5 pc"
             f"    |    real angular scale mapped at 3.4 kpc",
             color="lightgray", fontsize=10, ha="center")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIGURES_DIR / f"{args.prefix}_vs_real.png"
    fig.savefig(out, dpi=150, facecolor="black")
    plt.close(fig)
    print(f"[compare] saved {out}")


if __name__ == "__main__":
    main()
