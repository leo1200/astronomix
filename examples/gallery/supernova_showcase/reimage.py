"""
Offline re-imaging for the supernova showcase.

Rebuilds the three ``cassiopeia_realistic.py`` figures (slices/column density,
synthetic Chandra X-ray, X-ray + infrared composite) from a ``--save-state``
npz, without re-running the simulation and without touching a GPU. This is how
hero-resolution runs are imaged: the run itself only writes the npz; the
figures are produced (and iterated on) here.

Usage:
    python reimage.py casa_n512_dual_x32.npz --prefix cassiopeia_n512
"""

# ==== keep jax off the GPUs ====
# Figures are numpy/matplotlib; anything jax in _common runs fine on CPU.
# Setting CUDA_VISIBLE_DEVICES also disarms the showcase autocvd fallback.
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
# ruff: noqa: E402
# ===============================

# general
import argparse

# numerics
import numpy as np

# units
from astropy import units as u

# shared showcase helpers
from _common import (
    FIGURES_DIR,
    snr_code_units,
    temperature_K,
    realistic_figure,
    xray_figure,
    chandra_deep_figure,
    multiwavelength_figure,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state", help="npz from --save-state (rho, press, box, age, num_cells)")
    ap.add_argument("--prefix", default="cassiopeia",
                    help="output filename prefix (default reproduces the in-run names)")
    ap.add_argument("--los", choices=["x", "y", "z"], default="z",
                    help="projection line of sight; 'x' looks along the CSM-shell "
                         "asymmetry axis, putting the shocked dense shell in "
                         "front of the interior like the real near-side "
                         "'Green Monster'")
    ap.add_argument("--observe", action="store_true",
                    help="observational forward model for the deep-Chandra "
                         "image: Galactic absorption, ~1 arcsec PSF, Poisson "
                         "photon noise at deep-exposure depth")
    args = ap.parse_args()
    los_axis = {"x": 0, "y": 1, "z": 2}[args.los]

    d = np.load(args.state)
    rho, p = d["rho"], d["press"]
    box = float(d["box"])
    age_yr = float(d["age"])
    n = int(d["num_cells"])
    print(f"[reimage] {args.state}: N={n} box={box} pc age~{age_yr:.0f} yr "
          f"rho[{rho.min():.3e},{rho.max():.3e}]")

    cu = snr_code_units()
    T = temperature_K(rho, p, cu)
    # cell-centered radius grid (broadcast, so only r itself is a full cube)
    c = (np.arange(n) + 0.5) * (box / n) - box / 2
    r = np.sqrt(c[:, None, None] ** 2 + c[None, :, None] ** 2 + c[None, None, :] ** 2)
    dx_cm = (box / n) * float((1.0 * cu.code_length).to(u.cm).value)

    out = realistic_figure(
        rho, T, r, box,
        title=f"Cassiopeia A (realistic: clumpy ejecta, dense CSM shell, cooling, "
              f"N={n}, ~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / f"{args.prefix}_realistic.png",
    )
    print(f"[reimage] saved {out}")

    xout = xray_figure(
        rho, p, cu, box, dx_cm,
        title=f"Cassiopeia A -- synthetic Chandra X-ray (N={n}, ~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / f"{args.prefix}_xray.png", axis=los_axis,
    )
    print(f"[reimage] saved {xout}")

    cout = chandra_deep_figure(
        rho, p, cu, box, dx_cm,
        out_path=FIGURES_DIR / f"{args.prefix}_chandra.png", axis=los_axis,
        observe=args.observe,
    )
    print(f"[reimage] saved {cout}")

    mout = multiwavelength_figure(
        rho, p, cu, box, dx_cm,
        title=f"Cassiopeia A -- synthetic X-ray + infrared composite (N={n}, ~{age_yr:.0f} yr)",
        out_path=FIGURES_DIR / f"{args.prefix}_composite.png", axis=los_axis,
    )
    print(f"[reimage] saved {mout}")


if __name__ == "__main__":
    main()
