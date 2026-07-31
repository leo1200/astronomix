"""How structured is the remnant, scale by scale, against Chandra.

The spectral comparison in ``casa_observe.py`` scores the plasma; this scores the
STRUCTURE. Both are needed, and for the same reason: "the picture looks smoother
than the real one" is not a result until it is a number, and it is a number that
can move the wrong way while the picture improves.

The statistic is deliberately the simplest one that is fair between two images of
different exposure:

    at each scale, band-pass the counts image between a box of ``theta`` and one
    of ``2 theta``, take the RMS of that inside an annulus, subtract the Poisson
    contribution ANALYTICALLY, and divide by the local mean.

For nested boxcars over ``A1 < A2`` pixels the covariance of the two averages is
``m / A2``, so the Poisson variance of their difference is exactly
``m (1/A1 - 1/A2)`` -- no noise model to fit, no reference field to simulate.
What comes out is the fractional surface-brightness fluctuation carried by
structure of that angular size, which is comparable between a 20 ks synthetic
image and a 143 ks observation.

**The one trap**, and it is a serious one: at small scales the synthetic image is
noise-dominated, so the subtraction removes almost everything and what is left is
the difference of two nearly equal numbers. At 20 ks and 1 arcsec the Poisson
variance is ~96 % of the total, and the answer is not trustworthy. The signal-to-
noise of the subtraction is reported per scale for exactly this reason; treat any
scale below ~3 as indicative only, and generate the synthetic image at the real
exposure (``--exposure 143.5``) before quoting it.

Two more things the real image contains that the model does not, both of which
make the measured gap a LOWER bound: the Chandra PSF (~0.5 arcsec on axis, so
the real texture at 1 arcsec is itself suppressed) and the dust-scattering halo
(a smooth component that dilutes the real fluctuations).

Usage (CPU)::

    /export/home/lstorcks/xrayobs/bin/python casa_morphology.py \\
        obs_final_nei_synimg.npz --compare 2004
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np
from scipy.ndimage import uniform_filter

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PIXEL_ARCSEC = 0.492
REAL_EPOCH_DIR = Path("/export/data/lstorcks/chandra_casa/epoch_images")

#: box sizes in pixels; each measures structure between theta and 2 theta
BOXES = (2, 3, 5, 9, 17, 33)

#: the bright ejecta ring, well inside the forward shock (153") and outside the
#: centre, where the real image has the compact object
ANNULUS_ARCSEC = (60.0, 140.0)


def texture(counts, mask, box):
    """Fractional surface-brightness fluctuation between ``box`` and ``2 box``.

    Returns ``(amplitude, snr)`` where ``snr`` is the ratio of the structure
    variance to the Poisson variance that was subtracted -- below ~3 the
    amplitude is the difference of two nearly equal numbers and means little.
    """
    a1, a2 = box ** 2, (2 * box) ** 2
    d = uniform_filter(counts, box) - uniform_filter(counts, 2 * box)
    m = counts[mask].mean()
    var_noise = m * (1.0 / a1 - 1.0 / a2)
    var = d[mask].var() - var_noise
    return np.sqrt(max(var, 0.0)) / m, max(var, 0.0) / var_noise


def annulus(shape, lo, hi):
    n = shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c, yy - c) * PIXEL_ARCSEC
    return (r > lo) & (r < hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("synthetic", help="a casa_observe.py *_synimg.npz")
    ap.add_argument("--compare", default="2004", help="real epoch label")
    ap.add_argument("--annulus", type=float, nargs=2, default=ANNULUS_ARCSEC,
                    help="radial range (arcsec) the statistic is measured in")
    ap.add_argument("--out", default="casa_morphology", help="figure name stem")
    args = ap.parse_args()

    syn = np.load(args.synthetic)["counts"].astype(np.float64)
    real = np.load(REAL_EPOCH_DIR / f"epoch_{args.compare}.npz")["counts"].astype(np.float64)
    mask = annulus(syn.shape, *args.annulus)

    print(f"[morph] {args.synthetic} vs Chandra {args.compare}, "
          f"{args.annulus[0]:.0f}-{args.annulus[1]:.0f} arcsec annulus")
    print(f"    {'scale':>7} {'synthetic':>10} {'S/N':>6} {'Chandra':>9} "
          f"{'S/N':>6} {'real/syn':>9}")
    scales, ratios, rows = [], [], []
    for box in BOXES:
        s, s_snr = texture(syn, mask, box)
        r, r_snr = texture(real, mask, box)
        scales.append(box * PIXEL_ARCSEC)
        ratios.append(r / max(s, 1e-9))
        rows.append((s, s_snr, r, r_snr))
        flag = "  <- noise-dominated" if min(s_snr, r_snr) < 3.0 else ""
        print(f"    {box * PIXEL_ARCSEC:6.1f}\" {s:10.3f} {s_snr:6.1f} {r:9.3f} "
              f"{r_snr:6.1f} {r / max(s, 1e-9):9.2f}{flag}")

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    ax.loglog(scales, [x[0] for x in rows], "o-", color="tab:red",
              label="astronomix (synthetic)")
    ax.loglog(scales, [x[2] for x in rows], "s--", color="k",
              label=f"Chandra {args.compare}")
    ax.set(xlabel="angular scale [arcsec]",
           ylabel=r"fractional fluctuation $\sigma_I/I$",
           title="surface-brightness structure, scale by scale")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    out = FIGURES_DIR / f"{args.out}.png"
    fig.savefig(out, dpi=150)
    print(f"\n[morph] saved {out}")


if __name__ == "__main__":
    main()
