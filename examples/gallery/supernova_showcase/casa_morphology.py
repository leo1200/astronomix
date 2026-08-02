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
from scipy import ndimage
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


# =============================================================================
# ============ ↓ Topology: is it a web, or a few sharp edges? ↓ ===============
# =============================================================================
# The band-pass RMS above measures HOW MUCH fluctuation there is at a scale. It
# cannot say what SHAPE it has, and that turned out to matter: an edge is
# scale-free, so a handful of large features with sharp boundaries puts power
# into every octave and scores like a filamentary web. Measured, the 192^3 model
# scores BEST on sigma_I/I (1.46x) while looking like a smooth blob with two
# arcs across it, and the 512^3 contact-discontinuity model scores worse (1.80x)
# while carrying the fine cellular texture Cas A actually shows. A statistic
# that inverts the visual ordering is not measuring the thing we are chasing.
#
# These two add the missing axis. Both are pure topology/shape and carry no
# amplitude information, so they are complementary to the RMS rather than a
# replacement for it.


def poisson_match(real, syn, mask, seed=0):
    """Thin ``real`` so it carries the same counts -- and noise -- as ``syn``.

    Binomially thinning a Poisson image with probability ``p`` gives EXACTLY a
    Poisson image of mean ``p * lambda``, so matching the total counts in the
    annulus makes the two images statistically identical in their noise. That
    is what lets the statistics below be compared at all: unlike the band-pass
    RMS, a threshold count or an Euler characteristic has no analytic Poisson
    correction to subtract, and comparing a 143 ks image with a 20 ks one
    without matching would measure the exposure difference, not the remnant.
    """
    n_real, n_syn = float(real[mask].sum()), float(syn[mask].sum())
    if n_real <= n_syn:
        return real, 1.0
    p = n_syn / n_real
    rng = np.random.default_rng(seed)
    return rng.binomial(np.rint(real).astype(np.int64), p).astype(np.float64), p


def contrast_map(counts, box):
    """Local contrast at scale ``box``: the image over its own background."""
    band = uniform_filter(counts, box)
    bg = uniform_filter(counts, 4 * box)
    return band / np.maximum(bg, 1e-12) - 1.0


def euler_density(counts, mask, box, area_fraction=0.25):
    """Euler characteristic (components - holes) of the brightest ``area_fraction``.

    Thresholding at a fixed AREA FRACTION rather than a fixed contrast is what
    makes this a pure topology measure: both images light up the same number of
    pixels, so the only thing left to differ is how those pixels are connected.
    A web of filaments encircles voids and is dominated by holes (chi < 0); a
    few compact blobs or arcs are dominated by components (chi > 0).

    Returned per 1000 mask pixels, so it does not scale with the aperture.
    """
    c = contrast_map(counts, box)
    thr = np.quantile(c[mask], 1.0 - area_fraction)
    binary = (c > thr) & mask

    n_obj = ndimage.label(binary)[1]
    lab, n_comp = ndimage.label((~binary) & mask)
    # a hole is a background component that does not touch the outside of the
    # annulus; anything reaching the rim is the surrounding field, not a hole
    rim = ndimage.binary_dilation(~mask) & (lab > 0)
    n_holes = n_comp - len(np.unique(lab[rim]))
    return 1000.0 * (n_obj - n_holes) / float(mask.sum())


def filamentarity(counts, mask, box):
    """Mean structure-tensor coherence: how ORDERED the structure is locally.

    The gradient structure tensor of a single ridge has one large and one small
    eigenvalue; the coherence ``((l1 - l2) / (l1 + l2))^2`` is 1 for a clean
    ridge and 0 when several orientations meet inside the smoothing window. It
    is weighted by gradient power so flat, noise-dominated regions do not vote.

    **Read the direction carefully: HIGHER IS NOT MORE CAS A-LIKE.** Chandra
    measures 0.54 at 4.4 arcsec and every model here sits at 0.65-0.92. A few
    big clean arcs are locally very coherent; a dense tangle of filaments
    crossing at all orientations is not. So this discriminates in the same
    direction as the Euler characteristic -- towards the real remnant being
    made of many crossing structures rather than a few smooth ones -- and the
    model to prefer is the one with the LOWEST coherence, not the highest.
    """
    img = uniform_filter(counts, box)
    gy, gx = np.gradient(img)
    w = max(box, 2)
    jxx = uniform_filter(gx * gx, w)
    jyy = uniform_filter(gy * gy, w)
    jxy = uniform_filter(gx * gy, w)
    tr = jxx + jyy
    det = jxx * jyy - jxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 - 4.0 * det, 0.0))
    # Guard against the flat regions, where both eigenvalues vanish and the
    # ratio is 0/0. An absolute floor is not enough: with tr ~ 1e-20 and a
    # 1e-30 clamp the "coherence" comes out at 1e20 and the weighted mean
    # returns -4.8e13, which is what this guard exists to stop. The floor has
    # to be RELATIVE to the image's own gradient power.
    eps = 1e-6 * float(np.median(tr[mask]) + 1e-30)
    coh = np.where(tr > eps, (disc / np.maximum(tr, eps)) ** 2, 0.0)
    return float(np.average(coh[mask], weights=tr[mask]))


# =============================================================================
# ============ ↑ Topology: is it a web, or a few sharp edges? ↑ ===============
# =============================================================================


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

    # ---- topology, on a Poisson-matched pair -------------------------------
    real_m, p_thin = poisson_match(real, syn, mask)
    print(f"\n[morph] topology (real thinned by {p_thin:.3f} to match the "
          f"synthetic's {syn[mask].sum():.3g} counts, so the noise is identical)")
    print(f"    {'scale':>7} {'chi_syn':>9} {'chi_real':>9} "
          f"{'filam_syn':>10} {'filam_real':>11}")
    for box in (3, 5, 9, 17):
        cs = euler_density(syn, mask, box)
        cr = euler_density(real_m, mask, box)
        fs = filamentarity(syn, mask, box)
        fr = filamentarity(real_m, mask, box)
        print(f"    {box * PIXEL_ARCSEC:6.1f}\" {cs:9.2f} {cr:9.2f} "
              f"{fs:10.3f} {fr:11.3f}")
    print("    chi < 0 means holes dominate (a web); chi > 0 means components "
          "dominate (blobs/arcs)")

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
