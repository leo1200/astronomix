"""Is the model's structure real, or just a power spectrum?

``casa_morphology.py`` reports two topology statistics, and both improved as the
grid was refined -- chi from 2.42 to 0.84, coherence from 0.902 to 0.693. That
was read as the model converging on Cas A's structure. It might not be.

**Finer structure lowers coherence for an almost geometric reason:** once
features are smaller than the smoothing window, more orientations fall inside
it. So a model that merely makes FINER ISOTROPIC FOAM moves the same way as one
that makes a filamentary web, and neither statistic can tell them apart on its
own. Separating the two decides whether refining the grid is doing real work,
which in turn decides whether the expensive explosion-phase modelling is needed.

The control here is a **phase-randomised null**. For each image:

    1. divide out the smooth azimuthal profile, leaving the contrast field;
    2. Fourier transform it, replace every phase with a uniform random one, and
       transform back -- this preserves the power spectrum EXACTLY while
       destroying all topology, giving a Gaussian random field with identical
       fluctuation power at every scale;
    3. rebuild the image on the same radial profile and rescale to the same
       counts.

Any statistic computed on the null is therefore what that image would score if
its structure were nothing but its power spectrum. The quantity that matters is
the **excess** of the real statistic over its null:

    excess = statistic(image) - statistic(phase-randomised image)

A pure Gaussian foam has zero excess by construction, however fine it is. A
filamentary web has a large one. So:

* if the models' excess grows toward Chandra's as the grid is refined,
  resolution is generating genuine non-Gaussian structure and is worth pushing;
* if the excess stays flat near zero while the raw statistics march, the
  refinement is only changing the power spectrum -- finer foam -- and the
  missing structure has to come from somewhere else, i.e. the explosion.

Usage (CPU)::

    /export/home/lstorcks/xrayobs/bin/python casa_morph_null.py \\
        obs_n128_cd obs_n256_cd obs_n512_cd_adia --compare 2004
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

from casa_morphology import (
    ANNULUS_ARCSEC,
    PIXEL_ARCSEC,
    REAL_EPOCH_DIR,
    annulus,
    euler_density,
    filamentarity,
    poisson_match,
    texture,
)


def radial_profile(counts, n_bins=200):
    """The smooth azimuthal profile S(r), interpolated back onto the image.

    Phase randomisation must not be applied to the whole image: the remnant's
    shell is itself encoded in the phases, so randomising them would leave a
    structureless blob and the annulus statistics would be meaningless. Divide
    the profile out first, randomise only the FLUCTUATIONS, and put the profile
    back afterwards.
    """
    n = counts.shape[0]
    c = n // 2
    yy, xx = np.mgrid[:n, :n]
    r = np.hypot(xx - c, yy - c)
    edges = np.linspace(0.0, r.max(), n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    tot = np.bincount(idx.ravel(), weights=counts.ravel(), minlength=n_bins)
    cnt = np.bincount(idx.ravel(), minlength=n_bins).astype(float)
    prof = np.divide(tot, cnt, out=np.zeros(n_bins), where=cnt > 0)
    rc = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(r, rc, prof)


def phase_randomise(counts, rng):
    """Same power spectrum, no topology.

    Uses ``rfft2``/``irfft2`` so the Hermitian symmetry is enforced by
    construction and the result is exactly real -- randomising a full complex
    ``fft2`` by hand and hoping the conjugate pairs still match is how this
    goes subtly wrong.
    """
    smooth = radial_profile(counts)
    floor = max(smooth[smooth > 0].min() if np.any(smooth > 0) else 1e-6, 1e-6)
    contrast = counts / np.maximum(smooth, floor) - 1.0

    spec = np.fft.rfft2(contrast)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=spec.shape)
    spec = np.abs(spec) * np.exp(1j * phase)
    spec[0, 0] = np.abs(np.fft.rfft2(contrast)[0, 0])      # keep the mean real
    rand = np.fft.irfft2(spec, s=contrast.shape)

    out = np.maximum(smooth * (1.0 + rand), 0.0)
    scale = counts.sum() / max(out.sum(), 1e-30)
    return out * scale


def score(img, mask, box_chi=5, box_coh=9):
    """(chi, coherence, sigma_I/I) at the scales the ladder is quoted on."""
    return (euler_density(img, mask, box_chi),
            filamentarity(img, mask, box_coh),
            texture(img, mask, box_chi)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stems", nargs="+", help="obs_* stems (without _synimg.npz)")
    ap.add_argument("--compare", default="2004", help="real epoch label")
    ap.add_argument("--realisations", type=int, default=5,
                    help="phase-randomised realisations to average the null over")
    ap.add_argument("--annulus", type=float, nargs=2, default=ANNULUS_ARCSEC)
    args = ap.parse_args()

    real = np.load(REAL_EPOCH_DIR / f"epoch_{args.compare}.npz")["counts"].astype(float)
    rng = np.random.default_rng(12345)

    print(f"[null] phase-randomised control, {args.realisations} realisations, "
          f"{args.annulus[0]:.0f}-{args.annulus[1]:.0f} arcsec annulus")
    print(f"[null] chi at 2.5\", coherence at 4.4\"; EXCESS = image - its own null")
    print(f"    {'image':>18} {'chi':>7} {'chi_null':>9} {'d_chi':>7} "
          f"{'coh':>7} {'coh_null':>9} {'d_coh':>7}")

    # Load everything first, then thin EVERY image -- Chandra included -- to the
    # faintest one. Without this the real image sits at 143 ks and the models at
    # 20 ks, and the nulls come out at chi_null = 6.6 against 16.8: the
    # comparison would then be measuring the exposure difference, not the
    # structure. A phase-randomised null inherits the image's noise, so matched
    # noise is not a refinement here, it is the whole basis of the comparison.
    images = [(stem.replace("obs_", ""),
               np.load(f"{stem}_synimg.npz")["counts"].astype(float))
              for stem in args.stems]
    images.append((f"Chandra {args.compare}", real))
    mask0 = annulus(images[0][1].shape, *args.annulus)
    target = min(float(img[mask0].sum()) for _, img in images)
    print(f"[null] all images thinned to {target:.3g} counts in the annulus")

    rows = []
    for label, raw in images:
        mask = annulus(raw.shape, *args.annulus)
        faint = np.zeros_like(raw)
        faint[mask0] = target / max(float(raw[mask0].sum()), 1e-30) * raw[mask0]
        img, _ = poisson_match(raw, faint, mask, seed=1)
        chi, coh, sig = score(img, mask)

        nulls = np.array([score(phase_randomise(img, rng), mask)
                          for _ in range(args.realisations)])
        chi_n, coh_n = nulls[:, 0].mean(), nulls[:, 1].mean()
        rows.append((label, chi, chi_n, coh, coh_n, sig))
        print(f"    {label:>18} {chi:7.2f} {chi_n:9.2f} {chi - chi_n:+7.2f} "
              f"{coh:7.3f} {coh_n:9.3f} {coh - coh_n:+7.3f}")

    print()
    print("[null] READ IT LIKE THIS: a Gaussian foam has zero excess however "
          "fine it is.")
    print("       If the models' excess grows toward Chandra's with resolution, "
          "the grid is")
    print("       making real non-Gaussian structure. If it stays flat while "
          "the raw numbers")
    print("       march, refinement is only changing the power spectrum.")


if __name__ == "__main__":
    main()
