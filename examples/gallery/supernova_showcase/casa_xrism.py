"""
The model in XRISM's plane: per-element-group kT_e, n_e t, velocity and T_i.

Every spectral score in this study so far has been a **band ratio** -- six
numbers formed by integrating the whole remnant's emission through the ACIS
response. That is the right test of a *spectrum*, and it is a weak test of
*physics*, because a band ratio is one scalar per band and the model has many
ways to produce it. Four candidate explanations for the residual were falsified
against band ratios alone (structure, cooling, the dust halo, T_e) without any
of them being localised.

XRISM/Resolve changed what is available. Vink et al. (2026, arXiv:2602.06952)
fit **two pure-metal components per spatial region** -- one for the
intermediate-mass elements, one for the iron group -- and report, for each, the
electron temperature, the ionization age, the Doppler shift and the line width,
across the remnant. Those are precisely the quantities :mod:`_plasma` and
:mod:`_nei` already compute per cell. So the comparison can be made in the plane
the observation actually constrains, per element group, instead of after
integrating the distinction away.

WHY THIS IS A SHARPER TEST, IN ONE OBSERVATION
----------------------------------------------
``casa_plasma.py`` reports an ``n_e^2``-weighted ``kT_e`` of 3.05 keV. On the
same state it also reports an ``n_e^2``-weighted ``mu_e`` of **1.30**, against
2.0 for metal ejecta -- i.e. that average is dominated by shocked *wind*, not by
the ejecta whose lines XRISM measures. The single number is therefore not the
quantity the observation reports, and weighting by the emitting ions of one
element group is not a refinement of it but a different measurement.

WHAT IS COMPARED, AND WHY EACH CHOICE
-------------------------------------
**Weight.** A line's brightness goes as ``n_e`` times the density of the ion
that emits it, so the default weight for element ``el`` is
``n_e * n_el * f_line(el)``, with ``f_line`` the NEI population of the charge
states that carry its K-shell lines (:data:`LINE_CHARGE_MIN`). ``--weight
element`` drops the ionization factor and ``--weight ne2`` reproduces
``casa_plasma.py``'s convention; the three together show how much of any
disagreement is the weighting rather than the physics.

**Spatial binning is not optional.** XRISM's ranges are ranges *across
regions*, one value per 30" Resolve pixel, not the spread within a single
spectrum. Comparing our single global average against their range would compare
a mean to a distribution and could only ever "agree". So the model is projected
along the line of sight and binned onto pixels of the same angular size, each
pixel is reduced exactly as a fitted spectrum would be, and the RANGE ACROSS
PIXELS is what gets compared.

**Per-species ion temperatures.** The solver carries one temperature;
:func:`_plasma.species_ion_temperature` inverts the mean for one species under
mass-proportional shock heating. This is a genuine prediction of the calibrated
dynamics rather than a fit -- the Fe-minus-Si difference is essentially
``(3/16)(m_Fe - m_Si) v_s^2``, which is 176 keV at an 1800 km/s reverse shock --
and it is an UPPER bound, because nothing here relaxes the species towards each
other.

**The correlation sign is the discriminating test.** XRISM finds ``n_e t``
*anticorrelated* with ``kT_e``, robustly and against what a standard NEI picture
predicts, and reads it as dense clumps: denser gas is more ionized (higher
``n_e t``) *and* cooler, because the shock transmitted into it is slower. A
model whose density contrast is too low has no mechanism for that, so the sign
of this correlation is a pass/fail that no band ratio can express.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not fold through a response, so it is not a spectral fit and the numbers
are not "what XRISM would measure" -- they are the mass- and emission-weighted
truth of the model in the same variables. A fitted spectrum weights differently
(and XRISM's own two-component fit is itself a simplification of a continuous
distribution), so treat agreement inside a factor as agreement and do not read
significance into the third digit.

Usage (either environment -- this is numpy only)::

    ./run.sh casa_xrism.py orl_n256_final.npz
    /export/home/lstorcks/xrayobs/bin/python casa_xrism.py <state> --weight element
"""

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# the shared plasma physics
from _plasma import (
    ATOMIC,
    CODE_DENSITY,
    CODE_VELOCITY,
    KEV_IN_K,
    K_B,
    M_P,
    TRACER_SPLIT_PRESETS,
    load_diagnostic_state,
    plasma_state,
    set_tracer_split,
    species_ion_temperature,
    thermal_line_width,
    tracer_split_report,
)
import _nei
import _subgrid

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# ============ ↓ What XRISM measured ↓ ========================================
# =============================================================================
# Vink et al. 2026 (arXiv:2602.06952), Resolve, two pure-metal pshock components
# plus a power law, fitted per 30" pixel over two PV pointings. Ranges are
# ACROSS PIXELS, which is why the model has to be binned the same way.
#
# The Fe-group kT_e upper bound is quoted with a caveat in the paper itself:
# some pixels reach the 10 keV prior limit in regions where the synchrotron
# dominates, so 8.4 is "the largest well-constrained value", not a hard ceiling.
DISTANCE_KPC = 3.4                      # Reed et al. 1995, as casa_observe.py
RESOLVE_PIXEL_ARCSEC = 30.0             # Resolve is a 6x6 array of 30" pixels

#: element groups exactly as UltraSPEX splits them
GROUPS = {
    "IME": ("Si", "S", "Ar", "Ca"),
    "IGE": ("Fe",),                     # Ni is not carried as a scalar
}

XRISM = {
    "IME": dict(kT_e=(1.3, 2.1), net=(1.0e11, 3.4e11),
                v_shift=(-1250.0, 2000.0), sigma_v_max=2200.0),
    "IGE": dict(kT_e=(2.4, 8.4), net=(0.8e11, 3.0e11),
                v_shift=(-1700.0, 2400.0), sigma_v_max=3700.0),
}
#: T_i(Fe) - T_i(Si) from the THERMAL part of the line widths, two peripheral
#: regions where the kinematic broadening is smallest: (value, uncertainty) keV
XRISM_DELTA_TI_KEV = {"NW": (300.0, 180.0), "SE": (150.0, 60.0)}

#: Lowest charge state counted as carrying the element's K-shell line emission.
#:
#: For the intermediate-mass elements the bright K lines are the He-like triplet
#: and the H-like doublet, so only the top two ionization stages matter and
#: ``Z - 2`` is exact rather than a choice. Iron is different and the difference
#: is physical, not a convention: at ``kT_e ~ 3 keV`` and ``n_e t ~ 2e11`` its
#: mean charge is ~22, so requiring He-like Fe would discard essentially all of
#: it. Fe-K emission is produced from about Fe XVIII upward, hence 17. The
#: sensitivity to that choice is reported at the end of the run.
LINE_CHARGE_MIN = {"Si": 12, "S": 14, "Ar": 16, "Ca": 18, "Fe": 17}

#: contrasts for --subgrid-scan. 1.0 is the no-sub-grid control and must be
#: first, so every row can be read against it. The upper end is deliberately
#: past where the model stops improving: a scan that stops at the best value
#: cannot show that it IS the best value.
SCAN_CHI = (1.0, 1.5, 2.3, 4.0, 8.0, 16.0)
# =============================================================================
# ============ ↑ What XRISM measured ↑ ========================================
# =============================================================================


def arcsec_per_pc(distance_kpc=DISTANCE_KPC):
    """Angular size of one parsec at the remnant [arcsec/pc]."""
    return 206264.806 / (distance_kpc * 1000.0)


def line_emitting_fraction(element, kT_e, net, table):
    """NEI fraction of ``element`` in the charge states that emit its K lines.

    Interpolation is linear in ``(log kT_e, log n_e t)``, so summing the charge
    states in the TABLE and then interpolating one scalar is identical to
    interpolating every stage and summing afterwards -- and it is the difference
    between 500 MB and 14 GB of temporaries at 512^3. The sum is taken first for
    that reason and for no other.
    """
    kt_grid, net_grid, f = table
    z_min = LINE_CHARGE_MIN[element]
    g = f[element][..., z_min:].sum(axis=-1)[..., None]     # (nkt, nnet, 1)
    return _nei.interpolate_fractions(g, kt_grid, net_grid, kT_e, net)[0]


def group_weights(ps, group, table, *, mode="line"):
    """Emission weight per cell for one element group, and its composition.

    Returns ``(w, contributions)`` -- the weight array and, per element, the
    share of the total weight it carries, which is what says whether "IME" means
    silicon or has quietly become calcium.
    """
    n_e = ps["n_e"]
    kT_e_keV = ps["T_e"] / KEV_IN_K
    rho = ps["rho_cgs"]

    w = np.zeros_like(n_e)
    contributions = {}
    for el in GROUPS[group]:
        X_el = ps["X"].get(el)
        if X_el is None or np.ndim(X_el) == 0:
            continue
        n_el = rho * X_el / (ATOMIC[el][0] * M_P)
        if mode == "line":
            n_el = n_el * line_emitting_fraction(el, kT_e_keV, ps["net"], table)
        w_el = n_e * n_el
        w = w + w_el
        contributions[el] = float(w_el.sum())
    total = sum(contributions.values()) or 1.0
    return w, {el: v / total for el, v in contributions.items()}


def project_sums(w, wv, *, num_cells, box_pc, los="y",
                 pixel_arcsec=RESOLVE_PIXEL_ARCSEC):
    """Weighted means on sky pixels, from PRE-SUMMED numerators.

    ``w`` is the total weight per cell and ``wv`` a dict of ``w * value`` fields
    already accumulated over whatever phases contribute. Taking the numerators
    pre-summed is what lets the sub-grid split add a second phase without this
    function knowing about phases: two thermal components in one cell contribute
    to one spectrum, and a fit to that spectrum returns one number for the
    mixture, which is exactly a weight-summed mean.

    The line of sight is summed over and the two sky axes are block-averaged --
    the formation of a spatially resolved spectrum, in two lines.

    Returns ``(pixel_weight, {name: 2D weighted mean}, cells_per_pixel)``.
    """
    los_axis = {"x": 0, "y": 1, "z": 2}[los]
    dx_pc = box_pc / num_cells
    cells_per_pixel = max(1, int(round(pixel_arcsec / (arcsec_per_pc() * dx_pc))))
    # trim to a whole number of pixels rather than padding: a partial pixel would
    # carry less signal and read as a genuinely fainter region
    n_pix = num_cells // cells_per_pixel
    keep = n_pix * cells_per_pixel

    def block(a):
        a = np.sum(a, axis=los_axis)
        sl = tuple(slice(0, keep) for _ in range(2))
        a = a[sl]
        return a.reshape(n_pix, cells_per_pixel, n_pix, cells_per_pixel).sum(axis=(1, 3))

    W = block(w)
    out = {name: np.where(W > 0, block(v) / np.maximum(W, 1e-300), np.nan)
           for name, v in wv.items()}
    return W, out, cells_per_pixel


def project_to_pixels(w, values, **kw):
    """``project_sums`` for the single-phase case, multiplying the values here."""
    return project_sums(w, {k: w * v for k, v in values.items()}, **kw)


def pixel_velocity_moments(w, wv, wv2, *, num_cells, box_pc, los, pixel_arcsec):
    """Mean and 1-sigma spread of the line-of-sight velocity, per sky pixel.

    The spread is the KINEMATIC part of an observed line width: the emission in
    one pixel comes from many parcels at different line-of-sight velocities, and
    a fitted Gaussian absorbs that spread into its width. Thermal broadening is
    reported separately (:func:`_plasma.thermal_line_width`) because under
    mass-proportional heating it is the same for every species and so cannot
    explain a difference between them.
    """
    W, mom, _ = project_sums(
        w, {"v": wv, "v2": wv2}, num_cells=num_cells, box_pc=box_pc,
        los=los, pixel_arcsec=pixel_arcsec)
    mean = mom["v"]
    var = np.maximum(mom["v2"] - mean ** 2, 0.0)
    return W, mean, np.sqrt(var)


def subgrid_phases(fields, *, chi, f_mass, net_mode):
    """The state, re-read as one or two phases -- a list of ``(name, fields, f_vol)``.

    Pressure equilibrium does the work here and is the reason this is only a few
    lines: a phase differs from its cell ONLY in density (and, through
    ``net_mode``, in how long it has been shocked). So ``press`` is copied
    unchanged and :func:`_plasma.plasma_state` recomputes ``T``, ``T_e``, ``T_i``,
    ``n_e`` and ``n_e t`` for the phase from its own density, with no separate
    code path and no chance of the phase temperature disagreeing with the
    equation of state that produced the cell one.

    The factors come from :func:`_subgrid.phase_factors` and NOT from a local
    copy: ``density_time`` is ``rho * t``, so its factor is not free once the
    density and time factors are chosen, and an earlier local version broke that
    tie -- making two of the three ``net_mode`` values identical while the scan
    claimed to be bounding the choice.
    """
    if chi is None:
        return [("cell", fields, 1.0)]

    out = []
    for name, (rho_f, t_f, net_f, f_vol) in _subgrid.phase_factors(
            chi, f_mass, net_mode).items():
        fp = dict(fields)
        fp["rho"] = fields["rho"] * rho_f
        fp["time_since_shock"] = fields["time_since_shock"] * t_f
        fp["density_time"] = fields["density_time"] * net_f
        out.append((name, fp, f_vol))
    return out


def bright_pixels(W, *, fraction=0.05):
    """Pixels carrying enough emission that a fit would converge there.

    A real analysis discards pixels without the counts to constrain two thermal
    components; keeping them here would extend every model range with values
    from the faint outskirts and make the comparison look worse than it is (or,
    just as bad, look better, since the outskirts are cool).
    """
    finite = W[np.isfinite(W) & (W > 0)]
    if finite.size == 0:
        return np.zeros_like(W, dtype=bool)
    return W >= fraction * finite.max()


def spearman(x, y):
    """Rank correlation, without scipy (the xrayobs venv has it, astx may not)."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan
    def rank(a):
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # average ties, otherwise a plateau biases the coefficient
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / counts)[inv]
    rx, ry = rank(x[ok]), rank(y[ok])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else np.nan


def clumping_factor(sum_w, sum_wn, sum_wn2):
    """``<n_e^2> / <n_e>^2`` over the emitting gas -- the model's own contrast.

    This is the number the sub-grid clumping question is about, measured rather
    than assumed. A log-normal of width ``sigma_ln`` gives ``exp(sigma_ln^2)``;
    a two-phase medium at contrast ``chi`` and mass fraction ``f_m`` gives
    :func:`_subgrid.clumping_factor`. Laming & Hwang's and XRISM's one-zone
    ``chi ~ 10-100`` correspond to values of order 10-25, so the gap between this
    number and those is an upper bound on the missing emissivity -- an upper
    bound, because a one-zone analysis has to attribute to ``chi`` all of the
    structure that this calculation already resolves.

    Takes accumulated SUMS rather than arrays so it can be built up across
    sub-grid phases without ever holding a concatenation of them.
    """
    if sum_w <= 0:
        return np.nan
    mean = sum_wn / sum_w
    return float((sum_wn2 / sum_w) / max(mean ** 2, 1e-300))


def fmt_range(lo, hi, fmt="{:.2f}"):
    return f"{fmt.format(lo)}-{fmt.format(hi)}"


def verdict(model_lo, model_hi, obs_lo, obs_hi):
    """Do the two ranges overlap, and if not by how much."""
    if model_hi < obs_lo:
        return f"LOW by {obs_lo / max(model_hi, 1e-300):.2f}x"
    if model_lo > obs_hi:
        return f"HIGH by {model_lo / max(obs_hi, 1e-300):.2f}x"
    return "overlaps"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("state", help="a --save-state npz carrying the shock history")
    ap.add_argument("--weight", default="line", choices=("line", "element", "ne2"),
                    help="emission weight: LINE uses the NEI population of the "
                         "K-shell-emitting charge states (default), ELEMENT "
                         "drops the ionization factor, NE2 reproduces "
                         "casa_plasma.py's n_e^2 convention")
    ap.add_argument("--los", default="y", choices=("x", "y", "z"),
                    help="line of sight; y matches casa_observe.py and Orlando")
    ap.add_argument("--pixel-arcsec", type=float, default=RESOLVE_PIXEL_ARCSEC,
                    help="sky pixel size; 30 is Resolve's")
    ap.add_argument("--bright-fraction", type=float, default=0.05,
                    help="keep pixels above this fraction of the brightest")
    ap.add_argument("--te-model", default="ghavamian",
                    help="passed to _plasma.plasma_state")
    ap.add_argument("--kt-e-shock", type=float, default=0.3)
    ap.add_argument("--subgrid-chi", type=float, nargs="+", default=None,
                    metavar="CHI",
                    help="re-read every cell as a two-phase medium of density "
                         "contrast CHI before measuring, via _subgrid. This is "
                         "an INTERPRETATION LAYER, not simulated structure; it "
                         "is here so CHI can be calibrated against XRISM in "
                         "minutes instead of against a 45-minute observation. "
                         "Give TWO values to use a different contrast per "
                         f"element group, in the order {tuple(GROUPS)} -- XRISM "
                         "infers ~10 for the iron group and ~100 for the "
                         "intermediate-mass elements, i.e. the contrast is "
                         "composition-dependent, which one number cannot say")
    ap.add_argument("--subgrid-fmass", type=float,
                    default=_subgrid.F_MASS_DEFAULT, metavar="F",
                    help="mass fraction of the dense phase")
    ap.add_argument("--subgrid-net-mode", default=_subgrid.NET_MODE_DEFAULT,
                    choices=_subgrid.NET_MODES,
                    help="how the dense phase's ionization age follows from the "
                         "cell's -- see _subgrid, and note that n_e t is already "
                         "at the TOP of XRISM's range with no boost, which "
                         "bounds this choice")
    ap.add_argument("--subgrid-scan", action="store_true",
                    help="scan CHI and report, for each, whether the "
                         "line-emission-weighted kT_e lands in XRISM's box. "
                         "This is the calibration; run it before observing")
    ap.add_argument("--tracer-split", default="hwang_laming",
                    choices=sorted(TRACER_SPLIT_PRESETS),
                    help="see _plasma.TRACER_SPLIT_PRESETS; the two presets\n"
                         "disagree, and every mass and abundance below depends\n"
                         "on which is used")
    ap.add_argument("--out", default="casa_xrism", help="figure name stem")
    args = ap.parse_args()

    set_tracer_split(args.tracer_split)
    fields, meta = load_diagnostic_state(args.state)
    n, box, age = meta["num_cells"], meta["box"], meta["age"]
    print(f"\n[xrism] {args.state}: {n}^3, age {age:.0f} yr")
    if "argv" in meta:
        print(f"[xrism] state was made by: {meta['argv']}")

    if f"v{args.los}" not in fields:
        raise SystemExit(f"this state carries no v{args.los}; "
                         "the velocity comparison needs it")
    v_los = fields[f"v{args.los}"] * CODE_VELOCITY / 1e5        # km/s

    print(f"[xrism] weight = {args.weight}, LOS = {args.los}, "
          f"pixels of {args.pixel_arcsec:.0f}\", tracer split "
          f"'{args.tracer_split}'")
    print(tracer_split_report())

    if args.subgrid_scan:
        scan(fields, meta, args, v_los, n, box)
        return

    results = measure(fields, args, v_los, n, box,
                      chi=args.subgrid_chi, f_mass=args.subgrid_fmass,
                      verbose=True)
    if args.subgrid_chi is not None and len(args.subgrid_chi) > 1:
        print("[xrism] per-group contrast: "
              + ", ".join(f"{g} chi = {results[g]['chi']:.1f}"
                          for g in results))
    if len(results) < 2:
        raise SystemExit("both element groups are needed for the comparison")
    report(results, args)
    make_figure(results, args, meta)


def chi_for_group(chi, group):
    """Resolve ``--subgrid-chi`` (scalar, or one value per group) for one group.

    XRISM infers a contrast of ~10 for the iron group and up to ~100 for the
    intermediate-mass elements, so a single number is a known simplification
    rather than a modelling choice. It is cheap to relax HERE because the
    measurement already loops over groups -- but note what it does and does not
    mean: this applies a different contrast to the same cells depending on which
    element's lines are being weighted, which is a statement about where each
    species SITS, not a second hydrodynamic phase. Two genuinely different
    contrasts in one cell would need the composition to be spatially separated
    below the grid scale, which nothing here models.
    """
    if chi is None:
        return None
    if len(chi) == 1:
        return float(chi[0])
    if len(chi) != len(GROUPS):
        raise SystemExit(f"--subgrid-chi takes 1 or {len(GROUPS)} values "
                         f"({tuple(GROUPS)}), got {len(chi)}")
    return float(chi[list(GROUPS).index(group)])


def measure(fields, args, v_los, n, box, *, chi, f_mass, verbose=False):
    """Everything XRISM reports, for one ``(chi, f_mass)``.

    Separated from :func:`main` so the scan can call it repeatedly -- the
    calibration of ``chi`` has to happen here, in minutes, and not against a
    45-minute observation.
    """
    # chi may differ per element group, so the phase decomposition does too.
    # Built once per UNIQUE value rather than once per group: the two groups
    # usually share a chi, and a plasma state is ~5 GB at 256^3.
    chi_by_group = {g: chi_for_group(chi, g) for g in GROUPS}
    table = _nei.load_table() if args.weight == "line" else None

    states_by_chi = {}
    for c in dict.fromkeys(chi_by_group.values()):       # unique, order-preserving
        if verbose and c is not None:
            print(_subgrid.describe(c, f_mass, net_mode=args.subgrid_net_mode))
        states = []
        for name, fp, f_vol in subgrid_phases(
                fields, chi=c, f_mass=f_mass, net_mode=args.subgrid_net_mode):
            ps = plasma_state(fp, te_model=args.te_model,
                              kT_e_shock_keV=args.kt_e_shock)
            if not ps["info"]["composition_tracked"]:
                raise SystemExit(
                    "this state carries no composition scalars, so it has no "
                    "element groups to split -- rerun casa_orlando.py with "
                    "--composition")
            if ps["net"] is None:
                raise SystemExit("no ionization age in this state")
            ps["rho_cgs"] = fp["rho"] * CODE_DENSITY
            states.append((name, ps, f_vol))
            if verbose and c is not None:
                kt = ps["T_e"] / KEV_IN_K
                m = ps["shocked"]
                print(f"[xrism]   phase '{name}': {100 * f_vol:.1f}% of the "
                      f"volume, median shocked kT_e = {np.median(kt[m]):.2f} "
                      f"keV, n_e t = {np.median(ps['net'][m]):.2e}")
        states_by_chi[c] = states

    # ---- per group, accumulated over phases -------------------------------
    # A pixel's spectrum is the SUM of what every phase along the ray emits, and
    # a two-component fit to it returns one number per component. So the phases
    # are combined by summing w and w*value BEFORE projecting -- see project_sums.
    results = {}
    for group in GROUPS:
        el_ref = "Si" if group == "IME" else "Fe"
        states = states_by_chi[chi_by_group[group]]
        w_tot = None
        wv = {}
        contrib = {}
        clump_num = clump_den = clump_sq = 0.0
        g_num = {k: 0.0 for k in ("kT_e", "net", "kT_s")}
        g_den = 0.0
        for name, ps, f_vol in states:
            if args.weight == "ne2":
                w = ps["n_e"] ** 2
                c = {"(n_e^2, no composition)": 1.0}
            else:
                w, c = group_weights(ps, group, table, mode=args.weight)
            # unshocked gas has no line emission and no measured temperature
            w = f_vol * np.where(ps["shocked"] & np.isfinite(w), w, 0.0)
            for el, v in c.items():
                contrib[el] = contrib.get(el, 0.0) + v * float(w.sum())
            vals = {"kT_e": ps["T_e"] / KEV_IN_K,
                    "net": ps["net"],
                    "kT_s": species_ion_temperature(
                        ps["T_i"], ps["moments"]["mu_i"], el_ref) / KEV_IN_K,
                    "v": v_los, "v2": v_los ** 2}
            w_tot = w if w_tot is None else w_tot + w
            for k, v in vals.items():
                wv[k] = wv.get(k, 0.0) + w * v
            # <n_e^2>/<n_e>^2 over the emitting gas, across phases
            ok = w > 0
            clump_den += float((w * ps["n_e"])[ok].sum())
            clump_sq += float((w * ps["n_e"] ** 2)[ok].sum())
            clump_num += float(w[ok].sum())
            g_den += float(w.sum())
            for k in g_num:
                g_num[k] += float((w * vals[k]).sum())

        if w_tot is None or not np.any(w_tot > 0):
            print(f"[xrism] {group}: no emitting cells, skipped")
            continue
        total_c = sum(contrib.values()) or 1.0
        contrib = {el: v / total_c for el, v in contrib.items()}

        W, mom, cpp = project_sums(
            w_tot, {k: wv[k] for k in ("kT_e", "net", "kT_s")},
            num_cells=n, box_pc=box, los=args.los,
            pixel_arcsec=args.pixel_arcsec)
        _, v_mean, v_sigma = pixel_velocity_moments(
            w_tot, wv["v"], wv["v2"], num_cells=n, box_pc=box, los=args.los,
            pixel_arcsec=args.pixel_arcsec)
        keep = bright_pixels(W, fraction=args.bright_fraction)

        results[group] = dict(
            W=W, keep=keep, contrib=contrib, el_ref=el_ref,
            cells_per_pixel=cpp,
            kT_e=mom["kT_e"], net=mom["net"], kT_s=mom["kT_s"],
            v_mean=v_mean, v_sigma=v_sigma,
            clumping=clumping_factor(clump_num, clump_den, clump_sq),
            chi=chi_by_group[group],
            **{f"global_{k}": g_num[k] / max(g_den, 1e-300) for k in g_num},
        )

    return results


def report(results, args):
    """Print the comparison. Split out of :func:`main` so the scan can skip it."""
    # ---- report -----------------------------------------------------------
    npix = results["IME"]["keep"]
    print(f"[xrism] {results['IME']['cells_per_pixel']} cells per pixel, "
          f"{int(npix.sum())} of {npix.size} pixels above "
          f"{100 * args.bright_fraction:.0f}% of the brightest")

    for group, r in results.items():
        obs = XRISM[group]
        k = r["keep"]
        print(f"\n[xrism] ==== {group} "
              f"({', '.join(f'{el} {100 * f:.0f}%' for el, f in r['contrib'].items())})"
              f" ====")

        for name, field, target, fmt, unit in (
                ("kT_e", r["kT_e"], obs["kT_e"], "{:.2f}", "keV"),
                ("n_e t", r["net"], obs["net"], "{:.2e}", "cm^-3 s")):
            v = field[k]
            v = v[np.isfinite(v)]
            lo, hi = np.percentile(v, [10, 90])
            print(f"    {name:6s} per pixel  {fmt_range(lo, hi, fmt)} {unit} "
                  f"(10-90%), full {fmt_range(v.min(), v.max(), fmt)}")
            print(f"    {'':6s} XRISM       {fmt_range(*target, fmt)} {unit}"
                  f"   -> {verdict(lo, hi, *target)}")

        v = r["v_mean"][k]
        s = r["v_sigma"][k]
        v, s = v[np.isfinite(v)], s[np.isfinite(s)]
        print(f"    Doppler shift per pixel  {v.min():+.0f} to {v.max():+.0f} km/s"
              f"   XRISM {obs['v_shift'][0]:+.0f} to {obs['v_shift'][1]:+.0f}")
        print(f"    kinematic sigma_v        up to {s.max():.0f} km/s"
              f"   XRISM up to {obs['sigma_v_max']:.0f}")

        rho = spearman(r["kT_e"][k], r["net"][k])
        expected = "NEGATIVE (XRISM, robust)"
        print(f"    Spearman(kT_e, n_e t) across pixels = {rho:+.3f}"
              f"   want {expected}")

        print(f"    <n_e^2>/<n_e>^2 over the emitting gas = {r['clumping']:.2f}"
              "   [a two-phase medium at chi=100, f=0.01 gives ~25]")

    # ---- the two things only the pair can say -----------------------------
    print("\n[xrism] ==== per-species ion temperatures (mass-proportional, "
          "UNRELAXED upper bound) ====")
    kt_si = results["IME"]["global_kT_s"]
    kt_fe = results["IGE"]["global_kT_s"]
    print(f"    kT_i(Si) = {kt_si:.0f} keV (Si-line-weighted), "
          f"kT_i(Fe) = {kt_fe:.0f} keV (Fe-line-weighted)")
    print(f"    difference {kt_fe - kt_si:+.0f} keV")
    for reg, (val, err) in XRISM_DELTA_TI_KEV.items():
        inside = abs(kt_fe - kt_si - val) <= err
        print(f"    XRISM {reg}: {val:.0f} +- {err:.0f} keV"
              f"   -> {'INSIDE' if inside else 'outside'} 1 sigma")
    # Mass-proportional heating gives every species in the SAME parcel the same
    # thermal width. It does NOT follow that the two groups here have the same
    # width, because they are weighted onto different parcels -- and the fact
    # that they come out different is itself the model saying the iron was
    # shocked harder than the silicon, which is what XRISM concludes.
    print("\n[xrism] ==== line widths, the quantity XRISM actually fits ====")
    for group, r in results.items():
        el = r["el_ref"]
        kt_s = r["global_kT_s"]
        s_th = float(thermal_line_width(kt_s * KEV_IN_K, el)) / 1e5
        k = r["keep"]
        s_kin = np.nanmax(r["v_sigma"][k])
        total = np.hypot(s_kin, s_th)
        obs = XRISM[group]["sigma_v_max"]
        print(f"    {group} ({el}): kinematic {s_kin:.0f} + thermal {s_th:.0f} "
              f"-> total {total:.0f} km/s   XRISM up to {obs:.0f} "
              f"({total / obs:.2f}x)")
    print("    (thermal widths differ between the groups only because the two "
          "weights\n     select different parcels; within one parcel "
          "mass-proportional heating\n     makes every species' thermal width "
          "identical)")

    # The ion temperature IS a shock-velocity measurement, so say so in km/s --
    # that is the number the hydrodynamics is responsible for.
    print("\n[xrism] ==== what those ion temperatures mean as shock speeds ====")
    for group, r in results.items():
        el = r["el_ref"]
        A = ATOMIC[el][0]
        v = np.sqrt(16.0 / 3.0 * r["global_kT_s"] * KEV_IN_K * K_B / (A * M_P))
        print(f"    {group}: kT_i({el}) = {r['global_kT_s']:.0f} keV implies a "
              f"shock at {v / 1e5:.0f} km/s")
    v_obs = np.sqrt(16.0 / 3.0 * 176.0 * KEV_IN_K * K_B
                    / (ATOMIC["Si"][0] * M_P)) / 1e5
    print(f"    XRISM's 150-300 keV Fe-Si difference corresponds to a reverse "
          f"shock\n    near {v_obs:.0f} km/s (150 keV) to "
          f"{v_obs * np.sqrt(300.0 / 176.0):.0f} km/s (300 keV), and the "
          "published\n    value is 1800 km/s.")

    print("\n[xrism] ==== the n_e t / kT_e anticorrelation ====")
    print("    XRISM reads it as dense clumps: denser gas is MORE ionized and")
    print("    COOLER, because the shock transmitted into it is slower")
    print("    (v ~ v_s/sqrt(chi)). A model with too little density contrast")
    print("    has no mechanism for it, which is why the SIGN is the test.")


def scan(fields, meta, args, v_los, n, box):
    """Calibrate ``chi`` against XRISM, one line per value.

    The point of running this before any observation: a 256^3 row costs a couple
    of minutes and a 256^3 NEI observation costs 45, so a five-point scan here is
    cheaper than one wrong guess there. What it reports is deliberately the FOUR
    quantities that pull against each other -- the electron temperature and the
    implied shock speed come DOWN with chi, while the ionization age and the
    clumping factor go UP, and n_e t is already at the top of the observed range
    at chi = 1. There is therefore an optimum rather than a direction, and a scan
    is the only honest way to find it.
    """
    print(f"\n[xrism] ==== calibration scan, f_mass = {args.subgrid_fmass:.2f}, "
          f"net_mode = {args.subgrid_net_mode} ====")
    obs = XRISM["IME"]
    print(f"    XRISM IME: kT_e {fmt_range(*obs['kT_e'])} keV, "
          f"n_e t {fmt_range(*obs['net'], '{:.1e}')}, reverse shock ~1800 km/s")
    header = (f"    {'chi':>5s} {'kT_e IME (10-90%)':>19s} "
              f"{'n_e t IME (10-90%)':>21s} {'v_Si':>7s} "
              f"{'rho_IME':>8s} {'rho_IGE':>8s} {'C':>6s}  verdict")
    print(header)
    print("    " + "-" * (len(header) - 4))

    for chi in SCAN_CHI:
        c = None if chi <= 1.0 else chi
        try:
            res = measure(fields, args, v_los, n, box,
                          chi=None if c is None else [c],
                          f_mass=args.subgrid_fmass)
        except SystemExit as exc:
            print(f"    {chi:5.1f}  refused: {exc}")
            continue
        if len(res) < 2:
            continue
        r, ri = res["IME"], res["IGE"]
        k = r["keep"]
        kt = r["kT_e"][k][np.isfinite(r["kT_e"][k])]
        net = r["net"][k][np.isfinite(r["net"][k])]
        kt_lo, kt_hi = np.percentile(kt, [10, 90])
        nt_lo, nt_hi = np.percentile(net, [10, 90])
        A = ATOMIC["Si"][0]
        v = np.sqrt(16.0 / 3.0 * r["global_kT_s"] * KEV_IN_K * K_B / (A * M_P)) / 1e5
        # both must be satisfied, and they pull opposite ways
        ok_kt = kt_lo <= obs["kT_e"][1]
        ok_net = nt_hi <= obs["net"][1]
        note = ("kT_e IN, n_e t IN" if (ok_kt and ok_net) else
                "kT_e IN, n_e t OVER" if ok_kt else
                "kT_e over, n_e t IN" if ok_net else "both over")
        print(f"    {chi:5.1f} {fmt_range(kt_lo, kt_hi):>19s} "
              f"{fmt_range(nt_lo, nt_hi, '{:.2e}'):>21s} {v:7.0f} "
              f"{spearman(r['kT_e'][k], r['net'][k]):+8.3f} "
              f"{spearman(ri['kT_e'][ri['keep']], ri['net'][ri['keep']]):+8.3f} "
              f"{r['clumping']:6.2f}  {note}")

    print("\n    rho columns are Spearman(kT_e, n_e t) across pixels; XRISM "
          "measures both NEGATIVE.")
    print("    C is <n_e^2>/<n_e>^2 over the emitting gas, resolved x sub-grid.")
    print("    A row that fixes kT_e while pushing n_e t over the observed "
          "range has\n    not fixed anything -- it has traded one residual "
          "for another.")


def make_figure(results, args, meta):
    """The (kT_e, n_e t) plane per group, with the XRISM boxes on it."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), constrained_layout=True)
    colours = {"IME": "tab:cyan", "IGE": "tab:orange"}

    # 1. per-cell emission-weighted distribution, both groups, XRISM boxes
    ax = axes[0]
    ax.set_facecolor("0.08")
    kt_bins = np.logspace(-1, 1.5, 70)
    net_bins = np.logspace(9.5, 12.5, 70)
    for group, r in results.items():
        k = r["keep"]
        # ONE mask for both axes: masking them separately can hand histogram2d
        # two different lengths, which is a crash waiting for the first pixel
        # that is finite in one and not the other
        ok = k & np.isfinite(r["kT_e"]) & np.isfinite(r["net"])
        H, _, _ = np.histogram2d(
            np.clip(r["kT_e"][ok], kt_bins[0], kt_bins[-1]),
            np.clip(r["net"][ok], net_bins[0], net_bins[-1]),
            bins=[kt_bins, net_bins])
        ax.contour(kt_bins[:-1], net_bins[:-1], H.T, levels=3,
                   colors=colours[group], linewidths=1.4)
        obs = XRISM[group]
        ax.add_patch(plt.Rectangle(
            (obs["kT_e"][0], obs["net"][0]),
            obs["kT_e"][1] - obs["kT_e"][0], obs["net"][1] - obs["net"][0],
            fill=False, ec=colours[group], ls="--", lw=2,
            label=f"XRISM {group}"))
        ax.plot([], [], "-", color=colours[group], label=f"model {group}")
    ax.set(xscale="log", yscale="log", xlabel="$kT_e$ [keV]",
           ylabel=r"$n_e t$ [cm$^{-3}$ s]",
           title=f'per {args.pixel_arcsec:.0f}" pixel, line-emission weighted')
    ax.legend(fontsize=8, loc="lower left")

    # 2. the correlation, which is the discriminating test
    ax = axes[1]
    for group, r in results.items():
        k = r["keep"]
        ax.scatter(r["kT_e"][k], r["net"][k], s=14, alpha=0.7,
                   color=colours[group],
                   label=f"{group}  rho = {spearman(r['kT_e'][k], r['net'][k]):+.2f}")
        obs = XRISM[group]
        ax.add_patch(plt.Rectangle(
            (obs["kT_e"][0], obs["net"][0]),
            obs["kT_e"][1] - obs["kT_e"][0], obs["net"][1] - obs["net"][0],
            fill=False, ec=colours[group], ls="--", lw=1.5))
    ax.set(xscale="log", yscale="log", xlabel="$kT_e$ [keV]",
           ylabel=r"$n_e t$ [cm$^{-3}$ s]",
           title="XRISM measures these ANTIcorrelated")
    ax.legend(fontsize=9)

    # 3. velocity: shift and kinematic width per pixel
    ax = axes[2]
    for group, r in results.items():
        k = r["keep"]
        ax.scatter(r["v_mean"][k], r["v_sigma"][k], s=14, alpha=0.7,
                   color=colours[group], label=f"model {group}")
        obs = XRISM[group]
        ax.add_patch(plt.Rectangle(
            (obs["v_shift"][0], 0.0),
            obs["v_shift"][1] - obs["v_shift"][0], obs["sigma_v_max"],
            fill=False, ec=colours[group], ls="--", lw=1.5,
            label=f"XRISM {group}"))
    ax.set(xlabel="line-of-sight shift [km/s]",
           ylabel=r"kinematic $\sigma_v$ [km/s]",
           title="Doppler shift and width per pixel")
    ax.legend(fontsize=8)

    fig.suptitle(f"{Path(args.state).name}  {meta['num_cells']}$^3$  "
                 f"{meta['age']:.0f} yr   vs XRISM/Resolve "
                 "(Vink et al. 2026)", fontsize=11)
    out = FIGURES_DIR / f"{args.out}.png"
    fig.savefig(out, dpi=150)
    print(f"\n[xrism] saved {out}")


if __name__ == "__main__":
    main()
