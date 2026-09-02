"""The metrics the dynamo convergence study is read off with.

Kept apart from the plotting so that every number in the README, in the printed
table and in the figures comes from one definition. Everything here works on the
reduced ``.npz`` files that ``dynamo_convergence.py`` (astronomix) and
``athenapk_turb.py`` (AthenaPK) write, which share a schema.
"""

# general
from pathlib import Path

# numerics
import numpy as np

#: Index of each spectrum inside the ``spectra`` array of a run file.
E_V, E_KIN, E_MAG = 0, 1, 2


# -------------------------------------------------------------
# ==================== ↓ Run file access ↓ ====================
# -------------------------------------------------------------
def load_runs(data_dir, pattern="*.npz", skip=("smoke", "calib")):
    """Load every run file in ``data_dir``, newest schema only, sorted by N."""
    runs = []
    for path in sorted(Path(data_dir).glob(pattern)):
        if any(s in path.name for s in skip):
            continue
        d = np.load(path, allow_pickle=True)
        if "spectra" not in d.files or "code" not in d.files:
            continue                       # a legacy paper_*.npz, not this study
        runs.append(dict(d, path=path))
    return sorted(runs, key=lambda r: (str(r["code"]), str(r.get("scheme_key", "")),
                                       int(r["N"])))


def run_key(run):
    """A short label like ``astronomix N=128`` / ``AthenaPK ppm N=128``."""
    return f"{str(run['label'])} N={int(run['N'])}"


def spectra_of(run, deconvolve=True):
    """The spectra to use for a run: FV runs get the cell-average deconvolution.

    astronomix stores point values and needs no correction; AthenaPK stores cell
    averages, whose spectrum is the true one times ``sinc^2`` per axis. Comparing
    the two raw compares a low-pass-filtered field against an unfiltered one.
    """
    if deconvolve and bool(run.get("finite_volume", False)) \
            and "spectra_deconv" in run:
        return np.asarray(run["spectra_deconv"])
    return np.asarray(run["spectra"])
# -------------------------------------------------------------
# ==================== ↑ Run file access ↑ ====================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =================== ↓ Spectral metrics ↓ ====================
# -------------------------------------------------------------
def time_average(run, spectra, t_lo, t_hi=np.inf):
    """Average a run's spectra over the window ``t_lo <= t/t_cross <= t_hi``.

    Returns ``(mean, standard_error, n_snapshots)``; the standard error across
    snapshots is the noise floor any code-to-code difference has to beat.
    """
    tc = np.asarray(run["t_over_tc"])
    m = (tc >= t_lo) & (tc <= t_hi)
    if m.sum() < 2:
        raise ValueError(f"{run_key(run)}: {m.sum()} snapshots in [{t_lo}, {t_hi}]")
    sel = spectra[m]
    return sel.mean(0), sel.std(0, ddof=1) / np.sqrt(m.sum()), int(m.sum())


def fit_slope(n, E, band):
    """Log-log slope of ``E(n)`` over the shell band ``(n_lo, n_hi)``."""
    m = (n >= band[0]) & (n <= band[1]) & (E > 0)
    if m.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(n[m]), np.log(E[m]), 1)[0])


def cutoff_shell(n, E, band=(3, 8), frac=0.25, exponent=None):
    """Shell at which the compensated spectrum falls to ``frac`` of its plateau.

    The spectrum is compensated by its own large-scale power law (fitted over
    ``band`` unless ``exponent`` is given), which flattens the inertial range;
    the shell where that plateau has decayed to ``frac`` is where the scheme's
    numerical dissipation has taken over. ``n_cut / N`` is then the fraction of
    the grid a scheme turns into resolved turbulence.
    """
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    p = fit_slope(n, E, band) if exponent is None else exponent
    if not np.isfinite(p):
        return np.nan, np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where((n > 0) & (E > 0), E * n ** (-p), np.nan)
    plateau = np.nanmean(C[(n >= band[0]) & (n <= band[1])])
    target = frac * plateau

    # First crossing strictly above the fitting band, interpolated in log-log.
    idx = np.where((n > band[1]) & (C < target))[0]
    if len(idx) == 0:
        return np.nan, p
    i = idx[0]
    if i == 0 or not np.isfinite(C[i - 1]) or C[i - 1] <= target:
        return float(n[i]), p
    f = (np.log(C[i - 1]) - np.log(target)) / (np.log(C[i - 1]) - np.log(C[i]))
    return float(np.exp(np.log(n[i - 1]) + f * (np.log(n[i]) - np.log(n[i - 1])))), p


def peak_shell(n, E):
    """Peak shell of a spectrum, parabolically interpolated in log-log."""
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    valid = (n > 0) & (E > 0)
    i = int(np.argmax(np.where(valid, E, -np.inf)))
    if i <= 1 or i >= len(n) - 1:
        return float(n[i])
    x = np.log(n[i - 1:i + 2])
    y = np.log(E[i - 1:i + 2])
    denom = (y[0] - 2 * y[1] + y[2])
    if denom == 0:
        return float(n[i])
    return float(np.exp(x[1] - 0.5 * (x[2] - x[1]) * (y[2] - y[0]) / denom))


def mean_shell(n, E):
    """Energy-weighted mean shell ``<n> = sum n E(n) / sum E(n)``.

    A single robust number for "how small-scale is this field", which -- unlike
    the peak or a drop-to-a-fraction threshold -- does not depend on the shape
    of the spectrum near its maximum. For the magnetic spectrum it is the
    cleanest statement of what the dynamo produced: a scheme with less numerical
    resistivity puts its magnetic energy at higher ``n``.
    """
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    m = n >= 1
    return float(np.sum(n[m] * E[m]) / np.sum(E[m]))


def high_shell_fraction(n, E, N, above=4):
    """Fraction of the energy sitting in the top octave, ``n > N / above``.

    The counterpart to :func:`cutoff_shell`: a compensated-plateau cutoff asks
    where a spectrum falls off, and is therefore fooled by a scheme that piles
    energy up at the grid scale instead of dissipating it -- such a spectrum
    never falls off, so it scores a *high* effective resolution. This number
    says how much energy is in the top octave, where a resolved cascade should
    have almost none. Read the two together: a large ``n_1/4`` with a small
    high-shell fraction is resolved cascade, a large ``n_1/4`` with a large one
    is grid-scale pile-up.
    """
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    keep = n >= 1
    return float(np.sum(E[keep & (n > N / above)]) / np.sum(E[keep]))


def half_width_shell(n, E, frac=0.25):
    """Shell above the peak at which ``E`` has fallen to ``frac`` of its peak."""
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    i = int(np.argmax(np.where(n > 0, E, -np.inf)))
    target = frac * E[i]
    idx = np.where((np.arange(len(n)) > i) & (E < target))[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    f = (np.log(E[j - 1]) - np.log(target)) / (np.log(E[j - 1]) - np.log(E[j]))
    return float(np.exp(np.log(n[j - 1]) + f * (np.log(n[j]) - np.log(n[j - 1]))))
# -------------------------------------------------------------
# =================== ↑ Spectral metrics ↑ ====================
# -------------------------------------------------------------


#: The window in which the magnetic field is still essentially a passive tracer,
#: in crossing times. Deliberately a *fixed* window rather than one chosen per
#: run from ``E_B/E_K``: an adaptive window would be shorter for the schemes and
#: resolutions with the fastest dynamos, so a spectrum measured in it would be
#: averaged over a different part of the spin-up than everyone else's, and the
#: comparison would partly measure the window. In [2.5, 5] the flow has spun up
#: (Mach is flat from t/t_cross ~ 2) and ``E_B/E_K`` is at most a few percent
#: even for the fastest dynamo in the ladder (astronomix at 256^3, 6%), so all
#: nine runs are compared over the same crossing times in the same regime.
KINEMATIC_WINDOW = (2.5, 5.0)


def kinematic_window(run, window=KINEMATIC_WINDOW):
    """Mask of the snapshots in ``KINEMATIC_WINDOW``, and the ``E_B/E_K`` there.

    Before the dynamo back-reacts, all three schemes carry the *same*
    hydrodynamic turbulence (they are matched to better than 1% in ``v_rms``
    there), so a kinetic-spectrum comparison in this window measures numerical
    dissipation and nothing else. Once ``E_B/E_K`` reaches tens of percent the
    schemes have genuinely different flows -- a stronger dynamo drains the
    small-scale kinetic energy -- and a comparison there is measuring the dynamo
    rather than the scheme's resolving power. The returned ratio is how far into
    the back-reaction the window actually reaches, and is reported so that claim
    can be checked rather than assumed.
    """
    tc = np.asarray(run["t_over_tc"])
    ratio = np.asarray(run["E_B"]) / np.maximum(np.asarray(run["E_K"]), 1e-30)
    m = (tc >= window[0]) & (tc <= window[1])
    return m, float(ratio[m].max()) if m.any() else np.nan


# -------------------------------------------------------------
# ========= ↓ Effective Reynolds numbers (numerical) ↓ ========
# -------------------------------------------------------------
#: The driving shell. Both codes' forcing spectra peak here, so it is the outer
#: scale that Re and Rm are referred to.
N_INJECTION = 2.0


def dissipation_shell(n, E):
    """Dissipation-weighted mean shell, ``<n> = int n^3 E dn / int n^2 E dn``.

    The viscous dissipation spectrum is ``2 nu k^2 E_v(k)`` and the ohmic one
    ``2 eta k^2 E_B(k)``, so this is literally "the shell at which this field is
    being dissipated" -- and it is *free of the diffusivity itself*, which is the
    whole point: neither code has an explicit one to read off.

    Chosen over the compensated-plateau cutoff or the spectral peak because it
    needs no threshold, no fitting band and no assumption about the spectral
    shape, so it means the same thing for a 2nd- and a 5th-order scheme.
    """
    n = np.asarray(n, dtype=float)
    E = np.asarray(E, dtype=float)
    m = (n >= 1) & (E > 0)
    w = n[m] ** 2 * E[m]
    return float(np.sum(n[m] * w) / np.sum(w))


def reynolds_numbers(n_nu, n_eta, n_inj=N_INJECTION):
    """Effective ``(Re, Rm, Pm)`` from the two dissipation shells.

    Uses the Kolmogorov scale relation ``l_diss / L ~ Re^(-3/4)``, i.e.
    ``Re = (n_nu / n_inj)^(4/3)`` and likewise for ``Rm`` with the resistive
    shell. The prefactor of that relation is order unity and convention
    dependent, so **the absolute numbers are indicative and the ratios between
    codes, and the scaling with N, are the results**. ``Pm = Rm / Re`` inherits
    the same caveat but is a pure ratio of measured shells,
    ``(n_eta / n_nu)^(4/3)``, so the convention cancels between the two codes.
    """
    Re = (n_nu / n_inj) ** (4.0 / 3.0)
    Rm = (n_eta / n_inj) ** (4.0 / 3.0)
    return Re, Rm, Rm / Re


def effective_diffusivities(run, n_nu, n_eta, spectra, window, v_rms, L_inj=0.5):
    """``(nu_eff, eta_eff)`` in code units, from the measured dissipation shells.

    ``nu = v_rms L_inj / Re`` and ``eta = v_rms L_inj / Rm`` -- the definitions
    that make :func:`reynolds_numbers` self-consistent. Reported because a
    diffusivity is the quantity a reader wants to compare against an explicit
    one, not because it is more fundamental than the shells it comes from.
    """
    Re, Rm, _ = reynolds_numbers(n_nu, n_eta)
    return v_rms * L_inj / Re, v_rms * L_inj / Rm


def eigenmode_window_mask(run, above_seed=1e3, max_ratio=1e-4, t_min=2.0):
    """Snapshots inside the kinematic eigenmode window (see the growth rate).

    The resistive shell has to be read off *there*: before it the spectrum is
    still the tangling transient, after it the back-reaction has moved the peak
    to large scales and it no longer marks the resistive scale at all.
    """
    tc = np.asarray(run["t_over_tc"])
    E_B, E_K = np.asarray(run["E_B"]), np.asarray(run["E_K"])
    ratio = E_B / np.maximum(E_K, 1e-30)
    seed = float(E_B[0]) / max(float(np.max(E_K)), 1e-30)
    return (tc >= t_min) & (ratio >= seed * above_seed) & (ratio <= max_ratio)
# -------------------------------------------------------------
# ========= ↑ Effective Reynolds numbers (numerical) ↑ ========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ==================== ↓ Dynamo metrics ↓ =====================
# -------------------------------------------------------------
#: The decade in ``E_B / E_K`` over which the kinematic growth rate is fitted,
#: and the earliest crossing time admitted into the fit.
GROWTH_BAND = (1e-3, 1e-2)
GROWTH_T_MIN = 2.0

#: Eigenmode window for :func:`eigenmode_growth_rate`, as a multiple of the seed
#: ``E_B/E_K`` (lower edge: the initial tangling transient must be over) and as an
#: absolute ``E_B/E_K`` (upper edge: the back-reaction must not have started).
EIGENMODE_ABOVE_SEED = 1e3
EIGENMODE_MAX_RATIO = 1e-4


def eigenmode_growth_rate(t_over_tc, E_B, E_K, above_seed=EIGENMODE_ABOVE_SEED,
                          max_ratio=EIGENMODE_MAX_RATIO, t_min=GROWTH_T_MIN):
    """Growth rate of the kinematic *eigenmode*, and the evidence that it is one.

    :func:`growth_rate` fits one decade and cannot tell an eigenmode from a
    decaying transient. This one fits the window between the two things that are
    not the eigenmode -- the initial transient in which the coherent seed field
    is tangled down to the grid scale, and the back-reaction at the top -- and
    additionally returns the *per-decade* rates inside it, which is the actual
    test: a genuine eigenmode grows at the same rate in every decade.

    The window only exists if the seed is weak enough to leave room for it. With
    the uniform net-flux seed of the original setup there is no window at all
    (the mean field is conserved, so the tangling never finishes); with a
    zero-net-flux seed at ``beta = 1e12`` there are 4-5 clean decades.

    Returns ``(gamma, spread, per_decade, n_points)`` where ``gamma`` is the
    fitted rate per crossing time, ``spread`` is the standard deviation of the
    per-decade rates (small = a real eigenmode), and ``per_decade`` is the list
    of ``(decade_lower_edge, rate)`` pairs.
    """
    t_over_tc = np.asarray(t_over_tc, dtype=float)
    E_B = np.asarray(E_B, dtype=float)
    ratio = E_B / np.maximum(np.asarray(E_K, dtype=float), 1e-30)
    steady = t_over_tc >= t_min
    if not steady.any():
        return np.nan, np.nan, [], 0
    # The seed level is the *initial* magnetic energy measured against the
    # kinetic energy the driving settles at -- not the minimum of the ratio,
    # which the spin-up (E_K rising from zero) puts in the wrong place.
    E_K_steady = float(np.max(np.asarray(E_K, dtype=float)))
    seed = float(E_B[0]) / max(E_K_steady, 1e-30)
    lo, hi = seed * above_seed, max_ratio
    m = steady & (ratio >= lo) & (ratio <= hi) & (E_B > 0)
    if m.sum() < 4 or lo >= hi:
        return np.nan, np.nan, [], int(m.sum())
    gamma = float(np.polyfit(t_over_tc[m], np.log(E_B[m]), 1)[0])

    per_decade = []
    edge = 10.0 ** np.ceil(np.log10(lo))
    while edge * 10.0 <= hi * (1.0 + 1e-9):
        d = steady & (ratio >= edge) & (ratio <= 10 * edge) & (E_B > 0)
        if d.sum() >= 4:
            per_decade.append((float(edge),
                               float(np.polyfit(t_over_tc[d], np.log(E_B[d]), 1)[0])))
        edge *= 10.0
    spread = float(np.std([r for _, r in per_decade], ddof=1)) \
        if len(per_decade) > 1 else np.nan
    return gamma, spread, per_decade, int(m.sum())


def growth_rate(t_over_tc, E_B, E_K, band=GROWTH_BAND, t_min=GROWTH_T_MIN):
    """Kinematic growth rate of the magnetic energy, ``d ln E_B / dt``.

    Fitted over a *fixed decade in* ``E_B / E_K`` rather than over a fixed time
    window, and only after ``t_min`` crossing times. Both restrictions matter and
    both were found the hard way:

    * The box starts at rest, and the driving needs ~2 crossing times to bring
      ``v_rms`` to its plateau. During that spin-up the uniform seed field is
      wound up by a *developing* flow and grows far faster than the small-scale
      dynamo eigenmode. A window chosen as "10x the seed to 10% of saturation"
      slides back into that spin-up as the resolution rises -- at 256^3 it lands
      entirely inside it -- and returns a growth rate 3-4x too large, with a
      spurious ``N^0.7`` trend attached.
    * The growth is not a clean exponential all the way up: the local rate falls
      as the field starts to react back. Fitting a fixed decade in ``E_B/E_K``
      compares the same stage of the dynamo in every run, so the numbers are
      resolution- and scheme-independent by construction.

    Returns ``(gamma, r_squared, t_lo, t_hi, n_points)``, with ``gamma`` the
    growth rate of ``E_B`` per crossing time (so ``|B|`` grows at ``gamma / 2``).
    """
    t_over_tc = np.asarray(t_over_tc, dtype=float)
    E_B = np.asarray(E_B, dtype=float)
    ratio = E_B / np.maximum(np.asarray(E_K, dtype=float), 1e-30)
    m = (t_over_tc >= t_min) & (ratio >= band[0]) & (ratio <= band[1]) & (E_B > 0)
    if m.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan, int(m.sum())
    x, y = t_over_tc[m], np.log(E_B[m])
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1.0 - np.sum(resid ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-300)
    return float(slope), float(r2), float(x[0]), float(x[-1]), int(m.sum())


def saturation(run, t_lo):
    """Mean ``E_B``, ``E_B/E_K`` and ``<|B|>`` over the saturated window.

    Also returns ``sat_growth``, the residual ``d ln E_B / dt`` (per crossing
    time) still present in that window. A run whose dynamo has not finished
    growing by the end of the run has a large ``sat_growth``, and its
    "saturated" numbers are a lower bound rather than a converged level -- the
    2nd-order scheme at low resolution is exactly that case.
    """
    tc = np.asarray(run["t_over_tc"])
    m = tc >= t_lo
    E_B = np.asarray(run["E_B"])[m]
    E_K = np.asarray(run["E_K"])[m]
    E_B_t, E_K_t = np.asarray(run["E_B"]), np.asarray(run["E_K"])
    growing = np.polyfit(tc[m], np.log(np.maximum(E_B_t[m], 1e-300)), 1)[0] \
        if m.sum() >= 3 else np.nan
    return dict(E_B=float(E_B.mean()), E_K=float(E_K.mean()),
                sat_growth=float(growing), saturated=bool(abs(growing) < 0.05),
                ratio=float(np.mean(E_B / E_K)),
                mean_absB=float(np.asarray(run["mean_absB"])[m].mean()),
                v_rms=float(np.asarray(run["v_rms"])[m].mean()),
                mach=float(np.asarray(run["mach"])[m].mean()),
                mach_alfven=float(np.asarray(run["mach_alfven"])[m].mean()),
                rho_rms=float(np.asarray(run["rho_rms"])[m].mean()),
                rel_divB=float(np.asarray(run["rel_divB"])[m].mean())
                if "rel_divB" in run else np.nan,
                n_snapshots=int(m.sum()))


def dynamo_time_series(run):
    """``(t/t_cross, E_B, E_K)`` at the finest cadence the run recorded.

    AthenaPK writes a ``.hst`` history every few cycles, ~20x finer in time than
    its hdf5 dumps; astronomix has only the snapshot cadence. Both are volume
    integrals over a unit box, so they are directly comparable.
    """
    if "hst_time" in run:
        return (np.asarray(run["hst_time"]) / float(run["t_cross"]),
                np.asarray(run["hst_ME"]), np.asarray(run["hst_KE"]))
    return (np.asarray(run["t_over_tc"]), np.asarray(run["E_B"]),
            np.asarray(run["E_K"]))
# -------------------------------------------------------------
# ==================== ↑ Dynamo metrics ↑ =====================
# -------------------------------------------------------------


def summarize(run, sat_start=28.0, kin_band=(3, 8), frac=0.25, deconvolve=True,
              growth_run=None):
    """Every headline number for one run, in one dict.

    Spectral quantities are measured twice: in the kinematic window, where the
    three codes carry the same flow and the kinetic spectrum therefore measures
    the scheme, and in the saturated window, which is the state the run ends in.

    ``growth_run`` optionally supplies a *different* run to take the growth-rate
    fit from -- a short, high-cadence, scalars-only repeat of the same setup with
    the same seed. astronomix's production snapshot cadence (0.5 crossing times)
    only lands one point inside the fitting decade at 256^3, where AthenaPK's
    ``.hst`` has dozens; the companion run removes that asymmetry so both codes'
    growth rates come from a comparably sampled curve.
    """
    n = np.asarray(run["n_shell"], dtype=float)
    spec = spectra_of(run, deconvolve)
    Ev, _, _ = time_average(run, spec[:, E_V], sat_start)
    Eb, _, _ = time_average(run, spec[:, E_MAG], sat_start)

    # One fixed early window for the matching check and the kinetic spectra
    # alike: both have to be read off identical crossing times.
    kin_mask, kin_ratio_max = kinematic_window(run)
    early = kin_mask
    Ev_kin = spec[kin_mask, E_V].mean(0)

    # Primary cutoff: the Kolmogorov compensation n^(5/3) E_v(n), the same
    # definition the hydrodynamic study in ../../hydro/turbulence uses, so the
    # two studies' effective resolutions are on one scale. The self-fitted
    # variant is carried alongside as a robustness check -- a saturated dynamo
    # steepens the kinetic spectrum away from -5/3, and if the two disagreed the
    # ranking would be an artefact of the compensation.
    n_cut, _ = cutoff_shell(n, Ev, band=kin_band, frac=frac, exponent=-5.0 / 3.0)
    n_cut_fitted, slope_v = cutoff_shell(n, Ev, band=kin_band, frac=frac)
    n_cut_kin, _ = cutoff_shell(n, Ev_kin, band=kin_band, frac=frac,
                                exponent=-5.0 / 3.0)
    slope_v_kin = fit_slope(n, Ev_kin, kin_band)

    t_over_tc, E_B_t, E_K_t = dynamo_time_series(growth_run if growth_run
                                                 is not None else run)
    gamma, r2, t_lo, t_hi, n_fit = growth_rate(t_over_tc, E_B_t, E_K_t)
    sat = saturation(run, sat_start)

    N = int(run["N"])
    return dict(
        key=run_key(run), code=str(run["code"]), label=str(run["label"]),
        scheme=str(run.get("scheme", "")), N=N,
        t_wall=float(run["t_wall"]), t_compile=float(run.get("t_compile", 0.0)),
        n_steps=float(run.get("n_steps_estimated", np.nan)),
        zone_updates_per_s=float(run.get("zone_updates_per_s", np.nan)),
        n_cut=n_cut, n_cut_over_N=n_cut / N, slope_v=slope_v,
        n_cut_fitted=n_cut_fitted,
        n_cut_kin=n_cut_kin, n_cut_kin_over_N=n_cut_kin / N,
        slope_v_kin=slope_v_kin, kin_ratio_max=kin_ratio_max,
        mach_kin=float(np.asarray(run["mach"])[kin_mask].mean()),
        v_rms_kin=float(np.asarray(run["v_rms"])[kin_mask].mean()),
        mach_early=float(np.asarray(run["mach"])[early].mean()),
        v_rms_early=float(np.asarray(run["v_rms"])[early].mean()),
        n_peak_mag=peak_shell(n, Eb), n_cut_mag=half_width_shell(n, Eb, frac),
        n_mean_mag=mean_shell(n, Eb), n_mean_mag_over_N=mean_shell(n, Eb) / N,
        n_mean_kin=mean_shell(n, Ev),
        high_frac_v=high_shell_fraction(n, Ev_kin, N),
        high_frac_b=high_shell_fraction(n, Eb, N),
        n_cut_mag_over_N=half_width_shell(n, Eb, frac) / N,
        gamma=gamma / float(run["t_cross"]), gamma_tcross=gamma, gamma_r2=r2,
        gamma_window=(t_lo, t_hi), gamma_n_fit=n_fit, **sat,
    )
