"""Numerical diffusivities, Prandtl and Reynolds numbers, per scheme and resolution.

One table, one definition, every scheme in the ladder. Everything comes from the
spectral energy budget of the saturated state, shell by shell,

    dE(n)/dt = T(n) - D(n)

with ``T(n)`` the exact ideal right-hand side (``_mhd_spectral.transfer_spectra``)
and ``D(n)`` what the scheme threw away. Dividing out the shell's own curvature,

    nu_eff(n)  = D_v(n) / (2 k^2 E_v(n)),   eta_eff(n) = D_B(n) / (2 k^2 E_B(n))

and averaging over the flat band gives the two diffusivities, from which

    Re = v_rms L_inj / nu_eff,   Rm = v_rms L_inj / eta_eff,   Pm = Rm / Re.

``--audit`` adds the checks that say whether those numbers may be believed:
Parseval, budget closure, band and window systematics, a bootstrap error, and
the resolvedness diagnostic ``n_K / n_Nyquist`` -- if the Kolmogorov scale
implied by the measured ``nu`` lies beyond the grid, a Laplacian reading of that
``nu`` is extrapolating past where the dissipation actually happens.

``--calibration`` reads the runs with an *explicit* Laplacian diffusivity and
fits measured against imposed: the estimator is only worth its numbers if that
line has slope one.

    python make_mechanism_table.py --audit
    python make_mechanism_table.py --calibration
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import N_INJECTION, dissipation_shell, load_runs
from make_dissipation_figure import dissipation_series

HERE = Path(__file__).resolve().parent

#: Band, in units of the Nyquist shell, over which the diffusivities are
#: averaged. Below it the forcing and the outer scale contaminate the budget;
#: above it the transfer is aliased, the products not being dealiased.
BAND = (0.2, 0.7)

#: Start of the saturated window, in crossing times.
SAT_START = 28.0

#: Outer scale ``Re`` and ``Rm`` are referred to: the driving wavelength. Both
#: codes force at ``n = kL/2pi ~ 2`` in a unit box, so that is 0.5. It is a
#: *convention* -- the same one the dynamo literature quotes ``Rm_crit`` with --
#: and every absolute Re/Rm scales with it. It cancels exactly from ``Pm``.
#: :func:`measure` also returns each run's own measured integral scale, which
#: is 0.18-0.21 here and 9-11% larger for astronomix than for the GLM schemes,
#: so a fixed ``L`` flatters astronomix's Rm by about that much.
L_INJ = 0.5

#: Above this fraction of Nyquist, the Kolmogorov scale implied by the measured
#: ``nu`` is not on the grid and the Laplacian reading is an extrapolation.
RESOLVED_MAX = 0.8

#: A conditioning choice independent of ``MATCH_RATIO``: a *fixed* time window
#: in which every scheme's field is still passive. Matching on ``E_B/E_K``
#: reaches each scheme at a different time (3 to 15 crossing times here), so if
#: the answer depended on the OU forcing phase or on the flow still settling,
#: the two conditionings would disagree. They do not.
FIXED_WINDOW = (2.5, 5.0)

#: Reference magnetic energy fractions at which the schemes are compared. The
#: saturated state is *not* a fair comparison point: each scheme saturates at its
#: own ``E_B/E_K`` (0.08 to 0.43 across this ladder) and ``Pm_eff`` drifts with
#: that fraction, so a saturated-state table partly measures the saturation level
#: rather than the scheme. ``0.01`` is inside the kinematic phase for every run
#: -- the field is still passive, so all schemes carry the same turbulence.
MATCH_RATIO = 0.01
MATCH_WINDOW = 5          # snapshots averaged around the crossing


# -------------------------------------------------------------
# ================== ↓ The measurement ↓ ======================
# -------------------------------------------------------------
def measure(run, band=BAND, sat_start=SAT_START, snapshots=None):
    """``nu_eff``, ``eta_eff`` and everything derived, for one run.

    ``snapshots`` overrides the saturated-window mask, which is how the
    bootstrap resamples; leave it ``None`` for the plain measurement.
    """
    ser = dissipation_series(run, deconvolve=False)
    if ser is None:
        return None
    N = int(run["N"])
    n, nyq = ser["n"], N / 2
    k = 2.0 * np.pi * n
    inband = (n / nyq >= band[0]) & (n / nyq <= band[1])
    sel = (np.where(ser["t_over_tc"] >= sat_start)[0]
           if snapshots is None else snapshots)
    if len(sel) < 3:
        return None

    D_v, D_B = ser["D_v"][sel].mean(0), ser["D_B"][sel].mean(0)
    E_v, E_B = ser["E_v"][sel].mean(0), ser["E_B"][sel].mean(0)
    nu = float(np.mean(D_v[inband] / (2.0 * k[inband] ** 2 * E_v[inband])))
    eta = float(np.mean(D_B[inband] / (2.0 * k[inband] ** 2 * E_B[inband])))

    v_rms = float(np.asarray(run["v_rms"])[sel].mean())
    # Longitudinal integral scale of isotropic turbulence, (3 pi / 4) times the
    # 1/k moment of the velocity spectrum. Measured rather than assumed, so the
    # L_INJ convention can be checked instead of trusted.
    keep = n >= 1
    L_spec = float((3.0 * np.pi / 4.0) * np.sum(E_v[keep] / k[keep])
                   / np.sum(E_v[keep]))
    # Dissipation rates, from n >= 4 outwards: below that the budget is the
    # forcing rather than the dissipation, and shows up with the opposite sign.
    eps_v = float(D_v[n >= 4].sum())
    eps_B = float(D_B[n >= 4].sum())
    return dict(
        N=N, label=str(run["label"]), code=str(run["code"]),
        scheme=str(run.get("scheme_key", "-")),
        nu=nu, eta=eta, Pm=nu / eta,
        Re=v_rms * L_INJ / nu, Rm=v_rms * L_INJ / eta,
        L_spec=L_spec,
        Re_L=v_rms * L_spec / nu, Rm_L=v_rms * L_spec / eta,
        v_rms=v_rms, mach=float(np.asarray(run["mach"])[sel].mean()),
        ratio=float(np.mean(np.asarray(run["E_B"])[sel]
                            / np.maximum(np.asarray(run["E_K"])[sel], 1e-30))),
        eps_v=eps_v, eps_B=eps_B,
        # Fraction of in-band shells where the "dissipation" has the wrong sign.
        # A dissipative reading requires this to be zero.
        neg_v=float(np.mean(D_v[inband] <= 0)),
        neg_B=float(np.mean(D_B[inband] <= 0)),
        # Pm shell by shell, rather than as one band mean: the spread says how
        # much the Laplacian compression is costing.
        Pm_shells=(D_v[inband] / (2.0 * k[inband] ** 2 * E_v[inband]))
        / (D_B[inband] / (2.0 * k[inband] ** 2 * E_B[inband])),
        # Kolmogorov and resistive scales the measured coefficients imply, as
        # shell numbers, against the Nyquist shell the grid actually offers.
        n_kolmogorov=(nu ** 3 / eps_v) ** -0.25 / (2.0 * np.pi),
        n_resistive=(eta ** 3 / eps_v) ** -0.25 / (2.0 * np.pi),
        n_nyquist=nyq,
        # Shell diagnostics, kept for continuity with make_reynolds_figure.py.
        n_nu=dissipation_shell(n, E_v), n_eta=dissipation_shell(n, E_B),
        n_snapshots=len(sel),
        # Audit residuals.
        forcing=float(D_v[(n >= 1) & (n <= 3)].sum()),
        closure=float((D_v + D_B)[n >= 1].sum()),
        # Against the time average of v_rms **squared**, not the square of the
        # time-averaged v_rms: Jensen makes the latter smaller and pushes this
        # diagnostic above 1, which a truncated shell sum can never legitimately
        # be. The gap is 0.08-0.21% here and was mistaken for an estimator error.
        parseval=float(E_v[n >= 1].sum()
                       / (0.5 * float(np.mean(np.asarray(run["v_rms"])[sel] ** 2)))),
        ohm_diff=float(run["ohm_diff"]) if "ohm_diff" in run else 0.0,
        mom_diff=float(run["mom_diff"]) if "mom_diff" in run else 0.0,
    )


def measure_in_window(run, window=FIXED_WINDOW, band=BAND):
    """The diffusivities over a fixed time window, whatever ``E_B/E_K`` is there."""
    ser = dissipation_series(run, deconvolve=False)
    if ser is None:
        return None
    sel = np.where((ser["t_over_tc"] >= window[0])
                   & (ser["t_over_tc"] <= window[1]))[0]
    if len(sel) < 3:
        return None
    return measure(run, band=band, snapshots=sel)


def measure_at_ratio(run, ratio=MATCH_RATIO, window=MATCH_WINDOW,
                     t_min=2.0, band=BAND):
    """The diffusivities where ``E_B/E_K`` first reaches ``ratio``.

    The comparison point every scheme can be brought to, unlike the saturated
    state. Snapshots before ``t_min`` crossing times are excluded so the
    measurement cannot land in the spin-up.
    """
    ser = dissipation_series(run, deconvolve=False)
    if ser is None:
        return None
    tc, r = ser["t_over_tc"], ser["ratio"]
    hits = np.where((tc >= t_min) & (r >= ratio))[0]
    if len(hits) == 0:
        return None
    # Symmetric about the crossing, so the window mean is not biased upward by
    # the growth and does not depend on which side the snapshot cadence lands.
    lo = hits[0] - window // 2
    if lo < 1 or lo + window > len(tc):
        return None
    sel = np.arange(lo, lo + window)

    N = int(run["N"])
    n = ser["n"]
    k = 2.0 * np.pi * n
    b = (n / (N / 2) >= band[0]) & (n / (N / 2) <= band[1])
    D_v, D_B = ser["D_v"][sel].mean(0), ser["D_B"][sel].mean(0)
    E_v, E_B = ser["E_v"][sel].mean(0), ser["E_B"][sel].mean(0)
    nu = float(np.mean(D_v[b] / (2.0 * k[b] ** 2 * E_v[b])))
    eta = float(np.mean(D_B[b] / (2.0 * k[b] ** 2 * E_B[b])))
    v_rms = float(np.asarray(run["v_rms"])[sel].mean())
    keep = n >= 1
    L_spec = float((3.0 * np.pi / 4.0) * np.sum(E_v[keep] / k[keep])
                   / np.sum(E_v[keep]))
    return dict(nu=nu, eta=eta, Pm=nu / eta, Re=v_rms * L_INJ / nu,
                Rm=v_rms * L_INJ / eta, L_spec=L_spec,
                Re_L=v_rms * L_spec / nu, Rm_L=v_rms * L_spec / eta,
                t_over_tc=float(tc[sel].mean()), ratio=float(r[sel].mean()),
                mach=float(np.asarray(run["mach"])[sel].mean()))


#: Block length for the bootstrap, in snapshots. The in-band dissipation has an
#: integrated autocorrelation time of 1.6-4.8 snapshots (the driving is an
#: Ornstein-Uhlenbeck process with tau = 1 crossing time and snapshots are
#: 0.5-0.67 crossing times apart), so resampling snapshots independently would
#: understate the error by a factor sqrt(tau_int) ~ 1.3-2.2.
BLOCK = 3


def bootstrap(run, n_resample=400, seed=0, block=BLOCK, **kw):
    """Standard deviation of ``(nu, eta, Pm)`` over a *moving-block* resample."""
    ser = dissipation_series(run, deconvolve=False)
    sel = np.where(ser["t_over_tc"] >= kw.get("sat_start", SAT_START))[0]
    rng = np.random.default_rng(seed)
    starts = np.arange(len(sel) - block + 1)
    n_blocks = max(1, int(np.ceil(len(sel) / block)))
    draws = []
    for _ in range(n_resample):
        idx = np.concatenate([sel[s:s + block]
                              for s in rng.choice(starts, n_blocks)])[:len(sel)]
        draws.append(measure(run, snapshots=idx, **kw))
    return {q: float(np.std([d[q] for d in draws])) for q in ("nu", "eta", "Pm")}


#: Every band and window the measurement could defensibly have used. The first
#: three lie entirely below the 2/3 dealiasing cutoff, so if the answer depended
#: on aliasing it would move between them and ``(0.2, 0.7)``.
BAND_VARIANTS = ((0.1, 0.3), (0.2, 0.4), (0.3, 0.6), (0.15, 0.6), (0.2, 0.7))


def systematic(run, **kw):
    """Spread of ``Pm`` over the defensible band and window choices."""
    variants = [dict(band=b) for b in BAND_VARIANTS] + [dict(sat_start=24.0),
                                                        dict(sat_start=32.0)]
    vals = [measure(run, **{**kw, **v})["Pm"] for v in variants]
    return float(np.max(vals) - np.min(vals)) / 2.0
# -------------------------------------------------------------
# ================== ↑ The measurement ↑ ======================
# -------------------------------------------------------------


ORDER = {"astronomix": 5, "plm": 2, "ppm": 3, "limo3": 3, "wenoz": 5}
DIVB = {"astronomix": "CT"}


def collect(dirs, sat_start=SAT_START):
    _index(dirs)
    rows = []
    for d in dirs:
        for run in load_runs(d, skip=("smoke", "calib")):
            m = measure(run, sat_start=sat_start)
            if m is None:
                continue
            m["order"] = ORDER.get(m["scheme"] if m["code"] == "athenapk"
                                   else "astronomix", 0)
            m["divb"] = DIVB.get(m["code"], "GLM")
            rows.append(m)
    return sorted(rows, key=lambda r: (r["order"], r["divb"], r["N"]))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="*",
                   default=[str(HERE / "data" / "dissipation"),
                            str(HERE / "data" / "dissipation_mech")])
    p.add_argument("--sat-start", type=float, default=SAT_START)
    p.add_argument("--audit", action="store_true",
                   help="print the validity checks alongside the table")
    p.add_argument("--calibration", action="store_true",
                   help="fit measured against explicitly imposed diffusivity")
    p.add_argument("--summary", action="store_true",
                   help="the Re / Rm / Pm table across schemes and resolutions, "
                        "with errors and both outer-scale conventions")
    p.add_argument("--accounting", action="store_true",
                   help="decompose the dynamo advantage into its nu, eta, Rm "
                        "and Pm factors, and price it in grid cells")
    p.add_argument("--collapse", action="store_true",
                   help="fit the kinematic growth rate against Rm and Pm, with "
                        "model comparison and leave-one-scheme-out prediction")
    p.add_argument("--matched", action="store_true",
                   help="compare the schemes at matched E_B/E_K instead of in "
                        "each one's own saturated state")
    p.add_argument("--ratios", nargs="*", type=float,
                   default=(0.003, 0.01, 0.03, 0.06, 0.10),
                   help="the E_B/E_K values --matched reports at")
    p.add_argument("--out", default=str(HERE / "data" / "mechanism.md"))
    args = p.parse_args()

    if args.calibration:
        return calibration(HERE / "data" / "calibration", args.sat_start)

    rows = collect(args.data, args.sat_start)
    if not rows:
        raise SystemExit("no run with transfer spectra found; run with --transfer")

    if args.summary:
        return summary(rows, args)
    if args.accounting:
        return accounting(rows)
    if args.collapse:
        return collapse(rows)
    if args.matched:
        return matched(rows, args)

    lines = ["| scheme | order | div.B | N | `nu_eff` | `eta_eff` | `Pm` | `Re` | `Rm` |",
             "|---|---|---|---|---|---|---|---|---|"]
    print(f"\n{'scheme':26s} {'ord':>3s} {'divB':>4s} {'N':>4s} "
          f"{'nu_eff':>9s} {'eta_eff':>9s} {'Pm':>14s} {'Re':>6s} {'Rm':>6s} "
          f"{'Mach':>5s} {'EB/EK':>6s}")
    for r in rows:
        err = bootstrap_and_systematic(r, args)
        print(f"{r['label'][:26]:26s} {r['order']:3d} {r['divb']:>4s} {r['N']:4d} "
              f"{r['nu']:9.2e} {r['eta']:9.2e} {r['Pm']:6.3f}+-{err:6.3f} "
              f"{r['Re']:6.0f} {r['Rm']:6.0f} {r['mach']:5.3f} {r['ratio']:6.3f}")
        lines.append(f"| {r['label']} | {r['order']} | {r['divb']} | {r['N']} | "
                     f"{r['nu']:.2e} | {r['eta']:.2e} | **{r['Pm']:.2f} "
                     f"± {err:.2f}** | {r['Re']:.0f} | {r['Rm']:.0f} |")

    if args.audit:
        audit(rows)

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


def bootstrap_and_systematic(r, args):
    """Total uncertainty on ``Pm``: snapshot noise and analysis-choice spread."""
    run = _reload(r)
    stat = bootstrap(run, sat_start=args.sat_start)["Pm"]
    return float(np.hypot(stat, systematic(run, sat_start=args.sat_start)))


_CACHE = {}


def _index(dirs):
    """Index every run in ``dirs`` by ``(code, scheme, N)`` for :func:`_reload`."""
    for d in dirs:
        for run in load_runs(d, skip=("smoke", "calib")):
            _CACHE[(str(run["code"]), str(run.get("scheme_key", "-")),
                    int(run["N"]))] = run


def _reload(r):
    """Re-open the run file a row came from (rows carry no arrays)."""
    key = (r["code"], r["scheme"], r["N"])
    if key not in _CACHE:
        _index([HERE / "data" / "dissipation", HERE / "data" / "dissipation_mech"])
    return _CACHE[key]


def matched(rows, args):
    """The mechanism table at matched magnetic energy fraction."""
    head = "  ".join(f"Pm@{r:.3f}" for r in args.ratios)
    print(f"\n{'scheme':26s} {'ord':>3s} {'divB':>4s} {'N':>4s}  {head}")
    lines = ["| scheme | order | div.B | N | "
             + " | ".join(f"`Pm` at `E_B/E_K` = {r:g}" for r in args.ratios) + " |",
             "|---|---|---|---|" + "---|" * len(args.ratios)]
    for r in rows:
        run = _reload(r)
        cells, md = [], []
        for target in args.ratios:
            m = measure_at_ratio(run, ratio=target)
            cells.append(f"{m['Pm']:8.3f}" if m else f"{'--':>8s}")
            md.append(f"{m['Pm']:.2f}" if m else "--")
        print(f"{r['label'][:26]:26s} {r['order']:3d} {r['divb']:>4s} {r['N']:4d}  "
              + "  ".join(cells))
        lines.append(f"| {r['label']} | {r['order']} | {r['divb']} | {r['N']} | "
                     + " | ".join(md) + " |")
    print("\nEach column is one physical state, reached at a different time by "
          "each scheme.\nAt E_B/E_K = 0.01 the field is still passive, so every "
          "run carries the same turbulence there.")
    out = Path(args.out).with_name("mechanism_matched.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


#: Decade of ``E_B/E_K`` the kinematic growth rate is fitted over, and the
#: earliest crossing time admitted, matching ``_mhd_metrics.GROWTH_BAND``.
COLLAPSE_BAND = (3e-3, 3e-2)


def growth(run, band=COLLAPSE_BAND, t_min=2.0, with_count=False):
    """``d ln E_B / dt`` per crossing time over the kinematic decade.

    Three points is the minimum: the fastest dynamo in the ladder crosses this
    decade in under two crossing times, and the snapshot cadence is half a
    crossing time, so a stricter cut would silently drop the highest-resolution
    astronomix run rather than measure it. The count is returned alongside so a
    thin fit can be seen for what it is.
    """
    tc = np.asarray(run["t_over_tc"])
    E_B, E_K = np.asarray(run["E_B"]), np.asarray(run["E_K"])
    r = E_B / np.maximum(E_K, 1e-30)
    m = (tc >= t_min) & (r >= band[0]) & (r <= band[1]) & (E_B > 0)
    g = (float(np.polyfit(tc[m], np.log(E_B[m]), 1)[0]) if m.sum() >= 3
         else np.nan)
    return (g, int(m.sum())) if with_count else g


#: Prefactor of the growth-rate collapse, from :func:`collapse`. Used to price
#: the advantage in grid cells, and to stand in for a growth rate the snapshot
#: cadence was too coarse to fit.
COLLAPSE_C = 0.0198

REFERENCE = "astronomix WENO5+CT"


def summary(rows, args):
    """``Re``, ``Rm`` and ``Pm`` per scheme and resolution, with everything a
    reader needs to decide how far to trust each row.

    Measured at matched ``E_B/E_K`` (see :data:`MATCH_RATIO`), because each
    scheme saturates at its own magnetic energy fraction and ``Pm`` drifts with
    it. Errors combine a moving-block bootstrap over snapshots with the spread
    over every defensible band and window. ``Re_L``/``Rm_L`` repeat the two
    Reynolds numbers with each run's own measured integral scale in place of the
    fixed driving wavelength, which is the only part of the convention that does
    not cancel between codes.
    """
    print(f"\nAt matched E_B/E_K = {MATCH_RATIO:g}. L = {L_INJ} (driving "
          f"wavelength) for Re/Rm; L_spec is each run's measured integral "
          f"scale, used for Re_L/Rm_L.")
    print(f"\n{'scheme':22s} {'ord':>3s} {'divB':>4s} {'N':>4s} {'nu_eff':>9s} "
          f"{'eta_eff':>9s} {'Re':>6s} {'Rm':>6s} {'Pm':>15s} {'L_spec':>7s} "
          f"{'Re_L':>6s} {'Rm_L':>6s} {'ok?':>4s}")
    lines = ["| scheme | order | div.B | N | `nu_eff` | `eta_eff` | `Re` | `Rm` "
             "| `Pm` | `Rm_L` |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        run = _reload(r)
        m = measure_at_ratio(run)
        if m is None:
            continue
        err = float(np.hypot(bootstrap(run, sat_start=args.sat_start)["Pm"],
                             systematic(run, sat_start=args.sat_start)))
        ok = "" if r["n_kolmogorov"] / r["n_nyquist"] <= RESOLVED_MAX else "n_K!"
        print(f"{r['label'][:22]:22s} {r['order']:3d} {r['divb']:>4s} {r['N']:4d} "
              f"{m['nu']:9.2e} {m['eta']:9.2e} {m['Re']:6.0f} {m['Rm']:6.0f} "
              f"{m['Pm']:7.3f}+-{err:6.3f} {m['L_spec']:7.3f} {m['Re_L']:6.0f} "
              f"{m['Rm_L']:6.0f} {ok:>4s}")
        lines.append(f"| {r['label']} | {r['order']} | {r['divb']} | {r['N']} | "
                     f"{m['nu']:.2e} | {m['eta']:.2e} | {m['Re']:.0f} | "
                     f"{m['Rm']:.0f} | **{m['Pm']:.2f} ± {err:.2f}** | "
                     f"{m['Rm_L']:.0f} |")
    print(f"\n'n_K!' marks a run whose implied Kolmogorov scale lies beyond "
          f"Nyquist, where the Laplacian reading of nu is an extrapolation.")
    out = Path(args.out).with_name("mechanism_summary.md")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def accounting(rows):
    """What the CT scheme's advantage is made of, and what it is worth in cells.

    Three questions, in order: which diffusivity actually differs, how the
    growth-rate ratio splits between ``Rm`` and ``Pm``, and what grid each
    AthenaPK scheme would need to reach the same dynamo.
    """
    d = {}
    for r in rows:
        run = _reload(r)
        m = measure_at_ratio(run)
        if m:
            d[(r["label"], r["N"])] = dict(m, gamma=growth(run))
    ref = {n: v for (l, n), v in d.items() if l == REFERENCE}
    if not ref:
        raise SystemExit(f"no {REFERENCE} run to compare against")

    print(f"\nRatios {REFERENCE} / AthenaPK, at matched E_B/E_K = "
          f"{MATCH_RATIO:g}. A ratio below 1 means astronomix is the less "
          f"dissipative of the two.")
    print(f"{'':30s} {'N':>4s} {'nu':>6s} {'eta':>6s} {'Rm':>6s} {'Pm':>6s} "
          f"{'Gamma':>6s} {'sqrt(Rm Pm)':>12s}")
    for (l, n), v in sorted(d.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        if l == REFERENCE or n not in ref:
            continue
        a = ref[n]
        g = a["gamma"] / v["gamma"] if np.isfinite(a["gamma"]) else np.nan
        print(f"{l[:30]:30s} {n:4d} {a['nu'] / v['nu']:6.2f} "
              f"{a['eta'] / v['eta']:6.2f} {a['Rm'] / v['Rm']:6.2f} "
              f"{a['Pm'] / v['Pm']:6.2f} {g:6.2f} "
              f"{np.sqrt(a['Rm'] / v['Rm'] * a['Pm'] / v['Pm']):12.2f}")

    print("\nHow Rm scales with the grid. Order moves the prefactor, not the "
          "exponent, so it buys a constant factor rather than a better slope.")
    fit = {}
    for l in sorted({l for l, _ in d}):
        pts = sorted((n, d[(l, n)]["Rm"]) for (ll, n) in d if ll == l)
        if len(pts) >= 2:
            fit[l] = np.polyfit(np.log([q[0] for q in pts]),
                                np.log([q[1] for q in pts]), 1)
            print(f"  {l[:28]:28s} Rm = {np.exp(fit[l][1]):5.2f} N^{fit[l][0]:.2f}"
                  f"   (N = " + ", ".join(str(q[0]) for q in pts) + ")")

    print(f"\nGrid each GLM scheme needs for astronomix's dynamo, from "
          f"Gamma = {COLLAPSE_C:g} sqrt(Rm Pm) and the fits above.")
    for l, a in sorted(fit.items()):
        if l == REFERENCE:
            continue
        Pm = float(np.mean([d[(ll, n)]["Pm"] for (ll, n) in d if ll == l]))
        for n in sorted(ref):
            g = ref[n]["gamma"]
            predicted = not np.isfinite(g)
            if predicted:
                g = COLLAPSE_C * np.sqrt(ref[n]["Rm"] * ref[n]["Pm"])
            need = np.exp((np.log((g / (COLLAPSE_C * np.sqrt(Pm))) ** 2)
                           - a[1]) / a[0])
            print(f"  astronomix {n:3d}^3 (Gamma = {g:.3f}"
                  f"{', predicted' if predicted else ', measured'})  ->  "
                  f"{l[:24]:24s} {need:5.0f}^3  "
                  f"({need / n:4.1f}x linear, {(need / n) ** 3:4.0f}x cells)")


def growth_from_history(run, band=COLLAPSE_BAND, t_min=2.0):
    """The same growth rate from the ``.hst`` series, which is ~800 rows deep.

    An independent check on the snapshot fit, which at the highest resolutions
    has only three points inside the decade. AthenaPK writes a history file;
    astronomix does not, and is checked instead against its densely sampled
    zero-net-flux run.
    """
    if "hst_ME" not in run or "hst_KE" not in run:
        return np.nan, 0
    t = np.asarray(run["hst_time"]) / float(run["t_cross"])
    E_B, E_K = np.asarray(run["hst_ME"]), np.asarray(run["hst_KE"])
    r = E_B / np.maximum(E_K, 1e-30)
    m = (t >= t_min) & (r >= band[0]) & (r <= band[1]) & (E_B > 0)
    if m.sum() < 3:
        return np.nan, 0
    return float(np.polyfit(t[m], np.log(E_B[m]), 1)[0]), int(m.sum())


def collapse(rows):
    """Which of ``Rm``, ``Rm Pm`` or ``N`` orders the growth rate.

    Fitting exponents on seven correlated points proves little on its own, so
    the test that matters is out-of-sample: hold a whole *scheme* out, fit on
    the rest, and see how badly the held-out scheme is predicted. A law that is
    really about ``Rm`` and ``Pm`` should transfer between schemes; one that is
    secretly about resolution, or about the estimator, should not.
    """
    pts = []
    for r in rows:
        run = _reload(r)
        m = measure_at_ratio(run)
        g, npts = growth(run, with_count=True)
        if m is None or not np.isfinite(g):
            continue
        gh, nh = growth_from_history(run)
        pts.append(dict(label=r["label"], N=r["N"], Rm=m["Rm"], Pm=m["Pm"], g=g,
                        npts=npts, g_hst=gh, npts_hst=nh))
    if len(pts) < 4:
        raise SystemExit("too few runs with both a growth rate and a diffusivity")

    def design(sel, terms):
        return np.column_stack([np.ones(len(sel))]
                               + [np.log([p[t] for p in sel]) for t in terms])

    print(f"\n{len(pts)} points. In-sample and leave-one-scheme-out residuals:")
    print(f"{'model':22s} {'exponents':>26s} {'in-sample':>10s} {'LOSO':>8s}")
    for name, terms in (("Gamma ~ Rm", ("Rm",)), ("Gamma ~ N", ("N",)),
                        ("Gamma ~ Rm Pm", ("Rm", "Pm")),
                        ("Gamma ~ Rm N", ("Rm", "N"))):
        y = np.log([p["g"] for p in pts])
        c = np.linalg.lstsq(design(pts, terms), y, rcond=None)[0]
        rms = np.std(y - design(pts, terms) @ c)
        loso = []
        for held in {p["label"] for p in pts}:
            tr = [p for p in pts if p["label"] != held]
            te = [p for p in pts if p["label"] == held]
            if len(tr) <= len(terms) + 1:
                continue
            ct = np.linalg.lstsq(design(tr, terms),
                                 np.log([p["g"] for p in tr]), rcond=None)[0]
            loso += list(np.log([p["g"] for p in te]) - design(te, terms) @ ct)
        exps = " ".join(f"{t}^{c[i + 1]:+.2f}" for i, t in enumerate(terms))
        print(f"{name:22s} {exps:>26s} {100 * rms:9.1f}% "
              f"{100 * np.std(loso) if loso else float('nan'):7.1f}%")

    # With the exponents fixed a priori only the normalisation is fitted, so
    # leave-one-scheme-out is well posed even with four schemes -- unlike the
    # free fits above, where holding a scheme out removes most of the leverage
    # on the exponents and the LOSO number says more about the sample size than
    # about the model.
    y = np.log([p["g"] for p in pts])
    print("\nexponents fixed a priori, only the prefactor fitted:")
    print(f"{'model':22s} {'prefactor':>10s} {'in-sample':>10s} {'LOSO':>8s}")
    for name, a, b in (("sqrt(Rm)", 0.5, 0.0), ("sqrt(Rm Pm)", 0.5, 0.5),
                       ("sqrt(Rm) Pm", 0.5, 1.0), ("Rm Pm", 1.0, 1.0)):
        x = np.array([a * np.log(p["Rm"]) + b * np.log(p["Pm"]) for p in pts])
        c = np.mean(y - x)
        loso = []
        for held in {p["label"] for p in pts}:
            tr = [i for i, p in enumerate(pts) if p["label"] != held]
            te = [i for i, p in enumerate(pts) if p["label"] == held]
            ct = np.mean(y[tr] - x[tr])
            loso += list(y[te] - x[te] - ct)
        print(f"Gamma ~ {name:14s} {np.exp(c):10.4f} {100 * np.std(y - x - c):9.1f}% "
              f"{100 * np.std(loso):7.1f}%")
    print(f"\n{'scheme':26s} {'N':>4s} {'Rm':>6s} {'Pm':>5s} {'Gamma':>7s} "
          f"{'G/sqrt(RmPm)':>12s} {'fit pts':>7s} {'Gamma from .hst':>15s}")
    for p_ in pts:
        chk = (f"{p_['g_hst']:7.3f} ({p_['npts_hst']:3d} pts)"
               if np.isfinite(p_["g_hst"]) else f"{'--':>15s}")
        print(f"{p_['label'][:26]:26s} {p_['N']:4d} {p_['Rm']:6.0f} {p_['Pm']:5.2f} "
              f"{p_['g']:7.3f} {p_['g'] / np.sqrt(p_['Rm'] * p_['Pm']):12.4f} "
              f"{p_['npts']:7d} {chk:>15s}")
    print("\nThe last column refits the same decade on the ~800-row history "
          "series; agreement to a few percent\nshows the thin snapshot fits at "
          "256^3 are not the reason any point sits off the line.")


def audit(rows):
    print("\n=== validity checks ===")
    print(f"{'run':26s} {'N':>4s} {'Parseval':>8s} {'closure/inj':>11s} "
          f"{'negD_v':>6s} {'negD_B':>6s} {'n_K/n_Nyq':>9s} {'n_eta/n_Nyq':>11s} "
          f"{'eps_v':>6s} {'eps_B':>6s} {'nsnap':>5s}")
    for r in rows:
        flag = ("  <-- unresolved"
                if r["n_kolmogorov"] / r["n_nyquist"] > RESOLVED_MAX else "")
        print(f"{r['label'][:26]:26s} {r['N']:4d} {r['parseval']:8.4f} "
              f"{r['closure'] / abs(r['forcing']):11.3f} "
              f"{r['neg_v']:6.2f} {r['neg_B']:6.2f} "
              f"{r['n_kolmogorov'] / r['n_nyquist']:9.2f} "
              f"{r['n_resistive'] / r['n_nyquist']:11.2f} "
              f"{r['eps_v']:6.3f} {r['eps_B']:6.3f} {r['n_snapshots']:5d}{flag}")
    print("\nParseval: sum_{n>=1} E_v(n) / (<v_rms^2> / 2), <= 1 always. The "
          "shortfall is the discarded\ncube corners (0.01%) plus the n = 0 mean "
          "flow, which is 1e-4 for AthenaPK (it zeroes the\nnet momentum every "
          "step) but 1e-2 for astronomix at 64^3, falling to 8e-4 at 256^3.")
    print("closure/inj: sum_n [D_v + D_B] over the net injection at n <= 3; in a "
          "steady state dissipation balances injection, so this should be 0.")
    print("negD: fraction of in-band shells where D has the wrong sign for a "
          "dissipation. A non-zero entry falsifies the dissipative reading.")
    print(f"n_K/n_Nyq: Kolmogorov shell implied by the measured nu, over Nyquist. "
          f"Above {RESOLVED_MAX} the Laplacian reading extrapolates off the grid.")

    print("\n=== Pm shell by shell, and over every defensible band ===")
    print(f"{'run':26s} {'N':>4s} {'Pm(n) min':>9s} {'max':>6s} {'slope':>6s} | "
          + " ".join(f"{str(b):>12s}" for b in BAND_VARIANTS))
    for r in rows:
        run = _reload(r)
        n = np.asarray(run["n_shell"], dtype=float)
        inb = (n / (r["N"] / 2) >= BAND[0]) & (n / (r["N"] / 2) <= BAND[1])
        pm = r["Pm_shells"]
        slope = np.polyfit(np.log(n[inb]), np.log(pm), 1)[0]
        cells = []
        for b in BAND_VARIANTS:
            m = measure(run, band=b)
            cells.append(f"{m['Pm']:12.3f}" if m else f"{'--':>12s}")
        print(f"{r['label'][:26]:26s} {r['N']:4d} {pm.min():9.3f} {pm.max():6.3f} "
              f"{slope:6.2f} | " + " ".join(cells))
    print("\nThe first three bands lie entirely below the 2/3 dealiasing cutoff.")

    print(f"\n=== two independent conditioning choices ===")
    print(f"{'run':26s} {'N':>4s} {'t/tc at match':>13s} {'Mach':>6s} {'Mach (fixed)':>12s} "
          f"{'Pm @ ratio':>10s} {'Pm @ fixed t':>12s} {'diff':>6s}")
    for r in rows:
        run = _reload(r)
        a, b = measure_at_ratio(run), measure_in_window(run)
        if a is None or b is None:
            continue
        print(f"{r['label'][:26]:26s} {r['N']:4d} {a['t_over_tc']:13.1f} "
              f"{a['mach']:6.3f} {b['mach']:12.3f} {a['Pm']:10.3f} {b['Pm']:12.3f} "
              f"{100 * (b['Pm'] / a['Pm'] - 1):5.1f}%")
    print(f"\nMatching on E_B/E_K = {MATCH_RATIO:g} reaches each scheme at a "
          f"different time; the fixed window {FIXED_WINDOW} reaches each at a\n"
          f"different E_B/E_K. Agreement between the two columns means neither "
          f"the OU forcing phase nor the choice of conditioning is doing the work.")


def calibration(data_dir, sat_start):
    """Measured diffusivity against an explicitly imposed one.

    The only test that pins the estimator to something known. AthenaPK is given
    a fixed Laplacian ``eta`` (or ``nu``) on top of its own numerical one, and
    the budget is asked to find it. The increment per unit imposed coefficient
    should be one -- *until* the imposed value starts to dominate, at which
    point the numerical part is genuinely displaced (a smoother field is less
    numerically dissipated) and the increment falls below one for a physical
    reason rather than a diagnostic one. Both effects are printed so they can be
    told apart.
    """
    rows = [m for m in (measure(r, sat_start=sat_start)
                        for r in load_runs(data_dir, skip=("smoke",))) if m]
    if not rows:
        raise SystemExit(f"no calibration run in {data_dir}")

    print(f"\n{'eta_imp':>9s} {'nu_imp':>9s} | {'eta_meas':>10s} {'nu_meas':>10s} | "
          f"{'eta_meas-eta_imp':>16s} {'nu_meas-nu_imp':>14s} | {'E_B/E_K':>7s}")
    for r in sorted(rows, key=lambda r: (r["mom_diff"], r["ohm_diff"])):
        print(f"{r['ohm_diff']:9.2e} {r['mom_diff']:9.2e} | {r['eta']:10.3e} "
              f"{r['nu']:10.3e} | {r['eta'] - r['ohm_diff']:16.3e} "
              f"{r['nu'] - r['mom_diff']:14.3e} | {r['ratio']:7.3f}")

    for q, imposed, other in (("eta", "ohm_diff", "mom_diff"),
                              ("nu", "mom_diff", "ohm_diff")):
        ladder = sorted((r for r in rows if r[other] == 0.0),
                        key=lambda r: r[imposed])
        if len(ladder) < 2:
            continue
        print(f"\n  {q}: increment per unit imposed coefficient, step by step")
        for lo, hi in zip(ladder, ladder[1:]):
            d = (hi[q] - lo[q]) / (hi[imposed] - lo[imposed])
            print(f"    {lo[imposed]:.1e} -> {hi[imposed]:.1e}:  {d:6.3f}"
                  + ("   <-- imposed now exceeds the numerical part"
                     if lo[imposed] >= ladder[0][q] else ""))
        # Cross-talk: what the *other* coefficient does when this one is imposed.
        cross = [r for r in rows if r[other] > 0]
        if cross:
            base = [r for r in rows if r[imposed] == 0 and r[other] == 0][0]
            worst = max(cross, key=lambda r: abs(r[q] / base[q] - 1.0))
            print(f"  cross-talk: imposing {other.split('_')[0]} = "
                  f"{worst[other]:.1e} moves the measured {q} by "
                  f"{100 * (worst[q] / base[q] - 1):+.0f}%")


if __name__ == "__main__":
    main()
