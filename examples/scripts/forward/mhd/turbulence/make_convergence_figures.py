"""Figures and convergence metrics for the astronomix / AthenaPK dynamo study.

Reads every reduced run in ``data/`` (written by ``dynamo_convergence.py`` and
``athenapk_turb.py``) and produces

    figures/dynamo_time_series.png   E_B(t), E_B/E_K(t), <|B|>(t), Mach(t)
    figures/dynamo_spectra_saturated.png       saturated state, all resolutions
    figures/dynamo_spectra_kinematic.png       kinematic phase, all resolutions
    figures/dynamo_spectra_saturated_N<max>.png  the finest grid alone, converged
    figures/dynamo_convergence.png             the metrics against resolution
    figures/dynamo_runtime.png                 raw wall clock (see README)
    data/metrics.md                            the table the figures visualise

Four things are being measured, all of them functions of resolution:

  *The dynamo itself* -- the kinematic growth rate ``d ln E_B / dt`` and the
  saturated ``E_B / E_K``. Both are resolution-dependent by construction: the
  small-scale dynamo lives at the grid scale, so what a scheme resolves there is
  what its dynamo does.

  *Effective resolution* -- the shell ``n_1/4`` at which the compensated kinetic
  spectrum has fallen to a quarter of its inertial-range plateau. Above it the
  spectrum is numerical dissipation, so ``n_1/4 / N`` is the fraction of the grid
  a scheme converts into resolved turbulence.

  *The magnetic spectrum's peak* -- where the magnetic energy sits. In an ideal
  MHD dynamo this tracks the resistive scale, which is numerical here, so it
  moves with resolution and with the scheme.

  *Cost* -- raw wall clock at fixed N. NOTE that the ladder runs astronomix in
  x32 and AthenaPK in x64, so those wall clocks are not a like-for-like speed
  comparison; the README carries a precision-matched timing grid instead.

    python examples/scripts/forward/mhd/turbulence/make_convergence_figures.py
"""

# general
import argparse
import sys
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_metrics import (
    E_MAG, E_V, cutoff_shell, dynamo_time_series, kinematic_window, load_runs,
    mean_shell, peak_shell, spectra_of, summarize,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIG_DIR = HERE / "figures"

#: One colour per scheme, one line style per resolution — so a figure reads as
#: "scheme = colour, resolution = dash", the same convention as the hydro study.
SERIES = {
    "astronomix": ("#1f77b4", "astronomix WENO5+CT (5th)"),
    "plm":        ("#d62728", "AthenaPK PLM+VL2 (2nd)"),
    "ppm":        ("#ff9e4a", "AthenaPK PPM+RK3 (3rd)"),
    "limo3":      ("#9467bd", "AthenaPK LimO3+RK3 (3rd)"),
    "wenoz":      ("#2ca02c", "AthenaPK WENO-Z+RK3 (5th)"),
}
STYLE = {32: dict(ls=(0, (1, 3)), lw=1.3, alpha=0.65),
         64: dict(ls=":", lw=1.6, alpha=0.75),
         128: dict(ls="--", lw=1.9, alpha=0.9),
         256: dict(ls="-", lw=2.3, alpha=1.0),
         512: dict(ls="-", lw=2.9, alpha=1.0)}


def series_of(run):
    """Series key of a run: the code for astronomix, the scheme for AthenaPK."""
    return "astronomix" if str(run["code"]) == "astronomix" \
        else str(run["scheme_key"])


def style_of(run):
    color, label = SERIES[series_of(run)]
    st = dict(STYLE.get(int(run["N"]), STYLE[128]))
    st["color"] = color
    return st, label


# -------------------------------------------------------------
# ================== ↓ Figure: time series ↓ ==================
# -------------------------------------------------------------
def figure_time_series(runs, sat_start, deconvolve, out):
    """The dynamo as it happens: growth, saturation, and the driven flow.

    The last panel is the convergence-in-time check the averages depend on:
    ``<n>_B``, the shell the magnetic energy sits at, has to stop moving before
    the shaded window, or the "saturated" spectra are still evolving.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.4), sharex=True)

    for run in runs:
        st, label = style_of(run)
        st["label"] = f"{label}, N={int(run['N'])}"

        t, E_B, E_K = dynamo_time_series(run)
        axes[0, 0].semilogy(t, E_B, **st)
        # The t = 0 point has E_K = 0 (the box starts at rest), so the ratio is
        # only defined once the driving has spun the flow up.
        alive = E_K > 1e-3 * np.max(E_K)
        axes[0, 1].semilogy(t[alive], (E_B / E_K)[alive], **st)
        tc = np.asarray(run["t_over_tc"])
        axes[0, 2].semilogy(tc, np.asarray(run["mean_absB"]), **st)
        axes[1, 0].plot(tc, np.asarray(run["mach"]), **st)
        axes[1, 1].semilogy(tc, np.asarray(run["mach_alfven"]), **st)

        n = np.asarray(run["n_shell"], dtype=float)
        spec = spectra_of(run, deconvolve)
        axes[1, 2].plot(tc, [mean_shell(n, s) for s in spec[:, E_MAG]], **st)

    t_max = max(float(np.asarray(r["t_over_tc"]).max()) for r in runs)
    for ax in axes.ravel():
        ax.set_xlim(0.0, t_max)
        ax.axvspan(sat_start, t_max, color="0.9", zorder=0)
        ax.grid(alpha=0.25)
    axes[0, 0].set_ylabel(r"$E_B = \langle B^2/2 \rangle$")
    axes[0, 0].set_title("magnetic energy: the small-scale dynamo")
    axes[0, 1].set_ylabel(r"$E_B / E_K$")
    axes[0, 1].set_title("saturation level")
    axes[0, 2].set_ylabel(r"$\langle |B| \rangle$")
    axes[0, 2].set_title("mean field strength")
    axes[1, 0].set_ylabel(r"$\mathcal{M} = v_{\rm rms} / a$")
    axes[1, 0].set_title("turbulent Mach number (the matching condition)")
    axes[1, 1].set_ylabel(r"$\mathcal{M}_A = v_{\rm rms} / v_A$")
    axes[1, 1].set_title("Alfvenic Mach number")
    axes[1, 2].set_ylabel(r"$\langle n \rangle_B$")
    axes[1, 2].set_title("shell the magnetic energy sits at\n(convergence in time)")
    for ax in axes[1]:
        ax.set_xlabel(r"$t / t_{\rm cross}$")
    axes[0, 0].legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.suptitle("Driven subsonic MHD turbulence: dynamo growth and saturation.  "
                 "Colour = method, dash = resolution.  "
                 f"Shaded: the window all averages are taken over "
                 f"($t/t_{{\\rm cross}} \\geq {sat_start:g}$).", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------
# ================== ↑ Figure: time series ↑ ==================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ==================== ↓ Figure: spectra ↓ ====================
# -------------------------------------------------------------
def _spectra_legend(ax, runs):
    """Two small legends — one for colour (method), one for dash (resolution).

    One entry per run would be nine near-identical lines; the figure encodes two
    independent things, so the legend should too.
    """
    from matplotlib.lines import Line2D
    seen_series, seen_N = [], []
    for run in runs:
        key, N = series_of(run), int(run["N"])
        if key not in seen_series:
            seen_series.append(key)
        if N not in seen_N:
            seen_N.append(N)
    method = [Line2D([], [], color=SERIES[k][0], lw=2.2, label=SERIES[k][1])
              for k in seen_series]
    if len(seen_N) == 1:
        # Single resolution: the dash carries no information, so one legend.
        ax.legend(handles=method, fontsize=9, loc="lower left",
                  title=f"method  (N = {seen_N[0]})", title_fontsize=9)
        return
    resolution = [Line2D([], [], color="0.35", label=f"N = {N}",
                         **{k: v for k, v in STYLE[N].items() if k != "alpha"})
                  for N in sorted(seen_N)]
    first = ax.legend(handles=method, fontsize=8, loc="lower left",
                      title="method", title_fontsize=8, frameon=True)
    ax.add_artist(first)
    ax.legend(handles=resolution, fontsize=8, loc="upper right",
              title="resolution", title_fontsize=8, frameon=True)


def figure_spectra(runs, window, kin_band, frac, deconvolve, out, title):
    """Time-averaged kinetic and magnetic spectra, method by colour, N by dash.

    Curves of one colour at different dashes show what refinement buys a given
    scheme; curves of one dash in different colours show what the scheme buys at
    fixed cost per cell. Dots mark the kinetic cutoff ``n_1/4`` (where the
    Kolmogorov-compensated spectrum has fallen to a quarter of its ``kin_band``
    plateau) and the peak of the magnetic spectrum.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.6))
    anchor = []

    for run in runs:
        st, _ = style_of(run)
        n = np.asarray(run["n_shell"], dtype=float)
        spec = spectra_of(run, deconvolve)
        mask = window(run)
        Ev, se_v = spec[mask, E_V].mean(0), spec[mask, E_V].std(0, ddof=1) \
            / np.sqrt(mask.sum())
        Eb, se_b = spec[mask, E_MAG].mean(0), spec[mask, E_MAG].std(0, ddof=1) \
            / np.sqrt(mask.sum())
        good = n >= 1
        ng, Evg, Ebg = n[good], Ev[good], Eb[good]

        axes[0].loglog(ng, Evg, **st)
        axes[0].fill_between(ng, (Ev - se_v)[good], (Ev + se_v)[good],
                             color=st["color"], alpha=0.15, lw=0)
        anchor.append(np.interp(3.0, ng, Evg))
        n_cut, _ = cutoff_shell(n, Ev, band=kin_band, frac=frac,
                                exponent=-5.0 / 3.0)
        if np.isfinite(n_cut):
            axes[0].plot([n_cut], [np.interp(n_cut, ng, Evg)], "o",
                         color=st["color"], ms=6, zorder=5)

        axes[1].loglog(ng, Ebg, **st)
        axes[1].fill_between(ng, (Eb - se_b)[good], (Eb + se_b)[good],
                             color=st["color"], alpha=0.15, lw=0)
        n_pk = peak_shell(n, Eb)
        axes[1].plot([n_pk], [np.interp(n_pk, ng, Ebg)], "o",
                     color=st["color"], ms=6, zorder=5)

    # Kolmogorov guide line, anchored a factor of three above the spectra at
    # n = 3 so it sits clear of the curves it is a reference for.
    lo, hi = axes[0].get_ylim()
    n_ref = np.array([3.0, 60.0])
    y_ref = 3.0 * max(anchor) * (n_ref / 3.0) ** (-5.0 / 3.0)
    axes[0].loglog(n_ref, y_ref, color="0.45", lw=1.0, ls=(0, (6, 4)), zorder=1)
    axes[0].annotate(r"$n^{-5/3}$", (n_ref[1] * 1.05, y_ref[1]),
                     color="0.35", fontsize=9, ha="left", va="center")
    axes[0].set_ylim(lo, hi)

    axes[0].set_title(r"kinetic energy spectrum $E_v(n)$"
                      f"    (dots: $n_{{1/{int(1 / frac)}}}$)")
    axes[0].set_ylabel(r"$E_v(n)$")
    axes[1].set_title(r"magnetic energy spectrum $E_B(n)$"
                      "    (dots: spectral peak)")
    axes[1].set_ylabel(r"$E_B(n)$")
    for ax in axes:
        ax.set_xlabel(r"mode number $n = k L / 2\pi$")
        ax.grid(alpha=0.25, which="both")
    _spectra_legend(axes[0], runs)
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.005,
             "shaded: $\\pm$ s.e. over snapshots" + ("; finite-volume spectra "
             "deconvolved for cell averaging" if deconvolve else "; raw spectra"),
             ha="center", fontsize=8, color="0.35")
    fig.tight_layout(rect=(0, 0.025, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------
# ==================== ↑ Figure: spectra ↑ ====================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================== ↓ Figure: convergence ↓ ==================
# -------------------------------------------------------------
def _by_series(summaries):
    out = {}
    for s in summaries:
        out.setdefault(s["series"], []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s["N"])
    return out


def figure_convergence(summaries, out):
    """Every metric against resolution — the point of the whole study."""
    panels = [
        ("n_cut_kin_over_N", r"$n_{1/4} / N$, kinematic phase"
                             "\n(resolved fraction of the grid)", False),
        ("n_cut_over_N", r"$n_{1/4} / N$, saturated state", False),
        ("n_mean_mag", "$\\langle n \\rangle_B$: where the magnetic"
                        "\nenergy sits", True),
        ("gamma_tcross", r"kinematic growth rate  $\Gamma\, t_{\rm cross}$"
                          "\n(fitted over $E_B/E_K = 10^{-3} - 10^{-2}$)", False),
        ("ratio", r"saturated $E_B / E_K$", False),
        ("mach_early", r"Mach number at $t/t_{\rm cross} = 2.5-5$"
                       "\n(the matching condition)", False),
        ("mach", r"saturated Mach number"
                 "\n(lower where the dynamo is stronger)", False),
        ("high_frac_v", "fraction of $E_v$ above $n = N/4$"
                        "\n(grid-scale pile-up)", True),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(19.0, 8.0))
    grouped = _by_series(summaries)

    for ax, (field, title, logy) in zip(axes.ravel(), panels):
        for key, group in grouped.items():
            color, label = SERIES[key]
            Ns = [s["N"] for s in group]
            ys = [s[field] for s in group]
            ax.plot(Ns, ys, "o-", color=color, label=label, lw=1.8, ms=6)
        ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log", base=2)
        ax.set_xlabel("N")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Convergence with resolution")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def figure_runtime(summaries, out, tcross):
    """Wall clock at fixed N, throughput, and wall clock at fixed dynamo quality.

    Raw wall clock as configured, i.e. including each code's diagnostic output
    (82 in-flight spectral reductions for astronomix, 41 HDF5 dumps for
    AthenaPK). Neither is free and they are not equally expensive; the README
    carries the per-output-event costs and the corrected solver-only times.

    The third panel deliberately puts cost against the *dynamo* growth rate
    rather than against the kinetic cutoff ``n_1/4``: a scheme that piles energy
    up at the grid scale scores a high ``n_1/4`` without resolving more cascade,
    so ``n_1/4`` is not a safe quality axis here (see the README).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    grouped = _by_series(summaries)

    for key, group in grouped.items():
        color, label = SERIES[key]
        Ns = np.array([s["N"] for s in group], dtype=float)
        t = np.array([s["t_wall"] for s in group], dtype=float)
        quality = np.array([s["gamma_tcross"] for s in group], dtype=float)
        lab = label
        if len(Ns) >= 2:
            slope = np.polyfit(np.log(Ns), np.log(t), 1)[0]
            lab = f"{label}  ($t \\propto N^{{{slope:.2f}}}$)"
        axes[0].loglog(Ns, t, "o-", color=color, label=lab, lw=1.8, ms=6)
        axes[1].loglog(Ns, np.array([s["zone_updates_per_s"] for s in group]),
                       "o-", color=color, label=label, lw=1.8, ms=6)
        ok = np.isfinite(quality)
        axes[2].loglog(quality[ok], t[ok], "o-", color=color, label=label, lw=1.8, ms=6)

    axes[0].set_xlabel("N"); axes[0].set_ylabel("wall clock [s]")
    axes[0].set_title("cost at fixed resolution")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("cell updates / s")
    axes[1].set_title("throughput")
    axes[2].set_xlabel(r"dynamo growth rate $\Gamma\, t_{\rm cross}$")
    axes[2].set_ylabel("wall clock [s]")
    axes[2].set_title("cost at fixed dynamo")
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    fig.suptitle(f"Runtime: RAW wall clock on a single A100 for {tcross:g} crossing "
                 "times, diagnostic output included, astronomix x32 vs AthenaPK "
                 "x64.\nNot a like-for-like speed comparison — see the "
                 "precision-matched timing grid in the README.", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")
# -------------------------------------------------------------
# ================== ↑ Figure: convergence ↑ ==================
# -------------------------------------------------------------


COLUMNS = [
    ("label", "scheme", "{}"), ("N", "N", "{:d}"),
    ("mach", "Mach sat", "{:.3f}"), ("rho_rms", "rho_rms", "{:.3f}"),
    ("mach_early", "Mach 2.5-5", "{:.3f}"),
    ("kin_ratio_max", "max E_B/E_K there", "{:.3f}"),
    ("n_cut_kin", "n_1/4 kin", "{:.2f}"),
    ("high_frac_v", "E_v(n>N/4)", "{:.3f}"),
    ("n_cut_kin_over_N", "n_1/4/N kin", "{:.3f}"),
    ("n_cut", "n_1/4 sat", "{:.2f}"), ("n_cut_over_N", "n_1/4/N sat", "{:.3f}"),
    ("slope_v_kin", "slope kin", "{:.2f}"), ("slope_v", "slope sat", "{:.2f}"),
    ("n_peak_mag", "peak E_B", "{:.2f}"), ("n_mean_mag", "<n> E_B", "{:.2f}"),
    ("gamma_tcross", "Gamma t_cr", "{:.3f}"), ("gamma_r2", "fit r2", "{:.3f}"),
    ("ratio", "E_B/E_K", "{:.3f}"), ("sat_growth", "resid growth", "{:.3f}"),
    ("mach_alfven", "M_A", "{:.2f}"),
    ("rel_divB", "rel divB", "{:.2e}"),
    ("t_wall", "wall [s] (see README)", "{:.0f}"),
    ("zone_updates_per_s", "cells/s", "{:.2e}"),
]


def metric_table(summaries):
    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    rule = "|" + "|".join("---" for _ in COLUMNS) + "|"
    rows = [head, rule]
    for s in summaries:
        cells = []
        for field, _, fmt in COLUMNS:
            value = s[field]
            if isinstance(value, str):
                cells.append(value)
            else:
                cells.append(fmt.format(value) if np.isfinite(value) else "-")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_DIR))
    p.add_argument("--figures", default=str(FIG_DIR))
    p.add_argument("--sat-start", type=float, default=28.0,
                   help="start of the saturated window, in crossing times")
    p.add_argument("--band", type=float, nargs=2, default=(3.0, 8.0),
                   help="shell band the kinetic spectrum's power law is fitted on")
    p.add_argument("--frac", type=float, default=0.25,
                   help="compensated-plateau fraction defining the cutoff shell")
    p.add_argument("--growth-from", default=str(DATA_DIR / "controls" / "kinematic"),
                   help="directory of short high-cadence scalars-only repeats; "
                        "when a run has a companion there (same code and N) the "
                        "growth-rate fit is taken from it")
    p.add_argument("--exclude", nargs="*", default=(),
                   help="substrings of run-file names to leave out, e.g. "
                        "--exclude ppm")
    p.add_argument("--no-deconvolve", action="store_true",
                   help="do NOT undo AthenaPK's finite-volume cell averaging")
    args = p.parse_args()

    runs = load_runs(args.data, skip=("smoke", "calib", *args.exclude))
    if not runs:
        raise SystemExit(f"no run files in {args.data}")
    deconvolve = not args.no_deconvolve
    growth = {}
    if Path(args.growth_from).is_dir():
        for g in load_runs(args.growth_from):
            growth[(str(g["code"]), int(g["N"]))] = g
        if growth:
            print(f"growth rates from the high-cadence repeats in "
                  f"{args.growth_from}: {sorted(growth)}")
    summaries = []
    for run in runs:
        s = summarize(run, sat_start=args.sat_start, kin_band=tuple(args.band),
                      frac=args.frac, deconvolve=deconvolve,
                      growth_run=growth.get((str(run["code"]), int(run["N"]))))
        s["series"] = series_of(run)
        summaries.append(s)

    fig_dir = Path(args.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_time_series(runs, args.sat_start, deconvolve,
                       fig_dir / "dynamo_time_series.png")
    figure_spectra(runs, lambda r: np.asarray(r["t_over_tc"]) >= args.sat_start,
                   tuple(args.band), args.frac, deconvolve,
                   fig_dir / "dynamo_spectra_saturated.png",
                   "Saturated state, time-averaged over "
                   f"$t/t_{{\\rm cross}} \\geq {args.sat_start:g}$")
    figure_spectra(runs, lambda r: kinematic_window(r)[0],
                   tuple(args.band), args.frac, deconvolve,
                   fig_dir / "dynamo_spectra_kinematic.png",
                   r"Kinematic phase, time-averaged over "
                   r"$t/t_{\rm cross} = 2.5-5$ "
                   r"(same flow in every run, $E_B/E_K \leq 6\%$)")
    # The head-to-head the study comes down to: the finest grid, converged.
    top_n = max(int(r["N"]) for r in runs)
    figure_spectra([r for r in runs if int(r["N"]) == top_n],
                   lambda r: np.asarray(r["t_over_tc"]) >= args.sat_start,
                   tuple(args.band), args.frac, deconvolve,
                   fig_dir / f"dynamo_spectra_saturated_N{top_n}.png",
                   f"Saturated state at ${top_n}^3$, time-averaged over "
                   f"$t/t_{{\\rm cross}} \\geq {args.sat_start:g}$")
    figure_convergence(summaries, fig_dir / "dynamo_convergence.png")
    tcross = max(float(np.asarray(r["t_over_tc"]).max()) for r in runs)
    figure_runtime(summaries, fig_dir / "dynamo_runtime.png", tcross)

    table = metric_table(summaries)
    print()
    print(table)
    (Path(args.data) / "metrics.md").write_text(
        "# Dynamo convergence metrics\n\n"
        f"Saturated window: `t/t_cross >= {args.sat_start:g}`. "
        f"Kinetic cutoff: compensated-plateau fraction {args.frac} over the fitted "
        f"band n = {args.band[0]:g}-{args.band[1]:g}. "
        + ("Finite-volume spectra are deconvolved for cell averaging.\n\n"
           if deconvolve else "Raw (un-deconvolved) spectra.\n\n")
        + table + "\n\n"
        "`wall [s]` is the raw wall clock of each production run as configured "
        "-- astronomix in x32, AthenaPK in x64, each with its own diagnostic "
        "output. It is NOT a like-for-like speed comparison; see the Runtime "
        "section of README.md for the precision-matched timing grid.\n")
    print(f"\nwrote {Path(args.data) / 'metrics.md'}")


if __name__ == "__main__":
    main()
