"""The numerical Reynolds and Prandtl numbers of astronomix's two MHD schemes.

``Re``, ``Rm`` and ``Pm`` against resolution for the finite-difference scheme
(WENO-Z + constrained transport), the same with the upwind dissipation of the
CT edge EMFs at its default strength 0.03 (the setup of the turbulence runs,
drawn when its runs are among ``--data``) and the finite-volume scheme (VL2 +
PLM + HLLD + GLM cleaning), all run by ``dynamo_convergence.py`` on the same driven,
subsonic, isothermal dynamo box with the same forcing and reduced by the same
estimator. The numbers are those of ``make_mechanism_table.py --summary``
(``summary_rows``): the diffusivities from the spectral energy budget,
averaged over ``n / n_Nyq = 0.2-0.7`` at matched ``E_B/E_K = 0.01``, with
``L = 0.5``. Error bars combine a moving-block bootstrap with the spread over
the defensible bands and windows; open markers are runs whose implied
Kolmogorov shell lies beyond 0.8 of Nyquist, where the band-mean ``nu`` is
partly an extrapolation.

    python make_fd_fv_reynolds_figure.py --data data/fd_fv data/fd_fv_uct
    python make_fd_fv_reynolds_figure.py --data data/fd_fv --with-pp
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
from matplotlib.lines import Line2D

# astronomix constants
from astronomix.plotting_helpers.paper_style import (
    FD_COLOR,
    FV_COLOR,
    GUIDE_COLOR,
    TEXT_WIDTH,
)

# astronomix functions
from astronomix.plotting_helpers.paper_style import (
    figure_legend,
    use_paper_style,
)

# the dynamo study's measurement
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_mechanism_table import RESOLVED_MAX, collect, summary_rows

HERE = Path(__file__).resolve().parent

#: The series of the figure, by scheme key: colour, marker, line style and
#: legend label. The schemes carry the colours of all figures of the paper; the
#: options of the finite-difference scheme keep its colour and change the
#: marker and the line style, so the identity never rests on colour alone.
SERIES = {
    "wenoz": (FD_COLOR, "o", "-", "FD: WENO-Z + CT (5th order)"),
    "wenoz_uct0.03": (FD_COLOR, "D", "--", "FD: WENO-Z + CT + upwind EMF"),
    "vl2": (FV_COLOR, "s", "-", "FV: VL2 + PLM + GLM (2nd order)"),
    "wenoz_pp": (FD_COLOR, "^", ":", "FD: WENO-Z + CT, positivity preserving"),
}

#: Series whose fitted exponent is written in the Re and Rm panels; the upwind
#: series runs within a few percent of the plain FD one, so its exponent would
#: repeat the FD one.
ANNOTATED_SERIES = ("wenoz", "vl2", "wenoz_pp")


def fit_power_law(resolutions, values):
    """Exponent ``a`` of ``values ~ N^a``, by least squares in log-log."""
    return float(np.polyfit(np.log(resolutions), np.log(values), 1)[0])


def draw(table, series_keys, figures, stem):
    """The three-panel figure, written as SVG, PDF and PNG.

    Args:
        table: The rows of ``summary_rows``.
        series_keys: Scheme keys to draw, in legend order.
        figures: Output directory.
        stem: Output file name without extension.

    Returns:
        The list of written paths.
    """
    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.75), layout="constrained")
    panels = (("Re", "Re_err", r"$\mathrm{Re} = v_{\rm rms} L / \nu_{\rm eff}$"),
              ("Rm", "Rm_err", r"$\mathrm{Rm} = v_{\rm rms} L / \eta_{\rm eff}$"),
              ("Pm", "Pm_err", r"$\mathrm{Pm} = \nu_{\rm eff} / \eta_{\rm eff}$"))
    all_resolutions = sorted({row["N"] for row in table})

    legend_handles = []
    exponent_labels = []  # (colour, Re exponent, Rm exponent) per annotated series
    for key in series_keys:
        rows = sorted((row for row in table if row["scheme"] == key), key=lambda row: row["N"])
        if not rows:
            continue
        colour, marker, line_style, label = SERIES[key]
        resolutions = np.array([row["N"] for row in rows], dtype=float)
        resolved = np.array([row["resolvedness"] <= RESOLVED_MAX for row in rows])
        for ax, (quantity, error, _) in zip(axes, panels):
            values = np.array([row[quantity] for row in rows])
            errors = np.array([row[error] for row in rows])
            ax.plot(resolutions, values, linestyle=line_style, color=colour, zorder=2)
            ax.errorbar(resolutions, values, yerr=errors, fmt="none", ecolor=colour,
                        elinewidth=1.0, capsize=2.5, zorder=2)
            for mask, face in ((resolved, colour), (~resolved, "white")):
                if mask.any():
                    ax.plot(resolutions[mask], values[mask], marker, color=colour,
                            mfc=face, ms=5, mew=1.2, ls="", zorder=3)
        # The fitted exponents of the two panels whose scaling with N is the
        # statement.
        if len(rows) >= 2 and key in ANNOTATED_SERIES:
            exponent_labels.append((colour, *(
                fit_power_law(resolutions, np.array([row[quantity] for row in rows]))
                for quantity in ("Re", "Rm")
            )))
        legend_handles.append(Line2D([], [], color=colour, marker=marker, mfc=colour,
                                     ms=5, linestyle=line_style, label=label))

    # The exponents inside the Re and Rm panels, in the empty lower right
    # corner (the lines rise to the right), in the colour of their scheme and
    # stacked in the order of the lines (the first series is the top line).
    for panel, ax in enumerate(axes[:2]):
        for row, (colour, *exponents) in enumerate(reversed(exponent_labels)):
            ax.text(0.97, 0.05 + 0.1 * row, rf"$\propto N^{{{exponents[panel]:.2f}}}$",
                    transform=ax.transAxes, fontsize=8, color=colour, ha="right", va="bottom")

    for ax, (quantity, _, title) in zip(axes, panels):
        ax.set_xscale("log", base=2)
        ax.set_xticks(all_resolutions)
        ax.set_xticklabels([f"${N}^3$" for N in all_resolutions])
        ax.minorticks_off()
        ax.set_xlim(all_resolutions[0] / 1.25, all_resolutions[-1] * 1.25)
        ax.set_xlabel("resolution")
        ax.set_title(title)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[0].set_ylabel(r"$\mathrm{Re}$")
    axes[1].set_ylabel(r"$\mathrm{Rm}$")
    axes[2].set_ylabel(r"$\mathrm{Pm}$")
    pm_top = max(row["Pm"] + row["Pm_err"] for row in table if row["scheme"] in series_keys)
    axes[2].set_ylim(0.0, max(1.45, 1.15 * pm_top))
    axes[2].axhline(1.0, color=GUIDE_COLOR, lw=0.8, ls="--", zorder=1)
    axes[2].text(all_resolutions[-1] * 1.2, 0.97, r"$\mathrm{Pm} = 1$", fontsize=8, color=GUIDE_COLOR,
                 va="top", ha="right")

    legend_handles.append(Line2D([], [], color="0.3", marker="o", mfc="white", ls="",
                                 ms=5, label=rf"open: $n_K / n_{{\rm Nyq}} > {RESOLVED_MAX:g}$"))
    figure_legend(fig, legend_handles, ncols=2)

    figures.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in ("svg", "pdf", "png"):
        path = figures / f"{stem}.{extension}"
        fig.savefig(path)
        written.append(path)
    plt.close(fig)
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="*", default=[str(HERE / "data" / "fd_fv")],
                        help="directories of dynamo_convergence.py --transfer runs")
    parser.add_argument("--figures", default=str(HERE / "figures"))
    parser.add_argument("--stem", default="fd_fv_reynolds_prandtl")
    parser.add_argument("--with-pp", action="store_true",
                        help="also draw the FD runs with the positivity-preserving WENO")
    args = parser.parse_args()

    rows = [row for row in collect(args.data) if row["code"] == "astronomix"]
    if not rows:
        raise SystemExit(f"no astronomix run with transfer spectra in {args.data}")
    table = summary_rows(rows)
    series_keys = ["wenoz", "wenoz_uct0.03", "vl2"] + (["wenoz_pp"] if args.with_pp else [])

    print(f"{'scheme':42s} {'N':>4s} {'Re':>12s} {'Rm':>12s} {'Pm':>13s} "
          f"{'G t_cross':>9s} {'n_K/n_Nyq':>9s}")
    for key in series_keys:
        for row in sorted((r for r in table if r["scheme"] == key), key=lambda r: r["N"]):
            print(f"{SERIES[key][3]:42s} {row['N']:4d} {row['Re']:6.0f}+-{row['Re_err']:4.0f} "
                  f"{row['Rm']:6.0f}+-{row['Rm_err']:4.0f} {row['Pm']:6.3f}+-{row['Pm_err']:5.3f} "
                  f"{row['gamma']:9.3f} {row['resolvedness']:9.2f}")
    for path in draw(table, series_keys, Path(args.figures), args.stem):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
