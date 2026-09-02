"""The three-panel summary of the numerical Reynolds and Prandtl numbers.

``Re``, ``Rm`` and ``Pm`` against resolution, one panel each, all measured from
the spectral energy budget at matched ``E_B/E_K`` (see
``make_mechanism_table.py`` for the definitions and the audit behind them).

The point of putting them side by side is that the first two panels look the
same for every scheme up to a prefactor -- both diffusivities fall together as
the order rises, so ``Re`` and ``Rm`` both scale as ``N^1.2`` -- while the third
splits the schemes into two groups that do not converge. Open markers are runs
the resolvedness check flags, where the Kolmogorov scale implied by the measured
``nu`` lies beyond Nyquist and ``Re`` is partly an extrapolation.

    python make_mechanism_figure.py
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
from make_convergence_figures import SERIES, series_of
from make_mechanism_table import (MATCH_RATIO, RESOLVED_MAX, bootstrap, collect,
                                  measure_at_ratio, systematic, _reload)

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", nargs="*",
                   default=[str(HERE / "data" / "dissipation"),
                            str(HERE / "data" / "dissipation_mech")])
    p.add_argument("--figures", default=str(HERE / "figures"))
    args = p.parse_args()

    rows = collect(args.data)
    per_scheme = {}
    for r in rows:
        run = _reload(r)
        m = measure_at_ratio(run)
        if m is None:
            continue
        err = float(np.hypot(bootstrap(run)["Pm"], systematic(run)))
        per_scheme.setdefault(series_of(run), []).append(
            (r["N"], m["Re"], m["Rm"], m["Pm"], err,
             r["n_kolmogorov"] / r["n_nyquist"]))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    for key, pts in per_scheme.items():
        pts.sort()
        N = np.array([q[0] for q in pts], dtype=float)
        colour, label = SERIES[key]
        resolved = np.array([q[5] <= RESOLVED_MAX for q in pts])
        for ax, col in zip(axes, (1, 2, 3)):
            y = np.array([q[col] for q in pts])
            # The line carries the legend entry, so a scheme with no resolved
            # point (PPM, WENO-Z) still appears.
            ax.plot(N, y, "-", color=colour, lw=1.8, alpha=0.9, zorder=2,
                    label=label if col == 1 else None)
            # Filled = the diffusivity measurement is resolved on this grid.
            for mask, face in ((resolved, colour), (~resolved, "white")):
                if mask.any():
                    ax.plot(N[mask], y[mask], "o", color=colour, mfc=face,
                            ms=7, mew=1.6, zorder=3)
        err = np.array([q[4] for q in pts])
        axes[2].errorbar(N, [q[3] for q in pts], yerr=err, fmt="none",
                         ecolor=colour, elinewidth=1.4, capsize=3, zorder=2)

    guide = np.array([64.0, 256.0])
    for ax, anchor in zip(axes[:2], (600.0, 320.0)):
        ax.plot(guide, anchor * (guide / 64.0) ** 1.2, "k:", lw=1.2, zorder=1)
        ax.text(150, anchor * (150 / 64.0) ** 1.2 * 0.55, r"$\propto N^{1.2}$",
                fontsize=9, color="0.35")

    for ax, ttl, ylab in zip(
            axes,
            (r"effective $Re = v_{\rm rms} L / \nu_{\rm eff}$",
             r"effective $Rm = v_{\rm rms} L / \eta_{\rm eff}$",
             r"magnetic Prandtl number $Pm = \nu_{\rm eff} / \eta_{\rm eff}$"),
            (r"$Re$", r"$Rm$", r"$Pm$")):
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 256])
        ax.set_xticklabels(["$64^3$", "$128^3$", "$256^3$"])
        ax.set_xlabel("resolution")
        ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_ylim(0.0, 1.45)
    axes[2].axhline(1.0, color="0.4", lw=1.0, ls="--")
    axes[2].text(70, 1.03, r"$Pm = 1$", fontsize=8, color="0.4")
    handles, labels = axes[0].get_legend_handles_labels()
    order = np.argsort([l.split("(")[-1] for l in labels])
    axes[0].legend([handles[i] for i in order], [labels[i] for i in order],
                   fontsize=8, loc="upper left")

    fig.suptitle(
        f"Numerical Reynolds and Prandtl numbers from the spectral energy "
        f"budget, at matched $E_B/E_K = {MATCH_RATIO:g}$.\n"
        f"Order sets the prefactor of $Re$ and $Rm$ and leaves $Pm$ alone; the "
        f"divergence treatment sets $Pm$. Open markers: "
        f"$n_K / n_{{\\rm Nyq}} > {RESOLVED_MAX:g}$, where $Re$ is an "
        f"extrapolation.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = Path(args.figures) / "dynamo_mechanism.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
