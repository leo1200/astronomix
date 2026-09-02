"""Effective (numerical) Reynolds numbers of each scheme, against resolution.

Neither code has an explicit viscosity or resistivity: both dissipate through
truncation error. This measures what that dissipation amounts to, by reading off
*where* each field is dissipated and converting to a Reynolds number:

    n_nu   dissipation-weighted mean shell of the kinetic spectrum
    n_eta  the same for the magnetic spectrum, in the kinematic eigenmode window
    Re     (n_nu / n_inj)^(4/3),  Rm  (n_eta / n_inj)^(4/3),  Pm  Rm / Re

Both shells are measured in the eigenmode window, where the field is still a
passive tracer -- afterwards the back-reaction moves the magnetic peak to large
scales and it stops marking the resistive scale. The runs therefore have to be
the zero-net-flux, weak-seed ones (`data/reynolds/`); the production ladder's
uniform seed has no eigenmode window at all (see the README).

An independent check comes free: the kinematic growth rate obeys
``Gamma ~ eps^(1/3) l_eta^(-2/3)``, so ``n_eta ~ Gamma^(3/2)``. If the shell
measured off the spectrum and the one implied by the growth rate scale together,
both are measuring the same resistive scale.

    python make_reynolds_figure.py --data data/reynolds
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
    E_MAG, E_V, N_INJECTION, dissipation_shell, dynamo_time_series,
    eigenmode_growth_rate, eigenmode_window_mask, load_runs, reynolds_numbers,
    spectra_of,
)
from make_convergence_figures import SERIES, series_of

HERE = Path(__file__).resolve().parent


def measure(run, deconvolve=True):
    """Every Reynolds-number quantity for one run."""
    n = np.asarray(run["n_shell"], dtype=float)
    spec = spectra_of(run, deconvolve)
    mask = eigenmode_window_mask(run)
    if mask.sum() < 2:
        return None
    Ev = spec[mask, E_V].mean(0)
    Eb = spec[mask, E_MAG].mean(0)
    n_nu = dissipation_shell(n, Ev)
    n_eta = dissipation_shell(n, Eb)
    Re, Rm, Pm = reynolds_numbers(n_nu, n_eta)
    tc, E_B, E_K = dynamo_time_series(run)
    gamma, spread, per_decade, _ = eigenmode_growth_rate(tc, E_B, E_K)
    v_rms = float(np.asarray(run["v_rms"])[mask].mean())
    return dict(
        label=str(run["label"]), series=series_of(run), N=int(run["N"]),
        n_nu=n_nu, n_eta=n_eta, Re=Re, Rm=Rm, Pm=Pm,
        nu_eff=v_rms * 0.5 / Re, eta_eff=v_rms * 0.5 / Rm,
        gamma=gamma, gamma_spread=spread, n_snapshots=int(mask.sum()),
        v_rms=v_rms,
    )


def _fit(x, y):
    good = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if good.sum() < 2:
        return np.nan
    return float(np.polyfit(np.log(x[good]), np.log(y[good]), 1)[0])


def figure(rows, out):
    panels = [("n_nu", r"viscous shell $n_\nu$", r"$n_\nu$"),
              ("n_eta", r"resistive shell $n_\eta$", r"$n_\eta$"),
              ("Re", r"effective $Re = (n_\nu/n_{\rm inj})^{4/3}$", "Re"),
              ("Rm", r"effective $Rm = (n_\eta/n_{\rm inj})^{4/3}$", "Rm"),
              ("Pm", r"effective $Pm = Rm/Re$", "Pm"),
              ("gamma", r"kinematic growth rate $\Gamma\, t_{\rm cross}$", r"$\Gamma t_{\rm cross}$")]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2))
    by_series = {}
    for r in rows:
        by_series.setdefault(r["series"], []).append(r)
    for v in by_series.values():
        v.sort(key=lambda r: r["N"])

    for ax, (field, title, ylab) in zip(axes.ravel(), panels):
        for key, group in by_series.items():
            color, label = SERIES[key]
            N = np.array([g["N"] for g in group], dtype=float)
            y = np.array([g[field] for g in group], dtype=float)
            slope = _fit(N, y)
            ax.plot(N, y, "o-", color=color, lw=1.9, ms=6,
                    label=f"{label}" + (f"  ($\\propto N^{{{slope:.2f}}}$)"
                                        if np.isfinite(slope) else ""))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7)
    axes[1, 1].axhline(1.0, color="0.5", lw=0.9, ls="--")
    fig.suptitle("Effective (numerical) Reynolds numbers, measured in the "
                 "kinematic eigenmode window.\nAbsolute values carry an "
                 "order-unity convention; the ratios between schemes and the "
                 "scaling with N are the results.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


COLUMNS = [("label", "scheme", "{}"), ("N", "N", "{:d}"),
           ("n_nu", "n_nu", "{:.2f}"), ("n_eta", "n_eta", "{:.2f}"),
           ("Re", "Re", "{:.0f}"), ("Rm", "Rm", "{:.0f}"), ("Pm", "Pm", "{:.2f}"),
           ("nu_eff", "nu_eff", "{:.2e}"), ("eta_eff", "eta_eff", "{:.2e}"),
           ("gamma", "Gamma t_cr", "{:.3f}"),
           ("n_snapshots", "snaps", "{:d}")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(HERE / "data" / "reynolds"))
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--exclude", nargs="*", default=())
    args = p.parse_args()

    runs = load_runs(args.data, skip=("smoke", "calib", *args.exclude))
    rows = [m for m in (measure(r) for r in runs) if m is not None]
    if not rows:
        raise SystemExit(f"no run in {args.data} has an eigenmode window with spectra")
    rows.sort(key=lambda r: (r["series"], r["N"]))

    out_dir = Path(args.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure(rows, out_dir / "dynamo_reynolds.png")

    head = "| " + " | ".join(c[1] for c in COLUMNS) + " |"
    lines = [head, "|" + "|".join("---" for _ in COLUMNS) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            (r[f] if isinstance(r[f], str) else
             (fmt.format(r[f]) if np.isfinite(r[f]) else "-"))
            for f, _, fmt in COLUMNS) + " |")
    table = "\n".join(lines)
    print()
    print(table)
    (Path(args.data) / "reynolds.md").write_text(
        "# Effective (numerical) Reynolds numbers\n\n"
        f"Measured in the kinematic eigenmode window, referred to the driving "
        f"shell n_inj = {N_INJECTION:g}. Absolute values carry an order-unity "
        "convention from the Kolmogorov scale relation; the ratios between "
        "schemes and the scaling with N are the results.\n\n" + table + "\n")
    print(f"\nwrote {Path(args.data) / 'reynolds.md'}")


if __name__ == "__main__":
    main()
