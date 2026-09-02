"""Dissipation spectra, and the scale-dependent numerical diffusivity they give.

Neither code has an explicit viscosity or resistivity, so the dissipation
spectrum cannot be formed as ``2 nu k^2 E(k)`` -- ``nu`` is what we are after.
It is measured instead from the spectral energy budget. For each shell,

    dE(n)/dt = T(n) - D(n)

with ``T(n)`` the *ideal* transfer (the exact non-dissipative right-hand side,
projected on the field and shell-summed by ``_mhd_spectral.transfer_spectra``)
and ``D(n)`` everything the scheme threw away. Both terms are measured: ``T(n)``
from each snapshot, ``dE(n)/dt`` by differencing consecutive ones. In the
saturated state the second term averages to zero and ``D(n) = <T(n)>``, which is
where this is cleanest, so that is the window used by default.

Dividing by the shell's own curvature gives the quantity that actually
distinguishes the schemes:

    nu_eff(n) = D_v(n)  / (2 k^2 E_v(n))
    eta_eff(n) = D_B(n) / (2 k^2 E_B(n))

A Laplacian diffusivity is a *constant* here. A p-th order scheme instead gives
``~ (k dx)^(p-1)``: flat-ish for 2nd order, rising steeply for 5th. That is the
"different functional form" claim, plotted rather than argued.

    python make_dissipation_figure.py --data data/dissipation
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
from _mhd_metrics import E_MAG, E_V, load_runs, spectra_of
from make_convergence_figures import SERIES, STYLE, series_of

HERE = Path(__file__).resolve().parent
T_V, T_MAG = 3, 4          # rows added by snapshot_spectra(transfer=True)


def _dEdt(E, t):
    """``dE/dt`` per shell, via the logarithmic derivative.

    ``dE/dt = E d(ln E)/dt`` is an identity, but differencing ``ln E`` and
    multiplying back is exact for exponential growth where differencing ``E``
    directly is not. During the kinematic phase a shell grows by a factor of two
    or more between snapshots, so the distinction is not cosmetic.
    """
    safe = np.maximum(E, 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nan_to_num(safe * np.gradient(np.log(safe), t, axis=0))


def dissipation_series(run, deconvolve=False):
    """Per-snapshot ``D(n, t)`` and ``E(n, t)`` for both fields.

    ``D(n) = T_ideal(n) - dE(n)/dt``, with no time averaging, so the dissipation
    can be watched changing character as the dynamo proceeds.

    ``deconvolve`` is refused: ``spectra_deconv`` divides the cell-average
    transfer function out of the three *energy* spectra but not out of the two
    transfer spectra, so a deconvolved ``D / 2k^2E`` would mix corrected and
    uncorrected rows and lands ~20% low on ``eta``. Both terms are taken raw,
    which makes the ratio a self-consistent property of the stored field.
    """
    if deconvolve:
        raise ValueError("dissipation_series: the transfer spectra are not "
                         "deconvolved, so a deconvolved budget is inconsistent")
    spec = spectra_of(run, deconvolve)
    if spec.shape[1] < 5:
        return None
    t = np.asarray(run["t_over_tc"]) * float(run["t_cross"])
    return dict(n=np.asarray(run["n_shell"], dtype=float),
                t_over_tc=np.asarray(run["t_over_tc"]),
                D_v=spec[:, T_V] - _dEdt(spec[:, E_V], t),
                D_B=spec[:, T_MAG] - _dEdt(spec[:, E_MAG], t),
                E_v=spec[:, E_V], E_B=spec[:, E_MAG],
                ratio=np.asarray(run["E_B"]) / np.maximum(np.asarray(run["E_K"]), 1e-30))


def dissipation(run, t_lo, deconvolve=False):
    """``(n, D_v, D_B, E_v, E_B)`` time-averaged over ``t/t_cross >= t_lo``."""
    ser = dissipation_series(run, deconvolve)
    if ser is None:
        return None
    m = ser["t_over_tc"] >= t_lo
    if m.sum() < 3:
        return None
    return dict(n=ser["n"], D_v=ser["D_v"][m].mean(0), D_B=ser["D_B"][m].mean(0),
                E_v=ser["E_v"][m].mean(0), E_B=ser["E_B"][m].mean(0),
                n_snapshots=int(m.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(HERE / "data" / "dissipation"))
    p.add_argument("--figures", default=str(HERE / "figures"))
    p.add_argument("--sat-start", type=float, default=28.0)
    p.add_argument("--exclude", nargs="*", default=("smoke",))
    args = p.parse_args()

    runs = load_runs(args.data, skip=("calib", *args.exclude))
    measured = [(r, dissipation(r, args.sat_start)) for r in runs]
    measured = [(r, d) for r, d in measured if d is not None]
    if not measured:
        raise SystemExit(f"no run in {args.data} carries transfer spectra "
                         f"(run with --transfer)")

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))
    for run, d in measured:
        st, _ = style_of = None, None
        st = dict(STYLE.get(int(run["N"]), STYLE[128]))
        st["color"] = SERIES[series_of(run)][0]
        st["label"] = f"{SERIES[series_of(run)][1]}, N={int(run['N'])}"
        n, x = d["n"], d["n"] / (int(run["N"]) / 2)
        k = 2.0 * np.pi * n
        good = n >= 2
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = d["D_v"] / (2.0 * k ** 2 * d["E_v"])
            eta = d["D_B"] / (2.0 * k ** 2 * d["E_B"])

        axes[0, 0].loglog(n[good], np.abs(d["D_v"])[good], **st)
        axes[0, 1].loglog(n[good], np.abs(d["D_B"])[good], **st)
        axes[1, 0].loglog(x[good], np.abs(nu)[good], **st)
        axes[1, 1].loglog(x[good], np.abs(eta)[good], **st)

    axes[0, 0].set_title(r"kinetic dissipation spectrum $D_v(n)$", fontsize=10)
    axes[0, 1].set_title(r"magnetic dissipation spectrum $D_B(n)$", fontsize=10)
    axes[1, 0].set_title(r"$\nu_{\rm eff}(n) = D_v / 2k^2E_v$"
                         "\n(a Laplacian viscosity would be flat)", fontsize=10)
    axes[1, 1].set_title(r"$\eta_{\rm eff}(n) = D_B / 2k^2E_B$"
                         "\n(a Laplacian resistivity would be flat)", fontsize=10)
    for ax in axes[0]:
        ax.set_xlabel(r"mode number $n = kL/2\pi$")
    for ax in axes[1]:
        ax.set_xlabel(r"$n / n_{\rm Nyquist}$")
    axes[0, 0].set_ylabel(r"$|D_v(n)|$")
    axes[0, 1].set_ylabel(r"$|D_B(n)|$")
    axes[1, 0].set_ylabel(r"$\nu_{\rm eff}$")
    axes[1, 1].set_ylabel(r"$\eta_{\rm eff}$")
    for ax in axes.ravel():
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Numerical dissipation, measured from the spectral energy "
                 f"budget in the saturated state ($t/t_{{\\rm cross}} \\geq "
                 f"{args.sat_start:g}$).\n"
                 r"$D(n) = T_{\rm ideal}(n) - dE(n)/dt$, both terms measured.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.figures) / "dynamo_dissipation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # The slope of eta_eff over the resolved range is the p-1 exponent.
    print(f"\n{'run':34s} {'nu_eff slope':>13s} {'eta_eff slope':>14s}  "
          f"(over n/n_Nyq = 0.15 - 0.6; Laplacian = 0, p-th order = p-1)")
    for run, d in measured:
        n = d["n"]; k = 2.0 * np.pi * n; x = n / (int(run["N"]) / 2)
        band = (x >= 0.15) & (x <= 0.6)
        out_row = []
        for D, E in ((d["D_v"], d["E_v"]), (d["D_B"], d["E_B"])):
            q = D / (2.0 * k ** 2 * E)
            ok = band & np.isfinite(q) & (q > 0)
            out_row.append(np.polyfit(np.log(n[ok]), np.log(q[ok]), 1)[0]
                           if ok.sum() >= 4 else np.nan)
        print(f"{str(run['label'])[:22]:22s} N={int(run['N']):3d} "
              f"{out_row[0]:13.2f} {out_row[1]:14.2f}   "
              f"({d['n_snapshots']} snapshots)")


if __name__ == "__main__":
    main()
