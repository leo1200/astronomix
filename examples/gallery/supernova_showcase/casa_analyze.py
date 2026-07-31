"""
Measure a saved 3D remnant against the Cas A observations.

Runs on the CPU from a ``--save-state`` npz, so a finished (or a mid-flight)
simulation can be scored without touching a GPU. Produces the diagnostics
Orlando et al. use to accept or reject a model:

  * **angle-averaged r_FS and r_RS**, using the same two criteria as the 1D
    calibration -- a density contrast against the known analytic wind, and the
    departure from homologous expansion -- so 1D and 3D are directly comparable.
    (The showcase's old estimator read the reverse shock off a median-temperature
    profile and broke as soon as the interior cooled: it has reported anything
    from 0.01 to 1.09 pc for states whose real r_RS is ~1.5 pc.)
  * **r_FS versus position angle in the plane of the sky**, which Orlando et al.
    (2022) identify as the single most discriminating diagnostic for the
    circumstellar-shell parameters. The Earth vantage point is on the -y axis,
    so the plane of the sky is (x, z).
  * the **radial density profile** against the calibrated 1D solution, which
    separates "the 3D run disagrees with the observations" from "the 3D run
    disagrees with its own 1D calibration" -- two very different failures.

Usage::

    CUDA_VISIBLE_DEVICES= ./run.sh casa_analyze.py \\
        /export/data/lstorcks/supernova_showcase/orl_n256_shell.npz \\
        --profile casa_1d_fiducial_350yr.npz --label shell
"""

# ==== CPU only ====
# The showcase scripts call autocvd when CUDA_VISIBLE_DEVICES is unset, which
# blocks waiting for a free GPU. Setting it empty (rather than leaving it None)
# is the documented way to import them for CPU-side analysis.
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
# ==================

# general
import argparse
from pathlib import Path

# numerics
import numpy as np

# plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# units and constants
from astropy import units as u
import astropy.constants as const

# shared showcase helpers
from _common import FIGURES_DIR, MASS_PER_NUCLEUS, snr_code_units, temperature_K
from casa_orlando import measure_shocks_3d, shock_speed_vs_position_angle


# observational targets (see casa_calibrate_1d.py for the references)
OBS = dict(r_fs=(2.52, 0.20), r_rs=(1.58, 0.16), v_fs=(5250.0, 250.0),
           n_post=(4.0, 1.0))
DISTANCE_KPC = 3.4


def load(path):
    d = np.load(path)
    return dict(rho=np.asarray(d["rho"], dtype=np.float64),
                press=np.asarray(d["press"], dtype=np.float64),
                vx=np.asarray(d["vx"]) if "vx" in d else None,
                vy=np.asarray(d["vy"]) if "vy" in d else None,
                vz=np.asarray(d["vz"]) if "vz" in d else None,
                box=float(d["box"]), age=float(d["age"]),
                n=int(d["num_cells"]))


def coords(box, n):
    c = (np.arange(n) + 0.5) / n * box - box / 2.0
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return np.sqrt(X ** 2 + Y ** 2 + Z ** 2), X, Y, Z


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", help="one or more --save-state npz files")
    ap.add_argument("--labels", nargs="*", default=None, help="legend labels")
    ap.add_argument("--profile", default=None,
                    help="the calibrated 1D profile at the same age, for comparison")
    ap.add_argument("--n-w", type=float, default=0.928, help="wind density at r_fs_ref")
    ap.add_argument("--r-fs-ref", type=float, default=2.5)
    ap.add_argument("--n-c", type=float, default=0.1)
    ap.add_argument("--out", default="casa_analysis", help="figure name stem")
    args = ap.parse_args()

    cu = snr_code_units()
    rho_per_n = float((MASS_PER_NUCLEUS * const.m_p / u.cm ** 3).to(cu.code_density).value)
    labels = args.labels or [Path(s).stem for s in args.states]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    ax_prof, ax_pa, ax_sc = axes
    summary = []

    for path, label in zip(args.states, labels):
        st = load(path)
        r, X, Y, Z = coords(st["box"], st["n"])
        r_safe = np.maximum(r, 0.5 * st["box"] / st["n"])
        if st["vx"] is None:
            raise SystemExit(f"{path} has no velocities -- the reverse-shock "
                             f"criterion needs them; re-save with a newer --save-state")
        v_r = (st["vx"] * X + st["vy"] * Y + st["vz"] * Z) / r_safe

        d_all = np.load(path)
        f_sh = np.asarray(d_all["shocked_fraction"]) if "shocked_fraction" in d_all else None
        f_ej = np.asarray(d_all["C_ej"]) if "C_ej" in d_all else None
        m = measure_shocks_3d(st["rho"], v_r, r, age_yr=st["age"], code_units=cu,
                              n_w=args.n_w, r_fs_ref=args.r_fs_ref, n_c=args.n_c,
                              rho_per_n=rho_per_n, shocked_fraction=f_sh,
                              ejecta_fraction=f_ej)
        angles, r_pa = shock_speed_vs_position_angle(
            st["rho"], r, X, Y, Z, n_w=args.n_w, r_fs_ref=args.r_fs_ref,
            n_c=args.n_c, rho_per_n=rho_per_n)

        T = temperature_K(st["rho"], st["press"], cu)
        summary.append(dict(label=label, age=st["age"], r_fs=m["r_fs"], r_rs=m["r_rs"],
                            pa_spread=float(np.nanmax(r_pa) - np.nanmin(r_pa)),
                            T_max=float(np.nanmax(T)), rho_max=float(st["rho"].max())))

        ax_prof.semilogy(m["rc"], m["rho_mean"] / rho_per_n, lw=1.4, label=label)
        ax_pa.plot(angles, r_pa, lw=1.4, label=label)

    if args.profile is not None:
        d = np.load(args.profile)
        ax_prof.semilogy(np.asarray(d["r"]), np.asarray(d["rho"]) / rho_per_n,
                         "k--", lw=1.0, label="1D calibration")

    # ambient wind for reference
    rr = np.linspace(0.02, args.r_fs_ref * 1.4, 400)
    ax_prof.semilogy(rr, args.n_w * (args.r_fs_ref / rr) ** 2 + args.n_c,
                     color="0.6", ls=":", lw=1.0, label="unshocked wind")
    ax_prof.set(xlabel="r [pc]", ylabel=r"$\langle n\rangle$ [cm$^{-3}$]",
                xlim=(0, 3.4), ylim=(1e-2, 1e3), title="angle-averaged density")
    ax_prof.legend(fontsize=8)

    for key, col in (("r_fs", "tab:red"), ("r_rs", "tab:orange")):
        val, tol = OBS[key]
        ax_pa.axhspan(val - tol, val + tol, color=col, alpha=0.15)
        ax_pa.axhline(val, color=col, ls=":", lw=1.0)
    ax_pa.set(xlabel="position angle in the plane of the sky [deg]",
              ylabel="$r_{FS}$ [pc]", title="forward shock vs position angle\n"
                                            "(red band = observed $r_{FS}$)")
    ax_pa.legend(fontsize=8)

    # scorecard
    ax_sc.axis("off")
    lines = [f"{'model':<22}{'age':>6}{'r_FS':>8}{'r_RS':>8}{'FS/RS':>7}{'PA spread':>11}"]
    lines.append("-" * 62)
    for s in summary:
        ratio = s["r_fs"] / s["r_rs"] if s["r_rs"] and np.isfinite(s["r_rs"]) else np.nan
        lines.append(f"{s['label'][:21]:<22}{s['age']:>6.0f}{s['r_fs']:>8.3f}"
                     f"{s['r_rs']:>8.3f}{ratio:>7.3f}{s['pa_spread']:>11.3f}")
    lines.append("-" * 62)
    lines.append(f"{'OBSERVED':<22}{350:>6}{OBS['r_fs'][0]:>8.2f}{OBS['r_rs'][0]:>8.2f}"
                 f"{OBS['r_fs'][0] / OBS['r_rs'][0]:>7.3f}{'0.2-0.4':>11}")
    lines.append("")
    lines.append("PA spread = max - min r_FS over position angle;")
    lines.append("Cas A's forward shock is nearly circular (the FS/RS")
    lines.append("centre offset is 0.2 pc), so a LARGE spread means the")
    lines.append("shell is deforming the blast wave too much.")
    ax_sc.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=8.5,
               va="top", ha="left", transform=ax_sc.transAxes)

    out = Path(FIGURES_DIR) / f"{args.out}.png"
    fig.savefig(out, dpi=150)
    print("\n".join(lines))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
