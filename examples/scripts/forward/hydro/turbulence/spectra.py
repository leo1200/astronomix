"""Collect both codes' runs into one time-averaged spectrum table.

``driven_turbulence.py`` already reduces its astronomix snapshots to per-snapshot
spectra (via ``_spectral.reduce_snapshots``) because the raw cubes are far too
large to keep. This script does the same reduction for AthenaK's ``.bin`` dumps,
then time-averages every run over the stationary window and writes a single
small ``spectra.npz`` for ``make_figures.py``.

Both codes go through the identical estimator in ``_spectral.py``; the only
code-specific part here is reading the dump format and stitching AthenaK's
meshblocks back into a global cube.

    python examples/scripts/forward/hydro/turbulence/spectra.py --all
"""

# general
import argparse
import glob
import sys
from pathlib import Path

# numerics
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SCRATCH_DIR = Path("/export/data/lstorcks/turb_spectra")

sys.path.insert(0, str(HERE))
sys.path.insert(0, "/export/home/lstorcks/athena/athenak/vis/python")
from _spectral import reduce_snapshots


# -------------------------------------------------------------
# ============ ↓ AthenaK dump reduction ↓ =====================
# -------------------------------------------------------------
def _stitch(dump, var, n):
    """Assemble one global ``(N, N, N)`` cube from AthenaK's meshblock dump."""
    a = np.asarray(dump["mb_data"][var])
    out = np.zeros((n, n, n), dtype=np.float64)
    nz, ny, nx = a.shape[1], a.shape[2], a.shape[3]
    for m, lb in enumerate(dump["mb_logical"]):
        i, j, k = lb[0] * nx, lb[1] * ny, lb[2] * nz
        out[k:k + nz, j:j + ny, i:i + nx] = a[m]
    # AthenaK indexes (k, j, i) = (z, y, x); transpose so both codes hand the
    # estimator the same (x, y, z) axis order.
    return np.ascontiguousarray(out.transpose(2, 1, 0))


def reduce_athenak(rundir, weighted=False, deconvolve_cell_average=False):
    """Per-snapshot spectra for one AthenaK run directory."""
    import bin_convert

    rundir = Path(rundir)
    files = sorted(glob.glob(str(rundir / "bin" / "*.bin")))
    if not files:
        raise SystemExit(f"no .bin dumps under {rundir}/bin")
    m = np.load(rundir / "meta.npz", allow_pickle=True)
    n = int(m["n"])

    def snapshots():
        for f in files:
            d = bin_convert.read_binary(f)
            yield (float(d["time"]), _stitch(d, "dens", n), _stitch(d, "velx", n),
                   _stitch(d, "vely", n), _stitch(d, "velz", n))

    out = reduce_snapshots(snapshots(), weighted=weighted,
                           deconvolve_cell_average=deconvolve_cell_average)
    out.update(n=n, box_size=float(m["box_size"]), cs=float(m["cs"]),
               dedt=float(m["dedt"]), tcorr=float(m["tcorr"]),
               t_turnover=float(m["t_turnover"]), runtime=float(m["runtime"]),
               num_iterations=-1, weighted=weighted, label=str(m["label"]))
    return out
# -------------------------------------------------------------
# ============ ↑ AthenaK dump reduction ↑ =====================
# -------------------------------------------------------------


def average_window(run, t_start):
    """Time-average a run's per-snapshot spectra over ``time >= t_start``."""
    times = np.asarray(run["times"])
    sel = times >= t_start
    if not sel.any():
        raise SystemExit(f"no snapshots at t >= {t_start} (times: {times})")
    E = np.asarray(run["E_snap"])[sel]
    v_rms = np.asarray(run["v_rms"])[sel]
    return dict(
        n_shell=np.asarray(run["n_shell"]),
        E_mean=E.mean(axis=0),
        # Standard error over snapshots: the realisation-noise floor a
        # code-to-code difference has to beat to mean anything. The two codes
        # draw independent forcing realisations, so this is the relevant bar.
        E_err=(E.std(axis=0, ddof=1) / np.sqrt(len(E)) if len(E) > 1
               else np.zeros_like(E[0])),
        n_avg=len(E), t_used=times[sel], v_rms=v_rms,
        mach=float(v_rms.mean() / run["cs"]),
        n=int(run["n"]), box_size=float(run["box_size"]), cs=float(run["cs"]),
        dedt=float(run["dedt"]), t_turnover=float(run["t_turnover"]),
        runtime=float(run["runtime"]), num_iterations=int(run["num_iterations"]),
        label=str(run["label"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--astronomix", nargs="*", default=[],
                   help="astronomix npz files (default: all in data/)")
    p.add_argument("--athenak", nargs="*", default=[],
                   help="AthenaK run directories (default: all under the scratch root)")
    p.add_argument("--all", action="store_true", help="pick up every run found")
    p.add_argument("--tstart", type=float, default=5.0,
                   help="start of the averaging window, in turnover times")
    p.add_argument("--weighted", action="store_true",
                   help="use the sqrt(rho)-weighted spectrum for the AthenaK reduction "
                        "(must match how the astronomix runs were reduced)")
    p.add_argument("--deconvolve-fv", action="store_true",
                   help="divide AthenaK's spectra by the cell-averaging transfer "
                        "function, so finite-volume cell averages are compared "
                        "against astronomix's finite-difference point values on "
                        "equal footing (a ~1-3%% effect over the relevant shells)")
    p.add_argument("--out", type=str, default="spectra.npz")
    args = p.parse_args()

    astronomix_files = list(args.astronomix)
    athenak_dirs = list(args.athenak)
    if args.all or (not astronomix_files and not athenak_dirs):
        astronomix_files = sorted(glob.glob(str(DATA_DIR / "astronomix_*.npz")))
        # Only production runs: the scratch root also holds the calibration
        # iterations (cal_*) and the driving diagnostics (drvchk_*), which are
        # short, deliberately off-target, and must not enter the comparison.
        athenak_dirs = sorted(
            str(d) for d in SCRATCH_DIR.glob("n[0-9]*")
            if (d / "meta.npz").exists()
        )

    runs = {}
    for f in astronomix_files:
        d = np.load(f, allow_pickle=True)
        if bool(d["weighted"]) != args.weighted:
            raise SystemExit(
                f"{f} was reduced with weighted={bool(d['weighted'])} but --weighted="
                f"{args.weighted}; the two codes must use the same weighting")
        key = f"astronomix_{Path(f).stem.replace('astronomix_', '')}"
        runs[key] = average_window(d, args.tstart * float(d["t_turnover"]))
    for rd in athenak_dirs:
        run = reduce_athenak(rd, weighted=args.weighted,
                             deconvolve_cell_average=args.deconvolve_fv)
        key = f"athenak_{Path(rd).name}"
        runs[key] = average_window(run, args.tstart * run["t_turnover"])

    if not runs:
        raise SystemExit("no runs found")
    for key in sorted(runs):
        r = runs[key]
        print(f"[spectra] {key:26s} N={r['n']:4d}  Mach={r['mach']:.4f}  "
              f"{r['n_avg']:3d} snapshots  runtime={r['runtime']:.0f}s")

    flat = {f"{key}|{field}": value
            for key, r in runs.items() for field, value in r.items()}
    flat["run_keys"] = np.array(sorted(runs))
    flat["weighted"] = args.weighted
    flat["tstart_turnovers"] = args.tstart
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / args.out
    np.savez_compressed(out, **flat)
    print(f"[spectra] wrote {out}")


if __name__ == "__main__":
    main()
