"""Calibrate AthenaK's ``dedt`` per resolution so every run sits at one Mach number.

``check_athenak_driving.py`` shows that AthenaK's ``dedt`` is not a
resolution-invariant control: the stationary Mach number rises with N under OU
driving (and falls under white driving), and changes with ``cfl`` at fixed N.
The cause is in ``turb_driver``: the normalisation ``s`` solving
``m0 s^2 + m1 s = dedt`` is computed from the *fresh, uncorrelated* increment
``force_tmp``, while the field actually applied is the OU accumulation
``fcorr*force + gcorr*s*force_tmp``. astronomix normalises against the field it
actually applies, so its Mach number is resolution-independent at fixed ``dedt``.

Feeding both codes the same ``dedt`` would therefore compare turbulence at
*different Mach numbers at every resolution*. What has to be matched instead is
the achieved flow: same driving band, same solenoidal projection, same
correlation time, and the same stationary Mach number. This script finds the
``dedt`` that puts each AthenaK resolution at the target Mach.

The naive ``dedt <- dedt * (M_target/M)^3`` update (from ``eps ~ v^3/2L`` with
power linear in ``dedt``) does *not* work here: the measured local exponent of
``M(dedt)`` is close to 1/2, not 1/3, because AthenaK's normalisation puts a
``1/sqrt(dt)`` into the applied amplitude and ``dt`` itself responds to the flow.
Rather than assume an exponent, this fits one from the two most recent samples
and secants on it in log-log space, which converges in two or three iterations.

    python calibrate_athenak.py --n 64 128 256 --target 0.3175
"""

# general
import argparse
import json
import subprocess
import sys
from pathlib import Path

# numerics
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
sys.path.insert(0, str(HERE))
from athenak_turb import ATHENAK_BIN, RUN_ROOT, BOX_SIZE  # noqa: E402

#: One file per resolution rather than one shared file, so the per-N calibration
#: jobs can run concurrently on the queue without racing each other's writes.
#: ``load_calibration`` merges them.
CALIBRATION_GLOB = "athenak_dedt_calibration_n*.json"


def calibration_path(n):
    return DATA_DIR / f"athenak_dedt_calibration_n{n}.json"


def load_calibration(data_dir=DATA_DIR):
    """Merge the per-resolution calibration files into one ``{N: entry}`` dict."""
    merged = {}
    for f in sorted(Path(data_dir).glob(CALIBRATION_GLOB)):
        merged.update(json.loads(f.read_text()))
    return merged


def measure_mach(tag, frac=0.4):
    """Stationary Mach number from a run's history file (last ``frac`` of the run)."""
    hst = sorted((RUN_ROOT / tag).glob("*.hst"))
    if not hst:
        raise SystemExit(f"no .hst in {RUN_ROOT / tag}")
    with open(hst[0]) as fh:
        header = [ln for ln in fh if ln.startswith("#")]
    names = [c.split("=")[-1] for c in header[-1].lstrip("#").split()]
    data = np.loadtxt(hst[0])
    idx = {nm: i for i, nm in enumerate(names)}
    t = data[:, idx["time"]]
    mass = data[:, idx["mass"]]
    ke = sum(data[:, idx[k]] for k in ("1-KE", "2-KE", "3-KE") if k in idx)
    v = np.sqrt(2.0 * ke / np.maximum(mass, 1e-30))
    return float(v[t >= (1.0 - frac) * t.max()].mean())    # cs = 1


def run(n, dedt, nturn, tcorr, tag):
    cmd = [sys.executable, str(HERE / "athenak_turb.py"), "--n", str(n),
           "--dedt", repr(dedt), "--nturn", str(nturn), "--nsnap", "3",
           "--tag", tag]
    if tcorr >= 0:
        cmd += ["--tcorr", str(tcorr)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit(f"AthenaK calibration run {tag} failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--target", type=float, required=True,
                   help="target stationary Mach number (use astronomix's measured value)")
    p.add_argument("--nturn", type=float, default=8.0,
                   help="length of each calibration run, in turnover times")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--dedt0", type=float, default=-1.0,
                   help="initial dedt guess; <0 starts from the astronomix value. "
                        "Pass a value extrapolated from a coarser grid to keep the "
                        "expensive resolutions to two or three iterations.")
    p.add_argument("--tcorr", type=float, default=-1.0)
    p.add_argument("--tol", type=float, default=0.03,
                   help="stop early once |M/M_target - 1| is below this")
    args = p.parse_args()

    if not ATHENAK_BIN.exists():
        raise SystemExit(f"AthenaK binary not found: {ATHENAK_BIN}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        dedt = args.dedt0 if args.dedt0 > 0 else args.target ** 3 / (2.0 * BOX_SIZE)
        print(f"\n=== N = {n} (target Mach {args.target:.4f}) ===")
        samples = []                       # (log dedt, log Mach) history
        best = None                        # (|rel|, dedt, mach) seen so far
        for it in range(args.iters):
            tag = f"cal_n{n}_it{it}"
            run(n, dedt, args.nturn, args.tcorr, tag)
            mach = measure_mach(tag)
            rel = mach / args.target - 1.0
            samples.append((np.log(dedt), np.log(mach)))
            if best is None or abs(rel) < best[0]:
                best = (abs(rel), dedt, mach)
            print(f"  iter {it}: dedt={dedt:.6g}  ->  Mach={mach:.4f}  ({rel * 100:+.1f}%)")
            if abs(rel) < args.tol:
                print("  within tolerance")
                break
            if it == args.iters - 1:
                break
            # Local power-law exponent p in M ~ dedt^p, from the two most recent
            # samples; fall back to the 1/2 that this driver empirically shows
            # until there are two distinct points to fit.
            if len(samples) >= 2 and abs(samples[-1][0] - samples[-2][0]) > 1e-12:
                p = (samples[-1][1] - samples[-2][1]) / (samples[-1][0] - samples[-2][0])
                p = float(np.clip(p, 0.15, 1.5))       # guard against a noisy fit
            else:
                p = 0.5
            log_dedt = samples[-1][0] + (np.log(args.target) - samples[-1][1]) / p
            dedt = float(np.exp(log_dedt))
        _, dedt, mach = best                # keep the closest sample, not the last step
        entry = {str(n): dict(dedt=dedt, mach=mach, target=args.target,
                              tcorr=args.tcorr, nturn=args.nturn)}
        calibration_path(n).write_text(json.dumps(entry, indent=2, sort_keys=True))
        print(f"  wrote {calibration_path(n)}")

    print("\nmerged calibration:")
    for n, c in sorted(load_calibration().items(), key=lambda kv: int(kv[0])):
        print(f"  N={int(n):4d}  dedt={c['dedt']:.6g}  (last measured Mach {c['mach']:.4f})")


if __name__ == "__main__":
    main()
