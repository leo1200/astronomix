"""Is AthenaK's ``dedt`` driving independent of the time step (hence of N)?

``turb_driver`` computes its normalisation ``s`` from the *fresh, uncorrelated*
random increment ``force_tmp`` (solving ``m0 s^2 + m1 s = dedt`` with
``m0 ~ dt``), but the field it actually applies is the OU-accumulated ``force =
fcorr*force + gcorr*s*force_tmp``. Because ``force_tmp`` is uncorrelated with
the flow, the cross term ``m1`` averages to zero and the normalisation collapses
to the white-noise branch ``s = sqrt(dedt/m0) ~ 1/sqrt(dt)``. The applied
acceleration then grows as the time step shrinks, and the true energy input rate
scales like ``dedt * tcorr / dt`` rather than ``dedt``.

If that reading is right, the stationary Mach number depends on ``cfl`` at fixed
``N`` and on ``N`` at fixed ``cfl`` — which would make a resolution-convergence
study meaningless. This script measures both.

    python check_athenak_driving.py
"""

# general
import argparse
import subprocess
import sys
from pathlib import Path

# numerics
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from athenak_turb import ATHENAK_BIN, RUN_ROOT  # noqa: E402


def run_case(n, cfl, nturn, tag, tcorr):
    """Run one AthenaK case and return the history-file v_rms time series."""
    cmd = [sys.executable, str(HERE / "athenak_turb.py"), "--n", str(n),
           "--cfl", str(cfl), "--nturn", str(nturn), "--nsnap", "3",
           "--tcorr", str(tcorr), "--tag", tag]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit(f"case {tag} failed")

    # The .hst file carries the mass-weighted kinetic energy every few steps.
    hst = sorted((RUN_ROOT / tag).glob("*.hst"))
    if not hst:
        raise SystemExit(f"no .hst in {RUN_ROOT / tag}")
    with open(hst[0]) as fh:
        header = [ln for ln in fh if ln.startswith("#")]
    cols = header[-1].lstrip("#").split()
    # AthenaK labels columns like "[1]=time"; strip the bracketed index.
    names = [c.split("=")[-1] for c in cols]
    data = np.loadtxt(hst[0])
    idx = {nm: i for i, nm in enumerate(names)}
    t = data[:, idx["time"]]
    mass = data[:, idx["mass"]]
    ke = sum(data[:, idx[k]] for k in ("1-KE", "2-KE", "3-KE") if k in idx)
    v_rms = np.sqrt(2.0 * ke / np.maximum(mass, 1e-30))
    return t, v_rms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nturn", type=float, default=4.0)
    p.add_argument("--tcorr", type=float, default=-1.0,
                   help="-1 = the OU default L/(2 pi v_rms); 0 = white-in-time")
    args = p.parse_args()

    if not ATHENAK_BIN.exists():
        raise SystemExit(f"AthenaK binary not found: {ATHENAK_BIN}")

    mode = "white" if args.tcorr == 0.0 else "ou"
    cases = [(32, 0.30, f"drvchk_{mode}_n32_cfl30"),
             (32, 0.15, f"drvchk_{mode}_n32_cfl15"),
             (64, 0.30, f"drvchk_{mode}_n64_cfl30"),
             (128, 0.30, f"drvchk_{mode}_n128_cfl30")]

    print(f"driving mode: {mode} (tcorr={args.tcorr})")
    print(f"{'case':>26} {'N':>5} {'cfl':>6} {'final v_rms':>12} {'late-mean':>11}")
    results = {}
    for n, cfl, tag in cases:
        t, v = run_case(n, cfl, args.nturn, tag, args.tcorr)
        late = v[t >= 0.5 * t.max()].mean()
        results[tag] = late
        print(f"{tag:>26} {n:>5} {cfl:>6.2f} {v[-1]:>12.4f} {late:>11.4f}")

    print("\nIf dedt is a true injection rate, all four numbers agree.")
    a, b = results[f"drvchk_{mode}_n32_cfl30"], results[f"drvchk_{mode}_n32_cfl15"]
    c, d = results[f"drvchk_{mode}_n64_cfl30"], results[f"drvchk_{mode}_n128_cfl30"]
    print(f"  halving cfl at N=32:          {a:.4f} -> {b:.4f}   ratio {b / a:.3f}")
    print(f"  N=32 -> 64 -> 128 at cfl=0.3: {a:.4f} -> {c:.4f} -> {d:.4f}   "
          f"spread {(max(a, c, d) / min(a, c, d) - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
