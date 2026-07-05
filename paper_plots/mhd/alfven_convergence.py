"""3D circularly-polarized Alfvén-wave convergence figure (used as-is, double
precision).

This figure (``alfven_wave3D_dp_convergence.svg``) is produced by the existing
methods-paper benchmark in ``pytests/mhd/alfven_wave3D.py`` — an L1 convergence
sweep over N = 8..128 (double precision) for FV (JAX), FD (JAX) and FD
(Pallas), with the AthenaPK reference overlaid.  A copy of the validated figure
already lives in ``figures/``; running this script re-runs the (double
precision) benchmark and refreshes the copy.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/alfven_convergence.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "pytests" / "mhd" / "alfven_wave3D.py"
GENERATED = SRC.parent / "figures" / "alfven_wave3D_dp_convergence.svg"
DEST = HERE / "figures" / "alfven_wave3D_dp_convergence.svg"


def main():
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {SRC} --convergence --dp (re-runs the 3D convergence sweep) ...")
    subprocess.run(
        [sys.executable, str(SRC), "--convergence", "--dp"],
        cwd=SRC.parent, env=env, check=True,
    )
    shutil.copy(GENERATED, DEST)
    print(f"copied {GENERATED} -> {DEST}")


if __name__ == "__main__":
    main()
