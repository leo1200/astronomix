"""3D linear sound-wave convergence figure (used as-is in the paper).

This figure (``sound_wave3D_convergence.svg``) is produced by the existing
methods-paper benchmark in ``pytests/hydrodynamics/sound_wave3D.py`` — an L1
convergence sweep over N = 8..128 for FV (JAX), FD (JAX) and FD (Pallas).  A
copy of the validated figure already lives in ``figures/``; running this
script re-runs that benchmark and refreshes the copy.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/sound_wave_convergence.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "pytests" / "hydrodynamics" / "sound_wave3D.py"
GENERATED = REPO / "pytests" / "hydrodynamics" / "figures" / "sound_wave3D_convergence.svg"
DEST = HERE / "figures" / "sound_wave3D_convergence.svg"


def main():
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {SRC} --convergence (this re-runs the 3D convergence sweep) ...")
    subprocess.run(
        [sys.executable, str(SRC), "--convergence"], cwd=SRC.parent, env=env, check=True
    )
    shutil.copy(GENERATED, DEST)
    print(f"copied {GENERATED} -> {DEST}")


if __name__ == "__main__":
    main()
