"""1D Sod shock-tube figure (used as-is in the paper).

This figure (``shock_tube1D_test.svg``) is produced by the existing test in
``pytests/hydrodynamics/shock_tube1D.py`` (density / velocity / pressure of the
Sod problem vs. the exact Riemann solution for the various solver
configurations).  A copy of the validated figure already lives in ``figures/``;
running this script re-runs that test and refreshes the copy.

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/shock_tube.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SRC = REPO / "pytests" / "hydrodynamics" / "shock_tube1D.py"
GENERATED = SRC.parent / "figures" / "shock_tube1D_test.svg"
DEST = HERE / "figures" / "shock_tube1D_test.svg"


def main():
    # shock_tube1D.py is a pytest module (no __main__ guard); call its test
    # function directly. It writes figures/shock_tube1D_test.svg relative to cwd.
    env = dict(os.environ, PYTHONPATH=str(REPO))
    print(f"running {SRC}::test_shock_tube1D ...")
    subprocess.run(
        [sys.executable, "-c", "import shock_tube1D as m; m.test_shock_tube1D()"],
        cwd=SRC.parent, env=env, check=True,
    )
    shutil.copy(GENERATED, DEST)
    print(f"copied {GENERATED} -> {DEST}")


if __name__ == "__main__":
    main()
