"""Where this study's shared pieces live.

The setup helpers (`GAMMA`, `make_fd_config`, `ism_ti_cooling_setup`,
`fd_positivity`) are the same ones the Cas A showcase uses, and they stay there:
they are one solver configuration serving two studies, and forking them would
mean two configurations drifting apart. Importing this module puts that
directory on `sys.path`, so `from _common import ...` resolves from here.
"""

import sys
from pathlib import Path

#: the Cas A showcase, next door — home of `_common.py` and `run.sh`.
SHOWCASE_DIR = Path(__file__).resolve().parent.parent / "supernova_showcase"

if str(SHOWCASE_DIR) not in sys.path:
    sys.path.insert(0, str(SHOWCASE_DIR))

#: figures from this study go here, not into the showcase's directory.
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
