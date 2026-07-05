# Hydrodynamics forward-test figures (methods paper)

Scripts that generate the pure-hydrodynamics forward-test figures.  Each
simulation script caches its output under `data/` and regenerates the figure
from that cache unless `--rerun` is passed, so figures can be re-styled without
re-running any simulation.  `astronomix` is installed non-editably, so run with
the repo on `PYTHONPATH` (GPU is picked automatically via `autocvd`):

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/hydrodynamics/<script>.py [--rerun]

| script | figure | description |
| --- | --- | --- |
| `sound_wave_convergence.py` | `sound_wave3D_convergence.svg` | 3D linear sound-wave L1 convergence (FV-JAX, FD-JAX, FD-Pallas). Wraps `pytests/hydrodynamics/sound_wave3D.py` (used as-is). |
| `shock_tube.py` | `shock_tube1D_test.svg` | 1D Sod shock tube vs. exact Riemann solution. Wraps `pytests/hydrodynamics/shock_tube1D.py` (used as-is). |
| `double_blast.py` | `double_blast.pdf` | Woodward & Colella double blast, density at t=0.038: FV(HLL)/FV(HLLC)/FD at 400 cells + FV(HLL) at 10000 cells (reference). |
| `sedov_blast.py` | `sedov_blast_256.png` | Sedov-Taylor blast, 4×3 radial profiles (rho, \|v\|, p) vs. exact, one row each for FV(HLL), FV(HLLC), FV(AM-HLLC), FD — all 256³, with a smoothly-tapered, energy-normalised injection region. FD runs through Pallas. Use `--res N` for a quicker smoke test. |

`_common.py` holds shared paths and the consistent per-solver colours.

Figure titles are intentionally omitted — that information goes into the paper
captions.
