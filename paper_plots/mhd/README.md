# MHD forward-test figures (methods paper)

Scripts that generate the magnetohydrodynamics forward-test figures.  Each
simulation script caches its output under `data/` and regenerates the figure
from that cache unless `--rerun` is passed.  Run with the repo on `PYTHONPATH`
(GPU picked automatically via `autocvd`):

    PYTHONPATH=$(git rev-parse --show-toplevel) python paper_plots/mhd/<script>.py [--rerun]

| script | figure | description |
| --- | --- | --- |
| `alfven_convergence.py` | `alfven_wave3D_dp_convergence.svg` | 3D circularly-polarized Alfvén-wave L1 convergence (double precision), AthenaPK overlaid. Wraps `pytests/mhd/alfven_wave3D.py` (used as-is). |
| `orszag_tang.py` | `orszag_tang.svg` | Orszag-Tang vortex: high-resolution FD density field at 1024² (left, rasterized) + density cut at y=0.625π comparing FV(HLL) 200 and FD 200 against the FD 1024² reference (right). |
| `mhd_blast.py` | `mhd_blast_oscillations_comparison.png`, `mhd_blast_test1_256cells.png` | 3D MHD blast (Seo & Ryu): resolution×scheme oscillation grid, and FD slices (256³) + diagonal profiles with FV(HLL) overlay. Built from cached `arena/results/` data — no simulation is run. |
| `mhd_jet.py` | `mhd_jet_fd_256.png` | Magnetically driven jet, finite-difference solver only. Density slice through the jet axis at t=5. Use `--res N` for a quicker smoke test. |

`_common.py` holds shared paths, consistent solver labels/colours and the
`mhd_registered_variables(solver_mode)` helper used to index cached states by
name (`rv.density_index`, …) rather than hard-coded integers.

Figure titles are intentionally omitted — that information goes into the paper
captions.
