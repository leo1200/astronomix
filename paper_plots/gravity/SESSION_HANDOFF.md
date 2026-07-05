# Session handoff — self-gravity paper figures (resume on a free-GPU node)

Context: continuing work on the self-gravity methods-paper figures. compgpu10 ran
out of free GPUs mid-run; this file is the single source of truth to resume.

## 0. Environment (do this first, every time)

```bash
cd /export/home/lstorcks/agent-home/astronomix-refactor
export PYTHONPATH=$(pwd)          # astronomix is installed non-editable; this shadows the stale site-packages copy
```

Branch: `refactor` (worktree `astronomix-refactor`). Scripts pick a free GPU via
`autocvd(num_gpus=1)` automatically. Run any script in the background and poll its
log; don't block.

Key precision rule: **`--double` runs use the NATIVE_JAX backend, not Pallas.**
The Pallas/Triton kernel does some math in reduced precision and floors at
~1e-11; native float64 reaches the true ~1e-13 floor. Native fp64 at 128³ is
memory-heavy — make sure the chosen GPU is mostly free (we hit an OOM on a
co-tenant GPU). Finite-volume always runs native (Pallas WENO is FD-only).

## 1. TWO THINGS LEFT TO RUN

### (a) Slab figure — re-add power-law indicators  [plot-only, ~30 s]
Edit already applied (`add_power_law_indicators`, anchor (20, 1e-5), exponents
[-2,-5], kept alongside the direct fit). Just regenerate from the cached npz:

```bash
python paper_plots/gravity/slab_convergence.py        # no --rerun: uses data/slab_convergence.npz
```
Then eyeball `figures/slab_error_convergence.svg`: the two dashed reference
slopes (N^-2, N^-5) should sit in the lower-left, the solid black direct-fit line
(≈ N^-4.87) on the corrected-flux data, integer x-ticks 16…256. If the indicators
look misplaced with the extended N range, lower the anchor (e.g. (18, 1e-6)).

### (b) Evrard energy figure — fp64, 4 schemes  [~10 min, needs a free GPU]
The script now compares FOUR schemes: FD simple / FD flux-based / FD corrected
flux-based / **FV simple source** (newly added). Run:

```bash
python paper_plots/gravity/energy_conservation_comparison.py --rerun --double
```
Produces `data/energy_conservation_comparison_fp64.npz` and
`figures/energy_conservation_comparison_fp64.svg`.

Validation after it finishes (load the npz, keys are `time_<k>`, `total_<k>` …
with `k` in {`fd_0`, `fd_5`, `fd_6`, `fv_0`}):
- every scheme's `time_*[-1]` ≈ 1.20 (t_end) and NO zero slots  ← confirms the snapshot fix
- FD simple `|ΔE|/|E0|` ≈ 1.8e-1, FV simple ≈ 1.2e-1 (non-conservative)
- FD flux-based ≈ FD corrected ≈ 4e-9 (machine-precision conservative, overlap)
Render a PNG to inspect (matplotlib can't be shown as SVG here):
```bash
python - <<'PY'
import matplotlib; matplotlib.use('Agg'); from pathlib import Path
import sys; sys.argv.append('--double')
import energy_conservation_comparison as E
E.FIG_FILE = Path('/tmp/evrard_energy_fp64.png'); E.plot()
PY
```

## 2. STATE: what is DONE (don't redo)

Four core fp32 paper figures in `paper_plots/gravity/figures/` (jeans, slab,
energy, radial). All use the consistent 3-FD-scheme naming from `_common.py`.

- **jeans_waves_error_convergence.svg** — done (fp32, Pallas). Integer ticks +
  direct fit p≈-4.88.
- **slab_error_convergence.svg** — done; just being re-plotted to add the
  reference indicators (step 1a).
- **collapse_radial_profiles_comparison.svg** (fp32) AND
  **collapse_radial_profiles_comparison_fp64.svg** (fp64) — both done. Scatter
  rasterized (`rasterized=True`, dpi=300) so the SVG isn't gigabytes.
- **energy_conservation_comparison_fp64.svg** — being regenerated with 4 schemes
  (step 1b). NOTE: the fp32 version `energy_conservation_comparison.svg` /
  `.npz` (no suffix) is STALE & MISLEADING (pre-snapshot-fix, and conservative
  schemes are fp32-unstable on Evrard) — delete it or ignore it; Evrard energy is
  fp64-only by decision.

Subsonic-turbulence energy experiment in `tests/self_gravity_tests/`:
- `subsonic_turbulence_energy.py` — done. fp32 (`--`) and fp64 (`--double`)
  convergence figures + structure figure, all cached. This is the clean
  energy-conservation-VS-resolution story (all schemes stable 32³–128³).

## 3. CORE CODE CHANGE — snapshot bug fix (uncommitted, in astronomix/)

`astronomix/time_stepping/_time_loop.py` + `time_integration.py`: the final state
at `t_end` was never recorded — the regular grid fills slots `0…(N-1)/N·t_end`
and the separate "final" record targeted index `num_snapshots` (out of bounds →
dropped); a step-alignment race could leave the last slot at its `jnp.zeros`
init → the spurious Eₓ=0 / relErr=1 seen for the corrected scheme.
Fix: `SnapshotSpec` gained `final_index`; the in-body `record_final` was replaced
by a single post-loop record into the reserved last slot (`num_snapshots-1`).
Verified at N=32/96/128: snapshots now end exactly at t_end with no zero slots.
This is a shared-driver change — worth running an existing snapshot test
(`pytests/hydrodynamics/sound_wave3D.py::test_sound_wave_convergence`) before
committing.

## 4. KEY FINDINGS / DECISIONS (so you don't relitigate)

- The corrected-flux scheme did NOT "blow up" benignly — TWO things overlapped:
  (1) the snapshot bug (now fixed), and (2) a REAL instability: the conservative
  FD flux schemes (flux-based, corrected) NaN on the cold Evrard collapse below
  128³ (both fp32 AND fp64), and are only marginally fp32-stable at 128³. They
  ARE clean at 128³ in fp64 (~4e-9). The simple source is robust everywhere.
- Decision (user): Evrard energy figure = **fp64 @ 128³ only**; do NOT attempt
  96³ for Evrard; rely on the subsonic-turbulence experiment for the
  energy-vs-resolution scan.
- Decision (user): add **FV + simple source** to the Evrard energy comparison
  (implemented: `_collapse.collapse_config(..., solver_mode=FINITE_VOLUME)` →
  UNSPLIT/MINMOD/RK2_SSP/HLLC, native backend; `_common.Scheme` gained
  `solver_mode` + `label_override`; `energy_conservation_comparison.py` keys data
  per-scheme via `_scheme_key`).
- Conservative schemes' fp32 floor is round-off (≈1 machine-eps/step, grows with
  N); fp64 drops them ~7 orders. Simple/FV-simple are truncation-limited
  (precision-independent).

## 5. OPTIONAL / NICE-TO-HAVE (ask user before doing)

- Delete stale fp32 Evrard energy artifacts (`energy_conservation_comparison.svg`,
  `.npz`) to avoid confusion.
- Update `paper_plots/gravity/README.md` to document fp64 energy figure + FV
  scheme + the `--double` flag.
- Commit the snapshot-fix once validated against existing tests.
