# Field-level inference (differentiable MHD)

Reconstruct the initial **velocity** field of a driven $128^3$ MHD turbulence box
so that its line-of-sight density projection forms the `astronomix` logo after
~half a turbulent crossing time. Gradients come from reverse-mode AD through the
differentiable WENO-MHD solver (hand-derived Pallas adjoint).

## Layout
- `data/`     — all optimisation outputs and saved states (`*.npy`, `*.npz`)
- `figures/`  — all rendered figures (`*.png`, `*.gif`)
- `_archive/` — exploratory / superseded scripts (kept for reference, not needed)
- `logo.png`  — the target image

## Scripts (run from this directory, `PYTHONPATH=<repo root>`, env `astx`)
Helper `_spectral_ops.py` provides the spectral restrict/prolong (FFT pad/truncate).

### 1. `run_inference.py` — the optimisation
Optimises the initial velocity by Adam through the differentiable MHD solver.
Three methods (`--only`):
- `naive` — all $3\cdot128^3$ modes at the target resolution (the costly baseline);
- `k-windowing` — k-expansion schedule (`8→16→32→64`) at the target resolution;
- `multigrid` — same k-schedule on a **resolution ladder** ($32^3\!\to\!64^3\!\to\!128^3$),
  each band on the coarsest grid resolving it with a half-Nyquist gap ($N\ge4k_{\rm cut}$).

Saves the convergence trace to `data/*.npz` and the best reconstructed full state
to `data/best_state_*128.npy`. `--table-only` prints the per-stage parameter table.
```bash
# costly baseline, save the reconstructed state for the figure:
python run_inference.py --only naive --bands "64:60" --budget 0 \
    --state-out data/best_state_full128.npy --out data/bench_full128_naive.npz
# the three-method money-plot benchmark (equal wall-time budget):
python run_inference.py --only all --budget 10800 --out data/bench_money128.npz
```

### 2. `make_panel_snaps.py` + `make_panels_fig.py` — the 2×2 figure
`make_panel_snaps.py` (env `astx`) evolves a saved state and writes snapshot cubes;
`make_panels_fig.py` (env `jf1uids`, needs pyvista) renders the volume+screen panels.
```bash
python make_panel_snaps.py --state data/best_state_full128.npy --resolution 128 \
    --t-end 1.1819 --out data/panel_snaps_full128.npz
<jf1uids python> make_panels_fig.py --snaps data/panel_snaps_full128.npz \
    --out figures/panels_full128.png
```

### 3. `plot_convergence.py` — the money plot
Reads the three `data/bench_money128_*.npz` traces and writes
`figures/money128.png` (terminal loss vs optimisation wall-time).

## Key setup numbers ($128^3$)
code time $t_0=29.3$ kyr; $T=1.5\times10^4$ K; $c_s=14.4$, $v_A=20.8$,
$v_{\rm rms}=42.3$ km/s (sonic Mach $2.9$, Alfvén Mach $2.0$); driving
$3.5$ crossing times; $t_{\rm end}=1.18\,t_0=34.7$ kyr (half crossing).
