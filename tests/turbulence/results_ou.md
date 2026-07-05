# OU-forced isothermal MHD turbulence (HOW-MHD reproduction)

Driven, triply-periodic, isothermal MHD turbulence reproducing the turbulent-flow
test of HOW-MHD (https://arxiv.org/pdf/2304.04360), but driven with **Ornstein–
Uhlenbeck (temporally correlated) forcing** instead of the paper's white-in-time,
constant-energy-injection scheme. All runs use the **Pallas backend** (isothermal-MHD
WENO kernel) and pick a genuinely-free GPU via `autocvd`.

## Setup
- Solver: finite-difference WENO + constrained transport, isothermal EoS, MHD, 3D periodic.
- Backend: `PALLAS`, block `(4,4,8)`, Triton. Env: `jf1uids` (jax 0.6.2); run with
  `PYTHONPATH=<repo root>` so the worktree `astronomix` shadows the older installed copy.
- Forcing: solenoidal OU field, peak wavenumber `k_f = 2π·2` (≈ injection scale L/2),
  correlation time `τ = 0.5`, amplitude `F0 = 3.5` (calibrated so stationary `v_rms ≈ 1`,
  since `v_rms ∝ √F0`). Applied as a clean state-independent acceleration `v += F0 f dt`.
- Grid: `M_s ∈ {0.5, 2.0}` (via `c_s = 1/M_s`) × `M_A ∈ {1.0, 100.0}` (via guide field
  `B_0 = 1/M_A` along z). Targets are nominal; measured stationary values differ.
- `C_cfl = 1.5`, `t_end = 5 t_cross`, 60 snapshots. Resolutions `64³` and `128³`.
- Supersonic (`M_s ≥ 1`) runs enable positivity flooring + the vacuum-protection routine;
  this is what keeps the challenging shock-dominated cases stable.

## Files
- `ou_turbulence.py` — runner, one `(M_s, M_A)` case per invocation, writes `data/ou_*.npz`.
- `make_fig_ou.py` — builds `figures/fig_ou_{spectra,slices,timeseries}[_N128].png/.svg`.
- Run a case: `PYTHONPATH=<repo> python ou_turbulence.py --ms 2.0 --ma 100.0 --N 128
  --tcross 5 --F0 3.5 --tau 0.5 --nsnap 60`.

## Stationary results (mean over last third of the run)

| case (N=128)        | v_rms | M_s  | M_A  | ρ_min | ρ_max | stable |
|---------------------|-------|------|------|-------|-------|--------|
| M_s→0.5, M_A→1.0    | 0.93  | 0.47 | 0.73 | 0.43  | 1.45  | ✓      |
| M_s→0.5, M_A→100    | 1.18  | 0.59 | 4.03 | 0.41  | 1.78  | ✓      |
| M_s→2.0, M_A→1.0    | 0.95  | 1.90 | 0.60 | 0.02  | 11.1  | ✓      |
| M_s→2.0, M_A→100    | 1.07  | 2.13 | 6.92 | 0.03  | 14.4  | ✓      |

(64³ values are very close; see `data/`.) Wall time per run: ~45 s at 64³, ~70–100 s at
128³ on one A100 — Pallas makes the full 8-run grid a few minutes of compute.

## Physics observed
- **All eight runs reach a clean statistically-stationary state** (v_rms and M_s plateau
  by ~1 t_cross and hold flat to 5 t_cross — `fig_ou_timeseries`), including the
  shock-dominated supersonic cases. The earlier stability problems are resolved by the
  positivity + vacuum-protection guards on the supersonic runs.
- Spectra (`fig_ou_spectra`): energy injected near `k≈13`, a `k^-5/3` inertial range, and
  a numerical-dissipation falloff toward the grid scale — sharper / longer inertial range
  at 128³.
- Strong guide field (`M_A→1`, left column) → kinetic–magnetic **equipartition** and flow
  organised along the field. Weak field (`M_A→100`, right) → magnetic energy well below
  kinetic; the field saturates at `M_A ≈ 4–7` (not 100) because the turbulence amplifies it
  via small-scale dynamo — visible as finely-tangled `|B|` in the slices.
- Density contrast grows with `M_s`: subsonic rows stay near ρ≈1; supersonic rows develop
  shocks with ρ spanning ~0.02–14.

## Notes / knobs
- `F0` sets v_rms (`∝√F0`); `k_f` sets injection scale; `τ` the correlation time.
- For higher M_s or resolution, if a run goes non-finite: lower `C_cfl`, raise
  `protection_max_velocity`/`minimum_density`, or raise the forcing-correlation time.
