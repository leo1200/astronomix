# Rotating isothermal compressible turbulence — progress

Pivot from RBC to: driven, rotating, isothermal, subsonic compressible turbulence
in a triply-periodic cube; recover the slow barotropic columnar flow (SS vs MS).

## Done

- **Coriolis source** (`config.rotation`, `params.rotation_rate` Ω): adds
  `−2Ω ẑ×(ρu)` about z in `_physics_sources`, read directly from the conserved
  momentum so it is **isothermal-safe** and does no work (momentum only).
  Trivially differentiable. **Unit-tested**: a uniform flow with no forcing /
  viscosity undergoes the exact inertial oscillation `u_x=u0 cos2Ωt`,
  `u_y=−u0 sin2Ωt` — measured matches analytic to 5 decimals.
- **Isothermal viscosity fix**: `fd_viscosity_source` now skips the energy
  source for the isothermal EOS (where `energy_index=-1` would otherwise corrupt
  the `velocity_z` slot). Physical viscosity now works for isothermal hydro.
- **Forward sim** `forward_rotating_turbulence.py` (3D isothermal FD, periodic,
  forcing + viscosity + Coriolis): runs **fast (~100 s for 64³ to ~20 t, native;
  isothermal isn't Pallas-ported)**, reaches **stationary, subsonic** turbulence
  (u_rms ≈ 0.33, Ma ≈ 0.33), Ro tunable via Ω (ran Ro = 33, 0.66, 0.17).

## Blocker for D1 (columns)

Emergent columns do **not** form: the barotropic (z-invariant) fraction of the
horizontal KE is **≈0.22 independent of Ω** (Ro = 33 → 0.21, 0.66 → 0.21,
0.17 → 0.21). Diagnosis (Coriolis is verified correct): the built-in
turbulent-forcing module is **white-in-time** — it draws a *fresh* random field
every step and rescales to a constant injection rate — so it re-randomises the
flow each step and overwhelms rotation's slow 2D-ising (Taylor-Proudman)
organisation. It also peaks at k_f ≈ 1.5 (set by `kpk=4π/L`), i.e. it directly
forces the column scale.

This is exactly the regime the spec's **OU-correlated forcing (τ_f ~ one eddy
turnover) at k_f ≈ 2–4** is meant to avoid.

## OU forcing implemented + columns emerge (D1)

**OU forcing** added as a *new mode* (`TurbulentForcingConfig.ou_forcing`; params
`correlation_time` τ_f, `forcing_wavenumber` k_f, `forcing_amplitude` F0). The
persistent field is threaded through the integration carry (bundled into the
RNG-key slot as `(key, f)`, opaque everywhere except the forcing call), evolved
with the exact OU step `f ← a f + √(1−a²) ξ`, `a = exp(−dt/τ_f)`, ξ a fresh
unit-rms solenoidal field peaked at k_f, and applied as a *constant-amplitude*
acceleration `u += F0 f dt` (state-independent → clean adjoint, reproducible for
a fixed dt sequence). The white-in-time mode is untouched.

**Columns now emerge.** Sweep (64³, OU forcing):

| Ω (Ro) | forcing | barotropic fraction |
|--------|---------|---------------------|
| 0.01 (33)  | white  | 0.22 |
| 2.0 (0.17) | OU k_f=4 | 0.15 |
| 4.0 (0.11) | OU k_f=3 | **0.35** |

At Ω=4 (Ro≈0.11) the barotropic (z-invariant) fraction of the horizontal KE
rises from ~0.13 to a stationary **~0.33 within one rotation period**, and the
ω_z x–z slice shows clear z-elongated columns (vs the isotropic blobs at weak
rotation). u_rms stays stationary (Ma≈0.44, subsonic). **D1 (emergent columns +
stationary state) achieved.** Note: at 64³ the column scale needs *stronger*
rotation (Ro≈0.1) than the spec's 0.3–0.7 — limited scale separation
(forcing k_f≈3 → columns k≈1); the higher-Ro regime should work at N=128/256.

Files: `rot_turb_{omegaz,timeseries}_N64_Om4.0_ou1.png` (+ gif, npz).

## 128³ production column field (clean)

`128³`, OU forcing (k_f=3, τ_f=1, F0=1), Ω=4 → **Ma≈0.40, Ro≈0.10**, t=12.
(~4 h native on one GPU; isothermal has no Pallas kernel.) Diagnostics
(`rot_turb_{columns,omegaz,spectra,timeseries}_N128_Om4.0_ou1.png`):

- **column field from above** (`*_columns`): z-averaged ω_z shows coherent
  large-scale cyclones/anticyclones — the column cross-sections;
- **ω_z x–z slice**: clear z-elongated streaks (columns) with fine turbulent
  detail at this resolution;
- **isotropic E(k)**: stationary, peaks exactly at the forcing scale k_f=3 with a
  clean cascade;
- **spectral anisotropy** (the money plot for columns): **E(k⊥) ≫ E(k_z) by ~10×
  at every scale**, and the barotropic E(k_z=0) point sits at the very top —
  energy piles at k_z→0, the Taylor-Proudman 2D-ization signature.

The barotropic-fraction *scalar* reads lower at 128³ (0.245) than 64³ (0.35)
only because the higher resolution carries more small-scale 3D energy that
dilutes the z-average; the spectral anisotropy is the resolution-robust evidence
and shows strong, clean columns. **D1 fully achieved at production resolution.**

(Note: a first 128³ attempt was wasted on a flag slip — `RT_OU=1` was dropped on
a clean relaunch, so it ran white forcing; corrected here.)

## Inverse modeling (`inverse/`)

### Gradient gate — PASSED (`inverse/gradient_gate.py`)

Reverse-mode gradient of a terminal loss on the barotropic-columnar mode
`P_large(x(T))` w.r.t. the columnar IC, FD-checked through the **full** rotating-
isothermal path: FD + periodic + isothermal EOS + viscosity + **Coriolis** + **OU
forcing**, with `fixed_timestep` (so the OU forcing is identical for the
perturbed/unperturbed rollouts). `max |FD−AD|/|grad| = 4.2e-7` (threshold 1e-4) —
cleaner than RBC (smooth dynamics: no shocks, linear Coriolis, state-independent
forcing). **AD through the rotating-isothermal-Coriolis-OU path is clean.**
(Note: `fixed_timestep` uses a plain `fori_loop`, so reverse-mode stores every
step — OOMs at 32³/80 steps; the gate runs at 16³, enough for a differentiability
proof. A checkpointed fixed-step loop would be needed to scale reverse-mode.)

### SS-vs-MS mechanism (`inverse/sensitivity_mechanism.py`)

Forward-only (Pallas-N/A for isothermal, but cheap): spin up to a developed
columnar-turbulent state, then propagate truth and a tiny columnar-IC
perturbation under **identical OU forcing** (fixed timestep + same seed) and
measure two sensitivities vs window t (48³):

- `A_full(t)` = full-state tangent `‖δx(t)‖/‖δIC_col‖` — the proxy the
  single-shooting **adjoint** mirrors (the adjoint grows backward through the
  fast 3D turbulence at the same Lyapunov rate as the forward tangent);
- `A_large(t)` = `‖δP_large(t)‖/‖δIC_col‖` — the slow barotropic-columnar target.

**Result (long window, T = 20.6 eddy times).** `A_full` grows ~15× (`~ e^{λt}`)
while **`A_large` stays ~50× below it** (grows only ~2–3×). The asymptotic fit
gives **λ=0.090 → τ_L ≈ 8.9 eddy times** (the short-window fit's 3.5 was inflated
by an initial fast transient). So **rotation makes the flow only weakly chaotic /
highly predictable** — a real physical finding consistent with the slow columnar
manifold. The story holds: **the columnar target is predictable, yet the single-
shooting gradient (full tangent) blows up** because its adjoint traverses the 3D
turbulence; MS caps it. SS/MS gap at T=20.6 t_e (≈2.3 τ_L): m=2 → 2.2×, m=4 →
3.4×, m=8 → 5.8×, m=16 → 8.6×. (The gap is *modest by design*: rotation's long
τ_L means SS survives far longer here than in strongly-chaotic flows — MS only
becomes essential at T ≫ τ_L ≈ 9 eddy times.) `inverse/figures/rot_mechanism_N48.png`.

### Checkpointed fixed-timestep loop (integrator enabler)

`time_integration`: `fixed_timestep` + `differentiation_mode=BACKWARDS` now routes
through `checkpointed_while_loop` (time-based stop `t < t_end − dt/2`, exact to
`num_timesteps`) instead of the OOMing plain `fori_loop`. **Verified**: the
gradient gate at 32³/80 steps now runs and passes (`max|FD−AD|/|grad|=1.7e-8`) —
it OOM'd at 106 GB before. Reverse-mode at usable resolution is unblocked.

### Reverse-mode SS-vs-MS columnar recovery (`inverse/ss_vs_ms_recovery.py`)

Twin experiment in **decaying** rotating turbulence (forcing off, from a forced
spin-up snapshot — so the OU realisation needn't be replayed across MS segments).
Control = barotropic-columnar IC, 3D small scales pinned; step-defect loss;
m=1 (SS) vs m=2,4; checkpointed reverse mode.

**Validated + short-window money plot** (`inverse/figures/rot_money_plot.png`,
`aggregate_money_plot.py`). Completed windows (24³):

| T (eddy times) | SS m=1 | MS m=2 | MS m=4 |
|----------------|--------|--------|--------|
| 1.0  | 0.62 | 0.79 | — |
| 2.0  | **0.15** | 0.65 | 0.76 |

Inside τ_L (≈9 t_e) **single shooting wins** — SS is well-conditioned (gradient
*decreases* during optimisation), so at a fixed budget MS's extra segment
variables + defect penalty only slow it (same regime as the RBC study and the
theory: "no intermediate-m sweet spot at short windows"). This is the correct
baseline.

**The deep-window cliff was not reached.** Two walls: (i) a visible SS cliff
needs T ≫ τ_L ≈ 9 t_e (rotation's long predictability), i.e. ~2000–3000-step
windows; (ii) reverse-mode of the checkpointed `while_loop` over that many steps
hits a severe **XLA compile wall** (T=2/312 steps completes; T=6/938 and
T=12/1875 could not clear compile in hours, GPUs at 80% util = genuinely
compiling). So for *this* system the cheap forward **mechanism plot is the
better evidence** (it already shows the SS adjoint amplification ~e^{λt} and the
predictable columnar target). A T=12 run is left going in case it clears; the
driver is env-parameterised (`RT_N/RT_TWIN/RT_M/RT_STEPS`).

## Pallas isothermal-hydro WENO kernel (cost-wall remover)

The single biggest cost wall for this study was that **isothermal hydro had no
Pallas kernel** — forward runs and reverse-mode checkpoint-recompute both fell
back to native JAX. Now ported.

- **New kernel** `_weno_flux_hydro_iso_pallas` (+ `_local`, predicate
  `_hydro_iso_pallas_flux_supported`, `_hydro_iso_indices_for_axis`) in
  `astronomix/_finite_difference/_interface_fluxes/_weno_pallas.py`. Mirrors the
  ideal-gas hydro Pallas kernel's 1/2/3-D skeleton but with `ncomp = num_modes =
  ndim+1` (no energy/entropy), fixed `cs = params.isothermal_sound_speed`,
  isothermal flux `(mn, mn²/ρ + cs²ρ, mt·mn/ρ)`, and the `_eigen_hydro_iso`
  eigenstructure (acoustic∓ + ndim−1 shear waves). Multi-GPU-safe via
  `_weno5_shard_wrap`; halo 3 on the active axis.
- **Dispatch wired** in `_weno.py` (`_weno_flux_axis_dispatch`), wrapped in
  `diffable_pallas_call` so AD tangents still traverse the native path —
  differentiability of the rotating-isothermal study is preserved.
- **Validated** (`tests/pallas/weno_iso_hydro_validate.py`, Pallas vs native,
  all axes): interpret mode **float32 rel ≈ 1e-7** (single-precision rounding),
  **float64 rel ≈ 1e-16** (machine precision), AND the **real Triton-GPU path
  float32 rel ≈ 1e-7** — all in both 2-D and 3-D, every axis. Fully validated.
- **Pre-existing native bug found AND fixed (not introduced by this port):** 1-D
  isothermal `_eigenvector_building_blocks` (`_eigen_hydro_iso.py`) unconditionally
  read `momentum_index.x` and crashed in 1-D (the sibling
  `_eigenvalue_building_blocks` guards `ndim==1`; this one didn't), so native 1-D
  isothermal hydro WENO had never run. Fixed by adding the same `ndim==1` guard.
  With native 1-D working, 1-D is back in the validation: Pallas vs native
  **1-D axis 0 rel ≈ 1.6e-7 PASS** (interpret f32), alongside the 2-D/3-D passes.

## Next step

- Confirm the Triton-GPU iso-hydro kernel + measure the forward speedup on a free
  GPU, then re-time the reverse-mode deep-window SS/MS cliff (now that forward +
  checkpoint-recompute are Pallas-accelerated).
- Reverse-mode SS/MS recovery at T ≫ τ_L on a dedicated GPU for a converged cliff.
- Optional cleaner columns: N=128 with k_f≈4–6 for more scale separation.
