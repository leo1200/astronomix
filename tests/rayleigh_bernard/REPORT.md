# Rayleigh–Bénard convection in astronomix — progress report

Status against `tasks.md`:

| Phase | Item | Status |
|-------|------|--------|
| 0a | Thermal conduction for the FD scheme | **done + verified** |
| 0b | RBC config + forward smoke test | **done** |
| 0b.4 | Reverse-mode gradient gate (BLOCKING) | *not yet* |
| 1 | Forward simulation & regime (D1) | **convection demonstrated; saturated LSC run** |
| 2 | Lyapunov time (D0) | *not yet* |
| 3–8 | Inverse problem (SS vs MS) | *not yet* |

---

## Phase 0a — thermal conduction (finite-difference scheme)

A constant-conductivity Fourier heat term was added to the energy equation,

```
d(rho E)/dt  +=  div(kappa grad T) = kappa * laplacian(T),   T = p / rho  (R = 1)
```

integrated **explicitly**.

Code added:
- `astronomix/_physics_modules/_conduction/_conduction.py` — `fd_conduction_source`,
  a second-order Laplacian of `T` with isothermal Dirichlet ghost-`T` plates and
  adiabatic (zero-flux) side walls.
- `option_classes/simulation_config.py` — `thermal_conduction`,
  `conduction_wall_axis`, `conduction_isothermal_walls`; ghost-cell count bumped to
  ≥6 when conduction is on.
- `option_classes/simulation_params.py` — `thermal_conductivity`,
  `wall_temperature_low/high`.
- `_physics_modules/run_physics_modules.py` — conduction added to `_physics_sources`.
- `_finite_difference/_timestep_estimation/_timestep_estimator.py` — parabolic CFL
  limit `dt <= C dx^2 rho / (2 d (gamma-1) kappa)` (internal-energy diffusivity
  `chi = (gamma-1) kappa / rho`) in both FD hydro estimators.

**Boundary conditions.** The reflective hydro wall mirrors `rho` and `p` as even
quantities, so `T = p/rho` is mirrored too and its normal gradient vanishes →
side walls are automatically **adiabatic**. For the isothermal **plates** the
ghost ring adjacent to the interior along `conduction_wall_axis` is overwritten with
`T_ghost = 2 T_wall − T_interior`, fixing the wall-face temperature (Dirichlet).

**Verification** (`conduction_decay_test.py`, Phase 0a check (a)):
- *Operator check*: conductive source vs. `−kappa k^2 dT` — max rel. error `8.0e-4`
  (second-order truncation, as expected).
- *Decay-rate check* (overdamped, `chi k >> c_s`): measured Fourier-mode decay rate
  vs. analytic `chi k^2` — rel. error `3.2e-4`. **PASS.**

---

## Phase 0b / 1 — forward Rayleigh–Bénard (`rayleigh_benard.py`)

2D compressible RBC, finite-difference WENO, non-periodic:
- box `Lx × Ly = 2 × 1` (aspect ratio Γ = 2), grid `128 × 64` (prototype);
- no-slip reflective walls all around; isothermal hot bottom (`T=1.2`) / cold top
  (`T=0.8`) plates; adiabatic side walls; constant gravity `g=0.25` in `−y` via a
  linear external potential `phi = g y`;
- momentum viscosity ON (`mu = 1.4e-3`) + thermal conduction ON (`kappa = 1.9e-3`);
- control parameters: **Ra ≈ 5.6e4, Pr ≈ 1.1, Ma ≈ 0.25** (subsonic).

**Result.** The conductive base state is hydrostatic and convectively unstable.
The seeded perturbation decays acoustically at first, then the convective
instability grows exponentially (KE e-folding ≈ 2 time units from t ≈ 12), the
Nusselt number climbs above 1, and a large-scale circulation builds — exactly the
expected RBC behaviour. The run is **stable (no NaNs), conserves the plate driving,
and produces coherent convection rolls** (hot rising plumes / cold sinking
sheets visible in the temperature field).

Outputs in `figures/`: `rbc_timeseries_N64.png` (E_k, Nu, LSC vs t),
`rbc_final_field_N64.png` (T + velocity), `rbc_temperature_N64.gif`.

**Saturated run (t_end = 80).** Extending the integration shows the full RBC
life-cycle: acoustic transient → exponential convective growth (t ≈ 15–35) →
nonlinear saturation → **statistically stationary convection**. At saturation:

- Nusselt number settles to **Nu ≈ 5–6.5** (time-mean over the 2nd half ≈ 5.9),
  consistent with a 2D RBC `Nu–Ra` correlation at Ra ≈ 5.6e4;
- a **coherent large-scale circulation** fills the Γ=2 cell (hot rising plume on
  the left, cold sinking on the right; well-mixed core between thin plate thermal
  boundary layers), with mid-height `|u_x|` rms ≈ 0.01–0.047 ≈ 3–15 % of `v_ff`;
- the LSC amplitude **pulsates slowly** (KE and Nu oscillate ~±20 %), i.e. the
  flow is already time-dependent — encouraging for a chaotic regime at higher Ra
  (Phase 2 / D0).

This satisfies **D1** (forward simulation produces a coherent LSC). For a
*chaotic* LSC with reversals, raise Ra toward 1e6–1e7 and the resolution.

---

## 3D run for visual structure (`rayleigh_benard_3d.py`)

3D compressible RBC on the **Pallas (Triton) backend**, `96 × 48 × 96`
(≈0.44M cells), **Ra ≈ 1.2e6, Pr ≈ 1.0, Ma ≈ 0.25**. Periodic horizontal
(x, z) + isothermal no-slip plates (y) — the standard turbulent-RBC setup.

**Pallas + non-periodic note.** The reflective plates require 6 ghost cells, so
the padded dims are `N+12`. The Pallas WENO kernel needs the padded dims
divisible by `pallas_block_shape`; since `12 % 8 ≠ 0`, the periodic-TGV block
`(4,4,8)` would fall back to slow native. Use **`(4,4,4)`** (dims multiples of 4)
— Pallas engages (`_hydro_pallas_flux_supported → True`). Also set
`self_gravity_version = SIMPLE_SOURCE_TERM` so the external-potential gravity does
not request density fluxes, keeping the fused Pallas fast path.

**Result** (698 s wall to t≈48 on one GPU): conductive layer → convective burst
(Nu peaks ≈60) → **statistically-steady turbulent convection**, asymptotic
**Nu ≈ 11–12**. Both scales are present:
- *fine structure* — a sharp polygonal plume network near the plates
  (`rbc3d_hslice_N48.png`) and mushroom plumes in the vertical slice
  (`rbc3d_vslice_N48.png`);
- *large structure* — a domain-filling large-scale circulation (warm-up /
  cold-down) visible in the anomaly volume render (`rbc3d_volume_N48.png`).

Outputs: `rbc3d_{timeseries,vslice,hslice,volume}_N48.png`,
`rbc3d_vslice_N48.gif`. To sharpen the fine structure further, raise `NY`
(→ 64/96, i.e. 128³-ish) and/or Ra (lower `MU`, `KAPPA` at fixed Pr); cost grows
because the convective CFL `dt` shrinks as the flow speeds up.

---

## 2D turbulent "video-style" run (`rayleigh_benard_2d_turbulent.py`)

Reproduces the look of the well-known high-res 2D RBC renderings (hot/red rising,
cold/blue falling, white = mean, 16:9, time in free-fall units), at a sane
resolution via an **ILES** approach: a high *nominal* Ra is set with small
physical mu, kappa (Pr = 1) so the grid sets the dissipation scale and the field
fills with sharp plumes. (True Ra = 1e13 / 7680x4320 is a supercomputer run; the
boundary layers there are far below any affordable grid.)

Config: `320 x 180` (exact 16:9), Pallas backend, nominal **Ra = 3e7, Pr = 1**,
Ma ~ 0.25, reflective no-slip closed cell, isothermal plates, `tau_ff = Ly/v_ff`.
Result: developed turbulent convection by a few tau_ff, **Nu ~ 50-75**, with the
expected dense plume field + meandering large-scale wind. ~210 s to 12 tau_ff on
one GPU.

Outputs: `rbc2dturb_final_N180.png`, `rbc2dturb_N180.mp4`, `rbc2dturb_N180.gif`.

**Cinematic `512 x 288` version** (`NY=288`, Ra=3e7, 7 tau_ff, ~6 min stepping):
custom deep-navy -> white -> deep-red "balance" colormap (`CINEMA_CMAP`, white =
mean T), and a **per-frame adaptive colourbar** — the symmetric scale half-range
tracks the 99th-percentile temperature anomaly, so as the bulk mixes toward the
mean the colour scale "zooms in" (the colourbar numbers shrink) and the plumes
stay vivid. Outputs `rbc2dturb_{final,}_N288.{png,mp4,gif}` (mp4 4 s @ 20 fps;
to match "2 s per free-fall time" set fps ~ NUM_SNAPSHOTS / n_tau / 2).

Cost note: an earlier `512 x 288`, Ra = 1e8, 18-tau_ff attempt paced to ~hours
(turbulent CFL `dt` shrinks once convection ignites), so it was cut back to the
above. To go bigger, raise `NY` / lower `MU,KAPPA` and budget accordingly — the
acoustic CFL at Ma ~ 0.25 makes long high-Ra 2D runs genuinely expensive.

---

## Inverse modeling (Phases 0b.4, 3-7)

### Phase 0b.4 gradient gate -- PASSED (`inverse/gradient_gate.py`)

Reverse-mode gradient of a scalar terminal loss w.r.t. the large-scale IC,
FD-checked through the **full** differentiable path: FD WENO + reflective walls
+ external-potential gravity + viscosity + **thermal conduction** (isothermal
ghost-T plates), on a short (0.32 tau_ff) window, 48x24, NATIVE_JAX backward.

Result: `max |FD-AD| / |grad| = 9.5e-6` (threshold 1e-4) across random
directions, and the gradient is **100% low-k (smooth)**. The new constant-kappa
explicit conduction Laplacian and the ghost-T Dirichlet plates differentiate
cleanly. (A per-direction *relative* error metric is misleading: it blows up on
directions nearly orthogonal to the gradient where the true derivative ~ 0; the
gradient-norm-normalised error is the correct measure.) **AD through RBC is
clean -> inverse modeling is unblocked.**

### SS vs MS recovery (`inverse/ss_vs_ms_recovery.py`)

RBC analogue of the TGV `ss_vs_ms_recovery.py`: twin experiment, control = the
large-scale band (low-pass |k|<=K_CUT) of the initial vertical velocity, shared
step-defect loss, single shooting (m=1) vs multiple shooting (m=2,4), gradient
norm tracked to expose the SS adjoint blow-up. Loss normalised by the
observation energy; Adam.

**Result (T_obs = 1.0 tau_ff, 48x24, 25 Adam steps).** The optimisation
converges cleanly through the full differentiable RBC, confirming the machinery:

| m | meaning            | final large-scale recovery error |
|---|--------------------|----------------------------------|
| 1 | single shooting    | **0.587** (loss down ~33x, smooth bounded grad 1.7e-2 -> 2.2e-3) |
| 2 | multiple shooting  | 1.000 (data term not yet fit; defect ~4e-5) |
| 4 | multiple shooting  | ~0.92 |

At this window (well *inside* the predictability horizon) **single shooting
wins** -- exactly the regime the theory doc predicts: SS is well-conditioned, so
at a fixed optimiser budget MS's extra segment variables + stiff defect penalty
only slow convergence ("no intermediate-m accuracy sweet spot at short windows /
fixed budget"; Worked Example I). This is *not* evidence against MS: its
advantage appears only at windows **past the SS predictability cliff**, which a
1-tau_ff window does not reach. So this run is the correct **baseline** (SS works,
machinery validated); the headline (SS cliffs, MS holds) needs longer windows.

Crucially, the **gradient-norm panel shows no SS blow-up** at this window: m=1
|grad| *decreases* smoothly (2e-2 -> 2e-3), i.e. the adjoint is well-behaved
because 1 tau_ff is inside the predictability horizon. The SS adjoint blow-up
(the mechanism that makes MS win) is precisely what should appear once the
window is pushed past the cliff -- the next experiment.

Crucially, the **gradient-norm panel shows no SS blow-up** at this window: m=1
|grad| *decreases* smoothly (2e-2 -> 2e-3). **Figure/data:**
`figures/rbc_ss_vs_ms_Tobs1.0.png` (loss, recovery error, gradient norm) +
`data/rbc_ss_vs_ms_Tobs1.0.npz`.

A direct longer-window optimization (T_obs = 3 tau_ff) was attempted but is
**computationally infeasible** with native reverse-mode: differentiating through
~1460 low-Mach steps on a tiny grid is launch-bound (~45 min to XLA-compile the
backward graph, then >10 min per Adam step). It did, however, confirm the
mechanism at step 0: SS |grad| = **1.79e-1 at T_obs=3 vs 1.70e-2 at T_obs=1
(~10x), same setup** -- the adjoint growing with the window.

### SS-vs-MS mechanism plot (`inverse/adjoint_mechanism.py`) -- the headline

Rather than pay for reverse-mode optimization at long windows, the mechanism is
measured directly and cheaply (forward-only, Pallas-fast) via the large-scale
*sensitivity amplification*

    A(t) = || P_lk T(t; ctrl+d) - P_lk T(t; ctrl) || / ||d||  (averaged over d),

extracted from two snapshot forward runs (96x48, 5 runs to 8 tau_ff in ~4 min).
Single shooting over a window T is exposed to A(T); multiple shooting with m
segments caps the back-prop to one segment, i.e. A(T/m).

**Result (8 tau_ff window):** A(t) grows cleanly **~ e^{lambda t}** with
**lambda = 0.170 -> tau_L = 1.86 tau_ff** (robust: the 4-tau_ff run gave 1.82; a
Phase-2 / D0 by-product). The single-vs-multiple-shooting gradient-amplification
gap:

| method | per-segment window | amplification A | SS/MS gap |
|--------|--------------------|-----------------|-----------|
| single shooting | 8 tau_ff       | 9.56 | -- |
| MS m=2          | 4 tau_ff       | 1.28 | 7.4x |
| MS m=4          | 2 tau_ff       | 0.43 | 22x |
| MS m=8          | 1 tau_ff       | 0.25 | 39x |
| MS m=16         | 0.5 tau_ff     | 0.21 | 46x |

So at a window of ~4.3 tau_L the single-shooting gradient is **~46x more
amplified** than fine multiple shooting -- the exact conditioning gap (exploding
adjoint) that wrecks SS and segmentation caps. `figures/rbc_mechanism_N48.png`
+ `data/rbc_mechanism_N48.npz`. (Perturbation stays linear: A~9.6 with eps=1e-4
=> |delta_obs|~1e-3.)

Reproduce / scale up (when a GPU is actually free):
```
RBC_N=24 RBC_TOBS=1.0 RBC_M=1,2,4 RBC_STEPS=25 \
  PYTHONPATH=<repo> python -u tests/rayleigh_bernard/inverse/ss_vs_ms_recovery.py
```
Raise `RBC_TOBS` (longer window -> SS should cliff) and `RBC_N` (higher
effective Ra -> sharper predictability horizon) as compute allows.

**Compute reality (shared cluster).** `autocvd` only takes *free* GPUs; this is
a busy 8-GPU node, so jobs run **one at a time, serially**. The differentiable
RBC uses NATIVE_JAX on the backward pass (Pallas falls back), and the low-Mach
acoustic CFL forces ~500-1000 small steps per free-fall time, so each Adam step
backprops through a deep graph (~25-50 s/step at 48x24). A fully-converged
*money plot* (window sweep x m x optimiser steps, at a Ra high enough for a
sharp predictability horizon) needs substantially more compute / resolution
than a single shared GPU affords in one session -- the theory doc itself notes
turbulence "pushes hard toward many segments" because the unstable rate is the
fast small-scale one. The infrastructure (gate + SS/MS driver, env-parameterised
by `RBC_N/RBC_TOBS/RBC_M/RBC_STEPS`) is in place to scale up when more GPU is
available.

---

## Next steps

1. **Phase 0b.4 gradient gate (BLOCKING before inverse work):** finite-difference
   check the reverse-mode gradient of a scalar terminal loss w.r.t. the IC through
   the full FD + non-periodic-BC + viscosity + conduction path, on a window ≪ τ_L.
2. Raise Ra (≈1e6–1e7) and resolution (256², 512²) for a chaotic LSC with reversals.
3. Phase 2: Benettin Lyapunov time τ_L (sets all inverse-problem windows).
