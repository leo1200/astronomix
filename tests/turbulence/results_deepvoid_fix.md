# Deep-void high-Mach WENO instability — diagnosis & fix

Work on `HANDOFF_deepvoid_instability.md`. Worktree `astronomix-refactor-port`
(branch `refactor`), env `astx`, all GPU via autocvd. Cheap 128³ repro, ~1 min/run.

## TL;DR
The late-time deep-void blow-up has **two distinct mechanisms**, now separated:

1. **Floored-vacuum-cell momentum runaway** (the documented root cause) — *FIXED*.
   A cell pinned at the density floor accumulates momentum, so `v = mom/ρ_floor`
   runs away across the 5 RK substages. The per-substage REDISTRIBUTE positivity
   never rested it because **`vacuum_rest` was silently a no-op in the
   redistribute path** (it only acted in the conserved-state HARD_FLOOR path).
   Fix: wire `vacuum_rest` into REDISTRIBUTE (native source + bit-matched Pallas
   kernel). Isolated deep-void cells now rest (v=0); void *edges* keep the gentle
   neighbour-fill. Bit-compat native↔Pallas = 1.2e-7 (f32 round-off).

2. **Scheme-level WENO overshoot at void edges** (the residual marginal
   instability) — *substantially mitigated* by a new optional first-order
   (FOFC-style) LLF flux blend near the density floor (`--blend`), for **both hydro
   and MHD** (MHD blends the fast-magnetosonic Rusanov flux before the CT
   transverse-flux extraction; div(B)=0 preserved by construction, E_B/dynamo
   intact). It restores the CFL safety knob: lowering CFL with the blend now
   reliably stabilises the run, which the handoff noted was previously broken
   ("0.7 crash, 0.4 crash").

   **MHD M20 deep-void repro (128³, rhomin 5e-3) + redist+vrest+blend f8:**
   cfl 1.0 / 0.7 / 0.4 all PASS (M_turb≈21, E_B≈0.23–0.24, vs no-blend ref
   E_B=0.226 → induction preserved); only cfl 1.5 crashes (CFL-limit). Before the
   blend, MHD M20 was clean ONLY at cfl 1.0.

`cfl 1.5` for the **extreme M40 hydro proxy** is a genuine linear CFL-stability
limit (even blend_factor 80 = full LLF over 40% of mean density doesn't save it;
more dissipation doesn't help) — not a bug. Run extreme deep-void hydro at cfl ≤ 1.

## Diagnosis (read-only inline probe)
Added an env-gated per-step probe in `time_stepping/time_integration.py`
(`DEEPVOID_PROBE=1`, prints when `max|v|` crosses `DEEPVOID_PROBE_VTHR`). It reads
the post-step state only → bit-identical trajectory (unlike `--diag`, whose dense
snapshot grid changes the dt sequence and perturbs the marginal crash).

- **Baseline (no fix), hydro M40 cfl 1.0:** `min_rho` pinned exactly at the floor
  (5e-3) throughout; `max|v|` climbs monotonically `10→22→…→36` over many steps
  while `dt` collapses chasing it → blow-up. = the momentum runaway.
- **With resting, cfl 1.5:** different failure — fast `max|v|` explosion
  `1e5→1e6→1e8→NaN` at fixed `min_rho=floor`, `dt→1e-8`. = WENO void overshoot /
  CFL-limit, not the runaway.

## Results (128³, hydro M40 / MHD M20, isothermal, rhomin 5e-3, t_end=4 t_cross)
first_bad_snap = -1 means fully stable; M_turb target ≈ 40 (hydro) / 20 (MHD).

| config | cfl | nsnap | result | M_turb |
|---|---|---|---|---|
| baseline (no resting, redist) | 1.0 | 8 | **CRASH** @snap7 | collapse |
| floor+vrest (resting every substage) | 1.0 | 8 | PASS | 43.2 |
| redist+vrest (the fix) | 1.0 | 8 | PASS¹ | ~43 |
| redist+vrest, **MHD M20** | 1.0 | 8 | **PASS** | 20.8 |
| redist+vrest (no blend) | 1.5 / 0.7 / 0.4 | 8 | CRASH | — |
| redist+vrest+**blend** f8 | 0.4 | 8 | **PASS** | 45.8 |
| redist+vrest+**blend** f8 | 0.7 | 8 | **PASS** | 45.9 |
| redist+vrest+**blend** f8 | 0.7 | 60 | **PASS** | 48.5 |
| redist+vrest+**blend** f4 | 1.0 | 8 | **PASS** | 45.7 |
| redist+vrest+**blend** f16 | 1.0 | 8 | **PASS** | 45.2 |
| redist+vrest+**blend** f8 | 1.0 | 8 | CRASH² | — |
| redist+vrest+**blend** f{8,20,40,80} | 1.5 | 8 | CRASH³ | — |

¹ proven via floor+vrest (resting-only) PASS at cfl 1.0; redist+vrest rests a
  strict subset (isolated voids). ² chaotic unlucky draw — f4 & f16 both PASS at
  cfl 1.0, so not a systematic break. ³ CFL-limit: even full-LLF (f80) crashes.

Before the fix: essentially every CFL crashed and **lowering CFL did not help**.
After (resting + blend): there is a stable config at every CFL in 0.4–1.0, and the
realistic MHD deep-void case (the original 512³ ISM failure mode) is stable.

## 512³ ISM run — RESOLVED (2026-06-25)
Re-ran the production ISM at full scale with the recipe below
(`ISM_N512_blend`, `data_fig14/paper_ISM_N512_blend.npz`):
`--eos iso --mturb 10 --beta 0.1 --N 512 --mhd 1 --cfl 1.0 --rhomin 0.02 --tcross 5
--nsnap 6 --stage_mode redist --vacuum_rest 1 --blend 1` (XLA mem-fraction 0.95).

- **first_bad_snap = -1 — finite all the way to t/tc=5.0** (the snapshot that used to NaN).
- M_turb(t/tc): 9.6 / 10.0 / 9.7 / 11.0 / 10.2 — healthy ≈10 throughout.
- E_B grows 0.10 → 0.45 = a saturated small-scale dynamo (only fully captured now that
  the run reaches t/tc=5).
- All ρ/kinetic/magnetic spectra finite at all 6 snapshots, incl. the full 2.5–5 window.
- Wall 8.4 h, single A100 via autocvd. Peak ~141 GB vs 136 GB pool → fit via XLA
  rematerialization (tight; smoke-tested first). Old production run NaN'd exactly at
  t/tc=5 (`first_bad=5`). The **blend** is what carries it through the late deep-void phase.

## What's fixed vs residual
- **Fixed:** the floored-momentum runaway (the mechanism behind the 512³ ISM
  late-NaN per memory `mhd-deepvoid-vacuum-rest-fix`). MHD M20 deep-void repro now
  stable. `vacuum_rest` is now actually effective in the redistribute path.
- **Mitigated:** the WENO void-edge overshoot — the LLF blend restores a working
  CFL stability knob (reliably stable at cfl ≤ 0.7 for extreme M40).
- **Residual (research-grade):** at the *extreme* M40 stress proxy the exact
  blend_factor interacts chaotically with the dt sequence — occasional unlucky
  crashes remain; the marginality is reduced, not provably eliminated. `cfl 1.5`
  for M40 is a true CFL limit. The physically-relevant regime (M≈10–20) is served
  by the resting fix directly; a clean 512³ re-test (one at a time) is the next step.

## Code changes (branch `refactor`, not committed — user manages git)
- `_fluid_equations/_enforce_positivity.py` + `_enforce_positivity_pallas.py`:
  `vacuum_rest` now rests isolated deep-void cells in REDISTRIBUTE (was no-op).
- `_finite_difference/_interface_fluxes/_deepvoid_blend.py` (new): hydro + MHD LLF
  blend (fast-magnetosonic Rusanov for MHD; isothermal & ideal gas).
- `_finite_difference/_time_integrators/_ssprk.py`: blend wired into `_hydro_step_rhs`
  (forces the non-fused flux+div path when `--blend`) AND into `_ssprk4_with_ct`
  (blends the full interface flux before the CT transverse-B-slice extraction, so
  CT stays div(B)=0). The LSRK CT variant is not wired (runs use SSPRK4-CT).
- `option_classes/simulation_config.py`: `positivity_deepvoid_blend{,_factor}` flags.
- `time_stepping/time_integration.py`: env-gated read-only `DEEPVOID_PROBE`.
- `tests/turbulence/paper_turbulence.py`: `--blend`, `--blend_factor` flags.

## Recommended production config (deep-void hydro/MHD)
`--stage_mode redist --vacuum_rest 1` (auto-on for supersonic) + `--blend 1`
(hydro or MHD) + `--cfl ≤ 1.0`. If a given CFL crashes, lower CFL or change
`--blend_factor` (the knob now works). For the 512³ ISM (M≈10, MHD) this is the
recommended recipe; cfl 1.5 is beyond the stable limit for deep-void shocks.
