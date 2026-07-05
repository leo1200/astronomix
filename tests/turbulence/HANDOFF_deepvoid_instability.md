# Hand-off: high-Mach deep-void WENO+CT instability (the core robustness problem)

> **UPDATE (2026-06-24): two mechanisms separated; one FIXED, one mitigated. See
> `results_deepvoid_fix.md` for the full table.**
> The blow-up is TWO things, not one:
> 1. **Floored-vacuum-cell momentum runaway** = the root cause — *FIXED*. Confirmed
>    with a read-only inline probe (`DEEPVOID_PROBE=1`): a cell pinned at the floor
>    accumulates momentum, `v=mom/ρ_floor` runs away across the RK substages while
>    `dt` collapses. Root reason: **`vacuum_rest` was a silent NO-OP in the
>    REDISTRIBUTE positivity path** (only the conserved HARD_FLOOR path honoured it),
>    so the per-substage redistribute never rested the void momentum. Fix: wired
>    `vacuum_rest` into REDISTRIBUTE (native `_enforce_positivity.py` + bit-matched
>    Pallas kernel, native↔Pallas = 1.2e-7). **The MHD M20 deep-void repro is now
>    stable** (the realistic case = the original 512³ ISM failure mode), and hydro
>    M40 cfl 1.0 is stable.
> 2. **WENO void-edge overshoot** = the residual marginal instability — *mitigated*
>    by a new optional first-order LLF flux blend near the floor (`--blend`, **hydro
>    AND MHD**; FOFC-style; ruled-out item #4 was a *positivity* limiter, this is a
>    *robustness* blend triggered on density-near-floor not on positivity). MHD uses
>    the fast-magnetosonic Rusanov flux, blended before the CT transverse-flux
>    extraction → div(B)=0 preserved, **E_B/dynamo intact** (E_B 0.23–0.24 vs
>    no-blend 0.226). It **restores the CFL stability knob** that the bullet below
>    says was broken: **MHD M20 redist+vrest+blend is stable at cfl 1.0/0.7/0.4**
>    (only 1.0 worked before); hydro likewise at 0.7 & 0.4, across nsnap 8/60.
>    `cfl 1.5` for the *extreme proxy* is a genuine CFL-limit (even full-LLF f80
>    crashes — more dissipation doesn't help), not a bug.
> **512³ DONE (2026-06-25): RESOLVED.** Production ISM (M10, β0.1, rhomin0.02, MHD, cfl1.0,
> t/tc=5) with `--stage_mode redist --vacuum_rest 1 --blend 1` ran CLEAN: first_bad=-1 to
> t/tc=5.0 (old run NaN'd there), M≈10 throughout, E_B 0.10→0.45 (saturated dynamo), all
> spectra finite incl. the 2.5–5 window. 8.4h, single A100. npz `data_fig14/paper_ISM_N512_blend.npz`
> — figure can be restored from this.
> RESIDUAL (still research-grade): at the extreme M40 proxy the blend_factor
> interacts chaotically with the dt sequence (cfl1.0: f4✓ f8✗ f16✓), so occasional
> unlucky crashes remain — marginality reduced, not provably eliminated. Next:
> a clean 512³ ISM re-test (one at a time) with redist+vrest (+blend, cfl≤1).
> New code is on branch `refactor`, NOT committed (user manages git).

---


**Worktree:** `/export/home/lstorcks/agent-home/astronomix-refactor-port` (branch `refactor`).
**Env:** `astx` (`.../envs/astx/bin/python`, jax 0.10.2). Always `PYTHONPATH=<worktree root>`.
**GPU:** ONLY via autocvd (`eval $(autocvd -q)` or `autocvd(num_gpus=1)`); never set
CUDA_VISIBLE_DEVICES by hand, never `-l`, never co-tenant another user's GPU. 512³ jobs ONE AT A
TIME (two concurrent 512³ host-OOM'd, exit 137). Full history: memory [[mhd-deepvoid-vacuum-rest-fix]].

## The problem
High-Mach turbulence in deep voids goes NaN at LATE time in the FD WENO(+CT) solver. Seen first as
the 512³ ISM (M_S~10, β=0.1, isothermal MHD) blowing up at the final snapshot (t/tc=5) while 256³/448³
were stable. It is a GENERAL high-Mach deep-void instability — pure hydro crashes too (just at higher
Mach). It is chaotically marginal (tiny dt-sequence changes flip crash↔no-crash).

## Cheap reproductions (~1.5 min each, 128³)
```bash
cd tests/turbulence            # paper_turbulence.py
# HYDRO crash (cleanest test bed — no magnetic complications):
python paper_turbulence.py --tag dbg --outdir data_repro --eos iso --N 128 --mhd 0 \
  --mturb 40 --beta 0.1 --F0 3.5 --cfl 1.0 --tcross 4 --nsnap 8 \
  --stage_mode redist --protect 1 --rhomin 0.005 --vmaxcap 1000000000 --vacuum_rest 0
#   -> first_bad_snap=7 (CRASH). (M20 is stable in hydro; M40 crashes.)
# MHD crash at lower Mach (low-beta adds stiffness): --mhd 1 --mturb 20 --rhomin 0.005 (same else).
```
A correct fix must make these `first_bad_snap=-1` WITHOUT masking, keep M_turb correct (~40 / ~20),
and stay clean across `--nsnap 8/60/200` AND `--cfl 1.5/1.0/0.7/0.4` (the marginality must be GONE,
not dodged).

## What's RULED OUT (do NOT re-chase — each was directly tested)
1. **Eigensystem degeneracy** — NO. Adversarial probe `tests/turbulence/_eigen_probe.py` (Bx→0, Bt→0,
   cf≈cs, ρ=floor, |v|=50): `_eigen_{L_row,R_col,all_lambdas}_iso` all FINITE and L·R=I to fp32
   round-off. The iso eigensystem already guards every sqrt (`diff_safe_sqrt`).
2. **Float precision/overflow** — NO. float64 (was `--x64`, reverted) makes it WORSE: fp32 ns200 was
   clean, fp64 ns8/60/200 all crash. fp32 round-off was adding dissipation that occasionally masked it.
3. **Parameter tuning** — NO. cfl is NON-monotonic (1.5 crash, 1.0 clean, 0.7 crash, 0.4 crash);
   nsnap flips it via the segment-boundary dt clamping. No robust setting exists.
4. **Positivity-preserving (Zhang-Shu / Hu-Adams-Shu) flux limiting** — NO (this was the big one).
   Implemented + validated-applied (was `_pp_flux.py` + `--pp_flux`, reverted). Does NOT fix the hydro
   M40 crash at cfl 1.0 OR 0.5. REASON: for ISOTHERMAL there's no positivity left to enforce — cell-avg
   density is already pinned at the floor by REDISTRIBUTE and p=cs²ρ>0; where ρ hits the floor the
   limiter degenerates to pure Lax-Friedrichs, which STILL crashes. So the blow-up is NOT a
   density/pressure overshoot.
5. **NaN firewall** (was `--nan_safe`, reverted) — "works" only by MASKING: on a severe event it zeroes
   the flow (finite-but-WRONG, first_bad_snap misreports clean). Not a fix.

## The diagnosis (where to aim)
It is **velocity / wave-speed driven**: in deep voids `v = mom/ρ_floor` gets large (diagnostics show
ρ pinned at the floor, |v|~40-50), and the LOCAL fast wave speed transiently outruns the step's CFL
budget — `dt` is computed ONCE per step (`_timestep_estimator.py:_cfl_time_step_fd_mhd_fast`) and is
NOT re-evaluated across the 5 SSP-RK substages, while per-substage positivity can inject these high-v
floored cells (see `_ssprk.py` pre_stage/`_apply_stage_positivity`). The high-order WENO characteristic
reconstruction then oscillates/overshoots at the high LOCAL Mach and eventually produces non-finite.

## Promising directions to try (untested)
- **Velocity/wave-speed-aware limiting**: cap the velocity that positivity injects into floored cells,
  AND/OR cap the reconstructed interface velocity, to a CFL-consistent bound (~`C_cfl·dx/dt` minus the
  local fast speed). NB an earlier crude `min(vmaxcap, C_cfl·dx/dt)` per-substage cap (tested on MHD,
  reverted) was too loose (≈λ_max); needs to be tighter and account for the cell's own fast speed.
- **Diffusive-flux blending in flagged cells**: detect deep-void / high-local-Mach interfaces and blend
  the WENO flux toward a robust LLF/HLL flux there (a robustness fix-up, not a positivity one). Differs
  from #4 above: trigger on velocity/Mach, not on density positivity.
- **Sub-stepping / dt re-evaluation**: recompute the CFL dt INSIDE the RK substages (or after the
  per-step vacuum protection), so the step can't over-advance a cell whose wave speed grew mid-step.
- **Lower WENO order / more dissipation locally** in flagged cells.
Validate any of these against the cheap repro bar above (clean across nsnap AND cfl, physics preserved).

## State of the code (after revert)
- KEPT (validated/useful): `vacuum_rest` auto-on for supersonic (`paper_turbulence.py`) — reduces void
  severity, physics-preserving, the one thing that genuinely helped (pushed the failure later); the
  NaN-safe progress bar (`astronomix/time_stepping/_progress_bar.py`, guards `int(NaN)`); diagnostic
  flags `--diag` (per-snapshot min_rho/max|v|/max|B|/max(b²/ρ)/nan via snapshot callback + progress bar;
  NB the dense callback perturbs the marginal crash, so use it on a robustly-crashing config or add an
  inline per-step isfinite check) and `--mhd 0/1` (hydro vs MHD); probe `tests/turbulence/_eigen_probe.py`.
- REVERTED (didn't help): `_pp_flux.py` + `--pp_flux`; `--x64`; the `--nan_safe` firewall
  (`_apply_stage_positivity` nan_to_num + `_nan_safe_interface_fields`). Core solver is back to baseline.

## Key files
- Timestep/CFL: `astronomix/_finite_difference/_timestep_estimation/_timestep_estimator.py`
  (`_cfl_time_step_fd_mhd_fast` / `_cfl_time_step_fd_hydro_fast`).
- RK + per-substage positivity: `astronomix/_finite_difference/_time_integrators/_ssprk.py`
  (`_ssprk4_hydro`/`_ssprk4_with_ct`, `pre_stage`/`finalize`, `_hydro_step_rhs`).
- Positivity: `astronomix/_fluid_equations/_enforce_positivity.py` (`_redistribute_positivity_native`,
  `_apply_stage_positivity`); flags `positivity_*` in `option_classes/simulation_{config,params}.py`.
- Iso eigensystem (already-guarded, NOT the bug): `astronomix/_fluid_equations/_eigen_mhd_iso.py`.
- Run driver + repro: `tests/turbulence/paper_turbulence.py`. Figures (unaffected): `make_fig14.py`,
  `make_fig15.py`. The figure's 512³ ISM npz is currently a corrupted firewall run — NOT YET restored
  (user deferred; restore via a clean `--tcross 3` run or `--tcross 5` with vacuum_rest, no `--nan_safe`).
