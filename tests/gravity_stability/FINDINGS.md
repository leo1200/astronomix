# Self-gravity energy-conserving scheme: Evrard-collapse stability

Branch `gravity-stability` (off `refactor`). Worktree
`/export/home/lstorcks/agent-home/astronomix-gravity-stability`.
All runs: `astx` env, Pallas backend, fp32, free GPU via `autocvd`.

## The scheme

The conservative FD self-gravity energy source (`WENO_FLUX_GRAVITY`, the
"corrected flux-based" scheme; also `FD_FLUX_GRAVITY`) writes the gravitational
work as a flux divergence:

    q_hat_{i+1/2} = F_rho,{i+1/2} * phi_face_{i+1/2} - (dx^2/24) * corr_face
    S_energy_i    = -(q_hat_{i+1/2} - q_hat_{i-1/2}) / dx   ( + (-drho * phi)_i )

This conserves total energy to round-off because q_hat is a telescoping face
flux. The non-conserving `SIMPLE_SOURCE_TERM` instead uses `rho * v * a`, which
equals `v . (momentum source)` and so changes ONLY the kinetic energy — it never
touches the internal energy, which is why it cannot produce negative pressure.

## Crash mechanism (diagnosed, not guessed)

Cold Evrard collapse (e0=0.05), `enforce_positivity=False`, N=32/64:

- The conservative scheme drives the pressure slightly **negative at the
  cloud/ambient contact discontinuity** (r ~ 1.0–1.1, density ~1e-3) from
  t ~ 0.08 — NOT in the dense core (rho_max only ~9 at crash). The simple
  source's energy term is ~0 there (rho*v tiny in the ambient), so it stays
  positive.
- The negative-pressure pocket festers (p_min: -3e-6 -> -1e-4) and at t ~ 0.47
  (N=64) triggers negative density + a velocity blow-up (|v| -> 229) -> NaN.
- The within-stage `primitive_state_from_conserved` runs with
  `enforce_positivity=False`, so negative pressure -> `sqrt(neg)` -> NaN with no
  clamp to catch it.

Baseline survival (corrected diagnostic; conservation excellent until it dies):
- N=32: t_final = 0.40, |dE|/E ~ 2.6e-6 until crash
- N=64: t_final = 0.47, |dE|/E ~ 2.9e-6 until crash

(A subtle measurement trap: a crash leaves the remaining snapshot-buffer slots
bit-exactly zero, whose `total_energy` is "finite" and gives rel-err = 1.0 — the
naive energy-only diagnostic reads this as a late crash with 100% energy error.
The real crash is at peak in-fall. `diagnose()` now flags `total_energy == 0`.)

## What does NOT fix it (negative results)

- **phi_face upwind blend** (`gravity_energy_upwind`): no help; full upwind makes
  it worse — it breaks the cancellation between `-div(F phi)` and the cell-centred
  `-drho*phi` term, which must use a consistent phi.
- **Inter-cell flux distribution variants** (`FD_FLUX_GRAVITY`,
  `DONOR_ACCOUNTING`): crash identically — the instability is internal-energy
  positivity, not how the face flux is split between neighbours.
- **Global blend toward simple** (`gravity_energy_simple_blend`): only survives at
  blend=1.0 (= the simple scheme, ~39% energy error). Partial blends still crash.
- **Per-stage positivity** alone: HARD_FLOOR still crashes (within-stage
  conversion NaNs before the between-stage clamp); REDISTRIBUTE survives N=64 but
  injects 634x the energy — useless for conservation.

## What helps (the fix)

- **Adaptive local blend** (`gravity_energy_adaptive_blend = frac`): cell-by-cell,
  blend the conservative energy source toward the simple form only where it would
  remove more than `frac` of the cell's internal energy this stage:
      removal = max(0, -delta),  allowed = max(frac * e_int, 0)
      beta    = clip(1 - allowed/(removal+eps), 0, 1)
  with delta = (conservative - simple) energy, e_int = p/(gamma-1). Robust to
  p<=0 (allowed=0 -> beta=1 -> safe simple source there). Leaves the strictly
  conserving scheme untouched everywhere else.
  - N=64, frac=0.25: t_final 0.47 -> 0.68, |dE|/E ~ 1.5e-4, cloud-edge pressure
    held positive (~+1e-7 instead of -1e-4); collapse reaches rho_max ~26.
  - Delays but does not fully cure: residual pressure hovers at ~1e-7, BELOW
    minimum_pressure (3e-6).

- **Adaptive blend + a minimal pressure clamp** (`enforce_positivity`): the blend
  removes the gravity-driven drain so the floor only catches the residual
  contact-discontinuity undershoot, rather than firing constantly. At N=64 the
  floor extends survival 0.47 -> 0.78; floor-ALONE wrecks conservation (rel-err
  4.1 = 410%) while floor+adaptive keeps it at 6.4e-4 — a ~6000x reduction in the
  floor's energy injection.

## Resolution / survival table (WENO, fp32, no floor; t_end target = 1.2)

| N   | baseline t_crash | adaptive(0.25) t_crash | conservation |
|-----|------------------|------------------------|--------------|
|  32 | 0.40             | 0.56                   | ~2e-3        |
|  64 | 0.47             | 0.68                   | ~1.5e-4      |
|  96 | 0.76             | 0.82                   | ~5e-5        |
| 128 | 1.16 (~complete) | 1.16 (~complete)       | ~7e-5        |

The adaptive blend buys ~15–45% more in-fall time at every under-resolved N while
keeping energy conservation intact, and is benign (within round-off) at N=128
where the baseline already completes. It does NOT fully cure cold Evrard to
t=1.2 at N<=96 in fp32 — a residual deep-collapse void mechanism + the fp32
contact-discontinuity undershoot remain (the paper figures use fp64 native at
N=128, where the WENO scheme is fully stable; the fp32 Pallas blow-up on violent
collapse is a known single-precision artefact).

## Recommendation

Use `gravity_energy_adaptive_blend ~ 0.25` as a default-on stability guard for the
conservative FD self-gravity schemes: it is conservation-preserving (untouched
away from collapse cores), robustly handles p<=0, and materially raises the
crash time / lowers the effective resolution threshold. Pair with a minimal
`enforce_positivity` floor when an un-crashable run is required — the blend keeps
the floor's conservation damage negligible.

## MHD turbulence + self-gravity (adiabatic, driven)

Runner `mhd_grav_turb.py`: uniform magnetised box (rho0=1, beta=1, Bz0), OU
forcing to sonic Mach ~2, periodic self-gravity (Jeans swindle), ideal-gas EOS
(so the energy equation — hence the conservative gravity source — is actually
exercised; isothermal turbulence would not test it). G sets the Jeans length;
G=4 -> lam_J/L=0.44, G=8 -> 0.31. N=64, 2.5 t_cross. Figure
`figures/mhd_selfgravity_stability.png`.

**Without positivity protection** (enforce_positivity off): the supersonic
adiabatic turbulence itself (M_s -> 1.9) crashes ~t/tc 0.75–1.0 regardless of the
gravity scheme — the SIMPLE source crashes earliest. So the bare turbulence, not
the gravity scheme, is the limiter; positivity protection is required (this is
also the realistic production setting, cf. [[how-mhd-ou-turbulence]]).

**With positivity protection** (enforce_positivity + per-stage HARD_FLOOR + OU
`prot`, rhomin=0.02): the turbulence is stable and self-gravity drives a deep
collapse. Then the gravity scheme matters:

| case (G=8, protected)        | result   | rho_max |
|------------------------------|----------|---------|
| simple                       | FINITE   | 173     |
| WENO base (adapt 0)          | CRASH ~1.5 t/tc | 211 (peak before death) |
| WENO adapt 0.25              | CRASH    | 88      |
| WENO adapt 0.10              | **FINITE** | 209   |
| WENO adapt 0.05              | CRASH    | 191     |
| WENO global blend 0.5        | CRASH    | —       |
| G=4 WENO base                | FINITE   | 9.5     |
| G=4 WENO adapt 0.10          | FINITE   | 9.7     |

- The conservative WENO scheme crashes in DEEP MHD collapse (rho_max ~ 200) just
  like Evrard; the non-conserving simple source is unconditionally stable.
- The adaptive blend at frac=0.10 **stabilises the conservative scheme through the
  full deep collapse** where the baseline crashes. At milder gravity (G=4) the
  baseline already completes and the blend is benign.
- CAVEAT (honest): the deep-MHD-core instability sits near a threshold — frac=0.10
  survives but 0.05 and 0.25 do not (non-monotonic). Unlike Evrard's clean
  cloud-edge case, the blend here is a useful but delicately-tuned lever, not a
  guaranteed cure. For a robust production MHD+gravity run use the simple source,
  or WENO + adapt~0.1 + positivity protection, validated per setup.

## Positivity-preserving conservative energy redistribution (`gravity_energy_pp_redistribute`)

The cleanest realisation of "redistribute the energy between adjacent cells":
a conservative, activation-gated diffusion of the (gravity-predicted) internal
energy,
    f_{i+1/2} = w * A_{i+1/2} * (e_pred_i - e_pred_{i+1}),
applied as a few Jacobi passes, with A on only at faces touching a near-floor
cell. Because f is an antisymmetric FACE flux, the global energy sum is conserved
EXACTLY for any activation pattern — so unlike the simple/adaptive blends, this
keeps STRICT total-energy conservation while pulling internal energy into cold
cells from hotter neighbours. (Note: a fixed-donor phi_face was tried first and
made Evrard WORSE — concentrating the per-face work into one cell over-drains it;
the redistribution must be state-aware, toward cells that can afford it.)

Evrard N=64 (WENO, fp32):
- pp=0.10, passes=8, act=4 (no floor): t_crash 0.47 -> 0.58, |dE|/E = 6.7e-6
  (STRICTLY conservative — ~20x better than the adaptive blend's 1.5e-4).
- pp=0.10, passes=12, act=8 + floor: t_crash 0.47 -> 0.88, |dE|/E = 8.7e-5
  (best combination found: better survival AND ~7x better conservation than
  adaptive+floor's 0.78 @ 6.4e-4).

Head-to-head (WENO, fp32, t_end target 1.2), t_crash / |dE|/E:

| N   | floor only        | pp(0.1,8,4) + floor | pp alone (0.1,8,4) |
|-----|-------------------|---------------------|--------------------|
|  32 | 0.46 / 6e-2       | 0.47 / 1.2e-4       | 0.50 / 8e-6        |
|  64 | 0.78 / **4.1**    | 0.62 / 8e-5         | 0.58 / 6.7e-6      |
| 128 | **1.20 / 1.8e-4 (COMPLETE)** | 0.82 / 7e-5 | 0.74 / 8e-5     |

The decisive trade-off: floor-ONLY is the most ROBUST (completes N=128, survives
longest at N=64) but its conservation is destroyed where it matters (rel-err 4.1 =
410% at N=64 — the floor fires constantly, defeating the purpose of an
energy-conserving scheme). The pp redistribution gives STRICT conservation
(1e-4..1e-6) everywhere, but in its current form it can shorten survival and even
HURTS the well-resolved N=128 run (0.82/0.74 vs 1.20). So: your idea is the
correct one (it is the ONLY approach that keeps strict conservation while
improving positivity), but the source-level/gravity-only-approx implementation is
not yet a robust drop-in.

CAVEATS (honest): the redistribution is tuning-sensitive. Too many passes
(passes=20) or too-eager activation (act=8, no floor) DEsTabilises — the latter
even hurts the well-resolved N=128 run (0.74 vs 1.16 baseline). Root cause: e_pred
is built in the GRAVITY-ONLY approximation (it ignores the in-stage hydro flux
divergence), so the correction can fight the hydro update. The robust home for a
positivity-preserving conservative limiter is the SSPRK stage, acting on the FULL
post-update predicted state (a conservative cousin of the per-stage floor) — a
worthwhile follow-up, not yet implemented.

## Full-state conservative positivity limiter (POSITIVITY_CONSERVATIVE)

The robust home for the redistribution: a per-SSPRK-stage positivity mode
(`positivity_per_stage_mode = POSITIVITY_CONSERVATIVE`) that acts on the TRUE
post-update internal energy e_int = E - KE - E_mag (not the gravity-only
prediction). Same antisymmetric face-flux diffusion (exact energy conservation),
gated at faces touching a near-floor cell, + density floor/vacuum-rest for voids
+ a minimal residual pressure floor as the unconditional backstop. Implemented in
`_fluid_equations/_enforce_positivity.py::_conservative_energy_positivity`,
dispatched from `_apply_stage_positivity`.

KEY TUNING LESSON: the activation margin must be ~1.0 (genuine near-violations).
A larger margin (the initial default 4.0) switches the diffusion on across the
cold-but-HEALTHY Evrard ambient (e_int = rho*e0 = 5e-6 ~ 1.1 e_floor), which
over-diffuses and DEstabilises the well-resolved runs (N=96/128 crashed at t=0.02
with margin 4.0). With margin 1.0 it only fires at real violations.

Evrard, WENO, fp32, POSITIVITY_CONSERVATIVE (activation 1.0), t_crash / |dE|/E:

| N   | baseline | CONSERVATIVE (act 1.0) | conservation |
|-----|----------|------------------------|--------------|
|  32 | 0.40     | 0.27 (worse)           | 1.5e-6       |
|  64 | 0.47     | 0.62                   | 1.6e-5       |
|  96 | 0.76     | 0.86                   | 6.3e-5       |
| 128 | 1.16     | **1.20 COMPLETE**      | 1.6e-4       |

So the full-state limiter makes the previously-marginal N=128 run COMPLETE
cleanly and improves N=64/96, all with strict conservation — but no single
activation cures both ends (margin 1.0 helps high res yet hurts N=32; margin 4.0
the reverse). A NOTE on the integrator: applying the (nonlinear) positivity
operator to the SSPRK 5th-stage input breaks the Spiteri-Ruuth consistency and
crashed every N -> reverted; positivity must only touch stages 0..3 + finalize.

Adding a VELOCITY CAP to the limiter (clip |v| to positivity_max_velocity,
adjusting E so e_int is preserved) breaks the negative-pressure -> velocity-spike
cascade and helps further (best fp32 numbers, CONSERVATIVE act=1.0 + vcap=50):

| N   | baseline | CONSERVATIVE + vcap | conservation |
|-----|----------|---------------------|--------------|
|  32 | 0.40     | 0.40                | blows at end |
|  64 | 0.47     | 0.56                | 9.4e-6       |
|  96 | 0.76     | **1.00**            | 2.6e-4       |
| 128 | 1.16     | **1.20 COMPLETE**   | 1.6e-4       |

So the full-state limiter + velocity cap raises the crash-free resolution floor
from N~128 to N~96 and improves every resolution, with strict conservation while
healthy. N<=64 cold Evrard still does not reach t=1.2.

PRECISION IS NOT THE BLOCKER (key finding). fp64 native, cold Evrard:
- baseline: N=32 crash @0.22, N=64 @0.32 — EARLIER than fp32 (0.40/0.47)! fp32's
  extra round-off diffusion was accidentally stabilising.
- + CONSERVATIVE limiter: N=32 0.49, N=64 0.65 (extends), but |dE|/E blows to
  1e5/1e2 at the end (the residual floor firing hard at the catastrophic core).
So the crash is a genuine ALGORITHMIC positivity failure of the high-order WENO
reconstruction at the under-resolved collapse/ambient contact — not round-off.

REMAINING LIMIT / proper fix: a positivity-preserving WENO *reconstruction/flux*
limiter (e.g. Hu-Adams-Shu: blend the interface flux toward a positivity-
preserving Lax-Friedrichs flux so the first-order substep keeps rho, p >= floor).
That is the rigorous route to UNCONDITIONAL stability at all N; it is a separate,
substantial component (native + the fused Pallas kernel) not yet implemented here.
The FD WENO path has no reconstruction fallback (first_order_fallback is FV-only).

## THE FIX: positivity-preserving WENO flux limiter (`positivity_preserving_flux`)

The reconstruction-level cure (recovered + extended from the removed git commit
320d383 "Positivity-preserving flux limiter for FD MHD WENO"). A Hu-Adams-Shu
(2013) / Zalesak-FCT limiter blends each WENO interface flux toward the
first-order Lax-Friedrichs flux,
    F_hat = F_LF + theta * (F_WENO - F_LF),  theta in [0,1] per interface,
choosing theta so the cell updated with the limited flux keeps BOTH
density >= minimum_density AND pressure >= minimum_pressure. The recovered MHD
limiter only did density; the hydro sibling here
(`_finite_difference/_interface_fluxes/_pp_flux_limiter.py::pp_limit_flux_axis_hydro`)
ADDS the pressure constraint (a quadratic in theta, q(t)=rho(t)(E(t)-e_floor)-
0.5|m(t)|^2 >= 0, solved by vectorised bisection) — and the pressure constraint
is the one that binds on a cold self-gravity collapse. Wired into the hydro FD
SSPRK non-fused path (forces non-fused when on); gated, default off.

**RESULT — cold Evrard (e0=0.05) now COMPLETES t=1.2 at every resolution, with
strict conservation, no other knobs:**

| N   | baseline   | PP flux limiter |
|-----|------------|-----------------|
|  32 | 0.40 crash | **1.20, |dE|/E 9.8e-6** |
|  64 | 0.47 crash | **1.20, |dE|/E 3.2e-5** |

This is the unconditional-stability fix the others approximated: it acts BEFORE
the divergence (where the negative pressure is created), is exactly conservative
(it only redistributes the high-order flux correction between adjacent cells, the
LF base being conservative), and keeps full WENO accuracy where the flow is
admissible (theta=1).

**UNCONDITIONAL across resolution (cold Evrard e0=0.05, t_end=1.2, fp32, PP flux
limiter ONLY — no state limiter, no enforce_positivity):**

| N   | crashed | t_final | |dE|/E   |
|-----|---------|---------|----------|
|  16 | False   | 1.20    | 7.3e-6   |
|  24 | False   | 1.20    | 8.7e-6   |
|  32 | False   | 1.20    | 9.8e-6   |
|  48 | False   | 1.20    | 2.1e-5   |
|  64 | False   | 1.20    | 3.2e-5   |
|  96 | False   | 1.20    | 5.1e-5   |
| 128 | False   | 1.20    | 7.0e-5   |

Every resolution from 16^3 to 128^3 completes cleanly with strict conservation
(error grows mildly + monotonically with N = the round-off accumulation of more
steps, NOT a stability trend). This is the deliverable: cold Evrard is now
unconditionally stable with the energy-conserving self-gravity scheme.

## MHD turbulence + self-gravity with the PP flux limiter

The PP flux limiter has an MHD sibling (`pp_limit_flux_axis_mhd`, density +
pressure-incl-magnetic constraint, the latter solved by bisecting the
internal-energy residual which is cubic in theta for MHD), wired into the MHD CT
SSPRK/LSRK integrators. Same gate `positivity_preserving_flux`.

Driven adiabatic MHD turbulence + self-gravity (N=64, M_s~2 forcing, beta=1,
strong gravity G=8 -> lam_J/L=0.31, 2.5 t_cross):
- baseline WENO, NO positivity: turbulence itself crashes ~t/tc 0.75-1.0.
- WENO + HARD_FLOOR protection: crashes at the deep collapse ~t/tc 1.5.
- **WENO + PP flux limiter ONLY (no floor, no protection): ALL FINITE** — full
  2.5 t_cross, rho_max ~ 149, M_s -> 3.9. One conservative reconstruction-level
  mechanism replaces the non-conservative floor for BOTH the turbulent shocks and
  the gravitational collapse. (A tiny residual p_min ~ -3e-3 appears from the MHD
  interface reconstruction; it stays finite and completes — a small hard pressure
  floor or per-step CONSERVATIVE mode mops it up if strict positivity is needed.)

## Structure formation in MHD turbulence + self-gravity (science run)

With the PP flux limiter providing stability, a production run:
`structure_formation.py` — driven adiabatic MHD turbulence + self-gravity,
N=128, M_turb~3, beta=1, G=6 (lam_J/L=0.24), OU-forced, 4 t_cross, NO floors /
protection (positivity_preserving_flux only). 711 s on one GPU, ALL FINITE.
Figure `figures/structure_N128_G6.png` (make_fig_structure.py).

Physics (the star-formation signature):
- density PDF starts ~log-normal (turbulence) and develops a high-density
  POWER-LAW TAIL as self-gravity drives the densest fluctuations to collapse;
  the mass-weighted PDF shifts to high rho.
- rho_max: 1 -> ~240 (collapse runaway; plateaus at the grid-resolved core peak
  ~160-240, as expected for grid self-gravity without sink particles).
- dense-gas mass fraction builds up: f(rho>10) -> 0.40, f(rho>30) -> 0.23,
  f(rho>100) -> 0.045 by t/tc~3.6.
- magnetic amplification: <|B|> rises with rho between the rho^0.5 (strong-field)
  and rho^2/3 (isotropic-compression) references; box E_B 0.067 -> 0.33 (~5x).
- mid-plane rho slice = turbulent filaments + dense knots; column density (the
  observable) shows the projected structure.

This is the payoff of the stability work: the energy-conserving high-order
self-gravity scheme runs a full magnetised structure-formation calculation to
deep collapse with no non-conservative floors, on the conservative
reconstruction-level (PP flux) limiter alone.

## Resolution study (structure_formation.py --final_only, make_fig_resolution.py)

Memory-safe `--final_only` mode (no per-snapshot states; diagnostics from the
final state — required at 256^3/512^3 where storing snapshots is hundreds of GB).
All stabilised by the PP flux limiter only (no floors). G=6, M_turb~3, beta=1.

| N   | t/tc | wall    | rho_max | f(>10) | f(>30) | f(>100) | E_B/E_B0 |
|-----|------|---------|---------|--------|--------|---------|----------|
| 128 | 4.0  | 711 s   | 262     | 0.404  | 0.234  | 0.045   | 4.9      |
| 256 | 2.5  | 6823 s  | 212     | 0.438  | 0.270  | 0.071   | 8.4      |
| 512 | 1.0  | 10134 s | 16      | 0.004  | 0.000  | 0.000   | (onset)  |

All ALL-FINITE, no floors. 512^3 used ~108 GB (fits 143 GB) — final-state only.
NOTE the table mixes evolutionary stages (128@4.0 & 256@2.5 are deep in collapse;
512@1.0 is only at collapse onset, hence small dense-mass-frac). For a fair
resolution test, an ISO-TIME comparison at t/tc=1.0 (turbulent PDF + B-amplification
= the resolution-CONVERGENT quantities, before the resolution-divergent runaway
collapse) is done across N=128/256/512 (tags *_t1, `structure_resolution_t1.png`).
The deep-collapse pair (128@4.0 vs 256@2.5) separately shows finer grids resolve
DENSER cores (f(>100) 0.045->0.071) and a STRONGER small-scale dynamo
(E_B/E_B0 4.9->8.4). Figures `figures/structure_resolution{,_t1}.png`.

ISO-TIME convergence at t/tc=1.0 (turbulent + collapse onset), N=128/256/512:
rho_max 16/21/16, f(>10) 0.003/0.011/0.004, E_B/E_B0 4.31/4.52/4.33. The density
PDF (volume & mass) and the rho-B relation OVERLAP across resolution -> the
turbulent magnetised state is resolution-CONVERGED. So the divergence in the
deep-collapse comparison is the (expected) resolution-dependent core collapse, not
a turbulence/scheme artefact. Confirms the PP-flux-stabilised scheme is
well-behaved and convergent in the regime where convergence is meaningful.

Wall-time note: cost ~ cells x timesteps, and the deep collapse drives dt down
hard, so 512^3 to the full collapse (t/tc~2.5) is a ~1-day calc; N=512 is run to
1.0 t_cross (turbulent PDF + collapse onset — the resolution-CONVERGENT part; the
deep collapse is resolution-divergent without sink particles). 256^3 completed in
1.9 h, all finite, no floors — a non-trivial magnetised self-gravitating collapse
run entirely on the conservative PP-flux stabilisation.

## Config knobs added (all default to current behaviour)

`SimulationConfig`:
- `gravity_energy_upwind: float = 0.0`           (donor-cell phi_face blend)
- `gravity_flux_correction_factor: float = 1.0`  (scales the dx^2/24 correction)
- `gravity_energy_simple_blend: float = 0.0`     (global blend toward simple)
- `gravity_energy_adaptive_blend: float = 0.0`   (local positivity-aware blend)
- `gravity_energy_pp_redistribute: float = 0.0`  (CONSERVATIVE positivity diffusion) <-- best conservation
- `gravity_energy_pp_passes: int = 8`            (Jacobi passes for the above)
- `gravity_energy_pp_activate_factor: float = 4.0` (activation margin in floor units)

Implemented in `astronomix/_modules/_gravity/_gravity.py`. The blends are in both
`WENO_FLUX_GRAVITY` and `FD_FLUX_GRAVITY`; the pp-redistribution is in
`WENO_FLUX_GRAVITY`.
