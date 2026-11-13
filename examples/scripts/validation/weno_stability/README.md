# WENO stability without the patchwork

Branch `weno-stability`, worktree `/export/home/lstorcks/jf1uids-weno-stability`.

**Goal:** replace the post-hoc stabilisers that the high-Mach turbulence and
self-gravity runs need by a principled change inside the WENO scheme. The
stabilisers in question: vacuum `prot`, per-stage/per-step hard floors,
`vacuum_rest`, deep-void LLF blending, the Zalesak flux limiter,
and a 0.02 density floor.

All measurements below use **no** positivity machinery
(`minimum_density = 1e-10`, no `prot`, no floors, no blending) unless stated.

## In one paragraph

The bare scheme has two structural defects, both inside WENO:

1. The characteristic basis is evaluated at an interface "state" that is not
   the state of any gas.
2. The split-flux reconstruction is not positivity-preserving: next to a
   low-density cell it is *anti-diffusive* and drains it.

The fix is two options (native kernel and all three Pallas kernels):

* `weno_admissible_face_state`: average the pressure, not the enthalpy.
* `weno_positivity_preserving`: Zhang & Shu (2012) split-flux scaling inside
  the reconstruction. Implies the face state. For ideal MHD the scaling acts
  on the weighted *pairs* of each cell's update (Defect 4): a single
  Lax-Friedrichs split state of MHD is not admissible at low beta, and the
  first version silently fell back to first order there.

With them, Mach-10 driven turbulence completes without any other stabiliser:
isothermal MHD at 64³, 128³ and 256³; adiabatic MHD; isothermal hydro at the
provable CFL. So does the cold Evrard collapse, down to 32³. Measured
side-by-side, the current production recipe survives or aborts at 128³
depending on run-to-run variation.

A third failure, specific to self-gravity, is **not** a WENO defect. The
energy-conserving gravity coupling digs internal-energy holes wherever the
grid does not resolve the pressure scale height. The WENO fixes make those
runs complete and conserve energy, but only resolution or a different
coupling removes the holes.

## Status of the options

* `weno_admissible_face_state` is **on by default** (bug fix; smooth-flow
  results agree with the old basis to the WENO dissipation level).
* `weno_positivity_preserving` is opt-in: switch it on for supersonic and
  self-gravity runs. Its price is low-Mach contact accuracy (see Costs).

## Relation to the existing Hu-Adams-Shu limiter (`preserving_flux`)

Both come from the same positivity framework (Zhang & Shu 2010-12; Hu, Adams
& Shu 2013), but they act at different places.

* **Where.** `preserving_flux` limits the *finished* WENO flux. It blends the
  flux toward a separately computed two-cell Lax-Friedrichs flux, with `theta`
  chosen from Zalesak sums of the *updated cell values* at the current stage
  `dt`. PP-WENO scales each *split flux's face value* toward its upwind split
  state inside the reconstruction. Its constraint (face value and mirror
  admissible) does not involve `dt`.
* **Why that matters.** `preserving_flux` checks each axis on its own, with no
  dimensional factor, so three admissible directional updates can still sum to
  a negative density in 3D (seen: rho = -4.9e3). And because the bound is on
  the update, it loosens as `dt` shrinks, which lets a one-cell minimum drain
  over many steps. The face-value bound forbids reversing the inflow at every
  step.
* **Splitting speed.** `preserving_flux` keeps the per-field (Roe-like)
  splitting in the high-order flux. PP-WENO splits every mass-carrying field
  with the stencil's spectral radius, so the frozen-basis splitting is
  monotone for every stencil cell. Without that (commit 96a191f) Mach-10 MHD
  still blows up.
* **History.** A density-only Hu-Adams-Shu limiter was added in June
  (320d383) and removed (2d1301c) because 1D tests never needed it. That is
  consistent with the findings here: the failures are multi-D one-cell minima,
  which 1D Riemann problems do not produce.

## Defect 1: the characteristic basis is evaluated at a non-state

`_eigenvector_building_blocks` averaged `rho` and `m`, so `v = <m>/<rho>` is
mass-weighted, while the specific enthalpy `h` was averaged *unweighted*. It
then set `c^2 = (gamma-1)(<h> - v^2/2)`. For two cells of equal temperature
this gives the values below (`face_c2_demo.py`; true `c^2 = 1`):

| velocity jump | density ratio 1 | 10 | 100 | 10^4 |
|---|---|---|---|---|
| 3 c  | 1.75 | 0.02 | **-0.44** | **-0.50** |
| 10 c | 9.3 | **-9.9** | **-15** | **-16** |
| 100 c | 834 | **-1087** | **-1600** | **-1665** |

The values also change in a boosted frame, so the average is not Galilean
invariant. When `c^2 < 0` the clamp sets `inv_c2 = 0`. That **removes the
acoustic upwind correction** and leaves the undissipated central flux at
exactly the strongest jumps.

A cold dense slab rammed at Mach ~800 into tenuous gas (`riemann1d.py`,
`cold_ram`) blows up within ~2 steps. With `c^2 = gamma <p>/<rho>` (enthalpy
rebuilt from it) it gives the exact solution. The change is free elsewhere:

* identical L1 on Sod, Einfeldt 123, near-vacuum, Toro-3 and LeBlanc;
* still 5th order;
* ~+4 % runtime.

Alone it does not stabilise the turbulence: adiabatic beta = 0.1 MHD fails
because of pressure positivity.

## Defect 2: the split-flux reconstruction drains one-cell minima

Bare Mach-10 driven turbulence (HOW-MHD ISM case, `turb.py`) dies at
t/t_c ~ 0.6-0.85, isothermal hydro and MHD alike. The dying cell (`turb_diag.py`)
is a *single* density minimum (rho ~ 0.02 among 0.3-3) whose velocity is
unrelated to its neighbours'. Restarting that state without forcing
(`turb_restart.py`) reproduces the blow-up in ~15 steps: the cell's density
falls 2-3x **per step**, while the local divergence allows ~4 %.

Mechanism:

1. The Lax-Friedrichs speed is set by the low-density cell's large fast speed.
2. That makes the dense neighbours' split fluxes `(F - alpha q)/2` large.
3. The fifth-order extrapolation from the dense side then reverses the split
   flux that should flow into the minimum.

| change on the restart | outcome |
|---|---|
| float64 | NaN at the same time |
| CFL 1.5 -> 0.75 | NaN (later) |
| one splitting speed, no limiter | NaN (later) |
| existing Zalesak limiter (`preserving_flux`) | **density goes negative** (-4.9e3), NaN at the same time |
| LLF in cells with rho < 0.1 (probe) | stable |
| **positivity-preserving reconstruction** | **stable** |

The existing limiter fails for two reasons:

* it checks admissibility per axis with no dimensional factor;
* it bounds the *update*, so at small time steps a gradual drain continues.

### The fix: positivity-preserving split-flux reconstruction

Let `alpha` be the stencil's spectral radius. Splitting field `s` with speed
`alpha_s <= alpha` makes each split flux a scaled vector:

    f^+- = +-(alpha/2) w^+-,   w^+- = q +- F/alpha + sum_s (alpha_s/alpha - 1) R_s L_s q.

The WENO face value `w_hat` is replaced by `w + theta (w_hat - w)`. The
scaling is the largest `theta` in [0, 1] for which both of these are
admissible:

* the face value `w + theta d`;
* its mirror `(q +- F/alpha) - theta d` about the unshifted state.

The forward-Euler update of a cell is then a convex combination of admissible
states for `lambda (alpha_left + alpha_right) <= 1`, i.e. C_cfl <= 1/2 in the
code's sum-of-speeds CFL, or 0.75 with SSPRK4 (Zhang & Shu 2012). The cell's
own flux cancels between its two faces, so face-local speeds are fine. The
density bound is closed form. The pressure is concave along the segment, so
the chord's root is admissible: one closed-form step, no iteration.

`theta < 1` is a first-order candidate entering the WENO combination with a
weight set by admissibility instead of smoothness. In smooth flow `theta = 1`.

**Which splitting speeds.** Fields that carry mass use the common `alpha`.
That makes the splitting in the frozen basis monotone for *every* cell on the
stencil, however badly the face basis represents it (a void with a large fast
speed next to dense gas). For an isothermal gas, whose admissibility is a
density condition only, the fields that carry no mass keep their own speed
(shear waves, MHD Alfven waves), so vortical modes keep the default
dissipation. For an ideal gas every field uses `alpha` (since 2026-10-27).
The ideal-gas shear waves could keep their own speed too: in a cell's update
the shifts of the two cells at a face combine into the shift jump
`z_nb - z_own`, which raises the pressure of the first-order inflow state by
`2(gamma-1) kappa (1 - kappa/s) rho_i^2 rho_nb/(rho_i + rho_nb)^2 |dv_t|^2 >= 0`,
and the face check has to include that jump (it does,
`test_positivity_shear_shift.py`).

*Explored and rejected* (commits 25e1312, 96a191f): moving every field's speed
toward `alpha` only as far as needed to keep the split states admissible.
This leaves smooth flow exactly at the default errors, including low-Mach
contacts. But Mach-10 beta-0.1 isothermal MHD then NaNs at t/t_c ~ 1.05 at
128³ and 256³, at CFL 1.5, 1.0 and 0.75 alike, with rho >= 0.02 just before.
Admissibility of the split states is not the monotonicity that the common
speed provides. The rigorous per-field bound (Gershgorin on `L A_m R` over
the stencil) needs a Jacobian product per cell and field.

## Why the CFL limit is 0.75

* **The bound.** The forward-Euler stage is a convex combination of
  admissible states if `2 dt sum_d alpha_d / dx <= 1`. In the code's
  sum-of-speeds CFL (`dt = C dx / sum_d max lambda_d`) that is C <= 1/2.
  SSPRK(5,4) multiplies this by its SSP coefficient 1.508, giving
  **C <= 0.754**. The code's three-register final stage carries a -0.0208
  weight on u0, but it is algebraically identical to Spiteri-Ruuth's
  all-positive Shu-Osher form (the L(u3) term was eliminated via u4), so the
  SSP property holds.
* **Two caveats.**
  * In floating point, that subtraction can round a nearly emptied cell
    slightly negative; the Shu-Osher form would not, at the cost of extra
    registers.
  * `dt` is set from the speeds at the start of the step, and near-vacuum
    velocities can grow during the stages.
* **Measured**, deep-void isothermal hydro, Mach 10, 128³ (density reaches
  1e-8 to 1e-10):

  | C | 0.5 | 0.6 | 0.75 | 0.9 | 1.0 | 1.5 |
  |---|---|---|---|---|---|---|
  | result | complete | complete | complete | NaN 3.46 | complete | NaN 3.37 |

  Below the bound, every run completes. Above it, the outcome depends on the
  trajectory, like the non-monotonic CFL behaviour recorded before this work.
* **Why MHD survives C = 1.5.** The bound is sufficient, not necessary: it
  only bites when a cell is nearly emptied within one step. At beta = 0.1 the
  field keeps the voids near 1e-3 (min rho 9e-4 at 256³), far from that.
  Mach-10 isothermal hydro genuinely evacuates cells, since an isothermal
  rarefaction gives rho ~ exp(-dv / 2c).

## Defect 4: ideal MHD, where single split states are not admissible

The positivity argument needs the first-order split states q +- F/alpha to be
admissible. For the Euler equations that holds once alpha >= |v_n| + c. **For
ideal MHD it does not hold at any multiple of the fast speed** (Wu 2018, SIAM J.
Numer. Anal. 56, 2124). Take B along the normal and v = 0:

    p(q + F/alpha) / (gamma - 1) = p / (gamma - 1) - (p - B^2/2)^2 / (2 rho alpha^2) < 0
    at low beta for alpha = c_f.

Random low-beta states give inadmissible single split states in 49 % of cases
at beta ~ 0.1 and 73 % at 1e-3 to 1e-5 (`wu_pair_check.py`).

Consequences, measured with `alfven_pp.py` (CP Alfven wave, x64, N = 8/16/32):

* the theta-scaling has no admissible base and returns theta = 0, so the
  face is first-order Rusanov **in smooth flow**;
* orders were 0.2-0.6 (!);
* at N = 32 the error was 1.7e-3 (beta 0.2) and 2.5e-2 (beta 0.02), against
  8.7e-6 / 8.8e-6 without PP.

That is, the ideal-MHD runs of the previous sections were locally first order
at low beta, and they still produced negative pressures: min p = -5.7e-4
(Mach 10) and -6.1e-4 (Mach 20) at 256³ without the clamp.

The low-beta blast wave (`mhd_blast.py`, Balsara & Spicer, ambient
beta = 2.5e-4) exposed three more holes. Each was found by stepping the real
integrator stage by stage (`mhd_step_check.py`, `mhd_fe_check.py`):

1. **Zeroing the normal-B flux** (my own earlier addition) removed the
   Lax-Friedrichs diffusion of B_n from the update but not from the energy.
   At beta ~ 1e-6 that alone made one forward-Euler step at C_FE = 0.125
   negative. CT never uses that component, so it stays now.
2. **Stage states not re-synced.** The SSPRK increment was evaluated at the
   state whose cell-centred B is rebuilt from the faces, but was added to the
   stored stage state. Both have the same pressure, yet their sum moves p by
   (gamma-1) lambda dF_B . (B_stored - B_faces). Now every stage is re-synced
   with the pressure held.
3. **The 3-register SSPRK(5,4)** has a -0.0208 u0 weight. Its final
   combination is the convex Shu-Osher form only while u4 is unmodified.
   The 0.1169 share that stands in for 0.0961 u3 + 0.0637 dt L(u3) now
   multiplies u4 as formed. Bit-identical when there is no re-sync.

Two remedies:

**Rejected (87489da): raising alpha per cell** until both single split states
keep half the pressure.

* It is provable and restores the order (4.1 / 4.8).
* But alpha ~ 0.58 v_A / sqrt(beta) makes dt ~ sqrt(p)/B^2. In Mach-20
  adiabatic turbulence dt was 14x smaller at t = 0.25 t_c; the blast needed
  dt ~ 1e-7.

**Adopted: paired admissibility** (`_paired_scalings`). The per-cell
forward-Euler decomposition only needs two *weighted pairs* admissible:

    q_i^{n+1} = (1 - lambda S) q_i + (lambda S/2) (own_i + in_i),   S = alpha_L + alpha_R

* own_i is the cell's two mirror states. Its flux cancels, so the base is q_i
  itself.
* in_i is the two inflow states: Wu's generalized LF splitting, in which the
  neighbours' magnetic-tension terms cancel up to their B_n difference.
  Random low-beta pairs are inadmissible at the fast speed in ~1e-4 of cases
  (0 at 1.5x), against 70 % for single states.
* Each theta takes the smaller fraction of its two pairs (parallelogram
  corners).
* The Pallas kernel returns the split face fluxes, and the recombination is
  shared with native (5e-12 apart in x64).

| ideal MHD, C_cfl = 0.75 | Alfven N=32, beta 0.2 / 0.02 | orders | low-beta blast 50² / 100² | dt cost |
|---|---|---|---|---|
| no PP (face state) | 8.7e-6 / 8.8e-6 | 4.9 | fails at t = 0.0042 (also at C 0.375) | - |
| old PP (single states, B_n zeroed) | 1.7e-3 / 2.5e-2 | 0.2-0.6 | - | none |
| raised alpha (rejected) | 1.7e-5 / 7.1e-5 | 4.1-4.8 | completes | up to 14x+ |
| **paired** | **1.25e-5 / 1.31e-5** | **4.1-4.8** | **completes, min p 2.4e-2 / 2.9e-2** | **none** |

What is still not a proof: in multi-D the inflow pair's base keeps a residual
proportional to B_n(i+1) - B_n(i-1). Wu & Shu remove it with a discrete
div B = 0 condition or the Godunov-Powell source. Faces whose inflow base is
inadmissible get theta = 0, a first-order LF inflow, which is not guaranteed
positive either.

## Results

### Driven Mach-10 turbulence, beta = 0.1, CFL 1.5, no floors / prot / blending

| run | bare | PP-WENO |
|---|---|---|
| isothermal MHD 64³ | NaN t/t_c = 0.83 | **complete (5 t_c)** |
| isothermal MHD 128³ (native, Pallas x2) | NaN ~0.85 | **complete**, 133 s on an A100 |
| isothermal MHD 256³ (Pallas) | — | **complete** (x2), min rho 9e-4, 33 min |
| adiabatic MHD 64³ / 128³ | NaN 0.50 / 0.60 (face fix alone 0.36) | **complete** |
| isothermal hydro 128³ | NaN / dt collapse ~0.67 | CFL 1.5: NaN 3.4; **CFL 0.75: complete** |

The production recipe (prot + hard floors + `vacuum_rest` + rho_min 0.02),
same setup:

* 128³ native: aborted at 1.55;
* 128³ Pallas: one run completed (max|v| spiked to 12), an identical second
  run aborted at 4.6.

PP-WENO completed every MHD run and resolves physical voids an order of
magnitude below the recipe's floor. Isothermal hydro at Mach 10 evacuates
cells to 1e-9-1e-10, which is genuine for isothermal double rarefactions
(rho ~ exp(-dv/2c)). Their velocity is meaningless, so robustness there needs
the provable CFL.

Statistics (`turb_stats.py`, t > 2 t_c, 256³ PP-WENO as the reference):

| | PP-WENO 256³ | PP-WENO 128³ | recipe 128³ |
|---|---|---|---|
| sigma(ln rho) | 1.19 | 1.16 | 1.32 |
| min rho | 2.2e-3 | 2.4e-3 | 0.02 (floor) |

PP-WENO's density-PDF width converges from below. The recipe's 1.32 lies
*above* the 256³ value, which points at its floor wall and velocity spikes
rather than at physics. Near the grid scale (k >~ 100, the last factor ~3
below Nyquist) PP-WENO at 128³ has less density and kinetic power than the
recipe, and 256³ has more than both: the common splitting speed dissipates
slow magnetosonic modes more. Figure:
`out/stats_f_iso256_vs_f_iso128_vs_pstats_recipe128.png`.

### High resolution and Mach 20, every floor and clamp off

Setup: Pallas, CFL 1.5 unless noted, `clamp_in_estimates = False`,
rho_min = p_min = 1e-30. The last column counts cells with p <= 0, summed
over snapshots every 0.05 t_c.

| run | outcome | min rho | min p | p <= 0 |
|---|---|---|---|---|
| iso MHD M10 beta 0.1, 256³ | complete | 9.3e-4 | - | 0 |
| iso MHD M20 beta 0.1, 256³ | complete | 6.3e-4 | - | 0 |
| iso MHD M10 beta 1, 256³ | complete | 6.8e-4 | - | 0 |
| iso MHD M10 beta 0.1, **512³** (H100) | complete, 5.3 h | 6.5e-4 | - | 0 |
| adiabatic M10, 256³, old PP | complete | 4.7e-2 | -5.7e-4 | 67 |
| adiabatic M20, 256³, old PP | complete | 1.3e-2 | -6.1e-4 | 1024 |
| adiabatic M20, 128³, paired | complete, 208 s | 1.3e-2 | +3.5e-6 | 0 |
| adiabatic M10, 256³, paired | complete, 3327 s (old PP 1068 s) | 2.5e-2 | -1.1e-4 (t = 1.2) | 1 |
| adiabatic M20, 256³, paired | complete, 2869 s (old PP 1088 s) | 1.0e-2 | -2.9e-4 (t = 0.85) | 36 |
| adiabatic M20, 256³, paired, CFL 0.75 | complete, 5367 s | 1.6e-2 | -4.7e-4 (t = 0.8) | 7 |

**Isothermal MHD** never touches a floor, at any resolution or Mach number
tried; density is the only constraint and it is provable.

**Adiabatic MHD** with the paired limiter:

* the CFL estimate is unchanged, but 256³ wall time is ~3x the old PP's (probably the array-level
  paired recombination in the Pallas path: 17 output channels plus pair fractions on full arrays;
  not yet profiled);
* positive at 128³;
* at 256³, rare transient negative pressures in the cold early phase
  (t/t_c 0.8-1.2), even at CFL 0.75. That is ~30x fewer cells than the old PP.
  The cells recover (theta = 0 around them), and the wave-speed inputs stay
  real through the kernel's troubled-cell guard.

**Forensics (2026-10-06): the cause is the per-axis inflow pairs, not div B.**

Scripts: `negp_forensics.py`, `negp_decompose.py`, `inflow_census.py`. Each
negative event was stopped at its first occurrence (`turb.py
--stop-on-negative-p`) and stepped with the run's own numerics.

* CT keeps its own divergence (6th-order, on the faces) at round-off in every
  stage: 1e-5 of |B|/dx after 5 t_c. The fluid update sees a different one,
  the central difference of the interpolated cell-centred B: median
  0.06-0.09 % of |B|/dx, locally up to tens of %.
* The CFL 0.375 control (min p -7.7e-4) and the x64 control (-9.0e-4) also go
  negative. So it is neither the time step nor round-off.
* At the CFL 0.375 event, one forward-Euler step at C_FE = 0.25 is already
  negative. Rebuilt exactly (1e-16) from the decomposition:
  * the axis-summed own pairs are admissible;
  * the axis-summed inflow pairs are not (-1.05e-4). The x and y inflow
    pairs have inadmissible first-order bases (theta = 0), and the z pair
    spends the margin with theta ~ 1.
* Over the whole 256³ failing states, the per-axis first-order inflow bases
  are inadmissible in ~7 % of the cells. B_n varies along each axis, and that
  variation is mostly divergence-free, cancelling only across axes. The
  **axis-summed** first-order inflow is admissible in **every** cell, despite
  the nonzero central divergence.
* The Godunov-Powell counterfactual also "fixes" the event cell, but it adds
  more internal energy than the deficit there. It does not discriminate.

**Fix (9616fdb): limit each cell's inflow faces jointly over the axes.**

* A pre-pass (`mhd_inflow_reference`) builds each cell's axis-summed
  first-order inflow B_i from q, the physical fluxes and the face speeds. It
  needs no reconstruction and matches the flux's speeds to 8e-16.
* Each inflow face is bounded by theta_k <= frac(B_i, 2 dim s_k), a convex
  equal split. The own pairs stay per axis.
* No source term is added; the scheme stays conservative.
* Native == Pallas to 1.9e-10.

Results:

* Both forensic events now stay positive, in the full step and in every
  forward-Euler step:
  * CFL 0.375: -6.9e-6 -> +5.3e-5;
  * CFL 1.5: -3.4e-6 -> +8.9e-5.
* Alfven orders and errors are unchanged.
* The low-beta blast's min p rises from 2.9e-2 to 5.2e-2.

256³, Mach 20 unless noted, adiabatic, no clamp, floors 1e-30, every snapshot
checked:

| run | per-axis pairs | joint inflow |
|---|---|---|
| M20, CFL 0.375, to 1.5 t_c | min p -7.7e-4, 17 hits | **+1.3e-6, 0 hits** (complete, 1750 s) |
| M20, CFL 1.5 | -2.9e-4, 36 hits | **+1.2e-6, 0 hits** through 3.85 t_c |
| M10, CFL 1.5 | -1.1e-4, 1 hit | **+3.0e-5, 0 hits** through 3.1 t_c |

### Cold Evrard collapse (e0 = 0.05, fourth-order conservative gravity, fp32)

| N | bare | face state | PP-WENO |
|---|---|---|---|
| 32 | NaN t = 0.10 (fp64 too) | NaN 0.22 | **complete**, dE/E 1.2e-5 (fp64: 8e-8) |
| 64 | NaN t = 0.25 | — | **complete**, dE/E 3e-5 (min p -0.015, coupling) |
| 128 | complete, dE/E 7e-5 | — | complete, dE/E 7e-5 |

## Defect 3 (not WENO): the conservative gravity coupling

Even completed runs keep cells with `p < 0`:

* 32³: min p = -0.05 at r ~ 0.45 late in the infall;
* 64³: -0.015;
* 128³: -6e-4.

These cells' *total* energy is below their kinetic energy. The coupling charges
the work `F_rho (phi_face - phi_cell)` done on mass *in transit* to the
receiving cell. That mass's momentum is never decelerated, so the work comes
out of internal energy, at first order already. The transported gas can
afford the climb across half a cell only if

    e / rho  >~  |g| dx / 2,

i.e. **the grid must resolve the pressure scale height**. For cold Evrard
(e0 = 0.05, g ~ 1 at the edge) the half-cell climb is:

* 0.0625 at 32³ (criterion violated);
* 0.031 at 64³ (marginal);
* 0.016 at 128³ (satisfied).

This is why the conservative schemes were known to NaN below 128³, and why
warm Evrard (e0 = 0.2) is fine at 32³. Late in the collapse g ~ 5 at r ~ 0.45,
which violates the criterion again at 64³, exactly where the holes are.

### A principled conservative fix: flux-corrected gravitational work

Every conservative energy coupling is a choice of **potential-energy flux**
`q` at the faces:

    S_E,i = -(1/dx) sum_axes [ (q - F phi_i)_{i+1/2} - (q - F phi_i)_{i-1/2} ].

The face's total work is always `-F (phi_R - phi_L)`, so total energy is
conserved for *any* `q`. The scheme's fourth-order `q ~ F phi_face` (plus its
correction) charges half the climb to each side. **Donor accounting is exactly
`q = F phi_downwind`**: the donor pays the whole climb, at first order.

`GravityConfig.work_flux_correction` blends the two face by face with
flux-corrected transport:

    q = q_low + psi (q_high - q_low).

* **Conservation:** exact for every `psi`.
* **Order:** high wherever `psi = 1`.
* **Limiter:** `psi` comes from Zalesak budgets on the internal-energy loss
  *rate* (half the internal energy per wave-crossing time `dx / (|v| + c)`).
  `psi` therefore depends on the state only, never on `dt`.

**The residual no conservative split can fix.** The remaining drain is
`D = W - v.(rho a) ~ (F - m) . g`. Here `F - m` is the Lax-Friedrichs mass
diffusion at unresolved density gradients. In the failing cells (cloud
surface, density x4.5 per cell, cold infall) the face mass fluxes point
outward while the gas falls inward. `GravityConfig.limit_internal_energy_work`
is a **non-conservative** backstop for exactly this. It uses the same
dt-independent rate budget and is confined to those cells.

Cold Evrard (e0 = 0.05, PP-WENO hydro, fp32), dE/E and pressure:

| N | conservative | + flux-corrected work | + backstop | KE-only source |
|---|---|---|---|---|
| 32 | 1.2e-5, min p -0.046 (~1000 cells from t = 0.05) | **1.0e-5**, no p < 0 before t ~ 0.5, then <= 168 cells at -2e-6 | 3 %, p > 0 at every snapshot | 70 % (default WENO) |
| 64 | 3e-5, min p -0.015 | **2.9e-5, p > 0** at the end | 3e-4, p > 0 | 41 % |
| 128 | 7e-5, min p -6e-4 | **6.8e-5, p > 0** | 7.0e-5 (idle) | 20 % |

Tried and dropped: crediting the hydrodynamic stage heating in the budgets.
With the flux correction it let through splits whose heating did not
materialise (min p -5e-4). In the backstop, debiting expansion cooling made
it reject 10x more energy at 64³.

### Temporal convergence of conservation (fixed dt)

Mild Evrard (e0 = 0.2), 32³, float64, fourth-order coupling, dE/E for
250 / 500 / 1000 / 2000 / 4000 steps (`evrard_dt.py`). The repository's
`evrard_timestep_convergence` setup:

| scheme | dE/E | observed orders |
|---|---|---|
| legacy | 3.2e-9 ... 2.5e-13 | 2.8, 3.5, 5.0 (then round-off) |
| face state (default) | 4.8e-9 ... 1.0e-13 | 2.6, 3.1, 3.7, 6.0 |
| face + flux-corrected work | 5.1e-9 ... 5.5e-14 | 2.6, 3.2, 3.8, 6.9 |
| PP-WENO | 2.4e-9 ... 8.0e-13 | 2.3, 2.6, 2.9, 3.8 |
| PP + flux-corrected work | 4.0e-9 ... 1.1e-12 | 2.4, 2.7, 3.0, 3.7 |
| PP + flux-corrected + backstop | 4.7e-5 at every dt | plateau |
| first (per-stage, dt-dependent) limiter | 2.1e-3 at every dt | plateau |

What the table shows:

* **Flux correction:** the dt-independent flux correction leaves temporal
  convergence untouched.
* **PP-WENO:** its theta and speeds are also dt-independent. The scheme still
  conserves energy exactly in the dt -> 0 limit, but at ~3rd rather than ~4th
  order: theta's clipping makes the right-hand side only Lipschitz, which
  costs RK4 order at switching events.
* **Backstops:** any non-conservative backstop plateaus by construction. Use
  one only where strict positivity at unresolved resolution matters more than
  conservation.

## What is actually active in a "no stabiliser" MHD run

The audit traced every floor, clamp and limiter on the FD MHD path: Pallas,
SSPRK4 + CT, OU forcing, PP on, `clamp_in_estimates = False`, rho_min = p_min = 1e-30.

**Necessary, and active:**

1. `weno_positivity_preserving`. It implies the admissible face state, which
   is the default anyway. For ideal MHD it uses the paired scalings and turns
   on the stage re-sync of the cell-centred B.
2. Constrained transport (div B = 0).
3. The CFL condition:
   * provable at C_cfl <= 0.75;
   * MHD turbulence runs at 1.5 without incident;
   * isothermal hydro at Mach 10 needs 0.75.

**Always on, but not stabilisers:**

* the CT energy re-sync (ideal gas): holds p when B is rebuilt from the faces;
  total energy then moves by the B_WENO / B_faces mismatch;
* eigensystem guards, all read-only:
  * face density >= rho_min;
  * sqrt floors of 1e-12 to 1e-30;
  * the 1/sqrt(2) tangent when B_t -> 0;
  * fast/slow weights when lambda_f = lambda_s;
* the Pallas kernel's "troubled cell" floor on its *wave-speed inputs*
  (rho, p >= rho_min, p_min). At 1e-30 it acts only on rho <= 0 or p <= 0,
  which never occurred in the paired runs;
* the dt-collapse abort (control flow only).

**Off and not needed** (0 floor hits in every run below):

* `clamp_in_estimates`, rho_min / p_min floors;
* `prot`, hard floors, `vacuum_rest`;
* the deep-void blend, the Zalesak `preserving_flux`;
* dual energy.

**Caveat on the defaults:** `clamp_in_estimates = True` is the default. It is
documented as read-only but is not:

* For ideal MHD the step-end primitive recovery writes rho_min / p_min into
  the carried state. The momentum is rebuilt from the floored density, so it
  is amplified by rho_min/rho when 0 < rho < rho_min and flips sign when
  rho < 0.
* For isothermal MHD it divides by the floored density and damps the momentum.

**Other audit findings (not fixed):**

* the Pallas ideal-MHD kernel hard-codes WENO eps = 1e-7 and Jiang-Shu
  weights;
* the forcing kick is applied after dt is computed;
* `dt_max` is not applied with the snapshot callback;
* Pallas kernels silently fall back to native when a grid extent is not
  divisible by the block shape.

## Phase-out of the patchwork (5f23a0a and the migration commit)

**Removed** (FD; superseded by `weno_positivity_preserving`):
* per-stage positivity in all its modes (HARD_FLOOR / REDISTRIBUTE /
  CONSERVATIVE, native + Pallas) and `per_stage_specific_floor`;
* `vacuum_rest`, `nan_safe`, `cons_*`, `default_positivity_protection`,
  `positivity_max_velocity`;
* the forcing `prot` (`vacuum_protection` + params);
* the deep-void blend, the Zalesak `preserving_flux` and the HLLC fallback;
* the gravity work backstop.

**Kept**, none of them a positivity patch:
* the per-step HARD_FLOOR: the finite-volume solver's floor and the
  temperature floor of cooled runs;
* the read-only `clamp_in_estimates` (fixed: it no longer writes into the
  carried MHD state);
* dual energy (accuracy);
* the conservative `work_flux_correction`.

**Validation of the switch:**
* Showcase recipe vs PP-WENO only, 128³:
  * cooled SNR: final T min 1.9 K -> 6.6e3 K, the same energy;
  * Cas A: total energy within 0.01 %.
* Smoke runs at 64³, all complete and finite:
  * cooled SNR, Cas A, Cas A with cooling + tfloor;
  * the TI phase;
  * paper turbulence, iso and adiabatic (M10);
  * both self-gravity examples (the Evrard energy budget is unchanged).
* Asserting suite: 25 passed.
* Gradients (`pp_gradient_check.py`): reverse = forward = finite difference
  through PP for hydro, ideal and isothermal MHD.

## Costs

* Accuracy (smooth advected waves, x64, `smooth1d.py` / `smooth2d.py`): all
  5th order.
  * Shear (vortical) waves: unchanged.
  * Entropy waves (contacts) carry mass and use the common speed, so their
    error grows by (|v|+c)/|v|: 2.3x at Mach 0.77, 14x at Mach 0.08. That is
    ~1.2x / 1.7x in linear resolution. Nothing at high Mach.
* Runtime:
  * native, one x-flux, 64³, A100: ~1.2x for isothermal MHD, ~2.3x for ideal
    MHD (shared GPU, noisy);
  * Pallas, end to end: 128³ isothermal MHD in 133-136 s, *faster* than the
    recipe's 179 s, because no velocity spikes collapse dt.
  * Ideal MHD with the joint inflow limiter (`profile_pp.py`, 256³ fp32 A100,
    Mach-20 turbulence state, component medians):

    | | no PP | array recombination | fused kernels (69c1218+) |
    |---|---|---|---|
    | SSPRK(5,4)+CT step | 160 ms | 453 ms | **240 ms** (1.50x) |
    | WENO kernel per sweep | 12.5 ms | 13.9 | 14.0 |
    | recombination per sweep | - | 16 | ~2.5 (near its bandwidth floor) |
    | inflow reference per stage | - | 20 | 4.2 |
    | temporaries | 3.3 GB | 12.4 GB | 5.0 GB |

    The remaining 1.5x is structural: the 17-channel split output, one
    recombination pass per sweep and one reference pass per stage.

    End to end (256³ Mach 20, 5 t_c, A100), each run on its own limiter
    version:

    | limiter | wall time |
    |---|---|
    | first in-kernel per-face PP (first order at low beta, p < 0) | 1088 s |
    | joint, array recombination | 2772 s |
    | **joint, fused kernels** | **1570 s** |

    The fused run has the same min p over the run as the array run (+1.23e-6)
    and 0 floor hits.

## Is there a deeper result? One principle, two open assumptions

Every defect found here broke the same property: **each forward-Euler stage
must be a convex combination of admissible states** (invariant-domain
preservation). In order of discovery:

* the face basis was built from a non-state;
* per-field splitting speeds made the frozen-basis splitting non-monotone;
* unlimited WENO face values left the domain;
* the normal-B flux was zeroed, so the update was no longer the combination;
* stage states were not re-synced;
* the register-form SSPRK was non-convex under that re-sync;
* the per-axis inflow pairs used the wrong decomposition in multi-D;
* (gravity) the conservative work flux was not a convex split.

Each fix restores the property, and nothing else is needed.

With the joint limiter, two assumptions remain unenforced. Both were measured
on the hardest state (`invariant_domain_audit.py`: 256³ Mach 20, t = 0.70 t_c,
just before the old failure):

1. **The axis-summed first-order inflow B_i is admissible.**
   * It keeps at least **57 %** of the cell's pressure in every cell (71 %
     at the 1e-5 quantile).
   * A guarantee for all states needs Wu & Shu's div-B condition; the data
     say it is far from binding.
2. **The local Courant condition lambda_FE sum_d S_d,i <= 1** of the stiffest
   stage, with the stage's own speeds.
   * This is the sharp positivity time step,
     dt = 1.508 dx / max_i sum_d S_d,i. It is per cell; the code's CFL uses a
     sum over axes of global maxima.
   * In the code's units the sharp bound is C = 0.88 (isothermal MHD), 0.99
     (adiabatic MHD) and 1.05 (isothermal hydro), against the generic 0.754.
   * At the nominal C = 1.5 the local condition fails in 3.7 % (adiabatic)
     to 19 % (isothermal MHD) of the cells, including low-density ones, and
     the runs survive. So the bound is worst-case, not tight.

Things that are not hiding:

* **A minimum-entropy principle.** After one step, 0.34 % of cells undershoot
  the local minimum specific entropy (worst 0.84x). The lowest-pressure
  cells do not (1.01x). Positivity is the binding constraint where it
  matters.
* **Isothermal hydro at Mach 10, CFL 0.9.** It fails although, at the start
  of the step before its NaN, density is positive (min 6.7e-8) and the local
  Courant number is <= 0.86 everywhere. So the failure comes from inside the
  steps that follow. The likely cause is near-vacuum signal speeds growing with
  v = m/rho across the stages, which would break assumption 2 with stage
  speeds. This is NOT verified by stepping. A stage-level Courant check with
  step rejection would close it; open.

## Side findings (fixed)

* 1D finite difference converged at **first order** on periodic problems:
  1D kept the default 2 ghost cells, short of the WENO5 stencil. Fully
  periodic 1D runs now use `PERIODIC_ROLL`, and other 1D runs get >= 4 ghost
  cells, as in 2D/3D. Measured orders are now 4.93 / 5.00 / 5.00.
* Isothermal hydro crashed in 1D (`momentum_index.x` on an int). It now runs
  and matches the exact isothermal Riemann solutions.

## Open

* A cheap, rigorous per-field monotonicity bound would remove the low-Mach
  contact cost.
* The gravity coupling (above).
* Ideal MHD in multi-D: the joint inflow limiter is provable wherever the
  axis-summed first-order inflow B_i is admissible (all cells observed). Where
  it is not, only a div-B-consistent (Godunov-Powell) source would restore the
  guarantee.
* Costs of the Pallas paired path, both untested in production:
  * it writes 17 channels and recombines on arrays: more memory;
  * its shard halo is 4, never run on multiple GPUs.

## Reproduce

All scripts take `WENO_VARIANT=baseline|face|pp`. GPU jobs go through
`examples/scripts/validation/weno_stability/run.sh` and `pq`. Large outputs go to
`/export/data/lstorcks/weno_stability`.

    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/riemann1d.py
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/smooth2d.py shear 0.1
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/evrard.py --n 32
    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/pallas_equivalence_mhd.py
    WENO_VARIANT=pp pq sub -t a100 -n 1 --name t128 -- examples/scripts/validation/weno_stability/run.sh examples/scripts/validation/weno_stability/turb.py --N 128 --backend pallas --tag pp128
    python -m pytest pytests/hydrodynamics/test_positivity_preserving_weno.py pytests/mhd/test_positivity_preserving_mhd.py
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/alfven_pp.py --p0 0.01
    PYTHONPATH=. JAX_PLATFORMS=cpu WENO_VARIANT=pp python examples/scripts/validation/weno_stability/mhd_blast.py --n 100
    python examples/scripts/validation/weno_stability/wu_pair_check.py
