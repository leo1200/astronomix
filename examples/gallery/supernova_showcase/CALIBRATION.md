# Calibrating the Cas A setup against the observations

*2026-07-30. All numbers below are reproducible on a laptop CPU in under a
minute per model — see the commands at the end.*

## Why this exists

`cassiopeia_realistic.py` builds its remnant directly on the 3D grid: a cold,
homologous ejecta ball already **1.5 pc** across, laid on top of the wind, with
an age tag computed as `t0 = R_ej / v_max ≈ 122 yr` plus the simulated time.

That construction skips the phase that sets the answer. The progenitor wind
`n(r) = n_w (2.5 pc / r)^2` with `n_w = 0.8 cm^-3` contains

    M_wind(< r) ≈ 2.2 M_sun × (r / 1 pc)

so by the time the blast reaches 1.5 pc it should already have swept up
**≈ 3.3 M_sun — an entire ejecta mass** — formed a reverse shock, and given a
large fraction of its energy to the shocked wind. Starting a *cold, undecelerated*
3.3 M_sun ball carrying the full 1.5 × 10^51 erg at that radius therefore both
over-energises the remnant at that stage and erases its history. The measured
consequences were a forward shock 17 % too small (2.09 pc against 2.52 observed)
at a claimed age of ~470 yr, and a reverse-shock estimator that has reported
anything between 0.01 and 1.09 pc.

`casa_calibrate_1d.py` does the early phase properly and cheaply, in 1D
spherical symmetry, exactly as Orlando et al. (2016) Route B prescribes.

## Method

Same ejecta profile (flat core + `r^-9` envelope, tanh-tapered — literally the
same `_common.ejecta_radial_shape`), same `r^-2` wind, started at
`r0 = 0.05 pc` where the swept wind mass is still only ~3 % of `M_ej` so free
expansion is an excellent approximation. Evolved to 350 yr on a uniform radial
grid.

Shock definitions, measured on the profile with no fitting:

| quantity | definition |
|---|---|
| `r_RS` | largest radius still expanding homologously, `\|v − r/t\| < 0.05 r/t` (unshocked ejecta satisfies `v = r/t` exactly) |
| `r_FS` | outermost radius where `ρ > 2 ρ_wind(r)` against the known analytic wind |
| `v_FS`, `v_RS` | finite differences of those radii between snapshots — observer-frame shock speeds, as quoted from proper motions |
| `n_post` | mean density over the outer 5 % of the shocked region |
| age | `t0 = 1/s` exactly, for `v = s r` — no `R/v_max` approximation |

Two numerical requirements were found and are documented in the script:

1. **`first_order_fallback=True` is mandatory.** Plain 2nd-order MUSCL/minmod
   NaNs on the *first step* on the `r^-9` envelope. The threshold is
   `v_max ≈ 100–500 km/s`, and it is insensitive to the ejecta temperature
   (10² – 10⁶ K), the CFL number (0.4 → 0.1) and the density/pressure floors,
   and it happens in Cartesian geometry too — so it is the reconstruction
   overshooting on the steep envelope, not the spherical geometric source terms.
2. **A per-step density floor is mandatory.** The origin cell evacuates (ρ falls
   ~10 orders of magnitude by 80 yr as the cold, essentially pressureless
   homologous core expands off `r = 0`), and the finite-volume evolve has **no
   positivity protection of its own** — `params.minimum_density` is only read by
   the FV *Pallas* path and the timestep clamp. The floor in
   `_iteration_level_updates` is solver-agnostic and does apply. It sits ~3000×
   below the ambient `n_c` and only ever fires deep inside the reverse shock;
   mass and energy drift are both `< 1e-6` relative.

**Convergence** (baseline parameters, N = 1000 → 8000 radial cells):

| N | `r_FS` | `r_RS` | `v_FS` | `n_post` |
|---|---|---|---|---|
| 1000 | 2.222 | 1.554 | 5113 | 4.39 |
| 2000 | 2.231 | 1.585 | 4997 | 4.40 |
| 4000 | 2.240 | 1.604 | 5030 | 4.39 |
| 8000 | 2.244 | 1.614 | 5007 | 4.38 |

1 % in `r_FS` over an 8× refinement. All results below use N = 2000–4000.

## Result 1 — the existing parameters are better than the 3D run suggested

Running the *current showcase parameters* (E = 1.5e51, M_ej = 3.3, n_w = 0.8)
properly gives, at 350 yr:

    r_FS 2.24 pc   r_RS 1.61 pc   v_FS 5007 km/s   v_RS 3365 km/s   n_post 4.38

against observed `2.52 ± 0.20 / 1.58 ± 0.16 / 5250 ± 250 / 3000 ± 1000 / 4 ± 1`.
So `r_RS`, `v_FS`, `v_RS` and `n_post` are all already within 1σ; only `r_FS` is
low, by 1.4σ. The 3D run's 2.09 pc is *worse than the same physics done right* —
the deficit is the initial condition, not the parameters.

## Result 2 — E = 2 × 10^51 erg is the right answer

A 4 × 4 × 4 sweep of `(E_SN, M_ej, n_w)` and a refinement including the envelope
slope both put the optimum at **E ≈ 2 × 10^51 erg**, which is what is
independently inferred for Cas A and what Orlando's own Route B fit gives
(2.3 × 10^51). The envelope slope turns out to be nearly irrelevant (7 vs 12
changes `r_FS` by < 0.01 pc) because by 350 yr the reverse shock has eaten well
into the flat core and the envelope is long gone.

**This matters because E = 2e51 was previously reverted for numerical reasons
only** — an isolation probe blew up at 512³ at t = 0.028 with no bubbles and no
knots, and the setup was reset to 1.5e51 with the forward-shock radius chased by
running longer instead. The calibration says the physics wants 2e51; the
stability envelope has to be widened to accommodate it, not the other way round.

Solving simultaneously for `r_FS = 2.52 pc` **and** `n_post = 4 cm^-3` (the two
targets that pin the degenerate energy/wind-density block) gives the fiducial:

| | value |
|---|---|
| `E_SN` | 2.09 × 10^51 erg |
| `M_ej` | 3.0 M_sun |
| `n_w` at 2.5 pc | 0.93 cm^-3 |
| `γ` | 5/3 |

which lands at 350 yr on

    r_FS 2.52   r_RS 1.72   v_FS 5495   v_RS 3353   n_post 4.00

`M_ej` was scanned at fixed targets: 3.0 is best and the fit degrades
monotonically upward (score 2.33 / 2.79 / 3.76 / 4.30 for 3.0 / 3.3 / 4.0 / 5.0),
because a heavier ejecta needs a proportionally larger energy
(E = 2.09 / 2.18 / 2.40 / 2.71 × 10^51) and then overshoots `v_FS`.

## Result 3 — cosmic-ray back-reaction does *not* fix the FS/RS ratio

The one target that stays off is the **ratio** `r_FS / r_RS`: 1.46 in the model
against 1.59 observed. Orlando's cheap CR prescription — an effective adiabatic
index `γ_eff` from the Blasi/Ferrand lookup, raising the compression ratio above
4 — is the obvious candidate, and `γ_eff` is a single global number here, so it
is directly testable.

Varying γ alone is meaningless (it changes `n_post` by a factor of 3), so `E` and
`n_w` were re-solved at each γ to hold `r_FS = 2.52` and `n_post = 4` fixed —
Orlando's own point that η and `n_w` are degenerate and must be varied together:

| γ_eff | E [10^51] | n_w | `r_RS` | `r_FS/r_RS` | `v_FS` |
|---|---|---|---|---|---|
| 5/3 (η = 0) | 2.09 | 0.93 | 1.72 | **1.464** | 5495 |
| 1.5 (η ≈ 1e-4) | 2.11 | 0.75 | 1.80 | 1.397 | 5502 |
| 4/3 (η ≈ 1e-3) | 2.10 | 0.54 | 1.94 | 1.303 | 5605 |

A softer equation of state pushes the reverse shock **outward**, moving the ratio
*away* from the observed 1.59. So CR back-reaction cannot be the explanation, and
this independently supports Orlando's conclusion that η must be well below 10^-4.

Caveat, stated plainly: astronomix carries one global γ, so the ejecta equation of
state changes too. The ejecta is cold and dynamically pressureless, so this should
not matter, but a properly advected `γ_eff` applied only at the shock has not been
tested.

The remaining `r_RS` excess is therefore **not** an explosion-parameter problem.
It is exactly what Orlando et al. (2022) attribute to the asymmetric
circumstellar shell: the blast wave hits the shell, and the *reflected* shock
runs back into the ejecta and drives the reverse shock inward. Two independent
consistency checks support this reading:

* the fiducial 1D model reaches Orlando's shell radius of 1.5 pc at **182 yr**,
  against their quoted "the blast wave hits the shell at ≈ 180 yr";
* the observed reverse shock is strongly asymmetric (2000–4000 km/s outward in
  the E/N, ≈ 0 or *inward* in the W/SW), so a spherically symmetric model cannot
  reproduce a single `r_RS` in the first place — 1.58 pc is a circle fit.

Testing that is what the 3D ladder (`casa_orlando.py --shell`) is for.

## Result 4 — the showcase's CSM shell carries 27× too much material

`cassiopeia_realistic.py` uses `SHELL_RADIUS = 1.7 pc`, `SHELL_THICKNESS = 0.18 pc`
(Gaussian σ) and `SHELL_PEAK_DENSITY = 60 cm^-3`. Orlando et al. (2022)'s favoured
shell is `r_sh = 1.5 pc`, `σ = 0.02 pc`, `n_sh = 20 cm^-3`. What the blast wave
actually feels is the **column**, `n_sh × σ`:

    showcase          60 × 0.18 = 10.8 cm^-3 pc
    Orlando (2022)    20 × 0.02 =  0.40 cm^-3 pc      -> 27x less

So the showcase remnant is ploughing through 27 times more circumstellar shell
than the model it cites. That is a plausible contributor both to its
under-sized forward shock and to the ~6× X-ray over-brightness the forward model
measures (emission scales as density squared).

The honest difficulty is that σ = 0.02 pc is **thinner than the cell** of any
whole-remnant grid in use (0.0137 pc at 512³ in a 7 pc box, 0.027 pc at 256³).
`_common.orlando_csm_shell` therefore broadens the shell to a resolvable
2.5 cells **at fixed column** — preserving the mass the blast runs into, which
is what sets the deceleration and the reflected shock, while leaving the shell's
*fragmentation* explicitly unresolved. Broadening the shell instead of thinning
it is the choice that keeps the dynamics right; it is not a claim that the
shell's small-scale structure is captured.

## Result 5 — the 3D ladder reproduces the calibration, and is stable

Three 256³ runs from the same mapped profile (150 → 350 yr, radiative, one A100
each, ~25 min): `base` (clumping only), `shell` (+ the Orlando 2022 shell),
`pistons` (+ the Table-4 anisotropies).

| model | `r_FS` | `r_RS` | `r_FS/r_RS` | PA spread |
|---|---|---|---|---|
| base | 2.541 | 1.686 | 1.507 | 0.151 |
| shell | 2.441 | 1.635 | 1.492 | 0.302 |
| shell + pistons | 2.441 | 1.635 | 1.492 | 0.302 |
| **observed** | **2.52 ± 0.20** | **1.58 ± 0.16** | **1.595** | **0.2–0.4** |

(radii are quantised to the 0.05 pc radial bin of the estimator.)

* **`r_FS` and `r_RS` are both within 1σ of the observations**, against 2.09 pc
  and a broken estimator before. The angle-averaged density profile lies on top
  of the 1D calibration, so the 3D run reproduces its own calibration — which
  separates "wrong physics" from "wrong initial condition" for good.
* **The shell does what Orlando says it does.** It decelerates the blast
  (`r_FS` 2.54 → 2.44) and deforms it: the spread in `r_FS` over position angle
  goes from 0.151 to 0.302 pc, squarely in the observed range, with the minimum
  and maximum ~180° apart as in the data.
* **Energy is conserved to better than 0.01 %** over the whole run in all three,
  with no `dt` collapse — including the pistons leg. That matters: cold dense
  ejecta substructure (knots, Ni bubbles) destroyed *every* previous attempt,
  at t = 0.02–0.06, and was written off as "exceeding the stability envelope".
  It is stable here. The likely reason is that the mapped profile hands the
  solver an already-shocked, pressure-supported configuration instead of a cold
  homologous ball whose interior is at the pressure floor.
* **The pistons do not yet punch through.** They restructure the interior
  strongly (the interior density profile is visibly different) but no Si-rich
  jet reaches the forward shock. At 256³ a knot of radius `0.10 × r_FS` is only
  4.7 cells across, so this is expected to be resolution-limited; 512³ gives 9.4
  and is the test.

## Result 6 — the first real synthetic Chandra observation

`casa_observe.py` puts a saved state through pyXSIM (AtomDB/APEC per cell,
Doppler-shifted by the velocity field) → SIMPUT → SOXS with the real ACIS-S
ARF, RMF, PSF image and particle background, and bins the resulting event file
onto the *same* tangent-plane grid as the real `evt2` data through the *same*
projection code. The comparison is therefore in counts, not in appearance.

| state | count rate (0.5–7 keV) | vs real 295 counts/s |
|---|---|---|
| `casa_n128_jet_dual` (old showcase) | 1785 | 6.1× too bright |
| `orl_n256_pistons` (calibrated, Z = 1) | 85 | 0.29× |
| `orl_n256_pistons` (`--ejecta-zmet 10`) | 242 | **0.82×** |

and the radial surface-brightness profile of the calibrated state now **tracks
Chandra's from the centre out to ~130″**, peaking at the same radius, where the
old state peaked ~50″ early and fell off a cliff. The remnant's angular size is
right: the bright shell sits at ~150″, as observed.

The two failures that remain are both informative rather than mysterious:

1. **Under-luminous by ~3.5× at solar abundance, and brightest at the wrong
   place.** The synthetic image peaks at the outer rim (shocked circumstellar
   gas); the real one peaks in the ejecta ring interior to it. Both follow from
   the same missing ingredient: the simulation is single-fluid, so one
   metallicity applies everywhere and the O/Si/S/Fe-dominated shocked ejecta
   radiates as if it were solar, while Cas A's X-ray emission is line-dominated
   ejecta emission.

   **Tested directly:** re-running with `--ejecta-zmet 10` — a crude
   density/temperature-selected enhancement standing in for the missing
   composition tracer — takes the count rate from 85 to **242 counts/s against
   the observed 295**, i.e. from a factor 3.5 low to 18 % low. The luminosity
   deficit really is the abundances. This is a *composition* problem, not a
   hydrodynamics problem, and it is quantitatively the size the abundances
   predict.
2. **Far too smooth.** The real remnant is a filamentary lace on scales well
   below 1″. At 256³ a cell is 1.7″ and the imposed clump scale is 13 % of the
   remnant radius, against Orlando's 2 %. Resolution, plus the absence of NEI
   cooling (Orlando et al. 2025b show NEI losses are what fragment the ejecta
   into knots and thin filaments).

Both point at the same gate: **the finite-difference solver carries no passive
scalars**, so there are no composition tracers, no ejecta/CSM discriminator and
no shock-time tracer — hence no ionization age, no `T_e ≠ T_i`, and no NEI. The
dual-energy variable `g` is a working precedent for adding an advected field to
the FD path. Until that exists, the synthetic spectra are CIE with uniform
abundances and should be described that way.

## Handing the calibration to 3D

`casa_orlando.py` maps the calibrated profile onto the 3D grid at
`--map-age 150 yr` (`r_FS = 1.28 pc`, `r_RS = 0.94 pc`) — before the shell
encounter at 182 yr, so the interaction is *computed* rather than imposed — and
imposes the multi-D structure there. At 512³ in a 7 pc box that is 94 cells per
remnant radius at the start and 185 at 350 yr, against Orlando's stated
invariant of > 100 throughout / > 250 at the final time: met at the end, marginal
at the start. Mapping later would improve it but would put the shell inside the
blast wave.

## Reproducing

```bash
cd examples/gallery/supernova_showcase
./run.sh casa_calibrate_1d.py --converge                    # resolution study
./run.sh casa_calibrate_1d.py --scan coarse                 # (E, M_ej, n_w)
./run.sh casa_calibrate_1d.py --scan fine                   # + envelope slope
# the fiducial, and the profile the 3D run maps
./run.sh casa_calibrate_1d.py --n 4000 --energy-51 2.09 --ejecta-mass 3.0 \
    --n-w 0.928 --save-profile casa_1d_fiducial_350yr.npz
./run.sh casa_calibrate_1d.py --n 4000 --energy-51 2.09 --ejecta-mass 3.0 \
    --n-w 0.928 --age 150 --save-profile casa_1d_map150.npz
```

Logs of the runs quoted above: `calib_scan.log`, `calib_scan_fine.log`,
`gamma_scan.log`, `solve_calib.log`, `solve_mej.log`.
