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


## Result 7 — the unshocked ejecta mass, and an inner-slope caveat

The calibration above fixed the shock radii and speeds but said nothing about how
far the reverse shock has eaten into the ejecta **in mass**. The 3D runs then came
out with 96 % of the ejecta shocked against the ~87–90 % observed, which shows up
twice over: shocked Si+S 3× too high (0.24 against 0.08 M☉), and a reverse shock
0.35 pc too far in. One cause, two symptoms.

`m_unshocked` is now a calibration target (0.35 ± 0.10 M☉; DeLaney et al. 2014,
Hwang & Laming 2012), and the parameter that controls it is the **inner** ejecta
density index δ (`ρ ∝ r^-δ` inside the core radius). A flat core has
`M(<r) ∝ r³`, i.e. almost no mass at low velocity, so the reverse shock reaches
the centre in mass long before it does in radius:

| δ | `M_unshocked` | `r_FS` | score |
|---|---|---|---|
| 0.0 (flat) | 0.096 | 2.519 | 4.69 |
| 0.5 | 0.186 | 2.531 | 4.01 |
| **1.0** | **0.321** | **2.549** | **2.81** |
| 1.5 | 0.527 | 2.579 | 4.37 |
| 2.0 | 0.853 | 2.637 | 8.48 |

δ = 1 lands on target while *improving* `r_FS`, and is the standard
core-collapse value.

Two caveats, both load-bearing:

* `ejecta_radial_shape` clipped the profile to a maximum of 1. That is harmless
  for a flat core, where the shape never exceeds 1 anyway, but it silently
  flattened any central peak — so `inner_slope` was a **no-op** until the clip
  was restricted to a lower bound. Any result computed before that fix is a
  δ = 0 result whatever the flag said.
* **δ = 1 has not yet been shown to work in 3D.** The 256³ run with pistons,
  shell and cooling aborts on a timestep collapse at t = 0.019. The likely
  interaction is that a centrally peaked core puts the Fe pistons
  (`D_knot = 0.15`) inside denser material and pushes the contrast over the
  crush threshold — the same mechanism that killed the tabulated-radius and
  capped-radius piston variants. Until that is bisected, **δ = 0 is the working
  configuration** and the Si / `r_RS` discrepancy stands.

## Result 8 — the plasma the X-rays actually see

Results 1–7 are hydrodynamics: radii, masses, deceleration. What a telescope
records is set by three quantities the solver does *not* evolve — the electron
temperature, the electron density, and the ionization state — and every one of
them depends on the composition. Reconstructing them (`_plasma.py`, `_nei.py`,
shared by `casa_plasma.py` and `casa_observe.py`) changes the predicted
observation by factors, not percents.

### The mean molecular weight is not 0.61

Fully ionized cosmic gas has `mu = 0.61` and `mu_e = 1.18`. The fully ionized
oxygen layer that carries most of Cas A's ejecta mass has `mu = 16/9 = 1.78` and
`mu_e = 2.0`. Three consequences, all measured on `orl_n256_final`:

| quantity | cosmic `mu` | composition-aware | note |
|---|---|---|---|
| EM-weighted single-fluid `kT` | 4.5 keV | **13.0 keV** | `= (3/16) mu m_p v^2` at a ~1900 km/s reverse shock |
| EM-weighted `n_e t` | 3.3e11 | **2.3e11** | against ~1e11 observed — 1.7x fewer electrons per gram |
| EM-weighted `kT_e` | 2.60 keV | **3.05 keV** | see below |

The 13 keV is not an error: heavy ions at a fast shock really are heated to tens
of keV, and it is the *electrons* that are cold.

### Electron-ion equilibration, and why the present snapshot is enough

The electrons take `kT_e ~ 0.3 keV` at the shock (Ghavamian et al. 2007) and
relax by Coulomb collisions. Two things were wrong before:

* the temperature DIFFERENCE decays on `t_eq n_i/(n_e + n_i)`, not on `t_eq` —
  a factor 2.1 in cosmic gas, 9 in fully ionized oxygen, because the electrons
  hold most of the heat capacity there;
* the rate scales with `sum n_i Z_i^2/m_i`, which per gram is 4x smaller in
  oxygen than in hydrogen.

The integration uses the parcel's *present* density and temperature, and for an
adiabatic parcel that is exact, not approximate: `t_eq ~ T^{3/2}/n_e` and
adiabatic expansion carries `T ~ rho^{2/3}`, so `T^{3/2}/n_e` is invariant along
the trajectory. (Integrating instead in the electron column `int n_e dt`, which
the solver happens to carry, assumes `t_eq n_e` is the invariant — that needs
constant temperature, and it over-equilibrates the ejecta by 40 %.) The result
is `T_e/T = 0.32` EM-weighted: equilibration is a third complete at 350 yr.

### The spectral comparison, which is the real test

`casa_observe.py --compare` now extracts a spectrum from the synthetic event
file and from the real `evt2` events in the *same* aperture, through the *same*
response (with the ACIS cycle matched to the epoch — the filter contamination
absorbs most of the sub-keV flux by cycle 20, so comparing a cycle-0 model with
a cycle-20 observation is a soft-band error, not a model error).

Count rates in 0.5–7 keV against the real 295 counts/s, and band ratios
(synthetic/real) inside r < 200″:

| model | rate | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 2.8–4.2 | 4.2–6.0 | 6.0–7.0 |
|---|---|---|---|---|---|---|---|
| single-fluid `T` (CIE) | 113.5 (0.39x) | 0.20 | — | — | — | — | 3.00 |
| `T_e` (CIE) | 137.3 (0.47x) | 0.30 | 0.34 | 0.58 | 1.23 | 1.70 | 2.88 |
| **`T_e` + NEI** | 254.2 (**0.86x**) | **0.70** | **0.74** | **1.16** | 1.64 | 1.60 | 2.51 |

Using `T_e` rather than the single-fluid temperature is worth 21 % in total rate
and 50 % in the soft band — hotter gas puts more of its power outside the ACIS
band and produces fewer line photons inside it.

The remaining shape error is unambiguous and one-sided: **too hard**. That is
the signature of collisional ionization equilibrium standing in for a plasma
that has not reached it. Measured directly with AtomDB (NEI/CIE, at
`kT_e ~ 2–3 keV`, `n_e t ~ 1e11`):

| | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 6.0–7.0 |
|---|---|---|---|---|
| O | 3.1x | 1.0 | 1.0 | 1.0 |
| Si | 2.4x | **5.0x** | 4.2x | 0.38x |
| Fe | **6.5x** | 2.0x | 0.7x | 0.38x |

Every entry moves the model toward the observation — the two bands where it was
3x low go up by 3–6x, and Fe-K, where it was 2.9x high, comes down by a factor
2.6. That agreement in *sign and rough size across six independent bands* is the
reason to believe the diagnosis, and `--nei` implements it: the full run takes
the count rate from 0.47x to **0.86x** of the observed 295 counts/s and the two
line bands that carry most of Cas A's counts from 0.30/0.34 to 0.70/0.74.
`figures/obs_final_nei_spectrum_2004.png` shows the two spectra together — every
line complex (O/Ne/Fe-L, Si He-a, S He-a, Ar, Ca, Fe-K) lands at the right
energy with the right rough strength, and the ratio stays inside a factor ~2
across 0.4–8 keV.

What is left is still one-sided and still hard, but smaller: 1.6x too much
continuum at 4–6 keV, 2.5x at Fe-K, and 0.7x below 1.5 keV. Three candidates,
in the order they are worth testing:

1. **`T_e` is still too high.** Everything above is consistent with a bit too
   much hot gas; the Ghavamian post-shock value is calibrated on H-dominated ISM
   shocks and is extrapolated here to a reverse shock in metal ejecta.
2. **Unresolved clumping.** Emission goes as `<n^2>`; at 256^3 a cell is 1.7"
   and the imposed clumps are 13 % of the remnant radius against Orlando's 2 %.
   Denser, cooler knots would add soft emission without adding hard.
3. **The Ar and Ca lines are ~2x too strong** while Si is 0.74x, so the observed
   line ratios want less Ar and Ca per unit Si than Hwang & Laming's quoted
   masses imply. That is a `TRACER_SPLIT` input, and it is the one residual that
   is an assumption rather than a physical effect.

### What is assumed rather than simulated

* **four tracers, nine elements.** `_plasma.TRACER_SPLIT` divides the carried
  "O" layer into O/Ne/Mg and the "Si" layer into Si/S/Ar/Ca by Hwang & Laming's
  measured mass ratios. Comparing the raw "Si" tracer against measured Si read
  1.6x high; per element the model is now uniformly ~0.6x the observed shocked
  masses, which is an ejecta-composition input (`IIB_LAYERS`), not a
  hydrodynamic result.
* **full ionization in `mu_e`**, which is 10–20 % optimistic in Fe-rich cells.
* **single-shock ionization history**, and `T_e` held at its present value over
  that history.
* the reference hydrogen density that keeps APEC's `el/H` ratios finite in
  hydrogen-free ejecta — chosen per cell so that every metal density is exact,
  at the cost of 0.05 % of the free-free emission.

## Result 9 — how structured the remnant is, scale by scale

The image is the weakest part of the model, and "it looks too smooth" is not a
result until it is a number. `casa_morphology.py` band-passes the counts image
between `theta` and `2 theta`, takes the RMS in a 60–140″ annulus, subtracts the
Poisson variance analytically (for nested boxcars over `A1 < A2` pixels the
covariance is `m/A2`, so the noise variance of the difference is exactly
`m (1/A1 − 1/A2)` — nothing fitted), and divides by the local mean. That makes a
20 ks synthetic image comparable with a 143 ks observation.

| scale | synthetic | Chandra 2004 | real/syn | S/N of the subtraction |
|---|---|---|---|---|
| 1.0″ | 0.020 | 0.091 | 4.7x | 0.0 — meaningless |
| 1.5″ | 0.042 | 0.149 | 3.6x | 0.4 — meaningless |
| 2.5″ | 0.062 | 0.183 | 2.9x | 2.7 — marginal |
| 4.4″ | 0.120 | 0.219 | **1.83x** | 33 |
| 8.4″ | 0.218 | 0.248 | **1.14x** | 388 |
| 16.2″ | 0.308 | 0.273 | **0.89x** | 2932 |

**The model already matches Cas A's structure at ≥ 8″** and slightly exceeds it
at 16″ — the shell arcs and the pistons are, if anything, a touch too strong. It
is 1.83x too smooth at 4.4″. Below 2.5″ the synthetic image is Poisson-dominated
at 20 ks and the numbers mean nothing until it is regenerated at the real
exposure. Both the Chandra PSF (~0.5″) and the dust halo suppress the *real*
column, so the measured gap is a lower bound.

This is a resolution statement, and three independent facts say so:

1. one cell at 256³ in a 7 pc box is 0.0273 pc = **1.66″**, and WENO5 damps
   structure below ~3 cells, i.e. below ~5″;
2. the clump *seed* is grid-clamped, not physics-set —
   `k_hi = min(box/(0.02 r_FS), N/6)` is `min(273, 42)` at 256³, so the seeded
   clumps are 6 cells = **10″** against the intended 1.6″, a factor 6;
3. the measured deficit sets in at 4.4″ and is gone by 8.4″.

There is no evidence here for a missing large-scale mechanism. It also predicts
something checkable: in a roughly isobaric medium a denser knot is a *cooler*
knot, so the unresolved mass is sitting as smooth warm gas instead of cool dense
knots — which is exactly the residual spectral error (0.70x soft, 1.6x hard
continuum). Resolving the structure should move both at once, and if it moves
only one they are separate problems.

**Result 10 tested all three of those claims and two of them are wrong.** Read on
before quoting anything above about resolution.

## Result 10 — the structure deficit is not a resolution problem

Result 9 diagnosed the 4.4″ smoothness as a resolution statement and predicted
that resolving it would fix the residual spectral hardness too. Both halves were
measured. The diagnosis does not survive; the prediction is falsified in the
direction that matters.

### First, the measurement is real and not a noise artefact

The 20 ks numbers were regenerated at Cas A's actual 143.5 ks exposure. Nothing
moved, and two more scales became quotable:

| scale | 20 ks | 143.5 ks | S/N of the subtraction |
|---|---|---|---|
| 1.5″ | 3.59x | **3.64x** | 0.4 → **3.1** |
| 2.5″ | 2.94x | **2.97x** | 2.7 → **19** |
| 4.4″ | 1.83x | 1.84x | 33 → 235 |
| 8.4″ | 1.14x | 1.14x | 388 → 2784 |
| 16.2″ | 0.89x | 0.89x | 2932 → 21123 |

Count rate and all six band ratios are identical between the two exposures, so
this is the same model observed longer. The deficit therefore steepens
continuously inward: ~1.8x at 4.4″, ~3x at 2.5″, ~3.6x at 1.5″. The 1.0″ band
stays meaningless even at 143 ks, and for a *physical* reason rather than an
exposure one — the model has almost no 1″ structure, so its variance there is
essentially all Poisson and the subtraction has nothing left.

### A resolution ladder has to hold the seed fixed, and the first one did not

`turbulent_field` draws its Fourier coefficients on the run grid, so every rung
of a ladder gets a *different realisation* of the clumping, and `k_hi = N/6`
moves the seed *scale* with the grid as well. A ladder built that way varies
three things at once. `turbulent_field_on(..., seed_cells=)` now draws the
coefficients once and samples that same continuous field onto any grid — for a
field band-limited below the seed grid's Nyquist this is exact, not
interpolation (verified: subsampling the fine field reproduces the coarse one to
6e-19), and `--clump-seed-grid` / `--clump-kmax` expose it.

With the clumps held *literally identical* and only the grid changing:

| N | dx | 2.5″ | 4.4″ | 8.4″ | 16.2″ | rate |
|---|---|---|---|---|---|---|
| 128 | 3.3″ | 2.50x | 1.96x | 1.08x | 0.75x | 0.85 |
| 192 | 2.2″ | 2.40x | **1.46x** | 0.91x | 0.69x | 0.90 |
| 256 | 1.66″ | 2.67x | 1.63x | 0.97x | 0.73x | 0.89 |

**Non-monotone, with the realisation controlled.** 192³ develops more
small-scale texture than 256³ from the same initial condition. Doubling the
resolution moves the 4.4″ number by ±30 % in an unordered way, against a gap of
1.5–2x. Whatever sets the texture, it is not the cell size.

### Seeding smaller clumps makes the image smoother, not sharper

Result 9's second fact — that the grid clamps the seed 6x coarser than Orlando's
2 % target — was read as a limitation. Tested directly at fixed 256³, varying
only the seed band:

| seed `k_hi` | clump size | 2.5″ | 4.4″ | 8.4″ | rate |
|---|---|---|---|---|---|
| 21 | 25.4 % of r_FS | 2.67x | **1.63x** | 0.97x | 0.89 |
| 42 (grid-clamped default) | 12.7 % | 2.76x | 1.73x | 1.07x | 0.88 |
| 63 | 8.5 % | 3.01x | **1.95x** | 1.18x | 0.85 |

Monotone and backwards from the prediction. The 4.4″ structure is not seeded at
4.4″; it is produced by the nonlinear evolution — Rayleigh-Taylor fingering and
shock-clump interaction — of the *largest* seeded structures, while small seeded
clumps are damped by the scheme and smeared by the reverse shock before they can
grow. Removing the grid clamp would make the image worse.

**Confirmed out of sample at 512³.** That was predicted before the run was
scored: the default `k_hi = N/6` puts 512³ at `k_hi = 85`, the finest seed in
the whole study, so it should come out *smoother* than 256³ despite having half
the cell size. It does, on every metric at once:

| N | `k_hi` | dx | 4.4″ | 8.4″ | 16.2″ | rate | 0.5–1.5 |
|---|---|---|---|---|---|---|---|
| 128 | 21 | 3.3″ | 1.97x | 1.07x | 0.75x | 0.85 | 0.62 |
| 192 | 32 | 2.2″ | **1.66x** | 1.03x | 0.73x | 0.94 | 0.75 |
| 256 | 42 | 1.66″ | 1.83x | 1.14x | 0.88x | 0.86 | 0.70 |
| 512 | 85 | 0.83″ | **2.29x** | 1.54x | 1.21x | 0.77 | 0.58 |

**In the default configuration the image gets monotonically worse from 192³
upward, and the best remnant in this study is the second-coarsest.** The two
controlled experiments explain the whole curve: the seed-scale effect is
monotone and strong (1.63 → 1.95 for 3x in `k` at fixed 256³, and 2.29 by 4x at
512³), the pure-grid effect is non-monotone and weaker (1.96 / 1.46 / 1.63 with
the seed held fixed, optimum near 192³), and the default ties the two together
so that refining the grid also refines the seed and loses more than it gains.
The count rate falls with it (0.86 → 0.77) and the soft band with it (0.70 →
0.58), which is the `<n²>` signature of exactly this: less clumping, less dense
cool gas, less soft line emission.

This is also the point where the temptation to fit shows up, so it is worth
naming: the configuration that scores best on texture (`k_hi = 21`) is the one
furthest from Orlando's 2 % literature value, and it overshoots at 8.4″ and
16.2″. That is a reason to distrust it, not to adopt it. `CLUMP_SIZE_FRACTION`
stays at 0.02.

### The spectral coupling prediction is falsified

Result 9 predicted that more structure would raise the soft bands and lower the
hard continuum together, because isobaric dense knots are cool knots. Across the
same runs, ordered by how much 4.4″ texture they have:

| run | 4.4″ | rate | 0.5–1.5 | 4.2–6.0 | 6.0–7.0 |
|---|---|---|---|---|---|
| n256 k63 | 1.95x | 0.85 | 0.70 | 1.60 | 2.32 |
| n256 (default seed) | 1.73x | 0.88 | 0.71 | 1.64 | 2.45 |
| n256 k21 | 1.63x | 0.89 | 0.71 | 1.68 | 2.58 |
| n192 k21 | 1.46x | 0.90 | 0.72 | 1.65 | 2.64 |

More texture buys +0.02 in the soft band and costs +0.32 in Fe-K. Structure adds
emission everywhere and slightly *harder*, so the band ratios get worse, not
better. The two defects are separate.

By Result 9's own decision rule that puts the residual spectral error on the
electron temperature — the Ghavamian post-shock value is calibrated on
H-dominated ISM shocks and is being extrapolated to a reverse shock in metal
ejecta. **That, not resolution, is the next thing to work on.**

### Seeding the contact discontinuity: right physics, small effect

The clumping used to be imposed only inside the reverse shock. At the 150 yr
mapping time the reverse shock has already swept 2.3 of the 3.0 M☉ of ejecta
into the thin dense shell between r_RS and r_CD, so the interface that is
actually Rayleigh-Taylor unstable — and the material that produces essentially
all of the X-ray emission — was being handed to the solver perfectly smooth,
leaving the fingers to grow out of grid noise. The clumping now spans the whole
ejecta out to the contact discontinuity (`--clump-region`, default `ejecta`);
the perturbed mass goes from 1.17 to 2.79 M☉ at 256³, with the amplitude and
spectrum untouched.

Measured, at matched resolution and matched realisation:

| N | shell thickness | 4.4″ (r_RS window) | 4.4″ (r_CD window) | gain | rate |
|---|---|---|---|---|---|
| 128 | 1.4 cells | 1.97x | 1.96x | 0 % | 0.85 → 0.85 |
| 192 | 2.1 cells | 1.66x | 1.61x | 3 % | 0.94 → 0.96 |
| 256 | 2.8 cells | 1.83x | 1.73x | 5 % | 0.86 → 0.88 |
| 512 | 5.6 cells | 2.29x | 1.80x | **21 %** | 0.77 → 0.79 |

Real, in the right direction, and **the one change in this study whose benefit
grows cleanly with resolution** — precisely as the mechanism predicts, since
what is being perturbed is a shell that goes from 1.4 to 5.6 cells thick across
that range. (The 512³ pair is adiabatic, which the control above shows is worth
about 2 % on its own.) It is a correctness fix, not a solution: even at 512³ the
1.80x it reaches is no better than 256³'s old 1.83x, because the finer seed
costs more than the better-resolved shell wins back. The 256³ control reproduces
the previous `orl_n256_final` exactly (0.86 rate, 1.83x at 4.4″), which confirms
the refactor is otherwise a no-op.

### 512³ is stable again, but only with the old seeding

[[handoff-casa-orlando-calibrated]] recorded 512³ as dead. That was measured
with `POSITIVITY_HARD_FLOOR`. With `redistribute` the Orlando-route 512³ run
**completes to 350 yr** — the first one that ever has — with mass conserved to
6e-5 (17.3224 → 17.3214 M☉), r_FS = 2.465 pc and r_RS = 1.706 pc, both still
inside the observed 2.52 ± 0.20 / 1.58 ± 0.16.

The contact-discontinuity variant does *not*: it dies at t = 0.0215 (≈171 yr),
before the shell encounter, in a dt collapse with the mass still conserved to
1e-5 — so a genuine timestep collapse, not the old mass pump. The same
resolution with the old r_RS window passes that point and runs to completion.
The mechanism is already documented in `nickel_bubble_field`: at 512³ the
reverse shock meets a *fully resolved* dense cold structure at r ≈ 1.02 pc and
runs away radiatively, where at 256³ the same perturbation is grid-smeared and
survives. Seeding the contact discontinuity is exactly the change that puts a
resolved cold structure there.

**Dropping the cooling fixes it, and costs nothing.** `--cooling` off, same CD
seeding, 512³ runs to 350 yr: r_FS 2.465, r_RS 1.677, mass conserved. Result 8's
audit had already put `t_cool` at 2.8 Myr in the shocked gas — 4 orders above
the remnant's age — but that was an estimate. Measured on the observables at
256³, adiabatic against radiative with everything else fixed:

| | rate | 0.5–1.5 | 1.5–2.1 | 2.1–2.8 | 2.8–4.2 | 4.2–6.0 | 6.0–7.0 | 4.4″ |
|---|---|---|---|---|---|---|---|---|
| with cooling | 0.88 | 0.71 | 0.75 | 1.16 | 1.66 | 1.64 | 2.45 | 1.73x |
| adiabatic | 0.88 | 0.71 | 0.75 | 1.16 | 1.65 | 1.64 | 2.49 | 1.77x |

Identical to the third digit in the rate, within 0.04 in every band, 2 % in
texture, and the same r_FS and r_RS to the last digit. **Adiabatic + CD is
therefore the configuration for the top rung.**

### …but "cooling is irrelevant" holds only for the COSMIC curve

That comparison ran with `--clamp-floor`, the temperature floor, the cold-crush
LLF blend **and** the resolution limiter — and the limiter's entire job is to
switch cooling off in cells where the cooling length is unresolved, which is
exactly where thermal instability lives. It also used a cosmic-abundance curve
(X = 0.70, Z = 0.02) on metal-*dominated* ejecta. So what it measured was
suppressed cooling of the wrong plasma. Re-posed at 256³ with the guards off
(`--limiter-alpha 0`, no `--clamp-floor`) and Λ scaled directly:

| Λ | outcome | total energy | 4.4″ |
|---|---|---|---|
| ×1 (cosmic) | completes | 115.06 → 115.13 (*rises*) | 1.77x |
| **×10** | **destroyed: dt collapse to NaN at t = 0.0114 (≈161 yr)** | — | — |
| ×100 | completes | 115.06 → 114.92 (real loss) | 1.72x |

**At Λ×10 the shell undergoes a radiative crush runaway that kills the run.**
Cooling at a plausibly metal-enhanced rate is therefore *not* benign; the
resolution limiter was hiding it. The non-monotonicity is coherent rather than
suspicious: at ×100 cells cool straight through to the 10⁴ K floor where the
cooling gate switches them off, while at ×10 they linger in the thermally
unstable regime long enough for the shell to compress catastrophically —
intermediate rates are the most destabilising.

Whether that crush is physical fragmentation (the mechanism Orlando et al.
2025 invoke for the filamentary ejecta network) or an unresolved-radiative-shock
numerical runaway **cannot be settled by refining the grid** — refining makes it
worse. It needs a physical length scale for the cooling layer, i.e. the Field
length from `--conduction`, which has never been run on this configuration.
Until that is done, the honest statement is: *cosmic-abundance cooling is
irrelevant for Cas A; metal-enhanced cooling is an open question and the
radiative machinery must not be described as free.*

Note also that raising the metal mass fraction is **not** a way to ask this
question — the module is normalised to hydrogen (`Φ = n_H Γ − n_H² Λ`,
`mu_H = 1/X`), so raising Z lowers n_H and makes the gas cool *less* while the
tabulated Λ never moves. Use `--cooling-boost`; see `_boost_lambda`.

## Result 11 — what a real progenitor says about the fitted ejecta

Results 1–7 fitted the ejecta profile: its total mass, core radius, envelope
slope and inner slope δ are all parameters tuned until the shock radii came out
right. That is defensible as calibration but it means the profile is an input,
not a prediction — and the δ table above exists only because we were guessing.

The KEPLER solar-metallicity presupernova models of Sukhbold, Ertl, Woosley,
Brown & Janka (2016) — 200 stars from 9 to 120 M☉, with per-zone mass, radius,
velocity, density, temperature, pressure and 18 species — are distributed
**openly** by the Garching core-collapse archive at
`https://wwwmpa.mpa-garching.mpg.de/ccsnarchive/data/SEWBJ_2015/`. (The 3D
*explosion* models Orlando maps from, in the same archive, are not: those need
an access request.) `casa_progenitor.py` reads one, strips the hydrogen envelope
to a Type IIb as Orlando's W15-IIb is, and excises the iron core.

For **s16.0 stripped to 0.1 M☉ of hydrogen**:

| | as evolved | stripped to IIb |
|---|---|---|
| mass | 13.15 M☉ | **4.93 M☉** |
| radius | 887 R☉ (red supergiant) | **30.7 R☉ (compact)** |
| cores Fe / Si / C-O / He | 1.377 / 1.987 / 3.295 / 4.726 M☉ | unchanged |
| ejecta above the mass cut | — | **3.56 M☉** |

Two numbers fall out that were never put in.

**The fitted ejecta mass is nearly right.** We tuned `M_ej = 3.0` M☉ against the
shock radii; a real 16 M☉ star, stripped to a IIb and with its iron core
removed, gives **3.56 M☉** — within 18 %, and inside the observed 3–4 M☉. That
is the first independent physical support for a parameter that was pure
calibration.

**The energy budget is missing a quarter.** The material above the mass cut is
bound by **5.28e50 erg, 25 % of the calibrated 2.09e51 erg**. Route B's energy
is the kinetic energy at infinity, which is the right thing to compare with
observations — but any statement connecting it to an *explosion* energy has to
add the binding energy, i.e. the engine must deposit ~2.6e51 erg. Stripping
barely changes this (5.281 → 5.280e50): the binding energy lives in the deep
layers near the mass cut, not in the loosely bound envelope, so the IIb surgery
does not buy us out of it.

### Why the explosion itself is not simulated here

`casa_explode_1d.py` is written and its star maps correctly, but it cannot run:
**presupernova cores are supported by electron degeneracy and this solver has an
ideal-gas EOS.** The fraction of KEPLER's pressure that ideal gas plus radiation
accounts for in s16.0 is 2.7 % at the centre, 6 % at 0.7 M☉, 22 % at the 1.4 M☉
mass cut, 53 % at 2.3 M☉, and only reaches 80 % by 4 M☉. Handing those (ρ, p) to
a γ = 5/3 solver therefore describes a different star — the implied ideal-gas
central temperature is 2.7e11 K against KEPLER's 7.3e9 K — and it is not in
hydrostatic equilibrium under the EOS it is being integrated with, so it
disassembles on the first steps regardless of the bomb. Resolution, bomb mass
and bomb geometry are all irrelevant to this.

Closing it needs a degenerate-electron EOS. Starting instead above the
degenerate region (~3 M☉) would run today, but would impose the innermost ~2
M☉ of Si/O/Fe ejecta rather than computing it — and that is precisely the
material the reverse shock is processing at 350 yr, so it would concede the
point of the exercise.
