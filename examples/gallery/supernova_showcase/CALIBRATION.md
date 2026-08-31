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

## Result 12 — the metric was the problem, and Result 10 needs revising

Result 10's statistic is the band-pass RMS `sigma_I/I`. **An edge is scale-free**,
so a handful of large features with sharp boundaries deposits power into every
octave and scores exactly like a filamentary web. The symptom was visible and
was noted at the time: the 192³ model scores *best* on `sigma_I/I` (1.46x) while
looking like a smooth blob with two arcs across it, and the 512³
contact-discontinuity model scores *worse* (1.80x) while carrying the fine
cellular texture Cas A actually shows. A statistic that inverts the visual
ordering is not measuring the thing being chased.

`casa_morphology.py` now adds two statistics that carry **no amplitude
information at all**, so they are orthogonal to `sigma_I/I` rather than a
replacement:

* **Euler characteristic `chi`** (components − holes) of the brightest 25 % by
  *area*. Thresholding at fixed area fraction is what makes it pure topology:
  both images light up the same number of pixels, so only the connectivity can
  differ. A smooth image thresholded this way selects noise peaks — many small
  disconnected components, large `chi`; a genuinely structured one selects
  coherent structures — fewer, larger, lower `chi`.
* **Structure-tensor coherence.** Higher is *not* more Cas A-like: Chandra
  measures 0.54 at 4.4″ and every model sits at 0.65–0.92, because a few big
  clean arcs are locally coherent while a dense tangle of crossing filaments is
  not.

Both require the noise to be identical, which the analytic Poisson subtraction
cannot deliver for a threshold count. So the real image is **binomially thinned**
to the synthetic's counts: thinning a Poisson image with probability `p` gives
exactly a Poisson image of mean `p·lambda`, making the two statistically
indistinguishable in noise. (Consequence: `chi` is only comparable at matched
exposure — the 143.5 ks synthetic is not on the same scale as the 20 ks ones.)

### What changes

`chi` at 2.5″, synthetic/real, all at 20 ks:

| model | `sigma_I/I` at 4.4″ | **`chi` ratio at 2.5″** |
|---|---|---|
| 128³ CD | 1.96x | 2.42 |
| 192³ fixed seed | **1.46x — best** | 2.01 |
| 256³ seed k=21 | 1.63x | 1.76 |
| 256³ seed k=42 (default) | 1.73x | 1.29 |
| 256³ seed k=63 | 1.95x — worst | **1.25** |
| 512³ old r_RS window | 2.29x — worst | 1.49 |
| 512³ CD window | 1.80x | **0.84 — best** |

**Three of Result 10's conclusions reverse, and they now all point the same
way:**

1. **Resolution does help, monotonically.** 2.42 → 2.01 → 1.29 → 0.84 across
   128/192/256/512. Result 10's non-monotonic ladder was the RMS metric
   scrambling the ordering.
2. **Finer seeds are better, not worse.** At fixed 256³, `chi` goes 1.76 → 1.29
   → 1.25 for k_hi = 21 → 42 → 63, the exact opposite of the RMS result that
   "seeding smaller clumps makes the image smoother". Coarse-seed models score
   well on RMS *because* their few large sharp features are edge-rich, which is
   precisely the failure mode. **Result 10's headline on seed scale is
   withdrawn.**
3. **512³ CD is the best model in the study**, not merely "no better than
   256³".

### What does NOT change

The mechanism closures survive the better statistic, which is the reassuring
part. At 256³ against a 1.17 control: MHD at 500 µG gives 1.25, gentle Ni
bubbles 1.30, sharp Ni bubbles 1.16 — null or marginally negative on topology
exactly as on RMS. At 128³ the MHD trio is 2.40 / 2.28 / 2.57. So those
conclusions were not artefacts.

The synthesis the two statistics give together is more coherent than either
alone: **the model carries roughly the right amount of fluctuation power at
>= 8″ but the wrong organisation of it, and refining the grid adds little power
while steadily improving the organisation.** That is a different — and more
optimistic — statement than Result 10's, and it puts resolution back on the
table.

## Result 13 — the dust halo is real, mandatory, and NOT the outer flux

Every result up to here scores the model inside the forward shock, where it does
well. Outside it the model has been empty by construction, and Chandra is not.
Binned onto the same sky grid through the same response, `orl_n512_cd_adia`
against the 2004 epoch:

| radius | model | Chandra | ratio |
|---|---|---|---|
| 60–100″ | 1.10e-3 | 1.19e-3 | **1.1** |
| 100–140″ | 8.58e-4 | 9.93e-4 | **1.2** |
| 140–160″ | 1.03e-5 | 2.48e-4 | **24** |
| 160–180″ | 3.83e-7 | 9.83e-5 | **257** |
| 180–220″ | 3.85e-7 | 3.81e-5 | **99** |
| 220–260″ | 3.49e-7 | 1.66e-5 | **48** |

(counts/s/pixel, 0.5–7 keV). **8.8 % of Chandra's r < 260″ flux lies beyond the
observed forward shock at r = 153″; the model has 0.1 %**, and that 0.1 % is
only the instrumental and Galactic background SOXS adds. Quote that fraction
against a radius, not against an annulus edge — the profile falls by an order of
magnitude across the shock, so the same image gives 13.5 % beyond 140″, 8.8 %
beyond 153″ and 7.0 % beyond 160″.

This is a **sightline** problem, and `_dusthalo.py` models it as one: a
photon-level post-process that sits between `project_photons` (which applies
TBabs) and the telescope, because that is where interstellar dust sits
physically. It moves photons; it never creates them.

### Nothing in it is fitted

The dust column is the same one that does the absorbing. MRN's own normalisation
constants imply 1.564e-26 g of dust per hydrogen — a dust-to-H mass ratio of
0.0093 — so `--nh` fixes the scattering column with no knob left over.
`newdust` (Corrales et al. 2016, installed as `xdust`) then runs Mie theory on
that population, silicate plus graphite, `dn/da ∝ a^-3.5` from 0.005 to
0.25 µm.

Two independent published anchors say the answer is right:

* **Depth.** The model gives `tau_sca(1 keV) = 0.59 (N_H/1e22)`. Predehl &
  Schmitt (1995), from 25 measured ROSAT halos, give a mean `S ≈ 0.5`. 19 %,
  with nothing tuned.
* **Angular profile.** Draine (2003) Eqs. 9 and 11 give a median scattering
  angle `θ50 ≈ 360″ (keV/E)` and a cumulative `σ(<θ)/σ = s²/(1+s²)` in
  `s = θ/θ50`. Our MRN mixture reproduces that cumulative shape to a few
  percent at every quantile, with `θ50 ≈ 490″ (keV/E)` — a third wider, because
  MRN is not Draine's WD01 size distribution. `--halo-profile draine03` swaps
  the analytic form in, so that systematic is measured rather than argued.

### Two things the plan for this got wrong

**`tau_sca` is not `E^-2` where it matters.** It is asymptotically, and to
better than 5 % above 4 keV, but through the 0.5–2 keV band the local index
wanders between −1.4 and −2.2 and is not even monotonic: grains stop being small
compared with the wavelength. The band dependence is still steep — `tau` falls
7.0× from 0.5 to 2 keV — so "scattering is a soft-band effect" survives, but
`E^-2` must not be used to scale it onto the 0.5–1.5 keV residual.

**Single scattering is not good enough.** At `N_H = 1.2e22` the 1 keV depth is
0.71, so a quarter of the scattered photons scatter twice and single-scattering
would be a 25 % error there (a factor of two at 0.5 keV). The number of
scatterings is drawn from `Poisson(tau_sca(E))` and the small-angle deflections
add as vectors, each at its own position `x = 1 - d/D` along the sightline.

### The one real assumption

Where the dust sits along the 3.4 kpc. The default spreads it uniformly, which
needs no parameter; Cas A lies just beyond the Perseus arm, so a single screen
at ~2 kpc (`--halo-screen 0.41`) is the physically motivated alternative.
Running both is how the size of that assumption is measured.

### What it does: the right total in the wrong place, and a worse soft band

`orl_n512_cd_adia`, 20 ks, NEI, against Chandra 2004, with the *identical*
photon list — only the sightline changes:

| radius | model, no halo | model, halo | Chandra | real/model, no halo | real/model, halo |
|---|---|---|---|---|---|
| 60–100″ | 1.10e-3 | 9.51e-4 | 1.19e-3 | 1.1 | 1.3 |
| 100–140″ | 8.58e-4 | 7.33e-4 | 9.93e-4 | 1.2 | 1.4 |
| 140–160″ | 1.03e-5 | 4.65e-5 | 2.48e-4 | **24** | **5.3** |
| 160–180″ | 3.83e-7 | 2.97e-5 | 9.83e-5 | **257** | **3.3** |
| 180–220″ | 3.85e-7 | 2.06e-5 | 3.81e-5 | **99** | **1.8** |
| 220–260″ | 3.49e-7 | 1.34e-5 | 1.66e-5 | **48** | **1.2** |

The fraction of the r < 260″ flux beyond r_FS goes **0.09 % → 5.2 %** against
Chandra's 8.9 %. 18.3 % of photons scatter at least once, with a median
displacement of 141″.

**But the shape is wrong, and that matters more than the total.** The annulus
ratios above improve outward — 5.3, 3.3, 1.8, 1.2 — which looks like convergence
and is not. It is two curves with different slopes crossing. Measured as a local
power law `-d ln SB / d ln r` on the same images:

| | 160→180″ | 180→202″ | 202→227″ | 227→246″ | SB(150–170″)/SB(240–252″) |
|---|---|---|---|---|---|
| **Chandra 2004** | **7.2** | **5.6** | **4.2** | **4.9** | **10.9** |
| halo, uniform | 2.4 | 2.4 | 2.5 | 2.6 | 2.9 |
| halo, Perseus screen | 2.5 | 2.8 | 3.0 | 3.2 | 3.4 |

**The observed profile is far too steep to be a scattering halo.** This
falsifies the premise the work was started on, which was that the excess
declines "roughly as θ^-1.5 to θ^-2, the right shape for a dust-scattering
halo". It declines as r^-4 to r^-7. The error was in the sign as well as the
number: the data are too *steep* for a halo, not too shallow, and that could
have been measured in ten minutes before any of this was built.

### Can a halo be made compact enough? No

The only handle that narrows a scattering halo is grain size — `θ50 ∝ 1/(a E)`
— so MRN's 0.25 µm cutoff was pushed up and the profile re-measured:

| | τ_sca(1 keV) | 160→180″ | 202→227″ | SB(150″)/SB(246″) |
|---|---|---|---|---|
| Chandra | — | 7.2 | 4.2 | 10.9 |
| MRN, a_max = 0.25 µm | **0.71** | 2.4 | 2.5 | 2.9 |
| a_max = 0.50 µm | 1.52 | 3.4 | 3.3 | 4.1 |
| a_max = 1.0 µm | 2.39 | 4.0 | 3.5 | 4.8 |
| a_max = 1.0 µm + screen | 2.39 | 4.4 | 3.7 | 5.4 |

Quadrupling the maximum grain radius reaches half the observed contrast and
costs a factor 3.4 in optical depth: `S` goes from 0.59 — which is the value
Predehl & Schmitt measured — to **2.0**, four times the observational anchor.
There is no grain population that is simultaneously compact enough to make this
profile and shallow enough to be the dust we know is there. **The outer emission
is not the sightline.**

### It makes the count rate and the soft band WORSE, and that was the point

A halo cannot add photons. Inside a fixed aperture it can only remove them, and
the real observation lost the same photons long before we started modelling it.
So putting the model and the data through the same sightline moves the
comparison, and it moves it against us:

| | no halo | halo | |
|---|---|---|---|
| count rate ratio | 0.79 | **0.73** | |
| 0.5–1.5 keV (O, Ne, Fe-L) | 0.60 | **0.46** | |
| 1.5–2.1 keV (Si He-α) | 0.68 | 0.62 | |
| 2.1–2.8 keV (S He-α) | 1.06 | 1.02 | |
| 2.8–4.2 keV (Ar, Ca) | 1.58 | 1.56 | |
| 4.2–6.0 keV (continuum) | 1.71 | 1.71 | |
| 6.0–7.0 keV (Fe-K) | 2.43 | 2.42 | |

The count-rate question that was left open — whether the aperture correction
helps or hurts — is settled: it **hurts**, 0.79 → 0.73, because 7.9 % of the
model's photons leave r < 200″ and nothing scatters in from outside a source
that is entirely inside it.

The band effect is the one worth arguing about. The expectation was that
scattering would *relieve* the 0.5–1.5 keV deficit. It does the opposite:
scattering is strongest exactly where the model is already faintest, so the soft
ratio falls from 0.60 to **0.46** while everything above 2 keV moves by under
3 %. **The soft-band deficit is a factor of two, not 40 %**, and it is now a
deficit measured through the same sightline as the data rather than against a
model that pretended the sightline was empty.

### Guardrail

The emission model is untouched: the halo is a permutation of positions applied
to a photon list that is otherwise bit-identical, so every change above is
accounted for by photons leaving the aperture — 7.9 % of them at r < 200″, which
is the 0.79 → 0.73 in the count rate. Above 2 keV, where `tau_sca < 0.2`,
nothing moves by more than 3 %.

One trap is worth recording because it fails silently. `pyxsim.EventList`
**ignores the path it is given** and re-reads the filenames stored inside the
HDF5 file, so a copied-and-edited event list serves SOXS the original photons,
the run completes normally, and the count rate comes out identical to the
no-halo case — which is indistinguishable from "the halo does nothing".
`apply_dust_halo` rewrites `info/filenames` and then asserts the round trip.

### How big are the two assumptions? Both smaller than the result

The halo has exactly two things that are not computed from `--nh`: where the
dust sits along the sightline, and which grain size distribution it is. Both
were varied to the edge of plausibility:

| | count rate ratio | flux beyond r_FS | 0.5–1.5 keV | 1.5–2.1 keV |
|---|---|---|---|---|
| no halo | 0.79 | 0.09 % | 0.60 | 0.68 |
| **uniform dust, Mie/MRN** | **0.73** | **5.2 %** | **0.46** | **0.62** |
| Perseus screen, `x = 0.41` | 0.74 | 6.3 % | 0.47 | 0.62 |
| Draine (2003) profile | 0.75 | 5.2 % | 0.49 | 0.63 |
| Chandra 2004 | — | 8.9 % | — | — |

Putting all the dust in the Perseus arm instead of spreading it over 3.4 kpc
moves the outer flux by one point (5.2 → 6.3 %) and the band ratios by 0.01.
Swapping MRN's Mie profile for Draine's WD01 approximation — a third narrower —
moves them by less. The median displacement of a scattered photon is 141″, 129″
and 100″ across the three. **Every conclusion above survives both**, which is
what makes it worth stating that the depth itself is anchored on Predehl &
Schmitt rather than fitted: that is the number the result would actually be
sensitive to.

`figures/orl_n512_halo_uniform_absolute_2004.png` shows it as an image, and it
has to be a *different* image from the usual one: `comparison_figure` scales
each panel to its own 99.6th percentile and crops to 200″, which is right for
morphology and useless here, because the scattered light is three decades below
the shell and mostly outside that crop. `halo_image_figure` puts every panel on
the same absolute counts/s/pixel scale, logarithmic, out to 250″. Without the
halo the frame is black outside r_FS; with it a diffuse glow fills it, as in the
data. Two caveats when reading that panel: the vertical dark stripe is an ACIS
chip gap that only became visible once there were photons out there to fall into
it (the real image is a multi-obsid mosaic, so its gaps are dithered over), and
a three-decade log stretch cannot show the factor 1.2–1.8 that still separates
the outer profiles — read the numbers, not the picture.

`figures/orl_n512_halo_uniform_radial_2004.png` is the whole result in one plot:
without the halo the model falls off a cliff at 150″, with it the profile tracks
Chandra's outer tail and meets it by 250″, and the gap that remains is confined
to 150–200″.

### Reproducing

```bash
# the Mie table, once (~90 s on 40 cores; not keyed on N_H)
/export/home/lstorcks/xrayobs/bin/python _dusthalo.py --rebuild --selftest

# no halo, then halo, from the SAME cached photon list
S=/export/data/lstorcks/supernova_showcase
for v in "halo_off" "halo_uniform --halo" \
         "halo_screen --halo --halo-screen 0.41" \
         "halo_draine --halo --halo-profile draine03"; do
  set -- $v
  /export/home/lstorcks/xrayobs/bin/python casa_observe.py \
      $S/orl_n512_cd_adia.npz --exposure 20 --compare 2004 --nei \
      --pyxsim-events $S/xray_scratch/obs_n512_cd_adia_events.h5 \
      --out orl_n512_$1 "${@:2}"
done
```

### What this is, and what it is not

Three of the handoff's expectations for this work were wrong and one was right.
Wrong: that `tau_sca ~ E^-2` (not through the band where the argument was being
made); that the halo would *relieve* the soft-band residual (it doubles it); and
— the one that matters — that the outer excess had "the right shape for a
dust-scattering halo". It does not, by a factor of two in log slope, and no
admissible grain population closes that. Right: that the aperture correction
would push the count-rate discrepancy the wrong way.

**So this is not a model of the flux outside r_FS, and it should not be
presented as one.** It is a correction to the forward model that happens to be
mandatory rather than optional: `tau_sca(1 keV) = 0.71` is forced by the same
`N_H = 1.2e22` that the spectral fits already assume, so a fifth of the photons
*do* get moved and pretending otherwise is a modelling error. What it buys is
therefore not the outer profile — it is:

* **the aperture, honestly.** 7.9 % of the model's photons leave r < 200″ and
  the real observation lost the same photons before we ever saw it. Count rate
  0.79 → **0.73**.
* **the soft band, honestly.** 0.5–1.5 keV goes 0.60 → **0.46**. That is a
  factor-of-two deficit, not 40 %, and it is the single largest change any of
  this produced.

Both depend on the total scattered fraction and the aperture, not on the fine
angular profile, which is why they hold across every variant (0.73–0.75, 0.46–0.49)
even though the profile shape does not.

### What it leaves for the non-thermal rim — more, not less

The original reading was that the halo would take the diffuse outer component
and leave a sharp rim at 140–180″. The measurement says otherwise: **the whole
outer profile, from 150″ to 250″, is too steep to be sightline and is therefore
remnant.** That makes Task 2 bigger than it looked. A synchrotron model now has
to account for emission out to at least 250″ — where the halo contributes about
half the observed surface brightness but with the wrong gradient — not just a
thin rim at the shock. (Chandra's HRMA far-PSF wings are the other candidate and
are also not modelled here; they are shallower than r^-4 and so cannot be the
whole answer either.)

It also sharpens the T_e argument rather than softening it. The 4.2–6.0 keV
band is untouched by scattering (`tau_sca < 0.05` there), so the model's 1.71×
overprediction of that continuum stands unchanged — and if part of the *observed*
4.2–6.0 keV flux is non-thermal, the real thermal continuum is lower still and
the overprediction is worse than 1.71.

### And it is not the missing morphology

Worth stating because the image invites the opposite reading: a scattering halo
is smooth and azimuthally symmetric **by construction**. It cannot make a
filament, and it does not touch the 60–140″ annulus `casa_morphology.py` scores.
Nothing in Result 13 moves any structure number, and it must not be sold as if
it did.

---

## Result 14 — the model in XRISM's plane, per element group

*2026-08-31. `casa_xrism.py`, on `orl_n256_final` and `orl_n512_cd_adia`.*

Every spectral score in this study up to here has been a **band ratio**: six
numbers formed by integrating the whole remnant through the ACIS response. That
is the right test of a spectrum and a weak test of physics, and it is why four
candidate explanations for the residual (structure, cooling, the dust halo, T_e)
could each be falsified without any of them being *localised*.

**Vink et al. 2026 (arXiv:2602.06952), the XRISM/Resolve mapping of Cas A via
UltraSPEX, changed what is available** — and it was published after every result
above. It fits two pure-metal components per 30″ pixel, one for the
intermediate-mass elements and one for the iron group, and reports for each the
electron temperature, the ionization age, the Doppler shift and the line width.
Those are precisely the quantities `_plasma.py` and `_nei.py` compute per cell.

### Three method choices, each of which changes the answer

1. **Weight by the emitting ions, not by n_e².** `casa_plasma.py`'s
   n_e²-weighted `kT_e` is 3.05 keV — but on the same state its n_e²-weighted
   `mu_e` is **1.30** against 2.0 for metal ejecta, i.e. that average is
   dominated by shocked *wind*, not by the ejecta whose lines XRISM measures.
   The default weight here is `n_e × n_el × f_line(el)`, with `f_line` the NEI
   population of the charge states carrying the element's K-shell lines.
   *This matters:* the n_e² weighting understates the IME temperature error
   (1.14× against 1.44×) and gets the correlation sign **wrong** (below).
2. **Bin onto 30″ pixels.** XRISM's ranges are ranges *across regions*, one
   value per Resolve pixel. Comparing our single global average against their
   range would compare a mean to a distribution and could only ever "agree".
3. **Invert the mean ion temperature per species.**
   `_plasma.species_ion_temperature` gives each species its mass-proportional
   share, which is exact under adiabatic expansion because `T_s/T_i` is
   preserved along a parcel. It is an *unrelaxed upper bound*: nothing here
   relaxes the species towards each other.

### The measurement (256³ fiducial, line-emission weighted)

| | model, per pixel (10–90 %) | XRISM | |
|---|---|---|---|
| kT_e, IME | **3.03–5.56 keV** | 1.3–2.1 | **HIGH, 1.4–3×** |
| kT_e, Fe group | 2.59–5.43 keV | 2.4–8.4 | overlaps |
| n_e t, IME | 1.95–3.20×10¹¹ | 1.0–3.4×10¹¹ | overlaps |
| n_e t, Fe group | 1.42–2.76×10¹¹ | 0.8–3.0×10¹¹ | overlaps |
| Doppler shift, IME | −2087 to +2538 km/s | −1250 to +2000 | slightly broad |
| σ_v total, IME | 2934 km/s | ≤2200 | 1.33× |
| σ_v total, Fe group | 3419 km/s | ≤3700 | 0.92× |
| Spearman(kT_e, n_e t), IME | **−0.35** | negative | right sign |
| Spearman(kT_e, n_e t), Fe group | **+0.04** | negative, "pronounced" | **wrong** |
| ⟨n_e²⟩/⟨n_e⟩² over emitting gas | **1.45** | — | — |

### What it localises, in one sentence

**The iron is fine and the intermediate-mass elements are too hot** — which no
band ratio could have said, because the two are summed in every band.

And the ion temperatures turn that into a statement about the *hydrodynamics*
rather than about the plasma model. The Si-line-weighted ion temperature is
418 keV, and since `T_i = (3/16) m_Si v_s²/k`, that **is** a shock velocity:

> the reverse shock in the model's silicon-emitting gas runs at
> **2758 km/s** against a published ~1800.

A factor **1.53 in velocity**, i.e. 2.35 in temperature. Everything else in the
table follows from it: too-hot electrons, line widths 1.33× too broad, and a
soft band at 0.46× because the Si emissivity is off its peak.

*Sanity check on the machinery, which passes:* `(3/16)(m_Fe − m_Si) v²` at
1800 km/s is **176 keV**, inside XRISM's measured Fe−Si ion-temperature
difference of 150 ± 60 (SE) to 300 ± 180 keV (NW). Asserted in
`_plasma._assert_physics`.

### Resolution moves every number the right way

512³ CD against 256³, same weighting:

| | 256³ | 512³ CD |
|---|---|---|
| kT_e IME | 3.03–5.56 | **2.67–5.24** |
| v_Si implied | 2758 km/s | **2615** |
| ⟨n_e²⟩/⟨n_e⟩² | 1.45 | **1.87** |
| Spearman IME | −0.35 | **−0.54** |
| Spearman Fe group | +0.04 | **−0.28** |

**The Fe-group anticorrelation only appears at 512³.** That is the strongest
available evidence that the anticorrelation is a *density-structure* effect and
not a coincidence of the composition model — it switches on as the structure is
resolved. (Cost: 65 GB resident at 512³.)

### The weighting check, which is also a warning

Re-run with `--weight ne2`, i.e. `casa_plasma.py`'s convention: Spearman goes to
**+0.43**, the wrong sign, for both groups. The composition-aware weight is not
a refinement of the n_e² one; it is a different measurement, and the shocked
wind in the n_e² average reverses the correlation the observation is about.
`--weight element` (no ionization factor) gives 1.39× against 1.44×, so the
conclusion is robust to the *ionization* part of the weighting and sensitive to
the *composition* part.

---

## Result 15 — sub-grid density contrast: the temperature and the ionization age want different things

*2026-08-31. `_subgrid.py` + `casa_xrism.py --subgrid-scan`.*

Result 14 says the transmitted shock is 1.53× too fast. The literature says why:
Laming & Hwang (2003) needed ejecta overdensities ~100 to reconcile `n_e t` with
`kT_e`, and XRISM derives χ ≈ 10 (iron group) to ≈100 (IME) independently.

### Two things must be said before any number

**It cannot be fixed by refining the grid.** Cas A's knots are ~1″ = 0.016 pc; a
cell is 0.027 pc at 256³ and 0.014 pc at 512³. Resolving a χ = 100 knot at the
6–8 cells a shock interaction needs takes **N ≳ 1500** — 3–6× in linear
resolution, 30–200× in cost, past a memory wall that already stops this study at
512³. `CLUMP_SIZE_FRACTION = 0.02` already targets 3× larger than the observed
knots and is *still* grid-clamped.

**Our χ is not their χ, and quoting theirs here would double-count.** Their
inference is the offset of the data from a one-zone model with a single uniform
1800 km/s reverse shock. Ours is the offset from a 3D calculation that already
contains a distribution of shock velocities, clumping at contrast 5, pistons and
a reflected shock from the CSM shell. Most of what a one-zone analysis must
attribute to χ is already present here as resolved structure — which is why the
measured requirement below is **χ ≈ 4, not 100**.

### The model

Each cell is re-read as two phases at fixed cell mass, fixed cell volume and
**pressure equilibrium** — the post-crushing state (Klein, McKee & Colella
1994), reached in ≪ the 200 yr since the reverse shock arrived. Pressure
equilibrium is what makes the temperature split free of new parameters:
`ρT` equal across the phases gives `T_dense = T/χ` immediately, the same `v²/χ`
a transmitted shock gives. Two parameters, χ and `f_mass`; `χ = 1` is an exact
identity, asserted.

### The scan (256³, f_mass = 0.5, `net_mode = crossing`)

| χ | kT_e IME (10–90 %) | n_e t IME (10–90 %) | v_Si | ρ IME | ρ Fe | verdict |
|---|---|---|---|---|---|---|
| 1.0 | 3.03–5.56 | 1.95–3.20e11 | 2758 | −0.350 | +0.037 | kT_e over |
| 1.5 | 2.76–5.02 | 2.08–3.41e11 | 2583 | −0.356 | +0.020 | both over |
| 2.3 | 2.35–4.31 | 2.56–4.21e11 | 2222 | −0.389 | −0.062 | both over |
| **4.0** | **1.76–3.41** | 3.48–5.73e11 | **1730** | **−0.459** | **−0.350** | **kT_e IN, n_e t OVER** |
| 8.0 | 1.12–2.43 | 5.02–8.33e11 | 1214 | −0.479 | −0.479 | kT_e IN, n_e t OVER |
| 16.0 | 0.68–1.67 | 6.97e11–1.17e12 | 849 | −0.466 | −0.141 | kT_e IN, n_e t OVER |

*(XRISM: kT_e 1.3–2.1, n_e t 1.0–3.4e11, reverse shock ~1800 km/s, both ρ
negative. ρ is Spearman(kT_e, n_e t) across pixels.)*

*This table is `net_mode = crossing`; the `unchanged` one is below and is the
one that works.*

### The result: χ ≈ 4 satisfies all four XRISM constraints at once

**χ ≈ 4 lands on three targets immediately** — kT_e enters XRISM's range, the
implied shock velocity hits 1730 km/s against the published 1800, and *both*
correlation coefficients go negative, including the Fe-group one that Result 14
could not reproduce at 256³ at all.

**Under `crossing` it overshoots the ionization age**, 3.48–5.73e11 against an
observed 1.0–3.4e11 — `n_e t` was already at the top of the observed range at
χ = 1, so a mode that raises it further breaks it. That looked like a
show-stopper: two observables pulling against each other with no configuration
satisfying both.

**It is not, and `net_mode = unchanged` is the resolution.** Same scan, same χ:

| χ | kT_e IME | n_e t IME | v_Si | ρ IME | ρ Fe | |
|---|---|---|---|---|---|---|
| 1.0 | 3.03–5.56 | 1.95–3.20e11 | 2758 | −0.343 | +0.037 | kT_e over |
| 2.3 | 2.13–3.70 | 1.81–2.96e11 | 2245 | −0.363 | −0.007 | kT_e over |
| **4.0** | **1.55–2.69** | **1.87–3.05e11** | **1770** | **−0.410** | **−0.174** | **both IN** |
| 8.0 | 0.98–1.80 | 1.92–3.13e11 | 1267 | −0.498 | −0.222 | both IN |
| 16.0 | 0.61–1.20 | 1.94–3.16e11 | 905 | −0.477 | +0.020 | kT_e now too cold |

**At χ = 4 with the ionization age unchanged, the model satisfies the electron
temperature, the ionization age, the implied shock velocity and both correlation
signs simultaneously** — four independent XRISM constraints from one parameter,
none of them fitted to.

And `unchanged` is not a convenience: its physical content is sharp and
falsifiable. `n_e t` is `ρ × t`, so leaving it unchanged while the density rises
by χ means the elapsed time falls by χ — **the dense clumps were engulfed by the
reverse shock four times more recently than the mean**, which is a statement
about clump size and the shock crossing time, and a testable one. `crossing`
(engulfed only √χ later) gives a *stronger* Fe-group correlation, −0.350 against
−0.174, but overshoots `n_e t`; so the correlation mildly prefers `crossing` and
the level clearly prefers `unchanged`, and both are negative at χ = 4 either way.

χ ≈ 4–8 is the window: by χ = 16 the electrons are too *cold* (0.61–1.20 keV) and
the Fe-group correlation returns to zero.

The clumping factor is the other surprise: it peaks at ~1.63 near χ = 2.3 and
*falls* thereafter, never approaching the ~25 a one-zone two-phase estimate
gives. The mechanism is the weighting itself — as the dense phase cools, its
silicon stops being He-like and drops out of the line-emission weight. **An
emissivity argument made at fixed temperature does not survive being done at the
temperature the same model implies.**

### This is an interpretation layer

The simulation does not contain this structure. `_subgrid.describe` prints that
sentence on every run and any figure using it must carry it.

---

## Result 16 — synchrotron: anchored to a factor of two, by one measured number

*2026-08-31. `_synchrotron.py`.*

Helder & Vink (2008) measure ~54 % of Cas A's 4.2–6 keV flux as non-thermal
remnant-integrated; XRISM finds 47–90 % per pixel. **So the recorded residual
"our 4.2–6 keV is 1.71× the observed" is not a comparison between the same
quantity.** The observed *thermal* flux there is ~0.46× the total, making the
model's thermal continuum **~3.7× too bright**. Two of the six band ratios in
`OVERVIEW.md` §4 are not currently tests of anything.

### How much of this can be predicted rather than fitted

The obvious implementation has two free parameters and says nothing. Most of
that freedom can be removed:

* **the radio flux fixes the electrons.** 2720 Jy at 1 GHz, α = 0.77 (Baars et
  al. 1977), from the same population. Radio and X-ray emissivity both go as
  `K B^((s+1)/2)`, so normalising the summed radio emission cancels it.
* **the cutoff is field-independent.** `hν_cut ≈ 1.4 keV (v_s/3000 km/s)²/η`
  (Zirakashvili & Aharonian 2007): the acceleration and loss rates both scale as
  `1/B²`, so `E_max` depends only on `v_s` and the gyrofactor.
* **but the field returns through the emitting VOLUME.** The electrons radiating
  5 keV live 0.9 yr at 100 µG and **0.08 yr** at Cas A's ~500 µG rim field,
  against a 350 yr remnant. They occupy a thin layer at the shock; the radio
  electrons fill the shell.

| emitting volume assumed | predicted 4.2–6 keV | vs observed non-thermal |
|---|---|---|
| co-spatial with the radio (whole shell) | 2.5e-10 | **9.3×** |
| purely advected loss layer, 1.0e-4 pc | 3.2e-14 | **0.0012×** |
| **observed filament width, 1–3″** | 0.5–1.6e-11 | **0.19–0.58×** |
| the model's **own** fresh fraction, 0.0177 | — | **0.16×** |

**The picture is consistent to within a factor of a few, in the direction of too
faint.** With the measured radio flux, a loss-limited cutoff and the observed
filament width as the emitting thickness, the prediction is 0.19–0.58× the
measured flux with nothing fitted — and the model's *own* ram-pressure-weighted
fresh fraction gives 0.16×, so **the two independent routes to the emitting
volume agree with each other to a factor ~1.2–3.5.** That agreement is the
substantive result. The shared offset from unity is open; η > 1 would raise the
cutoff and the flux.

*A correction, recorded because it cost a debugging cycle:* the first version of
this table said 0.51–1.53× and "consistent within a factor of two", using a
**0.3 pc** emitting shell. That is the visually bright rim, not the shocked
region — the calibrated solution has r_FS − r_RS = **0.8 pc**. The wrong value
also made the wired component look ~30× fainter than the module predicted, which
read as a plumbing bug and was not one. `SHELL_THICKNESS_PC` is now tied to the
measured shocked thickness and asserted against it.

**What is not predicted is the width.** The advected loss layer is 2936× thinner
than the shell, so the observed filaments are 161–484× thicker than advection
allows — diffusive transport or magnetic damping, an open problem. So the honest
description is **one geometric parameter taken from an observation**, not two
free physics parameters and not a prediction.

### The part with nothing adjustable at all

Inverting XRISM's fitted photon index Γ = 2.94–3.43 through the loss-limited
relation at η = 1 gives a shock at **1708–2423 km/s**. That is Cas A's *reverse*
shock (1800–2000), not its forward shock (~5000) — so the non-thermal continuum
in those pointings is reverse-shock emission. Nothing was tuned to produce it,
and it agrees with the published association of the non-thermal continuum with
the reverse shock in the (south)west.

### And it makes the residual worse

Repeated because the above is encouraging and the consequence is not. Adding
synchrotron does not rescue the model; it makes the hard band mean something and
hands the problem to Result 15.

**One earlier claim of mine here was wrong and is corrected:** a first pass put
the advected loss layer at ~1e-9 pc and concluded the bracket spanned nine
orders of magnitude, making the normalisation hopeless. The arithmetic was
wrong — it is 1.0e-4 pc, the bracket is ~3000, and with the observed width the
answer lands within a factor of two. The conclusion changed from "hopeless" to
"one measured parameter".

---

## Result 17 — the Ar/Ca tracer split: a measured assumption, and fixing it works

*2026-08-31. `--tracer-split`, matched 256³ pair, NEI, no halo, vs Chandra 2004.*

`TRACER_SPLIT` divides the carried "Si" tracer among Si, S, Ar and Ca. It was set
from Hwang & Laming (2012)'s **remnant-integrated** shocked masses (Si 0.08,
S 0.06, Ar 0.02, Ca 0.02 M☉), because being comparable to that measurement is
the point of this pipeline. Expressed as ratios to solar, that is

    S/Si = 1.44x     Ar/Si = 1.72x     Ca/Si = 2.72x

**everywhere.** XRISM/Resolve measures S/Si = 0.88–1.12 solar per 30″ pixel, with
the Ar and Ca enhancement **confined to the NE and SW jet bases**. Both can be
true: the remnant-integrated Ar mass can be 0.02 M☉ with most of it in the jets
while the ratio is solar in the bulk. One "Si" tracer cannot say so, so the
enhancement is spread over *all* of the layer including the brightest smooth Si
— which is exactly the recorded residual "Ar and Ca ~2× too strong while Si is
0.74×". Not a bug and not a physical effect: the cost of four tracers standing
for nine elements.

`xrism_bulk` sets solar S/Si, Ar/Si and Ca/Si, derived from
`SOLAR_NUMBER_RATIO_TO_H` rather than typed in. **Only the Si layer changes** —
XRISM's band does not constrain Ne/O or Mg/O, and forcing those to solar would
put 15 % of the oxygen layer's mass into neon against Hwang & Laming's measured
Ne/O ≈ 0.08 solar, which is what an oxygen-burning layer should look like.
Changing a ratio no observation constrains, in the same commit as one it does, is
how a calibration stops being traceable.

### The matched pair

| band | `hwang_laming` | `xrism_bulk` | |
|---|---|---|---|
| 0.5–7 keV rate | 0.86 | **0.88** | |
| 0.5–1.5 (O, Ne, Fe-L) | 0.70 | 0.70 | — |
| **1.5–2.1 (Si He-α)** | 0.74 | **0.84** | **+0.10 →1** |
| 2.1–2.8 (S He-α) | 1.16 | 1.14 | −0.02 |
| **2.8–4.2 (Ar, Ca)** | 1.65 | **1.41** | **−0.24 →1** |
| 4.2–6.0 (continuum) | 1.61 | 1.52 | −0.09 |
| 6.0–7.0 (Fe-K) | 2.52 | 2.50 | −0.02 |

**The two residuals that were fingered move together, in the right directions,
and nothing else degrades.** Four of six bands improve; the count rate improves.
And the control reproduces the recorded guardrail to the last digit (0.86 and
0.70 / 0.74 / 1.16 / 1.65 / 1.61 / 2.52 against the recorded 0.86 and
0.70 / 0.74 / 1.16 / 1.64 / 1.60 / 2.51), which validates the whole chain through
the refactor.

**It accounts for about half the Ar/Ca excess** — 1.65 → 1.41 against a target of
1.0. The rest is not composition: the 2.8–4.2 keV band also carries continuum,
and the continuum is separately too bright (Results 15, 16).

**Which preset to use is a real choice, not a bug fix.** `hwang_laming` is right
if you want the remnant-integrated masses; `xrism_bulk` is right if you want the
per-pixel line ratios where the emission is. They disagree because the real
enhancement is spatially localised and this model cannot represent that. Both are
kept, `set_tracer_split` selects, and every consumer prints which it used.

*Side finding, fixed:* the hand-typed Si layer summed to **0.999**, quietly
losing 0.1 % of it. Both presets are now derived from their source numbers.

---

## Result 18 — the sub-grid contrast fixes the SHAPE, and exposes a normalisation error the old cancellation was hiding

*2026-08-31. `casa_observe.py --subgrid-chi 4 --subgrid-fmass 0.5
--subgrid-net-mode unchanged`, 256³, NEI, no halo, vs Chandra 2004; then an
`f_mass` scan through `casa_xrism.py`.*

Result 15 calibrated χ against the plasma diagnostics. This is the spectrum, which
the scan does not constrain, and it is the first thing in this study to move the
**one-sided energy-dependent residual** that Results 10, 7, 13 and the T_e bracket
all failed to move.

### The measurement

| band | χ = 1 (control) | **χ = 4** | target |
|---|---|---|---|
| 0.5–7 keV rate | 0.89 | **2.26** | 1.0 |
| 0.5–1.5 (O, Ne, Fe-L) | 0.70 | 2.48 | 1.0 |
| 1.5–2.1 (Si He-α) | 0.74 | 1.99 | 1.0 |
| 2.1–2.8 (S He-α) | 1.16 | 2.47 | 1.0 |
| 2.8–4.2 (Ar, Ca) | 1.65 | 2.48 | 1.0 |
| 4.2–6.0 (continuum) | 1.60 | 1.64 | 1.0 |
| **6.0–7.0 (Fe-K)** | **2.52** | **1.71** | 1.0 |

Read as absolute ratios that looks like a failure — everything got worse except
Fe-K. It is not, and the reason is that **the normalisation and the shape moved in
opposite directions.** Dividing each row by its own rate ratio separates them:

| band | shape, χ = 1 | shape, χ = 4 |
|---|---|---|
| 0.5–1.5 | 0.79 | 1.10 |
| 1.5–2.1 | 0.83 | 0.88 |
| 2.1–2.8 | 1.31 | 1.09 |
| 2.8–4.2 | 1.86 | 1.10 |
| 4.2–6.0 | 1.81 | 0.72 |
| 6.0–7.0 | 2.84 | 0.75 |
| **spread (max/min)** | **3.60** | **1.51** |
| **rms deviation** | **0.250 dex** | **0.084 dex** |

**The spectral shape improves by a factor of three, and its character changes.**
The control's residual is *monotone rising with energy* — 0.79 to 2.84, the
soft-deficit/hard-excess signature this study has been chasing since Result 8.
At χ = 4 there is no monotone trend below 4 keV and the whole 0.5–7 keV band sits
within ±25 %. Fe-K, the single worst residual in the study, goes 2.52 → 1.71 while
every other band rises — i.e. it improves *twice*, once absolutely and once
relative to the rest.

**So the density contrast was the right mechanism.** That is what four falsified
candidates and the "too much hot gas" diagnosis were pointing at, and it is now
measured rather than inferred.

### And the normalisation is a new, cleaner residual

2.26× is not a small error, and the honest reading is that **the control's 0.89
was a cancellation**: it under-predicted the soft bands and over-predicted the
hard ones, and the total came out near unity because the two errors had opposite
signs. Fixing the shape removed the cancellation and left the underlying problem
visible — **there is ~2.3× too much emitting material once the emission is placed
at the right temperature.**

### `f_mass` cannot absorb it, and that is the useful part

`f_mass` sets how much of the cell mass is in the dense phase, hence the clumping
factor, hence the normalisation. It was never scanned (0.5 was a placeholder). At
fixed χ = 4:

| f_mass | dense share of emission | kT_e IME | v_Si | Spearman IME / Fe | rate (∝ C) |
|---|---|---|---|---|---|
| 0.05 | 18 % | 2.75–4.99 | 2614 | −0.345 / +0.035 | ~1.10 |
| 0.10 | 32 % | 2.51–4.52 | 2484 | −0.354 / +0.032 | ~1.22 |
| 0.20 | 54 % | 2.15–3.81 | 2261 | −0.365 / +0.006 | ~1.46 |
| **0.50** | **88 %** | **1.55–2.69** | **1770** | **−0.410 / −0.174** | **2.26** |

*(XRISM: kT_e 1.3–2.1 keV, reverse shock ≈ 1800 km/s, both correlations negative.)*

**The two constraints select opposite ends of the range.** The `f_mass` that fixes
the normalisation (0.05–0.1) loses the temperature fix completely — kT_e back to
2.5–5.0 keV, the implied shock back to 2500–2600 km/s, and the Fe-group
correlation back to +0.03. The `f_mass` that fixes the temperature (0.5) overshoots
the rate by 2.3×. The mechanism is transparent: the dense phase is cool *and*
bright, so its share of the emission is what sets the temperature — and that share
is exactly what sets the normalisation.

**A two-phase model with one contrast therefore cannot satisfy the spectral shape
and the total emission at once.** That is a measured negative and it is more
useful than another tuning knob would have been, because it says the 2.3× is
**not absorbable by the sub-grid parameters** — it is a property of the
hydrodynamic state.

### Where that points, and it is somewhere the project already has the tool

Too much emitting material in the shocked ejecta is `OVERVIEW.md` §5.6: the
reverse shock has eaten too far into the ejecta *in mass* (96 % shocked against an
observed 87–90 %, 0.21 M☉ unshocked against 0.35 ± 0.10), controlled by the inner
ejecta slope δ, and recorded as *blocked* because δ = 1 has never run in 3D.

So the chain closes on itself: the XRISM diagnostic localised the residual to the
ejecta density structure; the sub-grid contrast fixed the shape and exposed the
normalisation; the normalisation points at δ; and δ is precisely the correlated,
multi-symptom parameter `casa_diff.py` exists to fit. **It is also the one place
in this whole study where PDE-constrained optimisation is well posed** (§5, "why
the residual is not fitted away with gradient descent"). Clear that module's
recorded blocker — reconcile the smooth `r_RS` and `M_unshocked` with
`casa_analyze.py`'s definitions — and this is the fit to run.

### A bug this cost, and the guard that was wrong

`merge_event_lists` refused the two phases after 80 minutes of photon generation,
on `emin` disagreeing (0.402 against 0.438 keV). Those are **outputs**, not
inputs: pyXSIM stores the actual observed-frame energy range of the photons it
drew, so two components at different temperatures legitimately differ there. The
merged list now spans their union and only `area`, `exp_time`, `nH` and `redshift`
are enforced. The phase event lists were reusable through `--pyxsim-events`, so
the cost was the guard's, not the run's — **but a guard strict about the wrong
quantity is as expensive as no guard at all.**

---

## Result 19 — the gradient fit: δ ≈ 1, and E, M_ej, n_w were already right

*2026-08-31. `casa_diff.py --validate/--check-grad/--fit`, 1D, 2000 cells, 350 yr,
float64.*

Result 18 left the spectrum limited by ~2.3× too much emitting material, which is
the δ / unshocked-mass residual `OVERVIEW.md` §5.6 records as *blocked*. This is
the fit that attacks it, and it is the one place in this study where
PDE-constrained optimisation is well posed: the 1D model is genuinely the right
physics, the gradient is exact, and the residuals are correlated in exactly the
way a one-at-a-time scan cannot handle.

### First, the recorded blocker was not a smoothing problem

`casa_diff.py` was recorded as unusable for `r_RS` and `M_unshocked`. The cause
was that they **measured different quantities** from the calibration's:

* `casa_calibrate_1d` defines the reverse shock by **homology** — the outer edge
  of the still-freely-expanding ejecta, |v − r/t| < 5 % of r/t. `casa_diff`
  defined it by **entropy**. On the fiducial the entropy version sits at
  **3.999 pc — the box edge**, because the outer wind is cold too, against a true
  r_RS of 1.721. A fit against that would have optimised a radius nobody
  measures, with a perfectly well-behaved gradient.
* `M_unshocked` multiplied by the entropy indicator *as well as* the radial
  window, integrating a strict subset of the hard region and then subtracting the
  hard version's full interior-wind correction. That mismatched numerator is why
  it went negative (−0.08 M☉).

`--validate` had the mirror-image defect: its "hard" r_RS was *also* an entropy
criterion, so it compared two entropy definitions with each other and reported
their agreement as validation. Both are now transcribed from
`casa_calibrate_1d.measure_snapshot`.

| observable | smooth | hard | rel |
|---|---|---|---|
| r_FS | 2.4797 | 2.5190 | −1.6 % |
| r_RS | 1.7223 | 1.7210 | **+0.1 %** |
| n_post | 4.0465 | 4.0097 | +0.9 % |
| M_unshocked | 0.1016 | 0.0961 | +5.7 % |

**And the gradients tightened as a consequence**: JVP against central differences
is now 0.5–10.9 % across the four parameters, against the 0.2–37 % recorded
before. A well-posed observable differentiates better than an ill-posed one.

### `n_post` is what makes the fit determined, and it matters

With three targets the system is under-determined and the fit walks up the E–n_w
degeneracy — both act through the deceleration. Observed: χ² fell 7.003 → 0.239
while **E rose 2.09 → 2.87e51 and n_w 0.93 → 1.50 together**, which is the
degeneracy and not a result. Adding `n_post` (Lee et al. 2014, 4.0 ± 1.0 cm⁻³)
pins n_w almost directly and takes the system to exactly determined.

*(v_FS and v_RS are deliberately not added: a velocity needs two epochs or a
shock-frame construction, and the post-shock gas velocity is not the shock
velocity. Adding a fifth observable that is subtly the wrong quantity is the
mistake this module already made once with r_RS.)*

### The fit, 4 targets / 4 parameters

| | fiducial (hand) | fitted | target |
|---|---|---|---|
| E | 2.09e51 erg | **2.00e51** | — |
| M_ej | 3.0 M☉ | 2.66 M☉ | — |
| n_w | 0.928 cm⁻³ | **0.988** | — |
| **δ (inner slope)** | **0.0** | **1.38** | — |
| r_FS | 2.480 | 2.426 | 2.52 ± 0.20 |
| r_RS | 1.722 | 1.673 | 1.58 ± 0.16 |
| n_post | 4.05 | 4.14 | 4.0 ± 1.0 |
| **M_unshocked** | **0.102** | **0.344** | **0.35 ± 0.10** |
| χ² | 7.005 | **0.561** | — |

**Two results, and the second is as useful as the first.**

**1. δ ≈ 1, and it fixes the unshocked mass.** 0.102 → 0.344 M☉ against an
observed 0.35 ± 0.10. δ ≈ 1 is the *standard core-collapse value*, which the
project's own notes had identified as the likely answer but only ever scanned.
Both fits agree on it: 1.36 with three targets, 1.38 with four. **This is the
stable, robust output.**

**2. E and n_w were already right.** With `n_post` in, the fit returns
E = 2.00e51 against the hand-calibrated 2.09e51 and n_w = 0.988 against 0.928 —
within 5–7 %. The hand calibration of Result 2 is *independently confirmed by a
gradient fit that was free to move it*, and the one thing it got wrong was δ,
which it never varied.

### What is NOT settled, stated plainly

**M_ej is the least constrained parameter and the two fits disagree about its
direction**: 3.46 M☉ with three targets, 2.66 M☉ with four, against 3.0 fiducial
and 3.56 from the KEPLER progenitor (Result 11). It is still drifting at the last
step while δ rises, which is a residual δ–M_ej degeneracy — both control how much
ejecta the reverse shock has processed. **Do not quote a fitted M_ej.** Breaking
that degeneracy needs an observable sensitive to the total ejecta mass
independently of the reverse-shock position; the shocked *element* masses are the
obvious candidate and would need the composition model in the loop.

Also: at exactly determined, χ² = 0.561 is **not a goodness-of-fit test** — it
only says the solver converged. The script now computes that note from the counts
rather than asserting it.

### The next step this hands over

δ ≈ 1 has still **never run in 3D** — that is the recorded §5.6 blocker, a
timestep collapse at t = 0.019 attributed to the Fe pistons sitting inside a
centrally peaked core. The 1D fit now says δ ≈ 1 is what the data want, with a
number and an uncertainty behind it, so bisecting that crash is no longer
speculative work.

---

## Result 20 — δ = 1 was already running, and the recorded blocker was a month out of date

*2026-08-31. Matched 256³ pair, adiabatic, shell + pistons + composition,
`--positivity redistribute --coldcrush-factor 16`, differing only in the inner
ejecta slope of the mapped 1D profile.*

Result 19 fitted δ ≈ 1.4 in 1D and handed over "run it in 3D — the recorded §5.6
blocker says it has never run". **It has been running for a month.**

### What the profile actually contained

`casa_calibrate_1d.py --inner-slope` **defaults to 1.0**, and the committed
`casa_1d_map150.npz` carries `cfg_inner_slope = 1.0` with a genuinely peaked core:

| profile | d ln ρ / d ln r, 0.05–0.5 pc | r_FS (150 yr) | r_RS |
|---|---|---|---|
| `--inner-slope 0.0` | **+1.25** (hollow) | 1.284 | 0.935 |
| committed `casa_1d_map150.npz` | **−0.55** (peaked) | 1.313 | 0.944 |

So every production mapping has been a δ = 1 mapping. The `ejecta_radial_shape`
clip that once made `--inner-slope` a no-op was fixed (`OVERVIEW.md` §6), and
after the fix the default took effect — silently, because nothing re-measured the
ejecta budget afterwards.

### The matched 3D pair

| | δ = 0 (`orl_n256_d0ctl_new`) | δ = 1 (`orl_n256_final`) | observed |
|---|---|---|---|
| total ejecta | 3.153 M☉ | 3.204 M☉ | — |
| shocked ejecta | 3.031 (**96 %**) | 2.787 (**87 %**) | 87–90 % |
| **unshocked ejecta** | **0.122 M☉ (2.28 σ)** | **0.417 M☉ (0.67 σ)** | 0.35 ± 0.10 |
| r_FS | 2.494 pc | 2.494 pc | 2.52 ± 0.20 |
| **r_RS** | **1.269 pc** | **1.531 pc** | 1.58 ± 0.16 |

**The δ = 0 control reproduces the recorded residual exactly** — 96 % shocked and
0.12 M☉ unshocked against the recorded "96 %, 0.096–0.21". So the numbers in §5.6
were δ = 0 numbers, and the state they were attributed to is δ = 1.

**And δ = 1 runs cleanly.** Mass 17.3676 → 17.3670 (6 digits), energy conserved to
1 part in 10⁴, completes to 350 yr. The recorded "δ = 1 has never run in 3D — a
timestep collapse at t = 0.019" was true when it was seen, on a configuration
predating `--positivity redistribute` as the default, the `ejecta_radial_shape`
clip fix, and `--coldcrush-factor 16`.

Note also that **δ moves r_RS and the ejecta budget but leaves r_FS untouched**
(2.494 in both), which is what makes it the right knob for this residual: it fixes
the reverse shock without disturbing the forward shock the calibration was built
on.

### The lesson, and it is not a small one

**A recorded blocker decays exactly like a recorded result, and this one had been
setting the priority list for a month.** `OVERVIEW.md` §6 collects "the run
completed and printed a plausible number" failures; this is the same disease in a
new organ — *the run was never attempted again, and the reason not to attempt it
stayed in the document after it stopped being true.* Re-check a blocker before
building around it. The check cost 25 minutes of GPU and would have cost the same
at any point in the last month.

### One inference in Result 18 is withdrawn

Result 18 attributed the 2.3× normalisation excess to this residual — "too much
emitting material in the shocked ejecta *is* the δ problem". **It cannot be:** the
ejecta mass budget is right to 0.67 σ. The excess needs another explanation, and
the measurement below is the candidate.

### What replaced it: the emission budget is circumstellar

On the same state, of the `n_e²`-weighted emission measure in shocked gas:

* **79 % from shocked wind/CSM, 21 % from ejecta**
* 8.52 M☉ of shocked wind against 2.79 M☉ of shocked ejecta
* EM-weighted μ_e = 1.297, against 2.0 for metal ejecta

`OVERVIEW.md` §1 opens by stating that Cas A's X-ray emission is *line-dominated
ejecta emission*. A model whose emission measure is 79 % circumstellar is suspect
on its own terms — and the signature matches: shocked wind is metal-free, so it
contributes **continuum and no lines**, and too much of it gives exactly *too
little soft line emission and too much hard continuum*.

**Stated as a hypothesis, because that is what it is.** The 79 % is a continuum
(`n_e²`) fraction; the forward model computes lines from per-element abundances,
so the ejecta may still dominate the line bands. The discriminating measurement is
the **band-resolved** ejecta/wind split, which `casa_observe.py` can produce by
observing the two populations separately and differencing.

It also does not retract Result 18's *measurement*: the sub-grid contrast does
flatten the shape (rms 0.250 → 0.084 dex). But it flattens it by **adding ejecta
line emissivity**, while this hypothesis says the shape is wrong because there is
**too much wind continuum**. Both flatten the same residual; only one can be the
cause. **So χ must not be tuned further until the band-resolved split is measured.**
