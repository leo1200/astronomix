# Cassiopeia A end to end: what the model contains, and what it does not

*A map of the pipeline for someone arriving cold. `CALIBRATION.md` has the
measurements and their derivations (Results 1–13); this says how the pieces fit
together, what physics is in each, and — the part that matters most — what is
still missing and why.*

*§5's ranking was revised on 2026-08-31 after a literature audit. One paper —
**Vink et al., XRISM/Resolve mapping of Cas A (arXiv:2602.06952, ApJ 2026)**,
published after every result in `CALIBRATION.md` — measures per element group
exactly the quantities `_plasma.py` computes, and it moved three items: ejecta
density contrast to the top, non-thermal synchrotron from last to third, and T_e
off the list entirely. **Start at §5.0.***

---

## 1. The object, and why it is modelled this way

Cassiopeia A is a ~350 yr old (Fesen et al. 2006) Galactic remnant at 3.4 kpc
(Reed et al. 1995) of a **Type IIb** supernova — a progenitor stripped to
~0.1 M☉ of hydrogen. Its X-ray emission is **line-dominated ejecta emission**
from metal-rich material heated by the *reverse* shock, not swept-up ISM. Three
consequences drive every design choice below:

1. **Composition is not a detail.** Fully ionised oxygen has μ = 1.78 and
   μ_e = 2.0, against cosmic 0.61 / 1.18. Assuming cosmic abundances gets the
   temperature, the electron density, the ionisation age and the emissivity all
   wrong, by factors.
2. **The plasma is far from equilibrium.** At n_e t ≈ 10¹¹ s cm⁻³ it is
   ionising, not in CIE; and the electrons are colder than the ions, because
   Coulomb equilibration takes longer than the remnant's age.
3. **The structure is inherited.** Most of the fine structure was imprinted by
   the explosion and is then *reprocessed* by the reverse shock (Orlando et al.
   2025) — it is not generated during the phase we simulate.

---

## 2. The pipeline

```
 casa_calibrate_1d.py   1D spherical, ejecta + r^-2 wind, calibrated to
        │               r_FS, r_RS, v_FS, v_RS, n_post          (seconds, CPU)
        ▼
 casa_orlando.py        map at 150 yr onto a 3D Cartesian grid, impose
        │               multi-D structure, evolve to 350 yr      (hours, GPU)
        ▼
 _plasma.py / _nei.py   composition → μ, μ_e, T_e, n_e t → ion fractions
        │
        ▼
 casa_observe.py        yt → pyXSIM (AtomDB, per cell, Doppler) → SIMPUT
        │               → _dusthalo.py → SOXS (ACIS ARF/RMF/PSF/bkg, Poisson)
        ▼
 casa_morphology.py     structure vs scale + topology, against real evt2
 casa_morph_null.py     phase-randomised null: is the structure real?
 casa_analyze.py        shock radii, masses    casa_plasma.py  plasma diagnostics
 casa_xrism.py          per-element-group kT_e, n_e t, sigma_v and per-species
                        T_i against XRISM/Resolve -- the sharpest scoring in the
                        directory, and the cheap place to calibrate (Result 14)

 casa_progenitor.py  →  casa_explode_1d.py     side branch, not in the main line:
                                               a real KEPLER progenitor stripped
                                               to IIb, exploded in 1D (Result 11)
```

`_common.py` holds the solver configuration and the initial-condition builders
the whole directory shares, and `run.sh` is the environment wrapper (§9).
`_subgrid.py` (density structure below the cell, Result 15) and
`_synchrotron.py` (the non-thermal component, Result 16) are physics modules for
the two gaps §5 ranks first and third; both are self-checking and neither is
wired into `casa_observe.py` yet.
`casa_wind.py` blows the CSM self-consistently and scores itself against the two
published features §5.7 names; it is not calibrated and not wired in.
`verify_sharded_turb.py` checks that the multi-GPU turbulent field is the same
field as the single-device one — a regression guard for the fix in §6.

The rest of the directory is the **pre-calibration gallery**: `cassiopeia.py`,
`cassiopeia_realistic.py`, `casa_turb_phase.py`, `snr_sedov.py`,
`young_snr_ism.py`, and the `reimage.py` / `compare_real.py` renderers. Those
are tuned to look right rather than to agree with data — no composition, no NEI,
a forward shock ~25 % small — and `README.md` documents them and their limits.
Do not take numbers from them. The thermal-instability / AthenaK cross-code study
that used to live here is now in `../thermal_instability/`, and the one-off
debugging probes have been deleted (their findings are in §6).

**Why two stages.** Building the remnant directly in 3D skips the phase that
sets the answer: the progenitor wind holds ~3 M☉ — an entire ejecta mass —
inside 1.5 pc, so by the time the blast arrives there it has already swept that
up, formed a reverse shock, and given much of its energy away. The 1D stage
does that history cheaply and *calibrates* against measurements; the 3D stage
adds the structure that only exists in 3D. This is Orlando et al.'s Route B.

**Calibrated fiducial** (Result 2): `E = 2.09e51 erg`, `M_ej = 3.0 M☉`,
`n_w = 0.928 cm⁻³` at 2.5 pc, γ = 5/3, mapped at 150 yr, evolved to 350 yr.

---

## 3. Physics that is in the model

### Hydrodynamics (`astronomix`)
WENO5 finite difference, SSP-RK4, 3D Cartesian, float32 with **Bryan+95 dual
energy** (the cold kinetic-energy-dominated ejecta core makes float32 pressure
recovery cancellation noise without it), positivity-preserving flux limiter,
cold-crush LLF blend. Positivity mode **`redistribute`** — `HARD_FLOOR`
manufactures mass (17 → 10¹² M☉) and was the cause of the old "512³ is
unstable" verdict.

### Initial condition at 150 yr (`casa_orlando.py`)
| ingredient | source | notes |
|---|---|---|
| ejecta + wind profile | calibrated 1D run | flat core + steep envelope; δ = 0 working value |
| small-scale clumping | Orlando et al. 2012 | log-normal, contrast ≈ 5, k-band, **spans the whole ejecta to r_CD** |
| 5 large-scale anisotropies | Orlando et al. 2016 Table 4 | Fe-rich knots + Si-rich NE jet/SW counter-jet; carry their own composition, which produces the observed Fe-outside-Si inversion |
| asymmetric CSM shell | Orlando et al. 2022 Eq. 1 | placed *ahead* of the blast so the interaction is computed; drives the reflected shock that pushes r_RS inward |
| wind clumping | — | k = 4–20, σ = 0.4 |
| Ni bubbles | optional, `--ni-bubbles` | measured null as imposed at 150 yr |
| magnetic field | optional, `--mhd-b0` | uniform, div-free; measured null (see §5) |
| explosion-era plumes | optional, `--plume-sigma` | direction-only (radially coherent) von Mises–Fisher lobes, the one anisotropic ingredient. Spectral gain, morphological null (see §5) |

### Passive scalars (9 + 4)
`C_ej` (ejecta/CSM discriminator), `C_Fe/C_Si/C_O/C_He` (H is the remainder),
plus solver-managed `entropy_initial`, `shocked_fraction`, `time_since_shock`,
`density_time`. **These are what make the observation possible**: without them
the forward model must assume one metallicity and CIE.

### Optional physics
Radiative cooling (Schure et al. 2009 curve, resolution limiter, T-floor,
`--clamp-floor`); thermal conduction (implemented, **15 days/run** — see §5).

### From state to observable (`_plasma.py`, `_nei.py`)
- **μ and μ_e per cell** from the carried composition.
- **T_e ≠ T_i** (`--te-model`, four prescriptions that bracket the physics —
  see §5.5): electrons take ~0.3 keV at the shock (Ghavamian et al. 2007)
  and relax by Coulomb collisions. The temperature *difference* decays on
  `t_eq · n_i/(n_e+n_i)` — a factor 9 in fully ionised oxygen — and the rate
  goes as Σ n_i Z_i²/m_i. Integrated in the parcel's *present* (ρ, T), which is
  exact for an adiabatic parcel because `T^{3/2}/n_e` is invariant along it.
  Result: **T_e/T = 0.32**, kT_e = 3.05 keV, EM-weighted.
- **NEI ion fractions** from each parcel's (kT_e, n_e t), 35 ion fields.

### The telescope (`casa_observe.py`, `_dusthalo.py`)
AtomDB/APEC emissivities per cell with the simulated abundances; Doppler shifts
from the velocity field; TBabs at N_H = 1.2×10²²; **dust scattering** (Mie on
an MRN population, nothing fitted — MRN's own normalisation ties the scattering
column to the same N_H as TBabs); then SOXS with the real ACIS ARF, RMF, PSF
image, instrumental and Galactic backgrounds, and Poisson statistics at the
actual exposure. Output is an event list binnable exactly like real `evt2`.

### Scoring
- **Spectral** — count rate and six band ratios in the same aperture through
  the same response, with the ACIS cycle matched to the epoch.
- **Structural** — `casa_morphology.py`: band-passed σ_I/I per scale (Poisson
  subtracted analytically), plus **Euler characteristic χ** and
  **structure-tensor coherence** on a binomially-thinned (noise-matched) pair.
- **Null** — `casa_morph_null.py`: phase randomisation preserves the power
  spectrum exactly and destroys topology, so `excess = statistic − its null`
  separates real structure from "the same power spectrum".

---

## 4. Where the model stands

**Two states carry these numbers and they are not interchangeable.** The
dynamics and plasma diagnostics were measured on the 256³ fiducial
`orl_n256_final`; the structural results and the halo-on spectrum on the 512³
contact-discontinuity run `orl_n512_cd_adia`, which is the best model in the
study (Result 12). Every row therefore names its state — a scoreboard that mixes
them silently reads as one model when it is two.

| quantity | model | state | observed |
|---|---|---|---|
| r_FS | 2.494 pc | n256 | 2.52 ± 0.20 |
| r_RS | 1.531 pc | n256 | 1.58 ± 0.16 |
| shocked ejecta | 2.79 M☉ | n256 | 2.8–3.7 |
| **unshocked ejecta** | **0.21 M☉** (1D δ = 0 scan: 0.096) | n256 / 1D | **0.35 ± 0.10 — off, and the fix is blocked; §5.6** |
| EM-weighted kT_e | 3.05 keV | n256 | — |
| EM-weighted n_e t | 2.3e11 | n256 | ~1e11 |
| 0.5–7 keV rate | 0.86× | n256, **no halo** | 1.0 |
| 0.5–7 keV rate | **0.73×** | n512 CD, **halo** | 1.0 |
| bands (0.5–1.5 … 6–7) | 0.70 / 0.74 / 1.16 / 1.64 / 1.60 / 2.51 | n256, no halo | 1.0 |
| bands (0.5–1.5 … 6–7) | **0.46 / 0.62 / 1.02 / 1.56 / 1.71 / 2.42** | n512 CD, halo | 1.0 |
| σ_I/I at 4.4″ | 1.80× | n512 CD | 1.0 |
| χ ratio at 2.5″ | 0.84 (best on the ladder) | n512 CD | 1.0 |
| coherence excess over the null | +0.56 | n512 CD | +0.36 |
| Euler excess over the null | −13.06 — **overshoots** | n512 CD | −10.30 |

**Two of these rows are not comparisons between the same quantity.** The
4.2–6.0 and 6–7 keV bands are ~half non-thermal in the real remnant and purely
thermal in the model (§5.0), so read 1.56 / 1.71 / 2.42 as *lower bounds on a
thermal excess of ~3.7×*, not as 1.7×. Everything below 2.8 keV is a fair
comparison.

**The halo-on row is the one to quote.** The dust halo is mandatory (Result 13),
so the 0.86× / 0.70 pair is a number from a sightline the model no longer claims
is empty. Scattering removes 7.9 % of the photons from r < 200″ and takes most
of them out of the soft band, so **the 0.5–1.5 keV deficit is a factor of two,
not 40 %** — measured through the same sightline as the data.

**The 512³ CD result is not settled.** It wins on χ — decisively, 0.84 against
1.25–2.42 for every other rung — but on the Euler *excess* over the
phase-randomised null it overshoots Chandra (−13.06 against −10.30) while the
512³ r_RS-window run nearly matches it. (On the amplitude statistic σ_I/I the
best score belongs to 192³, at 1.46×; Result 12 is the argument for why that
statistic ranks a smooth blob with two arcs above a filamentary web.) Different
statistics, different winners; the seeding window is still open.

**Independent support:** a real KEPLER progenitor (Sukhbold et al. 2016) at
16 M☉, stripped to IIb with its iron core excised, gives **3.56 M☉** of ejecta
against the **3.0 M☉ we fitted** — 18 %, unprompted (Result 11,
`casa_progenitor.py` / `casa_explode_1d.py`).

---

## 5. What is missing, and what it would cost

### Missing physics, ranked by how much of the residual it explains

*Re-ranked after the 2026-08-31 literature audit. Two of the items below moved,
and the reason is one paper: **Vink et al., XRISM/Resolve mapping of Cas A via
UltraSPEX (arXiv:2602.06952, ApJ 2026)** measures, per element group, exactly
the quantities `_plasma.py` computes — and it was published after every result
in `CALIBRATION.md`. §5.0 states what it says.*

**0. What XRISM now measures, and what it does to the residual.**

| | XRISM | this model |
|---|---|---|
| kT_e, IME (Si/S/Ar/Ca) | **1.3–2.1 keV** | 3.05 keV (EM-weighted, all species) |
| kT_e, Fe-group | 2.4–8.4 keV (>5 in W/central) | — (not split) |
| n_e t, IME | 1.0–3.4×10¹¹ | 2.3×10¹¹ |
| n_e t, Fe-group | 0.8–3×10¹¹ | — (not split) |
| σ_v, IME / Fe-group | ≤2200 / ≤3700 km s⁻¹ | not measured |
| T_i(Fe) − T_i(Si) | **150–300 keV** | one ion temperature |
| n_e t vs kT_e | **anticorrelated ("robust")** | — |
| nonthermal fraction, 4–6 keV | **47–90 %, flux-weighted 84 %** | **0 %** |
| implied ejecta overdensity | **χ ≈ 10 (Fe-group), up to ≈100 (IME)** | **`CLUMP_MAX_CONTRAST` = 5, species-independent** |

Two consequences, both of which change what the residual *means*:

*The hard band is ~3.7× off, not 1.71×, and the comparison as posed is not
valid.* Helder & Vink (2008) give ~54 % nonthermal in 4.2–6 keV
remnant-integrated; XRISM's 84 % is two PV pointings, so 54 % is the right
number for our full aperture. The observed **thermal** flux there is then 0.46×
the total, against our pure-thermal 1.71× — a thermal continuum **~3.7× too
bright**, up to ~10× locally. Two of the six band ratios in §4 are currently
not comparisons between the same quantity.

*T_e is closed as a **lever**, not as a **quantity**.* The full bracket (§5.3)
shows nothing in the admissible T_e family helps. But XRISM says the IME
electrons really are ~2× too hot. The fix is therefore upstream, in the density
structure: denser clumps slow the transmitted shock (v ≈ v_s/√χ), which lowers
kT_e *without* the emission-reweighting that defeated the scan.

**1. Ejecta density contrast — first among the spectral gaps, and now measured
(Result 15).** `_subgrid.py` re-reads each cell as two phases at fixed cell mass,
fixed cell volume and pressure equilibrium — the post-crushing state, which makes
the temperature split free of new parameters (`ρT` equal gives `T_dense = T/χ`,
the same `v²/χ` a transmitted shock gives). `casa_xrism.py --subgrid-scan`
calibrates χ in minutes rather than against a 45-minute observation:

| χ | kT_e IME | n_e t IME | v_Si | Spearman IME / Fe | |
|---|---|---|---|---|---|
| 1.0 | 3.03–5.56 | 1.95–3.20e11 | 2758 | −0.34 / +0.04 | kT_e over |
| 2.3 | 2.13–3.70 | 1.81–2.96e11 | 2245 | −0.36 / −0.01 | kT_e over |
| **4.0** | **1.55–2.69** | **1.87–3.05e11** | **1770** | **−0.41 / −0.17** | **both IN** |
| 8.0 | 0.98–1.80 | 1.92–3.13e11 | 1267 | −0.50 / −0.22 | both IN |
| 16.0 | 0.61–1.20 | 1.94–3.16e11 | 905 | −0.48 / +0.02 | kT_e too cold |

*(XRISM: kT_e 1.3–2.1, n_e t 1.0–3.4e11, v ≈ 1800, both correlations negative.
`net_mode = unchanged`, which is the one the data select — see below.)*

**At χ ≈ 4 the model satisfies four independent XRISM constraints at once** — the
electron temperature, the ionization age, the implied shock velocity (1770 against
1800 km s⁻¹), and *both* correlation signs including the Fe-group one it gets
wrong at χ = 1. One parameter, nothing fitted to any of the four.

**The ionization-age treatment is what makes that possible, and it is a physical
statement.** `n_e t` was already at the top of the observed range at χ = 1, so the
modes that raise it (`density` × χ, `crossing` × √χ) fix the temperature and break
the ionization age: at χ = 4, `crossing` gives 3.48–5.73e11 against an observed
ceiling of 3.4e11. `unchanged` holds it flat, and since `n_e t = ρ t`, holding it
flat while ρ rises by χ means the elapsed time *falls* by χ — **the dense clumps
were engulfed by the reverse shock four times more recently than the mean.** That
is a claim about clump size and the shock crossing time, and it is falsifiable.
(`crossing` does give a stronger Fe-group correlation, −0.35 against −0.17, so the
correlation mildly prefers it while the level clearly prefers `unchanged`; both are
negative at χ = 4 either way.) χ ≈ 4–8 is the window: by χ = 16 the electrons are
too *cold* and the Fe-group correlation returns to zero.

**χ ≈ 4, not the χ ≈ 10–100 of the literature, and that is not a disagreement.**
Laming & Hwang (2003) and XRISM infer χ from the offset of the data from a
*one-zone* model with a single uniform 1800 km s⁻¹ reverse shock. Ours is the
offset from a 3D calculation that already contains a distribution of shock
velocities, clumping at contrast 5, pistons and a reflected shock. Most of what a
one-zone analysis must attribute to χ is already here as resolved structure, and
quoting theirs would double-count it.

**The naive emissivity argument also does not survive being done properly.** A
one-zone estimate at χ = 100, f = 0.01 gives a clumping factor ≈25, hence "a
factor ~19 in emissivity". Measured, ⟨n²⟩/⟨n⟩² peaks at **1.63** near χ = 2.3 and
*falls* thereafter, because as the dense phase cools its silicon stops being
He-like and drops out of the line-emission weight. An emissivity argument made at
fixed temperature is invalid at the temperature the same model implies.

**The resolution wall is why this has to be sub-grid.** Cas A's knots are
~1″ = 0.016 pc; a cell is 0.027 pc at 256³ and 0.014 pc at 512³, so resolving a
high-contrast knot at the 6–8 cells a shock interaction needs takes **N ≳ 1500**
— 3–6× in linear resolution, 30–200× in cost, past a memory wall that already
stops this study at 512³. `CLUMP_SIZE_FRACTION = 0.02` already targets 3× larger
than the observed knots and is *still* grid-clamped.
**Wired through `casa_observe.py`** as two thermal components
(`--subgrid-chi/-fmass/-net-mode`), which works because the emission measure was
already explicit: a phase filling part of a cell is the same fields with the
emission measure scaled by its volume fraction. *Remaining:* the spectrum at
χ = 4 has not been measured yet — the scan constrains the plasma diagnostics, not
the band ratios. **It is an interpretation layer and `_subgrid.describe` prints
that sentence on every run.**

**2. Explosion-era structure — still the dominant *morphological* gap.**
The ejecta arrive already structured: neutrino-driven convection/SASI plumes,
Ni-bubble walls inflated after shock breakout, and RT fingers at the C/O–He and
He/H composition interfaces during shock propagation (Orlando et al. 2025). All
three are **coherent and anisotropic**. Our seed is a Gaussian random field —
**statistically isotropic by construction** — and the phase-randomised null
shows the consequence quantitatively: our structure *is* genuinely non-Gaussian
(Δcoherence +0.56 against a Gaussian null's 0), but **1.6× too ordered**
compared with Chandra's +0.36.
*Cost:* request the Wongwathanarat/Janka mapped model from the Garching archive
(one email; the progenitor models there are open, the explosion models are not),
or compute the RT-mixing part ourselves with an imposed l = 1,2 asymmetry
(2–3 sessions). A full in-house explosion needs a Yin-Yang/moving-mesh solver
rewrite (6–12 months) plus a degenerate EOS — **not recommended**; see §7.

**3. Non-thermal synchrotron — promoted from last to third, and it is no
longer optional.** Absent entirely. It costs the 140–180″ rim (a 3–5× residual
the dust halo demonstrably cannot absorb), and — the reason for the promotion —
**47–90 % of the observed 4–6 keV flux is non-thermal** (§5.0), so until it is
in the model those two band ratios are not tests of anything. Adding it does
*not* relieve the hard excess; it makes it ~3.7× instead of 1.71×, and points at
§5.1 for the cure.
`_synchrotron.py` now carries the physics, and it is **better anchored than
"a fit" (Result 16)**. The radio flux fixes the electrons (2720 Jy at 1 GHz,
α = 0.77, same population, and the `K B^((s+1)/2)` scaling cancels); the
loss-limited cutoff `hν_cut ≈ 1.4 keV (v_s/3000)²/η` is field-independent. What
does *not* cancel is the emitting **volume** — the 5 keV electrons live 0.08 yr at
Cas A's ~500 µG rim field against a 350 yr remnant:

| emitting volume assumed | vs observed non-thermal 4.2–6 keV |
|---|---|
| co-spatial with the radio (whole shell) | 9.3× |
| purely advected loss layer (1.0e-4 pc) | 0.0032× |
| **observed filament width, 1–3″** | **0.51–1.53×** |

With the observed width supplying the thickness the prediction is consistent
**within a factor of two, with nothing fitted** — one geometric parameter taken
from an observation, not two free physics parameters. The width itself is not
predicted: the observed filaments are 161–484× thicker than advection allows,
which needs diffusive transport or magnetic damping and is an open problem.

**And one result with nothing adjustable at all:** inverting XRISM's
Γ = 2.94–3.43 through the loss-limited relation at η = 1 gives a shock at
**1708–2423 km s⁻¹** — Cas A's *reverse* shock (1800–2000), not its forward shock
(~5000). The non-thermal continuum in those pointings is reverse-shock emission,
and nothing was tuned to produce that.
*Remaining cost:* ~1 week to add it to the photon list as a second source
(`pyxsim.PowerLawSourceModel` takes per-cell luminosity and index fields, which
is what the module produces), plus the event-list merge.

**4. `TRACER_SPLIT`'s Ar and Ca — an assumption with a measured error.**
Ours is Si:S:Ar:Ca = 0.444 / 0.333 / 0.111 / 0.111 by mass, from Hwang & Laming
(2012)'s *remnant-integrated* shocked masses, which puts Ar/Si and Ca/Si at
**2.1–2.5× solar everywhere**. XRISM localises that enhancement to the NE/SW jet
bases, with S/Si = 0.88–1.12 elsewhere. So the total Ar mass may well be right
while the *emission* is wrong, because the enhancement is spread over the
brightest Si instead of confined to the jets — which is exactly the "Ar/Ca ~2×
too strong while Si is 0.74×" residual, now with a size and a cause.

**DONE, and it works (Result 17).** `--tracer-split xrism_bulk` sets solar
S/Si, Ar/Si and Ca/Si in the Si layer only. Matched 256³ pair:

| band | `hwang_laming` | `xrism_bulk` |
|---|---|---|
| rate | 0.86 | **0.88** |
| 1.5–2.1 (Si He-α) | 0.74 | **0.84** |
| 2.8–4.2 (Ar, Ca) | 1.65 | **1.41** |
| 4.2–6.0 | 1.61 | 1.52 |
| others | 0.70 / 1.16 / 2.52 | 0.70 / 1.14 / 2.50 |

The two fingered residuals move together in the right directions, nothing
degrades, and the control reproduces the guardrail to the last digit. It accounts
for about **half** the Ar/Ca excess; the rest is continuum, not composition.
**Which preset to use is a real choice**: `hwang_laming` for the
remnant-integrated masses, `xrism_bulk` for the per-pixel line ratios where the
emission is. They disagree because the real enhancement is spatially localised
and one tracer cannot say so. Both are kept and every consumer prints which it
used.

**5. T_e — CLOSED as a lever (was "the headline open residual").**
The Ghavamian prescription is calibrated on H-dominated ISM shocks and
extrapolated to a reverse shock in metal ejecta, so it *was* the obvious
suspect. `--te-model` now brackets the whole admissible family, and the answer
is a bound rather than a direction:

| model | EM kT_e | rate | 0.5–1.5 | 6–7 (Fe-K) |
|---|---|---|---|---|
| `minimal` (no collisionless heating — cold bound) | 10.07 | 0.71 | 0.42 | 4.78 |
| `ghavamian` 0.15 keV | 4.50 | 0.80 | 0.57 | 3.65 |
| **`ghavamian` 0.30 keV (fiducial)** | **3.05** | **0.86** | **0.70** | 2.51 |
| `beta` = 0.03 | 3.16 | 0.87 | 0.70 | **2.39** |
| `equilibrated` (instant equipartition — hot bound) | 12.96 | 0.69 | 0.39 | 5.12 |

**Nothing in the family raises the soft band above 0.70 or lowers Fe-K below
2.39, and the fiducial is at or next to the optimum.** The EM-weighted kT_e is
*non-monotone* in the shock setting, with its minimum essentially at the
fiducial: halving the shock electron temperature *raises* it, 3.05 → 4.50 keV,
because T_e does not merely set the emitting temperature, it **weights** the
emission — colder electrons in freshly-shocked gas suppress its soft emission,
so the emission measure shifts onto the older, hotter, more equilibrated
interior. Every cell got cooler electrons and the mean went up. (`beta` = 0.03
also reproduces the published constant to <0.05 in five of six bands, so the
model is insensitive to *how* T_e is set as long as the EM-weighted value lands
near 3 keV.)
**But see §5.0:** XRISM says the IME electrons *are* ~2× too hot, so T_e is
closed as a knob and not as a discrepancy. §5.1 is the mechanism that moves it.

**6. The inner ejecta slope δ, and the unshocked mass — blocked, not unknown.**
This is the one gap where the fix is already identified and *fails to run*
(Result 7). The reverse shock has eaten too far into the ejecta *in mass*: 96 %
shocked against the observed ~87–90 %, i.e. 0.096–0.21 M☉ left unshocked against
a target of 0.35 ± 0.10 (DeLaney et al. 2014, Hwang & Laming 2012). One cause,
two symptoms — **shocked Si+S 3× too high (0.24 against 0.08 M☉) and r_RS
0.35 pc too far in**. The controlling parameter is the inner ejecta density index
δ, and in 1D δ = 1 (the standard core-collapse value) lands on the target *while
improving* r_FS. But **δ = 1 has never run in 3D**: the 256³ run with pistons,
shell and cooling dies on a timestep collapse at t = 0.019, most likely because
a centrally peaked core puts the Fe pistons inside denser material and pushes
the contrast over the crush threshold — the same mechanism that killed the
tabulated-radius and capped-radius piston variants. Until that is bisected,
**δ = 0 is the working configuration and the Si / r_RS discrepancy stands**.
*Cost:* one bisect session (the crash is reproducible and cheap at 256³); the
likely landing point is a piston-contrast gate rather than a solver change.

**7. Self-consistent CSM — and it now has two named targets.** The wind is an
imposed r⁻² law plus one smooth asymmetric shell. Two published measurements say
that is the wrong shape, not merely an idealisation:

* **The "Green Monster"** (Vink et al. 2024): dense *shocked CSM* projected onto
  the remnant's interior, pre-shock density **~12 cm⁻³** against our wind's
  0.93 cm⁻³ at 2.5 pc, n_e t = 1.5×10¹¹, v_r ≈ −2300 km s⁻¹. Their conclusion is
  a direct statement about our IC: the progenitor's environment "was not that of
  a smooth steady wind profile". It is now also implicated in the anomalous
  *western* Fe kinematics (Chandra + XRISM, 2026).
* **Reverse-shock asymmetry** (Fesen et al. 2025): 1900–2100 km s⁻¹ in the east
  but **stationary to −70 km s⁻¹ on the western limb**, with r_RS spanning
  90–130″. Our 1.531 pc = 93″ sits at the *inner edge* of that range, and the
  model has no east–west asymmetry at all.

`casa_wind.py` (built, smoke-tested at 64³, **not calibrated and deliberately
not wired in**) is the route. The blocker is a design problem, not plumbing: the
mapping stage consumes a 1D *spherical* profile and `ejecta_mass_coordinate` —
hence the whole composition model — rests on spherical symmetry. Adopting a 3D
CSM cube also requires re-deriving the calibration, since n_w is currently
*fitted*.

**8. Smaller items.** Fast ejecta knots beyond the forward shock (optically
bright, X-ray faint); dust destruction and IR emission; per-species ion
temperatures (XRISM measures T_i(Fe) − T_i(Si) = 150–300 keV; we carry one ion
temperature); multi-temperature and multi-ionisation structure within a cell.

### Measured negatives — do not re-litigate these

| tested | result |
|---|---|
| cosmic-ray back-reaction (γ_eff) | pushes r_RS the **wrong way**; supports η < 10⁻⁴ (Result 3) |
| magnetic fields | 20 µG (β = 3×10⁵): 0 %. 500 µG (25× over-strength): +2 % at 256³. Orlando et al. 2025 agree independently |
| Ni bubbles at 150 yr | null, gentle *and* sharp walls — the real mechanism acts just after breakout, not at 150 yr |
| clump amplitude **of the resolved log-normal** | χ saturates; coherence stalls at 0.787 vs 0.536; raising it degrades the hard continuum *and* drags r_RS off the observed value. **This does NOT close §5.1** — it says the *resolved, grid-clamped, species-independent* field cannot be pushed further, which is the reason §5.1 has to be sub-grid and two-phase |
| radially coherent plume field (`--plume-sigma`) | a modest **spectral** gain and essentially **no morphological** gain. Matched 256³ pair at `--coldcrush-factor 16`: rate 0.78 → 0.87, both soft bands improve, Fe-K improves substantially 3.05 → 2.54; but coherence is unchanged (0.865 → 0.858 at 4.4″) and the amplitude deficit gets slightly *worse* at every scale (1.83 → 2.02). r_RS moves the right way, 1.531 → 1.560. `--plume-vel` is a net negative — it adds 6.5 % kinetic energy, breaking the calibrated E, and does not change the outline |
| the plumes as a fix for the circular **outline** | refuted across four 256³ runs: the forward-shock position-angle spread is identical to three decimals (2.290–2.581, spread 0.292 pc). Ejecta structure does not reach the forward shock by 350 yr — the outline is set by the CSM shell and the wind, so it can only be fixed from the ambient side (§5.7) |
| cooling, cosmic Λ | changes nothing observable. **But Λ×10 with the guards off destroys the run** — metal-enhanced cooling is an *open* question |
| conduction (the Field length) | 15 days/run — unaffordable |
| projection / shell thickness | refuted: the emitting shell is already thin, dR/R = 0.12 at 109″ |
| dust halo as the outer flux | refuted: it puts 5.2 % beyond r_FS against Chandra's 8.8 %, in the wrong radial shape (observed r⁻⁴…⁻⁷ is too steep for any admissible halo). The halo is still **mandatory** — it just is not that flux (Result 13) |
| T_e, the entire admissible family | closed as a lever — §5.5. Non-monotone, fiducial at the optimum |

### Why the residual is not fitted away with gradient descent

`astronomix` is differentiable and `casa_diff.py` works, so the obvious question
is why the remaining parameters are not simply optimised against the images and
spectra. **Because the model is misspecified in ways that are now measured, not
suspected**, and an optimiser handed a biased forward model does not report the
bias — it absorbs it into whatever *is* free, and those are the parameters you
would then quote. Concretely: ~54 % of the 4–6 keV band is non-thermal and
absent (§5.0); the emitting density structure is below any affordable grid, so
⟨n²⟩ is biased low *by construction* (§5.1); and the T_e family is already
bracketed with the fiducial at its optimum (§5.5), so the only remaining slack
is in abundances, n_w and M_ej. On top of that, 200 yr of RT and shock–clump
interaction is chaotic, so gradients of image statistics are trajectory-specific
— which is why `casa_diff.py`'s JVPs agree with finite differences only to
0.2–37 %.

**Where it *is* well posed, and should be done: the 1D calibration.** There the
model — an adiabatic blast in a wind — really is the right physics, the gradient
is validated, and the target is precisely the correlated-residual problem that
one-at-a-time scanning cannot solve (§5.6: one cause, three symptoms, all
controlled by δ). Clear the recorded blocker first — reconcile the smooth `r_RS`
and `M_unshocked` with `casa_analyze.py`'s definitions — then fit 4–6 parameters
against 5–6 measurements.

A legitimate middle route is to make the **post-hydro** parameters
differentiable and fit *those* jointly: the clumping (χ, f), the non-thermal
normalisation and index, the T_e prescription, the `TRACER_SPLIT` ratios. They
are cheap, non-chaotic, and where the remaining freedom actually lives — but
they are **fitted interpretation parameters**, and any result must report their
count against the number of constraints.

---

## 6. Dead ends, and the mistakes that produced them

Every expensive mistake in this study has the same shape: **the run completed,
and printed a plausible number.** A crash is cheap. What cost weeks was code
that was silently not doing what its flag said, and metrics that ranked models
in the wrong order while looking rigorous. They are listed here because each one
was believed for a while, and any of them can come back.

### Silent bugs — the run finishes, the number is wrong

| what happened | what it looked like | how it was caught |
|---|---|---|
| **`POSITIVITY_HARD_FLOOR` manufactures mass**, 17 → 10¹² M☉ | "512³ is unstable" — a verdict that stood for weeks and was recorded in a handoff as settled | a mass-conservation check on a run that had *not* crashed. `--positivity redistribute` completes to 350 yr with mass conserved to 6e-5 |
| **`ejecta_radial_shape` clipped the profile to ≤ 1**, so `--inner-slope` did nothing | δ = 1 runs that "worked" and changed no observable | the unshocked mass refused to move. **Any δ result from before the fix is a δ = 0 result whatever the flag said** |
| **The LSRK fused Pallas path skipped the positivity flux limiter** — `use_fused_pallas` did not exclude the flux-blending flags, so `_blend_interface_flux` was never called | every `--low-mem` run blowing up, blamed on LSRK's non-SSP property | reading the dispatch, not the physics. LSRK + limiter is plausible again for 1024³, and untested |
| **`pyxsim.EventList` ignores the path it is given** and re-reads the filenames stored inside the HDF5 | the dust halo doing *nothing*: run completes, count rate identical to no-halo | the count rate was identical to the last digit, which is not what a physical process looks like. `apply_dust_halo` now rewrites `info/filenames` and asserts the round trip |
| **`casa_morph_null.py` v1 never thinned the real image**, though a comment said it did | a null table comparing Chandra at 143 ks against models at 20 ks — nulls 6.6 vs 16.8 | the numbers were too good. Every image is now thinned to matched counts |
| **The "sharded" turbulent field was not sharded** — it was a single-device array that replicated the whole cube the moment it multiplied a sharded density | 1024³ initial conditions that would not build, read as a memory-per-cell problem | threading `sharding` through the generators; `verify_sharded_turb.py` checks the field is bit-identical either way |
| **Multi-GPU was never broken.** It needed `AxisType.Auto` on the mesh (a jax 0.10 default drift) and `NCCL_NVLS_ENABLE=0` | "the solver does not shard", which put every rung above 512³ out of reach for months | trying the mesh option before touching the solver. Output is bit-identical to the single-device run |

The common lesson: **a completed run is not a passed run.** Mass, energy and the
scalar bounds are checked on every production run for this reason, and a result
that changes nothing at all is treated as a bug report, not as a null.

### Metrics that ranked models in the wrong order

* **Band-pass RMS `σ_I/I` cannot tell a filamentary web from a few sharp edges.**
  An edge is scale-free, so it deposits power in every octave. The statistic
  ranked the 192³ model — a smooth blob with two arcs — *best* at 1.46×, and the
  512³ contact-discontinuity model, which carries the cellular texture Cas A
  actually shows, *worse* at 1.80×. Adding the Euler characteristic and
  structure-tensor coherence **reversed three of Result 10's conclusions**:
  resolution does help monotonically, finer seeds are better not worse, and
  512³ CD is the best model. Result 10's seed-scale headline is withdrawn.
* **A resolution ladder that does not hold the seed fixed measures the seed.**
  The first ladder changed `k_hi` with N and produced a non-monotonic result
  that was read as "resolution does not help".
* **Topology needs matched noise, and a null.** χ is only comparable at matched
  exposure, which is why the real image is binomially thinned; and because finer
  structure lowers coherence for an almost geometric reason, "finer isotropic
  foam" and "filamentary web" move the same way until you subtract a
  phase-randomised null.
* **Quote a fraction against a radius, not an annulus edge.** The same image
  gives 13.5 % of the flux beyond 140″, 8.8 % beyond the observed forward shock
  at 153″ and 7.0 % beyond 160″.
* **`τ_sca ∝ E⁻²` is not usable in the soft band** (the local index wanders
  between −1.4 and −2.2 below 2 keV), and **single scattering is not good enough**
  at τ(1 keV) = 0.71, where a quarter of scattered photons scatter twice.

### Directions that were tried and are closed

The measured negatives are tabulated in §5 above and should not be re-run: cosmic
rays, magnetic fields, Ni bubbles imposed at 150 yr, raising the amplitude of the
resolved clumping field, the whole T_e family, the plume field as a fix for the
outline, cosmic-abundance cooling, conduction, a projection/shell-thickness
explanation for the smoothness, and the dust halo as the source of the outer
flux. **Note the one distinction that matters**: "raising the clump amplitude"
closes the *resolved, grid-clamped, species-independent* log-normal, not the
sub-grid two-phase contrast of §5.1, which is a different mechanism with a
different published calibration. Two more are structural rather than physical:

* **Building the remnant directly in 3D** (`cassiopeia_realistic.py`, the
  gallery track). It skips the phase that sets the answer — the wind holds an
  entire ejecta mass inside 1.5 pc — and its forward shock lands ~25 % small.
  That is what Route B and the 1D calibration exist to fix.
* **An in-house explosion.** Blocked by the degenerate EOS, not by effort; see
  §7.

---

## 7. Two hard walls

**Degenerate EOS.** A presupernova core is supported by electron degeneracy;
`astronomix` is ideal-gas. Ideal gas + radiation accounts for 2.7 % of KEPLER's
pressure at the centre, 22 % at the mass cut, 80 % only by 4 M☉. Feeding those
(ρ, p) to a γ = 5/3 solver describes a *different star* and it disassembles on
the first steps. This blocks in-house explosion modelling (Result 11) — and the
degenerate region is the part we do not need, which is why the EOS is not worth
building for this purpose.

**Resolution above 512³.** 566 bytes/cell/device (measured): 1024³ needs 8
GPUs. Multi-GPU itself is now fixed and verified (mesh `AxisType.Auto`,
`NCCL_NVLS_ENABLE=0`, and `sharding` threaded into the turbulent field —
bit-identical output), but the single H200 node is contended and a 1024³ run
could not be scheduled in 21 hours. 768³ additionally NaNs for reasons not yet
isolated. **512³ is the top of the verified ladder.**

---

## 8. Literature

**Setup and method** — Orlando et al. 2016 (Route B, Table 4 anisotropies),
2021 (Cas A as a fully developed remnant), 2022 (asymmetric CSM shell), 2025
(filamentary ejecta network; MHD has little effect on unshocked ejecta).
Wongwathanarat, Janka & Müller 2013/2015/2017 (3D neutrino-driven explosions to
shock breakout; PROMETHEUS-HOTB, excised PNS, grey ray-by-ray transport,
Yin-Yang grid — model `W15-2-cw-IIb` is what Orlando maps).
Sukhbold, Ertl, Woosley, Brown & Janka 2016 (200 KEPLER progenitors + calibrated
neutrino engines; **openly distributed**).

**Microphysics** — Schure et al. 2009 (cooling); Ghavamian et al. 2007
(electron heating at collisionless shocks); AtomDB/APEC; Timmes & Swesty 2000
(EOS); Draine 2003, Predehl & Schmitt 1995, Corrales et al. 2016 (dust
scattering).

**Observations** — Fesen et al. 2006 (age); Reed et al. 1995 (distance);
Hwang & Laming 2012 (shocked masses and abundances — the source of
`TRACER_SPLIT`); Laming & Hwang 2003 (ejecta overdensities ~100 needed to
reconcile n_e t with kT_e); DeLaney et al. 2010/2014 (3D structure, unshocked
ejecta mass); Milisavljevic & Fesen 2013 (3D kinematic map, 13 769 points);
Helder & Vink 2008 (~54 % of the 4.2–6 keV continuum is non-thermal,
remnant-integrated); Chandra ACIS evt2, 22 obsids 2000–2023, in
`/export/data/lstorcks/chandra_casa`.

**Added by the 2026-08-31 audit** — all published after the results in
`CALIBRATION.md`, and §5.0 is what they change:

* **Vink et al. 2026**, *Mapping plasma properties of Cassiopeia A with
  XRISM/Resolve: a Bayesian analysis via UltraSPEX* (arXiv:2602.06952, ApJ).
  **The key new reference.** Per-element-group kT_e, n_e t, σ_v, Doppler shifts,
  abundance ratios, non-thermal fraction and photon index; the n_e t–kT_e
  anticorrelation; and χ ≈ 10 (Fe-group) / ≈100 (IME) ejecta overdensities.
* **Vink et al. 2024**, *X-ray diagnostics of Cassiopeia A's "Green Monster"*
  (arXiv:2401.02491) — dense shocked CSM in the interior, ~12 cm⁻³ pre-shock.
* **Chandra + XRISM 2026**, *Three-dimensional expansion of iron ejecta in
  Cassiopeia A* (arXiv:2608.08412) — Fe to 5160 ± 320 km s⁻¹, layer inversion
  confirmed, western anomaly linked to the Green Monster.
* **Fesen et al. 2025**, *Cassiopeia A's reverse shock and its effects on the
  expanding SN ejecta* (arXiv:2501.07708) — optical r_RS 90–130″, 1000–2000
  km s⁻¹, stationary on the western limb; knot ablation tails.
* **XRISM 2025**, *Dynamics of the intermediate-mass-element ejecta*
  (arXiv:2503.23640) — pre-shock free-expansion velocities 2400–7100 km s⁻¹.
* **Orlando & Bocchino et al.**, REMLIGHT (arXiv:2012.13394) — the established
  recipe for synthesising synchrotron emission from an SNR MHD state (§5.3).

**Used by `_subgrid.py` and `_synchrotron.py`** — Klein, McKee & Colella 1994
(cloud crushing, and the pressure-equilibrium closure the two-phase split rests
on); Zirakashvili & Aharonian 2007 and Vink 2012 (the loss-limited cutoff and its
field-independence); Baars et al. 1977 (Cas A's 2720 Jy at 1 GHz, α = 0.77, the
anchor that removes the electron efficiency).

**Differentiable-solver prior art** — `diffhydro` (arXiv:2512.13403):
PDE-constrained inference in astrophysical flows, reverse mode with custom
FFT/multigrid adjoints and checkpointing through ~10³ steps at ≤512³. Its own
framing — "demonstrations with synthetic targets, not recovery of unknown real
systems" — is the reason `casa_diff.py` is aimed at the 1D calibration and not
at the images; see the note at the end of §5.

---

## 9. Reproducing

### Prerequisites — three things that live outside this directory

1. **Two environments.** The solver runs in the `astx` mamba env, wrapped by
   `./run.sh` (which strips the broken CUDA paths and puts the working tree
   ahead of the installed wheel). The observation runs in a *separate* CPU venv
   at `/export/home/lstorcks/xrayobs` — pyXSIM/SOXS/AtomDB, no JAX. They are not
   interchangeable and nothing warns you if you swap them.
2. **The real Chandra data.** `--compare <year>` reads pre-binned epoch images
   from `/export/data/lstorcks/chandra_casa/epoch_images/`, produced by
   `make_epoch_images.py` in that directory from 22 obsids of ACIS `evt2`. That
   pipeline is **outside this repository**; without it every `--compare`,
   `casa_morphology.py` and `casa_morph_null.py` call below has nothing to
   compare against.
3. **The Mie scattering table**, built once (~90 s on 40 cores) and cached next
   to the states. `--halo` cannot run without it:
   ```bash
   /export/home/lstorcks/xrayobs/bin/python _dusthalo.py --rebuild --selftest
   ```

### The fiducial

```bash
cd examples/gallery/supernova_showcase

# stage 1 — the calibration (seconds, CPU)
./run.sh casa_calibrate_1d.py --n 4000 --energy-51 2.09 --ejecta-mass 3.0 \
    --n-w 0.928 --age 150 --save-profile casa_1d_map150.npz

# stage 2 — the top rung: 512³, CD-seeded, ADIABATIC. --cooling is off
# deliberately: at 512³ the reverse shock meets a fully resolved cold structure
# and runs away radiatively (Result 10), and cooling changes no observable
# (t_cool = 2.8 Myr). ~76 GB at the measured 566 B/cell, so one large GPU.
pq sub -t a100 --gpus 1 -- ./run.sh casa_orlando.py casa_1d_map150.npz \
    --n 512 --shell --pistons --composition --positivity redistribute \
    --clump-region ejecta --save-state orl_n512_cd_adia.npz

# stage 3 — the observation, in the OTHER env. --halo is not optional (§3).
S=/export/data/lstorcks/supernova_showcase
/export/home/lstorcks/xrayobs/bin/python casa_observe.py $S/orl_n512_cd_adia.npz \
    --exposure 20 --nei --halo --compare 2004 --out orl_n512_halo_uniform

# stage 4 — the scores
/export/home/lstorcks/xrayobs/bin/python casa_morphology.py \
    orl_n512_halo_uniform_synimg.npz --compare 2004
/export/home/lstorcks/xrayobs/bin/python casa_morph_null.py \
    orl_n512_halo_uniform_synimg.npz --compare 2004
```

Swap `--n 512` for `--n 256` and add `--cooling` for the 25-minute 256³ rung —
that is the configuration the plasma and dynamics numbers in §4 were measured
on, and the cheap one to iterate against. Re-observing a state that has already
been photon-projected is far cheaper with `--pyxsim-events <cached>.h5`, which
reuses the photon list and re-runs only the sightline and the telescope; that is
how the halo variants in Result 13 were compared photon-for-photon.

**Guardrail — every change in this line of work is scored on all three, and
against the state that owns the number (§4):**

* **morphology** (n512 CD): σ_I/I 1.80× at 4.4″, χ 0.84 at 2.5″, coherence
  excess +0.56 against Chandra's +0.36, Euler excess −13.06 against −10.30;
* **spectrum** (n512 CD, **halo on**): rate 0.73× and the six band ratios
  0.46 / 0.62 / 1.02 / 1.56 / 1.71 / 2.42. *(The n256 no-halo pair — 0.86× and
  0.70 / 0.74 / 1.16 / 1.64 / 1.60 / 2.51 — remains the guardrail while
  iterating at 256³. Never compare one against the other.)*
* **dynamics** (n256): r_FS 2.494, r_RS 1.531, shocked ejecta 2.79 M☉,
  kT_e 3.05 keV, n_e t 2.3e11.

A change that improves the picture without improving a statistic is a failure,
not a success.
