# Cassiopeia A end to end: what the model contains, and what it does not

*A map of the pipeline for someone arriving cold. `CALIBRATION.md` has the
measurements and their derivations (Results 1–13); this says how the pieces fit
together, what physics is in each, and — the part that matters most — what is
still missing and why.*

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
```

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
- **T_e ≠ T_i**: electrons take ~0.3 keV at the shock (Ghavamian et al. 2007)
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

| quantity | model | observed |
|---|---|---|
| r_FS | 2.494 pc | 2.52 ± 0.20 |
| r_RS | 1.531 pc | 1.58 ± 0.16 |
| shocked ejecta | 2.79 M☉ | 2.8–3.7 |
| EM-weighted kT_e | 3.05 keV | — |
| EM-weighted n_e t | 2.3e11 | ~1e11 |
| 0.5–7 keV rate | 0.86× | 1.0 |
| bands (0.5–1.5 … 6–7) | 0.70 / 0.74 / 1.16 / 1.64 / 1.60 / 2.51 | 1.0 |
| structure ≥ 8″ | matched | — |
| structure at 2.5″ | ~3× too smooth | — |

**Independent support:** a real KEPLER progenitor (Sukhbold et al. 2016) at
16 M☉, stripped to IIb with its iron core excised, gives **3.56 M☉** of ejecta
against the **3.0 M☉ we fitted** — 18 %, unprompted (Result 11).

---

## 5. What is missing, and what it would cost

### Missing physics, ranked by how much of the residual it explains

**1. Explosion-era structure — the dominant gap.**
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
rewrite (6–12 months) plus a degenerate EOS — **not recommended**; see §6.

**2. Non-thermal synchrotron.**
Absent entirely. Costs the 140–180″ rim (a 3–5× residual that the dust halo
demonstrably cannot absorb) and, more importantly, means part of the *real*
4.2–6.0 keV flux is non-thermal — so our 1.64–1.71× overprediction of that
band understates the true thermal excess, sharpening the T_e problem.
*Cost:* 1–2 weeks for a DSA post-process on an MHD state. **It is a fit, not a
prediction:** Cas A's rim field is ~0.5 mG against our compressed 80 µG, so
field amplification enters as a free parameter.

**3. T_e — the headline open residual.**
The Ghavamian prescription is calibrated on H-dominated ISM shocks and
extrapolated to a reverse shock in metal ejecta. The signature to explain is
*too little soft line emission and too much hard continuum* — i.e. too much hot
gas, which is a T_e question, not a structure question.

**4. Self-consistent CSM.** The wind is an imposed r⁻² law plus one shell.
`astronomix` has a stellar-wind module; blowing the bubble self-consistently
would replace both, but would also require re-deriving the calibration (n_w is
currently *fitted*).

**5. Smaller items.** Fast ejecta knots beyond the forward shock (optically
bright, X-ray faint); dust destruction and IR emission; multi-temperature and
multi-ionisation structure within a cell; `TRACER_SPLIT` — four tracers stand
for nine elements, so relative abundances *within* a nucleosynthetic layer are
assumed, not simulated.

### Measured negatives — do not re-litigate these

| tested | result |
|---|---|
| cosmic-ray back-reaction (γ_eff) | pushes r_RS the **wrong way**; supports η < 10⁻⁴ (Result 3) |
| magnetic fields | 20 µG (β = 3×10⁵): 0 %. 500 µG (25× over-strength): +2 % at 256³. Orlando et al. 2025 agree independently |
| Ni bubbles at 150 yr | null, gentle *and* sharp walls — the real mechanism acts just after breakout, not at 150 yr |
| clump amplitude | χ saturates; coherence stalls at 0.787 vs 0.536; raising it degrades the hard continuum *and* drags r_RS off the observed value |
| cooling, cosmic Λ | changes nothing observable. **But Λ×10 with the guards off destroys the run** — metal-enhanced cooling is an *open* question |
| conduction (the Field length) | 15 days/run — unaffordable |
| projection / shell thickness | refuted: the emitting shell is already thin, dR/R = 0.12 at 109″ |

---

## 6. Two hard walls

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

## 7. Literature

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
Hwang & Laming 2012 (shocked masses and abundances); DeLaney et al. 2010/2014
(3D structure, unshocked ejecta mass); Milisavljevic & Fesen 2013 (3D kinematic
map, 13 769 points); Chandra ACIS evt2, 22 obsids 2000–2023, in
`/export/data/lstorcks/chandra_casa`.

---

## 8. Reproducing

```bash
cd examples/gallery/supernova_showcase
./run.sh casa_calibrate_1d.py --n 4000 --energy-51 2.09 --ejecta-mass 3.0 \
    --n-w 0.928 --age 150 --save-profile casa_1d_map150.npz
pq sub -t a100 --gpus 1 -- ./run.sh casa_orlando.py casa_1d_map150.npz \
    --n 256 --shell --pistons --composition --positivity redistribute \
    --clump-region ejecta --save-state orl_n256.npz
/export/home/lstorcks/xrayobs/bin/python casa_observe.py orl_n256.npz \
    --nei --halo --compare 2004                 # separate CPU venv
/export/home/lstorcks/xrayobs/bin/python casa_morphology.py obs_*_synimg.npz --compare 2004
/export/home/lstorcks/xrayobs/bin/python casa_morph_null.py obs_* --compare 2004
```

**Guardrail — every change in this line of work is scored on all three:**
morphology (σ_I/I, χ, coherence *and* their phase-randomised excess), spectrum
(rate 0.86× and the six band ratios), and dynamics (r_FS 2.494, r_RS 1.531,
shocked ejecta 2.79 M☉, kT_e 3.05 keV, n_e t 2.3e11). A change that improves
the picture without improving a statistic is a failure, not a success.
