# Supernova showcase

> **If you came for the calibrated Cas A model, start with
> [`OVERVIEW.md`](OVERVIEW.md), not here.** It maps the two-stage pipeline, what
> physics each piece contains, where the model stands against Chandra and what
> is missing; [`CALIBRATION.md`](CALIBRATION.md) holds the derivations
> (Results 1–13). **This README covers the older *showcase* scripts**
> (`cassiopeia*.py`, `snr_sedov.py`, `young_snr_ism.py`), which are tuned to
> look right rather than to agree with data, plus the numerics and the pq
> workflow that both tracks share. Where it says something is "not modelled",
> that is a statement about the showcase scripts — the calibrated track has
> since added composition, NEI, the electron temperature and the dust halo.

Three supernova-remnant setups, all solved with the **same consistent
high-order finite-difference scheme** (WENO reconstruction + RK4-SSP time
integration), running in single precision (float32):

| script | problem | ambient | reference |
|---|---|---|---|
| `snr_sedov.py`    | Sedov-Taylor thermal-bomb SNR | uniform box | Guo, Kim & Stone (2025) `snr.athinput` |
| `cassiopeia.py`   | ejecta-driven remnant of **Cassiopeia A** (idealized) | r⁻² progenitor wind | Orlando et al. (2021), A&A 645, A66 |
| `young_snr_ism.py`| the same ejecta as a contrast | uniform ISM | — |
| `cassiopeia_realistic.py` | **Cas A with realism**: clumpy ejecta, dense asymmetric CSM shell, cooling (+ optional conduction) | turbulent wind + shell | Orlando et al. (2021, 2025) |

Shared code-unit conventions, the solver configuration and the initial-condition
builders live in `_common.py`. Each script writes a figure to `figures/`.

The four form a progression: an idealized Sedov bomb, an idealized ejecta
remnant (Cas A) and its uniform-ISM contrast, and finally a physically
enriched Cas A that adds the ingredients that make young remnants look the way
they do in observations.

## The calibrated Cas A pipeline

The four scripts above are **showcases**: they are tuned to look right, not to
agree with data. A second, separate track does the Orlando et al. programme
properly — calibrate, then predict, then observe:

| script | role | runs on |
|---|---|---|
| `casa_calibrate_1d.py` | **Stage 1.** 1D spherical ejecta-into-wind from 0.05 pc, calibrated against `r_FS`, `r_RS`, `v_FS`, `v_RS` and the post-shock density. Scans and solves the degenerate `(E_SN, M_ej, n_w, γ_eff)` block. | CPU, seconds |
| `casa_orlando.py` | **Stage 2.** Maps the calibrated 1D profile into 3D at ~150 yr and imposes the multi-D structure there: Orlando (2012) ejecta clumping, the Table-4 pistons (`--pistons`), the Orlando (2022) asymmetric CSM shell (`--shell`), placed *ahead* of the blast so the interaction is computed. | GPU, hours |
| `casa_analyze.py` | Scores a saved state: angle-averaged `r_FS`/`r_RS` (same criteria as the 1D run), `r_FS` versus position angle, radial profile against the 1D solution. | CPU |
| `casa_plasma.py` | Non-equilibrium diagnostics from the shock history: ionization age `n_e t`, electron and ion temperatures, shocked ejecta/Fe/Si masses, and the EM(`kT_e`, `n_e t`) distribution Hwang & Laming fit the real data in. | CPU |
| `casa_observe.py` | **Real synthetic observations.** pyXSIM (AtomDB/APEC, per-cell simulated abundances) → SIMPUT → SOXS with the actual ACIS ARF/RMF/PSF/backgrounds → a Chandra event file, binned onto the same sky grid as the real `evt2` data and compared in counts/s. | CPU, separate `xrayobs` venv |
| `_plasma.py`, `_nei.py` | Composition → μ, μ_e and the electron temperature (Coulomb equilibration from the shock history), and (kT_e, n_e t) → non-equilibrium ion fractions. Shared by `casa_plasma.py` and `casa_observe.py`. | CPU |
| `_dusthalo.py` | The interstellar dust-scattering halo, applied to the photon list between absorption and the telescope (`--halo`). Mie on an MRN grain population; the column follows the same N_H that absorbs, so nothing is fitted. Build the table once with `--rebuild`. | CPU |
| `casa_morphology.py` | Structure vs scale against the real `evt2`: band-passed σ_I/I with the Poisson term subtracted analytically, plus Euler characteristic and structure-tensor coherence on a noise-matched pair. | CPU |
| `casa_morph_null.py` | The phase-randomised null for those statistics — same power spectrum, no topology — so the *excess* separates real structure from "the same power spectrum". | CPU |

### Composition and shock history (`--composition`)

`casa_orlando.py --composition` turns on the solver's passive scalars and
carries, per parcel:

* `C_ej` — an ejecta/circumstellar discriminator, so "shocked ejecta mass" is a
  measurement rather than a density-and-temperature guess;
* `C_Fe`, `C_Si`, `C_O`, `C_He` — the Type IIb chemical stratification, laid
  down at the mapping time by *enclosed ejecta mass coordinate* (which a 1D
  spherical flow preserves exactly, so it is the Lagrangian label) and then
  mixed by the 3D instabilities. Hydrogen is the remainder.
* `entropy_initial`, `time_since_shock`, `density_time` — the library-managed
  shock bookkeeping, from which the ionization age and the electron/ion
  temperature relaxation follow (Dwarkadas, Dewey & Bauer 2010).

These are what let `casa_observe.py` give pyXSIM the abundances the simulation
actually produced, per cell and per element, instead of one assumed metallicity.
See `tests/passive_scalars/validate.py` for the accuracy, boundedness,
conservation and passivity checks.

`CALIBRATION.md` records what the calibration established — including that
E ≈ 2 × 10⁵¹ erg is what the observations want (it had been reverted for
numerical reasons), and that cosmic-ray back-reaction *cannot* explain the
`r_FS`/`r_RS` ratio.

Note that `casa_observe.py` runs in `/export/home/lstorcks/xrayobs`, not `astx`:
yt/pyxsim/soxs bring their own numpy, and the X-ray post-processing has no
reason to share an environment with the solver.

## Running

Every script is self-contained. GPU selection is two-tier: if
`CUDA_VISIBLE_DEVICES` is already set (a scheduler like `pq` pins the assigned
GPUs there) it is respected as-is; otherwise the script grabs a free GPU with
`autocvd`. Interactively:

```bash
./run.sh cassiopeia.py            # 128^3, a few minutes on one GPU
./run.sh snr_sedov.py --n 256     # sharper, slower
./run.sh young_snr_ism.py --mhd   # run the FD backend in MHD mode (B = 0)
```

(`run.sh` cleans the CUDA env and imports the working tree instead of any
installed astronomix wheel.)

On the cluster, submit through the `pq` queue instead of running on the login
node — `pq stat` shows free GPUs per node, `pq log <jobid>` follows the output.
The progress bar detects the non-interactive stdout and logs one plain
progress line per 0.5 % instead of terminal bar frames. A hero-resolution
example (the 512³ dual-energy Cas A):

```bash
pq sub -t h200 -n 4 --name casa512 -- \
    env XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    ./run.sh cassiopeia_realistic.py --n 512 --t-end 0.25 --nsnap 21 \
    --dual-energy --jet --gpus 4 --save-state casa_n512_jet_dual_x32.npz
```

(Preallocation matters at the memory edge: on-demand allocation fragments the
cards and a run OOMs even though it fits.) Do NOT use `--low-mem` for the
blast runs without re-testing it: LSRK4 inflated the energy at N=64 and
collapsed `dt` to zero at the blast/shell interaction (t≈0.02), which was
attributed to the integrator not being SSP while the positivity-flux limiter's
guarantee needs SSP stage convexity. **That diagnosis was only half right.** The
fused Pallas path did not exclude the flux-blending flags, so every `--low-mem`
run silently skipped the positivity limiter altogether (`OVERVIEW.md` §6). The
exclusion is fixed; LSRK4 + limiter is plausible again and remains untested, so
prefer GPUs the SSP integrator fits on (~58 GiB/GPU at 512³ across 4 GPUs) until
someone measures it. The queued-log progress lines
include the per-step `dt`, and the time loop aborts with a loud
`[astronomix] ABORT` line if the clock ever stops advancing, so a dt collapse
fails fast instead of spinning on the GPUs.

Common flags: `--n` (cells per axis), `--t-end` (evolution time), `--mhd`
(MHD-mode FD path, B = 0). `snr_sedov.py` additionally takes `--no-cooling`.
`cassiopeia_realistic.py` adds `--dual-energy`, `--low-mem`, `--gpus N`
(x-axis domain decomposition; N must divide 8), and `--save-state <npz>`.

### Offline re-imaging (`reimage.py`)

Hero runs only need to write the `--save-state` npz; all figures can be
(re)produced afterwards without a GPU and without re-running:

```bash
python reimage.py casa_n512_dual_x32.npz --prefix cassiopeia_n512
```

writes `figures/cassiopeia_n512_{realistic,xray,composite}.png`. This is also
how the imaging (colour balance, bands, stretch) is iterated on.

## The numerics: a consistent high-order scheme for a strong blast

Athena's `snr.athinput` uses `ppm4 + hllc` finite-volume reconstruction with a
first-order flux correction (`fofc`). Here every setup instead uses astronomix's
high-order WENO **finite-difference** solver — no finite-volume path and no
first-order fallback.

A Sedov-strength thermal bomb (pressure contrast ~10⁶) or cold freely-expanding
ejecta (kinetic energy ≫ internal energy) makes a pure high-order WENO scheme
NaN within a handful of steps — in *both* single and double precision, so it is
not a floating-point problem but a missing positivity mechanism. Two ingredients
fix it while keeping the scheme high-order:

1. **Positivity-preserving flux limiter** (`PositivityConfig(preserving_flux=True)`):
   the Hu–Adams–Shu / Zalesak FCT limiter blends each WENO interface flux toward
   the first-order Lax–Friedrichs flux by the *minimal* amount that keeps density
   and pressure positive. This is a high-order positivity technique, **not** the
   FOFC / first-order fallback. It makes all setups here stable and
   energy-conserving in float32.
2. **A well-resolved, tanh-tapered injection region** with exact mass/energy
   renormalisation (a single-cell top-hat NaNs regardless of the limiter).

### Precision and the dual-energy formalism (`--dual-energy`)

The blast instability itself is not a precision effect — the positivity-flux
limiter fixes it in both precisions. But the cold, un-shocked, freely-expanding
ejecta **core** is: the pressure recovered as `p = (γ−1)(E − E_kin)` there is a
tiny difference of large numbers (catastrophic cancellation). At moderate
resolution this only corrupts the (dynamically negligible, cosmetically dark)
core temperature; globally x32 and x64 agree to a median Δp/p ≈ 1.6e−5, so the
smooth showcase setups at N ≤ 128 simply run in plain x32.

At high resolution the cancellation becomes fatal for the realistic setup:
x32 `cassiopeia_realistic.py` **vacuum-collapses at N ≥ 240** (the corrupt
core pressure cascades through the floor-injection feedback), and even where
it survives, the non-dual energy recovery sits on a compile-choice-sensitive
knife edge (bit-identical reruns can fall into either a flat or an
energy-jumping attractor, in both precisions).

The fix is the **Bryan et al. (1995) dual-energy formalism**
(`--dual-energy`): an internal-energy density `g` is advected alongside the
conserved variables (upwind `div(g v) − p div v`, floored, re-synced to
`p/(γ−1)` where the total-energy recovery is reliable), and the pressure
switches to `g` exactly in the cancellation regime (`e/E < η = 1e−3`). With it
the N = 256 x32 run matches x64, every dual run has a strictly monotone energy
series (the knife edge disappears), and the core temperature is physical —
which is also what quantitative cold-ejecta synthetic observables need.
The Pallas WENO kernels carry ``g`` natively (hydro and MHD): the dual field
rides the same multi-GPU halo exchange as the state and the Bryan+95 switch
runs inside the kernels, so dual energy costs neither the Pallas throughput
nor its memory advantage (validated by
``tests/dual_energy/pallas_dual_validate.py``).

## athinput → astronomix mapping (`snr_sedov.py`)

Faithfully reproduced from `snr.athinput`:

- triply-periodic Cartesian box, side 64 pc (`[-32, 32]³`);
- ideal gas, γ = 5/3, CFL = 0.3;
- code units 1 pc / (n = 1 cm⁻³ mass) / 1 Myr, μ = 0.618;
- ambient `damb = 0.862`, Athena code temperature `tamb = 93.13` (T = p/ρ);
- SNR = `etot_snr` ≈ 3.47e9 (= 1e51 erg = 1 bethe) thermal energy + `mass_snr`
  deposited in a central sphere;
- density/pressure floors `dfloor = 1e-4`, `pfloor = 1e-2`;
- radiative cooling (`ism_cooling`): the Schure et al. (2009) ISM cooling curve
  via the unconditionally-stable implicit method (`--no-cooling` disables it).

Deviations (none change the Sedov dynamics):

- **high-order WENO instead of `ppm4 + hllc + fofc`** — the point of the showcase;
- the injection sphere is a resolved, tanh-tapered 3 pc region (energy/mass
  exact) rather than the sharp 1 pc top-hat;
- no thermal conduction (the athinput sets `conductivity = 0` anyway) and no
  photoelectric heating term (astronomix has none; the temperature floor stands
  in for the heating balance);
- default `t_end = 0.01 Myr` keeps the blast inside the periodic box (the
  athinput's `tlim = 1 Myr` would overrun a 64 pc periodic domain);
- no passive scalars (`nscalars = 4` in the athinput) — astronomix has no
  passive-scalar API.

## Cassiopeia A (`cassiopeia.py`, Orlando et al. 2021)

Adopted from Orlando et al. (2021, Table 1):

- explosion (kinetic) energy **1.5 × 10⁵¹ erg** (= 1.5 bethe);
- ejecta mass **3.3 M⊙**;
- r⁻² progenitor wind `n(r) = n_w (r_fs/r)² + n_c` with `n_w = 0.8 cm⁻³` at
  `r_fs = 2.5 pc`, flattening to `n_c = 0.1 cm⁻³`.

Simplification: Orlando et al.'s ejecta comes from a full 3D neutrino-driven
core-collapse simulation (asymmetric, with radioactive-decay heating and, in
their MHD runs, a magnetic field). Here the ejecta is the standard **analytic
freely-expanding profile** — a flat inner core joined to a steep `ρ ∝ r⁻⁹`
envelope, spherically symmetric, homologous (`v ∝ r`), renormalised to the same
mass and energy. What it reproduces is the essential Cas A gas dynamics: a
forward shock into the wind, a reverse shock driven back into the ejecta, the
bright shocked main shell, and the cold expanding interior at an age of ~350 yr.
`young_snr_ism.py` runs the identical ejecta into a uniform ISM to show how the
ambient profile reshapes the remnant.

## Realistic Cas A (`cassiopeia_realistic.py`)

The idealized `cassiopeia.py` is a smooth, spherical, adiabatic remnant. The
realistic variant adds the physics that shapes a real young core-collapse SNR,
following the Orlando et al. (2021, 2025) Cas A models:

- **radiative cooling** — Schure et al. (2009) ISM curve, implicit; radiative
  losses in the shocked shell drive it toward thin, dense filaments;
- **a dense, asymmetric circumstellar shell** at ~1.7 pc — the pre-supernova
  eruptive-mass-loss shell Orlando et al. (2025) invoke for Cas A's "Green
  Monster" (they quote n ~ 180 cm⁻³, ~2× the shocked-shell density; softened to
  ~60 cm⁻³ here for the showcase grid), lopsided toward +x;
- **a turbulent / clumpy medium** — band-limited log-normal density fluctuations
  in the wind and shell, so the circumstellar gas is structured;
- **clumpy ejecta** — fractional density perturbations that grow into
  Rayleigh–Taylor fingers on contact with the reverse shock and the dense shell;
- **optional thermal conduction** (`--conduction`) — isotropic constant-κ; off
  by default because the explicit parabolic timestep in the near-vacuum hot
  bubble is expensive.

The ejecta perturbations are deliberately **large-scale-dominated** (low
wavenumber, steep red spectrum) — a few big plumes plus smaller clumps — rather
than fine speckle, matching the Orlando et al. picture in which the remnant
morphology is set by large-scale explosion asymmetries interacting with the
reverse shock. The instabilities and filaments are resolution-limited, so the
structure sharpens with `--n` (128³ shows the clumpy shell and onset of
fingering; 256³ resolves the filaments far better).

### Synthetic Chandra X-ray images

`cassiopeia_realistic.py` writes two Chandra-style views (also produced offline
by `reimage.py`):

**`cassiopeia_xray.png`** — a science-colour composite following the labeled
Chandra Cas A image: **red** = low energies (the Fe/Mg-dominated
shocked-ejecta lines; here the soft thermal band), **green** = intermediate
energies (Si; the medium thermal band), **blue** = the highest energies —
which in Cas A are **synchrotron emission** from shock-accelerated electrons,
not hot thermal gas. The nonthermal proxy localises emission at shocks with a
cell-scale pressure-jump detector scaled by the post-shock pressure, so the
blast wave lights up as the **thin arcs** seen in the real remnant, and the
shocked dense CSM reads blue-ish against the red/green ejecta shell — the
spectral distinction Vink et al. (2024) use to identify the "Green Monster"
as shocked CSM. Plus a broadband surface-brightness map.

**`cassiopeia_chandra.png`** — a deep single-band "press image": blue-on-black
full-bleed view with knot-weighted emissivity (the real blue image is
dominated by line emission from dense ejecta knots), a multi-scale unsharp
mask that renders the limb-brightened shell as thin tangled filaments, the
synchrotron component for the faint outer blast-wave rim, and an asinh
stretch.

All emissivities are visualisation proxies (thermal: ε ∝ n_e n_H √T), not
spectral-model synthetic observations (no NEI, no lines, no instrument
response). Frames are auto-cropped to the emitting remnant.

### X-ray + infrared composite (Chandra + JWST style)

`cassiopeia_realistic.py` also writes `figures/cassiopeia_composite.png` — three
panels (X-ray, infrared, and their overlay) emulating the multiwavelength Cas A
composites (e.g. the 2024 Chandra + JWST image). The channels are physically
**complementary**:

- **X-ray (blue)** traces the hot, diffuse, shock-heated plasma (T ≳ 10⁶ K)
  plus the thin synchrotron blast-wave arcs;
- **infrared, warm dust (gold)** traces dust in the dense, cooler shocked
  gas — the swept-up CSM shell and dense ejecta knots/filaments
  (ε_IR ∝ n_H² √T · exp[−(T/T_sput)^1.5]: suppressed where dust is sputtered
  in the hottest plasma);
- **infrared, pristine debris (deep red)** traces the cold **un-shocked**
  ejecta interior to the reverse shock (ε ∝ n_H² e^{−T/3·10⁴ K}) — the cool
  debris web Webb sees inside Cas A. This channel is only physically
  meaningful with `--dual-energy`: without it the cold interior's float32
  temperature is cancellation noise.

### The "Green Monster" (shocked near-side CSM)

The realistic setup's dense asymmetric shell is exactly the kind of structure
that Chandra + Webb identified as the origin of Cas A's "Green Monster": a
dense circumstellar shell from a pre-supernova mass-loss episode, shocked by
the blast wave, sitting on the **near side** of the remnant so it is seen
projected against the interior (Vink et al. 2024: pre-shock n ≈ 12 cm⁻³,
blueshifted −2300 km/s, X-ray properties matching the shocked CSM / blast
wave, not the Si/Fe ejecta). To view the analogue the same way, project along
the shell's asymmetry axis: `reimage.py --los x` puts the shocked dense shell
in front of the interior, where it appears as a filamentary dust structure
crossing the remnant's face in the IR panel. (The real GM is also pockmarked
by round holes from ejecta knots punching through it — a resolution-limited
effect to look for in the hero runs.)

Not modelled *by this script*: elemental abundances / NEI spectra (no passive
scalars or nucleosynthesis network), so the Fe/Si vs CSM abundance distinction
is only mimicked by the band proxies, and the radioactive Ti/Ni asymmetries that
shaped the pristine debris are out of scope. **The calibrated track has since
closed the first two**: `casa_orlando.py --composition` carries a four-element
stratification as passive scalars and `casa_observe.py --nei` builds the
non-equilibrium ion fractions from each parcel's (kT_e, n_e t) — see
`OVERVIEW.md` §3.

`--save-state <npz>` dumps the final density/pressure **and velocities** so
all images can be recomposed offline (`reimage.py`: colour balance, bands,
stretch, line of sight) without re-running; velocities also enable kinematic
views (e.g. the GM's blueshift) later.

What this *showcase* cannot represent (and the Orlando models include): a genuine
3D core-collapse ejecta structure, multiple chemical species / passive scalars,
radioactive-decay heating, non-equilibrium-ionization cooling, and
temperature-dependent/saturated conduction — so this is a gas-dynamics likeness,
not a spectral synthetic observation.

**Two of those are no longer library limitations.** Passive scalars and
NEI post-processing both exist and are used by the calibrated track; conduction
is implemented but costs ~15 days per run. What remains genuinely out of reach
is the explosion-era ejecta structure — see `OVERVIEW.md` §5 and §7, which also
records why an in-house explosion needs a degenerate EOS the solver does not
have.

## Comparison with the real observations (`compare_real.py`)

`compare_real.py <npz> --los x` builds a side-by-side figure
(`figures/<prefix>_vs_real.png`) of the synthetic deep X-ray and X-ray+IR
composite against the real Chandra 2024 press images (`real_obs/`), and
measures the forward/reverse shock radii from the state against the observed
values (FS ≈ 2.5 pc, RS ≈ 1.6 pc at 3.4 kpc). Known gaps of the 512³ model
vs the real remnant: knot-scale (0.02–0.05 pc) line-emitting ejecta clumps
(resolution + large-scale-only clump seeding + no NEI line emission), the
ring/bubble interior morphology (no radioactive Ni-bubble structure in the
ICs), a ~25 % undersized forward shock (CSM normalization / age mapping),
global asymmetries from the explosion engine (Fe-outside-Si overturn), and
observational forward-modelling (absorption, PSF, photon noise).

Every gap in that list except the knot scale is closed on the calibrated track:
`casa_calibrate_1d.py` fixes r_FS and r_RS against the measurements, the Table-4
pistons produce the Fe-outside-Si inversion, and `casa_observe.py` does the
forward modelling with the real ACIS response. Compare against `OVERVIEW.md` §4
before quoting any number from this section.

## Source material

The Athena input files and the Orlando et al. papers live in the sibling
`cassiopax/` repository (`athena_inputs/`, `literature/`).
