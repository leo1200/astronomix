# The small-scale dynamo, with resolution: astronomix vs AthenaPK

Driven, subsonic, isothermal MHD turbulence in a unit periodic box, started from
a seed field so weak (`beta_p = 1e6`) that every bit of magnetic energy in the
final state was made by the turbulence. Run to saturation at `N = 64, 128, 256`
in three schemes:

* **astronomix** — 5th-order finite-difference WENO5, characteristic-wise flux
  splitting, SSP-RK4, **constrained transport** (`dynamo_convergence.py`),
* **AthenaPK PLM + VL2 + HLLD** — 2nd order, GLM divergence cleaning, and
* **AthenaPK PPM + RK3 + HLLD** — 3rd order, GLM divergence cleaning
  (both `athenapk_turb.py --scheme plm|ppm`).

The question is what each scheme's *numerical dissipation* does to a dynamo:
the small-scale dynamo lives at the grid scale, so it is the sharpest available
probe of the difference between two discretisations, and everything it does is
a function of resolution.

## The setup

The physical setup is the **ICM case of `paper_turbulence.py`** already in this
directory (HOW-MHD, arXiv:2304.04360, section 3.10), unchanged apart from being
run for 40 crossing times instead of 30 so that every scheme in the ladder
reaches saturation:

| | |
|---|---|
| box | unit periodic cube, `rho0 = 1` |
| EOS | isothermal, `a = 1 / M_turb = 2` |
| seed field | uniform `B0 = sqrt(2 a^2 rho0 / beta) = 2.83e-3` along z, `beta_p = 1e6` |
| driving | solenoidal Ornstein-Uhlenbeck, `tau = 0.5`, peak at `n = kL/2pi ~ 2` |
| run length | `40 t_cross`, `t_cross = (L/2) / v_rms = 0.5` |
| achieved flow | `M_turb ~ 0.68` while the field is passive, `~ 0.6` once it saturates |

Subsonic and isothermal for the same reasons as the hydrodynamic study next
door (`../../hydro/turbulence`): no shocks, so the comparison measures
reconstruction and induction rather than shock capturing, and a fixed sound
speed so the Mach number does not drift over a 40-crossing-time average.

### How the two codes are matched

AthenaPK has no isothermal EOS, so the isothermal box is emulated with
`gamma = 1.0001` and `p0 = rho0 a^2 / gamma` — the standard AthenaPK turbulence
setup. It works: over the whole run the sound speed moves from 2.00000 to
2.00064, i.e. 0.03%, where an ideal-gas box at `gamma = 5/3` would have heated
until the Mach number fell by 3x.

The driving cannot be made identical — astronomix drives the smooth
`k^6 exp(-8k/kpk)` spectrum peaked at `n = 1.5`, AthenaPK drives its 30-mode
`kpeak = 2` set with a parabolic envelope over `1 <= |n| <= 3` — so the codes are
matched on the **achieved flow**, as in the hydrodynamic study. What is
remarkable is that no calibration turned out to be needed: handing AthenaPK
`accel_rms = 3.5`, the same number as astronomix's `F0`, lands all three schemes
at the same turbulence.

| | astronomix | AthenaPK PLM | AthenaPK PPM |
|---|---|---|---|
| Mach at `t/t_cross = 2.5-5`, N=64 | 0.685 | 0.687 | 0.680 |
| Mach at `t/t_cross = 2.5-5`, N=128 | 0.722 | 0.686 | 0.681 |

Both codes' amplitudes are set from the field they actually apply, which is why
this works at every resolution — unlike AthenaK's `dedt`, whose normalisation
collapses to the white-noise branch and drifts by 22-66% across a resolution
ladder (see the hydrodynamic study's README).

The remaining differences are the ones being measured: **FD/WENO5 +
constrained transport against FV/Godunov + GLM cleaning**, each at its own
standard CFL number (astronomix `C_cfl = 1.5`; AthenaPK 0.3 for PLM+VL2 and 0.4
for PPM+RK3, the values its own convergence tests use).

## What is measured

All of it through one estimator, `_mhd_spectral.py`, called by both runners —
astronomix's inside the snapshot callback while the state is still on the GPU,
AthenaPK's after reading the `.phdf` dumps back — so a difference in the numbers
is a difference in the solvers and not in the diagnostics.

**The dynamo.**
* `<|B|>(t)`, `E_B(t)`, `E_B/E_K(t)` — the growth curve and the saturation level.
* The kinematic growth rate `Gamma = d ln E_B / dt`, fitted over the genuinely
  exponential stretch (from 10x the seed energy to 10% of the saturated level),
  quoted as `Gamma t_cross`.
* The saturated `E_B / E_K`, with the residual growth rate still present in the
  averaging window reported alongside — a run whose dynamo has not finished
  growing gives a lower bound, not a level.

**The spectra**, time-averaged over two windows:
* the **kinematic phase**, the *fixed* window `t/t_cross = 2.5-5` in which the
  flow has spun up and `E_B/E_K` is at most 5% even for the fastest dynamo in the
  ladder, so all three schemes carry the same hydrodynamic turbulence and a
  kinetic-spectrum comparison measures numerical dissipation and nothing else.
  Deliberately fixed rather than chosen per run from `E_B/E_K`: an adaptive
  window would be shorter for the schemes with the faster dynamos, and the
  comparison would partly be measuring the window. And
* the **saturated state**, which is what the run ends in.

The figures plot the spectra themselves, with `n_1/4` and the magnetic peak
marked on the curves; the compensation is applied inside the metric rather than
in the figure. From them: the cutoff shell `n_1/4` where the Kolmogorov-compensated spectrum
`n^(5/3) E_v(n)` has fallen to a quarter of its `n = 3-8` plateau (so `n_1/4 / N`
is the fraction of the grid a scheme turns into resolved turbulence), the fitted
slope, the shell `<n>_B` the magnetic energy sits at, and the fraction of kinetic
energy above `n = N/4`.

That last one exists because `n_1/4` alone is not safe here. A compensated-plateau
cutoff asks where a spectrum *falls off*, so a scheme that piles energy up at the
grid scale instead of dissipating it scores a **high** effective resolution. The
top-octave energy fraction separates the two cases, and it has to: see the PPM
result below.

**Finite-volume vs finite-difference sampling.** astronomix stores point values,
AthenaPK stores cell averages, whose spectrum is the true one times
`sinc^2(pi n_i / N)` per axis — up to 19% suppression at `n = N/4`, in AthenaPK's
disfavour. `--no-deconvolve` turns the correction off; everything quoted here has
it on.

**div B.** Both codes are measured with AthenaPK's own `relDivB` definition,
`<L_cell |div B| / |B|>` on the **cell-centred** field (the reimplementation
reproduces AthenaPK's history output to 8 digits). For astronomix the *staggered*
divergence — the constraint constrained transport actually preserves, the
6th-order interface divergence — is reported as well. Read them together, and see
the caveat in the results.

**Runtime.** Wall clock of the integration loop on a single A100, with
astronomix's JIT compilation moved out of the timed region by a warm-up call
(same config, so the same compiled executable is reused) and AthenaPK's taken
from Parthenon's own `walltime used`. Both codes ran on the same GPU model, which
is why the whole ladder is on A100s: the AthenaPK binary is an `AMPERE80` build.

Each code's diagnostic output is subtracted using a *measured* per-event cost,
because neither is free and they are not equally expensive: astronomix's 82
in-flight spectral reductions and AthenaPK's 41 HDF5 dumps are each timed by a
pair of otherwise-identical short runs with 82 and 9 (resp. 41 and 5) output
events. The `timing/` controls hold those runs.

The ladder itself runs astronomix in x32 and AthenaPK in x64, so its wall clocks
are **not** a like-for-like speed comparison. A separate precision-matched timing
grid — every configuration back to back on one GPU, output disabled — is what the
Runtime section quotes; read that rather than the `wall [s]` column of
`data/metrics.md`.

## Running it

Both codes need an A100 (the AthenaPK binary is an `AMPERE80` build, and the
wall-clock numbers are only comparable on one GPU model), one GPU each:

```bash
bash run_dynamo_ladder.sh              # astronomix + both AthenaPK schemes, N = 64, 128, 256
bash run_dynamo_ladder.sh 64 128       # only the resolutions given
```

or a single leg by hand:

```bash
PYTHONPATH=$(git rev-parse --show-toplevel) python dynamo_convergence.py --n 128
python athenapk_turb.py --n 128 --scheme ppm
```

Then, once the queue has drained:

```bash
python make_convergence_figures.py                    # figures/ + data/metrics.md
python make_convergence_figures.py --data data/controls \
       --figures figures/controls                     # the scheme controls
python make_convergence_figures.py --data data/controls/seeds \
       --figures figures/controls/seeds               # the forcing realisations
```

The diffusivities need runs carrying the transfer spectra (`--transfer`), which
live in `data/dissipation/` and `data/dissipation_mech/`:

```bash
python make_dissipation_figure.py                     # figures/dynamo_dissipation.png
python make_mechanism_table.py --matched              # the mechanism table
python make_mechanism_table.py --audit                # and the checks behind it
python make_mechanism_table.py --calibration          # measured vs imposed diffusivity
```

The calibration set is AthenaPK given an explicit Laplacian coefficient, which
is the only run in the study whose answer is known in advance:

```bash
python athenapk_turb.py --n 64 --scheme plm --beta 1e2 --transfer \
       --ohm-diff 2e-3 --tag calib_eta2e3_plm_N64 --outdir data/calibration
```

`athenapk_turb.py` deletes its `.phdf` dumps after reducing them (`--keep-snapshots`
to keep them); a 256^3 run writes 25 GB of them. A single 256^3 meshblock
overflows the Kokkos team-scratch limit, so 256^3 is decomposed into eight 128^3
blocks — which is how AthenaPK is normally run anyway, and the block-boundary
cost that carries is part of what the code costs at that size.

## Where the data lives

Only the reduced per-run `.npz` files (a few hundred kB each: the scalar time
series, the per-snapshot spectra, and one mid-plane slice) are kept, in `data/`.
Raw snapshots go to `/export/data/lstorcks/mhd_dynamo/` and are deleted after
reduction. `data/metrics.md` is regenerated by `make_convergence_figures.py` and
is the table this README quotes.

## Files

| | |
|---|---|
| `dynamo_convergence.py` | astronomix runner; reduces snapshots in flight, so memory does not grow with the number of snapshots |
| `athenapk_turb.py` | AthenaPK runner: writes the input deck, runs the binary, reduces the `.phdf` dumps |
| `_mhd_spectral.py` | the one spectral estimator both codes go through |
| `_mhd_metrics.py` | the metric definitions every number in this README comes from |
| `make_convergence_figures.py` | figures + `data/metrics.md` |
| `make_dissipation_figure.py` | the spectral energy budget: `D(n)`, `nu_eff(n)`, `eta_eff(n)` |
| `make_mechanism_table.py` | `Pm`, `Re`, `Rm` per scheme; the matched-state table, the audit, the growth-rate model comparison and the explicit-diffusivity calibration |
| `make_mechanism_figure.py` | the three-panel `Re` / `Rm` / `Pm` summary figure |
| `make_dynamo_movie.py` | the side-by-side dynamo animation |
| `measure_glm_psi_term.py` | how much of a GLM scheme's `eta_eff` is the divergence cleaning |
| `DYNAMO_MECHANISM.md` | the written-up version of all of this: animation, derivation, table, discussion |
| `make_reynolds_figure.py` | the shell-based Reynolds numbers over resolution (read as *proportional to* `Re`/`Rm`; see the audit) |
| `run_dynamo_ladder.sh` | submits the ladder to the GPU queue |
| `paper_turbulence.py`, `make_fig14.py`, `make_fig15.py` | the earlier HOW-MHD reproduction this study's physical setup is taken from |

## Results

Every number below is in `data/metrics.md`, regenerated by
`make_convergence_figures.py`. Nine production runs, all complete. Error bars come
from the control set: four forcing realisations for astronomix at `N = 64`, three
each for the AthenaPK schemes.

### The turbulence is matched

The precondition for everything else. Over the fixed window
`t/t_cross = 2.5-5`, where the flow has spun up and the field is still nearly
passive:

| scheme | Mach at `t/t_cross = 2.5-5` | max `E_B/E_K` there |
|---|---|---|
| astronomix WENO5+CT | 0.693 ± 0.010 | 0.008 (N=64) ... 0.051 (N=256) |
| AthenaPK PLM+VL2 | 0.696 ± 0.013 | 0.001 ... 0.004 |
| AthenaPK PPM+RK3 | 0.696 ± 0.015 | 0.002 ... 0.007 |

The three schemes agree to 0.4% on the flow they are driving, inside the 1.5%
run-to-run scatter, at every resolution and with no calibration — `accel_rms` was
just set to astronomix's `F0`. So the differences that follow are not differences
in the turbulence being magnetised.

### The dynamo: growth rate is not a well-posed number in this setup

Saturation is measured cleanly. The *growth rate* is not, and finding out why is
the main methodological result here.

| scheme | saturated `E_B/E_K`, N=64 | N=128 | N=256 |
|---|---|---|---|
| astronomix WENO5+CT | 0.214 ± 0.031 | 0.299 | 0.356 |
| AthenaPK PLM+VL2 (2nd) | 0.113 ± 0.031 | 0.138 | 0.322 |

astronomix and PLM **converge onto each other** as the grid is refined, and their
saturated magnetic spectra at 256^3 agree to within 1% over `n = 3-8`. That part
is solid: the saturated state is a genuine, converging, code-independent
prediction, and the 2nd-order scheme simply needs more cells to reach it.

**The kinematic growth rate is a different story.** Fitting `d ln E_B / dt` over
successive decades in `E_B/E_K` — a true kinematic eigenmode would give the same
rate in every decade — gives (`data/weakseed/`, `beta = 1e8` seed):

| run | `1e-5-1e-4` | `1e-4-1e-3` | `1e-3-1e-2` | `1e-2-1e-1` |
|---|---|---|---|---|
| astronomix N=64 | 0.95 | 0.82 | 0.79 | 0.26 |
| astronomix N=128 | 1.62 | 1.21 | 1.01 | 0.37 |
| astronomix N=256 | 2.86 | 2.11 | 1.29 | 0.47 |
| AthenaPK PLM N=64 | 0.37 | 0.42 | 0.22 | 0.17 |
| AthenaPK PLM N=128 | 0.68 | 0.60 | 0.54 | 0.29 |

The rate falls monotonically through every decade, starting at `E_B/E_K ~ 1e-5`
where the field is dynamically irrelevant. **There is no exponential eigenmode
phase**, so "the growth rate" is whatever decade you chose to fit, and any
resolution trend read off it inherits that choice.

Two causes, both in the setup rather than in the schemes:

* **The seed is a uniform net-flux field in a periodic box, so its mean is
  exactly conserved** (measured: the `n = 0` shell holds 4.000e-6 at `t = 0` and
  3.981e-6 at the end). What the flow does first is *tangle* that mean field, not
  grow a dynamo eigenmode. The spectrum shows it directly: at 256^3 the mean
  shell of `E_B(n)` runs 3.3 -> 16 -> 37.7 by `t/t_cross = 2` — the field is
  shredded to the grid scale — and then migrates *back* to 17 over the following
  seven crossing times. A fixed-shape, growing-amplitude eigenmode would do
  neither. Because the tangling rate is set by the strain at the *grid* scale, a
  finer grid tangles faster, which is most of the apparent resolution trend.
* **The box starts at rest** and the driving needs ~2 crossing times to bring
  `v_rms` to its plateau, so with the original `beta = 1e6` seed the field passed
  `E_B/E_K = 1e-3` at `t/t_cross = 1.6` at 256^3 — inside the spin-up. The
  `beta = 1e8` runs in `data/weakseed/` push that to `t/t_cross = 3.2` and are
  what the table above uses.

An earlier version of this study quoted `Gamma t_cross` = 1.01 / 1.55 / 2.70 for
astronomix and claimed a `N^0.7` trend. Those numbers came from a fit window
defined as "10x the seed energy to 10% of saturation", which slides back into the
spin-up as the resolution rises — at 256^3 it landed entirely inside it. They are
wrong and have been removed; `growth_rate` in `_mhd_metrics.py` now fits a fixed
decade in `E_B/E_K` with `t/t_cross >= 2` and reports the point count, but no fit
window makes the quantity well posed in *this* setup.

**The fix, and what it shows.** A zero-net-flux seed (`--seed-field sin`,
AthenaPK `b_config = 2`: `B_x = sqrt(2) B0 sin(2 pi z / L)`, same magnetic energy,
no conserved mean) plus a seed weak enough to leave room between the tangling
transient and the back-reaction (`beta = 1e12`) does produce a genuine eigenmode.
Runs are in `data/zeroflux/`; `eigenmode_growth_rate` in `_mhd_metrics.py` fits
the window from 1000x the seed level up to `E_B/E_K = 1e-4` and reports the
per-decade rates, which is the test:

| run | 1e-8 | 1e-7 | 1e-6 | 1e-5 | fitted `Gamma t_cross` | spread |
|---|---|---|---|---|---|---|
| astronomix N=64 | 0.81 | 0.80 | 0.87 | 0.82 | **0.819** | 3.5% |
| astronomix N=128 | 1.30 | 1.36 | 1.47 | 1.35 | **1.385** | 5.2% |
| astronomix N=256 | 2.40 | 2.20 | 2.13 | 2.11 | **2.145** | 6.1% |
| AthenaPK PLM N=64 | 0.34 | 0.33 | 0.34 | 0.26 | **0.303** | 11.7% |
| AthenaPK PLM N=128 | 0.50 | 0.58 | 0.53 | 0.59 | **0.554** | 7.5% |
| astronomix N=64, x64 | 0.85 | 0.80 | 0.74 | 0.69 | 0.789 | 9.2% |

Four decades at a constant rate in every run: that is an eigenmode, where the
uniform seed gave a rate falling monotonically through every decade. The x64
control (0.789 against 0.819) shows x32 is not setting it even with a seed field
of 2.8e-6.

**And it does not converge — as it should not.** astronomix goes
0.819 -> 1.385 -> 2.145, factors of 1.69 and 1.55 per doubling, a fitted
`Gamma ~ N^0.70`. The ideal-MHD prediction is that the kinematic dynamo is driven
by the eddies at the resistive scale, `Gamma ~ eps^(1/3) l_eta^(-2/3)`, and with
numerical resistivity `l_eta ~ dx`, so `Gamma ~ N^(2/3) = N^0.667`, 1.587 per
doubling. The measurement lands on it. **A converged growth rate requires explicit
resistivity**, which fixes `Rm` independently of the grid; without it the growth
rate is a diagnostic of the scheme's dissipation rather than a physical
prediction, and the quantity that converges is the *saturated* state — which is
why that is what this study reports.

Read as a diagnostic, it is a sharp one. The astronomix/PLM ratio is nearly
resolution-independent (2.70 at 64^3, 2.50 at 128^3), so the two schemes differ by
a fixed factor in effective magnetic Reynolds number: `Gamma ~ l_eta^(-2/3)` turns
a factor 2.6 in rate into a factor `2.6^(3/2) = 4.2` in resistive scale.
Equivalently, **astronomix at 64^3 has the kinematic dynamo AthenaPK's 2nd-order
scheme reaches at roughly 200^3** — a factor ~3 in linear resolution, ~27 in cells.

**What would still be needed for a converged growth rate** — not done here:

**explicit resistivity** in both codes, so that `Rm` is a fixed physical number
rather than a function of the grid, and a resolution ladder run at fixed `Rm`.
Inserting the seed only after the turbulence is statistically steady
(`t/t_cross >~ 3`) would also remove the last of the spin-up from the
measurement, at the cost of a code change on the astronomix side.

### The velocity field, by contrast, barely distinguishes them

Measured in the same `t/t_cross = 2.5-5` window, the cutoff shell of the
Kolmogorov-compensated kinetic spectrum:

| scheme | `n_1/4 / N`, N=64 | N=128 | N=256 | `E_v` above `n = N/4` |
|---|---|---|---|---|
| astronomix WENO5+CT | 0.232 (`n_1/4` = 14.55 ± 0.08) | 0.209 | 0.180 | 0.006 / 0.002 / 0.001 |
| AthenaPK PLM+VL2 | 0.228 (14.46 ± 0.04) | 0.210 | 0.195 | 0.007 / 0.003 / 0.001 |
| AthenaPK PPM+RK3 | 0.371 (22.90 ± 0.40) | 0.354 | 0.303 | 0.027 / 0.013 / 0.005 |

astronomix and AthenaPK's 2nd-order scheme have **the same** kinetic cutoff, to
0.6% at `N = 64`, 0.5% at `N = 128` and 8% at `N = 256` (where PLM is the higher
of the two), while their dynamos differ by a factor of 2.3. Taken together with the magnetic spectra — where astronomix carries 2-3x
more power than PLM at `n > 10` at `N = 128` (`figures/dynamo_spectra_*.png`,
right panels) — that says the discriminator on this problem is **the induction
equation, not the momentum equation**: constrained transport plus
characteristic-wise WENO5 has a much lower numerical resistivity than GLM
cleaning plus a Godunov flux, at essentially equal numerical viscosity.

PPM's much higher `n_1/4` is not extra resolved cascade. It has 4-13x more of its
kinetic energy in the top octave than either other scheme, and its spectrum is
flat rather than falling there — grid-scale pile-up, which a compensated-plateau
cutoff scores as high effective resolution. Two controls confirm it is a property
of the scheme rather than of the configuration: at CFL 0.3 instead of 0.4 the tail
is unchanged (top-octave fraction 0.023 vs 0.024), and AthenaPK's other
high-order reconstructions show it too (LimO3 0.011, WENO-Z 0.023, all against
astronomix's 0.006).

The controls also show that within AthenaPK, reconstruction order does not order
the dynamo monotonically: `Gamma t_cross` at `N = 64` is 0.418 (PLM, 2nd), 0.517
(LimO3, 3rd), 0.559-0.705 (PPM, 3rd) but only 0.418 for WENO-Z (5th), which
nonetheless saturates highest of the four at `E_B/E_K = 0.249`. Whatever
astronomix is doing to get 0.958 is not simply "higher-order reconstruction".

### div B

| scheme | cell-centred `relDivB`, N=64 | N=128 | N=256 |
|---|---|---|---|
| astronomix WENO5+CT | 0.106 | 0.065 | 0.040 |
| AthenaPK PLM+VL2 | 0.058 | 0.037 | 0.017 |
| AthenaPK PPM+RK3 | 0.095 | 0.056 | 0.031 |
| astronomix, *staggered* | 1.0e-5 (x32) / 2.3e-14 (x64) | 1.5e-5 | 2.4e-5 |

**Do not read the first three rows as a quality ranking.** astronomix's
constrained transport keeps the divergence it actually constrains — the 6th-order
interface divergence of the staggered field — at floating-point round-off, in
both precisions, which is the last row. The cell-centred number is what a centred
difference of the *interpolated* cell-centred field gives, and for a staggered
scheme that is a property of the face-to-centre interpolation, not of a cleaning
failure. It is tabulated because the cell-centred field is what both codes
output and what an analysis pipeline downstream would use, and because it falls
off with resolution in every scheme.

### Dissipation spectra: the diffusivities, measured

Neither code has an explicit diffusivity, so the dissipation spectrum cannot be
formed as `2 nu k^2 E(k)` -- `nu` is the unknown. It is measured instead from the
spectral energy budget, shell by shell:

    dE(n)/dt = T(n) - D(n)

with `T(n)` the *ideal* transfer (the exact non-dissipative right-hand side
projected on the field and shell-summed, `_mhd_spectral.transfer_spectra`) and
`D(n)` everything the scheme threw away. Both terms are measured -- `T(n)` from
each snapshot, `dE(n)/dt` by differencing consecutive ones -- and in the
saturated state the second averages to zero, so `D(n) = <T(n)>`. The budget
closes: with `v = 0` the magnetic transfer vanishes identically, and during
growth `sum_n T_mag` exceeds the measured `dE_B/dt` by exactly the dissipated
fraction. `make_dissipation_figure.py` builds it; the figure is
`figures/dynamo_dissipation.png`.

Dividing out the shell's own curvature gives the diffusivities directly.
Averaged over the flat band `n/n_Nyq = 0.2-0.7` (below it the forcing and the
outer scale contaminate; above it the transfer is aliased, since the products
are not dealiased):

| | `nu_eff` | `eta_eff` | `Pm = nu/eta` | `nu` slope | `eta` slope |
|---|---|---|---|---|---|
| astronomix, N=64 | 1.13e-3 | **1.06e-3** | **1.07** | +0.25 | +0.35 |
| astronomix, N=128 | 4.38e-4 | **4.24e-4** | **1.03** | +0.20 | +0.33 |
| AthenaPK PLM, N=64 | 1.27e-3 | **2.03e-3** | **0.63** | +0.58 | +0.15 |
| AthenaPK PLM, N=128 | 5.46e-4 | **7.78e-4** | **0.70** | +0.40 | +0.06 |

ratios at fixed N, astronomix / PLM:

| | `nu` | `eta` | `Pm` |
|---|---|---|---|
| N=64 | 0.89 | **0.52** | 1.70 |
| N=128 | 0.80 | **0.54** | 1.47 |

**The central result of the whole study, now measured rather than inferred: at
the same grid the two schemes have the same numerical viscosity (0.89 and 0.80)
and astronomix's numerical resistivity is half (0.52 and 0.54).** The factor of
two holds at both resolutions. `Pm_num` is ~1.05 against ~0.66, and constant
with resolution for each scheme, confirming again that it is a property of the
scheme and not of the grid.

Both diffusivities fall as `~N^-1.3` in every case (astronomix `nu ~ N^-1.37`,
`eta ~ N^-1.32`; PLM `N^-1.22` and `N^-1.38`), close to the `N^-4/3` a fixed
dissipation-scale-to-grid ratio implies — an independent consistency check on
the whole measurement.

**One prediction this does *not* confirm.** The modified-equation ansatz gives
`eta_num(k) ~ (k dx)^(p-1)`, i.e. slope `+4` for the 5th-order scheme against
`+1` for the 2nd-order one. Measured on the saturated turbulence the slopes are
`+0.35` and `+0.15` — both essentially **flat, i.e. Laplacian-like**, and the
5th-order scheme is not the steeper of the two. So the hyper-dissipative
`k^(p-1)` form is a property of the *linear, smooth-field* limit, not of the
turbulent state: real turbulence is intermittent, the sharp structures are where
the dissipation happens, and there the limiters engage and the scheme is locally
low order. That is consistent with the linear measurement being eight orders of
magnitude smaller than the turbulent one, and it is the quantitative form of the
"numerical resistivity is flow-dependent" caveat.

What survives, then, is not "a different functional form in the turbulent state"
but **"the same functional form at half the amplitude, on the magnetic field
only"** — which is enough, because `Pm_num` is what the dynamo responds to. With
`Rm = v_rms L / eta_eff` this gives `Rm` = 580 / 1428 (astronomix) against
326 / 800 (PLM) and `Pm` = 1.05 against 0.66, and those two numbers reproduce
the measured growth-rate ratio: `sqrt(Rm ratio) = 1.33` times a `Pm` residual of
1.9 gives the observed 2.5-2.7.

### Summary: Re, Rm and Pm across schemes and resolutions

`make_mechanism_table.py --summary`. Measured at matched `E_B/E_K = 0.01`,
where every scheme's field is still passive and all of them carry the same
turbulence (`Mach` = 0.68-0.73 across every row). `Re` and `Rm` use the driving
wavelength `L = 0.5`; `Rm_L` repeats it with each run's own measured integral
scale, which is the only part of the convention that does not cancel between
codes. Errors on `Pm` combine a moving-block bootstrap with the spread over
every defensible band and window.

| scheme | order | div.B | N | `nu_eff` | `eta_eff` | `Re` | `Rm` | `Pm` | `Rm_L` |
|---|---|---|---|---|---|---|---|---|---|
| AthenaPK PLM+VL2 | 2 | GLM | 64 | 1.36e-3 | 2.25e-3 | 505 | 306 | **0.61 ± 0.07** | 107 |
| AthenaPK PLM+VL2 | 2 | GLM | 128 | 6.44e-4 | 1.02e-3 | 1092 | 687 | **0.63 ± 0.08** | 215 |
| AthenaPK PLM+VL2 | 2 | GLM | 256 | 2.88e-4 | 4.58e-4 | 2520 | 1585 | **0.63 ± 0.08** | 552 |
| AthenaPK PPM+RK3 | 3 | GLM | 64 | 6.48e-4 | 1.33e-3 | 1068 | 520 | **0.49 ± 0.03** | 165 |
| AthenaPK PPM+RK3 | 3 | GLM | 128 | 2.87e-4 | 5.73e-4 | 2427 | 1213 | **0.50 ± 0.13** | 375 |
| AthenaPK PPM+RK3 | 3 | GLM | 256 | 1.31e-4 | 2.47e-4 | 5468 | 2901 | **0.53 ± 0.12** | 890 |
| AthenaPK WENO-Z+RK3 | 5 | GLM | 64 | 6.88e-4 | 1.42e-3 | 989 | 480 | **0.49 ± 0.05** | 156 |
| AthenaPK WENO-Z+RK3 | 5 | GLM | 128 | 3.07e-4 | 6.25e-4 | 2260 | 1111 | **0.49 ± 0.06** | 345 |
| AthenaPK WENO-Z+RK3 | 5 | GLM | 256 | 1.40e-4 | 2.67e-4 | 5093 | 2664 | **0.52 ± 0.03** | 863 |
| astronomix WENO5 | 5 | **CT** | 64 | 1.14e-3 | 9.35e-4 | 604 | 735 | **1.22 ± 0.06** | 300 |
| astronomix WENO5 | 5 | **CT** | 128 | 5.22e-4 | 4.30e-4 | 1359 | 1650 | **1.21 ± 0.03** | 617 |
| astronomix WENO5 | 5 | **CT** | 256 | 2.33e-4 | 1.93e-4 | 3046 | 3675 | **1.21 ± 0.04** | 1166 |

PPM and WENO-Z are flagged `n_K!` at every resolution: the Kolmogorov scale
their `nu` implies sits at 1.14 / 0.97 / 0.90 and 1.18 / 0.94 / 0.89 of Nyquist,
so their `Re` -- and therefore their `Pm` -- is partly an extrapolation. Their
`Pm` also drifts up by 8% from 64^3 to 256^3, in step with that flag improving,
which is the direction an under-resolved `nu` predicts. PLM (0.72 / 0.59 / 0.54)
and astronomix (0.72 / 0.70 / 0.71) are resolved throughout, and those are the
two rows to lean on.

Reading it: `Re` and `Rm` both grow as `N^1.2` for every scheme; order moves the
prefactor; `Pm` is flat in `N` and splits the schemes into two groups by
divergence treatment.

### What this measurement rests on

Worth stating separately, because the parts have very different standing.

**Exact.** The magnetic budget. `dB/dt = curl(v x B)` *is* the complete ideal
induction equation -- no forcing acts on `B`, no term is dropped -- so
`D_B = T_mag - dE_B/dt` is precisely everything the discretisation did to the
magnetic energy that ideal MHD does not. For a GLM code that legitimately
includes the divergence-cleaning sink (`-grad psi` in the evolved equation),
which is the point rather than a contaminant: cleaning *is* part of the
induction discretisation. For CT there is no such term. Since `eta_eff` is the
measurement the whole result rests on, this is the part that matters most.

**Zero in the band, not merely small.** The kinetic budget omits the forcing.
Both codes force the *velocity* equation with a band-limited acceleration
(AthenaPK adds `dt * rho * a` to the momentum with `rho` untouched, astronomix
adds `amp * w` to the velocity), and `E_v` is the velocity spectrum, so the
omitted term is `Re[v-hat*(n) . a-hat(n)]` with `a-hat` supported only on
`n <= 3`. It is not a product with a varying field and does not spread.

**Approximate, with the size measured.** Aliasing: 2%, from repeating the runs
with the products formed on a 3/2-refined grid. The FD/FV representation
difference: 3%, from box-filtering astronomix's point values into cell averages
and re-reducing. Compressing a mildly scale-dependent `nu_eff(n)` into one
scalar: the `Pm(n)` ranges of the CT and GLM schemes do not overlap at any shell
in the band, so it is not what produces the separation.

**Convention, not measurement.** `L = 0.5`, the driving wavelength. Every
absolute `Re` and `Rm` scales with it; `Pm` does not depend on it at all. The
measured integral scale is 0.16-0.20 and is 9-11% larger for astronomix than for
the GLM schemes, so a fixed `L` flatters astronomix's `Rm` ratio by about that
much -- hence the `Rm_L` column.

**Calibrated.** Handed an explicit Laplacian `eta` comparable to its own
numerical one, the estimator recovers it to 2%. Independently, the threshold it
implies (`Rm_crit` = 176-225) is the literature value for `Pm <~ 1`.

**Cross-checked by an independent conditioning.** Matching on `E_B/E_K` reaches
each scheme at a different time (3 to 15 crossing times); a fixed window
`t/t_cross = 2.5-5` reaches each at a different `E_B/E_K`. The two give the same
`Pm` to within 4% for nine of ten runs, so neither the OU forcing phase nor the
choice of conditioning is doing the work.

**Error budget on `Pm`.** Moving-block bootstrap over snapshots (block 3, set by
the measured integrated autocorrelation time of 1.6-4.8 snapshots) 1-4%; band
and window choice up to 13%; conditioning choice up to 4%; aliasing 2%;
representation 3%; precision 5%; CFL 3%; forcing realisation 4%. Summed in
quadrature, ~17%. **The CT/GLM gap is 100%.**

**Known to fail.** The resolvedness check, for PPM and WENO-Z at every
resolution (`n_K/n_Nyquist` = 0.89-1.18), which is why the two rows to lean on
are PLM and astronomix. And the growth-rate collapse at astronomix 256^3, which
is unexplained and which makes the *speed* advantage resolution-dependent even
though `Pm` is not.

### Which part of the scheme is responsible: order, or constrained transport?

**The comparison point matters, and the obvious one is wrong.** Each scheme
saturates at its own `E_B/E_K` -- 0.08 for PLM at 64^3, 0.43 for astronomix at
256^3 -- and `Pm_eff` drifts with that fraction, upward for every GLM scheme as
the back-reaction sets in. A table read off "in the saturated state" therefore
partly measures *where each scheme happened to saturate*. Compared that way,
AthenaPK's PPM at 128^3 looks like it has almost closed the gap to CT
(`Pm` = 0.94 against 1.03) -- but it saturates at `E_B/E_K` = 0.36 where
astronomix saturates at 0.26, and the apparent convergence is entirely that
offset. `make_mechanism_table.py --matched` compares the schemes at the *same*
`E_B/E_K` instead, which for the reference value 0.01 is inside the kinematic
phase of every run, where the field is still passive and all four carry the same
hydrodynamic turbulence.

| scheme | order | div.B | N | `Pm` at `E_B/E_K` = 0.003 | 0.01 | 0.03 | 0.06 | 0.10 |
|---|---|---|---|---|---|---|---|---|
| AthenaPK PLM+VL2 | 2nd | GLM | 64 | 0.600 | **0.606** | 0.629 | 0.660 | -- |
| AthenaPK PLM+VL2 | 2nd | GLM | 128 | 0.618 | **0.629** | 0.633 | 0.635 | 0.659 |
| AthenaPK PLM+VL2 | 2nd | GLM | 256 | 0.633 | **0.629** | 0.653 | 0.680 | 0.695 |
| AthenaPK LimO3+RK3 | 3rd | GLM | 64 | 0.600 | **0.625** | 0.607 | 0.631 | 0.619 |
| AthenaPK PPM+RK3 | 3rd | GLM | 64 | 0.474 | **0.487** | 0.514 | 0.533 | 0.539 |
| AthenaPK PPM+RK3 | 3rd | GLM | 128 | 0.470 | **0.500** | 0.531 | 0.583 | 0.623 |
| AthenaPK PPM+RK3 | 3rd | GLM | 256 | 0.508 | **0.531** | 0.560 | 0.608 | 0.646 |
| AthenaPK WENO-Z+RK3 | 5th | GLM | 64 | 0.477 | **0.485** | 0.493 | 0.510 | -- |
| AthenaPK WENO-Z+RK3 | 5th | GLM | 128 | 0.482 | **0.492** | 0.506 | 0.562 | 0.657 |
| AthenaPK WENO-Z+RK3 | 5th | GLM | 256 | 0.512 | **0.523** | 0.556 | 0.607 | 0.660 |
| astronomix WENO5 | 5th | **CT** | 64 | 1.375 | **1.217** | 1.172 | 1.116 | 1.088 |
| astronomix WENO5 | 5th | **CT** | 128 | 1.296 | **1.214** | 1.160 | 1.076 | 1.057 |
| astronomix WENO5 | 5th | **CT** | 256 | 1.217 | **1.207** | 1.142 | 1.068 | 1.046 |

Read down the `E_B/E_K = 0.01` column:

**1. `Pm_num` is a scheme constant, and it does not converge away.** Each scheme
holds its value to within a few percent over a factor of four in resolution --
CT 1.217 / 1.214 / 1.207 and PLM 0.606 / 0.629 / 0.629, both over 64^3, 128^3
and 256^3 -- **the two schemes measured at all three resolutions are flat to
0.8% and 4% respectively over a factor of four in grid**. PPM gives 0.487 /
0.500 / 0.531 and WENO-Z 0.485 / 0.492 / 0.523, both drifting up 8% as their
resolvedness flag clears. The gap between constrained
transport and GLM cleaning is a factor 1.9-2.5 and shows no sign of closing
with N.

The contrast with the saturated-state reading is stark, and is the cleanest
statement of why the comparison point matters. Saturated, every GLM scheme
appears to march toward CT as the grid is refined -- PLM 0.639 / 0.708 / 0.793,
PPM 0.643 / 0.943, WENO-Z 0.534 / 0.886 -- while CT sits still at 1.063 / 1.028
/ 1.033. **That apparent convergence is entirely the saturation level rising
with resolution** (PLM's `E_B/E_K` goes 0.076 / 0.217 / 0.305): at matched
`E_B/E_K` nothing converges at all. This is the single number that separates the two families.

**2. Order is simply not the variable.** Four GLM schemes at 64^3 -- PLM (2nd,
0.62), LimO3 (3rd, 0.63), PPM (3rd, 0.48), WENO-Z (5th, 0.48) -- span
`Pm` = 0.48-0.63 with **no monotone trend in order**: the two 3rd-order schemes
sit at opposite ends of the range, and the 5th-order one is not distinguishable
from the lower of them. Order does buy `Rm` (`eta` falls from 2.2e-3 for PLM to
1.3e-3 for PPM) and it buys `Re` in step, but which side of the 0.48-0.63 band a
scheme lands on is a property of its limiter, not of its formal order. An
earlier version of this section, read off the saturated state, claimed order
left `Pm` untouched at 0.63/0.64; that was the saturation-level confound above,
and it happened to give the right conclusion for the wrong reason.

**3. So for the dynamo it is the induction discretisation, not the order.** A
5th-order GLM scheme (WENO-Z, `Pm` = 0.48) and a 2nd-order GLM scheme (PLM,
`Pm` = 0.62) are on the same side of the divide; the 5th-order CT scheme
(`Pm` = 1.22) is on the other, a factor 1.9 above the *highest* GLM value.
Order sets the *level* of both diffusivities, something else sets their *ratio*
-- see "What this does and does not establish" below for exactly how much of
that "something else" is pinned to constrained transport and how much rides
with it.

### The Prandtl number, demonstrated with a knob instead of a scheme

The step from "CT has a higher `Pm`" to "that is why its dynamo is faster" is an
inference across four schemes, and it can be tested by intervention instead.
AthenaPK takes an explicit Laplacian viscosity and resistivity (`--mom-diff`,
`--ohm-diff`), which move `Pm` inside one fixed discretisation. All at 64^3,
PLM+GLM, the production weak seed, everything measured at the *same* matched
state (`E_B/E_K = 0.01`) as the table above (`data/calibration_kin/`):

| | `nu` | `eta` | `Pm` | `Re` | `Rm` | `Gamma t_cross` | saturated `E_B/E_K` |
|---|---|---|---|---|---|---|---|
| PLM, nothing imposed | 1.36e-3 | 2.25e-3 | 0.61 | 505 | 306 | 0.269 | 0.076 |
| PLM, `nu` = 1e-3 imposed | 2.03e-3 | 1.90e-3 | **1.07** | 319 | 340 | **0.376** | **0.172** |
| PLM, `nu` = 2e-3 imposed | 2.72e-3 | 1.89e-3 | **1.44** | 239 | 344 | **0.463** | 0.166 |
| astronomix WENO5+CT | 1.14e-3 | 9.35e-4 | 1.22 | 604 | 735 | 0.577 | 0.154 |

**At an `Rm` that moves by 11%, raising `Pm` from 0.61 to 1.07 raises the
kinematic growth rate by 40% and the saturated magnetic energy by a factor
2.3.** The knob run at `Pm` = 1.07 lands on astronomix's saturated `E_B/E_K`
(0.172 against 0.154) from an `Rm` 2.2x lower. The `Pm` interpretation is
therefore not only a correlation across schemes; it survives an intervention
inside one scheme.

Two honest qualifications. `Re` necessarily changes -- that is what moving `Pm`
at fixed `Rm` means. And the imposed viscosity does cross-talk into the
resistivity: the measured `eta` falls 16% (a smoother flow makes less
small-scale field, so less numerical resistivity), which is where the 11% `Rm`
drift comes from. Both are reported rather than assumed away.

The **resistive** half of the ladder, run with a strong seed so the field stays
measurable whatever the dynamo does (`data/calibration/`), measures the
**dynamo threshold** as a by-product:

| imposed `eta` | `eta_tot` | `Rm` | `Pm` | saturated `E_B/E_K` | `d ln E_B / dt` |
|---|---|---|---|---|---|
| 0 | 2.07e-3 | 322 | 0.64 | 0.072 | -0.025 (saturated) |
| 1e-3 | 3.09e-3 | 225 | 0.45 | 0.017 | -0.039 |
| 2e-3 | 4.01e-3 | 176 | 0.35 | 0.0009 | -0.112 (decaying) |
| 4e-3 | 5.65e-3 | 124 | 0.25 | ~0 | -0.467 (dead) |

`Rm_crit` is between 176 and 225 at `Pm ~ 0.4`; the literature value for
`Pm <~ 1` is ~200 (Iskakov et al. 2007; Schekochihin et al. 2007). This is not
fully independent -- two thirds of `eta_tot` at the threshold is the *measured*
numerical part -- but the other third is exactly known, and the estimator
recovers it to 2% (below), so it is a real constraint on the absolute scale and
not a circular one. It is also the reason the shell-based estimate in
`make_reynolds_figure.py`, which would put every one of these runs at `Rm` < 20,
should not be read as an absolute number.

### Growth rate: one scaling for all four schemes

With `Rm` and `Pm` both measured at the matched state, the kinematic growth rate
collapses. Fitted over the decade `E_B/E_K = 3e-3` to `3e-2`
(`make_mechanism_table.py --collapse`):

| scheme | N | `Rm` | `Pm` | `Gamma t_cross` | `Gamma / sqrt(Rm Pm)` |
|---|---|---|---|---|---|
| AthenaPK PLM+VL2 | 64 | 306 | 0.61 | 0.269 | 0.0197 |
| AthenaPK PLM+VL2 | 128 | 687 | 0.63 | 0.463 | 0.0223 |
| AthenaPK PLM+VL2 | 256 | 1585 | 0.63 | 0.622 | 0.0197 |
| AthenaPK PPM+RK3 | 64 | 520 | 0.49 | 0.328 | 0.0206 |
| AthenaPK PPM+RK3 | 128 | 1213 | 0.50 | 0.645 | 0.0262 |
| AthenaPK PPM+RK3 | 256 | 2901 | 0.53 | 0.898 | 0.0229 |
| AthenaPK WENO-Z+RK3 | 64 | 480 | 0.49 | 0.232 | 0.0152 |
| AthenaPK WENO-Z+RK3 | 128 | 1111 | 0.49 | 0.426 | 0.0182 |
| AthenaPK WENO-Z+RK3 | 256 | 2664 | 0.52 | 0.685 | 0.0184 |
| astronomix WENO5+CT | 64 | 735 | 1.22 | 0.577 | 0.0193 |
| astronomix WENO5+CT | 128 | 1650 | 1.21 | 0.854 | 0.0191 |
| astronomix WENO5+CT | 256 | 3675 | 1.21 | 1.038 | **0.0156** |

Every growth rate is independently confirmed: the AthenaPK ones by refitting the
same decade on the ~800-row `.hst` history (46-199 points, agreeing to 2%), and
astronomix's 256^3 value -- fitted on only three snapshots -- by the densely
sampled zero-net-flux run, which gives the same 1.038 from 86 points. None of
the numbers below is a thin-fit artifact.

Twelve points is not enough to *fit* exponents: the free fit has a low
in-sample residual but a leave-one-scheme-out residual several times worse,
because holding a scheme out removes most of the leverage. The comparison that
is well posed is between laws whose exponents are fixed a priori, so that only
the prefactor is fitted:

| model | prefactor | in-sample | leave-one-scheme-out |
|---|---|---|---|
| `Gamma ~ sqrt(Rm)` | 0.0159 | 19.6% | 24.9% |
| **`Gamma ~ sqrt(Rm Pm)`** | **0.0195** | **14.6%** | **17.8%** |
| `Gamma ~ sqrt(Rm) Pm` | 0.0241 | 26.3% | 34.4% |
| `Gamma ~ Rm Pm` | 0.0007 | 50.1% | 59.1% |
| `Gamma ~ N^a` (fitted) | | 30%+ | 40%+ |

`sqrt(Rm Pm)` wins both in and out of sample, and resolution alone is the worst
model of the set, so the ordering is not a disguised `N` dependence. The same
constant covers the two explicit-viscosity runs whose `Pm` was moved by a
diffusivity knob rather than by a change of scheme (0.0198 and 0.0208), which is
the strongest evidence that the collapse is not fitting scheme labels.

**Where it breaks, and a retraction.** astronomix at 256^3 sits 20% below the
line at 0.0156, against 0.0191-0.0193 at 64^3 and 128^3. When only that point
was available we guessed a general high-`Rm` flattening -- `Gamma ~ Rm^(1/2)`
being a near-threshold law, and astronomix at 256^3 being the first run far
above threshold. **The 256^3 AthenaPK runs falsify that**: PPM at `Rm` = 2901
and WENO-Z at `Rm` = 2664 sit *on* the line (0.0229, 0.0184), and PLM at 1585
sits exactly on it (0.0197). The shortfall is specific to astronomix at 256^3
and we do not currently have an explanation for it.

It has a visible consequence, so it is not a detail. The *measured* growth-rate
advantage shrinks with resolution even though `Pm` does not:

| astronomix `Gamma` over | 64^3 | 128^3 | 256^3 |
|---|---|---|---|
| AthenaPK PLM (2nd) | 2.15 | 1.85 | 1.67 |
| AthenaPK PPM (3rd) | 1.76 | 1.32 | 1.16 |
| AthenaPK WENO-Z (5th) | 2.49 | 2.01 | 1.52 |

and with it the price in cells, computed from the *measured* `Gamma` rather than
the collapse:

| astronomix at | PLM (2nd) | PPM (3rd) | WENO-Z (5th) |
|---|---|---|---|
| 64^3 | 226^3 (44x cells) | 165^3 (17x) | 178^3 (22x) |
| 128^3 | 440^3 (41x) | 311^3 (14x) | 337^3 (18x) |
| 256^3 | 611^3 (14x) | 426^3 (5x) | 462^3 (6x) |

**So the honest headline is two-part: the `Pm` difference is robust, scheme-
specific and resolution-independent; the dynamo-speed advantage it buys is
none of those.** The advantage is real at every resolution measured, but it
falls from roughly 2x at 64^3 to 1.2-1.7x at 256^3, and whether it keeps falling
is the open question this ladder cannot answer. Any claim about what CT is worth
at production resolution should quote the 256^3 column, not the 64^3 one.

### Why does `Pm` matter? One mechanism tested, and rejected

`Gamma ~ sqrt(Rm Pm)` is an empirical collapse. The textbook mechanism behind a
`Pm` dependence is that the small-scale dynamo is stretched by the smallest
eddies the field can actually feel: below the *larger* of the viscous and
resistive scales there is either no flow structure left (`Pm > 1`) or no field
left (`Pm < 1`). With `l_D = (D^3 / eps)^(1/4)` and a turnover rate
`eps^(1/3) l^(-2/3)`, that predicts, with no free exponent,

    Gamma ~ sqrt( eps / max(nu, eta) ).

Every quantity in it is measured here, so it can simply be tried. It does not
work:

| model (only the prefactor fitted) | in-sample | leave-one-scheme-out |
|---|---|---|
| `sqrt(Rm)` | 19.6% | 24.9% |
| `sqrt(Re)` | 34.9% | 45.6% |
| **`sqrt(Rm Pm)`** | **14.6%** | **17.8%** |
| `sqrt(eps/nu)`, the Kolmogorov rate | 35.4% | 46.2% |
| `sqrt(eps/eta)`, the resistive-scale rate | 20.5% | 25.6% |
| `sqrt(eps/max(nu, eta))`, the mechanism above | 23.3% | 29.6% |

and the residuals are structured, not random: normalised by that rate, the CT
runs sit at 0.0187 / 0.0175 / 0.0133 and every GLM run between 0.0080 and
0.0136. **After dividing out the turnover rate at its own limiting dissipation
scale, the CT scheme is still 40-70% faster at 64^3 and 128^3.** So "`Pm > 1`
unlocks the viscous-scale eddies" is not sufficient to explain the advantage,
even though it points in the right direction.

What is left as the explanation for the `Pm` term is the *threshold*: `Rm_crit`
rises as `Pm` falls through unity, so at equal `Rm` a `Pm ~ 0.5` scheme is
running closer to its own critical point than a `Pm ~ 1.2` one and grows more
slowly for that reason. Our own threshold measurement (`Rm_crit` = 176-225 at
`Pm ~ 0.4`) is consistent with the published curve. That is an explanation of
the right size and sign, but it is borrowed rather than demonstrated here.

### Is any of this measurable? The audit

`make_mechanism_table.py --audit` runs the checks that decide whether the
diffusivities may be believed at all. Every one of them is a number, not an
argument.

**Parseval.** `sum_{n>=1} E_v(n)` against `<v_rms^2> / 2`: 0.9903-1.0000 across
the ladder, and necessarily `<= 1`, since the shell sum drops the `n = 0` mean
flow and the cube corners outside the inscribed Nyquist sphere. Checked in
float64 on a stored dump, the corner loss is 0.010% and Parseval over all modes
closes to 1.000000. The remaining shortfall is the mean flow, which is 1e-4 for
AthenaPK (it zeroes the net momentum every step) but 0.96% / 0.67% / 0.08% for
astronomix at `64^3` / `128^3` / `256^3` -- a genuine setup difference, Galilean-
irrelevant for the dynamo but not for a finite-difference scheme's advection
error, and shrinking with resolution while `Pm` does not.

An earlier version of this check divided by `<v_rms>^2` rather than
`<v_rms^2>` and so reported values up to 1.0020, which a sum over a subset of
non-negative modal energies can never be; Jensen's inequality accounts for
exactly the 0.08-0.21% excess.

**Sign.** A dissipative reading of `D(n)` requires `D > 0` in the band. The
fraction of in-band shells with the wrong sign is **0.00 for every run and both
fields**. (The figure plots `|D|`, which would have hidden this; the audit
reports the fraction so it cannot.)

**Budget closure.** In a steady state the total dissipation must balance the
injection, so `sum_n [D_v(n) + D_B(n)]` over *all* shells -- which includes the
forcing appearing at `n <= 3` with the opposite sign -- should be zero. It comes
out at -9% to -19% of the injection, consistently negative, which is the
expected size of the compressive `p dV` term the ideal transfer omits at
`Mach ~ 0.6`. The sign convention is confirmed independently: `D_v(n)` is
strongly negative for `n <= 3` and positive from `n >= 4` outwards, i.e. the
diagnostic sees the forcing as a source exactly where the forcing is.

**Why the omitted forcing cannot contaminate the band.** Both codes force the
*velocity* equation with a band-limited acceleration -- AthenaPK adds
`dt * rho * a` to the momentum with `rho` untouched, i.e. `dv/dt += a`, and
astronomix adds `amp * w` to the velocity directly. `E_v` is the velocity
spectrum, so the forcing enters `dE_v(n)/dt` as `Re[v-hat*(n) . a-hat(n)]`, and
`a-hat` has support only on `1 <= n <= 3`. Above `n = 3` the omitted term is
**exactly** zero, not merely small. It is not a product with a spatially varying
field, so it does not spread.

**Aliasing, measured rather than argued.** The nonlinear products are formed in
real space and not dealiased, so shells above 2/3 of Nyquist can be
contaminated, and the default band reaches 0.7. Two independent checks:
*(i)* the same measurement over three bands lying entirely below the 2/3 cutoff
-- `(0.1, 0.3)`, `(0.2, 0.4)`, `(0.3, 0.6)` -- and two crossing it, printed by
`--audit`. astronomix at 64^3 gives `Pm` = 0.970 / 1.096 / 1.078 / 1.062 /
1.063; PLM gives 0.506 / 0.538 / 0.646 / 0.604 / 0.639. The band matters at the
±13% level and the separation survives every choice.
*(ii)* the same runs repeated with the products formed on a 3/2-refined grid
(Orszag; `--dealias`), which is alias-free by construction. `Pm` at the matched
state moves from 1.217 to **1.241** for astronomix and from 0.606 to **0.607**
for PLM. **Aliasing is a 2% effect on a factor-of-two result.**

**`Pm` shell by shell, not as one band mean.** `nu_eff(n)` is not exactly flat
(log-slopes +0.2 to +0.6), so compressing it to a scalar is a choice. `--audit`
therefore also prints the range of `Pm(n)` across the band. At 64^3 astronomix
spans 0.96-1.20 and PLM 0.41-0.75; at 128^3, 0.85-1.13 against 0.55-0.81; the
5th-order GLM scheme spans 0.38-0.66. **The ranges do not overlap**, so the
CT/GLM separation is a statement about every shell in the band, not an artifact
of averaging a scale-dependent quantity.

**Everything else that could have produced it.** Each control is one 64^3 run,
identical but for the named change, read at the matched state:

| | `Pm` at `E_B/E_K` = 0.01 | shift |
|---|---|---|
| astronomix WENO5+CT, baseline | 1.217 | |
| ... products dealiased (Orszag 3/2) | 1.241 | +2.0% |
| ... state box-filtered to cell averages (FV representation) | 1.178 | -3.2% |
| ... float64 instead of float32 | 1.273 | +4.6% |
| ... a different forcing realisation | 1.245 | +2.3% |
| AthenaPK PLM+GLM, baseline | 0.606 | |
| ... products dealiased | 0.607 | +0.2% |
| ... CFL 0.15 instead of 0.3 | 0.626 | +3.3% |
| ... a different forcing realisation | 0.629 | +3.8% |
| AthenaPK LimO3+RK3 (3rd order, GLM) | 0.625 | |

**Every one of them is at or below 5%, against a CT/GLM gap of a factor 2.0.**
The box-filter control is the pointed one: handing astronomix's point values
through the same cell-average filter a finite-volume code's storage applies
moves both diffusivities by ~15% and their ratio by 3%, so the FD/FV
representation difference cannot be what separates the two codes.

**Calibration against a known coefficient.** The estimator is given an explicit
Laplacian `eta` on top of the numerical one and asked to find it: the measured
`eta` rises by 1.02, 0.92 and 0.82 per unit imposed over the three steps of the
ladder. The first step -- where the imposed value is comparable to the numerical
one -- **recovers it to 2%**. The fall-off at the top is physical rather than
diagnostic: once the explicit resistivity dominates, the field is smoother and
the scheme's own numerical resistivity is genuinely displaced. The viscous
ladder recovers only 0.62-0.76 per unit for the same reason, more strongly, and
the cross-talk is asymmetric: an imposed viscosity moves the measured `eta` by
-26% while an imposed resistivity moves the measured `nu` by +5%. **The `eta`
measurement, which the whole `Pm` story rests on, is the one that calibrates
cleanly.**

**Resolvedness -- the one check that fails.** The Kolmogorov shell implied by
the measured `nu` and the measured dissipation rate,
`n_K = (nu^3 / eps_v)^(-1/4) / 2pi`, should lie on the grid if `nu` is to be
read as a Laplacian coefficient at all. As a fraction of Nyquist:

| | N=64 | N=128 | N=256 |
|---|---|---|---|
| astronomix | 0.72 | 0.70 | **0.71** |
| AthenaPK PLM | 0.72 | 0.59 | 0.54 |
| AthenaPK PPM | **1.14** | **0.97** | |
| AthenaPK WENO-Z | **1.18** | **0.94** | |

astronomix sits at 0.71 at *every* resolution -- a fixed dissipation-scale-to-
grid ratio, which is what a self-similar numerical dissipation should give and
an independent confirmation that the measurement is internally consistent.
**PPM and WENO-Z at 64^3 exceed 1**: their implied Kolmogorov scale is off the
grid, so the flat `nu_eff` fitted in the band is being extrapolated past where
the dissipation actually happens. Those rows should not be used as evidence for
resolution-independence, and the WENO-Z point is also the outlier in the growth
collapse. What survives without them is still the whole result: PLM (resolved at all
three resolutions) sits at `Pm` = 0.61-0.63 and astronomix (also resolved at all
three) at 1.21. Two fully resolved schemes, a factor of four in grid, and a
factor two apart at every point.

**Two Reynolds numbers that disagree by a factor of 40.** The study contains two
routes to `Rm`. The shell route, `Rm = (n_eta / n_inj)^(4/3)` from the
dissipation-weighted shell (`make_reynolds_figure.py`), gives 12-52. The direct
route, `Rm = v_rms L / eta_eff` from the measured budget, gives 306-3531. They
disagree by 27-68x, and worse, the shell route puts *every* scheme at `Pm > 1`
and orders PLM (1.81) above astronomix (1.93) by a hair, where the direct route
separates them cleanly. **The direct route is the one to believe**: it is a
measurement rather than a Kolmogorov scaling relation with an assumed
order-unity prefactor, it calibrates against a known coefficient to 2%, and its
absolute scale is confirmed by the `Rm_crit` measurement landing on the
literature value. The shell-based figure is kept because the *scaling with N* it
shows is still meaningful, but its axis labels should be read as "proportional
to `Rm`", not as `Rm`.

**How noisy is the growth rate itself?** Much noisier than `Pm`. A second
forcing realisation moves astronomix's `Gamma t_cross` at 64^3 from 0.577 to
0.557 (4%) but PLM's from 0.269 to 0.186 (45%) -- PLM at 64^3 sits close to the
dynamo threshold, where the growth rate is a sensitive function of the
realisation. The 15% residual of the `sqrt(Rm Pm)` collapse is therefore
*within* the realisation scatter of its own inputs, and should be read as
"consistent with" rather than as a precision test. `Pm` itself is
realisation-stable to 2-4%, which is why the mechanism table is the robust part
of this study and the growth-rate collapse is the suggestive part.

**How much of a GLM scheme's `eta_eff` is the divergence cleaning? 0.14%.**
The evolved GLM equation is `dB/dt = curl(v x B) - grad psi` and the ideal
transfer carries only the first term, so the residual also holds the cleaning
coupling. AthenaPK dumps `psi`, so this is measurable rather than arguable:
`T_psi(n) = sum_shell Re[B-hat* . (-i k psi-hat)]` (`measure_glm_psi_term.py`)
comes out as a weak *sink* of magnitude 2.9e-6 in `eta` units against a measured
total of 2.03e-3 on a saturated `64^3` PLM run. The budget therefore overstates
that scheme's own numerical resistivity by 0.14%, which would move `Pm` 0.14%
*towards* CT. `psi` carries 1.0% of the magnetic energy. Measured on one run;
not asserted for the others.

**The Dedner damping is not a hidden channel either.** Varying `glmmhd_alpha`
over 0.02 / 0.1 / 0.5 -- a factor of 25 -- moves `eta_eff` by 3%, `Pm` by 2% and
`Rm` by 1%. A first pass with one forcing realisation each also showed `Gamma`
varying by 1.8x, which looked like a second channel; it is not. Those three runs
are identical to three digits out to `t/t_cross ~ 8` and only then bifurcate, and
repeating the ladder with four realisations at each `alpha` gives a one-way ANOVA
`F` = 1.72, `p` = 0.23 -- **no significant effect on the growth rate**.

**The realisation noise floor, from those twelve runs.** Pooled over all of
them, `Gamma` at `64^3` scatters by **20%** between forcing realisations while
`Pm` scatters by **1.7%**. That is the honest limit on the growth-rate collapse:
its 14.6% residual is *below* the scatter of its own dependent variable, so
`sqrt(Rm Pm)` is an empirical regression consistent within noise rather than a
validated law, and the grid-equivalence factors below inherit at least that 20%.
`Pm`, by contrast, is measured far more tightly than the effect it describes.

**One trap in the code, now closed.** `spectra_deconv` divides the
cell-averaging transfer function out of the three energy spectra but not out of
the two transfer spectra, so a deconvolved `D / 2k^2E` would mix corrected and
uncorrected rows and land ~20% low on `eta` (it happens to leave `Pm` alone to
1.4%, which is how it went unnoticed). The dissipation pipeline always used the
raw spectra, so no published number was affected; `dissipation_series` now
raises rather than accepting `deconvolve=True`.

### What this does and does not establish

**Established.** The four schemes separate into two groups by `Pm_num`, the
separation is a factor ~2, it holds shell by shell and at every resolution
measured, and none of aliasing, representation, precision, CFL, forcing
realisation or band choice accounts for more than 5% of it. Independently,
`Pm` is what the dynamo responds to: moving it with an explicit viscosity inside
one fixed scheme reproduces both the growth-rate and the saturation advantage.

**Not established.** That *constrained transport specifically* is the cause.
astronomix is the only CT code here and AthenaPK the only GLM one, so CT rides
with finite differences, characteristic-wise WENO flux splitting, a genuinely
isothermal EOS, SSP-RK4 and `C_cfl = 1.5`. The controls above rule out the
representation, the precision and the CFL as *individually* sufficient, and the
four-scheme order sweep inside AthenaPK rules out reconstruction order -- 2nd,
3rd (PLM, LimO3), 3rd (PPM) and 5th (WENO-Z) span `Pm` = 0.48-0.63 with no
monotone trend, so order simply is not the variable. What remains is
"the induction discretisation together with the characteristic decomposition",
and separating those two needs a code that offers both CT and GLM. Until then
this is a strong correlation with the confounds enumerated, not a proof of
mechanism.

### What the 5th-order scheme actually buys

Everything above, reduced to one accounting (`make_mechanism_table.py
--accounting`). All ratios are astronomix over the named AthenaPK scheme, at the
matched kinematic state; a ratio below 1 means astronomix is the less
dissipative of the two.

| | N | `nu` | `eta` | `Rm` | `Pm` | `Gamma` | `sqrt(Rm Pm)` |
|---|---|---|---|---|---|---|---|
| vs PLM+VL2 (2nd) | 64 | 0.84 | **0.42** | 2.40 | 2.01 | 2.15 | 2.19 |
| vs PLM+VL2 (2nd) | 128 | 0.81 | **0.42** | 2.40 | 1.93 | 1.85 | 2.15 |
| vs PLM+VL2 (2nd) | 256 | 0.81 | **0.42** | 2.32 | 1.92 | 1.67 | 2.11 |
| vs PPM+RK3 (3rd) | 64 | **1.76** | 0.70 | 1.41 | 2.50 | 1.76 | 1.88 |
| vs PPM+RK3 (3rd) | 128 | **1.82** | 0.75 | 1.36 | 2.43 | 1.32 | 1.82 |
| vs PPM+RK3 (3rd) | 256 | **1.78** | 0.78 | 1.27 | 2.27 | 1.16 | 1.70 |
| vs WENO-Z+RK3 (5th) | 64 | **1.65** | 0.66 | 1.53 | 2.51 | 2.49 | 1.96 |
| vs WENO-Z+RK3 (5th) | 128 | **1.70** | 0.69 | 1.48 | 2.47 | 2.01 | 1.91 |
| vs WENO-Z+RK3 (5th) | 256 | **1.67** | 0.72 | 1.38 | 2.31 | 1.52 | 1.78 |

**1. Against the 2nd-order scheme it is "the same viscosity, 0.42x the
resistivity".** The `eta` ratio is 0.42 at 64^3, 128^3 *and* 256^3 -- three
significant figures of resolution-independence -- while `nu` is within 20%. Both
the `Rm` gain and the `Pm` gain come from the resistivity alone.

**2. Against the higher-order schemes the story inverts, and only `Pm`
survives.** astronomix is 1.7-1.8x *more* viscous than PPM and WENO-Z and only
0.66-0.75x less resistive, so its `Rm` advantage collapses to 1.4-1.5x. What
does not move is `Pm`: 2.4-2.5x against both. **Raising the reconstruction order
inside AthenaPK closes most of the `Rm` gap and none of the `Pm` gap.** That is
the whole result in one line.

**3. Order buys a prefactor in `Rm`, not a better scaling.** Fitting
`Rm ~ N^a` per scheme gives a = 1.19 (PLM), 1.24 (PPM), 1.24 (WENO-Z), 1.16
(astronomix) -- indistinguishable -- with prefactors 2.20, 2.99, 2.78, 5.87.
Going 2nd to 3rd order multiplies `Rm` by 1.5 at every resolution and never
changes the slope.

**4. Priced in cells.** From `Gamma = 0.0198 sqrt(Rm Pm)` and those fits, the
grid each GLM scheme needs to reach astronomix's dynamo:

| astronomix at | PLM (2nd) | PPM (3rd) | WENO-Z (5th) |
|---|---|---|---|
| 64^3 | 226^3 (44x cells) | 165^3 (17x) | 178^3 (22x) |
| 128^3 | 440^3 (41x) | 311^3 (14x) | 337^3 (18x) |
| 256^3 | 611^3 (14x) | 426^3 (5x) | 462^3 (6x) |

Raising AthenaPK's order from 2nd to 3rd cuts the penalty by about 3x at every
resolution; it does not remove it, because the remaining factor is `Pm`. The
penalty itself falls sharply at 256^3 -- that is the astronomix growth-rate
shortfall of point 5, not a change in the diffusivities.

**5. Where it stops working.** astronomix at 256^3 falls 20% below the
collapse, and the AthenaPK 256^3 runs show that this is specific to it rather
than a general high-`Rm` effect. See "Growth rate: one scaling for all four
schemes" above -- the consequence is that the tables in points 1-4 of this
accounting describe the diffusivities correctly at every resolution, but the
*dynamo speed* they imply is only accurate up to 128^3.

### Numerical resistivity, viscosity and Reynolds numbers

Neither code has an explicit diffusivity, so what follows measures the numerical
one. The framework and its labels (R1-R7) follow the analysis in
`docs/`-less form here; each entry says what was measured and whether the
prediction held.

**The ansatz.** For a p-th order upwind-type scheme the leading dissipative
truncation term damps mode `k` at `gamma_d(k) = C_p V (k dx)^p k`, i.e.
`eta_num(k) = C_p V dx (k dx)^(p-1)`. For `p = 2` that is quasi-Laplacian and
eats into intermediate scales; for `p = 5` it is a `k^6` hyper-resistivity —
negligible over most of the resolved range, then a wall at the grid.

**(R1) Order `p`.** Not re-measured: the repo's own CP Alfven wave convergence
tests already pin it. AthenaPK `pytests/mhd/data/athenapk/`: `L1 ~ N^-1.80` for
PLM+VL2 and `N^-2.61` for PPM+RK3; astronomix's `alfven_wave3D` test gives 5th
order. So `p = 2` and `p = 5` for the two schemes compared here.

**(R1, null result) The passive-decay test does not work, and that is
informative.** A small-amplitude field in a *motionless* box should decay at
`2 eta_num k^2`. It does not: AthenaPK preserves a single smooth mode to
`ME(end)/ME(0) = 1.000000` at every `k dx` from 0.098 to 0.785, and astronomix's
apparent decay turns out to be round-off — the same measurement in double
precision drops `eta` from 3.89e-10 to **3.36e-11**, a factor 11.6, tracking the
precision rather than the scheme. **With no flow there is no numerical
resistivity at all**: it is entirely flow-driven, `~ V`, exactly as the ansatz
says. A decaying-wave measurement therefore has to use a *travelling* wave.
`measure_numerical_diffusivity.py` runs the test and documents the null.

**(R2) Dissipation scale.** Predicted `k_c/k_Nyq ~ N^(-1/(3p+1))`, i.e. `-0.062`
for `p = 5` and `-0.143` for `p = 2`; and `k_c L ~ N^(3p/(3p+1))`, i.e. `0.938`
and `0.857`. Measured on the zero-net-flux ladder (`data/reynolds/`), with `k_c`
the dissipation-weighted mean shell of the kinetic spectrum in the eigenmode
window:

| | `n_nu` | `k_c L` slope | predicted | `n_nu/N` slope | predicted |
|---|---|---|---|---|---|
| astronomix (p=5) | 9.24, 17.19, 33.10 | **+0.920** | +0.938 | **-0.080** | -0.062 |
| AthenaPK PLM (p=2) | 9.56, 17.03 | **+0.832** | +0.857 | **-0.168** | -0.143 |

Right ordering and within 0.03 of the prediction on `k_c L` for both schemes.
**The high-order scheme's resolved fraction really does decline more slowly.**

**(R3) Consistency — and the calibration.** The spectral-shell `Rm_eff` comes out
at 3-122, far below the `Rm_crit ~ 100-300` at which a dynamo should exist at
all, yet both codes dynamo: the Kolmogorov prefactor in
`Rm_eff = (k_c/k_inj)^(4/3)` is off by a factor of ~10-30. The **directly
measured** diffusivities fix that — `Rm = v_rms L / eta_eff` gives 326 to 1428
across the ladder, physically sensible and straddling `Rm_crit`. Use the direct
numbers for absolute statements and the spectral ones only for ratios.

**(R5) Threshold, indirectly.** With the direct `Rm`, AthenaPK PLM at 64^3 sits
at `Rm = 326` — the lowest in the ladder and closest to `Rm_crit` — which is the
natural explanation for its anomalously steep apparent `Gamma(N)` scaling
(R4a). Bracketing the actual threshold was not attempted.

**(R7) Not done.** The matched-`Rm` pair still needs running.

**(R4a) Growth-rate scaling.** Predicted `gamma tau_L ~ N^(2p/(3p+1))` = 0.625
(p=5) and 0.571 (p=2). Measured `N^0.696` (astronomix: 0.819, 1.386, 2.152) and
`N^0.872` (PLM: 0.303, 0.554, two points). astronomix is within 0.07 of the
prediction; PLM is well above it and the *ordering is inverted*. The likely
reason is R5: PLM's 64^3 point has the lowest `Rm_eff` in the whole ladder and
sits nearest threshold, where the apparent scaling steepens.

**(R4b) The collapse test — the sharpest result.** Plotting `gamma tau_L` against
`Rm_eff^(1/2)` for every run of both schemes:

| | N | `Rm_eff` | `gamma tau_L` | `gamma / sqrt(Rm_eff)` |
|---|---|---|---|---|
| astronomix | 64 | 7.28 | 0.819 | 0.304 |
| astronomix | 128 | 12.41 | 1.386 | **0.393** |
| astronomix | 256 | 30.17 | 2.152 | **0.392** |
| AthenaPK PLM | 64 | 3.15 | 0.303 | 0.171 |
| AthenaPK PLM | 128 | 7.42 | 0.554 | **0.204** |

Two things at once. **Within** astronomix the `Rm^(1/2)` scaling is verified —
`gamma/sqrt(Rm)` is 0.393 and 0.392 at 128^3 and 256^3, flat to 0.3% once the
lowest-`Rm` point is left behind. **Between** schemes it does not collapse: the
`p = 5` constant is 0.39 against the `p = 2` constant of ~0.20, a factor
**1.9**. The direct diffusivity measurement identifies that offset as the
**`Pm_num` difference** (1.05 against 0.66), not as a hyper-resistive cutoff
shape — see the dissipation-spectra section. Either way it is the quantitative
answer to "two codes at the same `Rm_eff` can still disagree".

**(R4b, second fingerprint) `Pm_num` is a fixed property of the scheme.**
Measured with the dissipation-weighted shells: astronomix 2.63, 2.81, 2.89 over
the 64-256^3 ladder (`~N^0.067`) and AthenaPK PLM 1.93, 1.96 (`~N^0.023`).
Constant to a few percent across a factor of four in resolution, order unity,
and **different between schemes by a factor 1.4** — exactly as predicted, and
not a free parameter.

**Where the difference lives, scale by scale.** The local log-slope of the
spectra through the dissipation range, in the eigenmode window at 128^3:

| `n/n_Nyq` | 0.15 | 0.25 | 0.35 | 0.50 | 0.70 |
|---|---|---|---|---|---|
| kinetic, astronomix | -2.40 | -2.18 | -3.82 | -5.01 | -7.11 |
| kinetic, PLM | -2.51 | -2.39 | -4.18 | -4.91 | -5.50 |
| **magnetic, astronomix** | **+0.03** | **+0.38** | -1.00 | -1.76 | -3.22 |
| **magnetic, PLM** | **-0.74** | **-0.52** | -2.60 | -3.40 | -4.39 |

The *kinetic* slopes agree between schemes at every scale — equal numerical
viscosity, as R2 says. The *magnetic* ones do not: at `n/n_Nyq = 0.15-0.25`
astronomix's spectrum is still **rising** (Kazantsev-like) while PLM's is already
falling. That is "Laplacian-like resistivity eating into intermediate scales"
against "flat, then a wall", read directly off the spectra.

**(R6) Weak-seed time to saturation.** `t_sat = T0 + A/gamma` with
`A = 0.5 ln(E_sat/E_seed)`. Comparing the `beta = 1e8` and `beta = 1e12`
zero-net-flux runs at 128^3 (four decades of seed): measured
`dt_sat = 5.00 t_cross` against a predicted `4.605/gamma = 3.32`. The additive
decomposition holds, but the implied `gamma = 0.92` is below the eigenmode
`1.385` — because the four extra decades a weaker seed must climb include the
tangling transient, which is slower than the eigenmode. So `A/gamma` needs
`gamma` evaluated over the decades actually traversed, not the eigenmode value.

### Runtime

**Read this section before quoting any speed number from the table above.** The
`wall [s]` column in `data/metrics.md` compares astronomix in **single**
precision against AthenaPK in **double**, because that is what each code's
production ladder was run in. That is a precision comparison as much as a method
comparison, and taking it at face value gets the answer backwards.

The measurement below is the controlled one: one crossing time at each
resolution, every configuration run back to back on the same GPU in a single
queue job, HDF5 output disabled, astronomix's JIT compilation outside the timed
region. Times are seconds per crossing time; both codes adapt their own step, so
this is cost per unit *physical* time, which is what a user pays.

| configuration | N=64 | N=128 | N=256 |
|---|---|---|---|
| astronomix WENO5+CT, x32 | 0.7 | 8.9 | 130.8 |
| astronomix WENO5+CT, x64 | 1.4 | 19.7 | 312.6 |
| AthenaPK PLM+VL2, x32 | 2.2 | 15.3 | **117.0** |
| AthenaPK PLM+VL2, x64 | 2.7 | 19.9 | 192.0 |
| AthenaPK PPM+RK3, x32 | 4.2 | 30.9 | 206.0 |
| AthenaPK PPM+RK3, x64 | 5.0 | 37.9 | 337.0 |

**At matched precision the two codes cross over between 128^3 and 256^3.** In
x32, astronomix is 3.1x faster at 64^3, 1.7x at 128^3, and **1.12x slower** at
256^3; in x64 it is 1.9x faster at 64^3, level at 128^3, and 1.62x slower at
256^3. The 3rd-order scheme costs 1.76x the 2nd-order one at 256^3 (206 s vs
117 s in x32), so it is 1.58x more expensive than astronomix there. AthenaPK's disadvantage at small grids is Parthenon per-cycle overhead,
not its MHD arithmetic — its throughput climbs from 4.4e7 to 2.08e8
cell-updates/s across the ladder (the 256^3 figure is in line with published
AthenaPK A100 numbers, so it is running properly) while astronomix's saturates
near 1.1e8.

Where the 256^3 cost goes, per crossing time in x32:

| | astronomix | AthenaPK PLM |
|---|---|---|
| steps / cycles | 840 | 1446 |
| per step / cycle | 156 ms | 81 ms |
| flux evaluations per step | 5 (SSP-RK4) | 2 (VL2) |
| **per flux evaluation** | **31 ms** | **40 ms**, of which ~7 ms is the driver |

So per flux evaluation the 5th-order characteristic-wise WENO5 and the 2nd-order
PLM+HLLD cost about the same, ~31 vs ~33 ms of solver. astronomix takes 1.68x
*fewer* steps — `C_cfl = 1.5` in its `dt = C dx / sum_i lambda_i` convention is
an effective 0.5 per direction, against AthenaPK's 0.3, which is already near
VL2's 3D stability limit; SSP-RK4's larger stability region is what buys that.
But it pays 5 flux evaluations per step against VL2's 2, so it ends up doing 1.45x
more flux evaluations per crossing time, and that is the whole story.

Three further things the audit pinned down, all in `data/controls/timing/`:

* **Precision is the biggest single lever**, and it is not the same for both:
  x64 costs astronomix **2.39x** at 256^3 but AthenaPK only **1.65x**. astronomix
  is further from being compute-bound, so it loses more to the halved bandwidth.
  (At 64^3 the astronomix x64 penalty is only 1.5x — the small grid is
  latency-bound, so precision matters less.)
* **AthenaPK's driving is not free.** Its `few_modes_ft` evaluates an explicit
  30-mode inverse transform per cell per cycle; with `num_modes = 1` the cycle
  drops from 133 ms to 109 ms, so the driver is **18% of an AthenaPK cycle** at
  256^3. astronomix does the same job with an FFT. That cost is real, but it
  belongs to the turbulence driver rather than to the MHD solver.
* **Meshblock size matters and the ladder used the right one**: 64^3 blocks are
  1.29x slower than 128^3 at `N = 256` (247 s vs 192 s), and a single 256^3 block
  is not possible at all. Dropping the history cadence 10x changed nothing, so
  the `.hst` reductions are free.

Node sharing is not contaminating any of this. Re-running the whole `N = 64`
trio serially in one job reproduced 46.4 / 169.2 / 303.6 s against the ladder's
46.1 / 167.0 / 305.0 s, and the 256^3 astronomix timings repeated in two separate
jobs to 0.2% (130.6 / 312.5 s and 130.8 / 312.6 s).

The honest one-line summary: **on this problem the two codes cost about the same
per flux evaluation, and at 256^3 in matched precision AthenaPK's 2nd-order
scheme is 1.1-1.6x cheaper per crossing time.** What astronomix buys for that
price is on the physics side — it reaches the converged saturated state at a
coarser grid (see above) — not on the clock.

The single-precision AthenaPK build was checked against the double-precision one
before being used for timing: at `N = 64` it gives the same Mach number (0.687)
and the same kinetic cutoff (14.61), with `Gamma t_cross` 0.507 against 0.418 ±
0.017 — a small but real speed-up of the dynamo from round-off noise, which is
why the science ladder uses the double-precision build and only the timing grid
uses the single-precision one.

### What the precision control says

astronomix runs the ladder in x32. In double precision at `N = 64` the growth
rate is 0.961 against 1.011 and the kinetic cutoff 14.66 against 14.52 — both
inside the realisation scatter, at 1.5x the wall clock. The x32 that the ladder
uses is not what makes its dissipation low.

### Caveats

* `n_1/4 / N` falls with resolution for astronomix (0.232, 0.209, 0.180) rather
  than holding constant as it does in the hydrodynamic study. Part of that is
  real and part is the window: astronomix's dynamo is fast enough at 256^3 that
  `E_B/E_K` reaches 5% inside `t/t_cross = 2.5-5`, so its "kinematic" kinetic
  spectrum there is already slightly drained by the field, which steepens it.
  The AthenaPK runs, with slower dynamos, do not pay this — it works against
  astronomix, not for it.
* One realisation per (scheme, N) above `N = 64`. The `N = 64` scatter (4.4% on
  `Gamma t_cross`, 14% on saturated `E_B/E_K`) is the best available estimate of
  the error bars at 128^3 and 256^3.
* GLM-MHD is run with AthenaPK's default `dedner_plain` source, as in its own
  turbulence input deck. `dedner_extended` was not tried.
* **AthenaPK's PPM+RK3 leg is excluded from the figures** (`--exclude ppm`)
  pending an explanation of its grid-scale pile-up; its data is still in `data/`
  and comes back by dropping the flag.
* The growth rate is not a well-posed quantity in this setup — see the dynamo
  section. The saturated state is.
* The physics ladder is astronomix x32 against AthenaPK x64. The precision
  controls say this does not matter for the physics (astronomix x64 at 64^3 is
  inside the realisation scatter; AthenaPK x32 at 64^3 reproduces the Mach number
  and the kinetic cutoff exactly and the growth rate to 1.5 sigma) but it matters
  a great deal for the cost, which is why the timing grid is measured separately
  at matched precision.

## Controls

Five control runs, all at `N = 64`, in `data/controls/` (figures under
`figures/controls/`, table in `data/controls/metrics.md`):

| control | what it rules out |
|---|---|
| `athenapk_ppm_cfl03` | PPM+RK3 at CFL 0.3 instead of 0.4 — is the grid-scale pile-up a time-step artefact? |
| `athenapk_limo3`, `athenapk_wenoz` | AthenaPK's other high-order reconstructions — is the ranking a property of *this* limiter? |
| `astronomix_x64` | astronomix in double precision — is the ladder's x32 the reason its dissipation is low? |
| `seeds/` | two more forcing realisations per scheme — how big is the run-to-run scatter? |
| `timing/` | the `N = 64` trio re-run serially in one job (did sharing a node distort the wall clock?), and the short paired runs the per-output-event costs were measured from |

