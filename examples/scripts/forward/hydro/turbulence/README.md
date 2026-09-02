# Spectral convergence: astronomix WENO5 vs AthenaK PLM

How much of the turbulent cascade does each scheme actually resolve, and how
fast does its power spectrum converge with resolution?

Driven, statistically stationary, **subsonic isothermal hydrodynamic
turbulence** in a unit periodic box is run at `N = 64, 128, 256` in both

* **astronomix** — 5th-order finite-difference WENO5, characteristic-wise flux
  splitting, SSP-RK4 (`driven_turbulence.py`), and
* **AthenaK** — 2nd-order finite volume, PLM reconstruction + Roe, RK2
  (`athenak_turb.py`),

and the time-averaged kinetic-energy spectra are compared.

## Why this setup

* **Subsonic (M ≈ 0.3)** — no shocks, so the flow has a genuine Kolmogorov
  `k^-5/3` inertial range and the comparison measures reconstruction order
  rather than shock capturing. In the supersonic regime both schemes are driven
  toward `k^-2` by shocks and the order of accuracy matters much less.
* **Isothermal** — an ideal-gas box heats up as the injected energy thermalises,
  so the Mach number drifts down by ~20% over ten turnovers. An isothermal EOS
  pins it, which matters when the whole point is to average over a long
  stationary window.
* **Driven, not decaying** — a decaying run would need identical initial
  conditions in both codes (AthenaK has no path to ingest an external IC without
  a custom problem generator), and its spectrum is never stationary. Driving to
  a steady state and time-averaging is the standard code-comparison protocol.

## How the two codes are matched

astronomix's Ornstein-Uhlenbeck forcing was written to mirror AthenaK's
`turb_driver`, and this study uses that path (`ou_forcing`,
`ou_exact_injection`, `banded_spectrum`). Term by term:

| | AthenaK `turb_driver` | astronomix OU forcing |
|---|---|---|
| driving band | discrete `nlow ≤ n ≤ nhigh` in `n = kL/2π` | same (`banded_spectrum=True`) |
| envelope | `k^-(expo+2)/2`, isotropic (`driving_type=0`) | same |
| projection | explicit incompressibility of the mode amplitudes | solenoidal projection in Fourier space |
| time correlation | `f ← fcorr·f + gcorr·ξ`, `fcorr = exp(-dt/tcorr)` | same |
| amplitude | `s` solving `m0 s² + m1 s = dedt`, giving exactly `dedt·dt` of kinetic energy per step | same quadratic (`_exact_injection_amplitude`) |
| CFL | `dt = cfl·min_i(dx/λ_i)`, `cfl = 0.3` | `dt = C_cfl·dx/Σ_i λ_i`, `C_cfl = 0.9` — the same dt in 3D |

The box has unit volume, so AthenaK's `dedt` (a rate per unit volume) and
astronomix's `energy_injection_rate` (a total rate) are the same number.

### …except that `dedt` is not a resolution-invariant control in AthenaK

Handing both codes the same `dedt` turns out **not** to give both codes the same
turbulence. `check_athenak_driving.py` measures the stationary Mach number at
fixed `dedt = 0.0135`:

| driving | N=32 | N=64 | N=128 | halving `cfl` at N=32 |
|---|---|---|---|---|
| OU (`tcorr = 0.53`) | 1.078 | 1.186 | 1.317 | 1.078 → 1.131 |
| white (`tcorr = 0`) | 0.183 | 0.145 | 0.110 | 0.183 → 0.135 |

The Mach number drifts by 22% (OU) to 66% (white) across the ladder, and changes
with the time step at *fixed* resolution — so the driving strength, not just the
numerics, would be varying along the convergence sequence.

The cause is in `turb_driver`: the normalisation `s` solving
`m0 s² + m1 s = dedt` is computed from the **fresh, uncorrelated** increment
`force_tmp` (with `m0 ∝ dt`), while the field actually applied is the OU
accumulation `fcorr·force + gcorr·s·force_tmp`. Because `force_tmp` is
uncorrelated with the flow, the cross term `m1` averages to zero and the
normalisation collapses to the white-noise branch `s = √(dedt/m0) ∝ 1/√dt`, so
the applied acceleration grows as the time step shrinks. astronomix computes its
amplitude from the field it actually applies (`_exact_injection_amplitude` is
handed the post-OU-update `f`), and is correspondingly stable across the same
N = 64 → 128 step: Mach 0.3132 → 0.3175 under OU driving (1.4%) and 0.3356 →
0.3608 under white driving (7.5%, and expected — with the injection rate
genuinely fixed, a better-resolved cascade dissipates slightly less efficiently
at the outer scale). Both are far below AthenaK's 22–66% drift. The runs in
`data/white_driving_check/` are what that comparison was measured on.

So the codes are matched on the **achieved flow** rather than on the nominal
input parameter: same driving band, same solenoidal projection, same correlation
time, and the same stationary Mach number. astronomix keeps `dedt = 0.0135` at
every resolution; `calibrate_athenak.py` tunes AthenaK's `dedt` per resolution
(`dedt ← dedt·(M_target/M)³`, converging in one or two iterations) so each of its
runs sits at astronomix's Mach number. This is the physically meaningful match —
the spectrum's normalisation is set by the dissipation rate, which the Mach
number fixes, not by the value of an input keyword.

The two codes necessarily draw **independent random forcing realisations**, so
the comparison is statistical: spectra are averaged over the snapshots in the
stationary window (the second half of the run), and `spectra.py` records the
standard error across snapshots as the noise floor that any code-to-code
difference has to beat. AthenaK seeds its driver identically at every resolution
(`rstate.idum = -1`, not settable from the input file) while astronomix draws a
fresh field per grid, so astronomix's resolution sequence carries slightly more
realisation scatter than AthenaK's.

## Two systematics that had to be handled, not assumed away

**Finite-volume vs finite-difference sampling.** astronomix's FD state holds
point values; AthenaK's FV output holds cell averages, which is the true field
multiplied by `sinc(π n_i / N)` along each axis. Comparing them naively compares
a low-pass-filtered field with an unfiltered one, in AthenaK's disfavour. The
suppression is under 2% in energy below `n = N/12` but reaches 19% at `n = N/4`
— and the peak of the code-to-code power ratio sits at high `n`, so it matters
there: it moves that peak from 3.07x to 2.26x at N=64. `spectra.py
--deconvolve-fv` divides the transfer function out, and the deconvolved numbers
are the ones quoted below.

**Forcing realisation.** AthenaK seeds its driver identically at every
resolution (`rstate.idum = -1`, not settable from the input file); astronomix
draws a fresh field per grid. A raw `‖E_N − E_2N‖` therefore charges astronomix
for realisation scatter AthenaK never pays, and made the 2nd-order code look
better converged (0.30 vs 0.19). Normalising each spectrum by its own plateau
before differencing removes the amplitude offset and gives 0.219 vs 0.196.

## What is measured

* **Effective resolution** `n_1/2` — the shell at which the compensated spectrum
  `n^{5/3} E(n)` has fallen to half its plateau. Above it the spectrum is
  numerical dissipation, not physics, so `n_1/2 / N` is the fraction of the grid
  a scheme converts into resolved turbulence, and the ratio between codes at
  equal `N` is how much further into the cascade the high-order scheme reaches.
* **Self-convergence** — the relative L2 difference between `E_N` and `E_{2N}`
  for the same code over the shells they share.

## Results

All seven runs sit at Mach 0.313–0.328 (±2.4%), so every comparison below is on
matched turbulence. Numbers are with `--deconvolve-fv`; `n_1/4` is the shell
where the compensated spectrum has fallen to a quarter of its `n = 3–6` plateau.

| code | N | Mach | n_1/4 | **n_1/4 / N** | runtime |
|---|---|---|---|---|---|
| astronomix WENO5 | 64 | 0.3134 | 14.10 | **0.220** | 75 s |
| astronomix WENO5 | 128 | 0.3175 | 28.02 | **0.219** | 724 s |
| astronomix WENO5 | 256 | 0.3279 | 55.89 | **0.218** | 12103 s |
| AthenaK PLM+Roe | 64 | 0.3165 | 11.95 | **0.187** | 19 s |
| AthenaK PLM+Roe | 128 | 0.3199 | 21.13 | **0.165** | 133 s |
| AthenaK PLM+Roe | 256 | 0.3248 | 39.36 | **0.154** | 1617 s |
| AthenaK PLM+HLLE | 128 | 0.3179 | 13.06 | **0.102** | 125 s |

**The 5th-order scheme resolves a fixed fraction of its grid; the 2nd-order one
does not.** astronomix holds 0.220 → 0.219 → 0.218 of the grid across a factor
of four in resolution (`n_1/2 ∝ N^1.04`), so refining buys proportionally more
cascade. AthenaK's resolved fraction *shrinks*, 0.187 → 0.165 → 0.154
(`n_1/2 ∝ N^0.85`): part of each refinement is spent re-resolving what the
scheme's own dissipation removed. The gap therefore widens with resolution —
the effective-resolution ratio goes 1.18x → 1.33x → 1.42x, and the peak
per-shell power ratio 2.26x → 3.07x → 4.48x, against ~2% error bars.

The Riemann solver is a secondary effect but not a negligible one: at N=128,
swapping Roe for the more diffusive HLLE costs more effective resolution
(0.165 → 0.102) than 2nd → 5th order buys at that resolution (0.165 → 0.219).
Reconstruction order and flux function are comparable levers here.

**Cost points the other way.** At equal N astronomix costs 4.0x / 5.5x / 7.5x
more wall clock (the ratio grows because its runtime scales as `N^4.06` against
AthenaK's measured `N^3.21`). Matching astronomix's dissipation scale needs
AthenaK at ~1.2–1.5x the linear grid (1.8–3.5x the cells), which on measured
runtimes is roughly *half* the cost:

| equal quality | astronomix | AthenaK |
|---|---|---|
| n_1/2 = 14.1 | 64³, 75 s | 78³, ~33 s |
| n_1/2 = 28.0 | 128³, 724 s | 174³, ~429 s |
| n_1/2 = 55.9 | 256³, 12103 s | 389³, ~5639 s |

So on this problem the high-order scheme wins decisively *per cell* and loses
*per second*. Two caveats on that second claim: the AthenaK costs are
extrapolations from a fitted `N^3.21` (the 389³ point is beyond the measured
ladder), and this compares two implementations — JAX/Pallas against
Kokkos/CUDA — not two discretisations in the abstract, so it is a statement
about these codes on an A100, not about high-order methods generally. The
per-cell result is the implementation-independent one.

## Running it

The `turb` problem generator is not part of AthenaK's `built_in_pgens`, so it
needs its own build (its `#include "pgen.hpp"` also needs `src/pgen` on the
include path):

```bash
cd /export/home/lstorcks/athena/athenak
export CUDA_HOME=/export/home/lstorcks/cuda PATH=$CUDA_HOME/bin:$PATH
export CUDAHOSTCXX=/usr/bin/g++-12 NVCC_WRAPPER_DEFAULT_COMPILER=/usr/bin/g++-12
cmake -S . -B build-a100-turb -D Kokkos_ENABLE_CUDA=On -D Kokkos_ARCH_AMPERE80=On \
      -D PROBLEM=fluids/turb -D CMAKE_CXX_COMPILER=$PWD/kokkos/bin/nvcc_wrapper \
      -D CMAKE_CXX_FLAGS="-I$PWD/src/pgen"
cmake --build build-a100-turb -j 16
```

Then, per resolution (both are single-GPU; submit through `pq`):

```bash
# astronomix — dedt is resolution-invariant, so one value for the whole ladder
PYTHONPATH=$(git rev-parse --show-toplevel) python driven_turbulence.py --n 128

# AthenaK — calibrate dedt to astronomix's Mach first, then run at that value
python calibrate_athenak.py --n 128 --target 0.3175
python athenak_turb.py --n 128 --calibrated
```

and finally

```bash
python spectra.py --all          # -> data/spectra.npz
python make_figures.py           # -> figures/, plus the metric table
```

`run_smoke.sh` runs the whole pipeline at `N = 32` for one turnover as a
configuration check.

## Where the data lives

The home filesystem is nearly full, so only the reduced per-snapshot spectra
(a few kB per run) are kept in `data/`. Raw snapshot cubes — AthenaK's `.bin`
dumps always, astronomix's only under `--save-states` — go to
`/export/data/lstorcks/turb_spectra/`.
