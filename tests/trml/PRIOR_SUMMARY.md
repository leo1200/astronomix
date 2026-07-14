# Lagrangian tracers + a Fokker–Planck model for the TRML temperature PDF

This summarises the tracer-particle implementation in `astronomix` and the
Fokker–Planck (FP) self-consistency test of the turbulent-radiative-mixing-layer
(TRML) temperature PDF, following the stochastic-temperature note. All figures
below are the **N = 128 fiducial** run; the analysis reuses the FP solver in
`~/diffmix`.

Reproduce with:
```bash
# N=128 two-phase run (spin-up, then equilibrium-seeded dense recording)
TRML_PRESET=fiducial python trml_tracers.py
TRML_PRESET=fiducial python fokker_planck_test.py     # writes figures/fp_*.png
```

---

## 1. The model

A Lagrangian tracer's temperature is modelled as an Itô SDE,
`dT = A(T) dt + sqrt(D(T)) dW`, with drift `A(T)` (radiative cooling) and
diffusion `D(T)` (turbulent mixing). In statistical steady state the temperature
PDF obeys the stationary Fokker–Planck equation, whose first integral ties the
**marginal** `P(T)` to the **transition** coefficients `A`, `D` and a constant
probability flux `J`. The test: measure `A`, `D`, `J` from tracer temperature
increments, reconstruct `P̂(T)`, and compare to the equilibrium histogram.

## 2. Implementation

- **Tracer module** `astronomix/_modules/_tracers/`: mass-weighted (∝ρ) seeding,
  trilinear velocity/temperature interpolation, RK2-in-space advection (periodic
  in x/y, clamp/optional re-injection in z), and an optional flux-matched
  boundary recycling. Threaded through the refactor-branch `LoopState` /
  snapshot machinery; default-off so existing runs are untouched. 7/7 unit tests
  in `pytests/test_tracers.py`.
- **Two-phase run** (`trml_tracers.py`): **phase 1** spins up to equilibrium with
  no tracers; **phase 2** seeds tracers ∝ the *equilibrium* density and records
  densely over a short window.
- **Regeneration thermostat**: a fixed mass-seeded tracer set drains out of the
  thin mixing layer, which turns over on the cooling time `t_cool ~ t_sh/ξ` (much
  faster than the slow through-flow) and is refed by *un-tracered* hot inflow. To
  keep the Lagrangian marginal equal to the instantaneous mass PDF, each step a
  fraction `dt / t_relax` of tracers is re-drawn ∝ the *current* density
  (`t_relax = 0.2 t_sh`, well above the diffusive-window lags). A per-tracer
  **generation counter** is recorded so the analysis drops increments spanning a
  regeneration — essential, since a position-jump heuristic misses a regeneration
  into a nearby cell and a single such spurious δT swamps the tiny cold-phase
  diffusion. Trajectory segments between regenerations are genuine and unbiased.

## 3. Efficiency — the bottleneck was the histogram, not the tracers

Profiling showed the tracer kernels are cheap (advance 0.18 ms, interpolate
0.05 ms; hydro ≈ 7 ms/iter at N=64). The cost was the per-snapshot temperature
**histogram**: `jnp.histogram` / `jnp.bincount` lower to a sort / scatter-add
that **serialises inside the integration `while_loop`** (~140 ms each, two per
snapshot → ~1.7 s/iter). Replacing it with a **scatter-free one-hot reduction**
(`astronomix/_snapshotting/_snapshot_diagnostics.py::_fast_uniform_histogram`)
brings the full recording loop from **1730 → 5.5 ms/iter (~300×)**. This is what
makes dense recording and high resolution tractable:

| run | before | after |
|---|---|---|
| N=64 / 8 t_sh recording | ~40 min | ~20 s |
| **N=128 fiducial (25 t_sh total)** | ~50 h (est.) | **~8 min** (4.5 spin-up + 3.5 record, 0.028 s/iter) |

No custom Pallas tracer backend is needed.

## 4. Results (N = 128)

### Summary figure
Left: the measured transition kernel `p(δT/Δt | T(t))` at the validity-window lag,
with the **drift `A(T)`** (cyan) and the **diffusion `D(T)`** as the
`±√(D/Δt)` band. Right: the three temperature PDFs — Eulerian mass-weighted,
Lagrangian tracers, and the **Fokker–Planck `P̂(T)`** reconstructed from *those
same* `A(T)`, `D(T)`, `J`. The Lagrangian and Eulerian curves overlay (Check 0),
and `P̂` reproduces them.

![Summary](figures/fp_summary.png)

### Check 0 — Lagrangian marginal vs mass-weighted Eulerian PDF
With the regeneration thermostat the mass-seeded tracers reproduce the Eulerian
mass-weighted PDF **across the full range, including the mixing band**:
**L1 ≈ 0.01**. (Without regeneration a fixed tracer set seeds correctly but then
drains from the thin, fast-turnover mixing band — the shaded region, holding only
~0.2–0.3 % of the mass — because the inflow that refeeds it is un-tracered;
`A(T)`, `D(T)`, `J` are local and unaffected, so the reconstruction recovered the
Eulerian `P(T)` either way, but the regenerated marginal now matches directly.)

![Check 0](figures/fp_check0_marginal.png)

### Drift and diffusion from tracer increments
`A(T)` (drift, lag-independent as expected) and `D(T; Δt)` from the connected
conditional moments of `δT = T(t+Δt) − T(t)`, binned in `T(t)`.

![Drift and diffusion](figures/fp_drift_diffusion.png)

### Conditional transition density p(δT/Δt | T(t))
The transition kernel the diffusion ansatz models as Gaussian. Plotting the
Lagrangian **rate** `δT/Δt` (column-normalized at each `T(t)`) makes the two
coefficients read off directly: the cyan solid line is the conditional mean,
which equals the **drift `A(T)`**, and the cyan dashed band is
`A(T) ± sqrt(D(T)/Δt)` — its half-width *is* the **diffusion `D(T)`**. As `Δt`
grows the band `sqrt(D/Δt)` shrinks and the cloud collapses onto `A(T)`
(random-walk increments averaging out into the deterministic drift) — the
ballistic→diffusive transition, panel by panel. The drift is negative (cooling)
through the mixing range and ≈0 at the cold floor.

![Transition density](figures/fp_transition_density.png)

### The diffusive window
`D(T; Δt)` rises from zero (ballistic, `Var(δT)∝Δt²`), **plateaus at
Δt ≈ 0.01–0.05 t_sh**, then bends over once Δt approaches the macroscopic scale —
the diffusive window the note predicts. Resolving it required the fine lag
spacing (`Δt_min ≈ 0.004 t_sh`) enabled by the efficiency fix.

![Diffusion plateau](figures/fp_diffusion_plateau.png)

### Reconstruction P̂(T) and the validity window
Reconstructing `P̂` from `(A, D, J)` (flux `J ≈ 3×10⁻³`, ~constant in T)
reproduces the full equilibrium PDF — cold peak, mixing plateau, hot rise —
**best at Δt ≈ 0.008 t_sh (support-L1 = 0.12; mixing-range L1 ≈ 6×10⁻⁴)**, in the
same window where `D` plateaus. The march is taken from the cumulative-mass
support edge (the implicit march is noise-sensitive in the empty tails where
`D ≈ 0`). `J ≠ 0` is essential — `J = 0` collapses `P̂` to the cold peak.

![Reconstruction](figures/fp_reconstruction.png)
![Validity window](figures/fp_validity_window.png)

## 5. Damköhler (ξ) scan — and a metric caveat

The note predicts the diffusion picture should *pass at high Damköhler number and
fail at low* — ξ = t_sh/t_cool,min sets the timescale separation
τ_c ≪ Δt ≪ τ_macro a diffusive window needs. Scanning ξ ∈ {3, 10, 30, 100, 300,
1000} at fixed N=64, M=0.5, χ=100 (`xi_scan.sh`, `xi_scan_summary.py`):

| ξ | abs L1 of dM/dlogT | **log-shape ⟨\|Δlog₁₀P\|⟩ (dex)** | KL | Check 0 L1 |
|---|---|---|---|---|
| 3 | 0.262 | **0.33** | 0.72 | 0.012 |
| 10 | 0.207 | **0.23** | 0.38 | 0.007 |
| 30 | 0.143 | **0.27** | 0.20 | 0.009 |
| 100 | 0.132 | **0.25** | 0.16 | 0.010 |
| 300 | 0.121 | **0.28** | 0.13 | 0.014 |
| 1000 | 0.103 | **0.28** | 0.07 | 0.017 |

**Metric caveat (important).** The *absolute* L1 falls with ξ, but that is largely
an artifact: at high ξ the PDF concentrates into the cold peak, so the
mixing-range values shrink and absolute deviations shrink with them (in the
mixing band the absolute L1 drops ~30× while the *relative* deviation is flat).
The **magnitude-independent log-shape deviation stays ~0.25–0.3 dex at every ξ**,
and KL falls only because it is mass-weighted by the (concentrating) cold peak.
So **the relative reconstruction quality is ~ξ-independent** over the range
probed — the data do *not* show the diffusion model becoming more faithful at
high Da.

The more direct test — the `D(T;Δt)` window — only **shifts to smaller Δt** as ξ
rises (faster cooling → shorter τ; peak Δt/t_sh goes 0.51 → 0.004), without
getting flatter/wider; by ξ=1000 the window sits at our `Δt_min = 0.004 t_sh`
floor (the ballistic rise is no longer resolved). Check 0 stays flat (~0.01,
ξ-independent), so this is a statement about the *physics/measurement*, not the
tracer sampling.

Caveat on the caveat: at high ξ we run out of *temporal* resolution, so a proper
high-Da test needs a **finer recording cadence** (smaller Δt_min) to resolve the
window where it has moved — that is the right follow-up before concluding either
way on the Da trend.

![ξ scan](figures/fp_xi_scan.png)

## 6. Takeaways

- Lagrangian tracers in `astronomix` work and, seeded at equilibrium, faithfully
  sample the mass-weighted temperature PDF (Check 0 passes, L1 < 0.01).
- The FP / diffusion picture **holds, approximately, in a window** Δt ≈ 0.01–0.05
  t_sh: there `D(T)` peaks/plateaus and `P̂` reproduces the *shape* of the
  equilibrium PDF to ~0.25–0.3 dex. It is not exact — a modest peak rather than a
  wide flat plateau — and the relative quality does **not** clearly improve with
  Damköhler number over ξ ∈ [3, 1000] (§5; the apparent abs-L1 trend was a
  magnitude artifact). Whether a genuinely clean window emerges at high Da is
  unresolved — it needs finer temporal cadence, since the window moves to smaller
  Δt as ξ grows.
- The headline engineering result is the ~300× recording speed-up, which is what
  makes the fine-Δt measurement — and high resolution — feasible.

### Caveats / next steps
- Report magnitude-independent metrics (log-shape, KL) alongside the absolute L1
  of the PDF — the absolute L1 is biased by how concentrated the distribution is.
- The regeneration thermostat keeps the Lagrangian marginal on the mass PDF; its
  timescale (`0.2 t_sh`) caps the usable increment lag (segments between
  regenerations), so the largest-Δt part of the validity scan loses statistics.
- **Finer recording cadence** (smaller Δt_min) is the key next step: at high ξ
  the diffusive window has moved below the current Δt_min = 0.004 t_sh, so the
  high-Da regime is under-resolved in time.
- The cold phase cools below the nominal `T_cold`; the PDF range must bracket the
  actual gas (`[10⁻³, 1.5]` here), not `[T_cold, T_hot]`.
