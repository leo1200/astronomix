# Scale-Resolved Initial-State Reconstruction in 2D Kelvin–Helmholtz Flow

*Twin (synthetic) experiment with `astronomix` (differentiable hydro in JAX). The
ground-truth initial perturbation `u0` is known, so all errors are measurable.*

## Claims under test
1. **Information frontier (mode 3).** A wavenumber `k_rec(T)` separates large
   scales that retain memory of `u0` from small scales scrambled by mixing; it
   recedes to lower `k` as `T` grows and tracks `1/h(T)` (mixing-layer width).
   Method-independent.
2. **Single-shooting failure (modes 1 & 2).** The cold-start gradient norm grows
   `~ e^{λT}`; single shooting fails to recover modes *inside* the recoverable
   band — the optimization frontier is strictly tighter than the information
   frontier.
3. **Soft multiple shooting closes the gap.** Segmenting + enforcing continuity
   only on large scales (low-pass) recovers the large-scale `u0` up to the
   information frontier, beating single shooting and hard (full-field) MS on the
   recoverable band.

## Setup (Stage 0–1)
- 2D viscous KH, box `[0,1]²`, `N` cells, periodic x (flow) / open y. Base tanh
  shear `vx=(dV/2)tanh((y-yc)/δ)`, `dV=1`, `δ=0.02`, low Mach (`Ma=0.3`), dynamic
  viscosity `μ = ρ dV δ / Re`, `Re=2000`. Growth time `t_g = δ/dV`.
- Unknown `u0`: broadband transverse-velocity seed, power across streamwise modes
  `kx ∈ [2,32]`, localized on the shear layer; rms `1e-2 dV`.
- Observation `H`: full final velocity `u_T` (options: Gaussian noise, masking).

## Figure ↔ claim map
| Figure | File | Supports |
|---|---|---|
| Roll-up / regime | `kh_rollup.png`, `stage1_vorticity.png`, `stage1_regime.png` | regime selection (coherent large + filamented small) |
| **RECOVERY (headline)** | `fig_recovery.png` | claims 2 & 3: (b) only observable-subspace control recovers at `T=5`; (a) two-frontier recovery-vs-`T` |
| **SVD frontier (headline)** | `frontier_svd.png`, `frontier_svd_Re500/250.png` | claim 1: structured-SVD `k_rec(T)` recedes; `σ_max~e^{λT}` |
| Tikhonov L-curve | `fig_lcurve.png` | claim 2/3: implicit null-space suppression (U-curve, min `lowk≈0.24`) |
| C — gradient norm vs T | `fig_C_gradient_mechanism.png` | claim 2 / mode 1 (`e^{λT}`, segmentation cap) |
| D — basin / warm-start | `fig_D_basin*.png` | mode-2 / over-parametrization (full-field pushed out) |
| A — recovery vs k (old) | `fig_A_recovery_vs_k.png` | superseded null (soft-MS untuned); kept for record |

## Findings (filled as runs land)
- **Stage 0 (forward + AD):** billows roll up; AD gradient finite. Random-direction
  FD matches AD to ~8%; the gradient-direction FD is off by ~2× — a WENO **kink**
  (piecewise-smooth solver → valid one-sided subgradient), the documented
  shock-capturing-AD caveat. Comparisons use the same gradients, so they remain
  fair; spectra/SVD use the local tangent.
- **Stage 1 (regime):** GO. Vorticity shows coherent large-scale vortices (with
  late pairing) plus thin filaments wound out and viscously erased; momentum
  thickness grows monotonically (`0→0.1`); perturbation spectrum develops a
  low-`kx` peak and a high-`kx` viscous cutoff. (The per-`kx` seed–final phase
  correlation is too noisy to use as a frontier; the tangent SVD is the rigorous
  measure.) Sweet-spot horizons `T ≈ 66–130 t_g`.
- **Stage 2 (tangent gain / mode 1):** The tangent gain grows **`σ_max(T) ~ e^{2.1 T}`**
  (`σ_max = 3 → 70 → 376 → 1306` at `T = 20/60/100/160 t_g`) — clean exponential
  gradient/adjoint explosion (**mode 1 confirmed**). Per-`kx` gain peaks at the
  unstable KH band (`kx≈2–3`) and rolls off ~1 decade toward high `kx`.
  - **Honest reframing of the "information frontier."** The per-`kx` forward gain
    is dominated by the leading Lyapunov direction (every input projects onto it
    and is amplified `~e^{λT}`), so it conflates "grows" with "recoverable" and
    does *not* by itself give a receding mode-3 edge. At `Re=2000`/`N=128` the
    small scales are still *forward-imprinted on `u_T` above noise* — they are
    lost to **optimization (modes 1 & 2)**, not to **information (mode 3)**. A
    sharp mode-3 frontier lives below the viscous scale (≈ grid here); to make it
    resolved would need **lower `Re`** (a clean future run). We therefore take the
    physical recoverable band as the **KH marginal-stability boundary
    `k_rec(T) ≈ 1/(2π h(T))`** (from the momentum thickness `h`, which grows so
    `1/h: 98→12`), and let the **reconstruction itself** be the operational
    recoverability test, overlaying `1/h`.
  - **Stage 2b — structured-SVD frontier (the fix; `frontier_svd.png`).** The
    per-`kx` gain's Lyapunov contamination is removed by taking the SINGULAR
    spectrum of the tangent map restricted to a shear-localized streamwise-mode
    basis (Gram matrix of forward jvps `G=(MV)(MV)^T`, eigendecomposed — uses
    *only* forward-mode AD, so it sidesteps the vmap-through-ghost-cell-padding
    crash of a naive vjp randomized SVD). The leading singular vector absorbs the
    Lyapunov direction; sub-leading `σ_a` expose the genuinely independent
    recoverable directions. Result (`Re=2000`, `N=64`): the singular spectrum
    **decays faster as `T` grows** and the recoverable edge **recedes**,
    `k_rec(T): 32→32→32→22` at `T=5/10/20/40 t_g`, while `σ_max` grows
    `2.0→1.4→2.9→9.9` (exponential at long `T`). So the SVD *does* show the
    receding mode-3 frontier the per-`kx` gain hid. Crucially, at **short `T` the
    within-band conditioning is benign** (`σ_max/σ_min ≈ 21` at `T=5`, all
    `kx≤32` above the 1% floor): recovery is information-feasible there, so a
    failure at short `T` is **optimization/regularization**, not information.
    (Lower-`Re` runs sharpen the recession — Re=500/250, `figures/frontier_svd_Re*.png`.)
- **Stage 4a — mode-1 fix by segmentation (`fig_C_gradient_mechanism.png`).** From
  the Stage-2 gain data: single shooting's back-prop gradient rides `σ_max(T)`,
  while `M`-segment multiple shooting only back-propagates one segment of length
  `T/M`, so its per-segment gradient is `σ_max(T/M)`. At `T=160 t_g` this is
  **`2355 (single) → 82 (M=2) → 15 (M=4) → 6.6 (M=8)`** — segmentation caps the
  gradient at `e^{λT/M}` (~360× smaller for `M=8`). Robust, independent of
  reconstruction success.
- **Stage 4b — cold-start reconstruction at `T=80 t_g` (mode 2).** Both single
  shooting *and* soft-MS (`μ=30`) **decrease the data misfit while diverging from
  truth** (`lowk_err: ~1 → 6–7` as `J` drops ~50×): they fit the noisy
  observation with the *wrong* IC. With mode-1 magnitude clipped, this is the
  **mode-2 multimodality** signature — at a long horizon from a cold start the
  basin is lost for *all* methods (consistent with the TGV study). So the
  soft-MS *recovery* advantage is not free at long horizons; it shows up in the
  **basin** test (warm start), below.
- **Stage 4c — warm-start basin test (`T=50 t_g`, `fig_D_basin.png`).** Started
  *near the truth* (`lowk_err≈0.3–0.4`), single shooting is **pushed out of the
  truth basin within ~20 steps** (`0.40 → 1.92`) while only marginally reducing
  the data misfit — it fits the noise and cannot return to truth (clean mode-2).
  Soft-MS (`μ=30`, untuned) is **also pushed out** (`0.31 → 1.34`, with the total
  loss even rising — an optimizer instability from the stiff defect term at this
  `lr`), so it does **not** rescue recovery here. A gentler retry
  (`lr=3e-4`, warm `0.15`, noise `3e-3`) **confirmed the null** robustly: single
  `0.14→0.65`, soft `0.13→0.56` — both pushed out, soft only marginally less
  (`fig_D_basin_warm2.png`). So the soft-MS advantage is absent at `μ=30` across
  cold + two warm settings.
- **Stage 4d — RECOVERY ACHIEVED: it was over-parametrization, not the optimizer
  (`recover_modes.py`, `fig_recovery.png`).** Root cause of the Stage-4b/4c null:
  the full-field control has ~768 DOF (after the `kx≤6` band-limit) but the truth
  seed lives in only **~10 DOF** — transverse only (`svx=0`), a single *known*
  shear-layer envelope `env(y)`, and `kx∈[2,6]`. The other ~758 DOF are an
  **unobservable null space**; with observation noise the optimizer fits the
  residual with null-space junk, so `lowk_err≈1` (cold) and a warm start is
  *pushed out* (`0.30→0.36`). Two diagnostics pin this down: (i) starting *at*
  truth with **no noise**, the full-field optimizer *stays* (`lowk≈0.08`) — truth
  **is** a minimizer; (ii) Tikhonov on the full field *does* recover the bulk once
  the penalty is large enough to suppress the null space — the `T=5` L-curve is a
  clean U bottoming at `lowk≈0.24` at `α≈1e-2` (`fig_lcurve.png`; the earlier
  `1e-8..1e-5` grid was inert because the reg term was ~50× below the misfit). The
  *cleaner* fix is to **parametrize the control in the observable subspace** — the
  physical KH prior (transverse, shear-localized `env(y)`, band-limited), giving a
  ~10-D, fully-observable inverse (SVD `cond≈21` at short `T`). Single shooting then
  **recovers cleanly and fast**: at `T=5 t_g`, `lowk_err: 0.876 → 0.003`,
  **early-stopping at iteration 89** when the data misfit hits the discrepancy
  floor. Method comparison at `T=5`: full-field cold `1.69`, warm `0.36`,
  Tikhonov-best `0.24`, **mode-space `0.003`** — explicit null-space removal beats
  implicit (Tikhonov) by ~80×.
  Operational frontier test (scan): mode-space recovery holds while the SVD
  `k_rec(T) > kmax` and degrades once the horizon pushes `k_rec` below the control
  band — *a good method reaches the information frontier; none crosses it*.
  - **Two frontiers (mode 2 vs mode 3).** The horizon scan shows the
    **optimization frontier is tighter than the information frontier**: at
    `Re=2000`, recovery is excellent at `T=5/20 t_g` (`lowk≈0.003/0.004`) but
    collapses by `T=40` (`0.76`) — yet the SVD says `k_rec=22>6` is still
    information-recoverable there. The gap (`T≈30→80`) is "lost to optimization,
    not information." Lowering `Re` (more viscous, smoother landscape) **extends
    the optimization frontier outward**: `Re=250` still recovers at `T=40`
    (`0.14`) where `Re=2000` has failed — the higher viscosity damps the chaotic
    multimodality even as it tightens the *information* frontier at very long `T`.
  - **Robustness of the prior (`T=20 t_g`, `Re=2000`).** Recovery degrades
    *gracefully* with prior mismatch, never back to the full-field failure: exact
    envelope `0.004`; a flexible 5-width `y`-basis (50 DOF) `0.31`; a wrong
    single envelope (2× too wide) `0.47` — all far below full-field `~1.6`. The
    lever is **dimensionality/observable-subspace match**: even mild
    over-parametrization (50 vs 10 DOF) costs an order of magnitude, so the result
    is not an artifact of knowing the exact envelope.
- **Stage 4e — does multiple shooting help recovery, tested fairly?
  (`recover_modes_ms.py`).** Earlier soft-MS runs were confounded by the full-field
  null space (every method failed for that reason). Re-tested with the **clean
  observable-subspace IC** (only the IC is mode-space; interior segment states are
  full fields pinned by data+continuity) at `T=40 t_g, Re=2000`, where mode-space
  single shooting fails by **mode 2** (`lowk≈0.78`, inside the info frontier).
  Result: MS **helps modestly but does not rescue** — `single 0.78`, `soft M=2
  (μ=30) 0.64` (best), `soft M=8 0.67`, `hard M=4 0.73`, `soft M=4 (μ=100) 0.78`;
  all stay deep in the failed band (`≫0.1`) and none reach the discrepancy floor.
  Notable: **`M=2` beats `M=4/8`** (more segments = more interior DOF that slow
  convergence), and gentle soft (μ=30) beats both hard continuity and high μ. So
  the `T=40` optimization frontier is genuine IC→data ill-conditioning that
  persists per-segment, not the long-horizon adjoint pathology segmentation cures.
  **Borderline check (`T=30`, single marginal):** `single 0.115`, `soft M=2 (μ=10)
  0.106` (a tie), `soft M=2 (μ=30) 0.175`, `soft M=4 (μ=30) 0.421` — MS gives **no
  reliable recovery gain**, and the extra interior-state DOF of `M=4` actively
  *hurt* (0.42 vs 0.12). **Verdict: in KH, multiple shooting does not provide a
  dependable recovery-accuracy advantage** (at best a marginal one for gentle
  `M=2`); its robust value is the **mode-1 gradient cap**, and the real recovery
  lever is observable-subspace parametrization with plain single shooting.
- **Stage 4f — attempting to CONSTRUCT a mode-2 advantage: vortex pairing +
  reduced-order MS (`mode2_*.py`, `fig_mode2.png`) — honest null.** To give MS its
  best shot, built a setting designed for mode-2: unknown = 2-mode seed (`kx=2`
  fundamental + `kx=1` subharmonic, 4 DOF), horizon `T*=65 t_g` spanning a
  vortex-**pairing** event (the `kx=1↔kx=2` phase selects which way billows merge →
  intended many-to-one folding), with **Path-A reduced-order MS**: interior
  segment states confined to a 28-mode POD basis of the KH flow manifold (99.88%
  ensemble variance) so they are determined (not the earlier null-space confound).
  Multi-restart (8 cold starts), mode-1 clipped. **Result: no advantage, two
  mechanisms.** (i) The setting is **not multimodal** — a 4-DOF seed is
  over-determined by the full terminal field, so single shooting recovers from
  *every* restart (`ic_err` median `0.05`, std `0.07`, data `~3e-6`); pairing did
  not fold the low-D inverse into separate basins. (ii) **Reduced-order MS is
  *worse*** (`ic_err≈0.58` all restarts, data floored at `1.4e-3`): the POD basis
  cannot represent the *specific* truth interior exactly, and that truncation error
  biases the continuity-coupled IC. The reduced basis that *determines* the problem
  also *biases* it. (Probe of longer horizons `T=100/130` to test whether any
  multimodal-but-identifiable terminal-only regime exists: in progress.)
- **Stage 4g — a terminal-only mode-2 regime DOES exist; soft+Adam MS still does
  not cure it (`make_fig_basin.py`, `fig_basin_mode2.png`).** Corrected the test:
  drop the biased POD interior, use **full-field interior + soft continuity** (the
  unbiased MS), and run the real diagnostic — **multi-restart with amplitude-scaled
  inits** (8 cold starts at init rms ~ seed amplitude, *fixed* truth), kx2-6
  (10 DOF), `T=45 t_g, Re=2000` (`k_rec=22` → identifiable). **Single shooting is
  cleanly multimodal:** ic_err scatters `{0.05, 0.06, 0.08, 0.18, 0.51, 0.64, 0.67,
  1.00}`, and the **loss discriminates the basins**: the truth basin (ic`~0.05`)
  sits at the noise floor `J≈1.2e-7`, the wrong basins are higher-loss local minima
  `J=0.8e-6..2e-6` (**7–17×** higher). So it is genuine mode 2 — a *unique global
  minimum at truth* with spurious higher-loss minima, **not** a mode-3 degeneracy —
  and the loss is a usable basin discriminator (multi-restart + pick-lowest-`J` →
  truth). So terminal-only KH *can* be mode-2. **But soft MS (M=2, μ=30)
  does not cure it:** ic_err `{0.34..0.94}`, std `0.34→0.18` (variance shrinks) but
  the floor *rises* — MS best `0.34` vs single best `0.05`. MS trades multimodality
  for **bias** (soft continuity at finite μ → a continuity-approx trajectory that
  fits the terminal off-truth): its solutions sit at `J=2e-7..9e-6` (**2–8× above
  the noise floor**) — MS **never reaches the global minimum from any init**, so it
  loses on *both* loss and ic_err. Practically, multi-restart single shooting +
  pick-lowest-`J` (→`0.06`) beats MS (best-`J` → `0.42`).
  - **Important caveat (why MS may still deserve the mode-2 credit elsewhere).**
    The classical MS basin-enlargement (Bock) is realized by a **constrained
    Gauss-Newton / SQP** solver (simultaneous Newton step on all states with
    continuity as hard equality constraints), *not* by a soft penalty + first-order
    (Adam) optimizer. Our soft+Adam MS is a surrogate; the negative result here is
    therefore "soft-penalty first-order MS does not cure KH mode-2," which may be an
    **optimizer-formulation** limitation, not proof that MS-as-a-formulation can't.
    A proper constrained-GN multiple shooting (per-segment AD Jacobians + KKT solve)
    is the untested route that could still show the textbook advantage.
  - **The structural tension.** A mode-2 advantage needs (a) an unknown high-D
    enough you can't multi-restart out of it, (b) a multimodal horizon, (c)
    *constrained* interior states. Terminal-only conflicts: low-D = unimodal (or
    multi-restartable, as here), high-D = null space, reduced-basis = bias,
    soft-penalty = bias. The cleanest realizations are **time-distributed obs**
    (interior pinned by data) and/or a **constrained-GN** MS solver — neither of
    which is the soft+Adam terminal-only setup that recurs throughout this study.
- **Stage 4h — the "mode-2 basins" were largely a FIRST-ORDER artifact
  (`recover_gn.py`, `fig_gn_vs_adam.png`).** Decisive control: rerun the same
  multimodal regime (`T=45 t_g, kx2-6, 8 cold inits`) with **single-shooting
  Gauss-Newton** (Levenberg-Marquardt; the low-D IC lets `J=du_T/dp` be formed by
  `np` forward-mode passes — no adjoint). **GN recovers from all 8 inits to the
  noise floor** (`ic_err 0.006..0.044`, 8/8) where **Adam recovered only 3/8**
  (`0.05..1.00`). So the basins that trapped first-order Adam **dissolve under a
  second-order optimizer** — they were an optimization artifact, not genuine
  multimodality. **Crucial consequence for MS:** by the **condensing equivalence**,
  terminal-only constrained-GN *multiple* shooting with cold interior init reduces
  to exactly this single-shooting GN step (the condensed IC subproblem uses the
  same full-horizon sensitivity `H'·∏Gⱼ·B`), so MS-GN would inherit GN-single's
  (already excellent) behavior and add nothing. The remaining MS value is the
  **mode-1 conditioning cap** at *long* horizons, where `J` itself becomes
  `e^{λT}`-ill-conditioned and per-segment GN blocks (`e^{λT/M}`) would help — the
  same robust mode-1 benefit, now in the Gauss-Newton Hessian rather than the
  clipped gradient.
- **Stage 4i — GENUINE mode-2 found at long T, and the MS principle (continuation)
  cures it (`gn_conditioning.py`, `recover_gn.py`, `fig_optimizer_horizon.png`,
  `fig_gn_conditioning.png`).** Pushing the GN multi-restart to longer horizons
  (kx2-6, Re=2000) maps a sharp transition: GN recovers **8/8** at `T=45`, **6/6**
  at `T=50`, ~`2/6` (borderline ~0.1) at `T=60–70`, and **0/6** at `T≥80` — there
  it converges to *wrong minima at ≈the noise floor* (`T=80` init-3: cost `2.5×`
  floor, ic_err `1.23`). This is **genuine nonlinear multimodality**, confirmed by
  the GN-Jacobian SVD: at the onset `κ(J)≤253` (vs the float32 wall `~1e7` — so
  **no mode-1 precision issue**, even at `Re=20000`/near-inviscid where `κ` only
  reaches `~1300` by `T=160`) and `σ_min≫noise` (**no mode-3**). So the long-`T`
  failure is *neither* conditioning *nor* information — it is mode 2. **The
  multiple-shooting basin-enlargement principle escapes it:** a **horizon-
  continuation** chain (warm-start GN along `T=50→60→…→120`, i.e. MS/homotopy
  realized in the horizon) recovers `ic_err: 0.04→0.03→0.02→0.01→0.004` at
  `T=60/70/80/100/120` — where *cold* GN fails (`T=80` best `0.62`, `T=120` best
  `1.63`). It even gets **more** accurate with horizon (longer window → more signal
  once inside the truth basin). So the genuine mode-2 obstacle is real and the MS
  idea genuinely cures it — but the effective, tractable form of MS here is
  **horizon continuation**, not a full constrained-MS-GN solver.

## Conclusion (honest)
On 2D viscous KH all three claims are now demonstrated, plus a sharper lesson
about *what* makes PDE-constrained reconstruction hard.

- **Mode 1 (gradient explosion) — SOLID.** The back-prop/adjoint tangent gain
  explodes `σ_max(T) ~ e^{λT}`, and **segmentation caps it at `e^{λT/M}`**
  (`2355→6.6` from single to `M=8` at `T=160 t_g`; `fig_C`).
- **Mode 3 (information frontier, claim 1) — SOLID via the structured SVD.** The
  per-`kx` forward gain is Lyapunov-contaminated and hides the frontier; the
  **singular spectrum** of the tangent map restricted to the shear-localized mode
  basis (forward-jvp Gram matrix) recovers it. The recoverable edge **recedes** —
  `k_rec: 32→32→22→6` (Re=500) and `→5` (Re=250) over `T=10/20/40/80 t_g` — while
  `σ_max→~70` (mode 1). Lower `Re` sharpens the recession (`frontier_svd*.png`).
- **Mode 2 — RE-DIAGNOSED twice, and now resolved.** Two distinct things were
  conflated under "mode 2". (i) At moderate `T` in the over-parametrized full-field
  control, the apparent failure was **over-parametrization** (control ~75× the
  observable-subspace DOF; noise fills the null space) — fixed by subspace
  parametrization. (ii) In the *clean* (mode-space) control, the multi-restart
  "basins" at `T=45` were a **first-order (Adam) artifact**: Gauss-Newton recovers
  8/8 there. **Genuine mode 2 does exist** — but only at **long horizons** (`T≳80`,
  `Re=2000`): cold GN converges to wrong minima at the noise floor (well-conditioned
  `κ≤253`, `σ_min≫noise`, so not mode 1/3). Two things cure it: **(i)** the MS
  basin-enlargement principle as **horizon continuation** (`ic_err 0.004` at `T=120`
  where cold GN fails); **(ii) literal constrained-GN multiple shooting** — at
  `T=80`, GN-MS (M=4, feasible interiors) recovers **3/12** cold starts (best
  `0.036`) where single-shooting GN recovers **0/12** (best `0.33`). So the MS
  *formulation* genuinely wins **but only with the right solver**: soft-penalty +
  Adam MS does not beat single shooting; constrained Gauss-Newton MS does (a real,
  partial basin enlargement). Moderate-`T` "mode 2" = optimizer/parametrization
  artifacts; genuine long-`T` mode 2 is real and curable by continuation **or**
  constrained-GN MS.
- **Recovery (claim 3 reframed) — ACHIEVED.** Restricting the control to the
  **observable subspace** (the physical KH prior: transverse, shear-localized,
  band-limited) makes the inverse ~10-D and well-conditioned; single shooting then
  recovers the seed to `lowk_err≈0.003` at `T=5 t_g`, converging at the
  discrepancy floor in ~90 iterations (vs full-field `1.7` / Tikhonov-best `0.24`).
  The horizon scan exposes **two frontiers**: an **optimization frontier** (where
  even the well-posed mode-space recovery fails: `T≈30` at `Re=2000`) that is
  *tighter* than the **information frontier** (SVD `k_rec>kmax` until `T≈80`).
  Lowering `Re` smooths the landscape and pushes the optimization frontier
  outward until recovery fails right at the information frontier (`Re=250` fails at
  `T≈80`, where `k_rec→5`). A good method reaches the information frontier; none
  crosses it (`fig_recovery.png`).

**Net.** The headline is not "multiple shooting beats single shooting" (that
advantage stayed regime/`μ`-dependent and modest, as in the TGV study) but the
cleaner result: *the binding constraints are (1) the exponential adjoint gain,
curable by segmentation, and (3) the information frontier, an information limit
no optimizer can beat — and between them, success hinges on parametrizing the
unknown in its observable subspace rather than on the shooting scheme.*

(The soft multiple-shooting *recovery* win specifically stayed `μ`-dependent and
modest — the `μ`-sweep at `T=80 t_g` was a uniform null, all methods past the
practical frontier — consistent with the TGV study. Segmentation's robust value
here is the **mode-1 gradient cap**, not a recovery advantage.)

## Reproducibility
Module layout: `problem.py` (forward/seed/observe), `metrics.py` (spectra,
filters, per-k error, mixing width), `information_frontier.py` (per-kx tangent
gain), `frontier_svd.py` (structured-SVD frontier, forward-jvp Gram),
`single_shooting.py`, `multiple_shooting.py` (full-field single/hard/soft + KH_ALPHA
Tikhonov), `recover_modes.py` (observable-subspace recovery; KH_CTRL_ENVW/KH_NYB
robustness), drivers `run_alpha_sweep.sh` / `run_modes_scan.sh`, figure scripts
`make_fig_recovery.py` / `make_fig_lcurve.py`. Noise scaled to the *perturbation*
imprint `rms(u_T-u_base)` (NOISE=1e-2 ⇒ SNR=100). RNG seeds fixed
(`jax.random.PRNGKey`). astronomix version recorded in `configs/`.

## Money tables + exact reproducibility

**Shared setup.** 2D viscous KH, `astronomix` finite-difference (RK4-SSP), `N=64`,
box `[0,1]²`, periodic-x / open-y. `KHParams(n=64, yc=0.5, delta=0.02, dV=1.0,
rho0=1.0, mach=0.3, gamma=5/3, reynolds=2000, k_min=2, k_max=6, seed_amp=1e-2,
env_width=0.04)`; growth time `t_g = delta/dV = 0.02`. Truth IC = broadband
transverse seed `random_broadband_seed(PRNGKey(0))` (vy only, `kx∈[2,6]`,
Gaussian-`env(y)` on the layer, rms `1e-2 dV`). Observation = terminal velocity
field `u_T` + Gaussian noise `nsd = NOISE·rms(u_T − u_base)`, `NOISE=1e-2`
(SNR=100 on the perturbation imprint), noise key `PRNGKey(12345)`. Control =
mode-space coeffs `c∈R^{nk×2}`, `seed(c)=env(y)·Σ_k[c_kᶜ cos(2πkx)+c_kˢ sin(2πkx)]`.

### Literal multiple-shooting vs single-shooting — depends entirely on the SOLVER
The verdict splits by *how* MS is solved:

**(a) soft-penalty + Adam MS — does NOT win** (`recover_modes_ms.py`, `make_fig_basin.py`).
- T=40 t_g (single init): single 0.784 | soft M2(μ30) 0.640 | M4 0.725 | M8 0.670 |
  M4(μ100) 0.780 | hard M4 0.726 — MS marginally less bad, **both fail** (≥0.6).
- T=45 t_g (8 restarts): single best **0.049**/median 0.346 ; soft M2 best 0.340 —
  **single shooting wins** (soft-MS shrinks variance, raises the floor via bias).

**(b) constrained Gauss-Newton MS — WINS** (`recover_msgn.py`, `fig_msgn_vs_single.png`).
Hard continuity (linearized each step), simultaneous condensed-Newton step (LM),
*feasible* (forward-propagated) interiors, at the **genuine long-T mode-2** regime.
Same cold inits, same truth/noise/parametrization as single-shooting GN:

| T/t_g | single-GN recovered | GN-MS (M=4) recovered | single best | GN-MS best |
|------:|--------------------:|----------------------:|------------:|-----------:|
| 80  | **0/12** | **3/12** | 0.332 | **0.036** |
| 120 | 0/6 | 0/6 (uniformly closer) | 1.626 | **0.283** |

→ At T=80, constrained GN-MS recovers cold starts (`ic_err 0.04–0.06`) that single
shooting recovers **none** of; at T=120 it is uniformly nearer truth. A **real,
partial basin enlargement** that single shooting structurally cannot achieve.
(The earlier "feasible ≡ single shooting" holds only for the *undamped first-order*
step; with LM damping + nonlinearity the interior DOF genuinely help. `base`/
infeasible interiors fail — defects too large to close — so dynamically-consistent
forward-propagated interiors are the right init.) Reproduce:
```
# single (baseline)
KH_INIT=s KH_SEED=0 KH_INITSCALE=1e-2 KH_TREC=80 KH_KMIN=2 KH_KMAX=6 KH_ITERS=50 \
  KH_OUT=data/gnh_T80_i$s.npz python recover_gn.py
# constrained GN multiple shooting
KH_INIT=s KH_SEED=0 KH_INITSCALE=1e-2 KH_TREC=80 KH_M=4 KH_INTERIOR=feasible \
  KH_KMIN=2 KH_KMAX=6 KH_ITERS=40 KH_OUT=data/msgn_feasible_M4_T80_i$s.npz python recover_msgn.py
```

### Money table 1 — recovery: MS PRINCIPLE (horizon continuation) vs cold GN
*(single-shooting GN bootstrapped across horizons — the basin-enlargement mechanism of MS, not literal segmentation)*
`ic_err = ‖seed_rec − seed_truth‖/‖seed_truth‖`. Cold GN = single-shooting
Levenberg–Marquardt from 6 cold inits (`INITSCALE=1e-2`); continuation = warm-start
GN along increasing T from a recovered `T=50` solution. Conditioning at these T:
`κ(J)≤780`, `σ_min≫noise` (so the cold-GN failure is genuine **mode 2**, not mode 1/3).

| T/t_g | continuation `ic_err` | cold-GN `ic_err` (6 inits) | cold best |
|------:|----------------------:|----------------------------|----------:|
| 60  | 0.040 | 0.05 0.06 0.10 0.11 0.14 0.15 | 0.055 |
| 70  | 0.029 | 0.02 0.10 0.15 0.15 0.20 1.89 | 0.023 |
| 80  | **0.021** | 0.62 1.23 1.62 2.75 6.66 20.4 | 0.624 |
| 100 | **0.010** | (not run) | — |
| 120 | **0.004** | 1.63 2.59 3.63 4.78 5.25 5.74 | 1.626 |
| 160 | — | 1.01 1.96 2.18 2.44 6.97 67.2 | 1.009 |

→ **MS principle wins at T≥80**: continuation recovers (`ic_err 0.004` at T=120)
where every cold-GN restart fails (`best 1.6`). It even sharpens with horizon.

Reproduce (cold GN at horizon T, restart s):
```
CUDA_VISIBLE_DEVICES=0 KH_INIT=s KH_SEED=0 KH_INITSCALE=1e-2 KH_TREC=T \
  KH_KMIN=2 KH_KMAX=6 KH_ITERS=50 KH_OUT=data/gnh_T${T}_i${s}.npz python recover_gn.py
```
Reproduce (continuation chain):
```
prev=data/gnh_T50_i1.npz           # a recovered short-horizon solution
for T in 60 70 80 100 120; do
  KH_SEED=0 KH_TREC=$T KH_KMIN=2 KH_KMAX=6 KH_ITERS=40 KH_WARMNPZ=$prev \
    KH_OUT=data/cont_T$T.npz python recover_gn.py
  prev=data/cont_T$T.npz
done
```

### Money table 2 — mode 1: segmentation caps the back-prop gain
State-space tangent gain `σ_max(τ)` measured by per-kx forward jvp (`frontier.npz`,
`information_frontier.py`): `σ_max = 2.99 / 70.3 / 376 / 1306` at `τ = 20/60/100/160
t_g` → `σ_max(τ)~e^{λτ}`, `λ≈2.1/t_phys`. `M`-segment multiple shooting only
back-propagates one segment of length `T/M`, so its per-segment gain is `σ_max(T/M)`:

| at T=160 t_g | M=1 | M=2 | M=4 | M=8 |
|---|---:|---:|---:|---:|
| per-segment horizon T/M (t_g) | 160 | 80 | 40 | 20 |
| back-prop gain σ_max(T/M) | 1306 | ≈29 | ≈5.4 | 2.99 |

(M=1 and M=8 are measured endpoints of `frontier.npz`; M=2,4 from the `e^{λτ}` fit.)
→ **~440× smaller gradient for M=8** — the difference between a trainable and an
untrainable objective for first-order optimizers. Reproduce: `python information_frontier.py`
(reads `σ_max(τ)`; segmentation cap = read off `σ_max(T/M)`).

### Where MS does NOT win (for honesty)
Low-D observable unknown at moderate T (`T≤50`): single-shooting GN already recovers
8/8 (`fig_gn_vs_adam.png`), and soft-penalty/terminal-only MS adds bias (Stages 4e–4g).
The wins above are specifically the long-horizon, gradient-explosion / genuine-mode-2 regime.

### Money table (100-init statistics) — recovery FRACTION vs M (`fig_campaign_fraction.png`)
Per-process campaign (`run_msgn_campaign.sh`): 100 cold inits each (identical RNG
across methods), T=80 t_g, kx2-6, single-shooting GN vs constrained GN-MS at
M=4/8/16 (feasible interiors). Recovery = ic_err<0.1; Wilson 68% CI. Individual
inits are float-path-sensitive in this mode-2 regime, so the FRACTION is the
robust statistic.

| method | recovered | fraction | 68% CI | best | median |
|---|---:|---:|---|---:|---:|
| single shooting | 1/100 | **0.01** | [0.00,0.03] | 0.041 | 2.40 |
| GN-MS M=4 | 13/100 | 0.13 | [0.10,0.17] | 0.015 | 1.34 |
| GN-MS M=8 | 19/100 | **0.19** | [0.15,0.23] | 0.007 | 0.68 |
| GN-MS M=16 | 13/100 | 0.13 | [0.10,0.17] | 0.013 | 0.61 |

→ **Constrained GN multiple shooting beats single shooting decisively** (13-19%
vs 1%, ~13-19x). The advantage is segmentation-driven but **does not scale with M**:
M=4≈M=16≈0.13 with a **weak optimum at M=8 (0.19)** whose 68% CI only marginally
clears M=4/16 (~1σ) — suggestive, not decisive. Physically: M=8 → T/M=10 t_g per
segment (inside the GN-recoverable regime AND short enough to enlarge the basin);
M=16 (T/M=5) adds interior DOF without further gain. Medians also favor M=8/16
(0.7/0.6) over M=4 (1.3). Reproduce: `NRUN=100 bash run_msgn_campaign.sh` then
`python make_fig_campaign.py`. Note: per-process is used (not the batched
`gn_batch.py`/`msgn_batch.py`) for safety + single/MS comparability; the batched
fully-JAX solvers run but the MS one needed convergence tuning.
