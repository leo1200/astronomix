# Agent Plan — Single vs Multiple Shooting on 2D Rayleigh–Benard (astronomix)

## Objective
Demonstrate, in a differentiable 2D Rayleigh-Benard simulation, that a terminal-time
large-scale inverse problem ("turn back the large-scale clock") is wrecked by single
shooting (SS) but solvable by multiple shooting / segmentation (MS), because segmentation
bounds the backward growth of the adjoint.

Physical dissipation: **momentum viscosity ON + thermal conduction (implemented in Phase 0a)**.
Driving: **conductive isothermal plates** (faithful RBC). Source/flux driving kept as fallback.

## Deliverables
- **D1** — forward simulation produces the expected large-scale structure (a coherent LSC).
- **D2** — mechanism plot: segmentation limits backward growth of the adjoint (norm + high-k), resetting at segments.
- **D3** — "money plot": large-scale recovery vs window length, SS vs MS.
- **D0 (prerequisite)** — Lyapunov time tau_L = 1/lambda1 (sets all window/segment lengths).

## Configuration (locked)
- Dimensionality: **2D** first.
- Scheme: **FD, default settings**, non-periodic BCs.
- Dissipation: **momentum viscosity ON + thermal conduction ON** (Phase 0a). Aim for physical nu, kappa to dominate numerical dissipation -> meaningful Ra, Pr; otherwise effectively ILES (still fine).
- Driving: **conductive isothermal plates** (hot bottom / cold top). Fallback: fixed-flux plates or volumetric heating/cooling source terms.
- Geometry: aspect ratio **Gamma = 2**, grid **512 x 512** (prototype 256 x 256), **Ma ~ 0.2**.
- Regime: **Ra ~ 1e7, Pr ~ 1** (effective if dissipation under-resolved).
- Inverse problem: **twin experiment (OSSE)**, **terminal-only** loss on the **large-scale projection**.
- `P_large`: **spectral low-pass at k_c** (also track LSC scalar).
- Control: **large-scale IC only**, small scales fixed to background.
- MS flavor: **hard-constraint primary**, weak-constraint secondary.
- Compute: single high-memory GPU assumed; prototype 256x256 then 512x512; window sweep bounded by backprop memory (checkpoint/remat).

---

## Phase 0a — Implement thermal conduction for the FD scheme (BLOCKING)
1. Add the conductive term to the energy equation: d(rho E)/dt += div(k grad T), with T from the EOS, **constant conductivity**, **explicit** time integration.
2. Timestep: enforce the parabolic limit dt <= C * dx^2 / kappa_diff alongside the acoustic limit; keep kappa modest so explicit stepping stays cheap (avoid an implicit solve, which also complicates AD).
3. Conductive BCs (the part that matters): isothermal **Dirichlet** T at top/bottom plates via ghost-T (this is what injects heat); **adiabatic** zero-flux (Neumann) at sidewalls; couple to the non-periodic BC machinery.
4. Verify: (a) 1D sinusoidal temperature perturbation decays at rate kappa_diff * k^2; (b) prescribed wall heat flux matches -k dT/dn at an isothermal wall; (c) finite-difference gradient check of a scalar loss w.r.t. the conduction-coupled state (AD must be clean — constant-kappa explicit Laplacian should be trivially differentiable).

## Phase 0b — Environment & full differentiability gate (BLOCKING)
1. Install astronomix + GPU JAX; confirm multi-GPU if available.
2. Build the RBC config (FD default, non-periodic BCs): confined box, no-slip walls; isothermal hot/cold plates; adiabatic no-slip sidewalls; constant-g via source-term machinery; viscosity ON; conduction ON (Phase 0a).
3. Smoke test: short forward run; check stability, conservation, plate heat flux, convection onset.
4. **Gradient check (GATE):** on a window << tau_L, finite-difference-verify the reverse-mode gradient of a scalar loss w.r.t. the IC through the full path (FD + non-periodic BCs + viscosity + conduction). Inspect gradient smoothness. Do NOT proceed until this passes.

## Phase 1 — Forward simulation & regime (D1)
1. Integrate to statistical stationarity at Ra ~ 1e7 (optionally 1e8).
2. Diagnostics:
   - temperature & vorticity snapshots showing the coherent LSC roll;
   - LSC amplitude + orientation time series (mid-height horizontal velocity, sinusoidal fit, or first POD mode);
   - capture at least one reversal if present;
   - Nu(t) and time-mean Nu vs a known 2D RBC Nu-Ra correlation (meaningful IF physical dissipation is resolved);
   - energy spectrum.
3. **Success:** clean LSC present + dynamics chaotic (Phase 2 confirms lambda1 > 0). If laminar/periodic, raise Ra.

## Phase 2 — Lyapunov time (D0, prerequisite)
1. Benettin-style estimate of lambda1: evolve a normalized tangent perturbation about the stationary attractor, periodically rescale, average the log-growth.
2. Report tau_L = 1/lambda1 in eddy-turnover units. All downstream windows/segments are in tau_L.

## Phase 3 — Inverse-problem setup (twin experiment)
1. Pick truth IC x0*; integrate to T; store **target = P_large(x(T))** (+ obs noise, cov R).
2. Define `P_large` (default spectral low-pass at k_c; also record LSC scalar).
3. Define prior/background: Gaussian toward climatological large-scale mean + covariance B; restrict control to the large-scale subspace.
4. Loss: J = || P_large(forward(control)|_T) - target ||^2_{R^-1} + (control - background)^T B^-1 (control - background).

## Phase 4 — Single shooting
1. Control = large-scale IC. Reverse-mode gradient through the full window; use jax.checkpoint/remat for memory.
2. Optimizer: Gauss-Newton/LM (or L-BFGS) — same family as MS.
3. Record per backward time: adjoint **norm** ||lambda(t)||, adjoint **spectrum / high-k fraction**, optimizer convergence, final recovery.

## Phase 5 — Multiple shooting (segmentation)
1. Partition [0,T] into segments of **dt <~ tau_L** (default dt = 0.5 tau_L). Controls = segment-boundary large-scale states.
2. **Hard-constraint** (continuity as equality constraints, Gauss-Newton/KKT, banded system) — primary.
   **Weak-constraint** (continuity penalty + model-error cov Q) — secondary.
3. Adjoints are independent per segment (span only dt) — this is the "reset."
4. Record the same diagnostics, per segment.

## Phase 6 — Mechanism plot (D2)
- **Primary:** ||adjoint(t)|| vs t — SS grows ~ e^{lambda1 (T-t)} monotonically toward t=0; MS sawtooths, each tooth bounded by e^{lambda1 dt}.
- **Corroborating:** adjoint energy spectrum / high-k fraction vs backward time — SS shifts toward small scales (now damped above the *physical* conduction/viscous scale, so the residual high-k growth is physical, not limiter noise); MS resets each segment.
- **Fair overlay:** compute SS and MS adjoints linearized about the *same* trajectory, so the only difference is integration horizon. Annotate tau_L and dt.

## Phase 7 — Money plot (D3)
1. Sweep window length T in [~0.5, ~20] tau_L (budget permitting). For each T, run SS and MS; compute **large-scale IC recovery error vs truth**.
2. **Money plot:** recovery error vs T (tau_L units), two curves.
   - Expect SS error to rise at the **conditioning limit (~1 tau_L)**.
   - Expect MS error to stay low until the **retrodiction/identifiability limit (>> tau_L)**.
   - The gap between the two thresholds is the headline result.
3. **Companion visual:** truth / SS-recovery / MS-recovery field triptych at one T between the thresholds.
4. **Metric validity:** distance-to-truth is meaningful only below the retrodiction horizon; beyond it, reinterpret via MAP/posterior spread (ensemble of MAPs from random inits), not distance-to-truth.

## Phase 8 — Controls & robustness
- SS and MS must share prior, optimizer family, target, and noise — only segmentation differs.
- Sensitivity to dt (segment length), obs noise R, low-pass cutoff k_c.
- Optional: posterior spread / retrodiction horizon via an ensemble of MAPs.

## Risks
- **Conduction parabolic CFL:** diffusion is stiff (dt ~ dx^2/kappa_diff). Keep kappa modest + cap dt; stay explicit. If you later want high-Ra with dominant physical kappa, you may need IMEX / super-time-stepping (RKL) — defer.
- **Conductive BC bugs:** the wall heat injection lives in the ghost-T treatment; verify wall flux (Phase 0a) before trusting D1.
- Controlled Ra/Pr only if physical nu, kappa dominate numerical dissipation (resolution cost); otherwise it's effectively ILES with conductive driving — still fine.
- **Discrete-adjoint noise:** reduced now that physical conduction+viscosity damp high-k in the backward adjoint, but still check gradient smoothness (Phase 0b).
- Low-Ma acoustic stiffness (small dt) — Ma ~ 0.2 compromise.
- MS fixes *conditioning*, not *identifiability* — the retrodiction horizon is physics, and is the correct place for MS to fail.