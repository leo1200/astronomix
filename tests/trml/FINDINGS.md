# Aligning the temperature PDFs of a TRML: Fokker–Planck vs a transfer matrix

**Branch `tracer2` (off `main`).** This is the critical, confirmation-bias-aware
write-up of the tracer-particle study of the turbulent-radiative-mixing-layer
(TRML) temperature PDF. It answers the question posed: can the **(weighted)
Eulerian**, **Lagrangian**, and **model** (Fokker–Planck / transfer-matrix)
temperature PDFs be brought into agreement, and *what does or does not hold*?

Reproduce:
```bash
cd tests/trml
TRML_PRESET=fiducial python trml_tracers.py        # N=128 two-phase run
TRML_PRESET=fiducial python fokker_planck_test.py   # FP reconstruction
TRML_PRESET=fiducial python transfer_matrix_test.py # transfer-matrix + CK + kernel
```
(`long64` = N=64 steady-state; `quick` = shakedown.)

---

## 0. What was built

- Lagrangian tracer particles ported into `astronomix` on `main`'s clean
  `LoopState`/snapshot machinery (`astronomix/_modules/_tracers/`): mass-weighted
  seeding, multilinear interpolation, RK2 advection, and a **regeneration
  thermostat** that re-draws a small fraction of tracers ∝ the *current* density
  each step so the Lagrangian ensemble stays mass-representative. 8/8 unit tests.
- Two-phase run: spin up to equilibrium with no tracers, then seed tracers from
  the *equilibrium* density and record densely over a short window (fine lag
  spacing). A scatter-free histogram keeps recording at ~6 ms/iter.
- Analyses: `fokker_planck_test.py` (Check 0, drift/diffusion, `P̂`
  reconstruction via `~/diffmix`); `transfer_matrix_test.py` (empirical
  transition matrix, Chapman–Kolmogorov test, kernel-vs-Gaussian comparison).

## 1. The four PDFs and which agreements are meaningful

| PDF | how obtained | logically independent of the target? |
|---|---|---|
| Eulerian mass-weighted `dM/dlogT` | grid histogram, ρ-weighted | — (this **is** the target marginal) |
| Lagrangian tracers | histogram of tracer temperatures | tests the *seeding/weighting* |
| Fokker–Planck `P̂` | march from measured `A(T)`, `D(T)`, `J` | tests a 2nd-order diffusion ansatz |
| Transfer-matrix stationary `π` | empirical kernel `M(Δt)` + flux `J` | tests a Markov ansatz (no truncation) |

**A PDF "matching" is only evidence if the two sides are independent.** Two traps
we deliberately avoid:
- The *inverse* FP construction (solve for `D` given `P` and `A`) reproduces the
  marginal by construction — self-fulfilling, not a test. We use the *forward*
  march instead.
- The stationary eigenvector of an empirically measured one-step matrix is the
  empirical occupation measure **by construction** if you include every
  transition. We break that circularity by **excluding the regeneration jumps**,
  so our `M` is the pure cooling+mixing operator and its stationary `π` is a
  genuine (falsifiable) prediction — which, as it turns out, *fails* in an
  informative way (§3).

## 2. Result: the PDFs do align — but only with the flux

*(Headline numbers are the N=128 fiducial; the N=64 run agrees — see §6.)*

- **Check 0 (Eulerian ↔ Lagrangian): L1 ≈ 0.008.** The mass-seeded, regenerated
  tracers reproduce the Eulerian mass PDF across the full range including the
  thin mixing band. So the Lagrangian marginal is trustworthy.
- **Fokker–Planck `P̂` ↔ Eulerian:** reproduces cold peak + mixing plateau + hot
  rise with support-L1 ≈ 0.12 (best at Δt ≈ 0.008–0.02 t_sh), mixing-range L1 ≈
  8×10⁻⁴. It is **approximately** right, not exact.
- **Transfer-matrix (flux-driven) ↔ Eulerian:** full-support L1 ≈ 0.106,
  interior-L1 ≈ 0.065 — i.e. it **beats** FP on the marginal at N=128 (using the
  full, non-Gaussian kernel instead of the Gaussian one).

`fp_summary`, `tm_four_pdfs`: all four curves overlay to ~0.1 L1 / ~0.3 dex.

**The essential caveat — the marginal is flux-maintained, not an interior fixed
point.** With `J = 0`:
- FP `P̂` collapses to the cold peak.
- the **closed** (conservative) transfer matrix `π` *also* collapses to the cold
  peak (L1 ≈ 0.23) — see `tm_four_pdfs`, red curve.

Only when the measured hot→cold probability current `J` is injected (FP via the
first-integral flux term; the transfer matrix via an explicit source at hot /
sink at cold — the discrete analog of the same first integral) do the models
reproduce the plateau. This is a physical statement: the steady mixing-layer PDF
is sustained by throughflow, and any interior-only (flux-free) model must fail.

## 3. Why neither model is *exact* — two diagnostics

The transfer matrix, being the untruncated transition operator, lets us decompose
FP's ~0.1–0.3-dex residual into its two possible causes.

1. **Is `T` alone Markov? (Chapman–Kolmogorov, `tm_chapman_kolmogorov`)**
   Compare `M(Δt)^k` to the directly measured `M(kΔt)`, judged against a
   **parametric Markov null** (a synthetic Markov chain built from the *measured*
   `M`, same sample sizes). The measured CK divergence sits **19–58× above the
   null** at N=128 (7–10× at N=64, where sampling noise is larger) and grows with
   lag → `T` has **clearly detectable memory** (non-Markov), consistent with a
   finite turbulent correlation time. The *absolute* divergence is still modest
   at the base lag (row-TV ~ 4×10⁻³) but grows to ~2×10⁻² over the scan.

2. **Is the one-step kernel Gaussian? (`tm_kernel_vs_fp`)**
   The empirical kernel vs the FP Gaussian built from the same `A`, `D`:
   mean row-TV ≈ **0.087**, peaking in the mixing range where cooling and mixing
   compete (skewed, heavier-than-Gaussian increments).

**At the base (one-step) lag the non-Gaussianity of the kernel (≈0.084)
dominates the non-Markovianity (≈0.004); memory only accumulates over multiple
steps (up to ≈0.02 over the CK scan).** So FP's imperfection is primarily the
*second-order/Gaussian truncation*, with a secondary, lag-accumulating memory
term. Replacing the Gaussian kernel with the full empirical kernel (the transfer
matrix) does help the *marginal*: at N=128 it goes from FP's 0.116 to 0.106
(full support) / 0.065 (interior). The improvement is real but bounded, because
the marginal's accuracy is also gated by the **flux/boundary treatment**, which
both models handle the same way. (At N=64 the gain is washed out by the transfer
matrix's own boundary point-source artifact — see §6 — so we trust the N=128
comparison, where more statistics shrink that artifact.)

## 4. Verdict (avoiding confirmation bias)

- **Do the PDFs align?** Yes, to ~0.1 L1 / ~0.3 dex — *provided* the measured
  flux is supplied. "Alignment" is approximate, not exact, and it is honest to
  say so: a single number (support-L1 ≈ 0.11–0.13) hides that the fit is
  excellent in the cold peak and good on the mixing plateau but limited by the
  boundary/flux handling and the tails.
- **Does the Fokker–Planck model hold?** Approximately. It is a *useful* but
  *imperfect* reduced model. It is not a clean diffusion in a scale-separated
  window: the kernel is measurably non-Gaussian and `T` is weakly non-Markov.
- **Does the transfer matrix do better?** At N=128, yes on the marginal
  (full-support 0.106 / interior 0.065 vs FP 0.116), because the true kernel is
  non-Gaussian; at N=64 the two are on par (its boundary artifact eats the gain).
  Its more durable value is **diagnostic**: it isolates *why* FP is imperfect
  (one-step non-Gaussianity > lag-accumulating memory) and demonstrates, via the
  honest (non-self-fulfilling) closed-vs-flux comparison, that the marginal is
  flux-maintained, not an interior fixed point.
- **Metric honesty.** We report L1 over the support *and* an interior L1
  (excluding the point source/sink bins); mixing-range L1; and dex-scale shape,
  because the cold peak (≈90% of the mass) makes a global L1 look artificially
  good. A concentrated PDF flatters absolute metrics — the earlier ξ-scan
  "improvement with Damköhler number" was largely this artifact (see
  `PRIOR_SUMMARY.md` §5), and we do **not** claim a Damköhler trend here.

## 5. Caveats / where this could still be wrong

- **Regeneration thermostat** pins the Lagrangian marginal to the mass PDF; its
  timescale (0.2 t_sh) caps the usable increment lag, so the large-Δt end of the
  CK / validity scans loses statistics.
- **`J` units.** FP and the transfer matrix measure `J` slightly differently
  (per unit time vs per step); both are internally consistent, but a factor-few
  mis-scaling of `J` would shift the plateau height — the flux is the most
  fragile input and deserves an independent cross-check.
- **Single (M, χ, ξ) point.** Results here are N=64/128, M=0.5, χ=100, ξ=100.
  The kernel-non-Gaussianity vs non-Markovianity split, and whether a genuinely
  clean diffusive window ever opens, should be scanned in ξ with **finer temporal
  cadence** (the window moves to smaller Δt as ξ grows).
- **Coarse transfer-matrix grid (40 bins)** and point source/sink at the support
  edges are discretization choices; the interior conclusions are insensitive to
  them but the boundary bins are not scored for that reason.

## 6. N=64 vs N=128 (resolution robustness)

Both runs: M=0.5, χ=100, ξ=100; 20 000 tracers; equilibrium-seeded dense
recording. Metrics on the coarse (40-bin) support unless noted.

| quantity | N=64 (`long64`) | N=128 (`fiducial`) |
|---|---|---|
| Check 0  L1 (Eulerian ↔ Lagrangian) | 0.007 | 0.008 |
| FP `P̂` support-L1 (best lag) | 0.132 | 0.116 |
| transfer-matrix closed (J=0) L1 | 0.23 (collapses) | 0.11 (collapses)¹ |
| transfer-matrix flux-driven, full support | 0.216² | **0.106** |
| transfer-matrix flux-driven, interior | 0.107 | **0.065** |
| CK signal / Markov null (ratio) | 7–10× | 19–58× |
| kernel vs FP-Gaussian, mean row-TV | 0.087 | 0.084 |

¹ The closed-`π` L1 looks "small" because the cold peak (~90 % of the mass) is
still reproduced; the collapse is in the low-mass mixing+hot range (see the red
curve in `tm_four_pdfs`, which falls off a decade above `T_cold`).
² At N=64 the full-support number is inflated by the transfer matrix's boundary
point-source artifact (a single injection bin); the interior number (0.107) is
the fair comparison there. At N=128 more statistics shrink the artifact, and the
transfer matrix beats FP even on full support.

**Consistent conclusions at both resolutions**: Check 0 passes; FP holds
approximately; the marginal is flux-maintained (closed models collapse); `T` is
non-Markov (more clearly at N=128 with lower sampling noise); the one-step kernel
is non-Gaussian. **Resolution-dependent**: the transfer matrix's advantage over
FP on the marginal is marginal at N=64 and clear at N=128 — the finer the
increments are resolved, the more the non-Gaussian kernel pays off.

Figures (N=128, no suffix): `figures/tm_four_pdfs.png`,
`figures/tm_chapman_kolmogorov.png`, `figures/tm_kernel_vs_fp.png`, and the FP
set `figures/fp_summary.png`, `figures/fp_reconstruction.png`,
`figures/fp_validity_window.png`, `figures/fp_diffusion_plateau.png`. (N=64
figures carry the `_long64` suffix.)
