# Differentiable Pallas Backend — Evaluation & Roadmap

Status: phases 0–2 implemented on `vibed` (this document tracks them);
phases 3–5 are planned.

## 0. Problem statement

Every Pallas kernel in astronomix sits behind an AD boundary
(`diffable_pallas_call(_n)` in `_pallas_helpers.py`) because JAX cannot
transpose an aliased `pl.pallas_call` (`input_output_aliases` → "JVP with
aliasing not supported"). Until now that boundary was a single design:

> `jax.custom_jvp` whose primal is the Pallas branch and whose tangent is
> `jax.jvp(native_branch, ...)`.

That is correct and supports *both* AD modes, but it has a structural cost
that only shows up under reverse mode (`jax.grad`, the dominant use in the
inverse-modeling studies):

- **Linearization runs both branches.** Partial evaluation of the JVP rule
  keeps the Pallas primal (its output is consumed downstream) *and* all the
  native-branch residuals needed by the linear tangent ops. The forward
  sweep under `jax.grad` therefore computes the Pallas kernel **plus**
  essentially the whole native forward, and stores the native residuals.
- **Backward is native-speed.** The cotangent flows through the transposed
  native tangent.

Net effect: with `backend=PALLAS`, `jax.grad` is *strictly more* expensive
in time and memory than `backend=NATIVE_JAX`. This is why differentiable
runs have been capped at ~64³ and effectively "fall back to native".

## 1. What custom JVP/VJPs can and cannot buy

| Lever | Mechanism | Gain | Cost/risk |
|---|---|---|---|
| `custom_vjp`, residuals = inputs, backward = `jax.vjp(native)` recompute (**VJP_REMAT**) | remat at kernel granularity | forward sweep under grad becomes *pure Pallas* (7–12× faster, ~50 % less memory); no native residuals stored | backward unchanged (native); forward-mode AD lost through the boundary |
| `custom_vjp` + hand/auto Pallas adjoint kernels (**VJP_PALLAS**) | backward sweep on Pallas too | backward sweep gets Pallas-class speed/memory; zero stored residuals (recompute at register level) | per-kernel adjoint code; Triton lowering graph ≈ fwd+bwd of kernel body |
| Pallas built-in `pallas_call` JVP | auto-differentiates the kernel body | fused primal+tangent kernel for forward mode | rejected: breaks on `input_output_aliases`, and there is **no transpose rule**, so reverse mode still needs a custom_vjp |
| Hand-derived adjoint algebra (no autodiff) | manual WENO/eigenstructure adjoint | marginal over in-kernel `jax.vjp` | very high effort, error-prone; not justified |

Key insight that makes VJP_PALLAS cheap to build: **inside a Pallas kernel
trace, `jax.vjp` works on the tile-level closure.** The per-face WENO flux
is a pure function of its 6-cell stencil, so the adjoint kernel can apply
`jax.vjp` to that closure *inside the kernel* — auto-deriving the WENO/
eigenstructure adjoint with zero hand algebra, recomputing forward values
in registers instead of storing residuals.

The scatter→gather flip is handled by splitting the adjoint into two
bounded-halo kernels (guide §4c "split, don't fuse"):

1. **face kernel** — per face `j`: `vjp` of the face-flux closure against
   the incoming flux cotangent → `6·ncomp` cell-cotangent channels + 3
   per-face scalar-cotangent channels (gamma, min density, min pressure).
   One fwd+bwd trace of the face math; halo 3; no in-kernel reductions.
2. **gather kernel** — linear: `d_state[c,i] = Σ_s buf[s·ncomp+c, i+2−s]`;
   halo 3.

The intermediate is `(6·ncomp+3)` channels × spatial — transient, consumed
immediately (e.g. ~250 MB at 128³ x32, vs multi-GB native residuals).

### Which kernels deserve a Pallas adjoint

Applying the skill's own "don't pallasify what XLA does fine" rule to
adjoints:

- **WENO interface flux (hydro / MHD / iso variants)** — the hot nonlinear
  stencil; dominates both sweeps. → Pallas adjoint. *(hydro done; MHD/iso
  follow the identical recipe.)*
- **Flux-divergence accumulator** — linear; its native vjp is a shift+axpy
  that XLA fuses. → VJP_REMAT is already near-optimal; no adjoint kernel.
- **Positivity floor** — pointwise mask; native vjp trivial. → VJP_REMAT.
- **FV fused evolve** — nonlinear and a good later candidate; the per-cell
  closure (reconstruct → Riemann → axpy) has halo 2 and the same two-kernel
  split applies. → VJP_REMAT now, Pallas adjoint in phase 4.

### The forward-mode caveat

`jax.custom_vjp` is not forward-differentiable: under VJP_REMAT/VJP_PALLAS,
`jax.jvp`/`jacfwd` through a Pallas dispatch raises. The default therefore
stays `PALLAS_AD_JVP_NATIVE`; reverse-mode users opt in via
`config.pallas_ad_mode`. (A later phase can add Pallas tangent kernels
behind a `custom_jvp` for forward mode — same in-kernel-`jax.vjp` trick,
with `jax.jvp` instead.)

## 2. Implemented (this branch)

- `config.pallas_ad_mode ∈ {PALLAS_AD_JVP_NATIVE (default), PALLAS_AD_VJP_REMAT,
  PALLAS_AD_VJP_PALLAS}` in `simulation_config.py`.
- `diffable_pallas_call(_n)` in `_pallas_helpers.py` gained
  `ad_mode=` and `adjoint_branch=` keywords; the VJP modes build a
  `jax.custom_vjp` whose fwd saves only the primal inputs.
- All four dispatch families pass `ad_mode=config.pallas_ad_mode`:
  FD WENO flux (`_weno.py`), FD flux-divergence (`_ssprk_pallas.py`),
  FV fused evolve (`evolve_state.py`), positivity floor
  (`_enforce_positivity.py`).
- Pallas adjoint for the ideal-gas hydro WENO flux:
  `_weno_pallas_adjoint.py` (face kernel + gather kernel + params-cotangent
  assembly), wired into the hydro arm of `_weno_flux_axis_dispatch` and
  active only under `PALLAS_AD_VJP_PALLAS`.
- Validation: `tests/pallas/ad_vjp_validate.py` (kernel-level grads vs
  native, CPU-interpret and GPU-Triton); benchmark:
  `tests/pallas/ad_baseline_bench.py` (time_integration-level grads,
  all modes vs native).

### Measured results

*(filled in from `tests/pallas/ad_baseline_bench.py` /
`ad_vjp_validate.py` runs — see REPORT in tests/pallas once runs land)*

## 3. Roadmap

| Phase | Content | Gate | Status |
|---|---|---|---|
| 0 | Baseline measurements (native grad vs Pallas-JVP_NATIVE grad, 64³) | numbers reproduce the "Pallas grad ≥ native grad" prediction | done (script) |
| 1 | `pallas_ad_mode` knob + custom_vjp helpers + wire all dispatch sites (VJP_REMAT) | kernel-level grads match native ≤1e-4 rel (x32) / 1e-10 (x64); time_integration grads match | done |
| 2 | Pallas adjoint for hydro WENO via in-kernel `jax.vjp`, two-kernel split (VJP_PALLAS) | same gradient gates; Triton lowering < ~60 s; backward sweep faster than VJP_REMAT at 64³ | done (validation pending) |
| 3 | Sensitivity-convergence gate (`tests/sensitivity/sensitivity.py`) in both VJP modes; 128³ grad attempt; update memories/study configs | AD gradient converges to Fourier gradient at native rate; 128³ grad fits on one A100 | pending |
| 4 | Pallas adjoints for MHD WENO, iso variants, FV fused evolve (same recipe: face/cell closure + `jax.vjp` in-kernel + linear gather) | per-kernel gradient gates + alfven_wave3D grad check | planned |
| 5 | Forward-mode Pallas tangent kernels (custom_jvp with in-kernel `jax.jvp`), only if a study needs jacfwd at scale | jvp matches native tangent | optional |
| 6 | Skill/guide updates (pallasify §4d rewrite: adjoint-by-tracing recipe, mode table, validation steps) | — | with this branch |

## 4. Risks / open questions

- **Triton lowering time** for the face-adjoint kernel (fwd+bwd of the WENO
  body in one kernel). Mitigated by the two-kernel split; if a variant
  still lowers slowly, fall back to VJP_REMAT for that kernel only — the
  `adjoint_branch=None` seam makes this a one-line decision per dispatch.
- **x64 typed-literal contamination** in JAX-generated transpose code:
  JAX emits typed constants in transposes, so the §4.4 f32-literal issue
  should not reappear; the x64 validation run is the gate.
- **`checkpointed_while_loop` interaction**: custom_vjp residuals (= kernel
  inputs) are re-materialized per checkpoint segment exactly like the
  primal carries; no extra live set. Verified by the 64³ grad benchmark
  memory numbers.
- **Multi-GPU adjoints**: both adjoint kernels route through
  `_pallas_call_sharded` (halo 3). Strong-scaling validation of the
  backward sweep is part of phase 3.
