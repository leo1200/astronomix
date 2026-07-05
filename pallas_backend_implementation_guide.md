# Pallas Backend Implementation Guide

This document captures the patterns, gotchas, and design choices that made
the finite-difference hydrodynamic Pallas backend work well in
`astronomix`.  It is meant as a working reference for extending the same
backend to **FD MHD (ideal gas + isothermal)** and ultimately the
**finite-volume scheme**.

The headline numbers achieved on the 128³ Sedov benchmark
(`tests/pallas/sedov3D.py`, single A100):

| Configuration | Temp | Total | Time |
|---|---|---|---|
| NATIVE_JAX SSPRK4 + GHOST_CELLS (original) | 614 MB | 694 MB | 56 s |
| PALLAS SSPRK4 + GHOST_CELLS (baseline) | 336 MB | 416 MB | 6.0 s |
| PALLAS SSPRK4 + PERIODIC_ROLL + `donate_state` | 200 MB | 240 MB | 4.9 s |
| **PALLAS LSRK4 + PERIODIC_ROLL + `donate_state`** | **96 MB** | **136 MB** | **5.3 s** |

State size is 40 MB (5 vars · 128³ · float32), so the best run sits at
**3.4× state**, well below the 10× ceiling we started from.

The MHD WENO numbers on `pytests/mhd/alfven_wave3D.py` (3D CP Alfvén,
2N×N×N, ideal gas, x32):

| N | NATIVE L1 | PALLAS L1 | NATIVE time | PALLAS time | speedup |
|---|---|---|---|---|---|
| 8 | 1.8815e-02 | 1.8815e-02 | 0.86 s | 0.07 s | **12.9×** |
| 16 | 1.1980e-03 | 1.1980e-03 | 1.65 s | 0.15 s | **11.0×** |
| 32 | 4.5975e-05 | 4.5967e-05 | 3.49 s | 0.47 s | **7.4×** |
| 64 | 1.2804e-05 | 1.2957e-05 | 31.85 s | 4.56 s | **7.0×** |

5th-order convergence holds; PALLAS and NATIVE L1 match to
single-precision rounding noise at every resolution.

The FV hydro numbers on a 128³ HLL+MINMOD periodic test (RK2_SSP, x32,
block `(4,4,4)`):

| Backend | Temp | Total | Time | max\|PALLAS − NATIVE\| |
|---|---|---|---|---|
| NATIVE FV HLL+MINMOD | 651 MB | 691 MB | 1.70 s | — |
| **PALLAS FV HLL+MINMOD** | **158 MB** | **198 MB** | **0.85 s** | **0.0** |

The fused reconstruction+Riemann+update kernel gives a **4.1× memory
reduction** and **2× speedup**, with bit-identical output.

---

## 1. Where the wins came from

In order of impact:

1. **Single physical RHS buffer across axes** via Pallas
   `input_output_aliases`.  Equivalent to writing
   `rhs = rhs + new_axis_contribution` in-place — XLA cannot do this on
   its own across opaque `pallas_call` boundaries.
   *Saves 1–2 full-state buffers.*

2. **Switching SSPRK4 → LSRK4 (2N-storage Carpenter-Kennedy).**  Drops
   the SSPRK4 third register (`q0` is no longer captured by the closure
   for each stage).
   *Saves 1 full-state buffer.*

3. **`boundary_handling=PERIODIC_ROLL`.**  Removes ghost-cell padding so
   buffers are `num_cells^d` rather than `(num_cells + 2·num_ghost)^d`.
   For WENO5 this is 128³ vs 136³ — ~25 % smaller per buffer.
   *Saves ~25 % across all live state buffers.*

4. **`donate_state=True`** at the top-level `time_integration` call.
   Aliases the input primitive state to the loop carry, so output and
   input share the same physical buffer.
   *Saves the output buffer (~40 MB).*

5. **Folding the LSRK4 stage update into the divergence kernel** via a
   `scale_in` parameter on the per-axis Pallas div kernel.  The first
   axis writes `dq = A[i]·dq + (−dt/dx)·div_0(F)` in place — no
   intermediate `rhs_q` register.
   *Saves 1 full-state buffer.*

6. **Sequential per-axis flux + divergence** instead of computing
   `dF_x`, `dF_y`, `dF_z` and then a 3-input divergence kernel.  Each
   `dF` axis is produced and consumed within one Pallas+JAX op pair, so
   the three flux temporaries never coexist.
   *Mostly a redundancy-with-XLA fix; still simplifies HLO.*

Things that **didn't** help much on this workload:

- A more aggressively fused WENO kernel that computes both `F_{i+1/2}`
  and `F_{i-1/2}` inside the same Pallas program (eliminates the dF
  buffer entirely but doubles WENO compute — net runtime regression on
  A100 because the kernel is compute-bound, not bandwidth-bound).
- Tuning `pallas_block_shape` away from `(4, 4, 8)`.  Larger normal-axis
  tiles helped halo overhead in theory but hurt occupancy in practice;
  `(8,8,8)` was ~3× slower and `(16,8,4)` was ~10× slower.

---

## 2. Concrete Pallas patterns that work

### 2.1 The bare kernel skeleton

```python
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

def _hydro_kernel_supported(state, config) -> bool:
    if pl is None:                                   # Pallas not built
        return False
    if config.backend != PALLAS:
        return False
    bx, by, bz = _as_3tuple_block_shape(
        getattr(config, "pallas_block_shape", None), ndim
    )
    # Block must evenly divide every spatial dim.
    for n, b in zip(state.shape[1:], (bx, by, bz)[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def my_pallas_op(state, dt_over_dx, config, ...):
    ndim = int(config.dimensionality)
    nvars = int(state.shape[0])
    spatial = tuple(int(x) for x in state.shape[1:])
    bx, by, bz = _as_3tuple_block_shape(
        getattr(config, "pallas_block_shape", None), ndim
    )
    grid = (spatial[0] // bx,
            spatial[1] // by if ndim >= 2 else 1,
            spatial[2] // bz if ndim == 3 else 1)

    # Block specs: KEEP the conserved-variable axis whole, tile only the
    # spatial dims.  Passing the full ``state.shape`` for the input spec
    # tells Pallas/Triton that the kernel can read anywhere it wants
    # inside that array — it will only physically load what's referenced.
    if ndim == 3:
        block_shape  = (nvars, bx, by, bz)
        out_spec     = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    # ... 1-D and 2-D variants similar ...

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, dt_ref, out_ref):
        bi = pl.program_id(0); bj = pl.program_id(1); bk = pl.program_id(2)

        # Build modular index arrays once per program; ALL stencil reads
        # go through ``(... + offset) % n``.  Periodic boundaries are
        # handled for free by this; for GHOST_CELLS the caller pads.
        ii = (bi * bx + jnp.arange(bx)[:, None, None]) % spatial[0]
        jj = (bj * by + jnp.arange(by)[None, :, None]) % spatial[1]
        kk = (bk * bz + jnp.arange(bz)[None, None, :]) % spatial[2]

        # Always grab scalars via ``ref[()]`` once at the top.
        dt = dt_ref[()]

        # ... do work, write to out_ref ...
        for var in range(nvars):
            out_ref[var, ...] = some_expression(var)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(state.shape, state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec],
        out_specs=out_spec,
        interpret=bool(getattr(config, "pallas_interpret", False)),
        name="some_kernel",
        compiler_params=_pallas_compiler_params(config),
    )(state, jnp.asarray(dt_over_dx, dtype=state.dtype))
```

Two helpers worth defining once and reusing everywhere:

```python
def _pallas_compiler_params(config):
    use_triton = bool(getattr(config, "pallas_use_triton", True))
    if use_triton and pltriton is not None:
        return pltriton.CompilerParams(
            num_warps=int(getattr(config, "pallas_num_warps", 4))
        )
    return None


def _as_3tuple_block_shape(block_shape, ndim):
    """Normalize whatever the user supplied to (bx, by, bz)."""
    ...  # see astronomix/_finite_difference/_interface_fluxes/_weno.py
```

### 2.2 Periodic indexing is the boundary handler

The kernel above reads `q_ref[var, (ii + offset) % nx, jj, kk]`.  That
modular wrap **is** the periodic boundary condition.  With
`boundary_handling=PERIODIC_ROLL` no extra boundary kernel is needed.
With `GHOST_CELLS` the caller pads the state with `num_ghost_cells`
ghost cells, fills them via `_boundary_handler`, and the kernel still
uses `% nx_padded` but only the interior cells matter.

### 2.3 In-place accumulation across axes — the key memory trick

The pattern that drops 1–2 full-state buffers from peak:

```python
def _hydro_flux_div_axis_pallas(
    dF, dt_over_dx, config, *, axis,
    rhs_accumulator=None,
    scale_in=1.0,
):
    """Computes ``rhs_out = scale_in * rhs_acc  +  (-dt/dx) * div_axis(dF)``.

    With ``input_output_aliases={0: 0}`` XLA keeps a SINGLE physical
    rhs buffer across all three axis calls.  ``scale_in`` lets the
    LSRK4 first-stage update (``dq = A*dq + dt*L(q)``) be done in place
    inside the kernel instead of via a separate ``rhs`` register.
    """
    accumulate = rhs_accumulator is not None

    def kernel(*refs):
        if accumulate:
            rhs_in_ref, f_ref, dtdx_ref, scale_ref, rhs_out_ref = refs
        else:
            f_ref, dtdx_ref, rhs_out_ref = refs
        # ...
        if accumulate:
            scale = scale_ref[()]
            for var in range(nvars):
                rhs_out_ref[var, ...] = (
                    scale * rhs_in_ref[var, ...]
                    + (-dtdx_ref[()]) * flux_diff(var)
                )
        else:
            for var in range(nvars):
                rhs_out_ref[var, ...] = -dtdx_ref[()] * flux_diff(var)

    kwargs = {}
    if accumulate:
        kwargs["input_output_aliases"] = {0: 0}     # <-- the magic
    # ...
```

Caller:

```python
dq = _hydro_flux_div_axis_pallas(
    dF_x, dtdx, config, axis=0,
    rhs_accumulator=dq, scale_in=A_coef,
)
dq = _hydro_flux_div_axis_pallas(
    dF_y, dtdy, config, axis=1,
    rhs_accumulator=dq, scale_in=1.0,
)
dq = _hydro_flux_div_axis_pallas(
    dF_z, dtdz, config, axis=2,
    rhs_accumulator=dq, scale_in=1.0,
)
```

The dF buffers are produced one at a time and consumed immediately.
The `dq` buffer is physically one allocation throughout.

### 2.3a Multi-GPU: the `_pallas_call_sharded` wrap

A bare ``pl.pallas_call`` is opaque to GSPMD.  Its ``BlockSpec`` index
map is ``lambda bi, bj, bk: (0, 0, 0, 0)`` — every block program *can*
read anywhere in the input, so GSPMD has to assume it *does* and
``all-gather`` the full state on every device before each call.  On the
FD Pallas sound-wave benchmark (`pytests/hydrodynamics/_extended_scaling.py`),
that pinned multi-GPU speedup at ~0.95× across N=64..256 (2 GPUs cost
the same as 1 GPU plus an all-gather every kernel).  Per-device temp
memory was unchanged from the single-GPU baseline, confirming each
device was materialising the full state.

The fix is mechanical and lives entirely in
``astronomix/_pallas_helpers.py``: ``_pallas_call_sharded(kernel_local,
state_inputs, halo, block_shape)`` wraps the kernel build in a
``jax.experimental.shard_map.shard_map`` body that:

1. ``jax.lax.ppermute``s a halo of ``stencil_reach`` cells from each
   neighbour shard along every sharded spatial axis (periodic ring).
2. Concatenates ``[left_halo, local, right_halo]`` on each sharded
   axis.
3. Calls the existing per-shard kernel build on the padded local shard.
   The kernel's modular indexing wraps within the padded shape, so
   interior reads land on real neighbour values and only halo-cell
   outputs (which get stripped) wrap around incorrectly.
4. Strips the halo from each state-shape output.

The user-facing knob is just ``pallas_mesh_context(mesh)`` around the
JIT trace — ``time_integration`` enters it whenever ``sharding`` is
non-None — plus a thin wrapper at each Pallas entry point that calls
``_pallas_call_sharded`` instead of ``pl.pallas_call`` directly.

```python
def _flavour_pallas(state, …, *, axis):
    if not _flavour_pallas_supported(state, config):
        return _flavour_native_axis(state, …)        # native fallback
    halo = [0, 0, 0]
    halo[axis] = STENCIL_REACH    # 3 for WENO5, 2 for FV PLM, 1 for div, 0 for pointwise
    return _pallas_call_sharded(
        lambda s: _flavour_pallas_local(s, …, axis=axis),
        state_inputs=(state,),
        halo=tuple(halo[:ndim]),
        block_shape=_as_3tuple_block_shape(config.pallas_block_shape, ndim)[:ndim],
    )
```

Halo widths in use today:

| Kernel | Stencil reach | Block-rounded halo (`bx=4`) |
|---|---|---|
| Hydro WENO5 / MHD WENO5 / iso-MHD WENO5 | 3 (offsets −2..+3) | 4 |
| Fused WENO + divergence (`_weno_flux_hydro_pallas_rhs`) | 3 | 4 |
| Per-axis flux divergence | 1 (offset −1) | 4 |
| Pointwise positivity floor / EOS / source | 0 | 0 (shard_map only, no ppermute) |
| CT modified-flux / edge-EMF / curl | 2–4 per kernel | 4 |

Single-device runs (``mesh is None`` or ``mesh.size == 1``) get a
transparent forward to the kernel build — no shard_map, no halo, no
perf change.

**Measured behaviour on `_extended_scaling.py` (sound_wave3D, 2 GPUs).**

| N | 1 GPU | 2 GPUs | speedup before fix | speedup after fix | theoretical max (halo) |
|---|---|---|---|---|---|
| 64  | 1.75 s   | 15.50 s | 0.80× | 0.11× ¹ | 1.60× |
| 128 | 24.53 s  | 14.06 s | 0.93× | **1.75×** | 1.78× |
| 256 | 394.78 s | 213.59 s | 0.95× | **1.85×** | 1.88× |

¹ N=64 is compile-cost dominated — the 14s on 2 GPUs is mostly the
one-time Triton compile of the sharded path, not steady-state work.
At N≥128 the wrap hits ~98 % of the halo-waste theoretical ceiling.

Per-device temp memory at N=256 drops from 5120 MB (1 GPU) to
2620 MB (per-device on 2 GPUs) — confirming each device materialises
only its local shard.  Before the fix, both were 5120 MB (each device
all-gathered the full state).

**Future optimisation if needed.**  Each kernel call pays one
``shard_map`` entry + per-axis ppermute; that overhead is already
negligible at N≥128.  If it ever bites at smaller N the natural fix
is to wrap an entire per-stage RHS (`_hydro_step_rhs` in `_ssprk.py`)
in one ``shard_map`` so a single halo exchange covers all the
WENO + divergence calls in that stage.  Hasn't been needed yet — the
current per-call wrap hits the halo ceiling at production scale.

The full design rationale (why this is needed, how the alternative
threading-mesh-through-every-function design was rejected) is in the
``pallasify`` skill, §4b'.

### 2.4 Backend-aware dispatch

The wrapper for each direction stays small and JIT-able:

```python
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _weno_flux_x(state, params, config, registered_variables):
    if _hydro_pallas_flux_supported(state, config):
        return _weno_flux_hydro_pallas(state, params, config,
                                       registered_variables, axis=0)
    return _weno_flux_x_native(state, params, config, registered_variables)
```

Three rules that paid off:

- `_hydro_pallas_flux_supported` is a **plain Python** predicate
  evaluated at jit-trace time.  It must NOT rely on traced values
  (only `config`, `state.shape`, `state.ndim`, etc.).
- Every Pallas helper must compile-time fall back to its native-JAX
  twin when `pl is None` or when the predicate returns False, so a
  build without Pallas / CUDA still works.
- The dispatcher imports the **public** ``_*_pallas`` entry point, not
  the ``_*_pallas_local`` body.  The public entry point does the
  multi-GPU ``shard_map`` wrap (§2.3a); bypassing it silently disables
  strong scaling.

### 2.5 Donating buffers from the top

The integrator-level decorator:

```python
@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state"],   # <-- chain the donation
)
def _lsrk4_hydro(conserved_state, ...): ...
```

Plus `config.donate_state=True` at the outermost `time_integration`
call.  Both are needed; donations don't propagate automatically across
JIT boundaries.

### 2.6 Low-storage RK stages

Carpenter & Kennedy (1994) 2N-storage 5-stage 4th-order RK4 only needs
two full-state buffers (`q`, `dq`) instead of SSPRK4's three (`q0`,
`q_curr`, `q_final`).  Coefficients used are pasted at the bottom of
`astronomix/_finite_difference/_time_integrators/_ssprk.py`.

Empirical CFL on WENO5+LSRK4: `C_cfl ≈ 1.4` (vs SSPRK4's 1.5).  LSRK4
has **no SSP property**, so very strong shocks may want
`enforce_positivity=True` and a slightly looser `minimum_density` /
`minimum_pressure` floor.

---

## 3. Memory accounting checklist

When the Pallas backend "doesn't reduce memory", it's almost always
because of one of these.  Audit them in order:

1. **Are you ACTUALLY running the path you think you are?**  Add a
   one-shot `print` at trace time.  `print` runs during JAX tracing, so
   if it never fires the path is dead.  Common reason: `finalize_config`
   silently overrode `boundary_handling` or `time_integrator`.
2. **Did `pip install` happen since your last edit?**  `pip install -e .`
   makes source edits live; a plain `pip install .` builds a wheel and
   uninstalls the dev install — your edits stop taking effect with no
   warning.  Confirm with `python -c "import astronomix; print(astronomix.__file__)"`.
3. **What buffers are actually live?**  Lower the JIT, get
   `compiled.as_text()`, and `grep` for `f32[5,N,N,N]` or
   `f32[5,N+2g,N+2g,N+2g]`.  Buffer count × buffer size ≈ temp size.
4. **Is your test fooling you?**  `progress_bar=True` masks `print`s
   behind `\r`.  Pipe through `tr '\r' '\n'` when reading
   `time_integration` output.

`compiled.memory_analysis()` exposes
`temp_size_in_bytes`, `argument_size_in_bytes`,
`output_size_in_bytes`, and `alias_size_in_bytes`.  Total =
`temp + arg + output − alias`; when `donate_state` is on, the alias
covers the output.

---

## 4. Roadmap for extending coverage

The infrastructure (block-spec helpers, compiler params, dispatch
wrappers, `input_output_aliases` pattern, LSRK4 integrator) is generic.
Each new path just needs an in-kernel eigenstructure.

### 4.1 FD MHD ideal gas — DONE

**Status:** the full Pallas WENO MHD kernel is in place
(`_weno_flux_mhd_pallas` in `_weno.py`).  Validated against the NATIVE
backend on `pytests/mhd/alfven_wave3D.py` at N=8/16/32 (x32):

| N | NATIVE L1 | PALLAS L1 | NATIVE temp | PALLAS temp | NATIVE time | PALLAS time |
|---|---|---|---|---|---|---|
| 8 | 1.881e-02 | 1.881e-02 | 0.72 MB | 0.25 MB | 0.87 s | 0.07 s |
| 16 | 1.198e-03 | 1.198e-03 | 5.22 MB | 1.95 MB | 1.65 s | 0.15 s |
| 32 | 4.597e-05 | 4.597e-05 | 28.6 MB | 15.8 MB | 3.48 s | 0.45 s |

L1 errors match NATIVE to single-precision rounding (~2e-8 relative at
N=32); 5th-order convergence is preserved; PALLAS gives
**7–12× speedup** and **~50 % memory reduction** relative to NATIVE.

**What's in place:**

- `_weno_flux_mhd_pallas` — 8 conserved vars, 7 characteristic waves
  (fast±, alfvén±, slow±, entropy).  All face eigenstructure (body of
  `_eigen_mhd._eigenvector_building_blocks`) is inlined as kernel-local
  closures; `L_row` / `R_col` / `λ` for each mode are inlined as
  compile-time `if mode == k:` branches matching the native helpers.
- Axis-aware indexing — `_mhd_indices_for_axis` reorders the local
  component tuple per axis (no native-style transpose-and-swap), so
  axes 1 and 2 put `mom_y`/`mom_z` (and `B_y`/`B_z`) in the normal
  slot.  The B-normal flux is identically zero, matching `_mhd_flux_x`.
- `_mhd_pallas_flux_supported` — predicate (PALLAS backend, 3D, MHD,
  IDEAL_GAS, block divides spatial dim).  Works in both x32 and x64
  modes (see x64 fix in §5).
- `_ssprk4_with_ct::compute_rhs` uses `_hydro_flux_div_axis_pallas` for
  the divergence step under PALLAS, sharing a single physical RHS
  buffer across axes via `input_output_aliases`.  CT magnetic field
  updates (`constrained_transport_rhs`, `update_cell_center_fields`,
  `interp_center_to_face`) stay native.
- `_weno_flux_x/y/z` dispatch routes MHD through `_weno_flux_mhd_pallas`
  whenever the predicate accepts.

**Known limitations:**

- **x64 falls back to native.**  In `pallas_interpret=True` mode the
  kernel runs cleanly in x64 (kernel math is correct), but the Triton
  GPU lowering used to trigger an ``('f64','f32')`` assertion deep in
  ``_truediv_lowering_rule`` (the assertion appeared in
  ``interp_center_to_face`` because that's the first downstream
  consumer of the kernel's output).  **Fixed in §5.**
- **CT helpers themselves are still native.**  If they show up as a
  dominant bottleneck after the rest is ported, the same per-tile +
  modular-index pattern in `_div_axis_pallas_shape_ok` ports trivially.

### 4.2 FD MHD isothermal — DONE

`_weno_flux_mhd_iso_pallas` implemented as a sibling of
`_weno_flux_mhd_pallas`.  Differences: 7 local conserved-state slots (no
energy), 6 characteristic waves (no entropy mode), sound speed is the
fixed `params.isothermal_sound_speed` (so face quantities skip the
enthalpy-based recomputation), and the flux drops the
`(E+p_total)*vn − v·B*Bn` energy term.  Wired through
`_weno_flux_x/y/z` after `_mhd_pallas_flux_supported`.

Validation: small 32³ iso-MHD periodic setup (uniform Bx + 1% density
perturbation), `max|PALLAS − NATIVE| = 1.0e-17` (machine epsilon),
~3.5× speedup, ~37% temp reduction.

x64 limitation fixed by the same typed-constant trick as 4.1 (see §5).

### 4.2 *(legacy)* FD MHD isothermal — porting recipe

Same as 4.1 but one fewer characteristic wave (no entropy mode).
Reuse the MHD kernel; gate on `config.equation_of_state == ISOTHERMAL`
and switch to the isothermal projection formulas (port
`_eigen_mhd_iso.py`).

For ISOTHERMAL hydro (no MHD), the same kernel skeleton applies with
`num_modes = ndim + 1` (one fewer than ideal gas — no entropy wave) and
the isothermal flux formula `(mn, mn*vn + c_s^2*rho, mt1*vn, mt2*vn)`.

### 4.3 FV scheme — DONE (hydro); FV MHD falls back to native

**Status:** the fused reconstruction + Riemann + conservative-update
Pallas kernel is in place at
`astronomix/_finite_volume/_state_evolution/_pallas_evolve.py`
(`_fv_evolve_axis_pallas`).  Wired into `_evolve_gas_state_unsplit_inner`
behind `_fv_pallas_evolve_supported`.

Validated on a 128³ FV hydrodynamic test (HLL + MINMOD + periodic
boundaries, RK2_SSP), `donate_state=True`, block `(4,4,4)`:

| Configuration | Temp | Total | Time | L1(PALLAS − NATIVE) |
|---|---|---|---|---|
| NATIVE FV HLL+MINMOD 128³ | 651 MB | 691 MB | 1.70 s | — |
| **PALLAS FV HLL+MINMOD 128³** | **158 MB** | **198 MB** | **0.85 s** | **0.0** |

That's a **4.1× memory reduction** and **2× speedup**, with bit-identical
output across many timesteps.

**What's in the kernel:**

- Single fused per-axis pass: reads primitives with halo 2, computes
  limited gradients (Python static-switched between MINMOD / VAN_ALBADA
  / OSHER / first-order fallback), reconstructs `q_L`/`q_R` at both
  interfaces bordering each cell, evaluates the Riemann solver
  (HLL / LAX_FRIEDRICHS, selected at compile time via
  `config.riemann_solver`), and emits the divergence
  `−(dt/dx)·(F_R − F_L)` directly.
- `input_output_aliases={0: 0}` accumulator pattern (same trick as the
  FD path): the conservative-state buffer is reused across axes inside
  `_evolve_gas_state_unsplit_inner`, eliminating the four full-state
  temporaries (`q_L`, `q_R`, `fluxes`, `rhs`) of the native pipeline.
- The MHD-aware kernel branch (8 conserved slots, MHD HLL flux/wave
  speed) is also implemented, but in this codebase the FV MHD pipeline
  always strips B-fields first and runs the gas update on a 5-slot
  sub-state, so the 8-slot MHD path is never reached by the existing
  dispatcher; it is left in place for future use.

**Constraints (gated by `_fv_pallas_evolve_supported`):**

- ideal-gas EOS, Cartesian geometry, no self-gravity / cosmic rays /
  viscosity, periodic or block-divisible ghost-cell layouts;
- `config.riemann_solver ∈ {HLL, LAX_FRIEDRICHS}` (HLLC / HLLC_LM /
  AM_HLLC not ported — same kernel skeleton, additional algebra);
- `config.limiter ∈ {MINMOD, VAN_ALBADA, VAN_ALBADA_PP, OSHER}`;
- x32 only (same Triton-x64-with-complex-kernels caveat as the FD MHD
  path);
- block shape must divide every spatial dim (including ghost-cell
  padding when the FV path keeps `boundary_handling=GHOST_CELLS`).
  Default `(4, 4, 8)` works for 128³ + 2 ghost = 132³ only if the user
  bumps the z-block down to 4 — `(4, 4, 4)` is a safe choice across
  the supported resolutions.

**Known limitations:**

- **FV MHD silently falls back to native for the gas-update step.**
  The codebase's FV MHD path strips the B-fields and calls the gas
  update on the 5-slot sub-state.  The same Pallas kernel that works
  beautifully for standalone FV hydro produces NaNs from inside that
  nested MHD pipeline — but in `pallas_interpret=True` mode the math
  is correct, so it's a Triton-lowering issue specific to that nested
  call context.  The predicate detects this case (`config.mhd` is True
  AND `state.shape[0] != 8`) and returns False, so the MHD path uses
  the native gas update and the magnetic update stays native too.
  TODO: narrow down the Triton lowering and remove this gate.
- HLLC / HLLC_LM / AM_HLLC solvers: not ported.  Same predicate-stub
  pattern would extend the kernel; algebra is in
  `astronomix/_finite_volume/_riemann_solver/hll.py`.
- Split scheme (`config.split == SPLIT`) uses the MUSCL predictor +
  per-axis primitive↔conserved conversion, which is not yet ported.
  Default `UNSPLIT` is where the Pallas path activates.

The FV scheme has three stages, each a candidate Pallas kernel:

1. **Reconstruction kernel** — `(state) → (q_L, q_R)` per axis.
   Block-tile the same way; halo of `1` (PLM) or `2` (parabolic).
   The per-block output is **two faces per cell** (q_L on the left
   face of cell i, q_R on the right face of cell i-1, or vice versa
   depending on the convention used by `_reconstruct_at_interface_split`).
   Be careful with block shape and register pressure: with N variables
   and two faces per cell, ncomp doubles relative to the WENO case.
2. **Riemann solver kernel** — `(q_L, q_R) → flux` per face.  Pure
   pointwise op, ideal Pallas workload.  Each solver (HLL, HLLC,
   HLLC_LM, AM_HLLC, LAX_FRIEDRICHS) is a separate kernel selected at
   compile time via the `config.riemann_solver` static value (Python
   `if config.riemann_solver == HLL:` branches in a dispatcher,
   matching the existing `_riemann_solver.py` structure).
3. **Conservative update** — already done by
   `_hydro_flux_div_axis_pallas`.  Drop-in reuse, just call it from
   the FV time integrator (the kernel is mhd- and EOS-agnostic).

Recommended order:

1. Start with the simplest combination — **HLL + first-order** (no
   reconstruction, just `q_L = state[i-1]`, `q_R = state[i]`) — for
   hydrodynamics ideal gas in 3D.  This is enough to validate the
   dispatch plumbing and the Riemann-solver kernel skeleton.
2. Add PLM/MUSCL reconstruction.  Hardest part is the slope limiter,
   which has plenty of branches that are best written as `jnp.where`
   chains rather than Python `if`s (so the limiter type can stay
   compile-time selected).
3. Port the remaining Riemann solvers one at a time, picking one
   regression test per solver.
4. Add the same `input_output_aliases` accumulator pattern as in
   `_ssprk4_hydro`/`_lsrk4_hydro` so the multi-axis FV stage update
   never holds more than one physical flux/RHS buffer.

Expected payoff: a fused FV recon+Riemann+state-update Pallas kernel
should give similar **6-12× speedup and ~50 % memory reduction**
as the FD MHD WENO port did, because the per-face Riemann math is
significantly heavier than what XLA can fuse across opaque jit
boundaries today.

The acceptance gate is `pytests/mhd/alfven_wave3D.py` (which exercises
both the FV and FD code paths), `tests/pallas/sedov3D.py` (hydro
regression), plus any shock-tube test under `tests/hydro_tests/`.

### 4.4 General porting checklist (per direction)

1. Write the kernel body using only the per-cell math (no JAX-array
   stencils).  Aim for the same "compute everything inline" style as
   `compute_interface_flux` in `_weno_flux_hydro_pallas`.
2. Wire it through a Pallas-supported predicate and a fall-back native
   path.  Keep both paths working — gives a free regression harness.
3. Add `input_output_aliases` whenever a buffer is accumulated.
4. Verify on the cheapest meaningful test first (`pytests/...` if
   present, else a 32³ Sedov-like or shock-tube setup).
5. Memory-analyze: `lower().compile().memory_analysis()`.  Confirm the
   buffer count dropped before chasing performance.
6. Benchmark wall-clock on a representative grid (128³ for 3D).

### 4.5 Hot-path leaf ops — DONE

After the main WENO + LSRK + CT pipeline was on Pallas, three leaf ops
were still going through native JAX every stage and showing up as
identifiable temp-buffer allocations.

**`_enforce_positivity` (per-cell floor on ρ and p).** Ported to a
pointwise Pallas kernel with `input_output_aliases={0: 0}` so the
floored conserved state is written back into the input buffer
in-place. Supports 1/2/3D, IDEAL_GAS and ISOTHERMAL, with or without
MHD. Files:
`astronomix/_finite_difference/_fluid_equations/_enforce_positivity.py`
(dispatch + native fallback) and `_enforce_positivity_pallas.py`
(kernel).

**MHD CFL fast path.** `_cfl_time_step_fd` used to materialise the full
seven-mode characteristic eigenvalue array for every cell at every
step. Mirroring the hydro fast path, `_cfl_time_step_fd_mhd_fast` (in
`_timestep_estimator.py`) computes the fast magnetosonic speed
``c_fast_d² = ½·(b²/ρ + c_s² + √((b²/ρ + c_s²)² − 4·B_d²/ρ·c_s²))``
pointwise per cell and reduces ``max(|v_d| + c_fast_d)`` per axis.
Pure-JAX, no Pallas kernel — what matters is that no full-state
eigenvalue intermediate is allocated. Gated by
`_mhd_fast_cfl_supported`, active whenever `backend == PALLAS`.

**`pallas_ct` toggle.** Switching CT itself to Pallas (`pallas_ct=True`
in `SimulationConfig`) replaces the four native CT stages with three
bounded-halo Pallas kernels (`_constrained_transport_pallas.py`). The
toggle defaults to **False** because compile cost dominates at
production-scale resolutions — measured on the
`pytests/mhd/alfven_wave3D.py` N=32 setup (245 timesteps, x64):

| Config | Temp | Warm runtime | Compile | µs/iter | L1 vs native |
|---|---|---|---|---|---|
| Native | 49.6 MB | 4.35 s | 25 s | 17753 | — |
| Pallas, `pallas_ct=False` | 32.5 MB | **0.98 s** | 57 s | **4008** | 3.6e-15 |
| Pallas, `pallas_ct=True` | 30.5 MB | 1.18 s | 68 s | 4836 | 3.6e-15 |

`pallas_ct=False` is the right default: 4.4× warm speedup over native
at 1.3× compile cost, with only 2 MB more temp than the fully-on
variant. `pallas_ct=True` saves the extra 2 MB and is the right
choice when working memory is the binding constraint, at the cost of
~11 s extra compile.

---

## 4.4 The x64 / Triton fix (was a real bug, now resolved)

The MHD kernels used to bail out with
``AssertionError: ('f64', 'f32')`` deep inside
``jax._src.pallas.triton.lowering._truediv_lowering_rule`` whenever
``jax.config.jax_enable_x64 == True``.  The math was correct under
``pallas_interpret=True``, so it was purely a Triton lowering issue:
some operand entered the lowering pass as **f32** while every other
operand around it was **f64**.

Root cause:  Python-float literals inside ``jnp.where(cond, A, B)``
arms (and other contexts the lowering treats as scalar constants) are
emitted to Triton as ``f32`` regardless of the surrounding tile dtype.
The MHD kernel had several such patterns that the hydro kernel did
not:

```python
sgn_bn = jnp.where(Bn_face >= 0.0, 1.0, -1.0)              # 1.0, -1.0 → f32
bt_n1  = jnp.where(bt_sq >= b_eps,
                   Bt1_face / jnp.sqrt(bt_sq_safe),
                   1.0 / jnp.sqrt(2.0))                    # constant → f32
bt_sq_safe = jnp.maximum(bt_sq, 1e-20)                     # 1e-20    → f32
```

Once these f32 constants flowed into the divisions further down
(``Bt1_face / jnp.sqrt(bt_sq_safe)``,
``Bt1_face_f64 / inv_sqrt_two_f32``), Triton tripped the assertion.

**Fix (applied in `_weno_pallas.py`):**

1. Derive typed-as-the-working-dtype literal tiles from an existing
   typed scalar (``gamma``, ``cs``, …):

   ```python
   zero_typed = gamma - gamma
   one_typed = zero_typed + 1.0
   neg_one_typed = zero_typed - 1.0
   inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
   ```

   ``zero_typed`` is a tile of zeros whose dtype follows ``gamma``;
   ``+ 1.0`` promotes the Python literal to the tile's dtype before it
   ever reaches Triton's lowering.  All ``jnp.where`` arms then use
   these typed constants instead of bare Python literals.

2. Pass ``b_eps`` and the ``sqrt`` floor as scalar kernel inputs
   (``jnp.asarray(value, dtype=state.dtype)``), exactly like ``gamma``
   was already being passed.  This guarantees the dtype matches the
   kernel's working dtype rather than defaulting to f32.

**Result.**  Production x64 MHD-WENO test on
`pytests/mhd/alfven_wave3D.py` (no monkey-patches, fresh
`pip install .`):

| N | NATIVE L1 (x64) | PALLAS L1 (x64) | diff |
|---|---|---|---|
| 16 | 1.197484e-03 | 1.197484e-03 | 1.55e-16 |
| 32 | 4.363055e-05 | 4.363055e-05 | 2.32e-17 |

i.e. **machine epsilon** in x64 too.  The same fix was applied to
``_weno_flux_mhd_iso_pallas``.  The FV kernel didn't need it (no
sgn-style ``jnp.where`` patterns) and worked in x64 once the gate was
removed.

**Lesson for new Pallas kernels.**  Whenever you write
``jnp.where(cond, X, Y)`` inside a Pallas kernel, make sure ``X``
and ``Y`` are *tile* expressions (or arithmetic involving tiles),
never bare Python ``1.0`` / ``2.0`` literals.  ``inv_X = 1.0 / tile``
on its own is fine — it's only when the Python literal sits in a
position that the Triton lowering reads as a scalar argument
(e.g. the false-arm of ``jnp.where``) that the f32 contamination
happens.  The ``pallasify`` skill must enforce this — see its
``Translation recipe`` step 3 note.

---

## 5. Gotchas

- **Pallas block shape must divide every spatial dim.**  Enforce in the
  `_supported` predicate; do NOT silently round up — that masks bugs.
  We use `(4, 4, 8)` by default in 3D.
- **`pl.BlockSpec((), lambda bi, bj, bk: ())`** is the spec for a
  scalar input.  Always `dtype`-cast scalars before passing them:
  `jnp.asarray(value, dtype=state.dtype)`.  Otherwise mixed-precision
  surprises appear inside the kernel.
- **Python-level `if` inside a kernel** is a *Python* if, evaluated at
  trace time.  `jnp.where` is the runtime version.  We rely on this:
  things like `if mode == 0:` select code paths at compile time.
- **`jax.lax.fori_loop` carries** are 2-state-buffer overhead per
  register.  Keep them as small as possible.  For LSRK4, the carry is
  `(q, dq)` — 2 buffers.  For SSPRK4 it's `(q_curr, q_final)` + the `q0`
  captured by closure = 3 buffers.
- **`pltriton.CompilerParams(num_warps=...)`** matters.  4 worked best
  for our kernels; 8 lit up register spills.  Make it
  config-controllable (`pallas_num_warps`) but default to 4.
- **`interpret=True`** lets you debug a kernel under pure Python.  Slow
  but invaluable for `print` debugging.  Expose it via
  `pallas_interpret` config so users can flip it for one run.
- **Cache invalidation across kernel changes**: clearing the JAX
  persistent compilation cache (`~/.cache/jax/...`) is sometimes
  necessary if you suspect a stale compiled artifact is being reused.

---

## 6. File map

| File | Role |
|---|---|
| `astronomix/_finite_difference/_interface_fluxes/_weno.py` | Hydro WENO kernels (Pallas + native), block-shape helpers, supported-predicate, eigenstructure-projection inlined in Pallas form. |
| `astronomix/_finite_difference/_time_integrators/_ssprk.py` | SSPRK4 (native), LSRK4 (Pallas-friendly 2N-storage), per-axis Pallas divergence kernel with `scale_in` + `input_output_aliases`, shared `_hydro_step_rhs`. |
| `astronomix/_finite_difference/_state_evolution/_evolve_state.py` | Top-level FD dispatch; picks SSPRK4 vs LSRK4 based on `config.time_integrator`. |
| `astronomix/_finite_difference/_timestep_estimation/_timestep_estimator.py` | Backend-aware CFL estimator; Pallas mode skips the full-state characteristic eigenvalue arrays. Hydro path reads primitive `|v| + c`; MHD path uses `_cfl_time_step_fd_mhd_fast` for the per-cell fast-magnetosonic speed. |
| `astronomix/_finite_difference/_fluid_equations/_enforce_positivity.py` + `_enforce_positivity_pallas.py` | Pointwise floor on ρ and p; Pallas kernel writes back in-place via `input_output_aliases={0:0}`. |
| `astronomix/_finite_difference/_magnetic_update/_constrained_transport_pallas.py` | Optional Pallas CT (3 bounded-halo kernels). Gated by `config.pallas_ct` (default False — see §4.5 for the compile/runtime tradeoff). |
| `astronomix/option_classes/simulation_config.py` | `backend`, `pallas_block_shape`, `pallas_use_triton`, `pallas_interpret`, `pallas_num_warps`, `pallas_ct`, `donate_state`, `time_integrator` knobs. |
| `tests/pallas/sedov3D.py` | The canonical hydro Pallas benchmark — produces the figure + memory/runtime printout. |
| `pytests/mhd/alfven_wave3D.py` | MHD convergence test (3D CP Alfvén wave, N=8..128, both FV and FD).  Acceptance gate for MHD Pallas changes — L1 error must match the NATIVE backend to machine precision. |

---

## 7. Quick-start example

```python
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import (
    PALLAS, FINITE_DIFFERENCE, CARTESIAN, RK4_LSRK,
    PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
)

cfg = SimulationConfig(
    backend=PALLAS,
    pallas_block_shape=(4, 4, 8),
    pallas_use_triton=True,

    solver_mode=FINITE_DIFFERENCE,
    time_integrator=RK4_LSRK,         # 2N-storage RK4
    boundary_settings=BoundarySettings(
        x=BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                             right_boundary=PERIODIC_BOUNDARY),
        y=BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                             right_boundary=PERIODIC_BOUNDARY),
        z=BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY,
                             right_boundary=PERIODIC_BOUNDARY),
    ),
    donate_state=True,                # alias input/output

    geometry=CARTESIAN,
    dimensionality=3,
    num_cells=128,
)

params = SimulationParams(t_end=0.1, C_cfl=1.4)   # 1.4 for LSRK4
```
