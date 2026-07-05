---
name: pallasify
description: Translate a native-JAX numerical kernel in this astronomix codebase into the matching Pallas backend kernel, following the conventions in `pallas_backend_implementation_guide.md`. Use when the user asks to "pallasify", "port to Pallas", "compile to Pallas", "regenerate the Pallas backend for X", or has just modified a `*_native` function and wants the `*_pallas` sibling refreshed. Pure code-translation: native JAX in, Pallas kernel out — the developer does NOT touch `_pallas` files by hand.
---

# pallasify — Compile native JAX to a Pallas backend kernel

This skill is the developer-loop mechanism for the astronomix Pallas
backend: **a developer writes / tweaks a native JAX function, runs this
skill, and the matching Pallas kernel in `_pallas` files is regenerated
to match**.  The Pallas side is treated as compiled output that mirrors
the native side — the user should not edit Pallas files by hand.

The full design rationale and the patterns this skill follows are in
`pallas_backend_implementation_guide.md` at the repo root.  Read it
before doing anything non-trivial; everything below assumes you have.

---

## When you are invoked

The user has either:

1. **Modified an existing native function** (e.g. `_weno_flux_x_native`,
   `_evolve_state_along_axis`, a Riemann solver, a reconstruction step)
   and wants its `_pallas` sibling refreshed; or
2. **Asked for a new native function to be pallasified**.

You will need to:

1. Identify the native function (the user names it; if ambiguous, ask).
2. Locate its `_pallas` module sibling (same-directory `_*_pallas.py`,
   or the top-level FV `_pallas_evolve.py`).  If none exists yet, create
   one — `astronomix/<package>/<subpackage>/_<kernel>_pallas.py`.
3. Either generate a new Pallas kernel from scratch or rewrite an
   existing one so its math, control flow and dispatch logic match the
   native version exactly.
4. Wire the dispatch in the *native* file at the bottom (lazy import
   pattern to avoid circular imports — see step 6).
5. Validate against the matching native version on the cheapest
   relevant test (`tests/pallas/sedov3D.py`,
   `pytests/mhd/alfven_wave3D.py`, or a small standalone smoke test).
   PALLAS and NATIVE must match to single-precision rounding.

---

## File layout to use

```
astronomix/
  _pallas_helpers.py                     ← block-shape, compiler-params, pl, pltriton
  _finite_difference/
    _interface_fluxes/
      _weno.py                           ← native + dispatcher (your edits go here)
      _weno_pallas.py                    ← generated; do not hand-edit
    _time_integrators/
      _ssprk.py                          ← native + integrator + dispatcher
      _ssprk_pallas.py                   ← generated; do not hand-edit
  _finite_volume/
    _state_evolution/
      evolve_state.py                    ← native + dispatcher
      _pallas_evolve.py                  ← generated; do not hand-edit
```

A `_pallas` module must:
- Import shared helpers from `astronomix._pallas_helpers`
  (`_as_3tuple_block_shape`, `_backend_is_pallas`,
  `_pallas_compiler_params`, `_pallas_call_sharded`, `pl`, `pltriton`).
- **Not** import from its native sibling at module load — use lazy
  imports inside the function body if a native fallback is needed.  The
  native module imports from the Pallas module at the **bottom** of
  its file.
- Expose a `_<flavour>_pallas_supported(state, config)` predicate that
  is a plain Python function and gates the Pallas kernel.
- Expose a `_<flavour>_indices_for_axis(config, registered_variables,
  axis)` if the algorithm uses characteristic projection / per-axis
  component permutation.
- Expose the actual public kernel (`_<flavour>_pallas`) as a **thin
  shard-aware wrapper** that:
  - takes the same arguments as the native function plus `axis` (if
    per-axis) and any accumulator buffers;
  - dispatches the native fallback if `_supported(...) is False`;
  - otherwise calls `_pallas_call_sharded(...)` so the actual
    `pl.pallas_call` is routed through ``shard_map`` + a periodic
    ppermute halo exchange whenever a multi-device mesh is active.
- Expose `_<flavour>_pallas_local` containing the actual
  ``pl.pallas_call`` build — this is the single-shard body that
  ``_pallas_call_sharded`` invokes either directly (single device) or
  inside the ``shard_map`` body on each shard (multi device).  Its
  ``BlockSpec``, ``out_shape`` and ``grid`` are derived from the input
  array's shape *inside the body* so the same code runs for the global
  shape or a halo-padded local shape.
- Use `pl.BlockSpec`, `pl.program_id`, modular indexing for the
  stencil reads, and `input_output_aliases={0:0}` for any
  accumulator buffer that is reused across axes.

---

## Decide first: pallasify, rewrite, or leave alone

Not every native op should become a Pallas kernel. Before opening the
recipe, classify the target:

- **Heavy stencil with characteristic projection** (WENO flux, FV
  reconstruct+Riemann, CT stages) → pallasify with the recipe below.
- **Pure pointwise op** (positivity floor, EOS conversion, source
  term) → pallasify, but use the *pointwise leaf-op* shape in §4b: no
  halo, `input_output_aliases={0:0}`, pass-through + selective
  overwrite. Cheap to compile, removes one full-state intermediate per
  call.
- **Op whose cost is a temp buffer, not arithmetic** (e.g. the MHD CFL
  estimator allocating a full-state 7-eigenvalue array) → **do not
  pallasify**. Rewrite it in pure JAX without the intermediate. The
  hot example here is `_cfl_time_step_fd_mhd_fast` —
  `max(|v_d| + c_fast_d)` per cell, no Pallas needed.
- **Multi-stage native pipeline with halo > ~6 if fused** (e.g.
  CT: modified flux → edge EMF → curl → cell-center update) → split
  into one Pallas kernel per stage with bounded halo each. A single
  fused kernel with halo ~8 made Triton lowering hang for >5 min;
  three kernels with halo ≤2 each compile in <1 s apiece.
- **Optional / opt-in Pallas variants**: if the pallasified version
  saves <5% runtime at production scale but costs >5 s extra
  compile (the CT case at production-N), expose a `config.<flag>`
  and have `_supported(...)` short-circuit on `if not config.<flag>:
  return False`. Predicate machinery is the right place to make a
  Pallas backend "available but off by default", not just to gate on
  hardware capability. See `pallas_ct` in `simulation_config.py`.

If the decision is "leave alone", stop here. Document the reasoning in
a one-line comment in the dispatcher so the next pass doesn't re-open
the question.

---

## Translation recipe

For each native function being pallasified:

### 1. Read the native body and classify each operation

Walk the native body top-to-bottom and tag each statement:

- **stencil read**: `_shift(x, k, axis=a)` or `jnp.roll(x, k, axis=a)`
  — becomes `q_ref[var, (ii + k) % nx, jj, kk]` (or analogous for the
  active axis) inside the kernel.  `axis` in the kernel is the *normal*
  direction; for off-axis variants, do an axis-aware
  `local_indices` permutation (see `_mhd_indices_for_axis` for the 8-var
  MHD example) rather than physical transpose+swap.
- **pointwise op**: `+`, `*`, `jnp.maximum`, `jnp.where`, `jnp.sqrt`,
  etc. — copy 1:1 into the kernel.
- **eos/eigenstructure call** (`primitive_state_from_conserved`,
  `_eigen_L_row_*`, `_eigen_R_col_*`, `_eigen_lambdas_*`,
  `_calculate_limited_gradients`, `_riemann_solver`, …) — **inline** the
  body as a kernel-local closure.  These calls cannot stay as function
  calls inside a Pallas kernel; their bodies must be expanded so every
  intermediate is per-tile compute.  `jax.lax.switch(mode, [f0, …, fk])`
  becomes `if mode == 0: … elif mode == 1: …` (Python-time dispatch,
  compiled away).
- **whole-state operation** (`F = _euler_flux(state, ...)`,
  `flux_minus_shift`, …) — pull the per-cell expression out of the
  whole-state function and replicate it per-tile.  If the function is
  used elsewhere, leave the native function alone — just hand-mirror
  its body inside the kernel.
- **boundary handler call** — for periodic boundaries the kernel's
  modular indexing handles them for free; the dispatcher only calls the
  native boundary handler ahead of the Pallas kernel when
  `config.boundary_handling == GHOST_CELLS`.

### 2. Build the kernel skeleton

The skeleton is **two functions, not one**, so the same kernel build
runs on either the global state (single device) or a halo-padded local
shard (multi device).  3-D variant shown; 1-D / 2-D drop the trailing
indices analogously — see `_weno_flux_hydro_pallas` /
`_weno_flux_hydro_pallas_local` for the canonical multi-dim form.

```python
def _flavour_pallas_supported(state, config) -> bool:
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    # …add flavour-specific gates: equation_of_state, mhd, ndim, dtype…
    if jax.config.jax_enable_x64 and not bool(getattr(config, "pallas_interpret", False)):
        return False  # Triton-x64 caveat — see guide §4
    block_shape = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    for n, b in zip(state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _flavour_indices_for_axis(config, registered_variables, axis):
    """Local component order for axis-aware kernels."""
    …  # axis=0 → (density, p_normal=mx, …); axis=1 → swap x/y; axis=2 → swap x/z.


def _flavour_pallas(state, …, *, axis):
    """Public entry point: predicate gate + multi-GPU shard_map wrap.

    On a single device this just calls ``_flavour_pallas_local`` directly.
    On a multi-device mesh ``_pallas_call_sharded`` halo-pads the state
    once per sharded spatial axis (via ppermute), calls
    ``_flavour_pallas_local`` on each shard, and strips the halo off the
    output — so the kernel body never has to know it's been sharded.
    """
    if not _flavour_pallas_supported(state, config):
        from astronomix.<…>._<…> import _flavour_native_x, _flavour_native_y, _flavour_native_z  # lazy
        return [_flavour_native_x, _flavour_native_y, _flavour_native_z][axis](state, …)

    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)

    # Per-axis stencil reach (== the deepest offset the kernel reads from
    # ``q_ref[var, (ii + offset) % nx, …]`` along the active axis).
    # WENO5 → 3, divergence → 1, pointwise → 0.  The wrapper rounds this
    # up to a multiple of the block size on each axis.
    halo_list = [0, 0, 0]
    if 0 <= int(axis) < ndim:
        halo_list[int(axis)] = STENCIL_REACH
    halo = tuple(halo_list[:ndim])

    def _local(state_local):
        return _flavour_pallas_local(state_local, …, axis=axis)

    return _pallas_call_sharded(
        _local,
        state_inputs=(state,),
        halo=halo,
        block_shape=block_shape[:ndim],
    )


def _flavour_pallas_local(state, …, *, axis):
    """Single-shard kernel build.

    Crucial property: every shape-derived value (``nx``, ``ny``, ``nz``,
    ``grid``, ``out_shape``, the ``in_state_spec`` shape) is read from
    *this* function's ``state.shape`` argument — never closed over from
    enclosing scope — so when the wrapper invokes it on a halo-padded
    local shard the kernel grid and modular indexing automatically resize.
    """
    ndim = int(config.dimensionality)
    nvars = int(state.shape[0])
    spatial_shape = tuple(int(x) for x in state.shape[1:])
    nx, ny, nz = (spatial_shape + (1, 1))[:3]
    bx, by, bz = _as_3tuple_block_shape(getattr(config, "pallas_block_shape", None), ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _flavour_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    # …flavour constants (num_modes, epsilon, tiny, b_eps if MHD)…

    if ndim == 3:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    # …1-D / 2-D variants…
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, *scalar_refs, out_ref):
        bi = pl.program_id(0); bj = pl.program_id(1); bk = pl.program_id(2)
        # Modular index arrays — periodic BC for free.  Inside ``shard_map``
        # the modular wrap is over the *padded* local size; halo cells at
        # the boundary do compute output that the wrapper later strips, so
        # only the interior is correctness-critical.
        ii = (bi*bx + jnp.arange(bx)[:, None, None]) % nx
        jj = (bj*by + jnp.arange(by)[None, :, None]) % ny
        kk = (bk*bz + jnp.arange(bz)[None, None, :]) % nz
        # Scalars: ref[()] once at the top.
        gamma = gamma_ref[()]
        …

        def q_at(var_index, offset):
            if axis == 0: return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1: return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        # Inline what the native function does — closures for primitive_from_q,
        # floored_cell, flux_from_q, left_project, add_right_correction, …
        …

        for var in range(nvars):
            out_ref[var, …] = …  # write every conserved slot

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(state.shape, state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, …],
        out_specs=out_spec,
        interpret=bool(getattr(config, "pallas_interpret", False)),
        name=f"flavour_axis_{axis}",
        **({"compiler_params": _pallas_compiler_params(config)} if _pallas_compiler_params(config) else {}),
    )(state, jnp.asarray(scalar_value, dtype=state.dtype), …)
```

### 3. Translate the native math

For each kernel-local closure that mirrors a native helper:

- `_shift(arr, k, axis=a)` reads → `q_at(var, k)` with the appropriate
  axis-conditional indexing.
- `state[index]` → take `index` from `local_indices` so the same kernel
  body works for any axis.
- `jnp.einsum('nxyz,nxyz->xyz', L_row, F)` → `sum_i L[i] * F[i]`
  per-tile, written as a Python sum over the 5–8 components.
- `jax.lax.switch(mode, [col_0, …, col_k])` →
  `if mode == 0: R = (…) elif mode == 1: R = (…) …` (Python-static).
- `jax.lax.cond(pred, t, f)` → `jnp.where(pred, t_value, f_value)` if
  you can predicate both sides; otherwise expand to a `jnp.where`
  multi-line equivalent.
- Whole-state allocations like `S = jnp.zeros_like(state)` followed by
  scattered writes → just emit zeros into `out_ref[var, …] = 0.0` at
  the top of the kernel and overwrite per local index.

### 4. Hook up `input_output_aliases` for any accumulator

Any kernel that gets called once per axis with a running buffer (rhs,
dq, conservative_change…) should expose an `accumulator=None` kwarg.
When provided:

- Put the accumulator first in the input list of `pl.pallas_call`.
- Add `kwargs["input_output_aliases"] = {0: 0}` so XLA reuses one
  physical buffer across calls.
- Inside the kernel, read `accumulator_in_ref[var, …]` and write
  `out_ref[var, …] = scale * accumulator_in_ref[…] + new_contrib`.
- In the **public** wrapper, pass both the accumulator and the
  flux/state input through ``_pallas_call_sharded`` with
  ``state_inputs=(accumulator, dF)`` (accumulator first, matching the
  ``input_output_aliases={0:0}`` order).  The wrapper applies the same
  halo to both — that's harmless for the accumulator (which is only
  read at the local cell) and correct for the stencil-reading flux.

`_hydro_flux_div_axis_pallas` in `_ssprk_pallas.py` is the canonical
example; `_fv_evolve_axis_pallas` shows the same trick on the FV side.

### 4b. Pointwise leaf-op shape (no halo, in-place)

For per-cell ops with no spatial dependence (positivity floor, EOS
conversion, source-term application), use this simpler shape rather
than the stencil+accumulator pattern in §4:

- Single `in_spec` for the conserved state, single `out_spec`, plus
  scalar specs for typed parameters (`gamma`, `minimum_density`, …).
- `kwargs["input_output_aliases"] = {0: 0}` so XLA reuses the input
  buffer for the output — one full-state buffer saved per call,
  every stage.
- Inside the kernel, close `ii, jj, kk` over the program ID and write a
  small `read(var)` helper. Then **pass every variable through, and
  selectively overwrite the ones the op touches**:
  ```python
  for var in range(nvars):
      if var == DENSITY:
          out_ref[var, ...] = rho_floored
      elif is_ideal and var == E:
          out_ref[var, ...] = energy_floored
      else:
          out_ref[var, ...] = read(var)
  ```
- Same `_as_3tuple_block_shape` / `_pallas_compiler_params` plumbing
  as the stencil case. The predicate is shorter — no per-axis halo
  check, just `state.ndim == ndim + 1` and block-divisibility.
- **Still wrap in ``_pallas_call_sharded`` with ``halo=(0,)*ndim``.**
  Even pointwise kernels are opaque to GSPMD and would trigger a
  full-state ``all-gather`` on a sharded input.  The helper with
  ``halo=0`` skips the ppermute but still routes through
  ``shard_map``, which is what tells GSPMD "this kernel can run
  locally on each shard, no collective needed".

`_enforce_positivity_pallas` in
`_finite_difference/_fluid_equations/` is the canonical example;
copy its skeleton for new leaf ops.

### 4b'. Multi-GPU: always route through `_pallas_call_sharded`

A ``pl.pallas_call`` is opaque to GSPMD.  Its ``BlockSpec`` index map is
``lambda bi, bj, bk: (0, 0, 0, 0)`` — every block program *can* read
anywhere in the input array, so GSPMD has to assume it *does* and
``all-gather`` the full state on every device before each call.  On the
FD Pallas sound-wave benchmark that pinned multi-GPU speedup at ~0.95×.

The fix is mechanical: when ``pallas_mesh_context(mesh)`` is active
around the JIT trace (``time_integration`` does this whenever the user
passes a ``sharding``), wrap the public kernel's ``pl.pallas_call`` in
``_pallas_call_sharded``.  That helper:

1. Reads the active mesh from the contextvar.
2. Infers which spatial axes are sharded from the state's
   ``NamedSharding.spec`` (falls back to a default mesh-axis PartitionSpec
   when the state is an intermediate with ``UnspecifiedValue`` sharding).
3. Rounds each natural halo width up to the nearest block-size multiple,
   so the padded local shard stays block-divisible.
4. ``jax.experimental.shard_map.shard_map`` wraps the kernel body; inside
   the body it ``jax.lax.ppermute``s a halo of that width on each sharded
   spatial axis (periodic ring), concatenates ``[left_halo, local,
   right_halo]``, calls the ``*_local`` kernel build on the padded
   shard, then strips the halo from each state-shape output.
5. Recurses safely: the helper re-enters with ``mesh=None`` inside the
   body, so the inner ``*_local`` call goes through the no-wrap path
   and just builds the ``pl.pallas_call`` for the local-padded shape.

Halo widths to use:

| Kernel family | Halo per active axis | Block-rounded (for `bx=4`) |
|---|---|---|
| Pointwise leaf op (positivity, EOS conversion) | 0 | 0 (shard_map only, no ppermute) |
| Divergence (`f[i] − f[i-1]`) | 1 | 4 |
| FV reconstruction + Riemann (PLM) | 2 | 4 |
| FV reconstruction (parabolic) | 3 | 4 |
| WENO5 (`q[i-2 .. i+3]`) | 3 | 4 |
| CT modified-flux / edge-EMF | 2 | 4 |
| CT curl (PVA + FD6) | 4 | 4 |

For a per-axis kernel only the active axis needs a non-zero halo (the
kernel reads ``ii``, ``jj``, ``kk`` with no offset along the other
axes).  Off-axis halos cost a bit of extra ppermute traffic but are
correct.  When in doubt, set ``halo_list[axis] = STENCIL_REACH`` and
leave the others at 0.

**What you don't have to change inside the kernel body.**  The modular
indexing ``(ii + offset) % nx`` still does the right thing — inside
``shard_map`` ``nx`` is the *padded local* size, and any wrap is
restricted to halo cells whose output the wrapper strips.  Only the
interior block outputs are correctness-critical.

**Single-device runs are unaffected.**  When
``pallas_mesh_context(mesh)`` is not entered (or ``mesh.size == 1``),
``_pallas_call_sharded`` is a transparent forward to
``kernel_build_fn(*state_inputs, *other_args)`` — same kernel, same
compile cache, same perf.

### 4c. Multi-stage pipeline: split, don't fuse

If a native pipeline computes A → B → C → D where each arrow is a
stencil of radius r, fusing the whole pipeline into one Pallas kernel
makes the effective halo `3r`. With `r=2` (typical WENO/CT spacing)
that pushes halo to 6–8, and Triton's lowering pass grinds to a halt
trying to schedule the closure graph (observed: >5 min stuck on a
single kernel).

Split at every natural intermediate. For CT we now have **three**
bounded-halo Pallas kernels in `_constrained_transport_pallas.py`
(`_ct_modified_flux_pallas`, `_ct_edge_emf_pallas`,
`_ct_curl_pallas`); each has halo ≤ 2 and compiles in well under a
second. The intermediates between kernels are real allocations, but
they're each 1/4 the size of the full state, and the alternative was
a kernel that never finished compiling.

Heuristic: target halo ≤ 4 per kernel. If your native pipeline has
more than two consecutive stencil stages, split.

### 4d. Differentiability: route AD through the native fallback

Pallas kernels in this codebase use ``input_output_aliases={0: 0}`` for
the memory win documented in §4. JAX's autodiff machinery can't transpose
an aliased ``pl.pallas_call`` — under ``jax.grad`` you get
``NotImplementedError: JVP with aliasing not supported``. So **every
Pallas dispatch site must be wrapped in a ``jax.custom_jvp`` whose
tangent rule delegates to the equivalent native-JAX kernel**.

Two helpers in ``astronomix/_pallas_helpers.py`` do the wrapping:

- ``diffable_pallas_call(state, params, *, pallas_branch, native_branch)``
  — the common case: two differentiable primals.
- ``diffable_pallas_call_n(primals, *, pallas_branch, native_branch)``
  — same idea for kernels with more than two diff primals (e.g. an extra
  accumulator / scale arg).

Both helpers:

- run ``pallas_branch(*primals)`` as the primal, so forward simulation
  performance is unchanged (no AD-time work outside ``jax.grad`` / etc.);
- on JVP, compute the primal via Pallas and the tangent via
  ``jax.jvp(native_branch, primals, tangents)``;
- on VJP, JAX transposes the JVP rule — the cotangent flow goes through
  the (transposable) native path with no aliasing.

Two pieces are needed at each dispatch site:

1. A **native fallback function** with the same positional signature
   as the Pallas branch. Often it already exists (the dispatcher's
   existing native arm); otherwise add one next to the Pallas kernel
   (e.g. ``_hydro_flux_div_axis_native`` in ``_ssprk_pallas.py``).
2. A **wrapped call** at the dispatcher level. Close over every static
   arg (``config``, ``registered_variables``, ``axis``, etc.) in the
   ``pallas_branch`` / ``native_branch`` lambdas — only differentiable
   tensors should appear as positional primals.

#### 4d.1 Dispatcher-level pattern (preferred)

```python
# In the native file (or the dispatcher entry point):

from astronomix._pallas_helpers import diffable_pallas_call

@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _flavour_flux_x(state, params, config, registered_variables):
    if _flavour_pallas_supported(state, config):
        pallas = lambda s, p: _flavour_pallas(
            s, p, config, registered_variables, axis=0,
        )
        native = lambda s, p: _flavour_native_x(
            s, p, config, registered_variables,
        )
        return diffable_pallas_call(
            state, params,
            pallas_branch=pallas, native_branch=native,
        )
    return _flavour_native_x(state, params, config, registered_variables)
```

#### 4d.2 Kernel-level pattern (when there is no dispatcher)

For kernels called directly from native integrator code (e.g.
``_hydro_flux_div_axis_pallas`` in ``_ssprk_pallas.py``), wrap inside
the public Pallas function itself so every existing call site picks up
differentiability automatically:

```python
# In _ssprk_pallas.py

def _hydro_flux_div_axis_native(dF, dt_over_dx, *, axis, rhs_accumulator=None, scale_in=1.0):
    """Native-JAX equivalent — used as the tangent branch."""
    div = -dt_over_dx * (dF - _shift(dF, 1, axis=axis + 1))
    if rhs_accumulator is None:
        return div
    return scale_in * rhs_accumulator + div


def _hydro_flux_div_axis_pallas(dF, dt_over_dx, config, *, axis,
                                 rhs_accumulator=None, scale_in=1.0):
    # …existing block/halo setup…

    if rhs_accumulator is None:
        def _pallas_branch(dF_in, dt_in):
            return _pallas_call_sharded(
                lambda d: _hydro_flux_div_axis_pallas_local(
                    d, dt_in, config, axis=axis,
                    rhs_accumulator=None, scale_in=scale_in,
                ),
                state_inputs=(dF_in,), halo=halo,
                block_shape=block_shape[:ndim],
            )
        def _native_branch(dF_in, dt_in):
            return _hydro_flux_div_axis_native(
                dF_in, dt_in, axis=axis,
                rhs_accumulator=None, scale_in=scale_in,
            )
        return diffable_pallas_call(
            dF, dt_over_dx,
            pallas_branch=_pallas_branch, native_branch=_native_branch,
        )

    # Accumulator path: extra diff primals (rhs_accumulator, scale_in).
    scale_in_arr = jnp.asarray(scale_in)
    return diffable_pallas_call_n(
        (dF, dt_over_dx, rhs_accumulator, scale_in_arr),
        pallas_branch=_pallas_branch_acc,
        native_branch=_native_branch_acc,
    )
```

#### 4d.3 Validation

For a new Pallas kernel, validate the gradient route alongside the
primal:

1. ``jax.grad`` of a scalar reduction of the kernel's output, with
   both ``backend=NATIVE_JAX`` and ``backend=PALLAS``. They should
   bit-match (because the JVP rule is the native path and the primal is
   bit-identical per the Pallas memory-table promise).
2. ``tests/sensitivity/sensitivity.py``'s
   ``run_gradient_convergence_test()`` — adds Pallas FD / FV curves
   beside native and checks that the AD gradient still converges to
   the analytical Fourier gradient at the same rate as native.

#### 4d.4 Limitations / when to hand-roll a Pallas adjoint instead

The native-JAX fallback gives correct gradients with zero new kernel
code, but the backward pass runs at native speed. Once a kernel's
backward pass is on the hot path, replace its ``native_branch`` with a
hand-rolled Pallas adjoint kernel paired through ``jax.custom_vjp``
(forward calls the aliased Pallas kernel, backward calls a paired
adjoint kernel). The call-site interface (``diffable_pallas_call`` /
``diffable_pallas_call_n``) is the seam; nothing in the dispatchers
changes when an individual kernel is later upgraded to a hand-rolled
adjoint.

### 5. Wire the dispatcher in the native file

At the **bottom** of the native file (after all native function
definitions, before the public dispatchers), add:

```python
from astronomix.<package>.<subpackage>._<flavour>_pallas import (  # noqa: E402
    _flavour_pallas_supported,
    _flavour_pallas,            # public, shard-aware wrapper
)
```

and update the dispatcher to route through the Pallas kernel when the
predicate accepts:

```python
@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _flavour_flux_x(state, params, config, registered_variables):
    if _flavour_pallas_supported(state, config):
        return _flavour_pallas(state, params, config, registered_variables, axis=0)
    return _flavour_native_x(state, params, config, registered_variables)
```

The dispatcher imports only the **public** ``_flavour_pallas`` entry
point — not the ``_flavour_pallas_local`` body — so it always picks up
the multi-GPU shard-map wrap.

The bottom-of-file import position is important: it lets the
`_pallas` module do a *lazy* import of native fallbacks from the
native file without tripping a circular import (the native names are
already bound in this module's globals by the time `_pallas` is
loaded).

### 6. Validate

Pick the cheapest meaningful regression test for the flavour:

- **FD hydro WENO** → `tests/pallas/sedov3D.py` (Pallas mode, 128³).
- **FD MHD WENO** → `pytests/mhd/alfven_wave3D.py` at N=8 or N=16,
  PALLAS vs NATIVE.
- **FV hydro** → a small periodic-box smoke test (32³ density wave is
  fine).
- **Anything else** → handcraft a 16³ or 32³ smoke test that exercises
  the path.

For *optional* Pallas variants (anything gated by a config flag like
`pallas_ct`) run an A/B/C harness — native baseline, flag off, flag
on — and report compile / warm runtime / temp memory / L1 diff vs
native side-by-side. The standalone scaffold in
`/tmp/compare_pallas_ct.py` is the working template: build three
configs from one base, run each twice (cold+warm) so compile time
falls out of the diff, then print compile/warm/iters/µs-per-iter and
L1 diff vs the baseline. Always reinstall (`python -m pip install .`)
before running — predicates are evaluated at module load and
monkey-patching them in a notebook is silently ineffective.

Acceptance criteria:

- `max|PALLAS − NATIVE|` matches to single-precision rounding (~1e-5
  relative, often much better).  For trivially smooth setups
  (uniform flow with tiny perturbation) demand machine-epsilon match.
- 5th-order or expected convergence rate preserved on a multi-N sweep
  if the native version is high-order.
- Memory analysis (`compiled.memory_analysis()`) shows the expected
  reduction (typically 30–60 %) and no regressions.
- **Multi-GPU strong scaling** — at the largest resolution the
  ``*_local`` kernel was designed for, ``pytests/hydrodynamics/_extended_scaling.py``
  (or the equivalent strong-scaling sweep for the kernel's flavour)
  should hit a 2-GPU speedup of roughly 1.5–1.9× depending on the
  compute/halo ratio.  Anything ≤ 1.0× means the ``_pallas_call_sharded``
  wrap is not being hit — usually a missed dispatcher rewire (still
  calling ``_local`` directly) or a missed ``pallas_mesh_context`` at
  the JIT call site.

### 7. Update the guide if the flavour is new

If you added a new `_pallas` module / kernel that didn't exist before,
add a short subsection in `pallas_backend_implementation_guide.md`
§4 with the headline numbers and any known limitations.

---

## Known limitations / things to never do

- **Don't forget to wrap the dispatch in `diffable_pallas_call` /
  `diffable_pallas_call_n` (§4d).** Without the wrap, any user who
  calls `jax.grad` / `jax.vjp` through `time_integration` with
  `backend=PALLAS` gets `NotImplementedError: JVP with aliasing not
  supported`. The wrap is cheap (one extra ``jax.custom_jvp`` boundary)
  and zero-cost outside AD, so apply it at every Pallas dispatch site
  by default. There is no scenario where Pallas + ``input_output_aliases``
  is differentiable without it.
- **Don't import `_*_pallas.py` symbols from the native file at the top
  of the file** — that re-introduces the circular import the
  bottom-of-file pattern fixes.  Always import Pallas symbols at the
  end of the native module.
- **Don't put `jax.lax.switch` / `jax.lax.cond` inside a Pallas
  kernel for compile-time selection** — Python `if` is what you want
  there.  `jax.lax.switch` should only show up if the decision genuinely
  needs to be runtime-dynamic (which is rare in these kernels).
- **Don't materialise a whole-state JAX array inside a Pallas
  kernel** — `jnp.zeros_like(conserved_state)`,
  `.at[idx].set(...)` chains on full arrays inside the kernel defeat
  the whole point.  Per-tile compute only.
- **Watch out for Python-float literals in `jnp.where` arms.** A bare
  `1.0` / `-1.0` / `1e-20` in the false-arm of `jnp.where` enters the
  Triton lowering as f32 regardless of the surrounding tile dtype, and
  trips a `('f64','f32')` assertion in `_truediv_lowering_rule` when
  the kernel is later run in x64.  Always derive typed scalars from
  an already-typed kernel input (e.g. `gamma`):
  ```python
  zero_typed = gamma - gamma
  one_typed = zero_typed + 1.0
  neg_one_typed = zero_typed - 1.0
  inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
  ```
  Then `jnp.where(Bn >= 0.0, one_typed, neg_one_typed)` is x64-safe.
  Same trick for `b_eps` / `sqrt` floors: pass them as scalar kernel
  args with `jnp.asarray(value, dtype=state.dtype)` rather than using
  bare Python floats inside the kernel.  See
  `pallas_backend_implementation_guide.md` §4.4 for the full
  diagnosis.
- **Don't change the native function's signature when pallasifying** —
  the Pallas kernel mirrors the signature so the dispatcher in the
  native file can call either path interchangeably.
- **Don't call `pl.pallas_call(…)(state, …)` directly from a public
  Pallas entry point.**  Always go through
  ``_pallas_call_sharded(_local, state_inputs=(state, …), halo=…,
  block_shape=…)``.  Bypassing it works on one device but silently
  re-introduces the full-state ``all-gather`` the moment the user
  passes a multi-device ``sharding`` to ``time_integration`` — strong
  scaling drops back to ~0.95× without any error message.
- **Don't capture ``nx``, ``ny``, ``nz``, ``grid`` or ``state.shape``
  in a closure that's reused across calls.**  Inside ``shard_map``
  these have the local halo-padded size, not the global size.  Always
  read them from the *current* ``state.shape`` argument inside the
  ``_local`` body.  If you write ``def _local(s): return pallas_call(…
  shape=state.shape)`` you've captured the outer shape and the kernel
  silently mis-sizes — the symptom is a runtime ``Block shape ... does
  not divide spatial dimension`` from Pallas.
- **Don't fuse a multi-stage pipeline into one Pallas kernel if the
  effective halo exceeds ~4–6.** Triton's lowering pass scales badly
  with closure graph depth × halo. Split at intermediates (see §4c)
  instead — three sub-second kernels beat one that never finishes.
- **Don't pallasify just because it's possible.** If the runtime gain
  on a production-scale benchmark is <5% and the compile-time hit is
  >5 s, either gate it behind a config flag (`pallas_ct` is the
  precedent) or skip the port and rewrite the native op to avoid the
  intermediate buffer that motivated it (`_cfl_time_step_fd_mhd_fast`
  is the precedent — same goal, pure JAX, zero compile cost).

---

## After you finish

Report to the user:

- which native function was pallasified,
- the path to the new / updated `_pallas` module (both the public
  shard-aware wrapper and the ``_local`` body),
- the validation test you ran and the `max|PALLAS − NATIVE|` it
  produced,
- the memory / runtime delta on that test,
- the **multi-GPU strong-scaling speedup** (1 GPU vs 2 GPUs at the
  same problem size).  If you didn't measure it, say so and recommend
  ``pytests/hydrodynamics/_extended_scaling.py`` (or the closest
  flavour equivalent),
- anything you had to leave on the native fallback (and why — usually
  an unsupported limiter or Riemann solver, or an x64 gate).

The user should never need to open the generated `_pallas` files; if
they do, that's a sign this skill missed a translation pattern and
should be improved.
