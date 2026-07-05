"""Pallas backend for ``_enforce_positivity``.

Per-cell pointwise op (no halo) with ``input_output_aliases={0: 0}`` so
the floored state is written back into the input buffer.  Removes one
full-state intermediate per stage on the FD time-integrator hot path.

Mirrors ``_enforce_positivity.py``; the developer never touches this
file by hand.  See ``pallas_backend_implementation_guide.md`` §4.4 for
the pallasify recipe.
"""

# general
import itertools

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_call_sharded,
    _pallas_compiler_params,
    pl,
)


def _enforce_positivity_pallas_supported(state, config: SimulationConfig) -> bool:
    """Pure pointwise op — no halo, no axis branches.  Supported for
    1/2/3D, IDEAL_GAS and ISOTHERMAL, with or without MHD, as long as
    the spatial block divides the spatial dims."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.equation_of_state not in (IDEAL_GAS, ISOTHERMAL):
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _enforce_positivity_pallas(
    conserved_state,
    config: SimulationConfig,
    gamma,
    minimum_density,
    minimum_pressure,
    registered_variables: RegisteredVariables,
):
    """In-place Pallas positivity floor.  Same arithmetic as the native
    ``_enforce_positivity``; the input buffer is donated via the
    Pallas-side ``input_output_aliases={0: 0}`` so XLA reuses it for
    the output (one full-state buffer saved per call).
    """
    assert _enforce_positivity_pallas_supported(conserved_state, config)

    # Multi-GPU: pure pointwise op, so halo=0.  Still route through the
    # shard_map wrapper so XLA does NOT all-gather the state before this
    # opaque pallas_call — the wrapper just runs the kernel locally on
    # each shard.
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)

    def _local(state_local):
        return _enforce_positivity_pallas_local(
            state_local,
            config,
            gamma,
            minimum_density,
            minimum_pressure,
            registered_variables,
        )

    return _pallas_call_sharded(
        _local,
        state_inputs=(conserved_state,),
        halo=(0,) * ndim,
        block_shape=block_shape[:ndim],
    )


def _enforce_positivity_pallas_local(
    conserved_state,
    config: SimulationConfig,
    gamma,
    minimum_density,
    minimum_pressure,
    registered_variables: RegisteredVariables,
):
    """Single-shard kernel build.  Called either directly (single device)
    or once per device from inside ``shard_map`` (multi-device).  The
    pallas_call's ``out_shape`` and ``grid`` are recomputed from the
    *local* ``conserved_state.shape`` so this works for both."""

    is_mhd = config.mhd
    is_ideal = (config.equation_of_state == IDEAL_GAS)
    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    DENSITY = int(registered_variables.density_index)
    if ndim == 1:
        MX = int(registered_variables.momentum_index)
    else:
        MX = int(registered_variables.momentum_index.x)
    if ndim >= 2:
        MY = int(registered_variables.momentum_index.y)
    if ndim == 3:
        MZ = int(registered_variables.momentum_index.z)
    if is_ideal:
        E = int(registered_variables.energy_index)
    if is_mhd:
        BX = int(registered_variables.magnetic_index.x)
        BY = int(registered_variables.magnetic_index.y)
        BZ = int(registered_variables.magnetic_index.z)

    vacuum_rest = bool(config.positivity_config.vacuum_rest)
    if ndim == 1:
        MOM_VARS = (MX,)
    elif ndim == 2:
        MOM_VARS = (MX, MY)
    else:
        MOM_VARS = (MX, MY, MZ)

    if ndim == 1:
        block_shape = (nvars, bx_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx_blk, by_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx_blk, by_blk, bz_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_in_ref, gamma_ref, rhomin_ref, pmin_ref, q_out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        if ndim == 1:
            ii = (bi * bx_blk + jnp.arange(bx_blk)) % nx
        elif ndim == 2:
            ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None]) % nx
            jj = (bj * by_blk + jnp.arange(by_blk)[None, :]) % ny
        else:
            ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
            jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
            kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pmin = pmin_ref[()]

        nan_safe = bool(config.positivity_config.nan_safe)

        def read(var):
            if ndim == 1:
                val = q_in_ref[var, ii]
            elif ndim == 2:
                val = q_in_ref[var, ii, jj]
            else:
                val = q_in_ref[var, ii, jj, kk]
            if nan_safe:
                # Triton has no is_finite; detect NaN via (x != x) and inf via
                # |x| >= ~f32-max, reset both to 0 (matches native nan_to_num).
                finite = (val == val) & (jnp.abs(val) < 3.0e38)
                val = jnp.where(finite, val, 0.0)
            return val

        rho = read(DENSITY)
        rho_floored = jnp.maximum(rho, rhomin)

        # Vacuum-rest: cells below the floor are vacuum -> zero their momentum so
        # the recovered velocity is 0 rather than momentum / rho_floored.
        below = rho < rhomin

        def mom_read(var):
            val = read(var)
            if vacuum_rest:
                val = jnp.where(below, 0.0, val)
            return val

        if is_ideal:
            mx = mom_read(MX)
            v2 = (mx * mx) / (rho_floored * rho_floored)
            if ndim >= 2:
                my = mom_read(MY)
                v2 = v2 + (my * my) / (rho_floored * rho_floored)
            if ndim == 3:
                mz = mom_read(MZ)
                v2 = v2 + (mz * mz) / (rho_floored * rho_floored)
            energy = read(E)
            if is_mhd:
                bxv = read(BX)
                byv = read(BY)
                bzv = read(BZ)
                b2 = bxv * bxv + byv * byv + bzv * bzv
                pressure = gm1 * (energy - 0.5 * rho_floored * v2 - 0.5 * b2)
            else:
                pressure = gm1 * (energy - 0.5 * rho_floored * v2)
            pressure_floored = jnp.maximum(pressure, pmin)
            if is_mhd:
                energy_floored = pressure_floored / gm1 + 0.5 * rho_floored * v2 + 0.5 * b2
            else:
                energy_floored = pressure_floored / gm1 + 0.5 * rho_floored * v2

        # Pass-through every variable, then overwrite the ones we touched.
        for var in range(nvars):
            if var == DENSITY:
                q_out_ref[var, ...] = rho_floored
            elif is_ideal and var == E:
                q_out_ref[var, ...] = energy_floored
            elif vacuum_rest and var in MOM_VARS:
                q_out_ref[var, ...] = mom_read(var)
            else:
                q_out_ref[var, ...] = read(var)

    kwargs = {"input_output_aliases": {0: 0}}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name="enforce_positivity",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(gamma, dtype=conserved_state.dtype),
        jnp.asarray(minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(minimum_pressure, dtype=conserved_state.dtype),
    )


# ---------------------------------------------------------------------------
# Pallas backend for ``_redistribute_positivity`` (3x3x3 neighbour stencil).
# Mirrors the native ``_redistribute_positivity_native``; do not hand-edit.
# Halo = 1 on every active axis (full 3^ndim neighbourhood).
# ---------------------------------------------------------------------------

def _redistribute_positivity_pallas_supported(state, config: SimulationConfig) -> bool:
    """1-cell-halo neighbour stencil. Supported for 1/2/3D, IDEAL_GAS and
    ISOTHERMAL, with/without MHD, block-divisible spatial dims. Gated off under
    x64 (the native fallback is used) to avoid the Triton f64/f32 literal
    caveat — turbulence runs are f32."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.equation_of_state not in (IDEAL_GAS, ISOTHERMAL):
        return False
    if jax.config.jax_enable_x64 and not bool(getattr(config, "pallas_interpret", False)):
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _redistribute_positivity_pallas(
    conserved_state,
    threshold,
    max_velocity,
    gamma,
    minimum_pressure,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Public shard-aware wrapper for the neighbour-redistribution kernel."""
    assert _redistribute_positivity_pallas_supported(conserved_state, config)
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)

    def _local(state_local):
        return _redistribute_positivity_pallas_local(
            state_local, threshold, max_velocity, gamma, minimum_pressure,
            config, registered_variables,
        )

    return _pallas_call_sharded(
        _local,
        state_inputs=(conserved_state,),
        halo=(1,) * ndim,
        block_shape=block_shape[:ndim],
    )


def _redistribute_positivity_pallas_local(
    conserved_state,
    threshold,
    max_velocity,
    gamma,
    minimum_pressure,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Single-shard kernel build; shapes read from ``conserved_state.shape``."""
    is_mhd = config.mhd
    is_ideal = (config.equation_of_state == IDEAL_GAS)
    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    vacuum_rest = bool(config.positivity_config.vacuum_rest)
    DENSITY = int(registered_variables.density_index)
    if ndim == 1:
        MOM = [int(registered_variables.momentum_index)]
    elif ndim == 2:
        MOM = [int(registered_variables.momentum_index.x),
               int(registered_variables.momentum_index.y)]
    else:
        MOM = [int(registered_variables.momentum_index.x),
               int(registered_variables.momentum_index.y),
               int(registered_variables.momentum_index.z)]
    if is_ideal:
        E = int(registered_variables.energy_index)
    if is_mhd:
        BX = int(registered_variables.magnetic_index.x)
        BY = int(registered_variables.magnetic_index.y)
        BZ = int(registered_variables.magnetic_index.z)

    if ndim == 1:
        block_shape = (nvars, bx_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx_blk, by_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx_blk, by_blk, bz_blk)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    offsets = list(itertools.product((-1, 0, 1), repeat=ndim))

    def kernel(q_in_ref, thr_ref, vmax_ref, gamma_ref, pmin_ref, q_out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        if ndim == 1:
            ii = (bi * bx_blk + jnp.arange(bx_blk)) % nx
        elif ndim == 2:
            ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None]) % nx
            jj = (bj * by_blk + jnp.arange(by_blk)[None, :]) % ny
        else:
            ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
            jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
            kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        threshold = thr_ref[()]
        vmax = vmax_ref[()]
        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        pmin = pmin_ref[()]

        def read(var, off):
            if ndim == 1:
                return q_in_ref[var, (ii + off[0]) % nx]
            if ndim == 2:
                return q_in_ref[var, (ii + off[0]) % nx, (jj + off[1]) % ny]
            return q_in_ref[var, (ii + off[0]) % nx, (jj + off[1]) % ny, (kk + off[2]) % nz]

        zero_off = (0,) * ndim
        rho_self = read(DENSITY, zero_off)
        # neighbour sums over valid (rho > threshold) cells
        rho_sum = rho_self * 0.0
        count = rho_self * 0.0
        mom_sum = [rho_self * 0.0 for _ in MOM]
        if is_ideal:
            E_sum = rho_self * 0.0
        for off in offsets:
            rho_n = read(DENSITY, off)
            vf = (rho_n > threshold).astype(rho_self.dtype)
            rho_sum = rho_sum + rho_n * vf
            count = count + vf
            for c, mvar in enumerate(MOM):
                mom_sum[c] = mom_sum[c] + read(mvar, off) * vf
            if is_ideal:
                E_sum = E_sum + read(E, off) * vf

        mom_self = [read(mvar, zero_off) for mvar in MOM]
        is_invalid = rho_self <= threshold
        has = count > 0
        count_safe = jnp.where(has, count, 1.0)
        rho_sum_safe = jnp.where(has, rho_sum, 1.0)
        rho_patched = jnp.where(has, rho_sum / count_safe, threshold)
        mom_patched = []
        for c, ms in enumerate(mom_self):
            # isolated (has=False) deep-void cell: rest it (v=0) under vacuum_rest,
            # else keep v = mom/threshold. See native `_redistribute_positivity_native`
            # for the run-away rationale; this kernel must stay bit-identical to it.
            isolated_v = (ms * 0.0) if vacuum_rest else (ms / threshold)
            v = jnp.where(has, mom_sum[c] / rho_sum_safe, isolated_v)
            v = jnp.clip(v, -vmax, vmax)
            mom_patched.append(rho_patched * v)

        rho_new = jnp.where(is_invalid, rho_patched, rho_self)
        mom_new = [jnp.where(is_invalid, mp, ms) for mp, ms in zip(mom_patched, mom_self)]

        if is_ideal:
            E_self = read(E, zero_off)
            E_patched = jnp.where(has, E_sum / count_safe, E_self)
            E_red = jnp.where(is_invalid, E_patched, E_self)
            # pressure floor (mirrors _enforce_positivity_native_impl on the result)
            rho_f = jnp.maximum(rho_new, threshold)
            v2 = mom_new[0] * mom_new[0] / (rho_f * rho_f)
            if ndim >= 2:
                v2 = v2 + mom_new[1] * mom_new[1] / (rho_f * rho_f)
            if ndim == 3:
                v2 = v2 + mom_new[2] * mom_new[2] / (rho_f * rho_f)
            if is_mhd:
                b2 = (read(BX, zero_off) ** 2 + read(BY, zero_off) ** 2
                      + read(BZ, zero_off) ** 2)
                pressure = gm1 * (E_red - 0.5 * rho_f * v2 - 0.5 * b2)
                pressure = jnp.maximum(pressure, pmin)
                E_new = pressure / gm1 + 0.5 * rho_f * v2 + 0.5 * b2
            else:
                pressure = gm1 * (E_red - 0.5 * rho_f * v2)
                pressure = jnp.maximum(pressure, pmin)
                E_new = pressure / gm1 + 0.5 * rho_f * v2
            rho_new = rho_f

        # pass through every variable, overwrite the ones we touched
        mom_set = {mvar: mom_new[c] for c, mvar in enumerate(MOM)}
        for var in range(nvars):
            if var == DENSITY:
                q_out_ref[var, ...] = rho_new
            elif var in mom_set:
                q_out_ref[var, ...] = mom_set[var]
            elif is_ideal and var == E:
                q_out_ref[var, ...] = E_new
            else:
                q_out_ref[var, ...] = read(var, zero_off)

    kwargs = {"input_output_aliases": {0: 0}}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name="redistribute_positivity",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(threshold, dtype=conserved_state.dtype),
        jnp.asarray(max_velocity, dtype=conserved_state.dtype),
        jnp.asarray(gamma, dtype=conserved_state.dtype),
        jnp.asarray(minimum_pressure, dtype=conserved_state.dtype),
    )
