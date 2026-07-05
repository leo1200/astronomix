"""Pallas backend for the hydro per-axis flux-divergence step.

This file holds the only Pallas kernel that the FD time-integrators in
``_ssprk.py`` need: ``_hydro_flux_div_axis_pallas``.  The integrator
``_ssprk4_hydro`` / ``_ssprk4_with_ct`` / ``_lsrk4_hydro`` in ``_ssprk.py``
all simply import this kernel and call it under the ``_backend_is_pallas``
predicate.  A developer never has to touch this file when writing the
native algorithm.

The kernel is mhd-agnostic — it just walks every variable channel — so
the same Pallas helper covers both the hydro and CT/MHD divergence steps.
Sharing it lets all FD integrators converge on a single buffer (via
``input_output_aliases``).
"""

# typing
from typing import Union

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _pallas_call_sharded,
    _pallas_compiler_params,
    diffable_pallas_call,
    diffable_pallas_call_n,
    pl,
)
from astronomix._stencil_operations._stencil_operations import _shift


def _hydro_flux_div_axis_native(
    dF,
    dt_over_dx,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
):
    """Native-JAX equivalent of :func:`_hydro_flux_div_axis_pallas`.

    Used as the tangent branch by ``diffable_pallas_call`` so that AD through
    the Pallas kernel goes through a transposable JAX expression. Must match
    the Pallas kernel's behaviour bit-for-bit on the primal output for the
    gradient to equal the gradient of the Pallas op at the input.
    """
    div = -dt_over_dx * (dF - _shift(dF, 1, axis=axis + 1))
    if rhs_accumulator is None:
        return div
    return scale_in * rhs_accumulator + div


def _div_axis_pallas_shape_ok(state, config: SimulationConfig) -> bool:
    """Lightweight predicate used by callers (e.g. the MHD CT integrator) that
    want the per-axis divergence Pallas kernel but cannot rely on the full
    hydro-WENO support predicate (which excludes MHD on its WENO step).

    The divergence kernel itself is mhd-agnostic — it just walks every
    variable channel — so this only checks the spatial block-divisibility
    constraint required by ``pl.pallas_call``.
    """
    if pl is None:
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(state.shape[1:], (bx, by, bz)[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _hydro_flux_div_axis_pallas(
    dF,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
):
    """Per-axis Pallas divergence kernel with optional in-place accumulation.

    Computes ``rhs_out = scale_in * (rhs_accumulator if provided else 0) +
    (-dt_over_dx) * (dF[..., i+1/2] - dF[..., (i+1/2)-1])`` along ``axis``.
    Calling it sequentially for each axis with ``rhs_accumulator=rhs_q`` lets
    XLA keep a single physical RHS buffer (via ``input_output_aliases``) across
    all three axes, eliminating both the chained ``rhs + ...`` adds and the
    transient buffers they would otherwise need.

    ``scale_in`` is folded into the kernel so the LSRK4 first-stage update
    ``dq = A[i] * dq + (-dt/dx) * div_0(F_0)`` can be done in place on the
    ``dq`` buffer without materialising a separate ``rhs`` register.

    This keeps the original 1-flux-per-cell WENO kernel (so peak compute is
    unchanged) while still consuming each ``dF_axis`` immediately after it is
    produced, instead of holding all three live for the original
    three-input divergence helper.
    """
    # Multi-GPU: divergence reads ``dF[i] - dF[i-1]`` along ``axis``, so the
    # only halo needed is 1 cell on the active axis (rounded up to the
    # Pallas block size by ``_pallas_call_sharded``).  The accumulator is
    # read at the local cell only — no halo — but we pass it through the
    # same wrapper to keep the input_output_aliases trick intact inside
    # each shard.
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    halo_list = [0, 0, 0]
    if 0 <= axis < ndim:
        halo_list[axis] = 1
    halo = tuple(halo_list[:ndim])

    if rhs_accumulator is None:
        def _local(dF_local):
            return _hydro_flux_div_axis_pallas_local(
                dF_local,
                dt_over_dx,
                config,
                axis=axis,
                rhs_accumulator=None,
                scale_in=scale_in,
            )

        def _pallas_branch(dF_in, dt_over_dx_in):
            return _pallas_call_sharded(
                lambda d: _hydro_flux_div_axis_pallas_local(
                    d, dt_over_dx_in, config,
                    axis=axis, rhs_accumulator=None, scale_in=scale_in,
                ),
                state_inputs=(dF_in,),
                halo=halo,
                block_shape=block_shape[:ndim],
            )

        def _native_branch(dF_in, dt_over_dx_in):
            return _hydro_flux_div_axis_native(
                dF_in, dt_over_dx_in, axis=axis,
                rhs_accumulator=None, scale_in=scale_in,
            )

        return diffable_pallas_call(
            dF, dt_over_dx,
            pallas_branch=_pallas_branch, native_branch=_native_branch,
        )

    def _pallas_branch_acc(dF_in, dt_over_dx_in, rhs_in, scale_in_arr):
        return _pallas_call_sharded(
            lambda r, d: _hydro_flux_div_axis_pallas_local(
                d, dt_over_dx_in, config,
                axis=axis, rhs_accumulator=r, scale_in=scale_in_arr,
            ),
            state_inputs=(rhs_in, dF_in),
            halo=halo,
            block_shape=block_shape[:ndim],
        )

    def _native_branch_acc(dF_in, dt_over_dx_in, rhs_in, scale_in_arr):
        return _hydro_flux_div_axis_native(
            dF_in, dt_over_dx_in, axis=axis,
            rhs_accumulator=rhs_in, scale_in=scale_in_arr,
        )

    # ``scale_in`` may be a traced scalar (LSRK4 stage coefficient) so route
    # it through the diffable primals tuple too.
    scale_in_arr = jnp.asarray(scale_in)
    return diffable_pallas_call_n(
        (dF, dt_over_dx, rhs_accumulator, scale_in_arr),
        pallas_branch=_pallas_branch_acc,
        native_branch=_native_branch_acc,
    )


def _hydro_flux_div_axis_pallas_local(
    dF,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
):
    """Single-shard ``pl.pallas_call`` build.  Called either directly or
    inside a ``shard_map`` body; ``dF.shape`` is the *local* (halo-padded)
    shape in the multi-device case so the kernel's grid/in-spec/out-spec
    re-derive automatically."""
    ndim = int(config.dimensionality)
    nvars = int(dF.shape[0])
    spatial_shape = tuple(int(x) for x in dF.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    accumulate = rhs_accumulator is not None

    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        flux_spec = pl.BlockSpec(dF.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        flux_spec = pl.BlockSpec(dF.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        flux_spec = pl.BlockSpec(dF.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(*refs):
        if accumulate:
            rhs_in_ref, f_ref, dtdx_ref, scale_in_ref, rhs_out_ref = refs
        else:
            f_ref, dtdx_ref, rhs_out_ref = refs
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        if ndim == 1:
            ii = (bi * bx + jnp.arange(bx)) % nx
        elif ndim == 2:
            ii = (bi * bx + jnp.arange(bx)[:, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :]) % ny
        else:
            ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
            jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
            kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        dtdx = dtdx_ref[()]

        def flux_diff(var):
            if axis == 0:
                if ndim == 1:
                    return f_ref[var, ii] - f_ref[var, (ii - 1) % nx]
                if ndim == 2:
                    return f_ref[var, ii, jj] - f_ref[var, (ii - 1) % nx, jj]
                return f_ref[var, ii, jj, kk] - f_ref[var, (ii - 1) % nx, jj, kk]
            if axis == 1:
                if ndim == 2:
                    return f_ref[var, ii, jj] - f_ref[var, ii, (jj - 1) % ny]
                return f_ref[var, ii, jj, kk] - f_ref[var, ii, (jj - 1) % ny, kk]
            return f_ref[var, ii, jj, kk] - f_ref[var, ii, jj, (kk - 1) % nz]

        if accumulate:
            scale = scale_in_ref[()]
            for var in range(nvars):
                rhs_out_ref[var, ...] = scale * rhs_in_ref[var, ...] + (-dtdx) * flux_diff(var)
        else:
            for var in range(nvars):
                rhs_out_ref[var, ...] = -dtdx * flux_diff(var)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    if accumulate:
        in_specs = [out_spec, flux_spec, scalar_spec, scalar_spec]
        kernel_args = (
            rhs_accumulator,
            dF,
            jnp.asarray(dt_over_dx, dtype=dF.dtype),
            jnp.asarray(scale_in, dtype=dF.dtype),
        )
        kwargs["input_output_aliases"] = {0: 0}
    else:
        in_specs = [flux_spec, scalar_spec]
        kernel_args = (dF, jnp.asarray(dt_over_dx, dtype=dF.dtype))

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(dF.shape, dF.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"hydro_flux_div_axis_{axis}{'_acc' if accumulate else ''}",
        **kwargs,
    )(*kernel_args)
