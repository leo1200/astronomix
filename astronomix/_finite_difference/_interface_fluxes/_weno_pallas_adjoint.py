"""Pallas adjoint (reverse-mode) kernels for the ideal-gas hydro WENO flux.

Used by the ``PALLAS_AD_VJP_PALLAS`` AD mode (``config.pallas_ad_mode``): the
forward sweep under ``jax.grad`` runs the existing aliased Pallas kernels, and
the backward sweep for the WENO interface flux runs the two kernels in this
module instead of recomputing the native-JAX flux and transposing it.

Design (see pallas_backend_implementation_guide.md, AD section):

The flux at face ``j`` is a pure per-face function of the six stencil cells
``q_{j-2..j+3}`` plus three scalars (gamma, minimum_density,
minimum_pressure).  The adjoint w.r.t. the state is therefore a gather over
the six faces that read each cell.  To keep the Triton closure graph bounded
(one forward + one backward trace of the WENO face math, not six) the adjoint
is split into two bounded-halo kernels:

1. **Face kernel** — for every face ``j``, run ``jax.vjp`` of the per-face
   flux closure *inside the kernel trace* against the incoming flux
   cotangent ``ct_j``.  This emits, per face, the cotangent w.r.t. each of
   its six stencil cells (``6*ncomp`` channels) plus per-face cotangents for
   the three scalars (3 channels).  Recompute happens at register level —
   no forward-sweep residuals are stored anywhere.
2. **Gather kernel** — the purely linear scatter→gather flip:
   ``d_state[c, i] = sum_s buf[s*ncomp + c, i + 2 - s]`` (the contribution
   of face ``i + 2 - s`` reading cell ``i`` at stencil slot ``s``).

The per-face flux closure ``_hydro_face_flux`` mirrors the kernel body of
``_weno_flux_hydro_pallas_local`` in ``_weno_pallas.py`` — both files are
generated artifacts of the pallasify skill; regenerate them together when the
native WENO math changes.
"""

import jax
import jax.numpy as jnp
import numpy as np

from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _pallas_call_sharded,
    _pallas_compiler_params,
    pl,
)
from astronomix.option_classes.simulation_config import (
    PALLAS_AD_VJP_PALLAS,
    SimulationConfig,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables


def _hydro_face_flux(q_stencil, gamma, rhomin, pgmin, *, ncomp, num_modes,
                     epsilon=1e-7, tiny=1e-14):
    """Per-face ideal-gas WENO5 flux as a pure function of the local stencil.

    ``q_stencil`` is a 6-tuple (offsets -2..+3 along the active axis) of
    ncomp-tuples of tiles in the local characteristic component order
    (density, normal momentum, transverse momenta, energy).  ``gamma`` /
    ``rhomin`` / ``pgmin`` are tiles (broadcast scalars) so that ``jax.vjp``
    of this function yields per-cell cotangents for them.

    Mirrors the kernel body of ``_weno_flux_hydro_pallas_local`` exactly —
    keep the two in sync (both are pallasify-generated).
    """
    gm1 = gamma - 1.0

    def primitive_from_q(q):
        rho = q[0]
        mn = q[1]
        if ncomp == 3:
            mt1 = 0.0
            mt2 = 0.0
            energy = q[2]
        elif ncomp == 4:
            mt1 = q[2]
            mt2 = 0.0
            energy = q[3]
        else:
            mt1 = q[2]
            mt2 = q[3]
            energy = q[4]

        inv_rho = 1.0 / rho
        vn = mn * inv_rho
        vt1 = mt1 * inv_rho
        vt2 = mt2 * inv_rho
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        pressure = gm1 * (energy - 0.5 * rho * v2)
        return rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure

    def floored_cell(q):
        rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
        troubled = (rho < rhomin) | (pressure < pgmin)
        rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
        pressure_f = jnp.where(troubled, jnp.maximum(pressure, pgmin), pressure)
        energy_f = jnp.where(troubled, pressure_f / gm1 + 0.5 * rho_f * v2, energy)
        specific_enthalpy = (energy_f + pressure_f) / rho_f
        sound_speed = jnp.sqrt(jnp.maximum(gamma * jnp.abs(pressure_f / rho_f), 1e-12))
        return rho_f, mn, mt1, mt2, energy_f, vn, vt1, vt2, v2, pressure_f, specific_enthalpy, sound_speed

    def flux_from_q(q):
        rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
        if ncomp == 3:
            return (mn, mn * vn + pressure, (energy + pressure) * vn)
        if ncomp == 4:
            return (mn, mn * vn + pressure, mt1 * vn, (energy + pressure) * vn)
        return (mn, mn * vn + pressure, mt1 * vn, mt2 * vn, (energy + pressure) * vn)

    f_stencil = tuple(flux_from_q(q) for q in q_stencil)
    floored_stencil = tuple(floored_cell(q) for q in q_stencil)

    cell_l = floored_stencil[2]
    cell_r = floored_stencil[3]
    rho_i, mn_i, mt1_i, mt2_i, energy_i, vn_i, vt1_i, vt2_i, v2_i, p_i, h_i, c_i = cell_l
    rho_j, mn_j, mt1_j, mt2_j, energy_j, vn_j, vt1_j, vt2_j, v2_j, p_j, h_j, c_j = cell_r
    rho_face = jnp.maximum(0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)), rhomin)
    vn_face = 0.5 * (mn_i + mn_j) / rho_face
    vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
    vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
    h_face = 0.5 * (h_i + h_j)
    v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
    c2_face = gm1 * (h_face - 0.5 * v2_face)
    c_face = jnp.sqrt(jnp.maximum(c2_face, 1e-12))
    inv_c2 = jnp.where(c2_face > 0.0, 1.0 / c2_face, 0.0)

    def left_project(mode: int, values):
        if mode == 0:
            acc = (0.5 * gm1 * v2_face + vn_face * c_face) * values[0]
            acc = acc - (gm1 * vn_face + c_face) * values[1]
            if ncomp == 3:
                acc = acc + gm1 * values[2]
            elif ncomp == 4:
                acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
            else:
                acc = (
                    acc
                    - gm1 * vt1_face * values[2]
                    - gm1 * vt2_face * values[3]
                    + gm1 * values[4]
                )
            return 0.5 * inv_c2 * acc

        if mode == 1:
            acc = (c2_face - 0.5 * gm1 * v2_face) * values[0]
            acc = acc + gm1 * vn_face * values[1]
            if ncomp == 3:
                acc = acc - gm1 * values[2]
            elif ncomp == 4:
                acc = acc + gm1 * vt1_face * values[2] - gm1 * values[3]
            else:
                acc = (
                    acc
                    + gm1 * vt1_face * values[2]
                    + gm1 * vt2_face * values[3]
                    - gm1 * values[4]
                )
            return inv_c2 * acc

        if mode == 2 and ncomp >= 4:
            return -vt1_face * values[0] + values[2]

        if mode == 3 and ncomp == 5:
            return -vt2_face * values[0] + values[3]

        acc = (0.5 * gm1 * v2_face - vn_face * c_face) * values[0]
        acc = acc - (gm1 * vn_face - c_face) * values[1]
        if ncomp == 3:
            acc = acc + gm1 * values[2]
        elif ncomp == 4:
            acc = acc - gm1 * vt1_face * values[2] + gm1 * values[3]
        else:
            acc = (
                acc
                - gm1 * vt1_face * values[2]
                - gm1 * vt2_face * values[3]
                + gm1 * values[4]
            )
        return 0.5 * inv_c2 * acc

    def add_right_correction(flux_acc, mode: int, Fs):
        if mode == 0:
            if ncomp == 3:
                R = (1.0, vn_face - c_face, h_face - vn_face * c_face)
            elif ncomp == 4:
                R = (1.0, vn_face - c_face, vt1_face, h_face - vn_face * c_face)
            else:
                R = (1.0, vn_face - c_face, vt1_face, vt2_face, h_face - vn_face * c_face)
        elif mode == 1:
            if ncomp == 3:
                R = (1.0, vn_face, 0.5 * v2_face)
            elif ncomp == 4:
                R = (1.0, vn_face, vt1_face, 0.5 * v2_face)
            else:
                R = (1.0, vn_face, vt1_face, vt2_face, 0.5 * v2_face)
        elif mode == 2 and ncomp >= 4:
            if ncomp == 4:
                R = (0.0, 0.0, 1.0, vt1_face)
            else:
                R = (0.0, 0.0, 1.0, 0.0, vt1_face)
        elif mode == 3 and ncomp == 5:
            R = (0.0, 0.0, 0.0, 1.0, vt2_face)
        else:
            if ncomp == 3:
                R = (1.0, vn_face + c_face, h_face + vn_face * c_face)
            elif ncomp == 4:
                R = (1.0, vn_face + c_face, vt1_face, h_face + vn_face * c_face)
            else:
                R = (1.0, vn_face + c_face, vt1_face, vt2_face, h_face + vn_face * c_face)
        return [flux_acc[slot] + R[slot] * Fs for slot in range(ncomp)]

    def lambda_from_floored_cell(cell, mode: int):
        vn = cell[5]
        c = cell[11]
        if mode == 0:
            return vn - c
        if mode == num_modes - 1:
            return vn + c
        return vn

    def alpha_for_mode(mode: int):
        amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
        for k in range(1, 6):
            amx = jnp.maximum(
                amx,
                jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode)),
            )
        return amx

    flux_acc = [
        (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot] + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) / 12.0
        for slot in range(ncomp)
    ]

    for mode in range(num_modes):
        s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
        qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

        d0 = s[1] - s[0]
        d1 = s[2] - s[1]
        d2 = s[3] - s[2]
        d3 = s[4] - s[3]
        d4 = s[5] - s[4]

        dq0 = qproj[1] - qproj[0]
        dq1 = qproj[2] - qproj[1]
        dq2 = qproj[3] - qproj[2]
        dq3 = qproj[4] - qproj[3]
        dq4 = qproj[5] - qproj[4]

        amx = alpha_for_mode(mode)

        aterm_p = 0.5 * (d0 + amx * dq0)
        bterm_p = 0.5 * (d1 + amx * dq1)
        cterm_p = 0.5 * (d2 + amx * dq2)
        dterm_p = 0.5 * (d3 + amx * dq3)

        IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
        IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
        IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
        alpha0_p = 1.0 / (epsilon + IS0_p) ** 2
        alpha1_p = 6.0 / (epsilon + IS1_p) ** 2
        alpha2_p = 3.0 / (epsilon + IS2_p) ** 2
        alpha_sum_p = jnp.maximum(alpha0_p + alpha1_p + alpha2_p, tiny)
        omega0_p = alpha0_p / alpha_sum_p
        omega2_p = alpha2_p / alpha_sum_p
        second = (
            omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) / 3.0
            + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) / 6.0
        )

        aterm_m = 0.5 * (d4 - amx * dq4)
        bterm_m = 0.5 * (d3 - amx * dq3)
        cterm_m = 0.5 * (d2 - amx * dq2)
        dterm_m = 0.5 * (d1 - amx * dq1)

        IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
        IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
        IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
        alpha0_m = 1.0 / (epsilon + IS0_m) ** 2
        alpha1_m = 6.0 / (epsilon + IS1_m) ** 2
        alpha2_m = 3.0 / (epsilon + IS2_m) ** 2
        alpha_sum_m = jnp.maximum(alpha0_m + alpha1_m + alpha2_m, tiny)
        omega0_m = alpha0_m / alpha_sum_m
        omega2_m = alpha2_m / alpha_sum_m
        third = (
            omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) / 3.0
            + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) / 6.0
        )

        Fs = -second + third
        flux_acc = add_right_correction(flux_acc, mode, Fs)

    return tuple(flux_acc)


def _weno_hydro_adjoint_face_pallas_local(
    conserved_state,
    flux_cotangent,
    gamma,
    rhomin,
    pgmin,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Per-face adjoint kernel build (single shard).

    Output channels: ``s * ncomp + c`` for stencil slot ``s`` (offset
    ``s - 2``) and local component ``c`` — the cotangent of the flux at this
    face w.r.t. stencil cell ``s`` — followed by three per-face scalar
    cotangent channels (gamma, minimum_density, minimum_pressure).
    """
    from astronomix._finite_difference._interface_fluxes._weno_pallas import (
        _hydro_indices_for_axis,
    )

    ndim = int(config.dimensionality)
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 2
    nchan = 6 * ncomp + 3

    if ndim == 1:
        out_block = (nchan, bx)
        out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        out_block = (nchan, bx, by)
        out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        out_block = (nchan, bx, by, bz)
        out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, ct_ref, gamma_ref, rhomin_ref, pgmin_ref, out_ref):
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

        gamma_s = gamma_ref[()]
        rhomin_s = rhomin_ref[()]
        pgmin_s = pgmin_ref[()]

        def read_at(ref, var_index: int, offset: int):
            if ndim == 1:
                return ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return ref[var_index, (ii + offset) % nx, jj]
                return ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return ref[var_index, ii, (jj + offset) % ny, kk]
            return ref[var_index, ii, jj, (kk + offset) % nz]

        q_stencil = tuple(
            tuple(read_at(q_ref, idx, off) for idx in local_indices)
            for off in range(-2, 4)
        )
        ct_local = tuple(read_at(ct_ref, idx, 0) for idx in local_indices)

        # Broadcast the scalars to tiles so the vjp emits per-face scalar
        # cotangents (summed outside the kernel) instead of in-kernel
        # cross-program reductions.
        zero_tile = q_stencil[2][0] * 0.0
        gamma_tile = gamma_s + zero_tile
        rhomin_tile = rhomin_s + zero_tile
        pgmin_tile = pgmin_s + zero_tile

        def face_flux(qs, g, rm, pm):
            return _hydro_face_flux(qs, g, rm, pm, ncomp=ncomp, num_modes=num_modes)

        _, vjp_fn = jax.vjp(face_flux, q_stencil, gamma_tile, rhomin_tile, pgmin_tile)
        dq_stencil, dgamma, drhomin, dpgmin = vjp_fn(ct_local)

        for s in range(6):
            for c in range(ncomp):
                out_ref[s * ncomp + c, ...] = dq_stencil[s][c]
        out_ref[6 * ncomp + 0, ...] = dgamma
        out_ref[6 * ncomp + 1, ...] = drhomin
        out_ref[6 * ncomp + 2, ...] = dpgmin

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = (nchan,) + spatial_shape
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, in_state_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"hydro_weno_adjoint_face_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        flux_cotangent,
        jnp.asarray(gamma, dtype=conserved_state.dtype),
        jnp.asarray(rhomin, dtype=conserved_state.dtype),
        jnp.asarray(pgmin, dtype=conserved_state.dtype),
    )


def _weno_hydro_adjoint_gather_pallas_local(
    face_buf,
    nvars: int,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Gather kernel build (single shard): flip the per-face scatter into a
    per-cell gather, ``d_state[c, i] = sum_s buf[s*ncomp + c, i + 2 - s]``."""
    from astronomix._finite_difference._interface_fluxes._weno_pallas import (
        _hydro_indices_for_axis,
    )

    ndim = int(config.dimensionality)
    spatial_shape = tuple(int(x) for x in face_buf.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)

    if ndim == 1:
        out_block = (nvars, bx)
        out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (0, bi))
        in_buf_spec = pl.BlockSpec(face_buf.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        out_block = (nvars, bx, by)
        out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (0, bi, bj))
        in_buf_spec = pl.BlockSpec(face_buf.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        out_block = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (0, bi, bj, bk))
        in_buf_spec = pl.BlockSpec(face_buf.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    def kernel(buf_ref, out_ref):
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

        def buf_at(chan: int, offset: int):
            if ndim == 1:
                return buf_ref[chan, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return buf_ref[chan, (ii + offset) % nx, jj]
                return buf_ref[chan, ii, (jj + offset) % ny]
            if axis == 0:
                return buf_ref[chan, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return buf_ref[chan, ii, (jj + offset) % ny, kk]
            return buf_ref[chan, ii, jj, (kk + offset) % nz]

        zero = buf_at(0, 0) * 0.0
        for var in range(nvars):
            out_ref[var, ...] = zero
        for c, var in enumerate(local_indices):
            acc = buf_at(0 * ncomp + c, 2)
            for s in range(1, 6):
                acc = acc + buf_at(s * ncomp + c, 2 - s)
            out_ref[var, ...] = acc

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = (nvars,) + spatial_shape
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(out_shape, face_buf.dtype),
        grid=grid,
        in_specs=[in_buf_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"hydro_weno_adjoint_gather_axis_{axis}",
        **kwargs,
    )(face_buf)


def _weno_flux_hydro_pallas_adjoint(
    conserved_state,
    flux_cotangent,
    params,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Public adjoint entry point (shard-aware).

    Returns ``(d_state, d_gamma, d_minimum_density, d_minimum_pressure)``
    for the WENO hydro flux along ``axis``, given the cotangent of the flux
    output.
    """
    from astronomix._finite_difference._interface_fluxes._weno_pallas import (
        _hydro_indices_for_axis,
    )

    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    ncomp = len(_hydro_indices_for_axis(config, registered_variables, axis))

    halo_list = [0, 0, 0]
    if 0 <= int(axis) < ndim:
        halo_list[int(axis)] = 3
    halo = tuple(halo_list[:ndim])

    def _face_local(state_local, ct_local):
        return _weno_hydro_adjoint_face_pallas_local(
            state_local, ct_local,
            params.gamma, params.minimum_density, params.minimum_pressure,
            config, registered_variables, axis=axis,
        )

    face_buf = _pallas_call_sharded(
        _face_local,
        state_inputs=(conserved_state, flux_cotangent),
        halo=halo,
        block_shape=block_shape[:ndim],
    )

    def _gather_local(buf_local):
        return _weno_hydro_adjoint_gather_pallas_local(
            buf_local, nvars, config, registered_variables, axis=axis,
        )

    d_state = _pallas_call_sharded(
        _gather_local,
        state_inputs=(face_buf,),
        halo=halo,
        block_shape=block_shape[:ndim],
    )

    d_gamma = jnp.sum(face_buf[6 * ncomp + 0])
    d_rhomin = jnp.sum(face_buf[6 * ncomp + 1])
    d_pgmin = jnp.sum(face_buf[6 * ncomp + 2])
    return d_state, d_gamma, d_rhomin, d_pgmin


def _zero_cotangent_like(x):
    arr = jnp.asarray(x)
    if jnp.issubdtype(arr.dtype, jnp.inexact):
        return jnp.zeros_like(arr)
    return np.zeros(arr.shape, dtype=jax.dtypes.float0)


def _weno_flux_hydro_pallas_adjoint_branch(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    axis: int,
):
    """Build the ``adjoint_branch`` for ``diffable_pallas_call`` at the hydro
    WENO dispatch site, or return None when the Pallas adjoint should not be
    used (mode is not VJP_PALLAS, or Pallas is unavailable)."""
    if pl is None:
        return None
    if int(config.pallas_ad_mode) != PALLAS_AD_VJP_PALLAS:
        return None

    def adjoint(primal_args, cotangent):
        conserved_state, params = primal_args
        d_state, d_gamma, d_rhomin, d_pgmin = _weno_flux_hydro_pallas_adjoint(
            conserved_state, cotangent, params, config, registered_variables,
            axis=axis,
        )
        d_params = jax.tree_util.tree_map(_zero_cotangent_like, params)
        d_params = d_params._replace(
            gamma=jnp.asarray(d_gamma, dtype=jnp.result_type(params.gamma)),
            minimum_density=jnp.asarray(
                d_rhomin, dtype=jnp.result_type(params.minimum_density)
            ),
            minimum_pressure=jnp.asarray(
                d_pgmin, dtype=jnp.result_type(params.minimum_pressure)
            ),
        )
        return (d_state, d_params)

    return adjoint
