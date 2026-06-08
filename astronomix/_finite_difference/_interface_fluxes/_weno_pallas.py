"""Pallas implementations of the 5th-order WENO interface flux.

This file is the **Pallas backend** for the WENO interface-flux step.  All
native-JAX implementations live in ``_weno.py``; the dispatchers there
import from this file and call into it when ``config.backend == PALLAS``
and the per-flavour ``_*_pallas_flux_supported`` predicate accepts.

A developer who only writes / modifies native JAX never needs to touch
this file.  See ``pallas_backend_implementation_guide.md`` (§2 for the
kernel skeleton, §4 for the per-flavour porting recipe) and the
``.claude/skills/pallasify`` skill, which mechanically translates a
native-JAX stencil function into the matching Pallas kernel that lives
here.

Currently covers:
- ideal-gas hydrodynamic WENO (``_weno_flux_hydro_pallas``)
- ideal-gas MHD WENO (``_weno_flux_mhd_pallas``)
- isothermal MHD WENO (``_weno_flux_mhd_iso_pallas``)
- a fused (no-rhs-buffer) variant for LSRK4 (``_weno_flux_hydro_pallas_rhs``)

The shared block-shape / compiler-params helpers live in
``astronomix._pallas_helpers``.
"""

import jax
import jax.numpy as jnp

from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _default_pallas_block_shape,
    _pallas_call_sharded,
    _pallas_compiler_params,
    pl,
    pltriton,
)


def _weno5_shard_wrap(kernel_local, conserved_state, config, axis):
    """Multi-GPU wrap for a per-axis 5th-order WENO Pallas kernel.

    The WENO5 stencil reads offsets ``-2..+3`` along the *active* axis only —
    so halo of 3 cells on that axis is enough.  Off-axis the kernel reads
    only its own cell index (``ii``, ``jj``, or ``kk``), so no halo is
    needed there even if those axes are sharded.

    Every WENO kernel here (hydro / MHD / iso-MHD / hydro_rhs) shares the
    same per-axis stencil reach, so all of them funnel through this single
    helper.  When ``pallas_mesh_context`` is not active the helper just
    forwards to ``kernel_local`` — single-device runs are unaffected.
    """
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    halo_list = [0, 0, 0]
    if 0 <= int(axis) < ndim:
        halo_list[int(axis)] = 3
    halo = tuple(halo_list[:ndim])

    def _call(state_local):
        return kernel_local(state_local)

    return _pallas_call_sharded(
        _call,
        state_inputs=(conserved_state,),
        halo=halo,
        block_shape=block_shape[:ndim],
    )
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL, PALLAS, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables


def _backend_name(config: SimulationConfig) -> str:
    """Return a robust string representation of config.backend.

    This intentionally does not import PALLAS/NATIVE_JAX constants.  It works with
    string constants, enum values, or small dataclass-like constant objects whose
    ``name`` or ``value`` carries the backend name.
    """
    backend = config.backend
    name = getattr(backend, "name", None)
    if name is not None:
        return str(name).upper()
    value = getattr(backend, "value", None)
    if isinstance(value, str):
        return value.upper()
    return str(backend).upper()


def _hydro_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """Whether the existing Pallas hydro WENO kernel can be used.

    Currently handles ideal-gas hydro only (ncomp = ndim+2, num_modes =
    ndim+2).  Isothermal hydro and MHD (ideal or isothermal) fall back to
    the native-JAX implementations.  See
    ``pallas_backend_implementation_guide.md`` §4 for the porting recipe
    for those variants.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.mhd:
        return False  # MHD WENO kernel not yet ported to Pallas (see guide §4.1)
    if config.equation_of_state != IDEAL_GAS:
        return False  # Isothermal hydro WENO Pallas kernel not yet ported (see guide §4.2)
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if conserved_state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    spatial_shape = conserved_state.shape[1:]
    for n, b in zip(spatial_shape, block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _hydro_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Return local Euler component indices for a flux normal to ``axis``.

    The returned order is the local characteristic order used by the Euler
    eigenvectors: density, normal momentum, first transverse momentum, optional
    second transverse momentum, energy.  The indices themselves refer to the
    original conserved-state component axis.
    """
    density_index = int(registered_variables.density_index)
    energy_index = int(registered_variables.energy_index)
    ndim = int(config.dimensionality)

    if ndim == 1:
        momentum_x = int(registered_variables.momentum_index)
        return (density_index, momentum_x, energy_index)

    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    if ndim == 2:
        if axis == 0:
            return (density_index, mx, my, energy_index)
        return (density_index, my, mx, energy_index)

    mz = int(registered_variables.momentum_index.z)
    if axis == 0:
        return (density_index, mx, my, mz, energy_index)
    if axis == 1:
        return (density_index, my, mx, mz, energy_index)
    return (density_index, mz, my, mx, energy_index)


def _weno_flux_hydro_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Pallas implementation of the ideal-gas hydrodynamic WENO flux.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  The arithmetic lives in
    ``_weno_flux_hydro_pallas_local`` so the same kernel build runs on
    either the global state (single device) or a local halo-padded shard
    (multi device) without changes.
    """
    if not _hydro_pallas_flux_supported(conserved_state, config):
        # Lazy import to break the circular dependency with _weno.py.
        from astronomix._finite_difference._interface_fluxes._weno import (
            _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
        )
        if axis == 0:
            return _weno_flux_x_native(conserved_state, params, config, registered_variables)
        if axis == 1:
            return _weno_flux_y_native(conserved_state, params, config, registered_variables)
        return _weno_flux_z_native(conserved_state, params, config, registered_variables)

    def _local(state_local):
        return _weno_flux_hydro_pallas_local(
            state_local, params, config, registered_variables, axis=axis
        )
    return _weno5_shard_wrap(_local, conserved_state, config, axis)


def _weno_flux_hydro_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Single-shard hydro-WENO kernel build.  When called from inside a
    ``shard_map`` body, ``conserved_state.shape`` is the local halo-padded
    shape and the kernel's grid / modular indexing wrap within that shape.
    Outside ``shard_map`` (single-device path) the shape is global."""
    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 2
    epsilon = 1e-7
    tiny = 1e-14

    # Output block specs keep the conserved-variable axis complete and block only
    # the spatial dimensions.
    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, gamma_ref, rhomin_ref, pgmin_ref, flux_out_ref):
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

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]

        def q_at(var_index: int, offset: int):
            if ndim == 1:
                return q_ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return q_ref[var_index, (ii + offset) % nx, jj]
                return q_ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            return tuple(q_at(idx, offset) for idx in local_indices)

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

        qm2 = q_local(-2)
        qm1 = q_local(-1)
        q0 = q_local(0)
        qp1 = q_local(1)
        qp2 = q_local(2)
        qp3 = q_local(3)
        q_stencil = (qm2, qm1, q0, qp1, qp2, qp3)
        f_stencil = tuple(flux_from_q(q) for q in q_stencil)

        # Compute floored primitive/eigenvalue data once for the six cells used
        # by the local Lax-Friedrichs alpha.  The earlier version recomputed this
        # data inside every characteristic mode; keeping it local here avoids both
        # global eigenvalue arrays and repeated per-mode work.
        floored_stencil = tuple(floored_cell(q) for q in q_stencil)

        # Interface eigenvector building blocks at i + 1/2, following
        # _eigenvector_building_blocks in _eigen_hydro.py.
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
            """Project one local vector onto one Euler left eigenvector.

            This is the local Pallas replacement for materialising
            ``_eigen_L_row_hydro(..., mode)`` followed by a full-array einsum.
            ``values`` is either a local conserved-state vector or a local flux
            vector in the normal/tangential component order used by this axis.
            """
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

            # Right acoustic wave.
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
            """Add Fs times one local Euler right eigenvector to flux_acc.

            This is the local Pallas replacement for materialising
            ``_eigen_R_col_hydro(..., mode)`` followed by an outer-product style
            einsum.  Returning a Python list keeps the component axis static and
            avoids building small dense eigenvector arrays inside the kernel.
            """
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

        # Set every output component.  Hydro should fill all components, but the
        # explicit zeroing makes failures obvious if a future registry adds fields.
        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"hydro_weno_flux_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.gamma, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
    )


# -----------------------------------------------------------------------------
# Pallas WENO for the ideal-gas MHD equations.
# -----------------------------------------------------------------------------


def _mhd_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """Whether the Pallas MHD ideal-gas WENO kernel can be used."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != IDEAL_GAS:
        return False  # isothermal MHD WENO Pallas kernel still TODO (guide §4.2)
    ndim = int(config.dimensionality)
    if ndim != 3:  # MHD WENO is 3D-only in this codebase
        return False
    if conserved_state.ndim != 4:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(conserved_state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _mhd_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Local conserved-variable order used by the MHD eigenvectors for a flux
    normal to ``axis``: (density, p_normal, p_trans1, p_trans2, B_normal,
    B_trans1, B_trans2, energy).  Returns the 8 indices into the original
    conserved-state component axis in that order.
    """
    density_index = int(registered_variables.density_index)
    energy_index = int(registered_variables.energy_index)
    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    mz = int(registered_variables.momentum_index.z)
    bx = int(registered_variables.magnetic_index.x)
    by = int(registered_variables.magnetic_index.y)
    bz = int(registered_variables.magnetic_index.z)

    if axis == 0:
        return (density_index, mx, my, mz, bx, by, bz, energy_index)
    if axis == 1:
        # Matches native ``_weno_flux_y_native``: swap mom_x↔mom_y, B_x↔B_y.
        return (density_index, my, mx, mz, by, bx, bz, energy_index)
    # axis == 2 — matches native ``_weno_flux_z_native`` transpose
    # (0, 3, 2, 1) followed by mom_x↔mom_z and B_x↔B_z swap.
    return (density_index, mz, my, mx, bz, by, bx, energy_index)


def _weno_flux_mhd_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Pallas implementation of the ideal-gas MHD WENO interface flux.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  Kernel arithmetic in
    ``_weno_flux_mhd_pallas_local``.
    """
    if not _mhd_pallas_flux_supported(conserved_state, config):
        # Lazy import to break the circular dependency with _weno.py.
        from astronomix._finite_difference._interface_fluxes._weno import (
            _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
        )
        if axis == 0:
            return _weno_flux_x_native(conserved_state, params, config, registered_variables)
        if axis == 1:
            return _weno_flux_y_native(conserved_state, params, config, registered_variables)
        return _weno_flux_z_native(conserved_state, params, config, registered_variables)

    def _local(state_local):
        return _weno_flux_mhd_pallas_local(
            state_local, params, config, registered_variables, axis=axis
        )
    return _weno5_shard_wrap(_local, conserved_state, config, axis)


def _weno_flux_mhd_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Single-shard ideal-gas MHD WENO build.  Mirrors
    ``_weno_flux_hydro_pallas`` but with 8 conserved variables and 7
    characteristic waves (fast-, alfvén-, slow-, entropy, slow+, alfvén+,
    fast+).  All face eigenstructure (the body of
    ``_eigen_mhd._eigenvector_building_blocks``) is inlined as kernel-local
    closures, and ``L_row``/``R_col``/``λ`` projections are dispatched at
    compile time via Python ``if mode == k`` branches inside the per-mode
    loop, matching the structure of the native ``_eigen_L_row`` /
    ``_eigen_R_col`` functions.  No full-domain projection matrices are ever
    materialised — every component is computed per-tile in registers."""
    ndim = 3
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx, ny, nz = spatial_shape
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _mhd_indices_for_axis(config, registered_variables, axis)
    ncomp = 8
    num_modes = 7
    epsilon = 1e-7
    tiny = 1e-14

    # Tile sizes / specs — identical to the hydro kernel.
    block_shape_out = (nvars, bx, by, bz)
    out_spec = pl.BlockSpec(block_shape_out, lambda bi, bj, bk: (0, bi, bj, bk))
    in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    # ``b_eps`` and the floors for ``sqrt`` are passed in as scalar kernel
    # arguments so they carry the same dtype as the input state.  This
    # matters under x64 + Triton: an untyped Python ``1e-20`` enters the
    # lowering as f32, which trips a ``('f64','f32')`` assertion in
    # ``_truediv_lowering_rule`` further down (see guide §5 x64 notes).
    b_eps_value = 1e-20
    sqrt_floor_value = 1e-12

    def kernel(q_ref, gamma_ref, rhomin_ref, pgmin_ref, b_eps_ref, sqrt_floor_ref, flux_out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
        jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
        kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        gam0 = 1.0 - gamma   # = -gm1
        gam1 = 0.5 * (gamma - 1.0)
        gam2 = (gamma - 2.0) / (gamma - 1.0)
        b_eps = b_eps_ref[()]
        sqrt_floor = sqrt_floor_ref[()]
        # Properly-typed literal scalars.  In x64 + Triton, bare Python
        # ``1.0`` / ``-1.0`` / ``1.0 / jnp.sqrt(2.0)`` inside ``jnp.where``
        # arrive in the lowering as f32 and trip a ``('f64', 'f32')``
        # assertion in ``_truediv_lowering_rule``.  Deriving them from
        # ``gamma`` ensures the dtype follows the kernel's working dtype.
        zero_typed = gamma - gamma
        one_typed = zero_typed + 1.0
        neg_one_typed = zero_typed - 1.0
        inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]

        def q_at(var_index: int, offset: int):
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            # local order: rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy
            return tuple(q_at(idx, offset) for idx in local_indices)

        def primitive_from_q(q):
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy = q
            inv_rho = 1.0 / rho
            vn = mn * inv_rho
            vt1 = mt1 * inv_rho
            vt2 = mt2 * inv_rho
            v2 = vn * vn + vt1 * vt1 + vt2 * vt2
            b2 = Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2
            p = gm1 * (energy - 0.5 * (rho * v2 + b2))
            return rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p

        def floored_cell(q):
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q)
            troubled = (rho < rhomin) | (p < pgmin)
            rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
            p_f = jnp.where(troubled, jnp.maximum(p, pgmin), p)
            energy_f = jnp.where(troubled, p_f / gm1 + 0.5 * (rho_f * v2 + b2), energy)
            # MHD enthalpy includes the magnetic contribution implicitly via
            # the (energy + p_gas) / rho average used in the native code.
            specific_enthalpy = (energy_f + p_f) / rho_f
            sound_speed_sq = jnp.maximum(0.0, gamma * jnp.abs(p_f / rho_f))
            sound_speed = jnp.sqrt(jnp.maximum(sound_speed_sq, sqrt_floor))
            # MHD characteristic speeds (cell-centered — used for the local
            # Lax-Friedrichs alpha; the FACE eigenstructure is computed
            # separately further down).
            bn2_over_rho = (Bn * Bn) / rho_f
            disc_root = jnp.sqrt(jnp.maximum(
                0.0,
                (b2 / rho_f + sound_speed_sq) ** 2 - 4.0 * bn2_over_rho * sound_speed_sq,
            ))
            c_fast = jnp.sqrt(jnp.maximum(
                0.0, 0.5 * (b2 / rho_f + sound_speed_sq + disc_root)
            ))
            c_alfven = jnp.sqrt(jnp.maximum(0.0, bn2_over_rho))
            c_slow = jnp.sqrt(jnp.maximum(
                0.0, 0.5 * (b2 / rho_f + sound_speed_sq - disc_root)
            ))
            return (rho_f, mn, mt1, mt2, Bn, Bt1, Bt2, energy_f,
                    vn, vt1, vt2, v2, b2, p_f, specific_enthalpy,
                    sound_speed, sound_speed_sq, c_fast, c_alfven, c_slow)

        def flux_from_q(q):
            """MHD flux along the normal direction (local x).  B_normal flux
            is identically zero (see ``_mhd_flux_x``)."""
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q)
            p_total = p + 0.5 * b2
            v_dot_B = vn * Bn + vt1 * Bt1 + vt2 * Bt2
            return (
                mn,                                  # density flux: rho * vn
                rho * vn * vn + p_total - Bn * Bn,    # normal momentum
                rho * vn * vt1 - Bn * Bt1,            # transverse 1
                rho * vn * vt2 - Bn * Bt2,            # transverse 2
                0.0,                                  # normal B flux is 0
                Bt1 * vn - Bn * vt1,                  # transverse 1 B
                Bt2 * vn - Bn * vt2,                  # transverse 2 B
                (energy + p_total) * vn - v_dot_B * Bn,  # energy
            )

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[8]; c_fast = cell[17]; c_alfven = cell[18]; c_slow = cell[19]
            if mode == 0:
                return vn - c_fast
            if mode == 1:
                return vn - c_alfven
            if mode == 2:
                return vn - c_slow
            if mode == 3:
                return vn
            if mode == 4:
                return vn + c_slow
            if mode == 5:
                return vn + c_alfven
            return vn + c_fast

        # ------------------------------------------------------------------
        # Build the FACE eigenstructure once per program (six cells, two of
        # which — at offsets 0 and 1 — straddle the interface i+1/2 we are
        # currently computing).  Matches the body of
        # ``_eigen_mhd._eigenvector_building_blocks``.
        # ------------------------------------------------------------------
        q_stencil = tuple(q_local(off) for off in range(-2, 4))     # offsets -2..3
        f_stencil = tuple(flux_from_q(q) for q in q_stencil)
        floored_stencil = tuple(floored_cell(q) for q in q_stencil)
        cell_l = floored_stencil[2]  # offset 0  (cell i)
        cell_r = floored_stencil[3]  # offset 1  (cell i+1)

        # Native MHD uses ``avg(momentum) / avg(max(rho, rhomin))`` to define
        # the interface velocity (NOT avg of velocities).  Same conventions
        # here.  Indices match the floored-cell tuple above.
        rho_i = cell_l[0]; mn_i = cell_l[1]; mt1_i = cell_l[2]; mt2_i = cell_l[3]
        Bn_i = cell_l[4]; Bt1_i = cell_l[5]; Bt2_i = cell_l[6]
        h_i = cell_l[14]
        rho_j = cell_r[0]; mn_j = cell_r[1]; mt1_j = cell_r[2]; mt2_j = cell_r[3]
        Bn_j = cell_r[4]; Bt1_j = cell_r[5]; Bt2_j = cell_r[6]
        h_j = cell_r[14]

        rho_face = jnp.maximum(
            0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)),
            rhomin,
        )
        vn_face = 0.5 * (mn_i + mn_j) / rho_face
        vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
        vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
        Bn_face = 0.5 * (Bn_i + Bn_j)
        Bt1_face = 0.5 * (Bt1_i + Bt1_j)
        Bt2_face = 0.5 * (Bt2_i + Bt2_j)
        h_face = 0.5 * (h_i + h_j)

        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
        b2_over_rho_face = b2_face / rho_face
        bn2_over_rho_face = (Bn_face * Bn_face) / rho_face

        c_sq_face = gm1 * (h_face - 0.5 * (v2_face + b2_over_rho_face))
        c_sq_face = jnp.maximum(c_sq_face, 0.0)
        c_face = jnp.sqrt(jnp.maximum(c_sq_face, sqrt_floor))
        inv_c_sq = jnp.where(c_sq_face > 0.0, 1.0 / c_sq_face, 0.0)

        ms_disc = (b2_over_rho_face + c_sq_face) ** 2 - 4.0 * bn2_over_rho_face * c_sq_face
        ms_disc_root = jnp.sqrt(jnp.maximum(ms_disc, 0.0))

        lambda_fast = jnp.sqrt(jnp.maximum(
            0.0, 0.5 * (b2_over_rho_face + c_sq_face + ms_disc_root)
        ))
        lambda_alfven = jnp.sqrt(jnp.maximum(0.0, bn2_over_rho_face))
        lambda_slow = jnp.sqrt(jnp.maximum(
            0.0, 0.5 * (b2_over_rho_face + c_sq_face - ms_disc_root)
        ))

        # Tangential normalisation with the degeneracy fix.
        bt_sq = Bt1_face * Bt1_face + Bt2_face * Bt2_face
        bt_sq_safe = jnp.maximum(bt_sq, b_eps)
        bt_n1 = jnp.where(
            bt_sq >= b_eps,
            Bt1_face / jnp.sqrt(bt_sq_safe),
            inv_sqrt_two_typed,
        )
        bt_n2 = jnp.where(
            bt_sq >= b_eps,
            Bt2_face / jnp.sqrt(bt_sq_safe),
            inv_sqrt_two_typed,
        )

        sgn_bn = jnp.where(Bn_face >= 0.0, one_typed, neg_one_typed)
        sgn_bt = jnp.where(
            Bt1_face != 0.0,
            jnp.where(Bt1_face >= 0.0, one_typed, neg_one_typed),
            jnp.where(Bt2_face >= 0.0, one_typed, neg_one_typed),
        )

        # Fast / slow mode weighting; same algebra as the native helper.
        denom = lambda_fast * lambda_fast - lambda_slow * lambda_slow
        denom_safe = jnp.maximum(denom, b_eps)
        am_fast = jnp.where(
            denom >= b_eps,
            jnp.sqrt(jnp.maximum(
                0.0, c_sq_face - lambda_slow * lambda_slow
            )) / jnp.sqrt(denom_safe),
            1.0,
        )
        am_slow = jnp.where(
            denom >= b_eps,
            jnp.sqrt(jnp.maximum(
                0.0, lambda_fast * lambda_fast - c_sq_face
            )) / jnp.sqrt(denom_safe),
            1.0,
        )

        sqrt_rho_face = jnp.sqrt(jnp.maximum(rho_face, rhomin))
        cs_geq_alfven = c_face >= lambda_alfven

        # ------------------------------------------------------------------
        # Inlined ``L_row`` and ``R_col`` for each of the 7 modes.  Each
        # ``left_project(mode, values)`` returns the scalar tile L_row · q,
        # ``add_right_correction(flux_acc, mode, Fs)`` adds Fs * R_col[:, mode]
        # to flux_acc.  All formulas are 1:1 translations of
        # ``_eigen_mhd._eigen_L_row`` and ``_eigen_R_col``.
        # ------------------------------------------------------------------
        def left_project(mode: int, values):
            """L_row[mode] · values.  ``values`` is an 8-tuple in local order:
            (rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy)."""
            rho_v, mn_v, mt1_v, mt2_v, Bn_v, Bt1_v, Bt2_v, e_v = values
            if mode == 0:  # fast-
                L_rho = (
                    am_fast * (gam1 * v2_face + lambda_fast * vn_face)
                    - am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = am_fast * (gam0 * vn_face - lambda_fast)
                L_mt1 = gam0 * am_fast * vt1_face + am_slow * lambda_slow * bt_n1 * sgn_bn
                L_mt2 = gam0 * am_fast * vt2_face + am_slow * lambda_slow * bt_n2 * sgn_bn
                L_Bt1 = gam0 * am_fast * Bt1_face + c_face * am_slow * bt_n1 * sqrt_rho_face
                L_Bt2 = gam0 * am_fast * Bt2_face + c_face * am_slow * bt_n2 * sqrt_rho_face
                L_E = -gam0 * am_fast
                acc = (
                    L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                    + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
                )
                acc = 0.5 * acc * inv_c_sq
                return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 1:  # alfvén-
                L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
                L_mt1 = -bt_n2
                L_mt2 = bt_n1
                L_Bt1 = -bt_n2 * sgn_bn * sqrt_rho_face
                L_Bt2 = bt_n1 * sgn_bn * sqrt_rho_face
                acc = (
                    L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                    + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v
                )
                return 0.5 * acc
            if mode == 2:  # slow-
                L_rho = (
                    am_slow * (gam1 * v2_face + lambda_slow * vn_face)
                    + am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = am_slow * (gam0 * vn_face) - am_slow * lambda_slow
                L_mt1 = gam0 * am_slow * vt1_face - am_fast * lambda_fast * bt_n1 * sgn_bn
                L_mt2 = gam0 * am_slow * vt2_face - am_fast * lambda_fast * bt_n2 * sgn_bn
                L_Bt1 = gam0 * am_slow * Bt1_face - c_face * am_fast * bt_n1 * sqrt_rho_face
                L_Bt2 = gam0 * am_slow * Bt2_face - c_face * am_fast * bt_n2 * sqrt_rho_face
                L_E = -gam0 * am_slow
                acc = (
                    L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                    + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
                )
                acc = 0.5 * acc * inv_c_sq
                return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 3:  # entropy
                L_rho = -c_sq_face / gam0 - 0.5 * v2_face
                L_mn = vn_face
                L_mt1 = vt1_face
                L_mt2 = vt2_face
                L_Bt1 = Bt1_face
                L_Bt2 = Bt2_face
                L_E = -1.0
                acc = (
                    L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                    + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
                )
                return -gam0 * acc * inv_c_sq
            if mode == 4:  # slow+
                L_rho = (
                    am_slow * (gam1 * v2_face - lambda_slow * vn_face)
                    - am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = am_slow * (gam0 * vn_face + lambda_slow)
                L_mt1 = gam0 * am_slow * vt1_face + am_fast * lambda_fast * bt_n1 * sgn_bn
                L_mt2 = gam0 * am_slow * vt2_face + am_fast * lambda_fast * bt_n2 * sgn_bn
                L_Bt1 = gam0 * am_slow * Bt1_face - c_face * am_fast * bt_n1 * sqrt_rho_face
                L_Bt2 = gam0 * am_slow * Bt2_face - c_face * am_fast * bt_n2 * sqrt_rho_face
                L_E = -gam0 * am_slow
                acc = (
                    L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                    + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
                )
                acc = 0.5 * acc * inv_c_sq
                return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 5:  # alfvén+
                L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
                L_mt1 = -bt_n2
                L_mt2 = bt_n1
                L_Bt1 = bt_n2 * sgn_bn * sqrt_rho_face
                L_Bt2 = -bt_n1 * sgn_bn * sqrt_rho_face
                acc = (
                    L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                    + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v
                )
                return 0.5 * acc
            # mode 6 — fast+
            L_rho = (
                am_fast * (gam1 * v2_face - lambda_fast * vn_face)
                + am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
            )
            L_mn = am_fast * (gam0 * vn_face + lambda_fast)
            L_mt1 = gam0 * am_fast * vt1_face - am_slow * lambda_slow * bt_n1 * sgn_bn
            L_mt2 = gam0 * am_fast * vt2_face - am_slow * lambda_slow * bt_n2 * sgn_bn
            L_Bt1 = gam0 * am_fast * Bt1_face + c_face * am_slow * bt_n1 * sqrt_rho_face
            L_Bt2 = gam0 * am_fast * Bt2_face + c_face * am_slow * bt_n2 * sqrt_rho_face
            L_E = -gam0 * am_fast
            acc = (
                L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v + L_E * e_v
            )
            acc = 0.5 * acc * inv_c_sq
            return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)

        def add_right_correction(flux_acc, mode: int, Fs):
            """flux_acc += Fs * R_col[:, mode] (local order, ncomp=8).
            B_normal slot (index 4) always gets 0."""
            if mode == 0:  # fast-
                R = (
                    am_fast,
                    am_fast * (vn_face - lambda_fast),
                    am_fast * vt1_face + am_slow * lambda_slow * bt_n1 * sgn_bn,
                    am_fast * vt2_face + am_slow * lambda_slow * bt_n2 * sgn_bn,
                    0.0,
                    c_face * am_slow * bt_n1 / sqrt_rho_face,
                    c_face * am_slow * bt_n2 / sqrt_rho_face,
                    am_fast * (
                        lambda_fast * lambda_fast
                        - lambda_fast * vn_face
                        + 0.5 * v2_face
                        - gam2 * c_sq_face
                    )
                    + am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
                )
                scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 1:  # alfvén-
                R = (
                    0.0,
                    0.0,
                    -bt_n2,
                    bt_n1,
                    0.0,
                    -bt_n2 * sgn_bn / sqrt_rho_face,
                    bt_n1 * sgn_bn / sqrt_rho_face,
                    bt_n1 * vt2_face - bt_n2 * vt1_face,
                )
                scale = 1.0
            elif mode == 2:  # slow-
                R = (
                    am_slow,
                    am_slow * (vn_face - lambda_slow),
                    am_slow * vt1_face - am_fast * lambda_fast * bt_n1 * sgn_bn,
                    am_slow * vt2_face - am_fast * lambda_fast * bt_n2 * sgn_bn,
                    0.0,
                    -c_face * am_fast * bt_n1 / sqrt_rho_face,
                    -c_face * am_fast * bt_n2 / sqrt_rho_face,
                    am_slow * (
                        lambda_slow * lambda_slow
                        - lambda_slow * vn_face
                        + 0.5 * v2_face
                        - gam2 * c_sq_face
                    )
                    - am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
                )
                scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 3:  # entropy
                R = (
                    1.0,
                    vn_face,
                    vt1_face,
                    vt2_face,
                    0.0,
                    0.0,
                    0.0,
                    0.5 * v2_face,
                )
                scale = 1.0
            elif mode == 4:  # slow+
                R = (
                    am_slow,
                    am_slow * (vn_face + lambda_slow),
                    am_slow * vt1_face + am_fast * lambda_fast * bt_n1 * sgn_bn,
                    am_slow * vt2_face + am_fast * lambda_fast * bt_n2 * sgn_bn,
                    0.0,
                    -c_face * am_fast * bt_n1 / sqrt_rho_face,
                    -c_face * am_fast * bt_n2 / sqrt_rho_face,
                    am_slow * (
                        lambda_slow * lambda_slow
                        + lambda_slow * vn_face
                        + 0.5 * v2_face
                        - gam2 * c_sq_face
                    )
                    + am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
                )
                scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 5:  # alfvén+
                R = (
                    0.0,
                    0.0,
                    -bt_n2,
                    bt_n1,
                    0.0,
                    bt_n2 * sgn_bn / sqrt_rho_face,
                    -bt_n1 * sgn_bn / sqrt_rho_face,
                    bt_n1 * vt2_face - bt_n2 * vt1_face,
                )
                scale = 1.0
            else:  # mode == 6 — fast+
                R = (
                    am_fast,
                    am_fast * (vn_face + lambda_fast),
                    am_fast * vt1_face - am_slow * lambda_slow * bt_n1 * sgn_bn,
                    am_fast * vt2_face - am_slow * lambda_slow * bt_n2 * sgn_bn,
                    0.0,
                    c_face * am_slow * bt_n1 / sqrt_rho_face,
                    c_face * am_slow * bt_n2 / sqrt_rho_face,
                    am_fast * (
                        lambda_fast * lambda_fast
                        + lambda_fast * vn_face
                        + 0.5 * v2_face
                        - gam2 * c_sq_face
                    )
                    - am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn,
                )
                scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
            return [flux_acc[slot] + (R[slot] * scale) * Fs for slot in range(ncomp)]

        def alpha_for_mode(mode: int):
            amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
            for k in range(1, 6):
                amx = jnp.maximum(
                    amx, jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode))
                )
            return amx

        # First-order centered part (1/12 stencil), one per component.
        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot]
             + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) / 12.0
            for slot in range(ncomp)
        ]

        for mode in range(num_modes):
            s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
            qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

            d0 = s[1] - s[0]; d1 = s[2] - s[1]; d2 = s[3] - s[2]
            d3 = s[4] - s[3]; d4 = s[5] - s[4]
            dq0 = qproj[1] - qproj[0]; dq1 = qproj[2] - qproj[1]
            dq2 = qproj[3] - qproj[2]; dq3 = qproj[4] - qproj[3]
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
            second = (omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) / 3.0
                      + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) / 6.0)

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
            third = (omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) / 3.0
                     + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) / 6.0)

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        # Write every output component.  Hydro/MHD covers all conserved
        # variables, but explicitly zero anything not in ``local_indices``
        # (defensive — also makes the B_normal-flux = 0 invariant explicit).
        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"mhd_weno_flux_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.gamma, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
        jnp.asarray(b_eps_value, dtype=conserved_state.dtype),
        jnp.asarray(sqrt_floor_value, dtype=conserved_state.dtype),
    )


# -----------------------------------------------------------------------------
# Pallas WENO for the isothermal MHD equations.
# -----------------------------------------------------------------------------


def _mhd_iso_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """Whether the Pallas isothermal MHD WENO kernel can be used."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != ISOTHERMAL:
        return False
    ndim = int(config.dimensionality)
    if ndim != 3:
        return False
    if conserved_state.ndim != 4:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(conserved_state.shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _mhd_iso_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Local conserved-variable order for isothermal MHD: (density, p_normal,
    p_trans1, p_trans2, B_normal, B_trans1, B_trans2).  Seven slots — no
    energy.  ``B_normal`` is the 0-coefficient placeholder so the
    L_row/R_col formulas can use the same projection structure as ideal-gas
    MHD, and its output flux slot is zeroed (matching ``_mhd_flux_isothermal_x``).
    """
    density_index = int(registered_variables.density_index)
    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    mz = int(registered_variables.momentum_index.z)
    bx = int(registered_variables.magnetic_index.x)
    by = int(registered_variables.magnetic_index.y)
    bz = int(registered_variables.magnetic_index.z)

    if axis == 0:
        return (density_index, mx, my, mz, bx, by, bz)
    if axis == 1:
        return (density_index, my, mx, mz, by, bx, bz)
    return (density_index, mz, my, mx, bz, by, bx)


def _weno_flux_mhd_iso_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Pallas implementation of the isothermal MHD WENO interface flux.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  Kernel arithmetic in
    ``_weno_flux_mhd_iso_pallas_local``.
    """
    if not _mhd_iso_pallas_flux_supported(conserved_state, config):
        # Lazy import to break the circular dependency with _weno.py.
        from astronomix._finite_difference._interface_fluxes._weno import (
            _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
        )
        if axis == 0:
            return _weno_flux_x_native(conserved_state, params, config, registered_variables)
        if axis == 1:
            return _weno_flux_y_native(conserved_state, params, config, registered_variables)
        return _weno_flux_z_native(conserved_state, params, config, registered_variables)

    def _local(state_local):
        return _weno_flux_mhd_iso_pallas_local(
            state_local, params, config, registered_variables, axis=axis
        )
    return _weno5_shard_wrap(_local, conserved_state, config, axis)


def _weno_flux_mhd_iso_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Single-shard isothermal MHD WENO build.  Mirrors
    ``_weno_flux_mhd_pallas`` but with 7 conserved-state slots (no
    energy) and 6 characteristic waves (no entropy mode): fast-,
    alfvén-, slow-, slow+, alfvén+, fast+.  Sound speed is the fixed
    ``params.isothermal_sound_speed``.  All face eigenstructure,
    ``L_row``, ``R_col``, and ``λ`` are inlined as kernel-local closures
    mirroring ``_eigen_mhd_iso`` line-for-line."""
    ndim = 3
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx, ny, nz = spatial_shape
    bx_, by_, bz_ = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx_, ny // by_, nz // bz_)

    local_indices = _mhd_iso_indices_for_axis(config, registered_variables, axis)
    ncomp = 7
    num_modes = 6
    epsilon = 1e-7
    tiny = 1e-14
    b_eps_value = 1e-20

    block_shape_out = (nvars, bx_, by_, bz_)
    out_spec = pl.BlockSpec(block_shape_out, lambda bi, bj, bk: (0, bi, bj, bk))
    in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, cs_ref, rhomin_ref, b_eps_ref, flux_out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        ii = (bi * bx_ + jnp.arange(bx_)[:, None, None]) % nx
        jj = (bj * by_ + jnp.arange(by_)[None, :, None]) % ny
        kk = (bk * bz_ + jnp.arange(bz_)[None, None, :]) % nz

        cs = cs_ref[()]
        cs2 = cs * cs
        cs2_inv = jnp.where(cs2 > 0.0, 1.0 / cs2, 0.0)
        rhomin = rhomin_ref[()]
        b_eps = b_eps_ref[()]
        # Properly-typed literal scalars (see x64-Triton workaround in the
        # ideal-gas MHD kernel for the rationale).
        zero_typed = cs - cs
        one_typed = zero_typed + 1.0
        neg_one_typed = zero_typed - 1.0
        inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)

        def q_at(var_index, offset):
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset):
            return tuple(q_at(idx, offset) for idx in local_indices)

        def primitive_from_q(q):
            rho, mn, mt1, mt2, Bn, Bt1, Bt2 = q
            inv_rho = 1.0 / rho
            vn = mn * inv_rho
            vt1 = mt1 * inv_rho
            vt2 = mt2 * inv_rho
            v2 = vn * vn + vt1 * vt1 + vt2 * vt2
            b2 = Bn * Bn + Bt1 * Bt1 + Bt2 * Bt2
            return rho, mn, mt1, mt2, Bn, Bt1, Bt2, vn, vt1, vt2, v2, b2

        def floored_cell(q):
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, vn, vt1, vt2, v2, b2 = primitive_from_q(q)
            rho_f = jnp.maximum(rho, rhomin)
            # Recompute primitives that depend on the floored density to keep
            # downstream arithmetic consistent.
            inv_rho = 1.0 / rho_f
            vn_f = mn * inv_rho
            vt1_f = mt1 * inv_rho
            vt2_f = mt2 * inv_rho
            bn2_over_rho = (Bn * Bn) / rho_f
            disc_root = jnp.sqrt(jnp.maximum(
                0.0, (b2 / rho_f + cs2) ** 2 - 4.0 * bn2_over_rho * cs2
            ))
            c_fast = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2 / rho_f + cs2 + disc_root)))
            c_alfven = jnp.sqrt(jnp.maximum(0.0, bn2_over_rho))
            c_slow = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2 / rho_f + cs2 - disc_root)))
            return (rho_f, mn, mt1, mt2, Bn, Bt1, Bt2,
                    vn_f, vt1_f, vt2_f, c_fast, c_alfven, c_slow)

        def flux_from_q(q):
            """Isothermal MHD x-flux in local order; B_normal flux is 0."""
            rho, mn, mt1, mt2, Bn, Bt1, Bt2, vn, vt1, vt2, v2, b2 = primitive_from_q(q)
            p_iso = cs2 * rho
            p_total = p_iso + 0.5 * b2
            return (
                mn,
                rho * vn * vn + p_total - Bn * Bn,
                rho * vn * vt1 - Bn * Bt1,
                rho * vn * vt2 - Bn * Bt2,
                0.0,
                Bt1 * vn - Bn * vt1,
                Bt2 * vn - Bn * vt2,
            )

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[7]; c_fast = cell[10]; c_alfven = cell[11]; c_slow = cell[12]
            if mode == 0:
                return vn - c_fast
            if mode == 1:
                return vn - c_alfven
            if mode == 2:
                return vn - c_slow
            if mode == 3:
                return vn + c_slow
            if mode == 4:
                return vn + c_alfven
            return vn + c_fast

        q_stencil = tuple(q_local(off) for off in range(-2, 4))
        f_stencil = tuple(flux_from_q(q) for q in q_stencil)
        floored_stencil = tuple(floored_cell(q) for q in q_stencil)
        cell_l = floored_stencil[2]
        cell_r = floored_stencil[3]

        rho_i, mn_i, mt1_i, mt2_i, Bn_i, Bt1_i, Bt2_i = cell_l[:7]
        rho_j, mn_j, mt1_j, mt2_j, Bn_j, Bt1_j, Bt2_j = cell_r[:7]
        rho_face = jnp.maximum(
            0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)),
            rhomin,
        )
        vn_face = 0.5 * (mn_i + mn_j) / rho_face
        vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
        vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
        Bn_face = 0.5 * (Bn_i + Bn_j)
        Bt1_face = 0.5 * (Bt1_i + Bt1_j)
        Bt2_face = 0.5 * (Bt2_i + Bt2_j)

        b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
        b2_over_rho = b2_face / rho_face
        bn2_over_rho = (Bn_face * Bn_face) / rho_face

        ms_disc = (b2_over_rho + cs2) ** 2 - 4.0 * bn2_over_rho * cs2
        ms_disc_root = jnp.sqrt(jnp.maximum(ms_disc, 0.0))
        lambda_fast = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2_over_rho + cs2 + ms_disc_root)))
        lambda_alfven = jnp.sqrt(jnp.maximum(0.0, bn2_over_rho))
        lambda_slow = jnp.sqrt(jnp.maximum(0.0, 0.5 * (b2_over_rho + cs2 - ms_disc_root)))

        bt_sq = Bt1_face * Bt1_face + Bt2_face * Bt2_face
        bt_sq_safe = jnp.maximum(bt_sq, b_eps)
        bt_n1 = jnp.where(bt_sq >= b_eps, Bt1_face / jnp.sqrt(bt_sq_safe), inv_sqrt_two_typed)
        bt_n2 = jnp.where(bt_sq >= b_eps, Bt2_face / jnp.sqrt(bt_sq_safe), inv_sqrt_two_typed)

        sgn_bn = jnp.where(Bn_face >= 0.0, one_typed, neg_one_typed)
        sgn_bt = jnp.where(
            Bt1_face != 0.0,
            jnp.where(Bt1_face >= 0.0, one_typed, neg_one_typed),
            jnp.where(Bt2_face >= 0.0, one_typed, neg_one_typed),
        )

        denom = lambda_fast * lambda_fast - lambda_slow * lambda_slow
        denom_safe = jnp.maximum(denom, b_eps)
        am_fast = jnp.where(
            denom >= b_eps,
            jnp.sqrt(jnp.maximum(0.0, cs2 - lambda_slow * lambda_slow)) / jnp.sqrt(denom_safe),
            1.0,
        )
        am_slow = jnp.where(
            denom >= b_eps,
            jnp.sqrt(jnp.maximum(0.0, lambda_fast * lambda_fast - cs2)) / jnp.sqrt(denom_safe),
            1.0,
        )

        sqrt_rho_face = jnp.sqrt(jnp.maximum(rho_face, rhomin))
        cs_geq_alfven = cs >= lambda_alfven

        def left_project(mode: int, values):
            """L_row[mode] · values for iso MHD.  ``values`` is a 7-tuple:
            (rho, mn, mt1, mt2, Bn, Bt1, Bt2)."""
            rho_v, mn_v, mt1_v, mt2_v, Bn_v, Bt1_v, Bt2_v = values
            if mode == 0:  # fast-
                L_rho = (
                    am_fast * (cs2 + lambda_fast * vn_face)
                    - am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = -am_fast * lambda_fast
                L_mt1 = am_slow * lambda_slow * bt_n1 * sgn_bn
                L_mt2 = am_slow * lambda_slow * bt_n2 * sgn_bn
                L_Bt1 = cs * am_slow * bt_n1 * sqrt_rho_face
                L_Bt2 = cs * am_slow * bt_n2 * sqrt_rho_face
                acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                acc = 0.5 * acc * cs2_inv
                return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 1:  # alfvén-
                L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
                L_mt1 = -bt_n2
                L_mt2 = bt_n1
                L_Bt1 = -bt_n2 * sgn_bn * sqrt_rho_face
                L_Bt2 = bt_n1 * sgn_bn * sqrt_rho_face
                acc = (L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                return 0.5 * acc
            if mode == 2:  # slow-
                L_rho = (
                    am_slow * (cs2 + lambda_slow * vn_face)
                    + am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = -am_slow * lambda_slow
                L_mt1 = -am_fast * lambda_fast * bt_n1 * sgn_bn
                L_mt2 = -am_fast * lambda_fast * bt_n2 * sgn_bn
                L_Bt1 = -cs * am_fast * bt_n1 * sqrt_rho_face
                L_Bt2 = -cs * am_fast * bt_n2 * sqrt_rho_face
                acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                acc = 0.5 * acc * cs2_inv
                return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 3:  # slow+
                L_rho = (
                    am_slow * (cs2 - lambda_slow * vn_face)
                    - am_fast * lambda_fast * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
                )
                L_mn = am_slow * lambda_slow
                L_mt1 = am_fast * lambda_fast * bt_n1 * sgn_bn
                L_mt2 = am_fast * lambda_fast * bt_n2 * sgn_bn
                L_Bt1 = -cs * am_fast * bt_n1 * sqrt_rho_face
                L_Bt2 = -cs * am_fast * bt_n2 * sqrt_rho_face
                acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                acc = 0.5 * acc * cs2_inv
                return jnp.where(cs_geq_alfven, acc * sgn_bt, acc)
            if mode == 4:  # alfvén+
                L_rho = bt_n2 * vt1_face - bt_n1 * vt2_face
                L_mt1 = -bt_n2
                L_mt2 = bt_n1
                L_Bt1 = bt_n2 * sgn_bn * sqrt_rho_face
                L_Bt2 = -bt_n1 * sgn_bn * sqrt_rho_face
                acc = (L_rho * rho_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                       + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
                return 0.5 * acc
            # mode 5 — fast+
            L_rho = (
                am_fast * (cs2 - lambda_fast * vn_face)
                + am_slow * lambda_slow * (bt_n1 * vt1_face + bt_n2 * vt2_face) * sgn_bn
            )
            L_mn = am_fast * lambda_fast
            L_mt1 = -am_slow * lambda_slow * bt_n1 * sgn_bn
            L_mt2 = -am_slow * lambda_slow * bt_n2 * sgn_bn
            L_Bt1 = cs * am_slow * bt_n1 * sqrt_rho_face
            L_Bt2 = cs * am_slow * bt_n2 * sqrt_rho_face
            acc = (L_rho * rho_v + L_mn * mn_v + L_mt1 * mt1_v + L_mt2 * mt2_v
                   + L_Bt1 * Bt1_v + L_Bt2 * Bt2_v)
            acc = 0.5 * acc * cs2_inv
            return jnp.where(~cs_geq_alfven, acc * sgn_bt, acc)

        def add_right_correction(flux_acc, mode: int, Fs):
            if mode == 0:  # fast-
                R = (
                    am_fast,
                    am_fast * (vn_face - lambda_fast),
                    am_fast * vt1_face + am_slow * lambda_slow * bt_n1 * sgn_bn,
                    am_fast * vt2_face + am_slow * lambda_slow * bt_n2 * sgn_bn,
                    0.0,
                    cs * am_slow * bt_n1 / sqrt_rho_face,
                    cs * am_slow * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 1:  # alfvén-
                R = (
                    0.0, 0.0,
                    -bt_n2, bt_n1, 0.0,
                    -bt_n2 * sgn_bn / sqrt_rho_face,
                    bt_n1 * sgn_bn / sqrt_rho_face,
                )
                scale = 1.0
            elif mode == 2:  # slow-
                R = (
                    am_slow,
                    am_slow * (vn_face - lambda_slow),
                    am_slow * vt1_face - am_fast * lambda_fast * bt_n1 * sgn_bn,
                    am_slow * vt2_face - am_fast * lambda_fast * bt_n2 * sgn_bn,
                    0.0,
                    -cs * am_fast * bt_n1 / sqrt_rho_face,
                    -cs * am_fast * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 3:  # slow+
                R = (
                    am_slow,
                    am_slow * (vn_face + lambda_slow),
                    am_slow * vt1_face + am_fast * lambda_fast * bt_n1 * sgn_bn,
                    am_slow * vt2_face + am_fast * lambda_fast * bt_n2 * sgn_bn,
                    0.0,
                    -cs * am_fast * bt_n1 / sqrt_rho_face,
                    -cs * am_fast * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(cs_geq_alfven, sgn_bt, 1.0)
            elif mode == 4:  # alfvén+
                R = (
                    0.0, 0.0,
                    -bt_n2, bt_n1, 0.0,
                    bt_n2 * sgn_bn / sqrt_rho_face,
                    -bt_n1 * sgn_bn / sqrt_rho_face,
                )
                scale = 1.0
            else:  # mode 5 — fast+
                R = (
                    am_fast,
                    am_fast * (vn_face + lambda_fast),
                    am_fast * vt1_face - am_slow * lambda_slow * bt_n1 * sgn_bn,
                    am_fast * vt2_face - am_slow * lambda_slow * bt_n2 * sgn_bn,
                    0.0,
                    cs * am_slow * bt_n1 / sqrt_rho_face,
                    cs * am_slow * bt_n2 / sqrt_rho_face,
                )
                scale = jnp.where(~cs_geq_alfven, sgn_bt, 1.0)
            return [flux_acc[slot] + (R[slot] * scale) * Fs for slot in range(ncomp)]

        def alpha_for_mode(mode: int):
            amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
            for k in range(1, 6):
                amx = jnp.maximum(
                    amx, jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode))
                )
            return amx

        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot]
             + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) / 12.0
            for slot in range(ncomp)
        ]

        for mode in range(num_modes):
            s = tuple(left_project(mode, f_stencil[k]) for k in range(6))
            qproj = tuple(left_project(mode, q_stencil[k]) for k in range(6))

            d0 = s[1] - s[0]; d1 = s[2] - s[1]; d2 = s[3] - s[2]
            d3 = s[4] - s[3]; d4 = s[5] - s[4]
            dq0 = qproj[1] - qproj[0]; dq1 = qproj[2] - qproj[1]
            dq2 = qproj[3] - qproj[2]; dq3 = qproj[4] - qproj[3]
            dq4 = qproj[5] - qproj[4]

            amx = alpha_for_mode(mode)

            aterm_p = 0.5 * (d0 + amx * dq0); bterm_p = 0.5 * (d1 + amx * dq1)
            cterm_p = 0.5 * (d2 + amx * dq2); dterm_p = 0.5 * (d3 + amx * dq3)
            IS0_p = 13.0 * (aterm_p - bterm_p) ** 2 + 3.0 * (aterm_p - 3.0 * bterm_p) ** 2
            IS1_p = 13.0 * (bterm_p - cterm_p) ** 2 + 3.0 * (bterm_p + cterm_p) ** 2
            IS2_p = 13.0 * (cterm_p - dterm_p) ** 2 + 3.0 * (3.0 * cterm_p - dterm_p) ** 2
            alpha0_p = 1.0 / (epsilon + IS0_p) ** 2
            alpha1_p = 6.0 / (epsilon + IS1_p) ** 2
            alpha2_p = 3.0 / (epsilon + IS2_p) ** 2
            alpha_sum_p = jnp.maximum(alpha0_p + alpha1_p + alpha2_p, tiny)
            omega0_p = alpha0_p / alpha_sum_p
            omega2_p = alpha2_p / alpha_sum_p
            second = (omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) / 3.0
                      + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) / 6.0)

            aterm_m = 0.5 * (d4 - amx * dq4); bterm_m = 0.5 * (d3 - amx * dq3)
            cterm_m = 0.5 * (d2 - amx * dq2); dterm_m = 0.5 * (d1 - amx * dq1)
            IS0_m = 13.0 * (aterm_m - bterm_m) ** 2 + 3.0 * (aterm_m - 3.0 * bterm_m) ** 2
            IS1_m = 13.0 * (bterm_m - cterm_m) ** 2 + 3.0 * (bterm_m + cterm_m) ** 2
            IS2_m = 13.0 * (cterm_m - dterm_m) ** 2 + 3.0 * (3.0 * cterm_m - dterm_m) ** 2
            alpha0_m = 1.0 / (epsilon + IS0_m) ** 2
            alpha1_m = 6.0 / (epsilon + IS1_m) ** 2
            alpha2_m = 3.0 / (epsilon + IS2_m) ** 2
            alpha_sum_m = jnp.maximum(alpha0_m + alpha1_m + alpha2_m, tiny)
            omega0_m = alpha0_m / alpha_sum_m
            omega2_m = alpha2_m / alpha_sum_m
            third = (omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) / 3.0
                     + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) / 6.0)

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"mhd_iso_weno_flux_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.isothermal_sound_speed, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
        jnp.asarray(b_eps_value, dtype=conserved_state.dtype),
    )


# -----------------------------------------------------------------------------
# Pallas WENO for the isothermal hydrodynamic (Euler) equations.
# -----------------------------------------------------------------------------


def _hydro_iso_pallas_flux_supported(conserved_state, config: SimulationConfig) -> bool:
    """Whether the Pallas isothermal-hydro WENO kernel can be used.

    Handles isothermal (no-energy) hydrodynamics in 1/2/3-D: ncomp =
    num_modes = ndim+1 (density + ndim momenta; acoustic-/shear/acoustic+
    waves, no entropy mode).  Ideal-gas hydro and MHD fall back to their own
    kernels / the native path.  See ``pallas_backend_implementation_guide.md``
    §4.2 and ``_eigen_hydro_iso`` for the isothermal eigenstructure.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.mhd:
        return False
    if config.equation_of_state != ISOTHERMAL:
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if conserved_state.ndim != ndim + 1:
        return False
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    spatial_shape = conserved_state.shape[1:]
    for n, b in zip(spatial_shape, block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


def _hydro_iso_indices_for_axis(config: SimulationConfig, registered_variables: RegisteredVariables, axis: int):
    """Local conserved-variable order for an isothermal-hydro flux normal to
    ``axis``: (density, normal momentum, transverse momenta…).  No energy.

    Mirrors ``_hydro_indices_for_axis`` with the energy slot dropped, so the
    same characteristic-projection structure applies with one fewer wave.
    """
    density_index = int(registered_variables.density_index)
    ndim = int(config.dimensionality)

    if ndim == 1:
        momentum_x = int(registered_variables.momentum_index)
        return (density_index, momentum_x)

    mx = int(registered_variables.momentum_index.x)
    my = int(registered_variables.momentum_index.y)
    if ndim == 2:
        if axis == 0:
            return (density_index, mx, my)
        return (density_index, my, mx)

    mz = int(registered_variables.momentum_index.z)
    if axis == 0:
        return (density_index, mx, my, mz)
    if axis == 1:
        return (density_index, my, mx, mz)
    return (density_index, mz, my, mx)


def _weno_flux_hydro_iso_pallas(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Pallas implementation of the isothermal hydrodynamic WENO flux.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  Kernel arithmetic in
    ``_weno_flux_hydro_iso_pallas_local``.
    """
    if not _hydro_iso_pallas_flux_supported(conserved_state, config):
        # Lazy import to break the circular dependency with _weno.py.
        from astronomix._finite_difference._interface_fluxes._weno import (
            _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
        )
        if axis == 0:
            return _weno_flux_x_native(conserved_state, params, config, registered_variables)
        if axis == 1:
            return _weno_flux_y_native(conserved_state, params, config, registered_variables)
        return _weno_flux_z_native(conserved_state, params, config, registered_variables)

    def _local(state_local):
        return _weno_flux_hydro_iso_pallas_local(
            state_local, params, config, registered_variables, axis=axis
        )
    return _weno5_shard_wrap(_local, conserved_state, config, axis)


def _weno_flux_hydro_iso_pallas_local(
    conserved_state,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Single-shard isothermal-hydro WENO build.  Mirrors
    ``_weno_flux_hydro_pallas_local`` but with ``ncomp = num_modes = ndim+1``
    (no energy), a fixed sound speed ``params.isothermal_sound_speed``, and the
    isothermal eigenstructure of ``_eigen_hydro_iso``: acoustic- (u−cs), the
    ndim−1 shear waves (u), and acoustic+ (u+cs).  No entropy mode."""
    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_iso_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 1
    epsilon = 1e-7
    tiny = 1e-14
    enforce_positivity = bool(config.enforce_positivity)

    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(q_ref, cs_ref, rhomin_ref, flux_out_ref):
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

        cs = cs_ref[()]
        cs2 = cs * cs
        cs2_inv = jnp.where(cs2 > 0.0, 1.0 / cs2, 0.0)
        rhomin = rhomin_ref[()]

        def q_at(var_index: int, offset: int):
            if ndim == 1:
                return q_ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return q_ref[var_index, (ii + offset) % nx, jj]
                return q_ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            return tuple(q_at(idx, offset) for idx in local_indices)

        def flux_from_q(q):
            """Isothermal Euler normal flux in local order, mirroring
            ``_euler_flux_isothermal_x``: density flux ``mn``, normal-momentum
            flux ``mn^2/rho + cs^2 rho``, transverse ``mn * (mt/rho)``.  The
            density floor is applied only when ``config.enforce_positivity``,
            exactly as in the native flux."""
            rho = q[0]
            mn = q[1]
            rho_flux = jnp.maximum(rho, rhomin) if enforce_positivity else rho
            p = cs2 * rho_flux
            if ncomp == 2:
                return (mn, mn * mn / rho_flux + p)
            if ncomp == 3:
                mt1 = q[2]
                return (mn, mn * mn / rho_flux + p, mn * (mt1 / rho_flux))
            mt1 = q[2]
            mt2 = q[3]
            return (mn, mn * mn / rho_flux + p, mn * (mt1 / rho_flux), mn * (mt2 / rho_flux))

        q_stencil = tuple(q_local(off) for off in range(-2, 4))  # offsets -2..3
        f_stencil = tuple(flux_from_q(q) for q in q_stencil)

        # Cell-centred normal velocities for the local Lax-Friedrichs alpha,
        # mirroring ``_eigenvalue_building_blocks`` (density floored to rhomin).
        vn_cells = tuple(qc[1] / jnp.maximum(qc[0], rhomin) for qc in q_stencil)

        # Interface (face at i+1/2) eigenstructure, mirroring
        # ``_eigenvector_building_blocks``: floor each cell density, average,
        # re-floor; the face velocity is avg(momentum) / face density.
        cell_l = q_stencil[2]  # offset 0  (cell i)
        cell_r = q_stencil[3]  # offset 1  (cell i+1)
        rho_face = jnp.maximum(
            0.5 * (jnp.maximum(cell_l[0], rhomin) + jnp.maximum(cell_r[0], rhomin)),
            rhomin,
        )
        vn_face = 0.5 * (cell_l[1] + cell_r[1]) / rho_face
        vt1_face = 0.5 * (cell_l[2] + cell_r[2]) / rho_face if ncomp >= 3 else None
        vt2_face = 0.5 * (cell_l[3] + cell_r[3]) / rho_face if ncomp >= 4 else None

        def lambda_at(vn_cell, mode: int):
            if mode == 0:
                return vn_cell - cs
            if mode == num_modes - 1:
                return vn_cell + cs
            return vn_cell

        def alpha_for_mode(mode: int):
            amx = jnp.abs(lambda_at(vn_cells[0], mode))
            for k in range(1, 6):
                amx = jnp.maximum(amx, jnp.abs(lambda_at(vn_cells[k], mode)))
            return amx

        def left_project(mode: int, values):
            """L_row[mode] · values, mirroring ``_eigen_L_row_hydro_iso``.
            ``values`` is a local vector (rho, mn, [mt1], [mt2])."""
            if mode == 0:  # acoustic minus (u - cs)
                acc = (cs2 + vn_face * cs) * values[0] - cs * values[1]
                return 0.5 * cs2_inv * acc
            if mode == num_modes - 1:  # acoustic plus (u + cs)
                acc = (cs2 - vn_face * cs) * values[0] + cs * values[1]
                return 0.5 * cs2_inv * acc
            if mode == 1:  # shear (first transverse)
                return -vt1_face * values[0] + values[2]
            # mode == 2 — shear (second transverse, 3-D only)
            return -vt2_face * values[0] + values[3]

        def add_right_correction(flux_acc, mode: int, Fs):
            """flux_acc += Fs * R_col[:, mode], mirroring ``_eigen_R_col_hydro_iso``."""
            if mode == 0:  # acoustic minus
                if ncomp == 2:
                    R = (1.0, vn_face - cs)
                elif ncomp == 3:
                    R = (1.0, vn_face - cs, vt1_face)
                else:
                    R = (1.0, vn_face - cs, vt1_face, vt2_face)
            elif mode == num_modes - 1:  # acoustic plus
                if ncomp == 2:
                    R = (1.0, vn_face + cs)
                elif ncomp == 3:
                    R = (1.0, vn_face + cs, vt1_face)
                else:
                    R = (1.0, vn_face + cs, vt1_face, vt2_face)
            elif mode == 1:  # shear (first transverse)
                if ncomp == 3:
                    R = (0.0, 0.0, 1.0)
                else:
                    R = (0.0, 0.0, 1.0, 0.0)
            else:  # mode == 2 — shear (second transverse, 3-D)
                R = (0.0, 0.0, 0.0, 1.0)
            return [flux_acc[slot] + R[slot] * Fs for slot in range(ncomp)]

        flux_acc = [
            (-f_stencil[1][slot] + 7.0 * f_stencil[2][slot]
             + 7.0 * f_stencil[3][slot] - f_stencil[4][slot]) / 12.0
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
            second = (omega0_p * (aterm_p - 2.0 * bterm_p + cterm_p) / 3.0
                      + (omega2_p - 0.5) * (bterm_p - 2.0 * cterm_p + dterm_p) / 6.0)

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
            third = (omega0_m * (aterm_m - 2.0 * bterm_m + cterm_m) / 3.0
                     + (omega2_m - 0.5) * (bterm_m - 2.0 * cterm_m + dterm_m) / 6.0)

            Fs = -second + third
            flux_acc = add_right_correction(flux_acc, mode, Fs)

        zero = flux_acc[0] * 0.0
        for var in range(nvars):
            flux_out_ref[var, ...] = zero
        for slot, var in enumerate(local_indices):
            flux_out_ref[var, ...] = flux_acc[slot]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[in_state_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"hydro_iso_weno_flux_axis_{axis}",
        **kwargs,
    )(
        conserved_state,
        jnp.asarray(params.isothermal_sound_speed, dtype=conserved_state.dtype),
        jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
    )


def _weno_flux_hydro_pallas_rhs(
    conserved_state,
    dt_over_dx,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    rhs_accumulator=None,
):
    """Fused WENO interface flux + axis-flux-divergence kernel.

    Computes ``rhs_out = (rhs_accumulator if provided else 0) +
    (-dt_over_dx) * d/dx_axis(F_axis(state))`` directly, without ever
    materialising the full-state-sized interface flux ``F_axis``.  Each Pallas
    block evaluates the two interface fluxes ``F_{i+1/2}`` and ``F_{i-1/2}``
    it needs locally and writes the divergence contribution (added to the
    accumulator, when present) into its output tile.

    When ``rhs_accumulator`` is provided, the kernel uses
    ``input_output_aliases`` so XLA can keep a single physical RHS buffer
    across all three axes — eliminating both the materialised ``dF``
    temporaries and the chained ``rhs + ...`` adds that would otherwise
    duplicate full-state buffers.

    The arithmetic matches a single pass through ``_weno_flux_hydro_pallas``
    followed by ``_hydro_flux_divergence_pallas``; the only change is that the
    left interface flux is also computed inside the same program rather than
    being read back from HBM.

    Public entry point: dispatches the supported-predicate check and the
    multi-GPU ``shard_map`` + halo wrap.  The same WENO5 halo as the
    pure-flux variant (3 cells on the active axis) suffices — the fused
    kernel evaluates both ``F_{i+1/2}`` and ``F_{i-1/2}``, and the deepest
    read inside ``F_{i-1/2}`` is at offset ``-3`` from the cell index.
    Arithmetic lives in ``_weno_flux_hydro_pallas_rhs_local``.
    """
    if not _hydro_pallas_flux_supported(conserved_state, config):
        raise RuntimeError(
            "_weno_flux_hydro_pallas_rhs called when Pallas WENO is unsupported."
        )

    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    halo_list = [0, 0, 0]
    if 0 <= int(axis) < ndim:
        halo_list[int(axis)] = 3
    halo = tuple(halo_list[:ndim])

    if rhs_accumulator is None:
        def _local(state_local):
            return _weno_flux_hydro_pallas_rhs_local(
                state_local, dt_over_dx, params, config, registered_variables,
                axis=axis, rhs_accumulator=None,
            )
        return _pallas_call_sharded(
            _local,
            state_inputs=(conserved_state,),
            halo=halo,
            block_shape=block_shape[:ndim],
        )

    def _local(rhs_local, state_local):
        return _weno_flux_hydro_pallas_rhs_local(
            state_local, dt_over_dx, params, config, registered_variables,
            axis=axis, rhs_accumulator=rhs_local,
        )
    return _pallas_call_sharded(
        _local,
        state_inputs=(rhs_accumulator, conserved_state),
        halo=halo,
        block_shape=block_shape[:ndim],
    )


def _weno_flux_hydro_pallas_rhs_local(
    conserved_state,
    dt_over_dx,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
    rhs_accumulator=None,
):
    """Single-shard fused WENO + divergence kernel build."""
    accumulate = rhs_accumulator is not None

    ndim = int(config.dimensionality)
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    ncomp = len(local_indices)
    num_modes = ndim + 2
    epsilon = 1e-7
    tiny = 1e-14

    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        in_state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(*refs):
        # The kernel accepts either 5 inputs (no accumulator) or 6 inputs
        # (with accumulator).  Both layouts end with the dt-over-dx scalar and
        # an output ref; the accumulator, when present, comes first so it can
        # be aliased to the output via ``input_output_aliases``.
        if accumulate:
            rhs_in_ref, q_ref, gamma_ref, rhomin_ref, pgmin_ref, dtdx_ref, rhs_out_ref = refs
        else:
            q_ref, gamma_ref, rhomin_ref, pgmin_ref, dtdx_ref, rhs_out_ref = refs
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

        gamma = gamma_ref[()]
        gm1 = gamma - 1.0
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]
        dtdx = dtdx_ref[()]

        def q_at(var_index: int, offset: int):
            if ndim == 1:
                return q_ref[var_index, (ii + offset) % nx]
            if ndim == 2:
                if axis == 0:
                    return q_ref[var_index, (ii + offset) % nx, jj]
                return q_ref[var_index, ii, (jj + offset) % ny]
            if axis == 0:
                return q_ref[var_index, (ii + offset) % nx, jj, kk]
            if axis == 1:
                return q_ref[var_index, ii, (jj + offset) % ny, kk]
            return q_ref[var_index, ii, jj, (kk + offset) % nz]

        def q_local(offset: int):
            return tuple(q_at(idx, offset) for idx in local_indices)

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

        def lambda_from_floored_cell(cell, mode: int):
            vn = cell[5]
            c = cell[11]
            if mode == 0:
                return vn - c
            if mode == num_modes - 1:
                return vn + c
            return vn

        # Pre-compute the union of the two interface stencils once.  The
        # left interface at ``i - 1/2`` needs cells at offsets ``-3..2`` and
        # the right interface at ``i + 1/2`` needs cells at offsets ``-2..3``,
        # so jointly we need offsets ``-3..3`` — seven cells per output
        # block.  Sharing the heavy ``primitive_from_q`` / ``floored_cell`` /
        # ``flux_from_q`` work across both flux evaluations cuts the
        # per-block compute roughly in half compared to evaluating each
        # interface independently.
        shared_q = tuple(q_local(off) for off in range(-3, 4))
        shared_f = tuple(flux_from_q(q) for q in shared_q)
        shared_floored = tuple(floored_cell(q) for q in shared_q)

        def compute_interface_flux(stencil_offset: int):
            """Compute the WENO interface flux ``F_{i + stencil_offset + 1/2}``.

            ``stencil_offset == 0`` evaluates ``F_{i+1/2}`` (left/right cells at
            offsets 0 and 1); ``stencil_offset == -1`` evaluates ``F_{i-1/2}``
            (left/right cells at offsets -1 and 0).  Returns a tuple of
            ``ncomp`` Pallas tiles, one per local Euler component slot.
            """
            # ``shared_*`` is indexed by absolute offset ``-3..3`` (i.e. slot
            # ``off + 3``).  The WENO stencil for this interface uses the six
            # cells at offsets ``stencil_offset - 2 .. stencil_offset + 3``.
            base = stencil_offset + 3 - 2  # absolute index of the first stencil cell
            q_stencil = tuple(shared_q[base + k] for k in range(6))
            f_stencil = tuple(shared_f[base + k] for k in range(6))
            floored_stencil = tuple(shared_floored[base + k] for k in range(6))

            cell_l = floored_stencil[2]
            cell_r = floored_stencil[3]
            (rho_i, mn_i, mt1_i, mt2_i, energy_i,
             vn_i, vt1_i, vt2_i, v2_i, p_i, h_i, c_i) = cell_l
            (rho_j, mn_j, mt1_j, mt2_j, energy_j,
             vn_j, vt1_j, vt2_j, v2_j, p_j, h_j, c_j) = cell_r
            rho_face = jnp.maximum(
                0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)),
                rhomin,
            )
            vn_face = 0.5 * (mn_i + mn_j) / rho_face
            vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
            vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
            h_face = 0.5 * (h_i + h_j)
            v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
            c2_face = gm1 * (h_face - 0.5 * v2_face)
            c_face = jnp.sqrt(jnp.maximum(c2_face, 1e-12))
            inv_c2 = jnp.where(c2_face > 0.0, 1.0 / c2_face, 0.0)

            def left_project(mode, values):
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

            def add_right_correction(flux_acc, mode, Fs):
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

            def alpha_for_mode(mode):
                amx = jnp.abs(lambda_from_floored_cell(floored_stencil[0], mode))
                for k in range(1, 6):
                    amx = jnp.maximum(
                        amx,
                        jnp.abs(lambda_from_floored_cell(floored_stencil[k], mode)),
                    )
                return amx

            flux_acc = [
                (
                    -f_stencil[1][slot]
                    + 7.0 * f_stencil[2][slot]
                    + 7.0 * f_stencil[3][slot]
                    - f_stencil[4][slot]
                )
                / 12.0
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

            return flux_acc

        flux_right = compute_interface_flux(0)   # F_{i+1/2}
        flux_left = compute_interface_flux(-1)   # F_{i-1/2}

        # local_indices covers every conserved component for hydro, so a
        # blanket zeroing pass is unnecessary; we set every output slot below.
        if accumulate:
            for slot, var in enumerate(local_indices):
                prior = rhs_in_ref[var, ...]
                rhs_out_ref[var, ...] = prior + (-dtdx) * (flux_right[slot] - flux_left[slot])
        else:
            for slot, var in enumerate(local_indices):
                rhs_out_ref[var, ...] = -dtdx * (flux_right[slot] - flux_left[slot])

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    if accumulate:
        # Same BlockSpec layout as the state/output (full conserved-variable
        # axis, blocked over spatial dims).  XLA is told to reuse the
        # accumulator buffer for the output so the RHS lives in a single
        # physical buffer across all three axis calls.
        rhs_in_spec = pl.BlockSpec(
            block_shape if not isinstance(block_shape, tuple)
            else block_shape,
            (
                (lambda bi, bj, bk: (0, bi))
                if ndim == 1
                else (lambda bi, bj, bk: (0, bi, bj))
                if ndim == 2
                else (lambda bi, bj, bk: (0, bi, bj, bk))
            ),
        )
        in_specs = [rhs_in_spec, in_state_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec]
        kernel_args = (
            rhs_accumulator,
            conserved_state,
            jnp.asarray(params.gamma, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
            jnp.asarray(dt_over_dx, dtype=conserved_state.dtype),
        )
        kwargs["input_output_aliases"] = {0: 0}
    else:
        in_specs = [in_state_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec]
        kernel_args = (
            conserved_state,
            jnp.asarray(params.gamma, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_density, dtype=conserved_state.dtype),
            jnp.asarray(params.minimum_pressure, dtype=conserved_state.dtype),
            jnp.asarray(dt_over_dx, dtype=conserved_state.dtype),
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=config.pallas_interpret,
        name=f"hydro_weno_rhs_axis_{axis}",
        **kwargs,
    )(*kernel_args)

