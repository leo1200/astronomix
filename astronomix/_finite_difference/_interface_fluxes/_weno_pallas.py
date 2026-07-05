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

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL, PALLAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
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


# Use the hand-derived explicit MHD WENO window adjoint
# (``_weno_mhd_flux_from_window_adjoint``) in the Pallas MHD backward kernel
# instead of an in-kernel jax.vjp of the whole window.  The whole-window vjp
# made the MHD backward ~63x the forward (vs ~9x for the hydro hand adjoint);
# the explicit transpose restores parity.  Set to ``False`` to fall back to the
# legacy jax.vjp path (kept for cross-checking / regression safety).
_MHD_VJP_USE_HAND_ADJOINT = True

# Within ``_weno_mhd_flux_from_window_adjoint`` the three residual in-kernel
# ``jax.vjp`` calls (the eigenstructure building-block map ``_eigen_bb``, and the
# per-mode R_col / L_row scalar functionals) are replaced by fully hand-derived
# elementwise transposes when this is ``True``.  ``False`` selects the HYBRID
# path (hand-transposed everywhere EXCEPT those three small local vjps).  Both
# are bit-exact in interpret mode AND measure IDENTICAL runtime (18.8x
# backward/forward at 64^3): a same-process A/B confirmed XLA/Triton already
# lowers the tiny per-mode scalar vjps as efficiently as the hand transpose, so
# the residual gap vs hydro's ~9x is the shared WENO-recon reverse (intrinsic to
# MHD's larger 7-mode/8-var system), NOT these vjps.  DEFAULT = HYBRID (simpler,
# equal speed, fewer lines); set ``True`` only to drop the in-kernel-autodiff
# dependency for jax/Triton-version robustness.
_MHD_VJP_FULL_HANDDERIVE = False


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


def _weno_hydro_flux_from_window(q_stencil, gamma, rhomin, pgmin, ncomp, num_modes):
    """Pure per-interface hydro-WENO flux from a gathered 6-cell stencil.

    ``q_stencil`` is the tuple ``(q[-2], q[-1], q[0], q[+1], q[+2], q[+3])``
    where each entry is a length-``ncomp`` tuple of the local conserved
    components (density, normal momentum, transverse momenta..., energy) in the
    per-axis characteristic order.  Returns the length-``ncomp`` list of WENO
    interface fluxes ``flux_acc`` at ``i + 1/2``.

    This is the *single source of truth* for the WENO arithmetic: the forward
    Pallas kernel gathers ``q_stencil`` from ``q_ref`` and calls this; the
    adjoint kernel gathers the same stencil and calls ``jax.vjp`` of this, so
    the Pallas backward is the exact transpose of the Pallas forward by
    construction (no separately-derived adjoint math).  Every operation here is
    elementwise on the gathered arrays — no ref reads, slices or rolls — which
    is what lets ``jax.vjp`` lower inside the Triton kernel.
    """
    gm1 = gamma - 1.0
    epsilon = 1e-7
    tiny = 1e-14

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

    # Interface eigenvector building blocks at i + 1/2.
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

    return flux_acc


def _weno_hydro_flux_from_window_adjoint(
    q_stencil, flux_bar, gamma, rhomin, pgmin, ncomp, num_modes
):
    """Explicit reverse pass (vector-Jacobian product) of
    :func:`_weno_hydro_flux_from_window`.

    Given the 6-cell window ``q_stencil`` and the output cotangent ``flux_bar``
    (length ``ncomp``) returns ``qbar_stencil`` — a length-6 list of length-
    ``ncomp`` lists, the cotangent w.r.t. every stencil input.  This is a
    HAND-DERIVED transpose written as plain elementwise arithmetic (no
    ``jax.vjp``): the auto-generated VJP of the full WENO window is miscompiled
    and slow to compile on the Triton GPU backend, whereas this explicit form
    is ordinary arithmetic that Pallas/Triton lowers reliably and fast.  It is
    validated bit-exact (~1e-15) against ``jax.vjp`` of the forward window in
    ``pytests/pallas/_weno_window_adjoint_check.py``.
    """
    gm1 = gamma - 1.0
    epsilon = 1e-7
    tiny = 1e-14

    # ---- forward per-cell maps (recomputed; needed for the reverse pass) ----
    def primitive_from_q(q):
        rho = q[0]; mn = q[1]
        if ncomp == 3:
            mt1 = 0.0; mt2 = 0.0; energy = q[2]
        elif ncomp == 4:
            mt1 = q[2]; mt2 = 0.0; energy = q[3]
        else:
            mt1 = q[2]; mt2 = q[3]; energy = q[4]
        inv_rho = 1.0 / rho
        vn = mn * inv_rho; vt1 = mt1 * inv_rho; vt2 = mt2 * inv_rho
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        pressure = gm1 * (energy - 0.5 * rho * v2)
        return rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure

    def floored_cell(q):
        rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
        troubled = (rho < rhomin) | (pressure < pgmin)
        rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
        pressure_f = jnp.where(troubled, jnp.maximum(pressure, pgmin), pressure)
        energy_f = jnp.where(troubled, pressure_f / gm1 + 0.5 * rho_f * v2, energy)
        h = (energy_f + pressure_f) / rho_f
        c = jnp.sqrt(jnp.maximum(gamma * jnp.abs(pressure_f / rho_f), 1e-12))
        return rho_f, mn, mt1, mt2, energy_f, vn, vt1, vt2, v2, pressure_f, h, c

    def flux_from_q(q):
        rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
        if ncomp == 3:
            return (mn, mn * vn + pressure, (energy + pressure) * vn)
        if ncomp == 4:
            return (mn, mn * vn + pressure, mt1 * vn, (energy + pressure) * vn)
        return (mn, mn * vn + pressure, mt1 * vn, mt2 * vn, (energy + pressure) * vn)

    # ---- per-cell adjoints ----
    def primitive_from_q_adj(q, bars):
        rho = q[0]; mn = q[1]
        if ncomp == 3:
            mt1 = 0.0; mt2 = 0.0
        elif ncomp == 4:
            mt1 = q[2]; mt2 = 0.0
        else:
            mt1 = q[2]; mt2 = q[3]
        inv_rho = 1.0 / rho
        vn = mn * inv_rho; vt1 = mt1 * inv_rho; vt2 = mt2 * inv_rho
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        (b_rho, b_mn, b_mt1, b_mt2, b_energy, b_vn, b_vt1, b_vt2, b_v2, b_pressure) = bars
        b_energy = b_energy + gm1 * b_pressure
        b_rho = b_rho + gm1 * (-0.5 * v2) * b_pressure
        b_v2 = b_v2 + gm1 * (-0.5 * rho) * b_pressure
        b_vn = b_vn + 2.0 * vn * b_v2
        b_vt1 = b_vt1 + 2.0 * vt1 * b_v2
        b_vt2 = b_vt2 + 2.0 * vt2 * b_v2
        b_inv_rho = mt2 * b_vt2 + mt1 * b_vt1 + mn * b_vn
        b_mt2 = b_mt2 + inv_rho * b_vt2
        b_mt1 = b_mt1 + inv_rho * b_vt1
        b_mn = b_mn + inv_rho * b_vn
        b_rho = b_rho + (-inv_rho * inv_rho) * b_inv_rho
        if ncomp == 3:
            return [b_rho, b_mn, b_energy]
        if ncomp == 4:
            return [b_rho, b_mn, b_mt1, b_energy]
        return [b_rho, b_mn, b_mt1, b_mt2, b_energy]

    def flux_from_q_adj(q, fbar):
        rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
        bd = dict(rho=0.0, mn=0.0, mt1=0.0, mt2=0.0, energy=0.0, vn=0.0,
                  vt1=0.0, vt2=0.0, v2=0.0, pressure=0.0)
        bd['mn'] += fbar[0]
        bd['mn'] += fbar[1] * vn; bd['vn'] += fbar[1] * mn; bd['pressure'] += fbar[1]
        if ncomp == 3:
            bd['energy'] += fbar[2] * vn; bd['pressure'] += fbar[2] * vn; bd['vn'] += fbar[2] * (energy + pressure)
        elif ncomp == 4:
            bd['mt1'] += fbar[2] * vn; bd['vn'] += fbar[2] * mt1
            bd['energy'] += fbar[3] * vn; bd['pressure'] += fbar[3] * vn; bd['vn'] += fbar[3] * (energy + pressure)
        else:
            bd['mt1'] += fbar[2] * vn; bd['vn'] += fbar[2] * mt1
            bd['mt2'] += fbar[3] * vn; bd['vn'] += fbar[3] * mt2
            bd['energy'] += fbar[4] * vn; bd['pressure'] += fbar[4] * vn; bd['vn'] += fbar[4] * (energy + pressure)
        bars = (bd['rho'], bd['mn'], bd['mt1'], bd['mt2'], bd['energy'], bd['vn'],
                bd['vt1'], bd['vt2'], bd['v2'], bd['pressure'])
        return primitive_from_q_adj(q, bars)

    def floored_cell_adj(q, bars12):
        rho, mn, mt1, mt2, energy, vn, vt1, vt2, v2, pressure = primitive_from_q(q)
        troubled = (rho < rhomin) | (pressure < pgmin)
        rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
        pressure_f = jnp.where(troubled, jnp.maximum(pressure, pgmin), pressure)
        energy_f = jnp.where(troubled, pressure_f / gm1 + 0.5 * rho_f * v2, energy)
        ratio = pressure_f / rho_f
        aval = gamma * jnp.abs(ratio)
        c = jnp.sqrt(jnp.maximum(aval, 1e-12))
        (b_rho_f, b_mn, b_mt1, b_mt2, b_energy_f, b_vn, b_vt1, b_vt2, b_v2,
         b_pressure_f, b_h, b_c) = bars12
        b_arg = b_c * 0.5 / c
        b_aval = jnp.where(aval > 1e-12, b_arg, 0.0)
        b_ratio = b_aval * gamma * jnp.sign(ratio)
        b_pressure_f = b_pressure_f + b_ratio / rho_f
        b_rho_f = b_rho_f + b_ratio * (-ratio / rho_f)
        b_energy_f = b_energy_f + b_h / rho_f
        b_pressure_f = b_pressure_f + b_h / rho_f
        b_rho_f = b_rho_f + b_h * (-(energy_f + pressure_f) / (rho_f * rho_f))
        b_pressure_f = b_pressure_f + jnp.where(troubled, b_energy_f / gm1, 0.0)
        b_rho_f = b_rho_f + jnp.where(troubled, b_energy_f * 0.5 * v2, 0.0)
        b_v2 = b_v2 + jnp.where(troubled, b_energy_f * 0.5 * rho_f, 0.0)
        b_energy = jnp.where(troubled, 0.0, b_energy_f)
        b_pressure = b_pressure_f * jnp.where(troubled, jnp.where(pressure > pgmin, 1.0, 0.0), 1.0)
        b_rho = b_rho_f * jnp.where(troubled, jnp.where(rho > rhomin, 1.0, 0.0), 1.0)
        bars = (b_rho, b_mn, b_mt1, b_mt2, b_energy, b_vn, b_vt1, b_vt2, b_v2, b_pressure)
        return primitive_from_q_adj(q, bars)

    def left_project_fwd(mode, values, fp):
        vnf, vt1f, vt2f, cf, v2f, c2f, inv_c2 = (fp['vnf'], fp['vt1f'], fp['vt2f'],
                                                 fp['cf'], fp['v2f'], fp['c2f'], fp['inv_c2'])
        if mode == 0:
            acc = (0.5 * gm1 * v2f + vnf * cf) * values[0] - (gm1 * vnf + cf) * values[1]
            if ncomp == 3:
                acc = acc + gm1 * values[2]
            elif ncomp == 4:
                acc = acc - gm1 * vt1f * values[2] + gm1 * values[3]
            else:
                acc = acc - gm1 * vt1f * values[2] - gm1 * vt2f * values[3] + gm1 * values[4]
            return 0.5 * inv_c2 * acc
        if mode == 1:
            acc = (c2f - 0.5 * gm1 * v2f) * values[0] + gm1 * vnf * values[1]
            if ncomp == 3:
                acc = acc - gm1 * values[2]
            elif ncomp == 4:
                acc = acc + gm1 * vt1f * values[2] - gm1 * values[3]
            else:
                acc = acc + gm1 * vt1f * values[2] + gm1 * vt2f * values[3] - gm1 * values[4]
            return inv_c2 * acc
        if mode == 2 and ncomp >= 4:
            return -vt1f * values[0] + values[2]
        if mode == 3 and ncomp == 5:
            return -vt2f * values[0] + values[3]
        acc = (0.5 * gm1 * v2f - vnf * cf) * values[0] - (gm1 * vnf - cf) * values[1]
        if ncomp == 3:
            acc = acc + gm1 * values[2]
        elif ncomp == 4:
            acc = acc - gm1 * vt1f * values[2] + gm1 * values[3]
        else:
            acc = acc - gm1 * vt1f * values[2] - gm1 * vt2f * values[3] + gm1 * values[4]
        return 0.5 * inv_c2 * acc

    def left_project_adj(mode, values, fp, s_bar, vbar, fpbar):
        vnf, vt1f, vt2f, cf, v2f, c2f, inv_c2 = (fp['vnf'], fp['vt1f'], fp['vt2f'],
                                                 fp['cf'], fp['v2f'], fp['c2f'], fp['inv_c2'])
        v = values
        is_last = not ((mode == 0) or (mode == 1) or (mode == 2 and ncomp >= 4)
                       or (mode == 3 and ncomp == 5))
        if mode == 0 or is_last:
            sgn = 1.0 if mode == 0 else -1.0
            A0 = 0.5 * gm1 * v2f + sgn * vnf * cf
            A1 = -(gm1 * vnf + sgn * cf)
            k = 0.5 * inv_c2
            acc = A0 * v[0] + A1 * v[1]
            if ncomp == 3:
                acc = acc + gm1 * v[2]
            elif ncomp == 4:
                acc = acc - gm1 * vt1f * v[2] + gm1 * v[3]
            else:
                acc = acc - gm1 * vt1f * v[2] - gm1 * vt2f * v[3] + gm1 * v[4]
            vbar[0] += s_bar * k * A0
            vbar[1] += s_bar * k * A1
            if ncomp == 3:
                vbar[2] += s_bar * k * gm1
            elif ncomp == 4:
                vbar[2] += s_bar * k * (-gm1 * vt1f); vbar[3] += s_bar * k * gm1
            else:
                vbar[2] += s_bar * k * (-gm1 * vt1f); vbar[3] += s_bar * k * (-gm1 * vt2f); vbar[4] += s_bar * k * gm1
            fpbar['inv_c2'] += s_bar * 0.5 * acc
            fpbar['v2f'] += s_bar * k * 0.5 * gm1 * v[0]
            fpbar['vnf'] += s_bar * k * (sgn * cf * v[0] - gm1 * v[1])
            fpbar['cf'] += s_bar * k * (sgn * vnf * v[0] - sgn * v[1])
            if ncomp >= 4:
                fpbar['vt1f'] += s_bar * k * (-gm1 * v[2])
            if ncomp == 5:
                fpbar['vt2f'] += s_bar * k * (-gm1 * v[3])
        elif mode == 1:
            B0 = c2f - 0.5 * gm1 * v2f
            acc = B0 * v[0] + gm1 * vnf * v[1]
            if ncomp == 3:
                acc = acc - gm1 * v[2]
            elif ncomp == 4:
                acc = acc + gm1 * vt1f * v[2] - gm1 * v[3]
            else:
                acc = acc + gm1 * vt1f * v[2] + gm1 * vt2f * v[3] - gm1 * v[4]
            vbar[0] += s_bar * inv_c2 * B0
            vbar[1] += s_bar * inv_c2 * gm1 * vnf
            if ncomp == 3:
                vbar[2] += s_bar * inv_c2 * (-gm1)
            elif ncomp == 4:
                vbar[2] += s_bar * inv_c2 * gm1 * vt1f; vbar[3] += s_bar * inv_c2 * (-gm1)
            else:
                vbar[2] += s_bar * inv_c2 * gm1 * vt1f; vbar[3] += s_bar * inv_c2 * gm1 * vt2f; vbar[4] += s_bar * inv_c2 * (-gm1)
            fpbar['inv_c2'] += s_bar * acc
            fpbar['c2f'] += s_bar * inv_c2 * v[0]
            fpbar['v2f'] += s_bar * inv_c2 * (-0.5 * gm1 * v[0])
            fpbar['vnf'] += s_bar * inv_c2 * gm1 * v[1]
            if ncomp >= 4:
                fpbar['vt1f'] += s_bar * inv_c2 * gm1 * v[2]
            if ncomp == 5:
                fpbar['vt2f'] += s_bar * inv_c2 * gm1 * v[3]
        elif mode == 2 and ncomp >= 4:
            vbar[0] += s_bar * (-vt1f); vbar[2] += s_bar
            fpbar['vt1f'] += s_bar * (-v[0])
        elif mode == 3 and ncomp == 5:
            vbar[0] += s_bar * (-vt2f); vbar[3] += s_bar
            fpbar['vt2f'] += s_bar * (-v[0])

    def weno_recon_fwd(aterm, bterm, cterm, dterm):
        IS0 = 13.0 * (aterm - bterm) ** 2 + 3.0 * (aterm - 3.0 * bterm) ** 2
        IS1 = 13.0 * (bterm - cterm) ** 2 + 3.0 * (bterm + cterm) ** 2
        IS2 = 13.0 * (cterm - dterm) ** 2 + 3.0 * (3.0 * cterm - dterm) ** 2
        a0 = 1.0 / (epsilon + IS0) ** 2; a1 = 6.0 / (epsilon + IS1) ** 2; a2 = 3.0 / (epsilon + IS2) ** 2
        asum = jnp.maximum(a0 + a1 + a2, tiny)
        om0 = a0 / asum; om2 = a2 / asum
        return om0 * (aterm - 2.0 * bterm + cterm) / 3.0 + (om2 - 0.5) * (bterm - 2.0 * cterm + dterm) / 6.0

    def weno_recon_adj(aterm, bterm, cterm, dterm, recon_bar):
        IS0 = 13.0 * (aterm - bterm) ** 2 + 3.0 * (aterm - 3.0 * bterm) ** 2
        IS1 = 13.0 * (bterm - cterm) ** 2 + 3.0 * (bterm + cterm) ** 2
        IS2 = 13.0 * (cterm - dterm) ** 2 + 3.0 * (3.0 * cterm - dterm) ** 2
        e0 = epsilon + IS0; e1 = epsilon + IS1; e2 = epsilon + IS2
        a0 = 1.0 / e0 ** 2; a1 = 6.0 / e1 ** 2; a2 = 3.0 / e2 ** 2
        s3 = a0 + a1 + a2; asum = jnp.maximum(s3, tiny)
        om0 = a0 / asum; om2 = a2 / asum
        P0 = (aterm - 2.0 * bterm + cterm) / 3.0
        P2 = (bterm - 2.0 * cterm + dterm) / 6.0
        ab = bb = cb = db = 0.0
        om0_bar = recon_bar * P0; P0_bar = recon_bar * om0
        om2_bar = recon_bar * P2; P2_bar = recon_bar * (om2 - 0.5)
        ab += P0_bar / 3.0; bb += -2.0 * P0_bar / 3.0; cb += P0_bar / 3.0
        bb += P2_bar / 6.0; cb += -2.0 * P2_bar / 6.0; db += P2_bar / 6.0
        a0_bar = om0_bar / asum; asum_bar = om0_bar * (-a0 / asum ** 2)
        a2_bar = om2_bar / asum; asum_bar += om2_bar * (-a2 / asum ** 2)
        s3_bar = jnp.where(s3 > tiny, asum_bar, 0.0)
        a0_bar += s3_bar; a1_bar = s3_bar; a2_bar += s3_bar
        IS0_bar = a0_bar * (-2.0) * e0 ** (-3)
        IS1_bar = a1_bar * 6.0 * (-2.0) * e1 ** (-3)
        IS2_bar = a2_bar * 3.0 * (-2.0) * e2 ** (-3)
        ab += IS0_bar * (26.0 * (aterm - bterm) + 6.0 * (aterm - 3.0 * bterm))
        bb += IS0_bar * (-26.0 * (aterm - bterm) - 18.0 * (aterm - 3.0 * bterm))
        bb += IS1_bar * (26.0 * (bterm - cterm) + 6.0 * (bterm + cterm))
        cb += IS1_bar * (-26.0 * (bterm - cterm) + 6.0 * (bterm + cterm))
        cb += IS2_bar * (26.0 * (cterm - dterm) + 18.0 * (3.0 * cterm - dterm))
        db += IS2_bar * (-26.0 * (cterm - dterm) - 6.0 * (3.0 * cterm - dterm))
        return ab, bb, cb, db

    def lam_of(cell, mode):
        vn = cell[5]; c = cell[11]
        if mode == 0:
            return vn - c
        if mode == num_modes - 1:
            return vn + c
        return vn

    # ---- forward recompute of shared intermediates ----
    f_st = [flux_from_q(q) for q in q_stencil]
    fl_st = [floored_cell(q) for q in q_stencil]
    cl, cr = fl_st[2], fl_st[3]
    rho_i, mn_i, mt1_i, mt2_i, e_i, vn_i, vt1_i, vt2_i, v2_i, p_i, h_i, c_i = cl
    rho_j, mn_j, mt1_j, mt2_j, e_j, vn_j, vt1_j, vt2_j, v2_j, p_j, h_j, c_j = cr
    rho_face = jnp.maximum(0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)), rhomin)
    vnf = 0.5 * (mn_i + mn_j) / rho_face
    vt1f = 0.5 * (mt1_i + mt1_j) / rho_face
    vt2f = 0.5 * (mt2_i + mt2_j) / rho_face
    hf = 0.5 * (h_i + h_j)
    v2f = vnf * vnf + vt1f * vt1f + vt2f * vt2f
    c2f = gm1 * (hf - 0.5 * v2f)
    cf = jnp.sqrt(jnp.maximum(c2f, 1e-12))
    inv_c2 = jnp.where(c2f > 0.0, 1.0 / c2f, 0.0)
    fp = dict(vnf=vnf, vt1f=vt1f, vt2f=vt2f, cf=cf, v2f=v2f, c2f=c2f, inv_c2=inv_c2, hf=hf)

    qbar = [[0.0] * ncomp for _ in range(6)]
    fbar_cell = [[0.0] * ncomp for _ in range(6)]
    fp_bar = dict(vnf=0.0, vt1f=0.0, vt2f=0.0, cf=0.0, v2f=0.0, c2f=0.0, inv_c2=0.0, hf=0.0)
    fl_bar = [[0.0] * 12 for _ in range(6)]

    cc = [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12]
    for j, k in enumerate((1, 2, 3, 4)):
        for slot in range(ncomp):
            fbar_cell[k][slot] += flux_bar[slot] * cc[j]

    last = ncomp - 1
    for mode in range(num_modes):
        s = [left_project_fwd(mode, f_st[k], fp) for k in range(6)]
        qp = [left_project_fwd(mode, q_stencil[k], fp) for k in range(6)]
        d = [s[i + 1] - s[i] for i in range(5)]
        dq = [qp[i + 1] - qp[i] for i in range(5)]
        lams = [lam_of(fl_st[k], mode) for k in range(6)]
        absl = [jnp.abs(x) for x in lams]
        amxs = [absl[0]]
        for k in range(1, 6):
            amxs.append(jnp.maximum(amxs[-1], absl[k]))
        amx = amxs[-1]
        ap = 0.5 * (d[0] + amx * dq[0]); bp = 0.5 * (d[1] + amx * dq[1])
        cp = 0.5 * (d[2] + amx * dq[2]); dp = 0.5 * (d[3] + amx * dq[3])
        am = 0.5 * (d[4] - amx * dq[4]); bm = 0.5 * (d[3] - amx * dq[3])
        cm = 0.5 * (d[2] - amx * dq[2]); dm = 0.5 * (d[1] - amx * dq[1])
        second = weno_recon_fwd(ap, bp, cp, dp)
        third = weno_recon_fwd(am, bm, cm, dm)
        Fs = -second + third

        if mode == 0:
            R = [1.0, vnf - cf] + ([vt1f] if ncomp >= 4 else []) + ([vt2f] if ncomp == 5 else []) + [hf - vnf * cf]
        elif mode == 1:
            R = [1.0, vnf] + ([vt1f] if ncomp >= 4 else []) + ([vt2f] if ncomp == 5 else []) + [0.5 * v2f]
        elif mode == 2 and ncomp >= 4:
            R = [0.0, 0.0, 1.0] + ([0.0] if ncomp == 5 else []) + [vt1f]
        elif mode == 3 and ncomp == 5:
            R = [0.0, 0.0, 0.0, 1.0, vt2f]
        else:
            R = [1.0, vnf + cf] + ([vt1f] if ncomp >= 4 else []) + ([vt2f] if ncomp == 5 else []) + [hf + vnf * cf]
        Fs_bar = sum(flux_bar[slot] * R[slot] for slot in range(ncomp))
        is_last = not ((mode == 0) or (mode == 1) or (mode == 2 and ncomp >= 4) or (mode == 3 and ncomp == 5))
        if mode == 0 or is_last:
            rs = -1.0 if mode == 0 else 1.0
            fp_bar['vnf'] += Fs * flux_bar[1] * 1.0
            fp_bar['cf'] += Fs * flux_bar[1] * rs
            fp_bar['hf'] += Fs * flux_bar[last] * 1.0
            fp_bar['vnf'] += Fs * flux_bar[last] * (rs * cf)
            fp_bar['cf'] += Fs * flux_bar[last] * (rs * vnf)
            if ncomp >= 4:
                fp_bar['vt1f'] += Fs * flux_bar[2]
            if ncomp == 5:
                fp_bar['vt2f'] += Fs * flux_bar[3]
        elif mode == 1:
            fp_bar['vnf'] += Fs * flux_bar[1]
            fp_bar['v2f'] += Fs * flux_bar[last] * 0.5
            if ncomp >= 4:
                fp_bar['vt1f'] += Fs * flux_bar[2]
            if ncomp == 5:
                fp_bar['vt2f'] += Fs * flux_bar[3]
        elif mode == 2 and ncomp >= 4:
            fp_bar['vt1f'] += Fs * flux_bar[last]
        elif mode == 3 and ncomp == 5:
            fp_bar['vt2f'] += Fs * flux_bar[last]

        second_bar = -Fs_bar; third_bar = Fs_bar
        ap_b, bp_b, cp_b, dp_b = weno_recon_adj(ap, bp, cp, dp, second_bar)
        am_b, bm_b, cm_b, dm_b = weno_recon_adj(am, bm, cm, dm, third_bar)
        d_b = [0.0] * 5; dq_b = [0.0] * 5; amx_b = 0.0
        for (tb, di) in ((ap_b, 0), (bp_b, 1), (cp_b, 2), (dp_b, 3)):
            d_b[di] += 0.5 * tb; dq_b[di] += 0.5 * amx * tb; amx_b += 0.5 * dq[di] * tb
        for (tb, di) in ((am_b, 4), (bm_b, 3), (cm_b, 2), (dm_b, 1)):
            d_b[di] += 0.5 * tb; dq_b[di] += -0.5 * amx * tb; amx_b += -0.5 * dq[di] * tb
        s_b = [0.0] * 6; qp_b = [0.0] * 6
        for i in range(5):
            s_b[i + 1] += d_b[i]; s_b[i] += -d_b[i]
            qp_b[i + 1] += dq_b[i]; qp_b[i] += -dq_b[i]
        for k in range(6):
            left_project_adj(mode, f_st[k], fp, s_b[k], fbar_cell[k], fp_bar)
            left_project_adj(mode, q_stencil[k], fp, qp_b[k], qbar[k], fp_bar)
        # amx = max_k |lambda_k|, built as a fold of jnp.maximum.  Reverse the
        # fold exactly as jax does (lax.max sends the cotangent to the SECOND
        # operand on ties) so the sub-gradient matches the native VJP at
        # eigenvalue ties -- e.g. the u=0 contact wave of a shock tube, where
        # all |lambda|=0 and the naive "route to every argmax" is wrong.
        acc = amx_b
        absl_bar = [0.0] * 6
        for k in range(5, 0, -1):
            prev_gets = amxs[k - 1] > absl[k]
            absl_bar[k] = absl_bar[k] + jnp.where(prev_gets, 0.0, acc)
            acc = jnp.where(prev_gets, acc, 0.0)
        absl_bar[0] = absl_bar[0] + acc
        for k in range(6):
            lam_b = absl_bar[k] * jnp.sign(lams[k])
            fl_bar[k][5] += lam_b
            if mode == 0:
                fl_bar[k][11] += lam_b * (-1.0)
            elif mode == num_modes - 1:
                fl_bar[k][11] += lam_b * (1.0)

    # ---- face quantities adjoint -> floored cells 2, 3 ----
    vnf_b = fp_bar['vnf']; vt1f_b = fp_bar['vt1f']; vt2f_b = fp_bar['vt2f']
    cf_b = fp_bar['cf']; v2f_b = fp_bar['v2f']; c2f_b = fp_bar['c2f']
    inv_c2_b = fp_bar['inv_c2']; hf_b = fp_bar['hf']
    c2f_b += jnp.where(c2f > 0.0, -1.0 / c2f ** 2, 0.0) * inv_c2_b
    c2f_b += jnp.where(c2f > 1e-12, 0.5 / cf, 0.0) * cf_b
    hf_b += gm1 * c2f_b; v2f_b += gm1 * (-0.5) * c2f_b
    vnf_b += 2.0 * vnf * v2f_b; vt1f_b += 2.0 * vt1f * v2f_b; vt2f_b += 2.0 * vt2f * v2f_b
    fl_bar[2][10] += 0.5 * hf_b; fl_bar[3][10] += 0.5 * hf_b
    rho_face_b = 0.0
    num2 = 0.5 * (mt2_i + mt2_j); num2_b = vt2f_b / rho_face; rho_face_b += vt2f_b * (-num2 / rho_face ** 2)
    fl_bar[2][3] += 0.5 * num2_b; fl_bar[3][3] += 0.5 * num2_b
    num1 = 0.5 * (mt1_i + mt1_j); num1_b = vt1f_b / rho_face; rho_face_b += vt1f_b * (-num1 / rho_face ** 2)
    fl_bar[2][2] += 0.5 * num1_b; fl_bar[3][2] += 0.5 * num1_b
    numn = 0.5 * (mn_i + mn_j); numn_b = vnf_b / rho_face; rho_face_b += vnf_b * (-numn / rho_face ** 2)
    fl_bar[2][1] += 0.5 * numn_b; fl_bar[3][1] += 0.5 * numn_b
    inner = 0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin))
    inner_b = jnp.where(inner > rhomin, rho_face_b, 0.0)
    fl_bar[2][0] += 0.5 * jnp.where(rho_i > rhomin, 1.0, 0.0) * inner_b
    fl_bar[3][0] += 0.5 * jnp.where(rho_j > rhomin, 1.0, 0.0) * inner_b

    for k in range(6):
        gq_fl = floored_cell_adj(q_stencil[k], fl_bar[k])
        gq_fx = flux_from_q_adj(q_stencil[k], fbar_cell[k])
        for c in range(ncomp):
            qbar[k][c] = qbar[k][c] + gq_fl[c] + gq_fx[c]
    return qbar


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


def _weno_flux_hydro_pallas_vjp_local(
    conserved_state,
    flux_bar,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Native Pallas adjoint (VJP) of :func:`_weno_flux_hydro_pallas_local`.

    Given the primal input ``conserved_state`` and the output cotangent
    ``flux_bar`` (same shape as the WENO flux), returns the input cotangent
    ``conserved_state_bar`` w.r.t. ``conserved_state``.

    Strategy (see ``pytests/pallas/_vjp_in_kernel_spike.py`` and
    ``pytests/pallas/_weno_vjp_check.py``): each grid block gathers its 6-cell
    WENO stencil from ``q_ref`` exactly as the forward kernel does, runs
    ``jax.vjp`` of the *shared* per-window flux function
    :func:`_weno_hydro_flux_from_window` against the tile's flux cotangent.
    Because forward and adjoint call the same window function, the Pallas
    backward is the exact transpose of the Pallas forward by construction — no
    separately-derived adjoint math.  Assembling the input cotangent is a
    stencil scatter; rather than cross-block atomics (which Triton
    mis-accumulates across blocks) the kernel emits the six per-offset
    contributions into six non-overlapping BlockSpec-tiled buffers and the
    scatter is a plain-JAX ``roll``-and-sum afterwards.

    The kernel always gathers/scatters along the *leading* spatial axis: the
    in-kernel ``jax.vjp`` is bit-exact in Pallas interpret mode for every axis,
    but the Triton GPU lowering miscompiles it when the stencil offset is on a
    non-leading spatial axis (validated).  For ``axis != 0`` we therefore
    transpose the flux axis to the front — exactly what the native y/z flux
    does — run the (GPU-correct) axis-0 path, and transpose the result back.
    The momentum-component permutation lives in the variable axis
    (``local_indices``) and is unaffected by the spatial transpose.

    Single-device build (no ``shard_map``): the differentiable / inverse-problem
    regime runs on one GPU.
    """
    ndim = int(config.dimensionality)
    local_indices = _hydro_indices_for_axis(config, registered_variables, axis)
    li = jnp.asarray(local_indices)
    ncomp = len(local_indices)

    # Run the kernel on a fully identity-indexed, axis-0 gather: bring the flux
    # spatial axis to the front AND permute the conserved components into
    # characteristic order, both in plain JAX.  Triton only lowers the in-kernel
    # VJP correctly for this configuration — a non-leading gather axis OR a
    # non-identity component permutation inside the kernel is miscompiled on the
    # GPU backend (each is bit-exact in Pallas interpret mode, confirming the
    # kernel logic; the discrepancy is purely the Triton lowering).  Both
    # permutations are undone afterwards, also in plain JAX.
    if axis == 0:
        cs_s, fb_s, inv_perm = conserved_state, flux_bar, None
    else:
        perm = [0, axis + 1] + [a for a in range(1, ndim + 1) if a != axis + 1]
        inv_perm = [perm.index(i) for i in range(ndim + 1)]
        cs_s = jnp.transpose(conserved_state, perm)
        fb_s = jnp.transpose(flux_bar, perm)
    cs = cs_s[li]  # characteristic component order -> identity inside the kernel
    fb = fb_s[li]

    nvars = int(cs.shape[0])
    spatial_shape = tuple(int(x) for x in cs.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    num_modes = ndim + 2
    offsets = tuple(range(-2, 4))  # WENO5 stencil window

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    full_spec = pl.BlockSpec(cs.shape, lambda bi, bj, bk: tuple([0] * (ndim + 1)))
    if ndim == 1:
        block_shape = (nvars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
    elif ndim == 2:
        block_shape = (nvars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
    else:
        block_shape = (nvars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(q_ref, fbar_ref, gamma_ref, rhomin_ref, pgmin_ref, *out_refs):
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
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]

        def shifted_index(offset: int):
            # Always offset the leading spatial axis (the flux axis, brought to
            # the front by the transpose above).
            if ndim == 1:
                return ((ii + offset) % nx,)
            if ndim == 2:
                return ((ii + offset) % nx, jj)
            return ((ii + offset) % nx, jj, kk)

        def q_local(offset: int):
            idx = shifted_index(offset)
            return tuple(q_ref[(var,) + idx] for var in range(ncomp))

        q_stencil = tuple(q_local(o) for o in offsets)

        own = shifted_index(0)
        flux_bar_slot = tuple(fbar_ref[(var,) + own] for var in range(ncomp))

        # Hand-derived explicit adjoint (pure elementwise arithmetic, no
        # jax.vjp): Triton miscompiles / is very slow to compile the
        # auto-generated VJP of the full WENO window, so the backward uses the
        # validated explicit transpose instead.
        qbar_stencil = _weno_hydro_flux_from_window_adjoint(
            q_stencil, flux_bar_slot, gamma, rhomin, pgmin, ncomp, num_modes
        )

        for o_idx in range(len(offsets)):
            out_ref = out_refs[o_idx]
            for comp in range(ncomp):
                out_ref[comp, ...] = qbar_stencil[o_idx][comp]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shapes = tuple(jax.ShapeDtypeStruct(cs.shape, cs.dtype) for _ in offsets)
    contributions = pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        grid=grid,
        in_specs=[full_spec, full_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=tuple(out_spec for _ in offsets),
        interpret=config.pallas_interpret,
        name=f"hydro_weno_flux_vjp_axis_{axis}",
        **kwargs,
    )(
        cs,
        fb,
        jnp.asarray(params.gamma, dtype=cs.dtype),
        jnp.asarray(params.minimum_density, dtype=cs.dtype),
        jnp.asarray(params.minimum_pressure, dtype=cs.dtype),
    )

    # contributions[o] holds, at interface cell i, the cotangent destined for
    # source cell i + offset (along the leading spatial axis = array axis 1).
    # U_bar[j] = sum_o contributions[o][j - offset] = sum_o roll(., offset).
    ubar_char = sum(
        jnp.roll(contributions[o_idx], offset, axis=1)
        for o_idx, offset in enumerate(offsets)
    )
    # Undo the component permutation, then the spatial transpose (plain JAX).
    conserved_state_bar = jnp.zeros_like(cs_s).at[li].set(ubar_char)
    if inv_perm is not None:
        conserved_state_bar = jnp.transpose(conserved_state_bar, inv_perm)
    return conserved_state_bar


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


def _weno_mhd_flux_from_window(q_stencil, gamma, rhomin, pgmin, b_eps, sqrt_floor,
                              ncomp, num_modes):
    """Pure per-interface ideal-gas MHD WENO flux from a gathered 6-cell stencil.

    ``q_stencil`` is the tuple ``(q[-2], q[-1], q[0], q[+1], q[+2], q[+3])`` where
    each entry is a length-8 tuple of the local conserved components in per-axis
    characteristic order ``(rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy)``.  Returns
    the length-8 list of WENO interface fluxes ``flux_acc`` at ``i + 1/2``.

    Single source of truth for the MHD WENO arithmetic: the forward Pallas kernel
    gathers ``q_stencil`` from ``q_ref`` and calls this; the adjoint kernel
    gathers the same stencil and calls ``jax.vjp`` of this, so the Pallas
    backward is the exact transpose of the Pallas forward by construction (no
    separately-derived adjoint math).  Every operation is elementwise on the
    gathered arrays — no ref reads, slices or rolls — which is what lets
    ``jax.vjp`` lower inside the Triton kernel (validated bit-exact and
    compile-at-parity on jax >= 0.10; the old auto-VJP Triton miscompile that
    forced the hydro hand-derivation is gone).  ``b_eps`` and ``sqrt_floor`` are
    passed in as already-typed scalars (x64 + Triton dtype hygiene)."""
    gm1 = gamma - 1.0
    gam0 = 1.0 - gamma   # = -gm1
    gam1 = 0.5 * (gamma - 1.0)
    gam2 = (gamma - 2.0) / (gamma - 1.0)
    epsilon = 1e-7
    tiny = 1e-14
    # Properly-typed literal scalars derived from gamma so the dtype follows the
    # working dtype (bare 1.0 / -1.0 / 1/sqrt(2) arrive as f32 under x64 + Triton
    # and trip a ('f64','f32') assertion in _truediv_lowering_rule).
    zero_typed = gamma - gamma
    one_typed = zero_typed + 1.0
    neg_one_typed = zero_typed - 1.0
    inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
    # AD-safe sqrt for non-negative-clamped quantities — mirrors the native
    # ``_eigen_mhd.diff_safe_sqrt``: a *positive* floor so the reverse pass never
    # forms ``sqrt'(0) = inf``.  ``jnp.sqrt(jnp.maximum(x, 0.0))`` is value-safe
    # but its gradient is ``inf`` at the clamp; under reverse-mode that ``inf``
    # meets a ``0`` cotangent and the in-kernel/multi-step backward yields NaN
    # (XLA folds inf*0->0 for a single call, which is why a one-shot VJP looked
    # clean).  The floor is below any physical value, so the forward stays
    # bit-exact with the native flux (which floors the same way).
    sqrt_eps = zero_typed + (1e-30 if jax.config.jax_enable_x64 else 1e-20)

    def ssqrt(x):
        return jnp.sqrt(jnp.maximum(x, sqrt_eps))

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
        disc_root = ssqrt(
            (b2 / rho_f + sound_speed_sq) ** 2 - 4.0 * bn2_over_rho * sound_speed_sq
        )
        c_fast = ssqrt(0.5 * (b2 / rho_f + sound_speed_sq + disc_root))
        c_alfven = ssqrt(bn2_over_rho)
        c_slow = ssqrt(0.5 * (b2 / rho_f + sound_speed_sq - disc_root))
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

    def lambda_from_floored_cell(cell, mode):
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

    f_stencil = tuple(flux_from_q(q) for q in q_stencil)
    floored_stencil = tuple(floored_cell(q) for q in q_stencil)
    cell_l = floored_stencil[2]  # offset 0  (cell i)
    cell_r = floored_stencil[3]  # offset 1  (cell i+1)

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
    c_sq_safe = jnp.where(c_sq_face > 0.0, c_sq_face, one_typed)
    inv_c_sq = jnp.where(c_sq_face > 0.0, 1.0 / c_sq_safe, 0.0)

    ms_disc = (b2_over_rho_face + c_sq_face) ** 2 - 4.0 * bn2_over_rho_face * c_sq_face
    ms_disc_root = ssqrt(ms_disc)

    lambda_fast = ssqrt(0.5 * (b2_over_rho_face + c_sq_face + ms_disc_root))
    lambda_alfven = ssqrt(bn2_over_rho_face)
    lambda_slow = ssqrt(0.5 * (b2_over_rho_face + c_sq_face - ms_disc_root))

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
        ssqrt(c_sq_face - lambda_slow * lambda_slow) / jnp.sqrt(denom_safe),
        1.0,
    )
    am_slow = jnp.where(
        denom >= b_eps,
        ssqrt(lambda_fast * lambda_fast - c_sq_face) / jnp.sqrt(denom_safe),
        1.0,
    )

    sqrt_rho_face = jnp.sqrt(jnp.maximum(rho_face, rhomin))
    cs_geq_alfven = c_face >= lambda_alfven

    def left_project(mode, values):
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

    def add_right_correction(flux_acc, mode, Fs):
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

    def alpha_for_mode(mode):
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

    return flux_acc


def _weno_mhd_flux_from_window_adjoint(
    q_stencil, flux_bar, gamma, rhomin, pgmin, b_eps, sqrt_floor, ncomp, num_modes
):
    """Explicit reverse pass (vector-Jacobian product) of
    :func:`_weno_mhd_flux_from_window`.

    Given the 6-cell window ``q_stencil`` (length-6 of length-8) and the output
    cotangent ``flux_bar`` (length-8) returns ``qbar_stencil`` — a length-6 list
    of length-8 lists, the cotangent w.r.t. every stencil input.

    HYBRID derivation (mirrors the hydro hand-derived adjoint
    :func:`_weno_hydro_flux_from_window_adjoint` section-by-section):

    * The per-mode characteristic projection (``left_project`` /
      ``add_right_correction``), the WENO smoothness reconstruction, the
      ``amx = max_k|lambda_k|`` Lax-Friedrichs fold, the centered flux, and the
      per-cell ``flux_from_q`` / ``floored_cell`` / ``primitive_from_q`` maps are
      ALL hand-transposed as plain elementwise arithmetic (no ``jax.vjp``).
    * The ONLY part wrapped in a small local ``jax.vjp`` is the MHD eigenvector
      *building-block* map ``_eigen_bb`` : the per-face scalar function from the
      eight linear-in-cell base face quantities
      ``(rho_face, vn_face, vt1_face, vt2_face, Bn_face, Bt1_face, Bt2_face,
      h_face)`` to the derived eigenstructure
      ``(v2_face, c_sq_face, inv_c_sq, c_face, lambda_fast, lambda_slow,
      am_fast, am_slow, bt_n1, bt_n2, sgn_bn, sgn_bt, sqrt_rho_face,
      cs_geq_alfven)``.  Hand-deriving that block (several nested ``ssqrt`` /
      degeneracy ``where`` branches: fast/slow split, tangential normalisation,
      sign conventions) bit-exact is intractable by hand; a local vjp of a tiny
      scalar-in/scalar-out function lowers far better in Triton than the
      whole-window vjp the old kernel used (which is the cause of the 63x
      forward/backward blow-up).  Everything downstream of those building blocks
      is explicit, so the in-kernel vjp scope shrinks from 48 stencil scalars
      through the entire WENO machinery to 8 face scalars through pure algebra.

    Validated bit-exact (~1e-12, x64) against ``jax.vjp`` of the forward window
    in ``pytests/pallas/_weno_mhd_window_adjoint_check.py``.  ``b_eps`` and
    ``sqrt_floor`` are typed scalars (x64 + Triton dtype hygiene)."""
    gm1 = gamma - 1.0
    gam0 = 1.0 - gamma
    gam1 = 0.5 * (gamma - 1.0)
    gam2 = (gamma - 2.0) / (gamma - 1.0)
    epsilon = 1e-7
    tiny = 1e-14
    zero_typed = gamma - gamma
    one_typed = zero_typed + 1.0
    neg_one_typed = zero_typed - 1.0
    inv_sqrt_two_typed = zero_typed + (1.0 / 2.0 ** 0.5)
    sqrt_eps = zero_typed + (1e-30 if jax.config.jax_enable_x64 else 1e-20)

    def ssqrt(x):
        return jnp.sqrt(jnp.maximum(x, sqrt_eps))

    # ---- per-cell forward maps (recomputed) ----
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
        specific_enthalpy = (energy_f + p_f) / rho_f
        sound_speed_sq = jnp.maximum(0.0, gamma * jnp.abs(p_f / rho_f))
        sound_speed = jnp.sqrt(jnp.maximum(sound_speed_sq, sqrt_floor))
        bn2_over_rho = (Bn * Bn) / rho_f
        disc_root = ssqrt(
            (b2 / rho_f + sound_speed_sq) ** 2 - 4.0 * bn2_over_rho * sound_speed_sq
        )
        c_fast = ssqrt(0.5 * (b2 / rho_f + sound_speed_sq + disc_root))
        c_alfven = ssqrt(bn2_over_rho)
        c_slow = ssqrt(0.5 * (b2 / rho_f + sound_speed_sq - disc_root))
        return (rho_f, mn, mt1, mt2, Bn, Bt1, Bt2, energy_f,
                vn, vt1, vt2, v2, b2, p_f, specific_enthalpy,
                sound_speed, sound_speed_sq, c_fast, c_alfven, c_slow)

    def flux_from_q(q):
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q)
        p_total = p + 0.5 * b2
        v_dot_B = vn * Bn + vt1 * Bt1 + vt2 * Bt2
        return (
            mn,
            rho * vn * vn + p_total - Bn * Bn,
            rho * vn * vt1 - Bn * Bt1,
            rho * vn * vt2 - Bn * Bt2,
            zero_typed,
            Bt1 * vn - Bn * vt1,
            Bt2 * vn - Bn * vt2,
            (energy + p_total) * vn - v_dot_B * Bn,
        )

    # ---- per-cell adjoints ----
    def primitive_from_q_adj(q, bars):
        # bars: cotangents for the 14-tuple primitive_from_q output, in order
        # (rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p).
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy = q
        inv_rho = 1.0 / rho
        vn = mn * inv_rho
        vt1 = mt1 * inv_rho
        vt2 = mt2 * inv_rho
        (b_rho, b_mn, b_mt1, b_mt2, b_Bn, b_Bt1, b_Bt2, b_energy,
         b_vn, b_vt1, b_vt2, b_v2, b_b2, b_p) = bars
        # p = gm1 * (energy - 0.5 * (rho * v2 + b2))
        #   d p/d energy = gm1; d p/d rho = gm1*(-0.5*v2);
        #   d p/d v2 = gm1*(-0.5*rho); d p/d b2 = gm1*(-0.5)
        b_energy = b_energy + gm1 * b_p
        v2 = vn * vn + vt1 * vt1 + vt2 * vt2
        b_rho = b_rho + gm1 * (-0.5 * v2) * b_p
        b_v2 = b_v2 + gm1 * (-0.5 * rho) * b_p
        b_b2 = b_b2 + gm1 * (-0.5) * b_p
        # b2 = Bn^2 + Bt1^2 + Bt2^2
        b_Bn = b_Bn + 2.0 * Bn * b_b2
        b_Bt1 = b_Bt1 + 2.0 * Bt1 * b_b2
        b_Bt2 = b_Bt2 + 2.0 * Bt2 * b_b2
        # v2 = vn^2 + vt1^2 + vt2^2
        b_vn = b_vn + 2.0 * vn * b_v2
        b_vt1 = b_vt1 + 2.0 * vt1 * b_v2
        b_vt2 = b_vt2 + 2.0 * vt2 * b_v2
        # vn = mn*inv_rho, etc.
        b_inv_rho = mt2 * b_vt2 + mt1 * b_vt1 + mn * b_vn
        b_mt2 = b_mt2 + inv_rho * b_vt2
        b_mt1 = b_mt1 + inv_rho * b_vt1
        b_mn = b_mn + inv_rho * b_vn
        b_rho = b_rho + (-inv_rho * inv_rho) * b_inv_rho
        return [b_rho, b_mn, b_mt1, b_mt2, b_Bn, b_Bt1, b_Bt2, b_energy]

    def flux_from_q_adj(q, fbar):
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q)
        bd = [0.0] * 14  # bars for primitive_from_q outputs
        # indices: 0 rho,1 mn,2 mt1,3 mt2,4 Bn,5 Bt1,6 Bt2,7 energy,
        #          8 vn,9 vt1,10 vt2,11 v2,12 b2,13 p
        p_total = p + 0.5 * b2
        # f0 = mn
        bd[1] += fbar[0]
        # f1 = rho*vn*vn + p_total - Bn*Bn ; p_total = p + 0.5*b2
        bd[0] += fbar[1] * (vn * vn)
        bd[8] += fbar[1] * (2.0 * rho * vn)
        bd[13] += fbar[1]
        bd[12] += fbar[1] * 0.5
        bd[4] += fbar[1] * (-2.0 * Bn)
        # f2 = rho*vn*vt1 - Bn*Bt1
        bd[0] += fbar[2] * (vn * vt1)
        bd[8] += fbar[2] * (rho * vt1)
        bd[9] += fbar[2] * (rho * vn)
        bd[4] += fbar[2] * (-Bt1)
        bd[5] += fbar[2] * (-Bn)
        # f3 = rho*vn*vt2 - Bn*Bt2
        bd[0] += fbar[3] * (vn * vt2)
        bd[8] += fbar[3] * (rho * vt2)
        bd[10] += fbar[3] * (rho * vn)
        bd[4] += fbar[3] * (-Bt2)
        bd[6] += fbar[3] * (-Bn)
        # f4 = 0  -> no contribution
        # f5 = Bt1*vn - Bn*vt1
        bd[5] += fbar[5] * vn
        bd[8] += fbar[5] * Bt1
        bd[4] += fbar[5] * (-vt1)
        bd[9] += fbar[5] * (-Bn)
        # f6 = Bt2*vn - Bn*vt2
        bd[6] += fbar[6] * vn
        bd[8] += fbar[6] * Bt2
        bd[4] += fbar[6] * (-vt2)
        bd[10] += fbar[6] * (-Bn)
        # f7 = (energy + p_total)*vn - v_dot_B*Bn ;
        #      v_dot_B = vn*Bn + vt1*Bt1 + vt2*Bt2
        v_dot_B = vn * Bn + vt1 * Bt1 + vt2 * Bt2
        bd[7] += fbar[7] * vn
        bd[13] += fbar[7] * vn          # p_total via p
        bd[12] += fbar[7] * 0.5 * vn    # p_total via 0.5*b2
        bd[8] += fbar[7] * (energy + p_total)
        # - v_dot_B * Bn
        bd[4] += fbar[7] * (-v_dot_B)
        bd[8] += fbar[7] * (-Bn * Bn)
        bd[4] += fbar[7] * (-Bn * vn)
        bd[9] += fbar[7] * (-Bn * Bt1)
        bd[5] += fbar[7] * (-Bn * vt1)
        bd[10] += fbar[7] * (-Bn * Bt2)
        bd[6] += fbar[7] * (-Bn * vt2)
        return primitive_from_q_adj(q, bd)

    def floored_cell_adj(q, bars20):
        # bars20: cotangents for the 20-tuple floored_cell output.
        rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy, vn, vt1, vt2, v2, b2, p = primitive_from_q(q)
        troubled = (rho < rhomin) | (p < pgmin)
        rho_f = jnp.where(troubled, jnp.maximum(rho, rhomin), rho)
        p_f = jnp.where(troubled, jnp.maximum(p, pgmin), p)
        energy_f = jnp.where(troubled, p_f / gm1 + 0.5 * (rho_f * v2 + b2), energy)
        ratio = p_f / rho_f
        ss_sq = jnp.maximum(0.0, gamma * jnp.abs(ratio))
        bn2_over_rho = (Bn * Bn) / rho_f
        b2_over_rho = b2 / rho_f
        disc_arg = (b2_over_rho + ss_sq) ** 2 - 4.0 * bn2_over_rho * ss_sq
        disc_root = ssqrt(disc_arg)
        cf_arg = 0.5 * (b2_over_rho + ss_sq + disc_root)
        cs_arg = 0.5 * (b2_over_rho + ss_sq - disc_root)
        ca_arg = bn2_over_rho
        ss_floor_arg = jnp.maximum(ss_sq, sqrt_floor)

        (b_rho_f, b_mn, b_mt1, b_mt2, b_Bn, b_Bt1, b_Bt2, b_energy_f,
         b_vn, b_vt1, b_vt2, b_v2, b_b2, b_p_f, b_h,
         b_sound, b_ss_sq, b_c_fast, b_c_alfven, b_c_slow) = bars20

        # --- characteristic speeds (reverse) ---
        # c_fast = ssqrt(cf_arg); ssqrt'(x)=0.5/sqrt(max(x,eps)) * [x>eps]
        def dssqrt(arg, bar):
            val = jnp.sqrt(jnp.maximum(arg, sqrt_eps))
            return jnp.where(arg > sqrt_eps, bar * 0.5 / val, 0.0)
        b_cf_arg = dssqrt(cf_arg, b_c_fast)
        b_cs_arg = dssqrt(cs_arg, b_c_slow)
        b_ca_arg = dssqrt(ca_arg, b_c_alfven)
        # cf_arg = 0.5*(b2_over_rho + ss_sq + disc_root)
        b_b2_over_rho = 0.5 * b_cf_arg
        b_ss_sq2 = 0.5 * b_cf_arg
        b_disc_root = 0.5 * b_cf_arg
        # cs_arg = 0.5*(b2_over_rho + ss_sq - disc_root)
        b_b2_over_rho += 0.5 * b_cs_arg
        b_ss_sq2 += 0.5 * b_cs_arg
        b_disc_root += -0.5 * b_cs_arg
        # ca_arg = bn2_over_rho
        b_bn2_over_rho = b_ca_arg
        # disc_root = ssqrt(disc_arg)
        b_disc_arg = dssqrt(disc_arg, b_disc_root)
        # disc_arg = (b2_over_rho + ss_sq)^2 - 4*bn2_over_rho*ss_sq
        s_bo = b2_over_rho + ss_sq
        b_b2_over_rho += b_disc_arg * 2.0 * s_bo
        b_ss_sq2 += b_disc_arg * 2.0 * s_bo
        b_bn2_over_rho += b_disc_arg * (-4.0 * ss_sq)
        b_ss_sq2 += b_disc_arg * (-4.0 * bn2_over_rho)
        # sound_speed = sqrt(max(ss_sq, sqrt_floor))
        sval = jnp.sqrt(ss_floor_arg)
        b_ss_sq2 += jnp.where(ss_sq > sqrt_floor, b_sound * 0.5 / sval, 0.0)
        # explicit ss_sq output bar
        b_ss_sq2 += b_ss_sq
        # bn2_over_rho = Bn^2 / rho_f
        b_Bn += b_bn2_over_rho * (2.0 * Bn / rho_f)
        b_rho_f += b_bn2_over_rho * (-(Bn * Bn) / (rho_f * rho_f))
        # b2_over_rho = b2 / rho_f
        b_b2 += b_b2_over_rho / rho_f
        b_rho_f += b_b2_over_rho * (-b2 / (rho_f * rho_f))
        # ss_sq = max(0, gamma*|ratio|)
        active = ss_sq > 0.0
        b_ratio = jnp.where(active, b_ss_sq2 * gamma * jnp.sign(ratio), 0.0)
        # ratio = p_f / rho_f
        b_p_f += b_ratio / rho_f
        b_rho_f += b_ratio * (-p_f / (rho_f * rho_f))

        # --- specific_enthalpy h = (energy_f + p_f) / rho_f ---
        b_energy_f += b_h / rho_f
        b_p_f += b_h / rho_f
        b_rho_f += b_h * (-(energy_f + p_f) / (rho_f * rho_f))

        # --- energy_f = where(troubled, p_f/gm1 + 0.5*(rho_f*v2 + b2), energy) ---
        b_p_f += jnp.where(troubled, b_energy_f / gm1, 0.0)
        b_rho_f += jnp.where(troubled, b_energy_f * 0.5 * v2, 0.0)
        b_v2 += jnp.where(troubled, b_energy_f * 0.5 * rho_f, 0.0)
        b_b2 += jnp.where(troubled, b_energy_f * 0.5, 0.0)
        b_energy = jnp.where(troubled, 0.0, b_energy_f)
        # --- p_f = where(troubled, max(p, pgmin), p) ---
        b_p = b_p_f * jnp.where(troubled, jnp.where(p > pgmin, 1.0, 0.0), 1.0)
        # --- rho_f = where(troubled, max(rho, rhomin), rho) ---
        b_rho = b_rho_f * jnp.where(troubled, jnp.where(rho > rhomin, 1.0, 0.0), 1.0)

        # Assemble primitive_from_q bars (14-tuple) and chain back.
        bd = [b_rho, b_mn, b_mt1, b_mt2, b_Bn, b_Bt1, b_Bt2, b_energy,
              b_vn, b_vt1, b_vt2, b_v2, b_b2, b_p]
        return primitive_from_q_adj(q, bd)

    # ---- eigenvector building blocks: base face quantities -> derived (local vjp) ----
    def _eigen_bb(rho_face, vn_face, vt1_face, vt2_face,
                  Bn_face, Bt1_face, Bt2_face, h_face):
        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
        b2_over_rho_face = b2_face / rho_face
        bn2_over_rho_face = (Bn_face * Bn_face) / rho_face
        c_sq_face = gm1 * (h_face - 0.5 * (v2_face + b2_over_rho_face))
        c_sq_face = jnp.maximum(c_sq_face, 0.0)
        c_face = jnp.sqrt(jnp.maximum(c_sq_face, sqrt_floor))
        c_sq_safe = jnp.where(c_sq_face > 0.0, c_sq_face, one_typed)
        inv_c_sq = jnp.where(c_sq_face > 0.0, 1.0 / c_sq_safe, 0.0)
        ms_disc = (b2_over_rho_face + c_sq_face) ** 2 - 4.0 * bn2_over_rho_face * c_sq_face
        ms_disc_root = ssqrt(ms_disc)
        lambda_fast = ssqrt(0.5 * (b2_over_rho_face + c_sq_face + ms_disc_root))
        lambda_alfven = ssqrt(bn2_over_rho_face)
        lambda_slow = ssqrt(0.5 * (b2_over_rho_face + c_sq_face - ms_disc_root))
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
            ssqrt(c_sq_face - lambda_slow * lambda_slow) / jnp.sqrt(denom_safe),
            1.0,
        )
        am_slow = jnp.where(
            denom >= b_eps,
            ssqrt(lambda_fast * lambda_fast - c_sq_face) / jnp.sqrt(denom_safe),
            1.0,
        )
        sqrt_rho_face = jnp.sqrt(jnp.maximum(rho_face, rhomin))
        # Only the *differentiable* float outputs are returned (sgn_bn / sgn_bt /
        # cs_geq_alfven are piecewise-constant -> zero cotangent, and lambda_alfven
        # only feeds the constant comparison; all three are computed outside the
        # differentiated map).
        return (v2_face, c_sq_face, inv_c_sq, c_face, lambda_fast, lambda_slow,
                am_fast, am_slow, bt_n1, bt_n2, sqrt_rho_face)

    def _eigen_consts(rho_face, vn_face, vt1_face, vt2_face,
                      Bn_face, Bt1_face, Bt2_face, h_face):
        """Non-differentiable companions of ``_eigen_bb`` (signs + the
        cs>=alfven branch flag)."""
        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
        b2_over_rho_face = b2_face / rho_face
        bn2_over_rho_face = (Bn_face * Bn_face) / rho_face
        c_sq_face = jnp.maximum(gm1 * (h_face - 0.5 * (v2_face + b2_over_rho_face)), 0.0)
        c_face = jnp.sqrt(jnp.maximum(c_sq_face, sqrt_floor))
        lambda_alfven = ssqrt(bn2_over_rho_face)
        sgn_bn = jnp.where(Bn_face >= 0.0, one_typed, neg_one_typed)
        sgn_bt = jnp.where(
            Bt1_face != 0.0,
            jnp.where(Bt1_face >= 0.0, one_typed, neg_one_typed),
            jnp.where(Bt2_face >= 0.0, one_typed, neg_one_typed),
        )
        cs_geq_alfven = c_face >= lambda_alfven
        return sgn_bn, sgn_bt, cs_geq_alfven

    def _eigen_bb_adj(base_q, ct):
        """Hand-derived Jacobian-transpose of :func:`_eigen_bb`.

        ``base_q`` is the 8-tuple of base face quantities (rho, vn, vt1, vt2, Bn,
        Bt1, Bt2, h); ``ct`` is the 11-tuple of output cotangents in the order
        returned by ``_eigen_bb`` (v2, csq, inv_c_sq, c_face, lambda_fast,
        lambda_slow, am_fast, am_slow, bt_n1, bt_n2, sqrt_rho_face).  Returns the
        length-8 cotangent over the base quantities.  Bit-exact vs
        ``jax.vjp(_eigen_bb)`` for non-degenerate states; matches it to FP-order
        on the measure-zero zero-tangential-B / no-field degeneracies (the
        ``1/sqrt(bt^2)`` near-singularity, where the native vjp is no more exact
        — both straddle central FD identically).  Validated in
        ``pytests/pallas/_weno_mhd_eigenbb_adjoint_isolated.py``."""
        (rho_face, vn_face, vt1_face, vt2_face,
         Bn_face, Bt1_face, Bt2_face, h_face) = base_q
        (b_v2, b_csq_out, b_invc, b_cface, b_lf, b_ls,
         b_amf, b_ams, b_bn1, b_bn2, b_srho) = ct

        # forward recompute (intermediates)
        v2_face = vn_face * vn_face + vt1_face * vt1_face + vt2_face * vt2_face
        b2_face = Bn_face * Bn_face + Bt1_face * Bt1_face + Bt2_face * Bt2_face
        b2_over_rho_face = b2_face / rho_face
        bn2_over_rho_face = (Bn_face * Bn_face) / rho_face
        c_sq_raw = gm1 * (h_face - 0.5 * (v2_face + b2_over_rho_face))
        c_sq_face = jnp.maximum(c_sq_raw, 0.0)
        ms_disc = (b2_over_rho_face + c_sq_face) ** 2 - 4.0 * bn2_over_rho_face * c_sq_face
        ms_disc_root = ssqrt(ms_disc)
        lf_arg = 0.5 * (b2_over_rho_face + c_sq_face + ms_disc_root)
        ls_arg = 0.5 * (b2_over_rho_face + c_sq_face - ms_disc_root)
        lambda_fast = ssqrt(lf_arg)
        lambda_slow = ssqrt(ls_arg)
        bt_sq = Bt1_face * Bt1_face + Bt2_face * Bt2_face
        bt_sq_safe = jnp.maximum(bt_sq, b_eps)
        sqrt_btss = jnp.sqrt(bt_sq_safe)
        denom = lambda_fast * lambda_fast - lambda_slow * lambda_slow
        denom_safe = jnp.maximum(denom, b_eps)
        sqrt_denom = jnp.sqrt(denom_safe)
        amf_num_arg = c_sq_face - lambda_slow * lambda_slow
        ams_num_arg = lambda_fast * lambda_fast - c_sq_face
        amf_num = ssqrt(amf_num_arg)
        ams_num = ssqrt(ams_num_arg)

        def dssqrt(arg, bar):
            val = jnp.sqrt(jnp.maximum(arg, sqrt_eps))
            return jnp.where(arg > sqrt_eps, bar * 0.5 / val, 0.0)

        b_rho = zero_typed; b_vn = zero_typed; b_vt1 = zero_typed; b_vt2 = zero_typed
        b_Bn = zero_typed; b_Bt1 = zero_typed; b_Bt2 = zero_typed; b_h = zero_typed
        b_b2or = zero_typed
        b_bn2or = zero_typed
        b_csq = b_csq_out               # direct output cotangent on c_sq_face
        b_lf_acc = b_lf
        b_ls_acc = b_ls

        # sqrt_rho_face = sqrt(max(rho, rhomin))
        srho_val = jnp.sqrt(jnp.maximum(rho_face, rhomin))
        b_rho += jnp.where(rho_face > rhomin, b_srho * 0.5 / srho_val, 0.0)

        # bt_n1 / bt_n2
        active_bt = bt_sq >= b_eps
        inv_sb = 1.0 / sqrt_btss
        b_btsq_safe = zero_typed
        b_Bt1 += jnp.where(active_bt, b_bn1 * inv_sb, 0.0)
        b_btsq_safe += jnp.where(active_bt, b_bn1 * (-0.5 * Bt1_face / bt_sq_safe ** 1.5), 0.0)
        b_Bt2 += jnp.where(active_bt, b_bn2 * inv_sb, 0.0)
        b_btsq_safe += jnp.where(active_bt, b_bn2 * (-0.5 * Bt2_face / bt_sq_safe ** 1.5), 0.0)
        b_btsq = jnp.where(bt_sq > b_eps, b_btsq_safe, 0.0)
        b_Bt1 += b_btsq * 2.0 * Bt1_face
        b_Bt2 += b_btsq * 2.0 * Bt2_face

        # am_fast / am_slow
        use = denom >= b_eps
        b_amf_num = jnp.where(use, b_amf / sqrt_denom, 0.0)
        b_sqrt_denom = jnp.where(use, b_amf * (-amf_num / sqrt_denom ** 2), 0.0)
        b_ams_num = jnp.where(use, b_ams / sqrt_denom, 0.0)
        b_sqrt_denom += jnp.where(use, b_ams * (-ams_num / sqrt_denom ** 2), 0.0)
        b_denom_safe = b_sqrt_denom * 0.5 / sqrt_denom
        b_denom = jnp.where(denom > b_eps, b_denom_safe, 0.0)
        b_amf_num_arg = dssqrt(amf_num_arg, b_amf_num)
        b_csq += b_amf_num_arg
        b_ls_acc += b_amf_num_arg * (-2.0 * lambda_slow)
        b_ams_num_arg = dssqrt(ams_num_arg, b_ams_num)
        b_lf_acc += b_ams_num_arg * (2.0 * lambda_fast)
        b_csq += b_ams_num_arg * (-1.0)
        b_lf_acc += b_denom * (2.0 * lambda_fast)
        b_ls_acc += b_denom * (-2.0 * lambda_slow)

        # lambda_fast / lambda_slow
        b_lf_arg = dssqrt(lf_arg, b_lf_acc)
        b_ls_arg = dssqrt(ls_arg, b_ls_acc)
        b_b2or += 0.5 * b_lf_arg + 0.5 * b_ls_arg
        b_csq += 0.5 * b_lf_arg + 0.5 * b_ls_arg
        b_ms_root = 0.5 * b_lf_arg - 0.5 * b_ls_arg
        b_ms_disc = dssqrt(ms_disc, b_ms_root)
        s_mc = b2_over_rho_face + c_sq_face
        b_b2or += b_ms_disc * 2.0 * s_mc
        b_csq += b_ms_disc * 2.0 * s_mc
        b_bn2or += b_ms_disc * (-4.0 * c_sq_face)
        b_csq += b_ms_disc * (-4.0 * bn2_over_rho_face)

        # c_face = sqrt(max(csq, sqrt_floor))
        cface_val = jnp.sqrt(jnp.maximum(c_sq_face, sqrt_floor))
        b_csq += jnp.where(c_sq_face > sqrt_floor, b_cface * 0.5 / cface_val, 0.0)
        # inv_c_sq = where(csq>0, 1/csq, 0)
        b_csq += jnp.where(c_sq_face > 0.0, b_invc * (-1.0 / c_sq_face ** 2), 0.0)

        # c_sq_face = max(c_sq_raw, 0)
        b_csq_raw = jnp.where(c_sq_raw > 0.0, b_csq, 0.0)
        b_h += b_csq_raw * gm1
        b_v2_internal = b_csq_raw * gm1 * (-0.5)
        b_b2or += b_csq_raw * gm1 * (-0.5)

        # bn2_over_rho_face = Bn^2 / rho
        b_Bn += b_bn2or * (2.0 * Bn_face / rho_face)
        b_rho += b_bn2or * (-(Bn_face * Bn_face) / rho_face ** 2)
        # b2_over_rho_face = b2 / rho
        b_b2 = b_b2or / rho_face
        b_rho += b_b2or * (-b2_face / rho_face ** 2)
        b_Bn += b_b2 * 2.0 * Bn_face
        b_Bt1 += b_b2 * 2.0 * Bt1_face
        b_Bt2 += b_b2 * 2.0 * Bt2_face

        # v2_face (output bar + internal)
        b_v2_total = b_v2 + b_v2_internal
        b_vn += b_v2_total * 2.0 * vn_face
        b_vt1 += b_v2_total * 2.0 * vt1_face
        b_vt2 += b_v2_total * 2.0 * vt2_face

        return [b_rho, b_vn, b_vt1, b_vt2, b_Bn, b_Bt1, b_Bt2, b_h]

    # ---- left_project given the eigenstructure (forward + adjoint) ----
    # ``fp`` carries every face quantity consumed by the projections.  The
    # adjoint accumulates cotangents into ``fpbar`` (same keys) and into the
    # per-cell values ``vbar``.  This mirrors left_project_adj in the hydro
    # template but for the 7 MHD waves.
    def _Lrows(mode, fp):
        """Return (coeffs, idxs): L_row[mode] as a list of (coeff, value_index)
        pairs, plus the per-mode overall scale.  value indices map to the
        8-tuple local order (rho, mn, mt1, mt2, Bn, Bt1, Bt2, energy)."""
        vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
        Bt1 = fp['Bt1']; Bt2 = fp['Bt2']
        v2 = fp['v2']; csq = fp['csq']; cface = fp['cface']
        lf = fp['lf']; ls = fp['ls']; amf = fp['amf']; ams = fp['ams']
        bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
        srho = fp['srho']; geq = fp['geq']; invc = fp['invc']
        if mode == 0:  # fast-
            L = [
                (amf * (gam1 * v2 + lf * vn) - ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
                (amf * (gam0 * vn - lf), 1),
                (gam0 * amf * vt1 + ams * ls * bn1 * sbn, 2),
                (gam0 * amf * vt2 + ams * ls * bn2 * sbn, 3),
                (gam0 * amf * Bt1 + cface * ams * bn1 * srho, 5),
                (gam0 * amf * Bt2 + cface * ams * bn2 * srho, 6),
                (-gam0 * amf, 7),
            ]
            scale = 0.5 * invc * jnp.where(~geq, sbt, one_typed)
            return L, scale
        if mode == 1:  # alfven-
            L = [
                (bn2 * vt1 - bn1 * vt2, 0),
                (-bn2, 2),
                (bn1, 3),
                (-bn2 * sbn * srho, 5),
                (bn1 * sbn * srho, 6),
            ]
            return L, 0.5 + zero_typed
        if mode == 2:  # slow-
            L = [
                (ams * (gam1 * v2 + ls * vn) + amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
                (ams * (gam0 * vn) - ams * ls, 1),
                (gam0 * ams * vt1 - amf * lf * bn1 * sbn, 2),
                (gam0 * ams * vt2 - amf * lf * bn2 * sbn, 3),
                (gam0 * ams * Bt1 - cface * amf * bn1 * srho, 5),
                (gam0 * ams * Bt2 - cface * amf * bn2 * srho, 6),
                (-gam0 * ams, 7),
            ]
            scale = 0.5 * invc * jnp.where(geq, sbt, one_typed)
            return L, scale
        if mode == 3:  # entropy
            L = [
                (-csq / gam0 - 0.5 * v2, 0),
                (vn, 1),
                (vt1, 2),
                (vt2, 3),
                (Bt1, 5),
                (Bt2, 6),
                (-1.0 + zero_typed, 7),
            ]
            return L, -gam0 * invc
        if mode == 4:  # slow+
            L = [
                (ams * (gam1 * v2 - ls * vn) - amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
                (ams * (gam0 * vn + ls), 1),
                (gam0 * ams * vt1 + amf * lf * bn1 * sbn, 2),
                (gam0 * ams * vt2 + amf * lf * bn2 * sbn, 3),
                (gam0 * ams * Bt1 - cface * amf * bn1 * srho, 5),
                (gam0 * ams * Bt2 - cface * amf * bn2 * srho, 6),
                (-gam0 * ams, 7),
            ]
            scale = 0.5 * invc * jnp.where(geq, sbt, one_typed)
            return L, scale
        if mode == 5:  # alfven+
            L = [
                (bn2 * vt1 - bn1 * vt2, 0),
                (-bn2, 2),
                (bn1, 3),
                (bn2 * sbn * srho, 5),
                (-bn1 * sbn * srho, 6),
            ]
            return L, 0.5 + zero_typed
        # mode 6 — fast+
        L = [
            (amf * (gam1 * v2 - lf * vn) + ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn, 0),
            (amf * (gam0 * vn + lf), 1),
            (gam0 * amf * vt1 - ams * ls * bn1 * sbn, 2),
            (gam0 * amf * vt2 - ams * ls * bn2 * sbn, 3),
            (gam0 * amf * Bt1 + cface * ams * bn1 * srho, 5),
            (gam0 * amf * Bt2 + cface * ams * bn2 * srho, 6),
            (-gam0 * amf, 7),
        ]
        scale = 0.5 * invc * jnp.where(~geq, sbt, one_typed)
        return L, scale

    def left_project_fwd(mode, values, fp):
        L, scale = _Lrows(mode, fp)
        acc = zero_typed
        for coeff, idx in L:
            acc = acc + coeff * values[idx]
        return acc * scale

    def _Rcols(mode, fp):
        """R_col[:, mode] as a length-8 list, plus the per-mode scale.  Mirrors
        ``add_right_correction`` in the forward window (B_normal slot is 0)."""
        vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
        v2 = fp['v2']; csq = fp['csq']; cface = fp['cface']
        lf = fp['lf']; ls = fp['ls']; amf = fp['amf']; ams = fp['ams']
        bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
        srho = fp['srho']; geq = fp['geq']
        if mode == 0:  # fast-
            R = [
                amf,
                amf * (vn - lf),
                amf * vt1 + ams * ls * bn1 * sbn,
                amf * vt2 + ams * ls * bn2 * sbn,
                zero_typed,
                cface * ams * bn1 / srho,
                cface * ams * bn2 / srho,
                amf * (lf * lf - lf * vn + 0.5 * v2 - gam2 * csq)
                + ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn,
            ]
            scale = jnp.where(~geq, sbt, one_typed)
        elif mode == 1:  # alfven-
            R = [
                zero_typed, zero_typed, -bn2, bn1, zero_typed,
                -bn2 * sbn / srho, bn1 * sbn / srho,
                bn1 * vt2 - bn2 * vt1,
            ]
            scale = one_typed
        elif mode == 2:  # slow-
            R = [
                ams,
                ams * (vn - ls),
                ams * vt1 - amf * lf * bn1 * sbn,
                ams * vt2 - amf * lf * bn2 * sbn,
                zero_typed,
                -cface * amf * bn1 / srho,
                -cface * amf * bn2 / srho,
                ams * (ls * ls - ls * vn + 0.5 * v2 - gam2 * csq)
                - amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn,
            ]
            scale = jnp.where(geq, sbt, one_typed)
        elif mode == 3:  # entropy
            R = [one_typed, vn, vt1, vt2, zero_typed, zero_typed, zero_typed, 0.5 * v2]
            scale = one_typed
        elif mode == 4:  # slow+
            R = [
                ams,
                ams * (vn + ls),
                ams * vt1 + amf * lf * bn1 * sbn,
                ams * vt2 + amf * lf * bn2 * sbn,
                zero_typed,
                -cface * amf * bn1 / srho,
                -cface * amf * bn2 / srho,
                ams * (ls * ls + ls * vn + 0.5 * v2 - gam2 * csq)
                + amf * lf * (bn1 * vt1 + bn2 * vt2) * sbn,
            ]
            scale = jnp.where(geq, sbt, one_typed)
        elif mode == 5:  # alfven+
            R = [
                zero_typed, zero_typed, -bn2, bn1, zero_typed,
                bn2 * sbn / srho, -bn1 * sbn / srho,
                bn1 * vt2 - bn2 * vt1,
            ]
            scale = one_typed
        else:  # mode == 6 — fast+
            R = [
                amf,
                amf * (vn + lf),
                amf * vt1 - ams * ls * bn1 * sbn,
                amf * vt2 - ams * ls * bn2 * sbn,
                zero_typed,
                cface * ams * bn1 / srho,
                cface * ams * bn2 / srho,
                amf * (lf * lf + lf * vn + 0.5 * v2 - gam2 * csq)
                - ams * ls * (bn1 * vt1 + bn2 * vt2) * sbn,
            ]
            scale = jnp.where(~geq, sbt, one_typed)
        return R, scale

    def _Rcol_apply(mode, fp_base, scal_tuple):
        """Return the length-8 tuple ``R[slot]*scale`` as a function of the
        differentiable scalar vector (used for the local vjp of R_col).  Every
        entry is broadcast to the tile shape (the scalar vector's leading entry
        is always tile-shaped) so jax.vjp's output/cotangent shapes match even
        for the structurally-constant R entries (the entropy/alfvén zeros)."""
        fpl = dict(fp_base)
        for key, val in zip(proj_keys, scal_tuple):
            fpl[key] = val
        R, scale = _Rcols(mode, fpl)
        ref = scal_tuple[0]
        return tuple(jnp.broadcast_to(R[slot] * scale, jnp.shape(ref))
                     for slot in range(ncomp))

    def _accum_scalar_bar(key, value, eig_bar, base_bar, base_index_of):
        """Route a cotangent on a consumed scalar (``key`` in ``proj_keys``) to
        either the base-face cotangent vector (vn/vt1/vt2/Bt1/Bt2 are base
        quantities) or the derived-eigenstructure accumulator ``eig_bar``."""
        if key == 'vn':
            base_bar[1] += value
        elif key == 'vt1':
            base_bar[2] += value
        elif key == 'vt2':
            base_bar[3] += value
        elif key == 'Bt1':
            base_bar[5] += value
        elif key == 'Bt2':
            base_bar[6] += value
        else:
            eig_bar[key] += value

    def _weno_recon_fwd(aterm, bterm, cterm, dterm):
        IS0 = 13.0 * (aterm - bterm) ** 2 + 3.0 * (aterm - 3.0 * bterm) ** 2
        IS1 = 13.0 * (bterm - cterm) ** 2 + 3.0 * (bterm + cterm) ** 2
        IS2 = 13.0 * (cterm - dterm) ** 2 + 3.0 * (3.0 * cterm - dterm) ** 2
        a0 = 1.0 / (epsilon + IS0) ** 2; a1 = 6.0 / (epsilon + IS1) ** 2; a2 = 3.0 / (epsilon + IS2) ** 2
        asum = jnp.maximum(a0 + a1 + a2, tiny)
        om0 = a0 / asum; om2 = a2 / asum
        return om0 * (aterm - 2.0 * bterm + cterm) / 3.0 + (om2 - 0.5) * (bterm - 2.0 * cterm + dterm) / 6.0

    def _weno_recon_adj(aterm, bterm, cterm, dterm, recon_bar):
        IS0 = 13.0 * (aterm - bterm) ** 2 + 3.0 * (aterm - 3.0 * bterm) ** 2
        IS1 = 13.0 * (bterm - cterm) ** 2 + 3.0 * (bterm + cterm) ** 2
        IS2 = 13.0 * (cterm - dterm) ** 2 + 3.0 * (3.0 * cterm - dterm) ** 2
        e0 = epsilon + IS0; e1 = epsilon + IS1; e2 = epsilon + IS2
        a0 = 1.0 / e0 ** 2; a1 = 6.0 / e1 ** 2; a2 = 3.0 / e2 ** 2
        s3 = a0 + a1 + a2; asum = jnp.maximum(s3, tiny)
        om0 = a0 / asum; om2 = a2 / asum
        P0 = (aterm - 2.0 * bterm + cterm) / 3.0
        P2 = (bterm - 2.0 * cterm + dterm) / 6.0
        ab = bb = cb = db = 0.0
        om0_bar = recon_bar * P0; P0_bar = recon_bar * om0
        om2_bar = recon_bar * P2; P2_bar = recon_bar * (om2 - 0.5)
        ab += P0_bar / 3.0; bb += -2.0 * P0_bar / 3.0; cb += P0_bar / 3.0
        bb += P2_bar / 6.0; cb += -2.0 * P2_bar / 6.0; db += P2_bar / 6.0
        a0_bar = om0_bar / asum; asum_bar = om0_bar * (-a0 / asum ** 2)
        a2_bar = om2_bar / asum; asum_bar += om2_bar * (-a2 / asum ** 2)
        s3_bar = jnp.where(s3 > tiny, asum_bar, 0.0)
        a0_bar += s3_bar; a1_bar = s3_bar; a2_bar += s3_bar
        IS0_bar = a0_bar * (-2.0) * e0 ** (-3)
        IS1_bar = a1_bar * 6.0 * (-2.0) * e1 ** (-3)
        IS2_bar = a2_bar * 3.0 * (-2.0) * e2 ** (-3)
        ab += IS0_bar * (26.0 * (aterm - bterm) + 6.0 * (aterm - 3.0 * bterm))
        bb += IS0_bar * (-26.0 * (aterm - bterm) - 18.0 * (aterm - 3.0 * bterm))
        bb += IS1_bar * (26.0 * (bterm - cterm) + 6.0 * (bterm + cterm))
        cb += IS1_bar * (-26.0 * (bterm - cterm) + 6.0 * (bterm + cterm))
        cb += IS2_bar * (26.0 * (cterm - dterm) + 18.0 * (3.0 * cterm - dterm))
        db += IS2_bar * (-26.0 * (cterm - dterm) - 6.0 * (3.0 * cterm - dterm))
        return ab, bb, cb, db

    def lam_of(cell, mode):
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

    # ============================ forward recompute ============================
    f_st = [flux_from_q(q) for q in q_stencil]
    fl_st = [floored_cell(q) for q in q_stencil]
    cl, cr = fl_st[2], fl_st[3]
    rho_i = cl[0]; mn_i = cl[1]; mt1_i = cl[2]; mt2_i = cl[3]
    Bn_i = cl[4]; Bt1_i = cl[5]; Bt2_i = cl[6]; h_i = cl[14]
    rho_j = cr[0]; mn_j = cr[1]; mt1_j = cr[2]; mt2_j = cr[3]
    Bn_j = cr[4]; Bt1_j = cr[5]; Bt2_j = cr[6]; h_j = cr[14]

    rho_face = jnp.maximum(0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin)), rhomin)
    vn_face = 0.5 * (mn_i + mn_j) / rho_face
    vt1_face = 0.5 * (mt1_i + mt1_j) / rho_face
    vt2_face = 0.5 * (mt2_i + mt2_j) / rho_face
    Bn_face = 0.5 * (Bn_i + Bn_j)
    Bt1_face = 0.5 * (Bt1_i + Bt1_j)
    Bt2_face = 0.5 * (Bt2_i + Bt2_j)
    h_face = 0.5 * (h_i + h_j)

    base = (rho_face, vn_face, vt1_face, vt2_face, Bn_face, Bt1_face, Bt2_face, h_face)
    if _MHD_VJP_FULL_HANDDERIVE:
        (v2_face, c_sq_face, inv_c_sq, c_face, lambda_fast, lambda_slow,
         am_fast, am_slow, bt_n1, bt_n2, sqrt_rho_face) = _eigen_bb(*base)
        eig_vjp = None  # replaced by the hand transpose _eigen_bb_adj below
    else:
        (v2_face, c_sq_face, inv_c_sq, c_face, lambda_fast, lambda_slow,
         am_fast, am_slow, bt_n1, bt_n2, sqrt_rho_face), eig_vjp = jax.vjp(_eigen_bb, *base)
    sgn_bn, sgn_bt, cs_geq_alfven = _eigen_consts(*base)

    fp = dict(vn=vn_face, vt1=vt1_face, vt2=vt2_face, Bt1=Bt1_face, Bt2=Bt2_face,
              v2=v2_face, csq=c_sq_face, cface=c_face, lf=lambda_fast, ls=lambda_slow,
              amf=am_fast, ams=am_slow, bn1=bt_n1, bn2=bt_n2, sbn=sgn_bn, sbt=sgn_bt,
              srho=sqrt_rho_face, geq=cs_geq_alfven, invc=inv_c_sq)

    # ---- left_project adjoint via local vjp over the consumed fp-scalars ----
    # The consumed *differentiable* fp scalars (sgn_bn/sgn_bt/cs_geq_alfven are
    # constants in the projection — their derivative is 0).  We differentiate the
    # whole projection w.r.t. this scalar vector with one tiny jax.vjp per (mode,
    # values); cheap and avoids re-deriving 7x messy L-rows by hand.
    proj_keys = ('vn', 'vt1', 'vt2', 'Bt1', 'Bt2', 'v2', 'csq', 'cface',
                 'lf', 'ls', 'amf', 'ams', 'bn1', 'bn2', 'srho', 'invc')

    def _project_scalar(mode, values_tuple, scal_tuple):
        fpl = dict(fp)
        for key, val in zip(proj_keys, scal_tuple):
            fpl[key] = val
        return left_project_fwd(mode, values_tuple, fpl)

    _pk_index = {k: i for i, k in enumerate(proj_keys)}

    def _rcol_apply_adj(mode, rsbar):
        """Hand transpose of :func:`_Rcol_apply` for one mode: given the length-8
        cotangents ``rsbar`` on the outputs ``R[slot]*scale``, return the
        length-16 cotangent over ``proj_keys``.  ``scale`` is the piecewise-const
        sign/branch factor (zero cotangent); only ``R[slot]`` carries scalar
        dependence.  Bit-exact vs ``jax.vjp(_Rcol_apply)`` (validated in
        ``pytests/pallas/_weno_mhd_RL_adjoint_isolated.py``)."""
        vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
        cface = fp['cface']; lf = fp['lf']; ls = fp['ls']
        amf = fp['amf']; ams = fp['ams']
        bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
        srho = fp['srho']; geq = fp['geq']
        g = [zero_typed] * 16
        if mode in (0, 6):
            scale = jnp.where(~geq, sbt, one_typed)
        elif mode in (2, 4):
            scale = jnp.where(geq, sbt, one_typed)
        else:
            scale = one_typed
        rb = [rsbar[slot] * scale for slot in range(ncomp)]

        def add(key, val):
            g[_pk_index[key]] = g[_pk_index[key]] + val

        if mode == 0:
            add('amf', rb[0])
            add('amf', rb[1] * (vn - lf)); add('vn', rb[1] * amf); add('lf', rb[1] * (-amf))
            add('amf', rb[2] * vt1); add('vt1', rb[2] * amf)
            add('ams', rb[2] * ls * bn1 * sbn); add('ls', rb[2] * ams * bn1 * sbn); add('bn1', rb[2] * ams * ls * sbn)
            add('amf', rb[3] * vt2); add('vt2', rb[3] * amf)
            add('ams', rb[3] * ls * bn2 * sbn); add('ls', rb[3] * ams * bn2 * sbn); add('bn2', rb[3] * ams * ls * sbn)
            add('cface', rb[5] * ams * bn1 / srho); add('ams', rb[5] * cface * bn1 / srho)
            add('bn1', rb[5] * cface * ams / srho); add('srho', rb[5] * (-cface * ams * bn1 / srho ** 2))
            add('cface', rb[6] * ams * bn2 / srho); add('ams', rb[6] * cface * bn2 / srho)
            add('bn2', rb[6] * cface * ams / srho); add('srho', rb[6] * (-cface * ams * bn2 / srho ** 2))
            E = (lf * lf - lf * vn + 0.5 * fp['v2'] - gam2 * fp['csq'])
            add('amf', rb[7] * E)
            add('lf', rb[7] * amf * (2.0 * lf - vn)); add('vn', rb[7] * amf * (-lf))
            add('v2', rb[7] * amf * 0.5); add('csq', rb[7] * amf * (-gam2))
            bsum = (bn1 * vt1 + bn2 * vt2)
            add('ams', rb[7] * ls * bsum * sbn); add('ls', rb[7] * ams * bsum * sbn)
            add('bn1', rb[7] * ams * ls * vt1 * sbn); add('vt1', rb[7] * ams * ls * bn1 * sbn)
            add('bn2', rb[7] * ams * ls * vt2 * sbn); add('vt2', rb[7] * ams * ls * bn2 * sbn)
        elif mode == 1:
            add('bn2', rb[2] * (-1.0)); add('bn1', rb[3] * 1.0)
            add('bn2', rb[5] * (-sbn / srho)); add('srho', rb[5] * (bn2 * sbn / srho ** 2))
            add('bn1', rb[6] * (sbn / srho)); add('srho', rb[6] * (-bn1 * sbn / srho ** 2))
            add('bn1', rb[7] * vt2); add('vt2', rb[7] * bn1); add('bn2', rb[7] * (-vt1)); add('vt1', rb[7] * (-bn2))
        elif mode == 2:
            add('ams', rb[0])
            add('ams', rb[1] * (vn - ls)); add('vn', rb[1] * ams); add('ls', rb[1] * (-ams))
            add('ams', rb[2] * vt1); add('vt1', rb[2] * ams)
            add('amf', rb[2] * (-lf * bn1 * sbn)); add('lf', rb[2] * (-amf * bn1 * sbn)); add('bn1', rb[2] * (-amf * lf * sbn))
            add('ams', rb[3] * vt2); add('vt2', rb[3] * ams)
            add('amf', rb[3] * (-lf * bn2 * sbn)); add('lf', rb[3] * (-amf * bn2 * sbn)); add('bn2', rb[3] * (-amf * lf * sbn))
            add('cface', rb[5] * (-amf * bn1 / srho)); add('amf', rb[5] * (-cface * bn1 / srho))
            add('bn1', rb[5] * (-cface * amf / srho)); add('srho', rb[5] * (cface * amf * bn1 / srho ** 2))
            add('cface', rb[6] * (-amf * bn2 / srho)); add('amf', rb[6] * (-cface * bn2 / srho))
            add('bn2', rb[6] * (-cface * amf / srho)); add('srho', rb[6] * (cface * amf * bn2 / srho ** 2))
            E = (ls * ls - ls * vn + 0.5 * fp['v2'] - gam2 * fp['csq'])
            add('ams', rb[7] * E)
            add('ls', rb[7] * ams * (2.0 * ls - vn)); add('vn', rb[7] * ams * (-ls))
            add('v2', rb[7] * ams * 0.5); add('csq', rb[7] * ams * (-gam2))
            bsum = (bn1 * vt1 + bn2 * vt2)
            add('amf', rb[7] * (-lf * bsum * sbn)); add('lf', rb[7] * (-amf * bsum * sbn))
            add('bn1', rb[7] * (-amf * lf * vt1 * sbn)); add('vt1', rb[7] * (-amf * lf * bn1 * sbn))
            add('bn2', rb[7] * (-amf * lf * vt2 * sbn)); add('vt2', rb[7] * (-amf * lf * bn2 * sbn))
        elif mode == 3:
            add('vn', rb[1]); add('vt1', rb[2]); add('vt2', rb[3]); add('v2', rb[7] * 0.5)
        elif mode == 4:
            add('ams', rb[0])
            add('ams', rb[1] * (vn + ls)); add('vn', rb[1] * ams); add('ls', rb[1] * ams)
            add('ams', rb[2] * vt1); add('vt1', rb[2] * ams)
            add('amf', rb[2] * (lf * bn1 * sbn)); add('lf', rb[2] * (amf * bn1 * sbn)); add('bn1', rb[2] * (amf * lf * sbn))
            add('ams', rb[3] * vt2); add('vt2', rb[3] * ams)
            add('amf', rb[3] * (lf * bn2 * sbn)); add('lf', rb[3] * (amf * bn2 * sbn)); add('bn2', rb[3] * (amf * lf * sbn))
            add('cface', rb[5] * (-amf * bn1 / srho)); add('amf', rb[5] * (-cface * bn1 / srho))
            add('bn1', rb[5] * (-cface * amf / srho)); add('srho', rb[5] * (cface * amf * bn1 / srho ** 2))
            add('cface', rb[6] * (-amf * bn2 / srho)); add('amf', rb[6] * (-cface * bn2 / srho))
            add('bn2', rb[6] * (-cface * amf / srho)); add('srho', rb[6] * (cface * amf * bn2 / srho ** 2))
            E = (ls * ls + ls * vn + 0.5 * fp['v2'] - gam2 * fp['csq'])
            add('ams', rb[7] * E)
            add('ls', rb[7] * ams * (2.0 * ls + vn)); add('vn', rb[7] * ams * ls)
            add('v2', rb[7] * ams * 0.5); add('csq', rb[7] * ams * (-gam2))
            bsum = (bn1 * vt1 + bn2 * vt2)
            add('amf', rb[7] * (lf * bsum * sbn)); add('lf', rb[7] * (amf * bsum * sbn))
            add('bn1', rb[7] * (amf * lf * vt1 * sbn)); add('vt1', rb[7] * (amf * lf * bn1 * sbn))
            add('bn2', rb[7] * (amf * lf * vt2 * sbn)); add('vt2', rb[7] * (amf * lf * bn2 * sbn))
        elif mode == 5:
            add('bn2', rb[2] * (-1.0)); add('bn1', rb[3] * 1.0)
            add('bn2', rb[5] * (sbn / srho)); add('srho', rb[5] * (-bn2 * sbn / srho ** 2))
            add('bn1', rb[6] * (-sbn / srho)); add('srho', rb[6] * (bn1 * sbn / srho ** 2))
            add('bn1', rb[7] * vt2); add('vt2', rb[7] * bn1); add('bn2', rb[7] * (-vt1)); add('vt1', rb[7] * (-bn2))
        else:  # mode 6
            add('amf', rb[0])
            add('amf', rb[1] * (vn + lf)); add('vn', rb[1] * amf); add('lf', rb[1] * amf)
            add('amf', rb[2] * vt1); add('vt1', rb[2] * amf)
            add('ams', rb[2] * (-ls * bn1 * sbn)); add('ls', rb[2] * (-ams * bn1 * sbn)); add('bn1', rb[2] * (-ams * ls * sbn))
            add('amf', rb[3] * vt2); add('vt2', rb[3] * amf)
            add('ams', rb[3] * (-ls * bn2 * sbn)); add('ls', rb[3] * (-ams * bn2 * sbn)); add('bn2', rb[3] * (-ams * ls * sbn))
            add('cface', rb[5] * ams * bn1 / srho); add('ams', rb[5] * cface * bn1 / srho)
            add('bn1', rb[5] * cface * ams / srho); add('srho', rb[5] * (-cface * ams * bn1 / srho ** 2))
            add('cface', rb[6] * ams * bn2 / srho); add('ams', rb[6] * cface * bn2 / srho)
            add('bn2', rb[6] * cface * ams / srho); add('srho', rb[6] * (-cface * ams * bn2 / srho ** 2))
            E = (lf * lf + lf * vn + 0.5 * fp['v2'] - gam2 * fp['csq'])
            add('amf', rb[7] * E)
            add('lf', rb[7] * amf * (2.0 * lf + vn)); add('vn', rb[7] * amf * lf)
            add('v2', rb[7] * amf * 0.5); add('csq', rb[7] * amf * (-gam2))
            bsum = (bn1 * vt1 + bn2 * vt2)
            add('ams', rb[7] * (-ls * bsum * sbn)); add('ls', rb[7] * (-ams * bsum * sbn))
            add('bn1', rb[7] * (-ams * ls * vt1 * sbn)); add('vt1', rb[7] * (-ams * ls * bn1 * sbn))
            add('bn2', rb[7] * (-ams * ls * vt2 * sbn)); add('vt2', rb[7] * (-ams * ls * bn2 * sbn))
        return g

    def _lrow_apply_adj(mode, values, out_bar):
        """Hand transpose of ``left_project_fwd`` w.r.t. the ``proj_keys`` scalars
        for one mode and one (fixed) 8-vector ``values``.  ``out_bar`` is the
        scalar cotangent on the projected output.  ``out = scale * sum_i
        coeff_i * values[idx_i]``; ``scale`` carries the differentiable ``invc``
        factor (for modes 0/2/3/4/6) plus piecewise-const sign/branch factors.
        Bit-exact vs ``jax.vjp`` of the cell-summed projection functional."""
        vn = fp['vn']; vt1 = fp['vt1']; vt2 = fp['vt2']
        cface = fp['cface']; lf = fp['lf']; ls = fp['ls']
        amf = fp['amf']; ams = fp['ams']
        bn1 = fp['bn1']; bn2 = fp['bn2']; sbn = fp['sbn']; sbt = fp['sbt']
        srho = fp['srho']; geq = fp['geq']
        g = [zero_typed] * 16
        L, scale = _Lrows(mode, fp)
        S = zero_typed
        for coeff, idx in L:
            S = S + coeff * values[idx]

        def add(key, val):
            g[_pk_index[key]] = g[_pk_index[key]] + val

        # invc dependence of scale
        if mode in (0, 6):
            sc_const = 0.5 * jnp.where(~geq, sbt, one_typed)
            add('invc', out_bar * sc_const * S)
        elif mode in (2, 4):
            sc_const = 0.5 * jnp.where(geq, sbt, one_typed)
            add('invc', out_bar * sc_const * S)
        elif mode == 3:
            add('invc', out_bar * (-gam0) * S)

        ob_S = out_bar * scale
        cb = {idx: ob_S * values[idx] for (_c, idx) in L}

        if mode == 0:
            c = cb[0]
            add('amf', c * (gam1 * fp['v2'] + lf * vn)); add('v2', c * amf * gam1); add('lf', c * amf * vn); add('vn', c * amf * lf)
            bsum = bn1 * vt1 + bn2 * vt2
            add('ams', c * (-ls * bsum * sbn)); add('ls', c * (-ams * bsum * sbn))
            add('bn1', c * (-ams * ls * vt1 * sbn)); add('vt1', c * (-ams * ls * bn1 * sbn))
            add('bn2', c * (-ams * ls * vt2 * sbn)); add('vt2', c * (-ams * ls * bn2 * sbn))
            c = cb[1]; add('amf', c * (gam0 * vn - lf)); add('vn', c * amf * gam0); add('lf', c * (-amf))
            c = cb[2]; add('amf', c * gam0 * vt1); add('vt1', c * gam0 * amf)
            add('ams', c * ls * bn1 * sbn); add('ls', c * ams * bn1 * sbn); add('bn1', c * ams * ls * sbn)
            c = cb[3]; add('amf', c * gam0 * vt2); add('vt2', c * gam0 * amf)
            add('ams', c * ls * bn2 * sbn); add('ls', c * ams * bn2 * sbn); add('bn2', c * ams * ls * sbn)
            c = cb[5]; add('amf', c * gam0 * fp['Bt1']); add('Bt1', c * gam0 * amf)
            add('cface', c * ams * bn1 * srho); add('ams', c * cface * bn1 * srho)
            add('bn1', c * cface * ams * srho); add('srho', c * cface * ams * bn1)
            c = cb[6]; add('amf', c * gam0 * fp['Bt2']); add('Bt2', c * gam0 * amf)
            add('cface', c * ams * bn2 * srho); add('ams', c * cface * bn2 * srho)
            add('bn2', c * cface * ams * srho); add('srho', c * cface * ams * bn2)
            c = cb[7]; add('amf', c * (-gam0))
        elif mode == 1:
            c = cb[0]; add('bn2', c * vt1); add('vt1', c * bn2); add('bn1', c * (-vt2)); add('vt2', c * (-bn1))
            c = cb[2]; add('bn2', c * (-1.0))
            c = cb[3]; add('bn1', c * 1.0)
            c = cb[5]; add('bn2', c * (-sbn * srho)); add('srho', c * (-bn2 * sbn))
            c = cb[6]; add('bn1', c * (sbn * srho)); add('srho', c * (bn1 * sbn))
        elif mode == 2:
            c = cb[0]
            add('ams', c * (gam1 * fp['v2'] + ls * vn)); add('v2', c * ams * gam1); add('ls', c * ams * vn); add('vn', c * ams * ls)
            bsum = bn1 * vt1 + bn2 * vt2
            add('amf', c * lf * bsum * sbn); add('lf', c * amf * bsum * sbn)
            add('bn1', c * amf * lf * vt1 * sbn); add('vt1', c * amf * lf * bn1 * sbn)
            add('bn2', c * amf * lf * vt2 * sbn); add('vt2', c * amf * lf * bn2 * sbn)
            c = cb[1]; add('ams', c * (gam0 * vn - ls)); add('vn', c * ams * gam0); add('ls', c * (-ams))
            c = cb[2]; add('ams', c * gam0 * vt1); add('vt1', c * gam0 * ams)
            add('amf', c * (-lf * bn1 * sbn)); add('lf', c * (-amf * bn1 * sbn)); add('bn1', c * (-amf * lf * sbn))
            c = cb[3]; add('ams', c * gam0 * vt2); add('vt2', c * gam0 * ams)
            add('amf', c * (-lf * bn2 * sbn)); add('lf', c * (-amf * bn2 * sbn)); add('bn2', c * (-amf * lf * sbn))
            c = cb[5]; add('ams', c * gam0 * fp['Bt1']); add('Bt1', c * gam0 * ams)
            add('cface', c * (-amf * bn1 * srho)); add('amf', c * (-cface * bn1 * srho))
            add('bn1', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn1))
            c = cb[6]; add('ams', c * gam0 * fp['Bt2']); add('Bt2', c * gam0 * ams)
            add('cface', c * (-amf * bn2 * srho)); add('amf', c * (-cface * bn2 * srho))
            add('bn2', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn2))
            c = cb[7]; add('ams', c * (-gam0))
        elif mode == 3:
            c = cb[0]; add('csq', c * (-1.0 / gam0)); add('v2', c * (-0.5))
            c = cb[1]; add('vn', c)
            c = cb[2]; add('vt1', c)
            c = cb[3]; add('vt2', c)
            c = cb[5]; add('Bt1', c)
            c = cb[6]; add('Bt2', c)
        elif mode == 4:
            c = cb[0]
            add('ams', c * (gam1 * fp['v2'] - ls * vn)); add('v2', c * ams * gam1); add('ls', c * (-ams * vn)); add('vn', c * (-ams * ls))
            bsum = bn1 * vt1 + bn2 * vt2
            add('amf', c * (-lf * bsum * sbn)); add('lf', c * (-amf * bsum * sbn))
            add('bn1', c * (-amf * lf * vt1 * sbn)); add('vt1', c * (-amf * lf * bn1 * sbn))
            add('bn2', c * (-amf * lf * vt2 * sbn)); add('vt2', c * (-amf * lf * bn2 * sbn))
            c = cb[1]; add('ams', c * (gam0 * vn + ls)); add('vn', c * ams * gam0); add('ls', c * ams)
            c = cb[2]; add('ams', c * gam0 * vt1); add('vt1', c * gam0 * ams)
            add('amf', c * lf * bn1 * sbn); add('lf', c * amf * bn1 * sbn); add('bn1', c * amf * lf * sbn)
            c = cb[3]; add('ams', c * gam0 * vt2); add('vt2', c * gam0 * ams)
            add('amf', c * lf * bn2 * sbn); add('lf', c * amf * bn2 * sbn); add('bn2', c * amf * lf * sbn)
            c = cb[5]; add('ams', c * gam0 * fp['Bt1']); add('Bt1', c * gam0 * ams)
            add('cface', c * (-amf * bn1 * srho)); add('amf', c * (-cface * bn1 * srho))
            add('bn1', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn1))
            c = cb[6]; add('ams', c * gam0 * fp['Bt2']); add('Bt2', c * gam0 * ams)
            add('cface', c * (-amf * bn2 * srho)); add('amf', c * (-cface * bn2 * srho))
            add('bn2', c * (-cface * amf * srho)); add('srho', c * (-cface * amf * bn2))
            c = cb[7]; add('ams', c * (-gam0))
        elif mode == 5:
            c = cb[0]; add('bn2', c * vt1); add('vt1', c * bn2); add('bn1', c * (-vt2)); add('vt2', c * (-bn1))
            c = cb[2]; add('bn2', c * (-1.0))
            c = cb[3]; add('bn1', c * 1.0)
            c = cb[5]; add('bn2', c * (sbn * srho)); add('srho', c * (bn2 * sbn))
            c = cb[6]; add('bn1', c * (-sbn * srho)); add('srho', c * (-bn1 * sbn))
        else:  # mode 6
            c = cb[0]
            add('amf', c * (gam1 * fp['v2'] - lf * vn)); add('v2', c * amf * gam1); add('lf', c * (-amf * vn)); add('vn', c * (-amf * lf))
            bsum = bn1 * vt1 + bn2 * vt2
            add('ams', c * ls * bsum * sbn); add('ls', c * ams * bsum * sbn)
            add('bn1', c * ams * ls * vt1 * sbn); add('vt1', c * ams * ls * bn1 * sbn)
            add('bn2', c * ams * ls * vt2 * sbn); add('vt2', c * ams * ls * bn2 * sbn)
            c = cb[1]; add('amf', c * (gam0 * vn + lf)); add('vn', c * amf * gam0); add('lf', c * amf)
            c = cb[2]; add('amf', c * gam0 * vt1); add('vt1', c * gam0 * amf)
            add('ams', c * (-ls * bn1 * sbn)); add('ls', c * (-ams * bn1 * sbn)); add('bn1', c * (-ams * ls * sbn))
            c = cb[3]; add('amf', c * gam0 * vt2); add('vt2', c * gam0 * amf)
            add('ams', c * (-ls * bn2 * sbn)); add('ls', c * (-ams * bn2 * sbn)); add('bn2', c * (-ams * ls * sbn))
            c = cb[5]; add('amf', c * gam0 * fp['Bt1']); add('Bt1', c * gam0 * amf)
            add('cface', c * ams * bn1 * srho); add('ams', c * cface * bn1 * srho)
            add('bn1', c * cface * ams * srho); add('srho', c * cface * ams * bn1)
            c = cb[6]; add('amf', c * gam0 * fp['Bt2']); add('Bt2', c * gam0 * amf)
            add('cface', c * ams * bn2 * srho); add('ams', c * cface * bn2 * srho)
            add('bn2', c * cface * ams * srho); add('srho', c * cface * ams * bn2)
            c = cb[7]; add('amf', c * (-gam0))
        return g

    # ============================ reverse accumulation ============================
    qbar = [[zero_typed] * ncomp for _ in range(6)]
    fbar_cell = [[zero_typed] * ncomp for _ in range(6)]
    fl_bar = [[zero_typed] * 20 for _ in range(6)]
    base_bar = [zero_typed] * 8  # cotangents for base face quantities
    eig_bar = {k: zero_typed for k in (
        'v2', 'csq', 'invc', 'cface', 'lf', 'ls', 'amf', 'ams', 'bn1', 'bn2',
        'srho')}  # differentiable derived eigenstructure
    # plus direct contributions to vn_face/vt1_face/vt2_face/Bt1_face/Bt2_face
    # which are base quantities -> accumulate into base_bar via proj.

    scal_vec = tuple(fp[k] for k in proj_keys)

    # Centered flux part.
    cc = [-1.0 / 12, 7.0 / 12, 7.0 / 12, -1.0 / 12]
    for j, k in enumerate((1, 2, 3, 4)):
        for slot in range(ncomp):
            fbar_cell[k][slot] += flux_bar[slot] * cc[j]

    # map proj_keys -> (base_bar index) or eig_bar key
    base_index_of = {'vn': 1, 'vt1': 2, 'vt2': 3}  # Bt1/Bt2 handled below
    for mode in range(num_modes):
        s = [left_project_fwd(mode, f_st[k], fp) for k in range(6)]
        qp = [left_project_fwd(mode, q_stencil[k], fp) for k in range(6)]
        d = [s[i + 1] - s[i] for i in range(5)]
        dq = [qp[i + 1] - qp[i] for i in range(5)]
        lams = [lam_of(fl_st[k], mode) for k in range(6)]
        absl = [jnp.abs(x) for x in lams]
        amxs = [absl[0]]
        for k in range(1, 6):
            amxs.append(jnp.maximum(amxs[-1], absl[k]))
        amx = amxs[-1]
        ap = 0.5 * (d[0] + amx * dq[0]); bp = 0.5 * (d[1] + amx * dq[1])
        cp = 0.5 * (d[2] + amx * dq[2]); dp = 0.5 * (d[3] + amx * dq[3])
        am = 0.5 * (d[4] - amx * dq[4]); bm = 0.5 * (d[3] - amx * dq[3])
        cm = 0.5 * (d[2] - amx * dq[2]); dm = 0.5 * (d[1] - amx * dq[1])
        second = _weno_recon_fwd(ap, bp, cp, dp)
        third = _weno_recon_fwd(am, bm, cm, dm)
        Fs = -second + third

        # --- add_right_correction adjoint: flux_acc[slot] += R[slot]*scale*Fs ---
        R, scaleR = _Rcols(mode, fp)
        Fs_bar = zero_typed
        for slot in range(ncomp):
            Fs_bar = Fs_bar + flux_bar[slot] * R[slot] * scaleR
        # cotangent on (R[slot]*scaleR): flux_bar[slot]*Fs
        rsbar = [flux_bar[slot] * Fs for slot in range(ncomp)]
        if _MHD_VJP_FULL_HANDDERIVE:
            rcol_grad = _rcol_apply_adj(mode, rsbar)
        else:
            _, rcol_vjp = jax.vjp(lambda sv: _Rcol_apply(mode, fp, sv), scal_vec)
            (rcol_grad,) = rcol_vjp(tuple(rsbar))
        for ki, key in enumerate(proj_keys):
            _accum_scalar_bar(key, rcol_grad[ki], eig_bar, base_bar, base_index_of)

        # --- WENO recon adjoint ---
        second_bar = -Fs_bar; third_bar = Fs_bar
        ap_b, bp_b, cp_b, dp_b = _weno_recon_adj(ap, bp, cp, dp, second_bar)
        am_b, bm_b, cm_b, dm_b = _weno_recon_adj(am, bm, cm, dm, third_bar)
        d_b = [zero_typed] * 5; dq_b = [zero_typed] * 5; amx_b = zero_typed
        for (tb, di) in ((ap_b, 0), (bp_b, 1), (cp_b, 2), (dp_b, 3)):
            d_b[di] += 0.5 * tb; dq_b[di] += 0.5 * amx * tb; amx_b += 0.5 * dq[di] * tb
        for (tb, di) in ((am_b, 4), (bm_b, 3), (cm_b, 2), (dm_b, 1)):
            d_b[di] += 0.5 * tb; dq_b[di] += -0.5 * amx * tb; amx_b += -0.5 * dq[di] * tb
        s_b = [zero_typed] * 6; qp_b = [zero_typed] * 6
        for i in range(5):
            s_b[i + 1] += d_b[i]; s_b[i] += -d_b[i]
            qp_b[i + 1] += dq_b[i]; qp_b[i] += -dq_b[i]

        # --- left_project adjoint ---
        # The projection is bilinear: out = scale(fp) * sum_i coeff_i(fp) * v[idx_i].
        # The cotangent w.r.t. the input ``values`` is hand-derived exactly from
        # the L-row coefficients (no vjp).  The cotangent w.r.t. the (messy) fp
        # eigenstructure scalars is obtained with a SINGLE local vjp per mode of
        # the cell-summed scalar functional (folding all 6 f-cells and 6 q-cells
        # in via their s_b / qp_b weights), instead of 12 per-cell vjps.
        L, scaleL = _Lrows(mode, fp)
        for k in range(6):
            for coeff, idx in L:
                fbar_cell[k][idx] += s_b[k] * scaleL * coeff
                qbar[k][idx] += qp_b[k] * scaleL * coeff

        if _MHD_VJP_FULL_HANDDERIVE:
            # The functional acc = sum_k s_b[k]*proj(f_st[k]) + qp_b[k]*proj(q[k])
            # is linear in each projection output, so its cotangent w.r.t. the
            # scalar vector is the s_b/qp_b-weighted sum of the per-(mode, values)
            # L-row transposes (out_bar = the cell weight).
            proj_grad = [zero_typed] * 16
            for k in range(6):
                gk = _lrow_apply_adj(mode, tuple(f_st[k]), s_b[k])
                gq = _lrow_apply_adj(mode, tuple(q_stencil[k]), qp_b[k])
                for ki in range(16):
                    proj_grad[ki] = proj_grad[ki] + gk[ki] + gq[ki]
        else:
            def _proj_functional(sv, _mode=mode, _sb=tuple(s_b), _qpb=tuple(qp_b)):
                acc = zero_typed
                for k in range(6):
                    acc = acc + _sb[k] * _project_scalar(_mode, tuple(f_st[k]), sv)
                    acc = acc + _qpb[k] * _project_scalar(_mode, tuple(q_stencil[k]), sv)
                return acc

            proj_out, proj_vjp = jax.vjp(_proj_functional, scal_vec)
            (proj_grad,) = proj_vjp(jnp.ones_like(proj_out))
        for ki, key in enumerate(proj_keys):
            _accum_scalar_bar(key, proj_grad[ki], eig_bar, base_bar, base_index_of)

        # --- amx fold adjoint (strict tie-break, matches lax.max VJP) ---
        acc = amx_b
        absl_bar = [zero_typed] * 6
        for k in range(5, 0, -1):
            prev_gets = amxs[k - 1] > absl[k]
            absl_bar[k] = absl_bar[k] + jnp.where(prev_gets, 0.0, acc)
            acc = jnp.where(prev_gets, acc, 0.0)
        absl_bar[0] = absl_bar[0] + acc
        for k in range(6):
            lam_b = absl_bar[k] * jnp.sign(lams[k])
            fl_bar[k][8] += lam_b                  # vn
            if mode == 0:
                fl_bar[k][17] += lam_b * (-1.0)    # c_fast
            elif mode == 1:
                fl_bar[k][18] += lam_b * (-1.0)    # c_alfven
            elif mode == 2:
                fl_bar[k][19] += lam_b * (-1.0)    # c_slow
            elif mode == 4:
                fl_bar[k][19] += lam_b * (1.0)
            elif mode == 5:
                fl_bar[k][18] += lam_b * (1.0)
            elif mode == 6:
                fl_bar[k][17] += lam_b * (1.0)
            # mode == 3 -> only vn

    # ---- eigenstructure adjoint: derived eig_bar -> base_bar via local vjp ----
    # jax.vjp wants cotangents matching outputs; sgn_bn/sgn_bt are non-diff
    # (sign) and cs_geq_alfven is bool -> zero cotangents for all three.
    eig_ct = (
        eig_bar['v2'], eig_bar['csq'], eig_bar['invc'], eig_bar['cface'],
        eig_bar['lf'], eig_bar['ls'], eig_bar['amf'], eig_bar['ams'],
        eig_bar['bn1'], eig_bar['bn2'], eig_bar['srho'],
    )
    if _MHD_VJP_FULL_HANDDERIVE:
        base_grad = _eigen_bb_adj(base, eig_ct)
    else:
        base_grad = eig_vjp(eig_ct)
    for i in range(8):
        base_bar[i] = base_bar[i] + base_grad[i]

    # ---- base face quantities adjoint -> floored cells 2, 3 ----
    (b_rho_face, b_vn_face, b_vt1_face, b_vt2_face,
     b_Bn_face, b_Bt1_face, b_Bt2_face, b_h_face) = base_bar
    # h_face = 0.5*(h_i + h_j)
    fl_bar[2][14] += 0.5 * b_h_face; fl_bar[3][14] += 0.5 * b_h_face
    # Bn_face = 0.5*(Bn_i + Bn_j)
    fl_bar[2][4] += 0.5 * b_Bn_face; fl_bar[3][4] += 0.5 * b_Bn_face
    # Bt1_face = 0.5*(Bt1_i + Bt1_j)
    fl_bar[2][5] += 0.5 * b_Bt1_face; fl_bar[3][5] += 0.5 * b_Bt1_face
    # Bt2_face = 0.5*(Bt2_i + Bt2_j)
    fl_bar[2][6] += 0.5 * b_Bt2_face; fl_bar[3][6] += 0.5 * b_Bt2_face
    # vt2_face = 0.5*(mt2_i+mt2_j)/rho_face
    rho_face_b = zero_typed
    num2 = 0.5 * (mt2_i + mt2_j); num2_b = b_vt2_face / rho_face
    rho_face_b += b_vt2_face * (-num2 / rho_face ** 2)
    fl_bar[2][3] += 0.5 * num2_b; fl_bar[3][3] += 0.5 * num2_b
    num1 = 0.5 * (mt1_i + mt1_j); num1_b = b_vt1_face / rho_face
    rho_face_b += b_vt1_face * (-num1 / rho_face ** 2)
    fl_bar[2][2] += 0.5 * num1_b; fl_bar[3][2] += 0.5 * num1_b
    numn = 0.5 * (mn_i + mn_j); numn_b = b_vn_face / rho_face
    rho_face_b += b_vn_face * (-numn / rho_face ** 2)
    fl_bar[2][1] += 0.5 * numn_b; fl_bar[3][1] += 0.5 * numn_b
    rho_face_b += b_rho_face
    inner = 0.5 * (jnp.maximum(rho_i, rhomin) + jnp.maximum(rho_j, rhomin))
    inner_b = jnp.where(inner > rhomin, rho_face_b, 0.0)
    fl_bar[2][0] += 0.5 * jnp.where(rho_i > rhomin, 1.0, 0.0) * inner_b
    fl_bar[3][0] += 0.5 * jnp.where(rho_j > rhomin, 1.0, 0.0) * inner_b

    # ---- per-cell floored + flux adjoints ----
    for k in range(6):
        gq_fl = floored_cell_adj(q_stencil[k], fl_bar[k])
        gq_fx = flux_from_q_adj(q_stencil[k], fbar_cell[k])
        for c in range(ncomp):
            qbar[k][c] = qbar[k][c] + gq_fl[c] + gq_fx[c]
    return qbar


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
    fast+).  The per-interface arithmetic — all face eigenstructure (the body
    of ``_eigen_mhd._eigenvector_building_blocks``) and the ``L_row``/``R_col``/
    ``λ`` projections dispatched at compile time via ``if mode == k`` branches —
    lives in the shared pure :func:`_weno_mhd_flux_from_window`, which the
    kernel calls on the gathered 6-cell stencil.  Factoring it into one function
    makes the kernel the single source of truth for both this forward pass and
    the :func:`_weno_flux_mhd_pallas_vjp_local` adjoint (which ``jax.vjp``\\s the
    same window).  No full-domain projection matrices are ever materialised —
    every component is computed per-tile in registers."""
    ndim = 3
    nvars = int(conserved_state.shape[0])
    spatial_shape = tuple(int(x) for x in conserved_state.shape[1:])
    nx, ny, nz = spatial_shape
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    local_indices = _mhd_indices_for_axis(config, registered_variables, axis)
    ncomp = 8
    num_modes = 7

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

        # Scalars are read here and passed to the shared window function, which
        # derives gm1/gam0/typed-literal scalars internally (the per-interface
        # arithmetic — including all x64/Triton dtype hygiene — lives there so
        # the forward kernel and the jax.vjp adjoint share one source of truth).
        gamma = gamma_ref[()]
        b_eps = b_eps_ref[()]
        sqrt_floor = sqrt_floor_ref[()]
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

        q_stencil = tuple(q_local(off) for off in range(-2, 4))     # offsets -2..3
        flux_acc = _weno_mhd_flux_from_window(
            q_stencil, gamma, rhomin, pgmin, b_eps, sqrt_floor, ncomp, num_modes
        )

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


def _weno_flux_mhd_pallas_vjp_local(
    conserved_state,
    flux_bar,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    *,
    axis: int,
):
    """Native Pallas adjoint (VJP) of :func:`_weno_flux_mhd_pallas_local`.

    Given the primal input ``conserved_state`` and the output cotangent
    ``flux_bar`` (same shape as the MHD WENO flux), returns the input cotangent
    w.r.t. ``conserved_state``.  Each grid block gathers its 6-cell WENO stencil
    from ``q_ref`` exactly as the forward kernel does and runs ``jax.vjp`` of the
    *shared* per-window flux :func:`_weno_mhd_flux_from_window` against the
    tile's flux cotangent, so the Pallas backward is the exact transpose of the
    Pallas forward by construction (no separately-derived MHD adjoint math — on
    jax >= 0.10 the in-kernel auto-VJP lowers cleanly on Triton and compiles at
    parity with the primal; the old miscompile that forced the hydro
    hand-derivation is gone).  The six per-offset stencil contributions are
    emitted into six non-overlapping BlockSpec-tiled buffers and assembled by a
    plain-JAX ``roll``-and-sum afterwards (avoids cross-block atomics).

    Like the hydro adjoint, the kernel always gathers/scatters along the
    *leading* spatial axis: for ``axis != 0`` the flux axis is transposed to the
    front (the in-kernel VJP is bit-exact for every axis in interpret mode, but
    older Triton miscompiled a non-leading gather axis; the transpose keeps the
    GPU-correct axis-0 path).  The momentum/magnetic component permutation lives
    in the variable axis (``local_indices``) and is applied/undone in plain JAX.

    Single-device build (no ``shard_map``): the inverse-problem regime runs on
    one GPU, differentiating w.r.t. the conserved STATE only (params are physical
    constants for the flux)."""
    ndim = int(config.dimensionality)
    assert ndim == 3, "MHD Pallas adjoint is built for 3D"
    local_indices = _mhd_indices_for_axis(config, registered_variables, axis)
    li = jnp.asarray(local_indices)
    ncomp = len(local_indices)  # 8
    num_modes = 7

    # Bring the flux spatial axis to the front AND permute the conserved
    # components into characteristic order, both in plain JAX (the in-kernel VJP
    # is only GPU-correct for an identity-indexed, axis-0 gather).  Undone after.
    if axis == 0:
        cs_s, fb_s, inv_perm = conserved_state, flux_bar, None
    else:
        perm = [0, axis + 1] + [a for a in range(1, ndim + 1) if a != axis + 1]
        inv_perm = [perm.index(i) for i in range(ndim + 1)]
        cs_s = jnp.transpose(conserved_state, perm)
        fb_s = jnp.transpose(flux_bar, perm)
    cs = cs_s[li]  # characteristic component order -> identity inside the kernel
    fb = fb_s[li]

    nvars = int(cs.shape[0])
    spatial_shape = tuple(int(x) for x in cs.shape[1:])
    nx, ny, nz = spatial_shape
    bx, by, bz = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    grid = (nx // bx, ny // by, nz // bz)

    offsets = tuple(range(-2, 4))  # WENO5 stencil window

    # Same typed floors as the forward kernel (x64 + Triton dtype hygiene).
    b_eps_value = 1e-20
    sqrt_floor_value = 1e-12

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    full_spec = pl.BlockSpec(cs.shape, lambda bi, bj, bk: tuple([0] * (ndim + 1)))
    block_shape = (nvars, bx, by, bz)
    out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(q_ref, fbar_ref, gamma_ref, rhomin_ref, pgmin_ref, b_eps_ref,
               sqrt_floor_ref, *out_refs):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)

        ii = (bi * bx + jnp.arange(bx)[:, None, None]) % nx
        jj = (bj * by + jnp.arange(by)[None, :, None]) % ny
        kk = (bk * bz + jnp.arange(bz)[None, None, :]) % nz

        gamma = gamma_ref[()]
        rhomin = rhomin_ref[()]
        pgmin = pgmin_ref[()]
        b_eps = b_eps_ref[()]
        sqrt_floor = sqrt_floor_ref[()]

        def shifted_index(offset: int):
            # Always offset the leading spatial axis (the flux axis, brought to
            # the front by the transpose above).
            return ((ii + offset) % nx, jj, kk)

        def q_local(offset: int):
            idx = shifted_index(offset)
            return tuple(q_ref[(var,) + idx] for var in range(ncomp))

        q_stencil = tuple(q_local(o) for o in offsets)
        own = shifted_index(0)
        flux_bar_slot = tuple(fbar_ref[(var,) + own] for var in range(ncomp))

        if _MHD_VJP_USE_HAND_ADJOINT:
            # Hand-derived explicit adjoint (mostly pure elementwise arithmetic;
            # the only in-kernel jax.vjp is of the small per-face eigenvector
            # building-block map).  This replaces the whole-window jax.vjp, whose
            # Triton lowering made the MHD backward ~63x the forward; the hand
            # adjoint shrinks the vjp scope to a handful of face scalars.
            qbar_stencil = _weno_mhd_flux_from_window_adjoint(
                q_stencil, flux_bar_slot, gamma, rhomin, pgmin, b_eps,
                sqrt_floor, ncomp, num_modes)
        else:
            # Legacy fallback: in-kernel jax.vjp of the whole shared window flux
            # (the exact transpose of the forward, but slow to lower on Triton).
            flat = [q_stencil[k][c] for k in range(6) for c in range(ncomp)]

            def wf(*flat_qs):
                qs = tuple(
                    tuple(flat_qs[k * ncomp + c] for c in range(ncomp))
                    for k in range(6)
                )
                return tuple(_weno_mhd_flux_from_window(
                    qs, gamma, rhomin, pgmin, b_eps, sqrt_floor, ncomp, num_modes))

            _, vjp_fn = jax.vjp(wf, *flat)
            cts = vjp_fn(tuple(flux_bar_slot))
            qbar_stencil = [[cts[k * ncomp + c] for c in range(ncomp)] for k in range(6)]

        for o_idx in range(len(offsets)):
            out_ref = out_refs[o_idx]
            for comp in range(ncomp):
                out_ref[comp, ...] = qbar_stencil[o_idx][comp]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shapes = tuple(jax.ShapeDtypeStruct(cs.shape, cs.dtype) for _ in offsets)
    contributions = pl.pallas_call(
        kernel,
        out_shape=out_shapes,
        grid=grid,
        in_specs=[full_spec, full_spec, scalar_spec, scalar_spec, scalar_spec,
                  scalar_spec, scalar_spec],
        out_specs=tuple(out_spec for _ in offsets),
        interpret=config.pallas_interpret,
        name=f"mhd_weno_flux_vjp_axis_{axis}",
        **kwargs,
    )(
        cs,
        fb,
        jnp.asarray(params.gamma, dtype=cs.dtype),
        jnp.asarray(params.minimum_density, dtype=cs.dtype),
        jnp.asarray(params.minimum_pressure, dtype=cs.dtype),
        jnp.asarray(b_eps_value, dtype=cs.dtype),
        jnp.asarray(sqrt_floor_value, dtype=cs.dtype),
    )

    # contributions[o] holds, at interface cell i, the cotangent destined for
    # source cell i + offset (along the leading spatial axis = array axis 1):
    # U_bar[j] = sum_o contributions[o][j - offset] = sum_o roll(., offset).
    ubar_char = sum(
        jnp.roll(contributions[o_idx], offset, axis=1)
        for o_idx, offset in enumerate(offsets)
    )
    # Undo the component permutation, then the spatial transpose (plain JAX).
    conserved_state_bar = jnp.zeros_like(cs_s).at[li].set(ubar_char)
    if inv_perm is not None:
        conserved_state_bar = jnp.transpose(conserved_state_bar, inv_perm)
    return conserved_state_bar


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

