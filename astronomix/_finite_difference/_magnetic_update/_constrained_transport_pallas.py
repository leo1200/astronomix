"""Pallas backend for the Constrained-Transport (CT) helpers.

This file is the **Pallas backend** for ``_constrained_transport.py``.
Two CT helpers are ported here:

* ``update_cell_center_fields`` — one Pallas kernel per call.  Just three
  ``interp_face_to_center`` stencils (halo 3 per axis, no chained
  closures) plus the cell-centered B / energy update.  Compiles in well
  under a second.

* ``constrained_transport_rhs_from_slices`` — split into **three**
  bounded-halo Pallas kernels so Triton never has to lower the full
  chained EMF closure tree at once.  Each kernel is a thin per-cell
  stencil with halo ≤ 4 per axis.  Their JAX-level glue materialises one
  intermediate per stage instead of the 12+ named intermediates the
  native code carries:

    Stage 1 — ``_ct_modified_flux_pallas``
        rho, mom_*, B_*  +  6 raw B-flux slices
          → 6 ``B_flux_axis_mod`` slices  (halo 2 along one axis each).

    Stage 2 — ``_ct_edge_emf_pallas``
        6 modified fluxes  →  Omega_z, Omega_x, Omega_y at cell edges
        (halo 2 per axis, two axes per output).

    Stage 3 — ``_ct_curl_pallas``
        Omega_z, Omega_x, Omega_y (edge values)  →  rhs_b{x,y,z}.
        Fuses ``point_values_to_averages`` smoothing with the
        ``finite_difference_int6`` curl in one tile; combined halo ≤ 4
        along the curl axis, ≤ 1 along the smoothing axis.

The developer never touches this file by hand; the pallasify skill
regenerates it from the native ``_constrained_transport.py`` when those
change.  See ``pallas_backend_implementation_guide.md`` §4.4 for the
recipe and the diagnostic that motivated the split.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_compiler_params,
    pl,
)


XAXIS = 0
YAXIS = 1
ZAXIS = 2


def _ct_pallas_block_ok(state_shape, config: SimulationConfig) -> bool:
    """Spatial-block divisibility check shared by all CT kernels."""
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(config.pallas_block_shape, ndim)
    for n, b in zip(state_shape[1:], block_shape[:ndim], strict=True):
        if int(n) % int(b) != 0:
            return False
    return True


# -----------------------------------------------------------------------------
# Cell-center reconstruction from interface B-fields (one Pallas kernel).
# -----------------------------------------------------------------------------


def _ct_update_cell_center_fields_pallas_supported(state, config: SimulationConfig) -> bool:
    """Whether the Pallas cell-center reconstruction can run.

    3D ideal-gas MHD only — the smaller iso/lower-dim paths fall through
    to the native version, which is short and unproblematic.  Enabled
    by default; the single-kernel implementation is just three
    independent face-to-center stencils with halo 3 each, so it compiles
    quickly.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.pallas_ct:
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != IDEAL_GAS:
        return False
    if int(config.dimensionality) != 3:
        return False
    if state.ndim != 4:
        return False
    return _ct_pallas_block_ok(state.shape, config)


def _ct_update_cell_center_fields_pallas(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Pallas ``update_cell_center_fields``.  3 face-to-center stencils
    (halo 3 per axis, independent stencils — no chained closures), one
    state pass-through + cell-centered B + energy fix-up.
    """
    assert _ct_update_cell_center_fields_pallas_supported(conserved_state, config)
    nvars = int(conserved_state.shape[0])
    nx, ny, nz = (int(x) for x in conserved_state.shape[1:])
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, 3)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    BX = int(registered_variables.magnetic_index.x)
    BY = int(registered_variables.magnetic_index.y)
    BZ = int(registered_variables.magnetic_index.z)
    E = int(registered_variables.pressure_index)

    state_out_spec = pl.BlockSpec((nvars, bx_blk, by_blk, bz_blk),
                                  lambda bi, bj, bk: (0, bi, bj, bk))
    state_in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    b_in_spec = pl.BlockSpec(bx_interface.shape, lambda bi, bj, bk: (0, 0, 0))

    def kernel(q_ref, bx_ref, by_ref, bz_ref, out_ref):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        # interp_face_to_center coefficients (3, -25, 150, 150, -25, 3) / 256
        # over offsets (-3, -2, -1, 0, 1, 2) along the axis: see the native
        # ``interp_face_to_center`` derivation in ``_interpolate.py``.
        def f2c_x():
            return (
                3.0  * bx_ref[(ii - 3) % nx, jj, kk]
              - 25.0  * bx_ref[(ii - 2) % nx, jj, kk]
              + 150.0 * bx_ref[(ii - 1) % nx, jj, kk]
              + 150.0 * bx_ref[ii,            jj, kk]
              - 25.0  * bx_ref[(ii + 1) % nx, jj, kk]
              + 3.0   * bx_ref[(ii + 2) % nx, jj, kk]
            ) / 256.0

        def f2c_y():
            return (
                3.0  * by_ref[ii, (jj - 3) % ny, kk]
              - 25.0  * by_ref[ii, (jj - 2) % ny, kk]
              + 150.0 * by_ref[ii, (jj - 1) % ny, kk]
              + 150.0 * by_ref[ii, jj,            kk]
              - 25.0  * by_ref[ii, (jj + 1) % ny, kk]
              + 3.0   * by_ref[ii, (jj + 2) % ny, kk]
            ) / 256.0

        def f2c_z():
            return (
                3.0  * bz_ref[ii, jj, (kk - 3) % nz]
              - 25.0  * bz_ref[ii, jj, (kk - 2) % nz]
              + 150.0 * bz_ref[ii, jj, (kk - 1) % nz]
              + 150.0 * bz_ref[ii, jj, kk]
              - 25.0  * bz_ref[ii, jj, (kk + 1) % nz]
              + 3.0   * bz_ref[ii, jj, (kk + 2) % nz]
            ) / 256.0

        Bx_center = f2c_x()
        By_center = f2c_y()
        Bz_center = f2c_z()

        Bx_old = q_ref[BX, ii, jj, kk]
        By_old = q_ref[BY, ii, jj, kk]
        Bz_old = q_ref[BZ, ii, jj, kk]
        b2_old = Bx_old * Bx_old + By_old * By_old + Bz_old * Bz_old
        b2_new = Bx_center * Bx_center + By_center * By_center + Bz_center * Bz_center
        E_old = q_ref[E, ii, jj, kk]
        E_new = E_old + 0.5 * (b2_new - b2_old)

        for var in range(nvars):
            if var == BX:
                out_ref[var, ...] = Bx_center
            elif var == BY:
                out_ref[var, ...] = By_center
            elif var == BZ:
                out_ref[var, ...] = Bz_center
            elif var == E:
                out_ref[var, ...] = E_new
            else:
                out_ref[var, ...] = q_ref[var, ii, jj, kk]

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(conserved_state.shape, conserved_state.dtype),
        grid=grid,
        in_specs=[state_in_spec, b_in_spec, b_in_spec, b_in_spec],
        out_specs=state_out_spec,
        interpret=config.pallas_interpret,
        name="ct_update_cell_center_fields",
        **kwargs,
    )(conserved_state, bx_interface, by_interface, bz_interface)


# -----------------------------------------------------------------------------
# CT EMF — split into three bounded-halo Pallas kernels.
# -----------------------------------------------------------------------------


def _ct_rhs_pallas_supported(state, config: SimulationConfig) -> bool:
    """Whether the staged Pallas CT-RHS path can run (3D only).  Gated
    on ``config.pallas_ct`` (default off): the staged kernel chain is
    correct and stable but adds ~25 s of one-time compile cost while
    giving only marginal memory savings at production grid sizes."""
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.pallas_ct:
        return False
    if not config.mhd:
        return False
    if int(config.dimensionality) != 3:
        return False
    if state.ndim != 4:
        return False
    return _ct_pallas_block_ok(state.shape, config)


def _ct_modified_flux_pallas(
    conserved_state,
    By_flux_x_interface,
    Bz_flux_x_interface,
    Bx_flux_y_interface,
    Bz_flux_y_interface,
    Bx_flux_z_interface,
    By_flux_z_interface,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Stage 1: per-axis modified magnetic-flux (Eq. 12–17).

    Six outputs, each obtained by adding to the raw WENO magnetic flux
    one ``interp_center_to_face`` of a cell-centered product:

      By_flux_x_mod = By_flux_x + interp_c2f_x(Bx * vy)
      Bz_flux_x_mod = Bz_flux_x + interp_c2f_x(Bx * vz)
      Bx_flux_y_mod = Bx_flux_y + interp_c2f_y(By * vx)
      Bz_flux_y_mod = Bz_flux_y + interp_c2f_y(By * vz)
      Bx_flux_z_mod = Bx_flux_z + interp_c2f_z(Bz * vx)
      By_flux_z_mod = By_flux_z + interp_c2f_z(Bz * vy)

    Worst-case stencil halo: 2 along each axis (independent per output).
    """
    nx, ny, nz = (int(x) for x in conserved_state.shape[1:])
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, 3)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    DENSITY = int(registered_variables.density_index)
    MX = int(registered_variables.momentum_index.x)
    MY = int(registered_variables.momentum_index.y)
    MZ = int(registered_variables.momentum_index.z)
    BX = int(registered_variables.magnetic_index.x)
    BY = int(registered_variables.magnetic_index.y)
    BZ = int(registered_variables.magnetic_index.z)

    out_block = (bx_blk, by_blk, bz_blk)
    state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    field_in_spec = pl.BlockSpec(conserved_state.shape[1:], lambda bi, bj, bk: (0, 0, 0))
    out_spec = pl.BlockSpec(out_block, lambda bi, bj, bk: (bi, bj, bk))

    def kernel(q_ref, byfx_ref, bzfx_ref, bxfy_ref, bzfy_ref, bxfz_ref, byfz_ref,
               byfx_mod_out, bzfx_mod_out, bxfy_mod_out, bzfy_mod_out,
               bxfz_mod_out, byfz_mod_out):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        # Product Bn*vm at a cell offset along ONE axis from (ii,jj,kk).
        def Bvy_at_x(off):
            rho = q_ref[DENSITY, (ii + off) % nx, jj, kk]
            return q_ref[BX, (ii + off) % nx, jj, kk] * q_ref[MY, (ii + off) % nx, jj, kk] / rho

        def Bvz_at_x(off):
            rho = q_ref[DENSITY, (ii + off) % nx, jj, kk]
            return q_ref[BX, (ii + off) % nx, jj, kk] * q_ref[MZ, (ii + off) % nx, jj, kk] / rho

        def Bvx_at_y(off):
            rho = q_ref[DENSITY, ii, (jj + off) % ny, kk]
            return q_ref[BY, ii, (jj + off) % ny, kk] * q_ref[MX, ii, (jj + off) % ny, kk] / rho

        def Bvz_at_y(off):
            rho = q_ref[DENSITY, ii, (jj + off) % ny, kk]
            return q_ref[BY, ii, (jj + off) % ny, kk] * q_ref[MZ, ii, (jj + off) % ny, kk] / rho

        def Bvx_at_z(off):
            rho = q_ref[DENSITY, ii, jj, (kk + off) % nz]
            return q_ref[BZ, ii, jj, (kk + off) % nz] * q_ref[MX, ii, jj, (kk + off) % nz] / rho

        def Bvy_at_z(off):
            rho = q_ref[DENSITY, ii, jj, (kk + off) % nz]
            return q_ref[BZ, ii, jj, (kk + off) % nz] * q_ref[MY, ii, jj, (kk + off) % nz] / rho

        # interp_center_to_face: (-f[i-1] + 9 f[i] + 9 f[i+1] - f[i+2]) / 16
        def c2f(prod):
            return (-prod(-1) + 9.0 * prod(0) + 9.0 * prod(1) - prod(2)) / 16.0

        byfx_mod_out[...] = byfx_ref[ii, jj, kk] + c2f(Bvy_at_x)
        bzfx_mod_out[...] = bzfx_ref[ii, jj, kk] + c2f(Bvz_at_x)
        bxfy_mod_out[...] = bxfy_ref[ii, jj, kk] + c2f(Bvx_at_y)
        bzfy_mod_out[...] = bzfy_ref[ii, jj, kk] + c2f(Bvz_at_y)
        bxfz_mod_out[...] = bxfz_ref[ii, jj, kk] + c2f(Bvx_at_z)
        byfz_mod_out[...] = byfz_ref[ii, jj, kk] + c2f(Bvy_at_z)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = tuple(
        jax.ShapeDtypeStruct(conserved_state.shape[1:], conserved_state.dtype)
        for _ in range(6)
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[state_spec] + [field_in_spec] * 6,
        out_specs=tuple(out_spec for _ in range(6)),
        interpret=config.pallas_interpret,
        name="ct_modified_flux",
        **kwargs,
    )(
        conserved_state,
        By_flux_x_interface, Bz_flux_x_interface,
        Bx_flux_y_interface, Bz_flux_y_interface,
        Bx_flux_z_interface, By_flux_z_interface,
    )


def _ct_edge_emf_pallas(
    By_flux_x_mod, Bz_flux_x_mod,
    Bx_flux_y_mod, Bz_flux_y_mod,
    Bx_flux_z_mod, By_flux_z_mod,
    config: SimulationConfig,
):
    """Stage 2: edge EMFs Omega_z, Omega_x, Omega_y (Eq. 19–21).

      Omega_z = interp_c2f_x(Bx_flux_y_mod) − interp_c2f_y(By_flux_x_mod)
      Omega_x = interp_c2f_y(By_flux_z_mod) − interp_c2f_z(Bz_flux_y_mod)
      Omega_y = interp_c2f_z(Bz_flux_x_mod) − interp_c2f_x(Bx_flux_z_mod)

    Each output uses interp_center_to_face along TWO different axes
    (one per input).  Halo: 2 per axis.
    """
    nx, ny, nz = (int(x) for x in By_flux_x_mod.shape)
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, 3)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    field_spec = pl.BlockSpec(By_flux_x_mod.shape, lambda bi, bj, bk: (0, 0, 0))
    out_spec = pl.BlockSpec((bx_blk, by_blk, bz_blk), lambda bi, bj, bk: (bi, bj, bk))

    def kernel(byfx_ref, bzfx_ref, bxfy_ref, bzfy_ref, bxfz_ref, byfz_ref,
               omz_out, omx_out, omy_out):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        # interp_center_to_face along each axis on a single-channel ref.
        def c2f_x(ref):
            return (
                -ref[(ii - 1) % nx, jj, kk]
              + 9.0 * ref[ii, jj, kk]
              + 9.0 * ref[(ii + 1) % nx, jj, kk]
              - ref[(ii + 2) % nx, jj, kk]
            ) / 16.0

        def c2f_y(ref):
            return (
                -ref[ii, (jj - 1) % ny, kk]
              + 9.0 * ref[ii, jj, kk]
              + 9.0 * ref[ii, (jj + 1) % ny, kk]
              - ref[ii, (jj + 2) % ny, kk]
            ) / 16.0

        def c2f_z(ref):
            return (
                -ref[ii, jj, (kk - 1) % nz]
              + 9.0 * ref[ii, jj, kk]
              + 9.0 * ref[ii, jj, (kk + 1) % nz]
              - ref[ii, jj, (kk + 2) % nz]
            ) / 16.0

        omz_out[...] = c2f_x(bxfy_ref) - c2f_y(byfx_ref)
        omx_out[...] = c2f_y(byfz_ref) - c2f_z(bzfy_ref)
        omy_out[...] = c2f_z(bzfx_ref) - c2f_x(bxfz_ref)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = tuple(
        jax.ShapeDtypeStruct(By_flux_x_mod.shape, By_flux_x_mod.dtype)
        for _ in range(3)
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec] * 6,
        out_specs=tuple(out_spec for _ in range(3)),
        interpret=config.pallas_interpret,
        name="ct_edge_emf",
        **kwargs,
    )(
        By_flux_x_mod, Bz_flux_x_mod,
        Bx_flux_y_mod, Bz_flux_y_mod,
        Bx_flux_z_mod, By_flux_z_mod,
    )


def _ct_curl_pallas(
    Omega_z_edge, Omega_x_edge, Omega_y_edge,
    dtdx, dtdy, dtdz,
    config: SimulationConfig,
):
    """Stage 3: edge-average smoothing (``point_values_to_averages``) +
    6th-order curl (``finite_difference_int6``).  Outputs the three
    interface-B RHS arrays rhs_b{x,y,z}.

    Fuses two short stencils per output: PVA (halo 1 on its two axes) and
    FD6 (halo 3 on its axis).  Worst-case combined halo along any axis is
    therefore ≤ 4, well inside Triton's comfort zone.
    """
    nx, ny, nz = (int(x) for x in Omega_z_edge.shape)
    bx_blk, by_blk, bz_blk = _as_3tuple_block_shape(config.pallas_block_shape, 3)
    grid = (nx // bx_blk, ny // by_blk, nz // bz_blk)

    field_spec = pl.BlockSpec(Omega_z_edge.shape, lambda bi, bj, bk: (0, 0, 0))
    out_spec = pl.BlockSpec((bx_blk, by_blk, bz_blk), lambda bi, bj, bk: (bi, bj, bk))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    c1 = 75.0 / 64.0
    c2 = -25.0 / 384.0
    c3 = 3.0 / 640.0

    def kernel(omz_ref, omx_ref, omy_ref, dtdx_ref, dtdy_ref, dtdz_ref,
               rhs_bx_out, rhs_by_out, rhs_bz_out):
        bi = pl.program_id(0)
        bj = pl.program_id(1)
        bk = pl.program_id(2)
        ii = (bi * bx_blk + jnp.arange(bx_blk)[:, None, None]) % nx
        jj = (bj * by_blk + jnp.arange(by_blk)[None, :, None]) % ny
        kk = (bk * bz_blk + jnp.arange(bz_blk)[None, None, :]) % nz

        dtdx_v = dtdx_ref[()]
        dtdy_v = dtdy_ref[()]
        dtdz_v = dtdz_ref[()]

        # ---- PVA helpers (Omega_bar at one specific offset along EACH
        # of its two smoothing axes; the curl below loops over offsets
        # of these PVA results along its differentiation axis).
        # 3D PVA is on two axes; the third axis stays at the cell.
        def pva_xy_omz(ox, oy, oz):
            # Omega_z is smoothed in (X, Y).
            q_c = omz_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz) % nz]
            sx = (
                omz_ref[(ii + ox + 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + omz_ref[(ii + ox - 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
            ) / 24.0
            sy = (
                omz_ref[(ii + ox) % nx, (jj + oy + 1) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + omz_ref[(ii + ox) % nx, (jj + oy - 1) % ny, (kk + oz) % nz]
            ) / 24.0
            return q_c + sx + sy

        def pva_yz_omx(ox, oy, oz):
            q_c = omx_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz) % nz]
            sy = (
                omx_ref[(ii + ox) % nx, (jj + oy + 1) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + omx_ref[(ii + ox) % nx, (jj + oy - 1) % ny, (kk + oz) % nz]
            ) / 24.0
            sz = (
                omx_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz + 1) % nz]
              - 2.0 * q_c
              + omx_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz - 1) % nz]
            ) / 24.0
            return q_c + sy + sz

        def pva_xz_omy(ox, oy, oz):
            q_c = omy_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz) % nz]
            sx = (
                omy_ref[(ii + ox + 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
              - 2.0 * q_c
              + omy_ref[(ii + ox - 1) % nx, (jj + oy) % ny, (kk + oz) % nz]
            ) / 24.0
            sz = (
                omy_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz + 1) % nz]
              - 2.0 * q_c
              + omy_ref[(ii + ox) % nx, (jj + oy) % ny, (kk + oz - 1) % nz]
            ) / 24.0
            return q_c + sx + sz

        # ---- 6th-order interface FD: c1·(f[i]−f[i−1]) + c2·(f[i+1]−f[i−2])
        # + c3·(f[i+2]−f[i−3]).
        def fd6_x(pva):
            return (
                c1 * (pva(0, 0, 0) - pva(-1, 0, 0))
              + c2 * (pva(1, 0, 0) - pva(-2, 0, 0))
              + c3 * (pva(2, 0, 0) - pva(-3, 0, 0))
            )

        def fd6_y(pva):
            return (
                c1 * (pva(0, 0, 0) - pva(0, -1, 0))
              + c2 * (pva(0, 1, 0) - pva(0, -2, 0))
              + c3 * (pva(0, 2, 0) - pva(0, -3, 0))
            )

        def fd6_z(pva):
            return (
                c1 * (pva(0, 0, 0) - pva(0, 0, -1))
              + c2 * (pva(0, 0, 1) - pva(0, 0, -2))
              + c3 * (pva(0, 0, 2) - pva(0, 0, -3))
            )

        rhs_bx_out[...] = -dtdy_v * fd6_y(pva_xy_omz) + dtdz_v * fd6_z(pva_xz_omy)
        rhs_by_out[...] = -dtdz_v * fd6_z(pva_yz_omx) + dtdx_v * fd6_x(pva_xy_omz)
        rhs_bz_out[...] = -dtdx_v * fd6_x(pva_xz_omy) + dtdy_v * fd6_y(pva_yz_omx)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    out_shape = tuple(
        jax.ShapeDtypeStruct(Omega_z_edge.shape, Omega_z_edge.dtype)
        for _ in range(3)
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec, field_spec, field_spec,
                  scalar_spec, scalar_spec, scalar_spec],
        out_specs=tuple(out_spec for _ in range(3)),
        interpret=config.pallas_interpret,
        name="ct_curl",
        **kwargs,
    )(
        Omega_z_edge, Omega_x_edge, Omega_y_edge,
        jnp.asarray(dtdx, dtype=Omega_z_edge.dtype),
        jnp.asarray(dtdy, dtype=Omega_z_edge.dtype),
        jnp.asarray(dtdz, dtype=Omega_z_edge.dtype),
    )


def _ct_rhs_pallas(
    conserved_state,
    By_flux_x_interface,
    Bz_flux_x_interface,
    Bx_flux_y_interface,
    Bz_flux_y_interface,
    Bx_flux_z_interface,
    By_flux_z_interface,
    dtdx,
    dtdy,
    dtdz,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Three-stage Pallas CT-RHS: chain ``_ct_modified_flux_pallas`` →
    ``_ct_edge_emf_pallas`` → ``_ct_curl_pallas``.

    Each stage is a single bounded-halo Pallas kernel — compile time
    stays sub-second per kernel.  Intermediate JAX arrays between
    stages (6 modified-flux slices, then 3 Omega slices) are single
    channel rather than 8-var so the peak temporary footprint stays
    well below the native code's 12+ intermediates.
    """
    assert _ct_rhs_pallas_supported(conserved_state, config)
    Byfx_mod, Bzfx_mod, Bxfy_mod, Bzfy_mod, Bxfz_mod, Byfz_mod = (
        _ct_modified_flux_pallas(
            conserved_state,
            By_flux_x_interface, Bz_flux_x_interface,
            Bx_flux_y_interface, Bz_flux_y_interface,
            Bx_flux_z_interface, By_flux_z_interface,
            config, registered_variables,
        )
    )
    Omega_z, Omega_x, Omega_y = _ct_edge_emf_pallas(
        Byfx_mod, Bzfx_mod, Bxfy_mod, Bzfy_mod, Bxfz_mod, Byfz_mod,
        config,
    )
    # Free the modified-flux intermediates as soon as Omega is built.
    del Byfx_mod, Bzfx_mod, Bxfy_mod, Bzfy_mod, Bxfz_mod, Byfz_mod
    return _ct_curl_pallas(
        Omega_z, Omega_x, Omega_y,
        dtdx, dtdy, dtdz,
        config,
    )
