"""
Pallas backend for the constrained-transport (CT) helpers of
``_constrained_transport.py``.

Three CT helpers are ported:

* ``update_cell_center_fields``: one Pallas kernel, three independent
  ``interp_face_to_center`` stencils (halo 3 per axis) plus the cell-centred
  magnetic field and energy update.

* ``constrained_transport_rhs_from_slices``: three bounded-halo Pallas kernels,
  so that Triton never has to lower the full chained EMF expression tree at
  once. Each kernel is a short per-cell stencil with a halo of at most 4 cells
  per axis, and the JAX-level glue materialises one intermediate per stage
  instead of the many named intermediates of the native code:

    Stage 1, ``_ct_modified_flux_pallas``
        rho, momenta, B and the six raw magnetic-flux slices
        -> the six modified fluxes (halo 2 along one axis each).

    Stage 2, ``_ct_edge_emf_pallas``
        the six modified fluxes -> Omega_z, Omega_x, Omega_y at the cell edges
        (halo 2 along the two axes of each output).

    Stage 3, ``_ct_curl_pallas``
        the edge EMFs -> the interface-field increments (rhs_bx, rhs_by,
        rhs_bz), fusing the ``point_values_to_averages`` smoothing with the
        ``finite_difference_int6`` curl (halo 4 along the curl axis).

* ``ct_upwind_dissipation_rhs``: two bounded-halo kernels (the upwind edge
  EMFs from the state and the stacked interface fields, then their
  sixth-order curl). Unlike the two above, which are opt-in
  (``backend_config.pallas_ct``), it runs whenever the backend is Pallas.

Multi-GPU: every entry point routes its ``pl.pallas_call`` through
``_pallas_call_sharded`` (a bare ``pl.pallas_call`` is opaque to GSPMD, which
would all-gather the full grid on every device). The single-channel field
slices are therefore stacked along a leading axis (``(6, nx, ny, nz)`` modified
fluxes, ``(3, nx, ny, nz)`` EMFs), so they ride the same variables-first halo
exchange as the conserved state, and each stage costs one ppermute halo
exchange instead of one collective permute per stencil shift of the native
roll-based stencils. When the mesh splits x, stages 2 and 3 compute the
x-dependent and x-free outputs in separate calls, so that the x-free part does
not wait for the x halo exchange; stage 1 can also be skipped for the x faces
when they were formed together with the x WENO flux
(``_ct_rhs_pallas_x_precomputed``).

The ``*_local`` builds derive every shape from their own arguments, so the
same kernel build runs on the global grid (single device) or on a halo-padded
local shard (multi device). Following the Pallas convention of this codebase
(see ``agent_guides/pallas_backend_implementation_guide.md``, §4.5), the
kernels are translations of the native ``_constrained_transport.py`` and are
regenerated from it when the native helpers change.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS, RIEMANN_FLUX

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import _riemann_flux_pallas_supported
from astronomix._finite_difference._magnetic_update._rebuild_energy import rebuilt_total_energy
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _pallas_call_sharded,
    _pallas_compiler_params,
    _pallas_mesh_splits_axis,
    _shard_cells,
    _shifted_index,
    pl,
)


# Coefficients of the sixth-order interface difference
# c1 (f_i - f_{i-1}) + c2 (f_{i+1} - f_{i-2}) + c3 (f_{i+2} - f_{i-3}) of
# ``finite_difference_int6``.
_FD6_COEFFICIENTS = (75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0)


# -------------------------------------------------------------
# ============ ↓ Shared helpers and stencil algebra ↓ =========
# -------------------------------------------------------------
#
# The stencil helpers below combine values that the kernels read from their
# refs; each takes a callable ``value_at(offset)``. They are plain Python and
# are inlined into every kernel at trace time.


def _ct_pallas_block_ok(state_shape, config: SimulationConfig) -> bool:
    """
    Whether the configured Pallas block shape divides every spatial dimension.

    Args:
        state_shape: The shape of a state-shaped array (variable axis leading).
        config: The simulation configuration.

    Returns:
        True if the CT kernels can tile the array.
    """
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=state_shape[1:],
    )
    for extent, block_size in zip(state_shape[1:], block_shape[:ndim], strict=True):
        if int(extent) % int(block_size) != 0:
            return False
    return True


def _ct_block_and_grid(spatial_shape, config: SimulationConfig):
    """
    Block shape and Pallas grid of a kernel whose inputs have the (possibly
    halo-padded) local spatial shape ``spatial_shape``. Called inside every
    ``*_local`` build, so that the grid resizes when ``_pallas_call_sharded``
    hands the build a shard: the blocks cover the cells the kernel writes
    (``_shard_cells``), the shard's own with ``local_output``.

    Args:
        spatial_shape: The spatial shape ``(nx, ny, nz)`` of the inputs.
        config: The simulation configuration.

    Returns:
        ``((nx, ny, nz), (block_x, block_y, block_z), grid)``.
    """
    nx, ny, nz = (int(extent) for extent in spatial_shape)
    _, written_extents = _shard_cells((nx, ny, nz))
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        3,
        spatial_shape=written_extents,
    )
    grid = tuple(extent // block for extent, block in zip(written_extents, block_shape))
    return (nx, ny, nz), block_shape, grid


def _block_cell_indices(block_shape, spatial_shape):
    """
    Cell indices ``(ii, jj, kk)`` of the current Pallas block in inputs of
    the spatial shape ``spatial_shape``, broadcastable over the 3D tile.

    Args:
        block_shape: The block shape ``(block_x, block_y, block_z)``.
        spatial_shape: The (local) spatial shape ``(nx, ny, nz)``.

    Returns:
        The three index arrays.
    """
    block_x, block_y, block_z = block_shape
    nx, ny, nz = spatial_shape
    first_x, first_y, first_z = _shard_cells(spatial_shape)[0]
    ii = _shifted_index(pl.program_id(0) * block_x + jnp.arange(block_x)[:, None, None], first_x) % nx
    jj = _shifted_index(pl.program_id(1) * block_y + jnp.arange(block_y)[None, :, None], first_y) % ny
    kk = _shifted_index(pl.program_id(2) * block_z + jnp.arange(block_z)[None, None, :], first_z) % nz
    return ii, jj, kk


def _wrapped_index(index, offset: int, extent: int):
    """``index`` shifted by ``offset`` with periodic wrap (``index`` itself for 0)."""
    if offset == 0:
        return index
    return (index + offset) % extent


def _compiler_kwargs(config: SimulationConfig) -> dict:
    """The ``compiler_params`` keyword of ``pl.pallas_call`` (empty without Triton)."""
    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params
    return kwargs


def _face_to_center(value_at):
    """
    ``interp_face_to_center``: the coefficients (3, -25, 150, 150, -25, 3) / 256
    over the face offsets (-3, -2, -1, 0, 1, 2) along one axis (derivation in
    ``_spatial_operators/_interpolate.py``).
    """
    return (
        3.0 * value_at(-3)
        - 25.0 * value_at(-2)
        + 150.0 * value_at(-1)
        + 150.0 * value_at(0)
        - 25.0 * value_at(1)
        + 3.0 * value_at(2)
    ) / 256.0


def _center_to_face(value_at):
    """``interp_center_to_face``: (-f[i-1] + 9 f[i] + 9 f[i+1] - f[i+2]) / 16."""
    return (-value_at(-1) + 9.0 * value_at(0) + 9.0 * value_at(1) - value_at(2)) / 16.0


def _point_to_average(value_at, offset, smoothing_axes):
    """
    ``point_values_to_averages`` at the 3D cell offset ``offset``: the point
    value plus (f[+1] - 2 f + f[-1]) / 24 along each of ``smoothing_axes``.

    Args:
        value_at: Callable returning the field at a 3D offset ``(ox, oy, oz)``.
        offset: The 3D offset of the evaluation point.
        smoothing_axes: The (two) axes of the smoothing.

    Returns:
        The smoothed value.
    """
    center = value_at(offset)
    smoothed = center
    for axis in smoothing_axes:
        plus_offset = list(offset)
        plus_offset[axis] += 1
        minus_offset = list(offset)
        minus_offset[axis] -= 1
        smoothed = smoothed + (
            value_at(tuple(plus_offset)) - 2.0 * center + value_at(tuple(minus_offset))
        ) / 24.0
    return smoothed


def _sixth_order_difference(value_at):
    """``finite_difference_int6`` at an interface; ``value_at(k)`` is f[i + k]."""
    first, second, third = _FD6_COEFFICIENTS
    return (
        first * (value_at(0) - value_at(-1))
        + second * (value_at(1) - value_at(-2))
        + third * (value_at(2) - value_at(-3))
    )


def _smoothed_curl(omega_z_at, omega_x_at, omega_y_at, dt_over_dx, dt_over_dy, dt_over_dz):
    """
    The interface-field increments from the edge EMFs: each EMF is smoothed
    in the two directions of its edge plane (Omega_z in x and y, Omega_x in y
    and z, Omega_y in x and z) and differentiated with the sixth-order
    interface difference along the curl direction.

    Args:
        omega_z_at: Callable returning Omega_z at a 3D offset.
        omega_x_at: Callable returning Omega_x at a 3D offset.
        omega_y_at: Callable returning Omega_y at a 3D offset.
        dt_over_dx: The time step over the grid spacing in x.
        dt_over_dy: The time step over the grid spacing in y.
        dt_over_dz: The time step over the grid spacing in z.

    Returns:
        ``(rhs_bx, rhs_by, rhs_bz)``.
    """

    def omega_z_average(offset):
        return _point_to_average(omega_z_at, offset, (0, 1))

    def omega_x_average(offset):
        return _point_to_average(omega_x_at, offset, (1, 2))

    def omega_y_average(offset):
        return _point_to_average(omega_y_at, offset, (0, 2))

    def difference(averaged_field, axis):
        """The sixth-order difference of ``averaged_field`` along ``axis``."""

        def value_along_axis(offset_along_axis):
            offset = [0, 0, 0]
            offset[axis] = offset_along_axis
            return averaged_field(tuple(offset))

        return _sixth_order_difference(value_along_axis)

    rhs_bx = (
        -dt_over_dy * difference(omega_z_average, 1)
        + dt_over_dz * difference(omega_y_average, 2)
    )
    rhs_by = (
        -dt_over_dz * difference(omega_x_average, 2)
        + dt_over_dx * difference(omega_z_average, 0)
    )
    rhs_bz = (
        -dt_over_dx * difference(omega_y_average, 0)
        + dt_over_dy * difference(omega_x_average, 1)
    )
    return rhs_bx, rhs_by, rhs_bz


# -------------------------------------------------------------
# ============ ↑ Shared helpers and stencil algebra ↑ =========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===== ↓ Cell-centred fields from the interface fields ↓ =====
# -------------------------------------------------------------


def _ct_update_cell_center_fields_pallas_supported(state, config: SimulationConfig) -> bool:
    """
    Whether the Pallas cell-centre reconstruction can run.

    3D ideal-gas MHD only; the isothermal and lower-dimensional paths use the
    short native version. Gated on ``config.backend_config.pallas_ct``, and
    under ``scheme = RIEMANN_FLUX`` on the condition of that scheme's kernels.

    Args:
        state: The conserved state.
        config: The simulation configuration.

    Returns:
        True if the Pallas kernel can be used.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.backend_config.pallas_ct:
        return False
    if not config.mhd:
        return False
    if config.equation_of_state != IDEAL_GAS:
        return False
    if int(config.dimensionality) != 3:
        return False
    if state.ndim != 4:
        return False
    if config.solver_config.scheme == RIEMANN_FLUX and not _riemann_flux_pallas_supported(state, config):
        return False
    return _ct_pallas_block_ok(state.shape, config)


def _ct_update_cell_center_fields_pallas(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams = None,
):
    """
    Pallas ``update_cell_center_fields``: the cell-centred magnetic field from
    the interface fields, with the total energy adjusted to the new magnetic
    energy (the pressure held), or under ``scheme = RIEMANN_FLUX`` the energy
    held (``rebuilt_total_energy``, which reads ``params``).

    The three interface fields are stacked along a leading axis so that they
    ride the variables-first halo exchange of ``_pallas_call_sharded`` (halo 3:
    the face-to-centre stencil reads the offsets -3..+2 along each axis).

    Args:
        conserved_state: The conserved state.
        bx_interface: The x-face magnetic field.
        by_interface: The y-face magnetic field.
        bz_interface: The z-face magnetic field.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters (gamma and the floors; read under ``RIEMANN_FLUX``).

    Returns:
        The conserved state with updated cell-centred field and energy.
    """
    assert _ct_update_cell_center_fields_pallas_supported(conserved_state, config)
    ndim = int(config.dimensionality)
    dtype = conserved_state.dtype
    _, block_shape, _ = _ct_block_and_grid(conserved_state.shape[1:], config)
    stacked_interface_fields = jnp.stack([bx_interface, by_interface, bz_interface])
    interface_field_halo = (3, 3, 3)[:ndim]
    zero_halo = (0,) * ndim
    hold_energy = config.solver_config.scheme == RIEMANN_FLUX
    scalars = (params.gamma, params.minimum_density, params.minimum_pressure) if hold_energy else ()

    def build_local(state_local, interface_fields_local, *scalars_local):
        return _ct_update_cell_center_fields_pallas_local(
            state_local,
            interface_fields_local,
            *scalars_local,
            config=config,
            registered_variables=registered_variables,
            hold_energy=hold_energy,
        )

    return _pallas_call_sharded(
        build_local,
        state_inputs=(conserved_state, stacked_interface_fields),
        other_args=tuple(jnp.asarray(scalar, dtype=dtype) for scalar in scalars),
        halo=interface_field_halo,
        # The kernel reads the conserved state only at the cell itself, so
        # only the interface fields (stencil -3..+2 on every axis) are
        # exchanged.
        input_halos=(zero_halo, interface_field_halo),
        block_shape=block_shape[:ndim],
        local_output=True,
    )


def _ct_update_cell_center_fields_pallas_local(
    conserved_state,
    interface_fields,
    *scalars,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    hold_energy: bool = False,
):
    """
    Single-shard build of ``_ct_update_cell_center_fields_pallas``: three
    independent face-to-centre stencils, the cell-centred field and the energy
    update, and a pass-through of every other variable. All shapes come from
    this call's arguments, so the build works on the shard of a device mesh,
    whose cells the padded interface fields hold after their halo
    (``_shard_cells``).

    Args:
        conserved_state: The conserved state.
        interface_fields: The stacked ``(3, nx, ny, nz)`` interface fields.
        *scalars: With ``hold_energy``: gamma, the minimum density and the
            minimum pressure (scalar arrays).
        config: The simulation configuration.
        registered_variables: The registered variables.
        hold_energy: Hold the total energy instead of the pressure (static).

    Returns:
        The updated conserved state.
    """
    num_vars = int(conserved_state.shape[0])
    state_shape, block_shape, grid = _ct_block_and_grid(conserved_state.shape[1:], config)
    interface_shape = tuple(int(extent) for extent in interface_fields.shape[1:])
    nx, ny, nz = interface_shape

    density_index = int(registered_variables.density_index)
    momentum_indices = tuple(int(index) for index in registered_variables.momentum_index)
    magnetic_x_index = int(registered_variables.magnetic_index.x)
    magnetic_y_index = int(registered_variables.magnetic_index.y)
    magnetic_z_index = int(registered_variables.magnetic_index.z)
    energy_index = int(registered_variables.energy_index)

    state_out_spec = pl.BlockSpec(
        (num_vars,) + block_shape,
        lambda bi, bj, bk: (0, bi, bj, bk),
    )
    state_in_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    interface_in_spec = pl.BlockSpec(interface_fields.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(state_ref, interface_ref, *refs):
        scalar_refs, out_ref = refs[:-1], refs[-1]
        ii, jj, kk = _block_cell_indices(block_shape, state_shape)
        same_cells = interface_shape == state_shape
        fi, fj, fk = (ii, jj, kk) if same_cells else _block_cell_indices(block_shape, interface_shape)

        bx_center = _face_to_center(
            lambda offset: interface_ref[0, _wrapped_index(fi, offset, nx), fj, fk]
        )
        by_center = _face_to_center(
            lambda offset: interface_ref[1, fi, _wrapped_index(fj, offset, ny), fk]
        )
        bz_center = _face_to_center(
            lambda offset: interface_ref[2, fi, fj, _wrapped_index(fk, offset, nz)]
        )

        bx_old = state_ref[magnetic_x_index, ii, jj, kk]
        by_old = state_ref[magnetic_y_index, ii, jj, kk]
        bz_old = state_ref[magnetic_z_index, ii, jj, kk]
        magnetic_energy_density_old = bx_old * bx_old + by_old * by_old + bz_old * bz_old
        magnetic_energy_density_new = (
            bx_center * bx_center + by_center * by_center + bz_center * bz_center
        )
        energy_old = state_ref[energy_index, ii, jj, kk]
        if hold_energy:
            gamma, minimum_density, minimum_pressure = (ref[()] for ref in scalar_refs)
            momentum_squared = sum(
                state_ref[index, ii, jj, kk] * state_ref[index, ii, jj, kk] for index in momentum_indices
            )
            energy_new = rebuilt_total_energy(
                energy_old,
                state_ref[density_index, ii, jj, kk],
                momentum_squared,
                magnetic_energy_density_old,
                magnetic_energy_density_new,
                gamma,
                minimum_density,
                minimum_pressure,
            )
        else:
            energy_new = energy_old + 0.5 * (magnetic_energy_density_new - magnetic_energy_density_old)

        for var in range(num_vars):
            if var == magnetic_x_index:
                out_ref[var, ...] = bx_center
            elif var == magnetic_y_index:
                out_ref[var, ...] = by_center
            elif var == magnetic_z_index:
                out_ref[var, ...] = bz_center
            elif var == energy_index:
                out_ref[var, ...] = energy_new
            else:
                out_ref[var, ...] = state_ref[var, ii, jj, kk]

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_vars,) + _shard_cells(state_shape)[1], conserved_state.dtype),
        grid=grid,
        in_specs=[state_in_spec, interface_in_spec] + [scalar_spec] * len(scalars),
        out_specs=state_out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_update_cell_center_fields",
        **_compiler_kwargs(config),
    )(conserved_state, interface_fields, *scalars)


# -------------------------------------------------------------
# ===== ↑ Cell-centred fields from the interface fields ↑ =====
# -------------------------------------------------------------


# -------------------------------------------------------------
# ======= ↓ Stage 1: modified magnetic-field fluxes ↓ =========
# -------------------------------------------------------------


def _ct_rhs_pallas_supported(state, config: SimulationConfig) -> bool:
    """
    Whether the staged Pallas CT right-hand side can run (3D MHD only).

    Gated on ``config.backend_config.pallas_ct`` (default off): the staged
    kernels add compile time and save little memory on a single device, but
    replace the many per-shift collective permutes of the native stencils by
    one halo exchange per stage in sharded runs.

    Args:
        state: The conserved state.
        config: The simulation configuration.

    Returns:
        True if the Pallas kernels can be used.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.backend_config.pallas_ct:
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
    flux_slices,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Stage 1: the modified magnetic-field fluxes of all three face directions.

    Halo 2: the centre-to-face stencil reads the offsets -1..+2 along one axis
    per slice.

    Args:
        conserved_state: The conserved state.
        flux_slices: The stacked ``(6, nx, ny, nz)`` raw WENO magnetic-flux
            slices in the order (By_fx, Bz_fx, Bx_fy, Bz_fy, Bx_fz, By_fz).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked modified fluxes, same layout as ``flux_slices``.
    """
    ndim = int(config.dimensionality)
    _, block_shape, _ = _ct_block_and_grid(conserved_state.shape[1:], config)
    state_halo = (2, 2, 2)[:ndim]
    zero_halo = (0,) * ndim

    def build_local(state_local, flux_local):
        return _ct_modified_flux_pallas_local(
            state_local,
            flux_local,
            config,
            registered_variables,
        )

    return _pallas_call_sharded(
        build_local,
        state_inputs=(conserved_state, flux_slices),
        halo=state_halo,
        # The flux slices are read only at the cell itself; only the
        # conserved state (stencil -1..+2) is exchanged.
        input_halos=(state_halo, zero_halo),
        block_shape=block_shape[:ndim],
    )


def _magnetic_velocity_product_reader(state_ref, registered_variables, ii, jj, kk, spatial_shape):
    """
    Return ``product(field_component, velocity_component, axis, offset)``, the
    cell-centred product B_field * v_velocity of the state at ``offset`` cells
    along ``axis`` from the block's cells.

    Args:
        state_ref: The conserved-state ref of the kernel.
        registered_variables: The registered variables.
        ii: The block's x cell indices.
        jj: The block's y cell indices.
        kk: The block's z cell indices.
        spatial_shape: The (local) spatial shape ``(nx, ny, nz)``.

    Returns:
        The product reader.
    """
    nx, ny, nz = spatial_shape
    density_index = int(registered_variables.density_index)
    momentum_indices = (
        int(registered_variables.momentum_index.x),
        int(registered_variables.momentum_index.y),
        int(registered_variables.momentum_index.z),
    )
    magnetic_indices = (
        int(registered_variables.magnetic_index.x),
        int(registered_variables.magnetic_index.y),
        int(registered_variables.magnetic_index.z),
    )

    def product(field_component, velocity_component, axis, offset):
        if axis == 0:
            cell = ((ii + offset) % nx, jj, kk)
        elif axis == 1:
            cell = (ii, (jj + offset) % ny, kk)
        else:
            cell = (ii, jj, (kk + offset) % nz)
        density = state_ref[(density_index,) + cell]
        return (
            state_ref[(magnetic_indices[field_component],) + cell]
            * state_ref[(momentum_indices[velocity_component],) + cell]
            / density
        )

    return product


def _ct_modified_flux_pallas_local(
    conserved_state,
    flux_slices,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Single-shard build of stage 1: the per-axis modified magnetic-field fluxes
    (Eqs. 12-17). Each output channel adds to a raw WENO magnetic flux the
    ``interp_center_to_face`` of a cell-centred product:

      out[0] = By_flux_x + interp_c2f_x(Bx * vy)
      out[1] = Bz_flux_x + interp_c2f_x(Bx * vz)
      out[2] = Bx_flux_y + interp_c2f_y(By * vx)
      out[3] = Bz_flux_y + interp_c2f_y(By * vz)
      out[4] = Bx_flux_z + interp_c2f_z(Bz * vx)
      out[5] = By_flux_z + interp_c2f_z(Bz * vy)

    The stencil reaches two cells along the face axis of each output.

    Args:
        conserved_state: The conserved state (possibly halo-padded).
        flux_slices: The stacked raw magnetic-flux slices.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked modified fluxes.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(conserved_state.shape[1:], config)

    state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    flux_in_spec = pl.BlockSpec(flux_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((6,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(state_ref, flux_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        product = _magnetic_velocity_product_reader(
            state_ref,
            registered_variables,
            ii,
            jj,
            kk,
            spatial_shape,
        )

        # (field component, velocity component, face axis) of each output.
        for channel, (field, velocity, axis) in enumerate(
            ((0, 1, 0), (0, 2, 0), (1, 0, 1), (1, 2, 1), (2, 0, 2), (2, 1, 2))
        ):
            out_ref[channel, ...] = flux_ref[channel, ii, jj, kk] + _center_to_face(
                lambda offset: product(field, velocity, axis, offset)
            )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(flux_slices.shape, flux_slices.dtype),
        grid=grid,
        in_specs=[state_spec, flux_in_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_modified_flux",
        **_compiler_kwargs(config),
    )(conserved_state, flux_slices)


def _ct_modified_flux_yz_pallas(
    conserved_state,
    flux_yz_slices,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Stage 1 for the y- and z-face fluxes only: the modified fluxes
    ``(Bx_fy, Bz_fy, Bx_fz, By_fz)`` when the x-face ones were already formed
    together with the x WENO flux (multi-GPU fast path along a split x axis).

    Args:
        conserved_state: The conserved state.
        flux_yz_slices: The stacked y- and z-face induction fluxes.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked modified fluxes, same layout as ``flux_yz_slices``.
    """
    ndim = int(config.dimensionality)
    _, block_shape, _ = _ct_block_and_grid(conserved_state.shape[1:], config)
    state_halo = (0, 2, 2)[:ndim]
    zero_halo = (0,) * ndim

    def build_local(state_local, flux_local):
        return _ct_modified_flux_yz_pallas_local(
            state_local,
            flux_local,
            config,
            registered_variables,
        )

    return _pallas_call_sharded(
        build_local,
        state_inputs=(conserved_state, flux_yz_slices),
        halo=state_halo,
        # The flux slices are read only at the cell itself and the conserved
        # state only along y and z (stencil -1..+2), so only its y/z halo is
        # exchanged.
        input_halos=(state_halo, zero_halo),
        block_shape=block_shape[:ndim],
    )


def _ct_modified_flux_yz_pallas_local(
    conserved_state,
    flux_yz_slices,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Single-shard build of ``_ct_modified_flux_yz_pallas`` (channels 2..5 of
    ``_ct_modified_flux_pallas_local``).

    Args:
        conserved_state: The conserved state (possibly halo-padded).
        flux_yz_slices: The stacked y- and z-face raw magnetic-flux slices.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked modified y- and z-face fluxes.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(conserved_state.shape[1:], config)

    state_spec = pl.BlockSpec(conserved_state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    flux_in_spec = pl.BlockSpec(flux_yz_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((4,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(state_ref, flux_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        product = _magnetic_velocity_product_reader(
            state_ref,
            registered_variables,
            ii,
            jj,
            kk,
            spatial_shape,
        )

        # (field component, velocity component, face axis) of each output.
        for channel, (field, velocity, axis) in enumerate(
            ((1, 0, 1), (1, 2, 1), (2, 0, 2), (2, 1, 2))
        ):
            out_ref[channel, ...] = flux_ref[channel, ii, jj, kk] + _center_to_face(
                lambda offset: product(field, velocity, axis, offset)
            )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(flux_yz_slices.shape, flux_yz_slices.dtype),
        grid=grid,
        in_specs=[state_spec, flux_in_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_modified_flux_yz",
        **_compiler_kwargs(config),
    )(conserved_state, flux_yz_slices)


# -------------------------------------------------------------
# ======= ↑ Stage 1: modified magnetic-field fluxes ↑ =========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ Stage 2: edge EMFs ↓ ====================
# -------------------------------------------------------------


def _center_to_face_reader(field_ref, ii, jj, kk, spatial_shape):
    """
    Return ``interpolate(channel, axis)``, the centre-to-face interpolation of
    one channel of ``field_ref`` along ``axis`` at the block's cells.

    Args:
        field_ref: The ref of the stacked input channels.
        ii: The block's x cell indices.
        jj: The block's y cell indices.
        kk: The block's z cell indices.
        spatial_shape: The (local) spatial shape ``(nx, ny, nz)``.

    Returns:
        The interpolation function.
    """
    nx, ny, nz = spatial_shape

    def interpolate(channel, axis):
        if axis == 0:
            return _center_to_face(
                lambda offset: field_ref[channel, _wrapped_index(ii, offset, nx), jj, kk]
            )
        if axis == 1:
            return _center_to_face(
                lambda offset: field_ref[channel, ii, _wrapped_index(jj, offset, ny), kk]
            )
        return _center_to_face(
            lambda offset: field_ref[channel, ii, jj, _wrapped_index(kk, offset, nz)]
        )

    return interpolate


def _ct_edge_emf_pallas(flux_mod_slices, config: SimulationConfig):
    """
    Stage 2: the edge EMFs from the stacked modified fluxes (halo 2: the
    centre-to-face stencil along two axes per output).

    Args:
        flux_mod_slices: The stacked ``(6, nx, ny, nz)`` modified fluxes.
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` EMFs (Omega_z, Omega_x, Omega_y).
    """
    # With x split over devices, the x-dependent and x-free parts are computed
    # separately so only the former waits for the x halo exchange.
    if _pallas_mesh_splits_axis(flux_mod_slices, 0):
        return _ct_edge_emf_pallas_split_x(flux_mod_slices, config)

    ndim = int(config.dimensionality)
    _, block_shape, _ = _ct_block_and_grid(flux_mod_slices.shape[1:], config)

    def build_local(flux_local):
        return _ct_edge_emf_pallas_local(flux_local, config)

    return _pallas_call_sharded(
        build_local,
        state_inputs=(flux_mod_slices,),
        halo=(2, 2, 2)[:ndim],
        block_shape=block_shape[:ndim],
    )


def _ct_edge_emf_pallas_local(flux_mod_slices, config: SimulationConfig):
    """
    Single-shard build of stage 2: the edge EMFs (Eqs. 19-21), stacked as the
    output channels (0, 1, 2):

      Omega_z = interp_c2f_x(Bx_flux_y_mod) - interp_c2f_y(By_flux_x_mod)
      Omega_x = interp_c2f_y(By_flux_z_mod) - interp_c2f_z(Bz_flux_y_mod)
      Omega_y = interp_c2f_z(Bz_flux_x_mod) - interp_c2f_x(Bx_flux_z_mod)

    The input channels follow stage 1's output, (By_fx, Bz_fx, Bx_fy, Bz_fy,
    Bx_fz, By_fz); the stencil reaches two cells along each axis.

    Args:
        flux_mod_slices: The stacked modified fluxes (possibly halo-padded).
        config: The simulation configuration.

    Returns:
        The stacked EMFs.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(flux_mod_slices.shape[1:], config)

    field_spec = pl.BlockSpec(flux_mod_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((3,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(flux_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        interpolate = _center_to_face_reader(flux_ref, ii, jj, kk, spatial_shape)

        out_ref[0, ...] = interpolate(2, 0) - interpolate(0, 1)  # Omega_z
        out_ref[1, ...] = interpolate(5, 1) - interpolate(3, 2)  # Omega_x
        out_ref[2, ...] = interpolate(1, 2) - interpolate(4, 0)  # Omega_y

    out_shape = jax.ShapeDtypeStruct(
        (3,) + tuple(flux_mod_slices.shape[1:]),
        flux_mod_slices.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_edge_emf",
        **_compiler_kwargs(config),
    )(flux_mod_slices)


def _ct_edge_emf_pallas_split_x(flux_mod_slices, config: SimulationConfig):
    """
    Stage 2 for a mesh that splits x: the edge EMFs of ``_ct_edge_emf_pallas``
    computed as an x-dependent part (Omega_z, Omega_y) and an x-free part
    (Omega_x), so the latter needs no x halo exchange.

    Args:
        flux_mod_slices: The stacked ``(6, nx, ny, nz)`` modified fluxes.
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` EMFs (Omega_z, Omega_x, Omega_y).
    """
    ndim = int(config.dimensionality)
    _, block_shape, _ = _ct_block_and_grid(flux_mod_slices.shape[1:], config)
    emf_halo = (2, 2, 2)[:ndim]
    x_free_halo = (0, 2, 2)[:ndim]

    # Channels interpolated along x (Bx_fy, Bx_fz), along y or z for the
    # x-dependent EMFs (By_fx, Bz_fx), and the two of the x-free EMF
    # (Bz_fy, By_fz).
    flux_x_line = jnp.stack([flux_mod_slices[2], flux_mod_slices[4]])
    flux_yz_line = jnp.stack([flux_mod_slices[0], flux_mod_slices[1]])
    flux_x_free = jnp.stack([flux_mod_slices[3], flux_mod_slices[5]])

    # Omega_z and Omega_y need x neighbours, Omega_x does not. Computing them
    # in two independent calls gives XLA the chance to overlap the x-free part
    # with the x halo exchange of the other.
    omega_z_and_y = _pallas_call_sharded(
        lambda flux_x_local, flux_yz_local: _ct_edge_emf_xdep_pallas_local(
            flux_x_local,
            flux_yz_local,
            config,
        ),
        state_inputs=(flux_x_line, flux_yz_line),
        halo=emf_halo,
        # The x-face fluxes are shifted only along x and the y/z-face fluxes
        # only along y and z (offsets -1..+2).
        input_halos=((2, 0, 0)[:ndim], x_free_halo),
        block_shape=block_shape[:ndim],
    )

    # Omega_x only reads y and z neighbours: with a mesh split along x alone
    # it is a purely local computation.
    omega_x = _pallas_call_sharded(
        lambda flux_local: _ct_edge_emf_xfree_pallas_local(flux_local, config),
        state_inputs=(flux_x_free,),
        halo=x_free_halo,
        input_halos=(x_free_halo,),
        block_shape=block_shape[:ndim],
    )
    return jnp.stack([omega_z_and_y[0], omega_x[0], omega_z_and_y[1]])


def _ct_edge_emf_xdep_pallas_local(flux_xline, flux_yzline, config: SimulationConfig):
    """
    Single-shard build of the x-dependent EMFs (Omega_z, Omega_y) of
    ``_ct_edge_emf_pallas_split_x``.

    Args:
        flux_xline: The stacked (Bx_fy, Bx_fz) modified fluxes.
        flux_yzline: The stacked (By_fx, Bz_fx) modified fluxes.
        config: The simulation configuration.

    Returns:
        The stacked ``(2, nx, ny, nz)`` EMFs (Omega_z, Omega_y).
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(flux_xline.shape[1:], config)

    x_line_spec = pl.BlockSpec(flux_xline.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    yz_line_spec = pl.BlockSpec(flux_yzline.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((2,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(flux_x_ref, flux_yz_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        interpolate_x_line = _center_to_face_reader(flux_x_ref, ii, jj, kk, spatial_shape)
        interpolate_yz_line = _center_to_face_reader(flux_yz_ref, ii, jj, kk, spatial_shape)

        out_ref[0, ...] = interpolate_x_line(0, 0) - interpolate_yz_line(0, 1)  # Omega_z
        out_ref[1, ...] = interpolate_yz_line(1, 2) - interpolate_x_line(1, 0)  # Omega_y

    out_shape = jax.ShapeDtypeStruct(
        (2,) + tuple(flux_xline.shape[1:]),
        flux_xline.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[x_line_spec, yz_line_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_edge_emf_xdep",
        **_compiler_kwargs(config),
    )(flux_xline, flux_yzline)


def _ct_edge_emf_xfree_pallas_local(flux_mod_slices, config: SimulationConfig):
    """
    Single-shard build of the x-free EMF (Omega_x) of
    ``_ct_edge_emf_pallas_split_x``.

    Args:
        flux_mod_slices: The stacked (Bz_fy, By_fz) modified fluxes.
        config: The simulation configuration.

    Returns:
        Omega_x as a ``(1, nx, ny, nz)`` array.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(flux_mod_slices.shape[1:], config)

    field_spec = pl.BlockSpec(flux_mod_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((1,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(flux_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        interpolate = _center_to_face_reader(flux_ref, ii, jj, kk, spatial_shape)

        out_ref[0, ...] = interpolate(1, 1) - interpolate(0, 2)  # Omega_x

    out_shape = jax.ShapeDtypeStruct(
        (1,) + tuple(flux_mod_slices.shape[1:]),
        flux_mod_slices.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_edge_emf_xfree",
        **_compiler_kwargs(config),
    )(flux_mod_slices)


# -------------------------------------------------------------
# ================= ↑ Stage 2: edge EMFs ↑ ====================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============= ↓ Stage 3: smoothed sixth-order curl ↓ ========
# -------------------------------------------------------------


def _channel_reader(field_ref, channel, ii, jj, kk, spatial_shape):
    """
    Return ``value_at((ox, oy, oz))``, one channel of ``field_ref`` at a 3D
    offset from the block's cells (periodic wrap).

    Args:
        field_ref: The ref of the stacked channels.
        channel: The channel to read.
        ii: The block's x cell indices.
        jj: The block's y cell indices.
        kk: The block's z cell indices.
        spatial_shape: The (local) spatial shape ``(nx, ny, nz)``.

    Returns:
        The reader.
    """
    nx, ny, nz = spatial_shape

    def value_at(offset):
        offset_x, offset_y, offset_z = offset
        return field_ref[
            channel,
            (ii + offset_x) % nx,
            (jj + offset_y) % ny,
            (kk + offset_z) % nz,
        ]

    return value_at


def _ct_curl_pallas(omega_slices, dtdx, dtdy, dtdz, config: SimulationConfig):
    """
    Stage 3: the smoothed sixth-order curl of the stacked edge EMFs (halo 4:
    the reach 3 of the sixth-order difference plus 1 for the fused smoothing).

    Args:
        omega_slices: The stacked ``(3, nx, ny, nz)`` EMFs (Omega_z, Omega_x,
            Omega_y).
        dtdx: The time step over the grid spacing in x.
        dtdy: The time step over the grid spacing in y.
        dtdz: The time step over the grid spacing in z.
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` interface-field increments.
    """
    # Same split of the x-dependent and x-free parts as for the edge EMFs.
    if _pallas_mesh_splits_axis(omega_slices, 0):
        return _ct_curl_pallas_split_x(omega_slices, dtdx, dtdy, dtdz, config)

    ndim = int(config.dimensionality)
    _, block_shape, _ = _ct_block_and_grid(omega_slices.shape[1:], config)

    def build_local(omega_local, dt_over_dx, dt_over_dy, dt_over_dz):
        return _ct_curl_pallas_local(
            omega_local,
            dt_over_dx,
            dt_over_dy,
            dt_over_dz,
            config,
        )

    return _pallas_call_sharded(
        build_local,
        state_inputs=(omega_slices,),
        other_args=(
            jnp.asarray(dtdx, dtype=omega_slices.dtype),
            jnp.asarray(dtdy, dtype=omega_slices.dtype),
            jnp.asarray(dtdz, dtype=omega_slices.dtype),
        ),
        halo=(4, 4, 4)[:ndim],
        block_shape=block_shape[:ndim],
    )


def _ct_curl_pallas_split_x(omega_slices, dtdx, dtdy, dtdz, config: SimulationConfig):
    """
    Stage 3 for a mesh that splits x: the smoothed curl of ``_ct_curl_pallas``
    with Omega_x exchanged only along y and z.

    Args:
        omega_slices: The stacked ``(3, nx, ny, nz)`` EMFs (Omega_z, Omega_x,
            Omega_y).
        dtdx: The time step over the grid spacing in x.
        dtdy: The time step over the grid spacing in y.
        dtdz: The time step over the grid spacing in z.
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` interface-field increments.
    """
    ndim = int(config.dimensionality)
    _, block_shape, _ = _ct_block_and_grid(omega_slices.shape[1:], config)
    curl_halo = (4, 4, 4)[:ndim]
    x_free_halo = (0, 4, 4)[:ndim]

    omega_x_dependent = jnp.stack([omega_slices[0], omega_slices[2]])
    omega_x_free = omega_slices[1:2]

    def build_local(
        omega_x_dependent_local,
        omega_x_free_local,
        dt_over_dx,
        dt_over_dy,
        dt_over_dz,
    ):
        return _ct_curl_split_x_pallas_local(
            omega_x_dependent_local,
            omega_x_free_local,
            dt_over_dx,
            dt_over_dy,
            dt_over_dz,
            config,
        )

    return _pallas_call_sharded(
        build_local,
        state_inputs=(omega_x_dependent, omega_x_free),
        other_args=(
            jnp.asarray(dtdx, dtype=omega_slices.dtype),
            jnp.asarray(dtdy, dtype=omega_slices.dtype),
            jnp.asarray(dtdz, dtype=omega_slices.dtype),
        ),
        halo=curl_halo,
        # Omega_z and Omega_y enter the x derivatives; Omega_x is only
        # differentiated (and smoothed) along y and z.
        input_halos=(curl_halo, x_free_halo),
        block_shape=block_shape[:ndim],
    )


def _ct_curl_split_x_pallas_local(
    omega_xdep,
    omega_xfree,
    dtdx,
    dtdy,
    dtdz,
    config: SimulationConfig,
):
    """
    Single-shard build of ``_ct_curl_pallas_split_x``.

    Args:
        omega_xdep: The stacked (Omega_z, Omega_y) EMFs.
        omega_xfree: Omega_x as a ``(1, nx, ny, nz)`` array.
        dtdx: The time step over the grid spacing in x (scalar array).
        dtdy: The time step over the grid spacing in y (scalar array).
        dtdz: The time step over the grid spacing in z (scalar array).
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` interface-field increments.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(omega_xdep.shape[1:], config)

    x_dependent_spec = pl.BlockSpec(omega_xdep.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    x_free_spec = pl.BlockSpec(omega_xfree.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((3,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(
        omega_x_dependent_ref,
        omega_x_free_ref,
        dt_over_dx_ref,
        dt_over_dy_ref,
        dt_over_dz_ref,
        out_ref,
    ):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        rhs_bx, rhs_by, rhs_bz = _smoothed_curl(
            _channel_reader(omega_x_dependent_ref, 0, ii, jj, kk, spatial_shape),
            _channel_reader(omega_x_free_ref, 0, ii, jj, kk, spatial_shape),
            _channel_reader(omega_x_dependent_ref, 1, ii, jj, kk, spatial_shape),
            dt_over_dx_ref[()],
            dt_over_dy_ref[()],
            dt_over_dz_ref[()],
        )
        out_ref[0, ...] = rhs_bx
        out_ref[1, ...] = rhs_by
        out_ref[2, ...] = rhs_bz

    out_shape = jax.ShapeDtypeStruct(
        (3,) + tuple(omega_xdep.shape[1:]),
        omega_xdep.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[x_dependent_spec, x_free_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_curl_split_x",
        **_compiler_kwargs(config),
    )(omega_xdep, omega_xfree, dtdx, dtdy, dtdz)


def _ct_curl_pallas_local(omega_slices, dtdx, dtdy, dtdz, config: SimulationConfig):
    """
    Single-shard build of stage 3: the edge-average smoothing
    (``point_values_to_averages``) and the sixth-order curl
    (``finite_difference_int6``) in one tile, with the stacked interface-field
    increments (rhs_bx, rhs_by, rhs_bz) as output.

    The input channels are 0 = Omega_z, 1 = Omega_x, 2 = Omega_y. Each output
    fuses two short stencils, the smoothing (reach 1 along its two axes) and
    the difference (reach 3 along its axis), so the combined reach along any
    axis is at most 4.

    Args:
        omega_slices: The stacked EMFs (possibly halo-padded).
        dtdx: The time step over the grid spacing in x (scalar array).
        dtdy: The time step over the grid spacing in y (scalar array).
        dtdz: The time step over the grid spacing in z (scalar array).
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` interface-field increments.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(omega_slices.shape[1:], config)

    field_spec = pl.BlockSpec(omega_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    out_spec = pl.BlockSpec((3,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(omega_ref, dt_over_dx_ref, dt_over_dy_ref, dt_over_dz_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)
        rhs_bx, rhs_by, rhs_bz = _smoothed_curl(
            _channel_reader(omega_ref, 0, ii, jj, kk, spatial_shape),
            _channel_reader(omega_ref, 1, ii, jj, kk, spatial_shape),
            _channel_reader(omega_ref, 2, ii, jj, kk, spatial_shape),
            dt_over_dx_ref[()],
            dt_over_dy_ref[()],
            dt_over_dz_ref[()],
        )
        out_ref[0, ...] = rhs_bx
        out_ref[1, ...] = rhs_by
        out_ref[2, ...] = rhs_bz

    out_shape = jax.ShapeDtypeStruct(
        (3,) + tuple(omega_slices.shape[1:]),
        omega_slices.dtype,
    )
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_curl",
        **_compiler_kwargs(config),
    )(omega_slices, dtdx, dtdy, dtdz)


# -------------------------------------------------------------
# ============= ↑ Stage 3: smoothed sixth-order curl ↑ ========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============== ↓ Three-stage CT right-hand side ↓ ===========
# -------------------------------------------------------------


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
    """
    Three-stage Pallas CT right-hand side: ``_ct_modified_flux_pallas`` ->
    ``_ct_edge_emf_pallas`` -> ``_ct_curl_pallas``.

    Each stage is one bounded-halo Pallas kernel and costs one halo exchange
    under a multi-device mesh. The flux slices travel stacked between the
    stages, so the peak temporary footprint stays well below that of the
    native code's many intermediates.

    Args:
        conserved_state: The conserved state.
        By_flux_x_interface: The raw x-face flux of B_y.
        Bz_flux_x_interface: The raw x-face flux of B_z.
        Bx_flux_y_interface: The raw y-face flux of B_x.
        Bz_flux_y_interface: The raw y-face flux of B_z.
        Bx_flux_z_interface: The raw z-face flux of B_x.
        By_flux_z_interface: The raw z-face flux of B_y.
        dtdx: The time step over the grid spacing in x.
        dtdy: The time step over the grid spacing in y.
        dtdz: The time step over the grid spacing in z.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The three interface-field increments ``(rhs_bx, rhs_by, rhs_bz)``.
    """
    assert _ct_rhs_pallas_supported(conserved_state, config)
    flux_slices = jnp.stack(
        [
            By_flux_x_interface,
            Bz_flux_x_interface,
            Bx_flux_y_interface,
            Bz_flux_y_interface,
            Bx_flux_z_interface,
            By_flux_z_interface,
        ]
    )
    flux_mod = _ct_modified_flux_pallas(
        conserved_state,
        flux_slices,
        config,
        registered_variables,
    )
    del flux_slices
    omega = _ct_edge_emf_pallas(flux_mod, config)
    del flux_mod
    rhs_b = _ct_curl_pallas(omega, dtdx, dtdy, dtdz, config)
    return rhs_b[0], rhs_b[1], rhs_b[2]


def _ct_rhs_pallas_x_precomputed(
    conserved_state,
    By_flux_x_interface_mod,
    Bz_flux_x_interface_mod,
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
    """
    Constrained-transport right-hand side when the x-face modified fluxes
    were computed together with the x WENO flux (see
    ``_update_cell_center_and_weno_flux_mhd_pallas_keep_halo_x_with_ct_mod``):
    stages 1 (y and z only), 2 and 3 of ``_ct_rhs_pallas``.

    Args:
        conserved_state: The conserved state.
        By_flux_x_interface_mod: The modified x-face flux of B_y.
        Bz_flux_x_interface_mod: The modified x-face flux of B_z.
        Bx_flux_y_interface: The raw y-face flux of B_x.
        Bz_flux_y_interface: The raw y-face flux of B_z.
        Bx_flux_z_interface: The raw z-face flux of B_x.
        By_flux_z_interface: The raw z-face flux of B_y.
        dtdx: The time step over the grid spacing in x.
        dtdy: The time step over the grid spacing in y.
        dtdz: The time step over the grid spacing in z.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The three interface-field increments ``(rhs_bx, rhs_by, rhs_bz)``.
    """
    assert _ct_rhs_pallas_supported(conserved_state, config)
    flux_yz_slices = jnp.stack(
        [
            Bx_flux_y_interface,
            Bz_flux_y_interface,
            Bx_flux_z_interface,
            By_flux_z_interface,
        ]
    )
    flux_yz_mod = _ct_modified_flux_yz_pallas(
        conserved_state,
        flux_yz_slices,
        config,
        registered_variables,
    )
    del flux_yz_slices
    flux_mod = jnp.concatenate(
        [
            jnp.stack([By_flux_x_interface_mod, Bz_flux_x_interface_mod]),
            flux_yz_mod,
        ],
        axis=0,
    )
    del flux_yz_mod
    omega = _ct_edge_emf_pallas(flux_mod, config)
    del flux_mod
    rhs_b = _ct_curl_pallas(omega, dtdx, dtdy, dtdz, config)
    return rhs_b[0], rhs_b[1], rhs_b[2]


# -------------------------------------------------------------
# ============== ↑ Three-stage CT right-hand side ↑ ===========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============ ↓ Upwind dissipation of the edge EMFs ↓ ========
# -------------------------------------------------------------
#
# The Pallas form of ``_constrained_transport.ct_upwind_dissipation_rhs``, the
# same arithmetic in two bounded-halo kernels: the upwind edge EMFs from the
# state and the stacked interface fields (halo 3: the six-point jump stencil),
# then their sixth-order curl (halo 3). Unlike the staged CT pipeline it runs
# whenever the backend is Pallas: it replaces some twenty full-grid
# temporaries of the native form by two outputs.

def _ct_upwind_pallas_supported(state, config: SimulationConfig) -> bool:
    """
    Whether the Pallas upwind EMF dissipation can run (3D MHD on the Pallas
    backend, block shape dividing the grid).

    Args:
        state: The conserved state.
        config: The simulation configuration.

    Returns:
        True if the Pallas kernels can be used.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if not config.mhd or int(config.dimensionality) != 3:
        return False
    if state.ndim != 4:
        return False
    return _ct_pallas_block_ok(state.shape, config)


def _upwind_jump(value_at):
    """
    The fifth-order upwind-biased jump at i + 1/2 (from the right minus from
    the left) of ``_constrained_transport._upwind_edge_jump``; ``value_at(k)``
    is the field at i + k.
    """
    from_left = (
        2.0 * value_at(-2)
        - 13.0 * value_at(-1)
        + 47.0 * value_at(0)
        + 27.0 * value_at(1)
        - 3.0 * value_at(2)
    ) / 60.0
    from_right = (
        -3.0 * value_at(-1)
        + 27.0 * value_at(0)
        + 47.0 * value_at(1)
        - 13.0 * value_at(2)
        + 2.0 * value_at(3)
    ) / 60.0
    return from_right - from_left


def _ct_upwind_dissipation_pallas(
    conserved_state,
    interface_fields,
    dt_over_dx,
    gamma,
    params,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The interface-field increments of the upwind EMF dissipation (Pallas).

    Args:
        conserved_state: The conserved state.
        interface_fields: The stacked ``(3, nx, ny, nz)`` interface fields
            (B_x, B_y, B_z).
        dt_over_dx: The (stage) time step over the cell size.
        gamma: The adiabatic index (ideal gas).
        params: The simulation parameters (minimum density and pressure,
            isothermal sound speed).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked ``(3, nx, ny, nz)`` increments (rhs_bx, rhs_by, rhs_bz).
    """
    ndim = int(config.dimensionality)
    dtype = interface_fields.dtype
    _, block_shape, _ = _ct_block_and_grid(interface_fields.shape[1:], config)
    if config.equation_of_state == IDEAL_GAS:
        sound_speed_squared = jnp.zeros((), dtype=dtype)
    else:
        sound_speed_squared = jnp.asarray(params.isothermal_sound_speed ** 2, dtype=dtype)

    def build_emf(state_local, interface_local, *scalars):
        return _ct_upwind_emf_pallas_local(
            state_local,
            interface_local,
            *scalars,
            config=config,
            registered_variables=registered_variables,
        )

    emfs = _pallas_call_sharded(
        build_emf,
        state_inputs=(conserved_state, interface_fields),
        other_args=(
            jnp.asarray(params.minimum_density, dtype=dtype),
            jnp.asarray(params.minimum_pressure, dtype=dtype),
            sound_speed_squared,
            jnp.asarray(gamma, dtype=dtype),
        ),
        halo=(3, 3, 3)[:ndim],
        block_shape=block_shape[:ndim],
    )

    def build_curl(emf_local, time_step_over_spacing):
        return _ct_plain_curl_pallas_local(emf_local, time_step_over_spacing, config)

    return _pallas_call_sharded(
        build_curl,
        state_inputs=(emfs,),
        other_args=(jnp.asarray(dt_over_dx, dtype=dtype),),
        halo=(3, 3, 3)[:ndim],
        block_shape=block_shape[:ndim],
    )


def _ct_upwind_emf_pallas_local(
    state,
    interface_fields,
    minimum_density,
    minimum_pressure,
    sound_speed_squared,
    gamma,
    *,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Single-shard build of the upwind edge EMFs, stacked as the output channels
    0 = Omega_z, 1 = Omega_x, 2 = Omega_y: per edge, minus half the strength
    times the largest signal speed across the jump of each of the two
    interface-field components meeting there (signs as in the native form).

    Args:
        state: The conserved state (possibly halo-padded).
        interface_fields: The stacked interface fields (possibly halo-padded).
        minimum_density: The density floor of the speed estimate (scalar array).
        minimum_pressure: The pressure floor of the speed estimate (scalar array).
        sound_speed_squared: The isothermal sound speed squared (scalar array;
            unused for an ideal gas).
        gamma: The adiabatic index (scalar array; ideal gas only).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The stacked ``(3, nx, ny, nz)`` upwind EMFs.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(interface_fields.shape[1:], config)
    nx, ny, nz = spatial_shape
    extents = (nx, ny, nz)
    ideal_gas = config.equation_of_state == IDEAL_GAS
    half_strength = 0.5 * config.solver_config.ct_upwind_strength
    density_index = registered_variables.density_index
    momentum_indices = (
        registered_variables.momentum_index.x,
        registered_variables.momentum_index.y,
        registered_variables.momentum_index.z,
    )
    magnetic_indices = (
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )
    energy_index = registered_variables.energy_index if ideal_gas else None

    state_spec = pl.BlockSpec(state.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    field_spec = pl.BlockSpec(interface_fields.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    out_spec = pl.BlockSpec((3,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(state_ref, field_ref, rhomin_ref, pmin_ref, cs2_ref, gamma_ref, out_ref):
        cell = _block_cell_indices(block_shape, spatial_shape)
        rhomin = rhomin_ref[()]

        def signal_speeds(offset):
            """|v_d| + c_fast,d along the three axes at a cell offset."""
            index = tuple(
                _wrapped_index(cell[axis], offset[axis], extents[axis]) for axis in range(3)
            )
            density = jnp.maximum(state_ref[(density_index,) + index], rhomin)
            velocity = [state_ref[(m,) + index] / density for m in momentum_indices]
            field = [state_ref[(b,) + index] for b in magnetic_indices]
            field_squared = field[0] * field[0] + field[1] * field[1] + field[2] * field[2]
            if ideal_gas:
                kinetic = 0.5 * density * (
                    velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]
                )
                pressure = (gamma_ref[()] - 1.0) * (
                    state_ref[(energy_index,) + index] - kinetic - 0.5 * field_squared
                )
                cs2 = gamma_ref[()] * jnp.maximum(pressure, pmin_ref[()]) / density
            else:
                cs2 = cs2_ref[()]
            total = cs2 + field_squared / density
            speeds = []
            for axis in range(3):
                normal = field[axis] * field[axis] / density
                discriminant = jnp.maximum(total * total - 4.0 * cs2 * normal, 0.0)
                speeds.append(jnp.abs(velocity[axis]) + jnp.sqrt(0.5 * (total + jnp.sqrt(discriminant))))
            return speeds

        # The cells around the three kinds of edge: the origin and its
        # neighbours one cell up along one or two of the edge-plane axes.
        speed_cache = {}

        def speeds_at(offset):
            if offset not in speed_cache:
                speed_cache[offset] = signal_speeds(offset)
            return speed_cache[offset]

        def edge_maximum(speed_axis, first_axis, second_axis):
            values = []
            for step_first in (0, 1):
                for step_second in (0, 1):
                    offset = [0, 0, 0]
                    offset[first_axis] += step_first
                    offset[second_axis] += step_second
                    values.append(speeds_at(tuple(offset))[speed_axis])
            return jnp.maximum(jnp.maximum(values[0], values[1]), jnp.maximum(values[2], values[3]))

        def jump(channel, axis):
            def value_at(shift):
                index = list(cell)
                index[axis] = _wrapped_index(cell[axis], shift, extents[axis])
                return field_ref[(channel,) + tuple(index)]

            return _upwind_jump(value_at)

        # Omega_z (x, y edges): B_x jumps along y, B_y along x; Omega_x
        # (y, z): B_y along z, B_z along y; Omega_y (z, x): B_z along x, B_x
        # along z.
        out_ref[0, ...] = half_strength * (
            -edge_maximum(1, 0, 1) * jump(0, 1) + edge_maximum(0, 0, 1) * jump(1, 0)
        )
        out_ref[1, ...] = half_strength * (
            -edge_maximum(2, 1, 2) * jump(1, 2) + edge_maximum(1, 1, 2) * jump(2, 1)
        )
        out_ref[2, ...] = half_strength * (
            -edge_maximum(0, 2, 0) * jump(2, 0) + edge_maximum(2, 2, 0) * jump(0, 2)
        )

    out_shape = jax.ShapeDtypeStruct((3,) + tuple(spatial_shape), interface_fields.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[state_spec, field_spec, scalar_spec, scalar_spec, scalar_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_upwind_emf",
        **_compiler_kwargs(config),
    )(state, interface_fields, minimum_density, minimum_pressure, sound_speed_squared, gamma)


def _ct_plain_curl_pallas_local(emf_slices, dt_over_dx, config: SimulationConfig):
    """
    Single-shard build of the sixth-order curl of the stacked upwind EMFs
    (``finite_difference_int6``, no edge smoothing), with the stacked
    interface-field increments as output:

      rhs_bx = dt/dx (-D_y Omega_z + D_z Omega_y)
      rhs_by = dt/dx (-D_z Omega_x + D_x Omega_z)
      rhs_bz = dt/dx (-D_x Omega_y + D_y Omega_x)

    Args:
        emf_slices: The stacked EMFs, channels 0 = Omega_z, 1 = Omega_x,
            2 = Omega_y (possibly halo-padded).
        dt_over_dx: The time step over the cell size (scalar array).
        config: The simulation configuration.

    Returns:
        The stacked ``(3, nx, ny, nz)`` increments.
    """
    spatial_shape, block_shape, grid = _ct_block_and_grid(emf_slices.shape[1:], config)

    field_spec = pl.BlockSpec(emf_slices.shape, lambda bi, bj, bk: (0, 0, 0, 0))
    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())
    out_spec = pl.BlockSpec((3,) + block_shape, lambda bi, bj, bk: (0, bi, bj, bk))

    def kernel(emf_ref, dt_over_dx_ref, out_ref):
        ii, jj, kk = _block_cell_indices(block_shape, spatial_shape)

        def difference(channel, axis):
            read = _channel_reader(emf_ref, channel, ii, jj, kk, spatial_shape)

            def value_at(shift):
                offset = [0, 0, 0]
                offset[axis] = shift
                return read(tuple(offset))

            return _sixth_order_difference(value_at)

        scale = dt_over_dx_ref[()]
        out_ref[0, ...] = scale * (-difference(0, 1) + difference(2, 2))
        out_ref[1, ...] = scale * (-difference(1, 2) + difference(0, 0))
        out_ref[2, ...] = scale * (-difference(2, 0) + difference(1, 1))

    out_shape = jax.ShapeDtypeStruct((3,) + _shard_cells(spatial_shape)[1], emf_slices.dtype)
    return pl.pallas_call(
        kernel,
        out_shape=out_shape,
        grid=grid,
        in_specs=[field_spec, scalar_spec],
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name="ct_upwind_curl",
        **_compiler_kwargs(config),
    )(emf_slices, dt_over_dx)


# -------------------------------------------------------------
# ============ ↑ Upwind dissipation of the edge EMFs ↑ ========
# -------------------------------------------------------------
