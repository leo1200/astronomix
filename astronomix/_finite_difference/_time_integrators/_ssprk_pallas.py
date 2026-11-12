"""
Pallas backend for the per-axis flux-divergence step of the finite-difference
time integrators.

The integrators in ``_ssprk.py`` (SSPRK4 and LSRK4, hydrodynamics and MHD with
constrained transport) call ``_hydro_flux_div_axis_pallas`` under the
``_backend_is_pallas`` predicate. The kernel is MHD-agnostic, since it walks
every variable channel, so one kernel serves all integrators and lets them keep
a single physical right-hand-side buffer (via ``input_output_aliases``).

With ``magnetic_state`` the energy row's flux loses the magnetic energy flux
of the fluid's own field rows, sum_c B_f[c] F_{B_c} with the two-point face
field of the stage state's cell-centred field: the fluid half of the
Poynting-consistent energy flux of the MHD integrator
(``_magnetic_update/_energy_flux.py``; the CT half is added after the CT).

The module also holds the native-JAX twin of the kernel, used as its tangent
for automatic differentiation, and the divergence of an x flux that already
carries its x halo (the multi-GPU fast path of ``_lsrk4_with_ct``).
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
from astronomix._finite_difference._magnetic_update._energy_flux import magnetic_energy_corrected_flux
from astronomix._stencil_operations._stencil_operations import _shift


def _hydro_flux_div_axis_native(
    interface_flux,
    dt_over_dx,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
    magnetic_state=None,
    energy_flux_channels=None,
):
    """
    Native-JAX equivalent of :func:`_hydro_flux_div_axis_pallas`.

    Used as the tangent branch of ``diffable_pallas_call`` so that AD through
    the Pallas kernel goes through a transposable JAX expression. It must
    reproduce the Pallas kernel's primal output for the gradient to equal the
    gradient of the Pallas operation at the input.

    Args:
        interface_flux: The flux through the right face of each cell
            (variable axis leading).
        dt_over_dx: The time step over the grid spacing along ``axis``.
        axis: The spatial axis of the divergence (0 = x).
        rhs_accumulator: Optional accumulator the divergence is added to.
        scale_in: The factor applied to the accumulator.
        magnetic_state: Optional state whose cell-centred field corrects the
            energy flux (``magnetic_energy_corrected_flux``).
        energy_flux_channels: ``(energy channel, (B_x, B_y, B_z) channels)``,
            with ``magnetic_state``.

    Returns:
        ``scale_in * rhs_accumulator - dt/dx * (F_{i+1/2} - F_{i-1/2})``, or the
        divergence term alone without an accumulator.
    """
    if magnetic_state is not None:
        interface_flux = magnetic_energy_corrected_flux(interface_flux, magnetic_state, energy_flux_channels, axis)
    divergence = -dt_over_dx * (interface_flux - _shift(interface_flux, 1, axis=axis + 1))
    if rhs_accumulator is None:
        return divergence
    return scale_in * rhs_accumulator + divergence


def _div_axis_pallas_shape_ok(state, config: SimulationConfig) -> bool:
    """
    Whether the per-axis divergence Pallas kernel can tile ``state``.

    Used by callers (e.g. the MHD constrained-transport integrators) that want
    the divergence kernel but cannot rely on the hydro WENO support predicate,
    which excludes MHD. The kernel itself is MHD-agnostic, so only the spatial
    block divisibility required by ``pl.pallas_call`` is checked.

    Args:
        state: A state-shaped array (variable axis leading).
        config: The simulation configuration.

    Returns:
        True if the kernel can run on ``state``.
    """
    if pl is None:
        return False
    ndim = int(config.dimensionality)
    if ndim not in (1, 2, 3):
        return False
    if state.ndim != ndim + 1:
        return False
    bx, by, bz = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=state.shape[1:],
    )
    for extent, block_size in zip(state.shape[1:], (bx, by, bz)[:ndim], strict=True):
        if int(extent) % int(block_size) != 0:
            return False
    return True


def _hydro_flux_div_axis_pallas(
    interface_flux,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
    magnetic_state=None,
    energy_flux_channels=None,
):
    """
    Per-axis Pallas divergence kernel with optional in-place accumulation.

    Computes ``rhs_out = scale_in * rhs_accumulator - dt/dx * (F_{i+1/2} -
    F_{i-1/2})`` along ``axis`` (without the accumulator term when none is
    given). Calling it sequentially for each axis with ``rhs_accumulator=rhs``
    lets XLA keep a single physical right-hand-side buffer (via
    ``input_output_aliases``) across all three axes, eliminating both the
    chained ``rhs + ...`` additions and the transient buffers they would need.

    ``scale_in`` is folded into the kernel so that the first LSRK4 stage update
    ``dq = A[i] * dq - dt/dx * div_0(F_0)`` can be done in place on the ``dq``
    buffer without materialising a separate right-hand-side register.

    Each flux is thereby consumed directly after it is produced, so the three
    axis fluxes never coexist, while the WENO kernel keeps computing one flux
    per cell.

    The call is wrapped for AD (native tangent) and for multi-GPU runs
    (``_pallas_call_sharded``).

    Args:
        interface_flux: The flux through the right face of each cell.
        dt_over_dx: The time step over the grid spacing along ``axis``.
        config: The simulation configuration.
        axis: The spatial axis of the divergence (0 = x).
        rhs_accumulator: Optional accumulator, updated in place.
        scale_in: The factor applied to the accumulator (may be traced).
        magnetic_state: Optional state whose cell-centred field removes the
            fluid's magnetic energy flux from the energy row
            (``magnetic_energy_corrected_flux``).
        energy_flux_channels: ``(energy channel, (B_x, B_y, B_z) channels)``,
            with ``magnetic_state``.

    Returns:
        The updated accumulator, or the divergence term without one.
    """
    # Multi-GPU: the divergence reads F[i] - F[i - 1] along ``axis`` (and the
    # field at i - 1 ... i + 1), so the only halo needed is one cell on that
    # axis (rounded up to the Pallas block size by ``_pallas_call_sharded``).
    ndim = int(config.dimensionality)
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=interface_flux.shape[1:],
    )
    halo_list = [0, 0, 0]
    if 0 <= axis < ndim:
        halo_list[axis] = 1
    halo = tuple(halo_list[:ndim])

    if magnetic_state is not None:
        return _hydro_flux_div_axis_with_energy_flux_pallas(
            interface_flux,
            dt_over_dx,
            config,
            axis=axis,
            rhs_accumulator=rhs_accumulator,
            scale_in=scale_in,
            magnetic_state=magnetic_state,
            energy_flux_channels=energy_flux_channels,
            halo=halo,
            block_shape=block_shape,
        )

    if rhs_accumulator is None:

        def pallas_branch(flux_in, dt_over_dx_in):
            return _pallas_call_sharded(
                lambda flux_local: _hydro_flux_div_axis_pallas_local(
                    flux_local,
                    dt_over_dx_in,
                    config,
                    axis=axis,
                    rhs_accumulator=None,
                    scale_in=scale_in,
                ),
                state_inputs=(flux_in,),
                halo=halo,
                block_shape=block_shape[:ndim],
            )

        def native_branch(flux_in, dt_over_dx_in):
            return _hydro_flux_div_axis_native(
                flux_in,
                dt_over_dx_in,
                axis=axis,
                rhs_accumulator=None,
                scale_in=scale_in,
            )

        return diffable_pallas_call(
            interface_flux,
            dt_over_dx,
            pallas_branch=pallas_branch,
            native_branch=native_branch,
        )

    zero_halo = (0,) * ndim

    def pallas_branch_accumulate(flux_in, dt_over_dx_in, rhs_in, scale_in_array):
        return _pallas_call_sharded(
            lambda rhs_local, flux_local: _hydro_flux_div_axis_pallas_local(
                flux_local,
                dt_over_dx_in,
                config,
                axis=axis,
                rhs_accumulator=rhs_local,
                scale_in=scale_in_array,
            ),
            state_inputs=(rhs_in, flux_in),
            halo=halo,
            # The accumulator is only read and written at the cell itself, so
            # only the flux is exchanged.
            input_halos=(zero_halo, halo),
            block_shape=block_shape[:ndim],
        )

    def native_branch_accumulate(flux_in, dt_over_dx_in, rhs_in, scale_in_array):
        return _hydro_flux_div_axis_native(
            flux_in,
            dt_over_dx_in,
            axis=axis,
            rhs_accumulator=rhs_in,
            scale_in=scale_in_array,
        )

    # ``scale_in`` may be a traced scalar (the LSRK4 stage coefficient), so it
    # is routed through the differentiable primals as well.
    scale_in_array = jnp.asarray(scale_in)
    return diffable_pallas_call_n(
        (interface_flux, dt_over_dx, rhs_accumulator, scale_in_array),
        pallas_branch=pallas_branch_accumulate,
        native_branch=native_branch_accumulate,
    )


def _hydro_flux_div_axis_with_energy_flux_pallas(
    interface_flux,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator,
    scale_in,
    magnetic_state,
    energy_flux_channels,
    halo,
    block_shape,
):
    """
    :func:`_hydro_flux_div_axis_pallas` with the fluid's magnetic energy flux
    removed from the energy row (``magnetic_state`` given); the same AD and
    multi-GPU wrapping, with the state as a further halo-exchanged input.
    """
    ndim = int(config.dimensionality)
    accumulate = rhs_accumulator is not None
    zero_halo = (0,) * ndim

    def pallas_branch(flux_in, dt_over_dx_in, state_in, *accumulation):
        if accumulate:
            rhs_in, scale_in_array = accumulation
            return _pallas_call_sharded(
                lambda rhs_local, flux_local, state_local: _hydro_flux_div_axis_pallas_local(
                    flux_local,
                    dt_over_dx_in,
                    config,
                    axis=axis,
                    rhs_accumulator=rhs_local,
                    scale_in=scale_in_array,
                    magnetic_state=state_local,
                    energy_flux_channels=energy_flux_channels,
                ),
                state_inputs=(rhs_in, flux_in, state_in),
                halo=halo,
                # The accumulator is only read and written at the cell itself.
                input_halos=(zero_halo, halo, halo),
                block_shape=block_shape[:ndim],
            )
        return _pallas_call_sharded(
            lambda flux_local, state_local: _hydro_flux_div_axis_pallas_local(
                flux_local,
                dt_over_dx_in,
                config,
                axis=axis,
                rhs_accumulator=None,
                scale_in=scale_in,
                magnetic_state=state_local,
                energy_flux_channels=energy_flux_channels,
            ),
            state_inputs=(flux_in, state_in),
            halo=halo,
            block_shape=block_shape[:ndim],
        )

    def native_branch(flux_in, dt_over_dx_in, state_in, *accumulation):
        rhs_in, scale_in_array = accumulation if accumulate else (None, scale_in)
        return _hydro_flux_div_axis_native(
            flux_in,
            dt_over_dx_in,
            axis=axis,
            rhs_accumulator=rhs_in,
            scale_in=scale_in_array,
            magnetic_state=state_in,
            energy_flux_channels=energy_flux_channels,
        )

    primals = (interface_flux, dt_over_dx, magnetic_state)
    if accumulate:
        primals = primals + (rhs_accumulator, jnp.asarray(scale_in))
    return diffable_pallas_call_n(primals, pallas_branch=pallas_branch, native_branch=native_branch)


def _hydro_flux_div_axis_native_from_kept_halo_sharded(
    interface_flux,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator,
    scale_in: Union[float, jnp.ndarray] = 1.0,
    kept_halo: int = 1,
):
    """
    Accumulate ``scale_in * rhs - dt/dx * (F_{i+1/2} - F_{i-1/2})`` along x
    from a flux that already carries ``kept_halo`` x-halo cells.

    The x-WENO kernel of the multi-GPU fast path keeps one x-halo cell of its
    flux, so the left face of each shard's first cell is already local and the
    divergence needs no further halo exchange; it runs as plain JAX inside a
    shard_map, which is cheaper than launching a separate Pallas kernel.

    Args:
        interface_flux: The interface flux with ``kept_halo`` extra x cells per
            side.
        dt_over_dx: The time step over the grid spacing.
        config: The simulation configuration.
        axis: The flux axis; only 0 (x) is supported.
        rhs_accumulator: The accumulator updated in place.
        scale_in: The factor applied to the accumulator (the LSRK coefficient).
        kept_halo: The number of kept x-halo cells of ``interface_flux``.

    Returns:
        The updated accumulator.
    """
    if axis != 0:
        raise RuntimeError("The kept-halo divergence is only implemented along x.")

    ndim = int(config.dimensionality)
    zero_halo = (0,) * ndim
    block_shape = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=rhs_accumulator.shape[1:],
    )

    return _pallas_call_sharded(
        lambda rhs_local, flux_local: _hydro_flux_div_axis_from_kept_halo_native(
            flux_local,
            dt_over_dx,
            axis=axis,
            rhs_accumulator=rhs_local,
            scale_in=scale_in,
            kept_halo=kept_halo,
        ),
        state_inputs=(rhs_accumulator, interface_flux),
        halo=zero_halo,
        # Nothing is exchanged: the flux already carries its x halo and the
        # accumulator is only written locally.
        input_halos=(zero_halo, zero_halo),
        block_shape=block_shape[:ndim],
    )


def _hydro_flux_div_axis_from_kept_halo_native(
    interface_flux,
    dt_over_dx,
    *,
    axis: int,
    rhs_accumulator,
    scale_in,
    kept_halo: int,
):
    """
    Local x divergence of a flux with ``kept_halo`` x-halo cells, by slicing.

    ``interface_flux[kept_halo + i]`` is the flux through the right face of
    cell ``i`` and ``interface_flux[kept_halo - 1 + i]`` the one through its
    left face.
    """
    array_axis = axis + 1
    num_cells = rhs_accumulator.shape[array_axis]
    right_face_flux = jax.lax.slice_in_dim(
        interface_flux,
        kept_halo,
        kept_halo + num_cells,
        axis=array_axis,
    )
    left_face_flux = jax.lax.slice_in_dim(
        interface_flux,
        kept_halo - 1,
        kept_halo - 1 + num_cells,
        axis=array_axis,
    )
    return scale_in * rhs_accumulator + (-dt_over_dx) * (right_face_flux - left_face_flux)


def _hydro_flux_div_axis_pallas_local(
    interface_flux,
    dt_over_dx,
    config: SimulationConfig,
    *,
    axis: int,
    rhs_accumulator=None,
    scale_in: Union[float, jnp.ndarray] = 1.0,
    magnetic_state=None,
    energy_flux_channels=None,
):
    """
    Single-shard ``pl.pallas_call`` build of the divergence kernel.

    Called either directly or inside a ``shard_map`` body; in the multi-device
    case ``interface_flux.shape`` is the local (halo-padded) shape, so the
    kernel's grid and block specs are derived from it automatically.

    Args:
        interface_flux: The flux through the right face of each cell.
        dt_over_dx: The time step over the grid spacing along ``axis``.
        config: The simulation configuration.
        axis: The spatial axis of the divergence (0 = x).
        rhs_accumulator: Optional accumulator, aliased to the output.
        scale_in: The factor applied to the accumulator.
        magnetic_state: Optional state whose cell-centred field removes the
            fluid's magnetic energy flux from the energy row.
        energy_flux_channels: ``(energy channel, (B_x, B_y, B_z) channels)``,
            with ``magnetic_state``.

    Returns:
        The updated accumulator, or the divergence term without one.
    """
    ndim = int(config.dimensionality)
    num_vars = int(interface_flux.shape[0])
    spatial_shape = tuple(int(extent) for extent in interface_flux.shape[1:])
    nx = spatial_shape[0]
    ny = spatial_shape[1] if ndim >= 2 else 1
    nz = spatial_shape[2] if ndim == 3 else 1
    bx, by, bz = _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=spatial_shape,
    )
    grid = (nx // bx, ny // by, nz // bz)

    accumulate = rhs_accumulator is not None
    corrected = magnetic_state is not None

    # The output (and the aliased accumulator) is tiled over the spatial axes
    # with the variable axis kept whole; the flux is passed whole so that each
    # program can read its left neighbour across the block boundary.
    if ndim == 1:
        block_shape = (num_vars, bx)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi))
        flux_spec = pl.BlockSpec(interface_flux.shape, lambda bi, bj, bk: (0, 0))
    elif ndim == 2:
        block_shape = (num_vars, bx, by)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj))
        flux_spec = pl.BlockSpec(interface_flux.shape, lambda bi, bj, bk: (0, 0, 0))
    else:
        block_shape = (num_vars, bx, by, bz)
        out_spec = pl.BlockSpec(block_shape, lambda bi, bj, bk: (0, bi, bj, bk))
        flux_spec = pl.BlockSpec(interface_flux.shape, lambda bi, bj, bk: (0, 0, 0, 0))

    scalar_spec = pl.BlockSpec((), lambda bi, bj, bk: ())

    def kernel(*refs):
        refs = list(refs)
        rhs_in_ref = refs.pop(0) if accumulate else None
        flux_ref = refs.pop(0)
        state_ref = refs.pop(0) if corrected else None
        if accumulate:
            dt_over_dx_ref, scale_in_ref, rhs_out_ref = refs
        else:
            dt_over_dx_ref, rhs_out_ref = refs

        # Block program ids and the (periodically wrapped) cell indices of the
        # block along each axis.
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

        dt_over_dx_value = dt_over_dx_ref[()]

        def read(ref, var, offset):
            """``ref[var]`` at the cell ``offset`` along ``axis``."""
            if axis == 0:
                index = ii if offset == 0 else (ii + offset) % nx
                if ndim == 1:
                    return ref[var, index]
                if ndim == 2:
                    return ref[var, index, jj]
                return ref[var, index, jj, kk]
            if axis == 1:
                index = jj if offset == 0 else (jj + offset) % ny
                if ndim == 2:
                    return ref[var, ii, index]
                return ref[var, ii, index, kk]
            index = kk if offset == 0 else (kk + offset) % nz
            return ref[var, ii, jj, index]

        def corrected_energy_flux(offset):
            """F_E - sum_c B_f[c] F_{B_c} at the face ``offset`` (0: i + 1/2)."""
            energy_channel, magnetic_channels = energy_flux_channels
            magnetic_flux = 0.0
            for channel in magnetic_channels:
                face_field = 0.5 * (read(state_ref, channel, offset) + read(state_ref, channel, offset + 1))
                magnetic_flux = magnetic_flux + face_field * read(flux_ref, channel, offset)
            return read(flux_ref, energy_channel, offset) - magnetic_flux

        def flux_difference(var):
            """F_{i+1/2} - F_{i-1/2} of one variable along ``axis``."""
            if corrected and var == energy_flux_channels[0]:
                return corrected_energy_flux(0) - corrected_energy_flux(-1)
            return read(flux_ref, var, 0) - read(flux_ref, var, -1)

        if accumulate:
            scale = scale_in_ref[()]
            for var in range(num_vars):
                rhs_out_ref[var, ...] = (
                    scale * rhs_in_ref[var, ...] + (-dt_over_dx_value) * flux_difference(var)
                )
        else:
            for var in range(num_vars):
                rhs_out_ref[var, ...] = -dt_over_dx_value * flux_difference(var)

    kwargs = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        kwargs["compiler_params"] = compiler_params

    # The state is passed whole, like the flux, for the field's neighbours.
    state_specs = [flux_spec] if corrected else []
    state_args = (magnetic_state,) if corrected else ()
    if accumulate:
        in_specs = [out_spec, flux_spec] + state_specs + [scalar_spec, scalar_spec]
        kernel_args = (
            rhs_accumulator,
            interface_flux,
            *state_args,
            jnp.asarray(dt_over_dx, dtype=interface_flux.dtype),
            jnp.asarray(scale_in, dtype=interface_flux.dtype),
        )
        kwargs["input_output_aliases"] = {0: 0}
    else:
        in_specs = [flux_spec] + state_specs + [scalar_spec]
        kernel_args = (
            interface_flux,
            *state_args,
            jnp.asarray(dt_over_dx, dtype=interface_flux.dtype),
        )

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(interface_flux.shape, interface_flux.dtype),
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        interpret=config.backend_config.pallas_interpret,
        name=f"hydro_flux_div_axis_{axis}{'_acc' if accumulate else ''}{'_energy' if corrected else ''}",
        **kwargs,
    )(*kernel_args)
