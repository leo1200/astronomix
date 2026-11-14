"""
The stage of the finite-difference scheme with a Riemann flux (``scheme =
RIEMANN_FLUX``) on a device mesh, with one halo exchange per stage and split
axis.

Called through ``_pallas_call_sharded``, every kernel of a stage exchanges
the halos of its own inputs, and the array-form steps between the kernels
(the transverse compression) add collective permutes of their own: about
nine exchanges per split axis and stage in the MHD form. Here the stage state,
the conserved state and (MHD) the interface fields, is exchanged once per
split axis with the halo of the whole stage (``shard_map_on_padded_shards``),
and every step of the stage runs on the padded shard and writes the cells its
consumers read: the shard widened along every split axis by

    rebuilt state             6  the cell-centred field from the faces
                                 (``update_cell_center_fields``); the face
                                 velocities read it 6 cells out, the fluxes
                                 4 (field 4 + 3 = 7 faces), the cold speed 3
    transverse compression    6  the array form on the rebuilt state (one
                                 roll: valid 5 cells out, read 3)
    flux F_H along axis a     2  along a only: the guard reads the faces
                                 i - 1/2 ... i + 3/2
    guarded flux along a      1  along a only: the divergence reads the
                                 faces i - 1/2 and i + 1/2
    face velocities           6  the EMFs read them 6 cells out along the
                                 transport axis
    edge EMFs                 3  the curl and the energy flux read them 3
                                 cells out
    increments                0

so the halo of the MHD form is ``MHD_HALO`` = 7 cells (4 for the fluid chain,
3 for the face-to-centre interpolation), of the hydrodynamic form
``HYDRO_HALO`` = 4. A kernel writes its whole region even where its own
stencil reaches past the padding (its modular indexing then reads the other
side of the padded array); those cells are never read.

The kernels are the ones of a single device, with the same arithmetic per cell,
and the maximum signal speeds of the guard are the maxima over all shards (one
all-reduce per stage for the three axes), so a stage gives on a mesh what it
gives on one device. The derivative of a stage is taken through the array
form (``_ssprk._riemann_flux_rhs_on_padded_shards``).
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import GHOST_CELLS, IDEAL_GAS, RIEMANN_FLUX

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._interface_fluxes._riemann_flux import (
    maximum_signal_speed,
    transverse_compression,
)
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import (
    _guard_pallas_local,
    _interface_flux_pallas_local,
    _kernel_scalars,
    _riemann_flux_pallas_supported,
)
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (
    _ct_plain_curl_pallas_local,
    _ct_update_cell_center_fields_pallas_local,
    _ct_update_cell_center_fields_pallas_supported,
)
from astronomix._finite_difference._magnetic_update._energy_flux_pallas import _ct_energy_flux_rhs_pallas_local
from astronomix._finite_difference._magnetic_update._flux_form_ct_pallas import (
    CT_WENO_EPSILON_RELATIVE,
    EPSILON_FLOOR,
    _cell_velocity_and_cold_speed_pallas_local,
    _edge_emfs_pallas_local,
    _flux_form_ct_pallas_supported,
)
from astronomix._finite_difference._ratio_rules import _floored_ratio
from astronomix._finite_difference._time_integrators._ssprk_pallas import _hydro_flux_div_axis_pallas_local
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _mesh_split_axes,
    _shard_context,
    shard_map_on_padded_shards,
)

#: The halo of the stage state per side of a split axis (module docstring).
MHD_HALO = 7
HYDRO_HALO = 4

#: The margins of the regions the steps write (module docstring).
REBUILT_STATE_MARGIN = 6
FLUX_HIGH_MARGIN = 2
GUARDED_FLUX_MARGIN = 1
FACE_VELOCITY_MARGIN = 6
EMF_MARGIN = 3


def _stage_halo(config: SimulationConfig) -> int:
    """The halo of the stage state of the form of ``config``."""
    return MHD_HALO if config.mhd else HYDRO_HALO


def _region(shard_shape, split_axes, margin: int, axes=None) -> tuple:
    """The spatial shape of the shard widened by ``margin`` cells per side along the split axes (in ``axes``)."""
    return tuple(
        extent + 2 * margin if axis in split_axes and (axes is None or axis in axes) else extent
        for axis, extent in enumerate(shard_shape)
    )


def _regions(shard_shape, split_axes, config: SimulationConfig) -> list:
    """The spatial shapes of every region a step of the stage writes."""
    margins = [0, FLUX_HIGH_MARGIN, GUARDED_FLUX_MARGIN]
    regions = [_region(shard_shape, split_axes, margin, (axis,)) for margin in margins for axis in range(3)]
    if config.mhd:
        regions += [_region(shard_shape, split_axes, margin) for margin in (REBUILT_STATE_MARGIN, EMF_MARGIN)]
    return regions


def padded_stage_supported(conserved_state, config: SimulationConfig) -> bool:
    """
    Whether a stage runs on padded shards: ``scheme = RIEMANN_FLUX``, the
    active Pallas mesh splits a spatial axis of the 3D state, the kernels of
    the scheme run (MHD: also the constrained-transport kernels), the
    boundaries are periodic, every split axis holds at least the halo per
    shard, and the block shape tiles every region of the stage.
    """
    if config.solver_config.scheme != RIEMANN_FLUX or config.dimensionality != 3 or conserved_state.ndim != 4:
        return False
    if config.boundary_handling == GHOST_CELLS or not _riemann_flux_pallas_supported(conserved_state, config):
        return False
    if config.mhd and not (
        _flux_form_ct_pallas_supported(conserved_state, config)
        and _ct_update_cell_center_fields_pallas_supported(conserved_state, config)
    ):
        return False
    split = _mesh_split_axes(conserved_state)
    if not split:
        return False
    halo = _stage_halo(config)
    shard_shape = [int(extent) for extent in conserved_state.shape[1:]]
    for axis, num_devices in split:
        if shard_shape[axis] % num_devices or shard_shape[axis] // num_devices < halo:
            return False
        shard_shape[axis] //= num_devices
    for region in _regions(shard_shape, {axis for axis, _ in split}, config):
        block = _as_3tuple_block_shape(config.backend_config.pallas_block_shape, 3, spatial_shape=region)
        if any(extent % size for extent, size in zip(region, block)):
            return False
    return True


def riemann_flux_stage_on_padded_shards(
    conserved_state,
    face_fields,
    dt_tilde,
    grid_spacing,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The increments of a stage on padded shards (``padded_stage_supported``).

    Args:
        conserved_state: The conserved state of the stage.
        face_fields: The interface fields (bx, by, bz); None in the
            hydrodynamic form.
        dt_tilde: The stage time step.
        grid_spacing: The cell size.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        ``(state, increment, field increments, density fluxes)``: the
        conserved state with the cell-centred field rebuilt from the faces
        (the state itself in the hydrodynamic form), the increment of the
        conserved state (with the CT half of the energy flux for the ideal
        gas), the increments of the interface fields (None in the
        hydrodynamic form) and the density fluxes of the three axes.
    """
    mhd = face_fields is not None
    state_inputs = (conserved_state,) + (tuple(field[None] for field in face_fields) if mhd else ())

    def body(*arguments, split_axes, global_maximum):
        padded_state = arguments[0]
        padded_fields = arguments[1:4] if mhd else None
        time_step, spacing, parameters = arguments[-3:]
        return _stage_on_padded_shard(
            padded_state,
            padded_fields,
            time_step,
            spacing,
            parameters,
            split_axes,
            global_maximum,
            config,
            registered_variables,
        )

    outputs = shard_map_on_padded_shards(
        body,
        state_inputs,
        (dt_tilde, grid_spacing, params),
        width=_stage_halo(config),
        num_outputs=4 if mhd else 2,
    )
    density_fluxes = tuple(outputs[-1][axis] for axis in range(3))
    if not mhd:
        return conserved_state, outputs[0], None, density_fluxes
    state, increment, field_increments, _ = outputs
    return state, increment, tuple(field_increments[axis] for axis in range(3)), density_fluxes


def _stage_on_padded_shard(
    padded_state,
    padded_fields,
    dt_tilde,
    grid_spacing,
    params: SimulationParams,
    split_axes,
    global_maximum,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The stage on one padded shard (module docstring): the steps of
    ``_ssprk._riemann_flux_rhs_on_padded_shards``'s array form, each on its
    region.

    Returns:
        MHD: (rebuilt state, increment of the conserved state, stacked
        increments of the interface fields, stacked density fluxes); the
        hydrodynamic form: (increment, stacked density fluxes); all on the
        cells of the shard.
    """
    halo = _stage_halo(config)
    shard_shape = tuple(
        extent - 2 * halo if axis in split_axes else extent for axis, extent in enumerate(padded_state.shape[1:])
    )
    dtype = padded_state.dtype

    def on_region(margin, build, axes=None):
        """Run a kernel build that writes the shard widened by ``margin`` (along ``axes``)."""
        with _shard_context((_region(shard_shape, split_axes, margin, axes), None)):
            return build()

    def shard_cells(array, margin, axes=None):
        """The cells of the shard in an array that covers the shard widened by ``margin``."""
        for axis in split_axes:
            if axes is None or axis in axes:
                array = array[(slice(None),) * (1 + axis) + (slice(margin, margin + shard_shape[axis]),)]
        return array

    # 1. the cell-centred field from the faces (MHD), on the region of the rebuilt state
    state, state_margin = padded_state, halo
    if padded_fields is not None:
        scalars = ()
        if config.equation_of_state == IDEAL_GAS:
            scalars = tuple(
                jnp.asarray(scalar, dtype=dtype)
                for scalar in (params.gamma, params.minimum_density, params.minimum_pressure)
            )
        stacked_fields = jnp.concatenate(padded_fields)
        state = on_region(
            REBUILT_STATE_MARGIN,
            lambda: _ct_update_cell_center_fields_pallas_local(
                padded_state,
                stacked_fields,
                *scalars,
                config=config,
                registered_variables=registered_variables,
                hold_energy=config.equation_of_state == IDEAL_GAS,
            ),
        )
        state_margin = REBUILT_STATE_MARGIN

    # 2. transverse compression and the maximum signal speeds (one all-reduce for the three axes)
    compression = transverse_compression(state, params, config, registered_variables)
    state_of_shard = shard_cells(state, state_margin)
    largest_speeds = global_maximum(
        jnp.stack(
            [maximum_signal_speed(state_of_shard, params, config, registered_variables, axis) for axis in range(3)]
        )
    )

    # 3. the flux-form constrained transport (``_flux_form_ct._increments``): it does not read the maxima, so it
    # runs while their all-reduce is under way
    energy_flux_channels = None
    if padded_fields is not None and config.equation_of_state == IDEAL_GAS:
        magnetic_index = registered_variables.magnetic_index
        energy_flux_channels = (
            registered_variables.energy_index,
            (magnetic_index.x, magnetic_index.y, magnetic_index.z),
        )
    if padded_fields is not None:
        face_velocities = on_region(
            FACE_VELOCITY_MARGIN,
            lambda: _cell_velocity_and_cold_speed_pallas_local(
                state,
                jnp.asarray(params.minimum_density, dtype=dtype),
                config=config,
                registered_variables=registered_variables,
            ),
        )
        emfs = on_region(
            EMF_MARGIN,
            lambda: _edge_emfs_pallas_local(
                stacked_fields,
                face_velocities,
                jnp.asarray(CT_WENO_EPSILON_RELATIVE, dtype=dtype),
                jnp.asarray(EPSILON_FLOOR, dtype=dtype),
                config=config,
            ),
        )
        time_step, spacing = jnp.asarray(dt_tilde, dtype=dtype), jnp.asarray(grid_spacing, dtype=dtype)
        field_increments = on_region(
            0, lambda: _ct_plain_curl_pallas_local(emfs, jnp.asarray(time_step / spacing, dtype=dtype), config)
        )
        if energy_flux_channels is not None:
            energy = on_region(
                0,
                lambda: _ct_energy_flux_rhs_pallas_local(
                    state,
                    emfs,
                    jnp.asarray(time_step / spacing, dtype=dtype),
                    config=config,
                    magnetic_channels=energy_flux_channels[1],
                ),
            )

    # 4. the fluid: flux, guard and divergence along each axis (``_ssprk._riemann_flux_increment``)
    scalars = _kernel_scalars(params, config, dtype)
    dt_over_dx = dt_tilde / grid_spacing
    increment, density_fluxes = None, []
    for axis in range(3):
        reference_step = _floored_ratio(1.0, largest_speeds[axis])
        fields = (compression[axis][None],)
        if padded_fields is not None:
            fields = (padded_fields[axis],) + fields
        flux_high = on_region(
            FLUX_HIGH_MARGIN,
            lambda: _interface_flux_pallas_local(
                state, fields, *scalars, config=config, registered_variables=registered_variables, axis=axis
            ),
            (axis,),
        )
        flux = on_region(
            GUARDED_FLUX_MARGIN,
            lambda: _guard_pallas_local(
                state,
                flux_high,
                *scalars,
                jnp.asarray(reference_step, dtype=dtype),
                config=config,
                registered_variables=registered_variables,
                axis=axis,
            ),
            (axis,),
        )
        density_fluxes.append(shard_cells(flux, GUARDED_FLUX_MARGIN, (axis,))[registered_variables.density_index])
        accumulator = increment
        increment = on_region(
            0,
            lambda: _hydro_flux_div_axis_pallas_local(
                flux,
                dt_over_dx,
                config,
                axis=axis,
                rhs_accumulator=accumulator,
                scale_in=1.0 if accumulator is None else jnp.asarray(1.0),
                magnetic_state=state if energy_flux_channels is not None else None,
                energy_flux_channels=energy_flux_channels,
            ),
        )
    density_fluxes = jnp.stack(density_fluxes)
    if padded_fields is None:
        return increment, density_fluxes
    if energy_flux_channels is not None:
        increment = increment.at[energy_flux_channels[0]].add(energy[0])
    return state_of_shard, increment, field_increments, density_fluxes
