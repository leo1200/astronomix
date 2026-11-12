"""
The kernels of the split MHD scheme with induction source: the Pallas backend of the sweeps of
``muscl_hancock_step`` with ``divergence_control = INDUCTION_SOURCE``, of the parabolic cleaning and of the
signal speeds of its time step.

A sweep is one kernel. For a block of cells it reads the five cells along the axis, forms the limited and
clipped slopes and the predicted edge values of the cell and of its two neighbours, solves the two interfaces
of the cell with HLL5R and updates the cell in the published form of the scheme (Waagan 2009),

    U_i <- U_i - dt/h (F_l[i+1/2] - F_r[i-1/2] + u_i (B_n,i^+ - B_n,i^-))    (the last term on the field rows),

with the flux for the left neighbour ``F_l`` at the right face and the flux for the right neighbour ``F_r`` at
the left face. The array form (``_divergence/_induction_source.py``) adds ``F_l[i-1/2]`` and takes it away
again, so the two forms agree to rounding on the field rows and have the same arithmetic on the others. Every
interface is solved twice, once for each neighbour, and the edge values of a cell three times: Triton cannot
shift tiles, and separate arrays of edge values and fluxes would cost three more states.

The cleaning is two kernels with a reach of one cell: the divergence as one field, then the diffusion of the
field in place. The time step takes the signal speeds as one field per cell and reduces it outside the kernel.
A step therefore holds two states and one field.

Nothing of the scheme is written a second time: the kernels call the elementwise functions of the array form
(limiter, slope clips, predictor, face states, HLL5R, conversions). ``_TileRows`` lets the functions that
index and update a state run on the tiles of a block. The derivative of every entry is the array form.
"""

# general
import operator

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    GHOST_CELLS,
    IDEAL_GAS,
    INDUCTION_SOURCE,
    ISOTHERMAL,
    PERIODIC_ROLL,
    PIECEWISE_LINEAR,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._divergence._induction_source import _hll5r_fluxes, _parabolic_cleaning
from astronomix._finite_volume._interface_frame import (
    FRAME_FIELD,
    FRAME_VELOCITY,
    _barotropic_law,
    _interface_frame_indices,
    _mhd_face_state,
    _state_components_from_frame,
)
from astronomix._finite_volume._reconstruction import _clipped_gradients, _hancock_predictor, _limited_gradients
from astronomix._finite_volume._riemann._hll5r import hll5r_signal_speed
from astronomix._finite_volume._stage import (
    _conserved_components_from_primitive,
    _hancock_terms,
    _primitive_components_from_conserved,
    _sweep_stage,
)
from astronomix._finite_volume._stage_pallas import _fv_stage_block_shape, _interior_spatial_shape
from astronomix._finite_volume._timestep_estimation._timestep_estimator import _maximum_hll5r_speed
from astronomix._pallas_helpers import (
    _backend_is_pallas,
    _current_pallas_mesh,
    _pallas_call_sharded,
    _pallas_compiler_params,
    diffable_pallas_call_n,
    pl,
)

#: The number of cells a sweep reads on each side of a cell along its axis.
_SWEEP_REACH = 2


# -------------------------------------------------------------
# ===================== ↓ Support predicate ↓ =================
# -------------------------------------------------------------


def _reference_kernels_supported(state, config: SimulationConfig) -> bool:
    """
    Return whether the kernels of this module run the configured scheme on ``state``.

    They cover the split MHD scheme with induction source and piecewise linear reconstruction for the ideal
    and the isothermal gas, in two and three dimensions, with a block shape that divides the updated cells.
    Compiled kernels need a GPU and a float32 state; with ``pallas_interpret`` every device and precision is
    served. With ghost cells the sweeps run on one device only (the halo exchange of a device mesh is
    periodic). What is refused runs in array form.

    Args:
        state: The primitive state.
        config: The simulation configuration.

    Returns:
        Whether the kernels are used.
    """
    solver_config = config.solver_config
    if pl is None or not _backend_is_pallas(config):
        return False
    if solver_config.divergence_control != INDUCTION_SOURCE or solver_config.reconstruction != PIECEWISE_LINEAR:
        return False
    if config.equation_of_state not in (IDEAL_GAS, ISOTHERMAL):
        return False
    dimensionality = int(config.dimensionality)
    if dimensionality == 1 or state.ndim != dimensionality + 1:
        return False
    if config.boundary_handling == GHOST_CELLS:
        mesh = _current_pallas_mesh()
        if config.num_ghost_cells < _SWEEP_REACH or (mesh is not None and mesh.size > 1):
            return False
    if not config.backend_config.pallas_interpret:
        if jax.default_backend() != "gpu" or state.dtype != jnp.float32:
            return False
    updated_extents = _interior_spatial_shape(state, config)
    block_extents = _fv_stage_block_shape(state, config)[:dimensionality]
    return all(int(extent) % int(block) == 0 for extent, block in zip(updated_extents, block_extents, strict=True))


def _periodic_kernels_supported(state, config: SimulationConfig) -> bool:
    """
    Return whether the kernels of the cleaning and of the time step run on ``state``: as the sweep kernel,
    and in the periodic layout only. With ghost cells both stay in array form: the cleaning needs the
    divergence in the first layer of ghost cells and the time step the faces of the boundary.
    """
    return config.boundary_handling == PERIODIC_ROLL and _reference_kernels_supported(state, config)


# -------------------------------------------------------------
# ===================== ↑ Support predicate ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Blocks of cells ↓ ===================
# -------------------------------------------------------------


class _TileRows(tuple):
    """
    The rows of a state on one block of cells: one tile per variable.

    A kernel cannot stack tiles and read a row back (Triton has neither slice nor scatter). The functions of
    the array form that index a state and update single rows get this tuple instead: it has the arithmetic
    between states and the ``.at[row].add(...)`` and ``.at[row].set(...)`` updates of an array.
    """

    def _combined(self, other, operation):
        if isinstance(other, _TileRows):
            return _TileRows(operation(tile, other_tile) for tile, other_tile in zip(self, other, strict=True))
        return _TileRows(operation(tile, other) for tile in self)

    def __add__(self, other):
        return self._combined(other, operator.add)

    def __sub__(self, other):
        return self._combined(other, operator.sub)

    def __mul__(self, other):
        return self._combined(other, operator.mul)

    __rmul__ = __mul__

    @property
    def at(self):
        """The row updates, as for arrays: ``rows.at[row].add(value)`` returns new rows."""
        return _RowUpdate(self)


class _RowUpdate:
    """The ``.at[row]`` updates of ``_TileRows``."""

    def __init__(self, rows: _TileRows, row: int = None):
        self.rows = rows
        self.row = row

    def __getitem__(self, row: int):
        return _RowUpdate(self.rows, row)

    def set(self, value) -> _TileRows:
        """The rows with ``value`` in place of the selected row."""
        return _TileRows(value if row == self.row else tile for row, tile in enumerate(self.rows))

    def add(self, value) -> _TileRows:
        """The rows with ``value`` added to the selected row."""
        return self.set(self.rows[self.row] + value)


def _block_kernel(block_rows, arrays, scalars, config: SimulationConfig, name: str, output_rows: int = None):
    """
    Run one function per block of cells as a kernel.

    All shapes are read from the first array, so the same build serves the whole grid and the padded shard of
    a device mesh. Indices along an axis wrap, which is the periodic boundary on one device. With ghost cells
    the blocks cover the cells inside them and the output has their shape.

    Args:
        block_rows: The function of a block, ``block_rows(read, *scalar_values) -> {row: tile}``, where
            ``read(position, row, array_axis=0, offset=0)`` is the tile of row ``row`` of the array at
            ``position``, shifted by ``offset`` cells along ``array_axis``.
        arrays: The arrays the kernel reads, each laid out ``(rows, *spatial)`` with the same spatial shape.
        scalars: The scalars the kernel reads (traced values enter here, never through a closure).
        config: The simulation configuration.
        name: The name of the kernel.
        output_rows: The number of rows of the output. ``None`` writes the returned rows into the first array
            in place (periodic layout only); that array may then be read at offset zero only, and its other
            rows keep their values.

    Returns:
        The output array ``(output_rows, *updated cells)``, or the updated first array.
    """
    dimensionality = int(config.dimensionality)
    dtype = arrays[0].dtype
    spatial_shape = tuple(int(extent) for extent in arrays[0].shape[1:])
    first_cell = config.num_ghost_cells if config.boundary_handling == GHOST_CELLS else 0
    updated_extents = _interior_spatial_shape(arrays[0], config)
    block_extents = tuple(_fv_stage_block_shape(arrays[0], config)[:dimensionality])
    grid = tuple(extent // block for extent, block in zip(updated_extents, block_extents, strict=True))
    in_place = output_rows is None
    output_shape = arrays[0].shape if in_place else (output_rows,) + updated_extents

    def whole_array_spec(array):
        return pl.BlockSpec(array.shape, lambda *program_ids: (0,) * array.ndim)

    scalar_spec = pl.BlockSpec((), lambda *program_ids: ())
    output_spec = pl.BlockSpec(
        (output_shape[0],) + block_extents, lambda *program_ids: (0,) + tuple(program_ids[:dimensionality])
    )

    def kernel(*references):
        array_references = references[: len(arrays)]
        scalar_values = [reference[()] for reference in references[len(arrays) : -1]]
        output_reference = references[-1]
        cell_indices = []
        for array_axis in range(dimensionality):
            shape = [1] * dimensionality
            shape[array_axis] = block_extents[array_axis]
            local = jnp.arange(block_extents[array_axis]).reshape(shape)
            cell_indices.append(first_cell + pl.program_id(array_axis) * block_extents[array_axis] + local)

        def read(position, row, array_axis=0, offset=0):
            indices = list(cell_indices)
            if offset != 0:
                indices[array_axis] = (indices[array_axis] + offset) % spatial_shape[array_axis]
            return array_references[position][(row, *indices)]

        for row, tile in block_rows(read, *scalar_values).items():
            output_reference[(row,) + (slice(None),) * dimensionality] = tile

    keyword_arguments = {"input_output_aliases": {0: 0}} if in_place else {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        keyword_arguments["compiler_params"] = compiler_params
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(output_shape, dtype),
        grid=grid,
        in_specs=[whole_array_spec(array) for array in arrays] + [scalar_spec] * len(scalars),
        out_specs=output_spec,
        interpret=config.backend_config.pallas_interpret,
        name=name,
        **keyword_arguments,
    )(*arrays, *(jnp.asarray(scalar, dtype=dtype) for scalar in scalars))


def _central_difference_at(read, position: int, row: int, array_axis: int):
    """The central difference ``(f[i+1] - f[i-1]) / 2`` of one row along an array axis on a block."""
    return 0.5 * (read(position, row, array_axis, 1) - read(position, row, array_axis, -1))


# -------------------------------------------------------------
# ===================== ↑ Blocks of cells ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Block functions ↓ ===================
# -------------------------------------------------------------


def _sweep_block(axis: int, config: SimulationConfig, registered_variables: RegisteredVariables):
    """
    The block function of one sweep along ``axis`` (1-based).

    The slopes, clips and predictor are the statements of ``_hancock_face_values`` with the limiter applied
    row by row (the limiter compares, which rows of tiles cannot). The source of a cell is the one of
    ``_source_of_cells`` with both edges of the cell at hand instead of a shifted array.

    Returns:
        ``block_rows(read, time_step, gamma, sound_speed, density_floor, pressure_floor)``: all rows of the
        primitive state after the sweep.
    """
    num_vars = registered_variables.num_vars
    frame_indices = _interface_frame_indices(axis, config, registered_variables)
    limiter = config.solver_config.limiter

    def block_rows(read, time_step, gamma, sound_speed, density_floor, pressure_floor):
        barotropic_law = _barotropic_law(SimulationParams(isothermal_sound_speed=sound_speed), config)
        hancock = _hancock_terms(axis, 0.5 * time_step, gamma, config, None, registered_variables, barotropic_law)
        distances = hancock.distances

        def edge_states(cell_minus, cell, cell_plus):
            gradients = _TileRows(
                _limited_gradients(minus, centre, plus, limiter, distances)
                for minus, centre, plus in zip(cell_minus, cell, cell_plus, strict=True)
            )
            gradients = _clipped_gradients(cell, gradients, hancock)
            predicted = cell - _hancock_predictor(cell, gradients, hancock) * hancock.half_step
            left_edge = _mhd_face_state(predicted - gradients * distances.to_left_face, frame_indices, barotropic_law)
            right_edge = _mhd_face_state(predicted + gradients * distances.to_right_face, frame_indices, barotropic_law)
            return left_edge, right_edge

        cells = [_TileRows(read(0, row, axis - 1, offset) for row in range(num_vars)) for offset in range(-2, 3)]
        _, right_edge_of_left_cell = edge_states(*cells[0:3])
        left_edge, right_edge = edge_states(*cells[1:4])
        left_edge_of_right_cell, _ = edge_states(*cells[2:5])
        _, flux_at_left_face = _hll5r_fluxes(right_edge_of_left_cell, left_edge, gamma, barotropic_law)
        flux_at_right_face, _ = _hll5r_fluxes(right_edge, left_edge_of_right_cell, gamma, barotropic_law)

        differences = [
            None if index is None else flux_at_right_face[position] - flux_at_left_face[position]
            for position, index in enumerate(frame_indices)
        ]
        normal_field_difference = right_edge.field_normal - left_edge.field_normal
        for position in range(FRAME_VELOCITY.start, FRAME_VELOCITY.stop):
            field_position = position + FRAME_FIELD.start - FRAME_VELOCITY.start
            # The velocity of the cell after the predictor is the mean of its two edge values.
            velocity = 0.5 * (right_edge[position] + left_edge[position])
            differences[field_position] = differences[field_position] + velocity * normal_field_difference

        rates = _state_components_from_frame(differences, frame_indices, num_vars)
        conserved = _conserved_components_from_primitive(list(cells[2]), gamma, config, registered_variables)
        conserved = [
            value - time_step * (rate / config.grid_spacing) for value, rate in zip(conserved, rates, strict=True)
        ]
        primitive = _primitive_components_from_conserved(
            conserved, gamma, density_floor, pressure_floor, config, registered_variables
        )
        return dict(enumerate(primitive))

    return block_rows


def _divergence_block(config: SimulationConfig, registered_variables: RegisteredVariables):
    """The block function of the central divergence of the field, as in ``_parabolic_cleaning``: one row."""
    field_indices = tuple(registered_variables.magnetic_index)[: config.dimensionality]

    def block_rows(read):
        divergence = 0.0
        for array_axis, index in enumerate(field_indices):
            divergence = divergence + _central_difference_at(read, 0, index, array_axis) / config.grid_spacing
        return {0: divergence}

    return block_rows


def _diffusion_block(config: SimulationConfig, registered_variables: RegisteredVariables):
    """
    The block function of the diffusion of the field, as in ``_parabolic_cleaning``: it reads the state (first
    array) at the cell and the divergence (second array) around it and returns the rows of the field.
    """
    field_indices = tuple(registered_variables.magnetic_index)[: config.dimensionality]

    def block_rows(read, step_diffusivity):
        rows = {}
        for array_axis, index in enumerate(field_indices):
            divergence_gradient = _central_difference_at(read, 1, 0, array_axis) / config.grid_spacing
            rows[index] = read(0, index) + step_diffusivity * divergence_gradient
        return rows

    return block_rows


def _signal_speed_block(config: SimulationConfig, registered_variables: RegisteredVariables):
    """
    The block function of the largest HLL5R signal speed at the left faces of a cell, as in
    ``_maximum_hll5r_speed``: one row.
    """
    num_vars = registered_variables.num_vars

    def block_rows(read, gamma, sound_speed):
        barotropic_law = _barotropic_law(SimulationParams(isothermal_sound_speed=sound_speed), config)
        exponent = gamma if barotropic_law is None else barotropic_law.exponent
        sound_speed_squared_of = None if barotropic_law is None else barotropic_law.sound_speed_squared_of
        cell = _TileRows(read(0, row) for row in range(num_vars))
        maximum_speed = None
        for axis in range(1, config.dimensionality + 1):
            frame_indices = _interface_frame_indices(axis, config, registered_variables)
            left_cell = _TileRows(read(0, row, axis - 1, -1) for row in range(num_vars))
            signal_speed = hll5r_signal_speed(
                _mhd_face_state(left_cell, frame_indices, barotropic_law),
                _mhd_face_state(cell, frame_indices, barotropic_law),
                exponent,
                sound_speed_squared_of,
            )
            maximum_speed = signal_speed if maximum_speed is None else jnp.maximum(maximum_speed, signal_speed)
        return {0: maximum_speed}

    return block_rows


# -------------------------------------------------------------
# ===================== ↑ Block functions ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Entries ↓ ===========================
# -------------------------------------------------------------


def _reference_sweep_pallas(
    primitive_state,
    axis: int,
    sweep_time_step,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    One sweep of the split MHD scheme along ``axis`` as one kernel (``_sweep_stage`` in array form, which is
    also its derivative). On a device mesh the shards exchange the reach of the sweep along its axis. Ghost
    cells keep their values.

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based) of the sweep.
        sweep_time_step: The time step of the sweep.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The primitive state after the sweep.
    """
    dimensionality = int(config.dimensionality)
    dtype = primitive_state.dtype
    num_vars = int(primitive_state.shape[0])
    ghost_cells = config.num_ghost_cells if config.boundary_handling == GHOST_CELLS else 0
    updated_cells = (slice(None),) + (slice(ghost_cells, -ghost_cells),) * dimensionality
    block_extents = _fv_stage_block_shape(primitive_state, config)[:dimensionality]
    halo = tuple(_SWEEP_REACH if array_axis == axis - 1 else 0 for array_axis in range(dimensionality))
    block_rows = _sweep_block(axis, config, registered_variables)

    def pallas_branch(state, time_step, gamma_value, simulation_params):
        scalars = (
            time_step,
            gamma_value,
            simulation_params.isothermal_sound_speed,
            simulation_params.minimum_density,
            simulation_params.minimum_pressure,
        )

        def local_build(local_state):
            return _block_kernel(block_rows, (local_state,), scalars, config, f"reference_sweep_{axis}", num_vars)

        updated = _pallas_call_sharded(local_build, state_inputs=(state,), halo=halo, block_shape=block_extents)
        return state.at[updated_cells].set(updated) if ghost_cells else updated

    def native_branch(state, time_step, gamma_value, simulation_params):
        return _sweep_stage(state, axis, time_step, gamma_value, config, simulation_params, registered_variables)

    return diffable_pallas_call_n(
        (primitive_state, jnp.asarray(sweep_time_step, dtype=dtype), jnp.asarray(gamma, dtype=dtype), params),
        pallas_branch=pallas_branch,
        native_branch=native_branch,
    )


def _parabolic_cleaning_pallas(
    primitive_state,
    signal_speed,
    dt,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The parabolic cleaning as two kernels (``_parabolic_cleaning`` in array form, which is also its
    derivative): the divergence as one field, then the diffusion of the field in place.

    Args:
        primitive_state: The primitive state.
        signal_speed: The speed ``lambda``: the largest signal speed of the step.
        dt: The time step.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive state with the cleaned field.
    """
    dimensionality = int(config.dimensionality)
    dtype = primitive_state.dtype
    block_extents = _fv_stage_block_shape(primitive_state, config)[:dimensionality]
    one_cell = (1,) * dimensionality
    no_cell = (0,) * dimensionality
    divergence_rows = _divergence_block(config, registered_variables)
    diffusion_rows = _diffusion_block(config, registered_variables)

    def pallas_branch(state, speed, time_step):
        step_diffusivity = time_step * (speed * config.grid_spacing / (2.0 * dimensionality))

        def divergence_build(local_state):
            return _block_kernel(divergence_rows, (local_state,), (), config, "reference_divergence", 1)

        def diffusion_build(local_state, local_divergence):
            arrays = (local_state, local_divergence)
            return _block_kernel(diffusion_rows, arrays, (step_diffusivity,), config, "reference_diffusion")

        divergence = _pallas_call_sharded(
            divergence_build, state_inputs=(state,), halo=one_cell, block_shape=block_extents
        )
        return _pallas_call_sharded(
            diffusion_build,
            state_inputs=(state, divergence),
            halo=one_cell,
            block_shape=block_extents,
            input_halos=(no_cell, one_cell),
        )

    def native_branch(state, speed, time_step):
        return _parabolic_cleaning(state, speed, time_step, config, registered_variables)

    return diffable_pallas_call_n(
        (primitive_state, jnp.asarray(signal_speed, dtype=dtype), jnp.asarray(dt, dtype=dtype)),
        pallas_branch=pallas_branch,
        native_branch=native_branch,
    )


def _maximum_hll5r_speed_pallas(
    primitive_state,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    The largest HLL5R signal speed between neighbouring cells (``_maximum_hll5r_speed`` in array form, which
    is also its derivative): one kernel for the speeds per cell and the reduction of that field.

    Args:
        primitive_state: The primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The largest signal speed.
    """
    dimensionality = int(config.dimensionality)
    block_extents = _fv_stage_block_shape(primitive_state, config)[:dimensionality]
    block_rows = _signal_speed_block(config, registered_variables)

    def pallas_branch(state, simulation_params):
        scalars = (simulation_params.gamma, simulation_params.isothermal_sound_speed)

        def local_build(local_state):
            return _block_kernel(block_rows, (local_state,), scalars, config, "reference_signal_speed", 1)

        signal_speeds = _pallas_call_sharded(
            local_build, state_inputs=(state,), halo=(1,) * dimensionality, block_shape=block_extents
        )
        return jnp.max(signal_speeds)

    def native_branch(state, simulation_params):
        return _maximum_hll5r_speed(state, config, simulation_params, registered_variables)

    return diffable_pallas_call_n((primitive_state, params), pallas_branch=pallas_branch, native_branch=native_branch)


# -------------------------------------------------------------
# ===================== ↑ Entries ↑ ===========================
# -------------------------------------------------------------
