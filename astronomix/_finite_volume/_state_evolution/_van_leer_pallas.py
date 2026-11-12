"""
Pallas backend of the VL2 finite-volume scheme.

Each VL2 stage is a single fused Pallas kernel: for a block of cells it reads
the stage's primitive stencil, reconstructs (donor cell or AthenaPK's PLM) the
two interface states on every face of the cell, evaluates the configured
Riemann solver on all ``2 * dim`` faces, forms the flux divergence, applies the
Dedner GLM source, and converts the updated conserved state back to primitives.
The initial conserved state ``U^n`` is recomputed in-kernel from ``W^n`` (a
pointwise read), so a full VL2 step is two kernel launches plus the time-step
reduction and holds only two state-sized buffers.

Every face flux is evaluated twice (once for each adjacent cell): Pallas'
Triton backend cannot shift register tiles, so faces cannot be shared between
the cells of a block. The alternative, AthenaPK's layout of separate flux
arrays, would halve the Riemann work but add three state-sized buffers and
roughly double the memory traffic.

AthenaPK's first-order flux correction runs as repeated stage evaluations: the
kernel reports which cells' updates lost positivity, and a re-evaluation with a
correction mask replaces the fluxes of the faces around those cells by
donor-cell local Lax-Friedrichs fluxes. In the common case that no cell fails,
this costs only one reduction over the reported codes.

The physics is not re-implemented here: the kernel calls the same elementwise
functions as the native path (``_athena_riemann_solvers``, the component
helpers of ``_van_leer_integrator`` and the extended Dedner source terms of
``_glm_divergence_cleaning``), so both backends evaluate the same expressions.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    GHOST_CELLS,
    LAX_FRIEDRICHS,
    PERIODIC_ROLL,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._magnetic_update._glm_divergence_cleaning import (
    _extended_dedner_source_terms,
)
from astronomix._pallas_helpers import (
    _as_3tuple_block_shape,
    _backend_is_pallas,
    _current_pallas_mesh,
    _pallas_call_sharded,
    _pallas_compiler_params,
    diffable_pallas_call_n,
    pl,
)


# -------------------------------------------------------------
# ===================== ↓ Support predicate ↓ =================
# -------------------------------------------------------------


def _interior_spatial_shape(state, config: SimulationConfig):
    """The spatial shape of the cells the kernel updates (ghost cells excluded)."""
    spatial_shape = tuple(int(extent) for extent in state.shape[1:])
    if config.boundary_handling == GHOST_CELLS:
        num_ghost_cells = config.num_ghost_cells
        return tuple(extent - 2 * num_ghost_cells for extent in spatial_shape)
    return spatial_shape


def _vl2_block_shape(state, config: SimulationConfig):
    """The Pallas block shape, resolved on the updated (interior) extents."""
    ndim = int(config.dimensionality)
    return _as_3tuple_block_shape(
        config.backend_config.pallas_block_shape,
        ndim,
        spatial_shape=_interior_spatial_shape(state, config),
    )


def _vl2_pallas_supported(state, config: SimulationConfig) -> bool:
    """
    Whether the fused VL2 stage kernel applies to this state and configuration.

    Args:
        state: The (padded) primitive state.
        config: The simulation configuration.

    Returns:
        True if the Pallas backend can run the VL2 stages.
    """
    if pl is None:
        return False
    if not _backend_is_pallas(config):
        return False
    if config.solver_config.time_integrator != VL2 or config.geometry != CARTESIAN:
        return False
    ndim = int(config.dimensionality)
    if state.ndim != ndim + 1:
        return False
    if config.boundary_handling not in (GHOST_CELLS, PERIODIC_ROLL):
        return False
    if config.boundary_handling == GHOST_CELLS:
        # The halo exchange of the multi-device wrapper is periodic, which only
        # matches the periodic-roll layout.
        mesh = _current_pallas_mesh()
        if mesh is not None and mesh.size > 1:
            return False
    block_shape = _vl2_block_shape(state, config)
    for extent, block in zip(_interior_spatial_shape(state, config), block_shape[:ndim], strict=True):
        if extent % block != 0:
            return False
    return True


# -------------------------------------------------------------
# ===================== ↑ Support predicate ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Fused stage kernel ↓ ================
# -------------------------------------------------------------


def _vl2_stage_pallas_local(
    stage_primitive_state,
    base_primitive_state,
    correction_mask,
    stage_time_step,
    cleaning_speed,
    damping_factor,
    gamma,
    density_floor,
    pressure_floor,
    *,
    piecewise_linear: bool,
    report_failures: bool,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Build and run the fused stage kernel on one (possibly halo-padded) shard.

    Every shape-dependent quantity is read from ``stage_primitive_state`` so the
    same build serves the global array and halo-padded local shards.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_primitive_state: The primitive state ``W^n`` whose conserved form
            the update starts from, or ``None`` when it is the stage state (the
            predictor stage).
        correction_mask: ``None``, or a ``(1, *grid)`` array that is one in the
            cells whose faces take first-order LLF fluxes.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        cleaning_speed: The GLM cleaning speed (zero for hydrodynamics).
        damping_factor: The stage's psi damping factor.
        gamma: The adiabatic index.
        density_floor: The density floor (used only with floors active).
        pressure_floor: The pressure floor (used only with floors active).
        piecewise_linear: PLM (else donor-cell) reconstruction.
        report_failures: Also return the positivity failure code of every
            cell's update (for the first-order flux correction).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive state after the stage (ghost cells not updated), and with
        ``report_failures`` the ``(1, *grid)`` failure codes.
    """
    # Imported here because the integrator module imports this module at its
    # top.
    from astronomix._finite_volume._state_evolution._van_leer_integrator import (
        _conserved_components_from_primitive,
        _interface_flux_from_states,
        _interface_frame_indices,
        _positivity_failure_code,
        _primitive_components_from_conserved,
        _state_components_from_frame,
        _van_leer_slopes,
    )

    # -------------------------------------------------------------
    # ===================== ↓ Static layout ↓ =====================
    # -------------------------------------------------------------

    ndim = int(config.dimensionality)
    num_vars = int(stage_primitive_state.shape[0])
    dtype = stage_primitive_state.dtype
    spatial_shape = tuple(int(extent) for extent in stage_primitive_state.shape[1:])
    extents_3d = spatial_shape + (1,) * (3 - ndim)
    ghost_cells = config.boundary_handling == GHOST_CELLS
    cell_offset = config.num_ghost_cells if ghost_cells else 0
    block_extents = _vl2_block_shape(stage_primitive_state, config)
    interior_extents = _interior_spatial_shape(stage_primitive_state, config)
    grid = tuple(interior_extents[axis] // block_extents[axis] for axis in range(ndim))

    frame_indices_per_axis = [
        _interface_frame_indices(axis, config, registered_variables) for axis in range(1, ndim + 1)
    ]
    stencil_reach = 2 if piecewise_linear else 1
    has_base = base_primitive_state is not None
    has_mask = correction_mask is not None
    mhd = config.mhd

    # -------------------------------------------------------------
    # ===================== ↑ Static layout ↑ =====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Block specifications ↓ ==================
    # -------------------------------------------------------------

    def whole_array_spec(array):
        """A block that is the whole array (read with explicit indices)."""
        return pl.BlockSpec(array.shape, lambda *program_ids: (0,) * array.ndim)

    def cell_block_spec(leading_extent):
        """The output block of a program: its cells (all of them with ghost cells)."""
        if ghost_cells:
            # Ghost-cell layout: the interior starts at an offset that is not a
            # multiple of the block, so the kernel stores with explicit indices
            # into the whole array; the ghost cells are filled afterwards.
            return pl.BlockSpec(
                (leading_extent,) + spatial_shape,
                lambda *program_ids: (0,) * (ndim + 1),
            )
        return pl.BlockSpec(
            (leading_extent,) + tuple(block_extents[:ndim]),
            lambda *program_ids: (0,) + tuple(program_ids[:ndim]),
        )

    scalar_spec = pl.BlockSpec((), lambda *program_ids: ())

    # -------------------------------------------------------------
    # ================= ↑ Block specifications ↑ ==================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ======================== ↓ Kernel ↓ =========================
    # -------------------------------------------------------------

    def kernel(*refs):

        # --------------- ↓ Ref unpacking ↓ ----------------

        refs = list(refs)
        stage_ref = refs.pop(0)
        base_ref = refs.pop(0) if has_base else stage_ref
        mask_ref = refs.pop(0) if has_mask else None
        if report_failures and ghost_cells:
            # The zero-initialised codes input is never read; it is aliased to
            # ``codes_ref`` so that the ghost cells, which the kernel does not
            # write, report POSITIVE.
            refs.pop(0)
        stage_time_step_ref = refs.pop(0)
        cleaning_speed_ref = refs.pop(0)
        damping_factor_ref = refs.pop(0)
        gamma_ref = refs.pop(0)
        density_floor_ref = refs.pop(0)
        pressure_floor_ref = refs.pop(0)
        output_ref = refs.pop(0)
        codes_ref = refs.pop(0) if report_failures else None

        stage_time_step_value = stage_time_step_ref[()]
        cleaning_speed_value = cleaning_speed_ref[()]
        damping_factor_value = damping_factor_ref[()]
        gamma_value = gamma_ref[()]
        minimum_density = density_floor_ref[()]
        minimum_pressure = pressure_floor_ref[()]

        # --------------- ↑ Ref unpacking ↑ ----------------

        # --------------- ↓ Cell indices ↓ ----------------

        cell_indices = []
        for axis in range(ndim):
            shape = [1] * ndim
            shape[axis] = block_extents[axis]
            local = jnp.arange(block_extents[axis]).reshape(shape)
            cell_indices.append(cell_offset + pl.program_id(axis) * block_extents[axis] + local)

        def load(ref, variable_index, axis, offset):
            """Value of ``variable_index`` at the cell shifted by ``offset`` along ``axis``."""
            indices = list(cell_indices)
            if offset != 0:
                shifted = indices[axis] + offset
                if not ghost_cells:
                    shifted = shifted % extents_3d[axis]
                indices[axis] = shifted
            return ref[(variable_index, *indices)]

        def store(ref, variable_index, value):
            """Write ``value`` of ``variable_index`` to the block's cells."""
            if ghost_cells:
                ref[(variable_index, *cell_indices)] = value
            else:
                ref[(variable_index,) + (slice(None),) * ndim] = value

        # --------------- ↑ Cell indices ↑ ----------------

        # --------------- ↓ Flux divergence ↓ ----------------

        def interface_flux(left_values, right_values, frame_indices, riemann_solver):
            """The flux (indexed like the state) of one face from its two states."""
            # Velocity components the layout lacks enter the solver as zeros.
            zero = jnp.zeros_like(left_values[registered_variables.density_index])
            return _state_components_from_frame(
                _interface_flux_from_states(
                    tuple(left_values[index] if index is not None else zero for index in frame_indices),
                    tuple(right_values[index] if index is not None else zero for index in frame_indices),
                    gamma_value,
                    cleaning_speed_value,
                    riemann_solver,
                    mhd,
                ),
                frame_indices,
                num_vars,
            )

        flux_difference = [None] * num_vars
        stencil_per_axis = []
        for axis in range(ndim):
            frame_indices = frame_indices_per_axis[axis]
            stencil = {
                offset: [load(stage_ref, variable, axis, offset) for variable in range(num_vars)]
                for offset in range(-stencil_reach, stencil_reach + 1)
            }
            stencil_per_axis.append(stencil)

            # The (left, right) states of the faces at i - 1/2 and i + 1/2.
            if piecewise_linear:
                right_face_values = {}
                left_face_values = {}
                for center in (-1, 0, 1):
                    faces = [
                        _van_leer_slopes(
                            stencil[center - 1][variable],
                            stencil[center][variable],
                            stencil[center + 1][variable],
                        )
                        for variable in range(num_vars)
                    ]
                    right_face_values[center], left_face_values[center] = zip(*faces)
                minus_face = (right_face_values[-1], left_face_values[0])
                plus_face = (right_face_values[0], left_face_values[1])
            else:
                minus_face = (stencil[-1], stencil[0])
                plus_face = (stencil[0], stencil[1])

            minus_flux = interface_flux(*minus_face, frame_indices, config.solver_config.riemann_solver)
            plus_flux = interface_flux(*plus_face, frame_indices, config.solver_config.riemann_solver)

            if has_mask:
                # The faces bordering a flagged cell take donor-cell LLF fluxes.
                mask_minus = load(mask_ref, 0, axis, -1)
                mask_center = load(mask_ref, 0, axis, 0)
                mask_plus = load(mask_ref, 0, axis, 1)
                correct_minus = (mask_minus > 0.5) | (mask_center > 0.5)
                correct_plus = (mask_center > 0.5) | (mask_plus > 0.5)
                minus_first_order = interface_flux(
                    stencil[-1],
                    stencil[0],
                    frame_indices,
                    LAX_FRIEDRICHS,
                )
                plus_first_order = interface_flux(
                    stencil[0],
                    stencil[1],
                    frame_indices,
                    LAX_FRIEDRICHS,
                )
                minus_flux = [
                    jnp.where(correct_minus, first_order, high_order)
                    for first_order, high_order in zip(minus_first_order, minus_flux)
                ]
                plus_flux = [
                    jnp.where(correct_plus, first_order, high_order)
                    for first_order, high_order in zip(plus_first_order, plus_flux)
                ]

            for variable in range(num_vars):
                axis_difference = plus_flux[variable] - minus_flux[variable]
                if flux_difference[variable] is None:
                    flux_difference[variable] = axis_difference
                else:
                    flux_difference[variable] = flux_difference[variable] + axis_difference

        # --------------- ↑ Flux divergence ↑ ----------------

        # --------------- ↓ Conserved update and sources ↓ ----------------

        # The base state is read at the cell itself (offset 0).
        base_primitive = [
            load(base_ref, variable, axis=0, offset=0) for variable in range(num_vars)
        ]
        base_conserved = _conserved_components_from_primitive(
            base_primitive,
            gamma_value,
            config,
            registered_variables,
        )
        conserved = [
            base_conserved[variable]
            - stage_time_step_value * (flux_difference[variable] / config.grid_spacing)
            for variable in range(num_vars)
        ]

        if report_failures:
            codes = _positivity_failure_code(conserved, config, registered_variables)
            store(codes_ref, 0, codes.astype(dtype))

        if mhd:
            psi_index = registered_variables.magnetic_psi_index
            if config.solver_config.glm_extended_source:
                # The extended Dedner source from central differences of the
                # stage state, as in the native ``_dedner_source``. Offset 0
                # of any axis' stencil is the cell itself.
                magnetic_indices = tuple(registered_variables.magnetic_index)
                momentum_indices = tuple(registered_variables.momentum_index)
                stage_center_values = stencil_per_axis[0][0]
                field_components = [stage_center_values[index] for index in magnetic_indices]
                field_differences = [
                    stencil_per_axis[axis][1][magnetic_indices[axis]]
                    - stencil_per_axis[axis][-1][magnetic_indices[axis]]
                    for axis in range(ndim)
                ]
                psi_differences = [
                    stencil_per_axis[axis][1][psi_index] - stencil_per_axis[axis][-1][psi_index]
                    for axis in range(ndim)
                ]
                magnetic_divergence, field_dot_psi_gradient = _extended_dedner_source_terms(
                    field_components,
                    field_differences,
                    psi_differences,
                    config.grid_spacing,
                )
                for momentum_index, field_component in zip(momentum_indices, field_components):
                    conserved[momentum_index] = (
                        conserved[momentum_index]
                        - stage_time_step_value * magnetic_divergence * field_component
                    )
                energy_index = registered_variables.energy_index
                conserved[energy_index] = (
                    conserved[energy_index] - stage_time_step_value * field_dot_psi_gradient
                )
            conserved[psi_index] = conserved[psi_index] * damping_factor_value

        primitive = _primitive_components_from_conserved(
            conserved,
            gamma_value,
            minimum_density,
            minimum_pressure,
            config,
            registered_variables,
        )

        # --------------- ↑ Conserved update and sources ↑ ----------------

        # --------------- ↓ Output ↓ ----------------

        for variable in range(num_vars):
            store(output_ref, variable, primitive[variable])

        # --------------- ↑ Output ↑ ----------------

    # -------------------------------------------------------------
    # ======================== ↑ Kernel ↑ =========================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ====================== ↓ Kernel call ↓ ======================
    # -------------------------------------------------------------

    state_arguments = [stage_primitive_state]
    in_specs = [whole_array_spec(stage_primitive_state)]
    if has_base:
        state_arguments.append(base_primitive_state)
        in_specs.append(whole_array_spec(base_primitive_state))
    if has_mask:
        state_arguments.append(correction_mask)
        in_specs.append(whole_array_spec(correction_mask))
    if report_failures and ghost_cells:
        # Zero-initialise the codes so that the ghost cells, which the kernel
        # never writes, read as POSITIVE.
        state_arguments.append(jnp.zeros((1,) + spatial_shape, dtype=dtype))
        in_specs.append(whole_array_spec(state_arguments[-1]))

    scalar_arguments = [
        jnp.asarray(value, dtype=dtype)
        for value in (
            stage_time_step,
            cleaning_speed,
            damping_factor,
            gamma,
            density_floor,
            pressure_floor,
        )
    ]
    in_specs += [scalar_spec] * len(scalar_arguments)

    out_shape = [jax.ShapeDtypeStruct(stage_primitive_state.shape, dtype)]
    out_specs = [cell_block_spec(num_vars)]
    if report_failures:
        out_shape.append(jax.ShapeDtypeStruct((1,) + spatial_shape, dtype))
        out_specs.append(cell_block_spec(1))

    input_output_aliases = {}
    if has_base:
        # The base state is only read at the cell being written, so the output
        # can reuse its buffer (in the ghost-cell layout its ghost cells keep
        # the old values until the boundary handler overwrites them).
        base_state_input_position = 1
        primitive_output_position = 0
        input_output_aliases[base_state_input_position] = primitive_output_position
    if report_failures and ghost_cells:
        initial_codes_input_position = len(state_arguments) - 1
        failure_codes_output_position = 1
        input_output_aliases[initial_codes_input_position] = failure_codes_output_position

    keyword_arguments = {}
    compiler_params = _pallas_compiler_params(config)
    if compiler_params is not None:
        keyword_arguments["compiler_params"] = compiler_params

    stage_name = "corrector" if has_base else "predictor"
    reconstruction_name = "piecewise_linear" if piecewise_linear else "donor_cell"
    kernel_name = f"vl2_{stage_name}_{reconstruction_name}"
    if has_mask:
        kernel_name += "_flux_correction"

    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(out_shape) if report_failures else out_shape[0],
        grid=grid,
        in_specs=in_specs,
        out_specs=tuple(out_specs) if report_failures else out_specs[0],
        input_output_aliases=input_output_aliases,
        interpret=config.backend_config.pallas_interpret,
        name=kernel_name,
        **keyword_arguments,
    )(*state_arguments, *scalar_arguments)

    # -------------------------------------------------------------
    # ====================== ↑ Kernel call ↑ ======================
    # -------------------------------------------------------------

    return outputs


# -------------------------------------------------------------
# ===================== ↑ Fused stage kernel ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Stage entry points ↓ ==================
# -------------------------------------------------------------


def _vl2_stage_pallas_once(
    stage_primitive_state,
    base_primitive_state,
    correction_mask,
    stage_time_step,
    cleaning_speed,
    damping_factor,
    gamma,
    piecewise_linear: bool,
    report_failures: bool,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    One evaluation of the stage kernel, multi-device aware and differentiable.

    The kernel runs through ``_pallas_call_sharded`` so that on a device mesh
    every shard receives a periodic halo of the stencil reach instead of an
    all-gather. Derivatives are taken through the native stage
    (``diffable_pallas_call_n``), which evaluates the same expressions.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_primitive_state: ``W^n``, or ``None`` when it is the stage state
            (the predictor stage).
        correction_mask: ``None``, or a ``(1, *grid)`` array that is one in the
            cells whose faces take first-order LLF fluxes.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        cleaning_speed: The GLM cleaning speed (zero for hydrodynamics).
        damping_factor: The stage's psi damping factor.
        gamma: The adiabatic index.
        piecewise_linear: PLM (else donor-cell) reconstruction.
        report_failures: Also return the positivity failure codes.
        config: The simulation configuration.
        params: The simulation parameters (floors).
        registered_variables: The registered variables.

    Returns:
        The primitive state after the stage (ghost cells not updated), and with
        ``report_failures`` the ``(1, *grid)`` failure codes.
    """
    ndim = int(config.dimensionality)
    block_shape = _vl2_block_shape(stage_primitive_state, config)
    halo = (2 if piecewise_linear else 1,) * ndim
    has_base = base_primitive_state is not None
    has_mask = correction_mask is not None
    dtype = stage_primitive_state.dtype
    placeholder = jnp.zeros((), dtype=dtype)

    # Everything traced enters as a primal (closing over traced values inside a
    # custom_jvp is not allowed); config and the flags are static.
    def pallas_branch(
        stage_state,
        base_state,
        mask,
        stage_time_step_value,
        cleaning_speed_value,
        damping_factor_value,
        gamma_value,
        simulation_params,
    ):
        """The stage on the Pallas backend (sharded over the device mesh)."""
        state_inputs = (
            [stage_state] + ([base_state] if has_base else []) + ([mask] if has_mask else [])
        )

        def local_build(*local_states):
            """The kernel on one halo-padded shard."""
            local_states = list(local_states)
            local_stage = local_states.pop(0)
            local_base = local_states.pop(0) if has_base else None
            local_mask = local_states.pop(0) if has_mask else None
            return _vl2_stage_pallas_local(
                local_stage,
                local_base,
                local_mask,
                stage_time_step_value,
                cleaning_speed_value,
                damping_factor_value,
                gamma_value,
                simulation_params.minimum_density,
                simulation_params.minimum_pressure,
                piecewise_linear=piecewise_linear,
                report_failures=report_failures,
                config=config,
                registered_variables=registered_variables,
            )

        # The kernel reads the base state U^n only in the cell it updates, so
        # its halo is never used: on a device mesh only the stage state (and
        # the correction mask, read across faces) exchange their neighbours'
        # cells, which saves a third of the corrector's communication.
        input_halos = (
            [halo]
            + ([(0,) * ndim] if has_base else [])
            + ([halo] if has_mask else [])
        )
        return _pallas_call_sharded(
            local_build,
            state_inputs=tuple(state_inputs),
            halo=halo,
            block_shape=block_shape[:ndim],
            num_state_outputs=2 if report_failures else 1,
            input_halos=tuple(input_halos),
        )

    def native_branch(
        stage_state,
        base_state,
        mask,
        stage_time_step_value,
        cleaning_speed_value,
        damping_factor_value,
        gamma_value,
        simulation_params,
    ):
        """The same stage in native JAX, whose tangent serves as the Pallas tangent."""
        # Imported here because the integrator module imports this module at
        # its top.
        from astronomix._finite_volume._state_evolution._van_leer_integrator import (
            _native_stage_from_primitive_base,
        )

        if has_mask:
            native_mask = mask
        elif report_failures:
            # Failures are reported before any correction: an all-zero mask
            # corrects nothing but makes the native stage return the failure
            # codes as well, matching the Pallas outputs.
            native_mask = jnp.zeros((1,) + stage_state.shape[1:], dtype)
        else:
            native_mask = None
        return _native_stage_from_primitive_base(
            stage_state,
            base_state if has_base else stage_state,
            stage_time_step_value,
            piecewise_linear,
            cleaning_speed_value,
            damping_factor_value,
            gamma_value,
            config,
            simulation_params,
            registered_variables,
            correction_mask=native_mask,
        )

    return diffable_pallas_call_n(
        (
            stage_primitive_state,
            base_primitive_state if has_base else placeholder,
            correction_mask if has_mask else placeholder,
            jnp.asarray(stage_time_step, dtype=dtype),
            jnp.asarray(cleaning_speed, dtype=dtype),
            jnp.asarray(damping_factor, dtype=dtype),
            jnp.asarray(gamma, dtype=dtype),
            params,
        ),
        pallas_branch=pallas_branch,
        native_branch=native_branch,
    )


def _vl2_stage_pallas(
    stage_primitive_state,
    base_primitive_state,
    stage_time_step,
    piecewise_linear: bool,
    cleaning_speed,
    damping_factor,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    One VL2 stage on the Pallas backend, including AthenaPK's first-order flux
    correction when it is configured.

    The flux correction mirrors AthenaPK's attempts: after an evaluation, the
    cells whose update lost positivity are added to the correction mask and
    the stage is re-evaluated with first-order fluxes around them (in the last
    attempt only density failures are added). An attempt that flags no new
    cell is skipped, so a stage without positivity problems costs one kernel
    evaluation and a reduction.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_primitive_state: ``W^n`` (``None`` in the predictor stage, where it
            is the stage state).
        stage_time_step: The stage's time-step weight ``beta * dt``.
        piecewise_linear: PLM (else donor-cell) reconstruction.
        cleaning_speed: The GLM cleaning speed (zero for hydrodynamics).
        damping_factor: The stage's psi damping factor.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The primitive state after the stage (ghost cells not updated).
    """
    # Imported here because the integrator module imports this module at its
    # top.
    from astronomix._finite_volume._state_evolution._van_leer_integrator import (
        NUM_FLUX_CORRECTION_ATTEMPTS,
        _newly_flagged_cells,
    )

    def evaluate(correction_mask, report_failures):
        """One kernel evaluation with the given correction mask."""
        return _vl2_stage_pallas_once(
            stage_primitive_state,
            base_primitive_state,
            correction_mask,
            stage_time_step,
            cleaning_speed,
            damping_factor,
            gamma,
            piecewise_linear,
            report_failures,
            config,
            params,
            registered_variables,
        )

    if not config.solver_config.first_order_flux_correction:
        return evaluate(None, False)

    primitive_state, failure_codes = evaluate(None, True)
    correction_mask = jnp.zeros_like(failure_codes)
    for attempt in range(NUM_FLUX_CORRECTION_ATTEMPTS):
        new_mask = jnp.where(_newly_flagged_cells(failure_codes, attempt), 1.0, correction_mask)
        # An attempt that flags no new cell keeps the previous evaluation.
        primitive_state, failure_codes = jax.lax.cond(
            jnp.any(new_mask != correction_mask),
            lambda mask: evaluate(mask, True),
            lambda mask: (primitive_state, failure_codes),
            new_mask,
        )
        correction_mask = new_mask
    return primitive_state


# -------------------------------------------------------------
# =================== ↑ Stage entry points ↑ ==================
# -------------------------------------------------------------
