# general
from contextlib import nullcontext
from types import NoneType
import jax
from jax.sharding import PartitionSpec
import jax.numpy as jnp
from functools import partial

from equinox.internal._loop.checkpointed import checkpointed_while_loop

# type checking
from jaxtyping import jaxtyped
from beartype import beartype as typechecker
from typing import Union

# runtime debugging
from jax.experimental import checkify

# astronomix constants
from astronomix._finite_difference._maths._differencing import _interface_field_divergence
from astronomix._finite_difference._state_evolution._evolve_state import _evolve_state_fd
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd, _cfl_time_step_fd_hydro
from astronomix._finite_volume._magnetic_update._vector_maths import divergence3D
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._pallas_helpers import pallas_mesh_context
from astronomix._physics_modules._frame_tracking._frame_tracking import _frame_tracking
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing import _apply_forcing, _apply_ou_forcing, _init_ou_forcing_state
from astronomix.analysis_helpers.energy_spectrum import _wavenumber_bins, get_kinetic_energy_spectrum, get_magnetic_energy_spectrum, get_magnetic_helicity_spectrum
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import BACKWARDS, FINITE_DIFFERENCE, FINITE_VOLUME, FORWARDS, GHOST_CELLS, IDEAL_GAS, PERIODIC_ROLL, STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_helper_data import (
    HelperData,
    _helper_data_requirements,
    _unpad_helper_data,
    get_helper_data,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.data_classes.simulation_snapshot_data import SnapshotData

# astronomix functions
from astronomix._finite_volume._state_evolution.evolve_state import _evolve_state_fv
from astronomix._physics_modules.run_physics_modules import _run_physics_modules
from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
    _cfl_time_step,
    _source_term_aware_time_step,
)
from astronomix._fluid_equations.total_quantities import (
    calculate_internal_energy,
    calculate_radial_momentum,
    calculate_total_mass,
)
from astronomix._fluid_equations.total_quantities import (
    calculate_total_energy,
    calculate_kinetic_energy,
    calculate_gravitational_energy,
)
from astronomix.time_stepping._utils import _pad, _unpad

# progress bar
from astronomix.time_stepping._progress_bar import _show_progress

# timing
from timeit import default_timer as timer


# @jaxtyped(typechecker=typechecker)
def time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    snapshot_callable = None,
    sharding: Union[NoneType, jax.NamedSharding] = None,
) -> Union[STATE_TYPE, SnapshotData]:
    """
    Integrate the fluid equations in time. For the options of
    the time integration see the simulation configuration and
    the simulation parameters.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        snapshot_callable: A callable which is called at certain time points
            if config.activate_snapshot_callback is True. The callable must
            have the signature
                callable(time: float, state: STATE_TYPE, registered_variables: RegisteredVariables) -> None
            and can be used to e.g. output the current state to disk or
            directly produce intermediate plots. Note that inside the callable,
            to pass data to memory, one must use
                jax.debug.callback(
                    function, args...
                )
            To avoid moving large amounts of data to the host, only pass
            the necessary data to the function in the jax.debug.callback call,
            e.g. only the slice or summary statistics you need.
        sharding: The sharding to use for the padded helper data. If None,
                  no sharding is applied.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots)
        either the final state of the fluid after the time
        integration of snapshots of the time evolution.

    """

    # Here we prepare everything for the actual time integration function,
    # _time_integration, which is jitted below. This includes setting up
    # runtime debugging via checkify if requested, printing the elapsed
    # time if requested, compiling the function for memory analysis if
    # requested, etc.

    # depending on the boundary handling, we might need to pad the state
    #  - for periodic boundaries implicitly enforced by only rolling arrays
    #    this is not necessary
    # Only build the helper-data fields actually consumed by the
    # active subsystems; the unpadded variant needed for snapshot
    # diagnostics is recovered by slicing the padded one inside the
    # update step (see _unpad_helper_data).
    requirements = _helper_data_requirements(config)
    helper_data_pad = get_helper_data(
        config,
        sharding,
        padded = config.boundary_handling != PERIODIC_ROLL,
        requirements = requirements,
    )

    # When the user supplies a multi-device sharding, pjit dispatch needs
    # every JIT input leaf to carry a sharding compatible with the target
    # mesh. SimulationParams has both Python-scalar fields (gamma, t_end,
    # C_cfl, ...) and size-(0,) placeholder arrays (the default
    # ``fixed_boundary_state``); JAX converts those into numpy 0-d /
    # empty arrays for the JIT call and pjit cannot infer a sharding for
    # them on a multi-device mesh, surfacing as
    # ``AttributeError: 'UnspecifiedValue' object has no attribute
    # '_addressable_device_assignment'`` at dispatch time. Promote every
    # leaf of ``params`` onto a fully-replicated NamedSharding on the
    # supplied mesh so pjit always sees a concrete sharding.
    if sharding is not None:
        replicated = jax.NamedSharding(sharding.mesh, PartitionSpec())
        params = jax.tree.map(
            lambda leaf: jax.device_put(leaf, replicated),
            params,
        )

    if config.donate_state:
        time_integration_jit = jax.jit(
            _time_integration,
            static_argnames=[
                "config",
                "registered_variables",
                "snapshot_callable"
            ],
            donate_argnames=["state"],
        )
    else:
        time_integration_jit = jax.jit(
            _time_integration,
            static_argnames=[
                "config",
                "registered_variables",
                "snapshot_callable"
            ],
        )

    if config.runtime_debugging:
        errors = (
            checkify.user_checks
            | checkify.index_checks
            | checkify.float_checks
            | checkify.nan_checks
            | checkify.div_checks
        )
        checked_integration = checkify.checkify(_time_integration, errors)

        err, final_state = checked_integration(
            primitive_state,
            config,
            params,
            registered_variables,
            helper_data_pad,
            snapshot_callable,
        )
        err.throw()

    else:
        memory_stats = None
        # Activate the user-provided mesh for every trace/compile of
        # ``_time_integration`` so any inner ``with_sharding_constraint``
        # calls (used to pin auxiliary scalar outputs to replicated
        # sharding) have a mesh to bind to.
        mesh_ctx = sharding.mesh if sharding is not None else nullcontext()
        # Multi-GPU Pallas: the Pallas kernels (WENO, divergence, positivity)
        # are opaque to GSPMD, so on a sharded input XLA would otherwise
        # all-gather the full state on every device before each
        # ``pallas_call``. ``pallas_mesh_context`` flips them into a
        # ``shard_map`` + ppermute halo-exchange shape instead, which is
        # the difference between ~0.95x and ~2x strong-scaling on FD
        # Pallas. The context only needs to be live while the JIT body is
        # traced; it is read by ``_pallas_call_sharded`` at trace time.
        pallas_mesh = sharding.mesh if sharding is not None else None
        if config.memory_analysis:
          with mesh_ctx, pallas_mesh_context(pallas_mesh):
            compiled_step = time_integration_jit.lower(
                primitive_state,
                config,
                params,
                registered_variables,
                helper_data_pad,
                snapshot_callable,
            ).compile()
            compiled_stats = compiled_step.memory_analysis()
            if compiled_stats is not None:
                # Calculate total memory usage including temporary storage,
                # arguments, and outputs (but excluding aliases)
                total = (
                    compiled_stats.temp_size_in_bytes
                    + compiled_stats.argument_size_in_bytes
                    + compiled_stats.output_size_in_bytes
                    - compiled_stats.alias_size_in_bytes
                )
                memory_stats = (
                    int(compiled_stats.temp_size_in_bytes),
                    int(compiled_stats.argument_size_in_bytes),
                    int(total),
                )
                print("=== Compiled memory usage PER DEVICE ===")
                print(
                    f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB"
                )
                print(
                    f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB"
                )
                print(f"Total size: {total / (1024**2):.2f} MB")
                print("========================================")

        if config.print_elapsed_time:
            if not config.memory_analysis:
                # compile the time integration function
                with mesh_ctx, pallas_mesh_context(pallas_mesh):
                    time_integration_jit.lower(
                        primitive_state,
                        config,
                        params,
                        registered_variables,
                        helper_data_pad,
                        snapshot_callable,
                    ).compile()

            start_time = timer()
            print("🚀 Starting simulation...")

        with mesh_ctx, pallas_mesh_context(pallas_mesh):
            final_state = time_integration_jit(
                primitive_state,
                config,
                params,
                registered_variables,
                helper_data_pad,
                snapshot_callable,
            )

        # For certain backend/size combinations (notably FD JAX at large
        # N with a multi-device mesh) pjit returns some scalar/auxiliary
        # output leaves with an ``UnspecifiedValue`` sharding. Their
        # device buffers are valid; the wrapper just never bound a
        # public Sharding, and every host-side accessor
        # (``is_fully_replicated``, ``is_fully_addressable``,
        # ``_value``) then crashes. Rebuild each such leaf as a regular
        # single-device array by going through its underlying per-device
        # buffer.
        if sharding is not None:
            from jax._src.sharding_impls import UnspecifiedValue as _Unspec

            def _force_concrete(leaf):
                if isinstance(leaf, jax.Array) and isinstance(leaf.sharding, _Unspec):
                    return jnp.asarray(leaf._arrays[0])
                return leaf

            final_state = jax.tree.map(_force_concrete, final_state)

        if config.print_elapsed_time:
            if config.return_snapshots and config.snapshot_settings.return_final_state:
                final_state.final_state.block_until_ready()
            else:
                final_state.block_until_ready()
            end_time = timer()
            print("🏁 Simulation finished!")
            print(f"⏱️ Time elapsed: {end_time - start_time:.2f} seconds")
            if config.return_snapshots:
                num_iterations = final_state.num_iterations
                print(f"🔄 Number of iterations: {num_iterations}")
                # print the time per iteration
                print(
                    f"⏱️ / 🔄 time per iteration: {(end_time - start_time) / num_iterations} seconds"
                )
                final_state = final_state._replace(runtime=end_time - start_time)

        if memory_stats is not None and config.return_snapshots:
            temp_b, arg_b, total_b = memory_stats
            final_state = final_state._replace(
                temporary_memory_bytes=temp_b,
                argument_memory_bytes=arg_b,
                total_memory_bytes=total_b,
            )

    return final_state


def _time_integration(
    state: Union[STATE_TYPE, StateStruct],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: Union[HelperData, NoneType],
    snapshot_callable = None,
) -> Union[STATE_TYPE, StateStruct, SnapshotData]:
    """
    Time integration.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.

    Returns:
        Depending on the configuration (return_snapshots, num_snapshots)
        either the final state of the fluid after the time integration
        of snapshots of the time evolution.
    """

    # in simulations, where we also follow e.g. star particles,
    # the state may be a struct containing the primitive state
    # and the star particle data
    if config.state_struct:
        primitive_state = state.primitive_state
    else:
        primitive_state = state

    # we must pad the state with ghost cells
    # pad the primitive state with two ghost cells on each side
    # to account for the periodic boundary conditions
    original_shape = primitive_state.shape

    if config.boundary_handling != PERIODIC_ROLL:
        primitive_state = _pad(primitive_state, config)

    if config.boundary_handling == GHOST_CELLS:
        # important for active boundaries influencing
        # the time step criterion for now only gas state
        if config.mhd:
            primitive_state = primitive_state.at[:-3, ...].set(
                _boundary_handler(primitive_state[:-3, ...], config, registered_variables, params)
            )
        else:
            primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)

    # -------------------------------------------------------------
    # =============== ↓ Setup of the snapshot array ↓ =============
    # -------------------------------------------------------------

    # In case the user requests the fluid state (or given
    # statistics) at certain time points (and not only a
    # final state at the end), we have to set up the arrays
    # to store this data.

    # The maximum timestep is also limited by the number of
    # snapshots we want to take.
    if config.return_snapshots:
        params = params._replace(
            dt_max=jnp.minimum(params.dt_max, params.t_end / config.num_snapshots)
        )

    if config.return_snapshots:
        time_points = jnp.zeros(config.num_snapshots)

        states = (
            jnp.zeros((config.num_snapshots, *original_shape))
            if config.snapshot_settings.return_states
            else None
        )
        total_mass = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_total_mass
            else None
        )
        total_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_total_energy
            else None
        )
        internal_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_internal_energy
            else None
        )
        kinetic_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_kinetic_energy
            else None
        )
        radial_momentum = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_radial_momentum
            else None
        )

        gravitational_energy = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_gravitational_energy
            and config.gravity
            else None
        )

        magnetic_divergence = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_magnetic_divergence
            and config.mhd
            else None
        )

        if (
            config.snapshot_settings.return_kinetic_energy_spectrum or 
            config.snapshot_settings.return_magnetic_energy_spectrum or
            config.snapshot_settings.return_helicity_spectrum
        ):
            k_idx, n_bins, k_centers = _wavenumber_bins(
                config.num_cells.x,
                config.num_cells.y,
                config.num_cells.z,
            )
            k_spectra = k_centers
        else:
            k_spectra = None

        if config.snapshot_settings.return_kinetic_energy_spectrum:
            kinetic_energy_spectrum = jnp.zeros((config.num_snapshots, n_bins))
        else:
            kinetic_energy_spectrum = None

        if config.snapshot_settings.return_magnetic_energy_spectrum:
            magnetic_energy_spectrum = jnp.zeros((config.num_snapshots, n_bins))
        else:
            magnetic_energy_spectrum = None

        if config.snapshot_settings.return_helicity_spectrum:
            helicity_spectrum = jnp.zeros((config.num_snapshots, n_bins))
        else:            
            helicity_spectrum = None

        temperature_pdf = (
            jnp.zeros((config.num_snapshots, config.snapshot_settings.num_temperature_bins))
            if config.snapshot_settings.return_temperature_pdf
            else None
        )

        current_checkpoint = 0

        snapshot_data = SnapshotData(
            time_points=time_points,
            states=states,
            total_mass=total_mass,
            total_energy=total_energy,
            internal_energy=internal_energy,
            kinetic_energy=kinetic_energy,
            gravitational_energy=gravitational_energy,
            current_checkpoint=current_checkpoint,
            radial_momentum=radial_momentum,
            magnetic_divergence=magnetic_divergence,
            k_spectra=k_spectra,
            kinetic_energy_spectrum=kinetic_energy_spectrum,
            magnetic_energy_spectrum=magnetic_energy_spectrum,
            helicity_spectrum=helicity_spectrum,
            temperature_pdf=temperature_pdf,
            final_state=None,
        )

    elif config.activate_snapshot_callback:
        current_checkpoint = 0
        snapshot_data = SnapshotData(
            time_points=None,
            states=None,
            total_mass=None,
            total_energy=None,
            current_checkpoint=current_checkpoint,
            kinetic_energy_spectrum=None,
            magnetic_energy_spectrum=None,
            helicity_spectrum=None,
            k_spectra=None,
            temperature_pdf=None,
        )

    # -------------------------------------------------------------
    # =============== ↑ Setup of the snapshot array ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ====================== ↓ Update step ↓ ======================
    # -------------------------------------------------------------

    # This is the actual update step of the data handled by the time
    # integration function. In the simplest case, this might just
    # take in the primitive state and return the updated primitive state
    # after a time step. However, the data which actually needs to be
    # updated may be more complex, e.g. the SnapshotData needs to be
    # updated appropriately if snapshots are requested.

    def update_step(carry):
        # --------------- ↓ Carry unpacking+ ↓ ----------------

        # Depending on the configuration, the carry might either contain
        #   - the time, the primitive state and the snapshot data
        #   - only the time and the primitive state

        # We need to appropriately unpack the carry and in case we
        # have snapshot data, we also directly update it here at
        # the beginning of the time step.

        # WARNING: Currently config.return_snapshots and 
        # config.activate_snapshot_callback are mutually
        # exclusive.

        if config.return_snapshots:
            # When SnapshotData is involved, we need to unpack the carry
            # correctly and update the SnapshotData if we are currently
            # at a point in time where we want to take a snapshot.

            time, key, primitive_state, snapshot_data = carry

            def update_snapshot_data(time, primitive_state, snapshot_data):
                time_points = snapshot_data.time_points.at[
                    snapshot_data.current_checkpoint
                ].set(time)

                if config.boundary_handling != PERIODIC_ROLL:
                    unpad_primitive_state = _unpad(primitive_state, config)
                else:
                    unpad_primitive_state = primitive_state

                # Recover the unpadded helper data by slicing — no
                # extra device storage, free under jit.
                helper_data_unpad = _unpad_helper_data(helper_data_pad, config)

                if config.snapshot_settings.return_states:
                    states = snapshot_data.states.at[
                        snapshot_data.current_checkpoint
                    ].set(unpad_primitive_state)
                else:
                    states = None

                if config.snapshot_settings.return_total_mass:
                    total_mass = snapshot_data.total_mass.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_total_mass(unpad_primitive_state, helper_data_unpad, config)
                    )
                else:
                    total_mass = None

                if config.snapshot_settings.return_total_energy:
                    total_energy = snapshot_data.total_energy.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_total_energy(
                            unpad_primitive_state,
                            helper_data_unpad,
                            params.gamma,
                            params.gravitational_constant,
                            params,
                            config,
                            registered_variables,
                        )
                    )
                else:
                    total_energy = None

                if config.snapshot_settings.return_internal_energy:
                    internal_energy = snapshot_data.internal_energy.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_internal_energy(
                            unpad_primitive_state,
                            helper_data_unpad,
                            params.gamma,
                            config,
                            registered_variables,
                        )
                    )
                else:
                    internal_energy = None

                if config.snapshot_settings.return_kinetic_energy:
                    kinetic_energy = snapshot_data.kinetic_energy.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_kinetic_energy(
                            unpad_primitive_state,
                            helper_data_unpad,
                            config,
                            registered_variables,
                        )
                    )
                else:
                    kinetic_energy = None

                if config.snapshot_settings.return_radial_momentum:
                    radial_momentum = snapshot_data.radial_momentum.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_radial_momentum(
                            unpad_primitive_state,
                            helper_data_unpad,
                            config,
                            registered_variables,
                        )
                    )
                else:
                    radial_momentum = None

                if (
                    config.gravity
                    and config.snapshot_settings.return_gravitational_energy
                ):
                    gravitational_energy = snapshot_data.gravitational_energy.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_gravitational_energy(
                            unpad_primitive_state,
                            helper_data_unpad,
                            params.gravitational_constant,
                            params,
                            config,
                            registered_variables,
                        )
                    )
                else:
                    gravitational_energy = None

                magnetic_divergence = snapshot_data.magnetic_divergence.at[
                    snapshot_data.current_checkpoint
                ].set(
                    jnp.max(jnp.abs(_interface_field_divergence(
                        unpad_primitive_state[registered_variables.interface_magnetic_field_index.x],
                        unpad_primitive_state[registered_variables.interface_magnetic_field_index.y],
                        unpad_primitive_state[registered_variables.interface_magnetic_field_index.z],
                        config.grid_spacing,
                    ))) if config.solver_mode == FINITE_DIFFERENCE else
                    jnp.max(jnp.abs(
                        divergence3D(
                            unpad_primitive_state[registered_variables.magnetic_index.x:registered_variables.magnetic_index.z+1],
                            config.grid_spacing,
                        )
                    ))
                ) if config.snapshot_settings.return_magnetic_divergence and config.mhd else None

                if config.snapshot_settings.return_kinetic_energy_spectrum:
                    _, kinetic_energy_spectrum_i = get_kinetic_energy_spectrum(
                        unpad_primitive_state[registered_variables.velocity_index.x],
                        unpad_primitive_state[registered_variables.velocity_index.y],
                        unpad_primitive_state[registered_variables.velocity_index.z],
                        unpad_primitive_state[registered_variables.density_index],
                    )
                    kinetic_energy_spectrum = snapshot_data.kinetic_energy_spectrum.at[
                        snapshot_data.current_checkpoint
                    ].set(kinetic_energy_spectrum_i)
                else:
                    kinetic_energy_spectrum = None
                
                if config.snapshot_settings.return_magnetic_energy_spectrum and config.mhd:
                    _, magnetic_energy_spectrum_i = get_magnetic_energy_spectrum(
                        unpad_primitive_state[registered_variables.magnetic_index.x],
                        unpad_primitive_state[registered_variables.magnetic_index.y],
                        unpad_primitive_state[registered_variables.magnetic_index.z],
                    )
                    magnetic_energy_spectrum = snapshot_data.magnetic_energy_spectrum.at[
                        snapshot_data.current_checkpoint
                    ].set(magnetic_energy_spectrum_i)
                else:
                    magnetic_energy_spectrum = None


                if config.snapshot_settings.return_helicity_spectrum and config.mhd:
                    _, helicity_spectrum_i = get_magnetic_helicity_spectrum(
                        unpad_primitive_state[registered_variables.magnetic_index.x],
                        unpad_primitive_state[registered_variables.magnetic_index.y],
                        unpad_primitive_state[registered_variables.magnetic_index.z],
                    )
                    helicity_spectrum = snapshot_data.helicity_spectrum.at[
                        snapshot_data.current_checkpoint
                    ].set(helicity_spectrum_i)
                else:
                    helicity_spectrum = None

                if config.snapshot_settings.return_temperature_pdf:
                    # calculate temperature from ideal gas law
                    # ASSUMING T = P / rho HERE!

                    # calculate the temperature
                    temperature = (
                        unpad_primitive_state[registered_variables.pressure_index] / 
                        unpad_primitive_state[registered_variables.density_index]
                    )
                    logT = jnp.log10(temperature)

                    # calculate the temperature PDF (dV/dlogT)
                    dV_dlogT, _ = jnp.histogram(
                        logT.flatten(),
                        range=(
                            jnp.log10(config.snapshot_settings.temperature_pdf_min),
                            jnp.log10(config.snapshot_settings.temperature_pdf_max)
                        ),
                        bins=config.snapshot_settings.num_temperature_bins,
                    )
                    temperature_pdf = snapshot_data.temperature_pdf.at[
                        snapshot_data.current_checkpoint
                    ].set(dV_dlogT)
                else:                    
                    temperature_pdf = None
                
                current_checkpoint = snapshot_data.current_checkpoint + 1
                snapshot_data = snapshot_data._replace(
                    time_points=time_points,
                    states=states,
                    current_checkpoint=current_checkpoint,
                    total_mass=total_mass,
                    total_energy=total_energy,
                    internal_energy=internal_energy,
                    kinetic_energy=kinetic_energy,
                    gravitational_energy=gravitational_energy,
                    radial_momentum=radial_momentum,
                    magnetic_divergence=magnetic_divergence,
                    kinetic_energy_spectrum=kinetic_energy_spectrum,
                    magnetic_energy_spectrum=magnetic_energy_spectrum,
                    helicity_spectrum=helicity_spectrum,
                    temperature_pdf=temperature_pdf
                )
                return snapshot_data

            def dont_update_snapshot_data(time, primitive_state, snapshot_data):
                return snapshot_data

            if config.use_specific_snapshot_timepoints:
                snapshot_data = jax.lax.cond(
                    jnp.abs(
                        time
                        - params.snapshot_timepoints[snapshot_data.current_checkpoint]
                    )
                    < 1e-12,
                    update_snapshot_data,
                    dont_update_snapshot_data,
                    time,
                    primitive_state,
                    snapshot_data,
                )
            else:
                snapshot_data = jax.lax.cond(
                    time
                    >= snapshot_data.current_checkpoint
                    * params.t_end
                    / config.num_snapshots,
                    update_snapshot_data,
                    dont_update_snapshot_data,
                    time,
                    primitive_state,
                    snapshot_data,
                )

            num_iterations = snapshot_data.num_iterations + 1
            snapshot_data = snapshot_data._replace(num_iterations=num_iterations)

        elif config.activate_snapshot_callback:
            # Here we deal with the case where the user passes
            # a callable which is applied at certain time points
            # - e.g. to output the current state to disk or
            # directly produce intermediate plots.

            time, key, primitive_state, snapshot_data = carry

            def update_snapshot_data(time, primitive_state, snapshot_data):
                current_checkpoint = snapshot_data.current_checkpoint + 1
                snapshot_data = snapshot_data._replace(
                    current_checkpoint=current_checkpoint
                )

                # call the user-defined snapshot callable
                # NOTE: to pass data to memory, one must use
                # jax.debug.callback(
                #     function, args...
                # )
                # inside the snapshot_callable. To avoid moving
                # large amounts of data to the host, only pass
                # the necessary data to the function in the
                # jax.debug.callback call, e.g. only the slice
                # or summary statistics you need.
                snapshot_callable(time, primitive_state, registered_variables)

                return snapshot_data

            def dont_update_snapshot_data(time, primitive_state, snapshot_data):
                return snapshot_data

            snapshot_data = jax.lax.cond(
                time
                >= snapshot_data.current_checkpoint
                * params.t_end
                / config.num_snapshots,
                update_snapshot_data,
                dont_update_snapshot_data,
                time,
                primitive_state,
                snapshot_data,
            )

            num_iterations = snapshot_data.num_iterations + 1
            snapshot_data = snapshot_data._replace(num_iterations=num_iterations)
        else:
            # This is the simplest case where we only have
            # the time and the primitive state in the carry.
            # We just unpack them accordingly.
            time, key, primitive_state = carry

        # --------------- ↑ Carry unpacking+ ↑ ----------------

        # ---------------- ↓ time step logic ↓ ----------------

        # This is the heart of the time integration function.
        # Here we determine the time step size and then evolve
        # the state and run the physics modules.

        # determine the time step size
        if not config.fixed_timestep:
            if config.solver_mode == FINITE_VOLUME:
                if config.source_term_aware_timestep:
                    dt = jax.lax.stop_gradient(
                        _source_term_aware_time_step(
                            primitive_state,
                            config,
                            params,
                            helper_data_pad,
                            registered_variables,
                            time,
                        )
                    )
                else:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step(
                            primitive_state,
                            config,
                            params,
                            registered_variables,
                        )
                    )
            elif config.solver_mode == FINITE_DIFFERENCE:
                if config.mhd:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step_fd(
                            primitive_state,
                            config.grid_spacing,
                            params.dt_max,
                            params.gamma,
                            config,
                            params,
                            registered_variables,
                            params.C_cfl,
                        )
                    )
                else:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step_fd_hydro(
                            primitive_state,
                            config.grid_spacing,
                            params.dt_max,
                            params.gamma,
                            config,
                            params,
                            registered_variables,
                            params.C_cfl,
                        )
                    )
        else:
            dt = params.t_end / config.num_timesteps

        # make sure we exactly hit the snapshot time points
        if config.use_specific_snapshot_timepoints and (config.return_snapshots or config.activate_snapshot_callback):
            dt = jnp.minimum(
                dt, params.snapshot_timepoints[snapshot_data.current_checkpoint] - time
            )

        # make sure we exactly hit the end time
        if config.exact_end_time and not config.use_specific_snapshot_timepoints:
            dt = jnp.minimum(dt, params.t_end - time)

        # ---------------- ↑ time step logic ↑ ----------------

        # ----------------- ↓ CENTRAL UPDATE ↓ ----------------

        if config.solver_mode == FINITE_VOLUME:
            # run physics modules
            # for now we mainly consider the stellar wind, a constant source term term,
            # so the source is handled via a simple Euler step but generally
            # a higher order method (in a split fashion) may be used
            primitive_state = _run_physics_modules(
                primitive_state,
                dt,
                config,
                params,
                helper_data_pad,
                registered_variables,
                time + dt,
            )

        # turbulence forcing, TODO: move to physics modules
        # NOTE: THE KEY IS CURRENTLY DIRECTLY IN THE CARRY
        # FOR THE CASE WITHOUT SNAPSHOT DATA AND NOT PRESENT
        # IN THE CARRY OTHERWISE. TODO: IMPROVE THIS.
        if config.turbulent_forcing_config.turbulent_forcing:
            if config.turbulent_forcing_config.ou_forcing:
                # the carry's "key" slot holds the OU forcing state (key, field)
                key, primitive_state = _apply_ou_forcing(
                    key,
                    primitive_state,
                    dt,
                    params.turbulent_forcing_params,
                    config,
                    registered_variables,
                )
            else:
                key, primitive_state = _apply_forcing(
                    key,
                    primitive_state,
                    dt,
                    params.turbulent_forcing_params,
                    config,
                    registered_variables,
                )

        # PRELIMINARY
        # Frame tracking, currently very specialized
        # I do not like this being here
        # CURRENTLY ONLY 3D
        if config.frame_tracking:
            primitive_state = _frame_tracking(
                primitive_state,
                config,
                params,
                registered_variables,
                helper_data_pad,
            )

        # better safe than sorry
        if config.enforce_positivity:
            primitive_state = primitive_state.at[registered_variables.density_index].set(
                jnp.maximum(
                    primitive_state[registered_variables.density_index], params.minimum_density
                )
            )
            if config.equation_of_state == IDEAL_GAS:
                primitive_state = primitive_state.at[registered_variables.pressure_index].set(
                    jnp.maximum(
                        primitive_state[registered_variables.pressure_index], params.minimum_pressure
                    )
                )

        # EVOLVE THE STATE
        if config.solver_mode == FINITE_VOLUME:
            primitive_state = _evolve_state_fv(
                primitive_state,
                dt,
                params.gamma,
                params.gravitational_constant,
                config,
                params,
                helper_data_pad,
                registered_variables,
            )
        elif config.solver_mode == FINITE_DIFFERENCE:
            primitive_state = _evolve_state_fd(
                primitive_state,
                dt,
                params.gamma,
                params.gravitational_constant,
                config,
                params,
                helper_data_pad,
                registered_variables,
            )

        time += dt

        # ----------------- ↑ CENTRAL UPDATE ↑ ----------------

        # If we are in the last time step, we also want to update the snapshot data.
        if config.return_snapshots or config.activate_snapshot_callback:
            snapshot_data = jax.lax.cond(
                jnp.abs(time - params.t_end) < 1e-12,
                update_snapshot_data,
                dont_update_snapshot_data,
                time,
                primitive_state,
                snapshot_data,
            )

        # progress bar update
        if config.progress_bar:
            jax.debug.callback(_show_progress, time, params.t_end)

        # packing the carry again
        if config.return_snapshots or config.activate_snapshot_callback:
            carry = (time, key, primitive_state, snapshot_data)
        else:
            carry = (time, key, primitive_state)

        return carry

    # -------------------------------------------------------------
    # ====================== ↑ Update step ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ loop-level logic ↓ ====================
    # -------------------------------------------------------------

    # Here we set up and start the actual time integration loops.
    # Depending on the configuration, this might be a fori loop
    # a while loop or a checkpointed while loop.

    def update_step_for(_, carry):
        return update_step(carry)

    def condition(carry):
        if config.return_snapshots or config.activate_snapshot_callback:
            t, _, _, _ = carry
        else:
            t, _, _ = carry
        return t < params.t_end

    # The carry's second slot threads the forcing RNG state. For OU forcing it
    # also holds the persistent forcing field, bundled as (key, field); it is
    # opaque everywhere except the forcing application.
    _rng_key = jax.random.key(config.random_seed)
    if (config.turbulent_forcing_config.turbulent_forcing
            and config.turbulent_forcing_config.ou_forcing):
        forcing_state = _init_ou_forcing_state(
            _rng_key, config, params.turbulent_forcing_params
        )
    else:
        forcing_state = _rng_key

    if config.return_snapshots or config.activate_snapshot_callback:
        carry = (0.0, forcing_state, primitive_state, snapshot_data)
    else:
        carry = (0.0, forcing_state, primitive_state)

    if not config.fixed_timestep:
        if config.differentiation_mode == BACKWARDS:
            carry = checkpointed_while_loop(
                condition, update_step, carry, checkpoints=config.num_checkpoints
            )
        elif config.differentiation_mode == FORWARDS:
            carry = jax.lax.while_loop(condition, update_step, carry)
        else:
            raise ValueError("Unknown differentiation mode.")
    else:
        if config.differentiation_mode == BACKWARDS:
            # Checkpointed fixed-timestep loop: bounded-memory reverse mode (a
            # plain fori_loop stores every step and OOMs through reverse mode).
            # The timestep is fixed, so a time-based stop is exact to
            # ``num_timesteps`` steps (compare against t_end - dt/2).
            _dt_fixed = params.t_end / config.num_timesteps

            def _fixed_condition(carry):
                return carry[0] < params.t_end - 0.5 * _dt_fixed

            carry = checkpointed_while_loop(
                _fixed_condition, update_step, carry,
                checkpoints=config.num_checkpoints,
            )
        else:
            carry = jax.lax.fori_loop(0, config.num_timesteps, update_step_for, carry)

    # -------------------------------------------------------------
    # =================== ↑ loop-level logic ↑ ====================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ===================== ↓ return logic ↓ ======================
    # -------------------------------------------------------------

    # Finally, we need to unpack the results from the loops and
    # return them in the appropriate format.

    if config.return_snapshots or config.activate_snapshot_callback:
        _, _, primitive_state, snapshot_data = carry

        if config.return_snapshots:
            if config.snapshot_settings.return_final_state:
                if config.boundary_handling != PERIODIC_ROLL:
                    unpad_primitive_state = _unpad(primitive_state, config)
                else:
                    unpad_primitive_state = primitive_state

                snapshot_data = snapshot_data._replace(
                    final_state=unpad_primitive_state
                )
            return snapshot_data
        else:
            if config.boundary_handling != PERIODIC_ROLL:
                primitive_state = _unpad(primitive_state, config)
            if config.state_struct:
                return StateStruct(primitive_state=primitive_state)

            return primitive_state
    else:
        _, _, primitive_state = carry

        # unpad the primitive state if we padded it
        if config.boundary_handling != PERIODIC_ROLL:
            primitive_state = _unpad(primitive_state, config)

        if config.state_struct:
            return StateStruct(primitive_state=primitive_state)

        return primitive_state

    # -------------------------------------------------------------
    # ===================== ↑ return logic ↑ ======================
    # -------------------------------------------------------------