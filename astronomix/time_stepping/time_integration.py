# general
from types import NoneType
import jax
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
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd
from astronomix._finite_volume._magnetic_update._vector_maths import divergence3D
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing import _apply_forcing
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_config import BACKWARDS, FINITE_DIFFERENCE, FINITE_VOLUME, FORWARDS, GHOST_CELLS, STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_helper_data import HelperData, get_helper_data
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

    helper_data_pad = get_helper_data(
        config,
        sharding,
        padded = config.boundary_handling == GHOST_CELLS,
        production = True
    )
    
    helper_data = get_helper_data(
        config,
        sharding,
        padded = False,
        production = True
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
            helper_data,
            helper_data_pad,
            snapshot_callable,
        )
        err.throw()

    else:
        if config.memory_analysis:
            compiled_step = time_integration_jit.lower(
                primitive_state,
                config,
                params,
                registered_variables,
                helper_data,
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
                time_integration_jit.lower(
                    primitive_state,
                    config,
                    params,
                    registered_variables,
                    helper_data,
                    helper_data_pad,
                    snapshot_callable,
                ).compile()

            start_time = timer()
            print("🚀 Starting simulation...")

        final_state = time_integration_jit(
            primitive_state,
            config,
            params,
            registered_variables,
            helper_data,
            helper_data_pad,
            snapshot_callable,
        )

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

    return final_state


def _time_integration(
    state: Union[STATE_TYPE, StateStruct],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_unpad: Union[HelperData, NoneType],
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

    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _pad(primitive_state, config)

        # important for active boundaries influencing
        # the time step criterion for now only gas state
        if config.mhd:
            primitive_state = primitive_state.at[:-3, ...].set(
                _boundary_handler(primitive_state[:-3, ...], config)
            )
        else:
            primitive_state = _boundary_handler(primitive_state, config)

    # -------------------------------------------------------------
    # =============== ↓ Setup of the snapshot array ↓ =============
    # -------------------------------------------------------------

    # In case the user requests the fluid state (or given
    # statistics) at certain time points (and not only a
    # final state at the end), we have to set up the arrays
    # to store this data.

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
            and config.self_gravity
            else None
        )

        magnetic_divergence = (
            jnp.zeros(config.num_snapshots)
            if config.snapshot_settings.return_magnetic_divergence
            and config.mhd
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

        if config.return_snapshots:
            # When SnapshotData is involved, we need to unpack the carry
            # correctly and update the SnapshotData if we are currently
            # at a point in time where we want to take a snapshot.

            time, key, primitive_state, snapshot_data = carry

            def update_snapshot_data(time, primitive_state, snapshot_data):
                time_points = snapshot_data.time_points.at[
                    snapshot_data.current_checkpoint
                ].set(time)

                if config.boundary_handling == GHOST_CELLS:
                    unpad_primitive_state = _unpad(primitive_state, config)
                else:
                    unpad_primitive_state = primitive_state

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
                    config.self_gravity
                    and config.snapshot_settings.return_gravitational_energy
                ):
                    gravitational_energy = snapshot_data.gravitational_energy.at[
                        snapshot_data.current_checkpoint
                    ].set(
                        calculate_gravitational_energy(
                            unpad_primitive_state,
                            helper_data_unpad,
                            params.gravitational_constant,
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

            def update_snapshot_data(snapshot_data):
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

            def dont_update_snapshot_data(snapshot_data):
                return snapshot_data

            snapshot_data = jax.lax.cond(
                time
                >= snapshot_data.current_checkpoint
                * params.t_end
                / config.num_snapshots,
                update_snapshot_data,
                dont_update_snapshot_data,
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
                            config.grid_spacing,
                            params.dt_max,
                            params.gamma,
                            config,
                            registered_variables,
                            params.C_cfl,
                        )
                    )
            elif config.solver_mode == FINITE_DIFFERENCE:
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
            dt = params.t_end / config.num_timesteps

        # make sure we exactly hit the snapshot time points
        if config.use_specific_snapshot_timepoints and config.return_snapshots:
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
            key, primitive_state = _apply_forcing(
                key,
                primitive_state,
                dt,
                params.turbulent_forcing_params,
                config,
                registered_variables,
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
        if config.use_specific_snapshot_timepoints and config.return_snapshots:
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

    if config.return_snapshots or config.activate_snapshot_callback:
        carry = (0.0, jax.random.key(42), primitive_state, snapshot_data)
    else:
        carry = (0.0, jax.random.key(42), primitive_state)

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
                if config.boundary_handling == GHOST_CELLS:
                    unpad_primitive_state = _unpad(primitive_state, config)
                else:
                    unpad_primitive_state = primitive_state
                
                snapshot_data = snapshot_data._replace(
                    final_state=unpad_primitive_state
                )
            return snapshot_data
        else:
            if config.boundary_handling == GHOST_CELLS:
                primitive_state = _unpad(primitive_state, config)
            if config.state_struct:
                return StateStruct(primitive_state=primitive_state)

            return primitive_state
    else:
        _, _, primitive_state = carry

        # unpad the primitive state if we padded it
        if config.boundary_handling == GHOST_CELLS:
            primitive_state = _unpad(primitive_state, config)

        if config.state_struct:
            return StateStruct(primitive_state=primitive_state)

        return primitive_state

    # -------------------------------------------------------------
    # ===================== ↑ return logic ↑ ======================
    # -------------------------------------------------------------