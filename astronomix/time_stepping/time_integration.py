"""
Time integration of the fluid equations.

This module wires together everything needed to advance a primitive state in
time: it prepares the helper data and sharding, optionally compiles for memory
analysis or runtime debugging, and then drives the per-step update through the
generic loop driver (fixed-step / adaptive while / checkpointed). The snapshot
machinery that records diagnostics along the way also lives here. For the
available options see the simulation configuration and the simulation parameters.
"""

# general
from contextlib import nullcontext

# typing
from typing import Any, NamedTuple, Union
from types import NoneType

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec
from jax.experimental import checkify

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    FORWARDS,
    GHOST_CELLS,
    ON_DEVICE,
    PERIODIC_ROLL,
    STATE_TYPE,
    TO_DISK
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.data_classes.simulation_snapshot_data import SnapshotData

# astronomix functions
from astronomix._finite_volume._state_evolution.evolve_state import _evolve_state_fv
from astronomix._finite_difference._state_evolution._evolve_state import _evolve_state_fd
from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
    _cfl_time_step,
    _source_term_aware_time_step,
)
from astronomix._finite_difference._timestep_estimation._timestep_estimator import (
    _cfl_time_step_fd,
    _cfl_time_step_fd_hydro
)
from astronomix._modules._iteration_level_updates import _iteration_level_updates
from astronomix._modules._turbulent_forcing._turbulent_forcing import _init_ou_forcing_state
from astronomix._snapshotting._snapshot_diagnostics import (
    build_snapshot_store,
    record_snapshot,
)
from astronomix.time_stepping._utils import _pad, _unpad
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements,
    _unpad_helper_data,
    get_helper_data,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._pallas_helpers import pallas_mesh_context

# progress bar
from astronomix.time_stepping._progress_bar import _show_progress

# generic time-integration loop driver
from astronomix.time_stepping._time_loop import (
    ADAPTIVE_CHECKPOINTED,
    ADAPTIVE_WHILE,
    FIXED_STEP,
    SnapshotSpec,
    integrate,
    times_close,
)

# timing
from timeit import default_timer as timer


class LoopState(NamedTuple):
    """The physics evolving-state threaded through the generic time-loop driver.

    The driver (``astronomix.time_stepping._time_loop.integrate``) treats this
    as an opaque pytree; only the closures here unpack it. Holding the PRNG key
    and the persistent Ornstein-Uhlenbeck forcing field here (rather than
    overloading a bare tuple slot) keeps the carry explicit and extensible.

    Attributes:
        primitive_state: The (padded) primitive fluid state being evolved.
        key: The PRNG key advanced by stochastic per-step modules (forcing, ...).
        forcing: The persistent OU forcing field ``f`` (shape (3, nx, ny, nz)),
            or ``None`` when OU forcing is inactive.
    """
    primitive_state: Any
    key: Any
    forcing: Any = None


# @jaxtyped(typechecker=typechecker)
def time_integration(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    snapshot_callable = None,
    sharding: Union[NoneType, jax.NamedSharding] = None,
    restart_state: Union[NoneType, "LoopState"] = None,
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
        restart_state: An optional :class:`LoopState` providing the PRNG key
            and persistent OU forcing field to resume from (typically obtained
            from ``astronomix.setup_helpers.restart_from_latest_checkpoint``).
            Its ``primitive_state`` slot is ignored; pass the restored state as
            the ``primitive_state`` argument. Mainly used together with
            ``snapshot_storage_mode == TO_DISK`` and ``params.t_start``.

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

    # Disk-checkpointing mode is driven on the host: it runs JIT'd segments
    # between snapshot times and streams each segment's loop carry to disk via
    # Orbax (sharding preserved per device). It reuses the helper data and the
    # promoted params built above.
    if config.snapshot_storage_mode == TO_DISK:
        return _time_integration_to_disk(
            primitive_state,
            config,
            params,
            registered_variables,
            helper_data_pad,
            snapshot_callable,
            sharding,
            restart_state,
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


def _prepare_padded_state(primitive_state, config, params, registered_variables):
    """Pad the primitive state with ghost cells and fill them via the boundary
    handler, exactly as a cold start does.

    The boundary handler only writes ghost cells (a deterministic function of
    the interior and ``params``), so re-running it on an unpadded state restored
    from disk reproduces the ghost cells the loop would have held — keeping a
    disk restart consistent with an uninterrupted run.
    """
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

    return primitive_state


def _seed_key_and_forcing(config, params):
    """Seed the PRNG key and (when OU forcing is active) the persistent forcing
    field from ``config.random_seed`` for a fresh run."""
    key0 = jax.random.key(config.random_seed)
    if (config.turbulent_forcing_config.turbulent_forcing
            and config.turbulent_forcing_config.ou_forcing):
        key0, forcing0 = _init_ou_forcing_state(
            key0, config, params.turbulent_forcing_params
        )
    else:
        forcing0 = None
    return key0, forcing0


def _build_initial_loop_state(primitive_state, config, params, restart_state=None):
    """Construct the initial loop carry for an (already padded) state.

    When ``restart_state`` is given, its PRNG key and persistent OU forcing
    field are reused so a resumed run continues the same stochastic realisation;
    otherwise they are seeded from ``config.random_seed``. The OU forcing (when
    active) needs a persistent solenoidal field; otherwise the forcing slot
    stays ``None`` and costs nothing in the carry.
    """
    if restart_state is not None:
        return LoopState(primitive_state, restart_state.key, restart_state.forcing)

    key0, forcing0 = _seed_key_and_forcing(config, params)
    return LoopState(primitive_state, key0, forcing0)


def _integrate_core(
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: Union[HelperData, NoneType],
    snapshot_callable,
    initial_loop_state: "LoopState",
    original_shape,
):
    """Drive the time loop on an already-padded initial carry.

    Shared by the in-memory return path (:func:`_time_integration`) and the
    disk-checkpointing segment runner (:func:`_run_segment`). Returns
    ``(t_final, loop_state, snapshot_store, num_iterations)``; ``snapshot_store``
    is ``None`` when no snapshots are collected.

    Args:
        original_shape: The unpadded state shape, used to size the on-device
            snapshot buffers (ignored when snapshots are disabled).
    """

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
            dt_max=jnp.minimum(
                params.dt_max,
                (params.t_end - params.t_start) / config.num_snapshots,
            )
        )

    if config.return_snapshots:
        snapshot_data = build_snapshot_store(
            config, config.num_snapshots, original_shape
        )
    elif config.activate_snapshot_callback:
        snapshot_data = SnapshotData(current_checkpoint=0)

    # -------------------------------------------------------------
    # =============== ↑ Setup of the snapshot array ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ step / record closures ↓ =================
    # -------------------------------------------------------------

    # The physics-specific pieces handed to the generic loop driver
    # (``astronomix.time_stepping._time_loop.integrate``): a ``step`` that
    # advances the state by one (adaptive) timestep, and — when snapshots are
    # requested — a recorder plus the predicate that decides when to record.

    def _step(time, state, snapshot_index):
        """Advance the state by one timestep.

        Estimates ``dt``, clamps it to land on the next snapshot time / the
        end time, runs the per-step modules and evolves the state.  Returns
        ``(dt, new_state)``; the driver advances the time.
        """
        primitive_state = state.primitive_state
        key = state.key
        forcing = state.forcing

        # determine the time step size
        if not config.fixed_timestep:
            if config.solver_mode == FINITE_VOLUME:
                if config.source_term_aware_timestep:
                    dt = jax.lax.stop_gradient(
                        _source_term_aware_time_step(
                            primitive_state, config, params, helper_data_pad,
                            registered_variables, time,
                        )
                    )
                else:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step(
                            primitive_state, config, params, registered_variables,
                        )
                    )
            elif config.solver_mode == FINITE_DIFFERENCE:
                if config.mhd:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step_fd(
                            primitive_state, config.grid_spacing, params.dt_max,
                            params.gamma, config, params, registered_variables,
                            params.C_cfl,
                        )
                    )
                else:
                    dt = jax.lax.stop_gradient(
                        _cfl_time_step_fd_hydro(
                            primitive_state, config.grid_spacing, params.dt_max,
                            params.gamma, config, params, registered_variables,
                            params.C_cfl,
                        )
                    )
        else:
            dt = params.t_end / config.num_timesteps

        # make sure we exactly hit the snapshot time points
        if config.use_specific_snapshot_timepoints and (
            config.return_snapshots or config.activate_snapshot_callback
        ):
            dt = jnp.minimum(
                dt, params.snapshot_timepoints[snapshot_index] - time
            )

        # make sure we exactly hit the end time
        if config.exact_end_time and not config.use_specific_snapshot_timepoints:
            dt = jnp.minimum(dt, params.t_end - time)

        # modules that run every time step
        key, forcing, primitive_state = _iteration_level_updates(
            primitive_state, key, forcing, dt, config, params, helper_data_pad,
            registered_variables, time + dt,
        )

        # evolve the state
        if config.solver_mode == FINITE_VOLUME:
            primitive_state = _evolve_state_fv(
                primitive_state, dt, params.gamma, config, params,
                helper_data_pad, registered_variables,
            )
        elif config.solver_mode == FINITE_DIFFERENCE:
            primitive_state = _evolve_state_fd(
                primitive_state, dt, params.gamma, config, params,
                helper_data_pad, registered_variables,
            )

        return dt, LoopState(primitive_state, key, forcing)

    def _record_snapshot(time, state, store, idx):
        """Record snapshot ``idx`` (the requested diagnostics)."""
        primitive_state = state.primitive_state

        if config.boundary_handling != PERIODIC_ROLL:
            unpad_primitive_state = _unpad(primitive_state, config)
        else:
            unpad_primitive_state = primitive_state

        # Recover the unpadded helper data by slicing — free under jit.
        helper_data_unpad = _unpad_helper_data(helper_data_pad, config)

        return record_snapshot(
            store,
            idx,
            time,
            unpad_primitive_state,
            helper_data_unpad,
            params,
            config,
            registered_variables,
        )

    def _should_record_snapshot(time, idx):
        """Whether snapshot ``idx`` is due at the start of a step at ``time``."""
        if config.use_specific_snapshot_timepoints:
            return times_close(time, params.snapshot_timepoints[idx])
        return time >= params.t_start + idx * (
            params.t_end - params.t_start
        ) / config.num_snapshots

    def _record_callback(time, state, store, _idx):
        """Snapshot recorder for ``activate_snapshot_callback``: invoke the
        user callable; no preallocated buffers are written.

        NOTE: to pass data to the host, the callable must use
        ``jax.debug.callback`` internally, and should only pass the slice /
        summary statistics actually needed to avoid moving large arrays.
        """
        primitive_state = state.primitive_state
        snapshot_callable(time, primitive_state, registered_variables)
        return store

    def _should_record_callback(time, idx):
        return time >= params.t_start + idx * (
            params.t_end - params.t_start
        ) / config.num_snapshots

    # -------------------------------------------------------------
    # ================ ↑ step / record closures ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ loop-level logic ↓ ====================
    # -------------------------------------------------------------

    # Assemble the snapshot collection (when requested) and pick the loop
    # backend, then hand it all to the generic time-loop driver.

    if config.return_snapshots:
        snapshot_spec = SnapshotSpec(
            store=snapshot_data,
            record=_record_snapshot,
            should_record=_should_record_snapshot,
            record_final=True,
            # Reserve the last buffer slot for the true final state at t_end, so
            # it is always written (the evenly spaced grid never lands on t_end).
            final_index=config.num_snapshots - 1,
        )
    elif config.activate_snapshot_callback:
        snapshot_spec = SnapshotSpec(
            store=snapshot_data,
            record=_record_callback,
            should_record=_should_record_callback,
            record_final=True,
        )
    else:
        snapshot_spec = None

    # Fixed-step runs use a plain fori_loop; adaptive runs use a while loop,
    # checkpointed for reverse-mode differentiability.
    if config.fixed_timestep:
        backend = FIXED_STEP
        num_steps = config.num_timesteps
        num_checkpoints = None
    elif config.differentiation_mode == BACKWARDS:
        backend = ADAPTIVE_CHECKPOINTED
        num_steps = None
        num_checkpoints = config.num_checkpoints
    elif config.differentiation_mode == FORWARDS:
        backend = ADAPTIVE_WHILE
        num_steps = None
        num_checkpoints = None
    else:
        raise ValueError("Unknown differentiation mode.")

    t_final, loop_state, snapshot_store, num_iterations = integrate(
        initial_loop_state,
        _step,
        params.t_end,
        backend=backend,
        t_start=params.t_start,
        num_steps=num_steps,
        num_checkpoints=num_checkpoints,
        snapshots=snapshot_spec,
        progress=_show_progress if config.progress_bar else None,
    )

    # -------------------------------------------------------------
    # =================== ↑ loop-level logic ↑ ====================
    # -------------------------------------------------------------

    return t_final, loop_state, snapshot_store, num_iterations


def _time_integration(
    state: Union[STATE_TYPE, StateStruct],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: Union[HelperData, NoneType],
    snapshot_callable = None,
    initial_loop_state: Union["LoopState", NoneType] = None,
) -> Union[STATE_TYPE, StateStruct, SnapshotData]:
    """
    Time integration.

    Args:
        state: The primitive state array (or state struct).
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data_pad: The padded helper data.
        initial_loop_state: An optional explicit initial loop carry (used to
            resume a run); seeded from ``config.random_seed`` when ``None``.

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

    # we must pad the state with ghost cells to account for the
    # boundary conditions (unless they are enforced by rolling)
    original_shape = primitive_state.shape
    primitive_state = _prepare_padded_state(
        primitive_state, config, params, registered_variables
    )

    if initial_loop_state is None:
        initial_loop_state = _build_initial_loop_state(
            primitive_state, config, params
        )

    _, loop_state, snapshot_store, num_iterations = _integrate_core(
        config,
        params,
        registered_variables,
        helper_data_pad,
        snapshot_callable,
        initial_loop_state,
        original_shape,
    )

    primitive_state = loop_state.primitive_state

    # -------------------------------------------------------------
    # ===================== ↓ return logic ↓ ======================
    # -------------------------------------------------------------

    # Finally, we need to unpack the results from the loops and
    # return them in the appropriate format.

    if config.return_snapshots:
        snapshot_data = snapshot_store._replace(num_iterations=num_iterations)
        if config.snapshot_settings.return_final_state:
            if config.boundary_handling != PERIODIC_ROLL:
                unpad_primitive_state = _unpad(primitive_state, config)
            else:
                unpad_primitive_state = primitive_state
            snapshot_data = snapshot_data._replace(final_state=unpad_primitive_state)
        return snapshot_data

    # No-snapshot path (also the snapshot-callback case): return the state.
    if config.boundary_handling != PERIODIC_ROLL:
        primitive_state = _unpad(primitive_state, config)

    if config.state_struct:
        return StateStruct(primitive_state=primitive_state)

    return primitive_state

    # -------------------------------------------------------------
    # ===================== ↑ return logic ↑ ======================
    # -------------------------------------------------------------


def _run_segment(
    primitive_state,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: Union[HelperData, NoneType],
    init_key,
    init_forcing,
):
    """Integrate one segment for the disk-checkpointing (TO_DISK) driver.

    Takes and returns the *unpadded* carry: it pads + fills boundaries on entry
    and unpads on exit, exactly as a cold start does. Because the boundary
    handler is a deterministic function of the interior, every segment boundary
    (whether reached in-memory or via a disk restart) regenerates identical
    ghost cells — so a run resumed from disk is bit-identical to the same run
    left uninterrupted. ``config`` always has snapshots disabled here (each
    segment end is itself a checkpoint), so no snapshot buffers are allocated.

    Returns ``(t_final, primitive_state_unpadded, key, forcing, num_iterations)``.
    """
    original_shape = primitive_state.shape
    primitive_state = _prepare_padded_state(
        primitive_state, config, params, registered_variables
    )
    initial_loop_state = LoopState(primitive_state, init_key, init_forcing)
    t_final, loop_state, _, num_iterations = _integrate_core(
        config,
        params,
        registered_variables,
        helper_data_pad,
        None,
        initial_loop_state,
        original_shape,
    )
    primitive_state = loop_state.primitive_state
    if config.boundary_handling != PERIODIC_ROLL:
        primitive_state = _unpad(primitive_state, config)
    return (
        t_final,
        primitive_state,
        loop_state.key,
        loop_state.forcing,
        num_iterations,
    )


def _time_integration_to_disk(
    state,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data_pad: Union[HelperData, NoneType],
    snapshot_callable,
    sharding: Union[NoneType, jax.NamedSharding],
    restart_state: Union["LoopState", NoneType],
):
    """Host-driven disk-checkpointing integration (``snapshot_storage_mode ==
    TO_DISK``).

    The run is split into ``config.num_snapshots`` segments spanning
    ``[params.t_start, params.t_end]``. Each segment is integrated by the JIT'd
    :func:`_run_segment`; afterwards the loop carry (primitive state, PRNG key,
    OU forcing) plus the time and cumulative iteration count are written to
    ``config.snapshot_storage_path`` via Orbax. The carry stays on device
    (sharded) between segments, and Orbax writes each device's shard
    independently, so this scales to multiple devices / nodes.

    Returns the final (unpadded) primitive state, like the no-snapshot path of
    :func:`_time_integration`; the per-snapshot data lives on disk.
    """
    # local import keeps the (optional) orbax dependency out of the import path
    # for runs that never touch disk checkpointing.
    from astronomix._snapshotting._orbax_storage import (
        latest_step,
        loop_checkpointer,
        save_loop_checkpoint,
    )

    if config.state_struct:
        primitive_state = state.primitive_state
    else:
        primitive_state = state

    # The carry between segments is the unpadded state plus the stochastic
    # bits (PRNG key, OU forcing). On a restart these come from the checkpoint.
    if restart_state is not None:
        key, forcing = restart_state.key, restart_state.forcing
    else:
        key, forcing = _seed_key_and_forcing(config, params)

    # Segment config: snapshots stay off (each segment end *is* a checkpoint)
    # and ON_DEVICE so the segment runner does not recurse into this driver.
    segment_config = config._replace(
        return_snapshots=False,
        activate_snapshot_callback=False,
        snapshot_storage_mode=ON_DEVICE,
        progress_bar=False,
    )

    # snapshot times spanning [t_start, t_end] (host-side floats)
    t0 = float(params.t_start)
    t1 = float(params.t_end)
    n = config.num_snapshots
    times = [t0 + (t1 - t0) * k / n for k in range(n + 1)]

    run_segment_jit = jax.jit(
        _run_segment,
        static_argnames=["config", "registered_variables"],
    )

    mesh_ctx = sharding.mesh if sharding is not None else nullcontext()
    pallas_mesh = sharding.mesh if sharding is not None else None
    replicated = (
        jax.NamedSharding(sharding.mesh, PartitionSpec())
        if sharding is not None
        else None
    )

    # Continue step numbering after any checkpoints already in the output
    # directory, so resuming into the same path extends one run history
    # (fresh directory -> starts at step 1).
    start_step = latest_step(config.snapshot_storage_path) or 0

    cumulative_iterations = 0
    with loop_checkpointer(config.snapshot_storage_path) as checkpointer:
        for i in range(n):
            # Canonicalise the carry onto the explicit sharding at every segment
            # boundary. A resumed run feeds in a state restored onto exactly this
            # sharding; pinning the in-memory carry the same way makes the two
            # paths use an identical device layout, so a disk restart reproduces
            # an uninterrupted run bit-for-bit (multi-device reductions are
            # otherwise sensitive to the layout XLA happens to pick).
            if sharding is not None:
                primitive_state = jax.device_put(primitive_state, sharding)

            segment_params = params._replace(t_start=times[i], t_end=times[i + 1])
            # Keep every params leaf on a concrete (replicated) sharding so pjit
            # dispatch on a multi-device mesh sees a sharding for the freshly
            # set t_start / t_end scalars too (see the note in time_integration).
            if replicated is not None:
                segment_params = jax.tree.map(
                    lambda leaf: jax.device_put(leaf, replicated), segment_params
                )

            with mesh_ctx, pallas_mesh_context(pallas_mesh):
                t_final, primitive_state, key, forcing, num_iterations = run_segment_jit(
                    primitive_state,
                    segment_config,
                    segment_params,
                    registered_variables,
                    helper_data_pad,
                    key,
                    forcing,
                )

            cumulative_iterations = cumulative_iterations + num_iterations

            # Pin the to-be-saved arrays onto the concrete device sharding. The
            # carry coming out of a jit under an active mesh can carry an
            # abstract-mesh sharding, which trips Orbax's shard-transfer path;
            # device_put onto the explicit NamedSharding makes it concrete (and
            # is a no-op data-movement-wise when already so placed).
            store_state = primitive_state
            store_forcing = forcing
            if sharding is not None:
                store_state = jax.device_put(primitive_state, sharding)
                if forcing is not None:
                    store_forcing = jax.device_put(forcing, sharding)

            save_loop_checkpoint(
                checkpointer,
                step=start_step + i + 1,
                time=t_final,
                primitive_state=store_state,
                key=key,
                forcing=store_forcing,
                num_iterations=cumulative_iterations,
            )

    if config.state_struct:
        return StateStruct(primitive_state=primitive_state)

    return primitive_state