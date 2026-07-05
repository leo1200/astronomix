"""Restart a simulation from the latest on-disk Orbax checkpoint.

Convenience around the disk-checkpointing (``snapshot_storage_mode == TO_DISK``)
path of the time integration: it reads the most recent checkpoint written to a
directory and returns everything needed to continue the run.
"""

# typing
from typing import Optional, Tuple

# jax
import jax

# astronomix containers
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping.time_integration import LoopState

# astronomix functions
from astronomix._snapshotting._orbax_storage import (
    latest_step,
    load_loop_checkpoint,
)


def restart_from_latest_checkpoint(
    directory,
    params: SimulationParams,
    *,
    step: Optional[int] = None,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> Tuple[jax.Array, SimulationParams, LoopState]:
    """Load the latest (or a specific) checkpoint and prepare a resumed run.

    Args:
        directory: The checkpoint directory used as ``snapshot_storage_path``.
        params: The simulation parameters of the run to resume. Returned with
            ``t_start`` set to the checkpoint's time so the integration picks up
            where it left off.
        step: The checkpoint step to restore. ``None`` (default) restores the
            latest one.
        sharding: Optional target sharding for the restored state / forcing
            (e.g. when resuming on a different device topology). When ``None``
            the sharding stored in the checkpoint is recovered.

    Returns:
        ``(primitive_state, params, restart_state)`` where ``primitive_state``
        is the restored (unpadded) state to pass as the first argument of
        :func:`~astronomix.time_stepping.time_integration.time_integration`,
        ``params`` has ``t_start`` set to the checkpoint time, and
        ``restart_state`` is the :class:`LoopState` carrying the PRNG key and OU
        forcing field to pass via ``restart_state=``.

    Example::

        ps, params, restart = restart_from_latest_checkpoint(
            path, params, sharding=sharding
        )
        final = time_integration(
            ps, config, params, registered_variables,
            sharding=sharding, restart_state=restart,
        )
    """
    checkpoint = load_loop_checkpoint(directory, step, sharding=sharding)

    params = params._replace(t_start=checkpoint.time)
    restart_state = LoopState(
        primitive_state=checkpoint.primitive_state,
        key=checkpoint.key,
        forcing=checkpoint.forcing,
    )
    return checkpoint.primitive_state, params, restart_state


def latest_checkpoint_step(directory) -> Optional[int]:
    """The most recent checkpoint step in ``directory`` (``None`` if empty)."""
    return latest_step(directory)
