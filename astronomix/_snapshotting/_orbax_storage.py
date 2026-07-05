"""Orbax-backed disk storage for the time-integration loop carry.

Used by the ``snapshot_storage_mode == TO_DISK`` path of the time integration
(see :func:`astronomix.time_stepping.time_integration.time_integration`) and by
the restart helper in :mod:`astronomix.setup_helpers`.

Each on-disk checkpoint stores the loop carry threaded through the integrator
— the (unpadded) primitive state, the PRNG key and the persistent OU forcing
field (when active) — plus the current simulation time and iteration count.
This mirrors :class:`~astronomix.time_stepping.time_integration.LoopState`, so a
checkpoint is everything needed to resume a run bit-reproducibly.

Implementation notes
--------------------
We use Orbax's **synchronous** ``StandardCheckpointer`` and lay out one
numbered sub-directory per step (``<root>/<step>``). Synchronous saving is a
deliberate choice: the asynchronous path coordinates step-directory creation
through a background signalling barrier that deadlocks in a single-process,
multi-device setting. The synchronous checkpointer writes one sharded array per
device directly (scaling to multiple devices / nodes) and, crucially, restores
straight into a sharded target without the async machinery.

The PRNG key is stored as raw key data (``jax.random.key_data`` -> a plain
uint32 array tensorstore can serialise) and rebuilt with
``jax.random.wrap_key_data`` on load. The OU forcing field is only written when
present; its absence on load is reported as ``forcing = None``.
"""

# general
from contextlib import contextmanager
from pathlib import Path

# typing
from typing import Any, NamedTuple, Optional

# jax
import jax
from jax.sharding import NamedSharding, PartitionSpec

# checkpointing (optional dependency — only needed for the TO_DISK snapshot and
# restart path). The import is guarded so that ``import astronomix`` keeps
# working when orbax-checkpoint is either absent OR installed at a version that
# is incompatible with the installed JAX. The latter is a real failure mode: an
# older orbax evaluates ``jax.experimental.layout.DeviceLocalLayout`` at import
# time, which JAX 0.10 removed, raising ``AttributeError`` from inside orbax. We
# therefore catch any import-time error here, not just ModuleNotFoundError, and
# defer the requirement to the point where the feature is actually used.
try:
    import orbax.checkpoint as ocp
except Exception:  # pragma: no cover - exercised only when orbax can't import
    ocp = None


def _require_orbax():
    """Raise a clear error if orbax-checkpoint is needed but unavailable."""
    if ocp is None:
        raise ModuleNotFoundError(
            "orbax-checkpoint is required for TO_DISK snapshotting and restart, "
            "but it could not be imported — it is either not installed or its "
            "version is incompatible with the installed JAX (e.g. an older "
            "orbax under JAX >= 0.10). Install/upgrade it, e.g. "
            "`pip install -U orbax-checkpoint`."
        )


class LoopCheckpoint(NamedTuple):
    """A loaded loop checkpoint (see module docstring)."""

    #: Simulation time the checkpoint was taken at.
    time: float
    #: The (unpadded) primitive fluid state.
    primitive_state: Any
    #: The reconstructed typed PRNG key.
    key: Any
    #: The persistent OU forcing field, or ``None`` if it was not stored.
    forcing: Any
    #: Cumulative number of integration steps taken up to this checkpoint.
    num_iterations: int
    #: The step index of this checkpoint.
    step: int


class _LoopCheckpointWriter:
    """Writes loop checkpoints into numbered sub-directories of a root."""

    def __init__(self, checkpointer, directory):
        self._checkpointer = checkpointer
        self._directory = Path(directory).resolve()

    def save(self, step, *, time, primitive_state, key, forcing, num_iterations):
        """Serialise one loop carry into the ``<root>/<step>`` sub-directory.

        The PRNG key is stored as raw key data (a plain uint32 array
        tensorstore can serialise) and the OU forcing field is only written
        when present, so its absence is unambiguous on load.
        """
        tree = {
            "time": time,
            "primitive_state": primitive_state,
            "key_data": jax.random.key_data(key),
            "num_iterations": num_iterations,
        }
        if forcing is not None:
            tree["forcing"] = forcing
        # Synchronous save; ``force`` overwrites a partially written step dir.
        self._checkpointer.save(self._directory / str(step), tree, force=True)


@contextmanager
def loop_checkpointer(directory):
    """Open a synchronous loop-checkpoint writer rooted at ``directory``.

    Use as a context manager so the checkpointer is closed on exit::

        with loop_checkpointer(path) as writer:
            save_loop_checkpoint(writer, step, time=t, primitive_state=ps, ...)
    """
    _require_orbax()
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    try:
        yield _LoopCheckpointWriter(checkpointer, directory)
    finally:
        checkpointer.close()


def save_loop_checkpoint(
    writer: _LoopCheckpointWriter,
    step: int,
    *,
    time,
    primitive_state,
    key,
    forcing,
    num_iterations,
) -> None:
    """Write one loop checkpoint at ``step`` through an open ``writer``.

    Array leaves keep their device sharding, so each device writes its own
    shard.
    """
    writer.save(
        step,
        time=time,
        primitive_state=primitive_state,
        key=key,
        forcing=forcing,
        num_iterations=num_iterations,
    )


def latest_step(directory) -> Optional[int]:
    """The most recent checkpoint step in ``directory``, or ``None`` if empty."""
    root = Path(directory)
    if not root.exists():
        return None
    steps = [
        int(child.name)
        for child in root.iterdir()
        if child.is_dir() and child.name.isdigit()
    ]
    return max(steps) if steps else None


def _abstract_pytree(tree_metadata, sharding):
    """Build an Orbax restore target (``ShapeDtypeStruct`` pytree) from saved
    array metadata, pinning each leaf to a concrete sharding.

    Passing an explicit target is what lets a checkpoint written on a
    multi-device mesh restore straight into its shards. Full-rank leaves (the
    state / forcing fields) get the requested ``sharding``; scalars and the
    PRNG key data are replicated. When ``sharding`` is ``None`` everything is
    placed on the default device.
    """
    if sharding is not None:
        spec_len = len(sharding.spec)
        replicated = NamedSharding(sharding.mesh, PartitionSpec())

        def pick(leaf):
            return sharding if len(leaf.shape) == spec_len else replicated
    else:
        single = jax.sharding.SingleDeviceSharding(jax.devices()[0])

        def pick(leaf):
            return single

    def to_struct(leaf):
        return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=pick(leaf))

    return jax.tree.map(
        to_struct,
        tree_metadata,
        is_leaf=lambda x: hasattr(x, "shape") and hasattr(x, "dtype"),
    )


def load_loop_checkpoint(
    directory,
    step: Optional[int] = None,
    *,
    sharding: Optional[jax.sharding.Sharding] = None,
) -> LoopCheckpoint:
    """Load a loop checkpoint from ``directory``.

    Args:
        directory: The checkpoint root directory.
        step: The step to load. ``None`` loads the latest checkpoint.
        sharding: Target :class:`~jax.sharding.NamedSharding` for the array
            leaves (``primitive_state`` / ``forcing``) — pass the sharding the
            resumed run will use so each device reads only its shard. When
            ``None`` the arrays are restored onto the default device.

    Returns:
        A :class:`LoopCheckpoint`.
    """
    _require_orbax()
    if step is None:
        step = latest_step(directory)
        if step is None:
            raise FileNotFoundError(f"No Orbax checkpoint found in {directory!r}.")

    path = Path(directory).resolve() / str(step)
    checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
    try:
        # The tree of per-array metadata: across Orbax versions ``metadata()``
        # returns either the tree-metadata directly (``.tree``) or a step
        # wrapper (``.item_metadata.tree``).
        metadata = checkpointer.metadata(path)
        tree_metadata = (
            metadata.tree if hasattr(metadata, "tree") else metadata.item_metadata.tree
        )
        target = _abstract_pytree(tree_metadata, sharding)
        tree = checkpointer.restore(path, target)
    finally:
        checkpointer.close()

    return LoopCheckpoint(
        time=tree["time"],
        primitive_state=tree["primitive_state"],
        key=jax.random.wrap_key_data(tree["key_data"]),
        forcing=tree.get("forcing", None),
        num_iterations=tree["num_iterations"],
        step=step,
    )
