"""Generic JAX time-integration loop driver.

Repo-agnostic: depends only on ``jax`` (and equinox's checkpointed while loop
for reverse-mode-friendly adaptive integration).  It owns the parts of a
time-stepping loop that are independent of the physics being integrated:

  * the loop backend — fixed-step ``fori_loop`` / adaptive ``while_loop`` /
    reverse-mode-friendly ``checkpointed_while_loop``;
  * collecting snapshots of the evolving state into preallocated buffers at
    chosen times (plus an optional final snapshot);
  * a host-side progress callback;
  * counting the number of steps taken.

The physics is supplied entirely through caller closures, so this module
carries no domain knowledge and can be reused by any project:

    step(t, state, snapshot_index) -> (dt, new_state)
        Advance ``state`` by one step.  Returns the timestep actually taken
        (the driver advances ``t``) and the new state.  ``snapshot_index`` is
        the current snapshot counter (0 when snapshots are disabled), passed
        through so the step can clamp ``dt`` to land on snapshot times.

    SnapshotSpec.record(t, state, store, index) -> store
        Write whatever diagnostics are wanted for snapshot ``index`` into the
        preallocated ``store`` pytree (typically ``store.field.at[index].set``).

    SnapshotSpec.should_record(t, index) -> bool
        Whether snapshot ``index`` is due at the start of a step at time ``t``
        (e.g. evenly spaced, or matching explicit timepoints).

``state`` and ``store`` are opaque pytrees to the driver.
"""

# typing
from typing import Any, Callable, NamedTuple, Optional

# jax
import jax
import jax.numpy as jnp

# checkpointed loop
from equinox.internal._loop.checkpointed import checkpointed_while_loop


# -------------------------------------------------------------
# =================== ↓ Loop backends ↓ =======================
# -------------------------------------------------------------
# Identifiers selecting which underlying JAX loop construct ``integrate`` uses.
# ``FIXED_STEP`` and ``ADAPTIVE_WHILE`` are not reverse-mode differentiable on
# their own; ``ADAPTIVE_CHECKPOINTED`` trades memory for a reverse-mode-friendly
# while loop via equinox.
FIXED_STEP = 0            # jax.lax.fori_loop over a fixed number of steps
ADAPTIVE_WHILE = 1        # jax.lax.while_loop until t >= t_end (forward-mode AD)
ADAPTIVE_CHECKPOINTED = 2  # checkpointed_while_loop until t >= t_end (reverse-mode AD)
# -------------------------------------------------------------
# =================== ↑ Loop backends ↑ =======================
# -------------------------------------------------------------


def times_close(t, target):
    """Float-precision-aware test that ``t`` has reached ``target``.

    The tolerance is scaled by the working float epsilon and the magnitude of
    ``target``, so the test behaves correctly in both float32 and float64.  A
    fixed absolute tolerance (e.g. ``1e-12``) silently fails in float32, where
    a step landing on ``target`` is typically only accurate to ~1e-7·|target|.
    """
    dtype = jnp.result_type(jnp.asarray(t), jnp.asarray(target))
    atol = 8.0 * jnp.finfo(dtype).eps * jnp.maximum(jnp.abs(target), 1.0)
    return jnp.abs(t - target) <= atol


class SnapshotSpec(NamedTuple):
    """Bundles everything the driver needs to collect snapshots.

    Attributes:
        store: Preallocated output pytree the ``record`` callback writes into.
        record: ``record(t, state, store, index) -> store``.
        should_record: ``should_record(t, index) -> bool`` crossing test.
        record_final: Also record the true final state once after the loop.
        final_index: Buffer slot the final state is written into. When ``None``
            the running snapshot counter is used; for fixed-size buffers this
            should be the last slot (``num_snapshots - 1``) so the final state
            at ``t_end`` is captured reliably regardless of step alignment.
    """

    store: Any
    record: Callable
    should_record: Callable
    record_final: bool = True
    final_index: Optional[int] = None


def integrate(
    state: Any,
    step: Callable,
    t_end,
    *,
    backend: int,
    t_start=0.0,
    num_steps: Optional[int] = None,
    num_checkpoints: Optional[int] = None,
    snapshots: Optional[SnapshotSpec] = None,
    progress: Optional[Callable] = None,
):
    """Run a time-integration loop.

    Args:
        state: Initial evolving-state pytree (opaque to the driver).
        step: ``step(t, state, snapshot_index) -> (dt, new_state)``.
        t_end: Integration end time.
        backend: One of ``FIXED_STEP`` / ``ADAPTIVE_WHILE`` /
            ``ADAPTIVE_CHECKPOINTED``.
        t_start: Initial integration time (the loop clock starts here).
        num_steps: Number of steps for ``FIXED_STEP``.
        num_checkpoints: Checkpoint count for ``ADAPTIVE_CHECKPOINTED``.
        snapshots: A :class:`SnapshotSpec`, or ``None`` to disable collection.
        progress: ``progress(t, t_end)`` host callback, or ``None``.

    Returns:
        ``(t, state, store, num_iterations)``.  ``store`` is the (possibly
        updated) snapshot store, or ``None`` when snapshots are disabled.
    """
    has_snapshots = snapshots is not None

    def body(carry):
        # The carry has an extra snapshot-store slot only when snapshots are
        # collected; the snapshot index is carried either way (and stays 0 when
        # snapshots are disabled) so ``step`` always receives a valid counter.
        if has_snapshots:
            t, state, snapshot_index, num_iterations, store = carry

            # Record the snapshot that is due at the start of this step. This
            # deliberately captures the state *before* this step's update, so a
            # snapshot reflects the field at exactly its recorded time.
            def record_due_snapshot(store_and_index):
                store, snapshot_index = store_and_index
                return (
                    snapshots.record(t, state, store, snapshot_index),
                    snapshot_index + 1,
                )

            store, snapshot_index = jax.lax.cond(
                snapshots.should_record(t, snapshot_index),
                record_due_snapshot,
                lambda store_and_index: store_and_index,
                (store, snapshot_index),
            )
        else:
            t, state, snapshot_index, num_iterations = carry

        dt, state = step(t, state, snapshot_index)
        t = t + dt
        num_iterations = num_iterations + 1

        if progress is not None:
            jax.debug.callback(progress, t, t_end)

        if has_snapshots:
            return (t, state, snapshot_index, num_iterations, store)
        return (t, state, snapshot_index, num_iterations)

    if has_snapshots:
        carry = (t_start, state, 0, 0, snapshots.store)
    else:
        carry = (t_start, state, 0, 0)

    # Fixed-step runs use a plain ``fori_loop``; adaptive runs use a while loop
    # whose predicate runs until the clock reaches ``t_end``, optionally in the
    # checkpointed variant for reverse-mode differentiability.
    if backend == FIXED_STEP:
        carry = jax.lax.fori_loop(0, num_steps, lambda _i, c: body(c), carry)
    elif backend == ADAPTIVE_WHILE:
        carry = jax.lax.while_loop(lambda c: c[0] < t_end, body, carry)
    elif backend == ADAPTIVE_CHECKPOINTED:
        carry = checkpointed_while_loop(
            lambda c: c[0] < t_end, body, carry, checkpoints=num_checkpoints
        )
    else:
        raise ValueError(f"Unknown loop backend: {backend}")

    if has_snapshots:
        t, state, snapshot_index, num_iterations, store = carry
        # The evenly spaced ``should_record`` grid never lands exactly on
        # ``t_end`` (the loop exits at the first step past it), so the final
        # state would otherwise be missing. Record it once here into the
        # reserved final slot — guaranteed written regardless of step
        # alignment, and differentiable since it acts on the loop output.
        if snapshots.record_final:
            final_snapshot_index = (
                snapshot_index
                if snapshots.final_index is None
                else snapshots.final_index
            )
            store = snapshots.record(t, state, store, final_snapshot_index)
        return t, state, store, num_iterations
    t, state, _snapshot_index, num_iterations = carry
    return t, state, None, num_iterations
