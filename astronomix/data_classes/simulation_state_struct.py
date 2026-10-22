"""
Container for an extended simulation state.

Wraps the primitive fluid state in a small struct so simulations that follow
additional quantities (e.g. sink particles) can carry them alongside the fluid.
Selected via ``config.state_struct``.
"""

# typing
from types import NoneType
from typing import NamedTuple, Union

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig


class SinkParticles(NamedTuple):
    """
    Sink particle data in fixed-size arrays.

    JIT requires array sizes known at compile time, so the arrays always have
    ``config.sink_particle_config.max_num_sinks`` slots. Sinks are only ever
    appended, so the filled slots are contiguous at the front; a slot is
    empty when its mass is zero.
    """

    #: Sink masses, shape (max_num_sinks,).
    mass: jax.Array

    #: Sink positions, shape (max_num_sinks, 3).
    position: jax.Array

    #: Sink velocities, shape (max_num_sinks, 3).
    velocity: jax.Array


class StateStruct(NamedTuple):
    """
    Struct bundling the fluid state with any extra simulation quantities.
    """

    #: The fluid (primitive) state.
    primitive_state: Union[STATE_TYPE, NoneType] = None

    #: The sink particles, or ``None`` when sink particles are not used. When
    #: sink particles are active and this is ``None``, the simulation starts
    #: without any sinks.
    sink_particles: Union[SinkParticles, NoneType] = None


def finalize_state(
    config: SimulationConfig,
    primitive_state: STATE_TYPE,
    sink_particles: Union[SinkParticles, NoneType] = None,
) -> Union[STATE_TYPE, StateStruct]:
    """
    Put the state in the form ``time_integration`` expects for ``config``.

    The state counterpart of ``finalize_config``: call it on the primitive
    state from ``construct_primitive_state``, or on the one returned by
    ``restart_from_latest_checkpoint``, before passing it to
    ``time_integration``.

    Args:
        config: The simulation configuration.
        primitive_state: The primitive state array.
        sink_particles: Sinks that already exist when the run starts, e.g. the
            final sinks of an earlier run or sinks placed by hand: arrays
            ``mass`` (N,), ``position`` (N, 3) and ``velocity`` (N, 3), where a
            slot with zero mass is empty. N may be smaller than
            ``max_num_sinks`` (empty slots are appended when the run starts)
            but not larger. ``None`` starts the run without sinks. A restart
            from disk does not need it: the restored sinks come with the
            ``restart_state`` of ``restart_from_latest_checkpoint``.

    Returns:
        A ``StateStruct`` holding the primitive state and the sinks when
        ``config.state_struct`` is set, otherwise the primitive state itself.

    Raises:
        ValueError: If sink particles are given but ``config.state_struct``
            is not set, since only the state struct carries sinks.
    """
    if config.state_struct:
        return StateStruct(
            primitive_state=primitive_state,
            sink_particles=sink_particles,
        )
    if sink_particles is not None:
        raise ValueError(
            "Sink particles were given, but config.state_struct is not set; "
            "only the state struct carries sink particles."
        )
    return primitive_state
