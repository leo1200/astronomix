"""
Container for an extended simulation state.

Wraps the primitive fluid state in a small struct so simulations that follow
additional quantities (e.g. star-particle positions) can carry them alongside
the fluid. Selected via ``config.state_struct``.
"""

# typing
from types import NoneType
from typing import NamedTuple, Union

import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE


class StateStruct(NamedTuple):
    """
    Struct bundling the fluid state with any extra simulation quantities.
    """

    #: The fluid (primitive) state.
    primitive_state: Union[STATE_TYPE, NoneType] = None

    #: Lagrangian tracer-particle positions, shape ``(num_tracers, dim)``,
    #: or ``None`` when the tracer module is inactive. Supplied as the initial
    #: positions on input; returned (no-snapshot path) as the final positions.
    tracers: Union[jnp.ndarray, NoneType] = None

    # Further fields (for example the positions of star particles) can be added
    # here as the simulation grows to follow additional quantities.
