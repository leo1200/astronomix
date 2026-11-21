"""
Configuration and parameter containers of the relaxation zone.

``RelaxationZoneConfig`` switches the zone on; ``RelaxationZoneParams`` holds the
target state and the relaxation rate, both as fields on the grid, so they can be
changed (for example to a time-dependent target fed from a larger simulation)
without recompiling.
"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp


class RelaxationZoneConfig(NamedTuple):
    """Static settings of the relaxation zone: on or off."""

    relaxation_zone: bool = False


class RelaxationZoneParams(NamedTuple):
    """
    The target state and the relaxation rate of the zone.

    Both live on the grid without ghost cells, like the external
    gravitational potential, and are given ghost cells internally (with a rate
    of zero, so the boundary conditions alone set the ghost cells).
    """

    #: The primitive state the zone relaxes to, of shape
    #: (num_variables, *spatial cells).
    target_state: jnp.ndarray = jnp.array([])

    #: The relaxation rate 1 / tau per cell, of shape (*spatial cells) for one
    #: rate of all variables or (num_variables, *spatial cells) for one rate per
    #: variable. Zero outside the zone; a smooth rise into the zone keeps the
    #: zone itself from reflecting waves.
    rate: jnp.ndarray = jnp.array([])
