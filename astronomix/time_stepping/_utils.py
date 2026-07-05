"""
Padding helpers for the time-integration loop.

Adds and removes the ghost-cell halo around a primitive state. Ghost cells hold
the boundary information the stencils read into, so the interior is padded
before integration (``_pad``) and the halo stripped off again before the state
is returned to the user (``_unpad``).
"""

# general
from functools import partial

# typing
from beartype import beartype as typechecker
from jaxtyping import jaxtyped

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    STATE_TYPE,
    STATE_TYPE_ALTERED,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config"])
def _unpad(state: STATE_TYPE, config: SimulationConfig) -> STATE_TYPE_ALTERED:
    """
    Strip the ghost-cell halo off every spatial axis of the state.

    Slices ``num_ghost_cells`` off both ends of each spatial axis, leaving the
    physical interior. Axis 0 holds the fluid variables and is never sliced.

    Args:
        state: The padded primitive state array (variables on axis 0, then one
            spatial axis per dimension).
        config: The simulation configuration; supplies the dimensionality and
            the number of ghost cells per side.

    Returns:
        The interior primitive state with the ghost halo removed.
    """
    if config.dimensionality == 1:
        state = jax.lax.slice_in_dim(
            state,
            config.num_ghost_cells,
            state.shape[1] - config.num_ghost_cells,
            axis=1,
        )
    elif config.dimensionality == 2:
        state = jax.lax.slice_in_dim(
            state,
            config.num_ghost_cells,
            state.shape[1] - config.num_ghost_cells,
            axis=1,
        )
        state = jax.lax.slice_in_dim(
            state,
            config.num_ghost_cells,
            state.shape[2] - config.num_ghost_cells,
            axis=2,
        )
    elif config.dimensionality == 3:
        state = jax.lax.slice_in_dim(
            state,
            config.num_ghost_cells,
            state.shape[1] - config.num_ghost_cells,
            axis=1,
        )
        state = jax.lax.slice_in_dim(
            state,
            config.num_ghost_cells,
            state.shape[2] - config.num_ghost_cells,
            axis=2,
        )
        state = jax.lax.slice_in_dim(
            state,
            config.num_ghost_cells,
            state.shape[3] - config.num_ghost_cells,
            axis=3,
        )

    return state


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config"])
def _pad(state: STATE_TYPE, config: SimulationConfig) -> STATE_TYPE_ALTERED:
    """
    Surround the state with a ghost-cell halo on every spatial axis.

    Pads ``num_ghost_cells`` cells onto both ends of each spatial axis using
    edge replication, which gives a sensible default for the ghost cells before
    the boundary handler fills them in. Axis 0 (the fluid variables) is left
    unpadded.

    Args:
        state: The interior primitive state array (variables on axis 0, then one
            spatial axis per dimension).
        config: The simulation configuration; supplies the dimensionality and
            the number of ghost cells per side.

    Returns:
        The primitive state padded with an edge-replicated ghost halo.
    """
    if config.dimensionality == 1:
        state = jnp.pad(
            state,
            ((0, 0), (config.num_ghost_cells, config.num_ghost_cells)),
            mode="edge",
        )

    elif config.dimensionality == 2:
        state = jnp.pad(
            state,
            (
                (0, 0),
                (config.num_ghost_cells, config.num_ghost_cells),
                (config.num_ghost_cells, config.num_ghost_cells),
            ),
            mode="edge",
        )

    elif config.dimensionality == 3:
        state = jnp.pad(
            state,
            (
                (0, 0),
                (config.num_ghost_cells, config.num_ghost_cells),
                (config.num_ghost_cells, config.num_ghost_cells),
                (config.num_ghost_cells, config.num_ghost_cells),
            ),
            mode="edge",
        )

    return state
