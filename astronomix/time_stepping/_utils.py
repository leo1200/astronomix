import jax.numpy as jnp
from astronomix.option_classes.simulation_config import (
    STATE_TYPE,
    STATE_TYPE_ALTERED,
    SimulationConfig,
)


import jax
from beartype import beartype as typechecker
from jaxtyping import jaxtyped


from functools import partial


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config"])
def _unpad(state: STATE_TYPE, config: SimulationConfig) -> STATE_TYPE_ALTERED:
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
    if config.dimensionality == 1:
        state = jnp.pad(state, ((0, 0), (2, 2)), mode="edge")

    elif config.dimensionality == 2:
        state = jnp.pad(state, ((0, 0), (2, 2), (2, 2)), mode="edge")

    elif config.dimensionality == 3:
        state = jnp.pad(state, ((0, 0), (2, 2), (2, 2), (2, 2)), mode="edge")

    return state
