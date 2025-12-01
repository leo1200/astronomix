# general
from functools import partial
import jax.numpy as jnp
import jax

import equinox as eqx

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams


class ForceNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=3,  # <- now takes (x, y, t)
            out_size=2,  # returns (Fx, Fy)
            width_size=128,
            depth=4,
            key=key,
        )

    def __call__(self, xyt):
        return self.mlp(xyt)


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _neural_net_force(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    time_step: Float[Array, ""],
    current_time: Float[Array, ""],  # <- new argument
):
    neural_net_params = params.neural_net_force_params.network_params
    neural_net_static = config.neural_net_force_config.network_static
    model = eqx.combine(neural_net_params, neural_net_static)

    N = config.num_cells
    positions = helper_data.geometric_centers
    positions_flat = positions.reshape(-1, 2)  # (N*N, 2)

    # Add time dimension: broadcast time to (N*N, 1) and concat
    time_broadcasted = jnp.full((positions_flat.shape[0], 1), current_time)
    positions_with_time = jnp.concatenate(
        [positions_flat, time_broadcasted], axis=1
    )  # (N*N, 3)

    # Apply model to (x, y, t)
    forces_flat = jax.vmap(model)(positions_with_time)  # shape (N*N, 2)

    # Reshape back to (2, N, N)
    forces = forces_flat.reshape(N, N, 2).transpose(2, 0, 1)

    nc = config.num_ghost_cells
    primitive_state = primitive_state.at[
        registered_variables.velocity_index.x, nc:-nc, nc:-nc
    ].add(forces[0] * time_step)
    primitive_state = primitive_state.at[
        registered_variables.velocity_index.y, nc:-nc, nc:-nc
    ].add(forces[1] * time_step)

    return primitive_state
