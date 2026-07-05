"""
Neural-network body force.

Evaluates a small Equinox MLP that maps a cell position and the current time
``(x, y, t)`` to a 2D force ``(Fx, Fy)``, and applies it as a velocity increment
to the interior cells. The trainable weights are carried in the simulation
parameters so the force can be learned through the differentiable solver.
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# neural networks
import equinox as eqx

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams


class ForceNet(eqx.Module):
    """An MLP mapping ``(x, y, t)`` to a planar force ``(Fx, Fy)``."""

    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=3,  # the input is the cell position and time (x, y, t)
            out_size=2,  # the output is the planar force (Fx, Fy)
            width_size=128,
            depth=4,
            key=key,
        )

    def __call__(self, xyt):
        return self.mlp(xyt)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _neural_net_force(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    time_step: Float[Array, ""],
    current_time: Float[Array, ""],
):
    """
    Apply the neural-network body force as a velocity increment.

    The network is reassembled from its trainable parameters and static
    structure, evaluated at every interior cell position and the current time,
    and the resulting force is integrated over one time step into the velocity.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters (carrying the trainable weights).
        helper_data: The helper data (providing the cell positions).
        time_step: The time step over which the force is applied.
        current_time: The current simulation time, fed to the network.

    Returns:
        The primitive state array with the neural-network force applied.
    """
    # Reassemble the network from its trainable parameters and its static
    # structure (the standard Equinox filter / combine split).
    neural_net_params = params.neural_net_force_params.network_params
    neural_net_static = config.neural_net_force_config.network_static
    model = eqx.combine(neural_net_params, neural_net_static)

    # TODO: this assumes the same number of cells in x and y; generalise to
    # non-square grids.
    num_cells_per_side = config.num_cells.x
    positions = helper_data.geometric_centers
    positions_flat = positions.reshape(-1, 2)

    # Broadcast the current time to every position and append it as the third
    # input feature, so the network sees (x, y, t) per cell.
    time_broadcasted = jnp.full((positions_flat.shape[0], 1), current_time)
    positions_with_time = jnp.concatenate(
        [positions_flat, time_broadcasted],
        axis=1,
    )

    # Evaluate the network at every cell, then fold the flat (N*N, 2) output
    # back into a (2, N, N) field of force components.
    forces_flat = jax.vmap(model)(positions_with_time)
    forces = forces_flat.reshape(
        num_cells_per_side, num_cells_per_side, 2
    ).transpose(2, 0, 1)

    # Apply the force to the interior cells only (skip the ghost cells), as a
    # velocity increment integrated over one time step.
    num_ghost_cells = config.num_ghost_cells
    primitive_state = primitive_state.at[
        registered_variables.velocity_index.x,
        num_ghost_cells:-num_ghost_cells,
        num_ghost_cells:-num_ghost_cells,
    ].add(forces[0] * time_step)
    primitive_state = primitive_state.at[
        registered_variables.velocity_index.y,
        num_ghost_cells:-num_ghost_cells,
        num_ghost_cells:-num_ghost_cells,
    ].add(forces[1] * time_step)

    return primitive_state
