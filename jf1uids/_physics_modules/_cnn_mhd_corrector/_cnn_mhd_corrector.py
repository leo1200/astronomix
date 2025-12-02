import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

# general
from functools import partial
import jax.numpy as jnp
import jax

import equinox as eqx

# typing
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Tuple, Union

from jf1uids._finite_volume._magnetic_update._vector_maths import curl2D
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.variable_registry.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams


class CorrectorCNN(eqx.Module):
    """
    A simple CNN that maps an input of shape (C, H, W) to an output of the same shape.
    """

    layers: eqx.nn.Sequential

    def __init__(self, in_channels: int, hidden_channels: int, *, key: PRNGKeyArray):
        # We need a key for each convolutional layer
        key1, key2, key3 = jax.random.split(key, 3)

        # A simple 3-layer CNN.
        # Note the use of padding=1 with kernel_size=3 to keep the
        # spatial dimensions (height and width) the same.
        self.layers = eqx.nn.Sequential(
            [
                # Layer 1: Expand channels from NUM_VARS to HIDDEN_CHANNELS
                eqx.nn.Conv2d(
                    in_channels, hidden_channels, kernel_size=3, padding=1, key=key1
                ),
                jax.nn.relu,
                # Layer 2: A hidden convolutional layer
                eqx.nn.Conv2d(
                    hidden_channels, hidden_channels, kernel_size=3, padding=1, key=key2
                ),
                jax.nn.relu,
                # Layer 3: Contract channels back to the original NUM_VARS
                # No activation function here, as we want to predict a raw correction value.
                eqx.nn.Conv2d(
                    hidden_channels, in_channels, kernel_size=3, padding=1, key=key3
                ),
            ]
        )

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ) -> Float[Array, "num_vars h w"]:
        """
        The forward pass of the model.
        """
        # Pass the input through the network to get the correction term
        correction = self.layers(primitive_state)

        # Add the learned correction to the original input
        electric_field_correction = correction[-3:, ...]
        magnetic_field_correction = curl2D(
            electric_field_correction, config.grid_spacing
        )
        correction = correction.at[-3:, ...].set(magnetic_field_correction)

        # update the primitive state with the correction
        primitive_state = primitive_state + correction * time_step

        # ensure that the pressure is larger than a minimum value
        p_min = 1e-4
        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
        )

        return primitive_state


# # @jaxtyped(typechecker=typechecker)
# @partial(jax.jit, static_argnames=["registered_variables", "config"])
# def _cnn_mhd_corrector(
#     primitive_state: STATE_TYPE,
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
#     params: SimulationParams,
#     time_step: Float[Array, ""],
# ):
#     neural_net_params = params.cnn_mhd_corrector_params.network_params
#     neural_net_static = config.cnn_mhd_corrector_config.network_static
#     model = eqx.combine(neural_net_params, neural_net_static)

#     correction = model(primitive_state)

#     # to not add divergence errors, we learn a correction for the electric field
#     # - and the divergence of a curl is zero
#     electric_field_correction = correction[-3:, ...]
#     magnetic_field_correction = curl2D(electric_field_correction, config.grid_spacing)
#     correction = correction.at[-3:, ...].set(magnetic_field_correction)

#     # update the primitive state with the correction
#     primitive_state = primitive_state + correction * time_step

#     # ensure that the pressure is larger than a minimum value
#     p_min = 1e-4
#     primitive_state = primitive_state.at[registered_variables.pressure_index].set(
#         jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
#     )

#     return primitive_state
