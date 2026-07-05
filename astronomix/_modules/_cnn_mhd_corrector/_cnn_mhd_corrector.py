"""
CNN-based MHD corrector.

A small convolutional network learns a per-step correction to the primitive
state. The magnetic-field part of the correction is obtained as the curl of a
learned electric field, so that the divergence-free constraint on the magnetic
field is preserved by construction (the divergence of a curl is zero).
"""

# general
from functools import partial

# typing
from jaxtyping import Array, Float, PRNGKeyArray

# jax
import jax
import jax.numpy as jnp

# neural networks
import equinox as eqx

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._magnetic_update._vector_maths import curl2D


class CorrectorCNN(eqx.Module):
    """
    A simple CNN that maps an input of shape (C, H, W) to an output of the same shape.
    """

    layers: list

    def __init__(self, in_channels: int, hidden_channels: int, *, key: PRNGKeyArray):
        # Each convolutional layer needs its own PRNG key for weight init.
        key1, key2, key3 = jax.random.split(key, 3)

        # A simple 3-layer CNN. Using padding=1 with kernel_size=3 keeps the
        # spatial dimensions (height and width) unchanged through each layer.
        self.layers = (
            # Layer 1: expand channels from the number of variables to the
            # hidden width.
            eqx.nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                key=key1,
            ),
            jax.nn.relu,
            # Layer 2: a hidden convolutional layer.
            eqx.nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                key=key2,
            ),
            jax.nn.relu,
            # Layer 3: contract channels back to the original number of
            # variables. No activation here, as we want to predict a raw
            # correction value.
            eqx.nn.Conv2d(
                hidden_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                key=key3,
            ),
        )

    def __call__(self, x: Float[Array, "num_vars h w"]) -> Float[Array, "num_vars h w"]:
        """
        The forward pass of the model.

        Args:
            x: The input field of shape (num_vars, h, w).

        Returns:
            The predicted correction of shape (num_vars, h, w).
        """
        # Pass the input through the network to get the correction term.
        correction = x
        for layer in self.layers:
            correction = layer(correction)

        return correction


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def _cnn_mhd_corrector(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    time_step: Float[Array, ""],
):
    """
    Apply the learned CNN correction to the primitive state for one time step.

    Args:
        primitive_state: The primitive state array.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters.
        time_step: The time step over which the correction is applied.

    Returns:
        The corrected primitive state.
    """
    neural_net_params = params.cnn_mhd_corrector_params.network_params
    neural_net_static = config.cnn_mhd_corrector_config.network_static
    model = eqx.combine(neural_net_params, neural_net_static)

    correction = model(primitive_state)

    # To avoid introducing divergence errors in the magnetic field, the network
    # predicts a correction for the electric field and we take its curl: the
    # divergence of a curl is zero, so the magnetic field stays divergence-free.
    electric_field_correction = correction[-3:, ...]
    magnetic_field_correction = curl2D(electric_field_correction, config.grid_spacing)
    correction = correction.at[-3:, ...].set(magnetic_field_correction)

    # Apply the correction over the time step.
    primitive_state = primitive_state + correction * time_step

    # Keep the pressure above a small floor so the correction cannot drive it
    # non-positive.
    minimum_pressure = 1e-4
    primitive_state = primitive_state.at[registered_variables.pressure_index].set(
        jnp.maximum(primitive_state[registered_variables.pressure_index], minimum_pressure)
    )

    return primitive_state
