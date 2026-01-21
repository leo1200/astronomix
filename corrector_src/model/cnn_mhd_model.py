from typing import Optional, Callable
import jax
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

# general
from functools import partial

from jax.experimental import checkify
from jf1uids._physics_modules._mhd._vector_maths import curl2D, curl3D, divergence3D
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
import jax.numpy as jnp


class CorrectorCNN(eqx.Module):
    """
    A simple CNN that maps an input of shape (C, H, W) to an output of the same shape.
    """

    layers: eqx.nn.Sequential
    active_snapshot_callable: bool = eqx.field(static=True)
    snapshot_callable: Callable[..., None] = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        key: PRNGKeyArray,
        snapshot_callable: Optional[Callable[..., None]] = None,
    ):
        # We need a key for each convolutional layer
        key1, key2, key3 = jax.random.split(key, 3)

        # A simple 3-layer CNN.
        # Note the use of padding=1 with kernel_size=3 to keep the
        # spatial dimensions (height and width) the same.
        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, key=key2),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Conv3d(
                    hidden_channels, in_channels, 3, padding=1, key=key3, use_bias=False
                ),
            ]
        )
        self.active_snapshot_callable = False
        if isinstance(snapshot_callable, Callable):
            self.active_snapshot_callable = True
            self.snapshot_callable = snapshot_callable

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ):
        """
        The forward pass of the model.
        """
        # Pass the input through the network to get the correction term
        correction = self.layers(primitive_state)
        electric_field_correction = correction[-3:, ...]
        if config.dimensionality == 2:
            magnetic_field_correction = curl2D(
                electric_field_correction, config.grid_spacing
            )
        elif config.dimensionality == 3:
            magnetic_field_correction = curl3D(
                electric_field_correction, config.grid_spacing
            )
        else:
            raise ValueError(
                "dimensionality was {config.dimensionality} instead of 2 or 3"
            )
        correction = correction.at[-3:, ...].set(magnetic_field_correction)

        # update the primitive state with the correction
        primitive_state = primitive_state + correction * time_step
        if self.active_snapshot_callable:
            jax.debug.callback(
                self.snapshot_callable, time_step, primitive_state, correction
            )
        # ensure that the pressure is larger than a minimum value
        p_min = 1e-12
        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.maximum(primitive_state[registered_variables.pressure_index], p_min)
        )
        rho_min = 1e-12
        primitive_state = primitive_state.at[registered_variables.density_index].set(
            jnp.maximum(primitive_state[registered_variables.density_index], rho_min)
        )

        return primitive_state
