import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

# typing
from beartype import beartype as typechecker
from typing import Tuple, Union

from jf1uids._finite_difference._magnetic_update._constrained_transport import (
    YAXIS,
    XAXIS,
    ZAXIS,
    update_cell_center_fields,
)
from jf1uids._finite_difference._maths._differencing import finite_difference_int6
from jf1uids.data_classes.simulation_helper_data import HelperData
from jf1uids.variable_registry.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from jf1uids.option_classes.simulation_params import SimulationParams
from functools import partial


def finite_difference_curl_3D(omega_bar, grid_spacing):
    dtdy = 1.0 / grid_spacing
    dtdz = 1.0 / grid_spacing
    dtdx = 1.0 / grid_spacing
    rhs_bx = -dtdy * finite_difference_int6(
        omega_bar[2], YAXIS
    ) + dtdz * finite_difference_int6(omega_bar[1], ZAXIS)

    rhs_by = -dtdz * finite_difference_int6(
        omega_bar[0], ZAXIS
    ) + dtdx * finite_difference_int6(omega_bar[2], XAXIS)

    rhs_bz = -dtdx * finite_difference_int6(
        omega_bar[1], XAXIS
    ) + dtdy * finite_difference_int6(omega_bar[0], YAXIS)
    return rhs_bx, rhs_by, rhs_bz


class CorrectorCNN(eqx.Module):
    """A simple CNN that maps an input of shape (C, H, W) to an output of the same shape."""

    layers: eqx.nn.Sequential

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        *,
        key: PRNGKeyArray,
        scale: float = 0.005,
    ):
        # We need a key for each convolutional layer
        key1, key2, key3, init_key = jax.random.split(key, 4)

        # Construct the CNN normally
        layers = [
            eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
            eqx.nn.Lambda(jax.nn.relu)]
        for _ in range(hidden_layers):
            layers.append(
             eqx.nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, key=key2)
            ) 
            layers.append(eqx.nn.Lambda(jax.nn.relu))
        layers.append(
            eqx.nn.Conv3d(
                hidden_channels,
                in_channels - 3,
                3,
                padding=1,
                key=key3,
                use_bias=False,
            ),
        )


        # After building the Sequential, we reinitialize all Conv3d weights
        seq = eqx.nn.Sequential(layers)

        # Reinit each conv.weight with scaled normal
        conv_indices = [
            i for i, l in enumerate(seq.layers) if isinstance(l, eqx.nn.Conv3d)
        ]

        for i in conv_indices:
            layer = seq.layers[i]
            wkey, init_key = jax.random.split(init_key)
            # Standard initialization is usually variance scaling,
            # here we force a specific scale normal distribution.
            new_w = scale * jax.random.normal(wkey, layer.weight.shape)

            # Update the model tree
            seq = eqx.tree_at(lambda s, idx=i: s.layers[idx].weight, seq, new_w)
        self.layers = seq

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ) -> Float[Array, "num_vars h w"]:
        """The forward pass of the model."""
        correction = self.layers(primitive_state)

        omega_bar = correction[-3:, ...]
        bx_interface_correction, by_interface_correction, bz_interface_correction = (
            finite_difference_curl_3D(omega_bar, config.grid_spacing)
        )
        interface_stack = jnp.stack(
            [bx_interface_correction, by_interface_correction, bz_interface_correction],
            axis=0,
        )
        correction = correction.at[-3:].set(interface_stack)
        # update the primitive state with the correction
        primitive_state = primitive_state.at[:5].add(correction[:5] * time_step)
        primitive_state = primitive_state.at[-3:].add(correction[-3:] * time_step)

        primitive_state = update_cell_center_fields(
            primitive_state,
            primitive_state[-3],
            primitive_state[-2],
            primitive_state[-1],
            registered_variables,
        )

        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.maximum(
                primitive_state[registered_variables.pressure_index],
                params.minimum_pressure,
            )
        )
        primitive_state = primitive_state.at[registered_variables.density_index].set(
            jnp.maximum(
                primitive_state[registered_variables.density_index],
                params.minimum_density,
            )
        )

        return primitive_state
