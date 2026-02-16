import time
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray

from astronomix._finite_difference._magnetic_update._constrained_transport import (
    YAXIS,
    XAXIS,
    ZAXIS,
)
from astronomix._finite_difference._maths._differencing import finite_difference_int6
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import STATE_TYPE, SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._finite_difference._maths._interpolate import interp_face_to_center
from typing import Optional, Callable
from enum import Enum


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
    active_snapshot_callable: bool = eqx.field(static=True, default=False)
    snapshot_callable: Callable[..., None] = eqx.field(static=True, default=None)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        *,
        key: PRNGKeyArray,
        scale: float = 0.005,
        snapshot_callable: Optional[Callable[..., None]] = None,
    ):
        # We need a key for each convolutional layer
        key1, key2, key3, init_key = jax.random.split(key, 4)

        # Construct the CNN normally
        layers = [
            eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
        ]
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

        if self.active_snapshot_callable:
            jax.debug.callback(
                self.snapshot_callable, time_step, primitive_state, correction
            )
        # update the primitive state with the correction
        primitive_state = primitive_state.at[:5].add(correction[:5] * time_step)
        primitive_state = primitive_state.at[-3:].add(correction[-3:] * time_step)

        Bx_center = interp_face_to_center(primitive_state[-3], XAXIS)
        By_center = interp_face_to_center(primitive_state[-2], YAXIS)
        Bz_center = interp_face_to_center(primitive_state[-1], ZAXIS)

        primitive_state = primitive_state.at[registered_variables.magnetic_index.x].set(
            Bx_center
        )
        primitive_state = primitive_state.at[registered_variables.magnetic_index.y].set(
            By_center
        )
        primitive_state = primitive_state.at[registered_variables.magnetic_index.z].set(
            Bz_center
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


class ScalarFieldCorrectorCNN(eqx.Module):
    """A simple CNN that maps an input of shape (C, H, W) to a static field."""

    layers: eqx.nn.Sequential
    channel_index_output: int
    active_snapshot_callable: bool = eqx.field(static=True, default=False)
    snapshot_callable: Callable[..., None] = eqx.field(static=True, default=None)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        channel_index_output: int,
        *,
        key: PRNGKeyArray,
        scale: float = 0.005,
        snapshot_callable: Optional[Callable[..., None]] = None,
    ):
        # We need a key for each convolutional layer
        key1, key2, key3, init_key = jax.random.split(key, 4)

        # Construct the CNN normally
        layers = [
            eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
        ]
        for _ in range(hidden_layers):
            layers.append(
                eqx.nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, key=key2)
            )
            layers.append(eqx.nn.Lambda(jax.nn.relu))
        layers.append(
            eqx.nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=1,
                kernel_size=3,
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
        self.active_snapshot_callable = False
        if isinstance(snapshot_callable, Callable):
            self.active_snapshot_callable = True
            self.snapshot_callable = snapshot_callable
        self.channel_index_output = channel_index_output

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Array,
    ) -> Array:
        """The forward pass of the model."""
        correction = self.layers(primitive_state)

        if self.active_snapshot_callable:
            jax.debug.callback(
                self.snapshot_callable, time_step, primitive_state, correction
            )
        # update the primitive state with the correction
        primitive_state = primitive_state.at[self.channel_index_output].add(
            correction[0] * time_step
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


class VectorField(Enum):
    VELOCITY = "velocity"
    MAGNETIC = "magnetic"


def update_model_vector_field(
    primitive_state: STATE_TYPE,
    correction: Array,
    time_step: Array,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    field_type: VectorField,
):
    if field_type is VectorField.VELOCITY:
        primitive_state = primitive_state.at[1:4].add(correction * time_step)
    if field_type is VectorField.MAGNETIC:
        omega_bar = correction
        bx_interface_correction, by_interface_correction, bz_interface_correction = (
            finite_difference_curl_3D(omega_bar, config.grid_spacing)
        )
        interface_stack = jnp.stack(
            [bx_interface_correction, by_interface_correction, bz_interface_correction],
            axis=0,
        )
        correction = correction.at[-3:].set(interface_stack)

        # update the primitive state with the correction
        primitive_state = primitive_state.at[-3:].add(correction[-3:] * time_step)

        Bx_center = interp_face_to_center(primitive_state[-3], XAXIS)
        By_center = interp_face_to_center(primitive_state[-2], YAXIS)
        Bz_center = interp_face_to_center(primitive_state[-1], ZAXIS)

        primitive_state = primitive_state.at[registered_variables.magnetic_index.x].set(
            Bx_center
        )
        primitive_state = primitive_state.at[registered_variables.magnetic_index.y].set(
            By_center
        )
        primitive_state = primitive_state.at[registered_variables.magnetic_index.z].set(
            Bz_center
        )

    return primitive_state


class VectorFieldCorrectorCNN(eqx.Module):
    """A simple CNN that maps an input of shape (C, H, W) to a static field."""

    layers: eqx.nn.Sequential
    vector_field_output: VectorField
    active_snapshot_callable: bool = eqx.field(static=True, default=False)
    snapshot_callable: Callable[..., None] = eqx.field(static=True, default=None)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        vector_field_output: VectorField,
        *,
        key: PRNGKeyArray,
        scale: float = 0.005,
        snapshot_callable: Optional[Callable[..., None]] = None,
    ):
        # We need a key for each convolutional layer
        key1, key2, key3, init_key = jax.random.split(key, 4)

        # Construct the CNN normally
        layers = [
            eqx.nn.Conv3d(in_channels, hidden_channels, 3, padding=1, key=key1),
            eqx.nn.Lambda(jax.nn.relu),
        ]
        for _ in range(hidden_layers):
            layers.append(
                eqx.nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1, key=key2)
            )
            layers.append(eqx.nn.Lambda(jax.nn.relu))
        layers.append(
            eqx.nn.Conv3d(
                in_channels=hidden_channels,
                out_channels=3,
                kernel_size=3,
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
        self.active_snapshot_callable = False
        if isinstance(snapshot_callable, Callable):
            self.active_snapshot_callable = True
            self.snapshot_callable = snapshot_callable
        self.vector_field_output = vector_field_output

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Array,
    ) -> Array:
        """The forward pass of the model."""
        correction = self.layers(primitive_state)

        if self.active_snapshot_callable:
            jax.debug.callback(
                self.snapshot_callable, time_step, primitive_state, correction
            )
        primitive_state = update_model_vector_field(
            primitive_state,
            correction,
            time_step=time_step,
            config=config,
            registered_variables=registered_variables,
            field_type=self.vector_field_output,
        )
        return primitive_state
