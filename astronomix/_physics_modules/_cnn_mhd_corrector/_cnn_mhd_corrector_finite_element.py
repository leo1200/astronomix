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
    normalize_input: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        *,
        key: PRNGKeyArray,
        scale: float = 0.005,
        snapshot_callable: Optional[Callable[..., None]] = None,
        normalize_input: bool = False,
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
            print("active snapshot callable")
            self.active_snapshot_callable = True
            self.snapshot_callable = snapshot_callable
        self.normalize_input = normalize_input

    def _compute_scales(
        self,
        primitive_state: STATE_TYPE,
        registered_variables: RegisteredVariables,
    ) -> Array:
        """Compute self-consistent normalization scales from the current state."""
        rho_mean = jnp.maximum(
            jnp.mean(primitive_state[registered_variables.density_index]), 1e-8
        )
        p_ref = jnp.maximum(
            jnp.std(primitive_state[registered_variables.pressure_index]), 1e-8
        )
        B2 = (
            primitive_state[-3] ** 2
            + primitive_state[-2] ** 2
            + primitive_state[-1] ** 2
        )
        # TODO: wrong normalizing for v here, it still works
        B_rms = jnp.maximum(jnp.sqrt(jnp.mean(B2)), 1e-8)
        v_ref = B_rms / jnp.sqrt(rho_mean)

        n = primitive_state.shape[0]
        scales = jnp.ones(n)
        scales = scales.at[registered_variables.density_index].set(rho_mean)
        scales = scales.at[registered_variables.pressure_index].set(p_ref)
        scales = scales.at[registered_variables.velocity_index.x].set(v_ref)
        scales = scales.at[registered_variables.velocity_index.y].set(v_ref)
        scales = scales.at[registered_variables.velocity_index.z].set(v_ref)
        scales = scales.at[registered_variables.magnetic_index.x].set(B_rms)
        scales = scales.at[registered_variables.magnetic_index.y].set(B_rms)
        scales = scales.at[registered_variables.magnetic_index.z].set(B_rms)
        scales = scales.at[-3].set(B_rms)
        scales = scales.at[-2].set(B_rms)
        scales = scales.at[-1].set(B_rms)
        return scales

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ) -> Float[Array, "num_vars h w"]:
        """The forward pass of the model."""

        if self.normalize_input:
            scales = self._compute_scales(primitive_state, registered_variables)
            n_out = primitive_state.shape[0] - 3
            net_input = primitive_state / scales[:, None, None, None]
            correction_raw = self.layers(net_input)
            correction = correction_raw * scales[:n_out, None, None, None]
        else:
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


class FiLMCorrectorCNN(eqx.Module):
    """CorrectorCNN extended with self-consistent input normalization and FiLM conditioning.

    Normalizes the input state by self-consistent Alfvénic scales, then injects
    local plasma β and Alfvénic Mach number as two extra spatial input channels.
    Between each hidden convolutional layer, Feature-wise Linear Modulation (FiLM)
    is applied, conditioned on global [log(mean_β+1), mean_M_A] scalars, allowing
    the model to adapt its corrections to the local physical regime without knowing
    any problem-specific parameters at instantiation time.

    Architecture:
        input_conv:   Conv3d(in_channels+2 → hidden_channels)  + ReLU
        hidden_convs: Conv3d(hidden_channels → hidden_channels) + FiLM + ReLU  (×hidden_layers)
        output_conv:  Conv3d(hidden_channels → in_channels-3)  (no bias, no activation)

    Output / state-update logic is identical to CorrectorCNN.
    """

    input_conv: eqx.nn.Conv3d
    hidden_convs: tuple
    output_conv: eqx.nn.Conv3d
    film_gamma: tuple
    film_beta: tuple
    normalize_input: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        hidden_layers: int,
        *,
        key: PRNGKeyArray,
        scale: float = 0.005,
        normalize_input: bool = True,
    ):
        def next_key():
            nonlocal key
            key, subkey = jax.random.split(key)
            return subkey

        def make_conv3d(in_ch, out_ch, use_bias=True):
            conv = eqx.nn.Conv3d(
                in_ch, out_ch, 3, padding=1, use_bias=use_bias, key=next_key()
            )
            new_w = scale * jax.random.normal(next_key(), conv.weight.shape)
            return eqx.tree_at(lambda m: m.weight, conv, new_w)

        def make_film_linear(out_ch):
            lin = eqx.nn.Linear(2, out_ch, key=next_key())
            new_w = scale * 0.1 * jax.random.normal(next_key(), lin.weight.shape)
            new_b = jnp.zeros(lin.bias.shape)
            lin = eqx.tree_at(lambda m: m.weight, lin, new_w)
            return eqx.tree_at(lambda m: m.bias, lin, new_b)

        self.input_conv = make_conv3d(in_channels + 2, hidden_channels)
        self.hidden_convs = tuple(
            make_conv3d(hidden_channels, hidden_channels) for _ in range(hidden_layers)
        )
        self.output_conv = make_conv3d(hidden_channels, in_channels - 3, use_bias=False)
        self.film_gamma = tuple(
            make_film_linear(hidden_channels) for _ in range(hidden_layers)
        )
        self.film_beta = tuple(
            make_film_linear(hidden_channels) for _ in range(hidden_layers)
        )
        self.normalize_input = normalize_input

    def _compute_scales(
        self,
        primitive_state: STATE_TYPE,
        registered_variables: RegisteredVariables,
    ) -> Array:
        """Self-consistent Alfvénic normalization scales (same logic as CorrectorCNN)."""
        rho_mean = jnp.maximum(
            jnp.mean(primitive_state[registered_variables.density_index]), 1e-8
        )
        p_ref = jnp.maximum(
            jnp.std(primitive_state[registered_variables.pressure_index]), 1e-8
        )
        B2 = (
            primitive_state[-3] ** 2
            + primitive_state[-2] ** 2
            + primitive_state[-1] ** 2
        )
        B_rms = jnp.maximum(jnp.sqrt(jnp.mean(B2)), 1e-8)
        v_ref = B_rms / jnp.sqrt(rho_mean)

        n = primitive_state.shape[0]
        scales = jnp.ones(n)
        scales = scales.at[registered_variables.density_index].set(rho_mean)
        scales = scales.at[registered_variables.pressure_index].set(p_ref)
        scales = scales.at[registered_variables.velocity_index.x].set(v_ref)
        scales = scales.at[registered_variables.velocity_index.y].set(v_ref)
        scales = scales.at[registered_variables.velocity_index.z].set(v_ref)
        scales = scales.at[registered_variables.magnetic_index.x].set(B_rms)
        scales = scales.at[registered_variables.magnetic_index.y].set(B_rms)
        scales = scales.at[registered_variables.magnetic_index.z].set(B_rms)
        scales = scales.at[-3].set(B_rms)
        scales = scales.at[-2].set(B_rms)
        scales = scales.at[-1].set(B_rms)
        return scales

    def __call__(
        self,
        primitive_state: STATE_TYPE,
        config: SimulationConfig,
        registered_variables: RegisteredVariables,
        params: SimulationParams,
        time_step: Float[Array, ""],
    ) -> Float[Array, "num_vars h w"]:
        """Forward pass with FiLM-conditioned correction."""
        # --- 1. Compute plasma regime fields from center B (channels 5-7) ---
        Bx = primitive_state[registered_variables.magnetic_index.x]
        By = primitive_state[registered_variables.magnetic_index.y]
        Bz = primitive_state[registered_variables.magnetic_index.z]
        B2_center = Bx**2 + By**2 + Bz**2

        p = primitive_state[registered_variables.pressure_index]
        rho = primitive_state[registered_variables.density_index]
        vx = primitive_state[registered_variables.velocity_index.x]
        vy = primitive_state[registered_variables.velocity_index.y]
        vz = primitive_state[registered_variables.velocity_index.z]

        beta_field = 2.0 * p / (B2_center + 1e-8)
        mach_a_field = jnp.sqrt((vx**2 + vy**2 + vz**2) * rho) / (
            jnp.sqrt(B2_center) + 1e-8
        )

        # Global conditioning vector for FiLM: [log(mean_β+1), mean_M_A]
        z = jnp.array(
            [
                jnp.log(jnp.mean(beta_field) + 1.0),
                jnp.mean(mach_a_field),
            ]
        )

        # --- 2. Normalize input state ---
        if self.normalize_input:
            scales = self._compute_scales(primitive_state, registered_variables)
            state_norm = primitive_state / scales[:, None, None, None]
        else:
            state_norm = primitive_state
            scales = jnp.ones(primitive_state.shape[0])

        # --- 3. Concatenate β and M_A as extra input channels ---
        net_input = jnp.concatenate(
            [state_norm, beta_field[None], mach_a_field[None]], axis=0
        )

        # --- 4. Forward through network with FiLM ---
        x = jax.nn.relu(self.input_conv(net_input))
        for conv, g_lin, b_lin in zip(
            self.hidden_convs, self.film_gamma, self.film_beta
        ):
            gamma = g_lin(z)[:, None, None, None] + 1.0  # identity residual init
            shift = b_lin(z)[:, None, None, None]
            x = jax.nn.relu(gamma * conv(x) + shift)
        correction_raw = self.output_conv(x)

        # --- 5. Unnormalize correction ---
        n_out = primitive_state.shape[0] - 3
        correction = correction_raw * scales[:n_out, None, None, None]

        # --- 6. Curl + state update (identical to CorrectorCNN) ---
        omega_bar = correction[-3:, ...]
        bx_interface_correction, by_interface_correction, bz_interface_correction = (
            finite_difference_curl_3D(omega_bar, config.grid_spacing)
        )
        interface_stack = jnp.stack(
            [bx_interface_correction, by_interface_correction, bz_interface_correction],
            axis=0,
        )
        correction = correction.at[-3:].set(interface_stack)

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
        return primitive_state
