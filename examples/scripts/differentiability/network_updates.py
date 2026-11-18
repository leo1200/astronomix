"""
Learned per-step updates for the differentiable solver.

Two small Equinox networks, each with the function that applies it once per
time step through ``config.per_step_update``:

- ``CorrectorCNN`` and ``cnn_mhd_corrector_update``: a learned correction of
  the primitive state of a 2D MHD run (used by ``solver_in_the_loop.py``). The
  magnetic-field part of the correction is the curl of a learned electric
  field, so the divergence-free constraint on the magnetic field is preserved
  by construction (the divergence of a curl is zero).
- ``ForceNet`` and ``neural_net_force_update``: a body force ``(Fx, Fy)`` as a
  function of position and time ``(x, y, t)``, applied as a velocity increment
  to the interior cells of a 2D run.

The trainable weights are read from ``params.per_step_update_params``, so they
can be learned through the differentiable solver; the static part of the
network (the standard Equinox filter / combine split) is closed over.
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

# astronomix functions
from astronomix._spatial_operators._vector_maths import curl2D


# -------------------------------------------------------------
# ================= ↓ CNN MHD corrector ↓ =====================
# -------------------------------------------------------------


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


def cnn_mhd_corrector_update(network_static):
    """
    Build the per-step update that applies the learned CNN correction.

    Args:
        network_static: The static part of a ``CorrectorCNN``, split off via
            ``eqx.partition(model, eqx.is_array)``. The trainable part is read
            from ``params.per_step_update_params``.

    Returns:
        The function to pass as ``config.per_step_update``.
    """

    @partial(jax.jit, static_argnames=["config", "registered_variables"])
    def per_step_update(
        primitive_state, dt, current_time, config, params, helper_data, registered_variables
    ):
        model = eqx.combine(params.per_step_update_params, network_static)

        correction = model(primitive_state)

        # To avoid introducing divergence errors in the magnetic field, the network
        # predicts a correction for the electric field and we take its curl: the
        # divergence of a curl is zero, so the magnetic field stays divergence-free.
        electric_field_correction = correction[-3:, ...]
        magnetic_field_correction = curl2D(electric_field_correction, config.grid_spacing)
        correction = correction.at[-3:, ...].set(magnetic_field_correction)

        # Apply the correction over the time step.
        primitive_state = primitive_state + correction * dt

        # Keep the pressure above a small floor so the correction cannot drive it
        # non-positive.
        minimum_pressure = 1e-4
        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.maximum(primitive_state[registered_variables.pressure_index], minimum_pressure)
        )

        return primitive_state

    return per_step_update


# -------------------------------------------------------------
# ================= ↓ Neural-network force ↓ ==================
# -------------------------------------------------------------


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


def neural_net_force_update(network_static, positions):
    """
    Build the per-step update that applies the neural-network body force.

    The network is evaluated at every interior cell position and the current
    time, and the resulting force is integrated over one time step into the
    velocity.

    Args:
        network_static: The static part of a ``ForceNet``, split off via
            ``eqx.partition(model, eqx.is_array)``. The trainable part is read
            from ``params.per_step_update_params``.
        positions: The centres of the interior cells, of shape ``(N, N, 2)``
            (``get_helper_data(config).geometric_centers``).

    Returns:
        The function to pass as ``config.per_step_update``.
    """

    @partial(jax.jit, static_argnames=["config", "registered_variables"])
    def per_step_update(
        primitive_state, dt, current_time, config, params, helper_data, registered_variables
    ):
        model = eqx.combine(params.per_step_update_params, network_static)

        # TODO: this assumes the same number of cells in x and y; generalise to
        # non-square grids.
        num_cells_per_side = config.num_cells.x
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
        ].add(forces[0] * dt)
        primitive_state = primitive_state.at[
            registered_variables.velocity_index.y,
            num_ghost_cells:-num_ghost_cells,
            num_ghost_cells:-num_ghost_cells,
        ].add(forces[1] * dt)

        return primitive_state

    return per_step_update
