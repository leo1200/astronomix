"""
Finite-volume magnetic-field (and coupled velocity) update.

Implements the implicit constrained-transport-style magnetic update of Pang &
Wu (PPCT, arXiv:2410.05173), advancing the magnetic field and the velocity
together with a fixed-point iteration and feeding the change back into the gas
pressure. Supports 2D and 3D (there is no 1D MHD).
"""

# general
from functools import partial

# typing
from jaxtyping import Array, Float, jaxtyped

# jax
import jax
import jax.numpy as jnp
from jax.experimental import checkify

# checkpointed loop
from equinox.internal._loop.checkpointed import checkpointed_while_loop

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    DOUBLE_PRECISION,
    FORWARDS,
    GHOST_CELLS,
    IMPLICIT_EULER,
    IMPLICIT_MIDPOINT,
    MAGNETIC_FIELD_ONLY,
    SINGLE_PRECISION,
    VELOCITY_ONLY,
)

# astronomix containers
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._finite_volume._magnetic_update._vector_maths import (
    cross,
    curl2D,
    curl3D,
    divergence2D,
)
from astronomix._fluid_equations._equations import (
    pressure_from_energy,
    total_energy_from_primitives,
)


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def magnetic_update(
    magnetic_field, gas_state, grid_spacing, dt, registered_variables, config, params
):
    """Update the magnetic field and gas state.

    Based on a simple fixed point iteration, as done in
    Pang, Dongwen, and Kailiang Wu. "Provably Positivity-Preserving
    Constrained Transport (PPCT) Second-Order Scheme for Ideal
    Magnetohydrodynamics." arXiv preprint arXiv:2410.05173 (2024).
    https://arxiv.org/abs/2410.05173, eq. (3.37).

    Args:
        magnetic_field: The magnetic field.
        gas_state: The gas state.
        grid_spacing: The width of the cells.
        dt: The time step.
        registered_variables: The registered variables.
        config: The simulation configuration.

    Returns:
        The updated magnetic field.
    """

    if config.dimensionality == 2:
        # In 2D only the in-plane (x, y) velocity components live on the grid,
        # so we pad to a full 3-vector and fill the first two slots from the
        # gas state; the out-of-plane component stays zero.
        velocity = jnp.zeros((3, *gas_state.shape[1:]), dtype=magnetic_field.dtype)
        velocity = velocity.at[0:2, :, :].set(
            gas_state[
                registered_variables.velocity_index.x : registered_variables.velocity_index.x
                + 2,
                ...,
            ]
        )
    elif config.dimensionality == 3:
        velocity = gas_state[
            registered_variables.velocity_index.x : registered_variables.velocity_index.x
            + 3,
            ...,
        ]
    else:
        raise ValueError("No 1D MHD.")

    density = gas_state[registered_variables.density_index, ...]

    def phi(velocity, magnetic_field):
        # calculate the electric field, electric field = -v x B
        electric_field = cross(magnetic_field, velocity)

        # calculate the curl of the electric field
        if config.dimensionality == 2:
            curl_electric_field = curl2D(electric_field, grid_spacing)
        elif config.dimensionality == 3:
            curl_electric_field = curl3D(electric_field, grid_spacing)
        else:
            raise ValueError("No 1D curl.")

        # calculate the curl of the magnetic field
        if config.dimensionality == 2:
            curl_magnetic_field = curl2D(magnetic_field, grid_spacing)
        elif config.dimensionality == 3:
            curl_magnetic_field = curl3D(magnetic_field, grid_spacing)
        else:
            raise ValueError("No 1D curl.")

        phi1 = curl_electric_field
        phi2 = cross(magnetic_field, curl_magnetic_field) / density

        return phi1, phi2


    if config.boundary_handling == GHOST_CELLS:
        magnetic_field = _boundary_handler(magnetic_field, config, registered_variables, params, MAGNETIC_FIELD_ONLY)
        velocity = _boundary_handler(velocity, config, registered_variables, params, VELOCITY_ONLY)

    B_0 = magnetic_field
    v_0 = velocity

    avg_B = (B_0 + magnetic_field) / 2
    avg_v = (v_0 + velocity) / 2

    phiA, phiB = phi(avg_v, avg_B)

    B_1 = magnetic_field - dt * phiA
    v_1 = velocity - dt * phiB

    if config.boundary_handling == GHOST_CELLS:
        B_1 = _boundary_handler(B_1, config, registered_variables, params, MAGNETIC_FIELD_ONLY)
        v_1 = _boundary_handler(v_1, config, registered_variables, params, VELOCITY_ONLY)

    def while_condition(state):
        B_k, v_k, B_kp1, v_kp1, current_iter = state
        max_change = jnp.maximum(
            jnp.max(jnp.linalg.norm(B_k - B_kp1, axis=0)),
            jnp.max(jnp.linalg.norm(v_k - v_kp1, axis=0)),
        )
        
        if config.numerical_precision == SINGLE_PRECISION:
            change_criterion = 1e-5
        elif config.numerical_precision == DOUBLE_PRECISION:
            change_criterion = 1e-10
        else:
            raise ValueError("Unknown numerical precision.")
        
        return (max_change > change_criterion) & (current_iter < 1000)

    def while_body(state):
        B_k, v_k, B_kp1, v_kp1, current_iter = state

        B_k = B_kp1
        v_k = v_kp1

        if config.fv_magnetic_integrator == IMPLICIT_MIDPOINT:
            avg_B = (B_k + magnetic_field) / 2
            avg_v = (v_k + velocity) / 2
        elif config.fv_magnetic_integrator == IMPLICIT_EULER:
            avg_B = B_k
            avg_v = v_k

        phiA, phiB = phi(avg_v, avg_B)

        B_kp1 = magnetic_field - dt * phiA
        v_kp1 = velocity - dt * phiB

        if config.boundary_handling == GHOST_CELLS:
            B_kp1 = _boundary_handler(B_kp1, config, registered_variables, params, MAGNETIC_FIELD_ONLY)
            v_kp1 = _boundary_handler(v_kp1, config, registered_variables, params, VELOCITY_ONLY)

        return B_k, v_k, B_kp1, v_kp1, current_iter + 1

    if config.differentiation_mode == FORWARDS:
        _, _, B_n, v_n, _ = jax.lax.while_loop(
            while_condition, while_body, (B_0, v_0, B_1, v_1, 0)
        )
    elif config.differentiation_mode == BACKWARDS:
        _, _, B_n, v_n, _ = checkpointed_while_loop(
            while_condition, while_body, (B_0, v_0, B_1, v_1, 0), checkpoints=3
        )

    if config.runtime_debugging:
        avg_B = (B_n + magnetic_field) / 2
        avg_v = (v_n + velocity) / 2

        phiA, phiB = phi(avg_v, avg_B)

        B_n_mod = magnetic_field - dt * phiA
        v_n_mod = velocity - dt * phiB

        checkify.check(
            jnp.sum(jnp.linalg.norm(B_n - B_n_mod, axis=0)) < 1e-8,
            "Eigeniteration for B not converged!",
        )
        checkify.check(
            jnp.sum(jnp.linalg.norm(v_n - v_n_mod, axis=0)) < 1e-8,
            "Eigeniteration for v not converged!",
        )
        checkify.check(
            jnp.all(
                jnp.abs(
                    divergence2D(B_n, grid_spacing) - divergence2D(B_0, grid_spacing)
                )
                < 1e-8
            ),
            "Divergence of magnetic field not conserved, by {divergence}",
            divergence=jnp.max(
                jnp.abs(
                    divergence2D(B_n, grid_spacing) - divergence2D(B_0, grid_spacing)
                )
            ),
        )

    # update the gas energy
    gas_energy = total_energy_from_primitives(
        density,
        jnp.linalg.norm(velocity, axis=0),
        gas_state[registered_variables.pressure_index, ...],
        5 / 3,
    )
    gas_energy_updated = (
        gas_energy
        - 0.5 * density * jnp.linalg.norm(velocity, axis=0) ** 2
        + 0.5 * density * jnp.linalg.norm(v_n, axis=0) ** 2
    )
    pressure_updated = pressure_from_energy(
        gas_energy_updated, density, jnp.linalg.norm(v_n, axis=0), 5 / 3
    )

    # update the gas state
    if config.dimensionality == 2:
        gas_state = gas_state.at[
            registered_variables.velocity_index.x : registered_variables.velocity_index.x
            + 2,
            ...,
        ].set(v_n[:2, ...])
    elif config.dimensionality == 3:
        gas_state = gas_state.at[
            registered_variables.velocity_index.x : registered_variables.velocity_index.x
            + 3,
            ...,
        ].set(v_n)

    gas_state = gas_state.at[registered_variables.pressure_index, ...].set(
        pressure_updated
    )

    return B_n, gas_state
