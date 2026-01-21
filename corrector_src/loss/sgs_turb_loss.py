import jax
import jax.numpy as jnp
from jf1uids.fluid_equations.fluid import (
    get_absolute_velocity,
    total_energy_from_primitives,
)
from jf1uids.fluid_equations.registered_variables import RegisteredVariables
from jf1uids.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
    SimulationConfig,
)
from jf1uids.option_classes.simulation_params import SimulationParams
import Pk_library as PKL
from jf1uids._physics_modules._mhd._vector_maths import divergence3D
from jf1uids.data_classes.simulation_helper_data import get_helper_data
from jf1uids.fluid_equations.total_quantities import (
    calculate_total_energy,
)
from corrector_src.utils.power_spectra_1d import pk_jax_1d
import numpy as np
from functools import partial


def mse_loss(predicted_state: jnp.ndarray, ground_truth: jnp.ndarray) -> float:
    """
    Default MSE loss function for ground truth training.

    Args:
        predicted_state: Current simulation state
        ground_truth:
        training_config: Training configuration containing ground truth data

    Returns:
        MSE loss value
    """

    # Compute MSE loss
    loss = jnp.mean(jnp.square(predicted_state - ground_truth))

    return loss


def get_energy(
    primitive_state: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
):
    """
    Calculate the total energy from the primitive state. The resolution doesnt matter, only the dimensionality
    """
    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    return total_energy_from_primitives(rho, u, p, params.gamma)


def spectral_energy_loss(
    predicted_state: jnp.ndarray,
    ground_truth: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> float:
    """
    Important caveat: predicted and ground truth must be same size, also same box size
    """
    use_log_ratio = True
    energy_predicted = get_energy(predicted_state, config, registered_variables, params)
    energy_gt = get_energy(ground_truth, config, registered_variables, params)

    k1D_pred, Pk1D_pred, Nmodes1D_pred = pk_jax_1d(
        energy_predicted, BoxSize=1.0, axis=0
    )

    k1D_gt, Pk1D_gt, Nmodes1D_gt = pk_jax_1d(energy_gt, BoxSize=1.0, axis=0)
    if use_log_ratio:
        # Avoid division by zero or log of 0
        eps = 1e-12
        P_pred = jnp.clip(Pk1D_pred, eps, None)
        P_true = jnp.clip(Pk1D_gt, eps, None)

        # Compute spectral log-ratio loss
        log_ratio = jnp.log(P_pred) - jnp.log(P_true)
        loss = jnp.sqrt(jnp.mean(log_ratio**2))
    else:
        loss = jnp.mean((Pk1D_pred - Pk1D_gt) ** 2)

    return loss


def rate_of_strain(
    state: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    velocity = state[registered_variables.velocity_index, ...]
    dx = config.grid_spacing
    # Compute ∂v_i/∂x_j using jnp.gradient
    dvx_dx, dvx_dy, dvx_dz = jnp.gradient(velocity[0], dx)
    dvy_dx, dvy_dy, dvy_dz = jnp.gradient(velocity[1], dx)
    dvz_dx, dvz_dy, dvz_dz = jnp.gradient(velocity[2], dx)

    grad_v = jnp.stack(
        [
            jnp.stack([dvx_dx, dvx_dy, dvx_dz]),
            jnp.stack([dvy_dx, dvy_dy, dvy_dz]),
            jnp.stack([dvz_dx, dvz_dy, dvz_dz]),
        ]
    )  # shape (3, 3, Nx, Ny, Nz)
    rate_of_strain = 0.5 * (grad_v + jnp.swapaxes(grad_v, 0, 1))
    return rate_of_strain


def rate_of_strain_loss(
    predicted_state: jnp.ndarray,
    ground_truth: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> float:
    strain_predicted = rate_of_strain(
        state=predicted_state, config=config, registered_variables=registered_variables
    )
    strain_gt = rate_of_strain(
        state=ground_truth, config=config, registered_variables=registered_variables
    )
    loss = mse_loss(strain_predicted, strain_gt)
    return loss


# def mean_flow_loss() need to think this one trough as the average over time, i think it should be covered by the rolling
# if i wanted to implement it, id have to play with the time integration fs (being fair they dont implement it in the cambridge 2022 paper)


def energy_loss(
    predicted_state: jnp.ndarray,
    ground_truth: jnp.ndarray,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> float:
    helper_data = get_helper_data(config)
    energy_predicted = calculate_total_energy(
        predicted_state,
        helper_data,
        params.gamma,
        params.gravitational_constant,
        config,
        registered_variables,
    )
    energy_ground_truth = calculate_total_energy(
        ground_truth,
        helper_data,
        params.gamma,
        params.gravitational_constant,
        config,
        registered_variables,
    )
    return energy_predicted - energy_ground_truth


def make_loss_function(cfg_training):
    """Builds a pure JAX-compatible loss function using values from cfg_training.
    Returns
        loss_function
        make_total_loss_from_components function
        active_loss_indices"""

    # Define loss components as a dict of {name: (weight, fn)}
    loss_fns = {
        "mse": (
            cfg_training["mse_loss"],
            lambda pred, gt, config, registered_vars, params: mse_loss(pred, gt),
        ),
        "strain": (
            cfg_training["rate_of_strain_loss"],
            lambda pred, gt, config, registered_vars, params: rate_of_strain_loss(
                pred, gt, config, registered_vars
            ),
        ),
        "spectral": (
            cfg_training["spectral_energy_loss"],
            lambda pred, gt, config, registered_vars, params: spectral_energy_loss(
                pred, gt, config, registered_vars, params
            ),
        ),
    }
    active_loss_indices = {
        i: (name.replace("_loss", ""), w)
        for i, (name, (w, _)) in enumerate(loss_fns.items())
        if w != 0
    }
    active_weights = {
        i: w for i, (name, (w, _)) in enumerate(loss_fns.items()) if w != 0
    }

    def compute_loss_from_components(loss_components):
        # need to make this for more than 1 loss lol
        if len(loss_components.shape) == 1:
            total_loss = 0.0
            for i, weight in active_weights.items():
                total_loss += loss_components[i] * weight

        else:
            total_loss = np.zeros(loss_components.shape[0])
            for j in range(loss_components.shape[0]):
                for i, weight in active_weights.items():
                    total_loss[j] += loss_components[j, i] * weight
        return total_loss

    @partial(jax.jit, static_argnames=["config", "registered_variables"])
    def loss_function(predicted, ground_truth, config, registered_variables, params):
        total = 0.0
        components = {}

        for name, (weight, fn) in loss_fns.items():
            if weight > 0:
                val = fn(predicted, ground_truth, config, registered_variables, params)
                components[name] = val
                total += weight * val

        # for name, value in components.items():
        #    jax.debug.print("{name}: {value}", name=name, value=value)

        return total, components

    return loss_function, compute_loss_from_components, active_loss_indices
