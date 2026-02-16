# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

# numerics
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config
import numpy as np
from matplotlib.colors import LogNorm

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

def run_blast_simulation(num_cells):

    # spatial domain
    box_size = 1.0

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = True,
        mhd = True,
        dimensionality = 2,
        box_size = box_size, 
        num_cells = num_cells,
        differentiation_mode = BACKWARDS,
        limiter = MINMOD,
        riemann_solver = HLL,
        exact_end_time = True,
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(
        t_end = 0.2,
        C_cfl = 0.9
    )

    registered_variables = get_registered_variables(config)

    # Grid size and configuration
    num_cells = config.num_cells
    # --- Initial Conditions ---
    grid_spacing = config.box_size / config.num_cells
    x = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)
    y = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    r = helper_data.r

    # Initialize state
    rho = jnp.ones_like(X)
    P = jnp.ones_like(X) * 0.1
    r_inj = 0.1 * box_size
    p_inj = 10.0
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    V_x = jnp.zeros_like(X)
    V_y = jnp.zeros_like(X)

    B_0 = 1 / jnp.sqrt(2)
    B_x = B_0 * jnp.ones_like(X)
    B_y = B_0 * jnp.ones_like(X)
    B_z = jnp.zeros_like(X)

    initial_magnetic_field = jnp.stack([B_x, B_y, B_z], axis=0)

    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = V_x,
        velocity_y = V_y,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        gas_pressure = P
    )

    config = finalize_config(config, initial_state.shape)

    final_state = time_integration(initial_state, config, params, registered_variables)

    return final_state

def downaverage_state(state: jnp.ndarray, target_shape: tuple[int, int]) -> jnp.ndarray:
    """
    Downaverages the spatial dimensions of a state array using block reshaping.

    This function is designed for a state array with the shape
    (NUM_VARS, H, W) and reduces it to (NUM_VARS, h, w) by
    averaging over non-overlapping blocks.

    Args:
        state: The input JAX array with shape (NUM_VARS, H, W).
        target_shape: A tuple (h, w) representing the desired output
                      spatial dimensions. H must be divisible by h, and W
                      must be divisible by w.

    Returns:
        The downaveraged JAX array with shape (NUM_VARS, h, w).
    """
    # 1. Get input and output dimensions
    num_vars, h_in, w_in = state.shape
    h_out, w_out = target_shape

    # 2. Assert that the downsampling is possible (dimensions are divisible)
    if h_in % h_out != 0 or w_in % w_out != 0:
        raise ValueError(
            f"Input shape {(h_in, w_in)} is not divisible by target shape {(h_out, w_out)}"
        )
    
    # 3. Calculate the block size (or downsampling factor)
    h_factor = h_in // h_out
    w_factor = w_in // w_out
    
    # 4. Reshape to create blocks and then take the mean
    # Original shape: (V, H, W)
    # Reshape to: (V, h_out, h_factor, w_out, w_factor)
    # This groups the original grid into blocks.
    reshaped = state.reshape(num_vars, h_out, h_factor, w_out, w_factor)
    
    # Take the mean over the block axes (h_factor and w_factor).
    # The axes are 2 and 4 in the reshaped array.
    downaveraged = reshaped.mean(axis=(2, 4))
    
    return downaveraged

simulation_result_high_res = run_blast_simulation(num_cells=128)
simulation_result_high_res_downsampled = downaverage_state(simulation_result_high_res, target_shape=(64, 64))
simulation_result_low_res = run_blast_simulation(num_cells=128)

# just plot the density with a linear colormap
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(simulation_result_low_res[0], extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
ax[0].set_title('Low Resolution Density')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_aspect('equal')

ax[1].imshow(simulation_result_high_res_downsampled[0], extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
ax[1].set_title('High Resolution Density (Downsampled)')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[1].set_aspect('equal')

ax[2].imshow(simulation_result_high_res[0], extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
ax[2].set_title('High Resolution Density')
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
ax[2].set_aspect('equal')
plt.tight_layout()
plt.savefig('mhd_blast_comparison.png', dpi=300)