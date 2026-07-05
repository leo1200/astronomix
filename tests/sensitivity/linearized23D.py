# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp

# Enable 64-bit precision for exact gradients and stable numerics
jax.config.update("jax_enable_x64", True)

# Debug nans
jax.config.update("jax_debug_nans", True)

import matplotlib.pyplot as plt
import os

from astronomix import CARTESIAN
from astronomix import get_helper_data, time_integration, get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, FINITE_VOLUME, PERIODIC_ROLL, SimulationConfig, finalize_config,
    PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    StaticFloatVector, StaticIntVector, GravityConfig, PositivityConfig
)
from astronomix.option_classes.simulation_params import SimulationParams

def run_nd_adjoint_test(dim=2):
    print(f"\n{'='*40}\nRunning {dim}D Adjoint Sensitivity Test\n{'='*40}")
    
    # ---------------------------------------------------------
    # 1. Physics & Grid Setup
    # ---------------------------------------------------------
    rho_B = 1.0
    c_s = 2.0
    gamma = 5/3
    P_B = (c_s**2) * rho_B / gamma
    eps = 1e-6  
    
    L = 1.0
    
    # Configure resolution, wavevector, and grid shapes based on dimensionality
    if dim == 2:
        Nx, Ny = 64, 64
        k_vec = jnp.array([2 * jnp.pi * 2, 2 * jnp.pi * 4])
        
        box_size = StaticFloatVector(x=L, y=L)
        num_cells = StaticIntVector(x=Nx, y=Ny)
        
        boundary_settings = BoundarySettings(
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
        )
        
        dV = (L / Nx) * (L / Ny)
        
    elif dim == 3:
        Nx, Ny, Nz = 32, 32, 32
        k_vec = jnp.array([2 * jnp.pi * 2, 2 * jnp.pi * 4, 2 * jnp.pi * 4])
        
        box_size = StaticFloatVector(x=L, y=L, z=L)
        num_cells = StaticIntVector(x=Nx, y=Ny, z=Nz)
        
        boundary_settings = BoundarySettings(
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        )
        
        dV = (L / Nx) * (L / Ny) * (L / Nz)
    else:
        raise ValueError("Only dim=2 or dim=3 supported in this test.")
    
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        geometry=CARTESIAN,
        progress_bar=True,
        boundary_handling=PERIODIC_ROLL,
        differentiation_mode=BACKWARDS,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=dim,
        box_size=box_size,
        num_cells=num_cells,
        boundary_settings=boundary_settings,
        return_snapshots=False,
    )
    
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    
    # Wave properties
    k_mag = jnp.linalg.norm(k_vec)
    k_hat = k_vec / k_mag
    omega = c_s * k_mag
    t_end = 0.02  # Run for a fraction of a wave period
    
    params = SimulationParams(
        C_cfl=0.4,
        t_end=t_end,
        gamma=gamma,
        gravitational_constant=0.0
    )

    # ---------------------------------------------------------
    # 2. Extract Coordinates and Phase
    # ---------------------------------------------------------
    if dim == 2:
        # geometric_centers shape: (Nx, Ny, 2)
        coords = jnp.transpose(helper_data.geometric_centers, (2, 0, 1))
        X, Y = coords[0], coords[1]
        phase = k_vec[0] * X + k_vec[1] * Y
    elif dim == 3:
        # geometric_centers shape: (Nx, Ny, Nz, 3)
        coords = jnp.transpose(helper_data.geometric_centers, (3, 0, 1, 2))
        X, Y, Z = coords[0], coords[1], coords[2]
        phase = k_vec[0] * X + k_vec[1] * Y + k_vec[2] * Z

    # ---------------------------------------------------------
    # 3. Define the Differentiable Simulator (Dimension Specific)
    # ---------------------------------------------------------
    if dim == 2:
        def cost_fn(rho_P0, vx_P0, vy_P0):
            rho = rho_B + rho_P0
            p = P_B + (c_s**2) * rho_P0
            
            initial_state = construct_primitive_state(
                config=config,
                registered_variables=registered_variables,
                density=rho,
                velocity_x=vx_P0,
                velocity_y=vy_P0,
                gas_pressure=p,
            )
            
            config_final = finalize_config(config, initial_state.shape)
            final_state = time_integration(initial_state, config_final, params, registered_variables)
            
            final_rho_P = final_state[registered_variables.density_index] - rho_B
            final_vx_P  = final_state[registered_variables.velocity_index.x]
            final_vy_P  = final_state[registered_variables.velocity_index.y]
            
            J = 0.5 * jnp.sum(final_rho_P**2 + final_vx_P**2 + final_vy_P**2) * dV
            return J

        grad_fn = jax.grad(cost_fn, argnums=(0, 1, 2))

    elif dim == 3:
        def cost_fn(rho_P0, vx_P0, vy_P0, vz_P0):
            rho = rho_B + rho_P0
            p = P_B + (c_s**2) * rho_P0
            
            initial_state = construct_primitive_state(
                config=config,
                registered_variables=registered_variables,
                density=rho,
                velocity_x=vx_P0,
                velocity_y=vy_P0,
                velocity_z=vz_P0,
                gas_pressure=p,
            )
            
            config_final = finalize_config(config, initial_state.shape)
            final_state = time_integration(initial_state, config_final, params, registered_variables)
            
            final_rho_P = final_state[registered_variables.density_index] - rho_B
            final_vx_P  = final_state[registered_variables.velocity_index.x]
            final_vy_P  = final_state[registered_variables.velocity_index.y]
            final_vz_P  = final_state[registered_variables.velocity_index.z]

            J = 0.5 * jnp.sum(final_rho_P**2 + final_vx_P**2 + final_vy_P**2 + final_vz_P**2) * dV
            return J

        grad_fn = jax.grad(cost_fn, argnums=(0, 1, 2, 3))

    # ---------------------------------------------------------
    # 4. Generate Initial Perturbations and Compute AD Gradients
    # ---------------------------------------------------------
    print("Computing JAX AD gradients...")
    rho_P0_init = eps * jnp.sin(phase)
    vx_P0_init  = jnp.zeros_like(rho_P0_init)
    vy_P0_init  = jnp.zeros_like(rho_P0_init)

    if dim == 2:
        g_rho_raw, g_vx_raw, g_vy_raw = grad_fn(rho_P0_init, vx_P0_init, vy_P0_init)
        
        # Scale to continuous Fréchet derivative
        ad_grad_rho = g_rho_raw / dV
        ad_grad_vx  = g_vx_raw / dV
        ad_grad_vy  = g_vy_raw / dV
        
        # Project AD velocity gradients onto longitudinal wave direction
        ad_grad_v_long = ad_grad_vx * k_hat[0] + ad_grad_vy * k_hat[1]
        
    elif dim == 3:
        vz_P0_init = jnp.zeros_like(rho_P0_init)
        g_rho_raw, g_vx_raw, g_vy_raw, g_vz_raw = grad_fn(rho_P0_init, vx_P0_init, vy_P0_init, vz_P0_init)
        
        # Scale to continuous Fréchet derivative
        ad_grad_rho = g_rho_raw / dV
        ad_grad_vx  = g_vx_raw / dV
        ad_grad_vy  = g_vy_raw / dV
        ad_grad_vz  = g_vz_raw / dV
        
        # Project AD velocity gradients onto longitudinal wave direction
        ad_grad_v_long = ad_grad_vx * k_hat[0] + ad_grad_vy * k_hat[1] + ad_grad_vz * k_hat[2]
        
    # Normalize by eps for O(1) plotting
    ad_grad_rho /= eps
    ad_grad_v_long /= eps

    # ---------------------------------------------------------
    # 5. Evaluate the Exact Analytical Gradients
    # ---------------------------------------------------------
    print("Evaluating analytical expressions...")
    term_rho_1 = 0.5 * (1 + c_s**2 / rho_B**2)
    term_rho_2 = 0.5 * (1 - c_s**2 / rho_B**2) * jnp.cos(2 * omega * t_end)
    ana_grad_rho = jnp.sin(phase) * (term_rho_1 + term_rho_2)
    
    term_v = -0.5 * (c_s / rho_B - rho_B / c_s) * jnp.sin(2 * omega * t_end)
    ana_grad_v_long = jnp.cos(phase) * term_v
    
    # ---------------------------------------------------------
    # 6. Plot 1D Collapse Slices
    # ---------------------------------------------------------
    # Flatten arrays to collapse N-dimensional points against the 1D phase profile
    phase_1d = phase.flatten() / (2 * jnp.pi)  # Normalize phase to cycle numbers
    ad_rho_1d = ad_grad_rho.flatten()
    ad_v_1d = ad_grad_v_long.flatten()
    ana_rho_1d = ana_grad_rho.flatten()
    ana_v_1d = ana_grad_v_long.flatten()
    
    # Sort for the analytical line plot to draw cleanly
    sort_idx = jnp.argsort(phase_1d)
    phase_sorted = phase_1d[sort_idx]
    ana_rho_sorted = ana_rho_1d[sort_idx]
    ana_v_sorted = ana_v_1d[sort_idx]
    
    os.makedirs("figures", exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Density Gradient Plot
    axs[0].plot(phase_sorted, ana_rho_sorted, '-', color='orange', linewidth=3, label='Analytical Expression')
    axs[0].scatter(phase_1d, ad_rho_1d, s=20, color='blue', alpha=0.5, label=f'JAX AD Gradients ({dim}D points)')
    
    # Note the raw f-string `rf"..."` to avoid \r parsing issues
    axs[0].set_title(rf'{dim}D Sensitivity to initial density: $\partial_{{U_0}} J$ for $\rho$')
    axs[0].set_ylabel('Gradient Amplitude')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    # Velocity Gradient Plot
    axs[1].plot(phase_sorted, ana_v_sorted, '-', color='red', linewidth=3, label='Analytical Expression')
    axs[1].scatter(phase_1d, ad_v_1d, s=20, color='green', alpha=0.5, label=f'JAX AD Gradients ({dim}D points)')
    
    axs[1].set_title(rf'{dim}D Sensitivity to initial velocity (longitudinal): $\partial_{{U_0}} J$ for $v_{{\parallel}}$')
    axs[1].set_xlabel(r'Wave Phase cycles ($\vec{k} \cdot \vec{x} / 2\pi$)')
    axs[1].set_ylabel('Gradient Amplitude')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = f'figures/euler_adjoint_sensitivities_{dim}d.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Success! {dim}D plot saved to {plot_path}")

if __name__ == "__main__":
    run_nd_adjoint_test(dim=2)
    run_nd_adjoint_test(dim=3)