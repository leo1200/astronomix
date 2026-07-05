# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp

# Enable 64-bit precision for stable finite differences and exact AD
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import os

# astronomix imports based on 1D usage
from astronomix import CARTESIAN
from astronomix import get_helper_data, time_integration, get_registered_variables
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, PERIODIC_ROLL, SimulationConfig, finalize_config,
    PERIODIC_BOUNDARY, BoundarySettings1D, GravityConfig, PositivityConfig
)
from astronomix.option_classes.simulation_params import SimulationParams

def run_1d_adjoint_test():
    # ---------------------------------------------------------
    # 1. Physics & Grid Setup
    # ---------------------------------------------------------
    rho_B = 1.0
    c_s = 2.0
    gamma = 5/3
    P_B = (c_s**2) * rho_B / gamma
    eps = 1e-6  # small perturbation limit
    
    L = 10.0
    Nx = 400
    t_end = 1.0
    dx = L / Nx
    
    # Setup 1D configuration
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        geometry=CARTESIAN,
        memory_analysis=False,
        progress_bar=True,
        boundary_handling=PERIODIC_ROLL,
        differentiation_mode=BACKWARDS,
        num_ghost_cells=0,
        mhd=False,
        dimensionality=1,
        box_size=L,
        num_cells=Nx,
        boundary_settings=BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        return_snapshots=False, # We only need the final state for the gradient
    )
    
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    
    params = SimulationParams(
        C_cfl=0.4, # Strict CFL for clear, crisp wave propagation
        t_end=t_end,
        gamma=gamma,
        gravitational_constant=0.0
    )
    
    # Get 1D coordinates and squeeze any trailing dimensions
    x_1d = jnp.squeeze(helper_data.geometric_centers)

    # ---------------------------------------------------------
    # 2. Define the Differentiable Simulator
    # ---------------------------------------------------------
    def cost_fn_1d(rho_P0, vx_P0):
        """ Evaluates L2 norm of the perturbation fields at target time t_end """
        
        # Construct full state (Coupling P to rho to enforce purely acoustic waves)
        rho = rho_B + rho_P0
        vx = vx_P0
        p = P_B + (c_s**2) * rho_P0
        
        initial_state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=rho,
            velocity_x=vx,
            gas_pressure=p,
        )
        
        # Finalize config internally for the time integrator
        config_final = finalize_config(config, initial_state.shape)
        
        # Run forward simulation
        final_state = time_integration(
            initial_state, config_final, params, registered_variables
        )
        
        # Extract perturbation fields from the final state tensor
        final_rho_P = final_state[registered_variables.density_index] - rho_B
        final_vx_P  = final_state[registered_variables.velocity_index]
        
        # Objective: J = 0.5 * integral(rho_P^2 + v_xP^2) dx
        J = 0.5 * jnp.sum(final_rho_P**2 + final_vx_P**2) * dx
        return J

    # ---------------------------------------------------------
    # 3. Generate Initial Perturbations (Gaussian Pulse)
    # ---------------------------------------------------------
    def rho_P0_func(x_pos):
        return eps * jnp.exp(- (x_pos - L/2)**2 / (2 * 0.5**2))
        
    def vx_P0_func(x_pos):
        return jnp.zeros_like(x_pos)
        
    rho_P0_initial = rho_P0_func(x_1d)
    vx_P0_initial  = vx_P0_func(x_1d)
    
    # ---------------------------------------------------------
    # 4. Compute Automatic Differentiation (AD) Gradients
    # ---------------------------------------------------------
    print("Computing JAX AD gradients backward through simulation...")
    # argnums=(0,1) tells JAX to differentiate w.r.t rho_P0 and vx_P0
    grad_fn = jax.grad(cost_fn_1d, argnums=(0, 1)) 
    ad_grad_rho_raw, ad_grad_vx_raw = grad_fn(rho_P0_initial, vx_P0_initial)
    
    # Convert JAX discrete gradient back to a continuous spatial Fréchet derivative
    ad_grad_rho = ad_grad_rho_raw / dx
    ad_grad_vx  = ad_grad_vx_raw / dx
    
    # ---------------------------------------------------------
    # 5. Evaluate the Exact Analytical Gradients
    # ---------------------------------------------------------
    print("Evaluating exact analytical gradients...")
    def analytical_gradients(x_arr, t):
        # We wrap queries properly to simulate periodic waves
        def get_rho(x_pos): return rho_P0_func(jnp.mod(x_pos, L))
        def get_vx(x_pos): return vx_P0_func(jnp.mod(x_pos, L))
            
        x_plus = x_arr + 2 * c_s * t
        x_minus = x_arr - 2 * c_s * t
        
        term1_rho = 0.5 * (1 + c_s**2 / rho_B**2) * get_rho(x_arr)
        term2_rho = 0.25 * (1 - c_s**2 / rho_B**2) * (get_rho(x_plus) + get_rho(x_minus))
        term3_rho = -0.25 * (rho_B / c_s - c_s / rho_B) * (get_vx(x_plus) - get_vx(x_minus))
        
        grad_rho = term1_rho + term2_rho + term3_rho
        
        term1_vx = 0.5 * (1 + rho_B**2 / c_s**2) * get_vx(x_arr)
        term2_vx = 0.25 * (1 - rho_B**2 / c_s**2) * (get_vx(x_plus) + get_vx(x_minus))
        term3_vx = -0.25 * (c_s / rho_B - rho_B / c_s) * (get_rho(x_plus) - get_rho(x_minus))
        
        grad_vx = term1_vx + term2_vx + term3_vx
        
        return grad_rho, grad_vx

    ana_grad_rho, ana_grad_vx = analytical_gradients(x_1d, t_end)
    
    # Normalize everything by eps to make the plot scale O(1)
    ad_grad_rho /= eps
    ana_grad_rho /= eps
    ad_grad_vx /= eps
    ana_grad_vx /= eps
    
    # ---------------------------------------------------------
    # 6. Plotting
    # ---------------------------------------------------------
    os.makedirs("figures", exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    axs[0].plot(x_1d, ad_grad_rho, 'o', color='blue', label='JAX AD Gradient', markersize=4, alpha=0.7)
    axs[0].plot(x_1d, ana_grad_rho, '-', color='orange', label='Analytical Expression', linewidth=2)
    axs[0].set_title(r'Sensitivity to initial density: $\partial_{U_0} J$ for $\rho$ (scaled by $1/\epsilon$)')
    axs[0].set_ylabel('Gradient Amplitude')
    axs[0].legend()
    axs[0].grid(True, linestyle='--', alpha=0.6)
    
    axs[1].plot(x_1d, ad_grad_vx, 'o', color='green', label='JAX AD Gradient', markersize=4, alpha=0.7)
    axs[1].plot(x_1d, ana_grad_vx, '-', color='red', label='Analytical Expression', linewidth=2)
    axs[1].set_title(r'Sensitivity to initial velocity: $\partial_{U_0} J$ for $v_x$ (scaled by $1/\epsilon$)')
    axs[1].set_xlabel('Spatial Coordinate $x$')
    axs[1].set_ylabel('Gradient Amplitude')
    axs[1].legend()
    axs[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_path = 'figures/euler_adjoint_sensitivities_1d.svg'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Success! Plot saved to {plot_path}")

if __name__ == "__main__":
    run_1d_adjoint_test()