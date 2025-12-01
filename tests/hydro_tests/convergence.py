# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, HLLC, BoundarySettings1D

# 64-bit floating point precision
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# constants and options
from astronomix import CARTESIAN, PERIODIC_BOUNDARY
from astronomix import SimulationConfig, SimulationParams

# simulation setup and execution
from astronomix import get_helper_data, finalize_config, get_registered_variables, time_integration

# plotting
import matplotlib.pyplot as plt

# ===================================================
# ============== ↓ astronomix simulation ↓ =============
# ===================================================

# -- Simulation Parameters from the Specification --
box_size = 1.0
advection_velocity = 1.0
t_end = 2.0  # Two full rotations
gamma_val = 1.4
pressure_val = 10.0

params = SimulationParams(t_end=t_end, gamma=gamma_val, C_cfl=0.1)

# -- Gaussian Pulse Parameters from the Specification --
pulse_center_initial = 0.5
pulse_width = 0.0625

def get_exact_solution(r, t):
    """Calculates the exact solution for the specified Gaussian pulse."""
    final_center = (pulse_center_initial + advection_velocity * t) % box_size
    distance = jnp.abs(r - final_center)
    return 1.0 + jnp.exp(-distance**2 / (2 * pulse_width**2))

def simulate_and_get_error(num_cells, use_first_order):
    """
    Runs the Gaussian pulse advection simulation and returns the L1 error.
    Switches between 1st and 2nd order based on the `use_first_order` flag.
    """
    config = SimulationConfig(
        geometry=CARTESIAN,
        boundary_settings=BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        # Use first_order_fallback to select the order of the scheme
        first_order_fallback=use_first_order,
        riemann_solver=HLLC,
        limiter=DOUBLE_MINMOD,  # Limiter is ignored if first_order_fallback=True
        box_size=box_size,
        num_cells=num_cells,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)
    r_centers = helper_data.geometric_centers

    # -- Initial Conditions --
    rho = get_exact_solution(r_centers, t=0.0)
    u = jnp.full_like(r_centers, advection_velocity)
    p = jnp.full_like(r_centers, pressure_val)

    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=u, gas_pressure=p
    )

    config = finalize_config(config, initial_state.shape)
    final_state = time_integration(initial_state, config, params, registered_variables)
    rho_final = final_state[registered_variables.density_index]
    
    # -- Error Calculation --
    rho_exact = get_exact_solution(r_centers, t=t_end)
    dx = box_size / num_cells
    l1_error = jnp.sum(jnp.abs(rho_final - rho_exact)) * dx

    return r_centers, rho_final, rho_exact, l1_error

# ===================================================
# ============== ↑ astronomix simulation ↑ =============
# ===================================================

# --- Convergence Study ---
resolutions = [64, 128, 256, 512, 1024, 2048]
l1_errors_1st = []
l1_errors_2nd = []

print("Running 1st and 2nd Order Gaussian Pulse Advection Test...")
for N in resolutions:
    print(f"  Simulating with {N} cells...")
    # First-order simulation
    _, _, _, error_1st = simulate_and_get_error(N, use_first_order=True)
    l1_errors_1st.append(error_1st)
    
    # Second-order simulation
    _, _, _, error_2nd = simulate_and_get_error(N, use_first_order=False)
    l1_errors_2nd.append(error_2nd)

l1_errors_1st = jnp.array(l1_errors_1st)
l1_errors_2nd = jnp.array(l1_errors_2nd)

convergence_rates_1st = jnp.log2(l1_errors_1st[:-1] / l1_errors_1st[1:])
convergence_rates_2nd = jnp.log2(l1_errors_2nd[:-1] / l1_errors_2nd[1:])

print("\n--- Results ---")
print("Resolution | L1 Error (1st) | Rate (1st) | L1 Error (2nd) | Rate (2nd)")
print("--------------------------------------------------------------------------")
print(f"{resolutions[0]:<10} | {l1_errors_1st[0]:<14.4e} |            | {l1_errors_2nd[0]:<14.4e} |")
for i in range(len(convergence_rates_1st)):
    print(f"{resolutions[i+1]:<10} | {l1_errors_1st[i+1]:<14.4e} | {convergence_rates_1st[i]:<10.2f} | {l1_errors_2nd[i+1]:<14.4e} | {convergence_rates_2nd[i]:.2f}")


# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"1st vs 2nd Order Gaussian Pulse Advection (t = {t_end}, v$_{{adv}}$ = {advection_velocity})", fontsize=16)

# Plot 1: Final Density Profile
ax1.set_title(f"density profile at t = {t_end} (N = {resolutions[-1]})")
r_fine, rho_1st_fine, rho_exact_fine, _ = simulate_and_get_error(resolutions[-1], use_first_order=True)
_, rho_2nd_fine, _, _ = simulate_and_get_error(resolutions[-1], use_first_order=False)

ax1.plot(r_fine, rho_exact_fine, 'k-', label='exact solution')
ax1.plot(r_fine, rho_1st_fine, 'b--', label='1st order (simulation)')
ax1.plot(r_fine, rho_2nd_fine, 'r:', label='2nd order (simulation)')
ax1.set_xlabel("position")
ax1.set_ylabel("density")
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: L1 Error vs. Resolution
ax2.set_title("L1 error vs. resolution")
ax2.loglog(resolutions, l1_errors_1st, 'bo-', label='1st order (simulation)')
ax2.loglog(resolutions, l1_errors_2nd, 'rs-', label='2nd order (simulation)')

# Reference lines
first_order_ref = l1_errors_1st[0] * (resolutions[0] / jnp.array(resolutions))
ax2.loglog(resolutions, first_order_ref, 'k:', alpha=0.7, label='1st order (ideal)')
second_order_ref = l1_errors_2nd[0] * (resolutions[0] / jnp.array(resolutions))**2
ax2.loglog(resolutions, second_order_ref, 'k--', alpha=0.7, label='2nd order (ideal)')

ax2.set_xlabel("resolution in number of cells")
ax2.set_ylabel("L1 error")
ax2.legend()
ax2.grid(True, which="both", linestyle=':', alpha=0.6)
ax2.set_xticks(resolutions)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())

plt.tight_layout()
plt.savefig('figures/convergence.pdf', bbox_inches='tight')
print("\nPlots saved to figures/convergence.pdf")