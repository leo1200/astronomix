# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

import jax.random as jr

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# astronomix data structures
from astronomix import SimulationConfig
from astronomix import SimulationParams

# astronomix constants
from astronomix.option_classes.simulation_config import AM_HLLC, CARTESIAN, HLLC, HLLC_LM, HYBRID_HLLC, MUSCL, SPHERICAL, HLL, MINMOD, SPLIT


# astronomix functions
from astronomix import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config

from astronomix import time_integration

# --- Additional imports for analysis and plotting ---
# For radially averaging the simulation data
from scipy.stats import binned_statistic
# For the exact Sedov-Taylor solution
from exactpack.solvers.sedov.sedov import Sedov


config = SimulationConfig(

    
    geometry = CARTESIAN,
    progress_bar = True,
    runtime_debugging = False,

    riemann_solver = HYBRID_HLLC,
    # split = SPLIT,
    # time_integrator = MUSCL,
    
    dimensionality = 3,

    exact_end_time = True,

    # ====== RESOLUTION =======
    num_cells = 256,
    # =========================

    # ==== snapshotting ====
    return_snapshots = False,
    # ======================

)


params = SimulationParams(
    t_end = 0.1
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# total explosion energy
E_explosion = 1.0

E_gas = E_explosion

# Ambient (background) physical conditions (adjust as needed)
rho_ambient  = 1.0         # typical ISM density
p_ambient    = 1e-4          # low gas pressure

# Pressures in code units
p_ambient = p_ambient

# --- Set Up the Explosion Injection Region ---

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_ambient

u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

# currently, we take 10 injection cells
r_explosion = 0.02

# Compute the injection volume (spherical volume in code units)
injection_volume = (4/3) * jnp.pi * r_explosion**3

# Adiabatic indices:
gamma_gas = params.gamma   # for the thermal gas
gamma_cr  = 4/3   # for cosmic rays

# The energy contained in a uniform pressure region is related by:
#   E = p * V / (gamma - 1)
# Hence, the effective explosion pressure in the injection region (in code units)
p_explosion_gas = E_gas * (gamma_gas - 1) / injection_volume

# Convert to code units
p_explosion_gas = p_explosion_gas

# --- Define the Radial Profiles ---
# Get the radial coordinate array (assumed already available)
r = helper_data.r

# Gas pressure: high within the explosion region, ambient elsewhere
p_gas = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p_gas
)

config = finalize_config(config, initial_state.shape)

# Run the simulation
result = time_integration(initial_state, config, params, registered_variables)


# =========================================================================
# === Analysis and Plotting ===============================================
# =========================================================================

print("Simulation finished. Starting analysis and plotting...")

# --- Parameters ---
num_bins = 200
num_scatter_samples = 100_000
key = jr.PRNGKey(42)  # Random seed for reproducibility

# --- Flatten simulation data ---
r_flat = helper_data.r.flatten()
rho_flat = result[registered_variables.density_index].flatten()
p_flat = result[registered_variables.pressure_index].flatten()
vx = result[registered_variables.velocity_index.x].flatten()
vy = result[registered_variables.velocity_index.y].flatten()
vz = result[registered_variables.velocity_index.z].flatten()
v_abs_flat = jnp.sqrt(vx**2 + vy**2 + vz**2)

domain_max_r = jnp.max(r_flat)

# --- Radial Binning (in JAX) ---
bins = jnp.linspace(0, domain_max_r, num_bins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])

bin_indices = jnp.searchsorted(bins, r_flat, side='right') - 1
bin_indices = jnp.clip(bin_indices, 0, num_bins - 1)

rho_sum = jnp.zeros(num_bins)
v_sum = jnp.zeros(num_bins)
p_sum = jnp.zeros(num_bins)
counts = jnp.zeros(num_bins)

rho_sum = rho_sum.at[bin_indices].add(rho_flat)
v_sum = v_sum.at[bin_indices].add(v_abs_flat)
p_sum = p_sum.at[bin_indices].add(p_flat)
counts = counts.at[bin_indices].add(1)

counts = jnp.where(counts == 0, 1, counts)
mean_rho = rho_sum / counts
mean_v_abs = v_sum / counts
mean_p = p_sum / counts

# --- JAX-based Random Subsampling for Scatter Plot ---
total_points = r_flat.shape[0]
perm = jr.permutation(key, total_points)
indices = perm[:num_scatter_samples]

r_scatter = r_flat[indices]
rho_scatter = rho_flat[indices]
v_abs_scatter = v_abs_flat[indices]
p_scatter = p_flat[indices]

# --- Exact Sedov-Taylor Solution ---
eblast_exact = 1.0 / 1.0  # E_explosion / rho_ambient
sedov_solver = Sedov(geometry=3, eblast=eblast_exact, gamma=params.gamma, omega=0.0)
r_exact = jnp.linspace(0.0, domain_max_r, 500)
solution_exact = sedov_solver(r=r_exact, t=params.t_end)

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=120)
fig.suptitle(f'Sedov Blast Wave at t = {params.t_end:.2f} (Grid: {config.num_cells}³)', fontsize=16)

# --- Density Plot ---
ax = axes[0]
ax.scatter(r_scatter, rho_scatter, color='lightgray', alpha=0.5, s=1, rasterized=True, label='simulation (sampled)')
ax.plot(bin_centers, mean_rho, '-', color='royalblue', label='binned average')
ax.plot(r_exact, solution_exact['density'], ls='--', lw=2, c='red', label='exact solution')
ax.set_title('Density')
ax.set_xlabel('Radius')
ax.set_ylabel('Density')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, domain_max_r)

# --- Velocity Plot ---
ax = axes[1]
ax.scatter(r_scatter, v_abs_scatter, color='lightgray', alpha=0.5, s=1, rasterized=True, label='simulation (sampled)')
ax.plot(bin_centers, mean_v_abs, '-', color='royalblue', label='binned average')
ax.plot(r_exact, solution_exact['velocity'], ls='--', lw=2, c='red', label='exact solution')
ax.set_title('Absolute Velocity')
ax.set_xlabel('Radius')
ax.set_ylabel(r'$|v|$')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, domain_max_r)
ax.set_ylim(0, None)

# --- Pressure Plot ---
ax = axes[2]
ax.scatter(r_scatter, p_scatter, color='lightgray', alpha=0.5, s=1, rasterized=True, label='simulation (sampled)')
ax.plot(bin_centers, mean_p, '-', color='royalblue', label='binned average')
ax.plot(r_exact, solution_exact['pressure'], ls='--', lw=2, c='red', label='exact solution')
ax.set_title('Pressure')
ax.set_xlabel('Radius')
ax.set_ylabel('Pressure')
ax.set_yscale('log')
ax.grid(True, linestyle=':', alpha=0.7)
ax.set_xlim(0, domain_max_r)
ax.set_ylim(bottom=p_ambient / 2)

# --- Legend ---
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

plt.tight_layout(rect=[0, 0.1, 1, 0.96])
plt.savefig('figures/sedovAM_HLLC.png', dpi=300)
print("Plot with binned profiles and JAX-sampled scatter saved as 'sedovAM_HLLC.png'.")