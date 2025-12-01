# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

import jax

# debug nans
# jax.config.update("jax_debug_nans", True)

import jax.random as jr

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

from astronomix._finite_difference._time_integrators._ssprk import _ssprk4

from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_x, _weno_flux_y, _weno_flux_z

# astronomix data structures
from astronomix import SimulationConfig
from astronomix import SimulationParams

# astronomix constants
from astronomix.option_classes.simulation_config import AM_HLLC, CARTESIAN, FINITE_DIFFERENCE, HLLC, HLLC_LM, HYBRID_HLLC, MUSCL, PERIODIC_BOUNDARY, SPHERICAL, HLL, MINMOD, SPLIT, BoundarySettings, BoundarySettings1D


# astronomix functions
from astronomix import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config

from astronomix import time_integration


from astronomix._finite_difference._fluid_equations._equations import conserved_state_from_primitive_mhd
from astronomix._finite_difference._fluid_equations._fluxes import _mhd_flux_x

# --- Additional imports for analysis and plotting ---
# For radially averaging the simulation data
from scipy.stats import binned_statistic
# For the exact Sedov-Taylor solution
# from exactpack.solvers.sedov.sedov import Sedov


config = SimulationConfig(

    solver_mode = FINITE_DIFFERENCE,

    # fixed_timestep = True,
    # num_timesteps = 2,

    mhd = True,

    geometry = CARTESIAN,
    progress_bar = True,
    runtime_debugging = False,

    riemann_solver = HYBRID_HLLC,
    # split = SPLIT,
    # time_integrator = MUSCL,
    
    dimensionality = 3,

    exact_end_time = True,

    # ====== RESOLUTION =======
    num_cells = 32,
    # =========================

    # ==== snapshotting ====
    return_snapshots = False,
    # ======================

    boundary_settings = BoundarySettings(
        x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    ),

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
r_explosion = 0.2

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
    density = rho, # jnp.ones_like(rho) + 0.1 * jnp.exp(-r**2 / (2 * (0.1)**2)),
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    magnetic_field_x = jnp.zeros_like(rho),
    magnetic_field_y = jnp.zeros_like(rho),
    magnetic_field_z = jnp.zeros_like(rho),
    gas_pressure = p_gas
)


config = finalize_config(config, initial_state.shape)

conserved_state = conserved_state_from_primitive_mhd(
    primitive_state = initial_state,
    gamma = params.gamma,
    registered_variables = registered_variables,
)

fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.imshow(conserved_state[registered_variables.density_index, :, :, config.num_cells//2], origin='lower', extent=[0,1,0,1])
plt.savefig("figures/start.png", dpi=300)


# f = _weno_flux_y(
#     conserved_state,
#     gamma = params.gamma,
#     registered_variables = registered_variables,
# )

# # print(conserved_state)

# for i in range(20):
#     print(f"Step {i+1}/20")
#     conserved_state = _ssprk4(
#         conserved_state,
#         gamma = params.gamma,
#         grid_spacing = config.grid_spacing,
#         dt = 0.01,
#         registered_variables = registered_variables,
#     )
#     fig, ax = plt.subplots(1, 1, figsize=(6,6))
#     ax.imshow(conserved_state[registered_variables.density_index, :, :, config.num_cells//2], origin='lower', extent=[0,1,0,1])
#     plt.savefig(f"figures/slice_{i+1}.png", dpi=300)

# print(conserved_state)

# Run the simulation
result = time_integration(initial_state, config, params, registered_variables)

# print(result)


# =========================================================================
# === Analysis and Plotting ===============================================
# =========================================================================

# imshow
fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.imshow(result[registered_variables.density_index, :, :, config.num_cells//2], origin='lower', extent=[0,1,0,1])
plt.savefig("figures/slice.png", dpi=300)

# print(result[registered_variables.density_index, :, :, config.num_cells//2])