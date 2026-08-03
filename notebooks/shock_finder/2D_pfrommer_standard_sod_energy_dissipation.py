# ============================================================================
# 2D Shock Finder Test — 2D Sod Tube
# ============================================================================
# Same initial conditions as the 1D Sod problem, extended uniformly in y.
# The shock travels in x only, so the finder must detect a vertical shock
# surface (a line of cells at constant x) across all y rows.
#
# Ground truth (exact Sod solution at t=0.2):
#   - shock front at x ≈ 0.87
#   - Mach number   M ≈ 1.75
#   - no structure in y
# ============================================================================

#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import cast

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD, StaticIntVector, StaticFloatVector
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer

from jaxtyping import Array, Float

from astronomix.option_classes.simulation_config import (
    GEOMETRY_TYPE,
    FIELD_TYPE,
)
from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_diagnostics_2d


#%%
# CONFIGURATION

num_cells  = 128          # cells per axis (nx = ny = 128)
box_size   = 1.0
shock_pos  = 0.5

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=2,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
)
params = SimulationParams(t_end=0.2)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

"""
Dimensions:
geometric_centers[i, j] = [x_value, y_value]   →  "this cell is located here"
rho[i, j]               = density value        →  "this cell has this much density"
p[i, j]                 = pressure value       →  "this cell has this much pressure"
"""

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = cast(
    GEOMETRY_TYPE, # type: ignore
    helper_data.geometric_centers,
)
# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x: FIELD_TYPE = geometric_centers[..., 0] # (nx, ny)
geometry_y: FIELD_TYPE = geometric_centers[..., 1] # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — 2D Sod tube (discontinuity in x only)
# ============================================================================

# INITIAL STATE
#         Left side: high pressure, high density        Right side: low pressure, low density
#         pL = 1,  ρL = 1                               pR = 0.1,  ρR = 0.125
#         ┌──────────────────────────────┬──────────────────────────────┐
#         │                              │                              │
#         │   High-pressure gas          │       Low-pressure gas       │
#         │   High-density gas           │       Low-density gas        │   ----> Sod tube
#         │                              │                              │
#         └──────────────────────────────┴──────────────────────────────┘
#                                       diaphragm
#                                       x = 0  
                               
rho = jnp.where(geometry_x < shock_pos, 1.0,   0.125)
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_x)
p   = jnp.where(geometry_x < shock_pos, 1.0,   0.1)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    gas_pressure=p,
)
config = finalize_config(config, initial_state.shape)


#%%
# RUN SIMULATION

final_state = time_integration(initial_state, config, params, registered_variables)

rho_final = final_state[registered_variables.density_index]       # (nx, ny)
vx_final  = final_state[registered_variables.velocity_index.x]    # (nx, ny)
vy_final  = final_state[registered_variables.velocity_index.y]    # (nx, ny)
p_final   = final_state[registered_variables.pressure_index]      # (nx, ny)


#%%
# RUN SHOCK FINDER
result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

# shock_surface_cells: (nx, ny) bool
# shock_direction:     (2, nx, ny) float  — (dx, dy) unit vector per cell
# mach_numbers:        (nx, ny) float
# shock_zones:         (nx, ny) bool

#%%
# DIAGNOSTICS

print("=== Shock Finder 2D Diagnostics ===")

surface_mask = result.shock_surface_cells   # (nx, ny)bool value -> True where shock surface cells are detected, False elsewhere
surface_x    = geometry_x[surface_mask]     # x-positions of surface cells        

# thermal energy flux at surface cells should be nonzero, and zero elsewhere
thermal_flux = result.thermal_energy_flux

print(
    "Flux at surface:",
    float(thermal_flux[surface_mask].mean()),
)

# %%
# PLOTS

fig, axes = plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=box_size,
    suptitle="2D Sod Shock — Thermal-Energy Dissipation Validation",
)

# ----------------------------------------------------------------------
# 6. Problem-specific panel: thermal-energy flux at surface cells
# ----------------------------------------------------------------------
surface_np      = np.array(result.shock_surface_cells).astype(bool)
thermal_flux_np = np.array(result.thermal_energy_flux)

thermal_flux_surface = np.where(surface_np, thermal_flux_np, np.nan)
mean_flux = thermal_flux_np[surface_np].mean() if surface_np.any() else np.nan

ax6 = axes[1, 2]
im6 = ax6.pcolormesh(
    np.array(geometry_x), np.array(geometry_y),
    thermal_flux_surface,
    cmap="inferno",
    shading="auto",
    vmin=0.0,
    vmax=1.05 * mean_flux,
)
ax6.set_title("Thermal-energy flux")
ax6.set_xlabel("x")
ax6.set_ylabel("y")
ax6.set_aspect("equal", adjustable="box")
ax6.set_xlim(0, box_size)
ax6.set_ylim(0, box_size)
plt.colorbar(im6, ax=ax6, label="Thermal-energy flux")

plt.show()
# %%