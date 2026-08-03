# ============================================================================
# 2D Shock Finder Test — Point Explosion (Sedov-like, single outward shock)
# ============================================================================
# A single point-like energy injection at the domain center drives one
# outward-propagating circular shock (Sedov-Taylor-like blast wave).
#
# Initial conditions:
#   A small disk of radius r_explosion at the center is set to a high
#   pressure p_explosion_gas, chosen so that integrating p/(gamma-1) over
#   the disk area gives back the target explosion energy E_explosion.
#   Everywhere else: ambient density/pressure, no initial velocity.
#
# Stress test goals (NOT a clean Sedov validation):
#   1. A single closed circular shock surface is detected
#   2. Shock finder does not crash or produce garbage everywhere
#   3. Detected shock_direction points radially outward from the center
#      along the shock front, at all angles (azimuthal symmetry check)
#   4. The very center (~r < r_explosion, pre-shock-formation region) and
#      the exact center point itself are expected to be ambiguous/noisy
# ============================================================================

#%%
from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_diagnostics_2d
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer

from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_diagnostics_2d


#%%
# CONFIGURATION

num_cells = 128
box_size  = 1.0

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=2,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
)
params = SimulationParams(t_end=0.1)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = helper_data.geometric_centers

# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x = geometric_centers[..., 0]  # (nx, ny)
geometry_y = geometric_centers[..., 1]  # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — point explosion (single outward-propagating shock)
#
# A small disk of radius r_explosion centered at TARGET_CENTER is given a
# uniform high pressure such that integrating p/(gamma-1) over the disk
# area reproduces E_explosion. Outside the disk: ambient density/pressure.
# No initial velocity anywhere.
# The center of the domain is the region of interest (single point source).
# ============================================================================

# center of the explosion
TARGET_CENTER = (0.5, 0.5)
center_x, center_y = TARGET_CENTER

# total explosion energy (code units)
E_explosion = 1.0

# ambient (background) physical conditions
rho_ambient = 1.0
p_ambient   = 1e-4

# radius of the injection disk (code units)
r_explosion = 0.05

# distance of every cell from the explosion center
dx_from_center = geometry_x - center_x
dy_from_center = geometry_y - center_y

r = jnp.sqrt(dx_from_center**2 + dy_from_center**2)

# injection area (2D analog of the 3D injection_volume in the point-explosion setup)
injection_area = jnp.pi * r_explosion**2

# adiabatic index of the gas
gamma_gas = params.gamma

# E = p * A / (gamma - 1)  =>  p = E * (gamma - 1) / A
p_explosion_gas = E_explosion * (gamma_gas - 1) / injection_area

# pressure: high within the explosion disk, ambient elsewhere
p   = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)
rho = jnp.ones_like(geometry_x) * rho_ambient
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    gas_pressure=p,
)
config = finalize_config(config, initial_state.shape)


# ============================================================================
# RUN SIMULATION
# ============================================================================

#%%
final_state = time_integration(initial_state, config, params, registered_variables)

rho_final = final_state[registered_variables.density_index]
p_final   = final_state[registered_variables.pressure_index]


#%%
# RUN SHOCK FINDER

result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

shock_dir_x = result.shock_direction[0]   # (nx, ny)
shock_dir_y = result.shock_direction[1]   # (nx, ny)

#%%
# DIAGNOSTICS
# notice one thing that all diagnostics below having some calculation
# based on our assumptions
# or based on visualize results
# as ofcourse we cannot counter all scenarios that can happen
print(f"=== Point Explosion at center {TARGET_CENTER} ===")

surface_mask = result.shock_surface_cells
thermal_flux = result.thermal_energy_flux
valid_flux = surface_mask & (thermal_flux > 0.0)

print(
    "Thermal flux:",
    "min =", float(thermal_flux[valid_flux].min()),
    "max =", float(thermal_flux[valid_flux].max()),
    "mean =", float(thermal_flux[valid_flux].mean()),
)

#%%
# PLOTS

fig, axes = plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=box_size,
    suptitle="2D Sedov Blast — Thermal-Energy Dissipation Validation",
)

# ----------------------------------------------------------------------
# 6. Problem-specific panel: thermal-energy flux at surface cells
# ----------------------------------------------------------------------
thermal_flux_np = np.array(result.thermal_energy_flux)
valid_flux_mask = thermal_flux_np > 0.0
thermal_flux_plot = np.where(valid_flux_mask, thermal_flux_np, np.nan)

ax6 = axes[1, 2]
im6 = ax6.pcolormesh(
    np.array(geometry_x), np.array(geometry_y),
    thermal_flux_plot,
    cmap="inferno",
    shading="auto",
    vmin=0.0,
    vmax=float(thermal_flux_np[valid_flux_mask].max()),
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