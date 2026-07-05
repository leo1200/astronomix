# ============================================================================
# 2D Shock Finder Test — Rotated Sod Tube
# ============================================================================
# The initial discontinuity is a straight line rotated by SHOCK_ANGLE degrees
# from the x-axis. 
#
# EXPECTED ground truth (same as 1D Sod at t=0.2, just rotated):
#   - shock front: a line perpendicular to the shock normal, at signed distance
#     ≈ 0.37 from the center along the normal direction
#   - Mach number: M ≈ 1.75
#   - shock_direction should align with the normal:
#       n = (cos θ, sin θ), up to an overall sign
#   - direction ratio should satisfy:
#       shock_dir_x / shock_dir_y ≈ tan(θ)
# ============================================================================

#%%
from ctypes import cast

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.option_classes.simulation_config import (
    GEOMETRY_TYPE,
    FIELD_TYPE,
)

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
params = SimulationParams(t_end=0.2)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = helper_data.geometric_centers

# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x = geometric_centers[..., 0] # (nx, ny)
geometry_y = geometric_centers[..., 1] # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — rotated Sod discontinuity
#
# A straight pressure discontinuity passes through (0.5, 0.5) with normal
# n̂ = (cos θ, sin θ) at θ = FRONT_NORMAL_ANGLE degrees.
#
# High pressure (p=1.0, ρ=1.0) where signed distance < 0 (left of front).
# Low  pressure (p=0.1, ρ=0.125) where signed distance > 0 (right of front).
# No initial velocity anywhere.
#
# For a symmetric Sod IC, the shock propagates along n̂ — so FRONT_NORMAL_ANGLE
# is also the expected shock normal angle after time evolution.
# ============================================================================

FRONT_NORMAL_ANGLE = 30.0        # EXPECTED angle of shock NORMAL from x-axis - degrees

# n̂ = (cos θ, sin θ) is the shock normal, pointing outward from the high-pressure region
target_theta_rad = jnp.deg2rad(FRONT_NORMAL_ANGLE)
target_nx_hat    = jnp.cos(target_theta_rad)   # x-component of shock normal
target_ny_hat    = jnp.sin(target_theta_rad)   # y-component of shock normal

# (nx, ny) — signed distance from cell center to line through (0.5, 0.5) with normal n̂
target_signed_dist = (geometry_x - 0.5) * target_nx_hat + (geometry_y - 0.5) * target_ny_hat

rho = jnp.where(target_signed_dist < 0, 1.0,   0.125)
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)
p   = jnp.where(target_signed_dist < 0, 1.0,   0.1)

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
rho_final = final_state[registered_variables.density_index]
vx_final  = final_state[registered_variables.velocity_index.x]
vy_final  = final_state[registered_variables.velocity_index.y]
p_final   = final_state[registered_variables.pressure_index]

#%%
# RUN SHOCK FINDER
result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

# %%
# DIAGNOSTICS

print(f"=== Shock Finder 2D Diagnostics — Rotated Sod ({FRONT_NORMAL_ANGLE}°) ===")
print(f"Expected shock normal direction : ({float(target_nx_hat):.3f}, {float(target_ny_hat):.3f})")
print(f"Expected shock_dir_y / shock_dir_x     : {float(jnp.tan(target_theta_rad)):.3f}")

surface_mask = result.shock_surface_cells # boolean, (nx, ny)

rotated_mean_flux = float(
    result.thermal_energy_flux[surface_mask].mean()
)

print("Mean rotated-shock flux:", rotated_mean_flux)

#### Direction diagnostics
# Shock_dir give us local shock at each cell
# NEED to calculate is the overall shock orientation -> do this via mean

shock_dir = result.shock_direction
shock_dir_x = shock_dir[0]
shock_dir_y = shock_dir[1]

print(f"overall shock direction calculated via mean of shock_dir at surface cells, ONLY correct for 1 shock or symmetric shocks:")
overall_dir_x = shock_dir_x[surface_mask].mean()
overall_dir_y = shock_dir_y[surface_mask].mean()
overall_angle = jnp.rad2deg(
    jnp.arctan2(
        overall_dir_y,
        overall_dir_x
    )
)

print(f"overall shock angle at surface          : {float(overall_angle):.2f}°")
print(f"Expected angle            : {float(FRONT_NORMAL_ANGLE):.2f}°")

#%%
# PLOTS
# ============================================================================
# THREE-PANEL ENERGY-DISSIPATION VALIDATION
# ============================================================================

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)

surface_np = np.array(result.shock_surface_cells).astype(bool)
thermal_flux_np = np.array(result.thermal_energy_flux)

# Hide cells that are not on the shock surface
thermal_flux_surface = np.where(
    surface_np,
    thermal_flux_np,
    np.nan,
)

mean_flux = thermal_flux_np[surface_np].mean()

fig, axes = plt.subplots(
    1, 3,
    figsize=(15, 5),
    constrained_layout=True,
)

fig.suptitle(
    "2D Sod Shock — Thermal-Energy Dissipation Validation",
    fontsize=14,
)

# --------------------------------------------------------------------------
# 1. Pressure
# --------------------------------------------------------------------------
im0 = axes[0].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
)

axes[0].set_title("Pressure")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

plt.colorbar(
    im0,
    ax=axes[0],
    label="Pressure",
)

# --------------------------------------------------------------------------
# 2. Shock surface and shock zone
# --------------------------------------------------------------------------
axes[1].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.75,
)

axes[1].contourf(
    geometry_x_np,
    geometry_y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.25,
)

axes[1].contour(
    geometry_x_np,
    geometry_y_np,
    surface_np.astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.5,
)

axes[1].set_title("Shock surface and shock zone")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")

# --------------------------------------------------------------------------
# 3. Thermal-energy flux
# --------------------------------------------------------------------------
im2 = axes[2].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    thermal_flux_surface,
    cmap="inferno",
    shading="auto",
    vmin=0.0,
    vmax=1.05 * mean_flux,
)

axes[2].set_title("Thermal-energy flux")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")

plt.colorbar(
    im2,
    ax=axes[2],
    label="Thermal-energy flux",
)

for ax in axes:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_box_aspect(1)

plt.show()

# %%
