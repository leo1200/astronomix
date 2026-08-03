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


#### Mach number diagnostics
surface_mach = result.mach_numbers[surface_mask] # value, (nx, ny) flattened to 1D
print(f"num_shocks (surface cells): {result.num_shocks}")
print(f"Mach at surface           : mean={surface_mach.mean():.3f}")


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

print(f"overall shock direction x at surface    : overall_dir_x={float(overall_dir_x):.3f}  (expect ≈ ±{float(target_nx_hat):.3f})")
print(f"overall shock direction y at surface    : overall_dir_y={float(overall_dir_y):.3f}  (expect ≈ ±{float(target_ny_hat):.3f})")
print(f"overall shock angle at surface          : {float(overall_angle):.2f}°")
print(f"Expected angle            : {float(FRONT_NORMAL_ANGLE):.2f}°")

# Direction alignment with expected normal.
# Use absolute value because n and -n are both valid normal directions.
dot_normal = overall_dir_x * target_nx_hat + overall_dir_y * target_ny_hat
alignment = jnp.abs(dot_normal)
print(f"Alignment check for overall shock direction |via dot with normal|    : {float(alignment):.3f}  (expect ≈ 1)")


#%%
# PLOTS

fig, axes = plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=box_size,
    mach_vmin=1.0,
    mach_vmax=2.0,
    suptitle=f"2D Rotated Sod Shock Tube ({float(FRONT_NORMAL_ANGLE):.2f}°) — Shock Finder Validation",
)

# ----------------------------------------------------------------------
# 6. Problem-specific panel: diagonal slice along shock normal through center
# ----------------------------------------------------------------------
geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)

t_vals   = np.linspace(-0.5, 0.5, 300)
x_sample = np.clip(0.5 + t_vals * float(target_nx_hat), geometry_x_np.min(), geometry_x_np.max())
y_sample = np.clip(0.5 + t_vals * float(target_ny_hat), geometry_y_np.min(), geometry_y_np.max())

geometry_x_nearest_np = np.argmin(np.abs(geometry_x_np[:, 0:1] - x_sample[np.newaxis, :]), axis=0)
geometry_y_nearest_np = np.argmin(np.abs(geometry_y_np[0:1, :] - y_sample[:, np.newaxis]), axis=1)

p_arr    = np.array(p_final)
surf_arr = np.array(result.shock_surface_cells)
zone_arr = np.array(result.shock_zones)

p_nearest    = p_arr[geometry_x_nearest_np, geometry_y_nearest_np]
surf_nearest = surf_arr[geometry_x_nearest_np, geometry_y_nearest_np]
zone_nearest = zone_arr[geometry_x_nearest_np, geometry_y_nearest_np]

ax6 = axes[1, 2]

ax6.plot(t_vals, p_nearest, label="pressure")

ax6.fill_between(
    t_vals, 0, 1,
    where=zone_nearest,
    alpha=0.20,
    color="green",
    linewidth=5,
    label="shock zone"
)

first = True
for ti in t_vals[surf_nearest]:
    ax6.axvline(
        ti,
        color="red",
        linestyle="--",
        linewidth=0.5,
        label="shock surface" if first else None
    )
    first = False

ax6.axvline(
    0.37,
    color="gray",
    linestyle=":",
    linewidth=1.2,
    label="expected shock distance ≈ 0.37"
)

ax6.set_title(f"Pressure slice along shock normal, θ={float(FRONT_NORMAL_ANGLE):.2f}°")
ax6.set_xlabel("Signed distance along normal")
ax6.set_ylabel("P")
ax6.set_box_aspect(1)
ax6.legend(fontsize=8)

plt.show()

# %%
