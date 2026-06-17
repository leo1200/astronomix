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
from astronomix._physics_modules._shock_finder.shock_finder_2d import find_shocks_pfrommer
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
# INITIAL CONDITIONS — rotated discontinuity
# Set up to reach TARGET shock angle of TARGET_SHOCK_ANGLE degrees
# The shock normal direction: n = (cos θ, sin θ)
# A point (x, y) is on the "left" (high pressure) side if:
#     (x - 0.5) * cos θ + (y - 0.5) * sin θ < 0
# This places the discontinuity as a line through the center of the domain,

TARGET_SHOCK_ANGLE = 30.0        # EXPECTED angle of shock NORMAL from x-axis - degrees

# n̂ = (cos θ, sin θ) is the shock normal, pointing outward from the high-pressure region
target_theta_rad = jnp.deg2rad(TARGET_SHOCK_ANGLE)
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

print(f"=== Shock Finder 2D Diagnostics — Rotated Sod ({TARGET_SHOCK_ANGLE}°) ===")
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
print(f"Expected angle            : {float(TARGET_SHOCK_ANGLE):.2f}°")

# Direction alignment with expected normal.
# Use absolute value because n and -n are both valid normal directions.
dot_normal = overall_dir_x * target_nx_hat + overall_dir_y * target_ny_hat
alignment = jnp.abs(dot_normal)
print(f"Alignment check for overall shock direction |via dot with normal|    : {float(alignment):.3f}  (expect ≈ 1)")


#%%
# PLOTS
fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10),
    constrained_layout=True
)
fig.suptitle(
    f"2D Rotated Sod Shock Tube ({float(TARGET_SHOCK_ANGLE):.2f}°) — Shock Finder Validation",
    fontsize=13
)

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)

# 1. Pressure
im0 = axes[0, 0].pcolormesh(geometry_x_np, geometry_y_np, np.array(p_final), cmap="viridis")
axes[0, 0].set_title("Pressure")
axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])

# 2. Density
im1 = axes[0, 1].pcolormesh(geometry_x_np, geometry_y_np, np.array(rho_final), cmap="plasma")
axes[0, 1].set_title("Density")
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1])

# 3. Shock surface + zone overlaid on pressure

# pressure sketch
axes[0, 2].pcolormesh(geometry_x_np, geometry_y_np, np.array(p_final), cmap="viridis", alpha=0.8)

# draw zone
axes[0, 2].contourf(
    geometry_x_np, geometry_y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.25
)

# draw surface on top
axes[0, 2].contour(
    geometry_x_np, geometry_y_np,
    np.array(result.shock_surface_cells).astype(float),
    levels=[0.5],
    colors="red",
    linewidths=0.5
)

axes[0, 2].set_title("Shock surface and shock zone")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

# 4. Mach number at surface cells
mach_surface_only = np.array(result.mach_numbers)

im3 = axes[1, 0].pcolormesh(
    geometry_x_np, geometry_y_np,
    mach_surface_only,
    cmap="hot",
    vmin=1.0,
    vmax=2.0,
    shading="auto"
)

axes[1, 0].set_title("Shock Mach number at surface cells")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 0], label="Shock Mach number")

# --------------------------------------------------------------------------
# 5. Shock direction at surface cells
# --------------------------------------------------------------------------

# pressure sketch
axes[1, 1].pcolormesh(
    geometry_x_np, geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.55
)

# Shock surface mask
surface = np.array(result.shock_surface_cells)
# Draw shock surface
axes[1, 1].contour(
    geometry_x_np, geometry_y_np,
    surface.astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.8
)

# Surface-cell coordinates and directions
geometry_x_surface_np = geometry_x_np[surface]
geometry_y_surface_np = geometry_y_np[surface]
shock_dir_x_surface_np = np.array(shock_dir_x)[surface]
shock_dir_y_surface_np = np.array(shock_dir_y)[surface]

# remove 0 direction cells to avoid NaNs when normalizing
mag_shock_dir_surface = np.sqrt(shock_dir_x_surface_np**2 + shock_dir_y_surface_np**2)
valid = mag_shock_dir_surface > 0 # could be zero -> skip
geometry_x_surface_np = geometry_x_surface_np[valid]
geometry_y_surface_np = geometry_y_surface_np[valid]

# Normalize arrows to unit length
u_shock_dir_x_surface_np = shock_dir_x_surface_np[valid] / mag_shock_dir_surface[valid]
u_shock_dir_y_surface_np = shock_dir_y_surface_np[valid] / mag_shock_dir_surface[valid]

# For visualization only: orient all arrows toward the expected normal.
# The raw shock finder direction may be n or -n; both represent the same normal line.
dot = u_shock_dir_x_surface_np * float(target_nx_hat) + u_shock_dir_y_surface_np * float(target_ny_hat)
flip = dot < 0
u_shock_dir_x_surface_np[flip] *= -1
u_shock_dir_y_surface_np[flip] *= -1


# Draw only 10 arrows, evenly spaced
n_arrows = 10
if len(geometry_x_surface_np) > n_arrows:
    idx = np.linspace(0, len(geometry_x_surface_np) - 1, n_arrows).astype(int)
    xs_plot = geometry_x_surface_np[idx]
    ys_plot = geometry_y_surface_np[idx]
    ux_plot = u_shock_dir_x_surface_np[idx]
    uy_plot = u_shock_dir_y_surface_np[idx]
else:
    xs_plot = geometry_x_surface_np
    ys_plot = geometry_y_surface_np
    ux_plot = u_shock_dir_x_surface_np
    uy_plot = u_shock_dir_y_surface_np

axes[1, 1].quiver(
    xs_plot,
    ys_plot,
    ux_plot,
    uy_plot,
    angles="xy",
    scale_units="xy",
    scale=25,
    color="white",
    width=0.004,
    headwidth=4,
    headlength=5,
    pivot="middle",
    zorder=20
)

# Add one expected-normal arrow from the center
axes[1, 1].annotate(
    "expected normal",
    xy=(0.5 + 0.15 * float(target_nx_hat), 0.5 + 0.15 * float(target_ny_hat)),
    xytext=(0.5, 0.5),
    textcoords="data",
    arrowprops=dict(arrowstyle="->", color="black", lw=2),
    ha="center",
    va="center",
    color="black"
)

axes[1, 1].set_title(
    f"Shock direction at surface cells\nexpected normal ≈ ({float(target_nx_hat):.2f}, {float(target_ny_hat):.2f})"
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].set_aspect("equal")


# 6. Diagonal slice along shock normal through the center
t_vals   = np.linspace(-0.5, 0.5, 300)
x_sample = np.clip(0.5 + t_vals * float(target_nx_hat), geometry_x_np.min(), geometry_x_np.max())
y_sample = np.clip(0.5 + t_vals * float(target_ny_hat), geometry_y_np.min(), geometry_y_np.max())

# nearest-cell lookup using geometry
geometry_x_nearest_np = np.argmin(np.abs(geometry_x_np[:, 0:1] - x_sample[np.newaxis, :]), axis=0)
geometry_y_nearest_np = np.argmin(np.abs(geometry_y_np[0:1, :] - y_sample[:, np.newaxis]), axis=1)

p_arr      = np.array(p_final)
surf_arr   = np.array(result.shock_surface_cells)
zone_arr   = np.array(result.shock_zones)

p_nearest    = p_arr[geometry_x_nearest_np, geometry_y_nearest_np]
surf_nearest = surf_arr[geometry_x_nearest_np, geometry_y_nearest_np]
zone_nearest = zone_arr[geometry_x_nearest_np, geometry_y_nearest_np]

# pressure slice
axes[1, 2].plot(t_vals, p_nearest, label="pressure")

# zone
axes[1, 2].fill_between(
    t_vals, 0, 1,
    where=zone_nearest,
    alpha=0.20,
    color="green",
    linewidth=5,
    label="shock zone"
)

# labeling
# no need duplicate label
first = True
for ti in t_vals[surf_nearest]:
    axes[1, 2].axvline(
        ti,
        color="red",
        linestyle="--",
        linewidth=0.5,
        label="shock surface" if first else None
    )
    first = False

axes[1, 2].axvline(
    0.37,
    color="gray",
    linestyle=":",
    linewidth=1.2,
    label="expected shock distance ≈ 0.37"
)

axes[1, 2].set_title(f"Pressure slice along shock normal, θ={float(TARGET_SHOCK_ANGLE):.2f}°")
axes[1, 2].set_xlabel("Signed distance along normal")
axes[1, 2].set_ylabel("P")
axes[1, 2].legend(fontsize=8)

for ax in axes.flat:
    ax.set_box_aspect(1)

spatial_axes = [
    axes[0, 0],
    axes[0, 1],
    axes[0, 2],
    axes[1, 0],
    axes[1, 1],
]

for ax in spatial_axes:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.show()

# %%
