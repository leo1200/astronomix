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
geometric_centers = cast(
    GEOMETRY_TYPE, # type: ignore
    helper_data.geometric_centers,
)
# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x: FIELD_TYPE = geometric_centers[..., 0] # (nx, ny)
geometry_y: FIELD_TYPE = geometric_centers[..., 1] # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — rotated discontinuity
# Set up to reach EXPECTED shock angle of SHOCK_ANGLE degrees
# The shock normal direction: n = (cos θ, sin θ)
# A point (x, y) is on the "left" (high pressure) side if:
#     (x - 0.5) * cos θ + (y - 0.5) * sin θ < 0
# This places the discontinuity as a line through the center of the domain,
# perpendicular to the normal.

SHOCK_ANGLE = 30.0        # EXPECTED angle of shock NORMAL from x-axis - degrees

# n̂ = (cos θ, sin θ) is the shock normal, pointing outward from the high-pressure region
theta_rad = jnp.deg2rad(SHOCK_ANGLE)
nx_hat    = jnp.cos(theta_rad)   # x-component of shock normal
ny_hat    = jnp.sin(theta_rad)   # y-component of shock normal

# (nx, ny) — signed distance from cell center to line through (0.5, 0.5) with normal n̂
signed_dist = (geometry_x - 0.5) * nx_hat + (geometry_y - 0.5) * ny_hat

rho = jnp.where(signed_dist < 0, 1.0,   0.125)
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)
p   = jnp.where(signed_dist < 0, 1.0,   0.1)

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

print(f"=== Shock Finder 2D Diagnostics — Rotated Sod ({SHOCK_ANGLE}°) ===")
print(f"EXPECTED shock normal direction : ({float(nx_hat):.3f}, {float(ny_hat):.3f})")
print(f"EXPECTED shock_dir_y / shock_dir_x     : {float(jnp.tan(theta_rad)):.3f}")

surface_mask = result.shock_surface_cells # boolean, (nx, ny)


#### Mach number diagnostics
surface_mach = result.mach_numbers[surface_mask] # value, (nx, ny) flattened to 1D

print(f"num_shocks (surface cells): {result.num_shocks}")
print(f"Mach at surface           : min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")

strong_surface_mach = surface_mach[surface_mach > 1.1]
print(
    f"Mach at strong surface where mach > 1.1   : "
    f"min={strong_surface_mach.min():.3f}  "
    f"max={strong_surface_mach.max():.3f}  "
    f"mean={strong_surface_mach.mean():.3f}"
)

#### Direction diagnostics
# Shock_dir give us local shock at each cell
# NEED to calculate is the overall shock orientation -> do this via mean

shock_dir = result.shock_direction
shock_dir_x = shock_dir[0]
shock_dir_y = shock_dir[1]

mean_dir_at_surface_x = shock_dir_x[surface_mask].mean()
mean_dir_at_surface_y = shock_dir_y[surface_mask].mean()
mean_angle = jnp.rad2deg(
    jnp.arctan2(
        mean_dir_at_surface_y,
        mean_dir_at_surface_x
    )
)

print(f"overall shock direction x at surface    : mean={float(mean_dir_at_surface_x):.3f}  (expect ≈ ±{float(nx_hat):.3f})")
print(f"overall shock direction y at surface    : mean={float(mean_dir_at_surface_y):.3f}  (expect ≈ ±{float(ny_hat):.3f})")
print(f"overall shock angle at surface          : {float(mean_angle):.2f}°")
print(f"Expected angle            : {SHOCK_ANGLE:.2f}°")

# Direction alignment with expected normal.
# Use absolute value because n and -n are both valid normal directions.
dot_normal = mean_dir_at_surface_x * nx_hat + mean_dir_at_surface_y * ny_hat
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
    f"2D Rotated Sod Shock Tube ({SHOCK_ANGLE}°) — Shock Finder Validation",
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

axes[1, 1].pcolormesh(
    geometry_x_np, geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.55
)

# Shock surface mask
surface = np.array(result.shock_surface_cells)

# Surface-cell coordinates and directions
xs = geometry_x_np[surface]
ys = geometry_y_np[surface]
ux = np.array(shock_dir_x)[surface]
uy = np.array(shock_dir_y)[surface]

# Normalize arrows to unit length
mag = np.sqrt(ux**2 + uy**2)
valid = mag > 0

xs = xs[valid]
ys = ys[valid]
ux = ux[valid] / mag[valid]
uy = uy[valid] / mag[valid]

# For visualization only: orient all arrows toward the expected normal.
# The raw shock finder direction may be n or -n; both represent the same normal line.
dot = ux * float(nx_hat) + uy * float(ny_hat)
flip = dot < 0
ux[flip] *= -1
uy[flip] *= -1

# Keep only a few arrows, evenly spaced along the shock surface
n_arrows = 10
if len(xs) > n_arrows:
    idx = np.linspace(0, len(xs) - 1, n_arrows).astype(int)
    xs_plot = xs[idx]
    ys_plot = ys[idx]
    ux_plot = ux[idx]
    uy_plot = uy[idx]
else:
    xs_plot = xs
    ys_plot = ys
    ux_plot = ux
    uy_plot = uy

# Draw shock surface
axes[1, 1].contour(
    geometry_x_np, geometry_y_np,
    surface.astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.8
)

# Draw fewer, cleaner direction arrows
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

# Optional: add one expected-normal arrow from the center
axes[1, 1].annotate(
    "",
    xy=(0.5 + 0.15 * float(nx_hat), 0.5 + 0.15 * float(ny_hat)),
    xytext=(0.5, 0.5),
    arrowprops=dict(arrowstyle="->", color="black", lw=2)
)

axes[1, 1].set_title(
    f"Shock direction at surface cells\nexpected normal ≈ ({float(nx_hat):.2f}, {float(ny_hat):.2f})"
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].set_aspect("equal")

# 6. Diagonal slice along shock normal through the center
# sample pressure along the normal direction
t_vals   = np.linspace(-0.5, 0.5, 300)
x_sample = np.clip(0.5 + t_vals * float(nx_hat), 0.01, 0.99)
y_sample = np.clip(0.5 + t_vals * float(ny_hat), 0.01, 0.99)

# nearest-cell lookup
cell_size = box_size / num_cells
xi = np.clip((x_sample / cell_size).astype(int), 0, num_cells - 1)
yi = np.clip((y_sample / cell_size).astype(int), 0, num_cells - 1)

p_arr      = np.array(p_final)
surf_arr   = np.array(result.shock_surface_cells)
zone_arr   = np.array(result.shock_zones)

p_along    = p_arr[xi, yi]
surf_along = surf_arr[xi, yi]
zone_along = zone_arr[xi, yi]

axes[1, 2].plot(t_vals, p_along, label="pressure")

axes[1, 2].fill_between(
    t_vals, 0, 1,
    where=zone_along,
    alpha=0.20,
    color="green",
    linewidth=5,
    label="shock zone"
)

first = True
for ti in t_vals[surf_along]:
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

axes[1, 2].set_title(f"Pressure slice along shock normal, θ={SHOCK_ANGLE}°")
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
