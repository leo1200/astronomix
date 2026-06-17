# ============================================================================
# 2D Shock Finder Test — Two Intersecting Shocks (X shape)
# ============================================================================
# Two Sod-like pressure discontinuities pass through (0.5, 0.5) at
# FRONT_NORMAL_ANGLE_1 = 60° and FRONT_NORMAL_ANGLE_2 = -20°, forming an X.
#
# Initial conditions:
#   Two signed distances (dist1, dist2) divide the domain into four wedges.
#   XOR: high pressure only where a cell is on opposite sides of the two fronts.
#   → two alternating high-pressure wedges, two low-pressure wedges.
#   High: p=1.0, ρ=1.0 — Low: p=0.1, ρ=0.125 — no initial velocity.
#
# Stress test goals (NOT a clean Sod validation):
#   1. Both shock arms detected away from the intersection
#   2. Shock finder does not crash or produce garbage everywhere
#   3. Detected d_s aligns with the expected normal along each arm
#   4. Intersection region (~r < 0.1) is ambiguous — noisy d_s expected there
# ==================================================================

#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD
from astronomix._physics_modules._shock_finder.shock_finder_2d import find_shocks_pfrommer


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
params = SimulationParams(t_end=0.15)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = helper_data.geometric_centers

# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x = geometric_centers[..., 0] # (nx, ny)
geometry_y = geometric_centers[..., 1] # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — double Sod (two outward-propagating shocks)
#
# Two planar fronts pass through center (0.5, 0.5) at +60° and -20°, forming an X shape
# Each front divides the domain into two half-planes via signed distance (dist1, dist2)
# XOR: a cell is high-pressure only if it's on opposite sides of the two fronts — this creates two alternating high/low pressure wedges
# High pressure: p=1.0, ρ=1.0 — Low pressure: p=0.1, ρ=0.125 (standard Sod values)
# No initial velocity anywhere
# The intersection region at the center is interensted
# ============================================================================

FRONT_NORMAL_ANGLE_1 =  60.0    # degrees — normal 
FRONT_NORMAL_ANGLE_2 = -20.0    # degrees — normal of shock 2
# both pass through the center
TARGET_CENTER = (0.5, 0.5)
target_theta1 = jnp.deg2rad(FRONT_NORMAL_ANGLE_1)
target_theta2 = jnp.deg2rad(FRONT_NORMAL_ANGLE_2)

target_nx_hat_1, target_ny_hat_1 = jnp.cos(target_theta1), jnp.sin(target_theta1)
target_nx_hat_2, target_ny_hat_2 = jnp.cos(target_theta2), jnp.sin(target_theta2)

# signed distance from each front
dist1 = (geometry_x - TARGET_CENTER[0]) * target_nx_hat_1 + (geometry_y - TARGET_CENTER[1]) * target_ny_hat_1   # signed dist from shock 1
dist2 = (geometry_x - TARGET_CENTER[0]) * target_nx_hat_2 + (geometry_y - TARGET_CENTER[1]) * target_ny_hat_2   # signed dist from shock 2

high1 = dist1 < 0
high2 = dist2 < 0

# XOR: high pressure in alternating wedges → pressure jump along both diagonals
in_high = jnp.logical_xor(high1, high2)

p   = jnp.where(in_high, 1.0, 0.1  )
rho = jnp.where(in_high, 1.0, 0.125)
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
print(f"=== Two Intersecting Shocks at Center ({FRONT_NORMAL_ANGLE_1}° and {FRONT_NORMAL_ANGLE_2}°) ===")
print(f"num_shocks (surface cells): {result.num_shocks}")

surface_mask = result.shock_surface_cells

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found")
else:
    surface_mach = result.mach_numbers[surface_mask]
    print(f"Mach at surface: min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")

    # classify surface cells as near/far from intersection
    geometry_surface_x = geometry_x[surface_mask]
    geometry_surface_y = geometry_y[surface_mask]
    dist_from_center = jnp.sqrt((geometry_surface_x - TARGET_CENTER[0])**2 + (geometry_surface_y - TARGET_CENTER[1])**2)
    near_intersection = dist_from_center < 0.1
    far_from_intersection = ~near_intersection

    shock_dir_surface_x = shock_dir_x[surface_mask]
    shock_dir_surface_y = shock_dir_y[surface_mask]

    # compare each surface cell shock direction to the two expected normals via absolute dot product
    dot1 = shock_dir_surface_x * float(target_nx_hat_1) + shock_dir_surface_y * float(target_ny_hat_1)
    dot2 = shock_dir_surface_x * float(target_nx_hat_2) + shock_dir_surface_y * float(target_ny_hat_2)
    align1 = jnp.abs(dot1)
    align2 = jnp.abs(dot2)

    shock1_like = far_from_intersection & (align1 >= align2)
    shock2_like = far_from_intersection & (align2 > align1)

    print("\nDirection alignment away from intersection:")
    if shock1_like.sum() > 0:
        print(f"  shock-1-like cells: count={int(shock1_like.sum())}, mean |dot(ds,n1)|={float(align1[shock1_like].mean()):.3f}")
    if shock2_like.sum() > 0:
        print(f"  shock-2-like cells: count={int(shock2_like.sum())}, mean |dot(ds,n2)|={float(align2[shock2_like].mean()):.3f}")


#%%
# PLOTS
# expectation of ploting is that it must be environment independent
# means it must visualize the results withou put into consideration of what the shock should look like
# 1 -5 are standard plots with no environment assumptions, just showing the results
# 
# there are some points we put details relating to the expected results, but they are just for reference and not for validating the results
# 

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10),
    constrained_layout=True
)
fig.suptitle(f"Two Intersecting Shocks at Center ({FRONT_NORMAL_ANGLE_1}° and {FRONT_NORMAL_ANGLE_2}°) — X shape", fontsize=13)

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

# 3. Shock surface + zone on pressure
axes[0, 2].pcolormesh(
    geometry_x_np, geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.8
)

# Draw shock zone first
axes[0, 2].contourf(
    geometry_x_np, geometry_y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.25
)

# Draw shock surface on top
axes[0, 2].contour(
    geometry_x_np, geometry_y_np,
    np.array(result.shock_surface_cells).astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.5
)

axes[0, 2].set_title("Shock surfaces and shock zones")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

# 4. Mach number
mach_surface_only = np.array(result.mach_numbers)

im3 = axes[1, 0].pcolormesh(
    geometry_x_np,
    geometry_y_np,
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

# 5. shock_direction at surface cells only
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


# Subsample arrows
n_arrows = 30
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


# Direction arrows
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

axes[1, 1].set_title("Shock direction at surface cells\nexpect outward directions")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

for ax in axes.flat:
    ax.set_box_aspect(1)

spatial_axes = [
    axes[0, 0],
    axes[0, 1],
    axes[0, 2],
    axes[1, 0],
    axes[1, 1]
]

for ax in spatial_axes:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

plt.show()


# %%
