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
from astronomix._physics_modules._shock_finder.shock_finder_2d import find_shocks_pfrommer

from jaxtyping import Array, Float

from astronomix.option_classes.simulation_config import (
    GEOMETRY_TYPE,
    FIELD_TYPE,
)

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
surface_mach = result.mach_numbers[surface_mask]

# shock surface cells should be around x ≈ 0.87, Mach number should be around 1.75
print(f"\nSurface cell x positions  : min={surface_x.min():.4f}  max={surface_x.max():.4f}")
print(f"Mach numbers at surface   : min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")
print(f"Shock zone cell count     : {result.shock_zones.sum()}")


shock_dir = result.shock_direction   # (2, nx, ny) -> shock direction vector at each cell
# shock front itself is a vertical line (constant x, extending across all y)
# -> shock_direction x-component should be ≈ ±1, y-component ≈ 0 everywhere
shock_dir_x = shock_dir[0]   # (nx, ny)
print(f"\nshock_direction[x] at surface: mean={shock_dir_x[surface_mask].mean():.3f}  (expect ≈ ±1)")
shock_dir_y = shock_dir[1]   # (nx, ny)
print(f"shock_direction[y] at surface: mean={shock_dir_y[surface_mask].mean():.3f}  (expect ≈  0)")


# %%
# PLOTS
"""
In here, as shock is a vertical line, in the ideal case 
an entire column have same status
-> so we we group in same column (for dimension nx,ny then follow axis 1) and sketch thing by x
"""
fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10),
    constrained_layout=True
)

fig.suptitle("2D Sod Shock Tube - Shock Finder Validation", fontsize=13)

dx = float(geometry_x[1, 0] - geometry_x[0, 0])

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)


# --------------------------------------------------------------------------
# 1. Pressure
# --------------------------------------------------------------------------
im0 = axes[0, 0].pcolormesh(geometry_x, geometry_y, p_final, cmap="viridis", shading="auto")
axes[0, 0].set_title("Pressure")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])

# --------------------------------------------------------------------------
# 2. Velocity x
# --------------------------------------------------------------------------
im1 = axes[0, 1].pcolormesh(geometry_x, geometry_y, vx_final, cmap="plasma", shading="auto")
axes[0, 1].set_title("Velocity x (uniform in y)")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
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


axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")
axes[1, 1].set_aspect("equal")

# --------------------------------------------------------------------------
# 6. visulize all characteristics for 1D slice at mid-row (y = mid, x run)
# --------------------------------------------------------------------------
mid = num_cells // 2

x_slice    = geometry_x[:, mid]
p_slice    = p_final[:, mid]
surf_slice = result.shock_surface_cells[:, mid]
zone_slice = result.shock_zones[:, mid]
mach_slice = result.mach_numbers[:, mid]

p_arr_slice      = np.array(p_slice)
surf_arr_slice   = np.array(surf_slice)
zone_arr_slice   = np.array(zone_slice)

axes[1, 2].plot(x_slice, p_slice, label="pressure")

axes[1, 2].fill_between(
    x_slice, 0, 1,
    where=zone_arr_slice,
    alpha=0.20,
    color="green",
    linewidth=5,
    label="shock zone"
)

first = True
for ti in x_slice[surf_slice]:
    axes[1, 2].axvline(
        ti,
        color="red",
        linestyle="--",
        linewidth=0.5,
        label="shock surface" if first else None
    )
    first = False

axes[1, 2].set_title(f"1D slice at y={mid} (mid row)")
axes[1, 2].set_xlabel("x")
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

# ============================================================================
# MAKE PANELS SQUARE
# ============================================================================

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
