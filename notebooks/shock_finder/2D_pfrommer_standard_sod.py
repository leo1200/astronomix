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

# --------------------------------------------------------------------------
# Precompute shock-surface / shock-zone x locations
# --------------------------------------------------------------------------
surface_mask_cols = jnp.any(surface_mask, axis=1)        # (nx,) with bool value -> if this col has any surface cell, mark it True
zone_mask_cols    = jnp.any(result.shock_zones, axis=1)  # (nx,) with bool value -> if this col has any shock zone cell, mark it True

# from surface_mask_cols we get which columns (indices) have shock surface cells, 
# need to propagate back x geometry positions of those columns from geometry_x
# geometry_x shape = (nx, ny, 2)
# geometry_x[:, 0] get all x row at 0 column (because y will always same, no need take y) -> (nx, 2)
# geometry_x[:, 0][surface_mask_cols] -> keep only entry where surface cells
surface_x_cols = geometry_x[:, 0][surface_mask_cols]
zone_x_cols    = geometry_x[:, 0][zone_mask_cols]

print("\n number shock surface columns:", surface_x_cols.size)
indices = jnp.where(surface_mask_cols)[0]
print(" shock surface columns indices:", indices)


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

# shock-zone band limits
# this only correct when the shock zone is a contiguous band of columns, 
# which is the case here, but may not be in general
# [------ shock region ------]
# x0                        x1
if zone_x_cols.size > 0:
    zone_x0 = float(zone_x_cols.min()) - 0.5 * dx
    zone_x1 = float(zone_x_cols.max()) + 0.5 * dx
else:
    zone_x0, zone_x1 = None, None


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

# --------------------------------------------------------------------------
# 3. Shock zone only
# --------------------------------------------------------------------------
# sketch pressure
axes[0, 2].pcolormesh(
    geometry_x, geometry_y, p_final,
    cmap="viridis",
    shading="auto",
    alpha=0.70
)
# sketch shock zone as vertical band from zone_x0 to zone_x1
if zone_x0 is not None:
    axes[0, 2].axvspan(
        zone_x0, zone_x1,
        color="lime",
        alpha=0.28,
        zorder=10
    )

axes[0, 2].set_title("Shock zone + we have margin 0.5 on each side + position from geometry and col have shock cell")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

# --------------------------------------------------------------------------
# 4. Shock surface
# --------------------------------------------------------------------------
# sketch pressure
axes[1, 0].pcolormesh(
    geometry_x, geometry_y, p_final,
    cmap="viridis",
    shading="auto",
    alpha=0.70
)
# sketch shock surface as vertical red line(s) at x positions of surface cells
for i, xi in enumerate(surface_x_cols):
    axes[1, 0].axvline(
        float(xi),
        color="red",
        linewidth=2.5,
        zorder=12
    )

axes[1, 0].set_title("Shock surface + position from geometry and " \
" have shock cell + no margin")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")

# --------------------------------------------------------------------------
# 5. Shock Mach number at surface cells
# --------------------------------------------------------------------------

# Mask Mach number outside shock-surface cells
mach_surface_only = jnp.where(
    result.shock_surface_cells,
    result.mach_numbers,
    jnp.nan
)

# Optional pressure background for context
axes[1, 1].pcolormesh(
    geometry_x, geometry_y, p_final,
    cmap="viridis",
    shading="auto",
    alpha=0.35
)

# Plot only the Mach number at detected shock-surface cells
im_mach = axes[1, 1].pcolormesh(
    geometry_x, geometry_y, mach_surface_only,
    cmap="hot",
    vmin=1.0,
    vmax=2.0,
    shading="auto"
)

axes[1, 1].set_title("Shock Mach number at surface cells")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

cbar = plt.colorbar(im_mach, ax=axes[1, 1])
cbar.set_label("Shock Mach number")

# --------------------------------------------------------------------------
# 6. visulize all characteristics for 1D slice at mid-row (y = mid, x run)
# --------------------------------------------------------------------------
mid = num_cells // 2

x_slice    = geometry_x[:, mid]
p_slice    = p_final[:, mid]
surf_slice = result.shock_surface_cells[:, mid]
zone_slice = result.shock_zones[:, mid]
mach_slice = result.mach_numbers[:, mid]

# pressure
axes[1, 2].plot(x_slice, p_slice, label="pressure")

# shock zone as vertical band on 1D slice
if zone_x0 is not None:
    axes[1, 2].axvspan(
        zone_x0, zone_x1,
        color="green",
        alpha=0.20,
        label="shock zone"
    )

# shock surface as dashed red line(s)
for i, xi in enumerate(x_slice[surf_slice]):
    axes[1, 2].axvline(
        float(xi),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="shock surface" if i == 0 else None
    )

axes[1, 2].set_title(f"1D slice at y={mid} (mid row)")
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("P")


# Mach display
# it will be a point -> check by look at secondary y-axis
ax2 = axes[1, 2].twinx()
ax2.scatter(
    x_slice[surf_slice],
    mach_slice[surf_slice],
    color="orange",
    s=35,
    label="shock Mach",
    zorder=20
)
ax2.set_ylabel("Shock Mach number")
ax2.set_ylim(1.0, 2.5)

# combined legend
lines1, labels1 = axes[1, 2].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

# ============================================================================
# MAKE PANELS SQUARE
# ============================================================================

for ax in axes.flat:
    ax.set_box_aspect(1)

ax2.set_box_aspect(1)

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
