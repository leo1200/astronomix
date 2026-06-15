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

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD, StaticIntVector, StaticFloatVector
from astronomix._physics_modules._shock_finder.shock_finder_2d import find_shocks_pfrommer


#%%
# ============================================================================
# CONFIGURATION
# ============================================================================

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

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
x = helper_data.geometric_centers[..., 0]   # (nx, ny)
y = helper_data.geometric_centers[..., 1]   # (nx, ny)


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
                               
rho = jnp.where(x < shock_pos, 1.0,   0.125)
u_x = jnp.zeros_like(x)
u_y = jnp.zeros_like(x)
p   = jnp.where(x < shock_pos, 1.0,   0.1)

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
# ============================================================================
# RUN SIMULATION
# ============================================================================

final_state = time_integration(initial_state, config, params, registered_variables)

rho_final = final_state[registered_variables.density_index]       # (nx, ny)
vx_final  = final_state[registered_variables.velocity_index.x]    # (nx, ny)
vy_final  = final_state[registered_variables.velocity_index.y]    # (nx, ny)
p_final   = final_state[registered_variables.pressure_index]      # (nx, ny)


#%%
# ============================================================================
# RUN SHOCK FINDER
# ============================================================================

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
# ============================================================================
# DIAGNOSTICS
# ============================================================================

print("=== Shock Finder 2D Diagnostics ===")

# x-positions of surface cells — should all cluster near x ≈ 0.87
surface_mask = result.shock_surface_cells              # (nx, ny)
surface_x    = x[surface_mask]
surface_mach = result.mach_numbers[surface_mask]

print(f"\nSurface cell x positions  : min={surface_x.min():.4f}  max={surface_x.max():.4f}")
print(f"Expected shock position   : x ≈ 0.87")
print(f"Mach numbers at surface   : min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")
print(f"Expected Mach number      : M ≈ 1.75")
print(f"Shock zone cell count     : {result.shock_zones.sum()}")

# shock_direction x-component should be ≈ ±1, y-component ≈ 0 everywhere
ds_x = result.shock_direction[0]   # (nx, ny)
ds_y = result.shock_direction[1]   # (nx, ny)
print(f"\nshock_direction[x] at surface: mean={ds_x[surface_mask].mean():.3f}  (expect ≈ ±1)")
print(f"shock_direction[y] at surface: mean={ds_y[surface_mask].mean():.3f}  (expect ≈  0)")

# %%
# ============================================================================
# PLOTS
# ============================================================================
fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10),
    constrained_layout=True
)

fig.suptitle("2D Sod Shock Tube — Shock Finder Validation", fontsize=13)
# --------------------------------------------------------------------------
# Precompute shock-surface / shock-zone x locations
# --------------------------------------------------------------------------
surface_cols = jnp.any(result.shock_surface_cells, axis=1)   # (nx,)
zone_cols    = jnp.any(result.shock_zones, axis=1)           # (nx,)

surface_x_cols = x[:, 0][surface_cols]
zone_x_cols    = x[:, 0][zone_cols]

dx = float(x[1, 0] - x[0, 0])

# shock-zone band limits
if zone_x_cols.size > 0:
    zone_x0 = float(zone_x_cols.min()) - 0.5 * dx
    zone_x1 = float(zone_x_cols.max()) + 0.5 * dx
else:
    zone_x0, zone_x1 = None, None

# --------------------------------------------------------------------------
# 1. Pressure
# --------------------------------------------------------------------------
im0 = axes[0, 0].pcolormesh(x, y, p_final, cmap="viridis", shading="auto")
axes[0, 0].set_title("Pressure")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])

# --------------------------------------------------------------------------
# 2. Velocity x
# --------------------------------------------------------------------------
im1 = axes[0, 1].pcolormesh(x, y, vx_final, cmap="plasma", shading="auto")
axes[0, 1].set_title("Velocity x (uniform in y)")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1])

# --------------------------------------------------------------------------
# 3. Shock zone only
# --------------------------------------------------------------------------
axes[0, 2].pcolormesh(
    x, y, p_final,
    cmap="viridis",
    shading="auto",
    alpha=0.70
)

if zone_x0 is not None:
    axes[0, 2].axvspan(
        zone_x0, zone_x1,
        color="lime",
        alpha=0.28,
        zorder=10
    )

axes[0, 2].set_title("Shock zone only")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

# --------------------------------------------------------------------------
# 4. Shock surface
# --------------------------------------------------------------------------
axes[1, 0].pcolormesh(
    x, y, p_final,
    cmap="viridis",
    shading="auto",
    alpha=0.70
)

for i, xi in enumerate(surface_x_cols):
    axes[1, 0].axvline(
        float(xi),
        color="red",
        linewidth=2.5,
        zorder=12
    )

axes[1, 0].set_title("Shock surface only")
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
    x, y, p_final,
    cmap="viridis",
    shading="auto",
    alpha=0.35
)

# Plot only the Mach number at detected shock-surface cells
im_mach = axes[1, 1].pcolormesh(
    x, y, mach_surface_only,
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
# 6. 1D slice at mid-row
# --------------------------------------------------------------------------
mid = num_cells // 2

x_slice    = x[:, mid]
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

# Mach on secondary y-axis
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
