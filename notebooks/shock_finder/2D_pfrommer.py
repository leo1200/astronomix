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


# ============================================================================
# RUN SIMULATION
# ============================================================================

final_state = time_integration(initial_state, config, params, registered_variables)

rho_final = final_state[registered_variables.density_index]       # (nx, ny)
vx_final  = final_state[registered_variables.velocity_index.x]    # (nx, ny)
vy_final  = final_state[registered_variables.velocity_index.y]    # (nx, ny)
p_final   = final_state[registered_variables.pressure_index]      # (nx, ny)


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


# ============================================================================
# DIAGNOSTICS
# ============================================================================

print("=== Shock Finder 2D Diagnostics ===")
print(f"shock_surface_cells shape : {result.shock_surface_cells.shape}")
print(f"shock_direction shape     : {result.shock_direction.shape}")
print(f"mach_numbers shape        : {result.mach_numbers.shape}")
print(f"shock_zones shape         : {result.shock_zones.shape}")
print(f"num_shocks (surface cells): {result.num_shocks}")

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


# ============================================================================
# PLOTS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("2D Sod Shock Tube — Shock Finder Validation", fontsize=13)

# 1. Pressure 2D map
im0 = axes[0, 0].pcolormesh(x, y, p_final, cmap="viridis")
axes[0, 0].set_title("Pressure")
axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])

# 2. Density 2D map
im1 = axes[0, 1].pcolormesh(x, y, rho_final, cmap="plasma")
axes[0, 1].set_title("Density")
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1])

# 3. Shock surface overlay on pressure
axes[0, 2].pcolormesh(x, y, p_final, cmap="viridis", alpha=0.8)
axes[0, 2].contour(x, y, result.shock_surface_cells.astype(float), levels=[0.5], colors="red", linewidths=1.5)
axes[0, 2].contourf(x, y, result.shock_zones.astype(float), levels=[0.5, 1.5], colors=["green"], alpha=0.25)
axes[0, 2].set_title("Shock surface (red) & zone (green)")
axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y")

# 4. Mach number map
im3 = axes[1, 0].pcolormesh(x, y, result.mach_numbers, cmap="hot")
axes[1, 0].set_title("Mach number (surface cells only)")
axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 0])

# 5. shock_direction x-component
im4 = axes[1, 1].pcolormesh(x, y, ds_x, cmap="RdBu", vmin=-1, vmax=1)
axes[1, 1].set_title("shock_direction x-component (expect ≈ ±1)")
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")
plt.colorbar(im4, ax=axes[1, 1])

# 6. 1D slice through the middle (y = ny//2) — pressure + shock markers
mid = num_cells // 2
x_slice   = x[:, mid]
p_slice   = p_final[:, mid]
surf_slice = result.shock_surface_cells[:, mid]
zone_slice = result.shock_zones[:, mid]

axes[1, 2].plot(x_slice, p_slice, label="pressure")
axes[1, 2].fill_between(x_slice, 0, 1, where=zone_slice, alpha=0.2, color="green", label="shock zone")
for xi in x_slice[surf_slice]:
    axes[1, 2].axvline(xi, color="red", linestyle="--", linewidth=1.5, label="shock surface")
axes[1, 2].set_title(f"1D slice at y={mid} (mid row)")
axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("P")
axes[1, 2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("figures/shock_finder_2D_test.svg")
plt.show()

print("\nFigure saved to figures/shock_finder_2D_test.svg")
# %%
