# ============================================================================
# 2D Shock Finder Test — Two Shocks at Different Angles
# ============================================================================
# Two independent Sod-like discontinuities, each at a different angle and
# passing through a different point in the domain. They form a V shape and
# may intersect as the simulation evolves.
#
# Shock 1: normal at +30°, discontinuity passing through (0.3, 0.5)
#   high pressure on the left side of its front
# Shock 2: normal at -30°, discontinuity passing through (0.7, 0.5)
#   high pressure on the left side of its front
#
# A point is on the "high" side of a shock if its signed distance from the
# discontinuity line is negative (same convention as rotated Sod test).
#
# Ground truth:
#   - two distinct shock surfaces at different angles
#   - shock_direction at shock 1 ≈ (cos30°,  sin30°) = ( 0.866,  0.500)
#   - shock_direction at shock 2 ≈ (cos30°, -sin30°) = ( 0.866, -0.500)
#   - Mach numbers roughly similar on both fronts (same jump ratio)
#   - intersection region (if reached) is ambiguous — expect noisy d_s there
# ============================================================================

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


# ============================================================================
# CONFIGURATION
# ============================================================================

ANGLE_1 =  30.0    # degrees — normal direction of shock 1
ANGLE_2 = -30.0    # degrees — normal direction of shock 2
CENTER_1 = (0.3, 0.5)   # point the shock 1 front passes through
CENTER_2 = (0.7, 0.5)   # point the shock 2 front passes through

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

x = helper_data.geometric_centers[..., 0]   # (nx, ny)
y = helper_data.geometric_centers[..., 1]   # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
# For each shock, compute signed distance from its front.
# A cell is on the "high pressure" side if signed_dist < 0.
# The two high-pressure regions are combined with jnp.maximum (union).

theta1 = jnp.deg2rad(ANGLE_1)
theta2 = jnp.deg2rad(ANGLE_2)

nx1, ny1 = jnp.cos(theta1), jnp.sin(theta1)
nx2, ny2 = jnp.cos(theta2), jnp.sin(theta2)

# signed distance from each front
dist1 = (x - CENTER_1[0]) * nx1 + (y - CENTER_1[1]) * ny1
dist2 = (x - CENTER_2[0]) * nx2 + (y - CENTER_2[1]) * ny2

high1 = dist1 < 0   # high pressure side of shock 1
high2 = dist2 < 0   # high pressure side of shock 2

# each shock independently: high side p=1.0, low side p=0.1
# where both overlap, take the higher pressure
p = jnp.where(high1, 1.0, 0.1)
p = jnp.where(high2, jnp.maximum(p, 1.0), p)

rho = jnp.where(high1, 1.0, 0.125)
rho = jnp.where(high2, jnp.maximum(rho, 1.0), rho)

u_x = jnp.zeros_like(x)
u_y = jnp.zeros_like(x)

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


# ============================================================================
# RUN SHOCK FINDER
# ============================================================================

#%%
result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

ds_x = result.shock_direction[0]
ds_y = result.shock_direction[1]


# ============================================================================
# DIAGNOSTICS
# ============================================================================

#%%
print(f"=== Two Shocks at Different Angles ({ANGLE_1}° and {ANGLE_2}°) ===")
print(f"Shock 1 normal: ({float(nx1):.3f}, {float(ny1):.3f})  through {CENTER_1}")
print(f"Shock 2 normal: ({float(nx2):.3f}, {float(ny2):.3f})  through {CENTER_2}")
print(f"num_shocks (surface cells): {result.num_shocks}")

surface_mask = result.shock_surface_cells

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found")
else:
    surface_mach = result.mach_numbers[surface_mask]
    print(f"Mach at surface: min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")

    # split surface cells by x position to diagnose each shock separately
    surface_x = x[surface_mask]
    surface_y = y[surface_mask]
    left_shock  = surface_x < 0.5
    right_shock = surface_x >= 0.5

    ds_x_surf = ds_x[surface_mask]
    ds_y_surf = ds_y[surface_mask]

    print(f"\nShock 1 (left,  expect ds≈({float(nx1):.3f}, {float(ny1):.3f})):")
    if left_shock.sum() > 0:
        print(f"  cells={int(left_shock.sum())}  ds_x={float(ds_x_surf[left_shock].mean()):.3f}  ds_y={float(ds_y_surf[left_shock].mean()):.3f}")
    else:
        print("  no surface cells found on left side")

    print(f"Shock 2 (right, expect ds≈({float(nx2):.3f}, {float(ny2):.3f})):")
    if right_shock.sum() > 0:
        print(f"  cells={int(right_shock.sum())}  ds_x={float(ds_x_surf[right_shock].mean()):.3f}  ds_y={float(ds_y_surf[right_shock].mean()):.3f}")
    else:
        print("  no surface cells found on right side")


# ============================================================================
# PLOTS
# ============================================================================

#%%
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Two Shocks at Different Angles ({ANGLE_1}° and {ANGLE_2}°)", fontsize=13)

x_np = np.array(x)
y_np = np.array(y)

# 1. Pressure
im0 = axes[0, 0].pcolormesh(x_np, y_np, np.array(p_final), cmap="viridis")
axes[0, 0].set_title("Pressure")
axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])

# 2. Density
im1 = axes[0, 1].pcolormesh(x_np, y_np, np.array(rho_final), cmap="plasma")
axes[0, 1].set_title("Density")
axes[0, 1].set_xlabel("x"); axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1])

# 3. Shock surface + zone on pressure
axes[0, 2].pcolormesh(x_np, y_np, np.array(p_final), cmap="viridis", alpha=0.8)
axes[0, 2].contour(x_np, y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="red", linewidths=1.5)
axes[0, 2].contourf(x_np, y_np, np.array(result.shock_zones).astype(float),
                    levels=[0.5, 1.5], colors=["green"], alpha=0.25)
axes[0, 2].set_title("Shock surface (red) & zone (green)")
axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y")

# 4. Mach number
im3 = axes[1, 0].pcolormesh(x_np, y_np, np.array(result.mach_numbers), cmap="hot")
axes[1, 0].set_title("Mach number (surface cells only)")
axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 0])

# 5. shock_direction quiver — key visual: two different arrow directions
step = 8
axes[1, 1].pcolormesh(x_np, y_np, np.array(p_final), cmap="viridis", alpha=0.5)
axes[1, 1].quiver(
    x_np[::step, ::step], y_np[::step, ::step],
    np.array(ds_x)[::step, ::step], np.array(ds_y)[::step, ::step],
    scale=20, color="white", alpha=0.8,
)
axes[1, 1].contour(x_np, y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="red", linewidths=1.5)
axes[1, 1].set_title("shock_direction (quiver)\nleft arrows ↗, right arrows ↘")
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")

# 6. ds_y component — signed: left shock positive, right shock negative
im5 = axes[1, 2].pcolormesh(x_np, y_np, np.array(ds_y), cmap="RdBu", vmin=-1, vmax=1)
axes[1, 2].contour(x_np, y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="black", linewidths=1.0)
axes[1, 2].set_title(f"ds_y component\nexpect +{float(ny1):.2f} left, {float(ny2):.2f} right")
axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("y")
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.show()