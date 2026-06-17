# ============================================================================
# 2D Shock Finder Test — Two Parallel Rotated Shocks
# ============================================================================
# High pressure in the middle region, low pressure on both sides.
# This drives two shocks propagating outward in opposite directions
# along the shock normal — like a 1D blast wave, rotated by SHOCK_ANGLE.
#
# Initial conditions (three regions along the normal direction):
#   left region  (d < -1/6): rho=0.125, p=0.1   (low pressure)
#   mid  region  (-1/6<d<1/6): rho=1.0, p=1.0   (high pressure)
#   right region (d >  1/6): rho=0.125, p=0.1   (low pressure)
#
# Ground truth:
#   - two distinct shock surfaces, each perpendicular to the shock normal
#   - the shock normal has angle SHOCK_ANGLE from the x-axis
#   - shocks propagate in opposite directions → d_s points outward on each
#   - ds_x mean ≈ 0 (left shock cancels right shock), but per-shock ≈ ±cos θ
#   - Mach numbers should be roughly symmetric (same jump ratio on both sides)
# ============================================================================

#%%
from typing import cast

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
params = SimulationParams(t_end=0.15)   # slightly shorter — keeps shocks separated

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


# Initial conditions (three regions along the normal direction):
#   left region  (d < -1/6): rho=0.125, p=0.1   (low pressure)
#   mid  region  (-1/6<d<1/6): rho=1.0, p=1.0   (high pressure)
#   right region (d >  1/6): rho=0.125, p=0.1   (low pressure)

SHOCK_ANGLE = 30.0        # degrees — expected angle of shock normal from x-axis
                          # both shocks share the same angle

# n̂ = (cos θ, sin θ) is the shock normal, pointing outward from the high-pressure region
theta_rad = jnp.deg2rad(SHOCK_ANGLE)
nx_hat    = jnp.cos(theta_rad)
ny_hat    = jnp.sin(theta_rad)

signed_dist = (geometry_x - 0.5) * nx_hat + (geometry_y - 0.5) * ny_hat # (nx, ny) — signed distance from cell center to line through (0.5, 0.5) with normal n̂

# three regions
in_left_mask  = signed_dist < -1/6
in_right_mask = signed_dist >  1/6
in_mid_mask   = ~in_left_mask & ~in_right_mask

# high pressure driver in the middle → two shocks propagate outward
rho = jnp.where(in_mid_mask, 1.0,   0.125)
p   = jnp.where(in_mid_mask, 1.0,   0.1  )
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


#%%
# RUN SIMULATION
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
shock_dir = result.shock_direction   # (2, nx, ny)
shock_dir_x = result.shock_direction[0]   # (nx, ny)
shock_dir_y = result.shock_direction[1]   # (nx, ny)


#%%
# DIAGNOSTICS

print(f"=== Two Parallel Shocks ({SHOCK_ANGLE}°) ===")
print(f"Shock normal              : ({float(nx_hat):.3f}, {float(ny_hat):.3f})")
print(f"num_shocks (surface cells): {result.num_shocks}")
print("Expected                  : two separated diagonal shock-surface clusters")

surface_mask = result.shock_surface_cells
surface_mach = result.mach_numbers[surface_mask]

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found — shock may have left domain or is too weak")
else:
    print(f"Mach at surface           : min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")
    print(f"shock_dir_x at surface    : mean={float(shock_dir_x[surface_mask].mean()):.3f}  (expect ≈ 0 — left/right shocks cancel)")
    print(f"shock_dir_y at surface    : mean={float(shock_dir_y[surface_mask].mean()):.3f}  (expect ≈ 0 — left/right shocks cancel)")
    print(f"  (each shock individually should have |shock_dir_x|≈{float(nx_hat):.3f}, |shock_dir_y|≈{float(ny_hat):.3f})")

    # check two distinct clusters along normal
    surface_dist = signed_dist[surface_mask]

    left_surface_mask  = surface_mask & (signed_dist < 0)
    right_surface_mask = surface_mask & (signed_dist > 0)

    left_dist  = signed_dist[left_surface_mask]
    right_dist = signed_dist[right_surface_mask]

    print("\nSurface cell positions along normal:")

    print(
        f"  left shock : "
        f"count={left_dist.size}, "
        f"min={float(left_dist.min()):.3f}, "
        f"max={float(left_dist.max()):.3f}, "
        f"mean={float(left_dist.mean()):.3f}"
    )

    print(
        f"  right shock: "
        f"count={right_dist.size}, "
        f"min={float(right_dist.min()):.3f}, "
        f"max={float(right_dist.max()):.3f}, "
        f"mean={float(right_dist.mean()):.3f}"
    )

# ============================================================================
# PLOTS
# ============================================================================

#%%
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Two Parallel Rotated Shocks ({SHOCK_ANGLE}°) — Shock Finder Validation", fontsize=13)

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
axes[0, 2].pcolormesh(geometry_x_np, geometry_y_np, np.array(p_final), cmap="viridis", alpha=0.8)
axes[0, 2].contour(geometry_x_np, geometry_y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="red", linewidths=1.5)
axes[0, 2].contourf(geometry_x_np, geometry_y_np, np.array(result.shock_zones).astype(float),
                    levels=[0.5, 1.5], colors=["green"], alpha=0.25)
axes[0, 2].set_title("Shock surface (red) & zone (green)")
axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y")

# 4. Mach number
im3 = axes[1, 0].pcolormesh(geometry_x_np, geometry_y_np, np.array(result.mach_numbers), cmap="hot")
axes[1, 0].set_title("Mach number (surface cells only)")
axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 0])

# 5. shock_direction quiver
step = 8
axes[1, 1].pcolormesh(geometry_x_np, geometry_y_np, np.array(p_final), cmap="viridis", alpha=0.5)
axes[1, 1].quiver(
    geometry_x_np[::step, ::step], geometry_y_np[::step, ::step],
    np.array(shock_dir_x)[::step, ::step], np.array(shock_dir_y)[::step, ::step],
    scale=20, color="white", alpha=0.8,
)
axes[1, 1].contour(geometry_x_np, geometry_y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="red", linewidths=1.5)
axes[1, 1].set_title(f"shock_direction (quiver)\nexpect outward arrows on each shock front")
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")

# 6. Slice along shock normal through center
t_vals   = np.linspace(-0.5, 0.5, 300)
x_sample = np.clip(0.5 + t_vals * float(nx_hat), 0.01, 0.99)
y_sample = np.clip(0.5 + t_vals * float(ny_hat), 0.01, 0.99)

cell_size = box_size / num_cells
xi = np.clip((x_sample / cell_size).astype(int), 0, num_cells - 1)
yi = np.clip((y_sample / cell_size).astype(int), 0, num_cells - 1)

p_along    = np.array(p_final)[xi, yi]
surf_along = np.array(result.shock_surface_cells)[xi, yi]
zone_along = np.array(result.shock_zones)[xi, yi]

axes[1, 2].plot(t_vals, p_along, label="pressure")
axes[1, 2].fill_between(t_vals, 0, 1, where=zone_along,
                         alpha=0.2, color="green", label="shock zone")
first = True
for ti in t_vals[surf_along]:
    axes[1, 2].axvline(
        ti,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="shock surface" if first else None
    )
    first = False
    
axes[1, 2].set_title(f"Slice along normal (θ={SHOCK_ANGLE}°)\nexpect 2 red lines + 2 green zones")
axes[1, 2].set_xlabel("distance along normal"); axes[1, 2].set_ylabel("P")
axes[1, 2].legend(fontsize=8)

plt.tight_layout()
plt.show()