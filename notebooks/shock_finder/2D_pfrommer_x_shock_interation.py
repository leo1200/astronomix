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


# %%
# PLOTS

fig, axes = plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=box_size,
    mach_vmin=1.0,
    mach_vmax=2.0,
    suptitle="2D Sod Shock Tube - Shock Finder Validation",
)

# ----------------------------------------------------------------------
# 6. Problem-specific panel: 1D slice at mid-row (y = mid, x runs)
# ----------------------------------------------------------------------
mid = num_cells // 2

x_slice    = geometry_x[:, mid]
p_slice    = p_final[:, mid]
surf_slice = result.shock_surface_cells[:, mid]
zone_slice = result.shock_zones[:, mid]

p_arr_slice    = np.array(p_slice)
surf_arr_slice = np.array(surf_slice)
zone_arr_slice = np.array(zone_slice)

ax6 = axes[1, 2]

ax6.plot(x_slice, p_slice, label="pressure")

ax6.fill_between(
    x_slice, 0, 1,
    where=zone_arr_slice,
    alpha=0.20,
    color="green",
    linewidth=5,
    label="shock zone"
)

first = True
for ti in x_slice[surf_slice]:
    ax6.axvline(
        ti,
        color="red",
        linestyle="--",
        linewidth=0.5,
        label="shock surface" if first else None
    )
    first = False

ax6.set_title(f"1D slice at y={mid} (mid row)")
ax6.set_xlabel("x")
ax6.set_ylabel("P")
ax6.set_box_aspect(1)
ax6.legend(fontsize=8)

plt.show()

# ----------------------------------------------------------------------
# 7. Velocity x (separate panel, standalone figure)
# ----------------------------------------------------------------------
fig7, ax7 = plt.subplots(figsize=(5, 5), constrained_layout=True)

im7 = ax7.pcolormesh(
    np.array(geometry_x), np.array(geometry_y),
    np.array(vx_final),
    cmap="plasma",
    shading="auto",
)
ax7.set_title("Velocity x (uniform in y)")
ax7.set_xlabel("x")
ax7.set_ylabel("y")
ax7.set_aspect("equal", adjustable="box")
ax7.set_xlim(0, box_size)
ax7.set_ylim(0, box_size)
plt.colorbar(im7, ax=ax7, label="Velocity x")

plt.show()

# %%