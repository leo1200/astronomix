# ============================================================================
# 2D Shock Finder Test — Two Shocks Intersecting at Center (X shape)
# ============================================================================
# Two Sod-like discontinuities, both passing through (0.5, 0.5) but at
# different angles (+30° and -30°). They form a clean X shape.
#
# This is a stress test: the intersection region has competing shock normals
# and the shock finder must decide what to do there. There is no single
# correct answer at the intersection — we just want to verify:
#   1. both arms of each shock are detected away from the intersection
#   2. the finder does not crash or produce garbage everywhere
#   3. d_s is consistent along each arm (not at the intersection itself)
#
# Initial conditions:
#   A point is "high pressure" if it is on the high side of EITHER shock.
#   The four quadrants of the X get different combinations:
#     top/bottom  (high for shock 1, low for shock 2)  → p=1.0
#     left/right  (low for shock 1, high for shock 2)  → p=1.0
#     ... actually both shocks high on same side        → p=1.0
#   Net effect: two wedge-shaped high pressure regions separated by the X.
#
# Ground truth (away from intersection):
#   shock 1 arms: ds ≈ ( cos30°,  sin30°) = ( 0.866,  0.500)
#   shock 2 arms: ds ≈ ( cos30°, -sin30°) = ( 0.866, -0.500)
#   intersection: ambiguous — noisy d_s expected, not a failure
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

ANGLE_1 =  30.0    # degrees — normal of shock 1
ANGLE_2 = -30.0    # degrees — normal of shock 2
# both pass through the center
CENTER = (0.5, 0.5)

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

x = helper_data.geometric_centers[..., 0]
y = helper_data.geometric_centers[..., 1]


# ============================================================================
# INITIAL CONDITIONS
# ============================================================================
# signed distance from each front (both through CENTER)
# high pressure where dist < 0 for that shock

theta1 = jnp.deg2rad(ANGLE_1)
theta2 = jnp.deg2rad(ANGLE_2)

nx1, ny1 = jnp.cos(theta1), jnp.sin(theta1)
nx2, ny2 = jnp.cos(theta2), jnp.sin(theta2)

dist1 = (x - CENTER[0]) * nx1 + (y - CENTER[1]) * ny1   # signed dist from shock 1
dist2 = (x - CENTER[0]) * nx2 + (y - CENTER[1]) * ny2   # signed dist from shock 2

high1 = dist1 < 0
high2 = dist2 < 0

# high pressure if on the high side of either shock
in_high = high1 | high2

p   = jnp.where(in_high, 1.0, 0.1  )
rho = jnp.where(in_high, 1.0, 0.125)
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
print(f"=== Two Intersecting Shocks at Center ({ANGLE_1}° and {ANGLE_2}°) ===")
print(f"Shock 1 normal: ({float(nx1):.3f}, {float(ny1):.3f})")
print(f"Shock 2 normal: ({float(nx2):.3f}, {float(ny2):.3f})")
print(f"num_shocks (surface cells): {result.num_shocks}")

surface_mask = result.shock_surface_cells

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found")
else:
    surface_mach = result.mach_numbers[surface_mask]
    print(f"Mach at surface: min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")

    # define "near intersection" as within 0.1 of center
    surf_x = x[surface_mask]
    surf_y = y[surface_mask]
    dist_from_center = jnp.sqrt((surf_x - 0.5)**2 + (surf_y - 0.5)**2)
    near_intersection = dist_from_center < 0.1
    far_from_intersection = ~near_intersection

    ds_x_surf = ds_x[surface_mask]
    ds_y_surf = ds_y[surface_mask]

    print(f"\nAway from intersection ({int(far_from_intersection.sum())} cells):")
    if far_from_intersection.sum() > 0:
        print(f"  ds_x mean={float(ds_x_surf[far_from_intersection].mean()):.3f}  std={float(ds_x_surf[far_from_intersection].std()):.3f}")
        print(f"  ds_y mean={float(ds_y_surf[far_from_intersection].mean()):.3f}  std={float(ds_y_surf[far_from_intersection].std()):.3f}")
        print(f"  (low std = consistent direction along each arm)")

    print(f"\nNear intersection ({int(near_intersection.sum())} cells):")
    if near_intersection.sum() > 0:
        print(f"  ds_x mean={float(ds_x_surf[near_intersection].mean()):.3f}  std={float(ds_x_surf[near_intersection].std()):.3f}")
        print(f"  ds_y mean={float(ds_y_surf[near_intersection].mean()):.3f}  std={float(ds_y_surf[near_intersection].std()):.3f}")
        print(f"  (high std expected — ambiguous region)")


# ============================================================================
# PLOTS
# ============================================================================

#%%
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"Two Intersecting Shocks at Center ({ANGLE_1}° and {ANGLE_2}°) — X shape", fontsize=13)

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
# mark intersection region
circle = plt.Circle((0.5, 0.5), 0.1, color="white", fill=False,
                     linestyle="--", linewidth=1.5, label="intersection zone")
axes[0, 2].add_patch(circle)
axes[0, 2].set_title("Shock surface (red) & zone (green)\ndashed circle = intersection region")
axes[0, 2].set_xlabel("x"); axes[0, 2].set_ylabel("y")

# 4. Mach number
im3 = axes[1, 0].pcolormesh(x_np, y_np, np.array(result.mach_numbers), cmap="hot")
axes[1, 0].set_title("Mach number (surface cells only)")
axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 0])

# 5. shock_direction quiver — key visual: X pattern of arrows
step = 8
axes[1, 1].pcolormesh(x_np, y_np, np.array(p_final), cmap="viridis", alpha=0.5)
axes[1, 1].quiver(
    x_np[::step, ::step], y_np[::step, ::step],
    np.array(ds_x)[::step, ::step], np.array(ds_y)[::step, ::step],
    scale=20, color="white", alpha=0.8,
)
axes[1, 1].contour(x_np, y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="red", linewidths=1.5)
circle2 = plt.Circle((0.5, 0.5), 0.1, color="yellow", fill=False,
                      linestyle="--", linewidth=1.5)
axes[1, 1].add_patch(circle2)
axes[1, 1].set_title("shock_direction (quiver)\nexpect X pattern, noisy inside circle")
axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("y")

# 6. ds_y component — shows the two arms cleanly
im5 = axes[1, 2].pcolormesh(x_np, y_np, np.array(ds_y), cmap="RdBu", vmin=-1, vmax=1)
axes[1, 2].contour(x_np, y_np, np.array(result.shock_surface_cells).astype(float),
                   levels=[0.5], colors="black", linewidths=1.0)
circle3 = plt.Circle((0.5, 0.5), 0.1, color="black", fill=False,
                      linestyle="--", linewidth=1.5)
axes[1, 2].add_patch(circle3)
axes[1, 2].set_title(f"ds_y component\nexpect +{float(ny1):.2f} (shock 1 arm), {float(ny2):.2f} (shock 2 arm)")
axes[1, 2].set_xlabel("x"); axes[1, 2].set_ylabel("y")
plt.colorbar(im5, ax=axes[1, 2])

plt.tight_layout()
plt.show()