# ============================================================================
# 2D Shock Finder Test — Two Parallel Rotated Shocks (blast wave in 1D)
# ============================================================================
# High pressure in the middle region, low pressure on both sides.
# This drives two shocks propagating outward in opposite directions
# along the shock normal — like a 1D blast wave, rotated by SHOCK_ANGLE.
#
# Initial conditions (three regions along the normal direction):
#   left region  (d < -1/6): rho=0.125, p=0.1   (low pressure)
#   mid  region  (-1/6<d<1/6): rho=1.0, p=1.0   (high pressure — the driver)
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

SHOCK_ANGLE = 30.0        # degrees — angle of shock normal from x-axis
                          # both shocks share the same angle

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

x = helper_data.geometric_centers[..., 0]   # (nx, ny)
y = helper_data.geometric_centers[..., 1]   # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — two discontinuities along the normal
# ============================================================================
# signed distance from center along shock normal
# disc 1 at d = -1/6  (left third boundary)
# disc 2 at d = +1/6  (right third boundary)

theta_rad = jnp.deg2rad(SHOCK_ANGLE)
nx_hat    = jnp.cos(theta_rad)
ny_hat    = jnp.sin(theta_rad)

signed_dist = (x - 0.5) * nx_hat + (y - 0.5) * ny_hat

# three regions
in_left  = signed_dist < -1/6
in_right = signed_dist >  1/6
in_mid   = ~in_left & ~in_right

# high pressure driver in the middle → two shocks propagate outward
rho = jnp.where(in_mid, 1.0,   0.125)
p   = jnp.where(in_mid, 1.0,   0.1  )
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

ds_x = result.shock_direction[0]   # (nx, ny)
ds_y = result.shock_direction[1]   # (nx, ny)


# ============================================================================
# DIAGNOSTICS
# ============================================================================

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
    print(f"ds_x at surface           : mean={float(ds_x[surface_mask].mean()):.3f}  (expect ≈ 0 — left/right shocks cancel)")
    print(f"ds_y at surface           : mean={float(ds_y[surface_mask].mean()):.3f}  (expect ≈ 0 — left/right shocks cancel)")
    print(f"  (each shock individually should have |ds_x|≈{float(nx_hat):.3f}, |ds_y|≈{float(ny_hat):.3f})")

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
axes[0, 2].pcolormesh(
    x_np, y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.8
)

# Draw shock zone first
axes[0, 2].contourf(
    x_np, y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.25
)

# Draw shock surface on top
axes[0, 2].contour(
    x_np, y_np,
    np.array(result.shock_surface_cells).astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.5
)

axes[0, 2].set_title("Shock surfaces and shock zones")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

# 4. Mach number
mach_surface_only = np.where(
    np.array(result.shock_surface_cells),
    np.array(result.mach_numbers),
    np.nan
)

im3 = axes[1, 0].pcolormesh(
    x_np,
    y_np,
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
    x_np, y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.55
)

surface = np.array(result.shock_surface_cells)

xs = x_np[surface]
ys = y_np[surface]
ux = np.array(ds_x)[surface]
uy = np.array(ds_y)[surface]

# Normalize
mag = np.sqrt(ux**2 + uy**2)
valid = mag > 0

xs = xs[valid]
ys = ys[valid]
ux = ux[valid] / mag[valid]
uy = uy[valid] / mag[valid]

# Subsample arrows
n_arrows = 16
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

# Draw shock surfaces
axes[1, 1].contour(
    x_np, y_np,
    surface.astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.5
)

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