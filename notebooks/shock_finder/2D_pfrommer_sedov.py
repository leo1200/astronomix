# ============================================================================
# 2D Shock Finder Test — Point Explosion (Sedov-like, single outward shock)
# ============================================================================
# A single point-like energy injection at the domain center drives one
# outward-propagating circular shock (Sedov-Taylor-like blast wave).
#
# Initial conditions:
#   A small disk of radius r_explosion at the center is set to a high
#   pressure p_explosion_gas, chosen so that integrating p/(gamma-1) over
#   the disk area gives back the target explosion energy E_explosion.
#   Everywhere else: ambient density/pressure, no initial velocity.
#
# Stress test goals (NOT a clean Sedov validation):
#   1. A single closed circular shock surface is detected
#   2. Shock finder does not crash or produce garbage everywhere
#   3. Detected shock_direction points radially outward from the center
#      along the shock front, at all angles (azimuthal symmetry check)
#   4. The very center (~r < r_explosion, pre-shock-formation region) and
#      the exact center point itself are expected to be ambiguous/noisy
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
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
geometry_x = geometric_centers[..., 0]  # (nx, ny)
geometry_y = geometric_centers[..., 1]  # (nx, ny)


# ============================================================================
# INITIAL CONDITIONS — point explosion (single outward-propagating shock)
#
# A small disk of radius r_explosion centered at TARGET_CENTER is given a
# uniform high pressure such that integrating p/(gamma-1) over the disk
# area reproduces E_explosion. Outside the disk: ambient density/pressure.
# No initial velocity anywhere.
# The center of the domain is the region of interest (single point source).
# ============================================================================

# center of the explosion
TARGET_CENTER = (0.5, 0.5)
center_x, center_y = TARGET_CENTER

# total explosion energy (code units)
E_explosion = 1.0

# ambient (background) physical conditions
rho_ambient = 1.0
p_ambient   = 1e-4

# radius of the injection disk (code units)
r_explosion = 0.05

# distance of every cell from the explosion center
dx_from_center = geometry_x - center_x
dy_from_center = geometry_y - center_y

r = jnp.sqrt(dx_from_center**2 + dy_from_center**2)

# injection area (2D analog of the 3D injection_volume in the point-explosion setup)
injection_area = jnp.pi * r_explosion**2

# adiabatic index of the gas
gamma_gas = params.gamma

# E = p * A / (gamma - 1)  =>  p = E * (gamma - 1) / A
p_explosion_gas = E_explosion * (gamma_gas - 1) / injection_area

# pressure: high within the explosion disk, ambient elsewhere
p   = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)
rho = jnp.ones_like(geometry_x) * rho_ambient
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
print(f"=== Point Explosion at center {TARGET_CENTER} ===")
print(f"num_shocks (surface cells): {result.num_shocks}")

surface_mask = result.shock_surface_cells

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found")
else:
    surface_mach = result.mach_numbers[surface_mask]
    print(f"Mach at surface: min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")

#%%
# PLOTS
# 1-5 are standard spatial diagnostics.
# 6 is only the 0° horizontal slice through the explosion center.

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 10),
    constrained_layout=True
)

fig.suptitle(
    f"Point Explosion at center {TARGET_CENTER} — single outward shock",
    fontsize=13
)

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)


# ============================================================================
# Helper for 1D shock-surface markers
# ============================================================================
def get_mask_segments(s, mask, max_gap_samples=3):
    """
    Group nearby True samples into one continuous crossing.
    This avoids drawing many repeated red lines for one shock crossing.
    """

    mask = np.array(mask).astype(bool)
    true_idx = np.where(mask)[0]

    if len(true_idx) == 0:
        return []

    groups = []
    current_group = [true_idx[0]]

    for idx in true_idx[1:]:
        if idx - current_group[-1] <= max_gap_samples:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]

    groups.append(current_group)

    segments = []
    for group in groups:
        s0 = s[group[0]]
        s1 = s[group[-1]]
        segments.append((s0, s1))

    return segments


# ============================================================================
# 1. Pressure
# ============================================================================
im0 = axes[0, 0].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto"
)

axes[0, 0].set_title("Pressure")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("y")
plt.colorbar(im0, ax=axes[0, 0])


# ============================================================================
# 2. Density
# ============================================================================
im1 = axes[0, 1].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    np.array(rho_final),
    cmap="plasma",
    shading="auto"
)

axes[0, 1].set_title("Density")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("y")
plt.colorbar(im1, ax=axes[0, 1])


# ============================================================================
# 3. Shock surface + shock zone on pressure
# ============================================================================
axes[0, 2].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.8
)

axes[0, 2].contourf(
    geometry_x_np,
    geometry_y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.25
)

axes[0, 2].contour(
    geometry_x_np,
    geometry_y_np,
    np.array(result.shock_surface_cells).astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.5
)

axes[0, 2].set_title("Shock surfaces and shock zones")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

axes[0, 2].legend(
    handles=[
        Patch(facecolor="green", edgecolor="green", alpha=0.25, label="shock zone"),
        Line2D([0], [0], color="red", lw=1.5, label="shock surface"),
    ],
    loc="upper right",
    fontsize=8
)


# ============================================================================
# 4. Mach number
# ============================================================================
mach_surface_only = np.array(result.mach_numbers)

im3 = axes[1, 0].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    mach_surface_only,
    cmap="hot",
    vmin=1.0,
    vmax=100.0,
    shading="auto"
)

axes[1, 0].set_title("Shock Mach number at surface cells")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("y")
plt.colorbar(im3, ax=axes[1, 0], label="Shock Mach number")


# ============================================================================
# 5. Shock direction at surface cells
# ============================================================================
axes[1, 1].pcolormesh(
    geometry_x_np,
    geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.55
)

axes[1, 1].contourf(
    geometry_x_np,
    geometry_y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.20
)

surface = np.array(result.shock_surface_cells).astype(bool)

axes[1, 1].contour(
    geometry_x_np,
    geometry_y_np,
    surface.astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.8
)

geometry_x_surface_np = geometry_x_np[surface]
geometry_y_surface_np = geometry_y_np[surface]
shock_dir_x_surface_np = np.array(shock_dir_x)[surface]
shock_dir_y_surface_np = np.array(shock_dir_y)[surface]

mag_shock_dir_surface = np.sqrt(
    shock_dir_x_surface_np**2 + shock_dir_y_surface_np**2
)

valid = mag_shock_dir_surface > 0

geometry_x_surface_np = geometry_x_surface_np[valid]
geometry_y_surface_np = geometry_y_surface_np[valid]

u_shock_dir_x_surface_np = shock_dir_x_surface_np[valid] / mag_shock_dir_surface[valid]
u_shock_dir_y_surface_np = shock_dir_y_surface_np[valid] / mag_shock_dir_surface[valid]

# Expected radial direction from explosion center
radial_x = geometry_x_surface_np - center_x
radial_y = geometry_y_surface_np - center_y

radial_mag = np.sqrt(radial_x**2 + radial_y**2)
radial_valid = radial_mag > 0

u_radial_x = radial_x[radial_valid] / radial_mag[radial_valid]
u_radial_y = radial_y[radial_valid] / radial_mag[radial_valid]

u_detected_x = u_shock_dir_x_surface_np[radial_valid]
u_detected_y = u_shock_dir_y_surface_np[radial_valid]

# Compare detected direction with ideal radial direction.
# Use abs because shock_direction may point inward or outward depending on convention.
dot_radial = u_detected_x * u_radial_x + u_detected_y * u_radial_y
abs_dot_radial = np.abs(dot_radial)

angle_error_deg = np.degrees(
    np.arccos(np.clip(abs_dot_radial, 0.0, 1.0))
)

print("\nShock direction vs expected radial direction:")
print(f"  mean |dot| = {abs_dot_radial.mean():.4f}")
print(f"  min  |dot| = {abs_dot_radial.min():.4f}")
print(f"  mean angle error = {angle_error_deg.mean():.2f} deg")
print(f"  max  angle error = {angle_error_deg.max():.2f} deg")
print(f"  90th percentile angle error = {np.percentile(angle_error_deg, 90):.2f} deg")


# The quantitative check shows that the detected shock directions are mostly radial. 
# The mean absolute dot product with the expected radial direction is 0.9831, 
# which is close to 1, and the mean angle error is about 8 degrees. 
# This means that, on average, 
# the shock finder is giving directions close to the expected normal direction of the circular shock.

# The arrows do not look perfectly perpendicular everywhere 
# because the shock is represented on a Cartesian grid and is spread over several cells, 
# not an infinitely thin circular line. 
# The red contour is also drawn from a boolean mask of detected shock-surface cells, 
# so it shows the edge of a discrete cell band rather than the exact analytic shock front. 
# Since the shock direction is computed locally from numerical gradients, 
# small cell-to-cell variations can make the arrows look wiggly.

# The maximum angle error is about 27 degrees, 
# and 90% of the cells have an angle error below about 19 degrees. 
# So there are some noisy local cells, 
# but the overall direction calculation appears consistent with an outward circular shock.

n_arrows = 100

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

axes[1, 1].quiver(
    xs_plot,
    ys_plot,
    ux_plot,
    uy_plot,
    angles="xy",
    scale_units="xy",
    scale=20,
    color="white",
    width=0.004,
    headwidth=4,
    headlength=5,
    pivot="middle",
    zorder=20
)

axes[1, 1].set_title(
    "Shock direction at surface cells\nexpect radially outward directions"
)
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

axes[1, 1].legend(
    handles=[
        Patch(facecolor="green", edgecolor="green", alpha=0.20, label="shock zone"),
        Line2D([0], [0], color="red", lw=1.8, label="shock surface"),
        Line2D(
            [0],
            [0],
            color="white",
            lw=0,
            marker=r"$\rightarrow$",
            markersize=12,
            label="shock direction",
        ),
    ],
    loc="upper right",
    fontsize=8
)


# ============================================================================
# 6. One-dimensional cut through center at 0°
# ============================================================================
cx, cy = TARGET_CENTER

# The 0° cut is horizontal through the center: y ≈ cy.
# Since this cut is aligned with the grid, no bilinear interpolation is needed.
j_center = np.argmin(np.abs(geometry_y_np[0, :] - cy))

s_cut = geometry_x_np[:, j_center]
p_cut = np.array(p_final)[:, j_center]
zone_cut = np.array(result.shock_zones).astype(bool)[:, j_center]
surface_cut = np.array(result.shock_surface_cells).astype(bool)[:, j_center]

# Sort by x-coordinate, just in case the grid ordering changes.
sort_idx = np.argsort(s_cut)

s_cut = s_cut[sort_idx]
p_cut = p_cut[sort_idx]
zone_cut = zone_cut[sort_idx]
surface_cut = surface_cut[sort_idx]

ax = axes[1, 2]

ax.plot(
    s_cut,
    p_cut,
    label="pressure",
    linewidth=2.0
)

ax.fill_between(
    s_cut,
    0.0,
    np.nanmax(p_cut) * 1.05,
    where=zone_cut,
    alpha=0.20,
    color="green",
    label="shock zone"
)

surface_segments = get_mask_segments(s_cut, surface_cut)

first = True

for s0, s1 in surface_segments:
    s_mid = 0.5 * (s0 + s1)

    ax.axvline(
        s_mid,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label="shock surface" if first else None
    )

    first = False

ax.set_title(
    "Cut through center at 0°\n(horizontal diameter, full pressure profile)"
)
ax.set_xlabel("x along center cut")
ax.set_ylabel("P")
ax.set_ylim(0.0, np.nanmax(p_cut) * 1.10)
ax.grid(alpha=0.25)
ax.legend(fontsize=8)


# ============================================================================
# Final formatting
# ============================================================================
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