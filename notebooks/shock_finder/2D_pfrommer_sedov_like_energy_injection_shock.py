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
from astronomix._physics_modules._shock_finder.shock_finder_2d import find_shocks_pfrommer


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

TARGET_CENTER = (0.5, 0.5)

# total explosion energy (code units)
E_explosion = 1.0

# ambient (background) physical conditions
rho_ambient = 1.0
p_ambient   = 1e-4

# radius of the injection disk (code units)
r_explosion = 0.05

# distance of every cell from the explosion center
r = jnp.sqrt(
    (geometry_x - TARGET_CENTER[0]) ** 2
    + (geometry_y - TARGET_CENTER[1]) ** 2
)

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

    # classify surface cells as near/far from the explosion center
    geometry_surface_x = geometry_x[surface_mask]
    geometry_surface_y = geometry_y[surface_mask]
    dist_from_center = jnp.sqrt(
        (geometry_surface_x - TARGET_CENTER[0]) ** 2
        + (geometry_surface_y - TARGET_CENTER[1]) ** 2
    )
    near_center = dist_from_center < r_explosion
    far_from_center = ~near_center

    shock_dir_surface_x = shock_dir_x[surface_mask]
    shock_dir_surface_y = shock_dir_y[surface_mask]

    # expected direction at each surface cell: radially outward from the center
    radial_x = geometry_surface_x - TARGET_CENTER[0]
    radial_y = geometry_surface_y - TARGET_CENTER[1]
    radial_mag = jnp.sqrt(radial_x**2 + radial_y**2)
    radial_mag_safe = jnp.where(radial_mag == 0, 1.0, radial_mag)
    radial_hat_x = radial_x / radial_mag_safe
    radial_hat_y = radial_y / radial_mag_safe

    # compare each surface cell shock direction to the expected radial normal
    # via absolute dot product (shock_direction sign convention may vary)
    dot_radial = shock_dir_surface_x * radial_hat_x + shock_dir_surface_y * radial_hat_y
    align_radial = jnp.abs(dot_radial)

    print("\nRadial alignment away from the center:")
    if far_from_center.sum() > 0:
        print(
            f"  far-from-center cells: count={int(far_from_center.sum())}, "
            f"mean |dot(ds, r_hat)|={float(align_radial[far_from_center].mean()):.3f}"
        )
    if near_center.sum() > 0:
        print(
            f"  near-center cells:     count={int(near_center.sum())}, "
            f"mean |dot(ds, r_hat)|={float(align_radial[near_center].mean()):.3f}  "
            "(expected noisy/ambiguous)"
        )

    # rough estimate of shock front radius (mean distance of far surface cells)
    if far_from_center.sum() > 0:
        print(f"  mean shock front radius: {float(dist_from_center[far_from_center].mean()):.3f}")
        print(f"  std  shock front radius: {float(dist_from_center[far_from_center].std()):.3f}  "
              "(lower = more circular/azimuthally uniform detection)")


#%%
# PLOTS
# expectation of ploting is that it must be environment independent
# means it must visualize the results without put into consideration of what the shock should look like
# 1-5 are standard plots with no environment assumptions, just showing the results
#
# there are some points we put details relating to the expected results, but they are just for reference and not for validating the results
#

fig, axes = plt.subplots(
    3, 3,
    figsize=(15, 15),
    constrained_layout=True
)
fig.suptitle(f"Point Explosion at center {TARGET_CENTER} — single outward shock", fontsize=13)

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)

# ============================================================================
# Helper functions for arbitrary 1D cuts
# ============================================================================
def bilinear_sample(field, x_grid, y_grid, xs, ys):
    """
    Sample a scalar field along arbitrary points using bilinear interpolation.
    Use this for pressure.
    """

    field = np.array(field)

    nx, ny = field.shape

    x_axis = x_grid[:, 0]
    y_axis = y_grid[0, :]

    xs = np.clip(xs, x_axis.min(), x_axis.max())
    ys = np.clip(ys, y_axis.min(), y_axis.max())

    ix = np.interp(xs, x_axis, np.arange(nx))
    iy = np.interp(ys, y_axis, np.arange(ny))

    i0 = np.floor(ix).astype(int)
    j0 = np.floor(iy).astype(int)

    i1 = np.clip(i0 + 1, 0, nx - 1)
    j1 = np.clip(j0 + 1, 0, ny - 1)

    i0 = np.clip(i0, 0, nx - 1)
    j0 = np.clip(j0, 0, ny - 1)

    wx = ix - i0
    wy = iy - j0

    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j1]
    f11 = field[i1, j1]

    return (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f10
        + (1.0 - wx) * wy * f01
        + wx * wy * f11
    )


def nearest_sample(field, x_grid, y_grid, xs, ys):
    """
    Sample a boolean/discrete field along arbitrary points using nearest cells.
    Use this for shock_zones and shock_surface_cells.
    """

    field = np.array(field)

    nx, ny = field.shape

    x_axis = x_grid[:, 0]
    y_axis = y_grid[0, :]

    xs = np.clip(xs, x_axis.min(), x_axis.max())
    ys = np.clip(ys, y_axis.min(), y_axis.max())

    ix = np.interp(xs, x_axis, np.arange(nx))
    iy = np.interp(ys, y_axis, np.arange(ny))

    ii = np.clip(np.rint(ix).astype(int), 0, nx - 1)
    jj = np.clip(np.rint(iy).astype(int), 0, ny - 1)

    return field[ii, jj]


def extract_cut(p0, p1, n_samples=500):
    """
    Extract pressure, shock zone, and shock surface along a line from p0 to p1.

    p0 = (x0, y0)
    p1 = (x1, y1)
    """

    x0, y0 = p0
    x1, y1 = p1

    t = np.linspace(0.0, 1.0, n_samples)

    xs = x0 + t * (x1 - x0)
    ys = y0 + t * (y1 - y0)

    # Distance along the cut, like x_slice in standard Sod
    s = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2)

    p_cut = bilinear_sample(
        np.array(p_final),
        geometry_x_np,
        geometry_y_np,
        xs,
        ys,
    )

    zone_cut = nearest_sample(
        np.array(result.shock_zones).astype(bool),
        geometry_x_np,
        geometry_y_np,
        xs,
        ys,
    ).astype(bool)

    surface_cut = nearest_sample(
        np.array(result.shock_surface_cells).astype(bool),
        geometry_x_np,
        geometry_y_np,
        xs,
        ys,
    ).astype(bool)

    return s, p_cut, zone_cut, surface_cut


def get_mask_segments(s, mask, max_gap_samples=3):
    """
    Group nearby True samples into one continuous crossing.
    This prevents thick red blocks from many repeated axvline calls.
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
axes[0, 2].pcolormesh(
    geometry_x_np, geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.8
)

# Draw shock zone first
axes[0, 2].contourf(
    geometry_x_np, geometry_y_np,
    np.array(result.shock_zones).astype(float),
    levels=[0.5, 1.5],
    colors=["green"],
    alpha=0.25
)

# Draw shock surface on top
axes[0, 2].contour(
    geometry_x_np, geometry_y_np,
    np.array(result.shock_surface_cells).astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.5
)

axes[0, 2].set_title("Shock surfaces and shock zones")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("y")

# 4. Mach number
mach_surface_only = np.array(result.mach_numbers)

im3 = axes[1, 0].pcolormesh(
    geometry_x_np,
    geometry_y_np,
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
    geometry_x_np, geometry_y_np,
    np.array(p_final),
    cmap="viridis",
    shading="auto",
    alpha=0.55
)

# Shock surface mask
surface = np.array(result.shock_surface_cells)
# Draw shock surface
axes[1, 1].contour(
    geometry_x_np, geometry_y_np,
    surface.astype(float),
    levels=[0.5],
    colors="red",
    linewidths=1.8
)

# Surface-cell coordinates and directions
geometry_x_surface_np = geometry_x_np[surface]
geometry_y_surface_np = geometry_y_np[surface]
shock_dir_x_surface_np = np.array(shock_dir_x)[surface]
shock_dir_y_surface_np = np.array(shock_dir_y)[surface]

# remove 0 direction cells to avoid NaNs when normalizing
mag_shock_dir_surface = np.sqrt(shock_dir_x_surface_np**2 + shock_dir_y_surface_np**2)
valid = mag_shock_dir_surface > 0  # could be zero -> skip
geometry_x_surface_np = geometry_x_surface_np[valid]
geometry_y_surface_np = geometry_y_surface_np[valid]

# Normalize arrows to unit length
u_shock_dir_x_surface_np = shock_dir_x_surface_np[valid] / mag_shock_dir_surface[valid]
u_shock_dir_y_surface_np = shock_dir_y_surface_np[valid] / mag_shock_dir_surface[valid]


# Subsample arrows
n_arrows = 30
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

axes[1, 1].set_title("Shock direction at surface cells\nexpect radially outward directions")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("y")

# ============================================================================
# 6-8. One-dimensional cuts through the explosion center, at different angles
# ============================================================================
# For a point explosion the shock front is (statistically) circular, so cuts
# at different angles through the center should all show a pressure jump at
# roughly the same distance from the center — this is the azimuthal-symmetry
# check analogous to the angle-alignment check used for the X-shock case.

cx, cy = TARGET_CENTER
cut_radius = 0.5  # half-diagonal-ish reach within the unit box from the center

def endpoints_through_center(angle_deg, radius=cut_radius, center=(cx, cy)):
    theta = np.deg2rad(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    p0 = (center[0] - radius * dx, center[1] - radius * dy)
    p1 = (center[0] + radius * dx, center[1] + radius * dy)
    return p0, p1

cut_angles = [0.0, 60.0, 130.0]

cuts = []
for ang, ax_ in zip(cut_angles, [axes[1, 2], axes[2, 0], axes[2, 1]]):
    p0, p1 = endpoints_through_center(ang)
    cuts.append({
        "title": f"Cut through center at {ang:.0f}°\n(diameter, full pressure profile)",
        "p0": p0,
        "p1": p1,
        "ax": ax_,
    })

# 4th cut: straight horizontal line, offset from the center, to show a slice
# that does NOT pass through the explosion origin (sanity check: should show
# two crossings, symmetric about the center's x/y, since it clips the circle).
cuts.append({
    "title": "Cut: horizontal slice offset from center\n(clips the circular shock at two points)",
    "p0": (0.0, cy + 0.2),
    "p1": (1.0, cy + 0.2),
    "ax": axes[2, 2],
})

for cut in cuts:
    ax = cut["ax"]

    s_cut, p_cut, zone_cut, surface_cut = extract_cut(
        cut["p0"],
        cut["p1"],
        n_samples=500,
    )

    # Pressure line, same idea as standard Sod plot 6
    ax.plot(
        s_cut,
        p_cut,
        label="pressure",
        linewidth=2.0,
    )

    # Shock zone, same idea as standard Sod plot 6
    ax.fill_between(
        s_cut,
        0.0,
        np.nanmax(p_cut) * 1.05,
        where=zone_cut,
        alpha=0.20,
        color="green",
        label="shock zone",
    )

    # Shock surface markers: group nearby detections into one representative line
    surface_segments = get_mask_segments(s_cut, surface_cut)

    first = True
    for s0, s1 in surface_segments:
        s_mid = 0.5 * (s0 + s1)

        ax.axvline(
            s_mid,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="shock surface" if first else None,
        )

        first = False

    ax.set_title(cut["title"])
    ax.set_xlabel("distance along cut")
    ax.set_ylabel("P")
    ax.set_ylim(0.0, np.nanmax(p_cut) * 1.10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

for ax in axes.flat:
    ax.set_box_aspect(1)

# Only 2D spatial panels should use equal x-y aspect.
# The 1D cut plots should stay normal line plots.
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