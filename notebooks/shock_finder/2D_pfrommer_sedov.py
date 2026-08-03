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
# DIAGNOSTICS (radial-direction check, independent of plotting)
# ============================================================================
surface = np.array(result.shock_surface_cells).astype(bool)
shock_dir_x_np = np.array(shock_dir_x)
shock_dir_y_np = np.array(shock_dir_y)

geometry_x_surface_np = np.array(geometry_x)[surface]
geometry_y_surface_np = np.array(geometry_y)[surface]
shock_dir_x_surface_np = shock_dir_x_np[surface]
shock_dir_y_surface_np = shock_dir_y_np[surface]

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

dot_radial = u_detected_x * u_radial_x + u_detected_y * u_radial_y
abs_dot_radial = np.abs(dot_radial)

angle_error_deg = np.degrees(np.arccos(np.clip(abs_dot_radial, 0.0, 1.0)))

print("\nShock direction vs expected radial direction:")
print(f"  mean |dot| = {abs_dot_radial.mean():.4f}")
print(f"  min  |dot| = {abs_dot_radial.min():.4f}")
print(f"  mean angle error = {angle_error_deg.mean():.2f} deg")
print(f"  max  angle error = {angle_error_deg.max():.2f} deg")
print(f"  90th percentile angle error = {np.percentile(angle_error_deg, 90):.2f} deg")


#%%
# PLOTS

fig, axes = plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=box_size,
    mach_vmin=1.0,
    mach_vmax=100.0,
    suptitle=f"Point Explosion at center {TARGET_CENTER} — single outward shock",
)

# ----------------------------------------------------------------------
# 6. Problem-specific panel: one-dimensional cut through center at 0°
# ----------------------------------------------------------------------
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


cx, cy = TARGET_CENTER
geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)

j_center = np.argmin(np.abs(geometry_y_np[0, :] - cy))

s_cut = geometry_x_np[:, j_center]
p_cut = np.array(p_final)[:, j_center]
zone_cut = np.array(result.shock_zones).astype(bool)[:, j_center]
surface_cut = np.array(result.shock_surface_cells).astype(bool)[:, j_center]

sort_idx = np.argsort(s_cut)
s_cut = s_cut[sort_idx]
p_cut = p_cut[sort_idx]
zone_cut = zone_cut[sort_idx]
surface_cut = surface_cut[sort_idx]

ax6 = axes[1, 2]

ax6.plot(s_cut, p_cut, label="pressure", linewidth=2.0)

ax6.fill_between(
    s_cut, 0.0, np.nanmax(p_cut) * 1.05,
    where=zone_cut, alpha=0.20, color="green", label="shock zone"
)

surface_segments = get_mask_segments(s_cut, surface_cut)
first = True
for s0, s1 in surface_segments:
    s_mid = 0.5 * (s0 + s1)
    ax6.axvline(
        s_mid, color="red", linestyle="--", linewidth=1.2,
        label="shock surface" if first else None
    )
    first = False

ax6.set_title("Cut through center at 0°\n(horizontal diameter, full pressure profile)")
ax6.set_xlabel("x along center cut")
ax6.set_ylabel("P")
ax6.set_ylim(0.0, np.nanmax(p_cut) * 1.10)
ax6.grid(alpha=0.25)
ax6.legend(fontsize=8)

plt.show()

# %%