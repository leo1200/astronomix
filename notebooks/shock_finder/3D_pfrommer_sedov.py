# ============================================================================
# 3D Shock Finder Test — Point Explosion (Sedov-Taylor blast wave)
# ============================================================================
# A single point-like energy injection at the domain center drives one
# outward-propagating spherical shock (Sedov-Taylor blast wave).
#
# Same setup pattern as the 2D test, generalized to 3D:
#   - injection sphere of radius r_explosion at domain center
#   - pressure inside sphere set so that integrating p/(gamma-1) over the
#     sphere volume reproduces E_explosion
#   - ambient density/pressure elsewhere, zero initial velocity
#
# Diagnostics:
#   - three orthogonal mid-plane projections (xy @ z=0.5, xz @ y=0.5,
#     yz @ x=0.5) showing density/pressure with shock zones, shock
#     surface, and shock direction arrows overlaid
#   - 3D average shock radius, computed as the mean of
#     sqrt(x^2 + y^2 + z^2) (measured from the explosion center) over all
#     shock-surface cells, compared against the analytic Sedov-Taylor
#     radius R(t) = xi_0 * (E * t^2 / rho_ambient)^(1/5)
# ============================================================================

#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.option_classes.simulation_config import CARTESIAN, HYBRID_HLLC

from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_surface_3d
from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_projections_3d


#%%
# INITIAL CONDITIONS
# refer to \tests\hydro_tests\sedov_radial.py
num_cells = 64          # per-axis resolution (3D is expensive: 64^3 cells)
box_size  = 1.0
config = SimulationConfig(
    geometry = CARTESIAN,
    progress_bar = True,
    runtime_debugging = False,
    riemann_solver = HYBRID_HLLC,
    dimensionality = 3,
    exact_end_time = True,
    num_cells = num_cells,
    box_size=box_size,
    return_snapshots = False,
)
params = SimulationParams(t_end = 0.06)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# total explosion energy
E_explosion = 1.0
E_gas = E_explosion

# Ambient (background) physical conditions (adjust as needed)
rho_ambient  = 1.0         # typical ISM density
p_ambient    = 1e-4          # low gas pressure

# Pressures in code units
p_ambient = p_ambient

# --- Set Up the Explosion Injection Region ---
rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_ambient
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

# currently, we take 10 injection cells
r_explosion = 0.02

# Compute the injection volume (spherical volume in code units)
injection_volume = (4/3) * jnp.pi * r_explosion**3

# Adiabatic indices:
gamma_gas = params.gamma   # for the thermal gas
gamma_cr  = 4/3   # for cosmic rays

# The energy contained in a uniform pressure region is related by:
#   E = p * V / (gamma - 1)
# Hence, the effective explosion pressure in the injection region (in code units)
p_explosion_gas = E_gas * (gamma_gas - 1) / injection_volume

# Convert to code units
p_explosion_gas = p_explosion_gas

# --- Define the Radial Profiles ---
# Get the radial coordinate array (assumed already available)
r = helper_data.r

# Gas pressure: high within the explosion region, ambient elsewhere
p_gas = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p_gas
)
config = finalize_config(config, initial_state.shape)

# geometric_centers shape: (nx, ny, nz, 3)
geometric_centers = helper_data.geometric_centers
geometry_x = geometric_centers[..., 0]  # (nx, ny, nz)
geometry_y = geometric_centers[..., 1]  # (nx, ny, nz)
geometry_z = geometric_centers[..., 2]  # (nx, ny, nz)

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
shock_dir_x = result.shock_direction[0]   # (nx, ny, nz)
shock_dir_y = result.shock_direction[1]
shock_dir_z = result.shock_direction[2]

surface_mask = np.array(result.shock_surface_cells).astype(bool)


#%%
#-- DIAGNOSTICS
# print number of shock surface cells
print(f"num_shocks (surface cells): {result.num_shocks}")

# if exists shock surface cells, print min/max/mean Mach number at the surface
if result.num_shocks == 0:
    print("No shock surface cells found")
else:
    surface_mach = np.array(result.mach_numbers)[surface_mask]
    print(f"Mach at surface: min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")

## Compute measured shock radius
# geometric centers of environment, this is also explosion center
TARGET_CENTER = (
    float(config.box_size[0]) / 2,
    float(config.box_size[1]) / 2,
    float(config.box_size[2]) / 2,
)
center_x, center_y, center_z = TARGET_CENTER

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)
geometry_z_np = np.array(geometry_z)

# distances of surface cells from explosion center:
dx_surf = geometry_x_np[surface_mask] - center_x
dy_surf = geometry_y_np[surface_mask] - center_y
dz_surf = geometry_z_np[surface_mask] - center_z

# radial distances of surface cells from explosion center
r_surface = np.sqrt(dx_surf**2 + dy_surf**2 + dz_surf**2)

# compute mean, std, median of measured shock radius
if len(r_surface) > 0:
    r_measured_mean   = r_surface.mean()
    r_measured_std    = r_surface.std()
    r_measured_median = np.median(r_surface)
else:
    r_measured_mean = r_measured_std = r_measured_median = np.nan

## Expected result
xi_0 = 1.15167  # Sedov-Taylor similarity constant for gamma = 5/3
dx = config.box_size[0] / config.num_cells[0]
t_end = params.t_end
r_analytic = xi_0 * (E_explosion * t_end**2 / rho_ambient) ** (1.0 / 5.0)

print("\n=== Expected Sedov evolution ===")
print(f"Grid spacing:               {dx:.6f}")
print(f"Gamma used in sim:          {gamma_gas:.4f}  (xi_0={xi_0} assumes gamma= {gamma_gas})")
print(f"t_end:                      {t_end}")
print(f"Injection radius:           {r_explosion:.6f}")
print(f"Injection radius (cells):   {r_explosion/dx:.2f}")
print(f"Expected shock radius:      {r_analytic:.6f}")
print(f"Expected shock (cells):     {r_analytic/dx:.2f}")
print(f"Expansion factor:           {r_analytic/r_explosion:.2f}")
print(f"Distance to boundary:       {config.box_size[0]/2 - r_analytic:.6f}")

## Measured result
print("\n=== Measured Sedov evolution ===")
print(f"Measured mean shock radius:   {r_measured_mean:.4f}  (std={r_measured_std:.4f}, median={r_measured_median:.4f})")
if not np.isnan(r_measured_mean):
    rel_err = 100.0 * (r_measured_mean - r_analytic) / r_analytic
    print(f"Relative error:             {rel_err:+.2f} %")


#%%
# PLOTS
# header of plot, given measured vs analytic shock radius
if not np.isnan(r_measured_mean):
    rel_err = 100.0 * (r_measured_mean - r_analytic) / r_analytic
    proj_title = (
        f"3D Point Explosion (Sedov-Taylor)\n"+
        f"projections at t={params.t_end}\n"
        f"measured shock radius = {r_measured_mean:.4f}  |  analytic = {r_analytic:.4f}  "
        f"({rel_err:+.2f}%)"
    )
else:
    proj_title = f"3D Point Explosion (Sedov-Taylor) — mid-plane projections at t={params.t_end}"

# plot 3D shock finder projections (xy, xz, yz) with shock zones, shock surface, and shock direction arrows
fig, axes = plot_shock_projections_3d(
    rho_final, result,
    geometry_x, geometry_y, geometry_z,
    center=TARGET_CENTER,
    box_size=box_size,
    suptitle=proj_title,
)
plt.show()

# ----------------------------------------------------------------------
# 3D shock surface
# ----------------------------------------------------------------------
mach_surf = np.array(result.mach_numbers)[surface_mask]
fig3d, ax3d = plot_shock_surface_3d(
    geometry_x_np[surface_mask], geometry_y_np[surface_mask], geometry_z_np[surface_mask],
    shock_dir_x[surface_mask], shock_dir_y[surface_mask], shock_dir_z[surface_mask],
    mach_surf,
    center=TARGET_CENTER,
    box_size=box_size,
    title=f"3D Shock Surface — Sedov-Taylor at t={params.t_end}",
    mode="SCATTER"
)
plt.show()

fig3d_2, ax3d_2 = plot_shock_surface_3d(
    geometry_x_np[surface_mask], geometry_y_np[surface_mask], geometry_z_np[surface_mask],
    shock_dir_x[surface_mask], shock_dir_y[surface_mask], shock_dir_z[surface_mask],
    mach_surf,
    center=TARGET_CENTER,
    box_size=box_size,
    title=f"3D Shock Surface — Sedov-Taylor at t={params.t_end}",
    mode="SMOOTH"
)
plt.show()