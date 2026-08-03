#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer
from astronomix.option_classes.simulation_config import CARTESIAN, HLLC, HYBRID_HLLC, MINMOD

from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_surface_3d
from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_projections_3d
from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_surface_3d_interactive

#%%
# INITIAL CONFIGURATION
num_cells = 64
box_size = 1.0

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=3,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
)

params = SimulationParams(t_end=0.1)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

geometric_centers = helper_data.geometric_centers

geometry_x = geometric_centers[...,0]
geometry_y = geometric_centers[...,1]
geometry_z = geometric_centers[...,2]

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)
geometry_z_np = np.array(geometry_z)

# ---------------------------------------------------------------------------
# Two shock planes
# ---------------------------------------------------------------------------

TARGET_CENTER = (0.5, 0.5, 0.5)

# Plane 1 normal direction
theta1 = jnp.deg2rad(45.0)   # azimuth
phi1   = jnp.deg2rad(45.0)   # inclination

n1x = jnp.cos(theta1)*jnp.sin(phi1)
n1y = jnp.sin(theta1)*jnp.sin(phi1)
n1z = jnp.cos(phi1)


# Plane 2 normal direction
theta2 = jnp.deg2rad(-30.0)
phi2   = jnp.deg2rad(60.0)

n2x = jnp.cos(theta2)*jnp.sin(phi2)
n2y = jnp.sin(theta2)*jnp.sin(phi2)
n2z = jnp.cos(phi2)


# Signed distances

dist1 = (
    (geometry_x-TARGET_CENTER[0])*n1x
    +(geometry_y-TARGET_CENTER[1])*n1y
    +(geometry_z-TARGET_CENTER[2])*n1z
)

dist2 = (
    (geometry_x-TARGET_CENTER[0])*n2x
    +(geometry_y-TARGET_CENTER[1])*n2y
    +(geometry_z-TARGET_CENTER[2])*n2z
)


# Half-space selection

high1 = dist1 < 0
high2 = dist2 < 0


# XOR creates alternating wedges
in_high = jnp.logical_xor(high1, high2)


# ---------------------------------------------------------------------------
# Thermodynamic states
# ---------------------------------------------------------------------------

p = jnp.where(in_high, 1.0, 0.1)
rho = jnp.where(in_high, 1.0, 0.125)


u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)
u_z = jnp.zeros_like(geometry_z)


initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    velocity_z=u_z,
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

shock_dir_x = result.shock_direction[0]   # (nx, ny, nz)
shock_dir_y = result.shock_direction[1]
shock_dir_z = result.shock_direction[2]

surface_mask = np.array(result.shock_surface_cells).astype(bool)

#%%
# PLOTS — three orthogonal mid-plane projections + 3D surface

fig, axes = plot_shock_projections_3d(
    rho_final, result,
    geometry_x, geometry_y, geometry_z,
    center=TARGET_CENTER,
    box_size=box_size
)
plt.show()

# ----------------------------------------------------------------------
# 3D shock surface (smoothed)
# ----------------------------------------------------------------------
mach_surf = np.array(result.mach_numbers)[surface_mask]
fig3d, ax3d = plot_shock_surface_3d(
    geometry_x_np[surface_mask], geometry_y_np[surface_mask], geometry_z_np[surface_mask],
    shock_dir_x[surface_mask], shock_dir_y[surface_mask], shock_dir_z[surface_mask],
    mach_surf,
    center=TARGET_CENTER,
    box_size=box_size,
    title=f"3D Shock Surface — Sedov-Taylor at t={params.t_end}",
)
plt.show()

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

#%%
# INTERACTIVE 3D PLOT
fig = plot_shock_surface_3d_interactive(
    geometry_x[surface_mask], geometry_y[surface_mask], geometry_z[surface_mask],
    shock_dir_x[surface_mask], shock_dir_y[surface_mask], shock_dir_z[surface_mask],
    mach_surf,
    center=(0.5,0.5,0.5),
    box_size=1.0,
    title="3D Sedov Shock Surface",
)

fig.show()