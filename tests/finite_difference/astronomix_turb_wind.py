multi_gpu = True

if multi_gpu:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus=4)
    # =======================
else:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus=1)
    # =======================

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

# setup
from astronomix import SimulationConfig
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, PERIODIC_BOUNDARY, VARAXIS, XAXIS, YAXIS, ZAXIS, BoundarySettings, BoundarySettings1D
)
from astronomix import SimulationParams
from astronomix import get_registered_variables
from astronomix import get_helper_data
from astronomix import construct_primitive_state

# main time integration function
from astronomix import time_integration

# stellar wind
from astronomix import WindParams
from astronomix.option_classes import WindConfig

# turbulent forcing
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astronomix.option_classes.simulation_config import finalize_config

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if multi_gpu:

    # mesh with variable axis
    split = (1, 2, 2, 1)
    sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
    named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    # mesh no variable axis
    split = (2, 2, 1)
    sharding_mesh_no_var = jax.make_mesh(split, (XAXIS, YAXIS, ZAXIS))
    named_sharding_no_var = jax.NamedSharding(sharding_mesh_no_var, P(XAXIS, YAXIS, ZAXIS))


"""## Simulation setup"""

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 600

# activate stellar wind
stellar_wind = False

# turbulence
turbulence = True

# otherwise B = 0.0
mhd = True

app_string = "driven_turb_wind"

# baseline simulation config
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    mhd = True,
    progress_bar = True,
    enforce_positivity = True,
    donate_state = True, # save storage
    dimensionality = 3,
    box_size = box_size,
    num_cells = num_cells,
    turbulent_forcing_config = TurbulentForcingConfig(
        turbulent_forcing = turbulence,
    ),
    boundary_settings =  BoundarySettings(
        BoundarySettings1D(
            left_boundary = PERIODIC_BOUNDARY,
            right_boundary = PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = PERIODIC_BOUNDARY,
            right_boundary = PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary = PERIODIC_BOUNDARY,
            right_boundary = PERIODIC_BOUNDARY
        )
    ),
    # animation
    activate_snapshot_callback = True,
    num_snapshots = 100
)

# get the variable registry
registered_variables = get_registered_variables(config)

# unit setup
code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.8

# wind parameters
M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

wind_params = WindParams(
    wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
)

params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = 0.1,
    gamma = gamma,
    minimum_density=1e-3,
    minimum_pressure=1e-3,
    wind_params = wind_params,
    turbulent_forcing_params = TurbulentForcingParams(
        energy_injection_rate = 0.2
    ),
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

if multi_gpu:
    rho = jax.device_put(rho, named_sharding_no_var)
    u_x = jax.device_put(u_x, named_sharding_no_var)
    u_y = jax.device_put(u_y, named_sharding_no_var)
    u_z = jax.device_put(u_z, named_sharding_no_var)
    p = jax.device_put(p, named_sharding_no_var)

if mhd:
    B_0 = 13.5 * u.microgauss / c.mu0**0.5
    B_0 = B_0.to(code_units.code_magnetic_field).value
else:
    B_0 = 0.0

B_x = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * B_0
B_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
B_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

if multi_gpu:
    B_x = jax.device_put(B_x, named_sharding_no_var)
    B_y = jax.device_put(B_y, named_sharding_no_var)
    B_z = jax.device_put(B_z, named_sharding_no_var)

bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

# construct primitive state
initial_state = construct_primitive_state(
    config = config,
    registered_variables=registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p,
    magnetic_field_x = B_x,
    magnetic_field_y = B_y,
    magnetic_field_z = B_z,
    interface_magnetic_field_x = bxb,
    interface_magnetic_field_y = byb,
    interface_magnetic_field_z = bzb,
)

# set all individual fields to None
rho = None
u_x = None
u_y = None
u_z = None
p = None
B_x = None
B_y = None
B_z = None
bxb = None
byb = None
bzb = None

if multi_gpu:
    initial_state = jax.device_put(initial_state, named_sharding)

config = finalize_config(config, initial_state.shape)

"""## Prepare saving frames for GIF"""

import os

# create folder for frames
os.makedirs('turb_frames', exist_ok=True)
os.makedirs('wind_frames', exist_ok=True)

# empty the folders
for filename in os.listdir('turb_frames'):
  os.remove(os.path.join('turb_frames', filename))
for filename in os.listdir('wind_frames'):
  os.remove(os.path.join('wind_frames', filename))

def save_frame(
  time,
  state,
  registered_variables,
  directory
):

    def plot_slices(density, pressure, time):

        # print(f"plotting slice at t = {time}")

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 4))

        # equal aspect ratio
        ax1.set_aspect('equal', 'box')
        # ax2.set_aspect('equal', 'box')
        ax3.set_aspect('equal', 'box')

        # remove x and y ticks
        for ax in [ax1, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])

        z_level = num_cells // 2

        im1 = ax1.imshow(density.T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
        ax1.set_title("density")

        # im2 = ax2.imshow(velocity.T, origin = "lower", extent = [0, 1, 0, 1])
        # ax2.set_title("velocity magnitude")

        im3 = ax3.imshow(pressure.T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
        ax3.set_title("pressure")

        fig.savefig(
            f"{directory}/frame{time}.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

    num_vars, nx, ny, nz = state.shape
    z_level = num_cells // 2

    density = state[registered_variables.density_index, :, :, z_level]
    # velocity = jnp.sqrt(state[registered_variables.velocity_index.x, :, :, z_level]**2 + state[registered_variables.velocity_index.y, :, :, z_level]**2 + state[registered_variables.velocity_index.z, :, :, z_level]**2)
    pressure = state[registered_variables.pressure_index, :, :, z_level]

    jax.debug.callback(
        plot_slices,
        density, pressure, time,
    )

save_turb_frame = lambda time, state, registered_variables: save_frame(time, state, registered_variables, 'turb_frames')
save_wind_frame = lambda time, state, registered_variables: save_frame(time, state, registered_variables, 'wind_frames')

"""## Turbulence only run

### Parameter adaptation
"""

t_final = 12.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# set the final time
params = params._replace(
    t_end = t_end,
)

"""### Running the simulation"""

final_state = time_integration(initial_state, config, params, registered_variables, save_turb_frame, sharding = named_sharding if multi_gpu else None)

"""### Plotting"""

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# equal aspect ratio
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')

z_level = num_cells // 2

ax1.imshow(final_state[registered_variables.density_index, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax1.set_title("density")

ax2.imshow(jnp.sqrt(final_state[registered_variables.velocity_index.x, :, :, z_level]**2 + final_state[registered_variables.velocity_index.y, :, :, z_level]**2 + final_state[registered_variables.velocity_index.z, :, :, z_level]**2).T, origin = "lower", extent = [0, 1, 0, 1])
ax2.set_title("velocity magnitude")

ax3.imshow(final_state[registered_variables.pressure_index, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax3.set_title("pressure")

"""## Let us introduce stellar wind

### Parameter adaptation
"""

# then stellar wind + turbulence
t_final = 0.5 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value
print(t_end)
config = config._replace(
    wind_config = WindConfig(
        stellar_wind = True,
        num_injection_cells = 8,
    ),
)
params = params._replace(
    t_end = t_end,
    minimum_density = 1e-3,
    minimum_pressure = 1e-3,
)

"""### Running the simulation"""

final_state = time_integration(final_state, config, params, registered_variables, save_wind_frame, sharding = named_sharding if multi_gpu else None)

"""### Plotting"""

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# equal aspect ratio
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')

z_level = num_cells // 2

ax1.imshow(final_state[registered_variables.density_index, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax1.set_title("density")

ax2.imshow(jnp.sqrt(final_state[registered_variables.velocity_index.x, :, :, z_level]**2 + final_state[registered_variables.velocity_index.y, :, :, z_level]**2 + final_state[registered_variables.velocity_index.z, :, :, z_level]**2).T, origin = "lower", extent = [0, 1, 0, 1])
ax2.set_title("velocity magnitude")

ax3.imshow(final_state[registered_variables.pressure_index, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax3.set_title("pressure")

"""## Make animations"""

import imageio.v3 as iio
import os
import re

def get_sorted_frames(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    # Extract numerical timestamp from filename (e.g., frame0.123.png -> 0.123)
    # and sort by it
    files.sort(key=lambda f: float(re.search(r'frame([0-9.]+)\.png', f).group(1)))
    return [os.path.join(directory, f) for f in files]

turb_frames = get_sorted_frames('turb_frames')
wind_frames = get_sorted_frames('wind_frames')

all_frames = turb_frames + wind_frames

# Read all images into a list
images = [iio.imread(frame) for frame in all_frames]

# Create GIF
gif_path = 'simulation_animation.gif'
iio.imwrite(gif_path, images, duration=int(1000/30), loop=0)

print(f"GIF created at: {gif_path}")