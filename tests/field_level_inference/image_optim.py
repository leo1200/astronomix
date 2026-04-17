# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp

import optax

import os

import imageio.v3 as iio
import re
import numpy as np

import pyvista as pv

# jax.config.update("jax_enable_x64", True)

# setup
from astronomix import SimulationConfig
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, FINITE_VOLUME, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D, BACKWARDS
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
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# from vape4d import render

"""## Loading the target image"""

resolution = 64
only_animate = True

def load_image(path, height, width, flat_profile = False):
    img = jnp.array(Image.open(path).convert("L"))
    img = 1.0 - img / 255.0 # black → 1, white → 0
    h, w = img.shape

    pad_h = (-h) % height
    pad_w = (-w) % width
    pad = ((pad_h//2, pad_h - pad_h//2),
           (pad_w//2, pad_w - pad_w//2))

    img = jnp.pad(img, pad)
    hp, wp = img.shape

    result = img.reshape(height, hp//height, width, wp//width).mean(axis=(1, 3))

    if flat_profile:
        result = jnp.where(result > 0.01, 1.01, 1.0)
    else:
        # still crunch the range between 1.01 and 1.0
        # but with all the intermediates
        result = 1.0 + (result - 0.01) / (1.0 - 0.01) * 0.01

    return result

# image path
path = "logo.png"

original_rgb = Image.open(path).convert("RGB")
target = load_image(path, resolution, resolution)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(original_rgb)
ax[0].set_title("Image")
ax[0].axis("off")

ax[1].imshow(target, cmap="viridis")
ax[1].set_title("Optimization target")
ax[1].axis("off")

plt.savefig("target.png", dpi = 600)

"""## Field-level inference

### Generation of an initial state
"""

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = resolution

# turbulence
turbulence = True

# otherwise B = 0.0
mhd = True
solver_mode = FINITE_DIFFERENCE

# baseline simulation config
config = SimulationConfig(
    solver_mode = solver_mode,
    mhd = mhd,
    progress_bar = True,
    enforce_positivity = True,
    donate_state = False, # save storage
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
)

# get the variable registry
registered_variables = get_registered_variables(config)

# unit setup
code_length = 3 * u.parsec
code_mass = 100 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.8

# initial state
n_h = 2 # cm^-3
rho_0 = n_h * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

dE_dt_turb = 4.3e34 * u.erg / u.s

params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = 0.1,
    gamma = gamma,
    minimum_density=(1e-2 * rho_0).to(code_units.code_density).value,
    minimum_pressure=(1e-2 * p_0).to(code_units.code_pressure).value,
    turbulent_forcing_params = TurbulentForcingParams(
        energy_injection_rate = dE_dt_turb.to(code_units.code_energy / code_units.code_time).value,
    ),
)

rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

if mhd:
    B_0 = 13.5 * u.microgauss / c.mu0**0.5
    B_0 = B_0.to(code_units.code_magnetic_field).value
else:
    B_0 = 0.0

B_x = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * B_0
B_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
B_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
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

config = finalize_config(config, initial_state.shape)

t_final = 24 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# set the final time
params = params._replace(
    t_end = t_end,
)

# running a small turbulent simulation
turb_state = time_integration(initial_state, config, params, registered_variables)

# calculate the rms velocity and turbulent crossing time
rms_velocity = jnp.sqrt(jnp.mean(turb_state[registered_variables.velocity_index.x]**2 + turb_state[registered_variables.velocity_index.y]**2 + turb_state[registered_variables.velocity_index.z]**2))
rms_velocity = (rms_velocity * code_units.code_velocity).to(u.km/u.s)
crossing_time = box_size * code_length / rms_velocity
print(f"RMS velocity: {rms_velocity}")
print(f"Turbulent crossing time: {(crossing_time).to(u.yr)}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# equal aspect ratio
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')

z_level = num_cells // 2

ax1.imshow(turb_state[registered_variables.density_index, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax1.set_title("density")

ax2.imshow(jnp.sqrt(turb_state[registered_variables.velocity_index.x, :, :, z_level]**2 + turb_state[registered_variables.velocity_index.y, :, :, z_level]**2 + turb_state[registered_variables.velocity_index.z, :, :, z_level]**2).T, origin = "lower", extent = [0, 1, 0, 1])
ax2.set_title("velocity magnitude")

ax3.imshow(turb_state[registered_variables.pressure_index, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax3.set_title("pressure")

plt.savefig("initial_turbulent_state.png", dpi = 600)

"""### Adapting the target to the total mass in the simulation"""

total_mass = jnp.sum(turb_state[registered_variables.density_index])
target = target / jnp.sum(target) * total_mass

"""### Adapt the simulation, turn off the turbulent driving"""

config = config._replace(
    differentiation_mode = BACKWARDS,
    num_checkpoints = 10,
    turbulent_forcing_config = TurbulentForcingConfig(
        turbulent_forcing = False,
    ),
)

t_final = crossing_time / 2
t_end = t_final.to(code_units.code_time).value

params = params._replace(
    t_end = t_end,
)

"""### Define the loss"""
def loss(velocity):

  # set the current velocity
  initial_state = turb_state.at[registered_variables.velocity_index.x].set(velocity[0])
  initial_state = initial_state.at[registered_variables.velocity_index.y].set(velocity[1])
  initial_state = initial_state.at[registered_variables.velocity_index.z].set(velocity[2])

  # integrate in time
  final_state = time_integration(initial_state, config, params, registered_variables)

  # get the final density
  density = final_state[registered_variables.density_index]

  # sum density along z
  projected_density = jnp.sum(density, axis = (2,))
  density_loss = jnp.mean(jnp.square(projected_density - target))
  return density_loss

initial_loss = loss(turb_state[registered_variables.density_index])
print(f"Initial loss: {initial_loss}")

"""### Optimization"""
current_velocity = jnp.array([
  turb_state[registered_variables.velocity_index.x],
  turb_state[registered_variables.velocity_index.y],
  turb_state[registered_variables.velocity_index.z],
])

if not only_animate:
    # optimizer setup
    learning_rate = 1e-2
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(current_velocity)

    # gradient
    grad_loss = jax.grad(loss)

    best_loss = initial_loss
    best_velocity = current_velocity

    # optimization
    num_steps = 150

    for step in range(num_steps):
        grads = grad_loss(current_velocity)
        updates, opt_state = optimizer.update(grads, opt_state)
        current_velocity = optax.apply_updates(current_velocity, updates)

        current_loss = loss(current_velocity)
        print(f"Step {step}: loss = {current_loss}")

        if current_loss < best_loss:
            best_loss = current_loss
            best_velocity = current_velocity

    best_state = turb_state.at[registered_variables.velocity_index.x].set(best_velocity[0])
    best_state = best_state.at[registered_variables.velocity_index.y].set(best_velocity[1])
    best_state = best_state.at[registered_variables.velocity_index.z].set(best_velocity[2])

    print("Optimization finished.")
    final_loss = loss(best_velocity)
    print(f"Final loss: {final_loss}")

if only_animate:
    best_state = jnp.load("best_state.npy")
    best_velocity = jnp.array([
        best_state[registered_variables.velocity_index.x],
        best_state[registered_variables.velocity_index.y],
        best_state[registered_variables.velocity_index.z],
    ])
    rms_velocity = jnp.sqrt(jnp.mean(best_velocity[0]**2 + best_velocity[1]**2 + best_velocity[2]**2))
    cr_ratio = t_end / (box_size / rms_velocity)
    print(f"Crossing time ratio (final time / crossing time): {cr_ratio}")
    rms_velocity = (rms_velocity * code_units.code_velocity).to(u.km/u.s)
    crossing_time = box_size * code_length / rms_velocity

    print(f"Turbulent crossing time (optimized): {(crossing_time).to(u.yr)}")
else:
    # save the best state to disc
    jnp.save("best_state.npy", best_state)


final_state = time_integration(best_state, config, params, registered_variables)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

ax[0, 0].imshow(original_rgb)
ax[0, 0].set_title("Image")
ax[0, 0].axis("off")

ax[0, 1].imshow(target, cmap="viridis")
ax[0, 1].set_title("Optimization target")
ax[0, 1].axis("off")

ax[1, 0].imshow(jnp.sum(best_state[registered_variables.density_index], axis = (2,)))
ax[1, 0].set_title("Initial state (optimized)")
ax[1, 0].axis("off")

ax[1, 1].imshow(jnp.sum(final_state[registered_variables.density_index], axis = (2,)))
ax[1, 1].set_title(f"Final state (loss = {loss(best_velocity):3e})")
ax[1, 1].axis("off")

plt.savefig("overview.png", dpi = 600)
"""### Animation"""

config = config._replace(
    activate_snapshot_callback = True,
    num_snapshots = 800
)

# create folder for frames
os.makedirs('frames', exist_ok=True)

# empty the folder
for filename in os.listdir('frames'):
  os.remove(os.path.join('frames', filename))

def save_frame(
  time,
  state,
  registered_variables
):

    plot_3D = True

    def plot_slices(density, time):

        print(f"plotting slice at t = {time}")

        if plot_3D:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

            # 3D volumetric rendering in pyvista, displayed in ax1.
            density_3d = np.asarray(density)
            grid = pv.ImageData()
            grid.dimensions = np.array(density_3d.shape) + 1
            grid.cell_data["density"] = density_3d.ravel(order="F")

            plotter = pv.Plotter(off_screen=True, window_size=(700, 700))
            plotter.add_volume(
                    grid,
                    scalars="density",
                    cmap="viridis",
                    clim=(0.01, 0.02),
                    opacity="sigmoid",
                    blending="composite",
                    shade=True,
            )
            plotter.camera_position = "iso"
            volume_img = plotter.screenshot(return_img=True)
            plotter.close()

            ax1.imshow(volume_img)
            ax1.set_title("volumetric rendering")
            ax1.set_xticks([])
            ax1.set_yticks([])

            # projection
            ax2.set_aspect('equal', 'box')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.imshow(np.sum(density_3d, axis = (2,)))
            ax2.set_title("density projection")
        else:
            fig, ax2 = plt.subplots(1, 1, figsize=(4, 4))

            # projection
            ax2.set_aspect('equal', 'box')
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.imshow(density)
            ax2.set_title("density projection")

        fig.savefig(
            f"frames/frame{time}.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0
        )

        plt.close(fig)

    if plot_3D:
        jax.debug.callback(
            plot_slices,
            state[registered_variables.density_index], time,
        )
    else:
        jax.debug.callback(
            plot_slices,
            jnp.sum(state[registered_variables.density_index], axis = (2,)), time,
        )

final_state = time_integration(best_state, config, params, registered_variables, save_frame)

def get_sorted_frames(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    # Extract numerical timestamp from filename (e.g., frame0.123.png -> 0.123)
    # and sort by it
    files.sort(key=lambda f: float(re.search(r'frame([0-9.]+)\.png', f).group(1)))
    return [os.path.join(directory, f) for f in files]

all_frames = get_sorted_frames('frames')

# Read all images into a list
images = [iio.imread(frame) for frame in all_frames]

# Create GIF
gif_path = 'simulation_animation.gif'
iio.imwrite(gif_path, images, duration=int(1000/30))

print(f"GIF created at: {gif_path}")