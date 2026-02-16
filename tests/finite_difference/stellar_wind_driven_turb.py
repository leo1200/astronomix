import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

multi_gpu = True

if multi_gpu:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus = 2)
    # =======================
else:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus = 1)
    # =======================

# numerics

import jax

from matplotlib.colors import LogNorm
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

# plotting
import matplotlib.pyplot as plt

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state

from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams

from astronomix import get_registered_variables
from astronomix.option_classes import WindConfig

from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, PERIODIC_BOUNDARY, VARAXIS, XAXIS, YAXIS, ZAXIS, BoundarySettings, BoundarySettings1D

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c

from astronomix.option_classes.simulation_config import FORWARDS

from astronomix.option_classes.simulation_config import finalize_config


print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 512

# activate stellar wind
stellar_wind = False

# turbulence
turbulence = True

# otherwise B = 0.0
mhd = True

# wanted_rms = 50 * u.km / u.s

app_string = "driven_turb_wind"

print("Appended string for files:", app_string)


dt_max = 0.001

# setup simulation config
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    mhd = True,
    progress_bar = True,
    enforce_positivity = True,
    dimensionality = 3,
    box_size = box_size, 
    num_cells = num_cells,
    turbulent_forcing_config = TurbulentForcingConfig(
        turbulent_forcing = turbulence,
    ),
    differentiation_mode = FORWARDS,
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

registered_variables = get_registered_variables(config)

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
    dt_max = dt_max,
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

print(p_0.to(code_units.code_pressure).value)

if multi_gpu:

    # mesh with variable axis
    split = (1, 2, 1, 1)
    sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
    named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    # mesh no variable axis
    split = (2, 1, 1)
    sharding_mesh_no_var = jax.make_mesh(split, (XAXIS, YAXIS, ZAXIS))
    named_sharding_no_var = jax.NamedSharding(sharding_mesh_no_var, P(XAXIS, YAXIS, ZAXIS))


rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value
if multi_gpu:
    rho = jax.device_put(rho, named_sharding_no_var)

u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
if multi_gpu:
    u_x = jax.device_put(u_x, named_sharding_no_var)

u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
if multi_gpu:
    u_y = jax.device_put(u_y, named_sharding_no_var)

u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
if multi_gpu:
    u_z = jax.device_put(u_z, named_sharding_no_var)



p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

if mhd:
    B_0 = 13.5 * u.microgauss / c.mu0**0.5
    B_0 = B_0.to(code_units.code_magnetic_field).value
else:
    B_0 = 0.0

print("B_0", B_0)

# magnetic field in x direction
B_x = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * B_0
if multi_gpu:
    B_x = jax.device_put(B_x, named_sharding_no_var)

B_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
if multi_gpu:
    B_y = jax.device_put(B_y, named_sharding_no_var)

B_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
if multi_gpu:
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
    sharding = named_sharding if multi_gpu else None,
)

# set all single fields to none
u_x, u_y, u_z = None, None, None
rho, p = None, None
B_x, B_y, B_z = None, None, None
bxb, byb, bzb = None, None, None

if multi_gpu:
    initial_state = jax.device_put(initial_state, named_sharding)
    helper_data = get_helper_data(config, sharding = named_sharding)
else:
    helper_data = get_helper_data(config)
    named_sharding = None


config = finalize_config(config, initial_state.shape)

# first only turbulence
t_final = 24.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value
print(t_end)
params = params._replace(
    t_end = t_end,
)
final_state = time_integration(initial_state, config, params, registered_variables, sharding = named_sharding)

# save the intermediate state to disk
jnp.save("data/" + app_string + "2.npy", final_state)

# final_state = jnp.array(jnp.load("data/" + app_string + "2.npy"))

s = 45

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

plt.savefig("figures/interm_" + app_string + ".png", dpi = 1000)

# then stellar wind + turbulence
t_final = 0.5 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value
print(t_end)
config = config._replace(
    wind_config = WindConfig(
        stellar_wind = True,
        num_injection_cells = 18,
    ),
)
params = params._replace(
    t_end = t_end,
    minimum_density = 1e-3,
    minimum_pressure = 1e-3,
)
final_state = time_integration(final_state, config, params, registered_variables, sharding = named_sharding)

# print min and max density and pressure
print("Final min density:", jnp.min(final_state[registered_variables.density_index]))
print("Final max density:", jnp.max(final_state[registered_variables.density_index]))
print("Final min pressure:", jnp.min(final_state[registered_variables.pressure_index]))
print("Final max pressure:", jnp.max(final_state[registered_variables.pressure_index]))

# print mean squared velocity
velocity_squared = (final_state[registered_variables.velocity_index.x]**2 + 
                      final_state[registered_variables.velocity_index.y]**2 + 
                      final_state[registered_variables.velocity_index.z]**2)
mean_squared_velocity = jnp.mean(velocity_squared)
rms_velocity = jnp.sqrt(mean_squared_velocity)
print("Final RMS velocity:", (rms_velocity * code_units.code_velocity).to(u.km / u.s).value, "km/s")

# save the final state to disk
jnp.save("data/driven_turb_wind" + app_string + "2.npy", final_state)

s = 45

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

plt.savefig("figures/" + app_string + ".png", dpi = 1000)