import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

multi_gpu = True

if multi_gpu:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus = 4)
    # =======================
else:
    # ==== GPU selection ====
    from autocvd import autocvd
    autocvd(num_gpus = 1)
    # =======================

# numerics
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from astronomix.data_classes.simulation_helper_data import HelperData

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state

from astronomix import get_registered_variables
from astronomix.option_classes import WindConfig

from astronomix.option_classes.simulation_config import BACKWARDS, DONOR_ACCOUNTING, HLL, HLLC, MINMOD, OSHER, PERIODIC_BOUNDARY, SIMPLE_SOURCE_TERM, VARAXIS, XAXIS, YAXIS, ZAXIS, BoundarySettings, BoundarySettings1D, SnapshotSettings

from astronomix._physics_modules._cooling._cooling_tables import schure_cooling
from astronomix._physics_modules._cooling.cooling_options import IMPLICIT_COOLING, PIECEWISE_POWER_LAW, CoolingConfig, CoolingCurveConfig, CoolingParams

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from astronomix._physics_modules._stellar_wind.weaver import Weaver

# turbulence
from astronomix.initial_condition_generation.turbulent_ic_generator import create_turb_field

from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, FORWARDS, HYBRID_HLLC

from astronomix.option_classes.simulation_config import finalize_config


print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
box_size = 1.0
num_cells = 64

# activate stellar wind
stellar_wind = True

# turbulence
turbulence = True
wanted_rms = 50 * u.km / u.s

# cooling
cooling = True

# mhd
mhd = True

app_string = ""
if turbulence:
    app_string += "_turb"
if cooling:
    app_string += "_cooling"
if mhd:
    app_string += "_mhd"

print("Appended string for files:", app_string)

fixed_timestep = False
scale_time = False
dt_max = 0.1
num_timesteps = 2000

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    memory_analysis = True,
    print_elapsed_time = True,
    riemann_solver = HLL,
    mhd = mhd,
    limiter = DOUBLE_MINMOD,
    first_order_fallback = False,
    progress_bar = True,
    dimensionality = 3,
    num_ghost_cells = 2,
    box_size = box_size, 
    num_cells = num_cells,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = 12,
        trace_wind_density = False,
    ),
    fixed_timestep = fixed_timestep,
    differentiation_mode = FORWARDS,
    num_timesteps = num_timesteps,
    return_snapshots = True,
    snapshot_settings = SnapshotSettings(
        return_states = False,
        return_final_state = True
    ),
    num_snapshots = 5,
    cooling_config = CoolingConfig(
        cooling = cooling,
        cooling_method = IMPLICIT_COOLING,
        cooling_curve_config = CoolingCurveConfig(
            cooling_curve_type = PIECEWISE_POWER_LAW
        )
    ),
)

registered_variables = get_registered_variables(config)

code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# time domain
C_CFL = 0.4

t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

if scale_time:
    t_end = dt_max * num_timesteps
print(t_end)


# wind parameters
M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

wind_params = WindParams(
    wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
)

# cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
reference_temperature = (1e8 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
# without a floor temperature, the simulations crash
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
cooling_curve_params = schure_cooling(code_units)

gravitational_constant = (c.G).to(code_units.code_length**3 / (code_units.code_mass * code_units.code_time**2)).value


params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = dt_max,
    gamma = gamma,
    t_end = t_end,
    wind_params = wind_params,
        cooling_params = CoolingParams(
        hydrogen_mass_fraction = hydrogen_mass_fraction,
        metal_mass_fraction = metal_mass_fraction,
        floor_temperature = floor_temperature,
        cooling_curve_params = cooling_curve_params
    ),
    gravitational_constant = gravitational_constant
)

# homogeneous initial state
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

print(p_0.to(code_units.code_pressure).value)

turbulence_slope = -2.0
kmin = 2.0
kmax = int(0.4 * num_cells / 2)

if multi_gpu:

    # mesh with variable axis
    split = (1, 2, 2, 1)
    sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
    named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    # mesh no variable axis
    split = (2, 2, 1)
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

x = jnp.linspace(0, config.box_size, config.num_cells)
y = jnp.linspace(0, config.box_size, config.num_cells)
z = jnp.linspace(0, config.box_size, config.num_cells)


if turbulence:
    key = jax.random.PRNGKey(42)
    key, sk1, sk2, sk3 = jax.random.split(key, 4)
    ux = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk1, sharding=named_sharding_no_var if multi_gpu else None)
    uy = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk2, sharding=named_sharding_no_var if multi_gpu else None)
    uz = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk3, sharding=named_sharding_no_var if multi_gpu else None)

    a = num_cells // 2 - 10
    b = num_cells // 2 + 10

    u_x = ux# .at[a:b, a:b, a:b].set(ux[a:b, a:b, a:b])
    u_y = uy# .at[a:b, a:b, a:b].set(uy[a:b, a:b, a:b])
    u_z = uz# .at[a:b, a:b, a:b].set(uz[a:b, a:b, a:b])

    rms_vel = jnp.sqrt(jnp.mean(ux**2 + uy**2 + uz**2))

    u_x = u_x / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_y = u_y / rms_vel * wanted_rms.to(code_units.code_velocity).value
    u_z = u_z / rms_vel * wanted_rms.to(code_units.code_velocity).value

p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

if mhd:
    B_0 = 13.5 * u.microgauss / c.mu0**0.5
    B_0 = B_0.to(code_units.code_magnetic_field).value

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
else:
    B_x = None
    B_y = None
    B_z = None

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
    sharding = named_sharding if multi_gpu else None,
)

if multi_gpu:
    initial_state = jax.device_put(initial_state, named_sharding)
    helper_data = get_helper_data(config, sharding = named_sharding)
else:
    helper_data = get_helper_data(config)
    named_sharding = None


config = finalize_config(config, initial_state.shape)


result = time_integration(initial_state, config, params, registered_variables, sharding = named_sharding)

final_state = result.final_state

from matplotlib.colors import LogNorm

s = 45

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# equal aspect ratio
ax1.set_aspect('equal', 'box')
ax2.set_aspect('equal', 'box')
ax3.set_aspect('equal', 'box')

z_level = num_cells // 2

xm, ym = jnp.meshgrid(x, y)

ax1.imshow(final_state[0, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax1.set_title("Density")

ax2.imshow(jnp.sqrt(final_state[1, :, :, z_level]**2 + final_state[2, :, :, z_level]**2).T, origin = "lower", extent = [0, 1, 0, 1])
ax2.set_title("Velocity")

ax3.imshow(final_state[4, :, :, z_level].T, origin = "lower", extent = [0, 1, 0, 1], norm = LogNorm())
ax3.set_title("Pressure")

plt.savefig("figures/slices" + app_string + ".png", dpi = 1000)

# def plot_weaver_comparison(axs, final_state, params, helper_data, code_units, rho_0, p_0):
#     rho = final_state[registered_variables.density_index].flatten()
#     vel = jnp.sqrt(final_state[registered_variables.velocity_index.x].flatten()**2 + final_state[registered_variables.velocity_index.y].flatten()**2 + final_state[registered_variables.velocity_index.z].flatten()**2)
#     p = final_state[registered_variables.pressure_index].flatten()

#     rho = rho * code_units.code_density
#     vel = vel * code_units.code_velocity
#     p = p * code_units.code_pressure

#     r = helper_data.r.flatten() * code_units.code_length

#     # get weaver solution
#     weaver = Weaver(
#         params.wind_params.wind_final_velocity * code_units.code_velocity,
#         params.wind_params.wind_mass_loss_rate * code_units.code_mass / code_units.code_time,
#         rho_0,
#         p_0
#     )
    
#     current_time = params.t_end * code_units.code_time
    
#     # density
#     r_density_weaver, density_weaver = weaver.get_density_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
#     r_density_weaver = r_density_weaver.to(u.parsec)
#     density_weaver = (density_weaver / m_p).to(u.cm**-3)

#     # velocity
#     r_velocity_weaver, velocity_weaver = weaver.get_velocity_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
#     r_velocity_weaver = r_velocity_weaver.to(u.parsec)
#     velocity_weaver = velocity_weaver.to(u.km / u.s)

#     # pressure
#     r_pressure_weaver, pressure_weaver = weaver.get_pressure_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
#     r_pressure_weaver = r_pressure_weaver.to(u.parsec)
#     pressure_weaver = (pressure_weaver / c.k_B).to(u.cm**-3 * u.K)

#     axs[0].set_yscale("log")
#     axs[0].scatter(r.to(u.parsec), (rho / m_p).to(u.cm**-3), label="astronomix", s = 1)

#     axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution")

#     axs[0].set_title("density")
#     axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
#     axs[0].set_xlim(0, 3)

#     axs[0].legend(loc="upper right")

#     axs[0].set_xlabel("r in pc")

#     axs[1].set_yscale("log")
#     axs[1].scatter(r.to(u.parsec), (p / c.k_B).to(u.K / u.cm**3), label="astronomix", s = 1)
#     axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution")

#     axs[1].set_title("pressure")
#     axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
#     axs[1].set_xlim(0, 3)

#     axs[1].legend(loc="upper right")

#     axs[1].set_xlabel("r in pc")


#     axs[2].set_yscale("log")
#     axs[2].scatter(r.to(u.parsec), vel.to(u.km / u.s), label="astronomix", s = 1)
#     axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution")
#     axs[2].set_title("velocity")
#     # ylim 1 to 1e4 km/s
#     axs[2].set_ylim(1, 1e4)
#     axs[2].set_xlim(0, 3)
#     axs[2].set_ylabel("v in km/s")
#     # xlabel
#     # show legend upper left
#     axs[2].legend(loc="upper right")

#     axs[2].set_xlabel("r in pc")

# fig, axs = plt.subplots(1, 3, figsize=(15, 5))
# plot_weaver_comparison(axs, final_state, params, helper_data, code_units, rho_0, p_0)
# plt.savefig("figures/profiles" + app_string + ".png")