"""
We consider stellar wind with and without cooling.
The aim is to compare the 3D finite difference solution
to the Weaver solution for the ideal case without cooling
and to the highly resolved radial simulations for the case with
cooling.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# ruff: noqa: E402
# =======================

# numerics
import jax

# debug nans
# jax.config.update("jax_debug_nans", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

import equinox as eqx

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state

from astronomix import get_registered_variables
from astronomix.option_classes import WindConfig

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY, SPHERICAL,
    BoundarySettings, BoundarySettings1D,
)
from astronomix.option_classes.simulation_config import finalize_config

from astronomix._modules._cooling._cooling_tables import schure_cooling
from astronomix._modules._cooling.cooling_options import (
    PIECEWISE_POWER_LAW, CoolingConfig, CoolingCurveConfig, CoolingParams
)
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from astronomix._modules._stellar_wind.weaver import Weaver
from astronomix._modules._stellar_wind.stellar_wind_options import EI

# ==== Common physics setup ====

# setup of the units
code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# initial conditions
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

# only applies to the 3D setting
mhd = True
B_0 = 0.1 * u.microgauss / c.mu0**0.5 
# small magnetic fields stabilize the 3D MHD simulations
B_0 = B_0.to(code_units.code_magnetic_field).value

# time domain
C_CFL = 0.9

# time setup
t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

# wind setup
stellar_wind = True
num_injection_cells = 14
M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star
wind_params = WindParams(
    wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
)

# cooling setup
cooling = True
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
reference_temperature = (1e8 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
cooling_curve_params = schure_cooling(code_units)

# ==== 3D simulation ====

# spatial discretization
box_size = 1.0
num_cells = 256

# setup simulation config
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    mhd = True,
    progress_bar = True,
    dimensionality = 3,
    box_size = box_size, 
    num_cells = num_cells,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = num_injection_cells,
        trace_wind_density = False,
    ),
    cooling_config = CoolingConfig(
        cooling = cooling,
        cooling_curve_config = CoolingCurveConfig(
            cooling_curve_type = PIECEWISE_POWER_LAW
        )
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

registered_variables = get_registered_variables(config)

params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = 0.001,
    minimum_density=1e-8,
    minimum_pressure=1e-8,
    t_end = t_end,
    wind_params = wind_params,
        cooling_params = CoolingParams(
        hydrogen_mass_fraction = hydrogen_mass_fraction,
        metal_mass_fraction = metal_mass_fraction,
        floor_temperature = floor_temperature,
        cooling_curve_params = cooling_curve_params
    )
)

# homogeneous initial state setup
rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value
u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

# magnetic field setup
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

helper_data_3D = get_helper_data(config)
r_3D = helper_data_3D.r.flatten() * code_units.code_length
config = finalize_config(config, initial_state.shape)

# running the simulation
final_state_3D = time_integration(initial_state, config, params, registered_variables)

# retrieving the results of interest
density_3D = final_state_3D[registered_variables.density_index].flatten()
velocity_3D = jnp.sqrt(final_state_3D[registered_variables.velocity_index.x].flatten()**2 + final_state_3D[registered_variables.velocity_index.y].flatten()**2 + final_state_3D[registered_variables.velocity_index.z].flatten()**2)
pressure_3D = final_state_3D[registered_variables.pressure_index].flatten()

# applying the code units
density_3D = density_3D * code_units.code_density
velocity_3D = velocity_3D * code_units.code_velocity
pressure_3D = pressure_3D * code_units.code_pressure

# nicer units
density_3D = (density_3D / m_p).to(u.cm**-3)
velocity_3D = velocity_3D.to(u.km / u.s)
pressure_3D = (pressure_3D / c.k_B).to(u.K / u.cm**3)

# ==== 1D simulation ====

num_cells = 2048

# spatial domain
geometry = SPHERICAL

# time stepping
C_CFL = 0.8

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = False,
    geometry = geometry,
    box_size = box_size, 
    num_cells = num_cells,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = num_injection_cells,
        trace_wind_density = False,
        wind_injection_scheme = EI
    ),
    cooling_config = CoolingConfig(
        cooling = cooling,
        cooling_curve_config = CoolingCurveConfig(
            cooling_curve_type = PIECEWISE_POWER_LAW
        )
    ),
)

helper_data_1D = get_helper_data(config)
registered_variables = get_registered_variables(config)

# simulation params
params = SimulationParams(
    C_cfl = C_CFL,
    dt_max = 0.001,
    t_end = t_end,
    wind_params = wind_params,
    cooling_params = CoolingParams(
        hydrogen_mass_fraction = hydrogen_mass_fraction,
        metal_mass_fraction = metal_mass_fraction,
        floor_temperature = floor_temperature,
        cooling_curve_params = cooling_curve_params
    ),
)

# homogeneous initial state
rho_init = jnp.ones(num_cells) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(num_cells)
p_init = jnp.ones(num_cells) * p_0.to(code_units.code_pressure).value

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho_init,
    velocity_x = u_init,
    gas_pressure = p_init
)

config = finalize_config(config, initial_state.shape)

# running the simulation
final_state_1D = time_integration(initial_state, config, params, registered_variables)

# retrieving the results of interest
density_1D = final_state_1D[registered_variables.density_index].flatten()
velocity_1D = final_state_1D[registered_variables.velocity_index].flatten()
pressure_1D = final_state_1D[registered_variables.pressure_index].flatten()

# applying the code units
density_1D = density_1D * code_units.code_density
velocity_1D = velocity_1D * code_units.code_velocity
pressure_1D = pressure_1D * code_units.code_pressure

# nicer units
density_1D = (density_1D / m_p).to(u.cm**-3)
velocity_1D = velocity_1D.to(u.km / u.s)
pressure_1D = (pressure_1D / c.k_B).to(u.K / u.cm**3)

# ==== No cooling reference ====

r_1D = helper_data_1D.geometric_centers.flatten() * code_units.code_length

# get weaver solution
weaver = Weaver(
    params.wind_params.wind_final_velocity * code_units.code_velocity,
    params.wind_params.wind_mass_loss_rate * code_units.code_mass / code_units.code_time,
    rho_0,
    p_0
)

current_time = params.t_end * code_units.code_time

# density
r_density_weaver, density_weaver = weaver.get_density_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_density_weaver = r_density_weaver.to(u.parsec)
density_weaver = (density_weaver / m_p).to(u.cm**-3)

# velocity
r_velocity_weaver, velocity_weaver = weaver.get_velocity_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_velocity_weaver = r_velocity_weaver.to(u.parsec)
velocity_weaver = velocity_weaver.to(u.km / u.s)

# pressure
r_pressure_weaver, pressure_weaver = weaver.get_pressure_profile(0.01 * u.parsec, 3.5 * u.parsec, current_time)
r_pressure_weaver = r_pressure_weaver.to(u.parsec)
pressure_weaver = (pressure_weaver / c.k_B).to(u.cm**-3 * u.K)

# ==== Plotting ====

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].set_yscale("log")
axs[0].scatter(r_3D.to(u.parsec), density_3D, label="3D simulation", s = 1)
axs[0].scatter(r_1D.to(u.parsec), density_1D, label="1D simulation", s = 3)
# axs[0].plot(r_density_weaver, density_weaver, "--", label="Weaver solution (no cooling)")
axs[0].set_title("density")
axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
axs[0].set_xlim(0, 3)
axs[0].legend(loc="upper right")
axs[0].set_xlabel("r in pc")

axs[1].set_yscale("log")
axs[1].scatter(r_3D.to(u.parsec), pressure_3D, label="3D simulation", s = 1)
axs[1].scatter(r_1D.to(u.parsec), pressure_1D, label="1D simulation", s = 3)
# axs[1].plot(r_pressure_weaver, pressure_weaver, "--", label="Weaver solution (no cooling)")
axs[1].set_title("pressure")
axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
axs[1].set_xlim(0, 3)
axs[1].legend(loc="upper right")
axs[1].set_xlabel("r in pc")

axs[2].set_yscale("log")
axs[2].scatter(r_3D.to(u.parsec), velocity_3D, label="3D simulation", s = 1)
axs[2].scatter(r_1D.to(u.parsec), velocity_1D, label="1D simulation", s = 3)
# axs[2].plot(r_velocity_weaver, velocity_weaver, "--", label="Weaver solution (no cooling)")
axs[2].set_title("velocity")
axs[2].set_ylim(1, 1e4)
axs[2].set_xlim(0, 3)
axs[2].set_ylabel("v in km/s")
axs[2].legend(loc="upper right")
axs[2].set_xlabel("r in pc")

plt.tight_layout()

plt.savefig("figures/wind_comparison.png", dpi=600)