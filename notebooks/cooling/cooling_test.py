# TODO: fix units

from autocvd import autocvd
autocvd(num_gpus = 1)

from astronomix._physics_modules._cooling._cooling import get_pressure_from_temperature, get_temperature_from_pressure
from astronomix._physics_modules._cooling._cooling_tables import schure_cooling

import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes import WindConfig
from astronomix._physics_modules._cooling.cooling_options import PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, CoolingConfig, CoolingParams, PiecewisePowerLawParams, SimplePowerLawParams

from astronomix import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config

from astronomix import time_integration

# astronomix constants
from astronomix.option_classes.simulation_config import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, SPHERICAL

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from astronomix._physics_modules._stellar_wind.weaver import Weaver


print("👷 Setting up simulation...")

# simulation settings
gamma = 5/3

# spatial domain
geometry = SPHERICAL
box_size = 1.0
num_cells = 10001

left_boundary = REFLECTIVE_BOUNDARY
right_boundary = OPEN_BOUNDARY

# activate stellar wind
stellar_wind = True

# setup simulation config
config = SimulationConfig(
    runtime_debugging = False,
    progress_bar = False,
    geometry = geometry,
    box_size = box_size, 
    num_cells = num_cells,
    wind_config = WindConfig(
        stellar_wind = stellar_wind,
        num_injection_cells = 5,
        trace_wind_density = False,
    ),
    cooling_config = CoolingConfig(
        cooling = True,
        cooling_curve_type = PIECEWISE_POWER_LAW
    )
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# code units
code_length = 10e18 * u.cm
code_mass = 1e-3 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
reference_temperature = (1e8 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
# without a floor temperature, the simulations crash
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
factor = (1e-23 * u.erg * u.cm ** 3 / u.s / c.m_p ** 2).to(code_units.code_energy * code_units.code_length ** 3 / (code_units.code_time * code_units.code_mass ** 2)).value
exponent = -0.7

cooling_curve_paramsA = SimplePowerLawParams(
    factor = factor,
    exponent = exponent,
    reference_temperature = reference_temperature
)

cooling_curve_paramsB = schure_cooling(code_units)

print("reference temperature", reference_temperature)
print("factor", factor)

# time stepping
C_CFL = 0.8
t_final = 1.25e12 * u.s
t_end = t_final.to(code_units.code_time).value
dt_max = 0.1 * t_end

# wind parameters
wind_final_velocity = 1500 * u.km / u.s
wind_mass_loss_rate = 1e-6 * u.M_sun / u.yr
wind_params = WindParams(
    wind_mass_loss_rate = wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity = wind_final_velocity.to(code_units.code_velocity).value
)

# simulation params
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
        cooling_curve_params = cooling_curve_paramsB
    )
)

# homogeneous initial state
rho_0 = 10 ** (-22.5) * u.g / u.cm**3
T_0 = 100 * u.K * c.k_B / c.m_p
p_0 = get_pressure_from_temperature(rho_0, T_0, hydrogen_mass_fraction, metal_mass_fraction)
print(p_0)

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

# main simulation loop
final_state = time_integration(initial_state, config, params, registered_variables)

# compare with weaver solution
def plot_weaver_comparison(axs, final_state, params, helper_data, code_units, rho_0, p_0):
    print("👷 generating plots")

    rho = final_state[registered_variables.density_index]
    vel = final_state[registered_variables.velocity_index]
    p = final_state[registered_variables.pressure_index]

    rho = rho * code_units.code_density
    vel = vel * code_units.code_velocity
    p = p * code_units.code_pressure

    r = helper_data.geometric_centers * code_units.code_length

    # get weaver solution
    weaver = Weaver(
        params.wind_params.wind_final_velocity * code_units.code_velocity,
        params.wind_params.wind_mass_loss_rate * code_units.code_mass / code_units.code_time,
        rho_0,
        p_0
    )
    current_time = params.t_end * code_units.code_time
    
    # density
    r_density_weaver, density_weaver = weaver.get_density_profile(0.01 * u.parsec, 10e18 * u.cm, current_time)
    r_density_weaver = r_density_weaver.to(u.parsec)
    density_weaver = (density_weaver / m_p).to(u.cm**-3)

    # velocity
    r_velocity_weaver, velocity_weaver = weaver.get_velocity_profile(0.01 * u.parsec, 10e18 * u.cm, current_time)
    r_velocity_weaver = r_velocity_weaver.to(u.parsec)
    velocity_weaver = velocity_weaver.to(u.km / u.s)

    # pressure
    r_pressure_weaver, pressure_weaver = weaver.get_pressure_profile(0.01 * u.parsec, 10e18 * u.cm, current_time)
    r_pressure_weaver = r_pressure_weaver.to(u.parsec)
    pressure_weaver = (pressure_weaver / c.k_B).to(u.cm**-3 * u.K)

    # temperature

    T = get_temperature_from_pressure(rho, p * c.m_p / c.k_B, hydrogen_mass_fraction, metal_mass_fraction)
    T = T.to(u.K)

    axs[0].set_yscale("log")
    axs[0].plot(r.to(u.cm), (rho).to(u.g * u.cm**-3), label="astronomix")
    axs[0].set_title("density")
    axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
    axs[0].set_ylim(1e-27, 1e-21)
    axs[0].set_xlim(0, 1e19)
    axs[0].legend(loc="upper left")
    axs[0].set_xlabel("r in cm")

    axs[1].set_yscale("log")
    axs[1].plot(r.to(u.cm), (p / c.k_B).to(u.K / u.cm**3), label="astronomix")
    axs[1].set_title("pressure")
    axs[1].set_ylabel(r"$p$/k$_b$ in K cm$^{-3}$")
    axs[1].set_xlim(0, 1e19)
    axs[1].legend(loc="upper left")
    axs[1].set_xlabel("r in cm")


    axs[2].set_yscale("log")
    axs[2].plot(r.to(u.cm), vel.to(u.km / u.s), label="astronomix")
    axs[2].set_title("velocity")
    axs[2].set_ylim(1, 1e4)
    axs[2].set_xlim(0, 1e19)
    axs[2].set_ylabel("v in km/s")
    axs[2].legend(loc="upper right")
    axs[2].set_xlabel("r in cm")

    axs[3].set_yscale("log")
    axs[3].plot(r.to(u.cm), T.to(u.K), label="astronomix")
    axs[3].set_title("temperature")
    axs[3].set_ylim(10, 1e9)
    axs[3].set_xlim(0, 1e19)
    axs[3].set_ylabel("T in K")
    axs[3].legend(loc="upper right")
    axs[3].set_xlabel("r in cm")


fig, axs = plt.subplots(1, 4, figsize=(20, 5))
plot_weaver_comparison(axs, final_state, params, helper_data, code_units, rho_0, p_0)
plt.tight_layout()

plt.savefig("cooling_test.svg")