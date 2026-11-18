"""
1D spherical stellar wind bubble compared with the Weaver analytic solution.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# units and constants
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    FORWARDS,
    SPHERICAL,
)
from astronomix._modules._stellar_wind.stellar_wind_options import EI

# astronomix containers
from astronomix import (
    CodeUnits,
    WindParams,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes import WindConfig
from astronomix._modules._stellar_wind.weaver import Weaver

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    finalize_config,
)


# figures are written to the local figures/ directory
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# simulation settings
# ---------------------------------------------------------------------------
GAMMA = 5 / 3
BOX_SIZE = 1.0
NUM_CELLS = 2000

code_units = CodeUnits(20 * u.pc, 1 * u.M_sun, 100 * u.km / u.s)

config = SimulationConfig(
    geometry=SPHERICAL,
    solver_mode=FINITE_VOLUME,
    progress_bar=True,
    box_size=BOX_SIZE,
    num_cells=NUM_CELLS,
    wind_config=WindConfig(
        stellar_wind=True,
        num_injection_cells=20,
        trace_wind_density=False,
        wind_injection_scheme=EI,
    ),
    differentiation_mode=FORWARDS,
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

# ---------------------------------------------------------------------------
# physical parameters
# ---------------------------------------------------------------------------
t_final = 0.5 * 1e6 * u.yr
t_end = t_final.to(code_units.code_time).value
dt_max = 0.1 * t_end

M_star = 5000 * u.M_sun
wind_final_velocity = 3230 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 * M_star / (1e6 * u.yr)

wind_params = WindParams(
    wind_mass_loss_rate=wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity=wind_final_velocity.to(code_units.code_velocity).value,
)

params = SimulationParams(
    C_cfl=0.8,
    dt_max=dt_max,
    gamma=GAMMA,
    t_end=t_end,
    wind_params=wind_params,
)

# homogeneous ambient medium
rho_0 = 2 * m_p / u.cm ** 3
p_0 = 3e4 * u.K / u.cm ** 3 * c.k_B

rho_init = jnp.ones(NUM_CELLS) * rho_0.to(code_units.code_density).value
u_init = jnp.zeros(NUM_CELLS)
p_init = jnp.ones(NUM_CELLS) * p_0.to(code_units.code_pressure).value

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho_init,
    velocity_x=u_init,
    gas_pressure=p_init,
)

config = finalize_config(config, initial_state.shape)

final_state = time_integration(initial_state, config, params, registered_variables)

# ---------------------------------------------------------------------------
# physical profiles and the Weaver analytic wind-bubble solution
# ---------------------------------------------------------------------------
rho = final_state[registered_variables.density_index] * code_units.code_density
vel = final_state[registered_variables.velocity_index] * code_units.code_velocity
p = final_state[registered_variables.pressure_index] * code_units.code_pressure
r = helper_data.geometric_centers * code_units.code_length

weaver = Weaver(
    params.wind_params.wind_final_velocity * code_units.code_velocity,
    params.wind_params.wind_mass_loss_rate * code_units.code_mass / code_units.code_time,
    rho_0,
    p_0,
)
current_time = params.t_end * code_units.code_time
r_lo, r_hi = 0.01 * u.parsec, code_units.code_length.to(u.parsec)

r_rho_w, rho_w = weaver.get_density_profile(r_lo, r_hi, current_time)
r_vel_w, vel_w = weaver.get_velocity_profile(r_lo, r_hi, current_time)
r_p_w, p_w = weaver.get_pressure_profile(r_lo, r_hi, current_time)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].set_yscale("log")
axs[0].plot(r.to(u.pc), (rho / m_p).to(u.cm ** -3), label="astronomix")
axs[0].plot(r_rho_w.to(u.pc), (rho_w / m_p).to(u.cm ** -3), "--", label="Weaver")
axs[0].set_xlabel("r in pc")
axs[0].set_ylabel(r"$\rho$ in m$_p$ cm$^{-3}$")
axs[0].set_title("density")
axs[0].legend()

axs[1].set_yscale("log")
axs[1].plot(r.to(u.pc), (p / c.k_B).to(u.K / u.cm ** 3), label="astronomix")
axs[1].plot(r_p_w.to(u.pc), (p_w / c.k_B).to(u.cm ** -3 * u.K), "--", label="Weaver")
axs[1].set_xlabel("r in pc")
axs[1].set_ylabel(r"$p / k_B$ in K cm$^{-3}$")
axs[1].set_title("pressure")
axs[1].legend()

axs[2].set_yscale("log")
axs[2].plot(r.to(u.pc), vel.to(u.km / u.s), label="astronomix")
axs[2].plot(r_vel_w.to(u.pc), vel_w.to(u.km / u.s), "--", label="Weaver")
axs[2].set_xlabel("r in pc")
axs[2].set_ylabel("v in km/s")
axs[2].set_title("velocity")
axs[2].legend()

fig.suptitle("1D spherical stellar wind vs. Weaver solution")
fig.tight_layout()
fig.savefig(figures_dir / "stellar_wind_1d.png", dpi=150)
