# ==== GPU selection ====
from autocvd import autocvd

from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix._fluid_equations._equations import get_absolute_velocity, total_energy_from_primitives

from astronomix import get_registered_variables
from astronomix.option_classes import WindConfig

from astronomix.option_classes.simulation_config import BACKWARDS, HLL, HYBRID_HLLC, OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from astronomix._physics_modules._stellar_wind.weaver import Weaver

# turbulence
from astronomix.initial_condition_generation.turbulent_ic_generator import create_turb_field

from astronomix.option_classes.simulation_config import FORWARDS

from astronomix.option_classes.simulation_config import finalize_config

import matplotlib.pyplot as plt

# power spectra
import Pk_library as PKL

from matplotlib.colors import LogNorm


def run_turbulent_simulation(stellar_wind = True, turbulence = True, t_final = 2.5e4 * u.yr, initial_state_given = None):

    # simulation settings
    gamma = 5/3

    # spatial domain
    box_size = 3.0
    num_cells = 128

    wanted_rms = 50 * u.km / u.s

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        first_order_fallback = True,
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
        differentiation_mode = FORWARDS,
        boundary_settings = BoundarySettings(
            BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY),
            BoundarySettings1D(left_boundary = PERIODIC_BOUNDARY, right_boundary = PERIODIC_BOUNDARY)
        ),
        riemann_solver = HYBRID_HLLC,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    code_length = 3 * u.parsec
    code_mass = 1 * u.M_sun
    code_velocity = 100 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    # time domain
    C_CFL = 0.4

    t_end = t_final.to(code_units.code_time).value

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
        gamma = gamma,
        t_end = t_end,
        wind_params = wind_params
    )

    # homogeneous initial state
    rho_0 = 2 * c.m_p / u.cm**3
    p_0 = 3e4 * u.K / u.cm**3 * c.k_B

    rho = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * rho_0.to(code_units.code_density).value

    u_x = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
    u_y = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))
    u_z = jnp.zeros((config.num_cells, config.num_cells, config.num_cells))

    turbulence_slope = -2.0
    kmin = 2
    kmax = int(0.6 * (num_cells // 2))

    if turbulence:

        key = jax.random.PRNGKey(42)
        key, sk1, sk2, sk3 = jax.random.split(key, 4)

        ux = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key = sk1)
        uy = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key = sk2)
        uz = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key = sk3)

        rms_vel = jnp.sqrt(jnp.mean(ux**2 + uy**2 + uz**2))

        u_x = ux / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_y = uy / rms_vel * wanted_rms.to(code_units.code_velocity).value
        u_z = uz / rms_vel * wanted_rms.to(code_units.code_velocity).value

        # print the maximum velocity
        print(f'Maximum velocity: {(jnp.max(jnp.sqrt(u_x**2 + u_y**2 + uz**2)) * code_units.code_velocity).to(u.km / u.s).value:.2f} km/s')

    p = jnp.ones((config.num_cells, config.num_cells, config.num_cells)) * p_0.to(code_units.code_pressure).value

    # construct primitive state

    initial_state = construct_primitive_state(
        config = config,
        registered_variables=registered_variables,
        density = rho,
        velocity_x = u_x,
        velocity_y = u_y,
        velocity_z = u_z,
        gas_pressure = p
    )

    if initial_state_given is not None:
        # use the given initial state
        initial_state = initial_state_given

    config = finalize_config(config, initial_state.shape)

    return initial_state, time_integration(initial_state, config, params, registered_variables), config, registered_variables, params

def get_energy(primitive_state, config, registered_variables, params):
    """Calculate the total energy from the primitive state."""
    rho = primitive_state[registered_variables.density_index]
    u = get_absolute_velocity(primitive_state, config, registered_variables)
    p = primitive_state[registered_variables.pressure_index]
    return total_energy_from_primitives(rho, u, p, params.gamma)

# turbulence only simulation
(
    initial_state_turb,
    final_state_pure_turb,
    config,
    registered_variables,
    params
) = run_turbulent_simulation(stellar_wind = False, turbulence = True, t_final = 0.3e4 * u.yr)

num_cells = final_state_pure_turb.shape[-1]
initial_energy = get_energy(initial_state_turb, config, registered_variables, params)
final_energy = get_energy(final_state_pure_turb, config, registered_variables, params)

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
initial_energy_np = np.array(initial_energy, dtype = np.float32)
final_energy_np = np.array(final_energy, dtype = np.float32)
Pk_energy_initial = PKL.Pk(initial_energy_np, 1, 0, "None", 6, 0)
Pk_energy_final = PKL.Pk(final_energy_np, 1, 0, "None", 6, 0)

fig, axs = plt.subplots(2, 3, figsize = (15, 9))

# axs[0, 0] - the initial energy power spectrum
axs[0, 0].plot(Pk_energy_initial.k1D, Pk_energy_initial.Pk1D, label = "energy power spectrum, $P_{1D}(k_\parallel)$", color = "blue")
axs[0, 0].plot(Pk_energy_initial.k1D, Pk_energy_initial.k1D**-2, label = "$k^{-2}$ for reference", color = "red", linestyle = "--")
axs[0, 0].set_xscale("log")
axs[0, 0].set_yscale("log")
axs[0, 0].set_xlabel("k")
axs[0, 0].set_ylabel("P(k)")
axs[0, 0].set_ylim(1e-6, 1e-1)
axs[0, 0].set_title("initial energy power spectrum")
axs[0, 0].legend()

# axs[1, 0] - the final energy power spectrum
axs[1, 0].plot(Pk_energy_final.k1D, Pk_energy_final.Pk1D, label = "energy power spectrum, $P_{1D}(k_\parallel)$", color = "blue")
axs[1, 0].plot(Pk_energy_final.k1D, Pk_energy_final.k1D**-2, label = "$k^{-2}$ for reference", color = "red", linestyle = "--")
axs[1, 0].set_xscale("log")
axs[1, 0].set_yscale("log")
axs[1, 0].set_xlabel("k")
axs[1, 0].set_ylabel("P(k)")
axs[1, 0].set_title("final energy power spectrum")
axs[1, 0].legend()

# axs[0, 1] - the initial density field, as an imshow
im = axs[0, 1].imshow(
    initial_state_turb[registered_variables.density_index, :, :, num_cells // 2],
    origin="lower",
    cmap="viridis",
    extent=(0, config.box_size, 0, config.box_size)
)
axs[0, 1].set_title("initial density field")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")

divider = make_axes_locatable(axs[0, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, label="density")

# axs[1, 1] - the final density field, as an imshow
im = axs[1, 1].imshow(
    final_state_pure_turb[registered_variables.density_index, :, :, num_cells // 2],
    origin="lower",
    cmap="viridis",
    extent=(0, config.box_size, 0, config.box_size)
)
axs[1, 1].set_title("final density field")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")

divider = make_axes_locatable(axs[1, 1])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, label="density")

# axs[0, 2] - empty, remove
axs[0, 2].axis("off")

# axs[1, 2] - histogram of the final log density field
log_density_final = jnp.log(final_state_pure_turb[registered_variables.density_index].flatten() / jnp.mean(final_state_pure_turb[registered_variables.density_index]))
axs[1, 2].hist(log_density_final, bins = 100, density = True, color = "blue", alpha = 0.7, label='log density histogram')

# fit a Gaussian to the histogram
mean = jnp.mean(log_density_final)
std = jnp.std(log_density_final)
x = jnp.linspace(mean - 4 * std, mean + 4 * std, 100)
axs[1, 2].plot(x, 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mean)**2 / std**2), color='red', label='Gaussian fit\n$\mu$ = {:.2f}, $\sigma$ = {:.2f}\n$\mu_s = - \sigma^2/2$ = {:.2f}'.format(mean, std, -std**2 / 2))

axs[1, 2].set_title("histogram of final log density")
axs[1, 2].set_xlabel(r"log($\rho / \bar{\rho}$)")
axs[1, 2].set_ylabel("probability density")

axs[1, 2].legend()

plt.tight_layout()

plt.savefig("turbulence.png", dpi = 500)

# do a stellar wind simulation with turbulence, taking the final state of the turbulence simulation as initial state

(
    initial_state_wind,
    final_state_wind,
    config,
    registered_variables,
    params
) = run_turbulent_simulation(stellar_wind = True, turbulence = True, t_final = 2.5e4 * u.yr, initial_state_given = final_state_pure_turb)

num_cells = final_state_wind.shape[-1]

# only plot an imshow of the density field at the center of the box
fig, axs = plt.subplots(1, 1, figsize = (6, 6))
im = axs.imshow(
    final_state_wind[registered_variables.density_index, :, :, num_cells // 2],
    origin="lower",
    cmap="viridis",
    extent=(0, config.box_size, 0, config.box_size),
    norm = LogNorm()
)
# add a colorbar
divider = make_axes_locatable(axs)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, label="density")
axs.set_title("density field with stellar wind and turbulence")
axs.set_xlabel("x")
axs.set_ylabel("y")
plt.tight_layout()
plt.savefig("turbulence_wind.png", dpi = 500)