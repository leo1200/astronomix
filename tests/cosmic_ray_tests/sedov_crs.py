import jax.numpy as jnp

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# astronomix data structures
from astronomix import SimulationConfig
from astronomix import SimulationParams

# astronomix constants
from astronomix.option_classes.simulation_config import CARTESIAN, SPHERICAL, HLL, MINMOD

# astronomix functions
from astronomix import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config

from astronomix import time_integration

# cosmic rays
from astronomix._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from astronomix._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayParams

import matplotlib.pyplot as plt

# For the exact Sedov-Taylor solution
from exactpack.solvers.sedov.sedov import Sedov

from astronomix.shock_finder.shock_finder import shock_criteria

def sedov_cr_test():

    config = SimulationConfig(

        
        geometry = SPHERICAL,
        first_order_fallback = True,
        progress_bar = False,
        runtime_debugging = False,


        # ====== RESOLUTION =======
        num_cells = 801,
        # =========================


        # =========================
        # NOTE: THE INJECTION RADIUS CURRENTLY
        # IS AT THE 10th CELL. IF YOU CHOOSE
        # AN INJECTION RADIUS TOO BIG
        # THIS WILL LEAD TO A DIFFERENT SOLUTION
        # STRUCTURE AKIN TO A SHOCK TUBE.
        # =========================


        # ======= COSMIC RAY SWITCH =======
        # ONLY ADVECTIVE COSMIC RAYS,
        # FOLLOWING doi.org/10.1088/1742-6596/719/1/012021
        # BUT JUST USING A NORMAL HLL SOLVER
        # WHICH IS MORE DIFFUSIVE BUT DOES NOT REQUIRE
        # A FIX AGAINST OSCILLATIONS AS NECESSARY FOR
        # A ROE SOLVER.
        cosmic_ray_config = CosmicRayConfig(
                                cosmic_rays = True,
                                diffusive_shock_acceleration = True,
                            ),
        # =================================


        # ==== snapshotting ====
        return_snapshots = True,
        num_snapshots = 20,
        # ======================

    )

    params = SimulationParams(
        t_end = 0.1,
        cosmic_ray_params = CosmicRayParams(
            diffusive_shock_acceleration_start_time = 0.0001,
            diffusive_shock_acceleration_efficiency = 0.5
        ),
        dt_max = 0.00001
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # total explosion energy
    E_explosion = 1.0

    # partition of energy: fraction into cosmic rays
    if config.cosmic_ray_config.diffusive_shock_acceleration:
        f_cr = 0.0
    else:
        f_cr = 0.0  # 0% of energy into cosmic rays

    E_gas = (1 - f_cr) * E_explosion
    E_cr  = f_cr * E_explosion

    # Ambient (background) physical conditions (adjust as needed)
    rho_ambient_phys  = 1.0         # typical ISM density
    p_ambient_phys    = 1e-4          # low gas pressure
    p_cr_ambient_phys = 0          # low cosmic ray pressure

    # Density in code units
    rho_ambient = rho_ambient_phys

    # Pressures in code units
    p_ambient   = p_ambient_phys
    p_cr_ambient = p_cr_ambient_phys

    # --- Set Up the Explosion Injection Region ---

    num_injection_cells = 5
    r_explosion_phys = helper_data.outer_cell_boundaries[num_injection_cells]
    r_explosion = r_explosion_phys

    # Compute the injection volume (spherical volume in code units)
    injection_volume = (4/3) * jnp.pi * r_explosion_phys**3

    # Adiabatic indices:
    gamma_gas = params.gamma   # for the thermal gas
    gamma_cr  = 4/3   # for cosmic rays

    # The energy contained in a uniform pressure region is related by:
    #   E = p * V / (gamma - 1)
    # Hence, the effective explosion pressure in the injection region (in code units)
    p_explosion_gas_phys = E_gas * (gamma_gas - 1) / injection_volume
    p_explosion_cr_phys  = E_cr  * (gamma_cr  - 1) / injection_volume

    # Convert to code units
    p_explosion_gas = p_explosion_gas_phys
    p_explosion_cr  = p_explosion_cr_phys

    # --- Define the Radial Profiles ---
    # Get the radial coordinate array (assumed already available)
    r = helper_data.geometric_centers  # e.g. radial centers of the grid cells

    # Density: assume uniform ambient density everywhere
    rho = rho_ambient * jnp.ones_like(r)

    # Radial velocity: initially at rest
    u_r = jnp.zeros_like(r)

    # Gas pressure: high within the explosion region, ambient elsewhere
    p_gas = p_ambient * jnp.ones_like(r)
    p_gas = p_gas.at[:num_injection_cells + 1].set(p_explosion_gas)

    # Cosmic ray pressure: similarly high inside the explosion region, ambient outside
    p_cr = p_cr_ambient * jnp.ones_like(r)
    p_cr = p_cr.at[:num_injection_cells + 1].set(p_explosion_cr)

    # --- Build the Initial State ---
    initial_state = construct_primitive_state(
        registered_variables=registered_variables,
        config=config,
        density=rho,
        velocity_x=u_r,
        gas_pressure=p_gas,
        cosmic_ray_pressure=p_cr
    )

    config = finalize_config(config, initial_state.shape)

    result = time_integration(initial_state, config, params, registered_variables)

    final_state = result.states[-1]

    rho_final = final_state[registered_variables.density_index]
    u_final = final_state[registered_variables.velocity_index]
    p_final = final_state[registered_variables.pressure_index]
    n_cr_final = final_state[registered_variables.cosmic_ray_n_index]
    p_cr_final = n_cr_final ** gamma_cr
    if config.cosmic_ray_config.cosmic_rays:
        p_gas_final = p_final - p_cr_final
    else:
        p_gas_final = p_final

    from astronomix.shock_finder.shock_finder import find_shock_zone, shock_sensor

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))

    debug_state = result.states[1]

    shock_crit = shock_criteria(debug_state, config, registered_variables, helper_data)
    print(jnp.any(shock_crit))

    # sensors = shock_sensor(final_state[registered_variables.pressure_index])
    shock_index, shock_index_left, shock_index_right = find_shock_zone(debug_state, config, registered_variables, helper_data)

    print("shock_index", shock_index)
    print("shock_index_right", shock_index_right)
    print("shock_index_left", shock_index_left)

    gas_pressure = debug_state[registered_variables.pressure_index] - debug_state[registered_variables.cosmic_ray_n_index] ** gamma_cr
    cr_pressure = debug_state[registered_variables.cosmic_ray_n_index] ** gamma_cr

    ax1.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], debug_state[registered_variables.pressure_index, shock_index - 15:shock_index + 15], "o--", color='black', label='total pressure')
    ax1.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], gas_pressure[shock_index - 15:shock_index + 15], "o--", color='red', label='gas pressure')
    ax1.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], cr_pressure[shock_index - 15:shock_index + 15], "o--", color='blue', label='cosmic ray pressure')

    ax1.axvline(helper_data.geometric_centers[shock_index], color='red', label='cell with max shock sensor')
    ax1.axvline(helper_data.geometric_centers[shock_index_right], color='blue', label='right shock boundary')
    ax1.axvline(helper_data.geometric_centers[shock_index_left], color='blue', label='left shock boundary')

    ax1.set_ylabel('Pressure')
    ax1.set_xlabel('Radius')
    ax1.legend(loc = 'upper right')

    ax2 = ax1.twinx()
    ax2.plot(helper_data.geometric_centers[shock_index - 15:shock_index + 15], shock_sensor(debug_state[registered_variables.pressure_index])[shock_index - 15:shock_index + 15], "o--", color='green')
    ax2.set_ylabel('Shock Sensor')

    pressure = debug_state[registered_variables.pressure_index]
    pressure_differences = jnp.zeros_like(pressure)
    pressure_differences = pressure_differences.at[1:].set(pressure[:-1] - pressure[1:])

    plt.savefig('figures/shock_debug.svg')

    eblast_exact = 1.0 / 1.0  # E_explosion / rho_ambient
    sedov_solver = Sedov(geometry=3, eblast=eblast_exact, gamma = 7 / 5, omega=0.0)
    r_exact = jnp.linspace(0.0, 1.0, 500)
    solution_exact = sedov_solver(r=r_exact, t=params.t_end)

    fig, axs = plt.subplots(2, 3, figsize=(15, 5))

    a = 0
    b = -1

    axs[0, 0].plot(r[a:b], rho[a:b], label='initial')
    axs[0, 0].set_ylabel('Density')
    axs[1, 0].plot(r[a:b], rho_final[a:b], label='final')
    axs[1, 0].plot(r_exact, solution_exact['density'], ls='--', lw=2, c='red', label='exact solution for gamma = 7/5')
    axs[1, 0].set_ylabel('Density')
    axs[1, 0].set_xlabel('r in pc')
    axs[0, 0].set_title('Density')
    axs[0, 0].legend()
    axs[1, 0].legend()

    axs[0, 1].plot(r[a:b], u_r[a:b], label='initial')
    axs[0, 1].set_ylabel('Velocity')
    axs[1, 1].plot(r, u_final, label='final')
    axs[1, 1].plot(r_exact, solution_exact['velocity'], ls='--', lw=2, c='red', label='exact solution for gamma = 7/5')
    axs[1, 1].set_ylabel('Velocity')
    axs[1, 1].set_xlabel('r in pc')
    axs[0, 1].set_title('Velocity')
    axs[0, 1].legend()
    axs[1, 1].legend()

    axs[0, 2].plot(r[a:b], p_gas[a:b], label='initial, gas')
    axs[1, 2].plot(r[a:b], p_gas_final[a:b], label='final, gas')

    if config.cosmic_ray_config.cosmic_rays:
        axs[0, 2].plot(r[a:b], p_gas[a:b] + p_cr[a:b], label='initial, total')
        axs[1, 2].plot(r[a:b], p_final[a:b], label='final, total')
        axs[0, 2].plot(r[a:b], p_cr[a:b], label='initial, CRs')
        axs[1, 2].plot(r[a:b], p_cr_final[a:b], label='final, CRs')

    axs[1, 2].plot(r_exact, solution_exact['pressure'], ls='--', lw=2, c='red', label='exact solution for gamma = 7/5')

    axs[0, 2].set_ylabel('Pressure')
    axs[1, 2].set_ylabel('Pressure')
    axs[1, 2].set_xlabel('r')

    axs[0, 2].set_title('Pressure')
    axs[0, 2].legend()
    axs[1, 2].legend()

    # set all xlims to 0, 0.5
    for ax in axs.flat:
        ax.set_xlim(0, 0.5)

    plt.tight_layout()

    plt.savefig('figures/sedov_crs.svg')