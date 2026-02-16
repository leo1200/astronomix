import jax.numpy as jnp

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# astronomix data structures
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from astronomix._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayParams

# astronomix constants
from astronomix.option_classes.simulation_config import CARTESIAN, SPHERICAL, HLL, MINMOD

# astronomix functions
from astronomix import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.shock_finder.shock_finder import find_shock_zone, shock_sensor

from astronomix import time_integration

import numpy as np
from matplotlib.colors import LogNorm
from astropy import units as u, constants  as c
import matplotlib.pyplot as plt

# constants in cgs units
pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.to("s")
yr = u.yr.to("s")
kyr = u.kyr.to("s")
mp = c.m_p.cgs.value

def philipp_sedov():

    # identify useful parameters for Sedov explosions in the ISM

    # final ST radius
    def _rST(E51, n):
        return 19.1 * np.power(E51,5./17.) * np.power(n,-7./17.)

    # time at RST
    def _tST(E51, n, xi=1.15, mu=0.61):
        pref = np.power(19.1*pc/xi, 5./2.) * np.power(mu*mp/1e51, 1./2.)
        return pref * np.power(E51, 8./34.) * np.power(n, -18./34.)

    plot_grid = False

    # typical resolutions
    resdx = [0.1, 0.2, 0.4, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0]
    n     = np.logspace(-2, 0, 3)
    print(n)
    # compute rSR and time
    RST = _rST(1.0, n)
    print(RST)
    tST = _tST(1.0, n)
    print(tST/kyr)

    print()
    print("for a box with 300 pc size:")
    for resloc in resdx:
        print("res =",resloc, "domain resolved with", 300/resloc, "cells")

    print()
    print("for a total time of 0.5 Myr")
    for tloc in tST:
        print(500*kyr / tloc)

    # plot 3*ST radius for a few densities and check how large the box needs to be to get appropriate resolution
    if plot_grid:
        n  = np.logspace(-3,3,15)
        E  = np.logspace(-1,1,10)
        
        nn, EE = np.meshgrid(n, E, indexing="xy")
        R = _rST(EE, nn)  # Note: EE = y-axis, nn = x-axis
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(n, E, R, norm=LogNorm())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (cm^-3)")
        ax.set_ylabel("E (foe)")
        cbar = plt.colorbar(im, label="radius (pc)")
        ax.set_title("Sedov-Taylor radius")
        
        xi = 1.15
        mu = 0.61
        t = _tST(xi, mu, EE, nn)/kyr
        fig, ax = plt.subplots()
        im = ax.pcolormesh(n, E, t, norm=LogNorm())
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (cm^-3)")
        ax.set_ylabel("E (foe)")
        cbar = plt.colorbar(im, label="time to reach R_ST (kyr)")
        ax.set_title("time to reach Sedov-Taylor radius")

        plt.savefig("figures/sedov_taylor_radius.png")

    code_length = 3 * u.parsec
    code_mass = 1e-3 * u.M_sun
    code_velocity = 1 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)

    # encapsulate functions to do a loop later 

    # Adiabatic indices:
    gamma_gas = 5/3   # for the thermal gas
    gamma_cr  = 4/3   # for cosmic rays

    def _run_sim(code_units, Ncell=301, DSA=True, tend_yr=100, DSA_t0_yr=10, n0=1.0, p0_gas=1e-10, p0_cr=1e-12, \
                box_size=1.0, E51=1.0, Ninj=10, debug=False):

        print("box_size (pc) = ", box_size * code_units.code_length.to(u.pc))
        print("R_ST     (pc) = ", _rST(E51,n0))
        print("t_ST    (kyr) = ", _tST(E51,n0)/kyr)
        print("t_sim   (kyr) = ", tend_yr/1000)
        
        config = SimulationConfig(
            geometry = SPHERICAL,
            first_order_fallback = True,
            num_cells = Ncell,
            cosmic_ray_config = CosmicRayConfig(
                                    cosmic_rays = True,
                                    diffusive_shock_acceleration = DSA,
                                ),
            return_snapshots = True,
            num_snapshots = 200,
            box_size = box_size,
            runtime_debugging = debug,
            limiter = MINMOD
        )
        if debug:
            print(config)
        
        params = SimulationParams(
            t_end = (tend_yr * u.yr).to(code_units.code_time).value,
            cosmic_ray_params = CosmicRayParams(
                diffusive_shock_acceleration_start_time = (DSA_t0_yr * u.yr).to(code_units.code_time).value,
                diffusive_shock_acceleration_efficiency = 0.1
            ),
        )
        
        helper_data = get_helper_data(config)
        registered_variables = get_registered_variables(config)
        
        # total explosion energy
        E_explosion = E51 * 1e51 * u.erg
        
        # partition of energy: fraction into cosmic rays
        if config.cosmic_ray_config.diffusive_shock_acceleration:
            f_cr = 0.0
        else:
            f_cr = 0.1  # 10% of energy into cosmic rays
        
        E_gas = (1 - f_cr) * E_explosion
        E_cr  = f_cr * E_explosion

        # Ambient (background) physical conditions (adjust as needed)
        rho_ambient_phys  = n0 * 1.67e-24 * u.g / u.cm**3          # typical ISM density
        p_ambient_phys    = p0_gas * u.dyn / u.cm**2           # low gas pressure
        p_cr_ambient_phys = p0_cr  * u.dyn / u.cm**2           # low cosmic ray pressure
        
        # Density in code units
        rho_ambient = rho_ambient_phys.to(code_units.code_density).value
        
        # Pressures in code units
        p_ambient   = p_ambient_phys.to(code_units.code_pressure).value
        p_cr_ambient = p_cr_ambient_phys.to(code_units.code_pressure).value
        
        # --- Set Up the Explosion Injection Region ---
        
        # currently, we take Ninj injection cells
        r_explosion_phys = helper_data.outer_cell_boundaries[Ninj-1] * code_units.code_length
        r_explosion = r_explosion_phys.to(code_units.code_length).value
        
        # Compute the injection volume (spherical volume in code units)
        injection_volume = (4/3) * jnp.pi * r_explosion_phys**3
            
        # The energy contained in a uniform pressure region is related by:
        #   E = p * V / (gamma - 1)
        # Hence, the effective explosion pressure in the injection region (in code units)
        p_explosion_gas_phys = E_gas * (gamma_gas - 1) / injection_volume
        p_explosion_cr_phys  = E_cr  * (gamma_cr  - 1) / injection_volume
        
        # Convert to code units
        p_explosion_gas = p_explosion_gas_phys.to(code_units.code_pressure).value
        p_explosion_cr  = p_explosion_cr_phys.to(code_units.code_pressure).value
        
        # --- Define the Radial Profiles ---
        # Get the radial coordinate array (assumed already available)
        r = helper_data.geometric_centers  # e.g. radial centers of the grid cells
        
        # Density: assume uniform ambient density everywhere
        rho = rho_ambient * jnp.ones_like(r)
        
        # Radial velocity: initially at rest
        u_r = jnp.zeros_like(r)
        
        # Gas pressure: high within the explosion region, ambient elsewhere
        p_gas = jnp.where(r <= r_explosion, p_explosion_gas, p_ambient)
        
        # Cosmic ray pressure: similarly high inside the explosion region, ambient outside
        p_cr = jnp.where(r <= r_explosion, p_explosion_cr, p_cr_ambient)
        
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
        
        ## running the simulation
        result = time_integration(initial_state, config, params, registered_variables)
        print(len(result.states))
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
        
        
        # convert back to physical quantities
        r = (helper_data.geometric_centers * code_units.code_length).to(u.pc)
        rho = (rho * code_units.code_density).to(m_p / u.cm**3)
        rho_final = (rho_final * code_units.code_density).to(m_p / u.cm**3)
        u_r = (u_r * code_units.code_velocity).to(u.km / u.s)
        u_final = (u_final * code_units.code_velocity).to(u.km / u.s)
        p = (p_gas_final * code_units.code_pressure / c.k_B).to(u.K / u.cm**3)
        p_cr = (p_cr * code_units.code_pressure / c.k_B).to(u.K / u.cm**3)
        p_cr_final = (p_cr_final * code_units.code_pressure / c.k_B).to(u.K / u.cm**3)

        return config, r, rho, rho_final, u_r, u_final, p_gas, p_gas_final, p_final, p_cr, p_cr_final, \
                result, helper_data, registered_variables


    def extract_fields(helper_data, config, state0, final_state):

        rho = state0[registered_variables.density_index]
        u_r = state0[registered_variables.velocity_index]
        p = state0[registered_variables.pressure_index]
        n_cr = state0[registered_variables.cosmic_ray_n_index]
        p_cr = n_cr ** gamma_cr
        if config.cosmic_ray_config.cosmic_rays:
            p_gas = p - p_cr
        else:
            p_gas = p

        rho_final = final_state[registered_variables.density_index]
        u_final = final_state[registered_variables.velocity_index]
        p_final = final_state[registered_variables.pressure_index]
        n_cr_final = final_state[registered_variables.cosmic_ray_n_index]
        p_cr_final = n_cr_final ** gamma_cr
        if config.cosmic_ray_config.cosmic_rays:
            p_gas_final = p_final - p_cr_final
        else:
            p_gas_final = p_final
        
        # convert back to physical quantities
        r = (helper_data.geometric_centers * code_units.code_length).to(u.pc)
        rho = (rho * code_units.code_density).to(m_p / u.cm**3)
        rho_final = (rho_final * code_units.code_density).to(m_p / u.cm**3)
        u_r = (u_r * code_units.code_velocity).to(u.km / u.s)
        u_final = (u_final * code_units.code_velocity).to(u.km / u.s)
        p = (p_gas_final * code_units.code_pressure / c.k_B).to(u.K / u.cm**3)
        p_cr = (p_cr * code_units.code_pressure / c.k_B).to(u.K / u.cm**3)
        p_cr_final = (p_cr_final * code_units.code_pressure / c.k_B).to(u.K / u.cm**3)

        return r, rho, rho_final, u_r, u_final, p_gas, p_gas_final, p_final, p_cr, p_cr_final

    # set units to SNe in ISM for galaxy sims

    code_length = 3 * u.parsec
    code_mass = 1e-3 * u.M_sun
    code_velocity = 1 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)
    print(code_units.code_length.to(u.pc))

    fig, ax = plt.subplots()

    #########################
    # INFO FOR LEONARD
    # HERE THE CODE BREAKS
    # I testes several starting times for when the shock injections starts
    # I also tested several resolutions
    # here are working numbers
    # t0 in 5000, 10000
    # res_pc in 0.05, 0.1
    # it breaks for res_pc >= 0.2

    for t0, ls in zip([5000, 10000], ["-", ":"]):
        for res_pc, lc in zip([0.05, 0.1], ["C0", "C1"]):
            box_size = 10
            Ncell    = box_size * code_length.to(u.pc).value / res_pc
            print("Ncell: ", Ncell)
            label = str(res_pc)+" pc"+" t0="+str(t0/1000)+" kyr"
            config, r, rho, rho_final, u_r, u_final, p_gas, p_gas_final, p_final, p_cr, p_cr_final, res, helper_data, registered_variables = \
                _run_sim(code_units, Ncell=int(Ncell), tend_yr=50000, DSA_t0_yr=t0, box_size=box_size, Ninj=20, debug=True)
            
            res_states = res.states
            #print(res)
            
            if True:
                Ecr   = np.zeros(len(res.states))
                times = res.time_points * code_units.code_time.to(u.kyr)
                for i in range(len(res_states)):
                    Vol = np.float64(helper_data.cell_volumes) * np.float64(code_units.code_length.to(u.cm)**3)
                    Pcr = np.float64(res_states[i][registered_variables.cosmic_ray_n_index]**gamma_cr * code_units.code_pressure.to(u.erg * u.cm**-3))
                    Ecr[i] = np.sum(Pcr*Vol)
                    #print(res.time_points[i] * code_units.code_time.to(u.kyr), Ecr)
                ax.plot(times, Ecr, label=label, ls=ls, color=lc)
        ax.set_xlabel("time (kyr)")
        ax.set_ylabel("total CR energy")
        ax.legend()

    plt.savefig("figures/philipp_cr_energy_injection.svg")

    def _plot_shock(helper_data, state):
        shock_index = jnp.argmax(shock_sensor(state[registered_variables.pressure_index]))
        
        shock_index_right, shock_index_left, shock_index_right = find_shock_zone(state, config, registered_variables, helper_data)
        
        print(shock_index_right, shock_index_left)
        
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
        ax1.plot(helper_data.geometric_centers[shock_index - 10:shock_index + 10]*code_length, \
                state[registered_variables.pressure_index, shock_index - 10:shock_index + 10], "o--")
        # mark vertical line at the shock
        ax2 = ax1.twinx()
        ax2.plot(helper_data.geometric_centers[shock_index - 10:shock_index + 10]*code_length, \
                shock_sensor(state[registered_variables.pressure_index])[shock_index - 10:shock_index + 10], "o--", color='green')
        ax2.set_ylabel('Shock Sensor')
        ax1.axvline(helper_data.geometric_centers[shock_index]*code_length, color='red', label='cell with max shock sensor')
        ax1.axvline(helper_data.geometric_centers[shock_index_right]*code_length, color='blue', label='right shock boundary')
        ax1.axvline(helper_data.geometric_centers[shock_index_left]*code_length, color='blue', label='left shock boundary')
        ax1.set_ylabel('Pressure')
        ax1.set_xlabel('Radius')
        ax1.legend(loc = 'upper right')

        plt.savefig("figures/philipp_debug.png")


    def _plot_state(helper_data, config, state0, state1, include_shock=False):

        
        r, rho, rho_final, u_r, u_final, p_gas, p_gas_final, p_final, p_cr, p_cr_final = extract_fields(helper_data, config, state0, state1)
        fig, axs = plt.subplots(2, 3, figsize=(15, 5))
        axs[0, 0].plot(r, rho, label='initial',ls=ls,color="C0")
        axs[1, 0].plot(r, rho_final, label='final',ls=ls,color="C0")
        axs[0, 1].plot(r, u_r, label='initial',ls=ls,color="C0")
        axs[1, 1].plot(r, u_final, label='final',ls=ls,color="C0")
        axs[0, 2].plot(r, p_gas, label='initial, gas',ls=ls,color="C0")
        axs[1, 2].plot(r, p_gas_final, label='final, gas',ls=ls,color="C0")
        
        if include_shock:
            shock_index = jnp.argmax(shock_sensor(state1[registered_variables.pressure_index]))
            shock_index_right, shock_index_left, shock_index_right = find_shock_zone(state1, config, registered_variables, helper_data)
            #print(shock_index_right, shock_index_left)
            axs[1,2].axvline(helper_data.geometric_centers[shock_index]*code_length,       color='red',  label='cell with max shock sensor')
            axs[1,2].axvline(helper_data.geometric_centers[shock_index_right]*code_length, color='blue', label='right shock boundary')
            axs[1,2].axvline(helper_data.geometric_centers[shock_index_left]*code_length,  color='blue', label='left shock boundary')

        axs[0, 0].set_ylabel('Density in $m_p / cm^3$')
        axs[1, 0].set_ylabel('Density in $m_p / cm^3$')
        axs[1, 0].set_xlabel('r in pc')
        axs[0, 0].set_title('Density')
        axs[0, 0].legend()
        axs[1, 0].legend()
        
        axs[0, 1].set_ylabel('Velocity in km/s')
        axs[1, 1].set_ylabel('Velocity in km/s')
        axs[1, 1].set_xlabel('r in pc')
        axs[0, 1].set_title('Velocity')
        axs[0, 1].legend()
        axs[1, 1].legend()
        
        axs[0, 2].set_ylabel('Pressure / $k_B$ in $K / cm^3$')
        axs[1, 2].set_ylabel('Pressure / $k_B$ in $K / cm^3$')
        axs[1, 2].set_xlabel('r in pc')
        axs[0, 2].set_yscale("log")
        #axs[1, 2].set_yscale("log") 
        
        axs[0, 2].set_title('Pressure')
        axs[0, 2].legend()
        axs[1, 2].legend()
        
        plt.tight_layout()

        plt.savefig("figures/philipp_sedov_crs.png")

    code_length = 3 * u.parsec
    code_mass = 1e-3 * u.M_sun
    code_velocity = 1 * u.km / u.s
    code_units = CodeUnits(code_length, code_mass, code_velocity)
    print(code_units.code_length.to(u.pc))

    res_pc   = 0.1
    box_size = 10
    Ncell    = box_size * code_length.to(u.pc).value / res_pc
    print("Ncell: ", Ncell)
    t0 = 5000
    label = str(res_pc)+" pc"+" t0="+str(t0/1000)+" kyr"
    config, r, rho, rho_final, u_r, u_final, p_gas, p_gas_final, p_final, p_cr, p_cr_final, res, helper_data, registered_variables = \
        _run_sim(code_units, Ncell=int(Ncell), tend_yr=50000, DSA_t0_yr=t0, box_size=box_size, Ninj=20, debug=False)

    fig, ax = plt.subplots()
    _plot_shock(helper_data, res.states[16])
    # detect shock fronts

    for i in [16]:
        _plot_state(helper_data, config, res.states[0], res.states[i], include_shock=True)