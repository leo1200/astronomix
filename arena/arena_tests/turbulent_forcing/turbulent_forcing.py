"""
# 3D MHD Blast Wave Test Problem

Following Seo & Ryu (2023), Sec. 3.7.

## Test setup

The initial conditions are given by

density = 1
velocity = (0, 0, 0)
magnetic field = (B0/sqrt(2), B0/sqrt(2), 0)
pressure = 100 for r <= r0
         = 1 + 99 * (r1 - r) / (r1 - r0) for r0 < r <= r1
         = 1 for r > r1

with r0 = 0.125, r1 = 1.1 * r0, and B0 = 10. r is
the radial distance from the center of the box.

The simulation is run until t = 0.02 in a periodic box 
of size 1.0.

## Literature reference

Seo, Jeongbhin, and Dongsu Ryu. 
"HOW-MHD: a high-order WENO-based 
magnetohydrodynamic code with a 
high-order constrained transport 
algorithm for astrophysical 
applications." 
The Astrophysical Journal
953.1 (2023): 39.
https://arxiv.org/pdf/2304.04360
"""

import os

import jax

# basic numerics
import jax.numpy as jnp

# plotting
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# astronomixSimulationConfig
from astronomix import (
    # astronomix data structures
    SimulationConfig,
    SimulationParams,
    # setup functions
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    initialize_interface_fields,
    # time integration
    time_integration
)

from astronomix._physics_modules._turbulent_forcing._turbulent_forcing import _apply_forcing
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams
from astronomix.initial_condition_generation.turbulent_ic_generator import create_turb_field
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY, 
    BoundarySettings, 
    BoundarySettings1D,
    SnapshotSettings
)

def turbulent_forcing_test(
    config: SimulationConfig,
    params: SimulationParams,
    configuration_name: str
):
    
    # # start profiling
    # jax.profiler.start_trace("/tmp/profile-data")

    test_name = f"turbulent_forcing_{config.num_cells}cells"

    print("👷 setting up turbulent_forcing with configuration: ", configuration_name)

    num_cells = config.num_cells

    # adapt the params for the 
    # correct end time
    params = params._replace(
        t_end=3.0,
        C_cfl=0.8,
        turbulent_forcing_params = TurbulentForcingParams(
            energy_injection_rate = 2.0
        ),
        dt_max = 0.1,
    )

    # set periodic boundaries in all directions
    print("Setting periodic boundaries in all directions.")
    config = config._replace(
        return_snapshots=False,
        turbulent_forcing_config = TurbulentForcingConfig(
            turbulent_forcing = True,
        ),
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # setup the initial conditions

    r = helper_data.r

    rho = jnp.ones_like(r) * 0.5
    P = jnp.ones_like(r) * 0.1

    # speed of sound
    c_s = jnp.sqrt(params.gamma * P / rho)
    print("sound speed:", c_s.mean())

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    # turbulence_slope = -2.0
    # kmin = 2.0
    # kmax = int(0.8 * num_cells / 2)
    # key = jax.random.PRNGKey(42)
    # key, sk1, sk2, sk3 = jax.random.split(key, 4)
    # V_x = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk1)
    # V_y = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk2)
    # V_z = create_turb_field(config.num_cells, 1, turbulence_slope, kmin, kmax, key=sk3)
    # rms = jnp.sqrt(jnp.mean(V_x**2 + V_y**2 + V_z**2))
    # rms_wanted = 5.0
    # V_x = V_x * (rms_wanted / rms)
    # V_y = V_y * (rms_wanted / rms)
    # V_z = V_z * (rms_wanted / rms)

    B0 = 0.0

    B_x = B0 / jnp.sqrt(2)
    B_y = B0 / jnp.sqrt(2)
    B_z = 0.0

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

    if config.solver_mode == FINITE_DIFFERENCE:
        bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)
    else:
        bxb, byb, bzb = None, None, None

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)

    # run the simulation
    print("🚀 running turbulent_forcing simulation")
    final_state = time_integration(
        initial_state, config, params, registered_variables
    )

    # def _apply_forcing(
    # key,
    # primitive_state,
    # dt,
    # turbulent_forcing_params: TurbulentForcingParams,
    # config: SimulationConfig,
    # registered_variables: RegisteredVariables,
    # )

    # key, final_state = _apply_forcing(
    #     jax.random.key(42),
    #     initial_state,
    #     0.01,
    #     params.turbulent_forcing_params,
    #     config,
    #     registered_variables,
    # )

    # store the simulation result

    # create a folder configuration_name in results/
    # and a data folder within
    output_folder = os.path.join("results", configuration_name, "data")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, test_name + ".npz")
    jnp.savez(output_file, final_state=final_state)

    # plot the results
    fig, ax = plt.subplots(figsize=(6, 6))

    # density
    v_squared = final_state[registered_variables.velocity_index.x]**2 + \
                final_state[registered_variables.velocity_index.y]**2 + \
                final_state[registered_variables.velocity_index.z]**2

    rho = final_state[registered_variables.density_index]

    E_K = rho # * v_squared / 2.0
    

    # print the rms velocity
    v_rms = jnp.sqrt(jnp.mean(v_squared))
    print(f"RMS velocity: {v_rms:.4f}")

    im = ax.imshow(
        E_K[:, :, num_cells // 2],
        origin="lower",
        extent=(0, config.box_size, 0, config.box_size),
        norm=LogNorm()
    )
    cbar = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cbar, label="kinetic energy density")
    ax.set_title("kinetic energy density slice")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # create the results/configuration_name/figures
    # folder if it does not exist
    figures_folder = os.path.join("results", configuration_name, "figures")
    os.makedirs(figures_folder, exist_ok=True)
    figure_file = os.path.join(figures_folder, test_name + ".png")
    plt.savefig(figure_file)
    plt.close(fig)