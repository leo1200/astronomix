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

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY, 
    BoundarySettings, 
    BoundarySettings1D,
    SnapshotSettings
)

def mhd_blast_test1(
    config: SimulationConfig,
    params: SimulationParams,
    configuration_name: str
):
    
    # # start profiling
    # jax.profiler.start_trace("/tmp/profile-data")

    test_name = f"mhd_blast_test1_{config.num_cells}cells"

    print("👷 setting up mhd_blast_test1 with configuration: ", configuration_name)

    num_cells = config.num_cells

    # adapt the params for the 
    # correct end time
    params = params._replace(
        t_end=0.02,
    )

    # set periodic boundaries in all directions
    print("Setting periodic boundaries in all directions.")
    config = config._replace(
        return_snapshots=True,
        num_snapshots=40,
        snapshot_settings=SnapshotSettings(
            return_states=False,
            return_final_state=True,
            return_magnetic_divergence=True
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
    r0 = 0.125
    r1 = 1.1 * r0

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 1.0
    P = jnp.where(r <= r0, 100.0, P)
    P = jnp.where((r > r0) & (r <= r1), 1.0 + 99.0 * (r1 - r) / (r1 - r0), P)
    P = jnp.where(r > r1, 1.0, P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B0 = 10

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
    print("🚀 running mhd_blast_test1 simulation")
    result = time_integration(
        initial_state, config, params, registered_variables
    )
    final_state = result.final_state
    magnetic_divergence = result.magnetic_divergence
    time_points = result.time_points

    # store the simulation result

    # create a folder configuration_name in results/
    # and a data folder within
    output_folder = os.path.join("results", configuration_name, "data")
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, test_name + ".npz")
    jnp.savez(output_file, final_state=final_state, magnetic_divergence=magnetic_divergence)

    # plot the results
    density = final_state[registered_variables.density_index]
    pressure = final_state[registered_variables.pressure_index]
    Bx = final_state[registered_variables.magnetic_index.x]
    By = final_state[registered_variables.magnetic_index.y]
    Bz = final_state[registered_variables.magnetic_index.z]
    vx = final_state[registered_variables.velocity_index.x]
    vy = final_state[registered_variables.velocity_index.y]
    vz = final_state[registered_variables.velocity_index.z]
    b_squared = Bx**2 + By**2 + Bz**2
    v_squared = vx**2 + vy**2 + vz**2

    fig, axs = plt.subplots(2, 3, figsize=(9, 6))

    # density
    im = axs[0, 0].imshow(
        density[:, :, num_cells // 2],
        origin="lower",
        extent=(0, config.box_size, 0, config.box_size),
        cmap="jet",
    )
    cbar = make_axes_locatable(axs[0, 0]).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cbar, label="density")
    axs[0, 0].set_title("density slice")
    axs[0, 0].set_xlabel("x")
    axs[0, 0].set_ylabel("y")

    # pressure
    im = axs[1, 1].imshow(
        pressure[:, :, num_cells // 2],
        origin="lower",
        extent=(0, config.box_size, 0, config.box_size),
        cmap="jet",
    )
    cbar = make_axes_locatable(axs[1, 1]).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cbar, label="pressure")
    axs[1, 1].set_title("pressure slice")
    axs[1, 1].set_xlabel("x")
    axs[1, 1].set_ylabel("y")

    im = axs[0, 1].imshow(
        v_squared[:, :, num_cells // 2],
        origin="lower",
        extent=(0, config.box_size, 0, config.box_size),
        cmap="jet",
    )
    cbar = make_axes_locatable(axs[0, 1]).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cbar, label="v²")
    axs[0, 1].set_title("kinetic energy slice")
    axs[0, 1].set_xlabel("x")
    axs[0, 1].set_ylabel("y")

    im = axs[1, 0].imshow(
        b_squared[:, :, num_cells // 2],
        origin="lower",
        extent=(0, config.box_size, 0, config.box_size),
        cmap="jet",
    )
    cbar = make_axes_locatable(axs[1, 0]).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cbar, label="B²")
    axs[1, 0].set_title("magnetic pressure slice")
    axs[1, 0].set_xlabel("x")
    axs[1, 0].set_ylabel("y")

    # 0, 2: |B|^2 / 2 along the diagonal from the center
    diag_indices = jnp.arange(0, num_cells)
    B_diag = b_squared[diag_indices, diag_indices, num_cells // 2]
    r_diag = jnp.sqrt((diag_indices) ** 2 + (diag_indices) ** 2) * (
        config.box_size / num_cells
    )
    axs[0, 2].plot(r_diag, B_diag)
    axs[0, 2].set_ylabel("|B|²")
    axs[0, 2].set_xlabel("diagonal")
    axs[0, 2].set_title("|B|² along diagonal")

    # density along the vertical centerline
    pressure_diag = pressure[diag_indices, diag_indices, num_cells // 2]
    axs[1, 2].plot(r_diag, pressure_diag)
    axs[1, 2].set_ylabel("pressure")
    axs[1, 2].set_xlabel("diagonal")
    axs[1, 2].set_title("pressure along diagonal")

    plt.tight_layout()

    # create the results/configuration_name/figures
    # folder if it does not exist
    figures_folder = os.path.join("results", configuration_name, "figures")
    os.makedirs(figures_folder, exist_ok=True)
    figure_file = os.path.join(figures_folder, test_name + ".png")
    plt.savefig(figure_file, dpi=800)
    plt.close(fig)

    # plot the magnetic divergence over time
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(time_points, magnetic_divergence / (B0 / config.grid_spacing))
    ax.set_xlabel("time")
    ax.set_ylabel("max |∇·B| / (B0 / Δx)")
    ax.set_title("Magnetic Divergence Over Time")
    plt.tight_layout()
    figure_file = os.path.join(figures_folder, test_name + "_divergence.svg")
    plt.savefig(figure_file)
    plt.close(fig)

    print(f"Results saved in {output_folder} and {figures_folder}.")

    # jax.profiler.stop_trace()