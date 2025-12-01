"""
# 3D Scaling Test
"""

import os

from timeit import default_timer as timer

import jax

# basic numerics
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from jax.sharding import PartitionSpec as P

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

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS, 
    BoundarySettings, 
    BoundarySettings1D,
    SnapshotSettings
)
from astronomix.time_stepping.time_integration import _time_integration

def scaling_test(
    config: SimulationConfig,
    params: SimulationParams,
    resolutions: list[int],
    configuration_name: str,
    multi_gpu: bool = False,
):
    
    for num_cells in resolutions:

        test_name = f"scaling_{num_cells}_{'multiGPU' if multi_gpu else 'singleGPU'}"

        print("👷 setting up scaling test with configuration: ", configuration_name)

        # adapt the params for the 
        # correct end time
        params = params._replace(
            t_end=0.02,
        )

        # set periodic boundaries in all directions
        print("Setting periodic boundaries in all directions.")
        config = config._replace(
            num_cells=num_cells,
            return_snapshots=True,
            num_snapshots=5,
            snapshot_settings=SnapshotSettings(
                return_states=False,
                return_final_state=True,
            ),
            boundary_handling = PERIODIC_ROLL,
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

        if multi_gpu:
            # mesh with variable axis
            split = (1, 2, 2, 1)
            sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
            named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

        if multi_gpu:
            helper_data = get_helper_data(config, sharding=named_sharding)
        else:
            helper_data = get_helper_data(config)
        
        registered_variables = get_registered_variables(config)

        # setup the initial conditions

        r = helper_data.r
        r0 = 0.125
        r1 = 1.1 * r0

        rho = jnp.ones_like(r)
        p = jnp.ones_like(r) * 1.0
        p = jnp.where(r <= r0, 100.0, p)
        p = jnp.where((r > r0) & (r <= r1), 1.0 + 99.0 * (r1 - r) / (r1 - r0), p)
        p = jnp.where(r > r1, 1.0, p)

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

        helper_data = HelperData()

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
            gas_pressure=p,
        )

        # set all separate fields to None
        bxb, byb, bzb = None, None, None
        V_x, V_y, V_z = None, None, None
        B_x, B_y, B_z = None, None, None
        rho = None
        p = None
        r = None

        # the initial state should already be on multiple gpus
        # if multi_gpu, but to be sure
        if multi_gpu:
            initial_state = jax.device_put(initial_state, named_sharding)

        config = finalize_config(config, initial_state.shape)

        if config.donate_state:
            time_integration_jit = jax.jit(
                _time_integration,
                static_argnames=[
                    "config",
                    "registered_variables",
                    "snapshot_callable"
                ],
                donate_argnames=["state"],
            )
        else:
            time_integration_jit = jax.jit(
                _time_integration,
                static_argnames=[
                    "config",
                    "registered_variables",
                    "snapshot_callable"
                ],
            )

        # compile the time integration step and print memory usage
        compiled_step = time_integration_jit.lower(
            initial_state,
            config,
            params,
            registered_variables,
            helper_data,
            helper_data,
        ).compile()
        compiled_stats = compiled_step.memory_analysis()

        assert compiled_stats is not None
        
        # Calculate total memory usage including temporary storage,
        # arguments, and outputs (but excluding aliases)
        total = (
            compiled_stats.temp_size_in_bytes
            + compiled_stats.argument_size_in_bytes
            + compiled_stats.output_size_in_bytes
            - compiled_stats.alias_size_in_bytes
        )
        print("=== Compiled memory usage PER DEVICE ===")
        print(
            f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB"
        )
        print(
            f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB"
        )
        print(f"Total size: {total / (1024**2):.2f} MB")
        print("========================================")

        in_size_mb = (
            compiled_stats.argument_size_in_bytes / (1024**2)
        )
        temp_size_mb = (
            compiled_stats.temp_size_in_bytes / (1024**2)
        )
        total_size_mb = total / (1024**2)
        # ==========================================================

        # start timer
        start_time = timer()

        result = time_integration_jit(
            initial_state,
            config,
            params,
            registered_variables,
            helper_data,
            helper_data,
        )

        final_state = result.final_state.block_until_ready()
        num_iterations = result.num_iterations

        end_time = timer()
        total_time = end_time - start_time

        print(
            f"🏁 completed simulation with {num_cells} cells in {total_time:.2f} seconds, "
            f"averaging {total_time / num_iterations:.4f} seconds per time step over {num_iterations} iterations."
        )
        
        # store the simulation result
        # create a folder configuration_name in results/
        # and a data folder within
        output_folder = os.path.join("results", configuration_name, "data")
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, test_name + ".npz")
        jnp.savez(
            output_file,
            final_state=final_state,
            num_iterations=num_iterations,
            total_time=total_time,
            in_size_mb=in_size_mb,
            temp_size_mb=temp_size_mb,
            total_size_mb=total_size_mb,
        )
    
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
        fig.colorbar(im, cax=cbar, label="v^2")
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
        fig.colorbar(im, cax=cbar, label="B^2")
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
        axs[0, 2].set_ylabel("|B|^2")
        axs[0, 2].set_xlabel("diagonal")
        axs[0, 2].set_title("|B|^2 along diagonal")

        # density along the vertical centerline
        pressure_diag = pressure[diag_indices, diag_indices, num_cells // 2]
        axs[1, 2].plot(r_diag, pressure_diag)
        axs[1, 2].set_ylabel("pressure")
        axs[1, 2].set_xlabel("diagonal")
        axs[1, 2].set_title("Pressure along diagonal")

        plt.tight_layout()

        # create the results/configuration_name/figures
        # folder if it does not exist
        figures_folder = os.path.join("results", configuration_name, "figures")
        os.makedirs(figures_folder, exist_ok=True)
        figure_file = os.path.join(figures_folder, test_name + ".png")
        plt.savefig(figure_file)
        plt.close(fig)
