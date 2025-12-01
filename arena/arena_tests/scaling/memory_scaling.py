"""
# Static memory scaling test.
"""

import os

# basic numerics
import jax
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
)

from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY, 
    BoundarySettings, 
    BoundarySettings1D
)

from astronomix.time_stepping.time_integration import _time_integration

def memory_scaling(
    config: SimulationConfig,
    params: SimulationParams,
    resolutions: list[int],
    configuration_name: str
):

    test_name = "memory_scaling"

    print("👷 testing memory_scaling with configuration: ", configuration_name)

    input_memory = []
    temp_memory = []
    total_memory = []

    # set periodic boundaries in all directions
    print("Setting periodic boundaries in all directions.")
    config_run = config._replace(
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

    for num_cells in resolutions:

        print(f"--- Testing resolution: {num_cells} cells per dimension ---")

        # adapt the config
        config_run = config_run._replace(
            num_cells=num_cells,
        )

        helper_data = get_helper_data(config_run)
        registered_variables = get_registered_variables(config_run)

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

        if config.mhd:
            B0 = 10

            B_x = B0 / jnp.sqrt(2)
            B_y = B0 / jnp.sqrt(2)
            B_z = 0.0

            B_x = jnp.ones_like(r) * B_x
            B_y = jnp.ones_like(r) * B_y
            B_z = jnp.ones_like(r) * B_z
        else:
            B_x = None
            B_y = None
            B_z = None

        if config.solver_mode == FINITE_DIFFERENCE:
            bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)
        else:
            bxb, byb, bzb = None, None, None

        # we will use empty helper data here
        helper_data = HelperData()

        initial_state = construct_primitive_state(
            config=config_run,
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

        # we set all variables used for the construction of
        # the initial state to None to free up memory
        # not sure if that's necessary
        B_x = None; B_y = None; B_z = None
        bxb = None; byb = None; bzb = None
        V_x = None; V_y = None; V_z = None
        rho = None; P = None; r = None

        config_run = finalize_config(config_run, initial_state.shape)

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

        compiled_step = time_integration_jit.lower(
            initial_state,
            config_run,
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

        input_memory.append(compiled_stats.argument_size_in_bytes / (1024**2))
        temp_memory.append(compiled_stats.temp_size_in_bytes / (1024**2))
        total_memory.append(total / (1024**2))

    # save results
    results_dir = f"results/{configuration_name}/data"
    os.makedirs(results_dir, exist_ok=True)
    npz_path = os.path.join(results_dir, test_name + ".npz")
    jnp.savez(
        npz_path,
        resolutions=jnp.array(resolutions),
        input_memory=jnp.array(input_memory),
        temp_memory=jnp.array(temp_memory),
        total_memory=jnp.array(total_memory),
    )
    print(f"💾 saved memory scaling results to {npz_path}")

    # plot results
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        resolutions,
        input_memory,
        marker="o",
        label="input memory",
    )
    ax.plot(
        resolutions,
        temp_memory,
        marker="o",
        label="temporary memory",
    )
    ax.plot(
        resolutions,
        total_memory,
        marker="o",
        label="total memory",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of cells per dimension")
    ax.set_ylabel("memory usage in MB")
    ax.set_title("memory scaling test")
    ax.legend()

    fig_dir = f"results/{configuration_name}/figures"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, test_name + ".svg")
    fig.savefig(fig_path)

    # also plot temp_memory / input_memory over resolution
    fig, ax = plt.subplots(figsize=(8, 6))
    ratio = jnp.array(temp_memory) / jnp.array(input_memory)
    ax.plot(
        resolutions,
        ratio,
        marker="o",
        label="temp / input memory ratio",
    )
    ax.set_xlabel("number of cells per dimension")
    ax.set_ylabel("temp / input memory ratio")
    ax.set_title("memory scaling test")
    ax.legend()
    fig_path = os.path.join(fig_dir, test_name + "_ratio.svg")
    fig.savefig(fig_path)
    print(f"💾 saved memory scaling figure to {fig_path}")
