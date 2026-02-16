# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=2)
# =======================

import jax

from astronomix.option_classes.simulation_config import (
    FORWARDS,
    HLL,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
)

from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import finalize_config


import timeit
import jax.numpy as jnp

from jax.sharding import PartitionSpec as P, NamedSharding

import matplotlib.pyplot as plt


def setup_ics(num_cells, num_injection_cells=2):
    # setup simulation config
    config = SimulationConfig(
        runtime_debugging=False,
        memory_analysis=True,
        progress_bar=True,
        dimensionality=3,
        box_size=1.0,
        num_cells=num_cells,
        differentiation_mode=FORWARDS,
        riemann_solver=HLL,
    )

    params = SimulationParams(t_end=0.3, C_cfl=0.4, dt_max=0.2)

    registered_variables = get_registered_variables(config)

    rho = jnp.ones((num_cells, num_cells, num_cells)) * 0.125
    u_x = jnp.zeros((num_cells, num_cells, num_cells))
    u_y = jnp.zeros((num_cells, num_cells, num_cells))
    u_z = jnp.zeros((num_cells, num_cells, num_cells))
    p = jnp.ones((num_cells, num_cells, num_cells)) * 0.1

    center = num_cells // 2

    injection_slice = slice(center - num_injection_cells, center + num_injection_cells)

    rho = rho.at[injection_slice, injection_slice, injection_slice].set(1.0)
    p = p.at[injection_slice, injection_slice, injection_slice].set(1.0)

    primitive_state = jnp.stack([rho, u_x, u_y, u_z, p], axis=0)

    config = finalize_config(config, primitive_state.shape)

    return primitive_state, config, params, registered_variables


def measure_execution_time(
    primitive_state, config, params, helper_data, registered_variables, sharding=None
):
    # execute once for compilation and warmup

    final_state = time_integration(
        primitive_state,
        config,
        params,
        registered_variables,
        sharding=sharding,
    )
    final_state.block_until_ready()

    plot_results(final_state, registered_variables)

    # Create a function for timing
    def timed_execution():
        final_state = time_integration(
            primitive_state,
            config,
            params,
            registered_variables,
            sharding=sharding,
        )
        final_state.block_until_ready()  # Ensure execution completes before timing stops

    # Measure execution time
    times = timeit.repeat(
        timed_execution,
        repeat=1,  # More repeats for better statistics
        number=1,  # Number of calls per measurement
    )

    # Calculate statistics
    min_time = min(times)
    mean_time = sum(times) / len(times)

    print(f"Execution time: min={min_time:.6f}s, mean={mean_time:.6f}s")
    return min_time  # Minimum time is often most representative for benchmarking


def plot_results(final_state, registered_variables):
    num_cells = final_state.shape[1]
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(
        final_state[registered_variables.density_index, :, :, num_cells // 2],
        extent=[0, 1, 0, 1],
    )
    axs[0].set_title("Density")
    axs[1].imshow(
        final_state[registered_variables.pressure_index, :, :, num_cells // 2],
        extent=[0, 1, 0, 1],
    )
    axs[1].set_title("Pressure")
    plt.savefig("check_{:d}.png".format(num_cells))


def make_scaling_plots(sharding=False, num_cells_list=[32, 64, 128, 256, 512, 1024]):
    execution_times = []

    for num_cells in num_cells_list:
        primitive_state, config, params, registered_variables = setup_ics(
            num_cells, num_cells // 16
        )

        if sharding:
            split = (1, 2, 1, 1)
            sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
            named_sharding = jax.NamedSharding(
                sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS)
            )
            primitive_state = jax.device_put(primitive_state, named_sharding)

            jax.debug.visualize_array_sharding(primitive_state[0, :, :, 0])

            helper_data = get_helper_data(config, named_sharding)
        else:
            helper_data = get_helper_data(config)
            named_sharding = None

        execution_time = measure_execution_time(
            primitive_state,
            config,
            params,
            helper_data,
            registered_variables,
            sharding=named_sharding,
        )
        execution_times.append(execution_time)

    fig, axs = plt.subplots(1, 1, figsize=(12, 6))
    axs.plot(num_cells_list, execution_times, "o-", label="execution time")

    # set x and y log scale
    axs.set_xscale("log", base=2)
    axs.set_yscale("log", base=2)
    axs.legend()

    axs.set_title("Execution time")
    axs.set_xlabel("Number of cells")
    axs.set_ylabel("Execution time (s)")

    num_cells_list = jnp.array(num_cells_list, dtype=jnp.float32)
    execution_times = jnp.array(execution_times, dtype=jnp.float32)

    file_appendix = ""

    if sharding:
        file_appendix = "_sharding"

    # save the data for later analysis
    jnp.savez(
        "results/scaling_data{}.npz".format(file_appendix),
        num_cells_list=num_cells_list,
        execution_times=execution_times,
    )

    # save the figure
    plt.savefig("scaling{}.png".format(file_appendix))


def plot_scaling_results(plot_sharded=True):
    # load the data
    data_unsharded = jnp.load("results/scaling_data.npz")

    if plot_sharded:
        data_sharded = jnp.load("results/scaling_data_sharding.npz")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # set figure title "scaling for a 3D shock test"
    fig.suptitle("Scaling for a 3D shock test")

    axs[0].plot(
        data_unsharded["num_cells_list"],
        data_unsharded["execution_times"],
        "o-",
        label="unsharded",
    )

    if plot_sharded:
        axs[0].plot(
            data_sharded["num_cells_list"],
            data_sharded["execution_times"],
            "o-",
            label="sharded (4 GPUs)",
        )

    # set x and y log scale
    axs[0].set_xscale("log", base=2)
    axs[0].set_yscale("log", base=2)

    axs[0].legend()
    axs[0].set_title("Execution time")
    axs[0].set_xlabel("Number of cells per dimension")
    axs[0].set_ylabel("Execution time in s")

    # plot the speedup
    if plot_sharded:
        common_length = min(
            len(data_unsharded["num_cells_list"]), len(data_sharded["num_cells_list"])
        )

    if plot_sharded:
        speedup = (
            data_unsharded["execution_times"][0:common_length]
            / data_sharded["execution_times"][0:common_length]
        )

    if plot_sharded:
        axs[1].plot(
            data_unsharded["num_cells_list"][0:common_length],
            speedup,
            "o-",
            label="speedup",
        )

    # theoretical speedup is 4
    axs[1].plot(
        data_unsharded["num_cells_list"][0:common_length],
        jnp.ones_like(data_unsharded["num_cells_list"][0:common_length]) * 4,
        "--",
        label="theoretical speedup",
    )
    axs[1].legend()
    axs[1].set_title("Speedup")
    axs[1].set_xlabel("Number of cells per dimension")

    plt.tight_layout()

    plt.savefig("scaling_results.png")


make_scaling_plots(sharding=True, num_cells_list=[512])