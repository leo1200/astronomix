"""
Cost of a backward pass through the finite-difference scheme in 3D.

The checkpointing benchmark (``checkpoint_scaling.py``) uses a 1D linear
acoustic eigenmode. This script measures the corresponding 3D numbers quoted in
the paper: a sinusoidal density perturbation (amplitude 1e-6, two wavelengths
per axis) on a 64^3 periodic grid in single precision, WENO-Z, C_cfl = 0.4,
evolved to t = 1.5 with c_s = 2. It prints the forward runtime on the native-JAX
and on the Pallas backend and, for the Pallas backend with 10 binomial
checkpoints, the runtime and the compiled memory of the gradient of
J = 1/2 sum(rho_P^2 + |v|^2) dV with respect to the initial perturbations.

    sbatch --export=ALL,RUN_DIR=examples/scripts/scaling \\
        pytests/runners/gpu1.sh python backward_pass_3d.py
"""

# general
import argparse
from timeit import default_timer as timer

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    BACKWARDS,
    CARTESIAN,
    FINITE_DIFFERENCE,
    FORWARDS,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

BACKGROUND_DENSITY = 1.0
SOUND_SPEED = 2.0
GAMMA = 5.0 / 3.0
AMPLITUDE = 1.0e-6
BOX_LENGTH = 1.0
END_TIME = 1.5
REPETITIONS = 3


def configuration(num_cells, backend, differentiation_mode, num_checkpoints=1):
    """The configuration and parameters of the 3D acoustic test."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        weno_z=True,
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=BOX_LENGTH,
        num_cells=num_cells,
        boundary_settings=BoundarySettings(x=periodic, y=periodic, z=periodic),
        backend_config=BackendConfig(backend=backend),
        differentiation_mode=differentiation_mode,
        num_checkpoints=num_checkpoints,
        progress_bar=False,
    )
    params = SimulationParams(C_cfl=0.4, t_end=END_TIME, gamma=GAMMA)
    return config, params


def initial_state(density_perturbation, velocity_perturbation, config, registered_variables, helper_data):
    """The primitive state of the perturbed uniform background."""
    background_pressure = SOUND_SPEED**2 * BACKGROUND_DENSITY / GAMMA
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=BACKGROUND_DENSITY + density_perturbation,
        velocity_x=velocity_perturbation,
        velocity_y=jnp.zeros_like(velocity_perturbation),
        velocity_z=jnp.zeros_like(velocity_perturbation),
        gas_pressure=background_pressure + SOUND_SPEED**2 * density_perturbation,
    )


def perturbation(helper_data):
    """The initial density perturbation (two wavelengths along every axis) and zero velocity."""
    centers = helper_data.geometric_centers
    wavenumber = 2.0 * jnp.pi * 2.0 / BOX_LENGTH
    density = AMPLITUDE * jnp.sin(wavenumber * (centers[..., 0] + centers[..., 1] + centers[..., 2]))
    return density, jnp.zeros_like(density)


def timed(function, arguments):
    """The mean runtime of a compiled call after one warm-up call."""
    jax.block_until_ready(function(*arguments))
    times = []
    for _ in range(REPETITIONS):
        start = timer()
        jax.block_until_ready(function(*arguments))
        times.append(timer() - start)
    return sum(times) / len(times)


def main():
    """Measure the forward runtimes and the cost of the gradient."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--checkpoints", type=int, default=10)
    arguments = parser.parse_args()
    num_cells = arguments.resolution

    # -------------------------------------------------------------
    # =============== ↓ Forward runtime per backend ↓ =============
    # -------------------------------------------------------------
    for backend, name in ((NATIVE_JAX, "native JAX"), (PALLAS, "Pallas")):
        jax.clear_caches()
        config, params = configuration(num_cells, backend, FORWARDS)
        # The snapshot result carries the number of steps.
        config = config._replace(return_snapshots=True, num_snapshots=1)
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)
        density, velocity = perturbation(helper_data)
        state = initial_state(density, velocity, config, registered_variables, helper_data)
        config = finalize_config(config, state.shape)
        forward = jax.jit(lambda state: time_integration(state, config, params, registered_variables))
        steps = int(forward(state).num_iterations)
        print(f"forward, {name}: {steps} steps, {timed(forward, (state,)):.3f} s", flush=True)
    # -------------------------------------------------------------
    # =============== ↑ Forward runtime per backend ↑ =============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Gradient (Pallas backend) ↓ =============
    # -------------------------------------------------------------
    jax.clear_caches()
    config, params = configuration(num_cells, PALLAS, BACKWARDS, arguments.checkpoints)
    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    density, velocity = perturbation(helper_data)
    config = finalize_config(
        config,
        initial_state(density, velocity, config, registered_variables, helper_data).shape,
    )
    cell_volume = (BOX_LENGTH / num_cells) ** 3

    def cost(density_perturbation, velocity_perturbation):
        state = initial_state(density_perturbation, velocity_perturbation, config, registered_variables, helper_data)
        final_state = time_integration(state, config, params, registered_variables)
        final_density_perturbation = final_state[registered_variables.density_index] - BACKGROUND_DENSITY
        final_speed_squared = sum(
            final_state[index] ** 2
            for index in (
                registered_variables.velocity_index.x,
                registered_variables.velocity_index.y,
                registered_variables.velocity_index.z,
            )
        )
        return 0.5 * jnp.sum(final_density_perturbation**2 + final_speed_squared) * cell_volume

    gradient = jax.jit(jax.grad(cost, argnums=(0, 1)))
    compiled = gradient.lower(density, velocity).compile()
    memory = compiled.memory_analysis()
    total_bytes = (
        memory.temp_size_in_bytes
        + memory.argument_size_in_bytes
        + memory.output_size_in_bytes
        - memory.alias_size_in_bytes
    )
    print(
        f"gradient, Pallas forward / native tangent, {arguments.checkpoints} checkpoints: "
        f"{timed(compiled, (density, velocity)):.2f} s, compiled memory {total_bytes / 1e9:.2f} GB",
        flush=True,
    )
    # -------------------------------------------------------------
    # ================= ↑ Gradient (Pallas backend) ↑ =============
    # -------------------------------------------------------------


if __name__ == "__main__":
    main()
