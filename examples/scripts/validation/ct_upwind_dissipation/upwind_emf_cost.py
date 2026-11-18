"""
Cost per time step of the upwind dissipation of the constrained-transport EMFs.

Times the finite-difference MHD step of the supersonic turbulence setup
(isothermal, M_turb = 10, beta_p = 0.1, TurbGen driving, WENO-Z with the
positivity-preserving reconstruction, SSPRK4, Pallas backend, single precision)
with and without ``ct_upwind_dissipation`` at a fixed time step, and prints the
time per step of both and their ratio.

    sbatch --exclusive --export=ALL,RUN_DIR=examples/scripts/validation/ct_upwind_dissipation \\
        pytests/runners/gpu1.sh python upwind_emf_cost.py [--resolution 512]
"""

# general
import argparse
from timeit import default_timer as timer

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    PALLAS,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import ISOTHERMAL

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix import TurbulenceDrivingConfig, TurbulenceDrivingParams

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)

TURBULENT_MACH_NUMBER = 10.0
PLASMA_BETA = 0.1
NUM_STEPS = 20
REPETITIONS = 3


def time_per_step(resolution, upwind_dissipation):
    """
    The mean time per step of NUM_STEPS fixed steps after a warm-up run.

    Args:
        resolution: The number of cells per axis.
        upwind_dissipation: Switch the upwind EMF dissipation on.

    Returns:
        The time per step in seconds.
    """
    jax.clear_caches()
    periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
    sound_speed = 1.0 / TURBULENT_MACH_NUMBER
    config = SimulationConfig(
        equation_of_state=ISOTHERMAL,
        progress_bar=False,
        dimensionality=3,
        num_cells=resolution,
        box_size=1.0,
        mhd=True,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        turbulence_driving_config=TurbulenceDrivingConfig(turbulence_driving=True),
        solver_config=FDSolverConfig(
            weno_z=True,
            weno_positivity_preserving=True,
            ct_upwind_dissipation=upwind_dissipation,
        ),
        backend_config=BackendConfig(backend=PALLAS),
        fixed_timestep=True,
        num_timesteps=NUM_STEPS,
    )
    # A step of a tenth of the CFL step of the initial state at rest; the cost
    # of a step does not depend on its size.
    time_step = 0.75 / 3.0 / resolution * sound_speed
    params = SimulationParams(
        C_cfl=0.75,
        isothermal_sound_speed=sound_speed,
        t_end=NUM_STEPS * time_step,
        turbulence_driving_params=TurbulenceDrivingParams(velocity=1.0),
    )
    registered_variables = get_registered_variables(config)
    shape = (resolution, resolution, resolution)
    density = jnp.ones(shape, dtype=jnp.float32)
    zero = jnp.zeros_like(density)
    magnetic_field_z = jnp.full(shape, jnp.sqrt(2.0 * sound_speed**2 / PLASMA_BETA), dtype=jnp.float32)
    interface_x, interface_y, interface_z = initialize_interface_fields(zero, zero, magnetic_field_z)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        magnetic_field_x=zero,
        magnetic_field_y=zero,
        magnetic_field_z=magnetic_field_z,
        interface_magnetic_field_x=interface_x,
        interface_magnetic_field_y=interface_y,
        interface_magnetic_field_z=interface_z,
    )
    config = finalize_config(config, state.shape)
    run = jax.jit(lambda state: time_integration(state, config, params, registered_variables))
    jax.block_until_ready(run(state))
    times = []
    for _ in range(REPETITIONS):
        start = timer()
        jax.block_until_ready(run(state))
        times.append(timer() - start)
    return min(times) / NUM_STEPS


def main():
    """Print the time per step with and without the upwind dissipation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolution", type=int, default=512)
    arguments = parser.parse_args()
    without = time_per_step(arguments.resolution, False)
    print(f"N = {arguments.resolution}: without the upwind dissipation {1e3 * without:.1f} ms/step", flush=True)
    with_term = time_per_step(arguments.resolution, True)
    print(
        f"N = {arguments.resolution}: with the upwind dissipation {1e3 * with_term:.1f} ms/step, "
        f"ratio {with_term / without:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
