"""
1D Sod shock tube pytest.

Runs the classic Sod shock tube on a handful of solver configurations
(finite-volume HLLC with minmod / superbee limiters and the finite-difference
solver), compares each against the analytic Riemann solution to within a fixed
tolerance, and writes an overview figure of density, velocity and pressure.
"""

# general
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    HLLC,
    MINMOD,
)
from astronomix.option_classes.simulation_config import SUPERBEE

# astronomix containers
from astronomix import (
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    get_registered_variables,
)
from astronomix.option_classes.simulation_config import config_to_string
from astronomix.plotting_helpers.inset_box import add_inset_box
from astronomix.test_setups.hydrodynamics.shock_tube1D import (
    setup_sod_shock_tube,
    sod_shock_tube_solution,
)


num_cells = 200

# The configurations under test: two finite-volume HLLC runs differing only in
# their slope limiter, plus the finite-difference solver.
config_list = [
    SimulationConfig(
        solver_mode = FINITE_VOLUME,
        solver_config=FVSolverConfig(limiter = MINMOD, riemann_solver = HLLC),
        num_cells = num_cells,
    ),
    SimulationConfig(
        solver_mode = FINITE_VOLUME,
        solver_config=FVSolverConfig(limiter = SUPERBEE, riemann_solver = HLLC),
        num_cells = num_cells,
    ),
    SimulationConfig(
        num_cells = num_cells,
    ),
]


def test_shock_tube1D(tol = 1e-2):
    """
    Run the Sod shock tube for every configuration and assert convergence.

    Each configuration is integrated to the test end time, compared against the
    analytic Sod solution, and plotted alongside a high-resolution reference.
    The mean absolute error in density, velocity and pressure must stay below
    ``tol`` for the test to pass.

    Args:
        tol: The maximum allowed mean absolute error per primitive variable.
    """

    # -------------------------------------------------------------
    # =========== ↓ High-resolution reference solution ↓ ==========
    # -------------------------------------------------------------

    fig, (ax_density, ax_velocity, ax_pressure) = plt.subplots(1, 3, figsize=(15, 5))

    # A finely resolved analytic solution serves as the visual reference the
    # coarser runs are plotted against.
    config_high_res = SimulationConfig(
        dimensionality = 1,
        num_cells = 1000,
    )

    registered_variables = get_registered_variables(config_high_res)
    helper_data_high_res = get_helper_data(config_high_res)
    _, config_high_res, params_high_res = setup_sod_shock_tube(
        config_high_res,
        registered_variables,
        SimulationParams(),
        helper_data_high_res,
    )
    reference_solution_high_res = sod_shock_tube_solution(
        config_high_res,
        registered_variables,
        params_high_res,
        helper_data_high_res,
    )

    ax_density.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.density_index],
        label='Reference',
        color='black'
    )
    ax_velocity.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.velocity_index],
        label='Reference',
        color='black'
    )
    ax_pressure.plot(
        helper_data_high_res.geometric_centers,
        reference_solution_high_res[registered_variables.pressure_index],
        label='Reference',
        color='black'
    )
    ax_density.set_xlabel('x')
    ax_density.set_ylabel('Density')
    ax_velocity.set_xlabel('x')
    ax_velocity.set_ylabel('Velocity')
    ax_pressure.set_xlabel('x')
    ax_pressure.set_ylabel('Pressure')
    ax_density.set_title('Density')
    ax_velocity.set_title('Velocity')
    ax_pressure.set_title('Pressure')

    # -------------------------------------------------------------
    # =========== ↑ High-resolution reference solution ↑ ==========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Per-configuration run and check ↓ ============
    # -------------------------------------------------------------

    for config in config_list:

        # Set up the simulation for this configuration.
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        # Build the initial state. The finite-difference solver tolerates a
        # larger CFL number than the finite-volume solver here.
        initial_state, config, params = setup_sod_shock_tube(
            config,
            registered_variables,
            SimulationParams(
                C_cfl = 1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.8,
            ),
            helper_data,
        )

        # Run the simulation to the test end time.
        final_state = time_integration(
            initial_state,
            config,
            params,
            registered_variables
        )

        # The analytic Sod solution on this configuration's grid.
        true_final_state = sod_shock_tube_solution(
            config,
            registered_variables,
            params,
            helper_data,
        )

        # Mean absolute error per primitive variable against the analytic
        # solution, indexed through the registered variables.
        density_error = jnp.mean(
            jnp.abs(
                final_state[registered_variables.density_index]
                - true_final_state[registered_variables.density_index]
            )
        )
        velocity_error = jnp.mean(
            jnp.abs(
                final_state[registered_variables.velocity_index]
                - true_final_state[registered_variables.velocity_index]
            )
        )
        pressure_error = jnp.mean(
            jnp.abs(
                final_state[registered_variables.pressure_index]
                - true_final_state[registered_variables.pressure_index]
            )
        )

        # Every primitive variable must match the analytic solution within tol.
        assert density_error < tol, f"Density error {density_error} exceeds tolerance {tol}"
        assert velocity_error < tol, f"Velocity error {velocity_error} exceeds tolerance {tol}"
        assert pressure_error < tol, f"Pressure error {pressure_error} exceeds tolerance {tol}"

        # Overlay this run on the reference plots.
        ax_density.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.density_index],
            label=config_to_string(config),
        )
        ax_velocity.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.velocity_index],
            label=config_to_string(config),
        )
        ax_pressure.plot(
            helper_data.geometric_centers,
            final_state[registered_variables.pressure_index],
            label=config_to_string(config),
        )

    # -------------------------------------------------------------
    # ============ ↑ Per-configuration run and check ↑ ============
    # -------------------------------------------------------------

    # A zoomed inset around the contact discontinuity highlights how the
    # limiters differ where the solutions are hardest to resolve.
    add_inset_box(ax_density, x1=0.62, x2=0.72, y1=0.20, y2=0.30, connect_loc1=1)

    handles, labels = ax_density.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    fig.savefig(figures_dir / "shock_tube1D_test.svg")

if __name__ == "__main__":
    test_shock_tube1D()
