"""
Why the finite-volume and AthenaPK errors of the CP Alfvén wave differ slightly.

The finite-volume scheme reproduces AthenaPK's VL2 scheme to round-off from the
same initial state, yet on the CP Alfvén-wave convergence test (paper,
Fig. mhd_convergence) the two L1 errors differ by 0.04-0.3 %. The two codes do
not start from the same discrete state: ``setup_cp_alfven_wave`` builds the
cell-centered field of the finite-volume scheme as the central-difference curl
of the analytic vector potential (divergence-free under the central-difference
divergence), whereas AthenaPK's ``cpaw`` problem generator sets the point values
of the analytic field. This script runs the finite-volume scheme from both
initial fields and compares each error, measured as in the paper against the
run's own initial state, with AthenaPK's (``l1_errors_primitive`` of the stored
reference). Double precision, native-JAX backend; it runs on a CPU.

    JAX_PLATFORMS=cpu PYTHONPATH=$(git rev-parse --show-toplevel) \\
        python alfven_initial_field_check.py [--resolutions 8 16 32]
"""

# general
import argparse
from pathlib import Path

# jax
import jax

jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    HLLD,
    NATIVE_JAX,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import StaticIntVector

from astronomix import (
    BackendConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix.test_setups.mhd.alfven_wave3D import CPAlfvenWave3DSettings

# astronomix functions
from astronomix import (
    construct_primitive_state,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.mhd.alfven_wave3D import (
    _wave_primitive_state,
    setup_cp_alfven_wave,
)

REPOSITORY = Path(__file__).resolve().parents[4]
ATHENAPK_REFERENCE = (
    REPOSITORY / "pytests" / "mhd" / "data" / "athenapk" / "gh200" / "cfl03_teams256_singleblock"
    / "athenapk_alfven_convergence.npz"
)

#: The variables of the error norm (rho, v, p, B), as in the paper.
ERROR_FIELDS = ("density", "velocity_x", "velocity_y", "velocity_z", "pressure", "field_x", "field_y", "field_z")


def error_against_initial_state(final_state, initial_state, registered_variables):
    """The mean over rho, v, p and B of the mean absolute difference between two states."""
    indices = (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )
    return float(np.mean([np.mean(np.abs(np.asarray(final_state[index] - initial_state[index]))) for index in indices]))


def point_value_state(config, registered_variables, helper_data):
    """The initial state with the point values of the analytic field (AthenaPK's ``cpaw``)."""
    centers = helper_data.geometric_centers
    density, velocity_x, velocity_y, velocity_z, pressure, field_x, field_y, field_z = _wave_primitive_state(
        centers[..., 0],
        centers[..., 1],
        centers[..., 2],
        0.0,
        CPAlfvenWave3DSettings(),
    )
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        gas_pressure=pressure,
        magnetic_field_x=field_x,
        magnetic_field_y=field_y,
        magnetic_field_z=field_z,
    )


def main():
    """Run both initial fields at every resolution and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32])
    arguments = parser.parse_args()

    reference = np.load(ATHENAPK_REFERENCE)
    athenapk_errors = dict(zip(reference["N_values"].tolist(), reference["l1_errors_primitive"].tolist()))

    for resolution in arguments.resolutions:
        config = SimulationConfig(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLD),
            backend_config=BackendConfig(backend=NATIVE_JAX),
            num_cells=StaticIntVector(2 * resolution, resolution, resolution),
            progress_bar=False,
        )
        curl_state, config, params = setup_cp_alfven_wave(config, SimulationParams(C_cfl=0.3))
        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)
        initial_states = {
            "curl of the vector potential (paper)": curl_state,
            "point values (as AthenaPK)": point_value_state(config, registered_variables, helper_data),
        }
        athenapk_error = athenapk_errors.get(resolution, float("nan"))
        for name, initial_state in initial_states.items():
            final_state = time_integration(initial_state, config, params, registered_variables)
            error = error_against_initial_state(final_state, initial_state, registered_variables)
            print(
                f"N={resolution:3d} {name:38s} L1 = {error:.10e}  AthenaPK {athenapk_error:.10e}  "
                f"relative difference {abs(error - athenapk_error) / athenapk_error:.2e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
