"""
Where the small difference between the finite-volume and the AthenaPK errors of
the CP Alfvén wave comes from.

The finite-volume scheme reproduces AthenaPK's VL2 scheme to round-off when it
starts from AthenaPK's initial state (``compare_to_athenapk.py``), but in the
convergence test of the paper each code starts from its own initial condition,
and the two L1 errors differ by 0.04-0.3 %. For every resolution this script
runs AthenaPK's VL2 + PLM + HLLD setup of the paper (GPU build, double
precision) with outputs at t = 0 and t = 5 and

1. compares AthenaPK's initial primitive state with astronomix's
   (``setup_cp_alfven_wave``, finite volume), variable by variable;
2. runs astronomix's finite-volume scheme (Pallas backend, double precision)
   from its own and from AthenaPK's initial state and prints both L1 errors
   (mean over rho, v, p, B of the mean absolute difference between the final
   and the initial state, the paper's definition) next to AthenaPK's.

    sbatch --export=ALL,RUN_DIR=examples/scripts/validation/athenapk_vl2 pytests/runners/gpu1.sh \\
        bash jupiter_athenapk.sh python alfven_athenapk_state_check.py
"""

# general
import argparse
import os
import subprocess
import sys
import tempfile

# jax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

# numerics
import numpy as np  # noqa: E402

# astronomix constants
from astronomix.option_classes.simulation_config import (  # noqa: E402
    FINITE_VOLUME,
    HLLD,
    PALLAS,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import StaticIntVector  # noqa: E402
from astronomix import (  # noqa: E402
    BackendConfig,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (  # noqa: E402
    construct_primitive_state,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave  # noqa: E402

# validation helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from athenapk_runner import (  # noqa: E402
    athenapk_binary,
    athenapk_environment,
    cp_alfven_input,
    read_output,
)

#: AthenaPK's primitive variables in the order of its output.
ATHENAPK_VARIABLES = ("rho", "v1", "v2", "v3", "p", "B1", "B2", "B3")


def astronomix_indices(registered_variables):
    """The state indices of astronomix in AthenaPK's variable order."""
    return (
        registered_variables.density_index,
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
        registered_variables.pressure_index,
        registered_variables.magnetic_index.x,
        registered_variables.magnetic_index.y,
        registered_variables.magnetic_index.z,
    )


def l1_error(final_state, initial_state, indices):
    """The paper's L1 error between two states (mean over the variables of their mean absolute difference)."""
    return float(np.mean([np.mean(np.abs(np.asarray(final_state[i]) - np.asarray(initial_state[i]))) for i in indices]))


def run_athenapk(resolution):
    """
    Run AthenaPK's VL2 + PLM + HLLD CP Alfvén wave to t = 5.

    Args:
        resolution: N (grid 2N x N x N).

    Returns:
        ``(initial_state, final_state)``: AthenaPK's primitive states (its
        variable order) at t = 0 and t = 5.
    """
    with tempfile.TemporaryDirectory(prefix=f"athenapk_cpaw_check_N{resolution}_") as run_directory:
        input_file = os.path.join(run_directory, "cpaw.in")
        with open(input_file, "w") as file:
            file.write(
                cp_alfven_input(
                    resolution,
                    integrator="vl2",
                    reconstruction="plm",
                    cfl=0.3,
                    compute_error=True,
                    output_variables="prim",
                )
            )
        result = subprocess.run(
            [athenapk_binary(gpu=True, single_precision=False), "-i", input_file],
            cwd=run_directory,
            capture_output=True,
            text=True,
            env=athenapk_environment(),
        )
        if result.returncode != 0:
            raise RuntimeError(f"athenaPK failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")
        initial_state, _, _ = read_output(os.path.join(run_directory, "parthenon.out0.00000.phdf"), "prim")
        final_state, _, _ = read_output(os.path.join(run_directory, "parthenon.out0.final.phdf"), "prim")
    return np.asarray(initial_state, dtype=np.float64), np.asarray(final_state, dtype=np.float64)


def main():
    """Compare the initial states and the errors at every resolution."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32])
    arguments = parser.parse_args()
    for resolution in arguments.resolutions:
        athenapk_initial, athenapk_final = run_athenapk(resolution)
        athenapk_error = l1_error(athenapk_final, athenapk_initial, range(len(ATHENAPK_VARIABLES)))

        config = SimulationConfig(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLD),
            backend_config=BackendConfig(backend=PALLAS),
            num_cells=StaticIntVector(2 * resolution, resolution, resolution),
            progress_bar=False,
        )
        own_initial, config, params = setup_cp_alfven_wave(config, SimulationParams(C_cfl=0.3))
        registered_variables = get_registered_variables(config)
        indices = astronomix_indices(registered_variables)

        differences = ", ".join(
            f"{name} {np.max(np.abs(np.asarray(own_initial[index]) - athenapk_initial[position])):.1e}"
            for position, (name, index) in enumerate(zip(ATHENAPK_VARIABLES, indices))
        )
        print(f"N={resolution:3d} max |initial astronomix - initial AthenaPK|: {differences}", flush=True)

        fields = dict(zip(ATHENAPK_VARIABLES, (jnp.asarray(athenapk_initial[position]) for position in range(8))))
        athenapk_initial_in_astronomix = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=fields["rho"],
            velocity_x=fields["v1"],
            velocity_y=fields["v2"],
            velocity_z=fields["v3"],
            gas_pressure=fields["p"],
            magnetic_field_x=fields["B1"],
            magnetic_field_y=fields["B2"],
            magnetic_field_z=fields["B3"],
        )
        for name, initial_state in (
            ("own initial state", own_initial),
            ("AthenaPK's initial state", athenapk_initial_in_astronomix),
        ):
            final_state = time_integration(initial_state, config, params, registered_variables)
            error = l1_error(final_state, initial_state, indices)
            print(
                f"N={resolution:3d} astronomix from {name:24s}: L1 = {error:.10e}, AthenaPK {athenapk_error:.10e}, "
                f"relative difference {abs(error - athenapk_error) / athenapk_error:.1e}",
                flush=True,
            )


if __name__ == "__main__":
    main()
