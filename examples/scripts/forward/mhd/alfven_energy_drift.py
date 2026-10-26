"""
Total-energy change of the finite-difference MHD scheme on the CP Alfvén wave.

The constrained transport updates the cell-centred field from the interface
fields after every stage and corrects the total energy by the change of the
magnetic energy, so that the thermal pressure is kept; the total energy is
then conserved only to the truncation order of the scheme. This script runs the
3D circularly polarized Alfvén wave of the convergence test (double precision,
WENO-Z, C_cfl = 1.5, Pallas backend) to t = 5 and prints the relative change of
the total energy, |E(t_end) - E(0)| / |E(0)|, at every resolution (the numbers
quoted in the paper's constrained-transport section).

    sbatch --export=ALL,RUN_DIR=examples/scripts/forward/mhd \\
        pytests/runners/gpu1.sh python alfven_energy_drift.py
"""

# general
import argparse

# jax
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

# astronomix constants
from astronomix.option_classes.simulation_config import PALLAS  # noqa: E402

# astronomix containers
from astronomix.option_classes.simulation_config import StaticIntVector  # noqa: E402
from astronomix import (  # noqa: E402
    BackendConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (  # noqa: E402
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave  # noqa: E402


def total_energy(state, registered_variables, gamma):
    """The total energy (thermal, kinetic and magnetic) summed over the cells."""
    density = state[registered_variables.density_index]
    pressure = state[registered_variables.pressure_index]
    kinetic = 0.5 * density * sum(
        state[index] ** 2
        for index in (
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
        )
    )
    magnetic = 0.5 * sum(
        state[index] ** 2
        for index in (
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        )
    )
    return float(jnp.sum(pressure / (gamma - 1.0) + kinetic + magnetic))


def main():
    """Run every resolution and print the relative change of the total energy."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[16, 32, 64])
    arguments = parser.parse_args()
    for resolution in arguments.resolutions:
        config = SimulationConfig(
            weno_z=True,
            backend_config=BackendConfig(backend=PALLAS),
            num_cells=StaticIntVector(2 * resolution, resolution, resolution),
            progress_bar=False,
        )
        initial_state, config, params = setup_cp_alfven_wave(config, SimulationParams(C_cfl=1.5))
        registered_variables = get_registered_variables(config)
        final_state = time_integration(initial_state, config, params, registered_variables)
        initial_energy = total_energy(initial_state, registered_variables, params.gamma)
        final_energy = total_energy(final_state, registered_variables, params.gamma)
        print(
            f"N = {resolution:3d}: |E(t_end) - E(0)| / |E(0)| = "
            f"{abs(final_energy - initial_energy) / abs(initial_energy):.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
