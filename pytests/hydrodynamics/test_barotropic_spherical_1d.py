"""
The spherical 1D grid with a barotropic gas (isothermal, polytropic) in the finite-volume framework.

The radial flux divergence weights the momentum flux, pressure included, with the face areas; the pressure source
of the geometry gives back the part that does not push on the gas. A barotropic gas has no pressure slot, the
source takes the pressure the equation of state gives for the density. Checked with the split scheme, the integrator
of the spherical grid: a uniform gas at rest stays at rest, and a pulse that runs into the centre conserves the mass.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_barotropic_spherical_1d.py
"""

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, REFLECTIVE_BOUNDARY, SPHERICAL
from astronomix.option_classes.simulation_config import HLL, ISOTHERMAL, MUSCL_HANCOCK, POLYTROPIC

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

CELLS = 64
STEPS = 200
# name, equation of state
LAWS = (("isothermal", ISOTHERMAL), ("polytropic", POLYTROPIC))


def _closed_sphere(equation_of_state, density, time_step):
    """Initial and final state of ``STEPS`` fixed steps between reflective walls, and the cell volumes."""
    config = SimulationConfig(
        geometry=SPHERICAL,
        dimensionality=1,
        box_size=1.0,
        num_cells=CELLS,
        boundary_settings=BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=FVSolverConfig(time_integrator=MUSCL_HANCOCK, riemann_solver=HLL),
        mhd=False,
        fixed_timestep=True,
        num_timesteps=STEPS,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    initial_state = construct_primitive_state(
        config, registered_variables, density=density, velocity_x=jnp.zeros(CELLS)
    )
    config = finalize_config(config, initial_state.shape)
    cell_volumes = np.asarray(get_helper_data(config).cell_volumes)
    params = SimulationParams(
        t_end=STEPS * time_step, isothermal_sound_speed=0.8, polytropic_constant=0.6, polytropic_exponent=5.0 / 3.0
    )
    final_state = np.asarray(time_integration(initial_state, config, params, registered_variables).final_state)
    return np.asarray(initial_state), final_state, cell_volumes


def test_uniform_barotropic_gas_stays_at_rest():
    """The pressure source balances the area factors of the pressure in the momentum flux."""
    for name, equation_of_state in LAWS:
        _, (density, velocity), _ = _closed_sphere(equation_of_state, jnp.ones(CELLS), 0.4 / CELLS)
        deviation = max(np.max(np.abs(density - 1.0)), np.max(np.abs(velocity)))
        print(f"{name}: largest deviation from the uniform state at rest after {STEPS} steps: {deviation:.3e}")
        assert deviation < 1e-12


def test_barotropic_pulse_conserves_the_mass():
    """A density pulse runs into the centre and back; the total mass stays and the density stays positive."""
    radius = (np.arange(CELLS) + 0.5) / CELLS
    # The pulse does not reach the outer wall within the run (see ``test_spherical_1d.py`` on that wall).
    pulse = np.where(np.abs(radius - 0.2) < 0.1, np.cos(np.pi * (radius - 0.2) / 0.2) ** 2, 0.0)
    for name, equation_of_state in LAWS:
        initial_state, final_state, cell_volumes = _closed_sphere(
            equation_of_state, jnp.asarray(1.0 + 0.5 * pulse), 1.25e-3
        )
        mass_change = np.sum(final_state[0] * cell_volumes) / np.sum(initial_state[0] * cell_volumes) - 1.0
        centre_change = np.max(np.abs(final_state[:, :3] - initial_state[:, :3]))
        print(
            f"{name}: relative mass change {mass_change:.3e}, largest change in the three innermost cells "
            f"{centre_change:.3f}, smallest density {final_state[0].min():.3f}"
        )
        assert np.all(np.isfinite(final_state)) and final_state[0].min() > 0.0
        assert abs(mass_change) < 1e-12
        # The pulse has reached the centre.
        assert centre_change > 0.01


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_uniform_barotropic_gas_stays_at_rest()
        test_barotropic_pulse_conserves_the_mass()
