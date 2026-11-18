"""
An isothermal atmosphere at rest in an external potential.

In a fixed potential ``phi`` the isothermal gas has the equilibrium ``rho = rho_0 exp(-phi / c_s^2)``: the pressure
force ``-c_s^2 grad(rho)`` balances the weight ``-rho grad(phi)``. The schemes are not built to keep this state
exactly, they keep it up to their truncation error: the velocities that appear are small and fall at second order
with the cell size. Checked for the three finite-volume integrators with a periodic potential
``phi = phi_0 cos(2 pi x)``; the gas has no energy equation, gravity acts on the momentum alone.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/self_gravity/test_barotropic_external_potential.py
"""

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    ATHENA_SECOND_ORDER,
    HLL,
    ISOTHERMAL,
    MUSCL_HANCOCK,
    PERIODIC_ROLL,
    RK2_SSP,
    SIMPLE_SOURCE,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_config import StaticFloatVector, StaticIntVector

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

SOUND_SPEED = 1.0
# Depth of the potential well in units of c_s^2: the density varies by a factor exp(2 * 0.3) = 1.8.
POTENTIAL_DEPTH = 0.3
# Two sound-crossing times of the wavelength.
END_TIME = 2.0
# name, time integrator, gravity coupling
SCHEMES = (
    ("VL2", VL2, ATHENA_SECOND_ORDER),
    ("RK2_SSP", RK2_SSP, SIMPLE_SOURCE),
    ("MUSCL_HANCOCK", MUSCL_HANCOCK, SIMPLE_SOURCE),
)


def _atmosphere(time_integrator, gravity_version, cells_per_wavelength):
    """
    Run the hydrostatic atmosphere.

    Returns:
        The largest speed (in units of the sound speed) and the largest relative density change at the end.
    """
    cells = (cells_per_wavelength, 8, 8)
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        dimensionality=3,
        box_size=StaticFloatVector(1.0, 8.0 / cells_per_wavelength, 8.0 / cells_per_wavelength),
        num_cells=StaticIntVector(*cells),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        boundary_handling=PERIODIC_ROLL,
        solver_mode=FINITE_VOLUME,
        equation_of_state=ISOTHERMAL,
        solver_config=FVSolverConfig(time_integrator=time_integrator, riemann_solver=HLL),
        gravity_config=GravityConfig(external_potential=True, self_gravity_version=gravity_version),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    x, _, _ = jnp.meshgrid(*[(jnp.arange(count) + 0.5) / cells_per_wavelength for count in cells], indexing="ij")
    potential = POTENTIAL_DEPTH * SOUND_SPEED**2 * jnp.cos(2.0 * jnp.pi * x)
    density = jnp.exp(-potential / SOUND_SPEED**2)
    zero = jnp.zeros(cells)
    initial_state = construct_primitive_state(
        config, registered_variables, density=density, velocity_x=zero, velocity_y=zero, velocity_z=zero
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=END_TIME, isothermal_sound_speed=SOUND_SPEED, C_cfl=0.4, gravitational_potential=potential
    )
    final_state = np.asarray(time_integration(initial_state, config, params, registered_variables).final_state)
    assert np.all(np.isfinite(final_state))
    largest_speed = np.max(np.abs(final_state[1:4])) / SOUND_SPEED
    density_change = np.max(np.abs(final_state[0] / np.asarray(density) - 1.0))
    return largest_speed, density_change


def test_isothermal_atmosphere_stays_at_rest():
    """The departure from the equilibrium is the truncation error: small, and smaller on the finer grid."""
    for name, time_integrator, gravity_version in SCHEMES:
        speeds = []
        for cells_per_wavelength in (32, 64):
            speed, density_change = _atmosphere(time_integrator, gravity_version, cells_per_wavelength)
            speeds.append(speed)
            print(
                f"{name}, {cells_per_wavelength} cells: largest speed / sound speed {speed:.3e}, "
                f"largest relative density change {density_change:.3e}"
            )
            assert density_change < 2e-2
        print(f"{name}: the largest speed falls by a factor {speeds[0] / speeds[1]:.2f}")
        assert speeds[0] < 2e-2
        assert speeds[1] < 0.5 * speeds[0]


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_isothermal_atmosphere_stays_at_rest()
