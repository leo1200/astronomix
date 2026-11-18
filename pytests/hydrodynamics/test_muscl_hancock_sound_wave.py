"""
The split finite-volume scheme (MUSCL_HANCOCK) treats every axis alike.

A plane sound wave runs along the last axis of a 1D, a 2D and a 3D periodic box that is uniform along the other
axes. In the Strang sequence the sweep along the last axis takes the whole step and the sweeps along the uniform
axes have nothing to do, so the three runs give the same profile to rounding level. A Hancock predictor that
couples the pressure to the x velocity in every sweep (instead of the velocity along the sweep) misses this by
4 to 16 percent of the wave amplitude and loses the second order along y and z.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_muscl_hancock_sound_wave.py
"""

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import HLL, HLLC, MINMOD, MUSCL_HANCOCK, VAN_LEER

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix.option_classes.simulation_config import StaticFloatVector

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

GAMMA = 5.0 / 3.0
AMPLITUDE = 1e-6
UNIFORM_CELLS = 4
T_END = 0.5
CFL = 0.4
VELOCITY_NAMES = ("velocity_x", "velocity_y", "velocity_z")


def _linear_wave(position, time):
    """Density, normal velocity and pressure of the right-going sound wave on ``rho = c = 1``."""
    perturbation = AMPLITUDE * np.cos(2.0 * np.pi * (position - time))
    return np.stack([1.0 + perturbation, perturbation, 1.0 / GAMMA + perturbation])


def _wave_profile(dimensionality, cells, riemann_solver, limiter):
    """Final density, normal velocity and pressure along the last axis, one row of cells per uniform position."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    shape = (UNIFORM_CELLS,) * (dimensionality - 1) + (cells,)
    box = [count / cells for count in shape] + [1.0] * (3 - dimensionality)
    config = SimulationConfig(
        dimensionality=dimensionality,
        box_size=1.0 if dimensionality == 1 else StaticFloatVector(*box),
        boundary_settings=periodic if dimensionality == 1 else BoundarySettings(periodic, periodic, periodic),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=MUSCL_HANCOCK, riemann_solver=riemann_solver, limiter=limiter),
        mhd=False,
        # The same time step in every box, so that the runs can be compared cell by cell.
        fixed_timestep=True,
        num_timesteps=int(round(T_END * cells / CFL)),
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    density, normal_velocity, pressure = _linear_wave(np.broadcast_to((np.arange(cells) + 0.5) / cells, shape), 0.0)
    velocities = {name: jnp.zeros(shape) for name in VELOCITY_NAMES[:dimensionality]}
    velocities[VELOCITY_NAMES[dimensionality - 1]] = jnp.asarray(normal_velocity)
    initial_state = construct_primitive_state(
        config, registered_variables, density=jnp.asarray(density), gas_pressure=jnp.asarray(pressure), **velocities
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA, C_cfl=CFL)
    final_state = np.asarray(time_integration(initial_state, config, params, registered_variables).final_state)
    rows = [registered_variables.density_index, dimensionality, registered_variables.pressure_index]
    return final_state[rows].reshape(3, -1, cells)


def _error(profile):
    """L1 error against the linear solution in units of the wave amplitude."""
    cells = profile.shape[-1]
    exact = _linear_wave((np.arange(cells) + 0.5) / cells, T_END)
    return np.mean(np.abs(profile - exact[:, None, :])) / AMPLITUDE


def test_wave_along_the_last_axis_repeats_the_1d_run():
    """The 2D and 3D runs repeat the 1D run to rounding level."""
    cells = 32
    for solver_name, riemann_solver, limiter in (("HLL, minmod", HLL, MINMOD), ("HLLC, van Leer", HLLC, VAN_LEER)):
        reference = _wave_profile(1, cells, riemann_solver, limiter)
        for dimensionality in (2, 3):
            profile = _wave_profile(dimensionality, cells, riemann_solver, limiter)
            difference = np.max(np.abs(profile - reference)) / AMPLITUDE
            print(f"{solver_name}, {dimensionality}D against 1D: {difference:.3e} of the amplitude")
            # Rounding of the unit background is 1e-10 of the amplitude; the x-velocity predictor gives 4e-2.
            assert difference < 1e-6


def test_wave_along_y_converges_at_second_order():
    """Along y the wave converges to the linear solution as it does along x."""
    coarse = _error(_wave_profile(2, 32, HLLC, VAN_LEER))
    fine = _error(_wave_profile(2, 64, HLLC, VAN_LEER))
    order = np.log2(coarse / fine)
    print(f"L1 error / amplitude along y: {coarse:.3e} (32 cells), {fine:.3e} (64 cells), order {order:.2f}")
    assert fine < 1e-2
    assert order > 1.6


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_wave_along_the_last_axis_repeats_the_1d_run()
        test_wave_along_y_converges_at_second_order()
