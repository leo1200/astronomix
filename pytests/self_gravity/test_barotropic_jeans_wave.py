"""
An isothermal Jeans wave: self-gravity coupled to a gas without an energy equation.

A standing density wave ``rho = rho_0 (1 + eps cos(k x))`` at rest in a periodic box oscillates with

    omega^2 = c_s^2 k^2 - 4 pi G rho_0

as long as the right-hand side is positive (wavelength below the Jeans length). The mode amplitude is then
``eps cos(omega t)``; numerical damping changes its size, not its zeros, so the frequency is read off the times at
which the amplitude changes sign. Checked for the three finite-volume integrators: ``VL2`` couples gravity inside its
stages (Athena++'s coupling, here without the energy source), ``RK2_SSP`` and ``MUSCL_HANCOCK`` apply a half
gravity step before and after the hydrodynamic step. Gravity moves no net momentum.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/self_gravity/test_barotropic_jeans_wave.py
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
    VAN_LEER,
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

EPS = 1e-4
SOUND_SPEED = 1.0
WAVENUMBER = 2.0 * np.pi  # one wavelength along x in a box of length 1
# 4 pi G rho_0 is half of c_s^2 k^2 (rho_0 = 1): gravity lowers the frequency of the sound wave by a factor sqrt(2).
GRAVITATIONAL_CONSTANT = 0.5 * SOUND_SPEED**2 * WAVENUMBER**2 / (4.0 * np.pi)
FREQUENCY = np.sqrt(0.5) * SOUND_SPEED * WAVENUMBER
PERIOD = 2.0 * np.pi / FREQUENCY
NUM_SNAPSHOTS = 60
# name, time integrator, self-gravity coupling
SCHEMES = (
    ("VL2", VL2, ATHENA_SECOND_ORDER),
    ("RK2_SSP", RK2_SSP, SIMPLE_SOURCE),
    ("MUSCL_HANCOCK", MUSCL_HANCOCK, SIMPLE_SOURCE),
)


def _sign_changes(times, values):
    """The times at which a sampled signal changes sign (linear interpolation between the two samples)."""
    crossing = np.nonzero(values[:-1] * values[1:] < 0.0)[0]
    return times[crossing] - values[crossing] * (times[crossing + 1] - times[crossing]) / (
        values[crossing + 1] - values[crossing]
    )


def _jeans_wave(time_integrator, self_gravity_version, cells_per_wavelength):
    """
    Run the standing wave for 1.4 periods.

    Returns:
        The measured frequency (from the first and the third sign change of the mode amplitude, one period apart)
        and the largest mean momentum density of the snapshots, in units of the momentum amplitude of the wave.
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
        solver_config=FVSolverConfig(time_integrator=time_integrator, riemann_solver=HLL, limiter=VAN_LEER),
        gravity_config=GravityConfig(self_gravity=True, self_gravity_version=self_gravity_version),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
    )
    registered_variables = get_registered_variables(config)
    x, _, _ = jnp.meshgrid(*[(jnp.arange(count) + 0.5) / cells_per_wavelength for count in cells], indexing="ij")
    zero = jnp.zeros(cells)
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=1.0 + EPS * jnp.cos(WAVENUMBER * x),
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=1.4 * PERIOD,
        isothermal_sound_speed=SOUND_SPEED,
        C_cfl=0.4,
        gravitational_constant=GRAVITATIONAL_CONSTANT,
    )
    result = time_integration(initial_state, config, params, registered_variables)

    times = np.asarray(result.time_points)
    states = np.asarray(result.states)
    density = states[:, registered_variables.density_index]
    amplitudes = 2.0 * np.mean((density - 1.0) * np.cos(WAVENUMBER * np.asarray(x)), axis=(1, 2, 3))
    zeros = _sign_changes(times, amplitudes)
    assert len(zeros) == 3, f"expected three sign changes of the mode amplitude in 1.4 periods, found {len(zeros)}"
    measured_frequency = 2.0 * np.pi / (zeros[2] - zeros[0])

    momentum = density[:, None] * states[:, 1:4]
    mean_momentum = np.max(np.abs(np.mean(momentum, axis=(2, 3, 4))))
    momentum_amplitude = EPS * FREQUENCY / WAVENUMBER
    return measured_frequency, mean_momentum / momentum_amplitude


def test_isothermal_jeans_wave_frequency():
    """The frequency converges to the dispersion relation at second order; no net momentum appears."""
    for name, time_integrator, self_gravity_version in SCHEMES:
        errors = []
        for cells_per_wavelength in (32, 64):
            frequency, momentum = _jeans_wave(time_integrator, self_gravity_version, cells_per_wavelength)
            errors.append(frequency / FREQUENCY - 1.0)
            print(
                f"{name}, {cells_per_wavelength} cells: frequency {frequency:.5f} (dispersion relation "
                f"{FREQUENCY:.5f}, relative error {errors[-1]:+.3e}), mean momentum / wave momentum {momentum:.2e}"
            )
            assert momentum < 1e-10
        assert abs(errors[0]) < 2e-2
        assert abs(errors[1]) < 5e-3


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_isothermal_jeans_wave_frequency()
