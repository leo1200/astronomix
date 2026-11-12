"""
Self-gravity under the unsplit RK2_SSP finite-volume scheme: growth of a Jeans-unstable mode.

A standing density perturbation ``rho = rho_0 (1 + eps cos(k x))`` at rest in a periodic box with
``4 pi G rho_0 = 2 c_s^2 k^2`` is Jeans unstable: its amplitude grows as ``eps cosh(s t)`` with the growth rate
``s = sqrt(4 pi G rho_0 - c_s^2 k^2)``. Without gravity the mode is a standing sound wave whose amplitude never
exceeds ``eps``. The test runs to ``s t = 1.5`` and compares the amplitude of the mode with the linear solution.

The source is added once per step with the potential of the state before the step, which is first order in time:
at a CFL number of 0.2 and 32 cells per wavelength the growth factor is 4 % below the linear one. The box uses
``PERIODIC_ROLL``: with ghost cells the Poisson solve of this coupling acts on the padded field.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/self_gravity/test_rk2_ssp_gravity.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax

# The perturbation is linear (eps = 1e-4): double precision keeps round-off below it.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix import FINITE_VOLUME, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import PERIODIC_ROLL, RK2_SSP, SIMPLE_SOURCE

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

GAMMA = 5.0 / 3.0
EPS = 1e-4
CELLS = (32, 8, 8)
WAVENUMBER = 2.0 * jnp.pi  # one wavelength along x in a box of length 1
GROWTH_RATE = WAVENUMBER  # 4 pi G rho_0 = 2 c_s^2 k^2 with rho_0 = c_s = 1
GROWTH_TIME = 1.5 / GROWTH_RATE


def _mode_amplitude(density, x):
    """Amplitude of the ``cos(k x)`` mode of the density."""
    return float(2.0 * jnp.mean((density - 1.0) * jnp.cos(WAVENUMBER * x)))


def test_rk2_ssp_jeans_growth_rate():
    """The Jeans-unstable mode grows by ``cosh(s t)`` under RK2_SSP with self-gravity."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    config = SimulationConfig(
        dimensionality=3,
        box_size=StaticFloatVector(1.0, 0.25, 0.25),
        num_cells=StaticIntVector(*CELLS),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        boundary_handling=PERIODIC_ROLL,
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=RK2_SSP),
        gravity_config=GravityConfig(self_gravity=True, self_gravity_version=SIMPLE_SOURCE),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    x, _, _ = jnp.meshgrid(*[(jnp.arange(cells) + 0.5) / CELLS[0] for cells in CELLS], indexing="ij")
    zero = jnp.zeros(CELLS)
    initial_state = construct_primitive_state(
        config,
        registered_variables,
        density=1.0 + EPS * jnp.cos(WAVENUMBER * x),
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
        gas_pressure=1.0 / GAMMA + EPS * jnp.cos(WAVENUMBER * x),
    )
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(
        t_end=GROWTH_TIME,
        gamma=GAMMA,
        C_cfl=0.2,
        gravitational_constant=2.0 * WAVENUMBER**2 / (4.0 * jnp.pi),
    )
    result = time_integration(initial_state, config, params, registered_variables)

    growth = _mode_amplitude(result.final_state[registered_variables.density_index], x) / EPS
    expected = float(jnp.cosh(GROWTH_RATE * GROWTH_TIME))
    print(f"growth factor {growth:.4f}, linear solution {expected:.4f}")
    assert abs(growth / expected - 1.0) < 0.05


if __name__ == "__main__":
    test_rk2_ssp_jeans_growth_rate()
