"""
First-order flux correction under the RK2_SSP and MUSCL_HANCOCK finite-volume schemes (hydrodynamics).

Two receding streams (``u = -s`` and ``u = +s`` at ``rho = 1``, ``p = 0.4``) open a near-vacuum between them. With
the HLL solver and the superbee limiter the second-order update drives a cell there to a negative pressure (unsplit
scheme at ``s = 5``, split scheme at ``s = 40``); the flux correction replaces the fluxes at the faces of such a
cell by donor-cell fluxes and the state stays positive. The split scheme corrects every sweep on its own. In a
smooth flow the correction finds nothing to replace and the result is unchanged.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_fv_flux_correction.py
"""

# testing
import pytest

# numerics
import jax
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import HLL, MUSCL_HANCOCK, RK2_SSP, SUPERBEE

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
    get_registered_variables,
    time_integration,
)

pytestmark = pytest.mark.x64

CELLS = 100
# name, time integrator, speed of the receding streams at which the scheme fails without the correction
SCHEMES = (("RK2_SSP", RK2_SSP, 5.0), ("MUSCL_HANCOCK", MUSCL_HANCOCK, 40.0))


def _final_state(time_integrator, boundary, flux_correction, density, velocity, pressure, t_end):
    """Final primitive state of a 1D finite-volume run with HLL and superbee."""
    config = SimulationConfig(
        dimensionality=1,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(boundary, boundary),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(
            time_integrator=time_integrator,
            riemann_solver=HLL,
            limiter=SUPERBEE,
            first_order_flux_correction=flux_correction,
        ),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    initial_state = construct_primitive_state(
        config, registered_variables, density=density, velocity_x=velocity, gas_pressure=pressure
    )
    config = finalize_config(config, initial_state.shape)
    result = time_integration(initial_state, config, SimulationParams(t_end=t_end, C_cfl=0.4), registered_variables)
    return np.asarray(result.final_state), registered_variables


def _receding_streams(time_integrator, stream_speed, flux_correction):
    x = (jnp.arange(CELLS) + 0.5) / CELLS
    return _final_state(
        time_integrator,
        OPEN_BOUNDARY,
        flux_correction,
        density=jnp.ones(CELLS),
        velocity=jnp.where(x < 0.5, -stream_speed, stream_speed),
        pressure=jnp.full(CELLS, 0.4),
        t_end=0.1 / stream_speed,
    )


def _sound_wave(time_integrator, flux_correction):
    x = (jnp.arange(CELLS) + 0.5) / CELLS
    wave = 1e-3 * jnp.sin(2.0 * jnp.pi * x)
    return _final_state(
        time_integrator,
        PERIODIC_BOUNDARY,
        flux_correction,
        density=1.0 + wave,
        velocity=wave,
        pressure=0.6 + wave,
        t_end=0.2,
    )


def test_flux_correction_keeps_receding_streams_positive():
    """Without the correction the near-vacuum fails; with it density and pressure stay positive."""
    for name, time_integrator, stream_speed in SCHEMES:
        plain, registered_variables = _receding_streams(time_integrator, stream_speed, flux_correction=False)
        corrected, _ = _receding_streams(time_integrator, stream_speed, flux_correction=True)
        density = corrected[registered_variables.density_index]
        pressure = corrected[registered_variables.pressure_index]
        plain_fails = not np.all(np.isfinite(plain)) or plain[registered_variables.pressure_index].min() <= 0.0
        print(
            f"{name}, stream speed {stream_speed:g}: plain run fails: {plain_fails}; "
            f"corrected minima: density {density.min():.3e}, pressure {pressure.min():.3e}"
        )
        assert plain_fails
        assert np.all(np.isfinite(corrected))
        assert density.min() > 0.0 and pressure.min() > 0.0


def test_flux_correction_leaves_smooth_flow_unchanged():
    """A smooth sound wave gives the same result with and without the correction."""
    for name, time_integrator, _ in SCHEMES:
        plain, _ = _sound_wave(time_integrator, flux_correction=False)
        corrected, _ = _sound_wave(time_integrator, flux_correction=True)
        difference = np.max(np.abs(corrected - plain))
        print(f"{name}: maximum difference with against without the correction: {difference:.3e}")
        assert difference < 1e-12


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_flux_correction_keeps_receding_streams_positive()
        test_flux_correction_leaves_smooth_flow_unchanged()
