"""
First-order flux correction under the unsplit RK2_SSP finite-volume scheme.

Two receding streams (``u = -5`` and ``u = +5`` at ``rho = 1``, ``p = 0.4``) open a near-vacuum between them. With
the HLL solver and the superbee limiter the second-order stage drives a cell there to a negative pressure; the
flux correction replaces the fluxes at the faces of such a cell by donor-cell fluxes and the state stays positive.
In a smooth flow the correction finds nothing to replace and the result is unchanged.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_rk2_ssp_flux_correction.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# astronomix constants
from astronomix import FINITE_VOLUME, OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import HLL, RK2_SSP, SUPERBEE

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

CELLS = 100
STREAM_SPEED = 5.0


def _final_state(boundary, flux_correction, density, velocity, pressure, t_end):
    """Final primitive state of a 1D RK2_SSP run with HLL and superbee."""
    config = SimulationConfig(
        dimensionality=1,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(boundary, boundary),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(
            time_integrator=RK2_SSP,
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


def _receding_streams(flux_correction):
    x = (jnp.arange(CELLS) + 0.5) / CELLS
    return _final_state(
        OPEN_BOUNDARY,
        flux_correction,
        density=jnp.ones(CELLS),
        velocity=jnp.where(x < 0.5, -STREAM_SPEED, STREAM_SPEED),
        pressure=jnp.full(CELLS, 0.4),
        t_end=0.1 / STREAM_SPEED,
    )


def _sound_wave(flux_correction):
    x = (jnp.arange(CELLS) + 0.5) / CELLS
    wave = 1e-3 * jnp.sin(2.0 * jnp.pi * x)
    return _final_state(
        PERIODIC_BOUNDARY, flux_correction, density=1.0 + wave, velocity=wave, pressure=0.6 + wave, t_end=0.2
    )


def test_flux_correction_keeps_receding_streams_positive():
    """Without the correction the near-vacuum fails; with it density and pressure stay positive."""
    plain, registered_variables = _receding_streams(flux_correction=False)
    corrected, _ = _receding_streams(flux_correction=True)
    density, pressure = corrected[registered_variables.density_index], corrected[registered_variables.pressure_index]
    plain_fails = not np.all(np.isfinite(plain)) or plain[registered_variables.pressure_index].min() <= 0.0
    print(f"plain run fails: {plain_fails}; corrected minima: density {density.min():.3e}, pressure {pressure.min():.3e}")
    assert plain_fails
    assert np.all(np.isfinite(corrected))
    assert density.min() > 0.0 and pressure.min() > 0.0


def test_flux_correction_leaves_smooth_flow_unchanged():
    """A smooth sound wave gives the same result with and without the correction."""
    plain, _ = _sound_wave(flux_correction=False)
    corrected, _ = _sound_wave(flux_correction=True)
    difference = np.max(np.abs(corrected - plain))
    print(f"maximum difference with against without the correction: {difference:.3e}")
    assert difference < 1e-12


if __name__ == "__main__":
    test_flux_correction_keeps_receding_streams_positive()
    test_flux_correction_leaves_smooth_flow_unchanged()
