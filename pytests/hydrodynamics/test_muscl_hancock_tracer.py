"""
The split finite-volume scheme (MUSCL_HANCOCK) carries the wind density tracer with the gas.

A smooth compressive flow in a periodic 1D box starts with the tracer at three tenths of the gas density everywhere. The
tracer is a density, so its concentration (tracer over gas density) is constant along the flow and has to stay at
three tenths. This holds to rounding level when the Hancock predictor compresses the tracer as it compresses the gas;
a predictor that only advects the tracer misses it at first order in the cell size (9e-4 at 64 cells).

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_muscl_hancock_tracer.py
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
from astronomix import FINITE_VOLUME, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import HLL, HLLC, MINMOD, MUSCL_HANCOCK, VAN_LEER

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    FVSolverConfig,
    SimulationConfig,
    SimulationParams,
    SnapshotSettings,
)
from astronomix._modules._stellar_wind.stellar_wind_options import WindConfig

# astronomix functions
from astronomix import (
    finalize_config,
    get_registered_variables,
    time_integration,
)

GAMMA = 5.0 / 3.0
CELLS = 64
CONCENTRATION = 0.3
T_END = 0.2
CFL = 0.4


def _final_state(riemann_solver, limiter):
    """Final state of the compressive flow and the registered variables."""
    config = SimulationConfig(
        dimensionality=1,
        box_size=1.0,
        boundary_settings=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(time_integrator=MUSCL_HANCOCK, riemann_solver=riemann_solver, limiter=limiter),
        # The tracer row without the wind source.
        wind_config=WindConfig(stellar_wind=False, trace_wind_density=True),
        mhd=False,
        progress_bar=False,
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    registered_variables = get_registered_variables(config)
    position = (np.arange(CELLS) + 0.5) / CELLS
    initial_state = np.zeros((registered_variables.num_vars, CELLS))
    initial_state[registered_variables.density_index] = 1.0 + 0.3 * np.sin(2.0 * np.pi * position + 1.0)
    initial_state[registered_variables.velocity_index] = 0.5 * np.sin(2.0 * np.pi * position)
    initial_state[registered_variables.pressure_index] = 1.0
    initial_state[registered_variables.wind_density_index] = (
        CONCENTRATION * initial_state[registered_variables.density_index]
    )
    initial_state = jnp.asarray(initial_state)
    config = finalize_config(config, initial_state.shape)
    params = SimulationParams(t_end=T_END, gamma=GAMMA, C_cfl=CFL)
    result = time_integration(initial_state, config, params, registered_variables)
    return np.asarray(result.final_state), registered_variables


def test_uniform_concentration_stays_uniform():
    """The concentration stays at its initial value while the gas density changes by tens of percent."""
    for solver_name, riemann_solver, limiter in (("HLL, minmod", HLL, MINMOD), ("HLLC, van Leer", HLLC, VAN_LEER)):
        final_state, registered_variables = _final_state(riemann_solver, limiter)
        density = final_state[registered_variables.density_index]
        tracer = final_state[registered_variables.wind_density_index]
        deviation = np.max(np.abs(tracer / density - CONCENTRATION))
        density_change = np.max(np.abs(density - 1.0))
        print(f"{solver_name}: concentration off by {deviation:.3e}, density off its mean by {density_change:.2f}")
        assert density_change > 0.2
        assert deviation < 1e-12


if __name__ == "__main__":
    test_uniform_concentration_stays_uniform()
