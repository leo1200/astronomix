"""A/B performance probe — ONE config per process.

Usage: python _bench_perf.py <config_name> <out_json>
Measures compile time, runtime, per-device compiled memory for the named
config against whichever ``astronomix`` is importable from this repo root.
"""

import sys
import json
from time import perf_counter

from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402

import jax
jax.config.update("jax_enable_x64", True)
import astronomix

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, FINITE_VOLUME, RK4_SSP, SimulationConfig, SnapshotSettings, StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration
from astronomix.test_setups.hydrodynamics.shock_tube1D import setup_sod_shock_tube
from astronomix.test_setups.hydrodynamics.sound_wave3D import setup_sound_wave
from astronomix.test_setups.mhd.alfven_wave3D import setup_cp_alfven_wave, CPAlfvenWave3DSettings

_COMMON = dict(
    return_snapshots=True, num_snapshots=2,
    snapshot_settings=SnapshotSettings(return_final_state=True),
    print_elapsed_time=True, memory_analysis=True, progress_bar=False,
)


def build_shock_tube():
    config = SimulationConfig(solver_mode=FINITE_VOLUME, num_cells=400, dimensionality=1, **_COMMON)
    rv = get_registered_variables(config); hd = get_helper_data(config)
    state, config, params = setup_sod_shock_tube(config, rv, SimulationParams(C_cfl=0.4), hd)
    params = params._replace(t_end=0.05)
    return state, config, params, get_registered_variables(config)


def build_sound_wave():
    config = SimulationConfig(solver_mode=FINITE_VOLUME, mhd=False, dimensionality=3,
                              num_cells=StaticIntVector(32, 32, 32), **_COMMON)
    state, config, params = setup_sound_wave(config, SimulationParams(C_cfl=0.4))
    params = params._replace(t_end=0.1)
    return state, config, params, get_registered_variables(config)


def build_alfven():
    config = SimulationConfig(solver_mode=FINITE_DIFFERENCE, mhd=True, dimensionality=3,
                              num_cells=StaticIntVector(32, 16, 16), time_integrator=RK4_SSP, **_COMMON)
    state, config, params = setup_cp_alfven_wave(config, SimulationParams(C_cfl=0.8), CPAlfvenWave3DSettings(t_end=1.0))
    return state, config, params, get_registered_variables(config)


BUILDERS = {
    "FV_shocktube_1D": build_shock_tube,
    "FV_soundwave_3D": build_sound_wave,
    "FD_MHD_alfven_3D": build_alfven,
}

name, out_json = sys.argv[1], sys.argv[2]
print("astronomix:", astronomix.__file__, flush=True)
jax.device_put(0.0).block_until_ready()  # backend init, excluded from compile timing

state, config, params, rv = BUILDERS[name]()
t0 = perf_counter()
result = time_integration(state, config, params, rv)
result.final_state.block_until_ready()
wall = perf_counter() - t0
runtime = float(result.runtime)
out = {
    "compile_s": wall - runtime,
    "runtime_s": runtime,
    "total_mem_mb": float(result.total_memory_bytes) / 1024**2,
    "temp_mem_mb": float(result.temporary_memory_bytes) / 1024**2,
    "iters": int(result.num_iterations),
}
print(f"RESULT {name}: {out}", flush=True)
with open(out_json, "w") as f:
    json.dump(out, f)
