"""Per-component GPU microbenchmark of the TI step (WENO / cooling / forcing / ...).

Times each piece in isolation on a saturated-looking state, so the numbers are
not confounded by JIT compilation or the initial heating transient the way a
short end-to-end run is.
"""

import os
import sys
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402

import time
import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    SimulationParams, SnapshotSettings, get_registered_variables,
    finalize_config, construct_primitive_state,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing import (
    _apply_forcing, _apply_ou_forcing, _create_solenoidal_field,
)
from astronomix._modules._cooling._cooling import update_pressure_by_cooling
from astronomix._modules._conduction._conduction import fd_conduction_source
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_x
from astronomix._fluid_equations._equations import conserved_state_from_primitive

from _common import GAMMA, make_fd_config, ism_ti_cooling_setup
from casa_ti_phase import gks_code_units, MU_ATH, BOX_SIZE, HRATE_CGS, DFLOOR, PFLOOR

N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
REP = 20


def timeit(fn, *a, label=""):
    out = fn(*a)
    jax.block_until_ready(out)
    t0 = time.time()
    for _ in range(REP):
        out = fn(*a)
    jax.block_until_ready(out)
    dt = (time.time() - t0) / REP * 1e3
    print(f"  {label:38s} {dt:8.2f} ms")
    return dt


def main():
    cu = gks_code_units()
    for explicit in (False, True):
        tag = "EXPLICIT" if explicit else "IMPLICIT"
        cc, cp = ism_ti_cooling_setup(
            cu, hrate_cgs=HRATE_CGS, mu_athena=MU_ATH, floor_temperature_K=10.0,
            hydrogen_mass_fraction=0.7, metal_mass_fraction=0.02, explicit=explicit)
        config = make_fd_config(
            BOX_SIZE, N, mhd=False, cooling_config=cc,
            snapshot_settings=SnapshotSettings(return_final_state=True),
            num_snapshots=2, random_seed=7,
            turbulent_forcing_config=TurbulentForcingConfig(
                turbulent_forcing=True, ou_forcing=True),
            thermal_conduction=True, conduction_density_weighted=True,
        )
        rv = get_registered_variables(config)
        key = jax.random.PRNGKey(0)
        # saturated-looking state: warm branch with ~20% fluctuations
        rho = jnp.exp(jax.random.normal(key, (N, N, N)) * 0.18)
        T = 90.0 * jnp.exp(jax.random.normal(jax.random.PRNGKey(1), (N, N, N)) * 0.18)
        v = [jax.random.normal(jax.random.PRNGKey(2 + i), (N, N, N)) * 7.0 for i in range(3)]
        state = construct_primitive_state(
            config=config, registered_variables=rv, density=rho,
            velocity_x=v[0], velocity_y=v[1], velocity_z=v[2], gas_pressure=rho * T)
        config = finalize_config(config, state.shape)
        params = SimulationParams(
            gamma=GAMMA, C_cfl=0.3, t_end=1.0,
            minimum_density=DFLOOR, minimum_pressure=PFLOOR,
            thermal_conductivity=1.0, cooling_params=cp,
            turbulent_forcing_params=TurbulentForcingParams(
                energy_injection_rate=5e6, correlation_time=0.5,
                forcing_wavenumber=0.589, forcing_amplitude=10.7),
        )
        cons = conserved_state_from_primitive(state, GAMMA, config, rv)
        dt = 1.5e-3
        print(f"\n=== {tag} cooling, N={N} ({N**3/1e6:.2f} M cells) ===")
        f_weno = jax.jit(lambda c: _weno_flux_x(c, params, config, rv))
        timeit(f_weno, cons, label="WENO flux, x axis (x12 per step)")
        f_cool = jax.jit(lambda s: update_pressure_by_cooling(
            s, rv, cc, params, dt, grid_spacing=config.grid_spacing))
        timeit(f_cool, state, label="cooling update (x4 per step)")
        f_cond = jax.jit(lambda s: fd_conduction_source(s, params, config, rv))
        timeit(f_cond, state, label="conduction source (x4 per step)")
        fld = jax.jit(lambda k: _create_solenoidal_field(k, config, 0.589))
        timeit(fld, key, label="OU solenoidal field draw (x1 per step)")
        f_ou = jax.jit(lambda k, s: _apply_ou_forcing(
            (k, jnp.zeros((3, N, N, N))), s, dt,
            params.turbulent_forcing_params, config, rv))
        timeit(f_ou, key, state, label="OU forcing apply (x1 per step)")
        f_wh = jax.jit(lambda k, s: _apply_forcing(
            k, s, dt, params.turbulent_forcing_params, config, rv))
        timeit(f_wh, key, state, label="white forcing apply (x1 per step)")


if __name__ == "__main__":
    main()
