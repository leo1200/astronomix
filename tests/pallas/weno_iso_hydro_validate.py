"""Validate the Pallas isothermal-hydro WENO flux against the native JAX flux.

The Pallas kernel (`_weno_flux_hydro_iso_pallas` in `_weno_pallas.py`) is
generated to mirror the native isothermal WENO path in `_weno.py`
(`_euler_flux_isothermal_x` + `_eigen_hydro_iso`).  This script builds a
random periodic isothermal conserved state and checks that the Pallas and
native interface fluxes agree to single-precision rounding, for every axis
in 1-D / 2-D / 3-D.  It also confirms the Pallas predicate actually engages
(so we are really testing the kernel, not a silent native fallback).

Run with PYTHONPATH set to the repo root (astronomix is installed
non-editable):

    PYTHONPATH=/export/home/lstorcks/agent-home/astronomix \
        python tests/pallas/weno_iso_hydro_validate.py
"""

import os

# ==== device selection ====
# ISO_CPU=1 runs the Pallas kernel in interpret mode on the CPU — a
# GPU-free correctness check of the kernel translation (no Triton lowering).
# Otherwise grab one free GPU via autocvd and exercise the real Triton path.
_CPU = os.environ.get("ISO_CPU", "0") == "1"
if _CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# ==========================

import jax
if os.environ.get("ISO_X64", "0") == "1":
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from astronomix import SimulationConfig, SimulationParams, get_registered_variables
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, ISOTHERMAL, NATIVE_JAX, PALLAS, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, finalize_config,
)
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_iso_pallas_flux_supported, _weno_flux_hydro_iso_pallas,
)

CS = 0.8
RHOMIN = 1e-8
BLOCK = (4, 4, 8)


def _periodic_bcs(ndim):
    p = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return BoundarySettings(p, p, p)


def _make_configs(ndim, n):
    base = dict(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=ISOTHERMAL, mhd=False,
        dimensionality=ndim, num_cells=n, box_size=1.0,
        enforce_positivity=False, progress_bar=False,
        boundary_settings=_periodic_bcs(ndim),
    )
    native = SimulationConfig(backend=NATIVE_JAX, **base)
    pallas = SimulationConfig(
        backend=PALLAS, pallas_block_shape=BLOCK, pallas_use_triton=not _CPU,
        pallas_interpret=_CPU, **base,
    )
    return native, pallas


def _random_state(rv, ndim, n, seed):
    rng = np.random.default_rng(seed)
    shape = (n,) * ndim
    nvars = ndim + 1
    state = np.empty((nvars,) + shape, dtype=np.float64)
    # density: positive, O(1) with fluctuations
    state[rv.density_index] = 1.0 + 0.3 * rng.standard_normal(shape)
    state[rv.density_index] = np.maximum(state[rv.density_index], 0.05)
    dens = state[rv.density_index]
    if ndim == 1:
        state[rv.momentum_index] = dens * 0.4 * rng.standard_normal(shape)
    else:
        state[rv.momentum_index.x] = dens * 0.4 * rng.standard_normal(shape)
        state[rv.momentum_index.y] = dens * 0.4 * rng.standard_normal(shape)
        if ndim == 3:
            state[rv.momentum_index.z] = dens * 0.4 * rng.standard_normal(shape)
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jnp.asarray(state, dtype=dtype)


def main():
    print(f"x64 = {jax.config.jax_enable_x64}", flush=True)
    params = SimulationParams(isothermal_sound_speed=CS, minimum_density=RHOMIN)
    native_fns = (_weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native)

    all_ok = True
    for ndim, n in ((1, 32), (2, 16), (3, 16)):
        native_cfg, pallas_cfg = _make_configs(ndim, n)
        rv = get_registered_variables(native_cfg)
        state = _random_state(rv, ndim, n, seed=1234 + ndim)
        native_cfg = finalize_config(native_cfg, state.shape)
        pallas_cfg = finalize_config(pallas_cfg, state.shape)

        supported = _hydro_iso_pallas_flux_supported(state, pallas_cfg)
        print(f"\n=== ndim={ndim}, N={n} | Pallas supported: {supported} ===",
              flush=True)
        if not supported:
            print("  !! Pallas predicate rejected the state — kernel NOT exercised")
            all_ok = False
            continue

        for axis in range(ndim):
            f_native = np.asarray(
                native_fns[axis](state, params, native_cfg, rv)
            )
            f_pallas = np.asarray(
                _weno_flux_hydro_iso_pallas(state, params, pallas_cfg, rv, axis=axis)
            )
            scale = np.max(np.abs(f_native)) + 1e-30
            absdiff = np.max(np.abs(f_native - f_pallas))
            reldiff = absdiff / scale
            tol = 1e-5 if jax.config.jax_enable_x64 else 1e-4
            ok = reldiff < tol
            all_ok = all_ok and ok
            print(f"  axis {axis}: max|Δ|={absdiff:.3e}  rel={reldiff:.3e}  "
                  f"[{'PASS' if ok else 'FAIL'}]", flush=True)

    print("\n" + ("ALL PASS — Pallas iso-hydro WENO matches native."
                  if all_ok else "FAILURES present."), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
