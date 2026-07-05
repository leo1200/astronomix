"""Validate the native Pallas WENO adjoint kernel against the native-JAX VJP.

For a single per-axis WENO flux call F = weno_flux(U):
  * the Pallas forward primal must equal the native primal (already true), and
  * the Pallas adjoint  U_bar = (dF/dU)^T F_bar  (from the hand-built atomic-
    scatter kernel) must equal the native  jax.vjp(native_flux)(F_bar).
Checked in 1D (ncomp=3) and 3D (ncomp=5, all three axes).
"""
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from astronomix import get_registered_variables
from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _weno_flux_hydro_pallas_local,
    _weno_flux_hydro_pallas_vjp_local,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN, FINITE_DIFFERENCE, PALLAS, PERIODIC_BOUNDARY, PERIODIC_ROLL,
    BoundarySettings, BoundarySettings1D, SimulationConfig, finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams

_NATIVE = {0: _weno_flux_x_native, 1: _weno_flux_y_native, 2: _weno_flux_z_native}


def _make(dim, N, seed=0):
    if dim == 1:
        bs = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    else:
        bs = BoundarySettings(*([BoundarySettings1D(PERIODIC_BOUNDARY,
                                                    PERIODIC_BOUNDARY)] * 3))
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, geometry=CARTESIAN, progress_bar=False,
        boundary_handling=PERIODIC_ROLL, num_ghost_cells=0, mhd=False,
        dimensionality=dim, box_size=1.0, num_cells=N, boundary_settings=bs,
        backend=PALLAS, pallas_block_shape=(4, 4, 8),
    )
    rv = get_registered_variables(cfg)
    params = SimulationParams(gamma=5.0 / 3.0)
    key = jax.random.key(seed)
    sh = (N,) * dim
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    rho = 1.0 + 0.3 * jax.random.uniform(k1, sh)
    p = 1.0 + 0.3 * jax.random.uniform(k2, sh)
    vx = 0.2 * jax.random.normal(k3, sh)
    kw = dict(density=rho, gas_pressure=p, velocity_x=vx)
    if dim >= 2:
        kw["velocity_y"] = 0.2 * jax.random.normal(k4, sh)
        kw["velocity_z"] = 0.2 * jax.random.normal(k5, sh)
    ps = construct_primitive_state(cfg, rv, **kw)
    cfg = finalize_config(cfg, ps.shape)
    q = conserved_state_from_primitive(ps, params.gamma, cfg, rv)
    return cfg, params, rv, q


def _cmp(name, a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    rel = np.abs(a - b).max() / scale
    ok = rel < 1e-10
    print(f"    {name}: max|{np.abs(a).max():.3e}| reldiff={rel:.3e} "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def run(dim, N):
    print(f"=== dim={dim} N={N} ===")
    cfg, params, rv, q = _make(dim, N)
    assert _hydro_pallas_flux_supported(q, cfg), "Pallas predicate False!"
    ok = True
    for axis in range(dim):
        native = lambda u, ax=axis: _NATIVE[ax](u, params, cfg, rv)
        primal_n, vjp_n = jax.vjp(native, q)
        primal_p = _weno_flux_hydro_pallas_local(q, params, cfg, rv, axis=axis)
        fbar = jax.random.normal(jax.random.key(100 + axis), q.shape, q.dtype)
        (ubar_n,) = vjp_n(fbar)
        ubar_p = _weno_flux_hydro_pallas_vjp_local(q, fbar, params, cfg, rv, axis=axis)
        print(f"  axis={axis}")
        ok &= _cmp("primal", primal_p, primal_n)
        ok &= _cmp("vjp   ", ubar_p, ubar_n)
    return ok


def main():
    res = {}
    res["1D"] = run(1, 64)
    res["3D"] = run(3, 16)
    print("\nSUMMARY:", {k: ("OK" if v else "FAIL") for k, v in res.items()})


if __name__ == "__main__":
    main()
