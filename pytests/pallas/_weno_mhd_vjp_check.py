"""Validate the native Pallas MHD WENO adjoint kernel against the native-JAX VJP.

For a single per-axis ideal-gas MHD WENO flux call F = weno_flux(U):
  * the Pallas forward primal equals the native primal (see
    _weno_mhd_forward_check.py), and
  * the Pallas adjoint  U_bar = (dF/dU)^T F_bar  (the in-kernel jax.vjp of the
    shared window + roll-scatter) must equal native  jax.vjp(native_flux)(F_bar).
Because the Pallas forward is bit-exact vs native, the Pallas adjoint (its exact
transpose) matches the native VJP to FP-order (~1e-10..1e-7).  3D, all 3 axes.

Run in astx (jax 0.10.2) on a GPU.  Pass --interpret to additionally run the
Pallas correctness oracle (CPU lowering — allowed; it is not a compute run).
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--interpret", action="store_true")
ap.add_argument("-N", type=int, default=16)
args = ap.parse_args()

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
    _mhd_pallas_flux_supported,
    _weno_flux_mhd_pallas_local,
    _weno_flux_mhd_pallas_vjp_local,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
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


def _make(N, interpret, seed=0):
    bs = BoundarySettings(*([BoundarySettings1D(PERIODIC_BOUNDARY,
                                                PERIODIC_BOUNDARY)] * 3))
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, geometry=CARTESIAN, progress_bar=False,
        boundary_handling=PERIODIC_ROLL, num_ghost_cells=0, mhd=True,
        dimensionality=3, box_size=1.0, num_cells=N, boundary_settings=bs,
        backend=PALLAS, pallas_block_shape=(4, 4, 8), pallas_interpret=interpret,
    )
    rv = get_registered_variables(cfg)
    params = SimulationParams(gamma=5.0 / 3.0)
    key = jax.random.key(seed)
    sh = (N,) * 3
    ks = jax.random.split(key, 9)
    rho = 1.0 + 0.3 * jax.random.uniform(ks[0], sh)
    p = 1.0 + 0.3 * jax.random.uniform(ks[1], sh)
    vx = 0.2 * jax.random.normal(ks[2], sh)
    vy = 0.2 * jax.random.normal(ks[3], sh)
    vz = 0.2 * jax.random.normal(ks[4], sh)
    Bx = 0.5 + 0.2 * jax.random.normal(ks[5], sh)
    By = 0.3 + 0.2 * jax.random.normal(ks[6], sh)
    Bz = 0.1 + 0.2 * jax.random.normal(ks[7], sh)
    bxb, byb, bzb = initialize_interface_fields(Bx, By, Bz)
    ps = construct_primitive_state(
        cfg, rv, density=rho, gas_pressure=p,
        velocity_x=vx, velocity_y=vy, velocity_z=vz,
        magnetic_field_x=Bx, magnetic_field_y=By, magnetic_field_z=Bz,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
    )
    cfg = finalize_config(cfg, ps.shape)
    q = conserved_state_from_primitive(ps, params.gamma, cfg, rv)
    return cfg, params, rv, q


def _cmp(name, a, b, tol):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    rel = np.abs(a - b).max() / scale
    ok = rel < tol
    print(f"    {name}: max|{np.abs(a).max():.3e}| reldiff={rel:.3e} "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def run(interpret):
    label = "interpret" if interpret else "GPU/Triton"
    print(f"=== MHD VJP check ({label}) N={args.N} ===")
    cfg, params, rv, q = _make(args.N, interpret)
    assert _mhd_pallas_flux_supported(q, cfg), "MHD Pallas predicate False!"
    ok = True
    # primal bit-exact tol; adjoint FP-order tol.
    ptol = 1e-10 if not interpret else 1e-12
    vtol = 1e-9 if not interpret else 1e-11
    for axis in range(3):
        native = lambda u, ax=axis: _NATIVE[ax](u, params, cfg, rv)
        primal_n, vjp_n = jax.vjp(native, q)
        primal_p = _weno_flux_mhd_pallas_local(q, params, cfg, rv, axis=axis)
        fbar = jax.random.normal(jax.random.key(100 + axis), q.shape, q.dtype)
        (ubar_n,) = vjp_n(fbar)
        ubar_p = _weno_flux_mhd_pallas_vjp_local(q, fbar, params, cfg, rv, axis=axis)
        print(f"  axis={axis}")
        ok &= _cmp("primal", primal_p, primal_n, ptol)
        ok &= _cmp("vjp   ", ubar_p, ubar_n, vtol)
    return ok


def main():
    ok = run(interpret=args.interpret)
    print("\nSUMMARY:", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
