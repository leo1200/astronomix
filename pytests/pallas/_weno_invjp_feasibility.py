"""Decision spike: can in-kernel ``jax.vjp`` of the *full* WENO window flux
replace the hand-derived explicit adjoint on jax 0.10.2?

The hydro Pallas adjoint (``_weno_flux_hydro_pallas_vjp_local``) currently calls
the hand-derived ``_weno_hydro_flux_from_window_adjoint`` because, on the OLD
jaxlib (0.6.2), the auto-generated VJP of the full window was mis-lowered /
very slow to compile on Triton.  jax >= ~0.8 fixed the Triton autodiff bugs
(we are on 0.10.2).  If in-kernel ``jax.vjp`` of the window now lowers cleanly,
compiles in reasonable time, and is bit-exact vs the hand-derived adjoint, then
the MHD adjoint needs only the pure window function factored out + ``jax.vjp``
(NO ~500-line hand-derivation).

Method: monkeypatch the module-global ``_weno_hydro_flux_from_window_adjoint``
with a drop-in ``jax.vjp`` wrapper (same signature), run the EXISTING, validated
adjoint kernel, and compare to (a) the hand-derived adjoint and (b) native
``jax.vjp`` of the flux.  Also time the first (compile) call of each.

Run in astx (jax 0.10.2) on a GPU.
"""
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
import time
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from astronomix import get_registered_variables
from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x_native, _weno_flux_y_native, _weno_flux_z_native,
)
import astronomix._finite_difference._interface_fluxes._weno_pallas as wp
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _weno_flux_hydro_pallas_local,
    _weno_flux_hydro_pallas_vjp_local,
    _weno_hydro_flux_from_window,
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

# Keep a handle on the hand-derived adjoint (the module global gets swapped).
_HAND = wp._weno_hydro_flux_from_window_adjoint


def _adjoint_via_vjp(q_stencil, flux_bar, gamma, rhomin, pgmin, ncomp, num_modes):
    """Drop-in replacement for ``_weno_hydro_flux_from_window_adjoint`` that uses
    ``jax.vjp`` of the shared forward window instead of hand-derived arithmetic.
    Same signature / return shape (list[6] of list[ncomp])."""
    flat = [q_stencil[k][c] for k in range(6) for c in range(ncomp)]

    def wf(*flat_qs):
        qs = tuple(tuple(flat_qs[k * ncomp + c] for c in range(ncomp)) for k in range(6))
        return tuple(_weno_hydro_flux_from_window(qs, gamma, rhomin, pgmin, ncomp, num_modes))

    _, vjp_fn = jax.vjp(wf, *flat)
    cts = vjp_fn(tuple(flux_bar))
    return [[cts[k * ncomp + c] for c in range(ncomp)] for k in range(6)]


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


def _cmp(name, a, b, tol=1e-10):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    scale = max(np.abs(a).max(), np.abs(b).max(), 1e-30)
    rel = np.abs(a - b).max() / scale
    ok = rel < tol
    print(f"    {name}: reldiff={rel:.3e} {'OK' if ok else 'MISMATCH'}")
    return ok


def _timed(fn):
    t0 = time.time(); out = jax.block_until_ready(fn()); return out, time.time() - t0


def run(dim, N):
    print(f"=== dim={dim} N={N} ===")
    cfg, params, rv, q = _make(dim, N)
    assert _hydro_pallas_flux_supported(q, cfg), "Pallas predicate False!"
    ok = True
    for axis in range(dim):
        native = lambda u, ax=axis: _NATIVE[ax](u, params, cfg, rv)
        _, vjp_n = jax.vjp(native, q)
        fbar = jax.random.normal(jax.random.key(100 + axis), q.shape, q.dtype)
        (ubar_n,) = vjp_n(fbar)

        # hand-derived adjoint (current production path), timed cold.
        wp._weno_hydro_flux_from_window_adjoint = _HAND
        jax.clear_caches()
        ubar_hand, t_hand = _timed(
            lambda: _weno_flux_hydro_pallas_vjp_local(q, fbar, params, cfg, rv, axis=axis))

        # in-kernel jax.vjp adjoint, timed cold.
        wp._weno_hydro_flux_from_window_adjoint = _adjoint_via_vjp
        jax.clear_caches()
        try:
            ubar_vjp, t_vjp = _timed(
                lambda: _weno_flux_hydro_pallas_vjp_local(q, fbar, params, cfg, rv, axis=axis))
            lowered = True
        except Exception as e:
            print(f"  axis={axis}: in-kernel vjp RAISED {type(e).__name__}: {str(e)[:200]}")
            lowered = False
        finally:
            wp._weno_hydro_flux_from_window_adjoint = _HAND

        print(f"  axis={axis}  compile: hand={t_hand:.1f}s  invjp="
              f"{t_vjp if lowered else float('nan'):.1f}s")
        ok &= _cmp("hand vs native ", ubar_hand, ubar_n)
        if lowered:
            ok &= _cmp("invjp vs native", ubar_vjp, ubar_n)
            ok &= _cmp("invjp vs hand  ", ubar_vjp, ubar_hand, tol=1e-12)
        else:
            ok = False
    return ok


def main():
    res = {}
    res["1D"] = run(1, 64)
    res["3D"] = run(3, 16)
    print("\nSUMMARY:", {k: ("OK" if v else "FAIL") for k, v in res.items()})


if __name__ == "__main__":
    main()
