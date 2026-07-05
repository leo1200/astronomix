"""Validate the hand-derived explicit WENO window adjoint
``_weno_hydro_flux_from_window_adjoint`` bit-exact against ``jax.vjp`` of the
forward window ``_weno_hydro_flux_from_window``, for ncomp = 3/4/5 on random
positive states.  Pure CPU, no GPU needed."""
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _weno_hydro_flux_from_window,
    _weno_hydro_flux_from_window_adjoint,
)

GAMMA, RHOMIN, PGMIN = 1.4, 1e-10, 1e-10


def relerr(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return np.abs(a - b).max() / max(np.abs(a).max(), np.abs(b).max(), 1e-30)


def main():
    rng = np.random.default_rng(0)
    ok = True
    for ncomp in (3, 4, 5):
        nm = ncomp
        qs = [jnp.asarray(rng.uniform(0.6, 1.4, (ncomp,))) for _ in range(6)]

        def F(qsflat):
            qst = tuple(tuple(qsflat[k][c] for c in range(ncomp)) for k in range(6))
            return tuple(_weno_hydro_flux_from_window(qst, GAMMA, RHOMIN, PGMIN, ncomp, nm))

        out, vjp = jax.vjp(F, qs)
        fb = tuple(jnp.asarray(rng.uniform(-1, 1)) for _ in range(ncomp))
        (gref,) = vjp(fb)
        g = _weno_hydro_flux_from_window_adjoint(
            [list(q) for q in qs], list(fb), GAMMA, RHOMIN, PGMIN, ncomp, nm
        )
        rel = max(relerr(jnp.asarray(gref[k]), jnp.asarray(g[k])) for k in range(6))
        status = "OK" if rel < 1e-10 else "MISMATCH"
        ok &= rel < 1e-10
        print(f"ncomp={ncomp} explicit window adjoint vs jax.vjp: rel={rel:.2e} {status}")
    print("SUMMARY:", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
