"""Fast (pure-JAX, no Triton) AD-safety probe for _weno_mhd_flux_from_window.

The e2e jax.grad through time_integration NaN'd on the Pallas (jax.vjp) backward.
Root cause: zero-floor ``jnp.sqrt(jnp.maximum(0.0, X))`` in the window has an
``inf`` derivative at the clamp; under reverse mode that inf meets a 0 cotangent
-> NaN (XLA folds inf*0->0 for a single benign call, masking it).  Fix: AD-safe
``ssqrt`` (positive floor, mirrors native diff_safe_sqrt).

This probe drives the window into the guarded regions with ADVERSARIAL states
(low pressure -> c_sq<=0, zero tangential B, Bn=0, degenerate fast/slow speeds)
and checks the window VJP is finite.  Pure JAX -> iterates in seconds.

Run in astx (jax 0.10.2) on a GPU (short).
"""
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _weno_mhd_flux_from_window,
)

GAMMA, RHOMIN, PGMIN = 5.0 / 3.0, 1e-4, 1e-4
B_EPS, SQRT_FLOOR = 1e-20, 1e-12
OFFS = tuple(range(-2, 4))


def window_field(ql):
    stencil = tuple(
        tuple(jnp.roll(ql[c], -o, axis=0) for c in range(8)) for o in OFFS
    )
    flux = _weno_mhd_flux_from_window(
        stencil, GAMMA, RHOMIN, PGMIN, B_EPS, SQRT_FLOOR, 8, 7)
    return jnp.stack(flux)


def make_state(kind, N=24, seed=0):
    """Build an adversarial conserved state (8, N) in local order
    (rho, mn, mt1, mt2, Bn, Bt1, Bt2, E)."""
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.5, 1.5, N)
    mn = rng.normal(0, 0.5, N) * rho
    if kind == "lowp":
        # energy barely above floor -> p ~ 0 -> c_sq_face <= 0 in places
        mt1 = rng.normal(0, 0.3, N) * rho
        mt2 = rng.normal(0, 0.3, N) * rho
        Bn = rng.normal(0, 0.5, N)
        Bt1 = rng.normal(0, 0.5, N)
        Bt2 = rng.normal(0, 0.5, N)
        v2 = (mn**2 + mt1**2 + mt2**2) / rho**2
        b2 = Bn**2 + Bt1**2 + Bt2**2
        E = 0.5 * (rho * v2 + b2) + rng.uniform(-0.02, 0.02, N)  # p ~ 0±
    elif kind == "zerobt":
        mt1 = rng.normal(0, 0.3, N) * rho
        mt2 = rng.normal(0, 0.3, N) * rho
        Bn = rng.uniform(0.3, 0.8, N)
        Bt1 = np.zeros(N)  # bt_sq = 0 -> tangential-degeneracy branch
        Bt2 = np.zeros(N)
        E = 1.0 + 0.5 * (rho * (mn**2) / rho**2 + Bn**2)
    elif kind == "zerobn":
        mt1 = rng.normal(0, 0.3, N) * rho
        mt2 = rng.normal(0, 0.3, N) * rho
        Bn = np.zeros(N)  # lambda_alfven = 0, fast/slow degeneracy
        Bt1 = rng.normal(0, 0.5, N)
        Bt2 = rng.normal(0, 0.5, N)
        E = 1.0 + 0.5 * (rho * (mn**2) / rho**2 + Bt1**2 + Bt2**2)
    else:  # "nofield" -> pure hydro limit, denom -> 0
        mt1 = rng.normal(0, 0.3, N) * rho
        mt2 = rng.normal(0, 0.3, N) * rho
        Bn = np.zeros(N); Bt1 = np.zeros(N); Bt2 = np.zeros(N)
        E = 1.0 + 0.5 * rho * (mn**2) / rho**2
    return jnp.asarray(np.stack([rho, mn, mt1, mt2, Bn, Bt1, Bt2, E]))


def main():
    ok = True
    for kind in ("lowp", "zerobt", "zerobn", "nofield"):
        ql = make_state(kind)
        F = window_field(ql)
        fbar = jax.random.normal(jax.random.key(1), F.shape, F.dtype)
        _, vjp = jax.vjp(window_field, ql)
        (g,) = vjp(fbar)
        ffin = bool(jnp.all(jnp.isfinite(F)))
        gfin = bool(jnp.all(jnp.isfinite(g)))
        ok &= ffin and gfin
        print(f"  {kind:8s}: forward finite={ffin}  VJP finite={gfin}  "
              f"|g|max={float(jnp.max(jnp.abs(jnp.nan_to_num(g)))):.3e}")
    print("\nSUMMARY:", "OK" if ok else "FAIL")


if __name__ == "__main__":
    main()
