"""Validate the Pallas AD modes at the kernel level.

Checks, for the FD hydro WENO flux dispatcher (`_weno_flux_x/y/z`) on a
random smooth 3D state:

  - primal: PALLAS == NATIVE (bit / fp rounding)
  - grad wrt state and gamma under
      * PALLAS_AD_JVP_NATIVE  (existing custom_jvp design)
      * PALLAS_AD_VJP_REMAT   (custom_vjp, native recompute backward)
      * PALLAS_AD_VJP_PALLAS  (custom_vjp, Pallas adjoint kernels)
    against the NATIVE_JAX backend gradient.

Also validates the flux-divergence accumulator kernel path under VJP_REMAT.

Run on CPU (interpret mode, no GPU needed):
    ADV_CPU=1 PYTHONPATH=<repo> python tests/pallas/ad_vjp_validate.py
Run on GPU (real Triton lowering):
    PYTHONPATH=<repo> python tests/pallas/ad_vjp_validate.py
"""

import os

_CPU = os.environ.get("ADV_CPU", "0") == "1"
if _CPU:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402

import jax

if os.environ.get("ADV_X64", "0") == "1":
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    NATIVE_JAX,
    PALLAS,
    PALLAS_AD_JVP_NATIVE,
    PALLAS_AD_VJP_PALLAS,
    PALLAS_AD_VJP_REMAT,
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams

N = int(os.environ.get("ADV_N", "16"))
MODE_NAMES = {
    PALLAS_AD_JVP_NATIVE: "JVP_NATIVE",
    PALLAS_AD_VJP_REMAT: "VJP_REMAT",
    PALLAS_AD_VJP_PALLAS: "VJP_PALLAS",
}


def make_config(backend, ad_mode=PALLAS_AD_JVP_NATIVE):
    pb = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    cfg = SimulationConfig(
        backend=backend,
        pallas_block_shape=(4, 4, 8),
        pallas_interpret=_CPU,
        pallas_ad_mode=ad_mode,
        solver_mode=FINITE_DIFFERENCE,
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        dimensionality=3,
        box_size=StaticFloatVector(x=1.0, y=1.0, z=1.0),
        num_cells=StaticIntVector(x=N, y=N, z=N),
        boundary_settings=BoundarySettings(pb, pb, pb),
        progress_bar=False,
        return_snapshots=False,
    )
    return finalize_config(cfg, (5, N, N, N))


def make_state(key):
    kr, kvx, kvy, kvz, kp = jax.random.split(key, 5)
    shp = (N, N, N)
    rho = 1.0 + 0.1 * jax.random.normal(kr, shp)
    vx = 0.1 * jax.random.normal(kvx, shp)
    vy = 0.1 * jax.random.normal(kvy, shp)
    vz = 0.1 * jax.random.normal(kvz, shp)
    p = 1.0 + 0.1 * jax.random.normal(kp, shp)
    mom = jnp.stack([rho * vx, rho * vy, rho * vz])
    e = p / (5.0 / 3.0 - 1.0) + 0.5 * rho * (vx**2 + vy**2 + vz**2)
    return jnp.concatenate([rho[None], mom, e[None]], axis=0)


def check_weno_flux():
    from astronomix._finite_difference._interface_fluxes._weno import (
        _weno_flux_x, _weno_flux_y, _weno_flux_z,
    )
    from astronomix._finite_difference._interface_fluxes._weno_pallas import (
        _hydro_pallas_flux_supported,
    )

    key = jax.random.PRNGKey(0)
    state = make_state(key)
    weights = jax.random.normal(jax.random.PRNGKey(1), state.shape)
    params = SimulationParams(t_end=0.1, gamma=5.0 / 3.0)

    cfg_probe = make_config(PALLAS)
    assert _hydro_pallas_flux_supported(state, cfg_probe), (
        "Pallas hydro predicate rejected the test state — test is vacuous"
    )

    for axis, flux_fn in enumerate((_weno_flux_x, _weno_flux_y, _weno_flux_z)):
        rv_cache = {}

        def loss(s, g, cfg):
            rv = rv_cache.setdefault(id(cfg), get_registered_variables(cfg))
            p = params._replace(gamma=g)
            return jnp.sum(flux_fn(s, p, cfg, rv) * weights)

        cfg_nat = make_config(NATIVE_JAX)
        f_nat = loss(state, params.gamma, cfg_nat)
        g_nat_s, g_nat_g = jax.grad(loss, argnums=(0, 1))(state, params.gamma, cfg_nat)

        print(f"axis={axis}")
        for mode in (PALLAS_AD_JVP_NATIVE, PALLAS_AD_VJP_REMAT, PALLAS_AD_VJP_PALLAS):
            cfg = make_config(PALLAS, ad_mode=mode)
            f_pal = loss(state, params.gamma, cfg)
            g_pal_s, g_pal_g = jax.grad(loss, argnums=(0, 1))(state, params.gamma, cfg)
            ds = float(jnp.max(jnp.abs(g_pal_s - g_nat_s)))
            dsr = ds / float(jnp.max(jnp.abs(g_nat_s)))
            dg = abs(float(g_pal_g) - float(g_nat_g)) / max(abs(float(g_nat_g)), 1e-30)
            df = abs(float(f_pal) - float(f_nat)) / max(abs(float(f_nat)), 1e-30)
            print(
                f"  [{MODE_NAMES[mode]:10s}] |dJ|/J={df:.2e}  "
                f"max|dgrad_state| rel={dsr:.2e}  |dgrad_gamma| rel={dg:.2e}",
                flush=True,
            )
            assert df < 1e-5, f"primal mismatch axis={axis} mode={MODE_NAMES[mode]}"
            tol = 1e-4 if not jax.config.jax_enable_x64 else 1e-10
            assert dsr < tol, f"state grad mismatch axis={axis} mode={MODE_NAMES[mode]}"
            assert dg < tol, f"gamma grad mismatch axis={axis} mode={MODE_NAMES[mode]}"


def check_div_kernel():
    from astronomix._finite_difference._time_integrators._ssprk_pallas import (
        _hydro_flux_div_axis_native,
        _hydro_flux_div_axis_pallas,
    )

    key = jax.random.PRNGKey(2)
    dF = jax.random.normal(key, (5, N, N, N))
    rhs = jax.random.normal(jax.random.PRNGKey(3), (5, N, N, N))
    ct_w = jax.random.normal(jax.random.PRNGKey(4), (5, N, N, N))

    for mode in (PALLAS_AD_JVP_NATIVE, PALLAS_AD_VJP_REMAT):
        cfg = make_config(PALLAS, ad_mode=mode)

        def loss_pal(dF_, dtdx_, rhs_, scale_):
            out = _hydro_flux_div_axis_pallas(
                dF_, dtdx_, cfg, axis=0, rhs_accumulator=rhs_, scale_in=scale_,
            )
            return jnp.sum(out * ct_w)

        def loss_nat(dF_, dtdx_, rhs_, scale_):
            out = _hydro_flux_div_axis_native(
                dF_, dtdx_, axis=0, rhs_accumulator=rhs_, scale_in=scale_,
            )
            return jnp.sum(out * ct_w)

        gp = jax.grad(loss_pal, argnums=(0, 1, 2, 3))(dF, 0.5, rhs, 0.7)
        gn = jax.grad(loss_nat, argnums=(0, 1, 2, 3))(dF, 0.5, rhs, 0.7)
        rel = [
            float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-30))
            for a, b in zip(gp, gn)
        ]
        print(f"div acc kernel [{MODE_NAMES[mode]:10s}] rel grad diffs: "
              + " ".join(f"{r:.2e}" for r in rel), flush=True)
        assert all(r < 1e-5 for r in rel), f"div grad mismatch mode={MODE_NAMES[mode]}"


if __name__ == "__main__":
    print(f"platform={jax.default_backend()} x64={jax.config.jax_enable_x64} N={N}")
    check_div_kernel()
    check_weno_flux()
    print("ALL PASSED")
