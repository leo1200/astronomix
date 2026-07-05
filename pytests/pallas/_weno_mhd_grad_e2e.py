"""End-to-end check: jax.grad through time_integration for ideal-gas MHD,
BACKWARDS mode, on the PALLAS backend (using the new Pallas MHD reverse adjoint)
vs the NATIVE_JAX backend.  Mirrors the field-level-inference use: differentiate
a density functional of the evolved state w.r.t. the initial velocity field.

The Pallas forward flux is bit-exact vs native, so its exact transpose (the new
Pallas adjoint) yields a gradient matching the native backward to FP-order
(~1e-7).  This is the lever that keeps the 256^3 inverse problem on the GPU.

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

from astronomix import (
    CodeUnits, SimulationConfig, SimulationParams, construct_primitive_state,
    get_registered_variables, time_integration,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, NATIVE_JAX, PALLAS, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, finalize_config,
)
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
)


def _build(backend, N, t_end, num_checkpoints):
    bk = {}
    if backend == "pallas":
        bk = dict(backend=PALLAS, pallas_block_shape=(4, 4, 8),
                  pallas_use_triton=True, pallas_interpret=False)
    else:
        bk = dict(backend=NATIVE_JAX)
    cfg = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, mhd=True, progress_bar=False,
        donate_state=False, dimensionality=3,
        box_size=1.0, num_cells=N, differentiation_mode=BACKWARDS,
        num_checkpoints=num_checkpoints,
        turbulent_forcing_config=TurbulentForcingConfig(turbulent_forcing=False),
        boundary_settings=BoundarySettings(
            *[BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
              for _ in range(3)]),
        **bk,
    )
    rv = get_registered_variables(cfg)
    params = SimulationParams(C_cfl=0.8, dt_max=0.1, gamma=5.0 / 3.0,
                              minimum_density=1e-4, minimum_pressure=1e-4,
                              t_end=t_end)
    sh = (N,) * 3
    key = jax.random.key(0)
    ks = jax.random.split(key, 8)
    rho = 1.0 + 0.2 * jax.random.uniform(ks[0], sh)
    p = 1.0 + 0.2 * jax.random.uniform(ks[1], sh)
    vx = 0.2 * jax.random.normal(ks[2], sh)
    vy = 0.2 * jax.random.normal(ks[3], sh)
    vz = 0.2 * jax.random.normal(ks[4], sh)
    B0 = 0.5
    Bx = B0 + 0.1 * jax.random.normal(ks[5], sh)
    By = 0.1 * jax.random.normal(ks[6], sh)
    Bz = 0.1 * jax.random.normal(ks[7], sh)
    bxb, byb, bzb = initialize_interface_fields(Bx, By, Bz)
    ps = construct_primitive_state(
        config=cfg, registered_variables=rv, density=rho, gas_pressure=p,
        velocity_x=vx, velocity_y=vy, velocity_z=vz,
        magnetic_field_x=Bx, magnetic_field_y=By, magnetic_field_z=Bz,
        interface_magnetic_field_x=bxb, interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb)
    cfg = finalize_config(cfg, ps.shape)
    return cfg, params, rv, ps


def _loss_and_grad(backend, N, t_end, num_checkpoints):
    cfg, params, rv, ps0 = _build(backend, N, t_end, num_checkpoints)
    di = rv.density_index
    vi = rv.velocity_index
    v0 = jnp.stack([ps0[vi.x], ps0[vi.y], ps0[vi.z]])

    def loss(v):
        ps = ps0.at[vi.x].set(v[0]).at[vi.y].set(v[1]).at[vi.z].set(v[2])
        fin = time_integration(ps, cfg, params, rv)
        proj = jnp.sum(fin[di], axis=2)
        return jnp.mean(proj ** 2)

    g = jax.grad(loss)
    t0 = time.time()
    grad = jax.block_until_ready(g(v0))
    return float(jax.block_until_ready(loss(v0))), np.asarray(grad), time.time() - t0


def main():
    N, t_end, ckpt = 16, 0.05, 4
    print(f"=== MHD grad e2e  N={N} t_end={t_end} ===")
    l_n, g_n, t_n = _loss_and_grad("native", N, t_end, ckpt)
    print(f"native:  loss={l_n:.6e}  grad|max|={np.abs(g_n).max():.3e}  ({t_n:.1f}s)")
    l_p, g_p, t_p = _loss_and_grad("pallas", N, t_end, ckpt)
    print(f"pallas:  loss={l_p:.6e}  grad|max|={np.abs(g_p).max():.3e}  ({t_p:.1f}s)")
    scale = max(np.abs(g_n).max(), np.abs(g_p).max(), 1e-30)
    rel = np.abs(g_n - g_p).max() / scale
    lrel = abs(l_n - l_p) / max(abs(l_n), 1e-30)
    print(f"loss reldiff={lrel:.3e}  grad reldiff={rel:.3e}  "
          f"{'OK' if rel < 1e-6 else 'MISMATCH'}")
    print("\nSUMMARY:", "OK" if rel < 1e-6 else "FAIL")


if __name__ == "__main__":
    main()
