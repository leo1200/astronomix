"""Feasibility spike for native Pallas VJPs.

The make-or-break question for a hand-rolled WENO Pallas adjoint: can
reverse-mode AD (``jax.vjp`` / ``jax.grad``) be *traced and lowered inside a
Pallas kernel body* on the Triton GPU backend? If so we can let JAX
differentiate the per-block forward computation (bit-exact, no hand-derived
WENO adjoint) and only handle the inter-block scatter ourselves.

Three escalating tests:
  (A) elementwise: kernel writes  x_bar = vjp(f)(ct)  for a nonlinear f
  (B) stencil reduce: kernel computes a 3-point nonlinear stencil per cell
      and VJPs it w.r.t. the local neighbourhood (the WENO-shaped case)
  (C) atomic scatter: kernel atomic-adds into a shared output buffer
      (needed to assemble the scattered input cotangent)

Each is checked bit-/tol-exact against the native ``jax.vjp``.
"""
from autocvd import autocvd

autocvd(num_gpus=1)
# ruff: noqa: E402
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


def _f(x):
    # nonlinear, WENO-flavoured: sqrt + reciprocal + powers
    return jnp.sqrt(x * x + 1.0) * x + 1.0 / (x * x + 0.5)


# ---- (A) elementwise jax.vjp inside a kernel ------------------------------
def test_A():
    n = 256
    x = jnp.linspace(-2.0, 3.0, n)
    ct = jnp.sin(jnp.linspace(0, 6, n))

    def kernel(x_ref, ct_ref, out_ref):
        xv = x_ref[...]
        ctv = ct_ref[...]
        _, vjp_fn = jax.vjp(_f, xv)
        (xbar,) = vjp_fn(ctv)
        out_ref[...] = xbar

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n,), x.dtype),
        grid=(1,),
    )(x, ct)

    _, vjp_fn = jax.vjp(_f, x)
    (ref,) = vjp_fn(ct)
    rel = float(jnp.abs(out - ref).max() / jnp.abs(ref).max())
    print(f"(A) elementwise vjp-in-kernel: rel={rel:.2e}  "
          f"{'OK' if rel < 1e-12 else 'FAIL'}")
    return rel < 1e-12


# ---- (B) ref-gathered stencil + elementwise vjp (the real kernel shape) ---
def test_B():
    # Mimics the WENO kernel: gather neighbours via ref indexing (NOT roll),
    # then VJP only the elementwise post-gather arithmetic (which test A shows
    # lowers fine). out[i] = g(q[i-1], q[i], q[i+1]); we want the cotangent on
    # the three gathered values for each cell, returned as the own-tile qbar
    # contribution from the centre term only (full scatter tested in C).
    n = 256

    def g(qm, q0, qp):  # elementwise in the gathered scalars
        return _f(q0) + 0.5 * _f(qm) - 0.25 * _f(qp)

    q = jnp.linspace(0.1, 2.0, n)
    ct = jnp.cos(jnp.linspace(0, 5, n))

    def kernel(q_ref, ct_ref, out_ref):
        i = jnp.arange(n)
        qm = q_ref[(i - 1) % n]
        q0 = q_ref[i]
        qp = q_ref[(i + 1) % n]
        ctv = ct_ref[...]
        _, vjp_fn = jax.vjp(g, qm, q0, qp)
        gm_bar, g0_bar, gp_bar = vjp_fn(ctv)
        # assemble the full input cotangent by shifting contributions back to
        # the cells they came from (done here with modular gather, no roll)
        out_ref[...] = g0_bar + jnp.roll(gm_bar, -1) + jnp.roll(gp_bar, 1)

    try:
        out = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct((n,), q.dtype),
            grid=(1,),
        )(q, ct)
    except Exception as e:
        # roll-based reassembly may not lower; retry with the elementwise-only
        # contribution and reassemble outside the kernel.
        print(f"(B) note: in-kernel roll reassembly unavailable ({type(e).__name__}); "
              "VJP-of-elementwise still validated via B2")
        return _test_B2()

    def stencil_out(qq):
        return g(jnp.roll(qq, 1), qq, jnp.roll(qq, -1))

    _, vjp_fn = jax.vjp(stencil_out, q)
    (ref,) = vjp_fn(ct)
    rel = float(jnp.abs(out - ref).max() / jnp.abs(ref).max())
    print(f"(B) ref-gather + elementwise vjp: rel={rel:.2e}  "
          f"{'OK' if rel < 1e-10 else 'FAIL'}")
    return rel < 1e-10


def _test_B2():
    # Fallback: VJP the elementwise gather-compute, return the three per-cell
    # cotangents (no in-kernel reassembly); reassemble + compare outside.
    n = 256

    def g(qm, q0, qp):
        return _f(q0) + 0.5 * _f(qm) - 0.25 * _f(qp)

    q = jnp.linspace(0.1, 2.0, n)
    ct = jnp.cos(jnp.linspace(0, 5, n))

    def kernel(q_ref, ct_ref, m_ref, z_ref, p_ref):
        i = jnp.arange(n)
        qm = q_ref[(i - 1) % n]; q0 = q_ref[i]; qp = q_ref[(i + 1) % n]
        _, vjp_fn = jax.vjp(g, qm, q0, qp)
        gm_bar, g0_bar, gp_bar = vjp_fn(ct_ref[...])
        m_ref[...] = gm_bar; z_ref[...] = g0_bar; p_ref[...] = gp_bar

    sds = jax.ShapeDtypeStruct((n,), q.dtype)
    m, z, p = pl.pallas_call(kernel, out_shape=(sds, sds, sds), grid=(1,))(q, ct)
    out = z + jnp.roll(m, -1) + jnp.roll(p, 1)

    def stencil_out(qq):
        return g(jnp.roll(qq, 1), qq, jnp.roll(qq, -1))
    _, vjp_fn = jax.vjp(stencil_out, q)
    (ref,) = vjp_fn(ct)
    rel = float(jnp.abs(out - ref).max() / jnp.abs(ref).max())
    print(f"(B2) ref-gather + elementwise vjp (reassemble outside): "
          f"rel={rel:.2e}  {'OK' if rel < 1e-10 else 'FAIL'}")
    return rel < 1e-10


# ---- (C) atomic_add scatter into a ZERO-INITIALISED buffer ----------------
def test_C():
    n = 64
    nblk = 8
    blk = n // nblk
    src = jnp.arange(n, dtype=jnp.float64) + 1.0
    out_init = jnp.zeros(n, dtype=jnp.float64)  # alias -> starts zeroed

    def kernel(src_ref, init_ref, out_ref):
        # init_ref aliases out_ref's (zeroed) memory; we accumulate via atomics
        b = pl.program_id(0)
        idx = b * blk + jnp.arange(blk)
        pl.atomic_add(out_ref, (idx,), src_ref[idx])
        nbr = ((b * blk + blk) % n).reshape(1)
        pl.atomic_add(out_ref, (nbr,), src_ref[(b * blk).reshape(1)])

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((n,), src.dtype),
        grid=(nblk,),
        in_specs=[pl.BlockSpec(memory_space=pl.ANY),
                  pl.BlockSpec(memory_space=pl.ANY)],
        out_specs=pl.BlockSpec(memory_space=pl.ANY),
        input_output_aliases={1: 0},
    )(src, out_init)

    ref = src.copy()
    for b in range(nblk):
        nbr = (b * blk + blk) % n
        ref = ref.at[nbr].add(src[b * blk])
    rel = float(jnp.abs(out - ref).max() / jnp.abs(ref).max())
    print(f"(C) atomic_add scatter (zeroed): rel={rel:.2e}  "
          f"{'OK' if rel < 1e-12 else 'FAIL'}")
    return rel < 1e-12


def main():
    results = {}
    for name, fn in [("A", test_A), ("B", test_B), ("C", test_C)]:
        try:
            results[name] = fn()
        except Exception as e:
            print(f"({name}) RAISED: {type(e).__name__}: {str(e)[:300]}")
            results[name] = False
    print("\nSUMMARY:", {k: ("OK" if v else "FAIL") for k, v in results.items()})


if __name__ == "__main__":
    main()
