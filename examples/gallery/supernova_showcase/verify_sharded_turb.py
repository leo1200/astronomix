"""Is the sharded turbulent field the same field as the unsharded one?

``_common.turbulent_field`` now forwards ``sharding`` to ``create_turb_field``,
which is what let the 1024^3 initial condition build at all -- the field was
previously a single-device array and replicated the whole cube onto one device
the moment it multiplied a sharded density.

That fix is only worth having if the sharded field is the SAME field. It is
built with an inverse FFT, which needs all-to-all communication once the array
is distributed, so "it ran and produced numbers" is not evidence of anything:
a wrong distributed transform returns numbers too. A 768^3 initial condition
came out full of NaNs on 4 GPUs while 256^3 on 2 GPUs was fine, which is
exactly the signature of a transform that is right for some decompositions and
not others.

This compares the two directly -- same key, same band, same process, same
devices -- so any difference is the sharding and nothing else.

Usage (GPU)::

    ./run.sh verify_sharded_turb.py --gpus 2 --n 256
    ./run.sh verify_sharded_turb.py --gpus 4 --n 768      # the failing case
"""

# ==== GPU selection ====
import os
import sys
_NUM_GPUS = 2
if "--gpus" in sys.argv:
    _NUM_GPUS = int(sys.argv[sys.argv.index("--gpus") + 1])
if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=_NUM_GPUS)
# ruff: noqa: E402
# =======================

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from _common import turbulent_field


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=256, help="cells per axis")
    ap.add_argument("--gpus", type=int, default=2, help="devices to shard over")
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=20)
    args = ap.parse_args()

    from jax.sharding import PartitionSpec as P, AxisType
    mesh = jax.make_mesh((args.gpus,), ("x",), axis_types=(AxisType.Auto,))
    sharding = jax.sharding.NamedSharding(mesh, P("x", None, None))
    jax.config.update("jax_use_shardy_partitioner", False)

    key = jax.random.PRNGKey(7)
    kw = dict(kmin=args.kmin, kmax=args.kmax, slope=-1.0)

    print(f"[verify] {args.n}^3 over {args.gpus} device(s), k = "
          f"{args.kmin}-{args.kmax}")

    plain = turbulent_field(args.n, key, sharding=None, **kw)
    plain = np.asarray(jax.block_until_ready(plain))
    print(f"[verify] unsharded: finite={np.all(np.isfinite(plain))}, "
          f"mean={plain.mean():+.3e}, std={plain.std():.6f}, "
          f"range=[{plain.min():+.3f}, {plain.max():+.3f}]")

    shard = turbulent_field(args.n, key, sharding=sharding, **kw)
    shard = np.asarray(jax.block_until_ready(shard))
    print(f"[verify] sharded:   finite={np.all(np.isfinite(shard))}, "
          f"mean={shard.mean():+.3e}, std={shard.std():.6f}, "
          f"range=[{shard.min():+.3f}, {shard.max():+.3f}]")

    if not np.all(np.isfinite(shard)):
        n_bad = int(np.sum(~np.isfinite(shard)))
        print(f"[verify] FAIL: the sharded field has {n_bad} non-finite cells "
              f"({100 * n_bad / shard.size:.2f} %). The distributed transform "
              f"is wrong at this decomposition; do NOT use sharded clumping.")
        raise SystemExit(1)

    d = np.abs(shard - plain)
    rel = d.max() / max(plain.std(), 1e-30)
    print(f"[verify] max|sharded - unsharded| = {d.max():.3e} "
          f"({rel:.3e} of the field's own sigma)")
    # A correct distributed FFT differs only by float reassociation, which is
    # parts in 1e-6 for float32. Anything larger is a different field, not a
    # rounding difference, and would silently change the initial condition.
    if rel > 1e-4:
        print("[verify] FAIL: that is not a rounding difference. The sharded "
              "field is a DIFFERENT realisation, so multi-GPU runs would carry "
              "a different initial condition from single-GPU ones.")
        raise SystemExit(1)
    print("[verify] PASS: the sharded and unsharded fields agree to rounding.")


if __name__ == "__main__":
    main()
