"""
Gradient-accuracy probe: is the grad-direction FD/AD ~2x mismatch a WENO KINK
(piecewise-smooth solver -> valid one-sided subgradient, expected, proceed) or
a BUG (fix)?

For a short horizon we compute the AD directional derivative along the gradient
direction and along a random direction, then:
  * an eps-scan of the CENTRAL difference (a kink stays ~constant ratio as
    eps->0; a nonlinearity converges to AD);
  * the one-sided LEFT and RIGHT slopes (a kink shows left != right, with AD
    equal to one of them and central = their average).
Also reports the effect of higher viscosity (smoother solution -> smaller kink).
"""

# ==== GPU selection ====
import os, sys
from autocvd import autocvd
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp
import problem as P


def probe(khp, Tg, label):
    T = Tg * khp.growth_time
    cfg, par = P.make_config_params(khp, T, snapshots=0, backward=True)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0 = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0.shape)
    seed = P.random_broadband_seed(jax.random.PRNGKey(0), khp, X, Y)

    def loss(s):
        v = P.velocity_of(P.forward(s, khp, cfg, par, rv, X, Y), rv)
        return 0.5 * jnp.mean(v[1] ** 2)

    g = jax.grad(loss)(seed)
    gn = float(jnp.sqrt(jnp.sum(g ** 2)))
    ghat = g / (gn + 1e-30)
    rdir = jnp.zeros_like(seed).at[1].set(
        jax.random.normal(jax.random.PRNGKey(7), (khp.n, khp.n)))
    rdir = rdir / jnp.sqrt(jnp.sum(rdir ** 2))
    L0 = float(loss(seed))
    print(f"\n[{label}] Re={khp.reynolds:.0f} T={T:.3f}: loss={L0:.3e} ||g||={gn:.3e}")
    for name, d in [("grad-dir", ghat), ("rand-dir", rdir)]:
        ad = float(jnp.sum(g * d))
        print(f"  {name}: AD={ad:.4e}")
        for eps in (3e-2, 1e-2, 3e-3, 1e-3, 3e-4):
            lp = float(loss(seed + eps * d)); lm = float(loss(seed - eps * d))
            cen = (lp - lm) / (2 * eps)
            left = (L0 - lm) / eps; right = (lp - L0) / eps
            print(f"    eps={eps:.0e}: central={cen:+.3e} (rel {abs(cen-ad)/(abs(ad)+1e-30):.2f}) "
                  f"| left={left:+.3e} right={right:+.3e}")


def main():
    for Re in (2000.0, 500.0):
        probe(P.KHParams(n=192, reynolds=Re), Tg=8, label=f"Re{Re:.0f}")


if __name__ == "__main__":
    main()
