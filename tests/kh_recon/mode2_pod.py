"""
Path-A reduced-order multiple shooting: build the low-D basis the interior
segment-states will live in. The whole reason terminal-only MS failed before was
underdetermined full-field interior states; here we confine them to a POD basis
of the KH flow manifold so the lifted problem is DETERMINED and MS buys purely
the mode-2 (basin) advantage.

Basis is TRUTH-AGNOSTIC: an ensemble of forward runs from RANDOM 2-mode seeds
(kx=1 subharmonic + kx=2 fundamental, random amps/phases), snapshotted at the
M-1 interior boundary times t_j=j*T/M, with the unperturbed (zero-seed) trajectory
subtracted as the time-evolved reference. POD = SVD of those deviations.

Env: KH_N, KH_RE, KH_TREC (t_g), KH_M, KH_SEEDAMP, KH_KENS (ensemble size),
     KH_R (kept POD modes). Saves data/mode2_pod_M{M}.npz.
"""
# ==== GPU selection ====
import os, sys
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================
import jax, jax.numpy as jnp, numpy as np
import problem as P
from astronomix import time_integration

N = int(os.environ.get("KH_N", 64))
RE = float(os.environ.get("KH_RE", 2000))
TREC = float(os.environ.get("KH_TREC", 65))
M_ = int(os.environ.get("KH_M", 4))
SEEDAMP = float(os.environ.get("KH_SEEDAMP", 3e-2))
KENS = int(os.environ.get("KH_KENS", 16))
R = int(os.environ.get("KH_R", 28))


def two_mode_seed(khp, X, Y, c):   # c=(a2cos,a2sin,a1cos,a1sin)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    svy = (c[0] * jnp.cos(2 * jnp.pi * 2 * X) + c[1] * jnp.sin(2 * jnp.pi * 2 * X)
           + c[2] * jnp.cos(2 * jnp.pi * 1 * X) + c[3] * jnp.sin(2 * jnp.pi * 1 * X)) * env
    svy = khp.seed_amp * khp.dV * svy / jnp.sqrt(jnp.mean(svy ** 2) + 1e-30)
    return jnp.stack([jnp.zeros_like(svy), svy])


def main():
    khp = P.KHParams(n=N, reynolds=RE, seed_amp=SEEDAMP, k_min=1, k_max=2)
    T = TREC * khp.growth_time
    h = T / M_
    cfg, par = P.make_config_params(khp, h, snapshots=0, backward=False)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0p = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0p.shape)

    def seg(s):
        return time_integration(s, cfg, par._replace(t_end=h), rv)

    def interior_states(seed):
        s = P.build_state(seed, khp, cfg, rv, X, Y)
        outs = []
        for _ in range(M_ - 1):
            s = seg(s); outs.append(s)
        return outs                       # states at t_1..t_{M-1}

    # time-evolved reference (zero seed) at the interior times
    ref = interior_states(jnp.zeros((2, khp.n, khp.n)))
    ref = [np.asarray(r) for r in ref]

    # ensemble of random 2-mode seeds -> deviations from ref at interior times
    devs = []
    key = jax.random.PRNGKey(7)
    for k in range(KENS):
        key, sub = jax.random.split(key)
        c = jax.random.normal(sub, (4,))
        states = interior_states(two_mode_seed(khp, X, Y, c))
        for j in range(M_ - 1):
            devs.append(np.asarray(states[j]).ravel() - ref[j].ravel())
    Dm = np.stack(devs)                   # (KENS*(M-1), statesize)
    # POD via SVD of the deviation matrix (rows = samples)
    U, s, Vt = np.linalg.svd(Dm, full_matrices=False)
    r = min(R, Vt.shape[0])
    phi = Vt[:r]                          # (r, statesize) orthonormal spatial modes
    sshape = ref[0].shape
    np.savez(os.path.join(os.path.dirname(__file__), f"data/mode2_pod_M{M_}.npz"),
             phi=phi, sval=s[:r], ref=np.stack(ref), sshape=np.array(sshape),
             T=T, h=h, M=M_, n=N, re=RE, seedamp=SEEDAMP, trec=TREC)
    ev = s ** 2 / np.sum(s ** 2)
    print(f"POD M={M_} N={N} T={TREC} ens={KENS}: r={r}, captured var "
          f"top{r}={np.sum(ev[:r]):.4f}, sv1..5={s[:5]/s[0]}")
    print(f"-> data/mode2_pod_M{M_}.npz  statesize={Dm.shape[1]} samples={Dm.shape[0]}")


if __name__ == "__main__":
    main()
