"""
Mode-2 advantage test, Path A. Unknown = 2-mode seed (kx=2 fundamental + kx=1
subharmonic, 4 DOF), horizon T* spanning a vortex-PAIRING event -> the IC->terminal
map folds distinct seeds onto similar finals (multimodal, mode 2) while staying
identifiable (mode 3 generous). Compare:
  single : optimize the 4-DOF IC, integrate full T  (reduced space)
  redms  : IC + interior segment states confined to the POD basis (mode2_pod.npz),
           soft full-field continuity  (reduced-order multiple shooting)
Multi-restart (KH_SEED) exposes the basin structure: single should SCATTER across
basins; redms should collapse to truth. mode-1 clipped in both -> the difference
is the basin (mode 2).

Env: KH_MODE in {single,redms}, KH_M, KH_MU, KH_SEED (restart), KH_STEPS, KH_LR,
     KH_NOISE, KH_N, KH_RE, KH_TREC, KH_SEEDAMP, KH_OUT.
"""
# ==== GPU selection ====
import os, sys
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ruff: noqa: E402
# =======================
import time
import jax, jax.numpy as jnp, numpy as np, optax
import problem as P
from astronomix import time_integration

MODE = os.environ.get("KH_MODE", "redms")           # single | redms
M_ = 1 if MODE == "single" else int(os.environ.get("KH_M", 4))
MU = float(os.environ.get("KH_MU", 30.0))
RESTART = int(os.environ.get("KH_SEED", 0))
STEPS = int(os.environ.get("KH_STEPS", 250))
LR = float(os.environ.get("KH_LR", 3e-3))
NOISE = float(os.environ.get("KH_NOISE", 1e-2))
N = int(os.environ.get("KH_N", 64))
RE = float(os.environ.get("KH_RE", 2000))
TREC = float(os.environ.get("KH_TREC", 65))
SEEDAMP = float(os.environ.get("KH_SEEDAMP", 3e-2))
OUT = os.environ.get("KH_OUT", f"data/mode2_{MODE}_s{RESTART}.npz")

# truth 2-mode seed direction (raw coeffs a2cos,a2sin,a1cos,a1sin); scaled to rms below
C_TRUTH = np.array([1.0, 0.0, 1.5, 0.6])


def main():
    khp = P.KHParams(n=N, reynolds=RE, seed_amp=SEEDAMP, k_min=1, k_max=2)
    T = TREC * khp.growth_time
    h = T / M_
    cfg, par = P.make_config_params(khp, h, snapshots=0, backward=True)
    rv = P.get_registered_variables(cfg)
    X, Y = P.coords(khp, cfg)
    s0p = P.build_state(jnp.zeros((2, khp.n, khp.n)), khp, cfg, rv, X, Y)
    cfg = P.finalize_config(cfg, s0p.shape)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    b2c, b2s = jnp.cos(2 * jnp.pi * 2 * X) * env, jnp.sin(2 * jnp.pi * 2 * X) * env
    b1c, b1s = jnp.cos(2 * jnp.pi * 1 * X) * env, jnp.sin(2 * jnp.pi * 1 * X) * env

    def seed_of(c):                       # raw (no renorm): 4 DOF -> (2,Nx,Ny)
        svy = c[0] * b2c + c[1] * b2s + c[2] * b1c + c[3] * b1s
        return jnp.stack([jnp.zeros_like(svy), svy])

    # scale truth to target rms = seed_amp*dV
    c_t = jnp.asarray(C_TRUTH)
    rms = float(jnp.sqrt(jnp.mean(seed_of(c_t)[1] ** 2)))
    c_t = c_t * (SEEDAMP * khp.dV / rms)
    truth_seed = seed_of(c_t)

    def seg(s):
        return time_integration(s, cfg, par._replace(t_end=h), rv)

    s = P.build_state(truth_seed, khp, cfg, rv, X, Y)
    for _ in range(M_ - 1):
        s = seg(s)
    obs_clean = P.velocity_of(seg(s), rv)
    bs = P.build_state(jnp.zeros_like(truth_seed), khp, cfg, rv, X, Y)
    for _ in range(M_):
        bs = seg(bs)
    pert_rms = float(jnp.sqrt(jnp.mean((obs_clean - P.velocity_of(bs, rv)) ** 2)))
    nsd = NOISE * pert_rms
    obs = obs_clean + nsd * jax.random.normal(jax.random.PRNGKey(12345), obs_clean.shape)

    # POD reduced interior basis
    if M_ > 1:
        pod = np.load(os.path.join(os.path.dirname(__file__), f"data/mode2_pod_M{M_}.npz"))
        phi = jnp.asarray(pod["phi"]); ref = jnp.asarray(pod["ref"])  # (r,SS),(M-1,*sshape)
        sval = jnp.asarray(pod["sval"])
        sshape = tuple(int(x) for x in pod["sshape"]); r = phi.shape[0]
        # scale modes by singular values so the optimized coeffs b are O(1)
        # (raw phi-coeffs are O(sval)~O(10-100) -> Adam at lr=3e-3 can't move them)
        psi = phi * sval[:, None]

        def interior_of(b):               # b (M-1,r) O(1) -> list of states
            return [ref[j] + (b[j] @ psi).reshape(sshape) for j in range(M_ - 1)]

    def ic_err(c):
        return float(jnp.linalg.norm(seed_of(c) - truth_seed) / jnp.linalg.norm(truth_seed))

    def loss(theta):
        s0 = P.build_state(seed_of(theta["c"]), khp, cfg, rv, X, Y)
        if M_ == 1:
            data = 0.5 * jnp.mean((P.velocity_of(seg(s0), rv) - obs) ** 2)
            return data, data
        starts = [s0] + interior_of(theta["b"])
        finals = [seg(st) for st in starts]
        data = 0.5 * jnp.mean((P.velocity_of(finals[-1], rv) - obs) ** 2)
        defect = sum(jnp.mean((finals[j] - starts[j + 1]) ** 2) for j in range(M_ - 1))
        return data + 0.5 * MU * defect, data

    # cold restart init for the IC; interior warm-started from it (projected onto POD)
    c0 = jnp.asarray(0.5 * SEEDAMP * khp.dV * np.random.default_rng(RESTART).standard_normal(4))
    theta = {"c": c0}
    if M_ > 1:
        s = P.build_state(seed_of(c0), khp, cfg, rv, X, Y); bs0 = []
        for j in range(M_ - 1):
            s = seg(s); bs0.append((phi @ (s - ref[j]).reshape(-1)) / sval)  # b'=a/sval ~O(1)
        theta["b"] = jnp.stack(bs0)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(LR))
    st = opt.init(theta); vg = jax.value_and_grad(loss, has_aux=True)

    @jax.jit
    def step(theta, st):
        (val, data), g = vg(theta); upd, st = opt.update(g, st)
        return optax.apply_updates(theta, upd), st, val, data

    t0 = time.time()
    for it in range(STEPS):
        theta, st, val, data = step(theta, st)
        if it % 25 == 0 or it == STEPS - 1:
            print(f"  {MODE} M={M_} s{RESTART} it={it:3d}: J={float(val):.3e} "
                  f"data={float(data):.3e} ic_err={ic_err(theta['c']):.3f}")
    err = ic_err(theta["c"]); rt = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    np.savez(OUT, mode=MODE, M=M_, mu=MU, restart=RESTART, ic_err=err,
             c_rec=np.asarray(theta["c"]), c_truth=np.asarray(c_t), runtime=rt,
             final_data=float(data), pert_rms=pert_rms)
    print(f"[done] {MODE} M={M_} s{RESTART}: ic_err={err:.3f} data={float(data):.2e} "
          f"c_rec={np.asarray(theta['c'])} {rt:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
