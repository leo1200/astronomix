"""
Gradient gate for the rotating-isothermal-turbulence inverse problem.

Finite-difference-verify the reverse-mode gradient of a scalar terminal loss on
the **barotropic columnar mode** P_large(x(T)) w.r.t. the large-scale columnar
initial condition, through the *full* differentiable path:

    FD WENO + periodic + isothermal EOS + momentum viscosity + Coriolis
    (-2 Omega zhat x rho u) + OU forcing,

with a **fixed timestep** so the OU forcing realisation is identical for the
perturbed and unperturbed rollouts (a twin-experiment requirement, and what
makes the FD check meaningful).

  P_large = horizontal low-pass (|k_perp| <= k_c) of the vertical average
            (z-invariant projection) of the horizontal velocity  ->  the slow,
            barotropic, z-invariant columnar flow.

Coriolis is linear/local and OU forcing is state-independent, so AD should be
clean; this proves it before the SS-vs-MS study.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from astronomix import (
    SimulationConfig, SimulationParams,
    get_registered_variables, get_helper_data,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FINITE_DIFFERENCE, ISOTHERMAL, NATIVE_JAX, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, finalize_config,
)
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements, get_helper_data as _get_helper_data,
)

N = 32                # now feasible: checkpointed fixed-timestep reverse mode
L = 1.0
CS = 1.0
RHO0 = 1.0
NU = 1.0e-3
OMEGA = 4.0
KF_MODE = 3.0
TAU_F = 1.0
F0 = 1.0
K_CUT = 2.0           # barotropic horizontal low-pass cutoff (mode)
WINDOW = 0.5
NSTEPS = 80           # fixed timesteps (dt = WINDOW / NSTEPS, CFL-stable)
NUM_CHECKPOINTS = 16
N_FD_DIRS = 4
SEED = 0


def make_lowpass2d(nx, ny, kcut):
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    mask = (jnp.sqrt(KX ** 2 + KY ** 2) <= kcut).astype(jnp.float64)
    return lambda f: jnp.fft.ifftn(jnp.fft.fftn(f) * mask).real


def main():
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=ISOTHERMAL, mhd=False,
        dimensionality=3, num_cells=N, box_size=L, backend=NATIVE_JAX,
        differentiation_mode=BACKWARDS, enforce_positivity=False, progress_bar=False,
        fixed_timestep=True, num_timesteps=NSTEPS, num_checkpoints=NUM_CHECKPOINTS,
        diffusion=True, rotation=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=True),
        return_snapshots=False,
    )
    rv = get_registered_variables(config)

    density = RHO0 * jnp.ones((N, N, N))
    zero = jnp.zeros_like(density)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=density,
        velocity_x=zero, velocity_y=zero, velocity_z=zero)
    config = finalize_config(config, probe.shape)
    requirements = _helper_data_requirements(config)
    hd_pad = _get_helper_data(config, None, padded=False, requirements=requirements)

    params = SimulationParams(
        gamma=5.0 / 3.0, isothermal_sound_speed=CS, t_end=WINDOW, viscosity=NU,
        rotation_rate=OMEGA,
        turbulent_forcing_params=TurbulentForcingParams(
            correlation_time=TAU_F, forcing_wavenumber=2.0 * np.pi * KF_MODE / L,
            forcing_amplitude=F0),
    )

    lowpass = make_lowpass2d(N, N, K_CUT)
    vxi, vyi = rv.velocity_index.x, rv.velocity_index.y

    def build_ic(ctrl):
        # ctrl: (2, N, N) barotropic horizontal velocity (z-invariant columns)
        vx = jnp.broadcast_to(lowpass(ctrl[0])[:, :, None], (N, N, N))
        vy = jnp.broadcast_to(lowpass(ctrl[1])[:, :, None], (N, N, N))
        return construct_primitive_state(
            config=config, registered_variables=rv, density=density,
            velocity_x=vx, velocity_y=vy, velocity_z=zero)

    def p_large(state):
        # vertical average (barotropic) then horizontal low-pass
        vbar_x = state[vxi].mean(axis=2)
        vbar_y = state[vyi].mean(axis=2)
        return jnp.stack([lowpass(vbar_x), lowpass(vbar_y)])

    def loss(ctrl):
        s0 = build_ic(ctrl)
        sT = _time_integration(s0, config, params, rv, hd_pad)
        return jnp.sum(p_large(sT) ** 2)

    rng = np.random.default_rng(SEED)
    v_amp = 0.2
    ctrl0 = jnp.stack([
        jnp.asarray(lowpass(jnp.asarray(v_amp * rng.standard_normal((N, N))))),
        jnp.asarray(lowpass(jnp.asarray(v_amp * rng.standard_normal((N, N))))),
    ])

    print(f"=== rotating-turbulence gradient gate: {N}^3, window {WINDOW} "
          f"({NSTEPS} fixed steps), Omega={OMEGA}, k_f={KF_MODE} ===", flush=True)
    loss_jit = jax.jit(loss)
    vg_jit = jax.jit(jax.value_and_grad(loss))
    L0, g = vg_jit(ctrl0)
    L0 = float(L0); g = np.asarray(g)
    gnorm = float(np.linalg.norm(g))
    print(f"loss = {L0:.6e}, |grad| = {gnorm:.6e}", flush=True)

    print("\ndirectional finite-difference check (central, eps=1e-4):", flush=True)
    eps = 1e-4
    max_norm_err = 0.0
    for d_idx in range(N_FD_DIRS):
        d = np.stack([np.asarray(lowpass(jnp.asarray(rng.standard_normal((N, N))))),
                      np.asarray(lowpass(jnp.asarray(rng.standard_normal((N, N)))))])
        d = d / np.linalg.norm(d)
        dj = jnp.asarray(d)
        Lp = float(loss_jit(ctrl0 + eps * dj))
        Lm = float(loss_jit(ctrl0 - eps * dj))
        fd = (Lp - Lm) / (2 * eps)
        ad = float(np.sum(g * d))
        ne = abs(fd - ad) / gnorm
        max_norm_err = max(max_norm_err, ne)
        print(f"  dir {d_idx}: FD={fd:+.6e}  AD={ad:+.6e}  |FD-AD|/|grad|={ne:.2e}",
              flush=True)

    ok = max_norm_err < 1e-4
    print(f"\nmax |FD-AD|/|grad| = {max_norm_err:.2e}  (gate threshold 1e-4)")
    print("RESULT:", "PASS" if ok else "FAIL")
    assert ok, "rotating-turbulence gradient gate failed"


if __name__ == "__main__":
    main()
