"""
Phase 0b.4 -- reverse-mode gradient gate (BLOCKING) for the 2D RBC.

Finite-difference-verify the reverse-mode gradient of a scalar terminal loss
with respect to the large-scale initial condition, through the *full* path:
finite-difference WENO + non-periodic (reflective) boundaries + external-
potential gravity + momentum viscosity + the new thermal-conduction module.
The constant-kappa explicit conduction Laplacian and the ghost-T Dirichlet
plates are linear/smooth operations, so AD should be clean; this script proves
it before any inverse modeling is attempted, and inspects gradient smoothness.

Control = the large-scale band of the initial vertical velocity (a spectral
low-pass of a field, |k| <= K_CUT). Loss = energy of the large-scale terminal
temperature anomaly. We compare jax.grad against central finite differences
along several random directions; PASS if the directional derivatives agree.
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
    BACKWARDS, FINITE_DIFFERENCE, IDEAL_GAS, NATIVE_JAX,
    REFLECTIVE_BOUNDARY, SIMPLE_SOURCE_TERM,
    BoundarySettings, BoundarySettings1D, StaticFloatVector, StaticIntVector,
    finalize_config,
)
from astronomix.time_stepping.time_integration import _time_integration
from astronomix.data_classes.simulation_helper_data import (
    _helper_data_requirements, get_helper_data as _get_helper_data,
)

# ---- problem (small, viscous -> well-conditioned, fast, FD-clean) ----
NY = 24
GAMMA_ASPECT = 2
LY = 1.0
LX = GAMMA_ASPECT * LY
GAMMA = 5.0 / 3.0
G = 0.25
T_BOT, T_TOP, P_BOT = 1.2, 0.8, 1.0
MU = 4.0e-3
KAPPA = 6.0e-3
K_CUT = 4.0
WINDOW = 1.0            # short window, << instability time -> clean FD check
C_CFL = 0.4
SEED = 0
N_FD_DIRS = 4


def hydrostatic_background(y):
    dT = T_BOT - T_TOP
    T = T_BOT - (dT / LY) * y
    p = P_BOT * (T / T_BOT) ** (G * LY / dT)
    return T, p, p / T


def make_lowpass(nx, ny, kcut):
    kx = jnp.fft.fftfreq(nx) * nx
    ky = jnp.fft.fftfreq(ny) * ny
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    mask = (jnp.sqrt(KX ** 2 + KY ** 2) <= kcut).astype(jnp.float64)

    def lowpass(field):
        return jnp.fft.ifftn(jnp.fft.fftn(field) * mask).real

    return lowpass


def main():
    nx = GAMMA_ASPECT * NY
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=IDEAL_GAS,
        dimensionality=2, backend=NATIVE_JAX,
        differentiation_mode=BACKWARDS, num_checkpoints=16,
        enforce_positivity=False, progress_bar=False,
        num_cells=StaticIntVector(nx, NY, -1),
        box_size=StaticFloatVector(LX, LY, 1.0),
        external_potential=True, self_gravity_version=SIMPLE_SOURCE_TERM,
        diffusion=True, thermal_conduction=True,
        conduction_wall_axis=1, conduction_isothermal_walls=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(),
        ),
        return_snapshots=False,
    )
    rv = get_registered_variables(config)
    helper_data = get_helper_data(config)
    coords = helper_data.geometric_centers
    x, y = coords[..., 0], coords[..., 1]

    T_bg, p_bg, rho_bg = hydrostatic_background(y)
    probe = construct_primitive_state(
        config=config, registered_variables=rv, density=rho_bg,
        velocity_x=jnp.zeros_like(x), velocity_y=jnp.zeros_like(x),
        gas_pressure=p_bg,
    )
    config = finalize_config(config, probe.shape)

    requirements = _helper_data_requirements(config)
    hd_pad = _get_helper_data(config, None, padded=False, requirements=requirements)

    params = SimulationParams(
        C_cfl=C_CFL, gamma=GAMMA, t_end=WINDOW, viscosity=MU,
        thermal_conductivity=KAPPA, wall_temperature_low=T_BOT,
        wall_temperature_high=T_TOP, gravitational_potential=G * y,
    )

    lowpass = make_lowpass(nx, NY, K_CUT)
    T_mean = 0.5 * (T_BOT + T_TOP)
    v_ff = float(np.sqrt(G * (T_BOT - T_TOP) / T_mean * LY))

    def build_ic(ctrl):
        vy = lowpass(ctrl)
        return construct_primitive_state(
            config=config, registered_variables=rv, density=rho_bg,
            velocity_x=jnp.zeros_like(x), velocity_y=vy, gas_pressure=p_bg,
        )

    def loss(ctrl):
        s0 = build_ic(ctrl)
        sT = _time_integration(s0, config, params, rv, hd_pad)
        T = sT[rv.pressure_index] / sT[rv.density_index]
        anom_large = lowpass(T - T_mean)
        return jnp.sum(anom_large ** 2)

    # control: a small large-scale vy seed
    rng = np.random.default_rng(SEED)
    ctrl0 = jnp.asarray(0.05 * v_ff * rng.standard_normal((nx, NY)))
    ctrl0 = lowpass(ctrl0)

    print(f"=== gradient gate: 2D RBC {nx}x{NY}, window {WINDOW} "
          f"({WINDOW / (LY / v_ff):.2f} tau_ff) ===", flush=True)
    loss_jit = jax.jit(loss)
    vg_jit = jax.jit(jax.value_and_grad(loss))
    L0, g = vg_jit(ctrl0)
    L0 = float(L0)
    g = np.asarray(g)
    print(f"loss = {L0:.6e}, |grad| = {np.linalg.norm(g):.6e}", flush=True)

    # gradient smoothness: spectral content of the gradient field
    gk = np.abs(np.fft.fft2(g))
    hi = gk[(np.fft.fftfreq(nx)[:, None] ** 2 * nx ** 2 +
             np.fft.fftfreq(NY)[None, :] ** 2 * NY ** 2) > K_CUT ** 2].sum()
    lo = gk.sum() - hi
    print(f"gradient spectral content: low-k {lo / gk.sum():.3f}, "
          f"high-k {hi / gk.sum():.3f}  (smooth if low-k dominates)")

    # finite-difference directional check
    # Directional check.  The accuracy of an AD directional derivative is set
    # by the gradient *scale*, so the meaningful error metric is
    # |FD - AD| / |grad| (a per-direction relative error blows up spuriously on
    # directions nearly orthogonal to the gradient, where the true derivative
    # ~ 0 yet the absolute error is still only O(eps^2 + fp)).
    print("\ndirectional finite-difference check (central, eps=1e-4):", flush=True)
    eps = 1e-4
    gnorm = float(np.linalg.norm(g))
    max_norm_err = 0.0
    for d_idx in range(N_FD_DIRS):
        d = rng.standard_normal((nx, NY))
        d = lowpass(jnp.asarray(d))
        d = np.asarray(d) / np.linalg.norm(np.asarray(d))
        dj = jnp.asarray(d)
        Lp = float(loss_jit(ctrl0 + eps * dj))
        Lm = float(loss_jit(ctrl0 - eps * dj))
        fd = (Lp - Lm) / (2 * eps)
        ad = float(np.sum(g * d))
        abs_err = abs(fd - ad)
        norm_err = abs_err / gnorm
        max_norm_err = max(max_norm_err, norm_err)
        print(f"  dir {d_idx}: FD={fd:+.6e}  AD={ad:+.6e}  "
              f"|FD-AD|={abs_err:.2e}  |FD-AD|/|grad|={norm_err:.2e}", flush=True)

    ok = max_norm_err < 1e-4
    print(f"\nmax |FD-AD|/|grad| = {max_norm_err:.2e}  (gate threshold 1e-4)")
    print("RESULT:", "PASS" if ok else "FAIL")
    assert ok, "gradient gate failed -- AD through the RBC path is not clean"


if __name__ == "__main__":
    main()
