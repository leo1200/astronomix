"""
3D compressible Rayleigh-Benard convection -- forward problem, for visual
appeal (coexisting large-scale circulation + fine turbulent plume structure).

Same physics as the 2D ``rayleigh_benard.py`` but in 3D and at higher Ra so
that a turbulent plume network (the fine structure) coexists with a large-scale
circulation (the large structure):

  * finite-difference WENO, Pallas (Triton) backend for speed;
  * box Lx x Ly x Lz = 2 x 1 x 2 (Gamma = 2 horizontally), vertical = y;
  * PERIODIC horizontal (x, z), isothermal no-slip plates at top/bottom (y)
    -- the standard turbulent-RBC setup: clean plumes, no side-wall artefacts;
  * hot bottom plate / cold top plate via thermal conduction (Dirichlet ghost-T);
  * constant gravity g in -y via a linear external potential phi = g y
    (SIMPLE_SOURCE_TERM coupling -> keeps the fused Pallas fast path);
  * momentum viscosity ON + thermal conduction ON, Pr ~ 1.

Pallas block shape: with the non-periodic plates the state carries 6 ghost
cells, so the padded dims are N+12.  12 is divisible by 4 but not 8, hence
``pallas_block_shape = (4, 4, 4)`` (dims multiples of 4) rather than the
(4, 4, 8) used for the fully periodic Taylor-Green box.

Visual outputs: vertical & near-plate horizontal slices, a slice animation, and
an alpha-composited volume projection of the temperature anomaly (the plumes).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import time as _time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    PALLAS,
    PERIODIC_BOUNDARY,
    REFLECTIVE_BOUNDARY,
    SIMPLE_SOURCE_TERM,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
    StaticFloatVector,
    StaticIntVector,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration


# ---------------------------------------------------------------------------
#  Problem parameters
# ---------------------------------------------------------------------------
NY = 48                       # vertical cells; Nx = Nz = GAMMA * NY
GAMMA_ASPECT = 2
LY = 1.0
LX = LZ = GAMMA_ASPECT * LY

GAMMA = 5.0 / 3.0
G = 0.25
T_BOT = 1.2
T_TOP = 0.8
P_BOT = 1.0

MU = 2.9e-4                   # dynamic viscosity   -> Ra ~ 5e5, Pr ~ 1
KAPPA = 4.3e-4                # thermal conductivity

T_END = 50.0
NUM_SNAPSHOTS = 30
C_CFL = 0.4
SEED_AMP = 1e-2

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def hydrostatic_background(y):
    dT = T_BOT - T_TOP
    T = T_BOT - (dT / LY) * y
    exponent = G * LY / dT
    p = P_BOT * (T / T_BOT) ** exponent
    rho = p / T
    return T, p, rho


def build_initial_state(config, registered_variables, helper_data):
    coords = helper_data.geometric_centers          # (Nx, Ny, Nz, 3)
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    T_bg, p_bg, rho_bg = hydrostatic_background(y)

    v_ff = np.sqrt(G * (T_BOT - T_TOP) / ((T_BOT + T_TOP) / 2.0) * LY)
    rng = np.random.default_rng(0)
    vy = jnp.zeros_like(x)
    for nx_ in (1, 2, 3):
        for nz_ in (1, 2, 3):
            ph = float(rng.uniform(0, 2 * np.pi))
            amp = float(rng.uniform(0.3, 1.0))
            vy = vy + amp * jnp.sin(np.pi * y / LY) * jnp.cos(
                2.0 * np.pi * nx_ * x / LX + ph
            ) * jnp.cos(2.0 * np.pi * nz_ * z / LZ + ph)
    vy = SEED_AMP * v_ff * vy / jnp.max(jnp.abs(vy))

    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_bg,
        velocity_x=jnp.zeros_like(x),
        velocity_y=vy,
        velocity_z=jnp.zeros_like(x),
        gas_pressure=p_bg,
    )
    return state, x, y, z


def volume_render(theta, extent_h, extent_v, axis=2):
    """Cheap front-to-back alpha-composited volume render of a temperature
    anomaly field ``theta`` (Nx, Ny, Nz), integrating along ``axis`` (default z).

    Hot anomalies -> warm/red, cold -> cool/blue; opacity grows with |theta|.
    Returns an RGB image suitable for imshow with origin='lower'.
    """
    th = np.moveaxis(theta, axis, 0)          # (depth, A, B)
    depth = th.shape[0]
    scale = np.percentile(np.abs(theta), 99) + 1e-12
    t = np.clip(th / scale, -1.0, 1.0)
    # per-voxel colour (warm for hot, cool for cold)
    r = np.clip(0.5 + 0.5 * t, 0, 1)
    b = np.clip(0.5 - 0.5 * t, 0, 1)
    g = np.clip(0.5 - 0.5 * np.abs(t), 0, 1)
    col = np.stack([r, g, b], axis=-1)        # (depth, A, B, 3)
    alpha = (np.abs(t) ** 1.5) * 0.6          # opacity per voxel

    out = np.zeros(th.shape[1:] + (3,))
    trans = np.ones(th.shape[1:])             # remaining transparency
    for d in range(depth):
        a = alpha[d]
        out += (trans * a)[..., None] * col[d]
        trans *= (1.0 - a)
    # composite over white background
    out += trans[..., None] * 1.0
    return np.clip(np.transpose(out, (1, 0, 2)), 0, 1)  # (B, A, 3) -> imshow


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    nx = nz = GAMMA_ASPECT * NY
    dT = T_BOT - T_TOP
    T_mean = 0.5 * (T_BOT + T_TOP)
    nu_kin = MU / 1.0
    chi = (GAMMA - 1.0) * KAPPA / 1.0
    Ra = G * (dT / T_mean) * LY ** 3 / (nu_kin * chi)
    Pr = nu_kin / chi
    cs_mean = float(np.sqrt(GAMMA * T_mean))
    v_ff = float(np.sqrt(G * (dT / T_mean) * LY))
    print(f"=== 3D Rayleigh-Benard, FD+Pallas, {nx}x{NY}x{nz}, Gamma={GAMMA_ASPECT} ===")
    print(f"Ra ~ {Ra:.2e}, Pr ~ {Pr:.2f}, Ma ~ {v_ff / cs_mean:.3f}, "
          f"cells = {nx * NY * nz / 1e6:.2f}M")

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        dimensionality=3,
        backend=PALLAS,
        pallas_block_shape=(4, 4, 4),
        pallas_use_triton=True,
        pallas_interpret=False,
        num_cells=StaticIntVector(nx, NY, nz),
        box_size=StaticFloatVector(LX, LY, LZ),
        progress_bar=True,
        external_potential=True,
        self_gravity_version=SIMPLE_SOURCE_TERM,   # fused Pallas fast path
        diffusion=True,
        thermal_conduction=True,
        conduction_wall_axis=1,                    # y vertical
        conduction_isothermal_walls=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),    # x
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),  # y plates
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),    # z
        ),
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    state, x, y, z = build_initial_state(config, registered_variables, helper_data)
    config = finalize_config(config, state.shape)

    params = SimulationParams(
        C_cfl=C_CFL,
        gamma=GAMMA,
        t_end=T_END,
        viscosity=MU,
        thermal_conductivity=KAPPA,
        wall_temperature_low=T_BOT,
        wall_temperature_high=T_TOP,
        gravitational_potential=G * y,
    )
    assert config.gravity

    t0 = _time.time()
    result = time_integration(state, config, params, registered_variables)
    states = result.states
    states.block_until_ready() if hasattr(states, "block_until_ready") else None
    time_points = np.asarray(result.time_points)
    n_snap = states.shape[0]
    print(f"Integrated {n_snap} snapshots to t = {time_points[-1]:.2f} "
          f"in {_time.time() - t0:.1f} s wall")

    rho = np.asarray(states[:, registered_variables.density_index])
    vx = np.asarray(states[:, registered_variables.velocity_index.x])
    vy = np.asarray(states[:, registered_variables.velocity_index.y])
    vz = np.asarray(states[:, registered_variables.velocity_index.z])
    p = np.asarray(states[:, registered_variables.pressure_index])
    T = p / rho

    if np.any(np.isnan(rho)):
        print("WARNING: NaNs encountered in the run!")

    Ek = 0.5 * np.mean(rho * (vx ** 2 + vy ** 2 + vz ** 2), axis=(1, 2, 3))
    Nu = 1.0 + np.mean(vy * T, axis=(1, 2, 3)) / (chi * dT / LY)
    jmid = NY // 2
    lsc = np.sqrt(np.mean(vx[:, :, jmid, :] ** 2, axis=(1, 2)))

    print(f"Final Nu = {Nu[-1]:.2f}, time-mean Nu (2nd half) = "
          f"{np.mean(Nu[n_snap // 2:]):.2f}")
    print(f"Final E_k = {Ek[-1]:.3e}, LSC amplitude = {lsc[-1]:.3e} "
          f"(v_ff = {v_ff:.3e})")

    # downsampled final field for re-plotting later
    np.savez(DATA_DIR / f"rbc3d_N{NY}.npz",
             time_points=time_points, Ek=Ek, Nu=Nu, lsc=lsc,
             T_final=T[-1].astype(np.float32))

    x_np, y_np, z_np = np.asarray(x), np.asarray(y), np.asarray(z)

    # --- time series ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, ttl in zip(axes, (Ek, Nu, lsc),
                             ("Kinetic energy", "Nusselt number",
                              "Mid-height |u_x| rms (LSC)")):
        ax.plot(time_points, data, "-o", ms=3)
        ax.set_xlabel("t"); ax.set_title(ttl); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc3d_timeseries_N{NY}.png", dpi=150)
    plt.close(fig)

    # --- vertical slice (x-y plane at mid z) ---
    kmid = nz // 2
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(T[-1, :, :, kmid].T, origin="lower", extent=(0, LX, 0, LY),
                   cmap="RdBu_r", aspect="equal", vmin=T_TOP, vmax=T_BOT)
    fig.colorbar(im, ax=ax, label="T", shrink=0.8)
    ax.set_title(f"Vertical slice (z=Lz/2), t={time_points[-1]:.1f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc3d_vslice_N{NY}.png", dpi=150)
    plt.close(fig)

    # --- horizontal slice near the bottom plate (x-z), the plume network ---
    jplate = max(NY // 12, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(T[-1, :, jplate, :].T, origin="lower", extent=(0, LX, 0, LZ),
                   cmap="inferno", aspect="equal")
    fig.colorbar(im, ax=ax, label="T", shrink=0.8)
    ax.set_title(f"Near-plate horizontal slice (y={y_np[0, jplate, 0]:.2f}), "
                 f"t={time_points[-1]:.1f}")
    ax.set_xlabel("x"); ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc3d_hslice_N{NY}.png", dpi=150)
    plt.close(fig)

    # --- volume render of the temperature anomaly (the plumes) ---
    T_horiz_mean = T[-1].mean(axis=(0, 2), keepdims=True)   # mean vs height
    theta = T[-1] - T_horiz_mean
    img = volume_render(theta, None, None, axis=2)          # integrate along z
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.imshow(img, origin="lower", extent=(0, LX, 0, LY), aspect="equal")
    ax.set_title(f"Temperature-anomaly volume render (depth = z), "
                 f"t={time_points[-1]:.1f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc3d_volume_N{NY}.png", dpi=160)
    plt.close(fig)

    # --- vertical-slice animation ---
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(T[0, :, :, kmid].T, origin="lower", extent=(0, LX, 0, LY),
                   cmap="RdBu_r", aspect="equal", vmin=T_TOP, vmax=T_BOT)
    title = ax.set_title("t = 0.00")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="T", shrink=0.8)

    def update(i):
        im.set_data(T[i, :, :, kmid].T)
        title.set_text(f"t = {time_points[i]:.2f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=n_snap, interval=120, blit=False)
    anim.save(OUTPUT_DIR / f"rbc3d_vslice_N{NY}.gif",
              writer=PillowWriter(fps=10), dpi=100)
    plt.close(fig)

    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
