"""
Highly turbulent 2D Rayleigh-Benard convection -- visual run.

Inspired by the well-known high-resolution 2D RBC renderings (Ra = 1e13,
Pr = 1, aspect ratio 16:9, hot/red rising, cold/blue falling, white = mean
temperature, time in free-fall units).  Ra = 1e13 at 7680x4320 is a
supercomputer simulation; here we reproduce the *look* at a modest resolution
by running an effectively under-resolved (ILES) high-*nominal*-Ra case:
physical viscosity/conduction are set small (Pr = 1) so the grid sets the
dissipation scale and the flow is densely populated with sharp plumes.

  * 2D finite-difference WENO, Pallas (Triton) backend;
  * aspect ratio Lx:Ly = 16:9, vertical = y;
  * no-slip reflective walls (closed cell), isothermal hot bottom / cold top
    plates via thermal conduction (Dirichlet ghost-T), adiabatic side walls;
  * constant gravity g in -y via a linear external potential phi = g y;
  * Pr = 1 (mu = (gamma-1) kappa), nominal Ra set by NOMINAL_RA.

Time is reported in free-fall units  tau_ff = Ly / v_ff,
v_ff = sqrt(g (dT/T) Ly).
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
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

# Cinematic diverging "balance"-style colormap: deep navy (cold) -> white (mean)
# -> deep red (hot), smooth and saturated like high-end convection renderings.
CINEMA_CMAP = LinearSegmentedColormap.from_list("cinema_balance", [
    (0.00, "#070d33"),   # deepest blue  (coldest)
    (0.16, "#12347f"),
    (0.33, "#3b74c4"),
    (0.45, "#aacbec"),
    (0.50, "#f7f7f7"),   # white at the mean temperature
    (0.55, "#f6cbab"),
    (0.67, "#dc7a45"),
    (0.84, "#b3301a"),
    (1.00, "#3f0707"),   # deepest red   (hottest)
])


def adaptive_halfrange(field, center, floor=0.012, pct=99.2):
    """Symmetric half-range about ``center`` tracking the current anomaly
    amplitude (robust 99th percentile).  Shrinks as the bulk mixes toward the
    mean, so the colour scale "zooms in" and the plumes stay visible while the
    colourbar numbers fade -> the dynamically adapting colourbar."""
    return float(max(floor, np.percentile(np.abs(field - center), pct)))

from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    PALLAS,
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
NY = 288                      # vertical cells; Nx = 16/9 * NY  (16:9 -> 512x288)
LY = 1.0
LX = 16.0 / 9.0 * LY

GAMMA = 5.0 / 3.0
G = 0.25
T_BOT = 1.2                   # hot bottom (red)
T_TOP = 0.8                   # cold top (blue); mean 1.0 = white
P_BOT = 1.0

NOMINAL_RA = 1.0e8            # grid-limited (ILES) nominal Rayleigh number
PR = 1.0

T_END_TAU = 15.0              # integration length in free-fall times (long: dense turbulence)
NUM_SNAPSHOTS = 180
C_CFL = 0.4
SEED_AMP = 1e-2

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def transport_from_ra(rho_ref=0.9):
    """mu, kappa giving the requested nominal Ra and Pr.

    Ra = g (dT/T) Ly^3 / (nu_kin chi),  nu_kin = mu/rho,  chi = (g-1) kappa/rho,
    Pr = nu_kin/chi = mu / ((gamma-1) kappa).  With Pr fixed,
    nu_kin chi = Pr chi^2, and chi = (gamma-1) kappa / rho.
    """
    dT = T_BOT - T_TOP
    T_mean = 0.5 * (T_BOT + T_TOP)
    nu_chi = G * (dT / T_mean) * LY ** 3 / NOMINAL_RA      # = nu_kin * chi
    chi = np.sqrt(nu_chi / PR)                              # nu_kin = Pr*chi
    kappa = chi * rho_ref / (GAMMA - 1.0)
    mu = PR * (GAMMA - 1.0) * kappa
    return float(mu), float(kappa)


def hydrostatic_background(y):
    dT = T_BOT - T_TOP
    T = T_BOT - (dT / LY) * y
    exponent = G * LY / dT
    p = P_BOT * (T / T_BOT) ** exponent
    rho = p / T
    return T, p, rho


def build_initial_state(config, registered_variables, helper_data, v_ff):
    coords = helper_data.geometric_centers
    x = coords[..., 0]
    y = coords[..., 1]
    T_bg, p_bg, rho_bg = hydrostatic_background(y)

    rng = np.random.default_rng(0)
    vy = jnp.zeros_like(x)
    for n in range(1, 13):                      # many horizontal modes
        ph = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(0.3, 1.0))
        vy = vy + amp * jnp.sin(np.pi * y / LY) * jnp.cos(
            2.0 * np.pi * n * x / LX + ph
        )
    vy = SEED_AMP * v_ff * vy / jnp.max(jnp.abs(vy))

    state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho_bg, velocity_x=jnp.zeros_like(x), velocity_y=vy,
        gas_pressure=p_bg,
    )
    return state, x, y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    nx = int(round(16.0 / 9.0 * NY))
    nx -= nx % 4                                 # keep multiple of 4 for Pallas
    dT = T_BOT - T_TOP
    T_mean = 0.5 * (T_BOT + T_TOP)
    v_ff = float(np.sqrt(G * (dT / T_mean) * LY))
    tau_ff = LY / v_ff
    t_end = T_END_TAU * tau_ff

    mu, kappa = transport_from_ra()
    chi = (GAMMA - 1.0) * kappa / 0.9
    cs_mean = float(np.sqrt(GAMMA * T_mean))
    print(f"=== 2D turbulent RBC, FD+Pallas, {nx}x{NY} (16:9) ===")
    print(f"nominal Ra = {NOMINAL_RA:.1e}, Pr = {PR}, Ma ~ {v_ff / cs_mean:.3f}, "
          f"cells = {nx * NY / 1e3:.0f}k")
    print(f"mu = {mu:.2e}, kappa = {kappa:.2e}, tau_ff = {tau_ff:.3f}, "
          f"t_end = {t_end:.1f} ({T_END_TAU} tau_ff)")

    # Lx must make dx == dy exactly: dx = Lx/nx, dy = Ly/NY
    lx = LX
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        dimensionality=2,
        backend=PALLAS,
        pallas_block_shape=(4, 4, 4),
        pallas_use_triton=True,
        pallas_interpret=False,
        num_cells=StaticIntVector(nx, NY, -1),
        box_size=StaticFloatVector(lx, LY, 1.0),
        progress_bar=True,
        external_potential=True,
        self_gravity_version=SIMPLE_SOURCE_TERM,
        diffusion=True,
        thermal_conduction=True,
        conduction_wall_axis=1,
        conduction_isothermal_walls=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),
            BoundarySettings1D(),
        ),
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)
    state, x, y = build_initial_state(config, registered_variables, helper_data, v_ff)

    # enforce exact dx == dy (Lx adjusted to nx)
    config = config._replace(box_size=StaticFloatVector(LY / NY * nx, LY, 1.0))
    config = finalize_config(config, state.shape)

    params = SimulationParams(
        C_cfl=C_CFL, gamma=GAMMA, t_end=t_end, viscosity=mu,
        thermal_conductivity=kappa, wall_temperature_low=T_BOT,
        wall_temperature_high=T_TOP, gravitational_potential=G * y,
    )
    assert config.gravity

    t0 = _time.time()
    result = time_integration(state, config, params, registered_variables)
    states = result.states
    time_points = np.asarray(result.time_points)
    n_snap = states.shape[0]
    print(f"Integrated {n_snap} snapshots to t = {time_points[-1]:.1f} "
          f"({time_points[-1] / tau_ff:.1f} tau_ff) in {_time.time() - t0:.1f} s")

    rho = np.asarray(states[:, registered_variables.density_index])
    vx = np.asarray(states[:, registered_variables.velocity_index.x])
    vy = np.asarray(states[:, registered_variables.velocity_index.y])
    p = np.asarray(states[:, registered_variables.pressure_index])
    T = p / rho
    if np.any(np.isnan(rho)):
        print("WARNING: NaNs encountered!")

    Ek = 0.5 * np.mean(rho * (vx ** 2 + vy ** 2), axis=(1, 2))
    Nu = 1.0 + np.mean(vy * T, axis=(1, 2)) / (chi * dT / LY)
    print(f"Final Nu = {Nu[-1]:.1f}, time-mean Nu (2nd half) = "
          f"{np.mean(Nu[n_snap // 2:]):.1f}, final E_k = {Ek[-1]:.3e}")

    np.savez(DATA_DIR / f"rbc2dturb_N{NY}.npz",
             time_points=time_points, Ek=Ek, Nu=Nu,
             tau_ff=tau_ff, T_final=T[-1].astype(np.float32))

    center = 1.0                      # mean temperature -> white
    half = np.array([adaptive_halfrange(T[i], center) for i in range(n_snap)])

    # --- final frame (adaptive range) ---
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.imshow(T[-1].T, origin="lower", extent=(0, lx, 0, LY),
              cmap=CINEMA_CMAP, vmin=center - half[-1], vmax=center + half[-1],
              aspect="equal", interpolation="bilinear")
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(OUTPUT_DIR / f"rbc2dturb_final_N{NY}.png", dpi=200)
    plt.close(fig)

    # --- animation with a dynamically adapting colourbar ---
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    fig.subplots_adjust(left=0.005, right=0.93, top=0.995, bottom=0.005)
    im = ax.imshow(T[0].T, origin="lower", extent=(0, lx, 0, LY),
                   cmap=CINEMA_CMAP, vmin=center - half[0], vmax=center + half[0],
                   aspect="equal", interpolation="bilinear")
    ax.set_axis_off()
    cax = fig.add_axes([0.94, 0.08, 0.015, 0.84])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("temperature", color="0.25")
    cbar.ax.tick_params(colors="0.25", labelsize=8)
    txt = ax.text(0.012, 0.96, "", transform=ax.transAxes, va="top",
                  color="0.85", fontsize=12,
                  bbox=dict(facecolor="0.15", alpha=0.45, edgecolor="none",
                            boxstyle="round,pad=0.25"))

    def update(i):
        im.set_data(T[i].T)
        im.set_clim(center - half[i], center + half[i])   # adapt the scale
        cbar.update_normal(im)                            # adapt the colourbar
        txt.set_text(f"t = {time_points[i] / tau_ff:4.1f} free-fall times    "
                     f"|  colour scale  +/-{half[i]:.3f}")
        return im, txt

    # playback: 2 seconds of video per free-fall time (as in the reference)
    n_tau = float(time_points[-1] / tau_ff)
    fps = max(4.0, (n_snap / n_tau) / 2.0)
    print(f"animation: {n_snap} frames, {n_tau:.1f} tau_ff, fps = {fps:.1f} "
          f"(2 s per free-fall time)")
    anim = FuncAnimation(fig, update, frames=n_snap, interval=100, blit=False)
    try:
        anim.save(OUTPUT_DIR / f"rbc2dturb_N{NY}.mp4",
                  writer=FFMpegWriter(fps=fps, bitrate=8000), dpi=130)
        print("wrote mp4")
    except Exception as e:
        print(f"ffmpeg unavailable ({e}); writing gif")
    anim.save(OUTPUT_DIR / f"rbc2dturb_N{NY}.gif",
              writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)

    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
