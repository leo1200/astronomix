"""
2D compressible Rayleigh-Benard convection -- forward problem (D1).

Deliverable D1 of ``tests/rayleigh_bernard/tasks.md``: the forward simulation
must produce the expected large-scale structure (coherent convection rolls /
large-scale circulation) at a meaningful Rayleigh number.

Setup (faithful RBC, Phase 0b):
  * finite-difference WENO scheme, non-periodic boundaries;
  * confined box, aspect ratio Gamma = Lx / Ly = 2;
  * no-slip reflective walls all around;
  * isothermal hot (bottom) / cold (top) plates driven by the new thermal
    conduction module (Dirichlet ghost-T), adiabatic side walls (the
    reflective hydro boundary mirrors T as an even quantity -> zero flux);
  * constant gravity g in -y via a linear external potential phi = g * y;
  * momentum viscosity ON + thermal conduction ON (physical nu, kappa).

The background is a hydrostatic, conductive (linear-T) atmosphere that is
super-adiabatic and hence convectively unstable.  A small multi-mode velocity
perturbation seeds the rolls.

Non-dimensional control parameters (Boussinesq-style estimates about the mean
state, rho ~ T ~ 1):
    Ra = g (dT/T) Ly^3 / (nu_kin * chi),     Pr = nu_kin / chi,
with kinematic viscosity nu_kin = mu/rho and thermal diffusivity
chi = (gamma-1) kappa / rho.

Diagnostics: temperature & vorticity snapshots, Nusselt number Nu(t), kinetic
energy E_k(t), and the mid-height horizontal-velocity LSC amplitude.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

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
    REFLECTIVE_BOUNDARY,
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
NY = 64                       # cells in the vertical; Nx = GAMMA * NY
GAMMA_ASPECT = 2              # aspect ratio Lx / Ly
LY = 1.0
LX = GAMMA_ASPECT * LY

GAMMA = 5.0 / 3.0            # adiabatic index
G = 0.25                      # gravitational acceleration (-y)
T_BOT = 1.2                   # hot bottom plate temperature
T_TOP = 0.8                   # cold top plate temperature
P_BOT = 1.0                   # pressure at the bottom plate (sets density scale)

MU = 1.4e-3                   # dynamic viscosity  -> Ra ~ 5e4, Pr ~ 1
KAPPA = 1.9e-3                # thermal conductivity

T_END = 80.0
NUM_SNAPSHOTS = 80
C_CFL = 0.4

SEED_AMP = 1e-2               # velocity seed amplitude (fraction of v_ff)

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def hydrostatic_background(y):
    """Linear conductive temperature profile in hydrostatic balance.

    T(y) = T_bot - (dT/Ly) y;  dp/dy = -rho g = -(p/T) g  =>
    p(y) = P_BOT (T(y)/T_bot)^(g Ly / dT).
    """
    dT = T_BOT - T_TOP
    T = T_BOT - (dT / LY) * y
    exponent = G * LY / dT
    p = P_BOT * (T / T_BOT) ** exponent
    rho = p / T
    return T, p, rho


def build_initial_state(config, registered_variables, helper_data):
    coords = helper_data.geometric_centers          # (Nx, Ny, 2)
    x = coords[..., 0]
    y = coords[..., 1]

    T_bg, p_bg, rho_bg = hydrostatic_background(y)

    # velocity seed: a few horizontal modes, vanishing at the plates
    v_ff = np.sqrt(G * (T_BOT - T_TOP) / ((T_BOT + T_TOP) / 2.0) * LY)
    rng = np.random.default_rng(0)
    vy = jnp.zeros_like(x)
    for n in (1, 2, 3):
        phase = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(0.5, 1.0))
        vy = vy + amp * jnp.sin(np.pi * y / LY) * jnp.cos(
            2.0 * np.pi * n * x / LX + phase
        )
    vy = SEED_AMP * v_ff * vy / jnp.max(jnp.abs(vy))
    vx = jnp.zeros_like(x)

    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho_bg,
        velocity_x=vx,
        velocity_y=vy,
        gas_pressure=p_bg,
    )
    return state, x, y


def vorticity_z(vx, vy, dx):
    """omega_z = d vy/dx - d vx/dy (2nd-order centered, np arrays)."""
    dvy_dx = (np.roll(vy, -1, axis=0) - np.roll(vy, 1, axis=0)) / (2 * dx)
    dvx_dy = (np.roll(vx, -1, axis=1) - np.roll(vx, 1, axis=1)) / (2 * dx)
    return dvy_dx - dvx_dy


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    nx = GAMMA_ASPECT * NY
    dT = T_BOT - T_TOP
    T_mean = 0.5 * (T_BOT + T_TOP)
    nu_kin = MU / 1.0
    chi = (GAMMA - 1.0) * KAPPA / 1.0
    Ra = G * (dT / T_mean) * LY ** 3 / (nu_kin * chi)
    Pr = nu_kin / chi
    cs_mean = float(np.sqrt(GAMMA * T_mean))
    v_ff = float(np.sqrt(G * (dT / T_mean) * LY))
    print(f"=== 2D Rayleigh-Benard, FD, {nx}x{NY}, Gamma={GAMMA_ASPECT} ===")
    print(f"Ra ~ {Ra:.2e}, Pr ~ {Pr:.2f}, Ma ~ v_ff/cs = {v_ff / cs_mean:.3f}")

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        dimensionality=2,
        num_cells=StaticIntVector(nx, NY, -1),
        box_size=StaticFloatVector(LX, LY, 1.0),
        progress_bar=True,
        external_potential=True,
        diffusion=True,                 # momentum viscosity
        thermal_conduction=True,        # conductive plates
        conduction_wall_axis=1,         # y is the vertical (plate) axis
        conduction_isothermal_walls=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),  # sidewalls
            BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY),  # plates
            BoundarySettings1D(),
        ),
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    state, x, y = build_initial_state(config, registered_variables, helper_data)
    config = finalize_config(config, state.shape)

    # external potential phi = g * y (bare grid, no ghosts) -> a = -g in y
    external_phi = G * y

    params = SimulationParams(
        C_cfl=C_CFL,
        gamma=GAMMA,
        t_end=T_END,
        viscosity=MU,
        thermal_conductivity=KAPPA,
        wall_temperature_low=T_BOT,     # bottom plate (y low) = hot
        wall_temperature_high=T_TOP,    # top plate (y high) = cold
        gravitational_potential=external_phi,
    )

    assert config.gravity, "gravity master switch not set"

    result = time_integration(state, config, params, registered_variables)
    states = result.states
    time_points = np.asarray(result.time_points)
    n_snap = states.shape[0]
    print(f"Got {n_snap} snapshots up to t = {time_points[-1]:.2f}")

    rho = np.asarray(states[:, registered_variables.density_index])
    vx = np.asarray(states[:, registered_variables.velocity_index.x])
    vy = np.asarray(states[:, registered_variables.velocity_index.y])
    p = np.asarray(states[:, registered_variables.pressure_index])
    T = p / rho

    if np.any(np.isnan(rho)):
        print("WARNING: NaNs encountered in the run!")

    dx = LX / nx
    y1d = np.asarray(y)[0, :]

    # kinetic energy
    Ek = 0.5 * np.mean(rho * (vx ** 2 + vy ** 2), axis=(1, 2))

    # Nusselt number: Nu = 1 + <vy T> / (chi dT/Ly), horizontally + vertically
    # averaged convective flux normalised by the conductive flux.
    conv_flux = np.mean(vy * T, axis=(1, 2))
    Nu = 1.0 + conv_flux / (chi * dT / LY)

    # LSC amplitude: rms of mid-height horizontal velocity
    jmid = NY // 2
    lsc = np.sqrt(np.mean(vx[:, :, jmid] ** 2, axis=1))

    print(f"Initial E_k = {Ek[0]:.3e}, final E_k = {Ek[-1]:.3e}")
    print(f"Final Nu = {Nu[-1]:.2f}, time-mean Nu (2nd half) = "
          f"{np.mean(Nu[n_snap // 2:]):.2f}")
    print(f"Final LSC amplitude = {lsc[-1]:.3e} (v_ff = {v_ff:.3e})")

    np.savez(
        DATA_DIR / f"rbc_N{NY}.npz",
        time_points=time_points, Ek=Ek, Nu=Nu, lsc=lsc,
        T_final=T[-1], vx_final=vx[-1], vy_final=vy[-1],
    )

    # --- time series ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(time_points, Ek, "-o", ms=3)
    axes[0].set_xlabel("t"); axes[0].set_ylabel(r"$E_k$")
    axes[0].set_title("Kinetic energy"); axes[0].grid(alpha=0.3)
    axes[1].plot(time_points, Nu, "-o", ms=3)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Nu")
    axes[1].set_title("Nusselt number"); axes[1].grid(alpha=0.3)
    axes[2].plot(time_points, lsc, "-o", ms=3)
    axes[2].set_xlabel("t"); axes[2].set_ylabel("LSC amplitude")
    axes[2].set_title("Mid-height |u_x| rms"); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc_timeseries_N{NY}.png", dpi=150)
    plt.close(fig)

    # --- final temperature + velocity field ---
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(T[-1].T, origin="lower", extent=(0, LX, 0, LY),
                   cmap="RdBu_r", aspect="equal")
    skip = max(nx // 48, 1)
    ax.quiver(np.asarray(x)[::skip, ::skip], np.asarray(y)[::skip, ::skip],
              vx[-1][::skip, ::skip], vy[-1][::skip, ::skip],
              color="k", scale=20 * v_ff)
    fig.colorbar(im, ax=ax, label="T", shrink=0.8)
    ax.set_title(f"Temperature + velocity, t = {time_points[-1]:.1f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rbc_final_field_N{NY}.png", dpi=150)
    plt.close(fig)

    # --- temperature animation ---
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(T[0].T, origin="lower", extent=(0, LX, 0, LY),
                   cmap="RdBu_r", aspect="equal", vmin=T_TOP, vmax=T_BOT)
    title = ax.set_title("t = 0.00")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="T", shrink=0.8)

    def update(i):
        im.set_data(T[i].T)
        title.set_text(f"t = {time_points[i]:.2f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=n_snap, interval=100, blit=False)
    anim.save(OUTPUT_DIR / f"rbc_temperature_N{NY}.gif",
              writer=PillowWriter(fps=12), dpi=100)
    plt.close(fig)

    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
