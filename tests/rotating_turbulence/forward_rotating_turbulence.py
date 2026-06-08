"""
Driven, rotating, isothermal, subsonic compressible turbulence in a triply
periodic cube -- forward problem (D1: emergent columns + stationary spectrum).

Isothermal compressible Navier-Stokes in a rotating frame (rotation axis zhat):

    d_t rho + div(rho u) = 0
    d_t(rho u) + div(rho u x u) + grad P = -2 Omega zhat x (rho u) + rho f + div tau
    P = c_s^2 rho        (isothermal: no energy equation, no conduction)

with f a solenoidal large-scale forcing (constant injection rate) and tau the
momentum viscosity.  The Coriolis term does no work -> a pure momentum source
(new `config.rotation` / `params.rotation_rate`).  Rotation drives the flow
toward the slow, barotropic (z-invariant) columnar manifold; we confirm
columns by (a) omega_z slices, (b) the rising *barotropic fraction* of the
horizontal kinetic energy, and (c) the energy spectrum.

Prototype at 64^3 (native; the isothermal WENO kernel is not Pallas-ported).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import time as _time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE, ISOTHERMAL, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, SnapshotSettings,
    SimulationConfig, finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig, TurbulentForcingParams,
)
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration

import os
N = int(os.environ.get("RT_N", 64))
L = 1.0
CS = 1.0                  # isothermal sound speed
EDOT = float(os.environ.get("RT_EDOT", 0.05))   # energy injection rate (u_rms ~ 0.25-0.33)
NU = float(os.environ.get("RT_NU", 1.0e-3))     # kinematic viscosity (rho0=1)
OMEGA = float(os.environ.get("RT_OMEGA", 0.5))  # rotation rate about z (Ro ~ u_rms/(2 Omega L_f))
RHO0 = 1.0
C_CFL = 0.8
T_END = float(os.environ.get("RT_TEND", 14.0))
NUM_SNAP = int(os.environ.get("RT_SNAP", 70))
# OU forcing (temporally correlated). RT_OU=1 enables it.
OU = int(os.environ.get("RT_OU", 0))
TAU_F = float(os.environ.get("RT_TAUF", 1.0))     # correlation time (~ eddy turnover)
KF_MODE = float(os.environ.get("RT_KF", 4.0))     # forcing peak in mode number n (k=2 pi n/L)
F0 = float(os.environ.get("RT_F0", 1.0))          # forcing amplitude (tunes u_rms)

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def vorticity_z(vx, vy, dx):
    """omega_z = d vy/dx - d vx/dy via spectral derivatives (periodic)."""
    n = vx.shape[0]
    k = jnp.fft.fftfreq(n, d=dx) * 2 * jnp.pi
    kx = k[:, None, None]
    ky = k[None, :, None]
    vyh = jnp.fft.fftn(vy)
    vxh = jnp.fft.fftn(vx)
    return jnp.real(jnp.fft.ifftn(1j * kx * vyh - 1j * ky * vxh))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE, equation_of_state=ISOTHERMAL, mhd=False,
        dimensionality=3, num_cells=N, box_size=L, enforce_positivity=False,
        progress_bar=True,
        diffusion=True,                     # physical momentum viscosity
        rotation=True,                      # Coriolis source about z
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)),
        turbulent_forcing_config=TurbulentForcingConfig(
            turbulent_forcing=True, ou_forcing=bool(OU)),
        return_snapshots=True, num_snapshots=NUM_SNAP,
        snapshot_settings=SnapshotSettings(return_states=True),
    )
    params = SimulationParams(
        C_cfl=C_CFL, isothermal_sound_speed=CS, t_end=T_END, viscosity=NU,
        rotation_rate=OMEGA,
        turbulent_forcing_params=TurbulentForcingParams(
            energy_injection_rate=EDOT,
            correlation_time=TAU_F,
            forcing_wavenumber=2.0 * np.pi * KF_MODE / L,   # mode -> physical k
            forcing_amplitude=F0,
        ),
    )
    rv = get_registered_variables(config)

    density = RHO0 * jnp.ones((N, N, N))
    zero = jnp.zeros_like(density)
    state = construct_primitive_state(
        config=config, registered_variables=rv, density=density,
        velocity_x=zero, velocity_y=zero, velocity_z=zero)
    config = finalize_config(config, state.shape)

    print(f"=== rotating isothermal turbulence {N}^3, Omega={OMEGA}, nu={NU}, "
          f"c_s={CS}, Edot={EDOT} ===")
    t0 = _time.time()
    result = time_integration(state, config, params, rv)
    states = np.asarray(result.states)
    tp = np.asarray(result.time_points)
    n_snap = states.shape[0]
    print(f"integrated {n_snap} snapshots to t={tp[-1]:.2f} in {_time.time()-t0:.1f}s")

    rho = states[:, rv.density_index]
    vx = states[:, rv.velocity_index.x]
    vy = states[:, rv.velocity_index.y]
    vz = states[:, rv.velocity_index.z]
    if np.any(np.isnan(rho)):
        print("WARNING: NaNs in the run!")

    dx = L / N
    u_rms = np.sqrt(np.mean(vx ** 2 + vy ** 2 + vz ** 2, axis=(1, 2, 3)))
    Ma = u_rms / CS
    # barotropic (z-invariant) fraction of the HORIZONTAL kinetic energy
    vx_bar = vx.mean(axis=3, keepdims=True)     # z-average
    vy_bar = vy.mean(axis=3, keepdims=True)
    e_h = np.mean(vx ** 2 + vy ** 2, axis=(1, 2, 3))
    e_baro = np.mean(vx_bar ** 2 + vy_bar ** 2, axis=(1, 2, 3))
    baro_frac = e_baro / (e_h + 1e-30)
    L_f = L / 2
    Ro = u_rms / (2 * OMEGA * L_f + 1e-30)

    print(f"final u_rms={u_rms[-1]:.3f}  Ma={Ma[-1]:.3f}  Ro={Ro[-1]:.3f}  "
          f"barotropic fraction={baro_frac[-1]:.3f}")
    print(f"rotation period pi/Omega = {np.pi/OMEGA:.2f}, "
          f"eddy turnover L_f/u_rms = {L_f/(u_rms[-1]+1e-9):.2f}")

    np.savez(DATA_DIR / f"rot_turb_N{N}_Om{OMEGA}_ou{OU}.npz", t=tp, u_rms=u_rms, Ma=Ma,
             Ro=Ro, baro_frac=baro_frac, vx_final=vx[-1].astype(np.float32),
             vy_final=vy[-1].astype(np.float32), vz_final=vz[-1].astype(np.float32))

    # --- time series: u_rms, barotropic fraction ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(tp, u_rms, "-o", ms=3); ax[0].set(xlabel="t", ylabel="u_rms",
        title=f"rms velocity (Ma={Ma[-1]:.2f}, Ro={Ro[-1]:.2f})")
    ax[0].grid(alpha=0.3)
    ax[1].plot(tp, baro_frac, "-o", ms=3, color="C2")
    ax[1].axvline(np.pi / OMEGA, ls="--", color="k", alpha=0.5, label="1 rotation period")
    ax[1].set(xlabel="t", ylabel="barotropic fraction of horiz. KE",
              title="2D-ization (columns forming)")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"rot_turb_timeseries_N{N}_Om{OMEGA}_ou{OU}.png", dpi=150)
    plt.close(fig)

    # --- omega_z: x-y slice (mid z) and x-z slice (mid y) shows columns ---
    wz_final = np.asarray(vorticity_z(jnp.asarray(vx[-1]), jnp.asarray(vy[-1]), dx))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    vmax = np.percentile(np.abs(wz_final), 99)
    im0 = ax[0].imshow(wz_final[:, :, N // 2].T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=(0, L, 0, L))
    ax[0].set(title=r"$\omega_z$  x-y slice (perp. to rotation)", xlabel="x", ylabel="y")
    fig.colorbar(im0, ax=ax[0], shrink=0.8)
    im1 = ax[1].imshow(wz_final[:, N // 2, :].T, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=(0, L, 0, L))
    ax[1].set(title=r"$\omega_z$  x-z slice (along rotation: columns = vertical streaks)",
              xlabel="x", ylabel="z (rotation axis)")
    fig.colorbar(im1, ax=ax[1], shrink=0.8)
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"rot_turb_omegaz_N{N}_Om{OMEGA}_ou{OU}.png", dpi=150)
    plt.close(fig)

    # --- column field viewed down the rotation axis: z-averaged (barotropic)
    #     vorticity. Columns = coherent cyclones/anticyclones from above. ---
    wz_bar = wz_final.mean(axis=2)          # (Nx, Ny)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    vmb = np.percentile(np.abs(wz_bar), 99)
    im = ax.imshow(wz_bar.T, origin="lower", cmap="RdBu_r", vmin=-vmb, vmax=vmb,
                   extent=(0, L, 0, L), interpolation="bilinear")
    ax.set(title=r"barotropic (z-averaged) $\omega_z$ — columns from above",
           xlabel="x", ylabel="y")
    fig.colorbar(im, ax=ax, shrink=0.8, label=r"$\langle\omega_z\rangle_z$")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rot_turb_columns_N{N}_Om{OMEGA}_ou{OU}.png", dpi=160)
    plt.close(fig)

    # --- energy spectra: isotropic E(k) and the anisotropy E(k_perp) vs E(k_z)
    #     (energy piling toward k_z -> 0 is the columnar signature) ---
    def energy_spectra(ux, uy, uz):
        n = ux.shape[0]
        kk = np.fft.fftfreq(n, d=L / n) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kk, kk, kk, indexing="ij")
        E = 0.5 * (np.abs(np.fft.fftn(ux)) ** 2 + np.abs(np.fft.fftn(uy)) ** 2
                   + np.abs(np.fft.fftn(uz)) ** 2) / n ** 6
        k0 = 2 * np.pi / L
        nb = n // 2
        kbin = np.round(np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2) / k0).astype(int)
        kpbin = np.round(np.sqrt(KX ** 2 + KY ** 2) / k0).astype(int)
        kzbin = np.round(np.abs(KZ) / k0).astype(int)
        Ek = np.bincount(kbin.ravel(), weights=E.ravel(), minlength=nb + 1)[:nb]
        Ekp = np.bincount(kpbin.ravel(), weights=E.ravel(), minlength=nb + 1)[:nb]
        Ekz = np.bincount(kzbin.ravel(), weights=E.ravel(), minlength=nb + 1)[:nb]
        return np.arange(nb), Ek, Ekp, Ekz

    kmode, Ek, Ekp, Ekz = energy_spectra(vx[-1], vy[-1], vz[-1])
    np.savez(DATA_DIR / f"rot_turb_spectra_N{N}_Om{OMEGA}_ou{OU}.npz",
             kmode=kmode, Ek=Ek, Ekp=Ekp, Ekz=Ekz)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].loglog(kmode[1:], Ek[1:], "-o", ms=3)
    ax[0].axvline(KF_MODE, ls="--", color="r", alpha=0.5, label=f"k_f = {KF_MODE:g}")
    ax[0].set(xlabel="k (mode)", ylabel="E(k)", title="isotropic kinetic energy spectrum")
    ax[0].legend(); ax[0].grid(alpha=0.3, which="both")
    ax[1].loglog(kmode[1:], Ekp[1:], "-o", ms=3, label=r"$E(k_\perp)$")
    ax[1].loglog(kmode[1:], Ekz[1:], "-s", ms=3, label=r"$E(k_z)$")
    ax[1].scatter([1], [Ekz[0]], color="C1", zorder=5,
                  label=r"$E(k_z{=}0)$ (barotropic)")
    ax[1].set(xlabel="k (mode)", ylabel="E",
              title="spectral anisotropy: energy piles at k_z→0 (columns)")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"rot_turb_spectra_N{N}_Om{OMEGA}_ou{OU}.png", dpi=150)
    plt.close(fig)

    # --- omega_z x-z slice animation (watch columns form along z) ---
    wz = np.stack([np.asarray(vorticity_z(jnp.asarray(vx[i]), jnp.asarray(vy[i]), dx))[:, N // 2, :]
                   for i in range(n_snap)])
    vmax = np.percentile(np.abs(wz), 99)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(wz[0].T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=(0, L, 0, L))
    ttl = ax.set_title("t = 0.00"); ax.set(xlabel="x", ylabel="z (rotation axis)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    def update(i):
        im.set_data(wz[i].T); ttl.set_text(f"t = {tp[i]:.2f}"); return im, ttl
    anim = FuncAnimation(fig, update, frames=n_snap, interval=120, blit=False)
    anim.save(OUTPUT_DIR / f"rot_turb_omegaz_N{N}_Om{OMEGA}_ou{OU}.gif", writer=PillowWriter(fps=12), dpi=110)
    plt.close(fig)
    print(f"figures -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
