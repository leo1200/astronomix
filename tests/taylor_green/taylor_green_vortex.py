"""
3D compressible Taylor-Green vortex (TGV) -- forward problem.

This is the forward-problem first cut for worked example II of
``init_optim_theory.md`` (Section 8, "3D compressible Euler turbulence").
The theory names the compressible Taylor-Green vortex transitioning to
turbulence as the *clean demonstrator* for the reduced- vs full-space
inverse-problem study, and explicitly suggests

    "A clean first cut is the scale-resolved adjoint-spectrum experiment
     on the Taylor-Green vortex before committing to the full inverse
     problem."

Before any of that we need the *forward* problem to actually produce the
turbulence we want: a large-scale, smooth, analytically known initial
vortex that self-steepens, rolls up, and breaks down into a turbulent
cascade.  This script sets that up with the standalone Hydro
finite-difference (WENO) scheme and verifies the transition with four
diagnostics:

    1. kinetic-energy decay  E_k(t)  and dissipation rate  -dE_k/dt,
       which should peak near the classic  t ~ 9  breakdown time;
    2. enstrophy growth (vortex stretching) up to that same time;
    3. the kinetic-energy spectrum at early / peak / late times,
       developing a  k^{-5/3}  inertial range;
    4. a vorticity-magnitude slice animation showing the large-scale
       vortices shredding into small scales.

The classic (incompressible-limit) TGV initial condition on the periodic
box [0, 2*pi]^3 is

    u_x =  V0 sin(x) cos(y) cos(z)
    u_y = -V0 cos(x) sin(y) cos(z)
    u_z =  0
    rho =  rho0
    p   =  p0 + (rho0 V0^2 / 16) (cos(2x) + cos(2y)) (cos(2z) + 2)

with the background pressure set by the reference Mach number
Ma0 = V0 / c0, i.e.  p0 = rho0 V0^2 / (gamma Ma0^2).  We use a subsonic
Ma0 = 0.3 so the transition is essentially shock-free (the inviscid WENO
scheme acts as an implicit LES; numerical dissipation plays the role of
the molecular viscosity at an effectively high Reynolds number).

The natural time unit is the eddy-turnover time t_c = 1 / (V0 k0) = 1
(k0 = 1, V0 = 1), so the times below are directly in turnover units.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from pathlib import Path

import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# astronomix
from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._fluid_equations._equations import get_absolute_velocity
from astronomix.analysis_helpers.energy_spectrum import (
    LOG_BINNING,
    get_kinetic_energy_spectrum,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    PALLAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
    finalize_config,
)
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators


# ---------------------------------------------------------------------------
#  Physical / numerical parameters
# ---------------------------------------------------------------------------

NUM_CELLS = 128         # resolution per dimension (start small, scale up)
BOX_SIZE = 2.0 * jnp.pi  # [0, 2*pi]^3 periodic box
V0 = 1.0                 # vortex velocity amplitude
RHO0 = 1.0               # background density
MA0 = 0.3                # reference Mach number V0 / c0 (subsonic, clean)
GAMMA = 5.0 / 3.0        # adiabatic index
T_END = 20.0             # in eddy-turnover times t_c = 1/(V0 k0) = 1
NUM_SNAPSHOTS = 80
C_CFL = 0.4

OUTPUT_DIR = Path(__file__).parent / "figures"
DATA_DIR = Path(__file__).parent / "data"


def build_initial_state(config, registered_variables, helper_data):
    """Construct the analytic compressible TGV primitive state."""
    coords = helper_data.geometric_centers  # (Nx, Ny, Nz, 3), centers in [0, 2pi)
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    velocity_x = V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
    velocity_y = -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
    velocity_z = jnp.zeros_like(x)

    density = RHO0 * jnp.ones_like(x)

    # background pressure from the reference Mach number:
    # c0^2 = gamma p0 / rho0 and Ma0 = V0 / c0  =>  p0 = rho0 V0^2 / (gamma Ma0^2)
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    pressure = p0 + (RHO0 * V0**2 / 16.0) * (
        jnp.cos(2.0 * x) + jnp.cos(2.0 * y)
    ) * (jnp.cos(2.0 * z) + 2.0)

    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        gas_pressure=pressure,
    )


def vorticity_magnitude(vx, vy, vz, dx):
    """|curl u| from spectral derivatives on the periodic box."""
    n = vx.shape[0]
    # physical wavenumbers for a box of length n*dx
    k = jnp.fft.fftfreq(n, d=dx) * 2.0 * jnp.pi
    kx = k[:, None, None]
    ky = k[None, :, None]
    kz = k[None, None, :]

    vx_hat = jnp.fft.fftn(vx)
    vy_hat = jnp.fft.fftn(vy)
    vz_hat = jnp.fft.fftn(vz)

    # omega = curl u  (i * k x u_hat -> real space)
    wx = jnp.fft.ifftn(1j * (ky * vz_hat - kz * vy_hat)).real
    wy = jnp.fft.ifftn(1j * (kz * vx_hat - kx * vz_hat)).real
    wz = jnp.fft.ifftn(1j * (kx * vy_hat - ky * vx_hat)).real
    return jnp.sqrt(wx**2 + wy**2 + wz**2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Compressible Taylor-Green vortex, Hydro FD, N={NUM_CELLS}^3 ===")

    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=IDEAL_GAS,
        mhd=False,
        # Pallas (Triton) backend: ~10x faster and lower memory than native JAX
        backend=PALLAS,
        pallas_block_shape=(4, 4, 8),
        pallas_use_triton=True,
        pallas_interpret=False,
        dimensionality=3,
        num_cells=NUM_CELLS,
        box_size=BOX_SIZE,
        progress_bar=True,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(return_states=True),
    )

    params = SimulationParams(
        C_cfl=C_CFL,
        gamma=GAMMA,
        t_end=T_END,
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    initial_state = build_initial_state(config, registered_variables, helper_data)
    config = finalize_config(config, initial_state.shape)

    dx = float(BOX_SIZE / NUM_CELLS)

    # sanity: report initial Mach number
    p0 = RHO0 * V0**2 / (GAMMA * MA0**2)
    c0 = float(jnp.sqrt(GAMMA * p0 / RHO0))
    print(f"Background sound speed c0 = {c0:.3f}, reference Mach Ma0 = {V0 / c0:.3f}")

    result = time_integration(initial_state, config, params, registered_variables)

    states = result.states                      # (n_snap, n_vars, Nx, Ny, Nz)
    time_points = result.time_points
    n_snap = states.shape[0]
    print(f"Got {n_snap} snapshots up to t = {float(time_points[-1]):.2f}")

    rho = states[:, registered_variables.density_index]
    vx = states[:, registered_variables.velocity_index.x]
    vy = states[:, registered_variables.velocity_index.y]
    vz = states[:, registered_variables.velocity_index.z]

    # --- diagnostic 1 & 2: kinetic energy, dissipation rate, enstrophy ---
    kinetic_energy = 0.5 * jnp.mean(rho * (vx**2 + vy**2 + vz**2), axis=(1, 2, 3))

    enstrophy = jnp.stack([
        0.5 * jnp.mean(vorticity_magnitude(vx[i], vy[i], vz[i], dx) ** 2)
        for i in range(n_snap)
    ])

    # dissipation rate epsilon = -dE_k/dt
    dissipation = -jnp.gradient(kinetic_energy, time_points)

    t_peak = float(time_points[int(jnp.argmax(dissipation))])
    print(f"Initial E_k = {float(kinetic_energy[0]):.5f}, "
          f"final E_k = {float(kinetic_energy[-1]):.5f}")
    print(f"Dissipation rate peaks at t = {t_peak:.2f} (classic TGV: ~9)")
    print(f"Peak enstrophy at t = "
          f"{float(time_points[int(jnp.argmax(enstrophy))]):.2f}")

    # NaN guard
    if bool(jnp.any(jnp.isnan(kinetic_energy))):
        print("WARNING: NaNs encountered in the run!")

    jnp.savez(
        DATA_DIR / f"tgv_N{NUM_CELLS}.npz",
        time_points=time_points,
        kinetic_energy=kinetic_energy,
        enstrophy=enstrophy,
        dissipation=dissipation,
    )

    # --- plot 1: energy / dissipation / enstrophy time series ---
    fig, (ax_e, ax_eps) = plt.subplots(1, 2, figsize=(12, 5))
    ax_e.plot(time_points, kinetic_energy, "-o", ms=3)
    ax_e.set_xlabel("t / t_c")
    ax_e.set_ylabel(r"$E_k = \langle \frac{1}{2}\, \rho\, |u|^2 \rangle$")
    ax_e.set_title("Kinetic energy decay")
    ax_e.grid(alpha=0.3)

    ax_eps.plot(time_points, dissipation / dissipation.max(),
                "-o", ms=3, label=r"$-dE_k/dt$ (norm.)")
    ax_eps.plot(time_points, enstrophy / enstrophy.max(),
                "-s", ms=3, label="enstrophy (norm.)")
    ax_eps.axvline(9.0, color="k", ls="--", alpha=0.5, label="t = 9 (classic)")
    ax_eps.set_xlabel("t / t_c")
    ax_eps.set_ylabel("normalized")
    ax_eps.set_title("Dissipation rate & enstrophy")
    ax_eps.legend()
    ax_eps.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"tgv_energy_N{NUM_CELLS}.png", dpi=200)
    plt.close(fig)

    # --- plot 2: kinetic energy spectra at early / peak / late ---
    idx_early = 0
    idx_peak = int(jnp.argmax(dissipation))
    idx_late = n_snap - 1
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, label in [
        (idx_early, f"t = {float(time_points[idx_early]):.1f}"),
        (idx_peak, f"t = {float(time_points[idx_peak]):.1f} (peak)"),
        (idx_late, f"t = {float(time_points[idx_late]):.1f}"),
    ]:
        k, ek = get_kinetic_energy_spectrum(
            vx[idx], vy[idx], vz[idx], rho[idx], binning=LOG_BINNING
        )
        # the helper returns physical wavenumber 2*pi*mode; on the [0,2pi]
        # box the fundamental TGV mode is integer mode 1, so divide by 2*pi
        # to plot against integer mode number (k=1 = box fundamental).
        ax.loglog(k / (2.0 * jnp.pi), ek, "-o", ms=3, label=label)
    ax.set_xlabel("wavenumber k (integer mode number)")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Kinetic energy spectrum (N={NUM_CELLS})")
    ax.set_xlim(1, NUM_CELLS // 2)
    ax.set_ylim(bottom=1e-9)
    add_power_law_indicators(
        ax,
        anchor=(3, 3e-3),
        exponents=[-5 / 3],
        scales=[1.0],
        labels=[r" $k^{-5/3}$"],
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"tgv_spectrum_N{NUM_CELLS}.png", dpi=200)
    plt.close(fig)

    # --- plot 3: vorticity-magnitude slice animation ---
    z_slice = NUM_CELLS // 2
    vort_slices = jnp.stack([
        vorticity_magnitude(vx[i], vy[i], vz[i], dx)[:, :, z_slice]
        for i in range(n_snap)
    ])
    vmax = float(jnp.max(vort_slices))
    vmin = max(float(jnp.min(vort_slices[vort_slices > 0])), vmax * 1e-3)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        vort_slices[0].T,
        origin="lower",
        extent=(0, float(BOX_SIZE), 0, float(BOX_SIZE)),
        cmap="inferno",
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im, cax=cax, label=r"$|\omega|$")
    title = ax.set_title(f"t = {float(time_points[0]):.2f}")

    def update(i):
        im.set_data(vort_slices[i].T)
        title.set_text(f"t = {float(time_points[i]):.2f}")
        return im, title

    anim = FuncAnimation(fig, update, frames=n_snap, interval=100, blit=False)
    anim.save(OUTPUT_DIR / f"tgv_vorticity_N{NUM_CELLS}.gif",
              writer=PillowWriter(fps=12), dpi=150)
    plt.close(fig)

    print(f"Figures written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
