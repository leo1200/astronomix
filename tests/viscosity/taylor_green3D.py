# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os
from typing import NamedTuple

import jax

# to not run into finite precision in
# the convergence study at high resolution
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from astronomix import SimulationConfig, SimulationParams
from astronomix import get_helper_data, time_integration
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    initialize_interface_fields,
)
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FORWARDS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
    finalize_config,
    GravityConfig,
    PositivityConfig,
)
from astronomix import get_registered_variables

from astronomix._fluid_equations.total_quantities import calculate_kinetic_energy

os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# ==================  3-D Taylor–Green Vortex  ================
# ─────────────────────────────────────────────────────────────

gamma = 5.0 / 3.0
box_size = 2.0 * jnp.pi


def simulate_taylor_green(
    num_cells: int = 64,
    Re: float = 100.0,
    mach: float = 0.1,
    t_end: float = 10.0,
    C_cfl: float = 0.4,
    num_snapshots: int = 200,
    return_states: bool = False,
):
    """
    Run the 3-D Taylor–Green vortex at a given Reynolds number.

    The Reynolds number is defined as  Re = ρ₀ V₀ / (μ k₀)  with k₀ = 1,
    so μ = ρ₀ V₀ / Re.

    Parameters
    ----------
    num_cells : int
        Grid resolution (same in every direction).
    Re : float
        Reynolds number.
    mach : float
        Initial Mach number  V₀ / c_s.  Keep ≪ 1 so the incompressible
        analytical solution is a good reference.
    t_end : float
        End time (in code units where L = 2π, k₀ = 1).
    C_cfl : float
        CFL safety factor.
    num_snapshots : int
        Number of snapshots for the energy time-series.
    return_states : bool
        Whether to return the full state at each snapshot (needed for
        animations, but uses more memory).
    """

    # ── derived quantities ──────────────────────────────────────────────
    rho0 = 1.0
    V0 = mach                           # c_s = 1 when we set p0 below
    p0 = rho0 / gamma                   # ⇒ c_s = sqrt(γ p0/ρ0) = 1
    mu = rho0 * V0 / Re                 # dynamic viscosity
    nu = mu / rho0                      # kinematic viscosity

    print(f"V0 = {V0:.4f},  μ = {mu:.6f},  ν = {nu:.6f}")
    print(f"Viscous CFL timescale ~ dx²/(6ν) = "
          f"{(box_size / num_cells)**2 / (6 * nu):.4f}")

    # ── config ──────────────────────────────────────────────────────────
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        runtime_debugging=False,
        progress_bar=True,
        positivity_config=PositivityConfig(default_positivity_protection=False),
        mhd=False,
        diffusion=True,
        dimensionality=3,
        num_cells=num_cells,
        box_size=float(box_size),
        differentiation_mode=FORWARDS,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY,
                right_boundary=PERIODIC_BOUNDARY,
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY,
                right_boundary=PERIODIC_BOUNDARY,
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY,
                right_boundary=PERIODIC_BOUNDARY,
            ),
        ),
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(
            return_states=return_states,
            return_final_state=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
            return_gravitational_energy=False,
        ),
        num_snapshots=num_snapshots,
    )

    params = SimulationParams(
        t_end=t_end,
        C_cfl=C_cfl,
        dt_max=jnp.inf,
        minimum_density=1e-8,
        minimum_pressure=1e-10,
        viscosity=mu,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    # ── coordinate arrays ───────────────────────────────────────────────
    dx = config.box_size.x / num_cells
    coords = jnp.arange(num_cells) * dx + 0.5 * dx   # cell centres
    x, y, z = jnp.meshgrid(coords, coords, coords, indexing="ij")

    # ── initial primitive state ─────────────────────────────────────────
    rho = rho0 * jnp.ones_like(x)

    v_x =  V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
    v_y = -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
    v_z =  jnp.zeros_like(x)

    p = (
        p0
        + (rho0 * V0**2 / 16.0)
        * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y))
        * (jnp.cos(2.0 * z) + 2.0)
    )

    # zero magnetic fields (hydro only)
    B_x = jnp.zeros_like(rho)
    B_y = jnp.zeros_like(rho)
    B_z = jnp.zeros_like(rho)
    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v_x,
        velocity_y=v_y,
        velocity_z=v_z,
        gas_pressure=p,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
    )

    config = finalize_config(config, initial_state.shape)

    # check that config.grid_spacing == dx
    assert jnp.isclose(config.grid_spacing, dx), (
        f"Expected grid spacing {config.grid_spacing} to match dx = {dx}"
    )

    # ── analytical reference ────────────────────────────────────────────
    # At low Mach and for k₀ = 1 the kinetic energy decays as
    #   E_k(t) = E_k(0) exp(−6 ν t)
    Ek0 = rho0 * V0**2 * float(box_size) ** 3 / 8.0
    print(f"Analytical E_k(0) = {Ek0:.6e}")

    # print E_k(0) from the initial state as a sanity check
    Ek0_sim = calculate_kinetic_energy(initial_state, helper_data, config, registered_variables)
    print(f"Simulated E_k(0) = {Ek0_sim:.6e}")

    # ── run ─────────────────────────────────────────────────────────────
    print(f"Running TGV:  N = {num_cells},  Re = {Re},  "
          f"Mach = {mach},  t_end = {t_end}")
    snapshots = jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )

    return snapshots, config, params, registered_variables, nu, Ek0


# ─────────────────────────────────────────────────────────────
#  Verification: kinetic energy decay
# ─────────────────────────────────────────────────────────────

def kinetic_energy_decay_test(
    num_cells: int = 64,
    Re: float = 100.0,
    mach: float = 0.1,
    t_end: float = 10.0,
):
    """Compare measured E_k(t) against the analytical exponential decay."""

    snapshots, config, params, _, nu, Ek0 = simulate_taylor_green(
        num_cells=num_cells,
        Re=Re,
        mach=mach,
        t_end=t_end,
    )

    t = snapshots.time_points
    Ek_sim = snapshots.kinetic_energy
    Ek_exact = Ek0 * jnp.exp(-6.0 * nu * t)

    # ── figure 1: kinetic energy vs time ────────────────────────────────
    fig, (ax_ek, ax_err) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    ax_ek.plot(t, Ek_sim, label="Simulation", linewidth=2)
    ax_ek.plot(t, Ek_exact, "--", label=r"$E_k(0)\,e^{-6\nu t}$", linewidth=2)
    ax_ek.set_ylabel(r"$E_k$")
    ax_ek.legend()
    ax_ek.set_title(
        f"3-D Taylor–Green Vortex  "
        f"($N={num_cells}$,  Re$={Re:.0f}$,  Ma$={mach}$)"
    )

    # ── figure 2: relative error ────────────────────────────────────────
    rel_err = jnp.abs(Ek_sim - Ek_exact) / Ek0
    ax_err.semilogy(t, rel_err, linewidth=2)
    ax_err.set_xlabel(r"$t$")
    ax_err.set_ylabel(r"$|E_k - E_k^{\rm exact}| / E_k(0)$")
    ax_err.set_title("Relative Kinetic Energy Error")

    plt.tight_layout()
    fig.savefig(f"figures/tgv3d_Ek_decay_N{num_cells}_Re{Re:.0f}.png", dpi=150)
    fig.savefig(f"figures/tgv3d_Ek_decay_N{num_cells}_Re{Re:.0f}.svg")
    print(f"Saved figures/tgv3d_Ek_decay_N{num_cells}_Re{Re:.0f}.*")
    plt.close(fig)

    return snapshots


def convergence_study(
    resolutions=(32, 64, 128),
    Re: float = 100.0,
    mach: float = 0.1,
    t_end: float = 5.0,
):
    """Measure spatial convergence of the kinetic energy error."""

    errors = []

    for N in resolutions:
        snapshots, *_, nu, Ek0 = simulate_taylor_green(
            num_cells=N,
            Re=Re,
            mach=mach,
            t_end=t_end,
        )
        t = snapshots.time_points
        Ek_sim = snapshots.kinetic_energy
        print(Ek_sim)
        Ek_exact = Ek0 * jnp.exp(-6.0 * nu * t)
        # L-infinity relative error over time
        err = float(jnp.max(jnp.abs(Ek_sim - Ek_exact) / Ek0))
        errors.append(err)
        print(f"  N = {N:4d}  →  max |ΔEk|/Ek0 = {err:.4e}")

    resolutions = list(resolutions)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(resolutions, errors, "o-", linewidth=2, label="Measured error")

    # reference slopes
    N0, e0 = resolutions[0], errors[0]
    Ns = jnp.array(resolutions, dtype=float)
    for order, ls in [(2, ":"), (4, "--"), (6, "-.")]:
        ref = e0 * (N0 / Ns) ** order
        ax.loglog(Ns, ref, ls, color="grey", alpha=0.6,
                  label=rf"$\propto N^{{-{order}}}$")

    ax.set_xlabel("N")
    ax.set_ylabel(r"max $|E_k - E_k^{\rm exact}| / E_k(0)$")
    ax.set_title(f"TGV Convergence  (Re = {Re:.0f},  Ma = {mach})")
    ax.legend(fontsize="small")
    plt.tight_layout()
    fig.savefig(f"figures/tgv3d_convergence_Re{Re:.0f}.png", dpi=150)
    fig.savefig(f"figures/tgv3d_convergence_Re{Re:.0f}.svg")
    print(f"Saved figures/tgv3d_convergence_Re{Re:.0f}.*")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  Animation: viscosity comparison
# ─────────────────────────────────────────────────────────────

def viscosity_comparison_animation(
    num_cells: int = 128,
    Re_list: tuple = (50, 200, 1600),
    mach: float = 0.3,
    t_end: float = 10.0,
    num_snapshots: int = 120,
    fps: int = 15,
):
    """
    Animate z = π mid-plane slices of vorticity magnitude for three
    different Reynolds numbers side by side.

    Parameters
    ----------
    num_cells : int
        Grid resolution.
    Re_list : tuple of float
        Three Reynolds numbers to compare.
    mach : float
        Initial Mach number (slightly higher than the convergence tests
        so the nonlinear dynamics are visible).
    t_end : float
        End time.
    num_snapshots : int
        Number of frames in the animation.
    fps : int
        Frames per second for the saved GIF.
    """

    n_cases = len(Re_list)
    dx = float(box_size) / num_cells
    mid = num_cells // 2  # z ≈ π slice

    # ── run all cases ───────────────────────────────────────────────────
    all_snapshots = []
    all_rv = []
    for Re in Re_list:
        print(f"\n{'=' * 50}")
        print(f"  Running Re = {Re}")
        print(f"{'=' * 50}")
        snapshots, config, params, rv, nu, Ek0 = simulate_taylor_green(
            num_cells=num_cells,
            Re=Re,
            mach=mach,
            t_end=t_end,
            num_snapshots=num_snapshots,
            return_states=True,
        )
        all_snapshots.append(snapshots)
        all_rv.append(rv)

    time = all_snapshots[0].time_points

    # ── compute vorticity magnitude on the mid-plane ────────────────────
    def vorticity_magnitude_slices(states, rv):
        """Return |ω| at z = mid for every snapshot.  Shape (n_snaps, nx, ny)."""
        vx = states[:, rv.velocity_index.x]
        vy = states[:, rv.velocity_index.y]
        vz = states[:, rv.velocity_index.z]

        # centered finite differences (axes: 0=snap, 1=x, 2=y, 3=z)
        dvz_dy = (jnp.roll(vz, -1, axis=2) - jnp.roll(vz, 1, axis=2)) / (2 * dx)
        dvy_dz = (jnp.roll(vy, -1, axis=3) - jnp.roll(vy, 1, axis=3)) / (2 * dx)
        dvx_dz = (jnp.roll(vx, -1, axis=3) - jnp.roll(vx, 1, axis=3)) / (2 * dx)
        dvz_dx = (jnp.roll(vz, -1, axis=1) - jnp.roll(vz, 1, axis=1)) / (2 * dx)
        dvy_dx = (jnp.roll(vy, -1, axis=1) - jnp.roll(vy, 1, axis=1)) / (2 * dx)
        dvx_dy = (jnp.roll(vx, -1, axis=2) - jnp.roll(vx, 1, axis=2)) / (2 * dx)

        omega_mag = jnp.sqrt(
            (dvz_dy - dvy_dz) ** 2
            + (dvx_dz - dvz_dx) ** 2
            + (dvy_dx - dvx_dy) ** 2
        )
        return omega_mag[:, :, :, mid]

    vort = []
    for i in range(n_cases):
        print(f"Computing vorticity for Re = {Re_list[i]}...")
        vort.append(vorticity_magnitude_slices(all_snapshots[i].states, all_rv[i]))

    # ── global colour scale ─────────────────────────────────────────────
    vmax = float(max(jnp.max(v) for v in vort))

    # ── set up figure ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n_cases, figsize=(5 * n_cases + 1, 5))
    if n_cases == 1:
        axes = [axes]

    extent = [0, float(box_size), 0, float(box_size)]

    ims = []
    for i, ax in enumerate(axes):
        im = ax.imshow(
            vort[i][0].T,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0,
            vmax=vmax,
        )
        ax.set_title(f"Re = {Re_list[i]}", fontsize=14)
        ax.set_xlabel(r"$x$")
        if i == 0:
            ax.set_ylabel(r"$y$")
        else:
            ax.set_yticklabels([])
        ims.append(im)

    fig.colorbar(ims[-1], ax=axes, shrink=0.85, label=r"$|\omega|$")

    title = fig.suptitle(
        rf"Vorticity at $z = \pi$,  $t = {float(time[0]):.2f}$"
        f"   (Ma = {mach},  N = {num_cells})",
        fontsize=13,
    )

    # ── kinetic energy inset ────────────────────────────────────────────
    ax_inset = fig.add_axes([0.13, 0.78, 0.18, 0.14])
    colors = plt.cm.tab10.colors
    for i, Re in enumerate(Re_list):
        Ek_norm = all_snapshots[i].kinetic_energy / all_snapshots[i].kinetic_energy[0]
        ax_inset.plot(time, Ek_norm, color=colors[i], lw=1.2, label=f"Re={Re}")
    vline = ax_inset.axvline(float(time[0]), color="k", lw=0.8, ls="--")
    ax_inset.set_xlim(float(time[0]), float(time[-1]))
    ax_inset.set_ylim(0, 1.05)
    ax_inset.set_ylabel(r"$E_k / E_k(0)$", fontsize=7)
    ax_inset.set_xlabel(r"$t$", fontsize=7)
    ax_inset.tick_params(labelsize=6)
    ax_inset.legend(fontsize=5, loc="lower left")

    # ── animate ─────────────────────────────────────────────────────────
    n_frames = len(time)

    def update(frame):
        for i in range(n_cases):
            ims[i].set_data(vort[i][frame].T)
        title.set_text(
            rf"Vorticity at $z = \pi$,  $t = {float(time[frame]):.2f}$"
            f"   (Ma = {mach},  N = {num_cells})"
        )
        vline.set_xdata([float(time[frame])])
        return ims + [title, vline]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    outpath = f"figures/tgv3d_viscosity_comparison_N{num_cells}.gif"
    anim.save(outpath, writer="pillow", fps=fps)
    print(f"Saved {outpath}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # # Primary validation: kinetic energy decay at Re = 100
    # kinetic_energy_decay_test(
    #     num_cells=64,
    #     Re=100.0,
    #     mach=0.1,
    #     t_end=10.0,
    # )

    # # Spatial convergence study
    # convergence_study(
    #     resolutions=(32, 64, 128),
    #     Re=100.0,
    #     mach=0.01,
    #     t_end=1.0,
    # )

    # Viscosity comparison animation
    viscosity_comparison_animation(
        num_cells=64,
        Re_list=(10, 100, 1000),
        mach=0.8,
        t_end=10.0,
        num_snapshots=60,
        fps=15,
    )