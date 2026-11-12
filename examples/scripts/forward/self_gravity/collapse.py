"""
3D self-gravitating collapse of a uniform-density sphere.

Runs the collapse with per-snapshot diagnostics, then produces three figures:
a central density slice of the collapsed sphere, the energy budget over time
(total / internal / kinetic / gravitational), and an animation of the collapsing
density slice.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import SECOND_ORDER_CONSERVATIVE

# astronomix containers
from astronomix import (
    GravityConfig,
    SnapshotSettings,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
)

# astronomix functions
from astronomix import (
    time_integration,
    get_helper_data,
    get_registered_variables,
    construct_primitive_state,
    finalize_config,
)


# figures are written to the local figures/ directory
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# =============== ↓ Configuration ↓ ===========================
# -------------------------------------------------------------
gamma = 5 / 3
box_size = 4.0
num_cells = 64

config = SimulationConfig(
    progress_bar = True,
    dimensionality = 3,
    box_size = box_size,
    num_cells = num_cells,
    gravity_config = GravityConfig(
        self_gravity = True,
        self_gravity_version = SECOND_ORDER_CONSERVATIVE,
        poisson_manual_open_boundaries = True,
        # flux-corrected gravitational work: keeps the conservative energy
        # coupling from driving cold receiving cells to negative pressure
        work_flux_correction = True,
    ),
    solver_config=FDSolverConfig(
        # positivity-preserving WENO (Zhang & Shu) keeps the low-resolution
        # collapse from driving density/pressure negative
        weno_positivity_preserving = True,
    ),
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    # per-snapshot states (for the animation) plus the integrated energy budget
    return_snapshots = True,
    num_snapshots = 60,
    snapshot_settings = SnapshotSettings(
        return_states = True,
        return_final_state = True,
        return_total_energy = True,
        return_internal_energy = True,
        return_kinetic_energy = True,
        return_gravitational_energy = True,
    ),
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)

params = SimulationParams(
    t_end = 1.2,
    C_cfl = 0.4,
    gamma = gamma,
    minimum_density = 1e-5,
    minimum_pressure = 3e-6,
)
# -------------------------------------------------------------
# =============== ↑ Configuration ↑ ===========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Initial state ↓ ===========================
# -------------------------------------------------------------
# uniform-density sphere of radius R at rest in a tenuous background
R = 1.0
rho_sphere = 1.0
rho_background = 1e-4

rho = jnp.where(helper_data.r <= R, rho_sphere, rho_background)
u_x = jnp.zeros_like(rho)
u_y = jnp.zeros_like(rho)
u_z = jnp.zeros_like(rho)

# small, roughly uniform thermal support so the sphere collapses
internal_energy = 0.05
p = jnp.maximum((gamma - 1) * rho * internal_energy, params.minimum_pressure)

initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u_x,
    velocity_y = u_y,
    velocity_z = u_z,
    gas_pressure = p,
)

config = finalize_config(config, initial_state.shape)
# -------------------------------------------------------------
# =============== ↑ Initial state ↑ ===========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Run ↓ =====================================
# -------------------------------------------------------------
snapshots = jax.block_until_ready(
    time_integration(initial_state, config, params, registered_variables)
)
# -------------------------------------------------------------
# =============== ↑ Run ↑ =====================================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Central density slice ↓ ===================
# -------------------------------------------------------------
z = num_cells // 2
density_index = registered_variables.density_index
final_density = snapshots.states[-1, density_index][:, :, z]
print(f"final density slice range: min={float(jnp.min(final_density)):.3e} "
      f"max={float(jnp.max(final_density)):.3e}")

# explicit positive vmin/vmax so LogNorm/colorbar stay valid even if the slice
# contains non-positive floor values
vmax = float(jnp.max(final_density))
positive = final_density[final_density > 0]
vmin = float(jnp.min(positive)) if positive.size else vmax * 1e-6

fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(final_density.T, origin="lower", cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax))
ax.set_title("Collapsed density (central slice)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.savefig(figures_dir / "collapse_density.png", dpi=200, bbox_inches="tight")
# -------------------------------------------------------------
# =============== ↑ Central density slice ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Energy budget over time ↓ =================
# -------------------------------------------------------------
time = snapshots.time_points
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(time, snapshots.total_energy, label="Total", color="black")
ax.plot(time, snapshots.internal_energy, label="Internal", color="green")
ax.plot(time, snapshots.kinetic_energy, label="Kinetic", color="red")
ax.plot(time, snapshots.gravitational_energy, label="Gravitational", color="blue")
ax.set_xlabel("Time")
ax.set_ylabel("Energy")
ax.set_title(f"Collapse energy budget (N = {num_cells})")
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "collapse_energy.svg")
# -------------------------------------------------------------
# =============== ↑ Energy budget over time ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Density-slice animation ↓ =================
# -------------------------------------------------------------
# central density slice at every snapshot, on a shared log colour scale
slices = snapshots.states[:, density_index][:, :, :, z]
slices_pos = slices[slices > 0]
anim_vmin = float(jnp.min(slices_pos)) if slices_pos.size else 1e-6
anim_vmax = float(jnp.max(slices))
anim_norm = LogNorm(vmin=anim_vmin, vmax=anim_vmax)

fig_a, ax_a = plt.subplots(figsize=(7, 7))
im_a = ax_a.imshow(slices[0].T, origin="lower", cmap="inferno", norm=anim_norm)
title_a = ax_a.set_title("t = 0.00")
fig_a.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)


def _update(frame):
    im_a.set_data(slices[frame].T)
    title_a.set_text(f"t = {float(time[frame]):.3f}")
    return im_a, title_a


anim = FuncAnimation(fig_a, _update, frames=slices.shape[0], blit=False)
anim.save(figures_dir / "collapse_animation.gif", writer=PillowWriter(fps=15), dpi=100)
# -------------------------------------------------------------
# =============== ↑ Density-slice animation ↑ =================
# -------------------------------------------------------------
