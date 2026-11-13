"""
Stellar wind blown into a turbulently-driven, magnetized ISM (3D finite difference).

The run is split into two phases that share the same turbulent (Ornstein-Uhlenbeck)
driving, the TurbGen driving of the code (natural mixture, 1 <= k <= 3):

  1. an OPTIONAL turbulence spin-up phase (driving only, no wind) that lets the ISM
     develop a realistic turbulent density/velocity field, and
  2. a stellar-wind phase in which a wind source is switched on WHILE the turbulent
     driving keeps running.

Slices are dumped along the way and stitched into a single GIF that spans the whole
time evolution (spin-up followed by wind).

At the default 64^3 resolution this completes in a few minutes on a single GPU.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import re
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# gif assembly
import imageio.v3 as iio

# units and constants
from astropy import units as u
import astropy.constants as c

# astronomix constants
from astronomix import PERIODIC_BOUNDARY

# astronomix containers
from astronomix import (
    CodeUnits,
    WindParams,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix.option_classes import WindConfig
from astronomix import TurbulenceDrivingConfig, TurbulenceDrivingParams

# astronomix functions
from astronomix import (
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    initialize_interface_fields,
    finalize_config,
)


# -------------------------------------------------------------
# =============== ↓ Top-level toggles / knobs ↓ ===============
# -------------------------------------------------------------

# --- physics switches ---
run_turbulence_spinup_phase = True   # optional first phase: pure turbulent driving
stellar_wind = True                  # second phase: switch on the stellar wind source
turbulence = True                    # turbulent driving (active during BOTH phases)
mhd = True                           # otherwise the initial magnetic field is zero

# --- spatial domain ---
box_size = 1.0
num_cells = 64                       # 64^3 -> a few-minute single-GPU run

# --- physics parameters ---
gamma = 5 / 3
C_cfl = 0.8
dt_max = 0.1

# target velocity dispersion of the turbulent driving (sets its amplitude)
turbulent_velocity_dispersion = 30 * u.km / u.s

# stellar-wind injection
num_injection_cells = 4

# --- phase durations (physical) ---
spinup_duration = 6.0 * 1e4 * u.yr   # turbulence-only spin-up
wind_duration = 0.5 * 1e4 * u.yr     # wind + turbulence

# --- output ---
num_snapshots = 100                  # frames captured per phase for the GIF
gif_fps = 30

# -------------------------------------------------------------
# =============== ↑ Top-level toggles / knobs ↑ ===============
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Output directories ↓ ======================
# -------------------------------------------------------------

figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# per-phase frame directories (cleared on each run)
spinup_frames_dir = figures_dir / "_frames_spinup"
wind_frames_dir = figures_dir / "_frames_wind"
for frames_dir in (spinup_frames_dir, wind_frames_dir):
    frames_dir.mkdir(exist_ok=True)
    for stale in frames_dir.glob("*.png"):
        stale.unlink()

# -------------------------------------------------------------
# =============== ↑ Output directories ↑ ======================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Simulation configuration ↓ ================
# -------------------------------------------------------------

# baseline config: turbulent driving on, wind off (wind is enabled for phase 2)
config = SimulationConfig(
    mhd=True,
    progress_bar=True,
    donate_state=True,               # save storage
    dimensionality=3,
    box_size=box_size,
    num_cells=num_cells,
    turbulence_driving_config=TurbulenceDrivingConfig(
        turbulence_driving=turbulence,
    ),
    boundary_settings=BoundarySettings(
        BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
    ),
    # frame callback for the animation
    activate_snapshot_callback=True,
    num_snapshots=num_snapshots,
)

registered_variables = get_registered_variables(config)

# -------------------------------------------------------------
# =============== ↑ Simulation configuration ↑ ================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Code units and parameters ↓ ===============
# -------------------------------------------------------------

code_length = 3 * u.parsec
code_mass = 1 * u.M_sun
code_velocity = 100 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# stellar-wind parameters
M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

wind_params = WindParams(
    wind_mass_loss_rate=wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity=wind_final_velocity.to(code_units.code_velocity).value,
)

params = SimulationParams(
    C_cfl=C_cfl,
    dt_max=dt_max,
    gamma=gamma,
    minimum_density=1e-3,
    minimum_pressure=1e-3,
    wind_params=wind_params,
    turbulence_driving_params=TurbulenceDrivingParams(
        velocity=turbulent_velocity_dispersion.to(code_units.code_velocity).value,
    ),
)

print("Target velocity dispersion of the driving:", turbulent_velocity_dispersion)

# -------------------------------------------------------------
# =============== ↑ Code units and parameters ↑ ===============
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Homogeneous initial state ↓ ===============
# -------------------------------------------------------------

rho_0 = 2 * c.m_p / u.cm ** 3
p_0 = 3e4 * u.K / u.cm ** 3 * c.k_B

shape = (num_cells, num_cells, num_cells)
rho = jnp.ones(shape) * rho_0.to(code_units.code_density).value
u_x = jnp.zeros(shape)
u_y = jnp.zeros(shape)
u_z = jnp.zeros(shape)
p = jnp.ones(shape) * p_0.to(code_units.code_pressure).value

if mhd:
    B_0 = (13.5 * u.microgauss / c.mu0 ** 0.5).to(code_units.code_magnetic_field).value
else:
    B_0 = 0.0

B_x = jnp.ones(shape) * B_0
B_y = jnp.zeros(shape)
B_z = jnp.zeros(shape)
bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    velocity_z=u_z,
    gas_pressure=p,
    magnetic_field_x=B_x,
    magnetic_field_y=B_y,
    magnetic_field_z=B_z,
    interface_magnetic_field_x=bxb,
    interface_magnetic_field_y=byb,
    interface_magnetic_field_z=bzb,
)

config = finalize_config(config, initial_state.shape)

# -------------------------------------------------------------
# =============== ↑ Homogeneous initial state ↑ ===============
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Frame-saving callback ↓ ===================
# -------------------------------------------------------------

z_level = num_cells // 2


def save_frame(time, state, registered_variables, directory):
    """Snapshot callback: dump a z-midplane density/pressure slice as a PNG frame."""

    def plot_slices(density, pressure, time):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        for ax in (ax1, ax2):
            ax.set_aspect("equal", "box")
            ax.set_xticks([])
            ax.set_yticks([])

        ax1.imshow(density.T, origin="lower", extent=[0, 1, 0, 1], norm=LogNorm())
        ax1.set_title("density")
        ax2.imshow(pressure.T, origin="lower", extent=[0, 1, 0, 1], norm=LogNorm())
        ax2.set_title("pressure")

        fig.savefig(f"{directory}/frame{float(time):012.5f}.png", dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    density = state[registered_variables.density_index, :, :, z_level]
    pressure = state[registered_variables.pressure_index, :, :, z_level]

    jax.debug.callback(plot_slices, density, pressure, time)


save_spinup_frame = lambda time, state, rv: save_frame(time, state, rv, spinup_frames_dir)
save_wind_frame = lambda time, state, rv: save_frame(time, state, rv, wind_frames_dir)

# -------------------------------------------------------------
# =============== ↑ Frame-saving callback ↑ ===================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Phase 1: turbulence spin-up ↓ =============
# -------------------------------------------------------------

state = initial_state

if run_turbulence_spinup_phase:
    print("Phase 1: driving turbulence (no wind) ...")
    params = params._replace(t_end=spinup_duration.to(code_units.code_time).value)
    state = time_integration(state, config, params, registered_variables, save_spinup_frame)

# -------------------------------------------------------------
# =============== ↑ Phase 1: turbulence spin-up ↑ =============
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Phase 2: stellar wind + turbulence ↓ ======
# -------------------------------------------------------------

if stellar_wind:
    print("Phase 2: stellar wind with continued turbulent driving ...")
    # switch the wind on; turbulent driving stays active via the shared config
    config = config._replace(
        wind_config=WindConfig(
            stellar_wind=True,
            num_injection_cells=num_injection_cells,
        ),
    )
    params = params._replace(t_end=wind_duration.to(code_units.code_time).value)
    state = time_integration(state, config, params, registered_variables, save_wind_frame)

final_state = state

# -------------------------------------------------------------
# =============== ↑ Phase 2: stellar wind + turbulence ↑ ======
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Final slice plot ↓ ========================
# -------------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for ax in (ax1, ax2, ax3):
    ax.set_aspect("equal", "box")

ax1.imshow(
    final_state[registered_variables.density_index, :, :, z_level].T,
    origin="lower", extent=[0, 1, 0, 1], norm=LogNorm(),
)
ax1.set_title("density")

speed = jnp.sqrt(
    final_state[registered_variables.velocity_index.x, :, :, z_level] ** 2
    + final_state[registered_variables.velocity_index.y, :, :, z_level] ** 2
    + final_state[registered_variables.velocity_index.z, :, :, z_level] ** 2
)
ax2.imshow(speed.T, origin="lower", extent=[0, 1, 0, 1])
ax2.set_title("velocity magnitude")

ax3.imshow(
    final_state[registered_variables.pressure_index, :, :, z_level].T,
    origin="lower", extent=[0, 1, 0, 1], norm=LogNorm(),
)
ax3.set_title("pressure")

fig.suptitle("Stellar wind in the turbulent ISM (z-midplane)")
fig.tight_layout()
fig.savefig(figures_dir / "turbulent_stellar_wind_final.png", dpi=150)

# -------------------------------------------------------------
# =============== ↑ Final slice plot ↑ ========================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =============== ↓ Assemble GIF over the whole run ↓ =========
# -------------------------------------------------------------


def sorted_frames(directory):
    """Return the frame PNGs in a directory sorted by their embedded timestamp."""
    files = list(directory.glob("*.png"))
    files.sort(key=lambda f: float(re.search(r"frame([0-9.]+)\.png", f.name).group(1)))
    return files


# spin-up frames first, then wind frames -> a single timeline
all_frames = sorted_frames(spinup_frames_dir) + sorted_frames(wind_frames_dir)

if all_frames:
    images = [iio.imread(frame) for frame in all_frames]
    gif_path = figures_dir / "turbulent_stellar_wind.gif"
    iio.imwrite(gif_path, images, duration=int(1000 / gif_fps), loop=0)
    print(f"GIF created at: {gif_path}")
else:
    print("No frames were captured; skipping GIF assembly.")

# -------------------------------------------------------------
# =============== ↑ Assemble GIF over the whole run ↑ =========
# -------------------------------------------------------------
