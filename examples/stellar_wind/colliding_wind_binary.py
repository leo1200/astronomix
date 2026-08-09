"""
3D colliding-wind binary: two stellar winds collide inside a turbulent, warm
ambient medium, with the two stars on an eccentric, inclined orbit around
their common center of mass (N-body point-mass gravity).

Runs one representative simulation, then produces two figures: density /
temperature / pressure central slices at the end time, and an animation of
the colliding-wind density slice.
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
import numpy as np

# units
from astropy import units as u
import astropy.constants as c

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation, PillowWriter

# astronomix constants
from astronomix import OPEN_BOUNDARY, FORWARDS
from astronomix.option_classes.simulation_config import MUSCL, SPLIT, MINMOD, FINITE_VOLUME

# astronomix containers
from astronomix import (
    CodeUnits,
    WindParams,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
)
from astronomix.option_classes import WindConfig, NBodyConfig, NBodyParams, NGP
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
    TurbulentForcingParams,
)

# astronomix functions
from astronomix import (
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    finalize_config,
)
from astronomix._modules._nbody._nbody import binary_starting_orbits_at_phase


# figures are written to the local figures/ directory
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# =============== ↓ Configuration ↓ ===========================
# -------------------------------------------------------------
gamma = 5 / 3
box_size = 1.0
num_cells = 256

config = SimulationConfig(
    progress_bar=True,
    first_order_fallback=True,
    dimensionality=3,
    box_size=box_size,
    num_cells=num_cells,
    solver_mode=FINITE_VOLUME,
    split=SPLIT,
    limiter=MINMOD,
    time_integrator=MUSCL,
    differentiation_mode=FORWARDS,
    wind_config=WindConfig(
        stellar_wind=True,
        num_injection_cells=8,
        trace_wind_density=False,
    ),
    # the wind-injection positions are taken from the N-body state every
    # step (see astronomix._modules._stellar_wind.stellar_wind), so the two
    # wind sources follow the binary orbit computed below.
    nbody_config=NBodyConfig(
        nbody=True,
        deposit_particles=NGP,
        central_object_only=False,
    ),
    turbulent_forcing_config=TurbulentForcingConfig(
        turbulent_forcing=False,
    ),
    return_snapshots=True,
    num_snapshots=15,
    snapshot_settings=SnapshotSettings(
        return_states=True,
        return_final_state=True,
    ),
    boundary_settings=BoundarySettings(
        BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
    ),
)

registered_variables = get_registered_variables(config)
# -------------------------------------------------------------
# =============== ↑ Configuration ↑ ===========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Units and physical parameters ↓ ===========
# -------------------------------------------------------------
C_CFL = 0.4
dt_max = 0.1

sep_in_au = 20 # upper limit: 80, lower limit: 5
length_temp = sep_in_au / 10
mass_temp = 1
code_length = length_temp * u.au
code_mass = mass_temp * u.M_sun
code_velocity = np.sqrt(c.G * code_mass / code_length).to(u.km / u.s)
code_units = CodeUnits(code_length, code_mass, code_velocity)

a = box_size / 3.  # binary semi-major axis (code length)


# one representative pair of massive, wind-driving stars on an eccentric,
# inclined orbit -- the numbers below correspond to a "typical" draw from the
# parameter ranges the original script sampled at random.
M1 = 50 * u.M_sun
M2 = 40 * u.M_sun
mass_loss_rate_1 = 6.5e-7 * u.M_sun / u.yr
mass_loss_rate_2 = 6.5e-8 * u.M_sun / u.yr
wind_velocity_1 = 2200 * u.km / u.s
wind_velocity_2 = 1850 * u.km / u.s
eccentricity = 0.5
cos_inclination = 1.0
turbulence_strength = 0.0

m1 = M1.to(code_units.code_mass).value
m2 = M2.to(code_units.code_mass).value
Period = 2 * np.pi * np.sqrt(a**3 / ((m1 + m2)))

# T_end = (3.5 * u.yr).to(code_units.code_time).value
T_end = Period
print(f"End time in code units: {T_end}")

m1 = M1.to(code_units.code_mass).value
m2 = M2.to(code_units.code_mass).value
mlr1 = mass_loss_rate_1.to(code_units.code_mass / code_units.code_time).value
mlr2 = mass_loss_rate_2.to(code_units.code_mass / code_units.code_time).value
v_inf1 = wind_velocity_1.to(code_units.code_velocity).value
v_inf2 = wind_velocity_2.to(code_units.code_velocity).value
inclination_deg = float(jnp.rad2deg(jnp.arccos(cos_inclination)))

masses = jnp.array([m1, m2])
# initial N-body phase-space state [t, x, y, z, vx, vy, vz] per body,
# flattened -- seeds astronomix._modules._nbody._nbody's RK4 integrator,
# advanced jointly with the hydro update every step.
nbody_state = binary_starting_orbits_at_phase(
    a, eccentricity, inclination_deg, m1, m2, phi=0.0, true_anom_deg=0.0,
)

wind_luminosity = 0.5 * (mlr1 + mlr2) / 2 * ((v_inf1 + v_inf2) / 2) ** 2
energy_injection_rate = turbulence_strength * wind_luminosity

params = SimulationParams(
    C_cfl=C_CFL,
    dt_max=dt_max,
    gamma=gamma,
    t_end=T_end,
    wind_params=WindParams(
        wind_mass_loss_rates=jnp.array([mlr1, mlr2]),
        wind_final_velocities=jnp.array([v_inf1, v_inf2]),
    ),
    turbulent_forcing_params=TurbulentForcingParams(
        energy_injection_rate=energy_injection_rate,
    ),
    nbody_params=NBodyParams(
        masses=masses,
        nbody_state=nbody_state,
    ),
)
# -------------------------------------------------------------
# =============== ↑ Units and physical parameters ↑ ===========
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Initial state ↓ ============================
# -------------------------------------------------------------
# homogeneous, warm ambient medium; the wind sources carve out their bubbles
# and collide as the simulation evolves.
rho_0 = 2 * c.m_p / u.cm**3
p_0 = 3e4 * u.K / u.cm**3 * c.k_B

shape = (num_cells, num_cells, num_cells)
rho = jnp.ones(shape) * rho_0.to(code_units.code_density).value
u_x = jnp.zeros(shape)
u_y = jnp.zeros(shape)
u_z = jnp.zeros(shape)
p = jnp.ones(shape) * p_0.to(code_units.code_pressure).value

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    velocity_z=u_z,
    gas_pressure=p,
)

config = finalize_config(config, initial_state.shape)
# -------------------------------------------------------------
# =============== ↑ Initial state ↑ ============================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Run ↓ =======================================
# -------------------------------------------------------------
print("👷 Running the colliding-wind binary simulation...")
snapshots = jax.block_until_ready(
    time_integration(initial_state, config, params, registered_variables)
)
# -------------------------------------------------------------
# =============== ↑ Run ↑ =======================================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======= ↓ Density / temperature / pressure slices ↓ ==========
# -------------------------------------------------------------
z = num_cells // 2
density_index = registered_variables.density_index
pressure_index = registered_variables.pressure_index

final_density = snapshots.states[-1, density_index][:, :, z]
final_pressure = snapshots.states[-1, pressure_index][:, :, z]
final_internal_energy = final_pressure / ((gamma - 1) * final_density)
final_temperature = code_units.get_temperature_from_internal_energy(
    np.asarray(final_internal_energy), gamma=gamma
).to(u.K).value

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

panels = [
    (final_density, "Density", "inferno", axes[0]),
    (final_temperature, "Temperature [K]", "plasma", axes[1]),
    (final_pressure, "Pressure", "viridis", axes[2]),
]
for field, title, cmap, ax in panels:
    field = np.asarray(field)
    positive = field[field > 0]
    vmin = float(np.min(positive)) if positive.size else float(np.min(field))
    vmax = float(np.max(field))
    im = ax.imshow(field.T, origin="lower", cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    ax.set_title(f"{title} (t = {float(snapshots.time_points[-1]):.3f})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

fig.tight_layout()
fig.savefig(figures_dir / "colliding_wind_binary_slices.png", dpi=200, bbox_inches="tight")
plt.close(fig)
# -------------------------------------------------------------
# ======= ↑ Density / temperature / pressure slices ↑ ==========
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Density-slice animation ↓ ===================
# -------------------------------------------------------------
time = snapshots.time_points
slices = snapshots.states[:, density_index][:, :, :, z]
slices_pos = slices[slices > 0]
anim_vmin = float(jnp.min(slices_pos)) if slices_pos.size else 1e-6
anim_vmax = float(jnp.max(slices))
anim_norm = LogNorm(vmin=anim_vmin, vmax=anim_vmax)

fig_a, ax_a = plt.subplots(figsize=(7, 7))
im_a = ax_a.imshow(np.asarray(slices[0]).T, origin="lower", cmap="inferno", norm=anim_norm)
title_a = ax_a.set_title("t = 0.00")
fig_a.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.04)


def _update(frame):
    im_a.set_data(np.asarray(slices[frame]).T)
    title_a.set_text(f"t = {float(time[frame]):.3f}")
    return im_a, title_a


anim = FuncAnimation(fig_a, _update, frames=slices.shape[0], blit=False)
anim.save(figures_dir / "colliding_wind_binary_density.gif", writer=PillowWriter(fps=6), dpi=100)
plt.close(fig_a)
# -------------------------------------------------------------
# =============== ↑ Density-slice animation ↑ ===================
# -------------------------------------------------------------

# Save snapshot states to npz file
np.savez(data_dir / "colliding_wind_binary_states.npz", states=np.asarray(snapshots.states), time_points=np.asarray(snapshots.time_points))

