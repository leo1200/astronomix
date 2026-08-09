"""
3D MHD stellar wind blown into a homogeneous magnetized medium (finite difference).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
from pathlib import Path

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# units and constants
from astropy import units as u
import astropy.constants as c

# astronomix constants
from astronomix import (
    BACKWARDS,
    PERIODIC_BOUNDARY,
)

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
from astronomix.option_classes import WindConfig

# astronomix functions
from astronomix import (
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    initialize_interface_fields,
    finalize_config,
)


# figures are written to the local figures/ directory
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# simulation settings
# ---------------------------------------------------------------------------
GAMMA = 5 / 3
BOX_SIZE = 1.0
NUM_CELLS = 64

config = SimulationConfig(
    mhd=True,
    progress_bar=True,
    dimensionality=3,
    num_ghost_cells=2,
    box_size=BOX_SIZE,
    num_cells=NUM_CELLS,
    wind_config=WindConfig(
        stellar_wind=True,
        num_injection_cells=8,
        trace_wind_density=False,
    ),
    differentiation_mode=BACKWARDS,
    snapshot_settings=SnapshotSettings(
        return_states=False,
        return_final_state=True,
    ),
    boundary_settings=BoundarySettings(
        BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
        BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY),
    ),
)

registered_variables = get_registered_variables(config)

# ---------------------------------------------------------------------------
# code units and physical parameters
# ---------------------------------------------------------------------------
code_units = CodeUnits(3 * u.parsec, 1 * u.M_sun, 100 * u.km / u.s)

t_final = 1.0 * 1e4 * u.yr
t_end = t_final.to(code_units.code_time).value

M_star = 40 * u.M_sun
wind_final_velocity = 2000 * u.km / u.s
wind_mass_loss_rate = 2.965e-3 / (1e6 * u.yr) * M_star

wind_params = WindParams(
    wind_mass_loss_rate=wind_mass_loss_rate.to(code_units.code_mass / code_units.code_time).value,
    wind_final_velocity=wind_final_velocity.to(code_units.code_velocity).value,
)

params = SimulationParams(
    C_cfl=0.9,
    dt_max=0.001,
    gamma=GAMMA,
    minimum_density=1e-8,
    minimum_pressure=1e-8,
    t_end=t_end,
    wind_params=wind_params,
)

# ---------------------------------------------------------------------------
# homogeneous magnetized ambient medium
# ---------------------------------------------------------------------------
rho_0 = 2 * c.m_p / u.cm ** 3
p_0 = 3e4 * u.K / u.cm ** 3 * c.k_B
B_0 = (0.1 * u.microgauss / c.mu0 ** 0.5).to(code_units.code_magnetic_field).value

shape = (NUM_CELLS, NUM_CELLS, NUM_CELLS)
rho = jnp.ones(shape) * rho_0.to(code_units.code_density).value
u_x = jnp.zeros(shape)
u_y = jnp.zeros(shape)
u_z = jnp.zeros(shape)
p = jnp.ones(shape) * p_0.to(code_units.code_pressure).value

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

final_state = time_integration(initial_state, config, params, registered_variables)

# z-midplane slices of density, velocity and pressure
z_level = NUM_CELLS // 2
ext = [0, 1, 0, 1]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
for ax in (ax1, ax2, ax3):
    ax.set_aspect("equal", "box")

ax1.imshow(
    final_state[registered_variables.density_index, :, :, z_level].T,
    origin="lower",
    extent=ext,
    norm=LogNorm(),
)
ax1.set_title("density")

vel = jnp.sqrt(
    final_state[registered_variables.velocity_index.x, :, :, z_level] ** 2
    + final_state[registered_variables.velocity_index.y, :, :, z_level] ** 2
)
ax2.imshow(vel.T, origin="lower", extent=ext)
ax2.set_title("velocity")

ax3.imshow(
    final_state[registered_variables.pressure_index, :, :, z_level].T,
    origin="lower",
    extent=ext,
    norm=LogNorm(),
)
ax3.set_title("pressure")

fig.suptitle("3D MHD stellar wind (z-midplane)")
fig.tight_layout()
fig.savefig(figures_dir / "stellar_wind_3d.png", dpi=150)
