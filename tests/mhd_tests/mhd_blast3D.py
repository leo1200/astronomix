# This tests follows the one presented in Fig. 12 in
# https://doi.org/10.48550/arXiv.2004.10542

# TODO: test if Lax-Friedrichs flux removes the problem of the "small waves"

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, HLLC_LM, LAX_FRIEDRICHS, VAN_ALBADA_PP, finalize_config
import numpy as np
from matplotlib.colors import LogNorm

from astronomix._finite_volume._magnetic_update._magnetic_field_update import magnetic_update

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

def run_blast_simulation(num_cells, B0, theta, phi):

    # spatial domain
    box_size = 1.0

    # setup simulation config
    config = SimulationConfig(
        runtime_debugging = False,
        progress_bar = True,
        mhd = True,
        dimensionality = 3,
        box_size = box_size, 
        num_cells = num_cells,
        limiter = MINMOD,
        riemann_solver = HLL,
        exact_end_time = True,
        boundary_settings = BoundarySettings(
            BoundarySettings1D(
                left_boundary = PERIODIC_BOUNDARY,
                right_boundary = PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary = PERIODIC_BOUNDARY,
                right_boundary = PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary = PERIODIC_BOUNDARY,
                right_boundary = PERIODIC_BOUNDARY
            )
        ),
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(
        t_end = 0.01,
        C_cfl = 0.4,
        gamma = 5/3
    )

    registered_variables = get_registered_variables(config)

    r = helper_data.r

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 0.1
    r_inj = 0.1 * box_size
    p_inj = 1000
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B_x = B0 * jnp.sin(theta) * jnp.cos(phi)
    B_y = B0 * jnp.sin(theta) * jnp.sin(phi)
    B_z = B0 * jnp.cos(theta)

    print(f"Magnetic field: Bx={B_x}, By={B_y}, Bz={B_z}")

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

    initial_state = construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = V_x,
        velocity_y = V_y,
        velocity_z = V_z,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        gas_pressure = P
    )

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, registered_variables, params, helper_data

num_cells = 192
B0 = 100 / jnp.sqrt(4 * jnp.pi)
theta = jnp.pi / 2
phi = jnp.pi / 4

initial_state, config, registered_variables, params, helper_data = run_blast_simulation(num_cells, B0, theta, phi)

run_simulation = True

if run_simulation:
    final_state = time_integration(initial_state, config, params, registered_variables)
    # save final state
    jnp.save('data/mhd_blast3D.npy', final_state)
else:
    final_state = jnp.load('data/mhd_blast3D.npy')

# plot
density = final_state[registered_variables.density_index]
pressure = final_state[registered_variables.pressure_index]
Bx = final_state[registered_variables.magnetic_index.x]
By = final_state[registered_variables.magnetic_index.y]
Bz = final_state[registered_variables.magnetic_index.z]
vx = final_state[registered_variables.velocity_index.x]
vy = final_state[registered_variables.velocity_index.y]
vz = final_state[registered_variables.velocity_index.z]
magnetic_pressure = 0.5 * (Bx**2 + By**2 + Bz**2)
v2_half = 0.5 * (vx**2 + vy**2 + vz**2)

fig, axs = plt.subplots(2, 3, figsize=(9, 6))

# density
im = axs[0, 0].imshow(
    density[:, :, num_cells//2],
    origin='lower',
    extent=(0, config.box_size, 0, config.box_size),
    cmap = "jet",
    vmin = 0.2,
    vmax = 3.5
)
cbar = make_axes_locatable(axs[0, 0]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label='density')
axs[0, 0].set_title('density slice')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('y')

# log pressure
im = axs[0, 1].imshow(
    jnp.log10(pressure[:, :, num_cells//2]),
    origin='lower',
    extent=(0, config.box_size, 0, config.box_size),
    cmap="jet",
    vmin = -1.0,
    vmax = 2.3
)
cbar = make_axes_locatable(axs[0, 1]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label='pressure')
axs[0, 1].set_title('pressure slice')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('y')

# 1, 0: v^2/2
im = axs[1, 0].imshow(
    v2_half[:, :, num_cells//2],
    origin='lower',
    extent=(0, config.box_size, 0, config.box_size),
    cmap = "jet",
    vmin = 0.0,
    vmax = 160.0
)
cbar = make_axes_locatable(axs[1, 0]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label='v^2/2')
axs[1, 0].set_title('kinetic energy slice')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('y')

# 1, 1: B^2/2
im = axs[1, 1].imshow(
    magnetic_pressure[:, :, num_cells//2],
    origin='lower',
    extent=(0, config.box_size, 0, config.box_size),
    cmap = "jet",
    vmin = 170,
    vmax = 480
)
cbar = make_axes_locatable(axs[1, 1]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label='B^2/2')
axs[1, 1].set_title('magnetic pressure slice')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('y')

# 0, 2: |B|^2 / 2 along the diagonal from the center
diag_indices = jnp.arange(num_cells // 2, num_cells)
B_diag = magnetic_pressure[diag_indices, diag_indices, num_cells//2]
r_diag = jnp.sqrt((diag_indices - num_cells//2)**2 + (diag_indices - num_cells//2)**2) * (config.box_size / num_cells)
axs[0, 2].plot(r_diag, B_diag)
axs[0, 2].set_ylabel('|B|^2 / 2')
axs[0, 2].set_xlabel('r')
axs[0, 2].set_xlim(0, 0.3)
axs[0, 2].set_ylim(180, 270)
axs[0, 2].set_title('|B|^2 / 2 along diagonal')

# density along the vertical centerline
density_center = density[num_cells//2, num_cells//2, :]
axs[1, 2].plot(jnp.linspace(0, config.box_size, num_cells), density_center)
axs[1, 2].set_ylabel('density')
axs[1, 2].set_xlabel('z')
axs[1, 2].set_xlim(0.5, 1.0)
axs[1, 2].set_ylim(0.0, 1.5)
axs[1, 2].set_title('rho along vertical centerline')

plt.tight_layout()
plt.savefig('figures/mhd_blast3D.png', dpi=300)