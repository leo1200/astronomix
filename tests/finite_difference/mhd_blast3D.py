# This tests follows the one presented in Fig. 12 in
# https://doi.org/10.48550/arXiv.2004.10542

# TODO: test if Lax-Friedrichs flux removes the problem of the "small waves"

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# =======================

# numerics
import jax
import jax.numpy as jnp

double_precision = False

if double_precision:
    jax.config.update("jax_enable_x64", True)

from jax.numpy.fft import fftn, ifftn

from astronomix._finite_difference._maths._differencing import finite_difference_int6


from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)

from astronomix._finite_difference._magnetic_update._constrained_transport import (
    constrained_transport_rhs,
    initialize_interface_fields,
)

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD,
    FINITE_DIFFERENCE,
    GHOST_CELLS,
    HLLC_LM,
    LAX_FRIEDRICHS,
    PERIODIC_ROLL,
    VAN_ALBADA_PP,
    SnapshotSettings,
    finalize_config,
)
import numpy as np
from matplotlib.colors import LogNorm

from astronomix._finite_volume._magnetic_update._magnetic_field_update import (
    magnetic_update,
)

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


from astronomix._finite_difference._fluid_equations._equations import (
    conserved_state_from_primitive_mhd,
)
# from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_face_centered_b

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    FORWARDS,
    HLL,
    HLLC,
    MINMOD,
    OSHER,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)


def run_blast_simulation(num_cells, B0, theta, phi):
    # spatial domain
    box_size = 1.0

    # setup simulation config
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        runtime_debugging=False,
        progress_bar=True,
        mhd=True,
        dimensionality=3,
        box_size=box_size,
        num_cells=num_cells,
        boundary_settings=BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ),
        return_snapshots=True,
        num_snapshots = 30,
        snapshot_settings=SnapshotSettings(
            return_states = False,
            return_final_state = True,
            return_magnetic_divergence = True,
        )
    )

    helper_data = get_helper_data(config)

    params = SimulationParams(t_end=0.01, C_cfl=1.5, gamma=5 / 3)

    registered_variables = get_registered_variables(config)

    r = helper_data.r

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 0.1
    r_inj = 0.1 * box_size
    r_tap = 1.1 * r_inj
    p_inj = 1000
    P = jnp.where(r**2 < r_inj**2, p_inj, P)
    P = jnp.where(
        (r**2 >= r_inj**2) & (r**2 <= r_tap**2),
        0.1 + (p_inj - 0.1) * (r_tap - r) / (r_tap - r_inj),
        P,
    )

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

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=V_x,
        velocity_y=V_y,
        velocity_z=V_z,
        magnetic_field_x=B_x,
        magnetic_field_y=B_y,
        magnetic_field_z=B_z,
        interface_magnetic_field_x=bxb,
        interface_magnetic_field_y=byb,
        interface_magnetic_field_z=bzb,
        gas_pressure=P,
    )

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, registered_variables, params, helper_data


num_cells = 300
B0 = 100 / jnp.sqrt(4 * jnp.pi)
theta = jnp.pi / 2
phi = jnp.pi / 4

initial_state, config, registered_variables, params, helper_data = run_blast_simulation(
    num_cells, B0, theta, phi
)

bxb = initial_state[registered_variables.interface_magnetic_field_index.x]
byb = initial_state[registered_variables.interface_magnetic_field_index.y]
bzb = initial_state[registered_variables.interface_magnetic_field_index.z]

conserved_state = conserved_state_from_primitive_mhd(
    primitive_state=initial_state[:-3],
    gamma=params.gamma,
    registered_variables=registered_variables,
)

c1, c2, c3 = 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0
divergence = jnp.mean(
    jnp.abs(
        1.0
        / config.grid_spacing
        * (
            finite_difference_int6(bxb, axis=0)
            + finite_difference_int6(byb, axis=1)
            + finite_difference_int6(bzb, axis=2)
        )
    )
)
print(divergence)

# Calculate fluxes based on the state of the current stage
dF_x = _weno_flux_x(conserved_state, params.gamma, registered_variables)
dF_y = _weno_flux_y(conserved_state, params.gamma, registered_variables)
dF_z = _weno_flux_z(conserved_state, params.gamma, registered_variables)

# Calculate RHS for interface magnetic fields using Constrained Transport
rhs_bx, rhs_by, rhs_bz = constrained_transport_rhs(
    conserved_state, dF_x, dF_y, dF_z, 1.0, 1.0, 1.0, registered_variables
)

c1, c2, c3 = 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0
divergence = jnp.abs(
    1.0
    / config.grid_spacing
    * (
        finite_difference_int6(rhs_bx, axis=0)
        + finite_difference_int6(rhs_by, axis=1)
        + finite_difference_int6(rhs_bz, axis=2)
    )
)
print("rhs div B:", jnp.mean(divergence))

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(divergence[:, :, num_cells // 2], origin="lower")
fig.colorbar(im, ax=ax, label="|div B|")
ax.set_title("Divergence of RHS of B field at center slice")
plt.savefig("figures/3d_blast_divergence_rhs.png", dpi=300)
plt.tight_layout()
plt.close(fig)

run_simulation = True

if run_simulation:
    result = time_integration(
        initial_state, config, params, registered_variables
    )
    # save final state
    jnp.save("data/mhd_blast3D.npy", result.final_state)
else:
    final_state = jnp.load("data/mhd_blast3D.npy")

div_b = result.magnetic_divergence
time_points = result.time_points
final_state = result.final_state

# plot the divergence evolution
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(time_points, div_b)
ax.set_ylabel("<|div B|>")
ax.set_xlabel("time")
ax.set_title("Evolution of mean |div B|")
plt.savefig("figures/3D_blast_magnetic_divergence.svg")
plt.close(fig)

bxb, byb, bzb = final_state[-3:, :]

c1, c2, c3 = 75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0
divergence = jnp.abs(
    1.0
    / config.grid_spacing
    * (
        finite_difference_int6(bxb, axis=0)
        + finite_difference_int6(byb, axis=1)
        + finite_difference_int6(bzb, axis=2)
    )
)

print("final div B:", jnp.mean(divergence))

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(divergence[:, :, num_cells // 2], origin="lower", vmax=1e-2)
fig.colorbar(im, ax=ax, label="|div B|")
ax.set_title("Divergence of B field at center slice")
fig.savefig("figures/3d_blast_divergence_final.png", dpi=300)
plt.close(fig)

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
    density[:, :, num_cells // 2],
    origin="lower",
    extent=(0, config.box_size, 0, config.box_size),
    cmap="jet",
    vmin=0.2,
    vmax=3.5,
)
cbar = make_axes_locatable(axs[0, 0]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label="density")
axs[0, 0].set_title("density slice")
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel("y")

# log pressure
im = axs[0, 1].imshow(
    jnp.log10(pressure[:, :, num_cells // 2]),
    origin="lower",
    extent=(0, config.box_size, 0, config.box_size),
    cmap="jet",
    vmin=-1.0,
    vmax=2.3,
)
cbar = make_axes_locatable(axs[0, 1]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label="pressure")
axs[0, 1].set_title("pressure slice")
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel("y")

# 1, 0: v^2/2
im = axs[1, 0].imshow(
    v2_half[:, :, num_cells // 2],
    origin="lower",
    extent=(0, config.box_size, 0, config.box_size),
    cmap="jet",
    vmin=0.0,
    vmax=160.0,
)
cbar = make_axes_locatable(axs[1, 0]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label="v^2/2")
axs[1, 0].set_title("kinetic energy slice")
axs[1, 0].set_xlabel("x")
axs[1, 0].set_ylabel("y")

# 1, 1: B^2/2
im = axs[1, 1].imshow(
    magnetic_pressure[:, :, num_cells // 2],
    origin="lower",
    extent=(0, config.box_size, 0, config.box_size),
    cmap="jet",
    vmin=170,
    vmax=480,
)
cbar = make_axes_locatable(axs[1, 1]).append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cbar, label="B^2/2")
axs[1, 1].set_title("magnetic pressure slice")
axs[1, 1].set_xlabel("x")
axs[1, 1].set_ylabel("y")

# 0, 2: |B|^2 / 2 along the diagonal from the center
diag_indices = jnp.arange(num_cells // 2, num_cells)
B_diag = magnetic_pressure[diag_indices, diag_indices, num_cells // 2]
r_diag = jnp.sqrt(
    (diag_indices - num_cells // 2) ** 2 + (diag_indices - num_cells // 2) ** 2
) * (config.box_size / num_cells)
axs[0, 2].plot(r_diag, B_diag)
axs[0, 2].set_ylabel("|B|^2 / 2")
axs[0, 2].set_xlabel("r")
axs[0, 2].set_xlim(0, 0.3)
axs[0, 2].set_ylim(180, 270)
axs[0, 2].set_title("|B|^2 / 2 along diagonal")

# density along the vertical centerline
density_center = density[num_cells // 2, num_cells // 2, :]
axs[1, 2].plot(jnp.linspace(0, config.box_size, num_cells), density_center)
axs[1, 2].set_ylabel("density")
axs[1, 2].set_xlabel("z")
axs[1, 2].set_xlim(0.5, 1.0)
axs[1, 2].set_ylim(0.0, 1.5)
axs[1, 2].set_title("rho along vertical centerline")

plt.tight_layout()
plt.savefig("figures/mhd_blast3D.png", dpi=300)