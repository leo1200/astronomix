"""
Trubulent radiative mixing layer simulations.

Setup based on Lancaster 2026 (unpublished):

Basic setup:

- box with dimensions (Lx, Ly, Lz) = (L_box, L_box, 1.5 * L_box), L_box = 1.0
- boundaries
    - x: periodic
    - y: periodic
    - z: bottom outflow, top fixed to U_hot (see below)
- the state z > z_center is the hot state
- the state z < z_center is the cold state
- initialization with two constant states along z,
  with a smooth transition at z_center = L_box / 2
    - a tanh profile of thickness \Delta x / 2, with
      \Delta x = L_box / num_cells
- these states are given by
    - U_hot = (\rho, v_x, v_y, v_z, P)_hot = (\rho_0, v_rel / 2, 0, 0, P_0)
    - U_cold = (\rho, v_x, v_y, v_z, P)_cold = (χ \rho_0, -v_rel / 2, 0, 0, P_0)
  with \rho_0 = P_0 = 1. v_rel is specified by the Mach number M, where
    - M = v_rel / c_hot (c_hot as defined below)
- as a simplification, T := P / \rho is used in the setup
- the adiabatic sound speed in the hot medium is
    - c_hot = sqrt(γ P_0 / \rho_0) = sqrt(γ)
- the temperature of the two phases are
    - T_hot = P_0 / \rho_0 = 1
    - T_cold = P_0 / (χ \rho_0) = 1 / χ

Cooling function: TODO

Viscosity can be used to stability simulations.

"""

multi_gpu = False

if __name__ == "__main__":
    if multi_gpu:
        # ==== GPU selection ====
        from autocvd import autocvd
        autocvd(num_gpus=4)
        # ruff: noqa: E402
        # =======================
    else:
        # ==== GPU selection ====
        from autocvd import autocvd
        autocvd(num_gpus=1)
        # ruff: noqa: E402
        # =======================

# typing
from dataclasses import dataclass
from typing import NamedTuple

# numerics
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.colors as mcolors

# astronomix
from astronomix import (
    SimulationConfig,
    get_helper_data,
    SimulationParams,
    time_integration,
    construct_primitive_state,
    get_registered_variables,
)
from astronomix.option_classes.simulation_params import FixedBoundaryState, FixedBoundaryState1D
from astronomix.option_classes.simulation_config import (
    FINITE_VOLUME,
    FIXED_BOUNDARY,
    FIXED_BOUNDARY_OPEN_MOMENTUM,
    KINEMATIC_VISCOSITY,
    MINMOD,
    MUSCL,
    OPEN_BOUNDARY,
    PALLAS,
    SPLIT,
    UNSPLIT,
    VARAXIS,
    XAXIS,
    YAXIS,
    ZAXIS,
    SnapshotSettings,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix._modules._cooling.cooling_options import EXPLICIT_COOLING, IMPLICIT_COOLING, SIMPLE_MIXING_LAYER_COOLING, CoolingConfig, CoolingCurveConfig, CoolingParams, MixingCoolingParams

from jax.sharding import PartitionSpec as P

if multi_gpu:

    # mesh with variable axis
    split = (1, 1, 2, 2)
    sharding_mesh = jax.make_mesh(split, (VARAXIS, XAXIS, YAXIS, ZAXIS))
    named_sharding = jax.NamedSharding(sharding_mesh, P(VARAXIS, XAXIS, YAXIS, ZAXIS))

    # mesh no variable axis
    split = (1, 2, 2)
    sharding_mesh_no_var = jax.make_mesh(split, (XAXIS, YAXIS, ZAXIS))
    named_sharding_no_var = jax.NamedSharding(sharding_mesh_no_var, P(XAXIS, YAXIS, ZAXIS))

# Box setup
num_cells_x = 128
num_cells_y = num_cells_x
num_cells_z = int(1.5 * num_cells_x)
box_size = 1.0
grid_spacing = box_size / num_cells_x # the same in all directions
L_x = box_size
L_y = box_size
L_z = 1.5 * box_size

# Simulation time
t_end_in_t_sh = 30.0

# Mixing settings
density_contrast = 100.0
xi = 100.0
mach_number = 0.5
gamma = 5 / 3
P0 = 1.0

# Derived quantities
rho_hot = 1.0
rho_cold = density_contrast * rho_hot
T_hot = P0 / rho_hot
T_cold = P0 / rho_cold
c_hot = (gamma * P0 / rho_hot) ** 0.5
v_rel = mach_number * c_hot
t_sh = L_x / v_rel
t_coolmin = t_sh / xi

# Simulation config
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    backend = PALLAS,
    pallas_block_shape = (4, 4, 8),
    pallas_use_triton = True,
    pallas_interpret = False,
    memory_analysis = True,
    print_elapsed_time = True,
    progress_bar = True,
    dimensionality = 3,
    box_size = StaticFloatVector(L_x, L_y, L_z),
    num_cells = StaticIntVector(num_cells_x, num_cells_y, num_cells_z),
    # viscosity_type = KINEMATIC_VISCOSITY,
    boundary_settings = BoundarySettings(
        x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        z = BoundarySettings1D(OPEN_BOUNDARY, FIXED_BOUNDARY_OPEN_MOMENTUM),
    ),
    enforce_positivity = True,
	cooling_config = CoolingConfig(
        cooling = True,
        cooling_method = IMPLICIT_COOLING,
        cooling_curve_config = CoolingCurveConfig(
            cooling_curve_type = SIMPLE_MIXING_LAYER_COOLING,
        )
    ),
	frame_tracking = True,
    return_snapshots = True,
    num_snapshots = 100,
    snapshot_settings=SnapshotSettings(
        return_final_state=True,
        return_states=False,
        return_temperature_pdf=True,
        num_temperature_bins=100,
        temperature_pdf_min=T_cold,
        temperature_pdf_max=T_hot,
    )
)

# tanh transition
def single_interface(f_l, f_u, Z, z_center, smoothing_length):
	return 0.5 * (
		f_l * (1 - jnp.tanh((Z - z_center) / smoothing_length)) 
		+ f_u * (1 + jnp.tanh((Z - z_center) / smoothing_length))
	)

registered_variables = get_registered_variables(config)

setup_initial_state = True

if setup_initial_state:
    helper_data = get_helper_data(config, sharding=named_sharding if multi_gpu else None)
    # construct the initial state
    cell_centers = helper_data.geometric_centers
    X = cell_centers[:, :, :, 0]
    Y = cell_centers[:, :, :, 1]
    Z = cell_centers[:, :, :, 2]
    z_center = L_z / 2
    smoothing_length = grid_spacing / 2
    density = single_interface(rho_cold, rho_hot, Z, z_center, smoothing_length)
    pressure = P0 * jnp.ones_like(density)
    velocity_x = single_interface(-v_rel / 2, v_rel / 2, Z, z_center, smoothing_length)
    velocity_y = jnp.zeros_like(density)
    velocity_z = jnp.zeros_like(density)


    # --- Perturbation ----------------------------------------------------------
    # Envelope: peaks in the tanh tails (±2 * smoothing_length from z_center),
    # zero in the bulk phases and at the sharpest point of the interface itself
    # (where WENO's shock sensor would damp perturbations most aggressively).
    dz       = Z - z_center
    envelope = jnp.exp(-((jnp.abs(dz) - 2 * smoothing_length) / (3 * grid_spacing))**2)

    amp = 0.03 * v_rel  # 3% of shear velocity

    # Deterministic KH modes: low-k, well-resolved at N_res = 64 (10-30 cells/wavelength)
    mode_numbers = jnp.array([2, 4, 6])
    kx = 2 * jnp.pi * mode_numbers / L_x
    ky = 2 * jnp.pi * mode_numbers / L_y

    key           = jax.random.PRNGKey(42)
    key_ph, key_n = jax.random.split(key, 2)
    phases        = jax.random.uniform(key_ph, (3, 3), minval=0.0, maxval=2 * jnp.pi)

    modes = jnp.zeros_like(density)
    for i in range(3):
        for j in range(3):
            modes = modes + jnp.sin(kx[i] * X + ky[j] * Y + phases[i, j])
    modes = modes / 9.0  # normalize to O(1)

    # Broadband noise: low amplitude, fills the spectrum above the deterministic modes
    key_nx, key_ny, key_nz = jax.random.split(key_n, 3)
    noise_x = jax.random.normal(key_nx, density.shape)
    noise_y = jax.random.normal(key_ny, density.shape)
    noise_z = jax.random.normal(key_nz, density.shape)

    # v_z gets the full deterministic+noise treatment (this is the KH growth direction)
    # v_x and v_y get noise only (adding modes to v_x would fight the mean shear)
    velocity_x = velocity_x +       amp * envelope * 0.3 * noise_x
    velocity_y = velocity_y +       amp * envelope * 0.3 * noise_y
    velocity_z = velocity_z + amp * envelope * (modes + 0.3 * noise_z)

    initial_state = construct_primitive_state(
        config               = config,
        registered_variables = registered_variables,
        density              = density,
        velocity_x           = velocity_x,
        velocity_y           = velocity_y,
        velocity_z           = velocity_z,
        gas_pressure         = pressure,
    )

    # save initial state
    jnp.savez("data/trml_initial_state.npz", initial_state=initial_state)
else:
    # load initial state
    initial_state = jnp.load("data/trml_initial_state.npz")["initial_state"]

if multi_gpu:
    initial_state = jax.device_put(initial_state, named_sharding)

mixing_cooling_params = MixingCoolingParams(
    xi = xi,
    mach_number = mach_number,
    density_contrast = density_contrast,
)

# set simulation params
params = SimulationParams(
    # viscosity = viscosity,
    t_end = t_end_in_t_sh * t_sh,
    C_cfl = 1.5 if config.solver_mode == FINITE_DIFFERENCE else 0.4, 
    gamma = gamma,
    minimum_density = rho_cold / 100,
    minimum_pressure = P0 / 100,
	fixed_boundary_state = FixedBoundaryState(
		z = FixedBoundaryState1D(
			right_state = jnp.array([rho_hot, v_rel / 2, 0.0, 0.0, P0])
        )
    ),
	cooling_params = CoolingParams(
		cooling_curve_params = mixing_cooling_params,
        floor_temperature = T_cold,
    )
)

# finalize config
config = finalize_config(config, initial_state.shape)

if __name__ == "__main__":

    # run the simulation
    result = time_integration(
        initial_state,
        config,
        params,
        registered_variables,
        sharding = named_sharding if multi_gpu else None
    )

    # save the final state for later analysis
    jnp.savez("data/trml_final_state.npz", final_state=result.final_state)

    # retrieve the initial and final temperature
    initial_temperature = (
        initial_state[registered_variables.pressure_index] / initial_state[registered_variables.density_index]
    )
    final_temperature = (
        result.final_state[registered_variables.pressure_index] / result.final_state[registered_variables.density_index]
    )

    # plot the results
    y_index = num_cells_y // 2

    fig_temp, (ax_temp_initial, ax_temp_final) = plt.subplots(1, 2, figsize=(9, 5))

    im_initial = ax_temp_initial.imshow(
        initial_temperature[:, y_index, :].T,
        origin="lower",
        extent=[0, L_x, 0, L_z],
        norm = LogNorm()
    )
    ax_temp_initial.set_title("Initial Temperature")
    ax_temp_initial.set_xlabel("x")
    ax_temp_initial.set_ylabel("z")
    divider_initial = make_axes_locatable(ax_temp_initial)
    cax_initial = divider_initial.append_axes("right", size="5%", pad=0.1)
    fig_temp.colorbar(im_initial, cax=cax_initial, label="Temperature")

    im_final = ax_temp_final.imshow(
        final_temperature[:, y_index, :].T,
        origin="lower",
        extent=[0, L_x, 0, L_z],
        norm = LogNorm()
    )
    ax_temp_final.set_title("Final Temperature")
    ax_temp_final.set_xlabel("x")
    ax_temp_final.set_ylabel("z")
    divider_final = make_axes_locatable(ax_temp_final)
    cax_final = divider_final.append_axes("right", size="5%", pad=0.1)
    fig_temp.colorbar(im_final, cax=cax_final, label="Temperature")

    fig_temp.tight_layout()
    fig_temp.savefig("figures/trml_temperature.png", dpi=500)

    # plot the average of the temperature PDF for t >= 10 t_sh
    time_points = result.time_points
    temperature_pdf = result.temperature_pdf
    time_mask = time_points >= 10 * t_sh
    print("Number of snapshots with t >= 10 t_sh:", time_mask.sum())
    avg_dV_dlogT = temperature_pdf[time_mask].mean(axis=0)
    logT_bin_edges = jnp.linspace(jnp.log10(T_cold), jnp.log10(T_hot), config.snapshot_settings.num_temperature_bins + 1)
    logT_bin_centers = 0.5 * (logT_bin_edges[:-1] + logT_bin_edges[1:])
    T_bin_centers = 10 ** logT_bin_centers
    avg_dV_dT = avg_dV_dlogT / (T_bin_centers * jnp.log(10))  # convert from dV/dlogT to dV/dT
    # normalize dV/dT so that its integral over T is 1 (i.e. it becomes a proper PDF)
    V = jnp.sum(avg_dV_dT * (10 ** logT_bin_edges[1:] - 10 ** logT_bin_edges[:-1]))
    avg_dV_dT = avg_dV_dT / V
    fig_pdf, ax_pdf = plt.subplots()
    ax_pdf.plot(T_bin_centers, avg_dV_dT)
    ax_pdf.set_xlabel("log10(Temperature)")
    ax_pdf.set_ylabel("dV/dT")
    ax_pdf.set_title("Average Temperature PDF for t >= 10 t_sh")
    ax_pdf.set_xscale("log")
    ax_pdf.set_yscale("log")
    fig_pdf.savefig("figures/trml_temperature_pdf.png", dpi=500)