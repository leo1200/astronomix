"""TRML runs with explicit kinematic viscosity and thermal conduction.

Same setup as ``trml.py`` (turbulent radiative mixing layer, Mach 0.5,
density contrast 100, implicit mixing-layer cooling, FD/WENO + Pallas) but with
the explicit diffusion terms switched on:

    - constant kinematic viscosity   nu    = 1e-3   (config.diffusion,
      viscosity_type = KINEMATIC_VISCOSITY)
    - constant thermal conductivity  kappa = 1e-3   (config.thermal_conduction)

Run for a single resolution and dump the final primitive state (plus the
temperature PDF history) so the diffusivity distributions can be analysed
offline by ``diffusivity_analysis.py``.

Usage (from tests/trml, with the repo root on PYTHONPATH):

    PYTHONPATH=<repo-root> python trml_diffusivity.py --num-cells 64
"""

import argparse
import os

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
import jax.numpy as jnp

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
    FIXED_BOUNDARY_OPEN_MOMENTUM,
    KINEMATIC_VISCOSITY,
    OPEN_BOUNDARY,
    PALLAS,
    SnapshotSettings,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    PositivityConfig,
)
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    SIMPLE_MIXING_LAYER_COOLING,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    MixingCoolingParams,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data", "diffusivity")
os.makedirs(DATA_DIR, exist_ok=True)


def build(num_cells_x):
    # ---- Box setup -------------------------------------------------------
    num_cells_y = num_cells_x
    num_cells_z = int(1.5 * num_cells_x)
    box_size = 1.0
    grid_spacing = box_size / num_cells_x  # the same in all directions
    L_x = box_size
    L_y = box_size
    L_z = 1.5 * box_size

    # ---- Physics constants ----------------------------------------------
    t_end_in_t_sh = 30.0
    density_contrast = 100.0
    xi = 100.0
    mach_number = 0.5
    gamma = 5 / 3
    P0 = 1.0

    rho_hot = 1.0
    rho_cold = density_contrast * rho_hot
    T_hot = P0 / rho_hot
    T_cold = P0 / rho_cold
    c_hot = (gamma * P0 / rho_hot) ** 0.5
    v_rel = mach_number * c_hot
    t_sh = L_x / v_rel

    # ---- Explicit diffusion parameters ----------------------------------
    kinematic_viscosity = 1e-3
    thermal_conductivity = 1e-3

    # ---- Config ----------------------------------------------------------
    config = SimulationConfig(
        positivity_config=PositivityConfig(default_positivity_protection=True),
        solver_mode=FINITE_DIFFERENCE,
        backend=PALLAS,
        # diffusion bumps the ghost-cell ring to 6 -> padded dims are N+12,
        # which is divisible by 4 (not 8); use a (4,4,4) Pallas block.
        pallas_block_shape=(4, 4, 4),
        pallas_use_triton=True,
        pallas_interpret=False,
        memory_analysis=True,
        print_elapsed_time=True,
        progress_bar=True,
        dimensionality=3,
        box_size=StaticFloatVector(L_x, L_y, L_z),
        num_cells=StaticIntVector(num_cells_x, num_cells_y, num_cells_z),
        # explicit kinematic viscosity
        diffusion=True,
        viscosity_type=KINEMATIC_VISCOSITY,
        # explicit thermal conduction
        thermal_conduction=True,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, FIXED_BOUNDARY_OPEN_MOMENTUM),
        ),
        cooling_config=CoolingConfig(
            cooling=True,
            cooling_method=IMPLICIT_COOLING,
            cooling_curve_config=CoolingCurveConfig(
                cooling_curve_type=SIMPLE_MIXING_LAYER_COOLING,
            ),
        ),
        frame_tracking=True,
        return_snapshots=True,
        num_snapshots=100,
        snapshot_settings=SnapshotSettings(
            return_final_state=True,
            return_states=False,
            return_temperature_pdf=True,
            num_temperature_bins=100,
            temperature_pdf_min=T_cold,
            temperature_pdf_max=T_hot,
        ),
    )

    registered_variables = get_registered_variables(config)

    # ---- Initial state ---------------------------------------------------
    def single_interface(f_l, f_u, Z, z_center, smoothing_length):
        return 0.5 * (
            f_l * (1 - jnp.tanh((Z - z_center) / smoothing_length))
            + f_u * (1 + jnp.tanh((Z - z_center) / smoothing_length))
        )

    helper_data = get_helper_data(config)
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

    # --- Perturbation (identical to trml.py) -----------------------------
    dz = Z - z_center
    envelope = jnp.exp(-((jnp.abs(dz) - 2 * smoothing_length) / (3 * grid_spacing)) ** 2)
    amp = 0.03 * v_rel
    mode_numbers = jnp.array([2, 4, 6])
    kx = 2 * jnp.pi * mode_numbers / L_x
    ky = 2 * jnp.pi * mode_numbers / L_y

    key = jax.random.PRNGKey(42)
    key_ph, key_n = jax.random.split(key, 2)
    phases = jax.random.uniform(key_ph, (3, 3), minval=0.0, maxval=2 * jnp.pi)

    modes = jnp.zeros_like(density)
    for i in range(3):
        for j in range(3):
            modes = modes + jnp.sin(kx[i] * X + ky[j] * Y + phases[i, j])
    modes = modes / 9.0

    key_nx, key_ny, key_nz = jax.random.split(key_n, 3)
    noise_x = jax.random.normal(key_nx, density.shape)
    noise_y = jax.random.normal(key_ny, density.shape)
    noise_z = jax.random.normal(key_nz, density.shape)

    velocity_x = velocity_x + amp * envelope * 0.3 * noise_x
    velocity_y = velocity_y + amp * envelope * 0.3 * noise_y
    velocity_z = velocity_z + amp * envelope * (modes + 0.3 * noise_z)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        velocity_z=velocity_z,
        gas_pressure=pressure,
    )

    mixing_cooling_params = MixingCoolingParams(
        xi=xi,
        mach_number=mach_number,
        density_contrast=density_contrast,
    )

    params = SimulationParams(
        viscosity=kinematic_viscosity,
        thermal_conductivity=thermal_conductivity,
        t_end=t_end_in_t_sh * t_sh,
        C_cfl=1.5,
        gamma=gamma,
        minimum_density=rho_cold / 100,
        minimum_pressure=P0 / 100,
        fixed_boundary_state=FixedBoundaryState(
            z=FixedBoundaryState1D(
                right_state=jnp.array([rho_hot, v_rel / 2, 0.0, 0.0, P0])
            )
        ),
        cooling_params=CoolingParams(
            cooling_curve_params=mixing_cooling_params,
            floor_temperature=T_cold,
        ),
    )

    config = finalize_config(config, initial_state.shape)

    meta = dict(
        num_cells_x=num_cells_x,
        grid_spacing=grid_spacing,
        gamma=gamma,
        kinematic_viscosity=kinematic_viscosity,
        thermal_conductivity=thermal_conductivity,
        T_hot=T_hot,
        T_cold=T_cold,
        P0=P0,
        t_sh=t_sh,
        t_end=t_end_in_t_sh * t_sh,
    )
    return config, params, registered_variables, initial_state, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-cells", type=int, required=True)
    args = ap.parse_args()

    N = args.num_cells
    config, params, registered_variables, initial_state, meta = build(N)

    result = time_integration(
        initial_state, config, params, registered_variables
    )

    out = os.path.join(DATA_DIR, f"trml_N{N}.npz")
    jnp.savez(
        out,
        final_state=result.final_state,
        time_points=result.time_points,
        temperature_pdf=result.temperature_pdf,
        **meta,
    )
    print(f"[N={N}] saved -> {out}")


if __name__ == "__main__":
    main()
