# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax


# activate nan debugging
jax.config.update("jax_debug_nans", True)

# enable 64 bit precision
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import SimulationConfig
from astronomix import get_helper_data
from astronomix import SimulationParams
from astronomix import time_integration
from astronomix._finite_difference._fluid_equations._eigen import _eigen_x
from astronomix._finite_difference._fluid_equations._equations import _b_squared3D, conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, FINITE_DIFFERENCE, HLLC_LM, LAX_FRIEDRICHS, VAN_ALBADA_PP, finalize_config
import numpy as np
from matplotlib.colors import LogNorm

from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_x

from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields

from astronomix._finite_volume._magnetic_update._magnetic_field_update import magnetic_update

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS, FORWARDS, HLL, HLLC, MINMOD,
    OSHER, PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D
)

def setup_blast_simulation(num_cells, B0, theta, phi):

    # spatial domain
    box_size = 1.0

    # setup simulation config
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
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
        gamma = 1.4
    )

    registered_variables = get_registered_variables(config)

    r = helper_data.r

    rho = jnp.ones_like(r)
    P = jnp.ones_like(r) * 0.1
    r_inj = 0.1 * box_size
    p_inj = 1000
    P = jnp.where(r**2 < r_inj**2, p_inj, P)

    print("Pressure min: {}, max: {}".format(jnp.min(P), jnp.max(P)))

    V_x = jnp.zeros_like(r)
    V_y = jnp.zeros_like(r)
    V_z = jnp.zeros_like(r)

    B_x = B0 * jnp.sin(theta) * jnp.cos(phi)
    B_y = B0 * jnp.sin(theta) * jnp.sin(phi)
    B_z = B0 * jnp.cos(theta)

    B_x = jnp.ones_like(r) * B_x
    B_y = jnp.ones_like(r) * B_y
    B_z = jnp.ones_like(r) * B_z

    bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

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
        interface_magnetic_field_x = bxb,
        interface_magnetic_field_y = byb,
        interface_magnetic_field_z = bzb,
        gas_pressure = P
    )

    config = finalize_config(config, initial_state.shape)

    return initial_state, config, registered_variables, params, helper_data

num_cells = 400
B0 = 100 / jnp.sqrt(4 * jnp.pi)
theta = jnp.pi / 2
phi = jnp.pi / 4

initial_state, config, registered_variables, params, helper_data = setup_blast_simulation(num_cells, B0, theta, phi)

b_squared = _b_squared3D(
    initial_state,
    registered_variables,
)

conserved_state = conserved_state_from_primitive_mhd(
    primitive_state = initial_state[:-3],
    gamma = params.gamma,
    registered_variables = registered_variables,
)

compiled_step = _weno_flux_x.lower(
    conserved_state,
    params.gamma,
    registered_variables,
).compile()

compiled_stats = compiled_step.memory_analysis()
if compiled_stats is not None:
    # Calculate total memory usage including temporary storage,
    # arguments, and outputs (but excluding aliases)
    total = (
          compiled_stats.temp_size_in_bytes
        + compiled_stats.argument_size_in_bytes
        + compiled_stats.output_size_in_bytes
        - compiled_stats.alias_size_in_bytes
    )
    print("=== Compiled memory usage PER DEVICE ===")
    print(
        f"Temp size: {compiled_stats.temp_size_in_bytes / (1024**2):.2f} MB"
    )
    print(
        f"Argument size: {compiled_stats.argument_size_in_bytes / (1024**2):.2f} MB"
    )
    print(f"Total size: {total / (1024**2):.2f} MB")
    print("========================================")