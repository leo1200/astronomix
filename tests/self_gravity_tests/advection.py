"""
Advection of self-gravitating slabs in equilibrium.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from typing import NamedTuple

# numerics
import jax
import jax.numpy as jnp

# enable 64-bit precision for better error measurement
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# astronomix classes
from astronomix import SimulationConfig
from astronomix import SimulationParams

# astronomix functions
from astronomix import get_helper_data
from astronomix import time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import SECOND_ORDER_CONSERVATIVE, FINITE_VOLUME, PERIODIC_ROLL, FOURTH_ORDER_CONSERVATIVE, finalize_config
from astronomix import get_registered_variables

# astronomix constants
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY, BoundarySettings, BoundarySettings1D,
    FINITE_DIFFERENCE, SIMPLE_SOURCE, GravityConfig
)

def initial_slab(
    x, rho_0, p_0, eps, v_vec, k_vec, G
):
    
    # kx scalar product
    kx = k_vec[0] * x[0] + k_vec[1] * x[1] + k_vec[2] * x[2]
    k_squared = jnp.sum(k_vec ** 2)

    # angular frequency
    w = k_vec[0] * v_vec[0] + k_vec[1] * v_vec[1] + k_vec[2] * v_vec[2]

    # density field
    rho = rho_0 * (1 + eps * jnp.cos(kx) + eps ** 2 / 3 * jnp.cos(2 * kx))

    # velocity field
    vx = v_vec[0] * jnp.ones_like(x[0])
    vy = v_vec[1] * jnp.ones_like(x[1])
    vz = v_vec[2] * jnp.ones_like(x[2])

    # pressure field
    P = p_0 + 4 * jnp.pi * G * eps * rho_0 ** 2 / k_squared * (
        (1 - eps ** 2 / 12) * jnp.cos(kx) + 
        eps / 3 * jnp.cos(2 * kx) +
        eps ** 2 / 12 * jnp.cos(3 * kx) +
        eps ** 3 / 144 * jnp.cos(4 * kx)
    )

    return rho, (vx, vy, vz), P, w

def slab_simulation(
    num_cells,
    num_periods = 1,
    solver_mode = FINITE_DIFFERENCE,
    self_gravity_version = SIMPLE_SOURCE,
    animate = False
):
    """
    Return the L1 density error.
    """

    G = 1 / (4 * jnp.pi)
    rho_0 = 1.0
    p_0 = 6.0
    eps = 0.3
    # v_vec = jnp.array([0.8, 0.6, 0.0])
    # k_vec = jnp.array([1/3, 2/3, 2/3])

    v_vec = jnp.array([0.6, 0.6, 0.6])
    k_vec = jnp.array([2/3, 2/3, 2/3])

    box_size = float(2 * jnp.pi * jnp.max(1/k_vec))

    # setup simulation config
    config = SimulationConfig(
        solver_mode=solver_mode,
        memory_analysis=True,
        progress_bar=True,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
        ),
        boundary_handling=PERIODIC_ROLL,
        num_ghost_cells=0,
        mhd=False,
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
        return_snapshots=animate,
        num_snapshots = 48 * num_periods,
    )

    helper_data = get_helper_data(config)

    registered_variables = get_registered_variables(config)

    x = helper_data.geometric_centers
    # shape (num_cells, num_cells, num_cells, 3)
    # to shape (3, num_cells, num_cells, num_cells)
    x = jnp.transpose(x, (3, 0, 1, 2))

    initial_rho, initial_v, initial_P, w = initial_slab(
        x, rho_0=rho_0, p_0=p_0, eps=eps, v_vec=v_vec, k_vec=k_vec, G=G
    )

    period = 2 * jnp.pi / w
    # period = 30 * jnp.pi
    t_end = num_periods * period

    # setup simulation params
    params = SimulationParams(
        C_cfl= 1.5 if solver_mode == FINITE_DIFFERENCE else 0.8,
        t_end=t_end,
        gamma=5/3,
        gravitational_constant=G
    )

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=initial_rho,
        velocity_x=initial_v[0],
        velocity_y=initial_v[1],
        velocity_z=initial_v[2],
        gas_pressure=initial_P,
    )

    # finalize config
    config = finalize_config(config, initial_state.shape)

    # run the simulation
    result = time_integration(
        initial_state, config, params, registered_variables
    )

    if animate:
        # THERE IS SOME PROBLEM WITH USING THIS
        # FOR THE ERROR ANALYSIS!
        # THE LAST FRAME WHEN TIMEPOINTS FOR THE FRAMES
        # ARE NOT SPECIFIED MUST NOT NECESSARILY BE THE FINAL
        # TIMEPOINT!
        final_state = result.states[-1]
    else:
        final_state = result

    # plot the initial and final density slices
    fig_comp, axs_comp = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axs_comp[0].imshow(initial_state[registered_variables.density_index, :, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
    axs_comp[0].set_title('Initial Density Slice')
    fig_comp.colorbar(im0, ax=axs_comp[0])
    im1 = axs_comp[1].imshow(final_state[registered_variables.density_index, :, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
    axs_comp[1].set_title('Final Density Slice')
    fig_comp.colorbar(im1, ax=axs_comp[1])

    projected_abs_diff = jnp.abs(final_state[registered_variables.density_index] - initial_rho).sum(axis=2)
    im2 = axs_comp[2].imshow(projected_abs_diff, origin='lower', extent=(0, box_size, 0, box_size))
    axs_comp[2].set_title('Projected Absolute Density Difference')
    fig_comp.colorbar(im2, ax=axs_comp[2])

    fig_comp.tight_layout()
    fig_comp.savefig(f"figures/slab_comparison_{num_cells}cells_{'FD' if solver_mode == FINITE_DIFFERENCE else 'FV'}.svg")

    # if animate do a func animation of the density slice evolution
    if animate:
        fig_anim, axs_anim = plt.subplots(1, 1, figsize=(10, 5))

        im = axs_anim.imshow(initial_state[registered_variables.density_index, :, :, num_cells // 2], origin='lower', extent=(0, box_size, 0, box_size))
        axs_anim.set_title('Density Slice Evolution (Simulated)')
        fig_anim.colorbar(im, ax=axs_anim)

        def update(frame):
            im.set_data(result.states[frame][registered_variables.density_index, :, :, num_cells // 2])
            return im,

        anim = FuncAnimation(fig_anim, update, frames=len(result.states), blit=True)
        anim.save(f"figures/slab_evolution_{num_cells}cells_{'FD' if solver_mode == FINITE_DIFFERENCE else 'FV'}.gif")

    # calculate L1 error
    return jnp.mean(jnp.abs(final_state[registered_variables.density_index] - initial_rho))

resolutions = [16, 32, 64, 96,]

class TestSetup(NamedTuple):
    solver_mode: int
    self_gravity_version: int
    marker: str = 'o'
    linewidth: float = 2.0

    def __str__(self):
        # FD for finite difference, FV for finite volume
        # simple source for SIMPLE_SOURCE
        # flux-based source for SECOND_ORDER_CONSERVATIVE
        # corrected flux-based source for FOURTH_ORDER_CONSERVATIVE
        # total string e.g.: FD, simple source
        solver_str = 'FD' if self.solver_mode == FINITE_DIFFERENCE else 'FV'
        gravity_str = 'simple source' if self.self_gravity_version == SIMPLE_SOURCE else 'flux-based source' if self.self_gravity_version == SECOND_ORDER_CONSERVATIVE else 'corrected flux-based source'
        return f"{solver_str}, {gravity_str}"

test_setups = [
    TestSetup(solver_mode=FINITE_DIFFERENCE, self_gravity_version=SIMPLE_SOURCE, marker='o', linewidth=3.0),
    TestSetup(solver_mode=FINITE_DIFFERENCE, self_gravity_version=SECOND_ORDER_CONSERVATIVE, marker='s', linewidth=2.0),
    TestSetup(solver_mode=FINITE_DIFFERENCE, self_gravity_version=FOURTH_ORDER_CONSERVATIVE, marker='^', linewidth=1.0),
]

errors = {str(test_setup): [] for test_setup in test_setups}

for num_cells in resolutions:
    for test_setup in test_setups:
        error = slab_simulation(num_cells, solver_mode=test_setup.solver_mode, self_gravity_version=test_setup.self_gravity_version)
        errors[str(test_setup)].append(error)
        print(f"Resolution: {num_cells}, Test Setup: {test_setup}, L1 Density Error: {error}")

# save the errors to a file
# jnp.savez("slab_errors.npz", resolutions=resolutions, errors_fd=errors_fd, errors_fv=errors_fv)

# plot the errors
fig_errors, ax_errors = plt.subplots(figsize=(8, 5))

for test_setup in test_setups:
    ax_errors.plot(resolutions, errors[str(test_setup)], marker=test_setup.marker, linewidth=test_setup.linewidth, label=str(test_setup))

ax_errors.set_xscale('log')
ax_errors.set_yscale('log')

anchor = (20, 1e-5)

add_power_law_indicators(
    ax=ax_errors,
    anchor=anchor,
    exponents=[-2, -5],
    x_span=2.0,
    scales=[1.0, 1.0],
    x_label='N'
)

ax_errors.set_xlabel('number of cells per dimension')
ax_errors.set_ylabel('L1 density error')
ax_errors.set_title('Slab L1 Density Error vs Resolution')
ax_errors.legend()
fig_errors.tight_layout()
fig_errors.savefig(f"figures/slab_error_convergence.svg")