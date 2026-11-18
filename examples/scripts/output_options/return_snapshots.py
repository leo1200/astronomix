"""
Output options: return_snapshots for in-memory diagnostics.

With ``return_snapshots`` the time integrator returns a SnapshotData object
instead of just the final state. Everything stays on the GPU, which is ideal for
building losses over several points in time. The SnapshotSettings give
fine-grained control over what is returned — here we ask only for integrated
energies (cheap), not the full per-snapshot states (the biggest allocation).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import PERIODIC_BOUNDARY

# astronomix containers
from astronomix import (
    SnapshotSettings,
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
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
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# configure the 3D self-gravitating Evrard collapse with snapshot diagnostics
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
        poisson_manual_open_boundaries = True,
        # flux-corrected gravitational work (the default coupling is the
        # conservative FOURTH_ORDER_CONSERVATIVE one)
        work_flux_correction = True,
    ),
    solver_config=FDSolverConfig(
        # positivity-preserving WENO keeps the low-resolution collapse from
        # driving density / pressure negative
        weno_positivity_preserving = True,
    ),
    boundary_settings = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    return_snapshots = True,
    num_snapshots = 100,
    snapshot_settings = SnapshotSettings(
        # return_states = True, # if one wants the full states
        # cheap integrated diagnostics — no full per-snapshot states
        return_total_energy = True,
        return_kinetic_energy = True,
        return_internal_energy = True,
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

# Evrard initial condition: a cold gas sphere of mass M and radius R with the
# rho ~ 1/r density profile, at rest in a tenuous background.
R = 1.0
M = 1.0
rho = jnp.where(
    helper_data.r <= R, M / (2 * jnp.pi * R**2 * helper_data.r), 1e-4
)
u_x = jnp.zeros_like(rho)
u_y = jnp.zeros_like(rho)
u_z = jnp.zeros_like(rho)

# small thermal energy per unit mass so the sphere collapses (cold Evrard test)
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

# run the simulation — returns a SnapshotData object (kept on the GPU)
snapshots = time_integration(initial_state, config, params, registered_variables)

# plot the energy budget over time
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(snapshots.time_points, snapshots.total_energy, label="total", color="black")
ax.plot(snapshots.time_points, snapshots.internal_energy, label="internal", color="green")
ax.plot(snapshots.time_points, snapshots.kinetic_energy, label="kinetic", color="red")
ax.plot(snapshots.time_points, snapshots.gravitational_energy, label="gravitational", color="blue")
ax.set_xlabel("time")
ax.set_ylabel("energy")
ax.legend()
fig.savefig(figures_dir / "evrard_energy_budget.png", dpi=200, bbox_inches="tight")
