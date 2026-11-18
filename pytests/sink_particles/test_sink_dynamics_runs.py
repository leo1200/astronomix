"""
Sink particle dynamics in whole runs pytest (float32).

Runs the time loop with moving sink particles and checks the physics: two
sinks on a circular orbit (Gong & Ostriker 2013, Section 3.1), a light sink
drifting through uniform gas across a periodic boundary with the sink time-step
limit setting the steps, and gas falling toward a sink on every hydro path
that computes gravity. The momentum the gas and a sink exchange is checked in
``test_sink_dynamics_operators.py``, in float64.
The runs use the finite-difference solver unless a test says otherwise.

Run on the CPU with::

    JAX_PLATFORMS=cpu python -m pytest pytests/sink_particles/test_sink_dynamics_runs.py
"""

# ==== GPU selection ====
import os
if os.environ.get("JAX_PLATFORMS") == "cpu":
    # XLA:CPU compiles the unrolled step in seconds at optimisation level 0
    # (minutes and tens of GB otherwise); the numerics differ at round-off only.
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_backend_optimization_level=0 --xla_llvm_disable_expensive_passes=true",
    )
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None:
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    LSRK4,
    MUSCL_HANCOCK,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    RIEMANN_FLUX,
    RK2_SSP,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    FVSolverConfig,
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
    SinkParticles,
)
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix._modules._sink_particles._sink_particle_formation import (
    _empty_sink_particles,
)
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    SinkFormationSettings,
    setup_sink_formation,
)


NUM_CELLS = 32
CELL_SIZE = 1.0 / NUM_CELLS
MAX_NUM_SINKS = 4

PERIODIC_AXIS = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
OPEN_AXIS = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
PERIODIC = BoundarySettings(x=PERIODIC_AXIS, y=PERIODIC_AXIS, z=PERIODIC_AXIS)
OPEN = BoundarySettings(x=OPEN_AXIS, y=OPEN_AXIS, z=OPEN_AXIS)


# -------------------------------------------------------------
# ======================= ↓ Helpers ↓ =========================
# -------------------------------------------------------------


def _setup(settings, masses, positions, velocities=None, **config_options):
    """A 32³ unit box with sink particles on and the given sinks passed in.

    Args:
        settings: The problem constants of the gas (``SinkFormationSettings``).
        masses: The masses of the sinks, shape (N,).
        positions: Their positions, shape (N, 3).
        velocities: Their velocities, shape (N, 3); zero by default.
        **config_options: Further ``SimulationConfig`` fields.

    Returns:
        ``(state, config, params, registered_variables)``.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
        **config_options,
    )
    state, config, params = setup_sink_formation(config, SimulationParams(), settings)

    num_sinks = len(masses)
    if velocities is None:
        velocities = np.zeros((num_sinks, 3))
    sink_particles = _empty_sink_particles(config, state.primitive_state.dtype)
    sink_particles = SinkParticles(
        mass=sink_particles.mass.at[:num_sinks].set(jnp.asarray(masses)),
        position=sink_particles.position.at[:num_sinks].set(jnp.asarray(positions)),
        velocity=sink_particles.velocity.at[:num_sinks].set(jnp.asarray(velocities)),
    )
    state = state._replace(sink_particles=sink_particles)
    return state, config, params, get_registered_variables(config)


def _in_cells(*cell_coordinates):
    """A position given in units of the cell size."""
    return np.array(cell_coordinates) * CELL_SIZE


def _gas_velocity_and_momentum(primitive_state, registered_variables):
    """The gas velocity field and the total gas momentum of an unpadded state.

    Args:
        primitive_state: The unpadded primitive state.
        registered_variables: The registered variables.

    Returns:
        ``(velocity, momentum)``: shape (3, nx, ny, nz) and (3,).
    """
    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [primitive_state[index] for index in registered_variables.velocity_index]
    )
    momentum = jnp.sum(density * velocity, axis=(1, 2, 3)) * CELL_SIZE**3
    return velocity, momentum


def _wrapped(difference, box_length=1.0):
    """A difference of positions taken to its nearest periodic copy."""
    return difference - box_length * np.round(difference / box_length)


# -------------------------------------------------------------
# ======================= ↑ Helpers ↑ =========================
# -------------------------------------------------------------


def test_circular_orbit():
    """Two unit sinks 16 cells apart, started at their Newtonian circular
    speeds in an open box of thin hot gas, stay on the circular orbit for one
    period: constant separation, fixed centre of mass, and the Newtonian phase
    at every snapshot (Gong & Ostriker 2013, Section 3.1)."""
    separation = 16 * CELL_SIZE
    speed = np.sqrt(1.0 / (2.0 * separation))
    period = 2.0 * np.pi * (0.5 * separation) / speed
    # The centre of mass sits off the cell corners and faces, so that no
    # symmetry of the grid hides an error.
    center = 0.5 + _in_cells(0.3, 0.2, 0.1)
    offset = np.array([0.5 * separation, 0.0, 0.0])
    # Thin hot gas: its mass does not change the orbit, it does not collapse
    # onto the sinks, and it stays far below the density threshold.
    state, config, params, registered_variables = _setup(
        SinkFormationSettings(
            background_density=1e-6,
            peak_overdensity=0.0,
            sound_speed_squared=25.0,
            t_end=period,
            boundary_settings=OPEN,
        ),
        masses=[1.0, 1.0],
        positions=[center - offset, center + offset],
        velocities=[[0.0, -speed, 0.0], [0.0, speed, 0.0]],
        return_snapshots=True,
        num_snapshots=8,
    )

    snapshots = time_integration(state, config, params, registered_variables)

    positions = np.asarray(snapshots.sink_particles.position[:, :2], dtype=np.float64)
    times = np.asarray(snapshots.time_points, dtype=np.float64)
    separation_vector = positions[:, 1] - positions[:, 0]

    distance = np.linalg.norm(separation_vector, axis=-1)
    separation_error = np.max(np.abs(distance / separation - 1.0))
    assert separation_error <= 0.01, (
        f"the separation deviates from the initial one by up to {separation_error:.2%}, "
        "more than 1 %"
    )

    center_of_mass = 0.5 * (positions[:, 0] + positions[:, 1])
    center_of_mass_drift = np.linalg.norm(center_of_mass - center, axis=-1)
    assert np.max(center_of_mass_drift) <= 0.01 * CELL_SIZE, (
        f"the centre of mass drifts by up to {np.max(center_of_mass_drift) / CELL_SIZE:.3e} "
        "cells, more than 0.01"
    )

    # The separation starts along +x and turns counterclockwise in the x-y
    # plane, by 360° per period.
    angle = np.degrees(np.arctan2(separation_vector[:, 1], separation_vector[:, 0]))
    phase_error = (angle - 360.0 * times / period + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(phase_error)) <= 2.0, (
        f"the separation angle is up to {np.max(np.abs(phase_error)):.2f}° off the Newtonian "
        f"phase, more than 2°; per snapshot: {phase_error}"
    )


def test_drift_through_uniform_gas():
    """A light sink moves in a straight line through uniform gas, across the
    periodic boundary, and the sink time-step limit (half a cell per step),
    not the CFL condition, sets the number of steps."""
    initial_position = np.array([0.9, 0.4, 0.3])
    velocity = np.array([10.0, 3.0, -2.0])
    state, config, params, registered_variables = _setup(
        SinkFormationSettings(
            background_density=1.0,
            peak_overdensity=0.0,
            sound_speed_squared=1.0,
            t_end=0.03,
            boundary_settings=PERIODIC,
        ),
        masses=[1e-6],
        positions=[initial_position],
        velocities=[velocity],
        return_snapshots=True,
        num_snapshots=2,
    )

    snapshots = time_integration(state, config, params, registered_variables)

    final_time = float(snapshots.time_points[-1])
    final_position = np.asarray(snapshots.sink_particles.position[-1, 0], dtype=np.float64)
    final_velocity = np.asarray(snapshots.sink_particles.velocity[-1, 0], dtype=np.float64)

    # Along x the sink crosses the boundary: x0 + v t = 1.2, stored as 0.2.
    expected_position = np.mod(initial_position + velocity * final_time, 1.0)
    position_error = np.max(np.abs(_wrapped(final_position - expected_position)))
    assert position_error <= 1e-3 * CELL_SIZE, (
        f"the final position is {position_error / CELL_SIZE:.3e} cells from "
        "(x0 + v t) mod L, more than 1e-3"
    )
    assert 0.0 <= final_position[0] < 1.0, (
        f"the final x = {final_position[0]} was not wrapped into the box"
    )

    velocity_change = np.linalg.norm(final_velocity - velocity)
    assert velocity_change <= 1e-4 * np.linalg.norm(velocity), (
        f"the sink velocity changed by {velocity_change / np.linalg.norm(velocity):.3e} of "
        "its magnitude, more than 1e-4"
    )

    minimum_num_steps = np.linalg.norm(velocity) * final_time / (0.5 * CELL_SIZE)
    assert int(snapshots.num_iterations) >= minimum_num_steps, (
        f"the run took {int(snapshots.num_iterations)} steps, fewer than the "
        f"{minimum_num_steps:.1f} that half a cell per step needs: the sink time-step limit "
        "did not set the steps"
    )


@pytest.mark.parametrize(
    "solver_options",
    [
        dict(),
        dict(solver_config=FDSolverConfig(time_integrator=LSRK4)),
        dict(solver_config=FDSolverConfig(scheme=RIEMANN_FLUX)),
        dict(mhd=True),
        dict(mhd=True, solver_config=FDSolverConfig(time_integrator=LSRK4)),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=MUSCL_HANCOCK)),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=RK2_SSP)),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=VL2)),
    ],
    ids=[
        "fd_ssprk54",
        "fd_lsrk4",
        "fd_riemann_flux",
        "fd_ssprk54_mhd",
        "fd_lsrk4_mhd",
        "fv_muscl_hancock",
        "fv_rk2_ssp",
        "vl2",
    ],
)
def test_gas_falls_toward_sink(solver_options):
    """Uniform gas at rest around a sink at rest starts falling toward it with
    the sink's point-mass acceleration, on every hydro path that computes
    gravity, while the sink stays where it is. This checks that the sink
    potential reaches the gravity source of every solver.

    Args:
        solver_options: The ``SimulationConfig`` fields that select the solver.
    """
    sink_mass = 1.0
    t_end = 2e-3
    # The sink sits off the centre of its cell.
    sink_position = 0.5 + _in_cells(0.3, 0.2, -0.15)
    # The gas is warm, so that its thermal energy takes up the kinetic energy
    # the gas next to the sink gains in one step. In the 10 steps sound
    # crosses a third of a cell, so 6 to 10 cells away the gas feels only the
    # sink.
    state, config, params, registered_variables = _setup(
        SinkFormationSettings(
            background_density=1e-3,
            peak_overdensity=0.0,
            sound_speed_squared=25.0,
            magnetic_field_z=1e-3,
            t_end=t_end,
            boundary_settings=PERIODIC,
        ),
        masses=[sink_mass],
        positions=[sink_position],
        gravity_config=GravityConfig(
            self_gravity=True,
            poisson_manual_open_boundaries=True,
        ),
        fixed_timestep=True,
        num_timesteps=10,
        **solver_options,
    )

    final_state = time_integration(state, config, params, registered_variables)

    # Over 10 steps the gas moves a small fraction of a cell, so in the cells
    # 6 to 10 cells from the sink its velocity is the sink's acceleration
    # times the elapsed time.
    velocity, _ = _gas_velocity_and_momentum(final_state.primitive_state, registered_variables)
    separation = np.moveaxis(np.asarray(get_helper_data(config).geometric_centers), -1, 0)
    separation = separation - sink_position[:, None, None, None]
    distance = np.linalg.norm(separation, axis=0)
    expected_velocity = (
        -params.gravitational_constant * sink_mass * t_end * separation / distance**3
    )
    in_shell = (distance >= 6 * CELL_SIZE) & (distance <= 10 * CELL_SIZE)
    relative_error = np.linalg.norm(
        np.asarray(velocity, dtype=np.float64) - expected_velocity,
        axis=0,
    ) / np.linalg.norm(expected_velocity, axis=0)
    assert np.max(relative_error[in_shell]) <= 0.05, (
        f"the gas velocity 6 to 10 cells from the sink is up to "
        f"{np.max(relative_error[in_shell]):.1%} off −G M t r̂ / r², more than 5 %"
    )

    sink_displacement = np.linalg.norm(
        np.asarray(final_state.sink_particles.position[0], dtype=np.float64) - sink_position
    )
    assert sink_displacement <= 1e-3 * CELL_SIZE, (
        f"the sink moved by {sink_displacement / CELL_SIZE:.3e} cells, more than 1e-3"
    )
