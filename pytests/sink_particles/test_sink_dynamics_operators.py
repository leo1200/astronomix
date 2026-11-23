"""
Sink particle dynamics operators pytest (float64).

Checks the operators of ``_sink_particle_dynamics.py`` one by one: the sink
density of the triangular-shaped-cloud (TSC) mass assignment (total mass, ghost
cells, empty slots, and weights whose derivative is continuous across a cell
face), the zero self-force and the equal and opposite forces of the
particle-mesh gravity, the force of an external potential read back with the
6th-order (finite difference, MUSCL_HANCOCK, RK2_SSP) and the 2nd-order (VL2)
stencil, the balance between the force of the gas on the sinks and the force
on the gas that each coupling applies, and the kick and the drift. Two sets of
three runs of the time loop with different step sizes check that the
kick-drift-kick integration of two sinks on a circular orbit, and the
imbalance of the momentum the gas and a sink exchange, converge at second
order.

Run on the CPU with::

    JAX_PLATFORMS=cpu python -m pytest pytests/sink_particles/test_sink_dynamics_operators.py
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
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    MUSCL_HANCOCK,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    RK2_SSP,
    VL2,
)
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    GHOST_CELLS,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
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
from astronomix._finite_volume._gravity_coupling import (
    _add_athena_gravity_source,
    _vl2_gravitational_potential,
)
from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix._modules._explicit_sources import explicit_sources
from astronomix._modules._sink_particles._sink_particle_dynamics import (
    _drift,
    _kick,
    _sink_acceleration,
    _sink_density,
    _sink_potential,
    _tsc_weights,
)
from astronomix._modules._sink_particles._sink_particle_formation import (
    _empty_sink_particles,
)
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    SinkFormationSettings,
    setup_sink_formation,
)
from astronomix.time_stepping._utils import _unpad
from astronomix.time_stepping.time_integration import _prepare_padded_state


NUM_CELLS = 16
CELL_SIZE = 1.0 / NUM_CELLS
MAX_NUM_SINKS = 4

# The momentum-exchange runs use a finer grid, so that the clump and the sink
# 7 cells from it fit in the periodic box.
EXCHANGE_NUM_CELLS = 32
EXCHANGE_CELL_SIZE = 1.0 / EXCHANGE_NUM_CELLS
EXCHANGE_SINK_MASS = 0.5
EXCHANGE_CLUMP_CENTER = np.full((3,), 0.5 + 0.5 * EXCHANGE_CELL_SIZE)
# 7 cells from the clump centre, along a direction off the grid axes.
EXCHANGE_SINK_POSITION = EXCHANGE_CLUMP_CENTER + np.array([6.0, 3.0, 2.0]) * EXCHANGE_CELL_SIZE

PERIODIC_AXIS = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
OPEN_AXIS = BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
PERIODIC = BoundarySettings(x=PERIODIC_AXIS, y=PERIODIC_AXIS, z=PERIODIC_AXIS)
OPEN = BoundarySettings(x=OPEN_AXIS, y=OPEN_AXIS, z=OPEN_AXIS)


@pytest.fixture(autouse=True, scope="module")
def float64():
    """Run every test of this module in float64, without switching float64 on
    for the float32 tests collected in the same session."""
    with jax.enable_x64(True):
        yield


# -------------------------------------------------------------
# ======================= ↓ Helpers ↓ =========================
# -------------------------------------------------------------


def _setup(
    boundary_settings,
    gravity_config=GravityConfig(self_gravity=True),
    clump_options=None,
    **config_options,
):
    """A 16³ unit box of gas at rest, uniform unless a clump is asked for,
    with sink particles on.

    Args:
        boundary_settings: The boundaries of the box.
        gravity_config: The gravity configuration; self-gravity is always on.
        clump_options: ``SinkFormationSettings`` fields of a Gaussian clump
            (``peak_overdensity``, ``overdensity_width``, ...); no clump by
            default.
        **config_options: Further ``SimulationConfig`` fields, e.g. the solver.

    Returns:
        ``(primitive_state, config, params, registered_variables)``, with the
        primitive state padded and its ghost cells filled, as in the time loop.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        gravity_config=gravity_config,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
        **config_options,
    )
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SinkFormationSettings(
            peak_overdensity=0.0,
            boundary_settings=boundary_settings,
        )._replace(**(clump_options or {})),
    )
    registered_variables = get_registered_variables(config)
    primitive_state = _prepare_padded_state(
        state.primitive_state,
        config,
        params,
        registered_variables,
    )
    return primitive_state, config, params, registered_variables


def _sinks(masses, positions, config, velocities=None):
    """Sink particles with the given sinks in the first slots and the other
    slots empty.

    Args:
        masses: The masses of the sinks, shape (N,).
        positions: Their positions, shape (N, 3).
        config: The simulation configuration.
        velocities: Their velocities, shape (N, 3); zero by default.

    Returns:
        Sink particles with ``MAX_NUM_SINKS`` slots.
    """
    num_sinks = len(masses)
    if velocities is None:
        velocities = np.zeros((num_sinks, 3))
    sink_particles = _empty_sink_particles(config, jnp.float64)
    return SinkParticles(
        mass=sink_particles.mass.at[:num_sinks].set(jnp.asarray(masses)),
        position=sink_particles.position.at[:num_sinks].set(jnp.asarray(positions)),
        velocity=sink_particles.velocity.at[:num_sinks].set(jnp.asarray(velocities)),
    )


def _in_cells(*cell_coordinates):
    """A position given in units of the cell size."""
    return np.array(cell_coordinates) * CELL_SIZE


def _weight_derivatives_by_cell(position, grid_shape, config):
    """The derivative of every TSC weight of one sink with respect to its
    position, keyed by the cell the weight belongs to.

    The weights are stored by their offset from the cell that contains the
    sink, and that cell changes when the sink crosses a face, so derivatives
    on the two sides of a face can only be compared cell by cell.

    Args:
        position: The position of the sink, shape (3,).
        grid_shape: The shape of a padded field.
        config: The simulation configuration.

    Returns:
        A dict from the padded cell index (i, j, k) to the derivative of that
        cell's weight with respect to the position, shape (3,).
    """

    def weights(sink_position):
        _, cell_weights = _tsc_weights(sink_position[None, :], grid_shape, config)
        return cell_weights[0]

    cell_indices, _ = _tsc_weights(position[None, :], grid_shape, config)
    cells = zip(*(np.asarray(index[0]) for index in cell_indices))
    derivatives = np.asarray(jax.jacfwd(weights)(position))
    return {tuple(int(i) for i in cell): derivative for cell, derivative in zip(cells, derivatives)}


def _orbit_setup(num_timesteps):
    """Two unit sinks 8 cells apart on their Newtonian circular orbit in a 16³
    open box of thin hot gas, with fixed steps over a quarter orbit.

    Args:
        num_timesteps: The number of fixed steps.

    Returns:
        ``(state, config, params, registered_variables)``.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        fixed_timestep=True,
        num_timesteps=num_timesteps,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
    )
    # Thin hot gas: its mass does not change the orbit, and its pressure keeps
    # it from collapsing onto the sinks.
    separation = 8 * CELL_SIZE
    speed = np.sqrt(1.0 / (2.0 * separation))
    period = 2.0 * np.pi * (0.5 * separation) / speed
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SinkFormationSettings(
            background_density=1e-9,
            peak_overdensity=0.0,
            sound_speed_squared=11.0,
            t_end=0.25 * period,
            boundary_settings=OPEN,
        ),
    )
    # The centre of mass sits off the cell corners and faces, so that no
    # symmetry of the grid hides an error.
    center = 0.5 + _in_cells(0.3, 0.2, 0.1)
    offset = np.array([0.5 * separation, 0.0, 0.0])
    sink_particles = _sinks(
        [1.0, 1.0],
        [center - offset, center + offset],
        config,
        velocities=[[0.0, -speed, 0.0], [0.0, speed, 0.0]],
    )
    state = state._replace(sink_particles=sink_particles)
    return state, config, params, get_registered_variables(config)


def _momentum_exchange_run(num_timesteps):
    """A sink at rest next to a gas clump at rest, run with fixed steps up to
    t = 0.05, about the time the sink takes to move half a cell. Periodic 32³
    box, Gaussian clump of width 3.2 cells and peak 50 above a background of
    1, c_s² = 4, which stays far below the density threshold of about 515;
    default finite-difference self-gravity coupling.

    Args:
        num_timesteps: The number of fixed steps.

    Returns:
        ``(gas_momentum_change, sink_momentum_change, final_sink_particles)``,
        the first two of shape (3,).
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(EXCHANGE_NUM_CELLS, EXCHANGE_NUM_CELLS, EXCHANGE_NUM_CELLS),
        progress_bar=False,
        fixed_timestep=True,
        num_timesteps=num_timesteps,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
    )
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SinkFormationSettings(
            background_density=1.0,
            peak_overdensity=50.0,
            overdensity_width=3.2 * EXCHANGE_CELL_SIZE,
            overdensity_center=tuple(EXCHANGE_CLUMP_CENTER),
            sound_speed_squared=4.0,
            t_end=0.05,
            boundary_settings=PERIODIC,
        ),
    )
    state = state._replace(
        sink_particles=_sinks([EXCHANGE_SINK_MASS], [EXCHANGE_SINK_POSITION], config)
    )
    registered_variables = get_registered_variables(config)

    final_state = time_integration(state, config, params, registered_variables)

    # The gas and the sink start at rest, so their final momenta are the
    # momenta they exchanged.
    primitive_state = final_state.primitive_state
    density = primitive_state[registered_variables.density_index]
    gas_momentum_change = np.array(
        [
            float(jnp.sum(density * primitive_state[velocity_index]))
            for velocity_index in registered_variables.velocity_index
        ]
    ) * EXCHANGE_CELL_SIZE**3
    final_sink_particles = final_state.sink_particles
    sink_momentum_change = np.asarray(
        final_sink_particles.mass[0] * final_sink_particles.velocity[0]
    )
    return gas_momentum_change, sink_momentum_change, final_sink_particles


# -------------------------------------------------------------
# ======================= ↑ Helpers ↑ =========================
# -------------------------------------------------------------


@pytest.mark.parametrize(
    "boundary_settings",
    [PERIODIC, OPEN],
    ids=["periodic", "open"],
)
def test_sink_density(boundary_settings):
    """The sink density holds the sink masses, leaves the ghost cells empty and
    ignores empty slots, for sinks at a cell centre, on a cell face and, in the
    periodic box, across the boundary; the derivative of every TSC weight is
    continuous across a cell face.

    Args:
        boundary_settings: The boundaries of the box.
    """
    primitive_state, config, _, _ = _setup(boundary_settings)
    grid_shape = primitive_state.shape[1:]
    is_periodic = boundary_settings == PERIODIC

    # In the open box every sink stays at least 2 cells from the edge, so that
    # all of its mass lands inside the box.
    if is_periodic:
        third_position = _in_cells(0.2, 15.9, 3.6)
    else:
        third_position = _in_cells(2.2, 13.6, 3.4)
    masses = [0.7, 1.3, 0.4]
    positions = [_in_cells(7.5, 8.5, 6.5), _in_cells(9.0, 5.3, 10.7), third_position]
    sink_particles = _sinks(masses, positions, config)
    sink_density = _sink_density(sink_particles, grid_shape, config)

    total_mass = float(jnp.sum(sink_density)) * CELL_SIZE**3
    assert abs(total_mass - sum(masses)) <= 1e-12 * sum(masses), (
        f"the sink density holds a mass of {total_mass!r}, but the sinks have {sum(masses)!r}"
    )

    num_ghost_cells = config.num_ghost_cells
    if num_ghost_cells > 0:
        interior = np.zeros(grid_shape, dtype=bool)
        interior[(slice(num_ghost_cells, -num_ghost_cells),) * 3] = True
        ghost_cell_density = np.asarray(sink_density)[~interior]
        assert np.all(ghost_cell_density == 0.0), (
            f"the sink density is non-zero in {np.count_nonzero(ghost_cell_density)} ghost cells"
        )

    # An empty slot placed anywhere, and moving, adds nothing.
    moved_empty_slot = sink_particles._replace(
        position=sink_particles.position.at[-1].set(jnp.asarray(_in_cells(5.5, 5.5, 5.5))),
        velocity=sink_particles.velocity.at[-1].set(1.0),
    )
    assert jnp.array_equal(
        _sink_density(moved_empty_slot, grid_shape, config),
        sink_density,
    ), "moving an empty slot changed the sink density"

    # Across the face between two cells, along each axis, the derivative of
    # each cell's weight changes only by the second derivative of the weights
    # (at most 2 / Δx²) times the distance between the two points.
    for axis in range(3):
        face_position = _in_cells(7.3, 8.6, 6.4)
        face_position[axis] = 8.0 * CELL_SIZE
        derivatives = {}
        for side in (-1.0, 1.0):
            position = face_position.copy()
            position[axis] += side * 1e-4 * CELL_SIZE
            derivatives[side] = _weight_derivatives_by_cell(
                jnp.asarray(position),
                grid_shape,
                config,
            )
        for cell in derivatives[-1.0].keys() | derivatives[1.0].keys():
            left = derivatives[-1.0].get(cell, np.zeros(3))
            right = derivatives[1.0].get(cell, np.zeros(3))
            jump = np.max(np.abs(left - right))
            assert jump <= 1e-3 / CELL_SIZE, (
                f"the derivative of the TSC weight of cell {cell} jumps by {jump:.3e} across "
                f"the cell face along axis {axis}, more than 1e-3 / Δx = {1e-3 / CELL_SIZE:.3e}"
            )


@pytest.mark.parametrize(
    "boundary_settings",
    [PERIODIC, OPEN],
    ids=["periodic", "open"],
)
def test_force_balance(boundary_settings):
    """Without gas, a lone sink feels no force from itself, and the forces
    between three sinks of different masses add up to zero.

    Args:
        boundary_settings: The boundaries of the box.
    """
    primitive_state, config, params, registered_variables = _setup(boundary_settings)
    primitive_state = primitive_state.at[registered_variables.density_index].set(0.0)
    gravitational_constant = params.gravitational_constant

    lone_sink = _sinks([1.0], [_in_cells(6.3, 9.6, 7.2)], config)
    acceleration = _sink_acceleration(
        primitive_state,
        lone_sink,
        config,
        params,
        registered_variables,
    )
    force_scale = gravitational_constant * 1.0 / (10.0 * CELL_SIZE) ** 2
    self_acceleration = float(jnp.linalg.norm(acceleration[0]))
    assert self_acceleration <= 1e-10 * force_scale, (
        f"a lone sink feels an acceleration of {self_acceleration:.3e} from itself, more than "
        f"1e-10 of G M / (10 Δx)² = {float(force_scale):.3e}"
    )

    masses = jnp.array([1.0, 0.4, 2.3])
    three_sinks = _sinks(
        masses,
        [_in_cells(5.1, 8.3, 7.5), _in_cells(9.4, 5.8, 9.7), _in_cells(7.9, 10.7, 6.2)],
        config,
    )
    acceleration = _sink_acceleration(
        primitive_state,
        three_sinks,
        config,
        params,
        registered_variables,
    )
    forces = masses[:, None] * acceleration[:3]
    largest_force = float(jnp.max(jnp.linalg.norm(forces, axis=-1)))
    net_force = float(jnp.linalg.norm(jnp.sum(forces, axis=0)))
    assert net_force <= 1e-10 * largest_force, (
        f"the forces between the three sinks add up to {net_force:.3e}, more than 1e-10 of "
        f"the largest force {largest_force:.3e}"
    )


@pytest.mark.parametrize(
    "solver_options",
    [
        dict(),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=MUSCL_HANCOCK)),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=VL2)),
    ],
    ids=["finite_difference", "fv_muscl_hancock", "vl2"],
)
def test_external_potential_on_sink(solver_options):
    """In uniform gas, a sink feels exactly the force of a quadratic external
    potential, with the 6th-order read-back of the schemes that take gravity
    from the explicit sources (finite difference, MUSCL_HANCOCK, RK2_SSP) and
    the 2nd-order one of VL2: a centred difference of either order is exact for
    a quadratic, and the TSC reading of a linear field is exact.

    Args:
        solver_options: The ``SimulationConfig`` fields that select the solver.
    """
    primitive_state, config, params, registered_variables = _setup(
        PERIODIC,
        gravity_config=GravityConfig(self_gravity=True, external_potential=True),
        **solver_options,
    )
    center = jnp.full((3,), 0.5)
    omega_squared = 3.0
    cell_centers = get_helper_data(config).geometric_centers
    params = params._replace(
        gravitational_potential=0.5
        * omega_squared
        * jnp.sum((cell_centers - center) ** 2, axis=-1)
    )

    # At least 5 cells from the edges: the 6th-order stencil plus the TSC
    # weights reach 4 cells, and the quadratic is not periodic.
    position = jnp.array([0.37, 0.58, 0.63])
    sink_particles = _sinks([0.1], [position], config)
    acceleration = _sink_acceleration(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )[0]

    expected_acceleration = -omega_squared * (position - center)
    error = float(jnp.linalg.norm(acceleration - expected_acceleration))
    assert error <= 1e-10 * float(jnp.linalg.norm(expected_acceleration)), (
        f"the sink acceleration {np.asarray(acceleration)} differs from −ω² (x − c) = "
        f"{np.asarray(expected_acceleration)} by {error:.3e}, more than 1e-10 relative"
    )


@pytest.mark.parametrize(
    "solver_options",
    [
        dict(gravity_config=GravityConfig(self_gravity=True, self_gravity_version=SIMPLE_SOURCE)),
        dict(
            gravity_config=GravityConfig(
                self_gravity=True,
                self_gravity_version=SECOND_ORDER_CONSERVATIVE,
            )
        ),
        dict(
            gravity_config=GravityConfig(
                self_gravity=True,
                self_gravity_version=FOURTH_ORDER_CONSERVATIVE,
            )
        ),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=RK2_SSP)),
        dict(solver_mode=FINITE_VOLUME, solver_config=FVSolverConfig(time_integrator=VL2)),
    ],
    ids=[
        "fd_simple_source",
        "fd_second_order_conservative",
        "fd_fourth_order_conservative",
        "fv_rk2_ssp",
        "vl2",
    ],
)
def test_gas_sink_force_balance(solver_options):
    """At one instant, the force of the gas on two sinks and the force that
    the coupling applies to the gas add up to zero: on a periodic box the
    gas-gas forces cancel among themselves, and the sinks read the field back
    with the momentum stencil of the coupling.

    Args:
        solver_options: The ``SimulationConfig`` fields that select the solver
            and the coupling.
    """
    primitive_state, config, params, registered_variables = _setup(
        PERIODIC,
        clump_options=dict(
            peak_overdensity=20.0,
            overdensity_width=0.1,
            overdensity_center=(0.47, 0.52, 0.5),
        ),
        **solver_options,
    )
    sink_particles = _sinks(
        [0.5, 0.3],
        [[0.68, 0.41, 0.55], [0.33, 0.70, 0.37]],
        config,
    )
    density = primitive_state[registered_variables.density_index]
    sink_potential = _sink_potential(sink_particles, density.shape, config, params)

    # The momentum the coupling gives the gas over a unit time step, without
    # mass fluxes, is the gravitational force density on the gas.
    uses_vl2 = config.solver_mode == FINITE_VOLUME and config.solver_config.time_integrator == VL2
    if uses_vl2:
        potential = _vl2_gravitational_potential(
            density,
            config,
            params,
            registered_variables,
            sink_potential=sink_potential,
        )
        state_change = _add_athena_gravity_source(
            jnp.zeros_like(primitive_state),
            density,
            [jnp.zeros_like(density)] * 3,
            potential,
            potential,
            1.0,
            config,
            registered_variables,
        )
    else:
        no_flux = jnp.zeros_like(density)
        state_change = explicit_sources(
            conserved_state_from_primitive(
                primitive_state,
                params.gamma,
                config,
                registered_variables,
            ),
            1.0,
            params.gamma,
            config,
            params,
            get_helper_data(config),
            registered_variables,
            density_fluxes=(no_flux, no_flux, no_flux),
            density_increment=no_flux,
            primitive_state=primitive_state,
            sink_potential=sink_potential,
        )
    # Only the physical cells: the ghost cells of the finite-volume schemes
    # repeat them.
    if config.boundary_handling == GHOST_CELLS:
        state_change = _unpad(state_change, config)
    gas_force_density = jnp.stack(
        [state_change[index] for index in registered_variables.momentum_index]
    )
    gas_force = jnp.sum(gas_force_density, axis=(1, 2, 3)) * CELL_SIZE**3

    sink_forces = sink_particles.mass[:, None] * _sink_acceleration(
        primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )

    net_force = float(jnp.linalg.norm(gas_force + jnp.sum(sink_forces, axis=0)))
    force_scale = float(
        jnp.sum(jnp.linalg.norm(gas_force_density, axis=0)) * CELL_SIZE**3
        + jnp.sum(jnp.linalg.norm(sink_forces, axis=-1))
    )
    assert net_force <= 1e-10 * force_scale, (
        f"the forces on the gas and on the sinks add up to {net_force:.3e}, more than "
        f"1e-10 of the sum of their magnitudes {force_scale:.3e}"
    )


def test_kick_and_drift_skip_empty_slots():
    """A kick and a drift move the filled slots only, even with an
    acceleration in every slot. A sink crossing the periodic boundary along x
    is wrapped back into the box; a sink leaving through the open boundary
    along y is not."""
    mixed_boundaries = BoundarySettings(x=PERIODIC_AXIS, y=OPEN_AXIS, z=OPEN_AXIS)
    _, config, _, _ = _setup(mixed_boundaries)
    time_step = 0.05

    masses = [0.5, 0.8]
    positions = np.array([[0.95, 0.5, 0.5], [0.4, 0.97, 0.3]])
    velocities = np.array([[2.0, 0.3, -0.1], [0.1, 1.5, 0.2]])
    sink_particles = _sinks(masses, positions, config, velocities=velocities)
    acceleration = 1.0 + jnp.arange(3.0 * MAX_NUM_SINKS).reshape(MAX_NUM_SINKS, 3)

    kicked = _kick(sink_particles, acceleration, time_step)
    np.testing.assert_allclose(
        kicked.velocity[:2],
        velocities + np.asarray(acceleration[:2]) * time_step,
        rtol=1e-14,
        err_msg="the kicked velocities of the filled slots are not v + a Δt",
    )
    assert jnp.array_equal(kicked.position, sink_particles.position), "the kick moved the sinks"
    assert jnp.array_equal(kicked.mass, sink_particles.mass), "the kick changed the sink masses"
    assert jnp.all(kicked.velocity[2:] == 0.0), (
        f"the kick changed the velocities of the empty slots: {kicked.velocity[2:]}"
    )

    drifted = _drift(sink_particles, time_step, config)
    expected_positions = positions + velocities * time_step
    # x = 1.05 lies past the periodic boundary and is wrapped to 0.05; y =
    # 1.045 lies past the open boundary and stays there.
    expected_positions[0, 0] -= 1.0
    np.testing.assert_allclose(
        drifted.position[:2],
        expected_positions,
        rtol=1e-14,
        atol=1e-15,
        err_msg=(
            "the drifted positions of the filled slots are not x + v Δt, wrapped along the "
            "periodic x axis and not along the open y axis"
        ),
    )
    assert jnp.array_equal(drifted.velocity, sink_particles.velocity), (
        "the drift changed the sink velocities"
    )
    assert jnp.array_equal(drifted.mass, sink_particles.mass), "the drift changed the sink masses"
    assert jnp.all(drifted.position[2:] == 0.0), (
        f"the drift moved the empty slots: {drifted.position[2:]}"
    )


def test_kdk_second_order():
    """Two sinks on a circular orbit, integrated over a quarter orbit with
    fixed steps Δt, Δt/2 and Δt/4: the differences between the final
    positions shrink as Δt², so their ratio is 5 (a first-order scheme gives
    3)."""
    final_positions = []
    for num_timesteps in (128, 256, 512):
        state, config, params, registered_variables = _orbit_setup(num_timesteps)
        final_state = time_integration(state, config, params, registered_variables)
        final_positions.append(np.asarray(final_state.sink_particles.position[:2]))

    coarse, medium, fine = final_positions
    ratio = np.linalg.norm(coarse - fine) / np.linalg.norm(medium - fine)
    assert 4.0 <= ratio <= 6.0, (
        f"the convergence ratio is {ratio:.3f}, outside [4, 6] (second order gives 5, "
        "first order 3)"
    )


def test_momentum_exchange_converges():
    """A sink at rest next to a gas clump falls toward it and takes its
    momentum from the gas, up to the time sampling of the coupling: in every
    stage of the hydro update the gas feels the sink at its mid-step position,
    and each kick feels the gas at the start or at the end of the step, a
    midpoint and a trapezoid rule. Over fixed steps Δt, Δt/2 and Δt/4, the
    imbalance |Δp_gas + Δp_sink| shrinks by about 4 per halving of the step:
    2 would mean a first-order coupling, and a ratio near 1 an imbalance in
    space that no step size removes."""
    imbalances = []
    for num_timesteps in (16, 32, 64):
        gas_momentum_change, sink_momentum_change, final_sinks = _momentum_exchange_run(
            num_timesteps
        )
        assert float(final_sinks.mass[0]) == EXCHANGE_SINK_MASS, (
            f"with {num_timesteps} steps the sink mass changed from {EXCHANGE_SINK_MASS} "
            f"to {float(final_sinks.mass[0])}: gas was accreted"
        )
        assert jnp.all(final_sinks.mass[1:] == 0.0), (
            f"with {num_timesteps} steps a new sink formed, masses of the other slots: "
            f"{final_sinks.mass[1:]}"
        )
        imbalances.append(np.linalg.norm(gas_momentum_change + sink_momentum_change))

    toward_clump = EXCHANGE_CLUMP_CENTER - EXCHANGE_SINK_POSITION
    cosine = np.dot(sink_momentum_change, toward_clump) / (
        np.linalg.norm(sink_momentum_change) * np.linalg.norm(toward_clump)
    )
    assert cosine >= 0.99, (
        f"the sink's momentum change does not point at the clump: cosine {cosine:.4f} < 0.99"
    )

    for coarse_imbalance, fine_imbalance, num_timesteps in zip(
        imbalances[:-1],
        imbalances[1:],
        (16, 32),
    ):
        ratio = coarse_imbalance / fine_imbalance
        assert 3.0 <= ratio <= 5.0, (
            f"from {num_timesteps} to {2 * num_timesteps} steps the momentum imbalance "
            f"shrinks by {ratio:.3f}, outside [3, 5] (second order gives 4); imbalances: "
            f"{imbalances}"
        )
