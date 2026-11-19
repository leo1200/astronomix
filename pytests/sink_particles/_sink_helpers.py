"""
Shared setup and helpers for the sink particle pytests.

The tests run on the Gaussian overdensity setup in
``astronomix/test_setups/self_gravity/sink_particle_formation3D.py``.
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import OPTIMAL_BACKEND

# astronomix containers
from astronomix import (
    BackendConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
)
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix import get_registered_variables
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    SinkFormationSettings,
    setup_sink_formation,
)
from astronomix._modules._sink_particles._sink_particle_formation import (
    _empty_sink_particles,
    _form_sink_particles,
)
from astronomix._modules._sink_particles._sink_particle_accretion import _accrete_gas


NUM_CELLS = 32
NUM_TIMESTEPS = 3
CELL_SIZE = 1.0 / NUM_CELLS

# The clump is centred on the centre of cell (16, 16, 16), so that this cell is
# the unique potential minimum. With c_s² = 1, G = 1 and r_acc = 2.5 cells on a
# 32³ unit box, the density threshold (Eq. 32) is ρ_res = π / (4 r_acc²) ≈ 129.
# A peak overdensity of 400 puts 57 cells above it, all within the centre
# cell's control volume, and makes the control volume Jeans unstable
# (|E_grav| ≈ 1.26 × 2 E_th).
CLUMP_CENTER = 0.5 + 0.5 * CELL_SIZE
SETTINGS = SinkFormationSettings(
    peak_overdensity=400.0,
    overdensity_center=(CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER),
)


def _setup(
    settings=SETTINGS,
    mhd=False,
    backend=OPTIMAL_BACKEND,
    **sink_particle_options,
):
    """Set up the Gaussian overdensity with sink particles on and the given
    sink options.

    Args:
        settings: The problem constants of the setup.
        mhd: Whether MHD is switched on (the field is ``settings.magnetic_field_z``).
        backend: The compute backend.
        **sink_particle_options: Further ``SinkParticleConfig`` fields.

    Returns:
        ``(state, config, params, registered_variables)``.
    """
    config = SimulationConfig(
        backend_config=BackendConfig(backend=backend),
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        fixed_timestep=True,
        num_timesteps=NUM_TIMESTEPS,
        progress_bar=False,
        mhd=mhd,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            **sink_particle_options,
        ),
    )
    state, config, params = setup_sink_formation(config, SimulationParams(), settings)
    registered_variables = get_registered_variables(config)
    return state, config, params, registered_variables


def _density_threshold(config, params):
    """The density threshold of Eq. 32 for the setup's uniform c_s² = 1."""
    accretion_radius = (
        config.sink_particle_config.accretion_radius_in_cells * config.grid_spacing
    )
    return jnp.pi * SETTINGS.sound_speed_squared / (
        4.0 * params.gravitational_constant * accretion_radius**2
    )


def _update_sinks_once(state, config, params, registered_variables, sink_particles=None):
    """Apply one formation and accretion step, as in a time step, by default
    starting from no sinks.

    Returns:
        ``(primitive_state, sink_particles)`` after the step.
    """
    if sink_particles is None:
        sink_particles = _empty_sink_particles(config, state.primitive_state.dtype)
    sink_particles, num_active_sinks = _form_sink_particles(
        state.primitive_state,
        sink_particles,
        config,
        params,
        registered_variables,
    )
    return _accrete_gas(
        state.primitive_state,
        sink_particles,
        num_active_sinks,
        config,
        params,
        registered_variables,
    )


def _form_sinks_once(state, config, params, registered_variables, sink_particles=None):
    """Apply one formation and accretion step; return only the sinks."""
    _, sink_particles = _update_sinks_once(
        state,
        config,
        params,
        registered_variables,
        sink_particles,
    )
    return sink_particles


def _one_sink_at(position, config, dtype):
    """Sink particles with a single (unit-mass) sink at ``position``."""
    sink_particles = _empty_sink_particles(config, dtype)
    return sink_particles._replace(
        mass=sink_particles.mass.at[0].set(1.0),
        position=sink_particles.position.at[0].set(jnp.array(position)),
    )


def _num_sinks(sink_particles):
    """The number of filled sink slots."""
    return int(jnp.sum(sink_particles.mass > 0.0))


def _gas_mass_and_momentum(primitive_state, registered_variables):
    """Total gas mass and momentum of an (unpadded, periodic) state."""
    cell_volume = CELL_SIZE**3
    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [primitive_state[index] for index in registered_variables.velocity_index]
    )
    mass = jnp.sum(density) * cell_volume
    momentum = jnp.sum(density * velocity, axis=(1, 2, 3)) * cell_volume
    return mass, momentum


def _with_linear_velocity(state, registered_variables, velocity_gradient):
    """Set the gas velocity to v_d = g_d (x_d − x_c) along each axis d, a flow
    that expands (g_d > 0) or contracts (g_d < 0) about the clump centre x_c."""
    cell_centers = (jnp.arange(NUM_CELLS) + 0.5) * CELL_SIZE
    coordinates = jnp.meshgrid(cell_centers, cell_centers, cell_centers, indexing="ij")
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(
            velocity_gradient[axis] * (coordinates[axis] - CLUMP_CENTER)
        )
    return state._replace(primitive_state=primitive_state)


def _sink_at_clump_center(mass, config, dtype):
    """Sink particles with a single sink of ``mass`` at the clump centre, at
    rest."""
    sink_particles = _one_sink_at(
        (CLUMP_CENTER, CLUMP_CENTER, CLUMP_CENTER),
        config,
        dtype,
    )
    return sink_particles._replace(mass=sink_particles.mass.at[0].set(mass))


def _num_accreted_cells(primitive_state, new_primitive_state, registered_variables):
    """The number of cells that lost gas."""
    density_index = registered_variables.density_index
    return int(jnp.sum(new_primitive_state[density_index] < primitive_state[density_index]))
