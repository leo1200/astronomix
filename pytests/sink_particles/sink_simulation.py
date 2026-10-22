"""
Sink particles in whole simulations pytest.

Checks the configuration requirements enforced by ``finalize_config``, a run
of the time loop in which a sink forms (through the in-memory snapshots), and
a restart from the disk snapshots.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import tempfile
from pathlib import Path

# testing
import pytest

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import TO_DISK
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import (
    GravityConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
    SnapshotSettings,
    StateStruct,
)
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix import (
    finalize_state,
    get_registered_variables,
    restart_from_latest_checkpoint,
    time_integration,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    setup_sink_formation,
)

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    NUM_CELLS,
    SETTINGS,
    _num_sinks,
    _one_sink_at,
    _setup,
)


@pytest.mark.parametrize(
    "self_gravity_version, mhd",
    [
        (SIMPLE_SOURCE, False),
        (SECOND_ORDER_CONSERVATIVE, False),
        (FOURTH_ORDER_CONSERVATIVE, False),
        (FOURTH_ORDER_CONSERVATIVE, True),
    ],
    ids=[
        "simple_source",
        "second_order_conservative",
        "fourth_order_conservative",
        "fourth_order_conservative_mhd",
    ],
)
def test_conservation_in_run(self_gravity_version, mhd):
    """A sink forms during a run of the time loop, no second sink forms, and
    gas plus sink mass and momentum are conserved at every snapshot, for each
    finite-difference self-gravity treatment and, for one treatment, with MHD.
    The gas moves with a small uniform bulk velocity, so the momentum is not
    zero. The run goes through the in-memory snapshots, which record the
    sinks. Not checked: the physics of the collapse (the gravity of the sinks
    on the gas is not implemented), total energy (accretion removes the energy
    of the accreted gas) and angular momentum (sinks have no spin).

    Args:
        self_gravity_version: The finite-difference self-gravity treatment.
        mhd: Whether MHD is switched on, with a uniform field B_z = 1.
    """
    config = SimulationConfig(
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        mhd=mhd,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
        ),
        return_snapshots=True,
        num_snapshots=5,
        snapshot_settings=SnapshotSettings(return_states=True),
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    # About 30 adaptive steps, a few free-fall times of the clump peak.
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SETTINGS._replace(t_end=0.06, magnetic_field_z=1.0),
    )
    registered_variables = get_registered_variables(config)

    # A uniform bulk velocity makes the total momentum non-zero. Over the run
    # the gas moves less than half a cell, so it stays around the sink, which
    # does not move yet.
    bulk_velocity = jnp.array([0.1, -0.2, 0.05])
    primitive_state = state.primitive_state
    for axis, velocity_index in enumerate(registered_variables.velocity_index):
        primitive_state = primitive_state.at[velocity_index].set(bulk_velocity[axis])
    state = state._replace(primitive_state=primitive_state)

    snapshots = time_integration(state, config, params, registered_variables)

    # No sink at t = 0 (recorded before the first formation call); one sink
    # from the next snapshot on.
    num_sinks = jnp.sum(snapshots.sink_particles.mass > 0.0, axis=1)
    assert num_sinks[0] == 0
    assert jnp.all(num_sinks[1:] == 1)

    # Gas plus sink mass and momentum at every snapshot equal the initial gas
    # mass and momentum.
    cell_volume = CELL_SIZE**3
    density_index = registered_variables.density_index
    velocity_indices = jnp.array(registered_variables.velocity_index)
    initial_density = primitive_state[density_index]
    initial_mass = jnp.sum(initial_density) * cell_volume
    initial_momentum = jnp.sum(
        initial_density * primitive_state[velocity_indices],
        axis=(1, 2, 3),
    ) * cell_volume

    density = snapshots.states[:, density_index]
    velocity = snapshots.states[:, velocity_indices]
    gas_mass = jnp.sum(density, axis=(1, 2, 3)) * cell_volume
    gas_momentum = jnp.sum(density[:, None] * velocity, axis=(2, 3, 4)) * cell_volume
    sinks = snapshots.sink_particles
    sink_mass = jnp.sum(sinks.mass, axis=1)
    sink_momentum = jnp.sum(sinks.mass[..., None] * sinks.velocity, axis=1)

    assert jnp.allclose(gas_mass + sink_mass, initial_mass, rtol=1e-4)
    assert jnp.allclose(gas_momentum + sink_momentum, initial_momentum, rtol=1e-4)


@pytest.mark.parametrize(
    "slots_factor",
    [1, 2],
    ids=["same_slots", "more_slots"],
)
def test_restart_from_disk(tmp_path, slots_factor):
    """Disk snapshots carry the sinks: a run restarted from a disk checkpoint
    starts with the sinks of that checkpoint and ends with the same gas state
    and the same sinks as the uninterrupted run. A restart with more sink
    slots gets empty slots appended to the restored sinks.

    Args:
        tmp_path: pytest's temporary directory, for the checkpoints.
        slots_factor: The ratio of the restarted run's max_num_sinks to the
            uninterrupted run's.
    """
    pytest.importorskip("orbax.checkpoint")
    state, config, params, registered_variables = _setup()

    # Two segments of adaptive steps; each ends exactly on its snapshot time
    # and writes a checkpoint. The sink forms in the first step.
    uninterrupted_path = str(tmp_path / "uninterrupted")
    config = config._replace(
        fixed_timestep=False,
        snapshot_storage_mode=TO_DISK,
        snapshot_storage_path=uninterrupted_path,
        num_snapshots=2,
    )
    final_state = time_integration(state, config, params, registered_variables)
    assert _num_sinks(final_state.sink_particles) == 1

    # Restart from the checkpoint of the first segment and run the second
    # segment again, writing to a separate directory.
    primitive_state, restart_params, restart_state = restart_from_latest_checkpoint(
        uninterrupted_path,
        params,
        step=1,
    )
    assert _num_sinks(restart_state.sink_particles) == 1
    max_num_sinks = config.sink_particle_config.max_num_sinks
    restart_config = config._replace(
        snapshot_storage_path=str(tmp_path / "restarted"),
        num_snapshots=1,
        sink_particle_config=config.sink_particle_config._replace(
            max_num_sinks=slots_factor * max_num_sinks
        ),
    )
    restarted_state = time_integration(
        finalize_state(restart_config, primitive_state),
        restart_config,
        restart_params,
        registered_variables,
        restart_state=restart_state,
    )

    assert jnp.array_equal(restarted_state.primitive_state, final_state.primitive_state)
    # The restored sinks fill the first slots; the slots added in a restart
    # with more slots stay empty.
    for restarted_field, final_field in zip(
        restarted_state.sink_particles,
        final_state.sink_particles,
    ):
        assert restarted_field.shape[0] == slots_factor * max_num_sinks
        assert jnp.array_equal(restarted_field[:max_num_sinks], final_field)
        assert jnp.all(restarted_field[max_num_sinks:] == 0.0)


def test_finalize_state():
    """``finalize_state`` puts the state in the form ``time_integration``
    expects: a ``StateStruct`` holding the primitive state and the given sinks
    when ``config.state_struct`` is set, otherwise the primitive state itself.
    Sinks without ``state_struct`` are rejected."""
    state, config, _, _ = _setup()
    primitive_state = state.primitive_state
    sink_particles = _one_sink_at((0.1, 0.1, 0.1), config, primitive_state.dtype)

    finalized_state = finalize_state(config, primitive_state, sink_particles)
    assert isinstance(finalized_state, StateStruct)
    assert finalized_state.primitive_state is primitive_state
    assert finalized_state.sink_particles is sink_particles

    config = config._replace(state_struct=False)
    assert finalize_state(config, primitive_state) is primitive_state
    with pytest.raises(ValueError, match="state_struct"):
        finalize_state(config, primitive_state, sink_particles)


@pytest.mark.parametrize(
    "unsupported_options, state_shape",
    [
        (dict(dimensionality=2), (4, 16, 16)),
        (dict(gravity_config=GravityConfig(self_gravity=False)), (5, 16, 16, 16)),
        (dict(state_struct=False), (5, 16, 16, 16)),
    ],
    ids=["2d", "no_self_gravity", "no_state_struct"],
)
def test_sink_particle_config_requirements(unsupported_options, state_shape):
    """``finalize_config`` must reject each configuration that sink particle
    formation does not support.

    Args:
        unsupported_options: The config fields that make the configuration
            unsupported, on top of an otherwise valid one.
        state_shape: The primitive state shape matching the dimensionality.
    """
    supported_options = dict(
        dimensionality=3,
        gravity_config=GravityConfig(self_gravity=True),
        state_struct=True,
        sink_particle_config=SinkParticleConfig(sink_particles=True),
    )
    config = SimulationConfig(**{**supported_options, **unsupported_options})

    with pytest.raises(ValueError):
        finalize_config(config, state_shape)


if __name__ == "__main__":
    test_conservation_in_run(SIMPLE_SOURCE, False)
    test_conservation_in_run(SECOND_ORDER_CONSERVATIVE, False)
    test_conservation_in_run(FOURTH_ORDER_CONSERVATIVE, False)
    test_conservation_in_run(FOURTH_ORDER_CONSERVATIVE, True)
    test_restart_from_disk(Path(tempfile.mkdtemp()), 1)
    test_restart_from_disk(Path(tempfile.mkdtemp()), 2)
    test_finalize_state()
    test_sink_particle_config_requirements(dict(dimensionality=2), (4, 16, 16))
    test_sink_particle_config_requirements(
        dict(gravity_config=GravityConfig(self_gravity=False)),
        (5, 16, 16, 16),
    )
    test_sink_particle_config_requirements(dict(state_struct=False), (5, 16, 16, 16))
