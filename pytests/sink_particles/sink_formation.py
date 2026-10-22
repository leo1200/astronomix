"""
Sink particle formation pytest (fast).

Checks the sink particle creation of Federrath et al. (2010), Section 2.2:
the creation checks (converging flow, potential minimum, Jeans instability
with and without a magnetic field, bound state, proximity to existing sinks
and between sinks formed in the same call), the limit on the number of sink
slots, and the wrapping of a new sink's position into a periodic box.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# testing
import pytest

# jax
import jax.numpy as jnp

# sink particle test helpers
from _sink_helpers import (
    CELL_SIZE,
    CLUMP_CENTER,
    NUM_CELLS,
    SETTINGS,
    _form_sinks_once,
    _num_sinks,
    _one_sink_at,
    _setup,
    _with_linear_velocity,
)


def test_position_wrapping():
    """A sink whose centre of mass falls past a periodic boundary is stored
    inside the box."""
    # A clump centred on the second cell along x (index 1), and an external
    # potential well in the last cell along x (index 31), two cells away across
    # the x = 0 boundary. The well makes the last cell the potential minimum,
    # so the sink forms from it, while most of the gas in its control volume
    # lies past the edge: the centre of mass comes out at x ≈ 1.011.
    boundary_settings = SETTINGS._replace(
        peak_overdensity=400.0,
        overdensity_center=(1.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
    )
    state, config, params, registered_variables = _setup(settings=boundary_settings)
    config = config._replace(
        gravity_config=config.gravity_config._replace(external_potential=True)
    )
    well_index = (NUM_CELLS - 1, NUM_CELLS // 2, NUM_CELLS // 2)
    params = params._replace(
        gravitational_potential=jnp.zeros((NUM_CELLS,) * 3).at[well_index].set(-10.0)
    )

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    # The stored x lies within the first cell: past the edge, wrapped back.
    assert _num_sinks(sink_particles) == 1
    assert 0.0 <= sink_particles.position[0, 0] < CELL_SIZE


def test_proximity():
    """No sink forms within r_acc of an existing sink (Section 2.2.7), with
    distances taken across periodic boundaries."""
    state, config, params, registered_variables = _setup()
    dtype = state.primitive_state.dtype
    box_length = SETTINGS.box_length

    # The sink created by the first call blocks a second call on the same
    # state.
    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    assert _num_sinks(sink_particles) == 1
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, sink_particles
    )
    assert _num_sinks(sink_particles) == 1

    # A sink far from the clump does not block formation.
    far_sink = _one_sink_at((0.1, 0.1, 0.1), config, dtype)
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, far_sink
    )
    assert _num_sinks(sink_particles) == 2

    # A sink one box length away from the clump centre is its own periodic
    # copy, so it blocks formation.
    periodic_copy_sink = _one_sink_at(
        (CLUMP_CENTER + box_length, CLUMP_CENTER, CLUMP_CENTER),
        config,
        dtype,
    )
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, periodic_copy_sink
    )
    assert _num_sinks(sink_particles) == 1

    # A clump centred on the first cell along x, next to the x = 0 boundary.
    # An existing sink in the last cell along x is one cell away across the
    # boundary, so it blocks formation.
    boundary_settings = SETTINGS._replace(
        overdensity_center=(0.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
    )
    state, config, params, registered_variables = _setup(settings=boundary_settings)
    sink_particles = _form_sinks_once(state, config, params, registered_variables)
    assert _num_sinks(sink_particles) == 1
    across_boundary_sink = _one_sink_at(
        (box_length - 0.5 * CELL_SIZE, CLUMP_CENTER, CLUMP_CENTER),
        config,
        dtype,
    )
    sink_particles = _form_sinks_once(
        state, config, params, registered_variables, across_boundary_sink
    )
    assert _num_sinks(sink_particles) == 1


@pytest.mark.parametrize(
    "velocity_gradient, expected_num_sinks",
    [
        # Contracting along every axis: a sink forms.
        ((-1.0, -1.0, -1.0), 1),
        # Expanding along every axis: no sink.
        ((1.0, 1.0, 1.0), 0),
        # Contracting along y and z but expanding along x. The divergence is
        # negative, but the flow must converge along each axis: no sink.
        ((1.0, -2.0, -2.0), 0),
    ],
    ids=["contracting", "expanding", "expanding_along_x_only"],
)
def test_converging_flow(velocity_gradient, expected_num_sinks):
    """Sinks only form where the flow converges along every axis (Section
    2.2.3)."""
    state, config, params, registered_variables = _setup()
    state = _with_linear_velocity(state, registered_variables, velocity_gradient)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


def test_potential_minimum():
    """A sink forms only where the potential is lowest in the control volume
    (Section 2.2.4), which need not be the densest cell."""
    state, config, params, registered_variables = _setup()

    # Add a narrow density spike in a single cell, 2 cells from the clump
    # centre along x (within r_acc). The spike cell becomes the densest cell
    # (ρ ≈ 484 against ≈ 401 at the centre), but it holds too little mass to
    # move the potential minimum away from the clump centre. The pressure is
    # raised with the density, keeping c_s² = 1.
    spike_index = (NUM_CELLS // 2 + 2, NUM_CELLS // 2, NUM_CELLS // 2)
    density_index = registered_variables.density_index
    pressure_index = registered_variables.pressure_index
    primitive_state = state.primitive_state
    primitive_state = primitive_state.at[(density_index, *spike_index)].add(300.0)
    primitive_state = primitive_state.at[(pressure_index, *spike_index)].set(
        primitive_state[(density_index, *spike_index)]
        * SETTINGS.sound_speed_squared
        / SETTINGS.gamma
    )
    state = state._replace(primitive_state=primitive_state)
    density = primitive_state[density_index]
    assert density[spike_index] == jnp.max(density)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    # A single sink, formed from the centre cell: its centre of mass is pulled
    # slightly toward the spike, but stays within half a cell of the centre.
    assert _num_sinks(sink_particles) == 1
    assert jnp.all(jnp.abs(sink_particles.position[0] - CLUMP_CENTER) < 0.5 * CELL_SIZE)


def test_one_sink_per_control_volume():
    """Candidates that pass every check in the same call are kept at least
    r_acc apart (Section 2.2.7), so a clump with several equivalent potential
    minima forms one sink."""
    # The clump is centred on the corner shared by the 8 cells around cell
    # (16, 16, 16). These cells are equivalent, so their potentials differ
    # only by rounding, and several of them pass every creation check.
    corner = CLUMP_CENTER - 0.5 * CELL_SIZE
    settings = SETTINGS._replace(overdensity_center=(corner, corner, corner))
    state, config, params, registered_variables = _setup(settings)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    # A single sink, which accretes the gas around the corner: its centre of
    # mass stays within one cell of the corner.
    assert _num_sinks(sink_particles) == 1
    assert jnp.all(jnp.abs(sink_particles.position[0] - corner) < CELL_SIZE)


@pytest.mark.parametrize(
    "peak_overdensity, sound_speed_squared, expected_num_sinks",
    [
        # Cold and dense enough: |E_grav| ≈ 1.26 × 2 E_th, a sink forms.
        (400.0, 1.0, 1),
        # Above the density threshold, but not Jeans unstable:
        # |E_grav| ≈ 0.63 × 2 E_th.
        (200.0, 1.0, 0),
        # Hot: doubling c_s² doubles E_th (|E_grav| ≈ 0.63 × 2 E_th). The
        # threshold doubles too (ρ_res ≈ 257), but the centre (ρ ≈ 401) stays
        # above it, so only the Jeans check fails.
        (400.0, 2.0, 0),
    ],
    ids=["jeans_unstable", "not_dense_enough", "too_hot"],
)
def test_jeans_instability(peak_overdensity, sound_speed_squared, expected_num_sinks):
    """A sink forms only if the gas in the control volume is Jeans unstable,
    |E_grav| > 2 E_th + E_mag (Section 2.2.5)."""
    settings = SETTINGS._replace(
        peak_overdensity=peak_overdensity,
        sound_speed_squared=sound_speed_squared,
    )
    state, config, params, registered_variables = _setup(settings=settings)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


@pytest.mark.parametrize(
    "angular_velocity, expected_num_sinks",
    [
        # At rest, |E_grav| − E_th ≈ 1.52 E_th, and the rotation becomes
        # unbound at Ω ≈ 25.5 (where E_kin reaches |E_grav| − E_th).
        # About half of that: bound, a sink forms.
        (12.5, 1),
        # About twice that: unbound, no sink.
        (50.0, 0),
    ],
    ids=["slow_rotation_bound", "fast_rotation_unbound"],
)
def test_bound_state(angular_velocity, expected_num_sinks):
    """A sink forms only if the gas in the control volume is bound,
    E_grav + E_th + E_kin + E_mag < 0 (Section 2.2.6)."""
    state, config, params, registered_variables = _setup()

    # Solid-body rotation about the z axis through the clump centre. Along
    # each axis, the velocity component along that axis does not change, so
    # the converging-flow check still passes; the centre-of-mass velocity is
    # zero by symmetry, so all the motion counts as kinetic energy.
    cell_centers = (jnp.arange(NUM_CELLS) + 0.5) * CELL_SIZE
    x, y, _ = jnp.meshgrid(cell_centers, cell_centers, cell_centers, indexing="ij")
    velocity_index = registered_variables.velocity_index
    primitive_state = state.primitive_state
    primitive_state = primitive_state.at[velocity_index.x].set(
        -angular_velocity * (y - CLUMP_CENTER)
    )
    primitive_state = primitive_state.at[velocity_index.y].set(
        angular_velocity * (x - CLUMP_CENTER)
    )
    primitive_state = primitive_state.at[velocity_index.z].set(0.0)
    state = state._replace(primitive_state=primitive_state)

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


@pytest.mark.parametrize(
    "magnetic_field, expected_num_sinks",
    [
        # The control volume stops being Jeans unstable between B = 9 and
        # B = 10, and E_mag grows as B².
        # About 1% of the threshold magnetic energy: a sink forms.
        (1.0, 1),
        # About 10 times the threshold magnetic energy: no sink.
        (30.0, 0),
    ],
    ids=["weak_field", "strong_field"],
)
def test_magnetic_energy(magnetic_field, expected_num_sinks):
    """The magnetic energy E_mag = ½ Σ |B|² ΔV of the control volume counts
    against sink formation in the Jeans check, |E_grav| > 2 E_th + E_mag
    (Section 2.2.5)."""
    state, config, params, registered_variables = _setup(
        settings=SETTINGS._replace(magnetic_field_z=magnetic_field),
        mhd=True,
    )

    sink_particles = _form_sinks_once(state, config, params, registered_variables)

    assert _num_sinks(sink_particles) == expected_num_sinks


def test_error_when_sink_slots_full():
    """When a new sink passes every check but all slots are taken, the run
    stops with an error, since the sink and its gas would otherwise be lost."""
    state, config, params, registered_variables = _setup(max_num_sinks=1)
    far_sink = _one_sink_at((0.1, 0.1, 0.1), config, state.primitive_state.dtype)

    # JAX wraps the RuntimeError raised in the host callback in its own
    # runtime error type, which keeps the message.
    with pytest.raises(Exception, match="Increase max_num_sinks"):
        sink_particles = _form_sinks_once(
            state, config, params, registered_variables, far_sink
        )
        sink_particles.mass.block_until_ready()


if __name__ == "__main__":
    test_position_wrapping()
    test_proximity()
    test_converging_flow((-1.0, -1.0, -1.0), 1)
    test_converging_flow((1.0, 1.0, 1.0), 0)
    test_converging_flow((1.0, -2.0, -2.0), 0)
    test_potential_minimum()
    test_jeans_instability(400.0, 1.0, 1)
    test_jeans_instability(200.0, 1.0, 0)
    test_jeans_instability(400.0, 2.0, 0)
    test_bound_state(12.5, 1)
    test_bound_state(50.0, 0)
    test_magnetic_energy(1.0, 1)
    test_magnetic_energy(30.0, 0)
    test_error_when_sink_slots_full()
