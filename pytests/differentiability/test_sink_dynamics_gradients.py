"""
Automatic differentiation through moving sink particles.

The problem is a 16³ periodic box with a Gaussian gas clump below the density
threshold and two sinks moving through it, run in float64 with the NATIVE_JAX
backend. One sink starts exactly on a cell centre and crosses a cell face, the
other starts exactly on a cell face and crosses the periodic boundary, and 6 of
the 8 sink slots stay empty. No sink forms and no gas is accreted, so only the
dynamics is differentiated: the kicks and drifts, and the gravity between the
sinks and the gas in both directions.

What is checked:

* ``jax.grad`` through ``time_integration`` under ``BACKWARDS`` with fixed
  steps agrees with central finite differences along random directions in the
  space of the inputs (sink masses, positions and velocities, and the clump's
  peak overdensity), and gives exactly zero for the positions and velocities
  of the empty slots;
* the dot-product test ``<J^T w, u> == w . (J u)`` between the ``FORWARDS``
  JVP (while loop) and the ``BACKWARDS`` VJP (checkpointed while loop), with
  adaptive steps.

Run on the CPU (the preamble then selects fast-compiling XLA flags)::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_sink_dynamics_gradients.py
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
    BACKWARDS,
    FORWARDS,
    NATIVE_JAX,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    SimulationConfig,
    SimulationParams,
    SinkParticleConfig,
    SinkParticles,
)
from astronomix.option_classes.simulation_config import StaticIntVector

# astronomix functions
from astronomix import (
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.self_gravity.sink_particle_formation3D import (
    SinkFormationSettings,
    setup_sink_formation,
)


NUM_CELLS = 16
CELL_SIZE = 1.0 / NUM_CELLS
BOX_LENGTH = 1.0
MAX_NUM_SINKS = 8
NUM_FILLED_SLOTS = 2
NUM_FIXED_STEPS = 8

# The run lasts 0.02, well below the free-fall time of about 0.08 from 2 cells
# onto the heavier sink, so the gas stays below the density threshold of about
# 32 (c_s² = 1, r_acc = 2.5 cells at 16³). The clump is 1.6 cells wide.
SETTINGS = SinkFormationSettings(
    peak_overdensity=20.0,
    overdensity_width=0.1,
    t_end=0.02,
)

# Masses, positions and velocities of the two filled slots. The heavier sink
# starts on the centre of cell (5, 9, 6) and, along x, crosses the face 0.5 Δx
# ahead at t ≈ 0.016. The lighter one starts on the face x = Δx and crosses
# x = 0 at t ≈ 0.016.
SINK_MASSES = [0.3, 0.2]
SINK_POSITIONS = [
    [5.5 * CELL_SIZE, 9.5 * CELL_SIZE, 6.5 * CELL_SIZE],
    [1.0 * CELL_SIZE, 11.3 * CELL_SIZE, 4.6 * CELL_SIZE],
]
SINK_VELOCITIES = [
    [2.0, 0.5, -0.5],
    [-4.0, 1.0, 0.5],
]
NUM_OUTPUTS = 4 * 3 + 3


@pytest.fixture(autouse=True, scope="module")
def float64():
    """Run every test of this module in float64, without switching float64 on
    for the tests of other modules collected in the same session."""
    with jax.enable_x64(True):
        yield


# -------------------------------------------------------------
# ======================= ↓ Helpers ↓ =========================
# -------------------------------------------------------------


def _initial_inputs():
    """The differentiated inputs: the masses, positions and velocities of all
    sink slots, and the clump's peak overdensity.

    Returns:
        ``(masses, positions, velocities, peak_overdensity)``, shapes (8,),
        (8, 3), (8, 3) and ().
    """
    masses = jnp.zeros((MAX_NUM_SINKS,)).at[:NUM_FILLED_SLOTS].set(jnp.array(SINK_MASSES))
    positions = jnp.zeros((MAX_NUM_SINKS, 3)).at[:NUM_FILLED_SLOTS].set(
        jnp.array(SINK_POSITIONS)
    )
    velocities = jnp.zeros((MAX_NUM_SINKS, 3)).at[:NUM_FILLED_SLOTS].set(
        jnp.array(SINK_VELOCITIES)
    )
    return masses, positions, velocities, jnp.asarray(SETTINGS.peak_overdensity)


def _final_state(inputs, fixed_timestep, differentiation_mode):
    """Run the clump with the sinks of ``inputs`` to t = 0.02.

    Args:
        inputs: ``(masses, positions, velocities, peak_overdensity)``.
        fixed_timestep: Whether to take 8 fixed steps instead of adaptive ones.
        differentiation_mode: ``FORWARDS`` or ``BACKWARDS``.

    Returns:
        The final ``StateStruct``, and the registered variables.
    """
    masses, positions, velocities, peak_overdensity = inputs
    config = SimulationConfig(
        backend_config=BackendConfig(backend=NATIVE_JAX),
        num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, NUM_CELLS),
        progress_bar=False,
        fixed_timestep=fixed_timestep,
        num_timesteps=NUM_FIXED_STEPS,
        differentiation_mode=differentiation_mode,
        sink_particle_config=SinkParticleConfig(
            sink_particles=True,
            max_num_sinks=MAX_NUM_SINKS,
        ),
    )
    state, config, params = setup_sink_formation(
        config,
        SimulationParams(),
        SETTINGS._replace(peak_overdensity=peak_overdensity),
    )
    registered_variables = get_registered_variables(config)
    state = state._replace(
        sink_particles=SinkParticles(mass=masses, position=positions, velocity=velocities)
    )
    return time_integration(state, config, params, registered_variables), registered_variables


def _outputs(inputs, fixed_timestep, differentiation_mode):
    """The final positions and velocities of the two filled slots and the
    final gas momentum, as one vector of length 15.

    Args:
        inputs: ``(masses, positions, velocities, peak_overdensity)``.
        fixed_timestep: Whether to take 8 fixed steps instead of adaptive ones.
        differentiation_mode: ``FORWARDS`` or ``BACKWARDS``.

    Returns:
        The outputs, shape (15,).
    """
    final_state, registered_variables = _final_state(inputs, fixed_timestep, differentiation_mode)
    primitive_state = final_state.primitive_state
    density = primitive_state[registered_variables.density_index]
    gas_momentum = jnp.stack(
        [
            jnp.sum(density * primitive_state[index]) * CELL_SIZE**3
            for index in registered_variables.velocity_index
        ]
    )
    sink_particles = final_state.sink_particles
    return jnp.concatenate(
        [
            sink_particles.position[:NUM_FILLED_SLOTS].ravel(),
            sink_particles.velocity[:NUM_FILLED_SLOTS].ravel(),
            gas_momentum,
        ]
    )


def _assert_only_dynamics(inputs, fixed_timestep, differentiation_mode):
    """Check that the run forms no sink, accretes no gas, and makes both
    crossings it is meant to differentiate through.

    Args:
        inputs: ``(masses, positions, velocities, peak_overdensity)``.
        fixed_timestep: Whether the run takes fixed steps.
        differentiation_mode: ``FORWARDS`` or ``BACKWARDS``.
    """
    final_state = jax.jit(
        lambda inputs: _final_state(inputs, fixed_timestep, differentiation_mode)[0]
    )(inputs)
    sink_particles = final_state.sink_particles
    assert jnp.array_equal(sink_particles.mass, inputs[0]), (
        f"the sink masses changed from {inputs[0]} to {sink_particles.mass}: a sink formed or "
        "gas was accreted"
    )

    initial_positions = np.asarray(inputs[1])
    final_positions = np.asarray(sink_particles.position)
    initial_cell = int(np.floor(initial_positions[0, 0] / CELL_SIZE))
    final_cell = int(np.floor(final_positions[0, 0] / CELL_SIZE))
    assert final_cell == initial_cell + 1, (
        f"the heavier sink ended in cell {final_cell} along x, not {initial_cell + 1}: it did "
        "not cross exactly one cell face"
    )
    assert BOX_LENGTH - CELL_SIZE < final_positions[1, 0] < BOX_LENGTH, (
        f"the lighter sink ended at x = {final_positions[1, 0]}, not in the last cell along x: "
        "it did not cross the periodic boundary"
    )


def _random_direction(rng):
    """A random direction in the space of the inputs, scaled per kind of input
    (Δx for positions, the largest initial sink speed for velocities, the
    larger sink mass for masses, the peak overdensity for the clump). The
    masses of the empty slots get no component: any positive change would
    fill the slot.

    Args:
        rng: A NumPy random generator.

    Returns:
        A direction with the structure of the inputs.
    """
    largest_speed = float(np.max(np.linalg.norm(np.array(SINK_VELOCITIES), axis=-1)))
    mass_direction = np.zeros(MAX_NUM_SINKS)
    mass_direction[:NUM_FILLED_SLOTS] = rng.normal(size=NUM_FILLED_SLOTS)
    return (
        jnp.asarray(max(SINK_MASSES) * mass_direction),
        jnp.asarray(CELL_SIZE * rng.normal(size=(MAX_NUM_SINKS, 3))),
        jnp.asarray(largest_speed * rng.normal(size=(MAX_NUM_SINKS, 3))),
        jnp.asarray(SETTINGS.peak_overdensity * rng.normal()),
    )


def _tree_dot(first, second):
    """The dot product of two pytrees of arrays with the same structure."""
    return sum(
        jnp.vdot(first_leaf, second_leaf)
        for first_leaf, second_leaf in zip(jax.tree.leaves(first), jax.tree.leaves(second))
    )


# -------------------------------------------------------------
# ======================= ↑ Helpers ↑ =========================
# -------------------------------------------------------------


def test_gradient_matches_finite_differences():
    """With 8 fixed steps, the reverse-mode gradient of a random linear
    combination of the outputs is finite, exactly zero for the positions and
    velocities of the empty slots, and agrees with central finite differences
    along 4 random directions."""
    inputs = _initial_inputs()
    _assert_only_dynamics(inputs, True, BACKWARDS)

    rng = np.random.default_rng(0)
    weights = jnp.asarray(rng.normal(size=NUM_OUTPUTS))

    def loss(inputs):
        """The weighted sum of the outputs."""
        return jnp.vdot(weights, _outputs(inputs, True, BACKWARDS))

    gradient = jax.jit(jax.grad(loss))(inputs)
    input_names = ("masses", "positions", "velocities", "peak overdensity")
    for name, leaf in zip(input_names, gradient):
        assert jnp.all(jnp.isfinite(leaf)), f"the gradient for the {name} is not finite: {leaf}"
    _, position_gradient, velocity_gradient, _ = gradient
    assert jnp.all(position_gradient[NUM_FILLED_SLOTS:] == 0.0), (
        "the gradient for the positions of the empty slots is not exactly zero: "
        f"{position_gradient[NUM_FILLED_SLOTS:]}"
    )
    assert jnp.all(velocity_gradient[NUM_FILLED_SLOTS:] == 0.0), (
        "the gradient for the velocities of the empty slots is not exactly zero: "
        f"{velocity_gradient[NUM_FILLED_SLOTS:]}"
    )

    # Each direction takes two runs, instead of two for each of the 57 inputs.
    jitted_loss = jax.jit(loss)
    step = 1e-6
    for _ in range(4):
        direction = _random_direction(rng)
        forward_inputs = jax.tree.map(lambda leaf, d: leaf + step * d, inputs, direction)
        backward_inputs = jax.tree.map(lambda leaf, d: leaf - step * d, inputs, direction)
        finite_difference = float(
            (jitted_loss(forward_inputs) - jitted_loss(backward_inputs)) / (2.0 * step)
        )
        projected_gradient = float(_tree_dot(gradient, direction))
        assert abs(projected_gradient - finite_difference) <= 1e-5 * abs(projected_gradient), (
            f"along a random direction the gradient gives {projected_gradient!r} and the "
            f"central difference {finite_difference!r}: more than 1e-5 relative apart"
        )


def test_forward_and_reverse_agree():
    """With adaptive steps, the ``FORWARDS`` JVP and the ``BACKWARDS`` VJP are
    transposes of each other: ``<J^T w, u> == w . (J u)``."""
    inputs = _initial_inputs()
    _assert_only_dynamics(inputs, False, FORWARDS)

    rng = np.random.default_rng(1)
    direction = _random_direction(rng)
    cotangent = jnp.asarray(rng.normal(size=NUM_OUTPUTS))

    def forward_outputs(inputs):
        """The outputs of the forward-mode configuration."""
        return _outputs(inputs, False, FORWARDS)

    def backward_outputs(inputs):
        """The outputs of the reverse-mode configuration."""
        return _outputs(inputs, False, BACKWARDS)

    jacobian_vector_product = jax.jit(
        lambda inputs, direction: jax.jvp(forward_outputs, (inputs,), (direction,))
    )
    vector_jacobian_product = jax.jit(
        lambda inputs, cotangent: jax.vjp(backward_outputs, inputs)[1](cotangent)
    )
    _, jacobian_direction = jacobian_vector_product(inputs, direction)
    (transposed_cotangent,) = vector_jacobian_product(inputs, cotangent)

    lhs = float(_tree_dot(transposed_cotangent, direction))
    rhs = float(jnp.vdot(cotangent, jacobian_direction))
    assert abs(lhs - rhs) <= 1e-10 * abs(rhs), (
        f"<J^T w, u> = {lhs!r} (BACKWARDS) and w . (J u) = {rhs!r} (FORWARDS) are more than "
        "1e-10 relative apart"
    )
