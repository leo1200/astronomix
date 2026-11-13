"""
The upwind dissipation of the constrained-transport edge EMFs
(``FDSolverConfig.ct_upwind_dissipation``).

Checks that the term vanishes on a uniform field oblique to the grid, that on
a random divergence-free field it is dissipative (it lowers the interface-field
energy) and adds no divergence, that full steps with it keep div B at
round-off, that the Pallas kernels reproduce the native increments (run
interpreted on the CPU), that it is off by default with strength 0.03, that in 2D it is the 3D
term of the z-invariant field, and that the switch is rejected outside
finite-difference MHD.
The stability gain itself (uniform oblique fields no longer growing near the
Nyquist wavenumber) needs N >= 256 and several Alfven times; it is measured in
``examples/scripts/validation/ct_upwind_dissipation``.

    JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_ct_upwind_dissipation.py
"""

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    IDEAL_GAS,
    ISOTHERMAL,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    PositivityConfig,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    ct_upwind_dissipation_rhs,
)
from astronomix._spatial_operators._differencing import (
    _interface_field_divergence,
    finite_difference_int6,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

# The divergence and energy checks compare sums to round-off.
pytestmark = pytest.mark.x64

NUM_CELLS = 16
SOUND_SPEED = 0.1


def _config(ct_upwind_strength=1.0, backend=NATIVE_JAX):
    """A periodic 3D isothermal finite-difference MHD configuration (float64)
    with the upwind EMF dissipation on; the Pallas kernels run interpreted
    when JAX runs on the CPU."""
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        mhd=True,
        equation_of_state=ISOTHERMAL,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        backend_config=BackendConfig(
            backend=backend,
            pallas_interpret=jax.default_backend() == "cpu",
        ),
        solver_config=FDSolverConfig(
            weno_z=True,
            weno_positivity_preserving=True,
            ct_upwind_dissipation=True,
            ct_upwind_strength=ct_upwind_strength,
        ),
        positivity_config=PositivityConfig(),
        progress_bar=False,
    )


def _random_divergence_free_faces(seed, amplitude):
    """
    Interface fields of a uniform oblique field plus a random perturbation that
    is the discrete curl of a random edge vector potential, so its interface
    divergence vanishes to round-off.

    Args:
        seed: The random seed.
        amplitude: The amplitude of the vector potential.

    Returns:
        The interface fields (bx, by, bz).
    """
    keys = jax.random.split(jax.random.key(seed), 3)
    shape = (NUM_CELLS,) * 3
    potential_x, potential_y, potential_z = (amplitude * jax.random.normal(key, shape) for key in keys)
    spacing = 1.0 / NUM_CELLS
    # B = curl A with the CT's own staggered derivative (A_x on (y, z) edges,
    # A_y on (z, x) edges, A_z on (x, y) edges), which commutes with the
    # interface divergence.
    bx = (finite_difference_int6(potential_z, 1) - finite_difference_int6(potential_y, 2)) / spacing
    by = (finite_difference_int6(potential_x, 2) - finite_difference_int6(potential_z, 0)) / spacing
    bz = (finite_difference_int6(potential_y, 0) - finite_difference_int6(potential_x, 1)) / spacing
    direction = np.array([2.0, 1.0, 0.5]) / np.linalg.norm([2.0, 1.0, 0.5])
    return bx + direction[0], by + direction[1], bz + direction[2]


def _state(config, registered_variables, faces, flow_seed=None):
    """A state with the given interface fields, at rest with unit density, or
    with a random density (0.01 to 1) and velocity (rms 1) for ``flow_seed``."""
    bx, by, bz = faces
    shape = (NUM_CELLS,) * 3
    if flow_seed is None:
        density = jnp.ones(shape)
        velocity = (jnp.zeros(shape),) * 3
    else:
        keys = jax.random.split(jax.random.key(flow_seed), 4)
        density = 10.0 ** jax.random.uniform(keys[0], shape, minval=-2.0, maxval=0.0)
        velocity = tuple(jax.random.normal(key, shape) for key in keys[1:])
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=velocity[0],
        velocity_y=velocity[1],
        velocity_z=velocity[2],
        magnetic_field_x=bx,
        magnetic_field_y=by,
        magnetic_field_z=bz,
        interface_magnetic_field_x=bx,
        interface_magnetic_field_y=by,
        interface_magnetic_field_z=bz,
    )


def _conserved_and_faces(state, registered_variables):
    """The conserved state (momentum = density * velocity) and the interface fields."""
    conserved = state
    density = state[registered_variables.density_index]
    for velocity_index, momentum_index in zip(
        (registered_variables.velocity_index.x, registered_variables.velocity_index.y, registered_variables.velocity_index.z),
        (registered_variables.momentum_index.x, registered_variables.momentum_index.y, registered_variables.momentum_index.z),
    ):
        conserved = conserved.at[momentum_index].set(density * state[velocity_index])
    face_indices = registered_variables.interface_magnetic_field_index
    faces = (state[face_indices.x], state[face_indices.y], state[face_indices.z])
    return conserved, faces


def _upwind_increments(faces, dt_tilde=1e-3, backend=NATIVE_JAX, flow_seed=None):
    """The increments of the upwind EMF dissipation alone (see ``_state``)."""
    config = _config(backend=backend)
    registered_variables = get_registered_variables(config)
    state = _state(config, registered_variables, faces, flow_seed)
    config = finalize_config(config, state.shape)
    conserved, state_faces = _conserved_and_faces(state, registered_variables)
    params = SimulationParams(minimum_density=1e-10, isothermal_sound_speed=SOUND_SPEED)
    return ct_upwind_dissipation_rhs(
        conserved,
        *state_faces,
        dt_tilde,
        1.0 / NUM_CELLS,
        5.0 / 3.0,
        params,
        config,
        registered_variables,
    ), state_faces


def test_uniform_oblique_field_is_left_unchanged():
    """The jumps of a uniform field vanish, so the term does too."""
    direction = np.array([2.0, 1.0, 0.5]) / np.linalg.norm([2.0, 1.0, 0.5])
    shape = (NUM_CELLS,) * 3
    faces = tuple(jnp.full(shape, component) for component in direction)
    increments, _ = _upwind_increments(faces)
    for increment in increments:
        assert float(jnp.max(jnp.abs(increment))) < 1e-13


def test_random_field_is_damped_without_divergence():
    """On a random divergence-free field the term lowers the interface-field
    energy and adds no interface divergence."""
    faces = _random_divergence_free_faces(seed=3, amplitude=1e-3)
    increments, state_faces = _upwind_increments(faces)
    energy_change = sum(float(jnp.sum(face * increment)) for face, increment in zip(state_faces, increments))
    assert energy_change < 0.0
    divergence = _interface_field_divergence(*increments, 1.0 / NUM_CELLS)
    increment_scale = max(float(jnp.max(jnp.abs(increment))) for increment in increments)
    assert float(jnp.max(jnp.abs(divergence))) < 1e-10 * increment_scale * NUM_CELLS


def test_steps_keep_divergence_free():
    """A few full steps with the term keep div B at round-off."""
    config = _config()
    registered_variables = get_registered_variables(config)
    faces = _random_divergence_free_faces(seed=5, amplitude=1e-3)
    state = _state(config, registered_variables, faces)
    config = finalize_config(config, state.shape)
    params = SimulationParams(
        C_cfl=0.75,
        minimum_density=1e-10,
        isothermal_sound_speed=SOUND_SPEED,
        t_end=0.02,
    )
    final = time_integration(state, config, params, registered_variables)
    face_indices = registered_variables.interface_magnetic_field_index
    divergence = _interface_field_divergence(
        final[face_indices.x],
        final[face_indices.y],
        final[face_indices.z],
        1.0 / NUM_CELLS,
    )
    assert bool(jnp.all(jnp.isfinite(final)))
    assert float(jnp.max(jnp.abs(divergence))) < 1e-9


def test_pallas_matches_native():
    """The Pallas kernels give the native increments on a random flow."""
    faces = _random_divergence_free_faces(seed=7, amplitude=1e-2)
    native, _ = _upwind_increments(faces, flow_seed=11)
    pallas, _ = _upwind_increments(faces, backend=PALLAS, flow_seed=11)
    scale = max(float(jnp.max(jnp.abs(increment))) for increment in native)
    assert scale > 0.0
    for native_increment, pallas_increment in zip(native, pallas):
        assert float(jnp.max(jnp.abs(pallas_increment - native_increment))) < 1e-12 * scale


def test_default_strength():
    """The switch is off by default; switched on, its strength is 0.03."""
    solver_config = FDSolverConfig()
    assert not solver_config.ct_upwind_dissipation
    assert solver_config.ct_upwind_strength == 0.03


def test_two_dimensional_term_matches_z_invariant_three_dimensional_term():
    """In 2D the term is the 3D term of the z-invariant field."""
    keys = jax.random.split(jax.random.key(13), 6)
    plane = (NUM_CELLS, NUM_CELLS)
    density = 10.0 ** jax.random.uniform(keys[0], plane, minval=-1.0, maxval=0.0)
    velocity = [jax.random.normal(key, plane) for key in keys[1:4]]
    potential = 1e-2 * jax.random.normal(keys[4], plane)
    spacing = 1.0 / NUM_CELLS
    # In-plane field from a z vector potential (divergence-free), plus a B_z.
    bx = finite_difference_int6(potential, 1) / spacing + 0.8
    by = -finite_difference_int6(potential, 0) / spacing + 0.4
    bz = 0.3 + 0.1 * jax.random.normal(keys[5], plane)
    params = SimulationParams(minimum_density=1e-10, isothermal_sound_speed=SOUND_SPEED)
    increments = {}
    for dimensionality in (2, 3):
        periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        config = _config()._replace(
            dimensionality=dimensionality,
            boundary_settings=BoundarySettings(*(periodic,) * dimensionality),
        )
        if dimensionality == 3:
            # A periodic slab of eight cells along z, at the same cell size.
            config = config._replace(
                num_cells=StaticIntVector(NUM_CELLS, NUM_CELLS, 8),
                box_size=StaticFloatVector(1.0, 1.0, 8 * spacing),
            )

        def extend(field):
            return field if dimensionality == 2 else jnp.broadcast_to(field[:, :, None], plane + (8,))

        registered_variables = get_registered_variables(config)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=extend(density),
            velocity_x=extend(velocity[0]),
            velocity_y=extend(velocity[1]),
            velocity_z=extend(velocity[2]),
            magnetic_field_x=extend(bx),
            magnetic_field_y=extend(by),
            magnetic_field_z=extend(bz),
            interface_magnetic_field_x=extend(bx),
            interface_magnetic_field_y=extend(by),
            interface_magnetic_field_z=extend(bz),
        )
        config = finalize_config(config, state.shape)
        conserved, faces = _conserved_and_faces(state, registered_variables)
        increments[dimensionality] = ct_upwind_dissipation_rhs(
            conserved, *faces, 1e-3, spacing, 5.0 / 3.0, params, config, registered_variables,
        )
    scale = max(float(jnp.max(jnp.abs(increment))) for increment in increments[3])
    assert scale > 0.0
    for two_dimensional, three_dimensional in zip(increments[2], increments[3]):
        assert float(jnp.max(jnp.abs(three_dimensional[:, :, 0] - two_dimensional))) < 1e-13 * scale


@pytest.mark.parametrize("variant", ["finite_volume", "hydrodynamics"])
def test_switch_requires_finite_difference_mhd(variant):
    """The switch is rejected for the finite-volume solver and without MHD."""
    config = _config()
    if variant == "finite_volume":
        # The switch is a field of the finite-difference container, which the
        # finite-volume solver does not take.
        config = config._replace(solver_mode=FINITE_VOLUME, equation_of_state=IDEAL_GAS)
        shape = (8, NUM_CELLS, NUM_CELLS, NUM_CELLS)
        message = "needs an FVSolverConfig"
    else:
        config = config._replace(mhd=False)
        shape = (4, NUM_CELLS, NUM_CELLS, NUM_CELLS)
        message = "ct_upwind_dissipation"
    with pytest.raises(ValueError, match=message):
        finalize_config(config, shape)
