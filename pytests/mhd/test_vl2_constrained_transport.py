"""
VL2 with constrained transport (``FVSolverConfig.divergence_control``).

The option replaces the GLM (Dedner) cleaning of the VL2 finite-volume scheme
by the constrained transport of Stone & Gardiner (2009): the face-centred field
is evolved with edge EMFs, the cell-centred field is the face average.

Checks that the option is off by default and leaves the GLM layout alone, that
the state then holds the face field (rows 8 to 10) next to the cell-centred
one, that ``construct_primitive_state`` builds a consistent pair of fields,
that full runs (Orszag-Tang vortex; a low-beta blast with the first-order flux
correction) keep the face divergence at round-off, keep the cell-centred field
equal to the face average and conserve the total energy, that the oblique
circularly polarised Alfven wave of Gardiner & Stone (2008) converges at
second order, and that the switch is rejected where it is not implemented
(other integrators, non-periodic boundaries, gravity, the GLM source option,
hydrodynamics).

    JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_vl2_constrained_transport.py
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
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

# The divergence and energy checks compare sums to round-off.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    GLM,
    HLLD,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    POSITIVITY_HARD_FLOOR,
    RK2_SSP,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    GravityConfig,
    PositivityConfig,
    SimulationConfig,
    StaticFloatVector,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.time_stepping.time_integration import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables

GAMMA = 5.0 / 3.0


# ---------------------------------------------------------------------------
# Set-up helpers
# ---------------------------------------------------------------------------
def _config(
    dimensionality, box_size, constrained_transport=True, solver_overrides=None, **overrides
):
    """A periodic VL2 MHD configuration (HLLD, van Leer limiter, native JAX).

    ``solver_overrides`` replaces fields of the ``FVSolverConfig``, ``overrides``
    fields of the ``SimulationConfig``.
    """
    periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
    settings = dict(
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(
            time_integrator=VL2,
            limiter=VAN_LEER,
            riemann_solver=HLLD,
            divergence_control=CONSTRAINED_TRANSPORT if constrained_transport else GLM,
        )._replace(**(solver_overrides or {})),
        mhd=True,
        dimensionality=dimensionality,
        box_size=StaticFloatVector(*(list(box_size) + [1.0] * (3 - dimensionality))),
        boundary_settings=BoundarySettings(
            *([periodic] * dimensionality + [BoundarySettings1D()] * (3 - dimensionality))
        ),
        backend_config=BackendConfig(backend=NATIVE_JAX),
        progress_bar=False,
    )
    settings.update(overrides)
    return SimulationConfig(**settings)


def _faces_from_potential(potential, mean_field, num_cells, dx):
    """The face field curl A + mean field, A given at the cell edges.

    The two-point curl has zero face divergence identically. Index i of a face
    array holds the value at i + 1/2; an axis with one cell is ignorable.
    """
    centre = [(np.arange(num_cells[a]) + 0.5) * dx for a in range(3)]
    upper = [centre[a] + (0.5 * dx if num_cells[a] > 1 else 0.0) for a in range(3)]

    def grid(staggered):
        return np.meshgrid(
            *[upper[a] if staggered[a] else centre[a] for a in range(3)], indexing="ij"
        )

    def difference(values, axis):
        return (values - np.roll(values, 1, axis=axis)) / dx

    a_x = potential(*grid((False, True, True)))[0]
    a_y = potential(*grid((True, False, True)))[1]
    a_z = potential(*grid((True, True, False)))[2]
    return [
        mean_field[0] + difference(a_z, 1) - difference(a_y, 2),
        mean_field[1] + difference(a_x, 2) - difference(a_z, 0),
        mean_field[2] + difference(a_y, 0) - difference(a_x, 1),
    ]


def _state(density, velocity, pressure, faces, dimensionality):
    """The primitive state (rho, v, p, B, b) from three-dimensional arrays."""
    cell_field = [0.5 * (faces[a] + np.roll(faces[a], 1, axis=a)) for a in range(3)]
    rows = [density, *velocity, pressure, *cell_field, *faces]
    state = np.stack([np.broadcast_to(row, density.shape) for row in rows])
    return jnp.asarray(state.reshape(state.shape[: 1 + dimensionality]))


def _run(state, config, t_end, cfl=0.3, **params):
    config = finalize_config(config, state.shape)
    simulation_params = SimulationParams(C_cfl=cfl, gamma=GAMMA, t_end=t_end, **params)
    registered_variables = get_registered_variables(config)
    final = time_integration(state, config, simulation_params, registered_variables)
    return np.asarray(jax.block_until_ready(final))


def _relative_face_divergence(state, dx, dimensionality):
    """max |div b| dx / max |B| with the face divergence of rows 8 to 10."""
    state = np.asarray(state)
    divergence = sum(
        (state[8 + a] - np.roll(state[8 + a], 1, axis=a)) / dx for a in range(dimensionality)
    )
    return float(np.abs(divergence).max() * dx / np.abs(state[5:8]).max())


def _face_average_defect(state, dimensionality):
    """Largest difference between the cell-centred field and the face average."""
    state = np.asarray(state)
    return max(
        float(np.abs(state[5 + a] - 0.5 * (state[8 + a] + np.roll(state[8 + a], 1, axis=a))).max())
        for a in range(dimensionality)
    )


def _conserved(state):
    """(rho, momentum, total energy, cell-centred field) of a primitive state."""
    state = np.asarray(state)
    energy = (
        state[4] / (GAMMA - 1.0)
        + 0.5 * state[0] * (state[1:4] ** 2).sum(0)
        + 0.5 * (state[5:8] ** 2).sum(0)
    )
    return np.stack([state[0], *(state[0] * state[1:4]), energy, *state[5:8]])


def _orszag_tang(n):
    """The Orszag-Tang vortex on the unit square, field from A_z on the corners."""
    shape, dx = (n, n, 1), 1.0 / n
    b0 = 1.0 / np.sqrt(4.0 * np.pi)

    def potential(x, y, z):
        a_z = b0 * (np.cos(4 * np.pi * x) / (4 * np.pi) + np.cos(2 * np.pi * y) / (2 * np.pi))
        return np.zeros_like(a_z), np.zeros_like(a_z), a_z

    x, y, _ = np.meshgrid(*[(np.arange(m) + 0.5) * dx for m in shape], indexing="ij")
    velocity = [-np.sin(2 * np.pi * y), np.sin(2 * np.pi * x), np.zeros(shape)]
    faces = _faces_from_potential(potential, (0.0, 0.0, 0.0), shape, dx)
    density = np.full(shape, 25.0 / (36.0 * np.pi))
    pressure = np.full(shape, 5.0 / (12.0 * np.pi))
    return _state(density, velocity, pressure, faces, 2), dx


def _alfven_wave(n):
    """The oblique circularly polarised Alfven wave of Gardiner & Stone (2008).

    Box (3, 1.5, 1.5), grid (2n, n, n), wave vector along (1, 2, 2) / 3, one
    period is t = 1.
    """
    amplitude, b_parallel, pressure0 = 0.1, 1.0, 0.1
    shape, dx = (2 * n, n, n), 1.5 / n
    sqrt5 = np.sqrt(5.0)
    # Columns: propagation direction and the two transverse unit vectors.
    rotation = np.array(
        [
            [1 / 3, -2 / sqrt5, -2 / (3 * sqrt5)],
            [2 / 3, 1 / sqrt5, -4 / (3 * sqrt5)],
            [2 / 3, 0.0, sqrt5 / 3],
        ]
    )

    def phase(x, y, z):
        return 2 * np.pi * (rotation[0, 0] * x + rotation[1, 0] * y + rotation[2, 0] * z)

    def potential(x, y, z):
        a_2 = amplitude / (2 * np.pi) * np.sin(phase(x, y, z))
        a_3 = amplitude / (2 * np.pi) * np.cos(phase(x, y, z))
        return tuple(rotation[a, 1] * a_2 + rotation[a, 2] * a_3 for a in range(3))

    x, y, z = np.meshgrid(*[(np.arange(m) + 0.5) * dx for m in shape], indexing="ij")
    d_2, d_3 = amplitude * np.sin(phase(x, y, z)), amplitude * np.cos(phase(x, y, z))
    velocity = [rotation[a, 1] * d_2 + rotation[a, 2] * d_3 for a in range(3)]
    faces = _faces_from_potential(potential, b_parallel * rotation[:, 0], shape, dx)
    return _state(np.ones(shape), velocity, np.full(shape, pressure0), faces, 3), dx


# ---------------------------------------------------------------------------
# Layout and initial state
# ---------------------------------------------------------------------------
def test_off_by_default_and_glm_layout_unchanged():
    assert FVSolverConfig().divergence_control == GLM

    glm = get_registered_variables(_config(2, (1.0, 1.0), constrained_transport=False))
    assert glm.num_vars == 9
    assert glm.magnetic_psi_active

    ct = get_registered_variables(_config(2, (1.0, 1.0)))
    assert ct.num_vars == 11
    assert not ct.magnetic_psi_active
    assert tuple(ct.magnetic_index) == (5, 6, 7)
    assert tuple(ct.interface_magnetic_field_index) == (8, 9, 10)


def test_construct_primitive_state_fields_are_consistent():
    n = 16
    config = _config(2, (1.0, 1.0))
    registered_variables = get_registered_variables(config)
    reference, dx = _orszag_tang(n)
    reference = np.asarray(reference)
    density = jnp.asarray(reference[0])

    # Face field given: the cell-centred field is derived as the face average
    # and replaces whatever cell-centred values were passed.
    state = construct_primitive_state(
        config,
        registered_variables,
        density=density,
        velocity_x=jnp.asarray(reference[1]),
        velocity_y=jnp.asarray(reference[2]),
        velocity_z=jnp.asarray(reference[3]),
        magnetic_field_x=jnp.zeros_like(density),
        interface_magnetic_field_x=jnp.asarray(reference[8]),
        interface_magnetic_field_y=jnp.asarray(reference[9]),
        interface_magnetic_field_z=jnp.asarray(reference[10]),
        gas_pressure=jnp.asarray(reference[4]),
    )
    assert state.shape == (11, n, n)
    np.testing.assert_allclose(np.asarray(state), reference, rtol=0.0, atol=1e-15)
    assert _relative_face_divergence(state, dx, 2) < 1e-13

    # Only a cell-centred field given: the faces are the average of the two
    # adjacent cells, and the cell-centred field is then their average. For a
    # uniform field both equal the input.
    uniform = construct_primitive_state(
        config,
        registered_variables,
        density=density,
        velocity_x=jnp.zeros_like(density),
        velocity_y=jnp.zeros_like(density),
        velocity_z=jnp.zeros_like(density),
        magnetic_field_x=jnp.full_like(density, 0.3),
        magnetic_field_y=jnp.full_like(density, -0.2),
        magnetic_field_z=jnp.full_like(density, 0.1),
        gas_pressure=jnp.ones_like(density),
    )
    uniform = np.asarray(uniform)
    for row, value in zip(range(5, 11), (0.3, -0.2, 0.1, 0.3, -0.2, 0.1)):
        np.testing.assert_allclose(uniform[row], value, rtol=0.0, atol=1e-15)
    assert _face_average_defect(uniform, 2) < 1e-15


# ---------------------------------------------------------------------------
# Full runs
# ---------------------------------------------------------------------------
def test_orszag_tang_divergence_face_average_and_energy():
    n = 32
    state0, dx = _orszag_tang(n)
    final = _run(state0, _config(2, (1.0, 1.0)), t_end=0.5)

    assert np.isfinite(final).all()
    assert final[0].min() > 0.0 and final[4].min() > 0.0
    # The face divergence stays at round-off (the GLM scheme reaches 8e-2 here).
    assert _relative_face_divergence(final, dx, 2) < 1e-12
    assert _face_average_defect(final, 2) < 1e-14
    # Flux form: mass, momentum and total energy are conserved to round-off.
    before, after = _conserved(state0).mean(axis=(1, 2)), _conserved(final).mean(axis=(1, 2))
    np.testing.assert_allclose(after[[0, 4]], before[[0, 4]], rtol=1e-12)
    np.testing.assert_allclose(after[1:4], before[1:4], rtol=0.0, atol=1e-13)
    # The vortex has evolved: the field energy has grown by tens of percent.
    assert (final[5:8] ** 2).sum(0).mean() > 1.1 * (np.asarray(state0)[5:8] ** 2).sum(0).mean()


def test_low_beta_blast_with_first_order_flux_correction():
    # Pressure ratio 1e4 and plasma beta 2e-3 outside the blast: the run needs
    # the first-order flux correction; the corrected fluxes feed the same edge
    # EMFs, so the divergence must stay at round-off.
    n = 32
    shape, dx = (n, n, 1), 1.0 / n
    x, y, _ = np.meshgrid(*[(np.arange(m) + 0.5) * dx for m in shape], indexing="ij")
    radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    pressure = np.where(radius < 0.1, 10.0, 1e-3)
    field = 1.0 / np.sqrt(2.0)
    faces = [np.full(shape, field), np.full(shape, field), np.zeros(shape)]
    state0 = _state(np.ones(shape), [np.zeros(shape)] * 3, pressure, faces, 2)

    config = _config(
        2,
        (1.0, 1.0),
        solver_overrides=dict(first_order_flux_correction=True),
        positivity_config=PositivityConfig(per_step_mode=POSITIVITY_HARD_FLOOR),
    )
    final = _run(state0, config, t_end=0.05, minimum_density=1e-10, minimum_pressure=1e-10)

    assert np.isfinite(final).all()
    assert final[0].min() > 0.0 and final[4].min() > 0.0
    assert final[0].max() > 1.5  # the blast shell has formed
    assert _relative_face_divergence(final, dx, 2) < 1e-12
    assert _face_average_defect(final, 2) < 1e-14


def test_alfven_wave_converges_at_second_order():
    errors = []
    for n in (8, 16):
        state0, dx = _alfven_wave(n)
        final = _run(state0, _config(3, (3.0, 1.5, 1.5)), t_end=1.0)
        assert _relative_face_divergence(final, dx, 3) < 1e-12
        # The norm of Gardiner & Stone (2008): per conserved variable the mean
        # absolute difference to the initial state, combined in quadrature.
        difference = np.abs(_conserved(final) - _conserved(state0)).mean(axis=(1, 2, 3))
        errors.append(float(np.sqrt((difference**2).sum())))
    # Measured: 6.16e-2 and 1.37e-2 (ratio 4.5).
    assert errors[1] < 2.0e-2
    assert errors[0] / errors[1] > 3.5


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(solver_overrides=dict(time_integrator=RK2_SSP)), "available with the VL2 scheme"),
        (dict(solver_overrides=dict(time_integrator=RK2_SSP), mhd=False), "option of the VL2"),
        (dict(solver_mode=FINITE_DIFFERENCE), "needs an FDSolverConfig"),
        (dict(mhd=False), "requires mhd=True"),
        (
            dict(
                boundary_settings=BoundarySettings(
                    BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
                    BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
                    BoundarySettings1D(),
                )
            ),
            "periodic boundaries on every axis",
        ),
        (dict(gravity_config=GravityConfig(self_gravity=True)), "no gravity coupling"),
        (dict(gravity_config=GravityConfig(external_potential=True)), "no gravity coupling"),
        (dict(solver_overrides=dict(glm_extended_source=True)), "glm_extended_source"),
    ],
)
def test_rejected_where_not_implemented(overrides, message):
    config = _config(2, (1.0, 1.0), **overrides)
    with pytest.raises(ValueError, match=message):
        finalize_config(config, (11, 16, 16))
