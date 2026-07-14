"""Unit tests for the Lagrangian tracer module (astronomix._modules._tracers).

Run with:  python -m pytest pytests/test_tracers.py
These are CPU-friendly and need no GPU.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
)
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix._modules._tracers._tracer_options import TracerConfig
from astronomix._modules._tracers._tracers import (
    _interior_slice,
    advance_tracers,
    interpolate_field,
    recycle_tracers,
    regenerate_tracers,
    sample_tracer_temperature,
    seed_tracers,
)


def _make_config(num_cells, box=1.0, periodic_z=True, reinject=True):
    """A finalized 3D FD config; z periodic or open per ``periodic_z``."""
    z_boundary = (
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
        if periodic_z
        else BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY)
    )
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        box_size=StaticFloatVector(box, box, box),
        num_cells=StaticIntVector(num_cells, num_cells, num_cells),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=z_boundary,
        ),
        tracer_config=TracerConfig(tracers=True, num_tracers=1000, reinject=reinject),
    )
    registered_variables = get_registered_variables(config)
    state_shape = (registered_variables.num_vars, num_cells, num_cells, num_cells)
    config = finalize_config(config, state_shape)
    return config, registered_variables


def _padded_state_from_interior(interior_fields, config, registered_variables):
    """Build a padded primitive state from interior (Nx,Ny,Nz) component fields.

    ``interior_fields`` maps a variable index to its interior array. Ghost cells
    are filled by edge-padding (their exact values are irrelevant for the tests,
    which keep tracers in the interior)."""
    ngc = config.num_ghost_cells
    num_cells = config.num_cells.x
    num_vars = registered_variables.num_vars
    padded = jnp.zeros((num_vars, num_cells + 2 * ngc, num_cells + 2 * ngc, num_cells + 2 * ngc))
    for index, field in interior_fields.items():
        padded_field = jnp.pad(
            field, ((ngc, ngc), (ngc, ngc), (ngc, ngc)), mode="edge"
        )
        padded = padded.at[index].set(padded_field)
    return padded


def test_interpolate_linear_is_exact():
    """Trilinear interpolation reproduces a linear field exactly."""
    config, _ = _make_config(16, box=1.0)
    dx = config.grid_spacing
    n = config.num_cells.x
    axis = (jnp.arange(n) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    a, b, c, d = 2.0, -3.0, 0.7, 1.5
    field = a * X + b * Y + c * Z + d

    key = jax.random.PRNGKey(0)
    # sample well inside [dx, L-dx] to avoid boundary wrap/clamp effects
    points = jax.random.uniform(
        key, (200, 3), minval=2 * dx, maxval=1.0 - 2 * dx
    )
    interpolated = interpolate_field(field, points, config)
    expected = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    assert jnp.allclose(interpolated, expected, atol=1e-10)


def test_advection_uniform_flow():
    """In a uniform velocity field tracers translate by exactly v*dt (mod L)."""
    config, registered_variables = _make_config(16, box=1.0)
    n = config.num_cells.x
    vx_val, vy_val, vz_val = 0.3, -0.5, 0.2
    ones = jnp.ones((n, n, n))
    interior = {
        registered_variables.density_index: ones,
        registered_variables.pressure_index: ones,
        registered_variables.velocity_index.x: vx_val * ones,
        registered_variables.velocity_index.y: vy_val * ones,
        registered_variables.velocity_index.z: vz_val * ones,
    }
    padded = _padded_state_from_interior(interior, config, registered_variables)

    key = jax.random.PRNGKey(1)
    positions = jax.random.uniform(key, (500, 3), minval=0.2, maxval=0.8)
    dt = 0.05
    new_positions = advance_tracers(positions, padded, dt, config, registered_variables)

    velocity = jnp.array([vx_val, vy_val, vz_val])
    expected = jnp.mod(positions + dt * velocity, 1.0)
    assert jnp.allclose(new_positions, expected, atol=1e-10)


def test_sample_temperature_matches_field():
    """Sampling temperature returns the interpolated P/rho field."""
    config, registered_variables = _make_config(16, box=1.0)
    dx = config.grid_spacing
    n = config.num_cells.x
    axis = (jnp.arange(n) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    density = 1.0 + 0.5 * X        # smoothly varying, all positive
    pressure = 2.0 + 0.3 * Y
    interior = {
        registered_variables.density_index: density,
        registered_variables.pressure_index: pressure,
        registered_variables.velocity_index.x: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.y: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.z: jnp.zeros((n, n, n)),
    }
    padded = _padded_state_from_interior(interior, config, registered_variables)

    key = jax.random.PRNGKey(2)
    points = jax.random.uniform(key, (200, 3), minval=2 * dx, maxval=1.0 - 2 * dx)
    sampled = sample_tracer_temperature(points, padded, config, registered_variables)
    # field P/rho is not linear, so compare against the same trilinear interp of T
    temperature_field = pressure / density
    expected = interpolate_field(temperature_field, points, config)
    assert jnp.allclose(sampled, expected, atol=1e-12)


def test_seed_mass_weighted_distribution():
    """Mass-weighted seeding draws cells with probability proportional to rho."""
    config, _ = _make_config(8, box=1.0)
    n = config.num_cells.x
    # half the box (z > 0.5) is 10x denser; expected mass fraction there = 10/11
    density = jnp.ones((n, n, n))
    half = n // 2
    density = density.at[:, :, half:].set(10.0)

    key = jax.random.PRNGKey(3)
    num = 40000
    positions = seed_tracers(key, density, config, num)
    fraction_upper = jnp.mean(positions[:, 2] > 0.5)
    expected = 10.0 / 11.0
    assert abs(float(fraction_upper) - expected) < 0.02


def _upward_flow_state(config, registered_variables):
    n = config.num_cells.x
    ones = jnp.ones((n, n, n))
    interior = {
        registered_variables.density_index: ones,
        registered_variables.pressure_index: ones,
        registered_variables.velocity_index.x: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.y: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.z: 5.0 * ones,  # strong upward flow
    }
    return _padded_state_from_interior(interior, config, registered_variables)


def test_clamp_at_open_boundary():
    """With reinject=False, an upward flow clamps tracers at the top (z=L)."""
    config, registered_variables = _make_config(
        16, box=1.0, periodic_z=False, reinject=False
    )
    padded = _upward_flow_state(config, registered_variables)
    positions = jnp.array([[0.5, 0.5, 0.9]])
    new_positions = advance_tracers(positions, padded, 1.0, config, registered_variables)
    assert float(new_positions[0, 2]) <= 1.0 + 1e-12
    assert abs(float(new_positions[0, 2]) - 1.0) < 1e-9


def test_reinject_at_open_boundary():
    """With reinject=True, a tracer leaving the open top re-appears just inside
    the high (inflow) end at L - dx/2 (not clamped at the boundary)."""
    config, registered_variables = _make_config(
        16, box=1.0, periodic_z=False, reinject=True
    )
    padded = _upward_flow_state(config, registered_variables)
    dx = config.grid_spacing
    positions = jnp.array([[0.5, 0.5, 0.9]])
    new_positions = advance_tracers(positions, padded, 1.0, config, registered_variables)
    assert abs(float(new_positions[0, 2]) - (1.0 - 0.5 * dx)) < 1e-9


def test_recycle_conserves_count_and_injects_at_top():
    """Flux-matched recycling conserves the tracer count and relocates the
    recycled tracers into the top inflow cell."""
    config, registered_variables = _make_config(16, box=1.0, periodic_z=False)
    n = config.num_cells.x
    ones = jnp.ones((n, n, n))
    # strong downward inflow in the top interior cell -> non-zero inflow flux
    vz = jnp.zeros((n, n, n)).at[:, :, -1].set(-2.0)
    interior = {
        registered_variables.density_index: ones,
        registered_variables.pressure_index: ones,
        registered_variables.velocity_index.x: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.y: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.z: vz,
    }
    padded = _padded_state_from_interior(interior, config, registered_variables)
    key = jax.random.PRNGKey(7)
    positions = jax.random.uniform(jax.random.PRNGKey(8), (4000, 3), minval=0.05, maxval=0.95)
    _, new_positions = recycle_tracers(
        positions, key, padded, 0.05, config, registered_variables
    )
    assert new_positions.shape == positions.shape
    moved = jnp.any(jnp.abs(new_positions - positions) > 1e-9, axis=1)
    assert int(moved.sum()) > 0                       # some recycling happened
    dx = config.grid_spacing
    # all relocated tracers sit in the top inflow cell [L - dx, L)
    assert bool(jnp.all(new_positions[moved, 2] >= 1.0 - dx - 1e-9))


def test_regenerate_thermostat():
    """Regeneration relocates ~dt/t_relax of tracers to fresh ∝rho positions and
    bumps the generation counter exactly for those tracers."""
    config, registered_variables = _make_config(16, box=1.0)
    config = config._replace(
        tracer_config=config.tracer_config._replace(
            regenerate=True, regenerate_timescale=1.0
        )
    )
    n = config.num_cells.x
    ones = jnp.ones((n, n, n))
    interior = {
        registered_variables.density_index: ones,
        registered_variables.pressure_index: ones,
        registered_variables.velocity_index.x: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.y: jnp.zeros((n, n, n)),
        registered_variables.velocity_index.z: jnp.zeros((n, n, n)),
    }
    padded = _padded_state_from_interior(interior, config, registered_variables)
    num = config.tracer_config.num_tracers
    positions = jax.random.uniform(jax.random.PRNGKey(11), (num, 3), maxval=jnp.array([1.0, 1.0, 1.0]))
    generation = jnp.zeros(num, dtype=jnp.int32)
    dt = 0.25  # expect ~25% regenerated (dt / regenerate_timescale)
    _, new_pos, new_gen = regenerate_tracers(
        positions, generation, jax.random.PRNGKey(12), padded, dt, config, registered_variables
    )
    moved = jnp.any(jnp.abs(new_pos - positions) > 1e-12, axis=1)
    fraction = float(jnp.mean(moved))
    assert 0.18 < fraction < 0.32                      # ~0.25
    # generation bumped exactly for moved tracers
    assert bool(jnp.all((new_gen - generation) == moved.astype(jnp.int32)))


if __name__ == "__main__":
    # Allow running without pytest installed: execute every test_* function.
    tests = {name: obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)}
    failures = 0
    for name, fn in tests.items():
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
