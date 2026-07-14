"""Lagrangian tracer particles.

Massless points advected by the (interpolated) fluid velocity. They sample
fluid quantities along Lagrangian trajectories without back-reacting on the
flow. Used by the Fokker-Planck / stochastic-temperature analysis of turbulent
radiative mixing layers, which needs temperature increments measured *along
particle paths* (transition statistics) rather than on the Eulerian grid.

Conventions
-----------
* Positions are physical coordinates with shape ``(num_tracers, dim)``.
* Interpolation is multilinear and operates on the **interior** (unpadded)
  field. The fractional interior index of a physical coordinate ``x`` is
  ``x / dx - 0.5`` (cell centers sit at ``(i + 0.5) * dx``). Per-axis boundary
  behaviour is handled explicitly here (periodic wrap vs clamp), so the result
  does not depend on the contents of the ghost cells.
* Single-GPU only: the gather reads the whole field.
"""

import itertools
from functools import partial

import jax
import jax.numpy as jnp

from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    SimulationConfig,
)
from astronomix.variable_registry.registered_variables import RegisteredVariables

from astronomix._modules._tracers._tracer_options import EULER, MASS_WEIGHTED, RK2


def _axis_periodic(config: SimulationConfig):
    """Per-axis booleans: an axis is periodic iff both its ends are periodic."""
    boundary_settings = config.boundary_settings
    axes = (boundary_settings.x, boundary_settings.y, boundary_settings.z)
    periodic = tuple(
        (axis.left_boundary == PERIODIC_BOUNDARY)
        and (axis.right_boundary == PERIODIC_BOUNDARY)
        for axis in axes
    )
    return periodic[: config.dimensionality]


def _box_size(config: SimulationConfig):
    """Per-axis physical box size (tuple of length ``dim``)."""
    box = (config.box_size.x, config.box_size.y, config.box_size.z)
    return box[: config.dimensionality]


def interpolate_field(field_interior, positions, config: SimulationConfig):
    """Multilinear interpolation of a scalar ``field_interior`` at ``positions``.

    Args:
        field_interior: Unpadded scalar field, shape ``(Nx, Ny[, Nz])``.
        positions: Physical coordinates, shape ``(num_tracers, dim)``.
        config: Simulation configuration (grid spacing, dimensionality,
            per-axis boundary settings).

    Returns:
        Interpolated values, shape ``(num_tracers,)``.
    """
    dx = config.grid_spacing
    dim = config.dimensionality
    periodic = _axis_periodic(config)
    shape = field_interior.shape

    # fractional interior cell-center index per axis, shape (num_tracers, dim)
    fractional = positions / dx - 0.5
    lower = jnp.floor(fractional)
    frac = fractional - lower
    lower = lower.astype(jnp.int32)

    result = jnp.zeros(positions.shape[0], dtype=field_interior.dtype)
    # sum over the 2**dim corners of the cell containing each particle
    for corner in itertools.product((0, 1), repeat=dim):
        gather_index = []
        weight = jnp.ones(positions.shape[0], dtype=field_interior.dtype)
        for axis in range(dim):
            index_axis = lower[:, axis] + corner[axis]
            if periodic[axis]:
                index_axis = jnp.mod(index_axis, shape[axis])
            else:
                index_axis = jnp.clip(index_axis, 0, shape[axis] - 1)
            gather_index.append(index_axis)
            weight_axis = frac[:, axis] if corner[axis] == 1 else (1.0 - frac[:, axis])
            weight = weight * weight_axis
        result = result + weight * field_interior[tuple(gather_index)]

    return result


def _interior_slice(config: SimulationConfig):
    """Slice that strips ghost cells from the leading ``dim`` spatial axes."""
    ngc = config.num_ghost_cells
    if ngc == 0:
        return tuple(slice(None) for _ in range(config.dimensionality))
    return tuple(slice(ngc, -ngc) for _ in range(config.dimensionality))


def _velocity_interior(primitive_state, config: SimulationConfig, registered_variables):
    """Return the interior velocity components as a list of length ``dim``."""
    velocity_index = registered_variables.velocity_index
    component_indices = (velocity_index.x, velocity_index.y, velocity_index.z)[
        : config.dimensionality
    ]
    spatial = _interior_slice(config)
    return [primitive_state[index][spatial] for index in component_indices]


def temperature_interior(primitive_state, config: SimulationConfig, registered_variables):
    """Interior temperature field ``T = P / rho`` (ASSUMING ideal gas T = P/rho)."""
    spatial = _interior_slice(config)
    pressure = primitive_state[registered_variables.pressure_index][spatial]
    density = primitive_state[registered_variables.density_index][spatial]
    return pressure / density


def apply_boundary(positions, config: SimulationConfig, reinject: bool = False):
    """Map positions back into the box.

    Periodic axes wrap. Non-periodic axes either clamp to ``[0, L]`` or, when
    ``reinject`` is set, re-inject particles that have left the domain (on
    *either* side) at the high end of the axis (the inflow side) — a fixed point
    just inside ``L``. Re-injection is only applied to the *final* position
    update; the RK2 midpoint and the seeding use the plain wrap/clamp form so an
    intermediate excursion never teleports a particle.
    """
    periodic = _axis_periodic(config)
    box = _box_size(config)
    grid_spacing = config.grid_spacing
    columns = []
    for axis in range(config.dimensionality):
        coordinate = positions[:, axis]
        if periodic[axis]:
            coordinate = jnp.mod(coordinate, box[axis])
        elif reinject and config.tracer_config.reinject:
            exited = (coordinate < 0.0) | (coordinate > box[axis])
            inject_point = box[axis] - 0.5 * grid_spacing
            coordinate = jnp.where(
                exited, inject_point, jnp.clip(coordinate, 0.0, box[axis])
            )
        else:
            coordinate = jnp.clip(coordinate, 0.0, box[axis])
        columns.append(coordinate)
    return jnp.stack(columns, axis=-1)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def advance_tracers(
    positions,
    primitive_state: STATE_TYPE,
    dt,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Advance tracer positions by one step using the (padded) velocity field.

    Uses the end-of-step velocity field for an RK2-in-space (midpoint) update,
    or forward Euler. The velocity is interpolated on the interior field, so
    ghost-cell contents are irrelevant. Periodic axes wrap; other axes clamp.

    Args:
        positions: Tracer positions, shape ``(num_tracers, dim)``.
        primitive_state: The padded primitive state held in the loop carry.
        dt: Time step.
        config: Simulation configuration.
        registered_variables: Variable index registry.

    Returns:
        Updated tracer positions, shape ``(num_tracers, dim)``.
    """
    velocity_components = _velocity_interior(primitive_state, config, registered_variables)

    def velocity_at(points):
        columns = [
            interpolate_field(component, points, config)
            for component in velocity_components
        ]
        return jnp.stack(columns, axis=-1)

    k1 = velocity_at(positions)
    if config.tracer_config.integrator == EULER:
        positions = positions + dt * k1
    else:  # RK2
        midpoint = apply_boundary(positions + 0.5 * dt * k1, config)
        k2 = velocity_at(midpoint)
        positions = positions + dt * k2

    return apply_boundary(positions, config, reinject=True)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def recycle_tracers(
    positions,
    key,
    primitive_state: STATE_TYPE,
    dt,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Flux-matched boundary recycling for the open-through-flow TRML (3D, +z inflow).

    A fixed mass-seeded tracer set keeps its initial weighting: mass flowing in
    at the hot top boundary is never tracered, and (with frame tracking) cold
    tracers drift up and vacate the bottom outflow, so the ensemble never relaxes
    to the steady-state mass distribution. This relocates, each step, a number of
    tracers matched to the measured top-inflow mass flux to the hot inflow layer
    (random in-plane, within the top z-cell). The randomly chosen removed tracers
    stand in for the cold gas cycling out. Net effect: the Lagrangian temperature
    marginal tracks the steady-state mass PDF (Eulerian mass-weighted), which is
    the marginal the Fokker-Planck reconstruction is tested against.

    Returns ``(advanced_key, new_positions)``. A relocated tracer teleports, so
    increments spanning a recycle are dropped in the analysis (detected from the
    recorded positions).
    """
    dim = config.dimensionality
    dx = config.grid_spacing
    box_z = config.box_size.z
    num_tracers = positions.shape[0]

    spatial = _interior_slice(config)
    density = primitive_state[registered_variables.density_index][spatial]
    velocity_z = primitive_state[registered_variables.velocity_index.z][spatial]

    total_mass = jnp.sum(density) * dx ** dim
    mass_per_tracer = total_mass / num_tracers

    # inflow mass rate at the top z-boundary cell (gas entering downward, v_z<0)
    top_density = density[..., -1]
    top_inflow_speed = jnp.maximum(-velocity_z[..., -1], 0.0)
    inflow_mass_rate = jnp.sum(top_density * top_inflow_speed) * dx ** (dim - 1)

    number_to_recycle = inflow_mass_rate * dt / mass_per_tracer
    recycle_probability = jnp.clip(number_to_recycle / num_tracers, 0.0, 1.0)

    key, key_select, key_x, key_y, key_z = jax.random.split(key, 5)
    recycle_mask = jax.random.uniform(key_select, (num_tracers,)) < recycle_probability

    new_x = jax.random.uniform(key_x, (num_tracers,), maxval=config.box_size.x)
    new_y = jax.random.uniform(key_y, (num_tracers,), maxval=config.box_size.y)
    # within the top inflow cell: z in [box_z - dx, box_z)
    new_z = box_z - jax.random.uniform(key_z, (num_tracers,), maxval=dx)

    positions = positions.at[:, 0].set(jnp.where(recycle_mask, new_x, positions[:, 0]))
    positions = positions.at[:, 1].set(jnp.where(recycle_mask, new_y, positions[:, 1]))
    positions = positions.at[:, dim - 1].set(
        jnp.where(recycle_mask, new_z, positions[:, dim - 1])
    )
    return key, positions


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def regenerate_tracers(
    positions,
    generation,
    key,
    primitive_state: STATE_TYPE,
    dt,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Regeneration thermostat: re-draw a fraction of tracers ∝ current density.

    A fixed mass-seeded tracer set drifts away from the instantaneous mass
    distribution where the flow turns over faster than tracers can follow and is
    fed by un-tracered boundary inflow (the thin, fast-cooling mixing layer).
    Each step a fraction ``dt / regenerate_timescale`` of tracers is relocated to
    a fresh position drawn ∝ the current density, which pins the Lagrangian
    marginal to the instantaneous mass distribution. The timescale should stay
    well above the increment lags so the (genuine, unbiased) trajectory segments
    between regenerations remain long enough to measure A(T), D(T).

    A regenerated tracer's increment is not a real Lagrangian increment, so each
    regeneration bumps a per-tracer ``generation`` counter (recorded per
    snapshot): the analysis drops any increment whose endpoints differ in
    generation. A position-jump heuristic is *not* reliable here — a regeneration
    can land in a nearby ∝ρ cell, and even a rare undetected one injects a large
    spurious δT that swamps the tiny cold-phase diffusion.

    Returns ``(advanced_key, new_positions, new_generation)``.
    """
    spatial = _interior_slice(config)
    density = primitive_state[registered_variables.density_index][spatial]
    num_tracers = positions.shape[0]

    key, key_candidate, key_select = jax.random.split(key, 3)
    # fresh positions ∝ the current density for every tracer ...
    candidates = seed_tracers(key_candidate, density, config, num_tracers)
    # ... but only swap in the regenerated fraction
    probability = jnp.clip(dt / config.tracer_config.regenerate_timescale, 0.0, 1.0)
    regenerate = jax.random.uniform(key_select, (num_tracers,)) < probability
    positions = jnp.where(regenerate[:, None], candidates, positions)
    generation = generation + regenerate.astype(generation.dtype)
    return key, positions, generation


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def sample_tracer_temperature(
    positions,
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Interpolated temperature ``T = P / rho`` at the tracer positions."""
    temperature = temperature_interior(primitive_state, config, registered_variables)
    return interpolate_field(temperature, positions, config)


@partial(jax.jit, static_argnames=["config", "num_tracers"])
def seed_tracers(key, density_interior, config: SimulationConfig, num_tracers: int):
    """Draw initial tracer positions from the interior density field.

    ``MASS_WEIGHTED`` samples cells with probability proportional to their mass
    (``rho`` times the cell volume; Cartesian cells share a volume, so
    proportional to ``rho``); ``UNIFORM`` samples cells with equal probability.
    The chosen cell is then given a uniform sub-cell jitter so positions are not
    pinned to cell centers.

    Args:
        key: PRNG key.
        density_interior: Unpadded density field, shape ``(Nx, Ny[, Nz])``.
        config: Simulation configuration.
        num_tracers: Number of tracer particles to seed.

    Returns:
        Tracer positions, shape ``(num_tracers, dim)``.
    """
    dx = config.grid_spacing
    dim = config.dimensionality
    shape = density_interior.shape

    flat_density = density_interior.reshape(-1)
    if config.tracer_config.seed_mode == MASS_WEIGHTED:
        probability = flat_density / jnp.sum(flat_density)
    else:  # UNIFORM
        probability = jnp.full(flat_density.shape, 1.0 / flat_density.size)

    key_cell, key_jitter = jax.random.split(key)
    flat_index = jax.random.choice(
        key_cell, flat_density.size, shape=(num_tracers,), p=probability
    )
    multi_index = jnp.stack(jnp.unravel_index(flat_index, shape), axis=-1)  # (N_p, dim)

    cell_centers = (multi_index + 0.5) * dx
    jitter = jax.random.uniform(
        key_jitter, (num_tracers, dim), minval=-0.5, maxval=0.5
    )
    positions = cell_centers + jitter * dx
    return apply_boundary(positions, config)
