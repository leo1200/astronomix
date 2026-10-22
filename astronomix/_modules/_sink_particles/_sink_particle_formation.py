"""
Sink particle formation following Federrath et al. (2010), ApJ 713, 269.

Once per time step, the gas is checked for regions that should turn into sink
particles, using the creation checks of Section 2.2 of the paper. A new sink is
created with zero mass at the centre of its cell and gets its mass from the
accretion step that follows in the same time step
(``_sink_particle_accretion.py``), as in Federrath's FLASH code.

The formation runs in two stages. First, cheap checks are evaluated on the
whole grid, giving a mask of candidate cells. Then, for a fixed-size list of
candidates, the gas in the control volume around each candidate (all cells
within the accretion radius r_acc) is gathered and the remaining checks are
evaluated on it. Candidates that pass every check become new sinks.

The refinement check of Section 2.2.2 is not applied: astronomix uses a
uniform grid, so every cell is already on the highest level of refinement.
"""

# general
import inspect
from functools import partial

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

try:  # the stable alias (newer jax)
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover - older jax
    from jax.experimental.shard_map import shard_map as _shard_map

# The keyword that turns off shard_map's check that an output is the same on
# every device, which it cannot infer through the collectives used below.
_NO_REPLICATION_CHECK = (
    {"check_vma": False}
    if "check_vma" in inspect.signature(_shard_map).parameters
    else {"check_rep": False}
)

# astronomix helpers
from astronomix._pallas_helpers import _current_pallas_mesh

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    IDEAL_GAS,
    PERIODIC_BOUNDARY,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._gravity._gravity import _compute_total_potential
from astronomix._stencil_operations._stencil_operations import _shift


def _raise_sink_slots_full(num_discarded, max_num_sinks):
    """Stop the run when new sinks do not fit in the sink particle slots.

    Called from inside the jitted formation step through
    ``jax.debug.callback``; the raised error aborts the jitted computation.

    Args:
        num_discarded: The number of new sinks without a free slot.
        max_num_sinks: The number of sink particle slots.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError(
        f"{int(num_discarded)} new sink particles cannot be stored: all "
        f"{int(max_num_sinks)} sink particle slots are taken. Increase "
        "max_num_sinks in SinkParticleConfig."
    )


def _empty_sink_particles(
    config: SimulationConfig,
    dtype,
) -> SinkParticles:
    """
    Create sink particle arrays with every slot empty (zero mass).

    Args:
        config: The simulation configuration; supplies the number of slots.
        dtype: The floating-point type of the arrays.

    Returns:
        Sink particles with ``max_num_sinks`` empty slots.
    """
    max_num_sinks = config.sink_particle_config.max_num_sinks
    return SinkParticles(
        mass=jnp.zeros((max_num_sinks,), dtype=dtype),
        position=jnp.zeros((max_num_sinks, 3), dtype=dtype),
        velocity=jnp.zeros((max_num_sinks, 3), dtype=dtype),
    )


def _pad_sink_particles(
    sink_particles: SinkParticles,
    config: SimulationConfig,
) -> SinkParticles:
    """
    Append empty slots (zero mass) to sink particles with fewer slots than
    ``max_num_sinks``, e.g. sinks restored from a checkpoint of a run with
    fewer slots. Filled slots stay at the front.

    Args:
        sink_particles: The sink particles passed into the run.
        config: The simulation configuration; supplies the number of slots.

    Returns:
        Sink particles with ``max_num_sinks`` slots.

    Raises:
        ValueError: If the sink particles have more slots than
            ``max_num_sinks``; dropping slots could drop filled ones.
    """
    max_num_sinks = config.sink_particle_config.max_num_sinks
    num_slots = sink_particles.mass.shape[0]
    if num_slots > max_num_sinks:
        raise ValueError(
            f"{num_slots} sink particle slots passed in, but max_num_sinks is "
            f"{max_num_sinks}. Increase max_num_sinks in SinkParticleConfig."
        )
    num_missing_slots = max_num_sinks - num_slots
    return jax.tree.map(
        lambda field: jnp.pad(
            field,
            [(0, num_missing_slots)] + [(0, 0)] * (field.ndim - 1),
        ),
        sink_particles,
    )


def _control_volume_offsets(config: SimulationConfig) -> np.ndarray:
    """
    Integer cell offsets (i, j, k) of the control volume around a cell.

    The control volume holds every cell whose centre lies within the accretion
    radius of the central cell, i.e. i² + j² + k² ≤ (r_acc / Δx)² (Federrath et
    al. 2010, Eq. 3). The offsets are computed with NumPy at trace time, since
    they only depend on the static configuration.

    Args:
        config: The simulation configuration.

    Returns:
        The offsets, shape (num_offsets, 3), with (0, 0, 0) among them.
    """
    radius_in_cells = config.sink_particle_config.accretion_radius_in_cells
    max_offset = int(np.floor(radius_in_cells))
    offset_range = np.arange(-max_offset, max_offset + 1)
    offsets = np.stack(
        np.meshgrid(offset_range, offset_range, offset_range, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    inside_control_volume = np.sum(offsets**2, axis=-1) <= radius_in_cells**2
    return offsets[inside_control_volume]


def _sound_speed_squared(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """
    The squared sound speed in every cell.

    Args:
        primitive_state: The primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        c_s² = γP/ρ for an ideal gas, or the constant isothermal c_s².
    """
    density = primitive_state[registered_variables.density_index]
    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        return params.gamma * pressure / density
    return jnp.full_like(density, params.isothermal_sound_speed**2)


def _interior_mask(
    density: FIELD_TYPE,
    config: SimulationConfig,
) -> FIELD_TYPE:
    """
    Boolean mask that is True on the physical cells and False on ghost cells.

    Args:
        density: Any (padded) field, used for its shape.
        config: The simulation configuration.

    Returns:
        The interior mask, with the shape of ``density``.
    """
    num_ghost_cells = config.num_ghost_cells
    interior = jnp.zeros(density.shape, dtype=bool)
    interior_slices = tuple(
        slice(num_ghost_cells, size - num_ghost_cells) for size in density.shape
    )
    return interior.at[interior_slices].set(True)


def _converging_flow_mask(velocity: jax.Array) -> FIELD_TYPE:
    """
    Boolean mask of the cells toward which the flow converges along every axis.

    Along each axis d, the neighbour at +1 must not move away from the cell,
    v_d(+1) − v_d(0) ≤ 0, and the neighbour at −1 must not move away either,
    v_d(−1) − v_d(0) ≥ 0 (Federrath et al. 2010, Section 2.2.3). The
    inequalities are not strict, so that gas at rest or in uniform motion
    passes, as in Federrath's FLASH code; only flow away from the cell fails.

    Args:
        velocity: The (padded) velocity field, with the three components on
            the last axis.

    Returns:
        The converging-flow mask, with the shape of one field.
    """
    converging = jnp.ones(velocity.shape[:-1], dtype=bool)
    for axis in range(3):
        velocity_component = velocity[..., axis]
        # _shift(field, -1, axis) holds the value of the neighbour at +1.
        velocity_at_plus_one = _shift(velocity_component, -1, axis)
        velocity_at_minus_one = _shift(velocity_component, 1, axis)
        converging = (
            converging
            & (velocity_at_plus_one - velocity_component <= 0.0)
            & (velocity_at_minus_one - velocity_component >= 0.0)
        )
    return converging


def _potential_minimum_mask(
    gravitational_potential: FIELD_TYPE,
    offsets: np.ndarray,
) -> FIELD_TYPE:
    """
    Boolean mask of the cells where the potential is lowest in their control
    volume.

    A cell passes if its potential is not above the potential of any cell in
    its control volume, φ(0) ≤ min φ (Federrath et al. 2010, Eq. 4).

    Args:
        gravitational_potential: The (padded) gravitational potential.
        offsets: The control-volume offsets, shape (num_offsets, 3).

    Returns:
        The potential-minimum mask, with the shape of the potential.
    """
    minimum_in_volume = gravitational_potential
    for offset in offsets:
        # Shifting by −offset brings the value at cell + offset to each cell.
        potential_at_offset = gravitational_potential
        for axis in range(3):
            if offset[axis] != 0:
                potential_at_offset = _shift(
                    potential_at_offset,
                    -int(offset[axis]),
                    axis,
                )
        minimum_in_volume = jnp.minimum(minimum_in_volume, potential_at_offset)
    return gravitational_potential <= minimum_in_volume


def _periodic_box(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """
    Which axes are periodic, and the box length along each axis.

    Both only depend on the static configuration, so they are computed with
    NumPy at trace time.

    Args:
        config: The simulation configuration.

    Returns:
        ``(is_periodic, box_length)``, two arrays of shape (3,).
    """
    boundary_settings = config.boundary_settings
    is_periodic = np.array(
        [
            axis_settings.left_boundary == PERIODIC_BOUNDARY
            and axis_settings.right_boundary == PERIODIC_BOUNDARY
            for axis_settings in (
                boundary_settings.x,
                boundary_settings.y,
                boundary_settings.z,
            )
        ]
    )
    box_length = np.array(
        [config.box_size.x, config.box_size.y, config.box_size.z]
    )
    return is_periodic, box_length


def _select_separated_candidates(
    candidate_passes: jax.Array,
    candidate_positions: jax.Array,
    potential_at_candidates: jax.Array,
    accretion_radius: float,
    is_periodic: np.ndarray,
    box_length: np.ndarray,
) -> jax.Array:
    """
    Keep only passing candidates that lie farther than r_acc from every other
    kept candidate, so that new sinks formed in the same call respect the
    proximity rule (Section 2.2.7) among themselves.

    The candidates are taken in order of increasing potential, the deepest
    first, with exact ties going to the lower candidate index. Each one is kept
    if it passes and no candidate kept before it lies within r_acc. The pass is
    sequential: a single parallel rule ("drop a candidate if a higher-priority
    passing candidate is within r_acc") would also drop candidates that are
    only near a candidate dropped itself.

    Args:
        candidate_passes: Whether each candidate passes the creation checks,
            shape (num_candidates,).
        candidate_positions: The candidate cell centres, shape
            (num_candidates, 3).
        potential_at_candidates: The gravitational potential at each candidate
            cell, shape (num_candidates,).
        accretion_radius: The accretion radius r_acc.
        is_periodic: Which axes are periodic, shape (3,).
        box_length: The box length along each axis, shape (3,).

    Returns:
        Whether each candidate becomes a sink, shape (num_candidates,).
    """
    # Distances between candidates, to the nearest periodic copy along
    # periodic axes, as in the proximity check to existing sinks.
    separation = candidate_positions[:, None, :] - candidate_positions[None, :, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    within_accretion_radius = (
        jnp.linalg.norm(separation, axis=-1) <= accretion_radius
    )

    # Failing candidates go last; the stable sort keeps exact ties in
    # candidate order.
    order = jnp.argsort(
        jnp.where(candidate_passes, potential_at_candidates, jnp.inf),
        stable=True,
    )

    def keep_if_separated(rank, kept):
        candidate = order[rank]
        near_kept = jnp.any(kept & within_accretion_radius[candidate])
        return kept.at[candidate].set(candidate_passes[candidate] & ~near_kept)

    return jax.lax.fori_loop(
        0,
        candidate_passes.shape[0],
        keep_if_separated,
        jnp.zeros_like(candidate_passes),
    )


def _collect_candidates(
    candidate_mask: jax.Array,
    max_num_candidates: int,
) -> tuple[jax.Array, jax.Array]:
    """
    The first ``max_num_candidates`` candidate cells, in the grid's row-major
    order, and the number of candidate cells.

    On one device this is ``jnp.nonzero`` with a fixed size. When the grid is
    split across devices (a mesh set by ``pallas_mesh_context``, as
    ``time_integration`` does for a sharded run), ``jnp.nonzero`` on the
    global mask would make every device gather the whole mask. Instead, each
    device lists the first ``max_num_candidates`` candidates of its own part
    of the grid, in global indices, and only these short lists are gathered.
    Sorting them in row-major order gives the same list as on one device: a
    cell among the first ``max_num_candidates`` of the grid is also among the
    first ``max_num_candidates`` of its device.

    Args:
        candidate_mask: Boolean mask of the candidate cells, shape of the
            (padded) grid.
        max_num_candidates: The length of the list.

    Returns:
        ``(candidate_indices, num_candidates_found)``: the cell indices, shape
        (max_num_candidates, 3), with unused entries set to 0, and the total
        number of candidate cells.
    """
    grid_shape = candidate_mask.shape

    # The distributed list needs a 4-axis state mesh (variables, x, y, z) that
    # splits the grid evenly; otherwise the mask is searched as a whole.
    mesh = _current_pallas_mesh()
    split = False
    if mesh is not None and len(mesh.axis_names) == 4:
        devices_per_axis = [mesh.shape[name] for name in mesh.axis_names[1:]]
        split = any(num > 1 for num in devices_per_axis) and all(
            size % num == 0 for size, num in zip(grid_shape, devices_per_axis)
        )
    if not split:
        candidate_indices = jnp.stack(
            jnp.nonzero(candidate_mask, size=max_num_candidates, fill_value=0),
            axis=-1,
        )
        return candidate_indices, jnp.sum(candidate_mask)

    # The state mesh may use integer axis names, which JAX's named collectives
    # would read as array axes. Alias the same devices (same order, so no data
    # movement) under string names, as the distributed Poisson solve does.
    axis_names = ("x", "y", "z")
    named_mesh = jax.sharding.Mesh(mesh.devices, ("vars",) + axis_names)

    def _local_list(local_mask):
        # The first candidates of this device's part of the grid, in global
        # indices: local index + device index × local size along each axis.
        local_indices = jnp.stack(
            jnp.nonzero(local_mask, size=max_num_candidates, fill_value=0),
            axis=-1,
        )
        offsets = jnp.stack(
            [
                jax.lax.axis_index(name) * local_mask.shape[axis]
                for axis, name in enumerate(axis_names)
            ]
        )
        local_count = jnp.sum(local_mask)
        local_is_valid = jnp.arange(max_num_candidates) < local_count

        # Gather the short lists of all devices, shape
        # (num_devices × max_num_candidates, ...), and the total count.
        indices = jax.lax.all_gather(
            local_indices + offsets, axis_names, tiled=True
        )
        is_valid = jax.lax.all_gather(local_is_valid, axis_names, tiled=True)
        num_candidates_found = jax.lax.psum(local_count, axis_names)

        # Valid entries first, in row-major order (x, then y, then z). A
        # lexicographic sort avoids a flat index, which overflows int32 on
        # large grids.
        order = jnp.lexsort(
            (indices[:, 2], indices[:, 1], indices[:, 0], ~is_valid)
        )[:max_num_candidates]
        candidate_indices = jnp.where(
            is_valid[order, None],
            indices[order],
            0,
        )
        return candidate_indices, num_candidates_found

    return _shard_map(
        _local_list,
        mesh=named_mesh,
        in_specs=(PartitionSpec(*axis_names),),
        out_specs=(PartitionSpec(), PartitionSpec()),
        **_NO_REPLICATION_CHECK,
    )(candidate_mask)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _form_sink_particles(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> tuple[SinkParticles, jax.Array]:
    """
    Create new sink particles where the gas passes all creation checks.

    A new sink is placed at the centre of its cell with zero mass, and with the
    centre-of-mass velocity (Eq. 12) of the gas in its control volume. It gets
    its mass from the accretion step that follows. The starting velocity only
    sets the frame of the first accretion checks; FLASH starts from zero
    velocity, which would judge a clump moving as a whole in a different frame
    than the creation checks below.

    Args:
        primitive_state: The (padded) primitive state after the hydro update.
        sink_particles: The current sink particles.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(sink_particles, num_active_sinks)``: the sink particles with the new
        sinks appended, and the number of filled slots including the new
        (still massless) sinks.
    """

    sink_particle_config = config.sink_particle_config
    grid_spacing = config.grid_spacing
    cell_volume = grid_spacing**3
    accretion_radius = sink_particle_config.accretion_radius_in_cells * grid_spacing

    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [
            primitive_state[registered_variables.velocity_index.x],
            primitive_state[registered_variables.velocity_index.y],
            primitive_state[registered_variables.velocity_index.z],
        ],
        axis=-1,
    )

    # The density threshold is the density at which the Jeans length equals
    # 2 r_acc, the smallest Jeans length the grid resolves (Federrath et al.
    # 2010, Eq. 32). It depends on the local sound speed, so it is evaluated
    # in every cell.
    sound_speed_squared = _sound_speed_squared(
        primitive_state,
        config,
        params,
        registered_variables,
    )
    density_threshold = (
        jnp.pi
        * sound_speed_squared
        / (4.0 * params.gravitational_constant * accretion_radius**2)
    )

    # The gravitational potential of the gas (plus any external potential),
    # from a Poisson solve on the updated density. The hydro update computes it
    # internally but does not return it, and for intermediate states only.
    gravitational_potential = _compute_total_potential(
        density,
        grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )
    offsets = _control_volume_offsets(config)

    # -------------------------------------------------------------
    # =============== ↓ Stage 1: grid-wide checks ↓ ===============
    # -------------------------------------------------------------

    # Density threshold check (Section 2.2.1), converging flow check
    # (Section 2.2.3) and gravitational potential minimum check (Section
    # 2.2.4). Ghost cells are excluded, as they only mirror physical cells or
    # hold boundary values.
    candidate_mask = (
        (density > density_threshold)
        & _converging_flow_mask(velocity)
        & _potential_minimum_mask(gravitational_potential, offsets)
        & _interior_mask(density, config)
    )

    # -------------------------------------------------------------
    # =============== ↑ Stage 1: grid-wide checks ↑ ===============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =========== ↓ Stage 2: control-volume checks ↓ ==============
    # -------------------------------------------------------------

    # JIT needs fixed array sizes, so the candidate cells are collected into a
    # list of fixed length; unused entries are marked as invalid.
    max_num_candidates = sink_particle_config.max_num_candidates
    candidate_indices, num_candidates_found = _collect_candidates(
        candidate_mask,
        max_num_candidates,
    )
    candidate_is_valid = jnp.arange(max_num_candidates) < num_candidates_found

    # Cell indices of every candidate's control volume. The indices wrap
    # around the grid, which is the right neighbourhood for periodic
    # boundaries and never happens for ghost-cell boundaries, where the ghost
    # layer is wider than the control volume.
    grid_shape = np.array(density.shape)
    control_volume_indices = tuple(
        (candidate_indices[:, None, axis] + offsets[None, :, axis]) % grid_shape[axis]
        for axis in range(3)
    )

    # Fields over the control volumes, shape (num_candidates, num_offsets),
    # with a trailing vector axis for the velocity.
    density_in_volume = density[control_volume_indices]
    velocity_in_volume = velocity[control_volume_indices]
    cell_mass_in_volume = density_in_volume * cell_volume
    gas_mass_in_volume = jnp.sum(cell_mass_in_volume, axis=-1)

    # Centre of the candidate cells, and the centre-of-mass velocity of the
    # gas in their control volumes (Eq. 12).
    candidate_positions = (candidate_indices - config.num_ghost_cells + 0.5) * grid_spacing
    center_of_mass_velocity = jnp.einsum(
        "ck,ckd->cd",
        cell_mass_in_volume,
        velocity_in_volume,
    ) / gas_mass_in_volume[:, None]

    # Proximity check (Section 2.2.7): no new sink within r_acc of an existing
    # sink. Along periodic axes the separation is taken to the nearest
    # periodic copy of the sink (minimum-image convention), as in Federrath's
    # FLASH code. Only filled slots are compared: empty slots have zero mass
    # and sit at the origin, where they would otherwise block candidates.
    is_periodic, box_length = _periodic_box(config)
    separation = candidate_positions[:, None, :] - sink_particles.position[None, :, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    distance_to_sink = jnp.linalg.norm(separation, axis=-1)
    sink_is_filled = sink_particles.mass > 0.0
    sink_within_accretion_radius = jnp.logical_and(
        distance_to_sink <= accretion_radius,
        sink_is_filled[None, :],
    )
    far_from_existing_sinks = jnp.logical_not(
        jnp.any(sink_within_accretion_radius, axis=-1)
    )

    # Jeans instability check (Section 2.2.5): the gravitational energy of the
    # gas in the control volume must exceed twice its thermal energy plus its
    # magnetic energy, |E_grav| > 2 E_th + E_mag.
    #   - E_th = ½ Σ M c_s² (Eq. 5).
    #   - E_grav = −Σ M (φ_max − φ), with φ_max the largest potential in the
    #     control volume, as in Federrath's FLASH code. Measuring φ from its
    #     maximum removes the part of the potential that is constant across
    #     the volume, which the gas outside the control volume contributes.
    #   - E_mag = ½ Σ |B|² ΔV (Eq. 9, with the magnetic energy density ½ B²
    #     of astronomix's field units instead of the paper's B² / 8π).
    sound_speed_squared_in_volume = sound_speed_squared[control_volume_indices]
    thermal_energy = 0.5 * jnp.sum(
        cell_mass_in_volume * sound_speed_squared_in_volume,
        axis=-1,
    )

    potential_in_volume = gravitational_potential[control_volume_indices]
    maximum_potential_in_volume = jnp.max(potential_in_volume, axis=-1, keepdims=True)
    gravitational_energy_magnitude = jnp.sum(
        cell_mass_in_volume * (maximum_potential_in_volume - potential_in_volume),
        axis=-1,
    )

    if config.mhd:
        magnetic_field_squared = (
            primitive_state[registered_variables.magnetic_index.x] ** 2
            + primitive_state[registered_variables.magnetic_index.y] ** 2
            + primitive_state[registered_variables.magnetic_index.z] ** 2
        )
        magnetic_energy = 0.5 * jnp.sum(
            magnetic_field_squared[control_volume_indices] * cell_volume,
            axis=-1,
        )
    else:
        magnetic_energy = jnp.zeros_like(thermal_energy)

    jeans_unstable = (
        gravitational_energy_magnitude > 2.0 * thermal_energy + magnetic_energy
    )

    # Bound state check (Section 2.2.6): the total energy of the gas in the
    # control volume must be negative, E_grav + E_th + E_kin + E_mag < 0, with
    # E_kin = ½ Σ M |v − v_cm|² the kinetic energy relative to the
    # centre-of-mass motion (Eq. 11).
    velocity_relative_to_center_of_mass = (
        velocity_in_volume - center_of_mass_velocity[:, None, :]
    )
    kinetic_energy = 0.5 * jnp.sum(
        cell_mass_in_volume
        * jnp.sum(velocity_relative_to_center_of_mass**2, axis=-1),
        axis=-1,
    )
    bound = (
        -gravitational_energy_magnitude
        + thermal_energy
        + kinetic_energy
        + magnetic_energy
        < 0.0
    )

    candidate_passes = (
        candidate_is_valid
        & far_from_existing_sinks
        & jeans_unstable
        & bound
    )

    # Proximity among the new sinks (Section 2.2.7): several nearby cells can
    # pass every check in the same call, e.g. when the potential minimum is
    # shared by equivalent cells (the potential check is not strict). Only
    # candidates at least r_acc apart are kept, the deepest potential first.
    candidate_passes = _select_separated_candidates(
        candidate_passes,
        candidate_positions,
        gravitational_potential[tuple(candidate_indices.T)],
        accretion_radius,
        is_periodic,
        box_length,
    )

    # -------------------------------------------------------------
    # =========== ↑ Stage 2: control-volume checks ↑ ==============
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ Appending new sinks ↓ ===================
    # -------------------------------------------------------------

    # Sinks are only appended, so the filled slots are contiguous at the
    # front. Each passing candidate gets the next free slot; failing
    # candidates get the out-of-range slot max_num_sinks. With mode="drop",
    # writes to an out-of-range slot are discarded, which also discards new
    # sinks once every slot is taken. New sinks are massless until the
    # accretion step, so the filled slots are counted here and passed on.
    max_num_sinks = sink_particle_config.max_num_sinks
    num_existing_sinks = jnp.sum(sink_particles.mass > 0.0)
    slot = jnp.where(
        candidate_passes,
        num_existing_sinks + jnp.cumsum(candidate_passes) - 1,
        max_num_sinks,
    )

    sink_particles = SinkParticles(
        mass=sink_particles.mass.at[slot].set(0.0, mode="drop"),
        position=sink_particles.position.at[slot].set(
            candidate_positions,
            mode="drop",
        ),
        velocity=sink_particles.velocity.at[slot].set(
            center_of_mass_velocity,
            mode="drop",
        ),
    )
    num_new_sinks = jnp.sum(candidate_passes)
    num_active_sinks = jnp.minimum(num_existing_sinks + num_new_sinks, max_num_sinks)

    # Candidates beyond max_num_candidates are only skipped for this step:
    # they are still above the threshold at the next step and are checked
    # then, so a warning is enough. A new sink without a free slot would be
    # lost, leaving its gas above the threshold on the grid, so the run stops
    # with an error instead (a host callback that raises aborts the jitted
    # computation).
    jax.lax.cond(
        num_candidates_found > max_num_candidates,
        lambda: jax.debug.print(
            "WARNING: {} sink particle candidates found, only the first {} "
            "are checked this step (increase max_num_candidates).",
            num_candidates_found,
            max_num_candidates,
        ),
        lambda: None,
    )
    jax.lax.cond(
        num_existing_sinks + num_new_sinks > max_num_sinks,
        lambda: jax.debug.callback(
            _raise_sink_slots_full,
            num_existing_sinks + num_new_sinks - max_num_sinks,
            max_num_sinks,
        ),
        lambda: None,
    )

    # -------------------------------------------------------------
    # ================= ↑ Appending new sinks ↑ ===================
    # -------------------------------------------------------------

    return sink_particles, num_active_sinks
