"""
Gas accretion onto sink particles following Federrath et al. (2010), ApJ 713, 269.

Once per time step, after sink creation, every sink takes the gas above the
density threshold from the cells within its accretion radius (Section 2.3):
from a cell with ρ > ρ_res, the mass increment ΔM = (ρ − ρ_res) ΔV (Eq. 13)
moves to the sink and the cell is left at ρ_res. Mass and linear momentum are
conserved (Appendix B); the sink moves to the centre of mass of itself and the
accreted gas. New sinks, created with zero mass, get their mass here too.

The work is done per sink rather than per cell: each sink gathers the few cells
around it, so the cost scales with the number of sinks, not with the grid.
"""

# general
from functools import partial

# numerics
import numpy as np

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_state_struct import SinkParticles
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._modules._sink_particles._sink_particle_formation import (
    _periodic_box,
    _sound_speed_squared,
)


def _accretion_offsets(config: SimulationConfig) -> np.ndarray:
    """
    Integer cell offsets (i, j, k), from the cell containing a sink, of every
    cell that can lie within the accretion radius of that sink.

    A sink can sit anywhere inside its cell, at most √3/2 cells from the cell
    centre, so the offsets are those with i² + j² + k² ≤ (r_acc/Δx + √3/2)².
    Whether a cell is actually within r_acc is checked per sink. The offsets
    are computed with NumPy at trace time, since they only depend on the
    static configuration.

    Args:
        config: The simulation configuration.

    Returns:
        The offsets, shape (num_offsets, 3), with (0, 0, 0) among them.
    """
    reach_in_cells = (
        config.sink_particle_config.accretion_radius_in_cells + 0.5 * np.sqrt(3.0)
    )
    max_offset = int(np.floor(reach_in_cells))
    offset_range = np.arange(-max_offset, max_offset + 1)
    offsets = np.stack(
        np.meshgrid(offset_range, offset_range, offset_range, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    within_reach = np.sum(offsets**2, axis=-1) <= reach_in_cells**2
    return offsets[within_reach]


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _accrete_gas(
    primitive_state: STATE_TYPE,
    sink_particles: SinkParticles,
    num_active_sinks: jax.Array,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> tuple[STATE_TYPE, SinkParticles]:
    """
    Move the gas above the density threshold within each sink's accretion
    radius onto that sink.

    Args:
        primitive_state: The (padded) primitive state.
        sink_particles: The sink particles, including the new massless ones.
        num_active_sinks: The number of filled slots, including the new sinks.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        ``(primitive_state, sink_particles)`` after accretion.
    """

    sink_particle_config = config.sink_particle_config
    grid_spacing = config.grid_spacing
    cell_volume = grid_spacing**3
    accretion_radius = sink_particle_config.accretion_radius_in_cells * grid_spacing
    num_ghost_cells = config.num_ghost_cells

    density = primitive_state[registered_variables.density_index]
    velocity = jnp.stack(
        [
            primitive_state[registered_variables.velocity_index.x],
            primitive_state[registered_variables.velocity_index.y],
            primitive_state[registered_variables.velocity_index.z],
        ],
        axis=-1,
    )

    # The density threshold of Eq. 32, per cell, as in the creation step.
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

    # -------------------------------------------------------------
    # ================ ↓ Cells around each sink ↓ =================
    # -------------------------------------------------------------

    # Each sink gathers the cells around the cell that contains it, giving
    # arrays of shape (max_num_sinks, num_offsets). The filled slots are
    # contiguous at the front, so the active sinks are the first ones.
    max_num_sinks = sink_particle_config.max_num_sinks
    sink_is_active = jnp.arange(max_num_sinks) < num_active_sinks

    offsets = _accretion_offsets(config)
    grid_shape = np.array(density.shape)
    num_interior_cells = grid_shape - 2 * num_ghost_cells
    is_periodic, box_length = _periodic_box(config)

    sink_cell = jnp.floor(sink_particles.position / grid_spacing).astype(jnp.int32)
    interior_index = sink_cell[:, None, :] + offsets[None, :, :]

    # Map every offset to a real (interior) cell. Along a periodic axis the
    # index wraps around the interior, which reaches the real cell across the
    # boundary whether or not ghost cells are used (with ghost cells, the
    # ghost layer only holds copies, which must not be accreted from). Along
    # a non-periodic axis there is no gas beyond the boundary, so offsets
    # leaving the interior are dropped.
    interior_index = jnp.where(
        is_periodic,
        interior_index % num_interior_cells,
        interior_index,
    )
    is_real_cell = jnp.all(
        (interior_index >= 0) & (interior_index < num_interior_cells),
        axis=-1,
    )
    interior_index = jnp.clip(interior_index, 0, num_interior_cells - 1)
    cell_indices = tuple(
        interior_index[..., axis] + num_ghost_cells for axis in range(3)
    )

    # Separation from the sink to each cell centre, taken to the nearest
    # periodic copy along periodic axes (minimum-image convention).
    cell_centers = (interior_index + 0.5) * grid_spacing
    separation = cell_centers - sink_particles.position[:, None, :]
    separation = jnp.where(
        is_periodic,
        separation - box_length * jnp.round(separation / box_length),
        separation,
    )
    distance = jnp.linalg.norm(separation, axis=-1)

    density_in_reach = density[cell_indices]
    density_threshold_in_reach = density_threshold[cell_indices]
    velocity_in_reach = velocity[cell_indices]
    mass_increment = (density_in_reach - density_threshold_in_reach) * cell_volume

    # A (sink, cell) pair is a possible accretion if the sink is active, the
    # cell is a real cell within r_acc of the sink, and the cell is above the
    # density threshold (Section 2.3).
    possible_accretion = (
        sink_is_active[:, None]
        & is_real_cell
        & (distance <= accretion_radius)
        & (density_in_reach > density_threshold_in_reach)
    )

    # -------------------------------------------------------------
    # ================= ↓ Accretion checks ↓ ======================
    # -------------------------------------------------------------

    # The gas in the cell containing the sink is always accreted, with no
    # further check (Section 2.3); this also avoids r = 0 below.
    is_inner_cell = jnp.all(offsets == 0, axis=-1)[None, :]

    # Bound check (Section 2.3): ΔM must be bound to the sink and the gas
    # around it, E_grav + E_kin < 0, with E_kin = ½ ΔM |v − v_n|² in the sink's
    # frame. The paper gives no formula for E_grav; as in Federrath's FLASH
    # code, the gas within r_acc is a uniform sphere of density ρ_res and the
    # sink a point mass M_n, both measured from the edge of the accretion
    # sphere:
    #   E_grav = −G (2π/3) ρ_res (r_acc² − r²) ΔM − G M_n ΔM (1/r − 1/r_acc).
    gravitational_constant = params.gravitational_constant
    safe_distance = jnp.where(distance > 0.0, distance, accretion_radius)
    gravitational_energy = (
        -gravitational_constant
        * (2.0 * jnp.pi / 3.0)
        * density_threshold_in_reach
        * (accretion_radius**2 - distance**2)
        * mass_increment
        - gravitational_constant
        * sink_particles.mass[:, None]
        * mass_increment
        * (1.0 / safe_distance - 1.0 / accretion_radius)
    )
    relative_velocity = velocity_in_reach - sink_particles.velocity[:, None, :]
    kinetic_energy = 0.5 * mass_increment * jnp.sum(relative_velocity**2, axis=-1)
    total_energy = gravitational_energy + kinetic_energy
    bound = total_energy < 0.0

    # Radial velocity check (Section 2.3): ΔM must move toward the sink,
    # v_r = r · (v − v_n) / r, with r pointing from the sink to the cell. The
    # paper asks for a negative v_r; as in Federrath's FLASH code, v_r up to
    # 10⁻⁵ c_s is accepted. Gas moving with the sink has v_r = 0 only up to
    # rounding (the sink velocity is a mass-weighted average of gas
    # velocities), so a bound of exactly 0 would accept or reject it at
    # random.
    radial_velocity = jnp.sum(separation * relative_velocity, axis=-1) / safe_distance
    radial_velocity_tolerance = 1e-5 * jnp.sqrt(sound_speed_squared[cell_indices])
    moving_toward_sink = radial_velocity <= radial_velocity_tolerance

    eligible_accretion = possible_accretion & (
        is_inner_cell | (bound & moving_toward_sink)
    )

    # -------------------------------------------------------------
    # ================= ↑ Accretion checks ↑ ======================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↑ Cells around each sink ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============= ↓ Choosing one sink per cell ↓ ================
    # -------------------------------------------------------------

    # A cell within reach of several sinks is accreted by one of them only.
    # The sink whose position lies in the cell comes first; among the other
    # sinks, the cell goes to the one it is most strongly bound to, the lowest
    # E_grav + E_kin (Section 2.3). This order follows FLASH; the paper states
    # both rules but not their order. An exact tie goes to the lowest slot, as
    # FLASH's loop order does; otherwise every tied sink would take the same
    # ΔM. Each choice is made with a scatter-min onto a grid-shaped array,
    # which handles several pairs pointing to the same cell.
    priority = jnp.where(is_inner_cell, 0, 1)
    no_priority = 2
    best_priority = jnp.full(density.shape, no_priority).at[cell_indices].min(
        jnp.where(eligible_accretion, priority, no_priority)
    )
    has_best_priority = eligible_accretion & (priority == best_priority[cell_indices])

    # The energy of the inner cell is not used (r may be 0 there); it only
    # competes with other inner cells, which would need two sinks in one cell.
    selection_key = jnp.where(is_inner_cell, -jnp.inf, total_energy)
    best_key = jnp.full(density.shape, jnp.inf, dtype=distance.dtype).at[
        cell_indices
    ].min(jnp.where(has_best_priority, selection_key, jnp.inf))
    has_best_key = has_best_priority & (selection_key == best_key[cell_indices])

    slot = jnp.broadcast_to(jnp.arange(max_num_sinks)[:, None], distance.shape)
    best_slot = jnp.full(density.shape, max_num_sinks).at[cell_indices].min(
        jnp.where(has_best_key, slot, max_num_sinks)
    )
    accreted = has_best_key & (slot == best_slot[cell_indices])

    # -------------------------------------------------------------
    # ============= ↑ Choosing one sink per cell ↑ ================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ==================== ↓ Gas update ↓ =========================
    # -------------------------------------------------------------

    # An accreted cell is left at the density threshold. The velocity is
    # unchanged, so the cell's momentum drops with its mass. For an ideal gas
    # the pressure is scaled with the density, which keeps c_s² (the
    # temperature) unchanged, as FLASH does at fixed specific internal
    # energy. The update goes through a grid mask, so that several pairs
    # pointing to the same cell never write conflicting values.
    cell_is_accreted = jnp.zeros(density.shape, dtype=bool).at[cell_indices].max(
        accreted
    )
    primitive_state = primitive_state.at[registered_variables.density_index].set(
        jnp.where(cell_is_accreted, density_threshold, density)
    )
    if config.equation_of_state == IDEAL_GAS:
        pressure = primitive_state[registered_variables.pressure_index]
        primitive_state = primitive_state.at[registered_variables.pressure_index].set(
            jnp.where(cell_is_accreted, pressure * density_threshold / density, pressure)
        )

    # -------------------------------------------------------------
    # ==================== ↑ Gas update ↑ =========================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ==================== ↓ Sink update ↓ ========================
    # -------------------------------------------------------------

    # Mass and linear momentum are conserved, and the sink moves to the
    # centre of mass of itself and the accreted gas (Appendix B, Eqs. B1, B2
    # and B6):
    #   M' = M + Σ ΔM,
    #   x' = x + Σ ΔM r / M',
    #   v' = v + Σ ΔM (v_gas − v) / M'.
    accreted_mass = jnp.where(accreted, mass_increment, 0.0)
    new_mass = sink_particles.mass + jnp.sum(accreted_mass, axis=-1)
    safe_new_mass = jnp.where(new_mass > 0.0, new_mass, 1.0)

    position_shift = jnp.sum(
        accreted_mass[..., None] * separation,
        axis=1,
    ) / safe_new_mass[:, None]
    velocity_shift = jnp.sum(
        accreted_mass[..., None]
        * (velocity_in_reach - sink_particles.velocity[:, None, :]),
        axis=1,
    ) / safe_new_mass[:, None]

    # The new position can lie just past the edge of the box; it is wrapped
    # back along periodic axes.
    new_position = sink_particles.position + position_shift
    new_position = jnp.where(
        is_periodic,
        jnp.mod(new_position, box_length),
        new_position,
    )

    sink_particles = SinkParticles(
        mass=new_mass,
        position=new_position,
        velocity=sink_particles.velocity + velocity_shift,
    )

    # -------------------------------------------------------------
    # ==================== ↑ Sink update ↑ ========================
    # -------------------------------------------------------------

    return primitive_state, sink_particles
