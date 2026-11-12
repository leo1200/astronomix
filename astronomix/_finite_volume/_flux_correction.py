"""
AthenaPK's first-order flux correction of the finite-volume stage.

A cell whose update would leave a non-positive density or pressure
(``_positivity_failure_code``) gets the fluxes of all its faces replaced by
first-order ones, donor cell with the local Lax-Friedrichs solver
(``_first_order_fluxes``, ``_corrected_fluxes``). Replacing a face flux also
changes the neighbour's update, so check and correction are repeated
(``_first_order_flux_correction``). The fused Pallas kernel
(``_stage_pallas.py``) uses the same failure code and the same rule for the
flagged cells; the stage with constrained transport has its own loop
(``_divergence/_constrained_transport.py``).
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    GHOST_CELLS,
    LAX_FRIEDRICHS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables


# -------------------------------------------------------------
# =================== ↓ Module constants ↓ ====================
# -------------------------------------------------------------

#: Failure code of a cell whose update keeps density and pressure positive.
POSITIVE = 0

#: Failure code of a cell whose update keeps the density positive but not the
#: pressure.
PRESSURE_FAILURE = 1

#: Failure code of every other failing cell (mainly a non-positive density).
DENSITY_FAILURE = 2

#: Number of check-and-correct passes of the first-order flux correction, as in
#: AthenaPK. Correcting a face also changes the neighbour's update, so the
#: check is repeated; the last pass only corrects density failures.
NUM_FLUX_CORRECTION_ATTEMPTS = 4

# -------------------------------------------------------------
# =================== ↑ Module constants ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ First-order flux correction ↓ =============
# -------------------------------------------------------------


def _interior_mask(primitive_state: STATE_TYPE, config: SimulationConfig):
    """
    Boolean mask of the interior (non-ghost) cells of the spatial grid.

    Args:
        primitive_state: The (padded) primitive state.
        config: The simulation configuration.

    Returns:
        The mask, shaped like one variable of the state.
    """
    spatial_shape = primitive_state.shape[1:]
    if config.boundary_handling != GHOST_CELLS:
        return jnp.ones(spatial_shape, dtype=bool)
    num_ghost_cells = config.num_ghost_cells
    interior = tuple(
        slice(num_ghost_cells, extent - num_ghost_cells) for extent in spatial_shape
    )
    return jnp.zeros(spatial_shape, dtype=bool).at[interior].set(True)


def _positivity_failure_code(
    conserved_components,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Classify an updated conserved state as AthenaPK's flux correction does:
    ``POSITIVE`` if density and pressure are positive, ``PRESSURE_FAILURE`` if
    only the pressure is negative (AthenaPK leaves these to the floors in its
    last attempt), ``DENSITY_FAILURE`` otherwise. Elementwise on per-variable
    values, so shared by the native path and the Pallas kernels.

    Args:
        conserved_components: The updated conserved values, indexed like the state.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The failure code per cell.
    """
    # Imported here because the stage module imports this module at its top.
    from astronomix._finite_volume._stage import _velocity_component_indices

    density = conserved_components[registered_variables.density_index]

    # This is the internal energy density p / (gamma - 1); only its sign is
    # needed.
    internal_energy = conserved_components[registered_variables.energy_index]
    for index in _velocity_component_indices(registered_variables):
        if index is not None:
            internal_energy = internal_energy - 0.5 * conserved_components[index] ** 2 / density
    if config.mhd:
        for index in tuple(registered_variables.magnetic_index):
            internal_energy = internal_energy - 0.5 * conserved_components[index] ** 2

    is_positive = (density > 0.0) & (internal_energy > 0.0)
    only_pressure_negative = (density > 0.0) & (internal_energy < 0.0)
    return jnp.where(
        is_positive,
        POSITIVE,
        jnp.where(only_pressure_negative, PRESSURE_FAILURE, DENSITY_FAILURE),
    )


def _newly_flagged_cells(failure_codes, attempt: int):
    """
    The cells the flux correction flags in a given attempt: every failing
    cell, except in the last attempt, which leaves pressure-only failures to
    the floors (as AthenaPK).

    Args:
        failure_codes: The positivity failure codes of the attempt's update.
        attempt: The attempt number (0-based).

    Returns:
        A boolean mask of the flagged cells.
    """
    if attempt < NUM_FLUX_CORRECTION_ATTEMPTS - 1:
        return failure_codes != POSITIVE
    return failure_codes == DENSITY_FAILURE


def _corrected_fluxes(high_order_fluxes, first_order_fluxes, correction_mask):
    """
    Replace the fluxes of every face bordering a cell of ``correction_mask`` by
    the first-order ones (the interface ``i - 1/2`` borders cells ``i - 1`` and
    ``i``).

    Args:
        high_order_fluxes: The stage's interface fluxes per axis.
        first_order_fluxes: The donor-cell LLF interface fluxes per axis.
        correction_mask: Boolean mask of the flagged cells (no variable axis).

    Returns:
        The corrected interface fluxes per axis.
    """
    corrected_fluxes = []
    # The mask has no variable axis, so spatial axis d (1-based in the state)
    # is array axis d - 1 here.
    for mask_axis, (high_order, first_order) in enumerate(
        zip(high_order_fluxes, first_order_fluxes)
    ):
        face_mask = correction_mask | jnp.roll(correction_mask, 1, axis=mask_axis)
        corrected_fluxes.append(jnp.where(face_mask[None], first_order, high_order))
    return corrected_fluxes


def _first_order_fluxes(
    stage_primitive_state: STATE_TYPE,
    gamma,
    cleaning_speed,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The donor-cell local Lax-Friedrichs fluxes of every axis.

    Args:
        stage_primitive_state: The primitive state the stage started from.
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (MHD only).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The interface fluxes per axis.
    """
    # Imported here because the stage module imports this module at its top.
    from astronomix._finite_volume._stage import _interface_fluxes

    return [
        _interface_fluxes(
            stage_primitive_state,
            axis,
            piecewise_linear=False,
            riemann_solver=LAX_FRIEDRICHS,
            gamma=gamma,
            cleaning_speed=cleaning_speed,
            config=config,
            registered_variables=registered_variables,
        )
        for axis in range(1, config.dimensionality + 1)
    ]


def _first_order_flux_correction(
    fluxes_per_axis,
    stage_primitive_state: STATE_TYPE,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    gamma,
    cleaning_speed,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The first-order flux correction (AthenaPK: ``first_order_flux_correct``):
    every interior cell whose update would leave a non-positive density or
    pressure gets all its face fluxes replaced by donor-cell LLF fluxes of the
    stage's primitive state. Replacing a face flux also changes the
    neighbour's update, so the check is repeated (with the corrected fluxes)
    ``NUM_FLUX_CORRECTION_ATTEMPTS`` times; in the last attempt only density
    failures are corrected, pure pressure failures are left to the floors.

    Args:
        fluxes_per_axis: The stage's interface fluxes per axis.
        stage_primitive_state: The primitive state the stage started from.
        base_conserved_state: The register the stage update starts from
            (``U^n`` in both VL2 stages).
        stage_time_step: The stage's time-step weight ``beta * dt``.
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (MHD only).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The corrected interface fluxes per axis.
    """
    # Imported here because the stage module imports this module at its top.
    from astronomix._finite_volume._stage import _flux_divergence

    first_order_fluxes = _first_order_fluxes(
        stage_primitive_state,
        gamma,
        cleaning_speed,
        config,
        registered_variables,
    )
    interior = _interior_mask(stage_primitive_state, config)

    correction_mask = jnp.zeros_like(interior)
    for attempt in range(NUM_FLUX_CORRECTION_ATTEMPTS):
        fluxes = _corrected_fluxes(fluxes_per_axis, first_order_fluxes, correction_mask)
        new_state = base_conserved_state - stage_time_step * _flux_divergence(fluxes, config)
        failure_codes = _positivity_failure_code(list(new_state), config, registered_variables)
        correction_mask = correction_mask | (_newly_flagged_cells(failure_codes, attempt) & interior)

    return _corrected_fluxes(fluxes_per_axis, first_order_fluxes, correction_mask)


# -------------------------------------------------------------
# =============== ↑ First-order flux correction ↑ =============
# -------------------------------------------------------------
