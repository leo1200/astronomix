"""
AthenaPK's first-order flux correction of the finite-volume stage.

A cell whose update would leave a non-positive density or pressure (the
density alone for a barotropic gas; ``_positivity_failure_code``) gets the fluxes of all its faces replaced by
first-order ones, donor cell with the local Lax-Friedrichs solver
(``_corrected_fluxes``); the stage computes them with the flux function of its
divergence control and passes them in. Replacing a face flux also
changes the neighbour's update, so check and correction are repeated
(``_first_order_flux_correction``). The fused Pallas kernel
(``_stage_pallas.py``) uses the same failure code and the same rule for the
flagged cells. The flux divergence (``_flux_divergence``) is defined here: the
trial updates of the correction and the update of the stage both use it. Next
to it stands the divergence of the spherical 1D grid with the pressure source
of the geometry (``_spherical_flux_divergence``).
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    GHOST_CELLS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._interface_frame import _velocity_component_indices
from astronomix._finite_volume._reconstruction import (
    _cell_distances,
    _limited_gradients,
)
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._geometry.geometric_terms import _pressure_nozzling_source


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
    last attempt), ``DENSITY_FAILURE`` otherwise. A barotropic gas has the
    density criterion only. Elementwise on per-variable values, so shared by
    the native path and the Pallas kernels.

    Args:
        conserved_components: The updated conserved values, indexed like the state.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The failure code per cell.
    """
    density = conserved_components[registered_variables.density_index]
    if is_barotropic(config):
        return jnp.where(density > 0.0, POSITIVE, DENSITY_FAILURE)

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


def _corrected_fluxes(high_order_fluxes, first_order_fluxes, correction_mask, axes):
    """
    Replace the fluxes of every face bordering a cell of ``correction_mask`` by
    the first-order ones (the interface ``i - 1/2`` borders cells ``i - 1`` and
    ``i``).

    Args:
        high_order_fluxes: The stage's interface fluxes per axis.
        first_order_fluxes: The donor-cell LLF interface fluxes per axis.
        correction_mask: Boolean mask of the flagged cells (no variable axis).
        axes: The spatial axes (1-based) of the fluxes.

    Returns:
        The corrected interface fluxes per axis.
    """
    corrected_fluxes = []
    for axis, high_order, first_order in zip(axes, high_order_fluxes, first_order_fluxes):
        # The mask has no variable axis, so spatial axis d (1-based in the
        # state) is array axis d - 1 here.
        face_mask = correction_mask | jnp.roll(correction_mask, 1, axis=axis - 1)
        corrected_fluxes.append(jnp.where(face_mask[None], first_order, high_order))
    return corrected_fluxes


def _flux_divergence(fluxes_per_axis, axes, config: SimulationConfig):
    """
    The flux divergence ``(1/dx) Σ_d (F_{i+1/2} - F_{i-1/2})`` over the given
    axes.

    Args:
        fluxes_per_axis: The interface fluxes (at ``i - 1/2``) per axis.
        axes: The spatial axes (1-based) of the fluxes.
        config: The simulation configuration.

    Returns:
        The flux divergence, laid out like the state.
    """
    flux_difference = 0.0
    for axis, fluxes in zip(axes, fluxes_per_axis):
        flux_difference = flux_difference + (jnp.roll(fluxes, -1, axis=axis) - fluxes)
    return flux_difference / config.grid_spacing


def _spherical_flux_divergence(
    fluxes: STATE_TYPE,
    pressure,
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
):
    """
    The flux divergence of the spherical 1D grid with the pressure source of
    the geometry,

        (1 / r_hat^2) ((r_{i+1/2}^2 F_{i+1/2} - r_{i-1/2}^2 F_{i-1/2}) / dr - S_i),

    where ``r_hat^2`` is the cell average of ``r^2``. The area factors act on
    the whole momentum flux, pressure included; the source ``S`` gives back
    the part of the pressure that does not push on the gas
    (``_pressure_nozzling_source``), so a uniform gas at rest stays at rest.

    Args:
        fluxes: The radial interface fluxes (at ``i - 1/2``).
        pressure: The gas pressure of the state the fluxes were computed from.
        config: The simulation configuration.
        helper_data: The helper data (radii of the grid).
        registered_variables: The registered variables.

    Returns:
        The flux divergence minus the source, laid out like the state.
    """
    # The geometry constant is the exponent of the face area (SPHERICAL = 2: the area grows as r^2).
    area_exponent = config.geometry
    radius = helper_data.geometric_centers
    inner_area = (radius - config.grid_spacing / 2) ** area_exponent
    outer_area = (radius + config.grid_spacing / 2) ** area_exponent

    pressure_gradient = _limited_gradients(
        jnp.roll(pressure, 1),
        pressure,
        jnp.roll(pressure, -1),
        config.solver_config.limiter,
        _cell_distances(config, helper_data),
    )
    pressure_source = _pressure_nozzling_source(
        pressure,
        pressure_gradient,
        config,
        helper_data,
        registered_variables,
    )

    weighted_difference = (outer_area * jnp.roll(fluxes, -1, axis=1) - inner_area * fluxes) / config.grid_spacing
    return 1 / helper_data.r_hat_alpha * (weighted_difference - pressure_source)


def _first_order_flux_correction(
    fluxes_per_axis,
    first_order_fluxes,
    correctable_cells,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    axes,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The first-order flux correction (AthenaPK: ``first_order_flux_correct``):
    every correctable cell whose update would leave a non-positive density or
    pressure gets all its face fluxes replaced by the first-order fluxes
    (donor-cell LLF fluxes of the stage's primitive state). Replacing a face
    flux also changes the neighbour's update, so the check is repeated (with
    the corrected fluxes) ``NUM_FLUX_CORRECTION_ATTEMPTS`` times; in the last
    attempt only density failures are corrected, pure pressure failures are
    left to the floors.

    Args:
        fluxes_per_axis: The stage's interface fluxes per axis.
        first_order_fluxes: The donor-cell LLF interface fluxes per axis.
        correctable_cells: Boolean mask of the cells that may be flagged (the
            interior cells; no variable axis).
        base_conserved_state: The register the stage update starts from
            (``U^n`` in both VL2 stages, the stage state itself under ``RK2_SSP``).
        stage_time_step: The stage's time-step weight ``beta * dt``.
        axes: The spatial axes (1-based) of the fluxes.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The corrected interface fluxes per axis.
    """
    correction_mask = jnp.zeros_like(correctable_cells)
    for attempt in range(NUM_FLUX_CORRECTION_ATTEMPTS):
        fluxes = _corrected_fluxes(fluxes_per_axis, first_order_fluxes, correction_mask, axes)
        new_state = base_conserved_state - stage_time_step * _flux_divergence(fluxes, axes, config)
        failure_codes = _positivity_failure_code(list(new_state), config, registered_variables)
        correction_mask = correction_mask | (_newly_flagged_cells(failure_codes, attempt) & correctable_cells)

    return _corrected_fluxes(fluxes_per_axis, first_order_fluxes, correction_mask, axes)


# -------------------------------------------------------------
# =============== ↑ First-order flux correction ↑ =============
# -------------------------------------------------------------
