"""
The VL2 finite-volume scheme of AthenaPK.

A re-implementation of AthenaPK's default second-order scheme: the van Leer
predictor-corrector ("VL2", Stone & Gardiner 2009) with a donor-cell predictor
half step and a piecewise-linear (harmonic van Leer slope) corrector step,
AthenaPK's HLLD / HLLE / HLLC / LLF interface solvers, and — for MHD — the
cell-centred GLM-MHD system of Dedner et al. (2002) with the cleaning scalar psi
carried as a ninth variable. AthenaPK's first-order flux correction and its
density / pressure floors are available as options. Self-gravity (and an
external potential) couple as in Athena++, whose VL2 integrator this is: every
stage adds the source of ``_add_athena_gravity_source`` built from the
potential of the stage's initial state and the stage's Riemann mass fluxes,
optionally with the time-centred energy source of Mullen, Hanawa & Gammie
(2021). The gravity source runs on the native backend only.

One step of size ``dt`` from the primitive state ``W^n`` (conserved form
``U^n``) reads, exactly as AthenaPK's task list::

    c_h      = max over cells and axes of |v_d| + c_fast,d     (MHD only)
    U^{n+½}  = U^n - dt/2 L_DC(W^n)       then  psi *= exp(-α c_h (dt/2) / dx)
    W^{n+½}  = W(U^{n+½})
    U^{n+1}  = U^n - dt   L_PLM(W^{n+½})  then  psi *= exp(-α c_h dt / dx)
    W^{n+1}  = W(U^{n+1})

with the flux divergence ``L(W)_i = (1/dx) Σ_d (F_{i+½} - F_{i-½})``. With
gravity each stage adds ``beta dt S(rho_stage, F_stage, phi_stage)`` to its
update, where ``phi_stage`` is the potential of the stage's initial density
(two Poisson solves per step, as in Athena++; three with the time-centred
energy source).

The per-face physics (reconstruction, Riemann solvers, conversions) is written as
elementwise functions that the Pallas kernels in :mod:`._van_leer_pallas` call as
well, so both backends evaluate the same expressions. The scheme reproduces
AthenaPK to round-off on its standard test problems
(``pytests/mhd/vl2_athenapk_regression.py``; the full comparison is documented in
``examples/scripts/validation/athenapk_vl2/README.md``).
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DONOR_CELL,
    GHOST_CELLS,
    HLL,
    HLLC,
    HLLD,
    LAX_FRIEDRICHS,
    POSITIVITY_HARD_FLOOR,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix._finite_volume._riemann_solver._athena_riemann_solvers import (
    HydroFaceState,
    MHDFaceState,
)

# astronomix functions
from astronomix._finite_volume._riemann_solver._athena_riemann_solvers import (
    hllc_hydro_flux,
    hlld_flux,
    hlle_hydro_flux,
    hlle_mhd_flux,
    llf_hydro_flux,
    llf_mhd_flux,
)
from astronomix._finite_volume._magnetic_update._glm_divergence_cleaning import (
    _dedner_source,
    _glm_cleaning_speed,
    _psi_damping_factor,
)
from astronomix._finite_volume._state_evolution._van_leer_pallas import (
    _vl2_pallas_supported,
    _vl2_stage_pallas,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._modules._gravity._gravity import (
    _add_athena_gravity_source,
    _compute_total_potential,
)


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
# ===================== ↓ State layout ↓ ======================
# -------------------------------------------------------------


def _velocity_component_indices(registered_variables: RegisteredVariables):
    """
    The state indices of the x, y and z velocity. Components the layout does
    not carry (the dimension-reduced hydro layouts) are ``None``; a missing
    component behaves exactly like a component that is zero everywhere.

    Args:
        registered_variables: The registered variables.

    Returns:
        A tuple of three state indices (``None`` for absent components).
    """
    velocity_index = registered_variables.velocity_index
    if isinstance(velocity_index, int):
        return (velocity_index, None, None)
    return tuple(component if component >= 0 else None for component in tuple(velocity_index))


def _interface_frame_indices(
    axis: int,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The state indices of the variables in the frame of an interface normal to
    ``axis`` (1-based), in AthenaPK's cyclic order: normal, then transverse 1
    and transverse 2 (x: y, z; y: z, x; z: x, y).

    Args:
        axis: The spatial axis (1, 2 or 3) normal to the interface.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        A tuple of state indices (``None`` for components the layout lacks),
        ordered like :class:`MHDFaceState` (MHD) or :class:`HydroFaceState`.
    """
    velocity = _velocity_component_indices(registered_variables)
    normal = axis - 1
    transverse_1 = axis % 3
    transverse_2 = (axis + 1) % 3

    hydro_indices = (
        registered_variables.density_index,
        velocity[normal],
        velocity[transverse_1],
        velocity[transverse_2],
        registered_variables.pressure_index,
    )
    if not config.mhd:
        return hydro_indices

    field = tuple(registered_variables.magnetic_index)
    return hydro_indices + (
        field[normal],
        field[transverse_1],
        field[transverse_2],
        registered_variables.magnetic_psi_index,
    )


def _state_components_from_frame(frame_values, frame_indices, num_vars: int):
    """
    Scatter interface-frame values back to state order.

    Args:
        frame_values: The values in the interface frame (e.g. a flux tuple).
        frame_indices: The state index of every frame value (``None`` for
            components the layout lacks, which are dropped).
        num_vars: The number of state variables.

    Returns:
        A list of the values indexed like the state.
    """
    state_components = [None] * num_vars
    for index, value in zip(frame_indices, frame_values):
        if index is not None:
            state_components[index] = value
    return state_components


# -------------------------------------------------------------
# ===================== ↑ State layout ↑ ======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============= ↓ Conserved <-> primitive variables ↓ ==========
# -------------------------------------------------------------


def _uses_floors(config: SimulationConfig) -> bool:
    """
    Return whether the density / pressure floors of the VL2 scheme are active.

    The floors (off by default, as in AthenaPK) are switched on by the per-step
    hard floor, ``positivity_config.per_step_mode == POSITIVITY_HARD_FLOOR``;
    like AthenaPK, the VL2 scheme then also applies them in every stage's
    conversion to primitives.
    """
    return config.positivity_config.per_step_mode == POSITIVITY_HARD_FLOOR


def _conserved_components_from_primitive(
    primitive_components,
    gamma,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Per-variable conserved values ``(rho, rho v, E, B, psi)`` from per-variable
    primitive values. The values are given as lists indexed like the state, so
    the function serves whole arrays (native) and register tiles (Pallas) alike.

    Args:
        primitive_components: The primitive values, indexed like the state.
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved values, indexed like the state.
    """
    density = primitive_components[registered_variables.density_index]
    velocity_indices = [
        index for index in _velocity_component_indices(registered_variables) if index is not None
    ]

    conserved_components = list(primitive_components)
    kinetic_energy = 0.0
    for index in velocity_indices:
        velocity = primitive_components[index]
        conserved_components[index] = density * velocity
        kinetic_energy = kinetic_energy + 0.5 * density * velocity * velocity

    energy = (
        primitive_components[registered_variables.pressure_index] / (gamma - 1.0) + kinetic_energy
    )
    if config.mhd:
        for index in tuple(registered_variables.magnetic_index):
            field = primitive_components[index]
            energy = energy + 0.5 * field * field

    conserved_components[registered_variables.energy_index] = energy
    return conserved_components


def _primitive_components_from_conserved(
    conserved_components,
    gamma,
    density_floor,
    pressure_floor,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The conversion from conserved to primitive values with optional floors
    (AthenaPK: ``ConsToPrim``) on per-variable values (lists indexed like the
    state). With the floors active, the density floor leaves momentum and
    energy untouched and the pressure floor raises the pressure (AthenaPK also
    resets the energy accordingly, which is implicit here as the conserved
    state is not kept).

    Args:
        conserved_components: The conserved values, indexed like the state.
        gamma: The adiabatic index.
        density_floor: The density floor (ignored without floors).
        pressure_floor: The pressure floor (ignored without floors).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive values, indexed like the state.
    """
    density = conserved_components[registered_variables.density_index]
    if _uses_floors(config):
        density = jnp.maximum(density, density_floor)
    inverse_density = 1.0 / density

    primitive_components = list(conserved_components)
    primitive_components[registered_variables.density_index] = density

    internal_energy = conserved_components[registered_variables.energy_index]
    for index in _velocity_component_indices(registered_variables):
        if index is not None:
            momentum = conserved_components[index]
            primitive_components[index] = momentum * inverse_density
            internal_energy = internal_energy - 0.5 * inverse_density * momentum * momentum
    if config.mhd:
        for index in tuple(registered_variables.magnetic_index):
            field = conserved_components[index]
            internal_energy = internal_energy - 0.5 * field * field

    pressure = (gamma - 1.0) * internal_energy
    if _uses_floors(config):
        pressure = jnp.maximum(pressure, pressure_floor)

    primitive_components[registered_variables.pressure_index] = pressure
    return primitive_components


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _conserved_from_primitive_vl2(
    primitive_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The conserved state ``(rho, rho v, E, B, psi)`` of a primitive state.

    Args:
        primitive_state: The primitive state.
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state (same layout as the primitive state).
    """
    conserved_components = _conserved_components_from_primitive(
        list(primitive_state),
        gamma,
        config,
        registered_variables,
    )
    return jnp.stack(conserved_components, axis=0)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _primitive_from_conserved_vl2(
    conserved_state: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The primitive state of a conserved state, with AthenaPK's density and
    pressure floors (``params.minimum_density`` and ``params.minimum_pressure``)
    when the per-step positivity mode is ``POSITIVITY_HARD_FLOOR`` (see
    ``_uses_floors``).

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters (floors).
        registered_variables: The registered variables.

    Returns:
        The primitive state.
    """
    primitive_components = _primitive_components_from_conserved(
        list(conserved_state),
        gamma,
        params.minimum_density,
        params.minimum_pressure,
        config,
        registered_variables,
    )
    return jnp.stack(primitive_components, axis=0)


# -------------------------------------------------------------
# ============= ↑ Conserved <-> primitive variables ↑ ==========
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================== ↓ Interface fluxes ↓ =====================
# -------------------------------------------------------------


def _van_leer_slopes(cell_minus, cell, cell_plus):
    """
    The piecewise-linear reconstruction of one cell with the van Leer
    (harmonic mean) limited half slope ``Δ_L Δ_R / (Δ_L + Δ_R)`` where the
    one-sided differences agree in sign, zero otherwise (AthenaPK:
    ``plm_simple.hpp``).

    Args:
        cell_minus: The value in the left neighbour.
        cell: The value in the cell.
        cell_plus: The value in the right neighbour.

    Returns:
        ``(value at the cell's right face, value at the cell's left face)``.
    """
    difference_left = cell - cell_minus
    difference_right = cell_plus - cell
    difference_product = difference_left * difference_right
    is_monotone = difference_product > 0.0
    # The denominator is guarded so that the discarded branch stays finite.
    safe_sum = jnp.where(is_monotone, difference_left + difference_right, 1.0)
    limited_half_slope = jnp.where(is_monotone, difference_product / safe_sum, 0.0)
    return cell + limited_half_slope, cell - limited_half_slope


def _reconstructed_face_states(primitive_state: STATE_TYPE, axis: int, piecewise_linear: bool):
    """
    The left and right primitive states at every interface ``i - 1/2`` along
    ``axis`` (stored at index ``i``).

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based, i.e. the array axis).
        piecewise_linear: Use the van Leer limited PLM; otherwise donor cell.

    Returns:
        The (left, right) interface states, each shaped like the state.
    """
    if not piecewise_linear:
        return jnp.roll(primitive_state, 1, axis=axis), primitive_state

    right_face_values, left_face_values = _van_leer_slopes(
        jnp.roll(primitive_state, 1, axis=axis),
        primitive_state,
        jnp.roll(primitive_state, -1, axis=axis),
    )
    return jnp.roll(right_face_values, 1, axis=axis), left_face_values


def _interface_flux_from_states(
    left_values,
    right_values,
    gamma,
    cleaning_speed,
    riemann_solver: int,
    mhd: bool,
):
    """
    Evaluate the configured Riemann solver on interface-frame value tuples.

    Args:
        left_values: Tuple of left-state values (interface frame).
        right_values: Tuple of right-state values (interface frame).
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (MHD only).
        riemann_solver: The configured Riemann solver.
        mhd: Whether this is GLM-MHD.

    Returns:
        The interface flux (:class:`MHDFlux` or :class:`HydroFlux`, interface
        frame).
    """
    if mhd:
        left = MHDFaceState(*left_values)
        right = MHDFaceState(*right_values)
        if riemann_solver == HLLD:
            return hlld_flux(left, right, gamma, cleaning_speed)
        if riemann_solver == HLL:
            return hlle_mhd_flux(left, right, gamma, cleaning_speed)
        if riemann_solver == LAX_FRIEDRICHS:
            return llf_mhd_flux(left, right, gamma, cleaning_speed)
        raise ValueError("Unsupported Riemann solver for VL2 MHD.")

    left = HydroFaceState(*left_values)
    right = HydroFaceState(*right_values)
    if riemann_solver == HLLC:
        return hllc_hydro_flux(left, right, gamma)
    if riemann_solver == HLL:
        return hlle_hydro_flux(left, right, gamma)
    if riemann_solver == LAX_FRIEDRICHS:
        return llf_hydro_flux(left, right, gamma)
    raise ValueError("Unsupported Riemann solver for VL2 hydrodynamics.")


def _interface_fluxes(
    primitive_state: STATE_TYPE,
    axis: int,
    piecewise_linear: bool,
    riemann_solver: int,
    gamma,
    cleaning_speed,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The interface fluxes along one axis: ``flux[:, i]`` is the flux through
    the interface ``i - 1/2``.

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based).
        piecewise_linear: Use PLM (else donor-cell) reconstruction.
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (MHD only).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The flux array, laid out like the state.
    """
    left_states, right_states = _reconstructed_face_states(primitive_state, axis, piecewise_linear)
    frame_indices = _interface_frame_indices(axis, config, registered_variables)

    # Velocity components the layout lacks enter the solver as zeros.
    zero = jnp.zeros_like(primitive_state[registered_variables.density_index])
    left_values = tuple(
        left_states[index] if index is not None else zero for index in frame_indices
    )
    right_values = tuple(
        right_states[index] if index is not None else zero for index in frame_indices
    )

    flux_values = _interface_flux_from_states(
        left_values,
        right_values,
        gamma,
        cleaning_speed,
        riemann_solver,
        config.mhd,
    )
    return jnp.stack(
        _state_components_from_frame(flux_values, frame_indices, registered_variables.num_vars),
        axis=0,
    )


def _flux_divergence(fluxes_per_axis, config: SimulationConfig):
    """
    The flux divergence ``(1/dx) Σ_d (F_{i+1/2} - F_{i-1/2})``.

    Args:
        fluxes_per_axis: The interface fluxes (at ``i - 1/2``) of every axis.
        config: The simulation configuration.

    Returns:
        The flux divergence, laid out like the state.
    """
    flux_difference = 0.0
    for axis, fluxes in enumerate(fluxes_per_axis, start=1):
        flux_difference = flux_difference + (jnp.roll(fluxes, -1, axis=axis) - fluxes)
    return flux_difference / config.grid_spacing


# -------------------------------------------------------------
# ================== ↑ Interface fluxes ↑ =====================
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

# -------------------------------------------------------------
# ===================== ↓ VL2 time step ↓ =====================
# -------------------------------------------------------------


def _vl2_gravitational_potential(
    density,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    The total gravitational potential (self-gravity and external) of a density
    field of the VL2 scheme.

    Args:
        density: The density field.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The potential, shaped like the density.
    """
    return _compute_total_potential(
        density,
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )


def _vl2_stage(
    stage_primitive_state: STATE_TYPE,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    piecewise_linear: bool,
    cleaning_speed,
    damping_factor,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    stage_potential=None,
    initial_potential=None,
):
    """
    One VL2 stage (native JAX): ``U = U_base - beta dt L(W_stage)``, the
    gravity source, the Dedner source and the conversion back to primitives.
    Ghost cells are not updated here.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_conserved_state: The conserved register the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        piecewise_linear: PLM (corrector) or donor-cell (predictor) fluxes.
        cleaning_speed: The GLM cleaning speed (MHD only).
        damping_factor: The stage's psi damping factor (MHD only).
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        stage_potential: The potential of the stage's initial state (gravity
            only).
        initial_potential: The potential at the start of the step, ``t^n``
            (gravity with the time-centred energy source only).

    Returns:
        The primitive state after the stage, and the potential of the stage's
        new density if the time-centred energy source computed it (otherwise
        ``None``).
    """
    fluxes_per_axis = [
        _interface_fluxes(
            stage_primitive_state,
            axis,
            piecewise_linear,
            config.solver_config.riemann_solver,
            gamma,
            cleaning_speed,
            config,
            registered_variables,
        )
        for axis in range(1, config.dimensionality + 1)
    ]

    if config.solver_config.first_order_flux_correction:
        fluxes_per_axis = _first_order_flux_correction(
            fluxes_per_axis,
            stage_primitive_state,
            base_conserved_state,
            stage_time_step,
            gamma,
            cleaning_speed,
            config,
            registered_variables,
        )

    conserved_state = base_conserved_state - stage_time_step * _flux_divergence(
        fluxes_per_axis,
        config,
    )

    end_potential = None
    if config.gravity_config.gravity:
        density_index = registered_variables.density_index
        work_potential = stage_potential
        if config.gravity_config.time_centred_energy_source:
            # The density has no source term, so the stage's new density, and
            # with it the potential at the end of the stage, is known before
            # the gravity source is added.
            end_potential = _vl2_gravitational_potential(
                conserved_state[density_index],
                config,
                params,
                registered_variables,
            )
            work_potential = 0.5 * (initial_potential + end_potential)
        conserved_state = _add_athena_gravity_source(
            conserved_state,
            stage_primitive_state[density_index],
            [fluxes[density_index] for fluxes in fluxes_per_axis],
            stage_potential,
            work_potential,
            stage_time_step,
            config,
            registered_variables,
        )

    if config.mhd:
        conserved_state = _dedner_source(
            conserved_state,
            stage_primitive_state,
            damping_factor,
            stage_time_step,
            config,
            registered_variables,
        )

    primitive_state = _primitive_from_conserved_vl2(
        conserved_state,
        gamma,
        config,
        params,
        registered_variables,
    )
    return primitive_state, end_potential


def _native_stage_from_primitive_base(
    stage_primitive_state: STATE_TYPE,
    base_primitive_state: STATE_TYPE,
    stage_time_step,
    piecewise_linear: bool,
    cleaning_speed,
    damping_factor,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    correction_mask=None,
):
    """
    The native stage in the Pallas kernel's signature, used as the tangent of
    the Pallas stage: the base register is given in primitive form and the
    first-order flux correction, if any, uses the given cell mask (the Pallas
    path runs the correction attempts outside the kernel).

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_primitive_state: The primitive state ``W^n`` whose conserved form
            the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        piecewise_linear: PLM (corrector) or donor-cell (predictor) fluxes.
        cleaning_speed: The GLM cleaning speed (MHD only).
        damping_factor: The stage's psi damping factor (MHD only).
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        correction_mask: ``None``, or a float ``(1, *grid)`` array that is
            larger than 0.5 in the cells whose faces take first-order LLF
            fluxes.

    Returns:
        The primitive state after the stage, and with a ``correction_mask`` also
        the ``(1, *grid)`` positivity failure codes of the update.
    """
    base_conserved_state = _conserved_from_primitive_vl2(
        base_primitive_state,
        gamma,
        config,
        registered_variables,
    )
    fluxes_per_axis = [
        _interface_fluxes(
            stage_primitive_state,
            axis,
            piecewise_linear,
            config.solver_config.riemann_solver,
            gamma,
            cleaning_speed,
            config,
            registered_variables,
        )
        for axis in range(1, config.dimensionality + 1)
    ]
    if correction_mask is not None:
        first_order_fluxes = _first_order_fluxes(
            stage_primitive_state,
            gamma,
            cleaning_speed,
            config,
            registered_variables,
        )
        fluxes_per_axis = _corrected_fluxes(
            fluxes_per_axis,
            first_order_fluxes,
            correction_mask[0] > 0.5,
        )

    conserved_state = base_conserved_state - stage_time_step * _flux_divergence(
        fluxes_per_axis,
        config,
    )
    if correction_mask is not None:
        failure_codes = _positivity_failure_code(
            list(conserved_state),
            config,
            registered_variables,
        )
        failure_codes = jnp.where(
            _interior_mask(stage_primitive_state, config),
            failure_codes,
            POSITIVE,
        )
    if config.mhd:
        conserved_state = _dedner_source(
            conserved_state,
            stage_primitive_state,
            damping_factor,
            stage_time_step,
            config,
            registered_variables,
        )
    primitive_state = _primitive_from_conserved_vl2(
        conserved_state,
        gamma,
        config,
        params,
        registered_variables,
    )
    if correction_mask is None:
        return primitive_state
    return primitive_state, failure_codes[None].astype(primitive_state.dtype)


def _cleaning_speed_and_damping(
    primitive_state: STATE_TYPE,
    dt,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    The step's GLM cleaning speed and the psi damping factors of the two stages
    (``c_h = 0`` and no damping for hydrodynamics).

    Args:
        primitive_state: The primitive state at the start of the step.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The cleaning speed and the (predictor, corrector) damping factors.
    """
    if not config.mhd:
        zero = jnp.zeros((), dtype=primitive_state.dtype)
        return zero, (zero + 1.0, zero + 1.0)
    cleaning_speed = _glm_cleaning_speed(primitive_state, config, params, registered_variables)
    damping_factors = (
        _psi_damping_factor(cleaning_speed, 0.5 * dt, config, params),
        _psi_damping_factor(cleaning_speed, dt, config, params),
    )
    return cleaning_speed, damping_factors


def _update_ghost_cells(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Refill the ghost cells after a stage (no-op for the periodic-roll layout).

    Args:
        primitive_state: The (padded) primitive state.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The primitive state with up-to-date ghost cells.
    """
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)
    return primitive_state


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _evolve_state_vl2(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Advance the primitive state by one VL2 step: a donor-cell predictor to the
    half step, then the corrector with the van Leer limited PLM, or with donor
    cell when ``reconstruction=DONOR_CELL`` is set (``config.solver_config.limiter`` is not
    read). Runs on the Pallas backend when it supports the configuration and
    natively otherwise; with gravity always natively.

    Args:
        primitive_state: The primitive state ``W^n``.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data (unused, kept for the common signature).
        registered_variables: The registered variables.

    Returns:
        The primitive state ``W^{n+1}``.
    """
    corrector_piecewise_linear = config.solver_config.reconstruction != DONOR_CELL

    # Updates between steps (turbulent forcing, cooling, ...) may have changed
    # the physical cells without touching the halo.
    primitive_state = _update_ghost_cells(primitive_state, config, params, registered_variables)

    cleaning_speed, damping_factors = _cleaning_speed_and_damping(
        primitive_state,
        dt,
        config,
        params,
        registered_variables,
    )
    predictor_damping_factor, corrector_damping_factor = damping_factors

    # The gravity source needs every face's mass flux, which the fused Pallas
    # stage kernels never write out (they form the flux divergence in
    # registers), so with gravity the stages run natively.
    if not config.gravity_config.gravity and _vl2_pallas_supported(primitive_state, config):

        # --------------- ↓ Pallas stages ↓ ----------------

        # The Pallas stages take the base register in primitive form and
        # recompute U^n in-kernel, which saves a state-sized buffer. In the
        # predictor the base state is the stage state itself.
        half_step_primitive_state = _vl2_stage_pallas(
            primitive_state,
            base_primitive_state=None,
            stage_time_step=0.5 * dt,
            piecewise_linear=False,
            cleaning_speed=cleaning_speed,
            damping_factor=predictor_damping_factor,
            gamma=gamma,
            config=config,
            params=params,
            registered_variables=registered_variables,
        )
        half_step_primitive_state = _update_ghost_cells(
            half_step_primitive_state,
            config,
            params,
            registered_variables,
        )
        new_primitive_state = _vl2_stage_pallas(
            half_step_primitive_state,
            base_primitive_state=primitive_state,
            stage_time_step=dt,
            piecewise_linear=corrector_piecewise_linear,
            cleaning_speed=cleaning_speed,
            damping_factor=corrector_damping_factor,
            gamma=gamma,
            config=config,
            params=params,
            registered_variables=registered_variables,
        )

        # --------------- ↑ Pallas stages ↑ ----------------

        return _update_ghost_cells(new_primitive_state, config, params, registered_variables)

    # --------------- ↓ Native stages ↓ ----------------

    # The potential of each stage's initial state: Athena++ solves the
    # Poisson equation after every stage, so the predictor sees phi(rho^n) and
    # the corrector phi(rho^{n+1/2}).
    initial_potential = None
    if config.gravity_config.gravity:
        initial_potential = _vl2_gravitational_potential(
            primitive_state[registered_variables.density_index],
            config,
            params,
            registered_variables,
        )

    initial_conserved_state = _conserved_from_primitive_vl2(
        primitive_state,
        gamma,
        config,
        registered_variables,
    )
    half_step_primitive_state, half_step_end_potential = _vl2_stage(
        primitive_state,
        initial_conserved_state,
        stage_time_step=0.5 * dt,
        piecewise_linear=False,
        cleaning_speed=cleaning_speed,
        damping_factor=predictor_damping_factor,
        gamma=gamma,
        config=config,
        params=params,
        registered_variables=registered_variables,
        stage_potential=initial_potential,
        initial_potential=initial_potential,
    )
    half_step_primitive_state = _update_ghost_cells(
        half_step_primitive_state,
        config,
        params,
        registered_variables,
    )

    # The time-centred energy source has already solved for the potential of
    # the half-step density (before the floors, which only matters where they
    # act), so it is reused rather than solved again.
    half_step_potential = half_step_end_potential
    if config.gravity_config.gravity and half_step_potential is None:
        half_step_potential = _vl2_gravitational_potential(
            half_step_primitive_state[registered_variables.density_index],
            config,
            params,
            registered_variables,
        )

    new_primitive_state, _ = _vl2_stage(
        half_step_primitive_state,
        initial_conserved_state,
        stage_time_step=dt,
        piecewise_linear=corrector_piecewise_linear,
        cleaning_speed=cleaning_speed,
        damping_factor=corrector_damping_factor,
        gamma=gamma,
        config=config,
        params=params,
        registered_variables=registered_variables,
        stage_potential=half_step_potential,
        initial_potential=initial_potential,
    )

    # --------------- ↑ Native stages ↑ ----------------

    return _update_ghost_cells(new_primitive_state, config, params, registered_variables)


# -------------------------------------------------------------
# ===================== ↑ VL2 time step ↑ =====================
# -------------------------------------------------------------
