"""
The stage of the finite-volume framework.

A stage advances a base state by the fluxes of the state it is evaluated on,

    U_new = U_base - beta dt L(W_stage) + sources,

and converts the result back to primitive variables. Its parts, in the order
of the calculation: the two states at every interface
(``_reconstruction.py``), the interface fluxes of the Riemann solvers
(``_riemann/``), the optional first-order flux correction
(``_flux_correction.py``), the flux divergence, Athena++'s gravity source
(``_gravity_coupling.py``) and the Dedner source of the GLM cleaning
(``_divergence/_glm.py``). ``fv_stage`` is the stage on the native backend;
``_native_stage_from_primitive_base`` is the native form of the fused Pallas
kernel (``_stage_pallas.py``). The time integrators (``_integrators.py``) are
sequences of stages. The stage with constrained transport is a function of
its own in ``_divergence/_constrained_transport.py``.

The file also holds what the parts share: the layout of the state in the
frame of an interface and the conversions between primitive and conserved
variables. The per-face physics is written as elementwise functions that the
Pallas kernel calls as well, so both backends evaluate the same expressions.
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
    GHOST_CELLS,
    HLL,
    HLLC,
    HLLD,
    LAX_FRIEDRICHS,
    POSITIVITY_HARD_FLOOR,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._divergence._glm import _dedner_source
from astronomix._finite_volume._flux_correction import (
    POSITIVE,
    _corrected_fluxes,
    _first_order_flux_correction,
    _first_order_fluxes,
    _interior_mask,
    _positivity_failure_code,
)
from astronomix._finite_volume._gravity_coupling import (
    _add_athena_gravity_source,
    _vl2_gravitational_potential,
)
from astronomix._finite_volume._reconstruction import _reconstructed_face_states
from astronomix._finite_volume._riemann._face_state import (
    HydroFaceState,
    MHDFaceState,
)
from astronomix._finite_volume._riemann._hllc import hllc_hydro_flux
from astronomix._finite_volume._riemann._hlld import hlld_flux
from astronomix._finite_volume._riemann._hlle import (
    hlle_hydro_flux,
    hlle_mhd_flux,
)
from astronomix._finite_volume._riemann._llf import (
    llf_hydro_flux,
    llf_mhd_flux,
)
from astronomix._geometry.boundaries import _boundary_handler


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
# ========================= ↓ Stage ↓ =========================
# -------------------------------------------------------------


def fv_stage(
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


# -------------------------------------------------------------
# ========================= ↑ Stage ↑ =========================
# -------------------------------------------------------------
