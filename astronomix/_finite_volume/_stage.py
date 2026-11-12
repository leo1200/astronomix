"""
The stage of the finite-volume framework.

A stage advances a base state by the fluxes of the state it is evaluated on,

    U_new = U_base - beta dt L(W_stage) + sources,

and converts the result back to primitive variables. Its parts, in the order
of the calculation: the two states at every interface
(``_reconstruction.py``), the interface fluxes of the Riemann solvers
(``_riemann/``), the optional first-order flux correction
(``_flux_correction.py``), the flux divergence, Athena++'s gravity source
(``_gravity_coupling.py``) and the divergence control: the Dedner source of
the GLM cleaning (``_divergence/_glm.py``) or the field update of the
constrained transport (``_divergence/_constrained_transport.py``). The
divergence control enters at two points, the interface fluxes
(``_fluxes_per_axis``) and the update of the field after the flux divergence.
The stage has two halves: ``_flux_update`` (fluxes, correction, update) and
``_sources_and_conversion`` (sources, divergence control, conversion).
``fv_stage`` is the stage on the native backend;
``_native_stage_from_primitive_base`` is the native form of the fused Pallas
kernel (``_stage_pallas.py``). The time integrators (``_integrators.py``) are
sequences of stages.

The file also holds the conversions between primitive and conserved
variables; the layout of the state in the frame of an interface is in
``_interface_frame.py``. The per-face physics is written as elementwise
functions that the Pallas kernel calls as well, so both backends evaluate the
same expressions.
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
    CONSTRAINED_TRANSPORT,
    DONOR_CELL,
    GHOST_CELLS,
    LAX_FRIEDRICHS,
    POSITIVITY_HARD_FLOOR,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._divergence._constrained_transport import (
    _ct_field_update,
    _ct_fluxes_per_axis,
)
from astronomix._finite_volume._divergence._glm import _dedner_source
from astronomix._finite_volume._flux_correction import (
    POSITIVE,
    _corrected_fluxes,
    _first_order_flux_correction,
    _flux_divergence,
    _interior_mask,
    _positivity_failure_code,
)
from astronomix._finite_volume._gravity_coupling import (
    _add_athena_gravity_source,
    _vl2_gravitational_potential,
)
from astronomix._finite_volume._interface_frame import (
    _interface_flux_from_states,
    _interface_frame_indices,
    _state_components_from_frame,
    _velocity_component_indices,
)
from astronomix._finite_volume._reconstruction import _reconstructed_face_states
from astronomix._geometry.boundaries import _boundary_handler


# -------------------------------------------------------------
# ============= ↓ Conserved <-> primitive variables ↓ ==========
# -------------------------------------------------------------


def _uses_floors(config: SimulationConfig) -> bool:
    """
    Return whether the density / pressure floors of the stage are active.

    The floors (off by default, as in AthenaPK) are switched on by the per-step
    hard floor, ``positivity_config.per_step_mode == POSITIVITY_HARD_FLOOR``;
    like AthenaPK, the stage then also applies them in its conversion to
    primitives.
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
def _conserved_from_primitive(
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
def _primitive_from_conserved(
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


def _interface_fluxes(
    primitive_state: STATE_TYPE,
    axis: int,
    reconstruction: int,
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
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR`` (with ``config.solver_config.limiter``).
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (MHD only).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The flux array, laid out like the state.
    """
    left_states, right_states = _reconstructed_face_states(
        primitive_state,
        axis,
        reconstruction,
        config.solver_config.limiter,
    )
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


def _fluxes_per_axis(
    primitive_state: STATE_TYPE,
    reconstruction: int,
    riemann_solver: int,
    gamma,
    cleaning_speed,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The interface fluxes of every spatial axis, by the divergence control:
    ``_interface_fluxes`` per axis for hydrodynamics and GLM-MHD, the fluxes
    with the face field as normal field for constrained transport.

    Args:
        primitive_state: The primitive state.
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR`` (with ``config.solver_config.limiter``).
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (GLM-MHD only).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The interface fluxes (at ``i - 1/2``) per axis.
    """
    if config.solver_config.divergence_control == CONSTRAINED_TRANSPORT:
        return _ct_fluxes_per_axis(
            primitive_state,
            reconstruction,
            config.solver_config.limiter,
            riemann_solver,
            gamma,
            registered_variables,
        )
    return [
        _interface_fluxes(
            primitive_state,
            axis,
            reconstruction,
            riemann_solver,
            gamma,
            cleaning_speed,
            config,
            registered_variables,
        )
        for axis in range(1, config.dimensionality + 1)
    ]


# -------------------------------------------------------------
# ================== ↑ Interface fluxes ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ========================= ↓ Stage ↓ =========================
# -------------------------------------------------------------


def _flux_update(
    stage_primitive_state: STATE_TYPE,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    reconstruction: int,
    cleaning_speed,
    gamma,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    correction_mask=None,
):
    """
    The first half of a stage: the interface fluxes, the first-order flux
    correction and the update ``U = U_base - beta dt L(W_stage)``.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_conserved_state: The conserved register the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        reconstruction: The reconstruction of the stage (``DONOR_CELL`` or ``PIECEWISE_LINEAR``).
        cleaning_speed: The GLM cleaning speed (GLM-MHD only).
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        correction_mask: ``None`` for the stage's own correction attempts
            (with ``first_order_flux_correction``), or a float ``(1, *grid)``
            array that is larger than 0.5 in the cells whose faces take
            first-order LLF fluxes (the attempts then run outside the stage).

    Returns:
        The updated conserved state and the interface fluxes (at ``i - 1/2``)
        per axis.
    """
    fluxes_per_axis = _fluxes_per_axis(
        stage_primitive_state,
        reconstruction,
        config.solver_config.riemann_solver,
        gamma,
        cleaning_speed,
        config,
        registered_variables,
    )

    has_mask = correction_mask is not None
    if has_mask or config.solver_config.first_order_flux_correction:
        first_order_fluxes = _fluxes_per_axis(
            stage_primitive_state,
            DONOR_CELL,
            LAX_FRIEDRICHS,
            gamma,
            cleaning_speed,
            config,
            registered_variables,
        )
        if has_mask:
            fluxes_per_axis = _corrected_fluxes(
                fluxes_per_axis,
                first_order_fluxes,
                correction_mask[0] > 0.5,
            )
        else:
            fluxes_per_axis = _first_order_flux_correction(
                fluxes_per_axis,
                first_order_fluxes,
                _interior_mask(stage_primitive_state, config),
                base_conserved_state,
                stage_time_step,
                config,
                registered_variables,
            )

    conserved_state = base_conserved_state - stage_time_step * _flux_divergence(
        fluxes_per_axis,
        config,
    )
    return conserved_state, fluxes_per_axis


def _sources_and_conversion(
    conserved_state: STATE_TYPE,
    fluxes_per_axis,
    stage_primitive_state: STATE_TYPE,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    damping_factor,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    stage_potential=None,
    initial_potential=None,
):
    """
    The second half of a stage: Athena++'s gravity source (when the integrator
    passes the potential of the stage's initial state), the divergence control
    (the Dedner source of the GLM cleaning, or the constrained-transport
    update of the face field with its cell-centred field) and the conversion
    back to primitives.

    Args:
        conserved_state: The conserved state after the flux update.
        fluxes_per_axis: The interface fluxes of the flux update.
        stage_primitive_state: The primitive state the fluxes were computed from.
        base_conserved_state: The conserved register the update started from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        damping_factor: The stage's psi damping factor (GLM-MHD only).
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        stage_potential: The potential of the stage's initial state, or
            ``None`` for a stage without the gravity source.
        initial_potential: The potential at the start of the step, ``t^n``
            (gravity with the time-centred energy source only).

    Returns:
        The primitive state after the stage, and the potential of the stage's
        new density if the time-centred energy source computed it (otherwise
        ``None``).
    """
    end_potential = None
    if stage_potential is not None:
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

    if config.solver_config.divergence_control == CONSTRAINED_TRANSPORT:
        conserved_state = _ct_field_update(
            conserved_state,
            fluxes_per_axis,
            stage_primitive_state,
            base_conserved_state,
            stage_time_step,
            config,
            registered_variables,
        )
    elif config.mhd:
        conserved_state = _dedner_source(
            conserved_state,
            stage_primitive_state,
            damping_factor,
            stage_time_step,
            config,
            registered_variables,
        )

    primitive_state = _primitive_from_conserved(
        conserved_state,
        gamma,
        config,
        params,
        registered_variables,
    )
    return primitive_state, end_potential


def fv_stage(
    stage_primitive_state: STATE_TYPE,
    base_conserved_state: STATE_TYPE,
    stage_time_step,
    reconstruction: int,
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
    One stage (native JAX): ``U = U_base - beta dt L(W_stage)``, the gravity
    source, the divergence control and the conversion back to primitives.
    Ghost cells are not updated here.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_conserved_state: The conserved register the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        reconstruction: The reconstruction of the stage (``DONOR_CELL`` or ``PIECEWISE_LINEAR``).
        cleaning_speed: The GLM cleaning speed (GLM-MHD only).
        damping_factor: The stage's psi damping factor (GLM-MHD only).
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        stage_potential: The potential of the stage's initial state (stages
            with Athena++'s gravity source only).
        initial_potential: The potential at the start of the step, ``t^n``
            (gravity with the time-centred energy source only).

    Returns:
        The primitive state after the stage, and the potential of the stage's
        new density if the time-centred energy source computed it (otherwise
        ``None``).
    """
    conserved_state, fluxes_per_axis = _flux_update(
        stage_primitive_state,
        base_conserved_state,
        stage_time_step,
        reconstruction,
        cleaning_speed,
        gamma,
        config,
        registered_variables,
    )
    return _sources_and_conversion(
        conserved_state,
        fluxes_per_axis,
        stage_primitive_state,
        base_conserved_state,
        stage_time_step,
        damping_factor,
        gamma,
        config,
        params,
        registered_variables,
        stage_potential,
        initial_potential,
    )


def _native_stage_from_primitive_base(
    stage_primitive_state: STATE_TYPE,
    base_primitive_state: STATE_TYPE,
    stage_time_step,
    reconstruction: int,
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
    path runs the correction attempts outside the kernel). Like the kernel it
    serves hydrodynamics and GLM-MHD, without the stage's gravity source.

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_primitive_state: The primitive state ``W^n`` whose conserved form
            the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        reconstruction: The reconstruction of the stage (``DONOR_CELL`` or ``PIECEWISE_LINEAR``).
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
    base_conserved_state = _conserved_from_primitive(
        base_primitive_state,
        gamma,
        config,
        registered_variables,
    )
    conserved_state, fluxes_per_axis = _flux_update(
        stage_primitive_state,
        base_conserved_state,
        stage_time_step,
        reconstruction,
        cleaning_speed,
        gamma,
        config,
        registered_variables,
        correction_mask=correction_mask,
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
    primitive_state, _ = _sources_and_conversion(
        conserved_state,
        fluxes_per_axis,
        stage_primitive_state,
        base_conserved_state,
        stage_time_step,
        damping_factor,
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
