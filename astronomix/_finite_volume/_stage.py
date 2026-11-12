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
The split MHD scheme has a third form: the fluxes of the HLL5R solver with the
induction source (``_divergence/_induction_source.py``), which is added with
the flux divergence.
The stage has two halves: ``_flux_update`` (fluxes, correction, update) and
``_sources_and_conversion`` (sources, divergence control, conversion).
``fv_stage`` is the stage on the native backend;
``_native_stage_from_primitive_base`` is the native form of the fused Pallas
kernel (``_stage_pallas.py``). The time integrators (``_integrators.py``) are
sequences of stages.

A stage acts on all spatial axes, or on the axes it is given: the
dimensionally split scheme runs one stage per sweep, with the Hancock
predictor in the reconstruction. The spherical 1D grid enters at two points
only: the distances of the reconstruction and the flux divergence with its
area factors and pressure source (``_spherical_flux_divergence``).

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
    CARTESIAN,
    CONSTRAINED_TRANSPORT,
    DONOR_CELL,
    GHOST_CELLS,
    GLM,
    INDUCTION_SOURCE,
    LAX_FRIEDRICHS,
    POSITIVITY_HARD_FLOOR,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._divergence._constrained_transport import (
    _ct_field_update,
    _ct_fluxes_per_axis,
)
from astronomix._finite_volume._divergence._glm import _dedner_source
from astronomix._finite_volume._divergence._induction_source import _sweep_fluxes_and_induction_source
from astronomix._finite_volume._flux_correction import (
    POSITIVE,
    _corrected_fluxes,
    _first_order_flux_correction,
    _flux_divergence,
    _interior_mask,
    _positivity_failure_code,
    _spherical_flux_divergence,
)
from astronomix._finite_volume._gravity_coupling import (
    _add_athena_gravity_source,
    _vl2_gravitational_potential,
)
from astronomix._finite_volume._interface_frame import (
    FRAME_FIELD,
    FRAME_VELOCITY,
    BarotropicLaw,
    _barotropic_law,
    _interface_flux_from_states,
    _interface_frame_indices,
    _state_components_from_frame,
    _velocity_component_indices,
)
from astronomix._finite_volume._reconstruction import (
    FieldRows,
    HancockTerms,
    _cell_distances,
    _reconstructed_face_states,
)
from astronomix._fluid_equations._equation_of_state import (
    is_barotropic,
    pressure_of,
)
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
    primitive values (without ``E`` for a barotropic gas). The values are given
    as lists indexed like the state, so the function serves whole arrays
    (native) and register tiles (Pallas) alike.

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

    # A barotropic gas has no energy equation: the momenta are all there is to convert.
    if is_barotropic(config):
        return conserved_components

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
    state is not kept). A barotropic gas has no energy equation: its density
    floor is the only floor and the velocities are all there is to convert.

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

    if is_barotropic(config):
        for index in _velocity_component_indices(registered_variables):
            if index is not None:
                primitive_components[index] = conserved_components[index] * inverse_density
        return primitive_components

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


def _hancock_terms(
    axis: int,
    predictor_half_step,
    gamma,
    config: SimulationConfig,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
    barotropic_law: BarotropicLaw = None,
) -> HancockTerms:
    """
    The terms of the Hancock predictor of a sweep along ``axis`` (the split
    scheme; with the rows of the field for its MHD form).

    Args:
        axis: The spatial axis (1-based) of the sweep.
        predictor_half_step: Half the time step of the sweep.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data (the radii of the spherical 1D grid).
        registered_variables: The registered variables.
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The predictor terms.
    """
    # A state row outside the frame of the interface is a density carried with the gas.
    frame_indices = _interface_frame_indices(axis, config, registered_variables)
    carried_density_indices = tuple(
        index for index in range(registered_variables.num_vars) if index not in frame_indices
    )
    field_rows = None
    if config.mhd:
        field_rows = FieldRows(
            transverse_velocity_indices=frame_indices[FRAME_VELOCITY][1:],
            normal_field_index=frame_indices[FRAME_FIELD][0],
            transverse_field_indices=frame_indices[FRAME_FIELD][1:],
            pressure_exponent=gamma if barotropic_law is None else barotropic_law.exponent,
        )
    return HancockTerms(
        half_step=predictor_half_step,
        gamma=gamma,
        density_index=registered_variables.density_index,
        normal_velocity_index=_velocity_component_indices(registered_variables)[axis - 1],
        pressure_index=registered_variables.pressure_index,
        carried_density_indices=carried_density_indices,
        distances=_cell_distances(config, helper_data),
        sound_speed_squared_of=None if barotropic_law is None else barotropic_law.sound_speed_squared_of,
        field_rows=field_rows,
    )


def _carried_density_flux(mass_flux, left_density, right_density, left_value, right_value):
    """
    The flux of a density that is carried with the gas (the wind density
    tracer), as Athena++ treats passive scalars: the mass flux of the Riemann
    solver times the concentration on its upwind side. The same for every
    solver, and the tracer cannot exceed the gas density by the transport.

    Args:
        mass_flux: The mass flux through the interface.
        left_density: The gas density of the left state.
        right_density: The gas density of the right state.
        left_value: The carried density of the left state.
        right_value: The carried density of the right state.

    Returns:
        The flux of the carried density.
    """
    return mass_flux * jnp.where(mass_flux > 0.0, left_value / left_density, right_value / right_density)


def _interface_fluxes(
    primitive_state: STATE_TYPE,
    axis: int,
    reconstruction: int,
    riemann_solver: int,
    gamma,
    cleaning_speed,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    hancock: HancockTerms = None,
    barotropic_law: BarotropicLaw = None,
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
        hancock: The predictor terms of the split scheme (``None`` without predictor).
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The flux array, laid out like the state.
    """
    left_states, right_states = _reconstructed_face_states(
        primitive_state,
        axis,
        reconstruction,
        config.solver_config.limiter,
        hancock,
    )
    frame_indices = _interface_frame_indices(axis, config, registered_variables)

    # Velocity components the layout lacks enter the solver as zeros (and so does the pressure of a barotropic
    # gas, which the solver call fills in from the density).
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
        barotropic_law,
    )
    flux_components = _state_components_from_frame(flux_values, frame_indices, registered_variables.num_vars)

    # A state row outside the frame is a density carried with the gas.
    density_index = registered_variables.density_index
    for index, component in enumerate(flux_components):
        if component is None:
            flux_components[index] = _carried_density_flux(
                flux_components[density_index],
                left_states[density_index],
                right_states[density_index],
                left_states[index],
                right_states[index],
            )
    return jnp.stack(flux_components, axis=0)


def _fluxes_per_axis(
    primitive_state: STATE_TYPE,
    axes,
    reconstruction: int,
    riemann_solver: int,
    gamma,
    cleaning_speed,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    predictor_half_step=None,
    helper_data: HelperData = None,
):
    """
    The interface fluxes of the given spatial axes, by the divergence control:
    ``_interface_fluxes`` per axis for hydrodynamics and GLM-MHD, the fluxes
    with the face field as normal field for constrained transport (always all
    axes: the field update needs them).

    Args:
        primitive_state: The primitive state.
        axes: The spatial axes (1-based) the stage acts on.
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR`` (with ``config.solver_config.limiter``).
        riemann_solver: The Riemann solver.
        gamma: The adiabatic index.
        cleaning_speed: The GLM cleaning speed (GLM-MHD only).
        config: The simulation configuration.
        params: The simulation parameters (the barotropic law).
        registered_variables: The registered variables.
        predictor_half_step: Half the time step of the sweep (the split
            scheme), or ``None`` for face states at the time level of the state.
        helper_data: The helper data (the split scheme only).

    Returns:
        The interface fluxes (at ``i - 1/2``) per axis.
    """
    barotropic_law = _barotropic_law(params, config)
    if config.solver_config.divergence_control == CONSTRAINED_TRANSPORT:
        return _ct_fluxes_per_axis(
            primitive_state,
            reconstruction,
            config.solver_config.limiter,
            riemann_solver,
            gamma,
            registered_variables,
            barotropic_law,
        )
    fluxes_per_axis = []
    for axis in axes:
        hancock = None
        if predictor_half_step is not None:
            hancock = _hancock_terms(
                axis, predictor_half_step, gamma, config, helper_data, registered_variables, barotropic_law
            )
        fluxes_per_axis.append(
            _interface_fluxes(
                primitive_state,
                axis,
                reconstruction,
                riemann_solver,
                gamma,
                cleaning_speed,
                config,
                registered_variables,
                hancock,
                barotropic_law,
            )
        )
    return fluxes_per_axis


def _fluxes_and_induction_source(
    primitive_state: STATE_TYPE,
    axes,
    reconstruction: int,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    predictor_half_step=None,
    helper_data: HelperData = None,
):
    """
    The interface fluxes of the split MHD scheme (``divergence_control =
    INDUCTION_SOURCE``) along the given axes and the sum of their induction
    sources.

    Args:
        primitive_state: The primitive state.
        axes: The spatial axes (1-based) the stage acts on.
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR`` (with ``config.solver_config.limiter``).
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters (the barotropic law).
        registered_variables: The registered variables.
        predictor_half_step: Half the time step of the sweep, or ``None``
            for face states at the time level of the state.
        helper_data: The helper data.

    Returns:
        The interface fluxes (at ``i - 1/2``) per axis and the source array.
    """
    barotropic_law = _barotropic_law(params, config)
    fluxes_per_axis = []
    induction_source = 0.0
    for axis in axes:
        hancock = None
        if predictor_half_step is not None:
            hancock = _hancock_terms(
                axis, predictor_half_step, gamma, config, helper_data, registered_variables, barotropic_law
            )
        fluxes, source = _sweep_fluxes_and_induction_source(
            primitive_state, axis, reconstruction, gamma, config, registered_variables, hancock, barotropic_law
        )
        fluxes_per_axis.append(fluxes)
        induction_source = induction_source + source
    return fluxes_per_axis, induction_source


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
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    correction_mask=None,
    axes=None,
    predictor_half_step=None,
    helper_data: HelperData = None,
):
    """
    The first half of a stage: the interface fluxes, the first-order flux
    correction and the update ``U = U_base - beta dt L(W_stage)`` (with the
    induction source of the split MHD scheme in ``L``).

    Args:
        stage_primitive_state: The primitive state the fluxes are computed from.
        base_conserved_state: The conserved register the update starts from.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        reconstruction: The reconstruction of the stage (``DONOR_CELL`` or ``PIECEWISE_LINEAR``).
        cleaning_speed: The GLM cleaning speed (GLM-MHD only).
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters (the barotropic law).
        registered_variables: The registered variables.
        correction_mask: ``None`` for the stage's own correction attempts
            (with ``first_order_flux_correction``), or a float ``(1, *grid)``
            array that is larger than 0.5 in the cells whose faces take
            first-order LLF fluxes (the attempts then run outside the stage).
        axes: The spatial axes (1-based) the stage acts on (``None``: all).
        predictor_half_step: Half the time step of the sweep for the Hancock
            predictor of the split scheme (``None``: no predictor).
        helper_data: The helper data (the split scheme only: the radii of the
            spherical 1D grid).

    Returns:
        The updated conserved state and the interface fluxes (at ``i - 1/2``)
        per axis.
    """
    if axes is None:
        # A constrained-transport state has three axes in 2D as well (the third one cell wide); its fluxes and
        # its field update cover all of them.
        is_constrained_transport = config.solver_config.divergence_control == CONSTRAINED_TRANSPORT
        num_axes = stage_primitive_state.ndim - 1 if is_constrained_transport else config.dimensionality
        axes = tuple(range(1, num_axes + 1))

    induction_source = None
    if config.solver_config.divergence_control == INDUCTION_SOURCE:
        fluxes_per_axis, induction_source = _fluxes_and_induction_source(
            stage_primitive_state,
            axes,
            reconstruction,
            gamma,
            config,
            params,
            registered_variables,
            predictor_half_step,
            helper_data,
        )
    else:
        fluxes_per_axis = _fluxes_per_axis(
            stage_primitive_state,
            axes,
            reconstruction,
            config.solver_config.riemann_solver,
            gamma,
            cleaning_speed,
            config,
            params,
            registered_variables,
            predictor_half_step,
            helper_data,
        )

    has_mask = correction_mask is not None
    if has_mask or config.solver_config.first_order_flux_correction:
        first_order_fluxes = _fluxes_per_axis(
            stage_primitive_state,
            axes,
            DONOR_CELL,
            LAX_FRIEDRICHS,
            gamma,
            cleaning_speed,
            config,
            params,
            registered_variables,
        )
        if has_mask:
            fluxes_per_axis = _corrected_fluxes(
                fluxes_per_axis,
                first_order_fluxes,
                correction_mask[0] > 0.5,
                axes,
            )
        else:
            fluxes_per_axis = _first_order_flux_correction(
                fluxes_per_axis,
                first_order_fluxes,
                _interior_mask(stage_primitive_state, config),
                base_conserved_state,
                stage_time_step,
                axes,
                config,
                registered_variables,
            )

    if config.geometry == CARTESIAN:
        flux_divergence = _flux_divergence(fluxes_per_axis, axes, config)
    else:
        # Spherical 1D (the split scheme): one radial sweep.
        flux_divergence = _spherical_flux_divergence(
            fluxes_per_axis[0],
            pressure_of(stage_primitive_state, params, config, registered_variables),
            config,
            helper_data,
            registered_variables,
        )
    conserved_state = base_conserved_state - stage_time_step * flux_divergence
    if induction_source is not None:
        conserved_state = conserved_state - stage_time_step * induction_source
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
    elif config.mhd and config.solver_config.divergence_control == GLM:
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
    axes=None,
    predictor_half_step=None,
    helper_data: HelperData = None,
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
        axes: The spatial axes (1-based) the stage acts on (``None``: all).
        predictor_half_step: Half the time step of the sweep for the Hancock
            predictor of the split scheme (``None``: no predictor).
        helper_data: The helper data (the split scheme only).

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
        params,
        registered_variables,
        axes=axes,
        predictor_half_step=predictor_half_step,
        helper_data=helper_data,
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


def _sweep_stage(
    primitive_state: STATE_TYPE,
    axis: int,
    sweep_time_step,
    gamma,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    helper_data: HelperData = None,
) -> STATE_TYPE:
    """
    One sweep of a dimensionally split scheme: a one-axis stage from the state itself, with the predictor over
    half of its time step.

    Args:
        primitive_state: The primitive state, with up-to-date ghost cells.
        axis: The spatial axis (1-based) of the sweep.
        sweep_time_step: The time step of the sweep.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        helper_data: The helper data (radii of the spherical 1D grid).

    Returns:
        The primitive state after the sweep. Its ghost cells are not up to date.
    """
    conserved_state = _conserved_from_primitive(primitive_state, gamma, config, registered_variables)
    primitive_state, _ = fv_stage(
        primitive_state,
        conserved_state,
        stage_time_step=sweep_time_step,
        reconstruction=config.solver_config.reconstruction,
        # No cleaning scalar: a zero cleaning speed and no damping.
        cleaning_speed=0.0,
        damping_factor=1.0,
        gamma=gamma,
        config=config,
        params=params,
        registered_variables=registered_variables,
        axes=(axis,),
        predictor_half_step=0.5 * sweep_time_step,
        helper_data=helper_data,
    )
    return primitive_state


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
        params,
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
