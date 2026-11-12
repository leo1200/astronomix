"""
The gravity couplings of the finite-volume schemes.

VL2 couples as Athena++ does: every stage adds the source of
``_add_athena_gravity_source``, built from the potential of the stage's initial
state (``_vl2_gravitational_potential``) and the stage's Riemann mass fluxes,
optionally with the time-centred energy source of Mullen, Hanawa & Gammie
(2021).

``RK2_SSP`` and ``MUSCL_HANCOCK`` couple first order in time, around the step:
the source of ``_modules/_gravity/_gravity.py`` is built from the state before
the step (``_gravity_source_presolve``) and added to the state after it
(``_apply_gravity_source``).

A barotropic gas has no energy equation: both couplings then act on the
momenta alone.
"""

# jax
import jax.numpy as jnp

# typing
from typing import Union
from jaxtyping import Array, Float

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    GHOST_CELLS,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    primitive_state_from_conserved,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._modules._gravity._gravity import (
    _component_index,
    _compute_total_potential,
)
from astronomix._modules._time_integrator_sources import _time_integrator_sources
from astronomix._stencil_operations._stencil_operations import _shift

# -------------------------------------------------------------
# ============== ↓ Coupling inside the VL2 stage ↓ ============
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


def _add_athena_gravity_source(
    conserved_state: STATE_TYPE,
    stage_density: FIELD_TYPE,
    mass_fluxes_per_axis,
    stage_potential: FIELD_TYPE,
    work_potential: FIELD_TYPE,
    stage_time_step,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Add Athena++'s self-gravity source of one VL2 stage to the conserved state
    (Athena++: ``HydroSourceTerms::SelfGravity``, ``self_gravity.cpp``).

    With the face accelerations ``g_{i-1/2} = -(phi_i - phi_{i-1}) / dx``, cell
    ``i`` receives along every axis

        (rho v)_i += (beta dt / 2) rho_i (g_{i-1/2} + g_{i+1/2})
        E_i       += (beta dt / 2) (F_{i-1/2} g_{i-1/2} + F_{i+1/2} g_{i+1/2})

    where ``rho`` and the momentum potential are those of the stage's initial
    state and ``F`` are the stage's Riemann mass fluxes, the same fluxes that
    update the density. The momentum source is the cell-centred
    ``rho grad(phi)`` of Mullen, Hanawa & Gammie (2021); with the potential of
    the discrete Laplacian (``GravityConfig.poisson_discrete_laplacian``) it is
    the divergence of a discrete gravitational stress tensor. The energy
    source is ``-div(F phi_face) + phi_i div(F)`` with the face potential
    ``(phi_i + phi_{i+1}) / 2``, so summed over the domain it equals
    ``sum phi_i * (density change)``: total energy is conserved for the
    time-continuous update. The expressions follow Athena++'s arithmetic, axis
    by axis, so the two codes agree to round-off. A barotropic gas receives
    the momentum source only.

    Args:
        conserved_state: The conserved state after the stage's flux update.
        stage_density: The density of the stage's initial (primitive) state.
        mass_fluxes_per_axis: The stage's density fluxes per axis, at the faces
            ``i - 1/2``.
        stage_potential: The potential of the stage's initial state (momentum
            source).
        work_potential: The potential the mass fluxes are charged with (energy
            source): ``stage_potential`` in Athena++, the time-centred average
            with ``GravityConfig.time_centred_energy_source``.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state with the gravity source added.
    """
    half_step_over_spacing = 0.5 * stage_time_step / config.grid_spacing
    energy_index = registered_variables.energy_index
    has_energy_equation = not is_barotropic(config)

    for spatial_axis, mass_flux in enumerate(mass_fluxes_per_axis):
        momentum_index = _component_index(registered_variables.momentum_index, spatial_axis)

        # The face accelerations times dx at i - 1/2 and i + 1/2 (Athena++'s
        # dpl and dpr).
        stage_acceleration_left = -(
            stage_potential - _shift(stage_potential, 1, axis=spatial_axis)
        )
        stage_acceleration_right = -(
            _shift(stage_potential, -1, axis=spatial_axis) - stage_potential
        )
        momentum_change = (
            half_step_over_spacing
            * stage_density
            * (stage_acceleration_left + stage_acceleration_right)
        )
        conserved_state = conserved_state.at[momentum_index].add(momentum_change)

        if has_energy_equation:
            work_acceleration_left = -(
                work_potential - _shift(work_potential, 1, axis=spatial_axis)
            )
            work_acceleration_right = -(
                _shift(work_potential, -1, axis=spatial_axis) - work_potential
            )
            energy_change = half_step_over_spacing * (
                mass_flux * work_acceleration_left
                + _shift(mass_flux, -1, axis=spatial_axis) * work_acceleration_right
            )
            conserved_state = conserved_state.at[energy_index].add(energy_change)

    return conserved_state


# -------------------------------------------------------------
# ============== ↑ Coupling inside the VL2 stage ↑ ============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ Coupling around the step ↓ ===============
# -------------------------------------------------------------


def _barotropic_gravity_source(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The gravity source of one time step for a barotropic gas: the momenta
    receive ``dt rho g`` with the centred acceleration
    ``g_i = -(phi_{i+1} - phi_{i-1}) / (2 dx)``, the same momentum source as
    with an energy equation; there is no energy row to charge.

    Args:
        primitive_state: The primitive state before the step.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The source, as a change of the conserved state over the step.
    """
    density = primitive_state[registered_variables.density_index]
    potential = _vl2_gravitational_potential(density, config, params, registered_variables)
    source = jnp.zeros_like(primitive_state)
    for spatial_axis in range(config.dimensionality):
        momentum_index = _component_index(registered_variables.momentum_index, spatial_axis)
        potential_difference = _shift(potential, -1, axis=spatial_axis) - _shift(potential, 1, axis=spatial_axis)
        acceleration = -potential_difference / (2 * config.grid_spacing)
        source = source.at[momentum_index].set(dt * density * acceleration)
    return source


def _gravity_source_presolve(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    The gravity source of one time step, built from the state before the
    step.

    Args:
        primitive_state: The primitive state before the step.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.

    Returns:
        The source, as a change of the conserved state over the step.
    """
    if is_barotropic(config):
        return _barotropic_gravity_source(primitive_state, dt, config, params, registered_variables)
    return _time_integrator_sources(
        conserved_state_from_primitive(primitive_state, gamma, config, registered_variables),
        None,
        None,
        dt,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
    )


def _apply_gravity_source(
    primitive_state: STATE_TYPE,
    gravity_source: STATE_TYPE,
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Add the source of ``_gravity_source_presolve`` to the state after the step.

    Args:
        primitive_state: The primitive state after the step.
        gravity_source: The source, as a change of the conserved state.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The primitive state with the source added, ghost cells updated.
    """
    if is_barotropic(config):
        # The source changes the momenta only, so each velocity takes its share over the density (the source of
        # the density row is zero).
        primitive_state = primitive_state + gravity_source / primitive_state[registered_variables.density_index]
    else:
        conserved_state = conserved_state_from_primitive(primitive_state, gamma, config, registered_variables)
        conserved_state = conserved_state + gravity_source
        primitive_state = primitive_state_from_conserved(conserved_state, gamma, config, registered_variables)
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(primitive_state, config, registered_variables, params)
    return primitive_state


# -------------------------------------------------------------
# ================ ↑ Coupling around the step ↑ ===============
# -------------------------------------------------------------
