"""
The gravity coupling of the VL2 scheme, as Athena++ couples it: every stage
adds the source of ``_add_athena_gravity_source``, built from the potential of
the stage's initial state (``_vl2_gravitational_potential``) and the stage's
Riemann mass fluxes, optionally with the time-centred energy source of Mullen,
Hanawa & Gammie (2021). A barotropic gas has no energy equation: the coupling
then acts on the momenta alone.

``RK2_SSP`` and ``MUSCL_HANCOCK`` take gravity from the explicit sources of the
modules (``_modules/_explicit_sources.py``), as the finite-difference schemes
do.
"""

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    STATE_TYPE,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._modules._gravity._gravity import (
    _component_index,
    _compute_total_potential,
)
from astronomix._stencil_operations._stencil_operations import _shift

# -------------------------------------------------------------
# ============== ↓ Coupling inside the VL2 stage ↓ ============
# -------------------------------------------------------------


def _vl2_gravitational_potential(
    density,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    sink_potential=None,
):
    """
    The total gravitational potential (self-gravity, external and sink
    particles) of a density field of the VL2 scheme.

    Args:
        density: The density field.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.
        sink_potential: The potential of the sink particles, added to the
            total (``None``: no sink particles).

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
        sink_potential=sink_potential,
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
