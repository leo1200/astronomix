"""
GLM divergence cleaning of the finite-volume stage (the VL2 finite-volume MHD scheme).

AthenaPK controls the divergence of the cell-centred magnetic field with the
hyperbolic-parabolic generalised Lagrange multiplier (GLM) method of Dedner et
al. (2002). A scalar psi carries divergence errors away at the cleaning speed
``c_h`` — this happens inside the Riemann solver, which solves the decoupled
``(B_normal, psi)`` subsystem exactly — and psi is damped every stage by the
source term implemented here. As in AthenaPK, ``c_h`` is the largest
hyperbolic signal speed on the grid, so the cleaning waves never restrict the
time step.
"""

# typing
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._timestep_estimation._timestep_estimator import (
    _maximum_signal_speed_vl2,
)


def _glm_cleaning_speed(
    primitive_state: STATE_TYPE,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> Float[Array, ""]:
    """
    The hyperbolic divergence-cleaning speed ``c_h`` of the coming step: the
    largest signal speed ``|v_d| + c_fast,d`` of the state at its start.

    Args:
        primitive_state: The primitive state at the start of the step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The cleaning speed ``c_h``.
    """
    return _maximum_signal_speed_vl2(primitive_state, config, params, registered_variables)


def _psi_damping_factor(
    cleaning_speed,
    stage_time_step,
    config: SimulationConfig,
    params: SimulationParams,
):
    """
    The parabolic damping factor of psi for one stage,
    ``exp(-alpha * c_h * beta dt / dx)`` (Mignone & Tzeferacos 2010, eq. 27),
    where ``alpha`` (``params.glm_alpha``) is the ratio of the diffusive to the
    advective time scale of the cleaning.

    Args:
        cleaning_speed: The cleaning speed ``c_h``.
        stage_time_step: The stage's time-step weight ``beta * dt``.
        config: The simulation configuration.
        params: The simulation parameters.

    Returns:
        The multiplicative damping factor.
    """
    return jnp.exp(-params.glm_alpha * cleaning_speed * stage_time_step / config.grid_spacing)


def _central_difference(field, axis: int):
    """Return the undivided central difference ``field[i+1] - field[i-1]`` along an axis."""
    return jnp.roll(field, -1, axis=axis) - jnp.roll(field, 1, axis=axis)


def _extended_dedner_source_terms(
    field_components,
    field_differences,
    psi_differences,
    grid_spacing,
):
    """
    The divergence ``div B`` and ``B . grad(psi)`` of the extended Dedner
    source (Dedner et al. 2002) from second-order central differences. The
    function is elementwise, so the native path (whole arrays) and the Pallas
    stage kernel (register tiles) share it; each supplies the differences from
    its own stencil.

    Args:
        field_components: The magnetic field components ``(B_x, B_y, B_z)`` of
            the stage's starting state in the cell.
        field_differences: For every active axis ``d``, the undivided central
            difference ``B_d[i+1] - B_d[i-1]`` along ``d``.
        psi_differences: For every active axis ``d``, the undivided central
            difference ``psi[i+1] - psi[i-1]`` along ``d``.
        grid_spacing: The cell width.

    Returns:
        ``(div B, B . grad(psi))``.
    """
    # Inactive axes contribute nothing. The factor 1/2 of the central
    # derivative is applied once to the sums.
    divergence_sum = 0.0
    field_dot_psi_gradient = 0.0
    for axis, (field_difference, psi_difference) in enumerate(
        zip(field_differences, psi_differences)
    ):
        divergence_sum = divergence_sum + field_difference / grid_spacing
        field_dot_psi_gradient = field_dot_psi_gradient + (
            field_components[axis] * psi_difference / grid_spacing
        )
    return 0.5 * divergence_sum, 0.5 * field_dot_psi_gradient


def _dedner_source(
    conserved_state: STATE_TYPE,
    primitive_state: STATE_TYPE,
    damping_factor,
    stage_time_step,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Apply AthenaPK's Dedner source to the freshly updated conserved state.

    The plain variant (default; AthenaPK's ``dedner_plain``) only damps psi.
    With ``config.solver_config.glm_extended_source`` (AthenaPK's ``dedner_extended``) the
    non-conservative sources ``-(div B) B`` (momentum) and ``-B . grad(psi)``
    (energy) of Dedner et al. (2002) are added as well, evaluated with central
    differences of the primitive state the stage started from, as in AthenaPK.
    The Pallas stage kernel (``_stage_pallas.py``) applies the same source with
    the shared ``_extended_dedner_source_terms``.

    Args:
        conserved_state: The conserved state after the stage's flux update.
        primitive_state: The primitive state at the start of the stage.
        damping_factor: The stage's psi damping factor (``_psi_damping_factor``).
        stage_time_step: The stage's time-step weight ``beta * dt``.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The conserved state with the source applied.
    """
    psi_index = registered_variables.magnetic_psi_index

    if config.solver_config.glm_extended_source:
        magnetic_indices = tuple(registered_variables.magnetic_index)
        momentum_indices = tuple(registered_variables.momentum_index)
        field_components = [primitive_state[index] for index in magnetic_indices]
        psi = primitive_state[psi_index]

        # The fields carry no variable axis, so spatial axis d is array axis d.
        field_differences = [
            _central_difference(field_components[axis], axis)
            for axis in range(config.dimensionality)
        ]
        psi_differences = [
            _central_difference(psi, axis) for axis in range(config.dimensionality)
        ]
        magnetic_divergence, field_dot_psi_gradient = _extended_dedner_source_terms(
            field_components,
            field_differences,
            psi_differences,
            config.grid_spacing,
        )

        for momentum_index, field_component in zip(momentum_indices, field_components):
            conserved_state = conserved_state.at[momentum_index].add(
                -stage_time_step * magnetic_divergence * field_component
            )
        conserved_state = conserved_state.at[registered_variables.energy_index].add(
            -stage_time_step * field_dot_psi_gradient
        )

    return conserved_state.at[psi_index].multiply(damping_factor)


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
