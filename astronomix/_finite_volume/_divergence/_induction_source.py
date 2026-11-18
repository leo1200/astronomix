"""
The induction source of the split MHD scheme and its parabolic divergence cleaning.

The split scheme with ``divergence_control = INDUCTION_SOURCE`` keeps a cell-centred field without a constraint.
Of the divergence terms of Powell et al. (1999) it carries the one of the induction equation only (Waagan 2009;
Waagan, Federrath & Klingenberg 2011),

    dB/dt + div(u B - B u) = -u div B,

so that mass, momentum and energy stay in conservation form. In a sweep the term has two parts. Where the normal
field jumps at an interface, the HLL5R solver returns two fluxes of the field, one for each neighbour (Bouchut,
Klingenberg & Waagan 2010, appendix). Inside a cell the slope of the normal field gives the source
``u dB_n/dn``. Both parts are one source per cell here, next to the flux for the left neighbour:

    U_i <- U_i - dt/h (F_l[i+1/2] - F_l[i-1/2]) - dt S_i,
    S_i = ((F_l - F_r)[i-1/2] + u_i (B_n,i^+ - B_n,i^-)) / h    (rows of the field only),

with the edge values ``B_n,i^-`` and ``B_n,i^+`` of the cell and its velocity ``u_i`` after the predictor.

What remains of the divergence is diffused once per step (``_parabolic_cleaning``), as in the eight-wave unit
of FLASH.
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._interface_frame import (
    FRAME_FIELD,
    FRAME_VELOCITY,
    BarotropicLaw,
    _interface_frame_indices,
    _mhd_face_state,
    _state_components_from_frame,
)
from astronomix._finite_volume._reconstruction import (
    HancockTerms,
    _reconstructed_face_states,
)
from astronomix._finite_volume._riemann._face_state import MHDFaceState
from astronomix._finite_volume._riemann._hll5r import hll5r_flux


def _hll5r_fluxes(left: MHDFaceState, right: MHDFaceState, gamma, barotropic_law: BarotropicLaw = None):
    """
    The pair of HLL5R fluxes of the configured equation of state.

    Args:
        left: The states left of the interfaces.
        right: The states right of the interfaces.
        gamma: The adiabatic index (ideal gas only).
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The flux for the left neighbour and the flux for the right neighbour.
    """
    if barotropic_law is None:
        return hll5r_flux(left, right, gamma)
    return hll5r_flux(left, right, barotropic_law.exponent, barotropic_law.sound_speed_squared_of)


def _source_of_cells(left: MHDFaceState, right: MHDFaceState, flux_for_left, flux_for_right, axis: int, spacing):
    """
    The induction source of a sweep in the frame of its interfaces.

    Args:
        left: The states left of the interfaces ``i - 1/2``: the right edge values of the cells ``i - 1``.
        right: The states right of the interfaces ``i - 1/2``: the left edge values of the cells ``i``.
        flux_for_left: The HLL5R flux for the left neighbour.
        flux_for_right: The HLL5R flux for the right neighbour.
        axis: The spatial axis (1-based) of the sweep.
        spacing: The cell width.

    Returns:
        The source of the field (normal, transverse 1, transverse 2) per cell.
    """
    array_axis = axis - 1

    def right_edge_value(left_values):
        return jnp.roll(left_values, -1, axis=array_axis)

    normal_field_difference = right_edge_value(left.field_normal) - right.field_normal
    sources = []
    for position in range(FRAME_VELOCITY.start, FRAME_VELOCITY.stop):
        field_position = position + FRAME_FIELD.start - FRAME_VELOCITY.start
        # The velocity of the cell after the predictor is the mean of its two edge values.
        velocity = 0.5 * (right_edge_value(left[position]) + right[position])
        flux_difference = flux_for_left[field_position] - flux_for_right[field_position]
        sources.append((flux_difference + velocity * normal_field_difference) / spacing)
    return sources


def _sweep_fluxes_and_induction_source(
    primitive_state: STATE_TYPE,
    axis: int,
    reconstruction: int,
    gamma,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    hancock: HancockTerms = None,
    barotropic_law: BarotropicLaw = None,
):
    """
    The interface fluxes of a sweep of the split MHD scheme along ``axis`` and its induction source.

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based) of the sweep.
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR`` (with ``config.solver_config.limiter``).
        gamma: The adiabatic index.
        config: The simulation configuration.
        registered_variables: The registered variables.
        hancock: The predictor terms of the sweep (``None`` without predictor).
        barotropic_law: ``None`` for the ideal gas, or the law of a barotropic gas.

    Returns:
        The flux array (``flux[:, i]`` is the flux for the left neighbour of the interface ``i - 1/2``) and the
        source array, both laid out like the state.
    """
    left_states, right_states = _reconstructed_face_states(
        primitive_state,
        axis,
        reconstruction,
        config.solver_config.limiter,
        hancock,
    )
    frame_indices = _interface_frame_indices(axis, config, registered_variables)
    left = _mhd_face_state(left_states, frame_indices, barotropic_law)
    right = _mhd_face_state(right_states, frame_indices, barotropic_law)
    flux_for_left, flux_for_right = _hll5r_fluxes(left, right, gamma, barotropic_law)

    zero = jnp.zeros_like(left.density)
    field_sources = _source_of_cells(left, right, flux_for_left, flux_for_right, axis, config.grid_spacing)
    source_values = (zero,) * FRAME_FIELD.start + tuple(field_sources) + (zero,)

    num_vars = registered_variables.num_vars
    fluxes = jnp.stack(_state_components_from_frame(flux_for_left, frame_indices, num_vars), axis=0)
    source = jnp.stack(_state_components_from_frame(source_values, frame_indices, num_vars), axis=0)
    return fluxes, source


def _central_difference(values, array_axis: int):
    """The central difference ``(f[i+1] - f[i-1]) / 2`` along an array axis."""
    return 0.5 * (jnp.roll(values, -1, axis=array_axis) - jnp.roll(values, 1, axis=array_axis))


def _parabolic_cleaning(
    primitive_state: STATE_TYPE,
    signal_speed,
    dt,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    One explicit step of the diffusion of the divergence, ``B <- B + dt eta grad(div B)``, with central
    differences of the cell-centred field and ``eta = lambda h / (2 d)`` in ``d`` dimensions (FLASH user guide,
    eight-wave unit: ``eta = (lambda / 2) (1/dx + 1/dy + 1/dz)^-1``). The step is stable for
    ``lambda dt / h <= 4``. The pressure is left as it is.

    Args:
        primitive_state: The primitive state.
        signal_speed: The speed ``lambda``: the largest signal speed of the step.
        dt: The time step.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The primitive state with the cleaned field. Its ghost cells are not up to date.
    """
    spacing = config.grid_spacing
    field_indices = tuple(registered_variables.magnetic_index)[: config.dimensionality]
    divergence = 0.0
    for array_axis, index in enumerate(field_indices):
        divergence = divergence + _central_difference(primitive_state[index], array_axis) / spacing

    diffusivity = signal_speed * spacing / (2.0 * config.dimensionality)
    for array_axis, index in enumerate(field_indices):
        divergence_gradient = _central_difference(divergence, array_axis) / spacing
        primitive_state = primitive_state.at[index].add(dt * diffusivity * divergence_gradient)
    return primitive_state
