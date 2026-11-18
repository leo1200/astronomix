"""
Explicit ohmic resistivity: the resistive electric field ``E = eta J``, ``J = curl B``, in every scheme.

Faraday's law with the resistive field,

    dB/dt = -curl(eta J) = eta laplacian(B)      (for div B = 0, constant eta),

and the energy it moves, ``dE/dt = -div(eta J x B)``, of which ``eta J^2`` heats the gas.

* Constrained transport (both finite-difference schemes, VL2 with ``CONSTRAINED_TRANSPORT``): ``eta J`` is an edge
  electromotive force whose curl updates the face field (``resistive_face_increment``), so the discrete
  divergence stays zero to round-off. The cell-centred field is the sixth-order face-to-centre interpolation of
  the face field, ``J`` its curl by sixth-order central differences, ``eta J`` is carried to the edges by two
  fourth-order centre-to-face interpolations and its curl taken with ``finite_difference_int6``, the operator of
  the constrained-transport curl.
* A cell-centred field (VL2 with GLM, the split scheme with the induction source): ``-curl(eta J)`` by central
  differences enters the field rows of the stage (``resistive_source``).
* Energy (ideal gas, ``resistive_source``): the production finite-difference scheme rebuilds its cell-centred
  field from the faces with the pressure held, which removes the field energy the resistivity takes; it receives
  the ohmic heating ``eta J^2``. Every other scheme keeps the total energy through the field update and receives
  the divergence of the resistive Poynting flux, ``-div(eta J x B)``.

Axes beyond the dimensionality carry no derivative (the field there is constant).
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    WENO_FLUX_SPLITTING,
)

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig

# astronomix functions
from astronomix._fluid_equations._equation_of_state import is_barotropic
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import (
    interp_center_to_face,
    interp_face_to_center,
)
from astronomix._stencil_operations._stencil_operations import _stencil_add

# Axes of a single field array (no leading variable axis).
XAXIS = 0
YAXIS = 1
ZAXIS = 2


def _central_first_derivative(field, axis, grid_spacing, dimensionality):
    """Sixth-order central first derivative along ``axis``; zero along an axis beyond the dimensionality."""
    if axis >= dimensionality:
        return 0.0
    return _stencil_add(
        field,
        indices=(3, 2, 1, -1, -2, -3),
        factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
        axis=axis,
    ) / (60.0 * grid_spacing)


def _curl(field_x, field_y, field_z, grid_spacing, dimensionality):
    """The curl of a cell-centred vector field by sixth-order central differences."""
    return (
        _central_first_derivative(field_z, YAXIS, grid_spacing, dimensionality)
        - _central_first_derivative(field_y, ZAXIS, grid_spacing, dimensionality),
        _central_first_derivative(field_x, ZAXIS, grid_spacing, dimensionality)
        - _central_first_derivative(field_z, XAXIS, grid_spacing, dimensionality),
        _central_first_derivative(field_y, XAXIS, grid_spacing, dimensionality)
        - _central_first_derivative(field_x, YAXIS, grid_spacing, dimensionality),
    )


def _to_edge(field, first_axis, second_axis, dimensionality):
    """A cell-centred field carried to the edge between two axes (fourth order), along the axes that exist."""
    for axis in (first_axis, second_axis):
        if axis < dimensionality:
            field = interp_center_to_face(field, axis)
    return field


def _face_difference(edge_field, axis, dimensionality):
    """The difference of an edge field across the cell along ``axis`` (``finite_difference_int6``); zero along
    an axis beyond the dimensionality."""
    if axis >= dimensionality:
        return 0.0
    return finite_difference_int6(edge_field, axis)


@partial(jax.jit, static_argnames=["dimensionality"])
def resistive_face_increment(
    bx_interface: jnp.ndarray,
    by_interface: jnp.ndarray,
    bz_interface: jnp.ndarray,
    eta,
    dt_tilde,
    grid_spacing: float,
    dimensionality: int,
):
    """
    Resistive increment of the three face fields over ``dt_tilde``: minus the curl of the edge electromotive
    force ``eta J``.

    Args:
        bx_interface: The face field B_x (index ``i`` is the ``i+1/2`` face along x); along an axis beyond the
            dimensionality the cell value.
        by_interface: The face field B_y (index ``j`` is the ``j+1/2`` face along y).
        bz_interface: The face field B_z (index ``k`` is the ``k+1/2`` face along z).
        eta: The ohmic diffusivity.
        dt_tilde: The (stage) time step.
        grid_spacing: The cell size (cubic cells).
        dimensionality: The number of spatial axes with derivatives.

    Returns:
        ``(increment_bx, increment_by, increment_bz)``.
    """
    face_fields = (bx_interface, by_interface, bz_interface)
    magnetic_field_x, magnetic_field_y, magnetic_field_z = (
        interp_face_to_center(face_field, axis) if axis < dimensionality else face_field
        for axis, face_field in enumerate(face_fields)
    )
    current_x, current_y, current_z = _curl(
        magnetic_field_x, magnetic_field_y, magnetic_field_z, grid_spacing, dimensionality
    )

    # Edge-centred resistive electromotive force: E_x on the x-edges (j+1/2, k+1/2), and so on.
    emf_x = eta * _to_edge(current_x, YAXIS, ZAXIS, dimensionality)
    emf_y = eta * _to_edge(current_y, XAXIS, ZAXIS, dimensionality)
    emf_z = eta * _to_edge(current_z, XAXIS, YAXIS, dimensionality)

    # dB/dt = -curl E, each component landing on its own face: the int6 difference along y (z) maps j+1/2 to j
    # (k+1/2 to k), onto the x-face.
    dt_over_dx = dt_tilde / grid_spacing
    increment_bx = -dt_over_dx * (
        _face_difference(emf_z, YAXIS, dimensionality) - _face_difference(emf_y, ZAXIS, dimensionality)
    )
    increment_by = -dt_over_dx * (
        _face_difference(emf_x, ZAXIS, dimensionality) - _face_difference(emf_z, XAXIS, dimensionality)
    )
    increment_bz = -dt_over_dx * (
        _face_difference(emf_y, XAXIS, dimensionality) - _face_difference(emf_x, YAXIS, dimensionality)
    )
    return increment_bx, increment_by, increment_bz


def _cell_centred_field_scheme(config: SimulationConfig) -> bool:
    """Whether the scheme's field is cell-centred (finite volume without constrained transport)."""
    return config.solver_mode == FINITE_VOLUME and config.solver_config.divergence_control != CONSTRAINED_TRANSPORT


def resistive_source_active(config: SimulationConfig) -> bool:
    """Whether resistivity enters the explicit sources: the energy of an ideal gas, and the field of the
    finite-volume schemes (the finite-difference schemes have it in their constrained-transport update)."""
    return config.diffusion_config.resistivity and (
        not is_barotropic(config) or config.solver_mode == FINITE_VOLUME
    )


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def resistive_source(primitive_state, params, config, registered_variables):
    """
    The resistive source rate of a stage (per unit time) in the layout of the state: the field rows of a
    finite-volume scheme (minus the curl of ``eta J``; with constrained transport the face increment and its
    face average) and the energy row of an ideal gas (see the module docstring).

    Args:
        primitive_state: The primitive state of the stage.
        params: The simulation parameters (``resistivity``).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The source rate, laid out like the state.
    """
    eta = params.resistivity
    grid_spacing = config.grid_spacing
    # The layout of VL2 with constrained transport has three spatial axes in every dimensionality; an axis of
    # one cell carries no difference.
    dimensionality = primitive_state.ndim - 1
    magnetic_index = tuple(registered_variables.magnetic_index)
    magnetic_field = [primitive_state[index] for index in magnetic_index]
    current = _curl(*magnetic_field, grid_spacing, dimensionality)
    electric_field = [eta * component for component in current]

    source = jnp.zeros_like(primitive_state)
    if _cell_centred_field_scheme(config):
        field_rate = _curl(*electric_field, grid_spacing, dimensionality)
        for index, rate in zip(magnetic_index, field_rate):
            source = source.at[index].set(-rate)
    elif config.solver_mode == FINITE_VOLUME:
        # VL2 with constrained transport: the faces hold the value at i - 1/2; the increment works on the
        # value at i + 1/2. The cell-centred field is the face average, so its rate is the average of the
        # face rates.
        face_index = tuple(registered_variables.interface_magnetic_field_index)
        upper_faces = [jnp.roll(primitive_state[index], -1, axis=axis) for axis, index in enumerate(face_index)]
        upper_rates = resistive_face_increment(*upper_faces, eta, 1.0, grid_spacing, dimensionality)
        for axis, (index, rate) in enumerate(zip(face_index, upper_rates)):
            face_rate = jnp.roll(rate, 1, axis=axis)
            source = source.at[index].set(face_rate)
            source = source.at[magnetic_index[axis]].set(0.5 * (face_rate + rate))

    if not is_barotropic(config):
        if config.solver_mode == FINITE_DIFFERENCE and config.solver_config.scheme == WENO_FLUX_SPLITTING:
            energy_rate = eta * sum(component * component for component in current)
        else:
            poynting_flux = (
                electric_field[1] * magnetic_field[2] - electric_field[2] * magnetic_field[1],
                electric_field[2] * magnetic_field[0] - electric_field[0] * magnetic_field[2],
                electric_field[0] * magnetic_field[1] - electric_field[1] * magnetic_field[0],
            )
            energy_rate = -sum(
                _central_first_derivative(flux, axis, grid_spacing, dimensionality)
                for axis, flux in enumerate(poynting_flux)
            )
        source = source.at[registered_variables.energy_index].set(energy_rate)
    return source
