"""
Reconstruction of the finite-volume stage: the two states at every interface.

Donor cell (the cell values) or piecewise linear with a limited slope: the
harmonic van Leer slope of AthenaPK, minmod, superbee or double minmod. The
slopes are elementwise functions of the two one-sided differences that the
fused Pallas kernel (``_stage_pallas.py``) calls as well.

The dimensionally split scheme (``muscl_hancock_step``) adds the Hancock
predictor: the cell values are advanced by half the time step of the sweep
with the limited slopes before they are extrapolated to the faces
(``_hancock_face_values``). Its slopes are difference quotients over the
distances between the cell centres (``CellDistances``), which is one of the
two places where a curvilinear geometry enters the scheme; the other one is
the flux divergence (``_flux_correction.py``).
"""

# typing
from typing import (
    Callable,
    NamedTuple,
    Tuple,
    Union,
)
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
    DONOR_CELL,
    DOUBLE_MINMOD,
    MINMOD,
    SPHERICAL,
    STATE_TYPE,
    SUPERBEE,
    VAN_LEER,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig


# -------------------------------------------------------------
# ======================== ↓ Limiters ↓ =======================
# -------------------------------------------------------------


@jax.jit
def _minmod(a, b):
    """Return the two-argument minmod of ``a`` and ``b`` (zero on sign change)."""
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.minimum(jnp.abs(a), jnp.abs(b))


def _minmod3(a, b, c):
    """Minmod function for three arguments in JAX."""
    same_sign = (jnp.sign(a) == jnp.sign(b)) & (jnp.sign(b) == jnp.sign(c))
    return jnp.where(
        same_sign,
        jnp.sign(a) * jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c))),
        0.0,
    )


def _double_minmod(a, b):
    """
    Double minmod limiter.
    """
    return jnp.where(a * b > 0, _minmod3((a + b) / 2, 2 * a, 2 * b), 0.0)


@jax.jit
def _maxmod(a, b):
    """Return the two-argument maxmod of ``a`` and ``b`` (zero on sign change)."""
    return 0.5 * (jnp.sign(a) + jnp.sign(b)) * jnp.maximum(jnp.abs(a), jnp.abs(b))


def _superbee(a, b):
    """
    Superbee limiter.
    """
    return jnp.where(a * b > 0, _minmod(_maxmod(a, b), _minmod(2 * a, 2 * b)), 0.0)


# -------------------------------------------------------------
# ======================== ↑ Limiters ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ Interface states ↓ ==================
# -------------------------------------------------------------


def _van_leer_half_difference(difference_left, difference_right):
    """
    Half the van Leer (harmonic mean) limited difference, ``Δ_L Δ_R / (Δ_L + Δ_R)``
    where the one-sided differences agree in sign, zero otherwise (AthenaPK:
    ``plm_simple.hpp``).
    """
    difference_product = difference_left * difference_right
    is_monotone = difference_product > 0.0
    # The denominator is guarded so that the discarded branch stays finite.
    safe_sum = jnp.where(is_monotone, difference_left + difference_right, 1.0)
    return jnp.where(is_monotone, difference_product / safe_sum, 0.0)


def _van_leer_slopes(cell_minus, cell, cell_plus):
    """
    The piecewise-linear reconstruction of one cell with the van Leer limited
    half slope (``_van_leer_half_difference``).

    Args:
        cell_minus: The value in the left neighbour.
        cell: The value in the cell.
        cell_plus: The value in the right neighbour.

    Returns:
        ``(value at the cell's right face, value at the cell's left face)``.
    """
    limited_half_slope = _van_leer_half_difference(cell - cell_minus, cell_plus - cell)
    return cell + limited_half_slope, cell - limited_half_slope


def _limited_difference(difference_left, difference_right, limiter: int):
    """
    The limited value of the two one-sided differences (or difference
    quotients) of a cell.

    Args:
        difference_left: The difference to the left neighbour.
        difference_right: The difference to the right neighbour.
        limiter: The slope limiter (``MINMOD``, ``VAN_LEER``, ``SUPERBEE`` or
            ``DOUBLE_MINMOD``).

    Returns:
        The limited difference.
    """
    if limiter == MINMOD:
        return _minmod(difference_left, difference_right)
    if limiter == SUPERBEE:
        return _superbee(difference_left, difference_right)
    if limiter == DOUBLE_MINMOD:
        return _double_minmod(difference_left, difference_right)
    if limiter == VAN_LEER:
        return 2.0 * _van_leer_half_difference(difference_left, difference_right)
    raise ValueError("Unknown limiter.")


def _limited_slopes(cell_minus, cell, cell_plus, limiter: int):
    """
    The piecewise-linear reconstruction of one cell with the limited slope of
    the two one-sided differences ``Δ_L``, ``Δ_R``.

    Args:
        cell_minus: The value in the left neighbour.
        cell: The value in the cell.
        cell_plus: The value in the right neighbour.
        limiter: The slope limiter (``MINMOD``, ``VAN_LEER``, ``SUPERBEE`` or
            ``DOUBLE_MINMOD``).

    Returns:
        ``(value at the cell's right face, value at the cell's left face)``.
    """
    if limiter == VAN_LEER:
        # AthenaPK's form, the half slope directly (no factor of two and back).
        return _van_leer_slopes(cell_minus, cell, cell_plus)

    limited_half_slope = 0.5 * _limited_difference(cell - cell_minus, cell_plus - cell, limiter)
    return cell + limited_half_slope, cell - limited_half_slope


class CellDistances(NamedTuple):
    """
    The distances the slopes of a cell are measured over: from the point the
    cell value belongs to (the cell centre; the volumetric centre in spherical
    geometry) to that point of the two neighbours and to the two faces of the
    cell. Scalars on a Cartesian grid, one value per cell in spherical 1D.
    """

    to_left_centre: Union[float, Float[Array, "num_cells"]]
    to_right_centre: Union[float, Float[Array, "num_cells"]]
    to_left_face: Union[float, Float[Array, "num_cells"]]
    to_right_face: Union[float, Float[Array, "num_cells"]]


def _cell_distances(config: SimulationConfig, helper_data: HelperData) -> CellDistances:
    """
    The distances between cell centres and to the faces along the sweep axis.

    Args:
        config: The simulation configuration.
        helper_data: The helper data (the radii of the spherical 1D grid).

    Returns:
        The distances.
    """
    spacing = config.grid_spacing
    if config.geometry == CARTESIAN:
        return CellDistances(spacing, spacing, spacing / 2, spacing / 2)
    if config.geometry == SPHERICAL and config.dimensionality == 1:
        # A cell value is the volume average, which a linear profile takes at
        # the volumetric centre: this lies outwards of the middle of the cell.
        radius = helper_data.geometric_centers
        centre = helper_data.volumetric_centers
        return CellDistances(
            to_left_centre=centre - jnp.roll(centre, 1),
            to_right_centre=jnp.roll(centre, -1) - centre,
            to_left_face=centre - (radius - spacing / 2),
            to_right_face=(radius + spacing / 2) - centre,
        )
    raise ValueError("Geometry and dimensionality combination not supported.")


def _limited_gradients(cell_minus, cell, cell_plus, limiter: int, distances: CellDistances):
    """
    The limited gradient of one cell: the limited value of the two one-sided
    difference quotients.

    Args:
        cell_minus: The value in the left neighbour.
        cell: The value in the cell.
        cell_plus: The value in the right neighbour.
        limiter: The slope limiter.
        distances: The distances between the cell centres.

    Returns:
        The limited gradient.
    """
    return _limited_difference(
        (cell - cell_minus) / distances.to_left_centre,
        (cell_plus - cell) / distances.to_right_centre,
        limiter,
    )


class HancockTerms(NamedTuple):
    """
    What the Hancock predictor of a sweep needs beyond the reconstruction.

    Attributes:
        half_step: Half the time step of the sweep.
        gamma: The adiabatic index.
        density_index: The state index of the density.
        normal_velocity_index: The state index of the velocity along the sweep axis.
        pressure_index: The state index of the pressure (not read for a barotropic gas).
        carried_density_indices: The state indices of the densities carried with the gas (the wind density tracer).
        distances: The distances between cell centres and to the faces.
        sound_speed_squared_of: ``None`` for the ideal gas; for a barotropic gas the squared sound speed as a
            function of the density.
    """

    half_step: Union[float, Float[Array, ""]]
    gamma: Union[float, Float[Array, ""]]
    density_index: int
    normal_velocity_index: int
    pressure_index: int
    carried_density_indices: Tuple[int, ...]
    distances: CellDistances
    sound_speed_squared_of: Callable = None


def _hancock_predictor(primitive_state: STATE_TYPE, gradients: STATE_TYPE, hancock: HancockTerms) -> STATE_TYPE:
    """
    The rate ``A(W) dW/dn`` of the primitive Euler equations along the sweep
    axis, ``dW/dt = -A(W) dW/dn``, with the limited gradients: every variable
    is advected with the normal velocity, and the three couplings of the
    acoustic waves are added (Toro 2009, section 14.4 in primitive variables).
    A density carried with the gas is compressed like the gas density, so that
    a uniform concentration stays uniform at the faces. A barotropic gas has
    no pressure row: its pressure force follows from the density gradient,
    ``dp/dn = a^2 drho/dn``.

    Args:
        primitive_state: The primitive state.
        gradients: The limited gradients of the state along the sweep axis.
        hancock: The predictor terms of the sweep.

    Returns:
        The rate, laid out like the state.
    """
    density = primitive_state[hancock.density_index]
    normal_velocity = primitive_state[hancock.normal_velocity_index]
    normal_velocity_gradient = gradients[hancock.normal_velocity_index]

    rate = normal_velocity * gradients
    # compression: drho/dt = ... - rho du_n/dn
    rate = rate.at[hancock.density_index].add(density * normal_velocity_gradient)
    for index in hancock.carried_density_indices:
        rate = rate.at[index].add(primitive_state[index] * normal_velocity_gradient)

    if hancock.sound_speed_squared_of is not None:
        # pressure force of a barotropic gas: du_n/dt = ... - (a^2 / rho) drho/dn
        pressure_gradient = hancock.sound_speed_squared_of(density) * gradients[hancock.density_index]
        return rate.at[hancock.normal_velocity_index].add(pressure_gradient / density)

    pressure = primitive_state[hancock.pressure_index]
    pressure_gradient = gradients[hancock.pressure_index]
    # pressure force: du_n/dt = ... - (1 / rho) dp/dn
    rate = rate.at[hancock.normal_velocity_index].add(pressure_gradient / density)
    # compression work: dp/dt = ... - rho c^2 du_n/dn, with rho c^2 = gamma p
    rate = rate.at[hancock.pressure_index].add(hancock.gamma * pressure * normal_velocity_gradient)
    return rate


def _hancock_face_values(cell_minus, cell, cell_plus, limiter: int, hancock: HancockTerms):
    """
    The piecewise-linear reconstruction of one cell of the split scheme: the
    cell value, minus half a step of the predictor, plus or minus the distance
    to the face times the limited gradient (MUSCL-Hancock, van Leer 1984).

    Args:
        cell_minus: The state in the left neighbour.
        cell: The state in the cell.
        cell_plus: The state in the right neighbour.
        limiter: The slope limiter.
        hancock: The predictor terms of the sweep.

    Returns:
        ``(state at the cell's right face, state at the cell's left face)``.
    """
    distances = hancock.distances
    gradients = _limited_gradients(cell_minus, cell, cell_plus, limiter, distances)
    predicted = cell - hancock.half_step * _hancock_predictor(cell, gradients, hancock)
    return predicted + distances.to_right_face * gradients, predicted - distances.to_left_face * gradients


def _reconstructed_face_states(
    primitive_state: STATE_TYPE,
    axis: int,
    reconstruction: int,
    limiter: int,
    hancock: HancockTerms = None,
):
    """
    The left and right primitive states at every interface ``i - 1/2`` along
    ``axis`` (stored at index ``i``).

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based, i.e. the array axis).
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR``.
        limiter: The slope limiter of the piecewise-linear reconstruction.
        hancock: The predictor terms of the split scheme, or ``None`` for the
            reconstruction at the time level of the state. Donor cell has no
            predictor.

    Returns:
        The (left, right) interface states, each shaped like the state.
    """
    if reconstruction == DONOR_CELL:
        return jnp.roll(primitive_state, 1, axis=axis), primitive_state

    cell_minus = jnp.roll(primitive_state, 1, axis=axis)
    cell_plus = jnp.roll(primitive_state, -1, axis=axis)
    if hancock is None:
        right_face_values, left_face_values = _limited_slopes(cell_minus, primitive_state, cell_plus, limiter)
    else:
        right_face_values, left_face_values = _hancock_face_values(
            cell_minus, primitive_state, cell_plus, limiter, hancock
        )
    return jnp.roll(right_face_values, 1, axis=axis), left_face_values


# -------------------------------------------------------------
# ===================== ↑ Interface states ↑ ==================
# -------------------------------------------------------------
