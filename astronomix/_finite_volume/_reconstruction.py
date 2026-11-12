"""
Reconstruction of the finite-volume stage: the two states at every interface.

Donor cell (the cell values) or piecewise linear with a limited slope: the
harmonic van Leer slope of AthenaPK, minmod, superbee or double minmod. The
slopes are elementwise functions of the two one-sided differences that the
fused Pallas kernel (``_stage_pallas.py``) calls as well.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DONOR_CELL,
    DOUBLE_MINMOD,
    MINMOD,
    STATE_TYPE,
    SUPERBEE,
    VAN_LEER,
)


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
        return _van_leer_slopes(cell_minus, cell, cell_plus)

    difference_left = cell - cell_minus
    difference_right = cell_plus - cell
    if limiter == MINMOD:
        limited_difference = _minmod(difference_left, difference_right)
    elif limiter == SUPERBEE:
        limited_difference = _superbee(difference_left, difference_right)
    elif limiter == DOUBLE_MINMOD:
        limited_difference = _double_minmod(difference_left, difference_right)
    else:
        raise ValueError("Unknown limiter.")
    limited_half_slope = 0.5 * limited_difference
    return cell + limited_half_slope, cell - limited_half_slope


def _reconstructed_face_states(primitive_state: STATE_TYPE, axis: int, reconstruction: int, limiter: int):
    """
    The left and right primitive states at every interface ``i - 1/2`` along
    ``axis`` (stored at index ``i``).

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based, i.e. the array axis).
        reconstruction: ``DONOR_CELL`` or ``PIECEWISE_LINEAR``.
        limiter: The slope limiter of the piecewise-linear reconstruction.

    Returns:
        The (left, right) interface states, each shaped like the state.
    """
    if reconstruction == DONOR_CELL:
        return jnp.roll(primitive_state, 1, axis=axis), primitive_state

    right_face_values, left_face_values = _limited_slopes(
        jnp.roll(primitive_state, 1, axis=axis),
        primitive_state,
        jnp.roll(primitive_state, -1, axis=axis),
        limiter,
    )
    return jnp.roll(right_face_values, 1, axis=axis), left_face_values


# -------------------------------------------------------------
# ===================== ↑ Interface states ↑ ==================
# -------------------------------------------------------------
