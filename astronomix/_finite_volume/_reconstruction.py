"""
Reconstruction of the finite-volume stage: the two states at every interface.

Donor cell (the cell values) or piecewise linear with the harmonic van Leer
slope, as in AthenaPK. The slope is an elementwise function that the fused
Pallas kernel (``_stage_pallas.py``) calls as well.
"""

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import STATE_TYPE


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


def _reconstructed_face_states(primitive_state: STATE_TYPE, axis: int, piecewise_linear: bool):
    """
    The left and right primitive states at every interface ``i - 1/2`` along
    ``axis`` (stored at index ``i``).

    Args:
        primitive_state: The primitive state.
        axis: The spatial axis (1-based, i.e. the array axis).
        piecewise_linear: Use the van Leer limited PLM; otherwise donor cell.

    Returns:
        The (left, right) interface states, each shaped like the state.
    """
    if not piecewise_linear:
        return jnp.roll(primitive_state, 1, axis=axis), primitive_state

    right_face_values, left_face_values = _van_leer_slopes(
        jnp.roll(primitive_state, 1, axis=axis),
        primitive_state,
        jnp.roll(primitive_state, -1, axis=axis),
    )
    return jnp.roll(right_face_values, 1, axis=axis), left_face_values
