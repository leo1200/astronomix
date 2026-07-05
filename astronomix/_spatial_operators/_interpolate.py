"""
High-order interpolation between cell centres and cell faces.

Provides the 4th-order centre-to-face and 6th-order face-to-centre
interpolations, plus the point-value to cell-average correction used to retain
high-order accuracy in dimensionally split settings.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _shift


@partial(jax.jit, static_argnames=["axis"])
def interp_center_to_face(arr, axis):
    """
    Interpolate to x-interfaces using 4th order
        f_{i+1/2} = (-f_{i-1} + 9f_{i} + 9f_{i+1} - f_{i+2}) / 16
    The i-th array index in the output corresponds to the i+1/2 interface.
    """
    return (
        -_shift(arr, 1, axis=axis)
        + 9 * arr
        + 9 * _shift(arr, -1, axis=axis)
        - _shift(arr, -2, axis=axis)
    ) / 16.0


@partial(jax.jit, static_argnames=["axis"])
def interp_face_to_center(f_int, axis):
    """
    6th order interpolation from face to center.
    """
    return (
        3 * _shift(f_int, 3, axis=axis)
        - 25 * _shift(f_int, 2, axis=axis)
        + 150 * _shift(f_int, 1, axis=axis)
        + 150 * f_int
        - 25 * _shift(f_int, -1, axis=axis)
        + 3 * _shift(f_int, -2, axis=axis)
    ) / 256.0


@partial(jax.jit, static_argnames=["axisA", "axisB"])
def point_values_to_averages(q, axisA, axisB):
    """
    For point values q, we can approximate the cell-averaged
    values Q based on interpolation as

    Q_i = q_i + Δx^2/24 q''(x_i) - ...

    For point values, the second derivative
    can be approximated

    q''(x_i) = (q_{i+1} - 2 q_i + q_{i-1}) / Δx^2

    Here we apply this in two dimensions.
    Compare Buchmüller and Helzel 2014, Eq. 12, 13.

    Such smoothing can be used to retain high-order accuracy
    in dimensionally split settings.    
    """
    smooth_x = (
        _shift(q, 1, axis=axisA) - 2 * q + _shift(q, -1, axis=axisA)
    ) / 24.0
    smooth_y = (
        _shift(q, 1, axis=axisB) - 2 * q + _shift(q, -1, axis=axisB)
    ) / 24.0
    return q + smooth_x + smooth_y

@partial(jax.jit, static_argnames=["axisA"])
def point_values_to_averages_single_axis(q, axisA):
    """
    Single axis version of point_values_to_averages.
    """
    smooth = (
        _shift(q, 1, axis=axisA) - 2 * q + _shift(q, -1, axis=axisA)
    ) / 24.0
    return q + smooth