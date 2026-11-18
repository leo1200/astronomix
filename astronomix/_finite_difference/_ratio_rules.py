"""
The two quotients of the finite-difference scheme with a Riemann flux
(``FDSolverConfig.scheme = RIEMANN_FLUX``) that carry their own derivative rule.

JAX differentiates ``x / y`` as ``x_dot / y - x y_dot y**-2``. In float32 ``y**-2``
is infinite for ``y`` below 5.4e-20, and the zero factor it meets at a floored or
vanishing numerator turns it into NaN. The rules form the slope as the product
``(x / y) (1 / y)`` instead; the values are the plain quotients.

The slope is a product of two quotients and not ``(x / y) / y``: XLA merges the
latter into ``x / (y y)``, and ``y y`` underflows in float32 below 3.8e-23. Both
factors go through the helpers themselves, so a derivative of a derivative stays
finite where the first one is.

``_weno_positivity._floored_ratio`` of the production scheme has the same
values but another derivative (zero below 1.08e-19 in float32, where this one
keeps the true slope, and a different rounding of the slope above), so one
definition for both would change the derivatives of one of the two schemes.
"""

# jax
import jax
import jax.numpy as jnp

#: The floor of the denominators.
DIVISION_FLOOR = 1e-30


@jax.custom_jvp
def _floored_ratio(x, d):
    """``x / max(d, DIVISION_FLOOR)``."""
    return x / jnp.maximum(d, DIVISION_FLOOR)


def _floored_ratio_rule(primals, tangents):
    """
    With y = max(d, DIVISION_FLOOR): d(x / y) = dx (1 / y) - (x / y) (1 / y) dd
    above the floor and dx (1 / y) at it.

    At and below the floor the product (x / y) (1 / y) is x times 1e60, which is
    infinite in float32. The selection discards it, but a compiled reverse pass
    on the CPU was not finite with it, so the discarded product is formed from
    a denominator of one; above the floor the slope is the same product.
    """
    x, d = primals
    x_dot, d_dot = tangents
    ratio = _floored_ratio(x, d)
    inverse = _floored_ratio(jnp.ones_like(d), d)
    above = d > DIVISION_FLOOR
    # 1.0 bounds nothing: it only keeps the discarded product finite
    d_safe = jnp.where(above, d, 1.0)
    slope = jnp.where(above, _floored_ratio(x, d_safe) * _floored_ratio(jnp.ones_like(d), d_safe), 0.0)
    return ratio, x_dot * inverse - slope * d_dot


@jax.custom_jvp
def _ratio(x, y):
    """``x / y`` for a denominator that is positive by construction."""
    return x / y


def _ratio_rule(primals, tangents):
    """d(x / y) = dx (1 / y) - (x / y) (1 / y) dy."""
    x, y = primals
    x_dot, y_dot = tangents
    ratio = _ratio(x, y)
    inverse = _ratio(jnp.ones_like(y), y)
    return ratio, x_dot * inverse - ratio * inverse * y_dot


_floored_ratio.defjvp(_floored_ratio_rule)
_ratio.defjvp(_ratio_rule)
