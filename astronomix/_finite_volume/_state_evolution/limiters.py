"""
Slope limiter functions for the finite-volume reconstruction.

Collects the elementary limiter primitives (minmod, maxmod) and the composite
limiters (double minmod, superbee) used to limit the reconstructed
gradients and avoid spurious oscillations at discontinuities.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _stencil_add


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
