"""
Slope limiter functions for the finite-volume reconstruction.

Collects the elementary limiter primitives (minmod, maxmod) and the composite
limiters (double minmod, superbee, van Albada) used to limit the reconstructed
gradients and avoid spurious oscillations at discontinuities.
"""

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    STATE_TYPE,
    STATE_TYPE_ALTERED,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig

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


# TODO: bring into common interface
def _van_albada_limiter(
    backward_difference: STATE_TYPE,
    forward_difference: STATE_TYPE,
    config: SimulationConfig,
) -> STATE_TYPE_ALTERED:
    """
    van Albada limited gradients along an axis
    """

    grid_spacing = config.grid_spacing
    epsilon = 3 * grid_spacing
    limited_gradients = (
        (forward_difference**2 + epsilon) * backward_difference
        + (backward_difference**2 + epsilon) * forward_difference
    ) / (forward_difference**2 + backward_difference**2 + 2 * epsilon)

    return limited_gradients
