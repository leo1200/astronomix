import jax.numpy as jnp
import jax
import jax.flatten_util
from typing import Sequence
from jaxtyping import PyTree


def conFIG(grads: Sequence[PyTree], use_least_square: bool = True) -> PyTree:
    """
    Performs the standard ConFIG update step for JAX/Equinox PyTree gradients.

    Adapts the ConFIG algorithm (tum-pbs/ConFIG) to work with PyTree gradients
    produced by eqx.filter_value_and_grad. Each gradient PyTree is flattened to a
    1D vector, the conflict-free direction is computed, then unflattened back.

    Uses EqualWeight for direction and ProjectionLength for rescaling.

    Args:
        grads: A sequence of PyTree gradients (e.g. from multiple loss functions).
        use_least_square: Whether to use least squares (True) or pseudo-inverse (False)
            for computing the best direction. Defaults to True.

    Returns:
        A PyTree with the same structure as each input gradient, containing the
        conflict-free update direction.

    References:
        Tum-pbs ConFIG: https://github.com/tum-pbs/ConFIG
    """
    # Flatten each PyTree gradient into a 1D vector
    flat_grads = []
    unravel_fn = None
    for grad in grads:
        flat, unravel = jax.flatten_util.ravel_pytree(grad)
        flat_grads.append(flat)
        if unravel_fn is None:
            unravel_fn = unravel

    grad_matrix = jnp.stack(flat_grads)  # shape: (m, N)
    m = grad_matrix.shape[0]

    # Equal weights
    weights = jnp.ones(m)

    # Normalize each gradient to get unit vectors
    norms = jnp.linalg.norm(grad_matrix, axis=1, keepdims=True)
    units = jnp.where(norms > 0, grad_matrix / norms, 0.0)

    # Find the best conflict-free direction
    if use_least_square:
        best_direction = jnp.linalg.lstsq(units, weights)[0]
    else:
        best_direction = jnp.linalg.pinv(units) @ weights

    # Rescale length using ProjectionLength model:
    # length = sum of projections of each gradient onto the unit target direction
    unit_target = best_direction / jnp.linalg.norm(best_direction)
    length = jnp.sum(jnp.array([jnp.dot(g, unit_target) for g in flat_grads]))
    config_grad_flat = length * unit_target

    assert unravel_fn is not None
    return unravel_fn(config_grad_flat)
