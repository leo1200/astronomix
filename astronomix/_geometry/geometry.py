"""
Curvilinear metric factors for non-Cartesian geometries.

Provides the area/volume metric factors (``r_hat_alpha`` and the volumetric cell
centre) used to build geometric source terms in cylindrical and spherical
coordinates.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import CYLINDRICAL, SPHERICAL


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["geometry"])
def _r_hat_alpha(
    r: Float[Array, "num_cells"], dr: Union[float, Float[Array, ""]], geometry: int
) -> Float[Array, "num_cells"]:
    """Return the geometric area factor ``r_hat_alpha`` for the given geometry.

    Args:
        r: The cell-centre radial coordinates.
        dr: The (uniform) radial grid spacing.
        geometry: The geometry identifier (``SPHERICAL`` or ``CYLINDRICAL``).

    Returns:
        The area metric factor; this is undefined for Cartesian coordinates and
        raises in that case.
    """
    if geometry == SPHERICAL:
        return r**2 + 1 / 12 * dr**2
    elif geometry == CYLINDRICAL:
        return r
    else:
        raise ValueError("Unknown geometry / not for cartesian coordinates")


# @jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=["geometry"])
def _center_of_volume(
    r: Float[Array, "num_cells"], dr: Union[float, Float[Array, ""]], geometry: int
) -> Float[Array, "num_cells"]:
    """Return the volumetric cell-centre radius for the given geometry.

    The volumetric centre is the radius that splits each cell into equal
    volumes; it differs from the geometric centre by an ``O(dr^2)`` correction.

    Args:
        r: The geometric cell-centre radial coordinates.
        dr: The (uniform) radial grid spacing.
        geometry: The geometry identifier (``SPHERICAL`` or ``CYLINDRICAL``).

    Returns:
        The volumetric cell-centre radii.
    """
    if geometry == CYLINDRICAL:
        return (r**2 + 1 / 12 * dr**2) / r**2 * r
    elif geometry == SPHERICAL:
        r_hat = _r_hat_alpha(r, dr, geometry)
        return (r**2 + 1 / 4 * dr**2) / r_hat * r
