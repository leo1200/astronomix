from functools import partial
from types import NoneType
from typing import NamedTuple, Union
from jax import NamedSharding
import jax.numpy as jnp
from astronomix._geometry.geometry import _center_of_volume, _r_hat_alpha
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    CYLINDRICAL,
    SPHERICAL,
    SimulationConfig,
)
import jax

# Helper data like the radii and cell volumes
# in the simulation or cooling tables etc.


class HelperData(NamedTuple):
    """Helper data used throughout the simulation."""

    #: The geometric centers of the cells.
    geometric_centers: jnp.ndarray = None

    #: The volumetric centers of the cells.
    #: Same as the geometric centers for Cartesian geometry.
    volumetric_centers: jnp.ndarray = None

    #: cell center to box center distances
    #: only for config.dimensionality > 1
    r: jnp.ndarray = None

    #: A helper variable, defined as
    #: \hat{r}^\alpha = V_j / (2 * \alpha * \pi * \Delta r)
    #: with V_j the volume of cell j, \alpha the geometry factor
    #: and \Delta r the cell width.
    r_hat_alpha: jnp.ndarray = None

    #: The cell volumes.
    cell_volumes: jnp.ndarray = None

    #: Coordinates of the inner cell boundaries.
    inner_cell_boundaries: jnp.ndarray = None

    #: Coordinates of the outer cell boundaries.
    outer_cell_boundaries: jnp.ndarray = None

@partial(jax.jit, static_argnames=("config", "sharding", "padded", "production"))
def get_helper_data(
    config: SimulationConfig,
    sharding: Union[NoneType, NamedSharding] = None,
    padded: bool = False,
    production: bool = False,
) -> HelperData:
    """Generate the helper data for the simulation from the configuration."""

    if padded:
        ngc = config.num_ghost_cells
    else:
        ngc = 0

    grid_spacing = config.box_size / config.num_cells

    # in spherical or cylindrical symmetry, we always need the helper data
    if config.geometry == SPHERICAL or config.geometry == CYLINDRICAL:
        r = jnp.linspace(
            grid_spacing / 2 - ngc * grid_spacing,
            config.box_size + grid_spacing / 2 + ngc * grid_spacing,
            config.num_cells + 2 * ngc,
            endpoint=False,
        )
        inner_cell_boundaries = r - grid_spacing / 2
        outer_cell_boundaries = r + grid_spacing / 2
        volumetric_centers = _center_of_volume(r, grid_spacing, config.geometry)
        r_hat = _r_hat_alpha(r, grid_spacing, config.geometry)
        cell_volumes = 2 * config.geometry * jnp.pi * grid_spacing * r_hat
        helper_data_pad = HelperData(
            geometric_centers=r,
            volumetric_centers=volumetric_centers,
            r_hat_alpha=r_hat,
            cell_volumes=cell_volumes,
            inner_cell_boundaries=inner_cell_boundaries,
            outer_cell_boundaries=outer_cell_boundaries,
        )

    helper_data_necessary = (
        config.wind_config.stellar_wind or config.cooling_config.cooling or 
        config.return_snapshots
    )

    if not production or helper_data_necessary:
        
        if config.geometry == CARTESIAN:
            if config.dimensionality > 1:
                x = jnp.linspace(
                    grid_spacing / 2 - ngc * grid_spacing,
                    config.box_size + grid_spacing / 2 + ngc * grid_spacing,
                    config.num_cells + 2 * ngc,
                    endpoint=False,
                )
                y = jnp.linspace(
                    grid_spacing / 2 - ngc * grid_spacing,
                    config.box_size + grid_spacing / 2 + ngc * grid_spacing,
                    config.num_cells + 2 * ngc,
                    endpoint=False,
                )

                if config.dimensionality == 3:
                    z = jnp.linspace(
                        grid_spacing / 2 - ngc * grid_spacing,
                        config.box_size + grid_spacing / 2 + ngc * grid_spacing,
                        config.num_cells + 2 * ngc,
                        endpoint=False,
                    )
                    if sharding is not None:
                        geometric_centers = jax.lax.with_sharding_constraint(
                            jnp.array(jnp.meshgrid(x, y, z)), sharding
                        )
                    else:
                        geometric_centers = jnp.array(jnp.meshgrid(x, y, z))
                else:
                    geometric_centers = jnp.array(jnp.meshgrid(x, y))

                # calculate the distances from the cell centers to the box center
                box_center = jnp.zeros(config.dimensionality) + config.box_size / 2

                geometric_centers = jnp.moveaxis(geometric_centers, 0, -1)

                volumetric_centers = geometric_centers

                r = jnp.linalg.norm(geometric_centers - box_center, axis=-1)

                helper_data_pad = HelperData(
                    geometric_centers=geometric_centers,
                    volumetric_centers=volumetric_centers,
                    r=r,
                )
            else:
                r = jnp.linspace(
                    grid_spacing / 2 - ngc * grid_spacing,
                    config.box_size - grid_spacing / 2 + ngc * grid_spacing,
                    config.num_cells + 2 * ngc,
                )
                r_hat = grid_spacing * jnp.ones_like(r)  # not really
                cell_volumes = grid_spacing * jnp.ones_like(r)
                inner_cell_boundaries = r - grid_spacing / 2
                outer_cell_boundaries = r + grid_spacing / 2
                helper_data_pad = HelperData(
                    geometric_centers=r,
                    r_hat_alpha=r_hat,
                    cell_volumes=cell_volumes,
                    inner_cell_boundaries=inner_cell_boundaries,
                    outer_cell_boundaries=outer_cell_boundaries,
                    volumetric_centers=r,
                )
    
    else:
        helper_data_pad = HelperData()

    return helper_data_pad
