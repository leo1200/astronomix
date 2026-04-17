from functools import partial
import math
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
    StaticFloatVector,
    StaticIntVector,
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

    if isinstance(config.box_size, float):
        config = config._replace(
            box_size=StaticFloatVector(
                config.box_size,
                config.box_size,
                config.box_size
            )
        )

    if isinstance(config.num_cells, int):
        config = config._replace(
            num_cells=StaticIntVector(
                config.num_cells,
                config.num_cells,
                config.num_cells
            )
        )

    grid_spacing_vec = config.box_size / config.num_cells

    # as soon as we accept a grid spacing vector, 
    # this will not be necessary anymore
    if config.dimensionality == 1:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
    elif config.dimensionality == 2:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not math.isclose(grid_spacing_vec.x, grid_spacing_vec.y):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )
    elif config.dimensionality == 3:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not (math.isclose(grid_spacing_vec.x, grid_spacing_vec.y) and math.isclose(grid_spacing_vec.x, grid_spacing_vec.z)):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )

    grid_spacing = config.grid_spacing

    # in spherical or cylindrical symmetry, we always need the helper data
    if config.geometry == SPHERICAL or config.geometry == CYLINDRICAL:
        r = jnp.linspace(
            grid_spacing / 2 - ngc * grid_spacing,
            config.box_size.x + grid_spacing / 2 + ngc * grid_spacing,
            config.num_cells.x + 2 * ngc,
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
                    config.box_size.x + grid_spacing / 2 + ngc * grid_spacing,
                    config.num_cells.x + 2 * ngc,
                    endpoint=False,
                )
                y = jnp.linspace(
                    grid_spacing / 2 - ngc * grid_spacing,
                    config.box_size.y + grid_spacing / 2 + ngc * grid_spacing,
                    config.num_cells.y + 2 * ngc,
                    endpoint=False,
                )

                if config.dimensionality == 3:
                    z = jnp.linspace(
                        grid_spacing / 2 - ngc * grid_spacing,
                        config.box_size.z + grid_spacing / 2 + ngc * grid_spacing,
                        config.num_cells.z + 2 * ngc,
                        endpoint=False,
                    )
                    if sharding is not None:
                        geometric_centers = jax.lax.with_sharding_constraint(
                            jnp.array(jnp.meshgrid(x, y, z, indexing='ij')), sharding
                        )
                    else:
                        geometric_centers = jnp.array(jnp.meshgrid(x, y, z, indexing='ij'))
                else:
                    geometric_centers = jnp.array(jnp.meshgrid(x, y, indexing='ij'))

                # calculate the distances from the cell centers to the box center
                if config.dimensionality == 1:
                    box_center = jnp.array([config.box_size.x / 2])
                elif config.dimensionality == 2:
                    box_center = jnp.array([config.box_size.x / 2, config.box_size.y / 2])
                elif config.dimensionality == 3:
                    box_center = jnp.array([config.box_size.x / 2, config.box_size.y / 2, config.box_size.z / 2])

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
                    config.box_size.x - grid_spacing / 2 + ngc * grid_spacing,
                    config.num_cells.x + 2 * ngc,
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
