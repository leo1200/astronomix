"""
Sink particle formation setup (3D).

A periodic box of uniform gas with a Gaussian overdensity at its centre,
evolved with self-gravity. The setup is used to test the sink particle creation
checks of Federrath et al. (2010), Section 2.2.

## References

- Federrath, Banerjee, Clark & Klessen (2010), ApJ 713, 269.
"""

# typing
from types import NoneType
from typing import NamedTuple, Union

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import PERIODIC_BOUNDARY

# astronomix containers
from astronomix.data_classes.simulation_state_struct import StateStruct, finalize_state
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.variable_registry.registered_variables import get_registered_variables


class SinkFormationSettings(NamedTuple):
    """Problem constants for the sink particle formation setup."""

    #: Side length of the cubic box.
    box_length: float = 1.0

    #: Background density.
    background_density: float = 1.0

    #: Peak density of the Gaussian overdensity, on top of the background.
    peak_overdensity: float = 100.0

    #: Width (standard deviation) of the Gaussian overdensity.
    overdensity_width: float = 0.05

    #: Centre (x, y, z) of the Gaussian overdensity; ``None`` places it at
    #: the centre of the box.
    overdensity_center: Union[tuple, NoneType] = None

    #: Sound speed squared of the gas.
    sound_speed_squared: float = 1.0

    #: Uniform magnetic field along z, used when ``config.mhd`` is on.
    magnetic_field_z: float = 0.0

    #: Adiabatic index of the gas.
    gamma: float = 5.0 / 3.0

    #: Gravitational constant.
    gravitational_constant: float = 1.0

    #: End time of the run.
    t_end: float = 1e-3


def setup_sink_formation(
    config: SimulationConfig,
    params: SimulationParams,
    settings: SinkFormationSettings = SinkFormationSettings(),
) -> tuple[StateStruct, SimulationConfig, SimulationParams]:
    """
    Set up a Gaussian overdensity in a periodic, self-gravitating 3D box.

    Enforces the geometry (3D Cartesian), a cubic box with periodic boundaries,
    self-gravity, and the state struct (which carries the sink particles). The
    gas is at rest with a uniform sound speed. With ``config.mhd`` on, the gas
    is threaded by the uniform field ``settings.magnetic_field_z`` along z,
    which has zero divergence. The number of cells, the sink particle
    configuration, MHD and the time-stepping options are left to the caller;
    the number of cells must be the same along every axis.

    Args:
        config: Simulation configuration.
        params: Simulation parameters.
        settings: Problem constants.

    Returns:
        state: Initial state struct of the simulation (no sinks passed in).
        config: Finalized simulation configuration.
        params: Updated simulation parameters (t_end, gamma,
            gravitational_constant).
    """
    config = config._replace(
        geometry=CARTESIAN,
        dimensionality=3,
        box_size=StaticFloatVector(
            settings.box_length,
            settings.box_length,
            settings.box_length,
        ),
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        gravity_config=config.gravity_config._replace(self_gravity=True),
        state_struct=True,
    )
    params = params._replace(
        t_end=settings.t_end,
        gamma=settings.gamma,
        gravitational_constant=settings.gravitational_constant,
    )

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    # Gaussian overdensity with the gas at rest and at a uniform sound speed
    # (pressure proportional to density). Distances to the centre are taken to
    # its nearest periodic copy, so a clump near the boundary wraps around it.
    if settings.overdensity_center is None:
        overdensity_center = jnp.full((3,), 0.5 * settings.box_length)
    else:
        overdensity_center = jnp.array(settings.overdensity_center)
    separation = helper_data.geometric_centers - overdensity_center
    separation = separation - settings.box_length * jnp.round(
        separation / settings.box_length
    )
    distance_squared = jnp.sum(separation**2, axis=-1)
    density = settings.background_density + settings.peak_overdensity * jnp.exp(
        -0.5 * distance_squared / settings.overdensity_width**2
    )
    pressure = density * settings.sound_speed_squared / settings.gamma
    zero_velocity = jnp.zeros_like(density)

    if config.mhd:
        magnetic_field = dict(
            magnetic_field_x=jnp.zeros_like(density),
            magnetic_field_y=jnp.zeros_like(density),
            magnetic_field_z=jnp.full_like(density, settings.magnetic_field_z),
        )
    else:
        magnetic_field = {}

    primitive_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero_velocity,
        velocity_y=zero_velocity,
        velocity_z=zero_velocity,
        gas_pressure=pressure,
        **magnetic_field,
    )

    config = finalize_config(config, primitive_state.shape)

    return finalize_state(config, primitive_state), config, params
