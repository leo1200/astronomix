"""
Utility helpers for the gravity physics module.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import FIELD_TYPE

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions (reuse the same ghost-cell handling as the state)
from astronomix.time_stepping._utils import _pad
from astronomix._geometry.boundaries import _boundary_handler


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _pad_external_potential(
    external_potential: FIELD_TYPE,
    reference_field: FIELD_TYPE,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
) -> FIELD_TYPE:
    """
    Bring a cell-centered external gravitational potential (defined on the bare
    grid, without ghost cells) onto the same ghost-cell layout as reference_field.

    When reference_field is unpadded (e.g. in diagnostics on the real grid) the
    potential already matches and is returned unchanged. When reference_field is
    ghost-cell padded (e.g. a state field during evolution) the potential is
    given ghost cells via the very same _pad / _boundary_handler routines used
    for the state, so its ghost values respect the actual boundary conditions
    (periodic wrap-around, reflection, ...) rather than mere edge replication.

    A dummy variable axis is added so the state-shaped helpers ([num_vars, ...])
    apply, then removed again. With a single variable the reflective boundary's
    normal-velocity negation targets an out-of-bounds row and is dropped, so the
    scalar potential is correctly mirrored without a sign flip.

    Args:
        external_potential: The external potential on the bare grid.
        reference_field: A field whose (spatial) shape should be matched.
        config: The simulation configuration.
        registered_variables: The registered variables.
        params: The simulation parameters.

    Returns:
        The external potential matching the shape of reference_field.
    """
    if external_potential.shape == reference_field.shape:
        # reference field is on the bare grid; nothing to pad
        return external_potential

    # add a dummy variable axis -> [1, spatial...], reuse the state ghost-cell
    # handling, then drop the variable axis again
    potential = external_potential[jnp.newaxis, ...]
    potential = _pad(potential, config)
    potential = _boundary_handler(potential, config, registered_variables, params)

    return potential[0]
