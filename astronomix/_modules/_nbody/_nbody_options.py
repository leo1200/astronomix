"""
Configuration and parameter containers for the N-body point-mass gravity solver.

``NBodyConfig`` holds the static choices (whether the N-body integration is
active, the particle-to-grid mass deposition scheme used to feed the point
masses' gravity back onto the gas, and whether to treat the masses as a fixed,
non-evolving central source) and ``NBodyParams`` the physical N-body
parameters (masses and the N-body phase-space state). The module-level integer
constants name the available deposition schemes.
"""

# typing
from typing import NamedTuple

# jax
from jax import numpy as jnp

# particle-to-grid mass deposition schemes
NGP = 0  # nearest grid point
CIC = 1  # cloud in cell
TSC = 2  # triangular shaped cloud


class NBodyConfig(NamedTuple):
    """Static configuration of the N-body point-mass gravity solver."""

    #: N-body switch. When True, the N-body state is advanced by an RK4
    #: integrator jointly with (i.e. using the same time step as) the hydro
    #: update.
    nbody: bool = False

    #: Particle-to-grid mass deposition scheme used to couple the N-body
    #: masses to the gas when ``config.gravity_config.self_gravity`` is also
    #: enabled. One of ``NGP`` / ``CIC`` / ``TSC``.
    deposit_particles: int = NGP

    #: Treat the N-body masses as a single fixed source at the grid center
    #: instead of integrating their mutual-gravity orbits. Useful for a
    #: static central object providing a background potential.
    central_object_only: bool = False


class NBodyParams(NamedTuple):
    """Runtime parameters of the N-body point-mass gravity solver."""

    #: Masses of the N bodies, shape (n_bodies,).
    masses: jnp.ndarray = jnp.array([])

    #: The N-body phase-space state, [t, x, y, z, vx, vy, vz] per body, shape
    #: (n_bodies, 7) or flattened (n_bodies * 7,). Seeds the N-body state
    #: carried through the time-integration loop (see
    #: astronomix.time_stepping.time_integration.LoopState); not updated in
    #: place; the evolving state lives in the loop carry.
    nbody_state: jnp.ndarray = jnp.array([])
