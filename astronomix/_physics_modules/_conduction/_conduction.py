"""
Thermal conduction for the finite-difference scheme.

We add a Fourier heat-conduction term to the energy equation,

    d(rho E)/dt  +=  div(kappa grad T) ,

with a **constant** conductivity ``kappa = params.thermal_conductivity`` and
the temperature taken from the ideal-gas relation

    T = p / rho            (code units, specific gas constant R = 1).

The constant-kappa case reduces to ``kappa * laplacian(T)`` which we
discretise with the standard second-order seven-point (in 3D) Laplacian.
Second order is deliberate: the conductive boundary conditions then live in a
single ghost-cell ring, the stencil is trivially differentiable (a constant
linear operator on T) and the explicit parabolic time-step stays cheap.

Boundary conditions
-------------------
``config.conduction_wall_axis`` selects the spatial axis whose two ends carry
the (optional) isothermal **Dirichlet** plates -- the hot/cold plates that
drive Rayleigh-Benard convection.  The plate temperatures are
``params.wall_temperature_low`` (low-index side) and
``params.wall_temperature_high`` (high-index side).  We enforce the wall-face
temperature by overwriting the ghost-cell ring adjacent to the interior with
the linear reflection ``T_ghost = 2 * T_wall - T_interior`` so that the
arithmetic mean across the wall face equals ``T_wall``.

All *other* boundaries are **adiabatic** (zero conductive flux).  No special
handling is needed there: the reflective hydro boundary mirrors density and
pressure as even quantities, so ``T = p / rho`` is mirrored too and its normal
gradient -- hence the conductive flux -- vanishes at the wall.
"""

from functools import partial

import jax
import jax.numpy as jnp

from astronomix._stencil_operations._stencil_operations import _stencil_add


def _temperature(primitive_state, registered_variables):
    """Ideal-gas temperature T = p / rho (code units, R = 1)."""
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    return p / rho


def _set_isothermal_ghosts(temperature, config, params):
    """Overwrite the ghost ring adjacent to the interior along the
    conduction wall axis to enforce isothermal Dirichlet plates."""
    ng = config.num_ghost_cells
    ax = config.conduction_wall_axis  # spatial axis (0-based) of the T field
    ndim = temperature.ndim

    def _index(pos):
        return (slice(None),) * ax + (pos,) + (slice(None),) * (ndim - ax - 1)

    # low side: ghost layer at index ng-1 is adjacent to the first interior
    # cell at index ng; the wall face sits between them.
    t_low_interior = temperature[_index(ng)]
    temperature = temperature.at[_index(ng - 1)].set(
        2.0 * params.wall_temperature_low - t_low_interior
    )

    # high side: ghost layer at index -ng is adjacent to the last interior
    # cell at index -ng-1.
    t_high_interior = temperature[_index(-ng - 1)]
    temperature = temperature.at[_index(-ng)].set(
        2.0 * params.wall_temperature_high - t_high_interior
    )

    return temperature


@partial(jax.jit, static_argnames=("config", "registered_variables"))
def fd_conduction_source(primitive_state, params, config, registered_variables):
    """Conductive energy source ``kappa * laplacian(T)`` for the FD scheme.

    Returns a state-shaped array with only the energy slot populated; it is
    meant to be accumulated (times ``dt``) onto the conserved-state RHS in
    :func:`_physics_sources`.
    """
    kappa = params.thermal_conductivity
    dx = config.grid_spacing
    ndim = config.dimensionality

    temperature = _temperature(primitive_state, registered_variables)

    if config.conduction_isothermal_walls:
        temperature = _set_isothermal_ghosts(temperature, config, params)

    # second-order Laplacian: sum_axis (T_{i+1} - 2 T_i + T_{i-1}) / dx^2
    laplacian_t = sum(
        _stencil_add(temperature, indices=(1, 0, -1), factors=(1.0, -2.0, 1.0), axis=ax)
        for ax in range(ndim)
    ) / (dx * dx)

    energy_source = kappa * laplacian_t

    S = jnp.zeros_like(primitive_state)
    S = S.at[registered_variables.energy_index].set(energy_source)
    return S
