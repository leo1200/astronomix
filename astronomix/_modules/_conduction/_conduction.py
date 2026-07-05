"""
Thermal conduction for the finite-difference scheme.

We add a Fourier heat-conduction term to the energy equation,

    d(rho E)/dt  +=  div(kappa grad T) ,

with a **constant** conductivity ``kappa = params.thermal_conductivity`` and
the temperature taken from the ideal-gas relation

    T = p / rho            (code units, specific gas constant R = 1).

The constant-kappa case reduces to ``kappa * laplacian(T)`` which we
discretise with the standard second-order seven-point (in 3D) Laplacian.
Second order is deliberate: the stencil is trivially differentiable (a constant
linear operator on T) and the explicit parabolic time-step stays cheap.

Boundary conditions are **adiabatic** (zero conductive flux) at every wall: the
reflective hydro boundary mirrors density and pressure as even quantities, so
``T = p / rho`` is mirrored too and its normal gradient -- hence the conductive
flux -- vanishes at the wall.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix functions
from astronomix._stencil_operations._stencil_operations import _stencil_add


def _temperature(primitive_state, registered_variables):
    """Ideal-gas temperature T = p / rho (code units, R = 1)."""
    rho = primitive_state[registered_variables.density_index]
    p = primitive_state[registered_variables.pressure_index]
    return p / rho


@partial(jax.jit, static_argnames=("config", "registered_variables"))
def fd_conduction_source(primitive_state, params, config, registered_variables):
    """Conductive energy source ``kappa * laplacian(T)`` for the FD scheme.

    Returns a state-shaped array with only the energy slot populated; it is
    meant to be accumulated (times ``dt``) onto the conserved-state RHS in
    the time-integrator source assembly.
    """
    kappa = params.thermal_conductivity
    dx = config.grid_spacing
    ndim = config.dimensionality

    temperature = _temperature(primitive_state, registered_variables)

    # second-order Laplacian: sum_axis (T_{i+1} - 2 T_i + T_{i-1}) / dx^2
    laplacian_t = sum(
        _stencil_add(temperature, indices=(1, 0, -1), factors=(1.0, -2.0, 1.0), axis=ax)
        for ax in range(ndim)
    ) / (dx * dx)

    energy_source = kappa * laplacian_t

    S = jnp.zeros_like(primitive_state)
    S = S.at[registered_variables.energy_index].set(energy_source)
    return S
