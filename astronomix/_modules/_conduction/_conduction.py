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

    if config.conduction_order == 4:
        # 4th-order FINITE-DIFFERENCE conduction. In an FD scheme the state is
        # the pointwise value, so ``T = p/rho`` and ``kappa`` are evaluated
        # pointwise (exactly) and only the derivative stencils set the order:
        #   1. pointwise heat flux  F_i = -kappa_i (dT/dx)_i  with the 4th-order
        #      central derivative (-T_{i+2} + 8T_{i+1} - 8T_{i-1} + T_{i-2})/12dx
        #   2. its divergence via the 4th-order CONSERVATIVE face interpolation
        #      Fhat_{i+1/2} = (-F_{i-1} + 7F_i + 7F_{i+1} - F_{i+2})/12,
        #      source_i = -(Fhat_{i+1/2} - Fhat_{i-1/2})/dx
        # Step 2 is the same linear flux the WENO kernel uses, so conduction no
        # longer throttles the 5th-order hydro to 2nd order.
        if config.conduction_density_weighted:
            kappa_field = kappa * primitive_state[registered_variables.density_index]
        else:
            kappa_field = kappa
        energy_source = 0.0
        for ax in range(ndim):
            dtdx = _stencil_add(
                temperature,
                indices=(2, 1, -1, -2),
                factors=(-1.0 / 12.0, 8.0 / 12.0, -8.0 / 12.0, 1.0 / 12.0),
                axis=ax,
            ) / dx
            flux = -kappa_field * dtdx
            # conservative 4th-order face flux and its simple difference
            fhat_p = _stencil_add(
                flux,
                indices=(-1, 0, 1, 2),
                factors=(-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0),
                axis=ax,
            )
            fhat_m = _stencil_add(
                flux,
                indices=(-2, -1, 0, 1),
                factors=(-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0),
                axis=ax,
            )
            energy_source = energy_source - (fhat_p - fhat_m) / dx
    elif config.conduction_density_weighted:
        # kappa = rho * alpha (Athena ``alpha_iso``): conservative
        # face-flux discretisation of div(rho alpha grad T),
        #   F_{i+1/2} = -alpha * (rho_i + rho_{i+1})/2 * (T_{i+1} - T_i)/dx,
        #   source_i  = -(F_{i+1/2} - F_{i-1/2})/dx,
        # which keeps the temperature diffusivity (gamma - 1) alpha uniform.
        rho = primitive_state[registered_variables.density_index]
        energy_source = 0.0
        for ax in range(ndim):
            t_p = _stencil_add(temperature, indices=(1,), factors=(1.0,), axis=ax)
            t_m = _stencil_add(temperature, indices=(-1,), factors=(1.0,), axis=ax)
            rho_p = _stencil_add(rho, indices=(1,), factors=(1.0,), axis=ax)
            rho_m = _stencil_add(rho, indices=(-1,), factors=(1.0,), axis=ax)
            energy_source = energy_source + (
                0.5 * (rho + rho_p) * (t_p - temperature)
                - 0.5 * (rho_m + rho) * (temperature - t_m)
            )
        energy_source = kappa * energy_source / (dx * dx)
    else:
        # second-order Laplacian: sum_axis (T_{i+1} - 2 T_i + T_{i-1}) / dx^2
        laplacian_t = sum(
            _stencil_add(temperature, indices=(1, 0, -1), factors=(1.0, -2.0, 1.0), axis=ax)
            for ax in range(ndim)
        ) / (dx * dx)

        energy_source = kappa * laplacian_t

    S = jnp.zeros_like(primitive_state)
    S = S.at[registered_variables.energy_index].set(energy_source)
    return S
