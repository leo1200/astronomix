"""
Thermal conduction for the finite-difference scheme.

We add a Fourier heat-conduction term to the energy equation,

    d(rho E)/dt  +=  div(kappa grad T) ,

with the temperature taken from the ideal-gas relation

    T = p / rho            (code units, specific gas constant R = 1).

The conductivity is either the constant ``kappa = params.thermal_conductivity``
or, with ``config.diffusion_config.conduction_density_weighted``, ``kappa = rho * alpha`` with
the diffusivity ``alpha = params.thermal_conductivity``. Three discretisations
are provided:

* 2nd order, constant kappa: ``kappa * laplacian(T)`` with the standard
  three-point Laplacian per axis (seven-point in 3D). The stencil is a constant
  linear operator on T, hence trivially differentiable, and the explicit
  parabolic time step stays cheap.
* 2nd order, ``kappa = rho * alpha``: a conservative face-flux discretisation
  with the face density taken as the mean of the two adjacent cells.
* 4th order (``config.diffusion_config.conduction_order == 4``, either conductivity): the
  pointwise heat flux from the 4th-order central first derivative, and its
  divergence through the 4th-order conservative face interpolation. This is
  the same linear flux the WENO kernel uses, so conduction does not reduce the
  5th-order hydrodynamics to 2nd order.

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
    density = primitive_state[registered_variables.density_index]
    pressure = primitive_state[registered_variables.pressure_index]
    return pressure / density


def _fourth_order_conduction_source(
    temperature,
    conductivity,
    grid_spacing,
    dimensionality,
):
    """
    4th-order finite-difference conductive energy source div(kappa grad T).

    In a finite-difference scheme the state is the pointwise value, so
    ``T = p / rho`` and ``kappa`` are evaluated pointwise (exactly) and only the
    derivative stencils set the order:

    1. the pointwise heat flux ``F_i = -kappa_i (dT/dx)_i`` with the 4th-order
       central derivative ``(-T_{i+2} + 8 T_{i+1} - 8 T_{i-1} + T_{i-2}) / (12 dx)``;
    2. its divergence via the 4th-order conservative face interpolation
       ``Fhat_{i+1/2} = (-F_{i-1} + 7 F_i + 7 F_{i+1} - F_{i+2}) / 12``,
       ``source_i = -(Fhat_{i+1/2} - Fhat_{i-1/2}) / dx``.

    Args:
        temperature: The temperature field.
        conductivity: The conductivity kappa (a scalar or a field).
        grid_spacing: The grid spacing.
        dimensionality: The number of spatial dimensions.

    Returns:
        The conductive energy source field.
    """
    energy_source = 0.0
    for axis in range(dimensionality):
        temperature_gradient = _stencil_add(
            temperature,
            indices=(2, 1, -1, -2),
            factors=(-1.0 / 12.0, 8.0 / 12.0, -8.0 / 12.0, 1.0 / 12.0),
            axis=axis,
        ) / grid_spacing
        heat_flux = -conductivity * temperature_gradient

        # The conservative 4th-order face fluxes at i + 1/2 and i - 1/2; their
        # simple difference is the divergence.
        face_flux_right = _stencil_add(
            heat_flux,
            indices=(-1, 0, 1, 2),
            factors=(-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0),
            axis=axis,
        )
        face_flux_left = _stencil_add(
            heat_flux,
            indices=(-2, -1, 0, 1),
            factors=(-1.0 / 12.0, 7.0 / 12.0, 7.0 / 12.0, -1.0 / 12.0),
            axis=axis,
        )
        energy_source = energy_source - (face_flux_right - face_flux_left) / grid_spacing
    return energy_source


def _density_weighted_conduction_source(
    temperature,
    density,
    diffusivity,
    grid_spacing,
    dimensionality,
):
    """
    2nd-order conservative conductive energy source div(rho alpha grad T).

    Uses ``kappa = rho * alpha`` (Athena ``alpha_iso``) with the face fluxes

        F_{i+1/2} = -alpha * (rho_i + rho_{i+1}) / 2 * (T_{i+1} - T_i) / dx,
        source_i  = -(F_{i+1/2} - F_{i-1/2}) / dx,

    which keeps the temperature diffusivity (gamma - 1) alpha uniform.

    Args:
        temperature: The temperature field.
        density: The density field.
        diffusivity: The diffusivity alpha.
        grid_spacing: The grid spacing.
        dimensionality: The number of spatial dimensions.

    Returns:
        The conductive energy source field.
    """
    energy_source = 0.0
    for axis in range(dimensionality):
        temperature_right = _stencil_add(temperature, indices=(1,), factors=(1.0,), axis=axis)
        temperature_left = _stencil_add(temperature, indices=(-1,), factors=(1.0,), axis=axis)
        density_right = _stencil_add(density, indices=(1,), factors=(1.0,), axis=axis)
        density_left = _stencil_add(density, indices=(-1,), factors=(1.0,), axis=axis)
        energy_source = energy_source + (
            0.5 * (density + density_right) * (temperature_right - temperature)
            - 0.5 * (density_left + density) * (temperature - temperature_left)
        )
    return diffusivity * energy_source / (grid_spacing * grid_spacing)


def _laplacian_conduction_source(
    temperature,
    conductivity,
    grid_spacing,
    dimensionality,
):
    """
    2nd-order conductive energy source ``kappa * laplacian(T)`` for a constant
    conductivity, with the Laplacian
    ``sum_axis (T_{i+1} - 2 T_i + T_{i-1}) / dx^2``.

    Args:
        temperature: The temperature field.
        conductivity: The constant conductivity kappa.
        grid_spacing: The grid spacing.
        dimensionality: The number of spatial dimensions.

    Returns:
        The conductive energy source field.
    """
    temperature_laplacian = sum(
        _stencil_add(
            temperature,
            indices=(1, 0, -1),
            factors=(1.0, -2.0, 1.0),
            axis=axis,
        )
        for axis in range(dimensionality)
    ) / (grid_spacing * grid_spacing)

    return conductivity * temperature_laplacian


@partial(jax.jit, static_argnames=("config", "registered_variables"))
def fd_conduction_source(primitive_state, params, config, registered_variables):
    """
    Conductive energy source div(kappa grad T) for the finite-difference scheme.

    Selects the discretisation from ``config.diffusion_config.conduction_order`` and
    ``config.diffusion_config.conduction_density_weighted`` (see the module docstring). The
    result is meant to be accumulated (times ``dt``) onto the conserved-state
    right-hand side in the time-integrator source assembly.

    Args:
        primitive_state: The primitive state array.
        params: The simulation parameters (providing the conductivity, or the
            diffusivity for the density-weighted form).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        A state-shaped array with only the energy slot populated.
    """
    conductivity = params.thermal_conductivity
    grid_spacing = config.grid_spacing
    dimensionality = config.dimensionality

    temperature = _temperature(primitive_state, registered_variables)

    if config.diffusion_config.conduction_order == 4:
        if config.diffusion_config.conduction_density_weighted:
            conductivity_field = (
                conductivity * primitive_state[registered_variables.density_index]
            )
        else:
            conductivity_field = conductivity
        energy_source = _fourth_order_conduction_source(
            temperature,
            conductivity_field,
            grid_spacing,
            dimensionality,
        )
    elif config.diffusion_config.conduction_density_weighted:
        energy_source = _density_weighted_conduction_source(
            temperature,
            primitive_state[registered_variables.density_index],
            conductivity,
            grid_spacing,
            dimensionality,
        )
    else:
        energy_source = _laplacian_conduction_source(
            temperature,
            conductivity,
            grid_spacing,
            dimensionality,
        )

    source_term = jnp.zeros_like(primitive_state)
    source_term = source_term.at[registered_variables.energy_index].set(energy_source)
    return source_term
