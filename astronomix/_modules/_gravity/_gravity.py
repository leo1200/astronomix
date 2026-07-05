"""
Self-gravity source terms coupling the gravitational potential to the fluid.

Assembles the total gravitational potential (self-gravity from the FFT Poisson
solve plus any external potential) and turns it into momentum and energy source
terms for the fluid. Several couplings are supported: a simple non-conservative
source and two conservative flux-based formulations (second- and fourth-order)
used by the finite-difference solver.
"""

# general
from functools import partial

# typing
from typing import Tuple, Union
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FIELD_TYPE,
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._modules._gravity._utils import _pad_external_potential
from astronomix._stencil_operations._stencil_operations import _shift, _stencil_add

@partial(jax.jit, static_argnames=["grid_spacing", "config", "registered_variables"])
def _compute_total_potential(
    gas_density: FIELD_TYPE,
    grid_spacing: float,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
    G: Union[float, Float[Array, ""]] = 1.0,
) -> FIELD_TYPE:
    """
    Compute the total gravitational potential, including contributions from self-gravity and any external potentials.

    Args:
        gas_density: The gas density field (ghost-cell padded, i.e. the
            shape of a single state field).
        grid_spacing: The grid spacing.
        config: The simulation configuration.
        params: The simulation parameters (provides the external potential).
        registered_variables: The registered variables.
        G: The gravitational constant.

    Returns:
        The total gravitational potential, with the same shape as gas_density.
    """
    total_potential = jnp.zeros_like(gas_density)

    # Self-gravity contribution from the FFT Poisson solve.
    if config.gravity_config.self_gravity:
        total_potential = total_potential + _compute_gravitational_potential(
            gas_density,
            grid_spacing,
            config,
            G,
        )

    # External-potential contribution. The external potential is supplied on the
    # bare grid, so it is given ghost cells matching the (here padded) density
    # field, filled according to the boundary conditions.
    if config.gravity_config.external_potential:
        external_potential = _pad_external_potential(
            params.gravitational_potential,
            gas_density,
            config,
            registered_variables,
            params,
        )
        total_potential = total_potential + external_potential

    return total_potential

def _fd_gravity_source(
    primitive_state: STATE_TYPE,
    density_fluxes,
    drho,
    dt,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
):
    """
    Build the finite-difference self-gravity source term for the full state.

    Computes the total gravitational potential and assembles the momentum and
    energy source contributions for every spatial axis, according to the
    configured coupling (``SIMPLE_SOURCE`` or one of the conservative,
    flux-based schemes).

    Args:
        primitive_state: The primitive state array.
        density_fluxes: The per-axis density fluxes at the cell faces, used by
            the conservative energy couplings.
        drho: The density change over the step, used by the conservative energy
            couplings to keep the ``phi * drho`` term consistent.
        dt: The time step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The full-state source term to be added over this time step.
    """

    S = jnp.zeros_like(primitive_state)

    gravitational_potential = _compute_total_potential(
        primitive_state[registered_variables.density_index],
        config.grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )

    if config.gravity_config.self_gravity_version == SIMPLE_SOURCE:

        for axis in range(1, config.dimensionality + 1):
            rho = primitive_state[registered_variables.density_index]
            v_axis = primitive_state[axis]

            # 6th-order centered finite difference for the gravitational
            # acceleration, a_i = -(phi_{i+3} - 9 phi_{i+2} + 45 phi_{i+1}
            # - 45 phi_{i-1} + 9 phi_{i-2} - phi_{i-3}) / (60 dx). The stencil
            # axis is ``axis - 1`` because the leading state axis indexes the
            # fields, not the spatial dimensions.
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=axis - 1,
            ) / (60.0 * config.grid_spacing)

            # Simple (non-conservative) coupling: rho * a for momentum and
            # rho * v * a for energy.
            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)
            S_axis = S_axis.at[registered_variables.pressure_index].set(
                rho * v_axis * acceleration
            )

            S += S_axis * dt
    elif config.gravity_config.self_gravity_version == SECOND_ORDER_CONSERVATIVE:

        for axis in range(1, config.dimensionality + 1):
            rho = primitive_state[registered_variables.density_index]
            phi_cell = gravitational_potential

            # Momentum source from the 6th-order centered potential gradient.
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=axis - 1,
            ) / (60.0 * config.grid_spacing)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)

            # Energy source built from the density fluxes so it is consistent
            # with the conservative update (no separate ``drho`` term needed).
            # The potential is interpolated to the right cell face i+1/2 with a
            # 6th-order symmetric stencil; the left face value is obtained by a
            # shift.
            phi_face = _stencil_add(
                gravitational_potential,
                indices=(-2, -1, 0, 1, 2, 3),
                factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0),
                axis=axis - 1,
            ) / 256.0

            F_right = density_fluxes[axis - 1]  # density flux at i+1/2
            F_left = _shift(density_fluxes[axis - 1], 1, axis=axis - 1)  # at i-1/2
            phi_face_left = _shift(phi_face, 1, axis=axis - 1)  # phi at i-1/2

            # Energy source W_i = -[F_right (phi_right - phi_i)
            # + F_left (phi_i - phi_left)] / dx, which is the discrete form of
            # -div(F phi) + phi div(F) = -rho v grad(phi).
            energy_source = -(
                F_right * (phi_face - phi_cell)
                + F_left * (phi_cell - phi_face_left)
            ) / config.grid_spacing

            S_axis = S_axis.at[registered_variables.energy_index].set(energy_source)

            S += S_axis * dt

    elif config.gravity_config.self_gravity_version == FOURTH_ORDER_CONSERVATIVE:
        for axis in range(1, config.dimensionality + 1):
            spatial_axis = axis - 1

            rho = primitive_state[registered_variables.density_index]
            v_axis = primitive_state[axis]
            dx = config.grid_spacing

            # 6th-order interpolation of the potential to the cell faces.
            phi_face = _stencil_add(
                gravitational_potential,
                indices=(-2, -1, 0, 1, 2, 3),
                factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0),
                axis=spatial_axis,
            ) / 256.0

            # 6th-order gravitational acceleration at the cell centers.
            acceleration = -_stencil_add(
                gravitational_potential,
                indices=(3, 2, 1, -1, -2, -3),
                factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
                axis=spatial_axis,
            ) / (60.0 * dx)

            S_axis = jnp.zeros_like(primitive_state)
            S_axis = S_axis.at[axis].set(rho * acceleration)

            # Corrected product form for the energy source. The fourth-order
            # product flux needs a correction term built from the curvature of
            # the potential and the gradient of the momentum density; second
            # order on the correction is sufficient to reach the overall order.
            f = rho * v_axis  # momentum density (rho v) at cell centers
            dPhi = -acceleration  # phi' at cell centers (6th order, reused)

            d2Phi = (  # phi'' at cell centers (2nd order)
                _shift(gravitational_potential, -1, axis=spatial_axis)
                - 2.0 * gravitational_potential
                + _shift(gravitational_potential, 1, axis=spatial_axis)
            ) / dx**2

            df = (  # f' at cell centers (2nd order)
                _shift(f, -1, axis=spatial_axis) - _shift(f, 1, axis=spatial_axis)
            ) / (2.0 * dx)

            # Correction at the cell centers, then averaged onto the faces.
            corr_cc = d2Phi * f + 2.0 * dPhi * df
            corr_face = 0.5 * (corr_cc + _shift(corr_cc, -1, axis=spatial_axis))

            # Corrected product flux and the resulting energy source -div(q_hat).
            q_hat = density_fluxes[axis - 1] * phi_face - (dx**2 / 24.0) * corr_face
            S_energy = -1.0 / dx * (q_hat - _shift(q_hat, 1, axis=spatial_axis))

            S_axis = S_axis.at[registered_variables.pressure_index].set(S_energy)
            S += S_axis * dt

        # Account for the change in potential energy due to the density change.
        S = S.at[registered_variables.energy_index].add(
            -drho * gravitational_potential
        )
    else:
        raise NotImplementedError("This scheme is not implemented.")

    return S

# @jaxtyped(typechecker=typechecker)
@partial(
    jax.jit, static_argnames=["axis", "grid_spacing", "registered_variables", "config"]
)
def _gravitational_source_term_along_axis(
    gravitational_potential: FIELD_TYPE,
    primitive_state: STATE_TYPE,
    grid_spacing: float,
    registered_variables: RegisteredVariables,
    dt: Union[float, Float[Array, ""]],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    axis: int,
) -> STATE_TYPE:
    """
    Compute the source term for the self-gravity solver along a single axis.
    Currently, simply density * gravitational_acceleration for the momentum
    and density * velocity * gravitational_acceleration for the energy.

    Args:
        gravitational_potential: The gravitational potential.
        primitive_state: The primitive state.
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        helper_data: The helper data.
        axis: The axis along which to compute the source term.

    Returns:
        The source term.

    """

    rho = primitive_state[registered_variables.density_index]
    v_axis = primitive_state[axis]

    # 2nd-order centered gravitational acceleration, a_i = -(phi_{i+1}
    # - phi_{i-1}) / (2 dx). The stencil axis is ``axis - 1`` because the
    # leading state axis indexes the fields, not the spatial dimensions.
    acceleration = -_stencil_add(
        gravitational_potential,
        indices=(1, -1),
        factors=(1.0, -1.0),
        axis=axis - 1,
    ) / (2 * grid_spacing)

    source_term = jnp.zeros_like(primitive_state)

    # set momentum source
    source_term = source_term.at[axis].set(rho * acceleration)

    # finite-volume self-gravity supports only the SIMPLE_SOURCE coupling
    # (the FD-only conservative flux schemes live in _fd_gravity_source).
    source_term = source_term.at[registered_variables.pressure_index].set(
        rho * v_axis * acceleration
    )

    return source_term