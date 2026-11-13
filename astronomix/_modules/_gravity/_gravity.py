"""
Self-gravity source terms coupling the gravitational potential to the fluid.

Assembles the total gravitational potential (self-gravity from the FFT Poisson
solve plus any external potential) and turns it into momentum and energy source
terms for the fluid. Several couplings are supported: a simple non-conservative
source and two conservative flux-based formulations (second- and fourth-order)
used by the finite-difference solver; Athena++'s second-order coupling of the
finite-volume VL2 scheme is in ``_finite_volume/_gravity_coupling.py``. Optionally
(``GravityConfig.work_flux_correction``) the potential-energy flux of the
conservative finite-difference couplings is flux-corrected so that the
gravitational work cannot drive the internal energy negative.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

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
from astronomix.variable_registry.registered_variables import RegisteredVariables
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._modules._gravity._poisson_solver import (
    _compute_gravitational_potential,
)
from astronomix._modules._gravity._utils import _pad_external_potential
from astronomix._stencil_operations._stencil_operations import (
    _shift,
    _stencil_add,
)


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
    Compute the total gravitational potential, including contributions from
    self-gravity and any external potentials.

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


# -------------------------------------------------------------
# ============== ↓ Stencil and indexing helpers ↓ =============
# -------------------------------------------------------------


def _component_index(vector_index, spatial_axis: int) -> int:
    """
    The state index of the component of a vector variable (e.g. the velocity)
    along ``spatial_axis`` (0-based). In 1D the registry stores a single int.
    """
    if isinstance(vector_index, int):
        return vector_index
    return vector_index[spatial_axis]


def _gravitational_acceleration(
    gravitational_potential: FIELD_TYPE,
    spatial_axis: int,
    grid_spacing: float,
) -> FIELD_TYPE:
    """
    Gravitational acceleration at the cell centres from the 6th-order centred
    finite difference of the potential,
    a_i = -(phi_{i+3} - 9 phi_{i+2} + 45 phi_{i+1} - 45 phi_{i-1} + 9 phi_{i-2}
    - phi_{i-3}) / (60 dx).
    """
    return -_stencil_add(
        gravitational_potential,
        indices=(3, 2, 1, -1, -2, -3),
        factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
        axis=spatial_axis,
    ) / (60.0 * grid_spacing)


def _potential_at_right_face(
    gravitational_potential: FIELD_TYPE,
    spatial_axis: int,
) -> FIELD_TYPE:
    """
    The potential interpolated to the right cell face i + 1/2 with the
    6th-order symmetric stencil (3, -25, 150, 150, -25, 3) / 256.
    """
    return _stencil_add(
        gravitational_potential,
        indices=(-2, -1, 0, 1, 2, 3),
        factors=(3.0, -25.0, 150.0, 150.0, -25.0, 3.0),
        axis=spatial_axis,
    ) / 256.0


def _fourth_order_work_correction(
    primitive_state: STATE_TYPE,
    gravitational_potential: FIELD_TYPE,
    axis: int,
    grid_spacing: float,
    registered_variables: RegisteredVariables,
) -> FIELD_TYPE:
    """
    The face correction term of the fourth-order product flux.

    The fourth-order product flux F phi needs a correction built from the
    curvature of the potential and the gradient of the momentum density f;
    second order on the correction is sufficient to reach the overall order.
    The cell-centre term phi'' f + 2 phi' f' is averaged onto the face
    i + 1/2.

    Args:
        primitive_state: The primitive state.
        gravitational_potential: The total potential at the cell centres.
        axis: The state-array axis (1-based spatial axis).
        grid_spacing: The grid spacing.
        registered_variables: The registered variables.

    Returns:
        The correction term at the faces i + 1/2.
    """
    spatial_axis = axis - 1
    velocity_index = _component_index(registered_variables.velocity_index, spatial_axis)
    momentum = primitive_state[registered_variables.density_index] * primitive_state[velocity_index]

    # phi' (6th order) and phi'' (2nd order) at the cell centres.
    potential_slope = _stencil_add(
        gravitational_potential,
        indices=(3, 2, 1, -1, -2, -3),
        factors=(1.0, -9.0, 45.0, -45.0, 9.0, -1.0),
        axis=spatial_axis,
    ) / (60.0 * grid_spacing)
    potential_curvature = (
        _shift(gravitational_potential, -1, axis=spatial_axis)
        - 2.0 * gravitational_potential
        + _shift(gravitational_potential, 1, axis=spatial_axis)
    ) / grid_spacing**2

    # f' at the cell centres (2nd order).
    momentum_slope = (
        _shift(momentum, -1, axis=spatial_axis) - _shift(momentum, 1, axis=spatial_axis)
    ) / (2.0 * grid_spacing)

    centre = potential_curvature * momentum + 2.0 * potential_slope * momentum_slope
    return 0.5 * (centre + _shift(centre, -1, axis=spatial_axis))


# -------------------------------------------------------------
# ============== ↑ Stencil and indexing helpers ↑ =============
# -------------------------------------------------------------


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
    Build the gravity source term of the stages for the full state (the
    finite-difference schemes; ``RK2_SSP`` and ``MUSCL_HANCOCK`` with
    ``SIMPLE_SOURCE``).

    Computes the total gravitational potential and assembles the momentum and
    energy source contributions for every spatial axis, according to the
    configured coupling (``SIMPLE_SOURCE`` or one of the conservative,
    flux-based schemes of the finite-difference solver). With ``GravityConfig.work_flux_correction`` the
    potential-energy flux of the conservative couplings is then limited (see
    ``_flux_correct_gravitational_work``).

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

    gravity_source = jnp.zeros_like(primitive_state)
    grid_spacing = config.grid_spacing
    self_gravity_version = config.gravity_config.self_gravity_version

    gravitational_potential = _compute_total_potential(
        primitive_state[registered_variables.density_index],
        grid_spacing,
        config,
        params,
        registered_variables,
        params.gravitational_constant,
    )

    if self_gravity_version == SIMPLE_SOURCE:

        for spatial_axis in range(config.dimensionality):
            momentum_index = _component_index(registered_variables.momentum_index, spatial_axis)
            velocity_index = _component_index(registered_variables.velocity_index, spatial_axis)
            density = primitive_state[registered_variables.density_index]
            velocity = primitive_state[velocity_index]

            acceleration = _gravitational_acceleration(
                gravitational_potential,
                spatial_axis,
                grid_spacing,
            )

            # Simple (non-conservative) coupling: rho * a for momentum and
            # rho * v * a for energy (a barotropic gas has no energy row).
            axis_source = jnp.zeros_like(primitive_state)
            axis_source = axis_source.at[momentum_index].set(density * acceleration)
            if registered_variables.energy_index >= 0:
                axis_source = axis_source.at[registered_variables.energy_index].set(
                    density * velocity * acceleration
                )

            gravity_source += axis_source * dt

    elif self_gravity_version == SECOND_ORDER_CONSERVATIVE:

        for spatial_axis in range(config.dimensionality):
            momentum_index = _component_index(registered_variables.momentum_index, spatial_axis)
            density = primitive_state[registered_variables.density_index]
            potential_at_cell = gravitational_potential

            # Momentum source from the 6th-order centred potential gradient.
            acceleration = _gravitational_acceleration(
                gravitational_potential,
                spatial_axis,
                grid_spacing,
            )

            axis_source = jnp.zeros_like(primitive_state)
            axis_source = axis_source.at[momentum_index].set(density * acceleration)

            # Energy source built from the density fluxes so it is consistent
            # with the conservative update (no separate ``drho`` term needed).
            # The potential is interpolated to the right cell face i + 1/2; the
            # left face value is obtained by a shift.
            potential_at_right_face = _potential_at_right_face(
                gravitational_potential,
                spatial_axis,
            )
            density_flux_right = density_fluxes[spatial_axis]  # at i + 1/2
            density_flux_left = _shift(density_fluxes[spatial_axis], 1, axis=spatial_axis)
            potential_at_left_face = _shift(potential_at_right_face, 1, axis=spatial_axis)

            # Energy source W_i = -[F_right (phi_right - phi_i)
            # + F_left (phi_i - phi_left)] / dx, which is the discrete form of
            # -div(F phi) + phi div(F) = -rho v grad(phi).
            energy_source = -(
                density_flux_right * (potential_at_right_face - potential_at_cell)
                + density_flux_left * (potential_at_cell - potential_at_left_face)
            ) / grid_spacing

            axis_source = axis_source.at[registered_variables.energy_index].set(energy_source)

            gravity_source += axis_source * dt

    elif self_gravity_version == FOURTH_ORDER_CONSERVATIVE:

        for spatial_axis in range(config.dimensionality):
            momentum_index = _component_index(registered_variables.momentum_index, spatial_axis)
            density = primitive_state[registered_variables.density_index]

            potential_at_right_face = _potential_at_right_face(
                gravitational_potential,
                spatial_axis,
            )

            acceleration = _gravitational_acceleration(
                gravitational_potential,
                spatial_axis,
                grid_spacing,
            )

            axis_source = jnp.zeros_like(primitive_state)
            axis_source = axis_source.at[momentum_index].set(density * acceleration)

            # Corrected product flux q_hat = F phi_face - dx^2 / 24 * correction
            # and the resulting energy source -div(q_hat).
            work_correction = _fourth_order_work_correction(
                primitive_state,
                gravitational_potential,
                spatial_axis + 1,
                grid_spacing,
                registered_variables,
            )
            corrected_flux = (
                density_fluxes[spatial_axis] * potential_at_right_face
                - (grid_spacing**2 / 24.0) * work_correction
            )
            energy_source = -1.0 / grid_spacing * (
                corrected_flux - _shift(corrected_flux, 1, axis=spatial_axis)
            )

            axis_source = axis_source.at[registered_variables.energy_index].set(energy_source)
            gravity_source += axis_source * dt

        # Account for the change in potential energy due to the density change.
        gravity_source = gravity_source.at[registered_variables.energy_index].add(
            -drho * gravitational_potential
        )
    else:
        raise NotImplementedError("This scheme is not implemented.")

    if (
        config.gravity_config.work_flux_correction
        and self_gravity_version != SIMPLE_SOURCE
    ):
        gravity_source = _flux_correct_gravitational_work(
            gravity_source,
            primitive_state,
            gravitational_potential,
            density_fluxes,
            dt,
            config,
            params,
            registered_variables,
        )

    return gravity_source


def _flux_correct_gravitational_work(
    gravity_source: STATE_TYPE,
    primitive_state: STATE_TYPE,
    gravitational_potential: FIELD_TYPE,
    density_fluxes,
    dt,
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Flux-corrected transport of the potential-energy flux.

    The conservative energy source already in ``gravity_source`` corresponds to
    the scheme's high-order potential-energy flux. Writing the face's
    donor-charged flux q_low = F phi_downwind (the cell the mass leaves pays
    the whole climb), the difference A = q_high - q_low is an antidiffusive
    flux: replacing q_high by q_low + psi A changes the energy source by
    + d/dx [(1 - psi) A], which keeps total energy exactly conserved for any
    psi.

    psi follows a one-sided Zalesak (1979) limiter (lower bound only): per
    cell, the antidiffusive contributions that lower its internal energy are
    scaled so that, together with the low-order coupling's own internal-energy
    change, the loss rate stays below half the internal energy per
    wave-crossing time dx / (|v| + c); each face takes the scaling of the cell
    it drains. All quantities are rates, so psi is independent of dt. Where
    the low-order coupling alone already exceeds the bound, the budget is
    clipped to zero and the bound itself is not guaranteed.

    Args:
        gravity_source: The full-state gravity source over the stage time step.
        primitive_state: The primitive state of the stage.
        gravitational_potential: The total potential at the cell centres.
        density_fluxes: The per-axis density fluxes at the faces i + 1/2.
        dt: The stage time step.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The source with the flux-corrected energy component.
    """
    energy_index = registered_variables.energy_index
    grid_spacing = config.grid_spacing
    gamma = params.gamma

    # -------------------------------------------------------------
    # ============== ↓ Budget inputs ↓ ============================
    # -------------------------------------------------------------

    velocity_indices = [
        _component_index(registered_variables.velocity_index, spatial_axis)
        for spatial_axis in range(config.dimensionality)
    ]
    momentum_indices = [
        _component_index(registered_variables.momentum_index, spatial_axis)
        for spatial_axis in range(config.dimensionality)
    ]

    density = primitive_state[registered_variables.density_index]
    pressure = jnp.maximum(primitive_state[registered_variables.pressure_index], 0.0)
    internal_energy = pressure / (gamma - 1.0)
    speed = jnp.sqrt(sum(primitive_state[index] ** 2 for index in velocity_indices))
    wave_speed = speed + jnp.sqrt(gamma * pressure / jnp.maximum(density, 1e-30))
    safe_dt = jnp.maximum(dt, 1e-30)

    # The kinetic part of the gravitational work rate, v . (rho a);
    # subtracting it from the total energy source leaves the internal-energy
    # change.
    kinetic_rate = sum(
        primitive_state[velocity_index] * gravity_source[momentum_index]
        for velocity_index, momentum_index in zip(velocity_indices, momentum_indices)
    ) / safe_dt

    # -------------------------------------------------------------
    # ============== ↑ Budget inputs ↑ ============================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Antidiffusive potential-energy fluxes ↓ ====
    # -------------------------------------------------------------

    # The antidiffusive fluxes per axis and their effect on each cell.
    antidiffusive_fluxes = []
    low_order_extra_rate = jnp.zeros_like(density)
    lowering_rate = jnp.zeros_like(density)
    for spatial_axis in range(config.dimensionality):
        mass_flux = density_fluxes[spatial_axis]
        potential_left = gravitational_potential
        potential_right = _shift(gravitational_potential, -1, axis=spatial_axis)
        potential_at_right_face = _potential_at_right_face(
            gravitational_potential,
            spatial_axis,
        )
        high_order_flux = mass_flux * potential_at_right_face
        if config.gravity_config.self_gravity_version == FOURTH_ORDER_CONSERVATIVE:
            high_order_flux = high_order_flux - (
                grid_spacing**2 / 24.0
            ) * _fourth_order_work_correction(
                primitive_state,
                gravitational_potential,
                spatial_axis + 1,
                grid_spacing,
                registered_variables,
            )
        low_order_flux = jnp.where(
            mass_flux > 0.0,
            mass_flux * potential_right,
            mass_flux * potential_left,
        )
        antidiffusive = high_order_flux - low_order_flux
        antidiffusive_fluxes.append(antidiffusive)

        # Replacing the high-order by the low-order flux adds
        # (A_{i+1/2} - A_{i-1/2}) / dx to the energy source of cell i.
        low_order_extra_rate = low_order_extra_rate + (
            antidiffusive - _shift(antidiffusive, 1, axis=spatial_axis)
        ) / grid_spacing

        # With psi = 1, face i + 1/2 lowers cell i by A / dx when A > 0, and
        # cell i + 1 by -A / dx when A < 0.
        lowered_by_right_face = jnp.maximum(antidiffusive, 0.0) / grid_spacing
        lowered_by_left_face = jnp.maximum(
            -_shift(antidiffusive, 1, axis=spatial_axis),
            0.0,
        ) / grid_spacing
        lowering_rate = lowering_rate + lowered_by_right_face + lowered_by_left_face

    # -------------------------------------------------------------
    # ============== ↑ Antidiffusive potential-energy fluxes ↑ ====
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Zalesak cell fractions ↓ ===================
    # -------------------------------------------------------------

    high_order_rate = gravity_source[energy_index] / safe_dt
    low_order_internal_rate = high_order_rate + low_order_extra_rate - kinetic_rate

    # The budget deliberately ignores the hydrodynamic rate. Crediting the
    # per-stage (linearised) heating estimate lets through high-order splits
    # whose heating never materialises, which drives cold collapsing gas to
    # negative pressure; debiting expansion cooling makes the limiter reject
    # far more potential-energy flux than necessary.
    budget = jnp.maximum(
        0.5 * internal_energy * wave_speed / grid_spacing + low_order_internal_rate,
        0.0,
    )
    cell_fraction = jnp.where(
        lowering_rate > budget,
        budget / jnp.maximum(lowering_rate, 1e-30),
        1.0,
    )

    # -------------------------------------------------------------
    # ============== ↑ Zalesak cell fractions ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============== ↓ Limited energy correction ↓ ================
    # -------------------------------------------------------------

    # Each face takes the fraction of the cell it drains: the left cell for
    # A > 0, the right cell for A < 0.
    correction = jnp.zeros_like(density)
    for spatial_axis in range(config.dimensionality):
        antidiffusive = antidiffusive_fluxes[spatial_axis]
        psi = jnp.where(
            antidiffusive > 0.0,
            cell_fraction,
            _shift(cell_fraction, -1, axis=spatial_axis),
        )
        rejected = (1.0 - psi) * antidiffusive
        correction = correction + (
            rejected - _shift(rejected, 1, axis=spatial_axis)
        ) / grid_spacing

    # -------------------------------------------------------------
    # ============== ↑ Limited energy correction ↑ ================
    # -------------------------------------------------------------

    return gravity_source.at[energy_index].add(correction * dt)
