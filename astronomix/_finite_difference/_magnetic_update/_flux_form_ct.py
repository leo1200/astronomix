"""
Constrained transport of the finite-difference scheme with a Riemann flux
(``FDSolverConfig.scheme = RIEMANN_FLUX``): flux-form WENO CT.

Every edge EMF is the difference of two one-dimensional transport fluxes of
the face fields that meet at the edge,

    E_z = G[b_x along y] - G[b_y along x],
    E_x = G[b_y along z] - G[b_z along y],
    E_y = G[b_z along x] - G[b_x along z],

each the two-state HLL flux of the UCT-HLL form (Del Zanna et al. 2007,
eq. 61),

    G = [a+ v^L b^L + a- v^R b^R - a+ a- (b^R - b^L)] / (a+ + a-),
    a+ = max(0, v^L + c, v^R + c),  a- = max(0, c - v^L, c - v^R),

where v^L,R and b^L,R are the left- and right-biased seventh-order WENO-Z
point interpolations to the edge of the face velocity (the cell velocity
interpolated to the face with the sixth-order midpoint stencil) and of the
face field, and c is the cold-plasma speed |B| / sqrt(rho), the largest of the
four cells around the edge. Super-Alfvenic flow gives the upwind flux
v^up b^up, a fluid at rest the Lax-Friedrichs form with the speed c. The face
fields are advanced with the staggered sixth-order curl of the EMFs
(``_energy_flux.edge_emf_curl``), so div B = 0 is kept to rounding.

The dissipation reads and writes the face fields through the same staggered
geometry, which is negative definite on divergence-free face fields for every
field orientation; the EMF of the production scheme
(``_constrained_transport.py``) routes the WENO dissipation of the
cell-centred field through the face-to-centre interpolation instead. The WENO
weights compare the smoothness indicators with ``CT_WENO_EPSILON_RELATIVE``
times the largest squared value of the stencil, so they do not depend on the
field strength in code units.
"""

# general
from functools import partial

# jax
import jax
import jax.numpy as jnp

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._magnetic_update._energy_flux import ct_energy_flux_rhs, edge_emf_curl
from astronomix._finite_difference._ratio_rules import _floored_ratio, _ratio
from astronomix._fluid_equations._eigen_mhd_iso import zero_safe_sqrt
from astronomix._stencil_operations._stencil_operations import _shift

XAXIS = 0
YAXIS = 1
ZAXIS = 2

#: Added to the relative WENO epsilon so that a field-free stencil gives 0 / 0
#: nowhere (the weights are then linear); far below any squared field scale.
EPSILON_FLOOR = 1e-30

#: The WENO epsilon of the edge interpolations relative to the largest squared
#: value of the stencil.
CT_WENO_EPSILON_RELATIVE = 1e-6


def _value_at(field, offset, axis):
    """``field`` shifted so that index i holds the value at i + offset."""
    return _shift(field, -offset, axis=axis)


def _interp_center_to_face_6(field, axis):
    """
    Sixth-order midpoint interpolation from the cell centres to the faces:
    index i holds the value at i + 1/2.
    """
    return (
        3.0 * _value_at(field, -2, axis)
        - 25.0 * _value_at(field, -1, axis)
        + 150.0 * field
        + 150.0 * _value_at(field, 1, axis)
        - 25.0 * _value_at(field, 2, axis)
        + 3.0 * _value_at(field, 3, axis)
    ) / 256.0


def _weno5z_point_interpolation(f0, f1, f2, f3, f4, epsilon, frozen=False):
    """
    Fifth-order WENO-Z interpolation of point values f0 ... f4 (at the
    positions -2 ... 2) to the half point +1/2, with the weights of the
    fluid's WENO-Z (``alpha_k = gamma_k (1 + tau / (beta_k + epsilon))``).
    With the linear weights it is (3, -20, 90, 60, -5) / 128. ``frozen``
    stops the gradient through the weights (``FDSolverConfig.weno_ad_frozen_weights``).
    """
    p0 = (3.0 * f0 - 10.0 * f1 + 15.0 * f2) / 8.0
    p1 = (-f1 + 6.0 * f2 + 3.0 * f3) / 8.0
    p2 = (3.0 * f2 + 6.0 * f3 - f4) / 8.0
    beta0 = 13.0 / 12.0 * (f0 - 2.0 * f1 + f2) ** 2 + 0.25 * (f0 - 4.0 * f1 + 3.0 * f2) ** 2
    beta1 = 13.0 / 12.0 * (f1 - 2.0 * f2 + f3) ** 2 + 0.25 * (f1 - f3) ** 2
    beta2 = 13.0 / 12.0 * (f2 - 2.0 * f3 + f4) ** 2 + 0.25 * (3.0 * f2 - 4.0 * f3 + f4) ** 2
    tau = jnp.abs(beta0 - beta2)
    alpha0 = (1.0 / 16.0) * (1.0 + _ratio(tau, beta0 + epsilon))
    alpha1 = (10.0 / 16.0) * (1.0 + _ratio(tau, beta1 + epsilon))
    alpha2 = (5.0 / 16.0) * (1.0 + _ratio(tau, beta2 + epsilon))
    if frozen:
        alpha0, alpha1, alpha2 = (jax.lax.stop_gradient(a) for a in (alpha0, alpha1, alpha2))
    return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2) / (alpha0 + alpha1 + alpha2)


def _weno7z_point_interpolation(f0, f1, f2, f3, f4, f5, f6, epsilon, frozen=False):
    """
    Seventh-order WENO-Z interpolation of point values f0 ... f6 (at the
    positions -3 ... 3) to the half point +1/2: four cubic substencils with
    the linear weights (1, 21, 35, 7) / 64, the Balsara & Shu (2000)
    smoothness indicators and tau_7 = |beta_0 + 3 beta_1 - 3 beta_2 - beta_3|
    (Castro, Costa & Don 2011). With the linear weights it is
    (-5, 42, -175, 700, 525, -70, 7) / 1024. ``frozen`` stops the gradient
    through the weights.
    """
    p0 = (-5.0 * f0 + 21.0 * f1 - 35.0 * f2 + 35.0 * f3) / 16.0
    p1 = (f1 - 5.0 * f2 + 15.0 * f3 + 5.0 * f4) / 16.0
    p2 = (-f2 + 9.0 * f3 + 9.0 * f4 - f5) / 16.0
    p3 = (5.0 * f3 + 15.0 * f4 - 5.0 * f5 + f6) / 16.0
    # The Balsara & Shu (2000) smoothness indicators in their derivative form,
    # beta_k = (a1 + a3 / 24)^2 + 13/12 a2^2 + 781/720 a3^2 with a1, a2, a3 the
    # derivatives of the substencil cubic at the target point (scaled so that
    # linear data gives 1, as for the fifth-order ones): a1 + a3 / 24 is the
    # third-order one-sided first derivative, a2 and a3 the second and third
    # differences. Written in the first differences u they are sums of squares,
    # never negative; the expanded quadratic forms in f (coefficients up to
    # 17246 / 240) cancel to O(ulp f^2), which in float32 exceeds the
    # indicators of smooth data from N ~ 100 on and drove beta_k + epsilon
    # negative.
    u0, u1, u2, u3, u4, u5 = f1 - f0, f2 - f1, f3 - f2, f4 - f3, f5 - f4, f6 - f5
    beta0 = (
        ((2.0 * u0 - 7.0 * u1 + 11.0 * u2) / 6.0) ** 2
        + 13.0 / 12.0 * (u0 - 3.0 * u1 + 2.0 * u2) ** 2
        + 781.0 / 720.0 * (u0 - 2.0 * u1 + u2) ** 2
    )
    beta1 = (
        ((-u1 + 5.0 * u2 + 2.0 * u3) / 6.0) ** 2
        + 13.0 / 12.0 * (u3 - u2) ** 2
        + 781.0 / 720.0 * (u1 - 2.0 * u2 + u3) ** 2
    )
    beta2 = (
        ((2.0 * u2 + 5.0 * u3 - u4) / 6.0) ** 2
        + 13.0 / 12.0 * (u3 - u2) ** 2
        + 781.0 / 720.0 * (u2 - 2.0 * u3 + u4) ** 2
    )
    beta3 = (
        ((11.0 * u3 - 7.0 * u4 + 2.0 * u5) / 6.0) ** 2
        + 13.0 / 12.0 * (2.0 * u3 - 3.0 * u4 + u5) ** 2
        + 781.0 / 720.0 * (u3 - 2.0 * u4 + u5) ** 2
    )
    tau = jnp.abs(beta0 + 3.0 * beta1 - 3.0 * beta2 - beta3)
    alpha0 = (1.0 / 64.0) * (1.0 + _ratio(tau, beta0 + epsilon))
    alpha1 = (21.0 / 64.0) * (1.0 + _ratio(tau, beta1 + epsilon))
    alpha2 = (35.0 / 64.0) * (1.0 + _ratio(tau, beta2 + epsilon))
    alpha3 = (7.0 / 64.0) * (1.0 + _ratio(tau, beta3 + epsilon))
    if frozen:
        alpha0, alpha1, alpha2, alpha3 = (jax.lax.stop_gradient(a) for a in (alpha0, alpha1, alpha2, alpha3))
    return (alpha0 * p0 + alpha1 * p1 + alpha2 * p2 + alpha3 * p3) / (alpha0 + alpha1 + alpha2 + alpha3)


def _weno7z_point_pair(field, axis, frozen_weights):
    """
    Left- and right-biased seventh-order WENO-Z point interpolations of
    ``field`` to the half points i + 1/2 along ``axis``, with the epsilon
    ``CT_WENO_EPSILON_RELATIVE max(field^2)`` over the eight cells of both
    stencils.
    """
    left = [_value_at(field, o, axis) for o in range(-3, 4)]
    right = [_value_at(field, o, axis) for o in range(4, -3, -1)]
    scale2 = None
    for o in range(-3, 5):
        f2 = _value_at(field, o, axis) ** 2
        scale2 = f2 if scale2 is None else jnp.maximum(scale2, f2)
    epsilon = CT_WENO_EPSILON_RELATIVE * scale2 + EPSILON_FLOOR
    return (
        _weno7z_point_interpolation(*left, epsilon, frozen_weights),
        _weno7z_point_interpolation(*right, epsilon, frozen_weights),
    )


def _hll_transport_flux(face_velocity, field, cold_speed, axis, active, frozen_weights):
    """
    The two-state HLL transport flux G of ``field`` along ``axis`` at the half
    points i + 1/2 (module docstring), from the face velocity, the face field
    and the cold speed at the faces. Along a collapsed axis the field is
    uniform and the flux is ``face_velocity * field`` itself.
    """
    if not active:
        return face_velocity * field
    vL, vR = _weno7z_point_pair(face_velocity, axis, frozen_weights)
    bL, bR = _weno7z_point_pair(field, axis, frozen_weights)
    c = jnp.maximum(cold_speed, _value_at(cold_speed, 1, axis))
    a_plus = jnp.maximum(0.0, jnp.maximum(vL, vR) + c)
    a_minus = jnp.maximum(0.0, c - jnp.minimum(vL, vR))
    total = a_plus + a_minus
    gL, gR = vL * bL, vR * bR
    hll = _floored_ratio(a_plus * gL + a_minus * gR - a_plus * a_minus * (bR - bL), total)
    return jnp.where(total > 0.0, hll, 0.5 * (gL + gR))


def _cell_velocity_and_cold_speed(conserved_state, params, registered_variables):
    """The cell velocity components and the cold-plasma speed |B| / sqrt(rho)."""
    momentum_index = registered_variables.momentum_index
    magnetic_index = registered_variables.magnetic_index
    density = jnp.maximum(conserved_state[registered_variables.density_index], params.minimum_density)
    velocity = [conserved_state[index] / density for index in (momentum_index.x, momentum_index.y, momentum_index.z)]
    b2 = sum(conserved_state[index] ** 2 for index in (magnetic_index.x, magnetic_index.y, magnetic_index.z))
    return velocity, zero_safe_sqrt(b2 / density)


def _edge_emfs(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The edge EMFs of the flux-form WENO CT.

    Args:
        conserved_state: The conserved state (velocities and cold speed).
        bx_interface: The B_x field at the x-interfaces.
        by_interface: The B_y field at the y-interfaces.
        bz_interface: The B_z field at the z-interfaces.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The edge EMFs (E_x at the x-edges, E_y, E_z).
    """
    active = [axis < int(config.dimensionality) for axis in range(3)]
    frozen_weights = config.solver_config.weno_ad_frozen_weights
    velocity, cold_speed = _cell_velocity_and_cold_speed(conserved_state, params, registered_variables)
    faces = (bx_interface, by_interface, bz_interface)

    def transport_flux(component, along):
        """G[b_component transported along ``along``] at the edge."""
        face_velocity, face_speed = velocity[along], cold_speed
        if active[component]:
            face_velocity = _interp_center_to_face_6(face_velocity, component)
            face_speed = jnp.maximum(cold_speed, _value_at(cold_speed, 1, component))
        return _hll_transport_flux(face_velocity, faces[component], face_speed, along, active[along], frozen_weights)

    emf_z = transport_flux(XAXIS, YAXIS) - transport_flux(YAXIS, XAXIS)
    emf_x = transport_flux(YAXIS, ZAXIS) - transport_flux(ZAXIS, YAXIS)
    emf_y = transport_flux(ZAXIS, XAXIS) - transport_flux(XAXIS, ZAXIS)
    return emf_x, emf_y, emf_z


@partial(jax.jit, static_argnames=["registered_variables", "config"])
def flux_form_ct_rhs(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    dt_tilde,
    grid_spacing,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Increments of the interface fields over ``dt_tilde`` from the flux-form
    WENO CT: the staggered curl of the edge EMFs.

    Args:
        conserved_state: The conserved state (velocities and cold speed).
        bx_interface: The B_x field at the x-interfaces.
        by_interface: The B_y field at the y-interfaces.
        bz_interface: The B_z field at the z-interfaces.
        dt_tilde: The (stage) time step.
        grid_spacing: The cell size.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The increments of the interface fields (bx, by, bz).
    """
    increments, _ = _increments(
        conserved_state, (bx_interface, by_interface, bz_interface), dt_tilde, grid_spacing,
        params, config, registered_variables, with_energy=False,
    )
    return increments[0], increments[1], increments[2]


def flux_form_ct_rhs_with_energy(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    dt_tilde,
    grid_spacing,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    ``flux_form_ct_rhs`` together with the CT half of the Poynting-consistent
    energy flux (``_energy_flux.ct_energy_flux_rhs``) from the same EMFs.

    Args:
        conserved_state: The conserved state (velocities, cold speed and the
            cell-centred field of the energy flux).
        bx_interface: The B_x field at the x-interfaces.
        by_interface: The B_y field at the y-interfaces.
        bz_interface: The B_z field at the z-interfaces.
        dt_tilde: The (stage) time step.
        grid_spacing: The cell size.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The increments of the interface fields (bx, by, bz) and the energy
        increment.
    """
    increments, energy = _increments(
        conserved_state, (bx_interface, by_interface, bz_interface), dt_tilde, grid_spacing,
        params, config, registered_variables, with_energy=True,
    )
    return increments[0], increments[1], increments[2], energy


def _increments(
    conserved_state,
    interface_fields,
    dt_tilde,
    grid_spacing,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    with_energy: bool,
):
    """
    The increments of the interface fields and, with ``with_energy``, the CT
    half of the energy flux from the same EMFs. Where
    ``_flux_form_ct_pallas_supported`` holds, the kernels of
    ``_flux_form_ct_pallas.py`` and ``_energy_flux_pallas.py`` compute the
    values in one call (the EMFs exist only between them) and a derivative
    takes the array form below.

    Args:
        conserved_state: The conserved state.
        interface_fields: The interface fields (bx, by, bz).
        dt_tilde: The (stage) time step.
        grid_spacing: The cell size.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        with_energy: Whether the energy increment is computed.

    Returns:
        The increments of the interface fields and the energy increment
        (None without ``with_energy``).
    """
    magnetic_index = registered_variables.magnetic_index
    magnetic_channels = (magnetic_index.x, magnetic_index.y, magnetic_index.z)
    dimensionality = int(config.dimensionality)

    def array_form(state, fields, time_step, spacing, parameters):
        emfs = _edge_emfs(state, fields[0], fields[1], fields[2], parameters, config, registered_variables)
        increments = edge_emf_curl(emfs, time_step / spacing, dimensionality)
        if not with_energy:
            return increments, None
        cell_magnetic = tuple(state[channel] for channel in magnetic_channels)
        return increments, ct_energy_flux_rhs(cell_magnetic, emfs, time_step / spacing, dimensionality)

    if not _flux_form_ct_pallas_supported(conserved_state, config):
        return array_form(conserved_state, interface_fields, dt_tilde, grid_spacing, params)

    def pallas_branch(state, fields, time_step, spacing, parameters):
        emfs = _edge_emfs_pallas(state, fields, parameters, config, registered_variables)
        increments = _edge_emf_curl_pallas(emfs, time_step / spacing, config)
        if not with_energy:
            return increments
        return increments, _ct_energy_flux_rhs_pallas(state, emfs, time_step / spacing, config, magnetic_channels)

    def native_branch(state, fields, time_step, spacing, parameters):
        increments, energy = array_form(state, fields, time_step, spacing, parameters)
        if not with_energy:
            return jnp.stack(increments)
        return jnp.stack(increments), energy[None]

    dtype = interface_fields[0].dtype
    primals = (
        conserved_state,
        jnp.stack(interface_fields),
        jnp.asarray(dt_tilde, dtype=dtype),
        jnp.asarray(grid_spacing, dtype=dtype),
        params,
    )
    result = diffable_pallas_call_n(primals, pallas_branch=pallas_branch, native_branch=native_branch)
    if not with_energy:
        return result, None
    return result[0], result[1][0]


# The kernel modules are the compiled form of this module and of
# ``_energy_flux.py`` and are imported at the bottom, as everywhere in the
# package.
from astronomix._pallas_helpers import diffable_pallas_call_n  # noqa: E402
from astronomix._finite_difference._magnetic_update._energy_flux_pallas import (  # noqa: E402
    _ct_energy_flux_rhs_pallas,
)
from astronomix._finite_difference._magnetic_update._flux_form_ct_pallas import (  # noqa: E402
    _edge_emf_curl_pallas,
    _edge_emfs_pallas,
    _flux_form_ct_pallas_supported,
)
