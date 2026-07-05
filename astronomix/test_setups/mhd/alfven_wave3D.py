"""
Circularly polarized Alfvén wave (3D).

Three-dimensional convergence test using a circularly polarized Alfvén
wave propagating obliquely with respect to the grid axes. Polarized Alfvén
waves are common in astrophysical plasmas (e.g. the solar corona;
Goldstein 1978), and the CP Alfvén wave is a smooth, finite-amplitude
analytic solution of the ideal MHD equations -- a particularly stringent
target for measuring the convergence rate of an MHD scheme.

In the unrotated frame the wave state is

    rho = 1,    v_x = 0,
    v_y = 0.1 sin(2 pi x_unrot),    v_z = 0.1 cos(2 pi x_unrot),
    B_x = 1,
    B_y = 0.1 sin(2 pi x_unrot),    B_z = 0.1 cos(2 pi x_unrot),
    p   = 0.1.

Because v_perp and delta B_perp are parallel, the wave is left-going along
the unrotated x axis at the Alfvén speed v_A = B_x / sqrt(rho) = 1. The
wave is then rotated by the Euler angles -arctan(2/sqrt(5)) about the
y axis followed by arctan(2) about the z axis, mapping the unrotated +x
direction to the unit vector (1, 2, 2)/3 in the simulation frame.

The simulation domain is the periodic box (3, 3/2, 3/2) with grid
(2N, N, N), so that the wave fits exactly one wavelength along each axis
and the grid spacing is uniform. The standard end time is t = 5, after
which the wave has executed five full periods and the analytic state has
returned to the initial condition.

Depending on the chosen scheme:
- For FINITE_DIFFERENCE (constrained-transport), the magnetic field is
  initialized from the analytic vector potential evaluated on the staggered
  edge grid, so that the discrete face-centered B is divergence-free to
  machine precision under the corresponding discrete curl.
- For FINITE_VOLUME (cell-centered), a central difference curl is applied
  to the cell-centered vector potential to guarantee a 0-divergence field
  under a standard central difference discrete divergence operator.
The uniform B background is added analytically in both cases.

## References

- https://arxiv.org/pdf/2304.04360

"""

# typing
from typing import NamedTuple

# jax
import jax.numpy as jnp

# astronomix constants
from astronomix import CARTESIAN
from astronomix.option_classes.simulation_config import (
    PERIODIC_BOUNDARY,
    STATE_TYPE,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import (
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix._spatial_operators._differencing import finite_difference_int6
from astronomix._spatial_operators._interpolate import interp_face_to_center

_XAXIS, _YAXIS, _ZAXIS = 0, 1, 2

# Rotation matrix mapping the unrotated frame to the simulation frame:
# rotate by -arctan(2/sqrt(5)) about y, then by arctan(2) about z. Its first
# column is the simulation-frame propagation direction (1, 2, 2)/3. This is
# part of the fixed test geometry and not user-configurable.
_SQRT5 = 5.0**0.5
_R = jnp.array([
    [1.0 / 3.0,   -2.0 / _SQRT5,        -2.0 / (3.0 * _SQRT5)],
    [2.0 / 3.0,    1.0 / _SQRT5,        -4.0 / (3.0 * _SQRT5)],
    [2.0 / 3.0,    0.0,                  _SQRT5 / 3.0        ],
])


class CPAlfvenWave3DSettings(NamedTuple):
    """Problem constants for the 3D circularly polarized Alfvén wave test."""

    #: Simulation-frame box size ``(L_x, L_y, L_z)``.
    box_size: tuple = (3.0, 1.5, 1.5)

    #: Final time at which the solution is evaluated.
    t_end: float = 5.0

    #: Adiabatic index of the gas.
    gamma: float = 5.0 / 3.0

    #: Background density ``rho_0``.
    rho_0: float = 1.0

    #: Background pressure ``p_0``.
    p_0: float = 0.1

    #: Uniform background field along the propagation direction.
    b_parallel: float = 1.0

    #: Transverse perturbation amplitude.
    amplitude: float = 0.1

    #: Wavenumber in the unrotated frame; wavelength = 2 pi / k_wave.
    k_wave: float = 2.0 * jnp.pi

    #: Alfvén speed ``= b_parallel / sqrt(rho_0)``; the wave is left-going.
    v_alfven: float = 1.0


def _x_unrot(X, Y, Z):
    """Unrotated x-coordinate at simulation-frame points: (X + 2Y + 2Z) / 3."""
    return _R[0, 0] * X + _R[1, 0] * Y + _R[2, 0] * Z


def _phase(X, Y, Z, t, settings: CPAlfvenWave3DSettings):
    """Wave phase k * (x_unrot + v_A * t); the wave is left-going in the
    unrotated frame, so the value at (x, t) equals the initial value at
    x + v_A t."""
    return settings.k_wave * (_x_unrot(X, Y, Z) + settings.v_alfven * t)


def _wave_primitive_state(X, Y, Z, t, settings: CPAlfvenWave3DSettings):
    """
    Cell-centered primitive state of the rotated CP Alfvén wave.

    Returns ``(rho, v_x, v_y, v_z, p, B_x, B_y, B_z)`` at simulation-frame
    coordinates (X, Y, Z) and time ``t``. The magnetic field includes the
    uniform background along the propagation direction.
    """
    phi = _phase(X, Y, Z, t, settings)
    s, c = jnp.sin(phi), jnp.cos(phi)
    # In the unrotated frame: v_x = 0, B_x = uniform; the transverse
    # perturbations of v and delta B coincide for a left-going wave.
    dy = settings.amplitude * s
    dz = settings.amplitude * c

    v_x = _R[0, 1] * dy + _R[0, 2] * dz
    v_y = _R[1, 1] * dy + _R[1, 2] * dz
    v_z = _R[2, 1] * dy + _R[2, 2] * dz

    B_x = _R[0, 1] * dy + _R[0, 2] * dz + _R[0, 0] * settings.b_parallel
    B_y = _R[1, 1] * dy + _R[1, 2] * dz + _R[1, 0] * settings.b_parallel
    B_z = _R[2, 1] * dy + _R[2, 2] * dz + _R[2, 0] * settings.b_parallel

    rho = jnp.full_like(X, settings.rho_0)
    p   = jnp.full_like(X, settings.p_0)
    return rho, v_x, v_y, v_z, p, B_x, B_y, B_z


def _vector_potential(X, Y, Z, t, settings: CPAlfvenWave3DSettings):
    """
    Perturbation vector potential ``A`` in the simulation frame at points
    (X, Y, Z) and time ``t``. Its curl reproduces the perturbation field
    only -- the uniform B background must be added to B separately, since
    no periodic vector potential exists for a uniform 3D field.
    """
    phi = _phase(X, Y, Z, t, settings)
    s, c = jnp.sin(phi), jnp.cos(phi)
    A_y_un = (settings.amplitude / settings.k_wave) * s
    A_z_un = (settings.amplitude / settings.k_wave) * c
    A_x = _R[0, 1] * A_y_un + _R[0, 2] * A_z_un
    A_y = _R[1, 1] * A_y_un + _R[1, 2] * A_z_un
    A_z = _R[2, 1] * A_y_un + _R[2, 2] * A_z_un
    return A_x, A_y, A_z


def _generate_state(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    helper_data: HelperData,
    t: float,
    settings: CPAlfvenWave3DSettings,
) -> STATE_TYPE:
    """
    Generate the discrete state for the CP Alfvén wave at time `t`,
    applying the same discrete curl operators used during initialization.
    """
    cell_centers = helper_data.geometric_centers
    Xc, Yc, Zc = cell_centers[..., 0], cell_centers[..., 1], cell_centers[..., 2]

    Nx, Ny, Nz = Xc.shape
    Lx, Ly, Lz = settings.box_size
    dx = Lx / Nx                # uniform iff num_cells = (2N, N, N)

    if config.solver_mode == FINITE_DIFFERENCE:
        # build the staggered edge grid for the vector potential
        x_l = jnp.linspace(dx, Lx, Nx, endpoint=True)
        y_l = jnp.linspace(dx, Ly, Ny, endpoint=True)
        z_l = jnp.linspace(dx, Lz, Nz, endpoint=True)
        x_c = jnp.linspace(dx/2, Lx + dx/2, Nx, endpoint=False)
        y_c = jnp.linspace(dx/2, Ly + dx/2, Ny, endpoint=False)
        z_c = jnp.linspace(dx/2, Lz + dx/2, Nz, endpoint=False)

        # A_x lives on yz-edges  (i,   j+1/2, k+1/2)  ->  (x_c, y_l, z_l)
        Xax, Yax, Zax = jnp.meshgrid(x_c, y_l, z_l, indexing="ij")
        # A_y lives on zx-edges  (i+1/2, j,   k+1/2)  ->  (x_l, y_c, z_l)
        Xay, Yay, Zay = jnp.meshgrid(x_l, y_c, z_l, indexing="ij")
        # A_z lives on xy-edges  (i+1/2, j+1/2, k  )  ->  (x_l, y_l, z_c)
        Xaz, Yaz, Zaz = jnp.meshgrid(x_l, y_l, z_c, indexing="ij")

        A_x_edge, _, _ = _vector_potential(Xax, Yax, Zax, t = t, settings = settings)
        _, A_y_edge, _ = _vector_potential(Xay, Yay, Zay, t = t, settings = settings)
        _, _, A_z_edge = _vector_potential(Xaz, Yaz, Zaz, t = t, settings = settings)

        # discrete curl -> face-centered perturbation field
        bxb_pert = (1.0/dx) * finite_difference_int6(A_z_edge, _YAXIS) \
                 - (1.0/dx) * finite_difference_int6(A_y_edge, _ZAXIS)
        byb_pert = (1.0/dx) * finite_difference_int6(A_x_edge, _ZAXIS) \
                 - (1.0/dx) * finite_difference_int6(A_z_edge, _XAXIS)
        bzb_pert = (1.0/dx) * finite_difference_int6(A_y_edge, _XAXIS) \
                 - (1.0/dx) * finite_difference_int6(A_x_edge, _YAXIS)

        # add uniform background to the face fields
        bxb = bxb_pert + _R[0, 0] * settings.b_parallel
        byb = byb_pert + _R[1, 0] * settings.b_parallel
        bzb = bzb_pert + _R[2, 0] * settings.b_parallel

        # cell-centered B by face-to-center interpolation
        B_x = interp_face_to_center(bxb, _XAXIS)
        B_y = interp_face_to_center(byb, _YAXIS)
        B_z = interp_face_to_center(bzb, _ZAXIS)

    elif config.solver_mode == FINITE_VOLUME:
        # evaluate vector potential directly at cell centers
        A_x_c, A_y_c, A_z_c = _vector_potential(Xc, Yc, Zc, t = t, settings = settings)

        # use central differencing identical to our FV volume divB logic
        def central_diff(f, axis):
            # boundaries are guaranteed periodic in this setup
            return (jnp.roll(f, -1, axis=axis) - jnp.roll(f, 1, axis=axis)) / (2 * dx)

        # discrete central diff curl -> cell-centered perturbation field
        bxb_pert = central_diff(A_z_c, _YAXIS) - central_diff(A_y_c, _ZAXIS)
        byb_pert = central_diff(A_x_c, _ZAXIS) - central_diff(A_z_c, _XAXIS)
        bzb_pert = central_diff(A_y_c, _XAXIS) - central_diff(A_x_c, _YAXIS)

        # add uniform background to the cell-centered fields
        B_x = bxb_pert + _R[0, 0] * settings.b_parallel
        B_y = byb_pert + _R[1, 0] * settings.b_parallel
        B_z = bzb_pert + _R[2, 0] * settings.b_parallel

        # fallback structure for compatibility; interfaces carry same as cell centers
        bxb, byb, bzb = B_x, B_y, B_z
    else:
        raise ValueError(f"Unsupported solver_mode: {config.solver_mode}")

    # cell-centered hydro state at time t
    rho, v_x, v_y, v_z, p, _, _, _ = _wave_primitive_state(Xc, Yc, Zc, t = t, settings = settings)

    return construct_primitive_state(
        config = config,
        registered_variables = registered_variables,
        density = rho,
        velocity_x = v_x,
        velocity_y = v_y,
        velocity_z = v_z,
        gas_pressure = p,
        magnetic_field_x = B_x,
        magnetic_field_y = B_y,
        magnetic_field_z = B_z,
        interface_magnetic_field_x = bxb,
        interface_magnetic_field_y = byb,
        interface_magnetic_field_z = bzb,
    )


def setup_cp_alfven_wave(
    config: SimulationConfig,
    params: SimulationParams,
    settings: CPAlfvenWave3DSettings = CPAlfvenWave3DSettings(),
) -> tuple[STATE_TYPE, SimulationConfig, SimulationParams]:
    """
    Set up the 3D circularly polarized Alfvén wave test.

    Enforces the geometry (3D Cartesian), box size, periodic
    boundaries on all faces, end time, gamma, and MHD mode
    required by the standard problem. The number of cells, Riemann solver,
    slope limiter and CFL number are left untouched. The user is
    responsible for choosing num_cells = (2N, N, N) so that the grid
    spacing is uniform across axes.

    The discrete B-field is properly formulated depending on the underlying
    solver topology to guarantee div(B)=0 natively.

    Args:
        config: Simulation configuration.
        params: Simulation parameters.
        settings: Problem constants (defaults to the standard CP Alfvén
            wave values).

    Returns:
        state: Initial primitive state of the simulation.
        config: Updated simulation configuration (CARTESIAN geometry,
            box_size from ``settings``, 3D, periodic boundaries, MHD).
        params: Updated simulation parameters (t_end and gamma from
            ``settings``).
    """
    config = config._replace(
        geometry = CARTESIAN,
        dimensionality = 3,
        box_size = StaticFloatVector(*settings.box_size),
        boundary_settings = BoundarySettings(
            x = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        mhd = True,
    )
    params = params._replace(t_end = settings.t_end, gamma = settings.gamma)

    registered_variables = get_registered_variables(config)
    helper_data = get_helper_data(config)

    state = _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=0.0,
        settings=settings,
    )

    config = finalize_config(config, state.shape)

    return state, config, params


def cp_alfven_wave_solution(
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    params: SimulationParams,
    helper_data: HelperData,
    settings: CPAlfvenWave3DSettings = CPAlfvenWave3DSettings(),
) -> STATE_TYPE:
    """
    Exact CP Alfvén wave state at t = ``params.t_end``, evaluated on the
    cell centers in ``helper_data``. Because the wave propagates rigidly
    along (1, 2, 2)/3 at the Alfvén speed, the analytic state at any time
    is simply a translated copy of the initial condition.

    Note: the reference state applies the exact same discrete curl
    initialization as ``setup_cp_alfven_wave``, so that the convergence test
    measures only the evolution errors of the numerical scheme, free from
    grid initialization errors.

    Args:
        config: Simulation configuration.
        registered_variables: Registered variables in the simulation.
        params: Simulation parameters.
        helper_data: Helper data for the simulation.
        settings: Problem constants (must match those used in
            :func:`setup_cp_alfven_wave`).

    Returns:
        state: Exact primitive state at t = params.t_end.
    """
    return _generate_state(
        config=config,
        registered_variables=registered_variables,
        helper_data=helper_data,
        t=params.t_end,
        settings=settings,
    )
