"""
N-body point-mass gravity solver.

Advances a set of point masses ([t, x, y, z, vx, vy, vz] per body) with an RK4
integrator under their mutual gravity, and deposits their masses onto the gas
grid (nearest-grid-point / cloud-in-cell / triangular-shaped-cloud) so the
self-gravity Poisson solve can feed the combined gas-plus-N-body potential back
onto the gas. The coupling is one-directional: the N-body integration
(``rk4_step_nbody``) only ever depends on the other bodies' masses, never on
the gas, so the N-body masses gravitationally act on the gas but the gas never
acts back on them.

Also provides a couple of two-body Kepler-orbit helpers (``binary_starting_orbits``
and friends) for constructing N-body initial conditions for an (eccentric,
inclined) binary sub-system, and a Plummer-style ``integrate_nbody`` driver for
generating/inspecting N-body trajectories standalone, outside the main solver.
"""

# general
from functools import partial
import time

# typing
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped
from typing import Tuple, Union

# jax
import jax
import jax.numpy as jnp
from jax import jit, lax

# astronomix constants
from astronomix._modules._nbody._nbody_options import NGP, CIC, TSC

# astronomix containers
from astronomix.option_classes.simulation_config import SimulationConfig


def acceleration(positions: jnp.ndarray, masses: jnp.ndarray, eps: float = 1e-12):
    """
    Compute accelerations for n bodies due to mutual gravity.
    positions: shape (n,3)
    masses: shape (n,)
    returns: acc shape (n,3) where acc[i] = sum_{j != i} - masses[j] * (r_i - r_j) / |r_i - r_j|^3
    (G is assumed 1, consistent with the code's unit treatment)
    """
    diff = positions[:, None, :] - positions[None, :, :]  # (n, n, 3)
    r2 = jnp.sum(diff ** 2, axis=-1)  # (n, n)
    inv_r3 = jnp.where(r2 > 0, 1.0 / (r2 * jnp.sqrt(r2) + eps), 0.0)  # (n, n)
    mass_factors = masses[None, :]  # (1, n)
    # acceleration contribution from j on i: - mass_j * diff_ij * inv_r3_ij
    contrib = - (mass_factors[..., None] * diff) * inv_r3[..., None]  # (n, n, 3)
    contrib = contrib * (1.0 - jnp.eye(positions.shape[0])[:, :, None])
    acc = jnp.sum(contrib, axis=1)  # (n, 3)
    return acc

@jit
def rk4_step_nbody(state: jnp.ndarray, h: float, masses: jnp.ndarray):
    """
    One RK4 step for n-body system.
    state: array shape (n,7) where each row is [t, x, y, z, vx, vy, vz]
    masses: array shape (n,)
    returns: new_state shape (n,7)
    """
    hh = 0.5 * h
    h6 = h / 6.0

    def deriv(s):
        # s shape (n,7)
        n = s.shape[0]
        dt_col = jnp.ones((n, 1), dtype=s.dtype)  # time derivative
        positions = s[:, 1:4]
        velocities = s[:, 4:7]
        acc = acceleration(positions, masses)
        return jnp.concatenate([dt_col, velocities, acc], axis=1)

    k1 = deriv(state)
    k2 = deriv(state + hh * k1)
    k3 = deriv(state + hh * k2)
    k4 = deriv(state + h * k3)

    return state + h6 * (k1 + 2.0 * (k2 + k3) + k4)


@partial(jax.jit, static_argnames=['config'])
def _advance_nbody_state(
    nbody_state: jnp.ndarray,
    masses: jnp.ndarray,
    dt: Float[Array, ""],
    config: SimulationConfig,
) -> jnp.ndarray:
    """
    Advance the N-body phase-space state by one RK4 step of size ``dt``.

    Advances jointly with (i.e. using the same time step as) the hydro update;
    called once per outer time-integration step (see
    astronomix.time_stepping.time_integration). ``nbody_state`` holds
    [t, x, y, z, vx, vy, vz] per body, flattened to shape (n_bodies * 7,).

    When ``config.nbody_config.central_object_only``, the masses are treated
    as a fixed, non-evolving source and the state is returned unchanged.
    """
    if config.nbody_config.central_object_only:
        return nbody_state

    n_bodies = masses.size
    state = nbody_state.reshape((n_bodies, 7))
    new_state = rk4_step_nbody(state, dt, masses)
    return new_state.reshape(nbody_state.shape)


### Nearest Grid Point (NGP) particle deposition (might not be good with FFT poisson solver)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=['grid_shape', 'grid_spacing'])
def _deposit_particles_ngp(
    particle_positions: Float[Array, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[Array, "nx ny nz"]:
    """
    Deposit n point-masses to nearest grid cell (NGP).
    Positions in same units as grid, origin at (0,0,0).
    """
    grid_extent = jnp.array(grid_shape) * grid_spacing
    grid_min    = -0.5 * grid_extent
    particle_densities = particle_masses / (grid_spacing ** 3)
    # map world->grid indices by subtracting the minimum corner:
    idx = ((particle_positions - grid_min) // grid_spacing).astype(int)
    idx = jnp.clip(idx, 0, jnp.array(grid_shape) - 1)
    # Flatten grid and add masses
    flat_idx = idx[:,0] * (grid_shape[1]*grid_shape[2]) + idx[:,1] * grid_shape[2] + idx[:,2]
    rho_flat = jnp.zeros(grid_shape[0]*grid_shape[1]*grid_shape[2])
    rho_flat = rho_flat.at[flat_idx].add(particle_densities)
    return rho_flat.reshape(grid_shape)

### Cloud-In-Cell (CIC) particle deposition (might be better?)
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_cic(
    particle_positions: Float[jnp.ndarray, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    Cloud-In-Cell (CIC) deposit (3D).
    """
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent
    particle_densities = particle_masses / (grid_spacing ** 3)
    # relative index in grid coordinates
    rel = (particle_positions - grid_min) / grid_spacing
    i0 = jnp.floor(rel).astype(jnp.int32)
    f  = rel - i0.astype(rel.dtype)
    # 8 neighbor offsets for CIC
    offsets = jnp.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1],
    ], dtype=jnp.int32)

    neigh_idx = i0[:, None, :] + offsets[None, :, :]
    # clip indices to grid boundaries (non-periodic)
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # weights: for each dim weight is (1-f) if offset==0 else f; multiply over dims -> (N,8)
    f_b = f[:, None, :]
    # boolean mask of offsets==0 broadcasted -> choose (1-f) or f
    w_comp = jnp.where(offsets[None, :, :] == 0, 1.0 - f_b, f_b)
    weights = jnp.prod(w_comp, axis=-1)
    # linearize 3D indices to flat indices (row-major: x*(ny*nz) + y*nz + z)
    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])
    flat_idx_flat = flat_idx.reshape(-1)
    values_flat = (particle_densities[:, None] * weights).reshape(-1)
    n_cells = nx * ny * nz
    rho_flat = jnp.zeros(n_cells, dtype=particle_densities.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))

# Triangular-Shaped-Cloud (TSC) particle deposition (quadratic B-spline)
# This is a more accurate method than CIC, but more expensive. It spreads each particle’s mass to the
# 3 nearest grid points along each axis (3×3×3 = 27 cells in 3D) with quadratic weights.
@jaxtyped(typechecker=typechecker)
@partial(jax.jit, static_argnames=('grid_shape', 'grid_spacing'))
def _deposit_particles_tsc(
    particle_positions: Float[jnp.ndarray, "n 3"],
    particle_masses:    Union[Float[Array, ""], Float[Array, "n"]],
    grid_shape:         Tuple[int, int, int],
    grid_spacing:       float
) -> Float[jnp.ndarray, "nx ny nz"]:
    """
    TSC (Triangular-Shaped-Cloud) deposit in 3D
    """
    nx, ny, nz = grid_shape
    grid_extent = jnp.array([nx, ny, nz]) * grid_spacing
    grid_min = -0.5 * grid_extent
    particle_densities = particle_masses / (grid_spacing ** 3)
    rel = (particle_positions - grid_min) / grid_spacing
    # floor(rel) gives a central index; neighbors are floor(rel)-1, floor(rel), floor(rel)+1
    i_center = jnp.floor(rel).astype(jnp.int32)
    # Offsets for TSC: 27 neighbors
    offsets = jnp.array([[i, j, k]
                         for i in (-1, 0, 1)
                         for j in (-1, 0, 1)
                         for k in (-1, 0, 1)], dtype=jnp.int32)
    neigh_idx = i_center[:, None, :] + offsets[None, :, :]
    max_idx = jnp.array([nx - 1, ny - 1, nz - 1], dtype=jnp.int32)
    neigh_idx = jnp.clip(neigh_idx, 0, max_idx)
    # 1D distances
    r = rel[:, None, :] - neigh_idx.astype(rel.dtype)
    s = jnp.abs(r)
    # 1D TSC kernel evaluated vectorized:
    def W1D_from_s(s_component):
        w = jnp.where(s_component <= 0.5,
                      0.75 - s_component**2,
                      jnp.where(s_component <= 1.5,
                                0.5 * (1.5 - s_component)**2,
                                0.0))
        return w

    wx = W1D_from_s(s[..., 0])
    wy = W1D_from_s(s[..., 1])
    wz = W1D_from_s(s[..., 2])
    weights = wx * wy * wz

    flat_idx = (neigh_idx[..., 0] * (ny * nz)
                + neigh_idx[..., 1] * nz
                + neigh_idx[..., 2])

    flat_idx_flat = flat_idx.reshape(-1)
    values_flat = (particle_densities[:, None] * weights).reshape(-1)
    n_cells = nx * ny * nz
    rho_flat = jnp.zeros(n_cells, dtype=particle_densities.dtype)
    rho_flat = rho_flat.at[flat_idx_flat].add(values_flat)

    return rho_flat.reshape((nx, ny, nz))


@partial(jax.jit, static_argnames=['grid_shape', 'grid_spacing', 'config'])
def _deposit_nbody_density(
    nbody_state: jnp.ndarray,
    masses: jnp.ndarray,
    grid_shape: Tuple[int, int, int],
    grid_spacing: float,
    config: SimulationConfig,
) -> Float[Array, "nx ny nz"]:
    """
    Deposit the N-body point masses onto the gas grid.

    Used to couple the N-body masses' gravity into the gas self-gravity
    potential (``config.gravity_config.self_gravity``): the returned density
    is added to the gas density before the self-gravity Poisson solve (see
    ``astronomix._modules._gravity._gravity._compute_total_potential``), so
    the gas feels the combined gas-plus-N-body potential. The N-body dynamics
    (``rk4_step_nbody``) only ever depend on the other bodies' masses, so this
    coupling is one-directional: the gas feels the N-body masses, not the
    other way around.

    When ``config.nbody_config.central_object_only``, all mass is deposited
    at the grid center instead of the (unintegrated) N-body positions.
    """
    if config.nbody_config.central_object_only:
        particle_positions = jnp.zeros((1, 3), dtype=nbody_state.dtype)
        particle_masses = masses[:1]
    else:
        n_bodies = masses.size
        particle_positions = nbody_state.reshape((n_bodies, 7))[:, 1:4]
        particle_masses = masses

    if config.nbody_config.deposit_particles == NGP:
        return _deposit_particles_ngp(particle_positions, particle_masses, grid_shape, grid_spacing)
    elif config.nbody_config.deposit_particles == CIC:
        return _deposit_particles_cic(particle_positions, particle_masses, grid_shape, grid_spacing)
    elif config.nbody_config.deposit_particles == TSC:
        return _deposit_particles_tsc(particle_positions, particle_masses, grid_shape, grid_spacing)
    else:
        raise ValueError(f"Unknown deposit_particles method: {config.nbody_config.deposit_particles}")


@jit
def binary_starting_orbits(sep: float,
                           e: float,
                           inc_deg: float,
                           m1: float,
                           m2: float,
                           true_anom_deg: float = 0.0,
                           G: float = 1.0):
    """
    Construct initial state vectors [t, x, y, z, vx, vy, vz] for a binary system
    - sep: semi-major axis a
    - e: eccentricity (0 <= e < 1 for elliptic)
    - inc_deg: inclination in degrees (rotation about x-axis). Position stays on x-axis.
    - m1, m2: masses
    - true_anom_deg: starting true anomaly (degrees); default = 0 -> periapsis (relative position on +x)
    - G: gravitational constant (default 1)
    Returns: jnp.array shape (2,7) where each row is [t, x, y, z, vx, vy, vz]
    Notes:
      - Positions are given in the center-of-mass frame (COM at origin).
      - By construction initial positions have only an x-component (y=z=0).
    """

    # convert angles
    i = jnp.deg2rad(inc_deg)
    f = jnp.deg2rad(true_anom_deg)

    # semi-major axis
    a = sep

    mu = G * (m1 + m2)

    # radial distance at true anomaly f
    r = a * (1.0 - e**2) / (1.0 + e * jnp.cos(f))

    # specific angular momentum
    h = jnp.sqrt(mu * a * (1.0 - e**2))

    # velocity components in perifocal (r, theta, z=0)
    v_r = (mu / h) * e * jnp.sin(f)
    v_theta = (mu / h) * (1.0 + e * jnp.cos(f))

    # position and velocity in perifocal coords (relative vector)
    r_pf = jnp.array([r, 0.0, 0.0], dtype=jnp.float64)
    v_pf = jnp.array([v_r, v_theta, 0.0], dtype=jnp.float64)

    # rotation about x-axis by inclination i (perifocal -> inertial, with Omega=omega=0)
    ci = jnp.cos(i)
    si = jnp.sin(i)
    R_x = jnp.array([[1.0, 0.0, 0.0],
                     [0.0, ci, -si],
                     [0.0, si,  ci]], dtype=jnp.float64)

    r_eci = R_x @ r_pf
    v_eci = R_x @ v_pf

    # split into two-body COM coordinates (COM at origin)
    r1 = - (m2 / (m1 + m2)) * r_eci
    r2 =   (m1 / (m1 + m2)) * r_eci
    v1 = - (m2 / (m1 + m2)) * v_eci
    v2 =   (m1 / (m1 + m2)) * v_eci

    orbit1 = jnp.concatenate([jnp.array([0.0]), r1, v1])  # [t, x, y, z, vx, vy, vz]
    orbit2 = jnp.concatenate([jnp.array([0.0]), r2, v2])

    return jnp.stack([orbit1, orbit2])



@jit
def _eccentric_from_true_anomaly(f, e):
    """
    E = 2*arctan( sqrt((1-e)/(1+e)) * tan(f/2) )
    Handles f near ±pi robustly via atan2 formulation.
    """
    # Use half-angle formulation with atan2 to reduce issues with quadrants.
    s = jnp.sqrt((1.0 - e) / (1.0 + e))
    t_half = jnp.tan(0.5 * f)
    E = 2.0 * jnp.arctan(s * t_half)
    # Ensure E is continuous (map to principal branch)
    return E

@jit
def _solve_kepler_newton(M, e, n_iter=30):
    """
    Solve M = E - e*sin(E) for E using Newton iterations.
    M, e are scalars. Uses a good analytic initial guess.
    """
    # initial guess using Fourier series / first terms
    E0 = M + e * jnp.sin(M) + 0.5 * (e**2) * jnp.sin(2.0 * M)

    def body_fun(i, E):
        f = E - e * jnp.sin(E) - M
        fp = 1.0 - e * jnp.cos(E)
        E_new = E - f / fp
        return E_new

    E_final = lax.fori_loop(0, n_iter, body_fun, E0)
    return E_final

@jit
def binary_starting_orbits_at_phase(sep: float,
                           e: float,
                           inc_deg: float,
                           m1: float,
                           m2: float,
                           phi: float = 0.0,
                           true_anom_deg: float = 0.0,
                           G: float = 1.0):
    """
    Construct state vectors [t, x, y, z, vx, vy, vz] for a binary at orbital phase `phi`.
    - sep: semi-major axis a
    - e: eccentricity (0 <= e < 1 for elliptic)
    - inc_deg: inclination in degrees (rotation about x-axis). Positions are rotated about x.
    - m1, m2: masses
    - phi: orbital phase in [0,1). phi=0 corresponds to true_anom_deg at time zero.
    - true_anom_deg: true anomaly at phase phi=0 (degrees). Default 0 -> periapsis on +x.
    - G: gravitational constant (default 1)
    Returns: jnp.array shape (2,7) where each row is [t, x, y, z, vx, vy, vz]
    Notes:
      - Positions/velocities are returned in the center-of-mass frame (COM at origin).
      - By construction the initial reference periapsis (phi=0, true_anom_deg=0) lies on +x.
    """

    # convert angles and scalars
    i = jnp.deg2rad(inc_deg)
    f0 = jnp.deg2rad(true_anom_deg)
    a = sep
    mu = G * (m1 + m2)

    # 1) compute eccentric anomaly at phi=0 from provided true anomaly f0
    # handle circular case e==0 specially to avoid divisions by zero
    E0 = jnp.where(e == 0.0, f0, _eccentric_from_true_anomaly(f0, e))

    # 2) compute mean anomaly at phi=0
    M0 = E0 - e * jnp.sin(E0)

    # 3) advance mean anomaly by 2*pi*phi (wrap phi into [0,1))
    phi_wrap = phi - jnp.floor(phi)
    M_target = M0 + 2.0 * jnp.pi * phi_wrap

    # 4) solve Kepler's equation for target eccentric anomaly E_target
    E = _solve_kepler_newton(M_target, e, n_iter=40)

    # 5) compute position in perifocal coordinates from E
    cosE = jnp.cos(E)
    sinE = jnp.sin(E)
    sqrt_1_e2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - e**2))

    # Perifocal position (relative) (x_pf, y_pf, 0)
    x_pf = a * (cosE - e)
    y_pf = a * sqrt_1_e2 * sinE
    r_pf = jnp.array([x_pf, y_pf, 0.0])

    # Perifocal velocity (relative)
    # factor = sqrt(mu / a) / (1 - e*cosE)
    factor = jnp.sqrt(mu / a) / (1.0 - e * cosE)
    vx_pf = - factor * sinE
    vy_pf =   factor * sqrt_1_e2 * cosE
    v_pf = jnp.array([vx_pf, vy_pf, 0.0])

    # 6) rotate by inclination about x-axis (perifocal -> inertial, with Omega=omega=0)
    ci = jnp.cos(i)
    si = jnp.sin(i)
    R_x = jnp.array([[1.0, 0.0, 0.0],
                     [0.0, ci, -si],
                     [0.0, si,  ci]])

    r_eci = R_x @ r_pf
    v_eci = R_x @ v_pf

    # 7) split into two-body COM coordinates (COM at origin)
    r1 = - (m2 / (m1 + m2)) * r_eci
    r2 =   (m1 / (m1 + m2)) * r_eci
    v1 = - (m2 / (m1 + m2)) * v_eci
    v2 =   (m1 / (m1 + m2)) * v_eci

    orbit1 = jnp.concatenate([jnp.array([0.0]), r1, v1])  # [t, x, y, z, vx, vy, vz]
    orbit2 = jnp.concatenate([jnp.array([0.0]), r2, v2])

    return jnp.concatenate([orbit1, orbit2])

@jit
def kepler_period(a: float, m1: float, m2: float, G: float = 1.0):
    return 2.0 * jnp.pi * jnp.sqrt(a**3 / (G * (m1 + m2)))

@jit
def positions_at_phase_kepler(traj: jnp.ndarray,
                              phi: float,
                              a: float,
                              m1: float,
                              m2: float,
                              G: float = 1.0):
    """
    Return positions (x,y,z) of both stars at orbital phase phi (0..1)
    using analytic Kepler period.
    """
    times = traj[:, 0, 0]
    period = kepler_period(a, m1, m2, G)

    phi = phi - jnp.floor(phi)  # wrap
    target_time = times[0] + phi * period

    j = jnp.searchsorted(times, target_time, side="right") - 1
    j = jnp.clip(j, 0, traj.shape[0] - 2)

    t0 = times[j]
    t1 = times[j + 1]
    alpha = (target_time - t0) / (t1 - t0)

    # positions
    pos0 = traj[j, :, 1:4]
    pos1 = traj[j + 1, :, 1:4]

    # velocities
    vel0 = traj[j, :, 4:7]
    vel1 = traj[j + 1, :, 4:7]

    pos_phi = (1.0 - alpha) * pos0 + alpha * pos1
    vel_phi = (1.0 - alpha) * vel0 + alpha * vel1

    return pos_phi, vel_phi


def integrate_nbody(orbits: jnp.ndarray, masses: jnp.ndarray, h: float, T: float, eps: float = 1e-12):
    """
    orbits: jnp.array shape (n,7) initial rows [t, x, y, z, vx, vy, vz]
    masses: jnp.array shape (n,)
    h: timestep
    T: total integration time
    returns: traj_com shape (num_steps, n, 7) positions/velocities in COM frame
    """
    state0 = orbits
    num_steps = int(jnp.ceil(T / h))
    n = state0.shape[0]
    totalM = jnp.sum(masses)

    @jit
    def run_with_fori_loop(state0):
        def body_fn(i, carry):
            state, traj = carry
            new_state = rk4_step_nbody(state, h, masses)
            traj = traj.at[i].set(new_state)
            return new_state, traj

        traj = jnp.zeros((num_steps, n, 7), dtype=state0.dtype)
        _, traj = lax.fori_loop(0, num_steps, body_fn, (state0, traj))
        return traj

    t0 = time.perf_counter()
    traj = run_with_fori_loop(state0)
    t1 = time.perf_counter()
    print(f"n-body RK4 took {t1 - t0:.4f} seconds")

    # Transform trajectories into center-of-mass frame (positions only; times/vels adjusted)
    positions = traj[:, :, 1:4]  # (num_steps, n, 3)
    weighted = positions * masses[None, :, None]
    COM = jnp.sum(weighted, axis=1) / totalM
    # Subtract COM from each body's positions for all timesteps
    positions_com = positions - COM[:, None, :]

    traj_com = traj.at[:, :, 1:4].set(positions_com)

    return traj_com
