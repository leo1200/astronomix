"""
Plummer sampler for N-body initial conditions in 3d

Generates positions from a Plummer sphere and samples velocities from the
local Plummer velocity distribution using rejection sampling. It then
rescales velocities to reach a target virial ratio Q (default 0.5) and
optionally enforces a confinement box [-a,a]^3 by a similarity rescale.

Returns:
    txv : jnp.array - [t, x, y, z, vx, vy, vz]
    masses : jnp.array - masses for each particle

G defaults to 1.0.
The Plummer scale `b` defaults to a/2 so that most mass lies well
within the box radius `a`. Change `b` if you want a more/less
concentrated distribution.

Example:
    txv, m = plummer_sampler(n=100, M1=0.8, M2=1.2, a=1.0, seed=42)
"""


import jax
import jax.numpy as jnp
from jax import jit
from typing import Tuple, Any
from functools import partial


def _pairwise_distances(positions: jnp.ndarray, softening: float = 0.0) -> jnp.ndarray:
    """
    positions: (n,3)
    """
    diff = positions[:, None, :] - positions[None, :, :] 
    r2 = jnp.sum(diff ** 2, axis=-1)
    r = jnp.sqrt(r2 + softening ** 2)
    r = r + jnp.eye(positions.shape[0]) * 1e30
    return r

def _potential_energy(positions: jnp.ndarray, masses: jnp.ndarray, softening: float = 0.0) -> float:
    """
    Compute gravitational potential energy U = - sum_{i<j} m_i m_j / r_ij  (G=1)
    """
    r = _pairwise_distances(positions, softening=softening)  # (n,n)
    mm = masses[:, None] * masses[None, :]  # (n,n)
    U_full = - jnp.sum(mm / r)  # includes both i<j and j<i
    U = 0.5 * U_full
    return U

def _pack_orbits(pos: jnp.ndarray, vel: jnp.ndarray, t0: float = 0.0) -> jnp.ndarray:
    n = pos.shape[0]
    tcol = jnp.full((n, 1), float(t0), dtype=pos.dtype)
    return jnp.hstack([tcol, pos, vel])

# ------------------------------
# Plummer sampler
# ------------------------------
# @partial(jit, static_argnums=("n", "mass_dist"))
def plummer_sampler(n: int,
                    key: Any,             
                    M1: float = 1.0,
                    M2: float = 1.0,
                    mass_dist: str = "uniform",   # "uniform" or "powerlaw"
                    powerlaw_alpha: float = 2.35, # used only if mass_dist == "powerlaw"
                    a: float = 1.0,   # Plummer scale radius
                    t0: float = 0.0,
                    softening: float = 0.0
                   ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns:
        orbits: [t0, x,y,z, vx,vy,vz]
        masses: shape (n,)
    Notes:
        - G = 1 units
        - velocities are scaled so that 2*T + U = 0 (global virial equilibrium)
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # --- sample masses ---
    if mass_dist == "uniform":
        masses = M1 + (M2 - M1) * jax.random.uniform(k1, (n,))
    elif mass_dist == "powerlaw":
        a_pl = powerlaw_alpha
        u = jax.random.uniform(k1, (n,))
        if a_pl == 1.0:
            # special-case: p(m) ~ 1/m
            masses = jnp.exp(u * (jnp.log(M2) - jnp.log(M1))) * M1
        else:
            C1 = M1 ** (1.0 - a_pl)
            C2 = M2 ** (1.0 - a_pl)
            masses = ((u * (C2 - C1) + C1)) ** (1.0 / (1.0 - a_pl))
    else:
        raise ValueError("mass_dist must be 'uniform' or 'powerlaw'")

    # --- sample Plummer radii via inverse CDF ---
    u_r = jax.random.uniform(k2, (n,))  # in (0,1)
    # inverse CDF for Plummer cumulative mass: r = a * ( u^{-2/3} - 1 )^{-1/2}
    r = a * (u_r ** (-2.0 / 3.0) - 1.0) ** (-0.5)

    # sample isotropic directions
    xyz = jax.random.normal(k3, (n, 3))
    norms = jnp.linalg.norm(xyz, axis=1, keepdims=True)
    dirs = xyz / norms
    pos = dirs * r[:, None]  # (n,3)

    # sample velocities isotropically (initial guess)
    vel_raw = jax.random.normal(k4, (n, 3))

    # compute current kinetic energy T0 with this raw velocity (scale=1)
    T0 = 0.5 * jnp.sum(masses[:, None] * (vel_raw ** 2))

    # compute potential energy
    U = _potential_energy(pos, masses, softening=softening)  # negative

    # desired kinetic energy to satisfy virial: 2T = -U -> T_desired = -U/2
    T_desired = -0.5 * U

    # compute global scale factor safely
    # If T0 is tiny or negative numeric, clamp denominator to small positive
    scale = jnp.where(T0 > 0.0, jnp.sqrt(jnp.maximum(T_desired / T0, 0.0)), 0.0)
    vel = vel_raw * scale

    # center-of-mass shift and zero net momentum
    Mtot = jnp.sum(masses)
    COM_pos = jnp.sum(pos * masses[:, None], axis=0) / Mtot
    COM_vel = jnp.sum(vel * masses[:, None], axis=0) / Mtot

    pos = pos - COM_pos[None, :]
    vel = vel - COM_vel[None, :]

    orbits = _pack_orbits(pos, vel, t0=t0)
    return orbits, masses

# ------------------------------
# Example usage
# key = jax.random.PRNGKey(1234)
# orbits, masses = plummer_sampler(n=128, key=key, M1=0.5, M2=2.0, mass_dist="uniform", a=1.0)
# print(orbits.shape, masses.shape)

"""
Simple virialized sphere sampler for N-body initial conditions (3D).

Generates `n` bodies with masses in [M1, M2] placed inside a sphere of
radius R<=a and assigns velocities so the system has a target virial
ratio Q = T/|U| (default Q=0.5). Positions are isotropic (uniform in
sphere) and velocity directions are isotropic; speeds are initially
drawn from a Maxwell-like distribution and then globally rescaled to
achieve the virial ratio.

Returns:
    txv: [t, x, y, z, vx, vy, z]
    masses: masses for each particle

Example:
    txv, m = virialized_sphere_sampler(100, 0.8, 1.2, a=1.0, R=0.9, seed=1)
"""

from typing import Tuple, Optional
import numpy as np
import jax.numpy as jnp


def virialized_sphere_sampler(
    n: int,
    M1: float,
    M2: float,
    a: float,
    R: Optional[float] = None,
    G: float = 1.0,
    Q: float = 0.5,
    seed: Optional[int] = None,
    t0: float = 0.0,
    masses: Optional[np.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a simple virialized sphere of `n` particles.

    Parameters
    ----------
    n : int
        Number of particles.
    M1, M2 : float
        Min/max mass for uniform sampling in [M1, M2], unless `masses` is
        supplied (length n).
    a : float
        Half-box size. Positions will be confined to [-a,a]^3 (via a
        similarity rescale if needed).
    R : Optional[float]
        Sphere radius to place particles in. If None, defaults to 0.9*a.
    G : float
        Gravitational constant (default 1.0).
    Q : float
        Target virial ratio T/|U| (default 0.5).
    seed : Optional[int]
        RNG seed for reproducibility.
    t0 : float
        Time value for each txv row (default 0.0).
    masses : Optional[np.ndarray]
        If provided, used as masses (must have length n).

    Returns
    -------
    txv : jnp.ndarray shape (n,7)
    masses_jnp : jnp.ndarray shape (n,)
    """

    rng = np.random.default_rng(seed)

    if masses is None:
        masses_np = rng.uniform(low=M1, high=M2, size=n).astype(float)
    else:
        masses_np = np.asarray(masses, dtype=float).copy()
        if masses_np.shape != (n,):
            raise ValueError("masses must have shape (n,)")

    M_tot = masses_np.sum()

    if R is None:
        R = 0.9 * a

    # --- sample positions uniformly in sphere of radius R
    u = rng.random(n)
    r = R * u ** (1.0 / 3.0)  # invert CDF for uniform sphere
    cos_theta = rng.uniform(-1.0, 1.0, size=n)
    sin_theta = np.sqrt(1.0 - cos_theta ** 2)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta

    # --- initial velocity magnitudes: Maxwell-like draw using sigma~sqrt(G*M_tot/R)
    v_scale = np.sqrt(np.abs(G * M_tot / max(R, 1e-12)))
    # draw three independent normals and take magnitude
    vx0 = rng.normal(scale=v_scale, size=n)
    vy0 = rng.normal(scale=v_scale, size=n)
    vz0 = rng.normal(scale=v_scale, size=n)

    # now we will rescale speeds to reach target virial ratio
    # compute potential energy U
    coords = np.vstack([x, y, z]).T  # (n,3)
    dx = coords[:, None, 0] - coords[None, :, 0]
    dy = coords[:, None, 1] - coords[None, :, 1]
    dz = coords[:, None, 2] - coords[None, :, 2]
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    np.fill_diagonal(dist, np.inf)

    mprod = masses_np[:, None] * masses_np[None, :]
    U = -0.5 * G * np.sum(mprod / dist)

    speeds2 = vx0 * vx0 + vy0 * vy0 + vz0 * vz0
    T = 0.5 * np.sum(masses_np * speeds2)

    if T <= 0.0:
        raise RuntimeError("Initial kinetic energy non-positive; try different RNG/params")

    desired_T = Q * np.abs(U)
    scale_v = np.sqrt(desired_T / T)

    vx = vx0 * scale_v
    vy = vy0 * scale_v
    vz = vz0 * scale_v

    # subtract center-of-mass velocity so CM is at rest
    v_cm = np.sum(masses_np[:, None] * np.vstack([vx, vy, vz]).T, axis=0) / M_tot
    vx -= v_cm[0]
    vy -= v_cm[1]
    vz -= v_cm[2]

    # ensure confinement in box [-a,a]^3; if coordinates exceed a, similarity rescale
    max_coord = np.max(np.abs(np.vstack([x, y, z])))
    if max_coord > a:
        lam = a / max_coord
        x *= lam
        y *= lam
        z *= lam
        # velocities must scale as 1/sqrt(lam) to preserve dynamical similarity
        vel_scale = 1.0 / np.sqrt(lam)
        vx *= vel_scale
        vy *= vel_scale
        vz *= vel_scale

    txv_np = np.column_stack([np.full(n, t0), x, y, z, vx, vy, vz])

    txv = jnp.array(txv_np)
    masses_jnp = jnp.array(masses_np)

    return txv, masses_jnp
