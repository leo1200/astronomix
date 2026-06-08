"""
KH reconstruction -- forward operator, broadband seed, observation operator.

2D viscous Kelvin-Helmholtz twin experiment: a fixed tanh shear base flow plus
an unknown broadband transverse-velocity perturbation field u0 (localized on the
shear layer, power spread across a band of streamwise wavenumbers kx). The
forward map integrates to time T with the differentiable astronomix hydro-FD
solver; the observation operator returns the (optionally noised / masked) final
velocity field.

Everything here is a pure library (no GPU selection); entry scripts call
autocvd before importing this module.

Conventions: box [0,1]^2, x = flow direction (periodic), y = transverse (open).
Base shear v_x = (dV/2) tanh((y-yc)/delta); uniform rho, P (low Mach).
"""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    DYNAMIC_VISCOSITY,
    FINITE_DIFFERENCE,
    FORWARDS,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
    finalize_config,
)


class KHParams(NamedTuple):
    n: int = 256              # cells per dim
    box: float = 1.0
    yc: float = 0.5           # shear-layer center
    delta: float = 0.02       # shear-layer thickness
    dV: float = 1.0           # velocity jump across the layer
    rho0: float = 1.0
    mach: float = 0.3         # dV / c_s  (low -> ~incompressible, no acoustics)
    gamma: float = 5.0 / 3.0
    reynolds: float = 2000.0  # Re = rho dV delta / mu
    c_cfl: float = 1.0
    num_checkpoints: int = 800   # >= steps/segment -> store all, ~no recompute
    # broadband seed band (streamwise integer mode numbers kx) and envelope
    k_min: int = 2
    k_max: int = 32
    seed_amp: float = 1e-2    # rms of the seed transverse velocity (units of dV)
    env_width: float = 0.04   # Gaussian envelope half-width in y around yc

    @property
    def pressure(self):
        # c_s = dV / mach ; P = c_s^2 rho / gamma
        c_s = self.dV / self.mach
        return c_s**2 * self.rho0 / self.gamma

    @property
    def viscosity(self):  # dynamic viscosity mu = rho dV delta / Re
        return self.rho0 * self.dV * self.delta / self.reynolds

    @property
    def growth_time(self):
        # KH growth time ~ delta / dV (order of magnitude); used to scale horizons
        return self.delta / self.dV


def make_config_params(khp: KHParams, t_end: float, snapshots=0, backward=False):
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=2,
        num_cells=khp.n,
        box_size=khp.box,
        progress_bar=False,
        diffusion=True,
        viscosity_type=DYNAMIC_VISCOSITY,
        differentiation_mode=BACKWARDS if backward else FORWARDS,
        num_checkpoints=khp.num_checkpoints,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        return_snapshots=snapshots > 0,
        num_snapshots=max(snapshots, 1),
        snapshot_settings=SnapshotSettings(return_states=True) if snapshots > 0 else SnapshotSettings(),
    )
    params = SimulationParams(
        viscosity=khp.viscosity,
        t_end=t_end,
        C_cfl=khp.c_cfl,
        gamma=khp.gamma,
    )
    return config, params


def coords(khp: KHParams, config):
    hd = get_helper_data(config)
    centers = hd.geometric_centers      # (Nx, Ny, 2)
    return centers[..., 0], centers[..., 1]


def base_fields(khp: KHParams, X, Y):
    """Fixed tanh shear base flow (the known part)."""
    vx = (khp.dV / 2.0) * jnp.tanh((Y - khp.yc) / khp.delta)
    vy = jnp.zeros_like(vx)
    rho = khp.rho0 * jnp.ones_like(vx)
    P = khp.pressure * jnp.ones_like(vx)
    return rho, vx, vy, P


def random_broadband_seed(key, khp: KHParams, X, Y):
    """Broadband transverse-velocity perturbation localized on the shear layer.

    v_y'(x,y) = env(y) * sum_{k=k_min..k_max} a_k cos(2 pi k x + phi_k),
    flat power across the band, rms normalised to seed_amp * dV. Returned as a
    (2, Nx, Ny) field (svx, svy) with svx = 0 (classic KH transverse seed).
    """
    ks = jnp.arange(khp.k_min, khp.k_max + 1)
    ka, kp = jax.random.split(key)
    amps = jax.random.normal(ka, ks.shape)
    phis = jax.random.uniform(kp, ks.shape, minval=0.0, maxval=2 * jnp.pi)
    # sum of streamwise modes (broadcast over the grid)
    phases = 2 * jnp.pi * ks[None, None, :] * X[..., None] + phis[None, None, :]
    svy = jnp.sum(amps[None, None, :] * jnp.cos(phases), axis=-1)
    env = jnp.exp(-((Y - khp.yc) / khp.env_width) ** 2)
    svy = svy * env
    svy = khp.seed_amp * khp.dV * svy / jnp.sqrt(jnp.mean(svy**2) + 1e-30)
    return jnp.stack([jnp.zeros_like(svy), svy])


def build_state(seed, khp: KHParams, config, rv, X, Y):
    """Primitive state = base shear + seed velocity perturbation."""
    rho, vx, vy, P = base_fields(khp, X, Y)
    return construct_primitive_state(
        config=config, registered_variables=rv,
        density=rho, velocity_x=vx + seed[0], velocity_y=vy + seed[1],
        gas_pressure=P,
    )


def velocity_of(state, rv):
    return jnp.stack([state[rv.velocity_index.x], state[rv.velocity_index.y]])


def forward(seed, khp, config, params, rv, X, Y):
    """Integrate base+seed to params.t_end; return final primitive state."""
    state = build_state(seed, khp, config, rv, X, Y)
    return time_integration(state, config, params, rv)


def observe(state, rv, noise_key=None, sigma=0.0, mask=None):
    """Observation operator H: full final velocity, optional noise / mask."""
    v = velocity_of(state, rv)
    if sigma > 0 and noise_key is not None:
        v = v + sigma * jax.random.normal(noise_key, v.shape)
    if mask is not None:
        v = v * mask
    return v
