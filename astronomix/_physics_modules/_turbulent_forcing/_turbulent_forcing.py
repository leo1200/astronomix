"""
The turbulent forcing module also draws from 
https://arxiv.org/pdf/2304.04360
"""

import itertools

import jax
import jax.numpy as jnp
from functools import partial

from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingParams
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.variable_registry.registered_variables import RegisteredVariables

@partial(jax.jit, static_argnames=["config"])
def _create_forcing_field(
    key,
    config: SimulationConfig,
):
    
    xsize = config.box_size.x
    ysize = config.box_size.y
    zsize = config.box_size.z

    nx = config.num_cells.x + 2 * config.num_ghost_cells
    ny = config.num_cells.y + 2 * config.num_ghost_cells
    nz = config.num_cells.z + 2 * config.num_ghost_cells

    # wavenumbers using fftfreq
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=xsize/nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=ysize/ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=zsize/nz)
    
    # broadcasting instead of meshgrid
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)
    
    k_squared = kx_3d**2 + ky_3d**2 + kz_3d**2
    kk = jnp.sqrt(k_squared)
    
    # power spectrum of the forcing
    kpk = 4.0 * jnp.pi / config.box_size.x # correct?
    Pk = kk**6 * jnp.exp(-8.0 * kk / kpk)
    
    key, sk1, sk2 = jax.random.split(key, 3)

    raw_noise = jax.random.normal(sk1, shape=(3, nx, ny, nz)) + \
                1j * jax.random.normal(sk2, shape=(3, nx, ny, nz))

    cwx = jnp.sqrt(Pk) * raw_noise[0]
    cwy = jnp.sqrt(Pk) * raw_noise[1]
    cwz = jnp.sqrt(Pk) * raw_noise[2]
    
    # DC mode to zero
    cwx = cwx.at[0, 0, 0].set(0.0 + 0.0j)
    cwy = cwy.at[0, 0, 0].set(0.0 + 0.0j)
    cwz = cwz.at[0, 0, 0].set(0.0 + 0.0j)
    
    # project out compressible component
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    div_k = (kx_3d * cwx + ky_3d * cwy + kz_3d * cwz) / k_squared_safe
    div_k = div_k.at[0, 0, 0].set(0.0 + 0.0j)
    cwx = cwx - kx_3d * div_k
    cwy = cwy - ky_3d * div_k
    cwz = cwz - kz_3d * div_k

    # get real space fields
    wx_real = jnp.real(jnp.fft.ifftn(cwx))
    wy_real = jnp.real(jnp.fft.ifftn(cwy))
    wz_real = jnp.real(jnp.fft.ifftn(cwz))

    return key, wx_real, wy_real, wz_real


# ---------------------------------------------------------------------------
# Ornstein-Uhlenbeck (temporally correlated) forcing
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=["config"])
def _create_solenoidal_field(key, config, k_f):
    """A fresh solenoidal (divergence-free) random velocity field peaked at
    wavenumber ``k_f`` and normalised to unit rms (mean(|w|^2) = 1)."""
    nx = config.num_cells.x + 2 * config.num_ghost_cells
    ny = config.num_cells.y + 2 * config.num_ghost_cells
    nz = config.num_cells.z + 2 * config.num_ghost_cells

    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=config.box_size.x / nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=config.box_size.y / ny)
    kz = 2.0 * jnp.pi * jnp.fft.fftfreq(nz, d=config.box_size.z / nz)
    kx_3d = kx.reshape(nx, 1, 1)
    ky_3d = ky.reshape(1, ny, 1)
    kz_3d = kz.reshape(1, 1, nz)
    k_squared = kx_3d ** 2 + ky_3d ** 2 + kz_3d ** 2
    kk = jnp.sqrt(k_squared)

    # spectrum k^6 exp(-8 k / kpk) peaks at k = 0.75 kpk -> kpk = k_f / 0.75
    kpk = k_f / 0.75
    Pk = kk ** 6 * jnp.exp(-8.0 * kk / kpk)

    key, sk1, sk2 = jax.random.split(key, 3)
    raw = jax.random.normal(sk1, shape=(3, nx, ny, nz)) + \
        1j * jax.random.normal(sk2, shape=(3, nx, ny, nz))
    cwx = jnp.sqrt(Pk) * raw[0]
    cwy = jnp.sqrt(Pk) * raw[1]
    cwz = jnp.sqrt(Pk) * raw[2]
    cwx = cwx.at[0, 0, 0].set(0.0 + 0.0j)
    cwy = cwy.at[0, 0, 0].set(0.0 + 0.0j)
    cwz = cwz.at[0, 0, 0].set(0.0 + 0.0j)

    # project out the compressible (curl-free) component -> solenoidal
    k_squared_safe = jnp.where(k_squared == 0.0, 1.0, k_squared)
    div_k = (kx_3d * cwx + ky_3d * cwy + kz_3d * cwz) / k_squared_safe
    div_k = div_k.at[0, 0, 0].set(0.0 + 0.0j)
    cwx = cwx - kx_3d * div_k
    cwy = cwy - ky_3d * div_k
    cwz = cwz - kz_3d * div_k

    wx = jnp.real(jnp.fft.ifftn(cwx))
    wy = jnp.real(jnp.fft.ifftn(cwy))
    wz = jnp.real(jnp.fft.ifftn(cwz))

    # normalise to unit rms
    norm = jnp.sqrt(jnp.mean(wx ** 2 + wy ** 2 + wz ** 2) + 1e-30)
    field = jnp.stack([wx, wy, wz]) / norm
    return key, field


@partial(jax.jit, static_argnames=["config"])
def _init_ou_forcing_state(key, config, turbulent_forcing_params):
    """Initial OU forcing state ``(key, f0)`` with f0 a stationary draw."""
    key, field = _create_solenoidal_field(
        key, config, turbulent_forcing_params.forcing_wavenumber
    )
    return (key, field)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_ou_forcing(
    forcing_state,
    primitive_state,
    dt,
    turbulent_forcing_params,
    config,
    registered_variables,
):
    """Apply Ornstein-Uhlenbeck forcing.

    The persistent forcing field ``f`` (carried in ``forcing_state``) is evolved
    with the exact OU discretisation ``f <- a f + sqrt(1 - a^2) xi``,
    ``a = exp(-dt / tau_f)``, keeping it at unit rms, then applied as a
    constant-amplitude acceleration ``velocity += F0 f dt`` (state-independent,
    so the adjoint is clean and the realisation is reproducible for a fixed
    timestep sequence).
    """
    key, f = forcing_state
    tau_f = turbulent_forcing_params.correlation_time
    a = jnp.exp(-dt / tau_f)
    key, xi = _create_solenoidal_field(
        key, config, turbulent_forcing_params.forcing_wavenumber
    )
    f = a * f + jnp.sqrt(jnp.maximum(1.0 - a ** 2, 0.0)) * xi

    F0 = turbulent_forcing_params.forcing_amplitude
    vx_i = registered_variables.velocity_index.x
    vy_i = registered_variables.velocity_index.y
    vz_i = registered_variables.velocity_index.z
    primitive_state = primitive_state.at[vx_i].add(F0 * f[0] * dt)
    primitive_state = primitive_state.at[vy_i].add(F0 * f[1] * dt)
    primitive_state = primitive_state.at[vz_i].add(F0 * f[2] * dt)

    return (key, f), primitive_state


# experimental, taken from the HOW-MHD Fortran code, this kind of
# protection was not mentioned in the paper, otherwise 
# turbulent simulations crash
@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _vacuum_protection(
    primitive_state,
    rhopmin: float,
    vel_max: float,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Applies the vacuum protection routine.
    For cells where density < rhopmin, it averages the density and momentum 
    over the 3x3x3 neighborhood of valid cells, and clips velocities to vel_max.
    """
    # Extract current primitive state
    rho = primitive_state[registered_variables.density_index]
    vx = primitive_state[registered_variables.velocity_index.x]
    vy = primitive_state[registered_variables.velocity_index.y]
    vz = primitive_state[registered_variables.velocity_index.z]

    # Reconstruct conserved momentum (matches q(iw, 2:4) in Fortran)
    mom_x = rho * vx
    mom_y = rho * vy
    mom_z = rho * vz

    # Identify vacuum cells (invalid) and healthy cells (valid)
    is_invalid = rho <= rhopmin
    is_valid = ~is_invalid

    # Zero out invalid cells so they don't contribute to the neighborhood sum
    rho_valid = rho * is_valid
    mom_x_valid = mom_x * is_valid
    mom_y_valid = mom_y * is_valid
    mom_z_valid = mom_z * is_valid
    count_valid = is_valid.astype(rho.dtype)

    # Helper function to sum over the local 3x3x3 neighborhood using periodic shifts
    def sum_neighbors(arr):
        out = jnp.zeros_like(arr)
        offsets = [-1, 0, 1]
        axes = tuple(range(config.dimensionality))
        # Itertools handles 1D, 2D, and 3D dynamically
        for shift in itertools.product(offsets, repeat=config.dimensionality):
            out += jnp.roll(arr, shift=shift, axis=axes)
        return out

    # Compute sums over the valid neighbors
    rho_sum = sum_neighbors(rho_valid)
    mom_x_sum = sum_neighbors(mom_x_valid)
    mom_y_sum = sum_neighbors(mom_y_valid)
    mom_z_sum = sum_neighbors(mom_z_valid)
    count_sum = sum_neighbors(count_valid)

    # Handle safe division
    has_valid_neighbors = count_sum > 0
    count_safe = jnp.where(has_valid_neighbors, count_sum, 1.0)
    rho_sum_safe = jnp.where(has_valid_neighbors, rho_sum, 1.0)

    # Calculate patched values from neighbors
    rho_patched = jnp.where(has_valid_neighbors, rho_sum / count_safe, rhopmin)
    
    # If an invalid cell has NO valid neighbors, Fortran explicitly dampens 
    # the velocity by dividing the old momentum by the new rhopmin limit.
    vx_isolated = mom_x / rhopmin
    vy_isolated = mom_y / rhopmin
    vz_isolated = mom_z / rhopmin

    vx_patched = jnp.where(has_valid_neighbors, mom_x_sum / rho_sum_safe, vx_isolated)
    vy_patched = jnp.where(has_valid_neighbors, mom_y_sum / rho_sum_safe, vy_isolated)
    vz_patched = jnp.where(has_valid_neighbors, mom_z_sum / rho_sum_safe, vz_isolated)

    # Apply velocity ceiling strictly to the patched cells
    vx_patched = jnp.clip(vx_patched, -vel_max, vel_max)
    vy_patched = jnp.clip(vy_patched, -vel_max, vel_max)
    vz_patched = jnp.clip(vz_patched, -vel_max, vel_max)

    # Merge the patched cells back into the global state
    rho_new = jnp.where(is_invalid, rho_patched, rho)
    vx_new = jnp.where(is_invalid, vx_patched, vx)
    vy_new = jnp.where(is_invalid, vy_patched, vy)
    vz_new = jnp.where(is_invalid, vz_patched, vz)

    # Reconstruct the final primitive array
    primitive_new = primitive_state.at[registered_variables.density_index].set(rho_new)
    primitive_new = primitive_new.at[registered_variables.velocity_index.x].set(vx_new)
    primitive_new = primitive_new.at[registered_variables.velocity_index.y].set(vy_new)
    primitive_new = primitive_new.at[registered_variables.velocity_index.z].set(vz_new)

    return primitive_new

@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _apply_forcing(
    key,
    primitive_state,
    dt,
    turbulent_forcing_params: TurbulentForcingParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    
    key, wx_real, wy_real, wz_real = _create_forcing_field(key, config)

    Edot = turbulent_forcing_params.energy_injection_rate
    dtforc = dt
    dV = config.grid_spacing**3
    
    # get density and velocities
    rho = primitive_state[registered_variables.density_index]
    u = primitive_state[registered_variables.velocity_index.x]
    v = primitive_state[registered_variables.velocity_index.y]
    w = primitive_state[registered_variables.velocity_index.z]

    # compute terms for quadratic equation
    # a * amp^2 + b * amp + c = 0
    tempa = 0.5 * jnp.sum(rho * (wx_real**2 + wy_real**2 + wz_real**2))
    tempb = jnp.sum(rho * u * wx_real + rho * v * wy_real + rho * w * wz_real)
    tempc = -Edot * dtforc / dV
    
    # solve quadratic equation for amplitude
    discriminant = tempb**2 - 4.0 * tempa * tempc
    
    # conditional to handle edge cases
    amp = jax.lax.cond(
        (discriminant >= 0) & (jnp.abs(tempa) > 1e-10),
        lambda: (-tempb + jnp.sqrt(discriminant)) / (2.0 * tempa),
        lambda: 0.0
    )

    # apply new velocities directly to the primitive state
    primitive_state = primitive_state.at[registered_variables.velocity_index.x].add(amp * wx_real)
    primitive_state = primitive_state.at[registered_variables.velocity_index.y].add(amp * wy_real)
    primitive_state = primitive_state.at[registered_variables.velocity_index.z].add(amp * wz_real)

    # protection routine
    if config.turbulent_forcing_config.vacuum_protection:
        primitive_state = _vacuum_protection(
            primitive_state,
            turbulent_forcing_params.protection_density_threshold,
            turbulent_forcing_params.protection_max_velocity,
            config,
            registered_variables
        )

    return key, primitive_state


# @partial(jax.jit, static_argnames=["config", "registered_variables"])
# def apply_forcing(
#     key,
#     conserved_state,
#     dt,
#     turbulent_forcing_params: TurbulentForcingParams,
#     config: SimulationConfig,
#     registered_variables: RegisteredVariables,
# ):
    
#     key, wx_real, wy_real, wz_real = create_forcing_field(key, config)

#     Edot = turbulent_forcing_params.energy_injection_rate
#     dtforc = dt
#     dV = config.grid_spacing**3
    
#     # get density and velocities
#     rho = conserved_state[registered_variables.density_index]
#     rhou = conserved_state[registered_variables.momentum_index.x]
#     rhov = conserved_state[registered_variables.momentum_index.y]
#     rhow = conserved_state[registered_variables.momentum_index.z]

#     # compute terms for quadratic equation
#     # a * amp^2 + b * amp + c = 0
#     tempa = 0.5 * jnp.sum(rho * (wx_real**2 + wy_real**2 + wz_real**2))
#     tempb = jnp.sum(rhou * wx_real + rhov * wy_real + rhow * wz_real)
#     tempc = -Edot * dtforc / dV
    
#     # solve quadratic equation for amplitude
#     discriminant = tempb**2 - 4.0 * tempa * tempc
    
#     # conditional to handle edge cases
#     amp = jax.lax.cond(
#         (discriminant >= 0) & (jnp.abs(tempa) > 1e-10),
#         lambda: (-tempb + jnp.sqrt(discriminant)) / (2.0 * tempa),
#         lambda: 0.0
#     )
    
#     # apply forcing to momentum
#     conserved_state = conserved_state.at[registered_variables.momentum_index.x].add(
#         rho * amp * wx_real
#     )
#     conserved_state = conserved_state.at[registered_variables.momentum_index.y].add(
#         rho * amp * wy_real
#     )
#     conserved_state = conserved_state.at[registered_variables.momentum_index.z].add(
#         rho * amp * wz_real
#     )

#     # update the total energy
#     # kinetic energy = 0.5 * rho * u_squared
#     # where rho u -> rho (u + amp w)
#     # -> change in kinetic energy = 0.5 * rho * [ (u + amp w)^2 - u^2 ]
#     new_squared_velocity = (
#         (rhou / rho + amp * wx_real) ** 2
#         + (rhov / rho + amp * wy_real) ** 2
#         + (rhow / rho + amp * wz_real) ** 2
#     )
#     old_squared_velocity = (
#         (rhou / rho) ** 2
#         + (rhov / rho) ** 2
#         + (rhow / rho) ** 2
#     )
#     delta_kinetic_energy = 0.5 * rho * (new_squared_velocity - old_squared_velocity)
#     conserved_state = conserved_state.at[registered_variables.energy_index].add(delta_kinetic_energy)
    
#     return key, conserved_state