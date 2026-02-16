"""
The turbulent forcing module also draws from 
https://arxiv.org/pdf/2304.04360
"""

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
    
    # in our case we assume a cubic box
    # with equal number of cells in each direction

    xsize = config.box_size
    ysize = config.box_size
    zsize = config.box_size

    nx = config.num_cells
    ny = config.num_cells
    nz = config.num_cells
    
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
    kpk = 4.0 * jnp.pi / config.box_size
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
    
    # apply forcing to momentum
    primitive_state = primitive_state.at[registered_variables.velocity_index.x].add(
        amp * wx_real
    )
    primitive_state = primitive_state.at[registered_variables.velocity_index.y].add(
        amp * wy_real
    )
    primitive_state = primitive_state.at[registered_variables.velocity_index.z].add(
        amp * wz_real
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