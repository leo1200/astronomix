"""
Shared diagnostics for the KH reconstruction study.

Conventions: fields are (Nx, Ny) or (2, Nx, Ny); x (axis -2 for 2-comp, axis 0
for scalar) is the periodic flow direction, so spectral analysis is a real FFT
over the streamwise (x) wavenumber kx (integer mode number on the unit box).
"""

import jax.numpy as jnp
import numpy as np


# ----------------------------------------------------------------------
#  streamwise spectral tools (x periodic)
# ----------------------------------------------------------------------

def streamwise_modes(nx):
    """Integer streamwise wavenumbers for an rfft over x of length nx."""
    return jnp.arange(nx // 2 + 1)


def lowpass_x(field, kcut):
    """Keep streamwise modes |kx| <= kcut (real fft over axis -2)."""
    n = field.shape[-2]
    fk = jnp.fft.rfft(field, axis=-2)
    k = streamwise_modes(n)
    mask = (k <= kcut).astype(field.dtype)
    shape = [1] * field.ndim
    shape[-2] = mask.shape[0]
    return jnp.fft.irfft(fk * mask.reshape(shape), n=n, axis=-2)


def highpass_x(field, kcut):
    return field - lowpass_x(field, kcut)


def streamwise_energy_spectrum(vel, y_slab=None):
    """Per-kx kinetic-energy spectrum of a (2,Nx,Ny) velocity field.

    Power is summed over the two components and (optionally restricted then)
    averaged over the transverse y rows in ``y_slab`` (a boolean mask over y),
    so the spectrum reflects the shear-layer region rather than the quiescent
    free stream. Returns (kx, E(kx)).
    """
    nx = vel.shape[-2]
    vk = jnp.fft.rfft(vel, axis=-2)            # (2, nkx, Ny)
    p = (jnp.abs(vk) ** 2).sum(axis=0)          # (nkx, Ny)
    if y_slab is not None:
        p = p[:, y_slab]
    Ek = p.mean(axis=-1)                         # average over y
    return streamwise_modes(nx), Ek


def per_kx_relative_error(rec, truth, y_slab=None):
    """Per-kx relative reconstruction error ||rec(kx)-truth(kx)|| / ||truth(kx)||
    for two (2,Nx,Ny) fields, summed over components and y (in the slab)."""
    rk = jnp.fft.rfft(rec, axis=-2)
    tk = jnp.fft.rfft(truth, axis=-2)
    if y_slab is not None:
        rk = rk[:, :, y_slab]; tk = tk[:, :, y_slab]
    num = jnp.sqrt(jnp.sum(jnp.abs(rk - tk) ** 2, axis=(0, 2)))
    den = jnp.sqrt(jnp.sum(jnp.abs(tk) ** 2, axis=(0, 2)))
    return num / (den + 1e-30)


def per_kx_correlation(a, b, y_slab=None):
    """Per-kx normalized correlation between two (2,Nx,Ny) fields.

    corr(kx) = Re<â(kx), b̂(kx)> / (|â(kx)| |b̂(kx)|), summed over components and
    averaged over y (in the slab). 1 = perfectly aligned, 0 = decorrelated.
    """
    ak = jnp.fft.rfft(a, axis=-2)
    bk = jnp.fft.rfft(b, axis=-2)
    if y_slab is not None:
        ak = ak[:, :, y_slab]; bk = bk[:, :, y_slab]
    num = jnp.real(jnp.sum(jnp.conj(ak) * bk, axis=(0, 2)))
    na = jnp.sqrt(jnp.sum(jnp.abs(ak) ** 2, axis=(0, 2)))
    nb = jnp.sqrt(jnp.sum(jnp.abs(bk) ** 2, axis=(0, 2)))
    return num / (na * nb + 1e-30)


# ----------------------------------------------------------------------
#  physical-space diagnostics
# ----------------------------------------------------------------------

def vorticity(vel, dx):
    """omega_z = dvy/dx - dvx/dy via central differences (periodic in x)."""
    dvydx = (jnp.roll(vel[1], -1, 0) - jnp.roll(vel[1], 1, 0)) / (2 * dx)
    dvxdy = (jnp.roll(vel[0], -1, 1) - jnp.roll(vel[0], 1, 1)) / (2 * dx)
    return dvydx - dvxdy


def vorticity_thickness(vx, dx, dV):
    """Mixing-layer width h = dV / max_y |d<vx>_x/dy| (x-averaged mean profile)."""
    vbar = vx.mean(axis=0)                       # mean over x -> profile(y)
    dvdy = jnp.gradient(vbar, dx)
    return dV / (jnp.max(jnp.abs(dvdy)) + 1e-30)


def momentum_thickness(vx, dx, dV, y):
    """Momentum thickness of the x-averaged shear profile (units of y)."""
    vbar = vx.mean(axis=0)
    u = vbar / (0.5 * dV)                        # normalized to +/-1
    integ = 0.25 * (1.0 - u ** 2)
    return float(jnp.trapezoid(integ, y))


def shear_layer_slab(y, yc, half_width):
    """Boolean mask selecting |y - yc| <= half_width."""
    return jnp.abs(y - yc) <= half_width
