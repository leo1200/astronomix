"""
Shell-averaged energy, cross and helicity spectra of 3D vector fields.

Provides the FFT-based spectral diagnostics used by the snapshot machinery:
generic vector-field energy and cross spectra, plus the MHD physics wrappers
(density-weighted kinetic energy, magnetic energy, cross helicity, magnetic
helicity). All spectra share a single shell-binning of the wavenumber grid so
their bins line up.
"""

# general
import math

# jax
import jax.numpy as jnp


# ==========================================================================
#  Binning mode constants
# ==========================================================================

LOG_BINNING = 0       # Logarithmic bins — constant dk/k, smoothest at high k
INTEGER_BINNING = 1   # Integer mode shells — dk = 2*pi/L, good statistics per bin
PHYSICAL_BINNING = 2  # Physical wavenumber — dk = 1, finest resolution (default)


# ==========================================================================
#  Shared helpers
# ==========================================================================

def _wavenumber_bins(Nx, Ny, Nz, binning=PHYSICAL_BINNING):
    """
    Compute shell-binning indices for spectral accumulation.

    Args:
        Nx, Ny, Nz: Grid dimensions.
        binning: One of LOG_BINNING, INTEGER_BINNING, PHYSICAL_BINNING.

    Returns:
        bin_indices: Flat int32 shell-bin indices, shape (Nx*Ny*Nz,).
        number_of_bins: Number of shells.
        wavenumber_centers: Physical wavenumber bin centers, shape
            (number_of_bins,).
    """
    # Integer mode numbers along each axis (fftfreq with d = 1/N yields integers).
    freq_x = jnp.fft.fftfreq(Nx, d=1.0 / Nx)
    freq_y = jnp.fft.fftfreq(Ny, d=1.0 / Ny)
    freq_z = jnp.fft.fftfreq(Nz, d=1.0 / Nz)
    freq_grid_x, freq_grid_y, freq_grid_z = jnp.meshgrid(
        freq_x, freq_y, freq_z, indexing="ij"
    )

    # Magnitude of the integer mode vector at each grid point, and the largest
    # magnitude any mode can reach (the corner of the half-grid).
    mode_magnitude = jnp.sqrt(freq_grid_x**2 + freq_grid_y**2 + freq_grid_z**2)
    max_mode = math.sqrt((Nx // 2) ** 2 + (Ny // 2) ** 2 + (Nz // 2) ** 2)

    if binning == LOG_BINNING:
        k_min = 1.0
        k_max = max_mode
        number_of_bins = max(Nx, Ny, Nz) // 4
        log_edges = jnp.logspace(
            jnp.log10(k_min * 0.5), jnp.log10(k_max + 0.5), number_of_bins + 1
        )
        bin_indices = jnp.clip(
            jnp.digitize(mode_magnitude.ravel(), log_edges) - 1, 0, number_of_bins - 1
        )
        # Bin centers are the geometric mean of the edges, scaled to physical
        # wavenumber.
        wavenumber_centers = 2.0 * jnp.pi * jnp.sqrt(log_edges[:-1] * log_edges[1:])
        return bin_indices, number_of_bins, wavenumber_centers

    elif binning == INTEGER_BINNING:
        bin_indices = mode_magnitude.astype(jnp.int32).ravel()
        number_of_bins = int(max_mode) + 2
        wavenumber_centers = (jnp.arange(number_of_bins) + 0.5) * 2.0 * jnp.pi
        return bin_indices, number_of_bins, wavenumber_centers

    else:  # PHYSICAL_BINNING
        physical_wavenumber = 2.0 * jnp.pi * mode_magnitude
        bin_indices = physical_wavenumber.astype(jnp.int32).ravel()
        number_of_bins = int(2.0 * math.pi * max_mode) + 2
        wavenumber_centers = jnp.arange(number_of_bins) + 0.5
        return bin_indices, number_of_bins, wavenumber_centers


# ==========================================================================
#  Generic spectrum functions
# ==========================================================================

def vector_field_energy_spectrum(fx, fy, fz, energy_coeff=1.0,
                                binning=PHYSICAL_BINNING):
    """
    Energy spectrum E(k) of a vector field (fx, fy, fz).

    Satisfies: sum(E(k)) == mean(c * |f|^2) over the domain
    (shell-summed convention).

    Args:
        fx, fy, fz: Field components, each shaped (Nx, Ny, Nz).
        energy_coeff: Scalar multiplier c (default 1.0).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        wavenumber_centers: Physical wavenumber bin centers.
        Ek: Energy spectrum per bin.

    Based on: https://qiauil.github.io/blog/2026/tke_spectrum/
    """
    Nx, Ny, Nz = fx.shape
    N_total = float(Nx * Ny * Nz)

    fx_hat = jnp.fft.fftn(fx)
    fy_hat = jnp.fft.fftn(fy)
    fz_hat = jnp.fft.fftn(fz)

    energy_fft = energy_coeff * (
        jnp.abs(fx_hat) ** 2 + jnp.abs(fy_hat) ** 2 + jnp.abs(fz_hat) ** 2
    ) / N_total**2

    bin_indices, number_of_bins, wavenumber_centers = _wavenumber_bins(
        Nx, Ny, Nz, binning
    )
    Ek = jnp.zeros(number_of_bins).at[bin_indices].add(energy_fft.ravel())
    return wavenumber_centers, Ek


def vector_field_cross_spectrum(f1x, f1y, f1z, f2x, f2y, f2z, coeff=1.0,
                                binning=PHYSICAL_BINNING):
    """
    Cross spectrum C(k) between two vector fields f1 and f2.

    Satisfies: sum(C(k)) == mean(c * f1 . f2) over the domain.

    The spectrum is the shell-summed co-spectrum (real part of the
    conjugate dot product in Fourier space). Unlike energy spectra,
    cross spectra are not positive-definite.

    Args:
        f1x, f1y, f1z: First field components, each (Nx, Ny, Nz).
        f2x, f2y, f2z: Second field components, each (Nx, Ny, Nz).
        coeff: Scalar multiplier (default 1.0).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        wavenumber_centers: Physical wavenumber bin centers.
        Ck: Cross spectrum per bin.
    """
    Nx, Ny, Nz = f1x.shape
    N_total = float(Nx * Ny * Nz)

    f1x_hat = jnp.fft.fftn(f1x)
    f1y_hat = jnp.fft.fftn(f1y)
    f1z_hat = jnp.fft.fftn(f1z)

    f2x_hat = jnp.fft.fftn(f2x)
    f2y_hat = jnp.fft.fftn(f2y)
    f2z_hat = jnp.fft.fftn(f2z)

    cross_fft = coeff * jnp.real(
        jnp.conj(f1x_hat) * f2x_hat
        + jnp.conj(f1y_hat) * f2y_hat
        + jnp.conj(f1z_hat) * f2z_hat
    ) / N_total**2

    bin_indices, number_of_bins, wavenumber_centers = _wavenumber_bins(
        Nx, Ny, Nz, binning
    )
    Ck = jnp.zeros(number_of_bins).at[bin_indices].add(cross_fft.ravel())
    return wavenumber_centers, Ck


# ==========================================================================
#  MHD physics wrappers
# ==========================================================================

def get_kinetic_energy_spectrum(vx, vy, vz, rho, binning=PHYSICAL_BINNING):
    """
    Kinetic energy spectrum with density weighting.

    Uses w = sqrt(rho) * u so that sum(E_k) == mean(0.5 * rho * |u|^2).

    Args:
        vx, vy, vz: Velocity components, each (Nx, Ny, Nz).
        rho: Density field, (Nx, Ny, Nz).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        wavenumber_centers, Ek: Wavenumber bin centers and kinetic energy spectrum.
    """
    rho_sqrt = jnp.sqrt(rho)
    return vector_field_energy_spectrum(
        rho_sqrt * vx, rho_sqrt * vy, rho_sqrt * vz,
        energy_coeff=0.5, binning=binning,
    )


def get_magnetic_energy_spectrum(Bx, By, Bz, mu0=1.0, binning=PHYSICAL_BINNING):
    """
    Magnetic energy spectrum.

    sum(E_m) == mean(|B|^2 / (2 * mu0)).

    Args:
        Bx, By, Bz: Magnetic field components, each (Nx, Ny, Nz).
        mu0: Magnetic permeability (default 1.0 for Alfvénic units).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        wavenumber_centers, Em: Wavenumber bin centers and magnetic energy spectrum.
    """
    return vector_field_energy_spectrum(
        Bx, By, Bz,
        energy_coeff=1.0 / (2.0 * mu0), binning=binning,
    )


def get_cross_helicity_spectrum(vx, vy, vz, Bx, By, Bz, binning=PHYSICAL_BINNING):
    """
    Cross-helicity spectrum H_c(k).

    Cross helicity: H_c = integral(u . B) dV.
    Spectrum satisfies: sum(H_c(k)) == mean(u . B).

    Args:
        vx, vy, vz: Velocity components, each (Nx, Ny, Nz).
        Bx, By, Bz: Magnetic field components, each (Nx, Ny, Nz).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        wavenumber_centers, Hc: Wavenumber bin centers and cross-helicity spectrum.
    """
    return vector_field_cross_spectrum(
        vx, vy, vz, Bx, By, Bz,
        coeff=1.0, binning=binning,
    )


def get_magnetic_helicity_spectrum(Bx, By, Bz, binning=PHYSICAL_BINNING):
    """
    Magnetic helicity spectrum H_m(k).

    Magnetic helicity: H_m = integral(A . B) dV, where B = curl(A).
    In a periodic domain with Coulomb gauge, the vector potential is
    recovered in Fourier space as: A_hat(k) = +i k x B_hat(k) / |k|^2.

    Spectrum satisfies: sum(H_m(k)) == mean(A . B).

    Args:
        Bx, By, Bz: Magnetic field components, each (Nx, Ny, Nz).
        binning: LOG_BINNING, INTEGER_BINNING, or PHYSICAL_BINNING.

    Returns:
        wavenumber_centers, Hm: Wavenumber bin centers and magnetic helicity spectrum.
    """
    Nx, Ny, Nz = Bx.shape
    N_total = float(Nx * Ny * Nz)

    Bx_hat = jnp.fft.fftn(Bx)
    By_hat = jnp.fft.fftn(By)
    Bz_hat = jnp.fft.fftn(Bz)

    # Physical wavevector components on the FFT grid.
    freq_x = jnp.fft.fftfreq(Nx, d=1.0 / Nx)
    freq_y = jnp.fft.fftfreq(Ny, d=1.0 / Ny)
    freq_z = jnp.fft.fftfreq(Nz, d=1.0 / Nz)
    freq_grid_x, freq_grid_y, freq_grid_z = jnp.meshgrid(
        freq_x, freq_y, freq_z, indexing="ij"
    )

    kx = 2.0 * jnp.pi * freq_grid_x
    ky = 2.0 * jnp.pi * freq_grid_y
    kz = 2.0 * jnp.pi * freq_grid_z

    k_sq = kx**2 + ky**2 + kz**2
    # Guard the k=0 mode against division by zero; its contribution is zeroed
    # out below, so the placeholder value is irrelevant.
    k_sq_safe = jnp.where(k_sq == 0, 1.0, k_sq)

    # Vector potential in Coulomb gauge: A_hat = +i (k x B_hat) / |k|^2, built
    # from the cross product k x B_hat component-wise.
    curl_x = ky * Bz_hat - kz * By_hat
    curl_y = kz * Bx_hat - kx * Bz_hat
    curl_z = kx * By_hat - ky * Bx_hat

    Ax_hat = +1j * curl_x / k_sq_safe
    Ay_hat = +1j * curl_y / k_sq_safe
    Az_hat = +1j * curl_z / k_sq_safe

    # Drop the k=0 mode: it carries no helicity and would otherwise pick up the
    # placeholder denominator above.
    Ax_hat = jnp.where(k_sq == 0, 0.0, Ax_hat)
    Ay_hat = jnp.where(k_sq == 0, 0.0, Ay_hat)
    Az_hat = jnp.where(k_sq == 0, 0.0, Az_hat)

    # Helicity density in Fourier space: Re(A_hat* . B_hat).
    helicity_fft = jnp.real(
        jnp.conj(Ax_hat) * Bx_hat
        + jnp.conj(Ay_hat) * By_hat
        + jnp.conj(Az_hat) * Bz_hat
    ) / N_total**2

    # Accumulate the helicity density into wavenumber shells.
    bin_indices, number_of_bins, wavenumber_centers = _wavenumber_bins(
        Nx, Ny, Nz, binning
    )
    Hm = jnp.zeros(number_of_bins).at[bin_indices].add(helicity_fft.ravel())
    return wavenumber_centers, Hm