import jax.numpy as jnp
import math


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
        k_idx: Flat int32 bin indices, shape (Nx*Ny*Nz,).
        n_bins: Number of bins.
        k_centers: Physical wavenumber bin centers, shape (n_bins,).
    """
    freq_x = jnp.fft.fftfreq(Nx, d=1.0 / Nx)  # integer mode numbers
    freq_y = jnp.fft.fftfreq(Ny, d=1.0 / Ny)
    freq_z = jnp.fft.fftfreq(Nz, d=1.0 / Nz)
    FX, FY, FZ = jnp.meshgrid(freq_x, freq_y, freq_z, indexing="ij")

    m_mag = jnp.sqrt(FX**2 + FY**2 + FZ**2)  # integer mode magnitude
    max_mode = math.sqrt((Nx // 2) ** 2 + (Ny // 2) ** 2 + (Nz // 2) ** 2)

    if binning == LOG_BINNING:
        k_min = 1.0
        k_max = max_mode
        n_bins = max(Nx, Ny, Nz) // 4
        log_edges = jnp.logspace(
            jnp.log10(k_min * 0.5), jnp.log10(k_max + 0.5), n_bins + 1
        )
        k_idx = jnp.clip(
            jnp.digitize(m_mag.ravel(), log_edges) - 1, 0, n_bins - 1
        )
        # Geometric mean of edges, converted to physical wavenumber
        k_centers = 2.0 * jnp.pi * jnp.sqrt(log_edges[:-1] * log_edges[1:])
        return k_idx, n_bins, k_centers

    elif binning == INTEGER_BINNING:
        k_idx = m_mag.astype(jnp.int32).ravel()
        n_bins = int(max_mode) + 2
        k_centers = (jnp.arange(n_bins) + 0.5) * 2.0 * jnp.pi
        return k_idx, n_bins, k_centers

    else:  # PHYSICAL_BINNING
        k_phys = 2.0 * jnp.pi * m_mag
        k_idx = k_phys.astype(jnp.int32).ravel()
        n_bins = int(2.0 * math.pi * max_mode) + 2
        k_centers = jnp.arange(n_bins) + 0.5
        return k_idx, n_bins, k_centers


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
        k_centers: Physical wavenumber bin centers.
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

    k_idx, n_bins, k_centers = _wavenumber_bins(Nx, Ny, Nz, binning)
    Ek = jnp.zeros(n_bins).at[k_idx].add(energy_fft.ravel())
    return k_centers, Ek


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
        k_centers: Physical wavenumber bin centers.
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

    k_idx, n_bins, k_centers = _wavenumber_bins(Nx, Ny, Nz, binning)
    Ck = jnp.zeros(n_bins).at[k_idx].add(cross_fft.ravel())
    return k_centers, Ck


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
        k_centers, Ek: Wavenumber bin centers and kinetic energy spectrum.
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
        k_centers, Em: Wavenumber bin centers and magnetic energy spectrum.
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
        k_centers, Hc: Wavenumber bin centers and cross-helicity spectrum.
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
        k_centers, Hm: Wavenumber bin centers and magnetic helicity spectrum.
    """
    Nx, Ny, Nz = Bx.shape
    N_total = float(Nx * Ny * Nz)

    Bx_hat = jnp.fft.fftn(Bx)
    By_hat = jnp.fft.fftn(By)
    Bz_hat = jnp.fft.fftn(Bz)

    # Physical wavevector components
    freq_x = jnp.fft.fftfreq(Nx, d=1.0 / Nx)
    freq_y = jnp.fft.fftfreq(Ny, d=1.0 / Ny)
    freq_z = jnp.fft.fftfreq(Nz, d=1.0 / Nz)
    KX, KY, KZ = jnp.meshgrid(freq_x, freq_y, freq_z, indexing="ij")

    kx = 2.0 * jnp.pi * KX
    ky = 2.0 * jnp.pi * KY
    kz = 2.0 * jnp.pi * KZ

    k_sq = kx**2 + ky**2 + kz**2
    k_sq_safe = jnp.where(k_sq == 0, 1.0, k_sq)

    # A_hat = +i (k x B_hat) / |k|^2  (Coulomb gauge)
    cx = ky * Bz_hat - kz * By_hat
    cy = kz * Bx_hat - kx * Bz_hat
    cz = kx * By_hat - ky * Bx_hat

    Ax_hat = +1j * cx / k_sq_safe
    Ay_hat = +1j * cy / k_sq_safe
    Az_hat = +1j * cz / k_sq_safe

    # Zero out k=0 mode (no helicity contribution, avoids division artefact)
    Ax_hat = jnp.where(k_sq == 0, 0.0, Ax_hat)
    Ay_hat = jnp.where(k_sq == 0, 0.0, Ay_hat)
    Az_hat = jnp.where(k_sq == 0, 0.0, Az_hat)

    # Helicity density in Fourier space: Re(A_hat* . B_hat)
    hel_fft = jnp.real(
        jnp.conj(Ax_hat) * Bx_hat
        + jnp.conj(Ay_hat) * By_hat
        + jnp.conj(Az_hat) * Bz_hat
    ) / N_total**2

    # Shell accumulation
    k_idx, n_bins, k_centers = _wavenumber_bins(Nx, Ny, Nz, binning)
    Hm = jnp.zeros(n_bins).at[k_idx].add(hel_fft.ravel())
    return k_centers, Hm