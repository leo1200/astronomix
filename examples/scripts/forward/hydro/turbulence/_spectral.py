"""The single shell-averaged spectral estimator both codes are measured with.

Kept in its own module so that ``driven_turbulence.py`` (which reduces
astronomix snapshots while they are still in memory) and ``spectra.py`` (which
reduces AthenaK's ``.bin`` dumps from disk) provably go through the same code
path. Any difference in the resulting spectra is then a difference in the
solvers, not in the diagnostics.

Convention: ``E(n)`` is shell-summed over integer mode shells
``n = |k| L / 2pi``, normalised so that ``sum_n E(n) == mean(0.5 |v|^2)`` over
the box — the same convention as
``astronomix.analysis_helpers.energy_spectrum``.
"""

# numerics
import numpy as np


def _cell_average_transfer(N):
    """Modal transfer function of averaging a field over a cell, per 3D mode.

    A finite-volume code stores cell averages while a finite-difference code
    stores point values, and averaging a mode over one cell multiplies it by
    ``sinc(pi n_i / N)`` along each axis. Comparing an FV spectrum with an FD one
    therefore compares a slightly low-pass-filtered field with an unfiltered one.
    The suppression is only ~1-3% in energy over the shells where the schemes'
    dissipation scales sit, but it is a real systematic that favours the FD code,
    so it is made correctable rather than assumed away.
    """
    f = np.fft.fftfreq(N, d=1.0 / N)
    # np.sinc(x) is sin(pi x)/(pi x), so the per-axis factor is np.sinc(n_i / N).
    t = np.sinc(f / N)
    return t[:, None, None] * t[None, :, None] * t[None, None, :]


def shell_spectrum(vx, vy, vz, rho=None, deconvolve_cell_average=False):
    """Shell-summed kinetic-energy spectrum over integer mode shells.

    Args:
        vx, vy, vz: Velocity components, each ``(N, N, N)`` in ``(x, y, z)``
            axis order.
        rho: Optional density; when given the field is weighted as
            ``sqrt(rho) v`` so the sum is the mean kinetic energy *density*.
        deconvolve_cell_average: Divide out the cell-averaging transfer function
            before shell-binning. Use for finite-volume data (AthenaK) when
            comparing against finite-difference point values (astronomix).

    Returns:
        ``(n, E)``: integer shell numbers ``0 .. N/2`` and the shell-summed
        energy in each.
    """
    if rho is not None:
        w = np.sqrt(rho)
        vx, vy, vz = w * vx, w * vy, w * vz

    N = vx.shape[0]
    n_tot = float(N) ** 3
    energy = 0.5 * sum(np.abs(np.fft.fftn(c)) ** 2 for c in (vx, vy, vz)) / n_tot ** 2

    if deconvolve_cell_average:
        t2 = _cell_average_transfer(N) ** 2
        # The Nyquist row has sinc -> 2/pi, never 0, so no guard is needed; clip
        # only against pathological underflow.
        energy = energy / np.maximum(t2, 1e-6)

    # Integer mode number along each axis; a mode lands in shell n when
    # |n_vec| is within half a shell of n.
    f = np.fft.fftfreq(N, d=1.0 / N)
    mag = np.sqrt(f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2)
    shell = np.rint(mag).astype(np.int64).ravel()

    n_bins = N // 2 + 1
    keep = shell < n_bins
    E = np.bincount(shell[keep], weights=energy.ravel()[keep], minlength=n_bins)
    return np.arange(n_bins), E


def reduce_snapshots(snapshots, weighted=False, deconvolve_cell_average=False):
    """Per-snapshot spectra and rms velocity for an iterable of snapshots.

    Args:
        snapshots: Iterable of ``(time, rho, vx, vy, vz)``.
        weighted: Use the ``sqrt(rho)``-weighted spectrum.
        deconvolve_cell_average: Undo finite-volume cell averaging (see
            :func:`shell_spectrum`).

    Returns:
        dict with ``times``, ``n_shell``, ``E_snap`` ``(n_snap, n_bins)`` and
        ``v_rms`` ``(n_snap,)``.
    """
    times, spectra, v_rms = [], [], []
    n_shell = None
    for t, rho, vx, vy, vz in snapshots:
        n_shell, E = shell_spectrum(vx, vy, vz, rho=rho if weighted else None,
                                    deconvolve_cell_average=deconvolve_cell_average)
        times.append(t)
        spectra.append(E)
        v_rms.append(np.sqrt(np.mean(vx ** 2 + vy ** 2 + vz ** 2)))
    if n_shell is None:
        raise ValueError("no snapshots given")
    return dict(times=np.asarray(times), n_shell=n_shell,
                E_snap=np.asarray(spectra), v_rms=np.asarray(v_rms))
