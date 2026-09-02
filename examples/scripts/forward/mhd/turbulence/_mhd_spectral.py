"""The single shell-averaged spectral estimator both codes are measured with.

``dynamo_convergence.py`` reduces astronomix snapshots on the GPU while they are
still in memory; ``athenapk_turb.py`` reduces AthenaPK's Parthenon ``.phdf``
dumps from disk. Both call :func:`snapshot_spectra` here, so any difference in
the resulting spectra is a difference in the solvers, not in the diagnostics.

Convention: ``E(n)`` is shell-summed over integer mode shells ``n = |k| L / 2pi``
and normalised so that ``sum_n E_kin(n) == mean(0.5 rho |v|^2)`` and
``sum_n E_mag(n) == mean(0.5 |B|^2)`` over the box -- the same convention as
``astronomix.analysis_helpers.energy_spectrum`` and as the hydrodynamic study in
``examples/scripts/forward/hydro/turbulence/_spectral.py``.

Everything is written in ``jax.numpy`` so that the astronomix side can run it
inside the snapshot callback (no host round-trip of the full cube), and the
AthenaPK side gets the identical arithmetic on the same hardware.
"""

# numerics
import numpy as np
import jax.numpy as jnp


# -------------------------------------------------------------
# ================= ↓ Shell-binning kernel ↓ ==================
# -------------------------------------------------------------
def _shell_index(N):
    """Integer shell index ``n = rint(|n_vec|)`` for every mode of an N^3 grid."""
    f = jnp.fft.fftfreq(N, d=1.0 / N)
    mag = jnp.sqrt(
        f[:, None, None] ** 2 + f[None, :, None] ** 2 + f[None, None, :] ** 2
    )
    return jnp.rint(mag).astype(jnp.int32).ravel()


def _cell_average_transfer(N):
    """Modal transfer function of averaging a field over a cell, per 3D mode.

    A finite-volume code stores cell averages while a finite-difference code
    stores point values, and averaging a mode over one cell multiplies it by
    ``sinc(pi n_i / N)`` along each axis. Comparing an FV spectrum with an FD one
    therefore compares a low-pass-filtered field with an unfiltered one, which
    favours the FD code -- a real systematic, so it is made correctable rather
    than assumed away (it is 2% in energy at n = N/12 but 19% at n = N/4).
    """
    f = jnp.fft.fftfreq(N, d=1.0 / N)
    t = jnp.sinc(f / N)          # jnp.sinc(x) = sin(pi x) / (pi x)
    return t[:, None, None] * t[None, :, None] * t[None, None, :]


def shell_spectrum(fx, fy, fz, weight=None, deconvolve_cell_average=False):
    """Shell-summed energy spectrum ``0.5 |w f|^2`` of a 3D vector field.

    Args:
        fx, fy, fz: Field components, each ``(N, N, N)`` in ``(x, y, z)`` order.
        weight: Optional per-cell weight applied to each component before the
            transform (``sqrt(rho)`` for the kinetic energy spectrum).
        deconvolve_cell_average: Divide out the cell-averaging transfer function.
            Use for finite-volume data (AthenaPK) when comparing against
            finite-difference point values (astronomix).

    Returns:
        ``E``: shell-summed energy in shells ``n = 0 .. N/2``.
    """
    if weight is not None:
        fx, fy, fz = weight * fx, weight * fy, weight * fz

    N = fx.shape[0]
    n_tot = float(N) ** 3
    energy = 0.5 * sum(
        jnp.abs(jnp.fft.fftn(c)) ** 2 for c in (fx, fy, fz)
    ) / n_tot ** 2

    if deconvolve_cell_average:
        t2 = _cell_average_transfer(N) ** 2
        # The Nyquist row has sinc -> 2/pi, never 0, so no guard is needed; clip
        # only against pathological underflow.
        energy = energy / jnp.maximum(t2, 1e-6)

    n_bins = N // 2 + 1
    shell = _shell_index(N)
    inside = shell < n_bins
    return jnp.zeros(n_bins, dtype=energy.dtype).at[
        jnp.where(inside, shell, 0)
    ].add(jnp.where(inside, energy.ravel(), 0.0))


def shell_numbers(N):
    """The shell centres ``n = 0 .. N/2`` that :func:`shell_spectrum` returns."""
    return np.arange(N // 2 + 1)
# -------------------------------------------------------------
# ================= ↑ Shell-binning kernel ↑ ==================
# -------------------------------------------------------------


def relative_divb(bx, by, bz, grid_spacing):
    """AthenaPK's ``relDivB`` history diagnostic, ``<L_cell |div B| / |B|>``.

    Reimplemented here rather than taken from AthenaPK's ``.hst`` so that both
    codes' numbers come from one definition and can be put on the same axis.
    ``div B`` is the centred difference AthenaPK uses (its ``(B[i+1] - B[i-1])/dx``
    is twice the centred derivative, which its 0.5 prefactor undoes), evaluated
    on the *cell-centred* field in both codes -- astronomix's constrained
    transport keeps the staggered divergence at round-off by construction, so
    the cell-centred one is the only comparable measure.
    """
    dx = grid_spacing
    div = ((jnp.roll(bx, -1, 0) - jnp.roll(bx, 1, 0))
           + (jnp.roll(by, -1, 1) - jnp.roll(by, 1, 1))
           + (jnp.roll(bz, -1, 2) - jnp.roll(bz, 1, 2))) / dx
    abs_b = jnp.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    L = jnp.sqrt(3.0) * dx
    return jnp.mean(jnp.where(abs_b > 0.0,
                              0.5 * L * jnp.abs(div) / jnp.maximum(abs_b, 1e-300),
                              0.0))


# -------------------------------------------------------------
# ============= ↓ Spectral transfer / dissipation ↓ ===========
# -------------------------------------------------------------
def _fft_vec(fx, fy, fz):
    return jnp.stack([jnp.fft.fftn(fx), jnp.fft.fftn(fy), jnp.fft.fftn(fz)])


def _wavevectors(N, box=1.0):
    k1 = 2.0 * jnp.pi * jnp.fft.fftfreq(N, d=box / N)
    return k1[:, None, None], k1[None, :, None], k1[None, None, :]


def _shell_sum(field, N):
    """Shell-sum a per-mode real quantity onto the same shells as the spectra."""
    n_bins = N // 2 + 1
    shell = _shell_index(N)
    inside = shell < n_bins
    return jnp.zeros(n_bins, dtype=field.dtype).at[
        jnp.where(inside, shell, 0)
    ].add(jnp.where(inside, field.ravel(), 0.0))


def _pad_axis(hat, M, axis):
    """Zero-pad one axis of a spectrum from ``N`` to ``M`` modes.

    The Nyquist element is split in half between ``+N/2`` and ``-N/2``, which is
    what makes the interpolation exact: an unsplit Nyquist row is a cosine the
    padded transform would read as a complex exponential.
    """
    N, h = hat.shape[axis], hat.shape[axis] // 2
    shape = list(hat.shape)
    shape[axis] = M
    out = jnp.zeros(shape, dtype=hat.dtype)

    def sl(lo, hi):
        idx = [slice(None)] * hat.ndim
        idx[axis] = slice(lo, hi)
        return tuple(idx)

    out = out.at[sl(0, h)].set(hat[sl(0, h)])
    out = out.at[sl(M - h + 1, M)].set(hat[sl(h + 1, N)])
    half = 0.5 * hat[sl(h, h + 1)]
    out = out.at[sl(h, h + 1)].set(half)
    return out.at[sl(M - h, M - h + 1)].set(half)


def _refine(field, M):
    """Interpolate a periodic field from ``N^3`` onto a finer ``M^3`` grid.

    Spectral zero-padding, so the field is unchanged -- every mode it carries is
    carried exactly, and no new one is created. Used to form the nonlinear
    products on a grid fine enough that their aliases fall outside the shells
    that are kept (Orszag's 3/2 rule).
    """
    if M == field.shape[0]:
        return field
    hat = jnp.fft.fftn(field)
    for axis in range(3):
        hat = _pad_axis(hat, M, axis)
    return jnp.real(jnp.fft.ifftn(hat)) * (float(M) / float(field.shape[0])) ** 3


def box_filter(field):
    """Replace a point-value field by its cell averages on the same grid.

    Multiplying the spectrum by ``sinc(pi n_i / N)`` per axis is exactly the
    average of the trigonometric interpolant over one cell, i.e. what a
    finite-volume code would have stored. Used only by the representation
    control that asks whether the FD/FV difference, rather than the induction
    discretisation, is what separates the two codes.
    """
    N = field.shape[0]
    return jnp.real(jnp.fft.ifftn(jnp.fft.fftn(field)
                                  * _cell_average_transfer(N)))


def transfer_spectra(rho, vx, vy, vz, bx, by, bz, sound_speed, grid_spacing,
                     gamma=None, pressure=None, dealias=False):
    """Ideal (non-dissipative) rate of change of each shell's energy.

    ``T(n) = sum_shell Re(f-hat* . a-hat) / N^6`` where ``a`` is the exact
    right-hand side of the ideal equations. Combined with the *measured*
    ``dE(n)/dt`` from consecutive snapshots this isolates what the scheme threw
    away:

        ``D(n) = T(n) - dE(n)/dt``

    and hence the scale-dependent effective diffusivity ``D(n) / (2 k^2 E(n))``,
    which is the quantity that distinguishes a quasi-Laplacian 2nd-order scheme
    (roughly flat in ``k``) from a 5th-order hyper-resistive one (rising steeply
    towards the grid). Nothing here knows about either code: it is evaluated on
    the fields, so both go through it identically.

    The magnetic side is exact -- the ideal induction equation is
    ``dB/dt = curl(v x B)`` and nothing else. The kinetic side omits the
    turbulent forcing, which neither code exposes to a snapshot callback, so
    ``T_v`` is only meaningful above the driving band (``n >= 4`` here).

    Args:
        gamma, pressure: for an ideal-gas run, the adiabatic index and the
            pressure field. Omit both for an isothermal run, where
            ``grad p / rho = a^2 grad(ln rho)``.
    """
    N = rho.shape[0]
    box = grid_spacing * N
    if dealias:
        # Orszag 3/2: form every product on a grid half again as fine, so the
        # aliases of a quadratic term land above the shells that are kept.
        M = 3 * N // 2
        rho, vx, vy, vz, bx, by, bz = (_refine(f, M) for f in
                                       (rho, vx, vy, vz, bx, by, bz))
        if pressure is not None:
            pressure = _refine(pressure, M)
        N_keep, N = N, M
    else:
        N_keep = N
    n_tot = float(N) ** 3
    kx, ky, kz = _wavevectors(N, box)
    rho_safe = jnp.maximum(rho, 1e-12)

    # ---- magnetic: dB/dt = curl(v x B) ------------------------------------
    wx = vy * bz - vz * by
    wy = vz * bx - vx * bz
    wz = vx * by - vy * bx
    w_hat = _fft_vec(wx, wy, wz)
    curl = jnp.stack([
        1j * (ky * w_hat[2] - kz * w_hat[1]),
        1j * (kz * w_hat[0] - kx * w_hat[2]),
        1j * (kx * w_hat[1] - ky * w_hat[0]),
    ])
    b_hat = _fft_vec(bx, by, bz)
    T_mag = _shell_sum(
        jnp.real(jnp.sum(jnp.conj(b_hat) * curl, axis=0)) / n_tot ** 2,
        N)[:N_keep // 2 + 1]

    # ---- kinetic: dv/dt = -(v.grad)v - grad p / rho + (curl B) x B / rho ---
    def d(f, k):
        return jnp.real(jnp.fft.ifftn(1j * k * jnp.fft.fftn(f)))

    adv = []
    for c in (vx, vy, vz):
        adv.append(-(vx * d(c, kx) + vy * d(c, ky) + vz * d(c, kz)))
    if pressure is None:
        # isothermal: grad p / rho = a^2 grad(ln rho)
        lnrho = jnp.log(rho_safe)
        gradp = [sound_speed ** 2 * d(lnrho, k) for k in (kx, ky, kz)]
    else:
        gradp = [d(pressure, k) / rho_safe for k in (kx, ky, kz)]
    jx = d(bz, ky) - d(by, kz)
    jy = d(bx, kz) - d(bz, kx)
    jz = d(by, kx) - d(bx, ky)
    lorentz = [(jy * bz - jz * by) / rho_safe,
               (jz * bx - jx * bz) / rho_safe,
               (jx * by - jy * bx) / rho_safe]
    a_hat = _fft_vec(*[adv[i] - gradp[i] + lorentz[i] for i in range(3)])
    v_hat = _fft_vec(vx, vy, vz)
    T_kin = _shell_sum(
        jnp.real(jnp.sum(jnp.conj(v_hat) * a_hat, axis=0)) / n_tot ** 2,
        N)[:N_keep // 2 + 1]

    return jnp.stack([T_kin, T_mag])


#: Names of the two transfer spectra :func:`transfer_spectra` returns.
TRANSFER_NAMES = ("T_v", "T_mag")
# -------------------------------------------------------------
# ============= ↑ Spectral transfer / dissipation ↑ ===========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============ ↓ Per-snapshot reduction (shared) ↓ =============
# -------------------------------------------------------------
def snapshot_spectra(rho, vx, vy, vz, bx, by, bz, sound_speed, grid_spacing,
                     deconvolve_cell_average=False, spectra=True,
                     transfer=False, pressure=None, gamma=None,
                     dealias=False, cell_average_input=False):
    """Every per-snapshot diagnostic of the dynamo study, in one pass.

    Both codes are reduced through this function -- astronomix inside its
    snapshot callback, AthenaPK after reading a ``.phdf`` -- so the spectra and
    the scalars are defined identically for both.

    Args:
        rho, vx, vy, vz, bx, by, bz: Cell fields, each ``(N, N, N)``. The
            magnetic field is the cell-centred one in both codes (astronomix
            averages its staggered field to centres for output; AthenaPK's
            GLM-MHD field is cell-centred to begin with).
        sound_speed: Isothermal sound speed ``a``, for the Mach number.
        grid_spacing: Cell size, for the ``div B`` diagnostic.
        spectra: When False, return a zero-filled spectrum array instead of
            computing the transforms. The scalars are cheap reductions while the
            spectra are nine FFTs plus three shell scatter-adds, so this is what
            makes a high-cadence scalar time series affordable.
        transfer: Also return the two ideal transfer spectra (see
            :func:`transfer_spectra`), appended as rows 3 and 4. Doubles the
            per-snapshot cost; needed only for the dissipation-spectrum analysis.
        pressure, gamma: passed through to :func:`transfer_spectra` for an
            ideal-gas run.
        deconvolve_cell_average: See :func:`shell_spectrum`.

    Returns:
        ``(scalars, spectra)`` where ``scalars`` is a length-12 array
        ``[v_rms, b_rms, mean_absB, rho_rms, E_K, E_B, mach, mach_alfven,
        beta_plasma, rel_divB, min_rho, finite]`` and ``spectra`` is
        ``(3, N/2 + 1)`` holding ``[E_v, E_kin, E_mag]``:

            E_v   -- unweighted velocity spectrum, sum = mean(0.5 |v|^2)
            E_kin -- sqrt(rho)-weighted, sum = mean(0.5 rho |v|^2)
            E_mag -- magnetic,           sum = mean(0.5 |B|^2)
    """
    finite = jnp.all(jnp.isfinite(rho)) & jnp.all(jnp.isfinite(vx)) \
        & jnp.all(jnp.isfinite(bx))
    rho_safe = jnp.maximum(jnp.nan_to_num(rho), 1e-12)
    vx, vy, vz = (jnp.nan_to_num(c) for c in (vx, vy, vz))
    bx, by, bz = (jnp.nan_to_num(c) for c in (bx, by, bz))

    v2 = vx ** 2 + vy ** 2 + vz ** 2
    b2 = bx ** 2 + by ** 2 + bz ** 2
    v_rms = jnp.sqrt(jnp.mean(v2))
    b_rms = jnp.sqrt(jnp.mean(b2))
    mean_absB = jnp.mean(jnp.sqrt(b2))
    rho_mean = jnp.mean(rho_safe)
    rho_rms = jnp.sqrt(jnp.mean((rho_safe - rho_mean) ** 2))
    E_K = jnp.mean(0.5 * rho_safe * v2)
    E_B = jnp.mean(0.5 * b2)
    # Alfven Mach number from the volume-averaged Alfven speed, and the plasma
    # beta of the *thermal* pressure a^2 rho against the mean magnetic pressure.
    v_alfven = b_rms / jnp.sqrt(rho_mean)
    mach_alfven = v_rms / jnp.maximum(v_alfven, 1e-30)
    beta_plasma = sound_speed ** 2 * rho_mean / jnp.maximum(E_B, 1e-30)

    scalars = jnp.stack([
        v_rms, b_rms, mean_absB, rho_rms, E_K, E_B,
        v_rms / sound_speed, mach_alfven, beta_plasma,
        relative_divb(bx, by, bz, grid_spacing),
        jnp.min(rho_safe), finite.astype(rho_safe.dtype),
    ])

    n_rows = 5 if transfer else 3
    if not spectra:
        n_bins = rho.shape[0] // 2 + 1
        return scalars, jnp.zeros((n_rows, n_bins), dtype=rho.dtype)

    if cell_average_input:
        # Representation control: hand a finite-difference code's point values
        # through the same box filter a finite-volume code's storage applies.
        rho_safe = jnp.maximum(box_filter(rho_safe), 1e-12)
        vx, vy, vz, bx, by, bz = (box_filter(f) for f in (vx, vy, vz, bx, by, bz))
        if pressure is not None:
            pressure = box_filter(pressure)

    dc = deconvolve_cell_average
    spectra = jnp.stack([
        shell_spectrum(vx, vy, vz, deconvolve_cell_average=dc),
        shell_spectrum(vx, vy, vz, weight=jnp.sqrt(rho_safe),
                       deconvolve_cell_average=dc),
        shell_spectrum(bx, by, bz, deconvolve_cell_average=dc),
    ])
    if transfer:
        spectra = jnp.concatenate([spectra, transfer_spectra(
            rho_safe, vx, vy, vz, bx, by, bz, sound_speed, grid_spacing,
            gamma=gamma, pressure=pressure, dealias=dealias)])
    return scalars, spectra


#: Names of the twelve scalars :func:`snapshot_spectra` returns, in order.
SCALAR_NAMES = ("v_rms", "b_rms", "mean_absB", "rho_rms", "E_K", "E_B",
                "mach", "mach_alfven", "beta_plasma", "rel_divB",
                "min_rho", "finite")

#: Names of the spectra :func:`snapshot_spectra` returns, in order. The last two
#: are present only with ``transfer=True``.
SPECTRUM_NAMES = ("E_v", "E_kin", "E_mag", "T_v", "T_mag")
# -------------------------------------------------------------
# ============ ↑ Per-snapshot reduction (shared) ↑ =============
# -------------------------------------------------------------
