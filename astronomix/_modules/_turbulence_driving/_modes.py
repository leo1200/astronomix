"""
Driving modes: wave vectors and amplitudes, built once on the host.

Port of ``TurbGen::init_modes`` (TurbGen.h, lines 756-930; licence notice
of TurbGen in ``LICENSE_TurbGen`` in this directory). The operations are
kept in the order of the source so that the list agrees with the C++ code
to round-off.
"""

# general
import math
from functools import lru_cache
from typing import NamedTuple, Tuple

# numerics
import numpy as np

# astronomix
from astronomix._modules._turbulence_driving._turbulence_driving_options import (
    BAND,
    PARABOLA,
    POWER_LAW,
    TurbulenceDrivingConfig,
)


class DrivingModes(NamedTuple):
    """The driven Fourier modes."""

    #: Wave vectors k = 2 pi n / L, shape (num_modes, dimensionality).
    wave_vectors: np.ndarray

    #: Amplitude of every mode, shape (num_modes,).
    amplitudes: np.ndarray


class SeparableLattice(NamedTuple):
    """
    The modes on the lattice of the distinct wave numbers of every axis.
    Modes with a negative wave number along the last axis are mirrored,
    k -> -k, which leaves the real field unchanged if their coefficient is
    conjugated.
    """

    #: Per axis: the distinct wave numbers, shape (lattice size of the axis,).
    wave_numbers: Tuple[np.ndarray, ...]

    #: Per axis: lattice index of every mode, shape (num_modes,).
    index: Tuple[np.ndarray, ...]

    #: Whether a mode was mirrored, shape (num_modes,).
    mirrored: np.ndarray


def driving_wavenumber_range(config: TurbulenceDrivingConfig) -> Tuple[float, float]:
    """
    Physical wave number range of the driving (TurbGen.h, 331-332):
    kmin = (k_min - eps) 2 pi / L_x, kmax = (k_max + eps) 2 pi / L_x
    with the machine epsilon eps, so that modes on the edges are included.
    """
    epsilon = float(np.finfo(np.float64).eps)
    kmin = (config.k_min - epsilon) * 2 * math.pi / config.box_size[0]
    kmax = (config.k_max + epsilon) * 2 * math.pi / config.box_size[0]
    return kmin, kmax


@lru_cache(maxsize=None)
def build_driving_modes(config: TurbulenceDrivingConfig) -> DrivingModes:
    """The list of modes of a configuration (cached, numpy float64)."""
    if config.dimensionality not in (1, 2, 3):
        raise ValueError("turbulence driving: dimensionality must be 1, 2 or 3")
    if config.spectral_form in (BAND, PARABOLA):
        modes = _full_sampling_modes(config)
    elif config.spectral_form == POWER_LAW:
        modes = _power_law_modes(config)
    else:
        raise ValueError("turbulence driving: unknown spectral form")
    if modes.amplitudes.size == 0:
        raise ValueError("turbulence driving: no mode between k_min and k_max")
    return modes


def _full_sampling_modes(config: TurbulenceDrivingConfig) -> DrivingModes:
    """
    Band and parabola (TurbGen.h, 813-846): all k = 2 pi n / L, n integer
    (the full k-space, not a half space), with kmin <= |k| <= kmax and

        band:      A = (kc / |k|)^((ndim - 1) / 2),               kc = kmin
        parabola:  A = sqrt(|1 - 4 (|k| - kc)^2 / (kmax - kmin)^2|)
                       (kc / |k|)^((ndim - 1) / 2),               kc = (kmin + kmax) / 2

    in the order of the source (n_x outermost, n_z innermost). The parabola
    vanishes at the edges: those modes carry amplitudes of round-off size.
    """
    ndim, box_size = config.dimensionality, config.box_size
    kmin, kmax = driving_wavenumber_range(config)
    # TurbGen scans -256 <= n <= 256 per axis; no larger |n| than this limit can lie within kmax
    limits = [min(256, int(kmax * box_size[d] / (2 * math.pi)) + 1) for d in range(ndim)]
    numbers = np.meshgrid(*[np.arange(-limit, limit + 1) for limit in limits], indexing="ij")
    wave_vectors = np.stack([2 * math.pi * numbers[d].ravel() / box_size[d] for d in range(ndim)], axis=1)
    k = np.sqrt(sum(wave_vectors[:, d] * wave_vectors[:, d] for d in range(ndim)))
    driven = (k >= kmin) & (k <= kmax)
    wave_vectors, k = wave_vectors[driven], k[driven]
    if config.spectral_form == BAND:
        kc, spectrum = kmin, np.ones_like(k)
    else:
        kc = 0.5 * (kmin + kmax)
        spectrum = np.abs(-4.0 / np.square(kmax - kmin) * np.square(k - kc) + 1.0)
    amplitudes = np.sqrt(spectrum) * (kc / k) ** ((ndim - 1) / 2.0)
    return DrivingModes(wave_vectors, amplitudes)


def _power_law_modes(config: TurbulenceDrivingConfig) -> DrivingModes:
    """
    Power law (TurbGen.h, 851-925): for every integer shell kmin <= s <= kmax
    (in units of 2 pi / L_x), 2^ndim ceil(s^angles_exponent) random directions
    are drawn; each gives the lattice vector nearest to a point at radius
    s + u - 1/2 (u uniform), kept if it lies in the driving range, with

        A = sqrt((|k| / kc)^power_law_exponent s^(ndim - 1) / num_angles 4 sqrt(3))
            (kc / |k|)^((ndim - 1) / 2),     kc = kmin.

    A lattice vector drawn twice gives two modes, as in the source. The
    angles use TurbGen's generator (``_ran2``), so that the same seed gives
    the same modes as the C++ code.
    """
    ndim, box_size, exponent = config.dimensionality, config.box_size, config.power_law_exponent
    kmin, kmax = driving_wavenumber_range(config)
    kc, kmid = kmin, kmax  # kmid: start of an optional second power-law section, unused by the driving
    generator = _Ran2()
    generator.draw(-config.random_seed)  # the initialising call of the source; its deviate is not used
    seed = config.random_seed
    wave_vectors, amplitudes = [], []
    first_shell = max(1, _round_half_away(kmin * box_size[0] / (2 * math.pi)))
    last_shell = _round_half_away(kmax * box_size[0] / (2 * math.pi))
    for shell in range(first_shell, last_shell + 1):
        num_angles = int(2.0**ndim * math.ceil(float(shell) ** config.angles_exponent))
        for _ in range(num_angles):
            uniform, seed = generator.draw(seed)
            phi = 2 * math.pi * uniform
            if ndim == 1:
                phi = 0.0 if phi < math.pi else math.pi
            theta = math.pi / 2.0
            if ndim > 2:
                uniform, seed = generator.draw(seed)
                theta = math.acos(1.0 - 2.0 * uniform)
            uniform, seed = generator.draw(seed)
            radius = shell + uniform - 0.5
            point = (
                radius * math.sin(theta) * math.cos(phi),
                radius * math.sin(theta) * math.sin(phi),
                radius * math.cos(theta),
            )
            wave_vector = [2 * math.pi * _round_half_away(point[d]) / box_size[d] for d in range(ndim)]
            k = math.sqrt(sum(component * component for component in wave_vector))
            if kmin <= k <= kmax:
                spectrum = (k / kc) ** exponent
                if k >= kmid:
                    spectrum = (kmid / kmin) ** exponent * (k / kmid) ** exponent
                amplitude = math.sqrt(spectrum * shell ** (ndim - 1) / num_angles * 4.0 * math.sqrt(3.0))
                amplitudes.append(amplitude * (kc / k) ** ((ndim - 1) / 2.0))
                wave_vectors.append(wave_vector)
    return DrivingModes(np.array(wave_vectors, dtype=np.float64).reshape(-1, ndim), np.array(amplitudes))


def _round_half_away(value: float) -> int:
    """The C function ``round``: to the nearest integer, halves away from zero."""
    magnitude = math.floor(abs(value))
    magnitude += abs(value) - magnitude >= 0.5
    return -magnitude if value < 0 else magnitude


class _Ran2:
    """
    The generator ``ran2`` of TurbGen (TurbGen.h, 1071-1110; L'Ecuyer with a
    Bays-Durham shuffle, Numerical Recipes). The shuffle table lives in the
    generator and the linear congruential state ``idum`` with the caller,
    exactly as in the source. Used for the power-law mode angles only.
    """

    IM1, IM2, IA1, IA2, IQ1, IQ2, IR1, IR2, NTAB = 2147483563, 2147483399, 40014, 40692, 53668, 52774, 12211, 3791, 32

    def __init__(self):
        self.idum2, self.iy, self.table = 123456789, 0, [0] * self.NTAB

    def _advance(self, idum: int) -> int:
        k = idum // self.IQ1
        idum = self.IA1 * (idum - k * self.IQ1) - k * self.IR1
        return idum + self.IM1 if idum < 0 else idum

    def draw(self, idum: int) -> Tuple[float, int]:
        """Returns a uniform deviate in (0, 1) and the new ``idum``."""
        if idum <= 0:
            idum = max(-idum, 1)
            self.idum2 = idum
            for j in range(self.NTAB + 7, -1, -1):
                idum = self._advance(idum)
                if j < self.NTAB:
                    self.table[j] = idum
            self.iy = self.table[0]
        idum = self._advance(idum)
        k = self.idum2 // self.IQ2
        self.idum2 = self.IA2 * (self.idum2 - k * self.IQ2) - k * self.IR2
        if self.idum2 < 0:
            self.idum2 += self.IM2
        j = self.iy // (1 + (self.IM1 - 1) // self.NTAB)
        self.iy = self.table[j] - self.idum2
        self.table[j] = idum
        if self.iy < 1:
            self.iy += self.IM1 - 1
        return min((1.0 / self.IM1) * self.iy, 1.0 - 1.2e-7), idum


@lru_cache(maxsize=None)
def separable_lattice(config: TurbulenceDrivingConfig) -> SeparableLattice:
    """The modes of a configuration arranged for the separable sum (cached)."""
    wave_vectors = build_driving_modes(config).wave_vectors
    mirrored = wave_vectors[:, -1] < 0
    wave_vectors = np.where(mirrored[:, None], -wave_vectors, wave_vectors)
    axes = [np.unique(wave_vectors[:, d], return_inverse=True) for d in range(config.dimensionality)]
    wave_numbers, index = zip(*axes)
    return SeparableLattice(wave_numbers, tuple(i.reshape(-1).astype(np.int32) for i in index), mirrored)
