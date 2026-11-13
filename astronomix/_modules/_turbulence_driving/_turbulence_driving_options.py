"""
Configuration, parameters and state of the turbulence driving.

The names follow the parameter file of TurbGen (``TurbGen.par``); the TurbGen
name is given with every field.
"""

# typing
from typing import NamedTuple, Tuple, Union

# jax
import jax.numpy as jnp

# spectral forms of the driving amplitude (TurbGen ``spect_form``)
BAND = 0
PARABOLA = 1
POWER_LAW = 2


class TurbulenceDrivingConfig(NamedTuple):
    """
    Static part of the turbulence driving: everything that fixes the list of
    modes and the array shapes. The defaults are the driving of Beattie et
    al. (2025): parabola over 1 <= k <= 3, peak at k = 2 (the parameter
    ``k_driving``), natural mixture. Their run, isothermal at rms Mach 4, is

        SimulationConfig(..., turbulence_driving_config=TurbulenceDrivingConfig(turbulence_driving=True))
        SimulationParams(..., turbulence_driving_params=TurbulenceDrivingParams(velocity=4.0 * sound_speed))

    with ``amplitude_auto_adjust=True`` or a calibrated ``amplitude_factor``
    (the factor that holds Mach 4 depends on the scheme and the resolution;
    1.2 for BEATTIE_REFERENCE and FD_RIEMANN_ISOTHERMAL at 128^3).

    In a simulation, ``finalize_config`` sets ``dimensionality`` and
    ``box_size`` from the simulation configuration.
    """

    #: Drive turbulence with the Ornstein-Uhlenbeck acceleration field.
    turbulence_driving: bool = False

    #: Number of spatial dimensions, 1, 2 or 3 (``ndim``; the variants 1.5
    #: and 2.5 of TurbGen are not ported). The field has as many components.
    dimensionality: int = 3

    #: Box size per axis (``L``); the lower box edge is at the origin.
    box_size: Tuple[float, ...] = (1.0, 1.0, 1.0)

    #: Spectral form of the amplitudes: BAND, PARABOLA or POWER_LAW
    #: (``spect_form``).
    spectral_form: int = PARABOLA

    #: Smallest driven wave number in units of 2 pi / L_x (``k_min``).
    k_min: float = 1.0

    #: Largest driven wave number in units of 2 pi / L_x (``k_max``).
    k_max: float = 3.0

    #: Exponent of the power-law spectrum (``power_law_exp``; -5/3
    #: Kolmogorov, -2 Burgers). POWER_LAW only.
    power_law_exponent: float = -2.0

    #: The number of modes per shell of the power-law spectrum grows as
    #: k^angles_exponent (``angles_exp``; 2 samples k-space fully).
    angles_exponent: float = 1.0

    #: Pattern updates per turnover time (``nsteps_per_t_turb``).
    num_steps_per_turnover: int = 10

    #: Seed of the driving sequence and of the power-law mode angles
    #: (``random_seed``).
    random_seed: int = 42

    #: Adjust the amplitude towards the target velocity dispersion with the
    #: measured dispersion handed to ``advance_turbulence_driving``
    #: (``ampl_auto_adjust``); in a simulation the time loop measures it at
    #: the start of every step.
    amplitude_auto_adjust: bool = False


class TurbulenceDrivingParams(NamedTuple):
    """Differentiable parameters of the turbulence driving."""

    #: Target velocity dispersion (``velocity``). With ``k_driving`` it sets
    #: the turnover time L_x / (k_driving velocity); the acceleration scales
    #: as velocity^2.
    velocity: Union[float, jnp.ndarray] = 1.0

    #: Characteristic driving wave number in units of 2 pi / L_x
    #: (``k_driv``).
    k_driving: Union[float, jnp.ndarray] = 2.0

    #: 1 solenoidal, 0 compressive, 0.5 natural mixture (``sol_weight``).
    solenoidal_weight: Union[float, jnp.ndarray] = 0.5

    #: Amplitude factor, a scalar or one value per component
    #: (``ampl_factor``). The acceleration scales as amplitude_factor^1.5, so
    #: that the velocity dispersion reached is about proportional to it.
    amplitude_factor: Union[float, jnp.ndarray] = 1.0


class TurbulenceDrivingState(NamedTuple):
    """State of the driving carried through the time loop (and a restart)."""

    #: Ornstein-Uhlenbeck state of every mode in units of its stationary
    #: standard deviation (TurbGen ``OUphases / OUvar``), shape
    #: (num_modes, num_components, 2) with the real and imaginary part last.
    ou_phases: jnp.ndarray

    #: Number of the current pattern (TurbGen ``step``); -1 before the first.
    step: jnp.ndarray

    #: Base key of the sequence; pattern n uses ``fold_in(key, n)``.
    key: jnp.ndarray

    #: Factor per component from the automatic amplitude adjustment (1 if
    #: unused), shape (num_components,).
    amplitude_adjustment: jnp.ndarray
