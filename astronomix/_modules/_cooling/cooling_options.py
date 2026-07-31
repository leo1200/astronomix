"""
Configuration and parameter containers for radiative cooling.

Defines the integer tags that select a cooling-curve type and a cooling method
(explicit / implicit), together with the NamedTuples that carry the parameters
of each cooling curve and the overall cooling configuration.
"""

# typing
from typing import NamedTuple, Union
from types import NoneType
from jaxtyping import PyTree

# jax
import jax.numpy as jnp

# Cooling-curve type tags (select which Lambda(T) model is used).
SIMPLE_POWER_LAW = 1
PIECEWISE_POWER_LAW = 2
NEURAL_NET_COOLING = 3
NEURAL_NET_COOLING_WITH_DENSITY = 4
SIMPLE_MIXING_LAYER_COOLING = 5

# Cooling-method tags (how the temperature update is integrated in time).
EXPLICIT_COOLING = 1
IMPLICIT_COOLING = 2


class SimplePowerLawParams(NamedTuple):
    """Parameters of a single power-law cooling curve Lambda(T)."""

    factor: float = 1.0
    exponent: float = 1.0
    reference_temperature: float = 1e8


class PiecewisePowerLawParams(NamedTuple):
    """Tabulated parameters of a piecewise power-law cooling curve.

    The tables hold, per temperature bin, the curve value and slope plus the
    Townsend temporal-evolution coefficients (``Y_table``).
    """

    log10_T_table: jnp.ndarray = jnp.array([])
    log10_Lambda_table: jnp.ndarray = jnp.array([])
    alpha_table: jnp.ndarray = jnp.array([])
    Y_table: jnp.ndarray = jnp.array([])
    reference_temperature: float = 1e8


class CoolingNetConfig(NamedTuple):
    """Static configuration of a neural-network cooling curve."""

    network_static: Union[PyTree, NoneType] = None


class CoolingNetParams(NamedTuple):
    """Trainable parameters of a neural-network cooling curve."""

    network_params: Union[PyTree, NoneType] = None


class MixingCoolingParams(NamedTuple):
    """Parameters of the simple mixing-layer cooling model (Lancaster 2026)."""

    xi: float = 0.5  # xi = t_sh / t_coolmin
    mach_number: float = 0.5
    density_contrast: float = 10.0


# Union of every cooling-curve parameter container; the active variant is
# selected by the cooling-curve type tag in CoolingCurveConfig.
COOLING_CURVE_TYPE = Union[SimplePowerLawParams, PiecewisePowerLawParams, CoolingNetParams, MixingCoolingParams]


class CoolingCurveConfig(NamedTuple):
    """Static configuration selecting the cooling-curve model."""

    cooling_curve_type: int = SIMPLE_POWER_LAW

    #: In case of neural the cooling the network architecture
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()


class CoolingConfig(NamedTuple):
    """Top-level cooling configuration (activation, method and curve)."""

    cooling: bool = False
    cooling_method: int = IMPLICIT_COOLING
    cooling_curve_config: CoolingCurveConfig = CoolingCurveConfig()


class CoolingParams(NamedTuple):
    """Runtime cooling parameters (composition, temperature floor, curve)."""

    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02

    floor_temperature: float = 1e4

    #: Cooling-resolution limiter (0 = off). Where the cooling length
    #: ``l_cool = c_s * t_cool`` is unresolved (below ``alpha`` grid cells) the
    #: cooling rate is suppressed by ``min(1, (l_cool / (alpha*dx))^2)``. An
    #: unresolved radiative shock otherwise collapses into a cell-scale cold
    #: dense layer with no pressure support (the ram-pressure crush runaway
    #: that killed the 512^3 SNR runs); suppressing the un-representable
    #: cooling keeps such layers at the resolved adiabatic solution, while
    #: resolved cooling regions are untouched.
    resolution_limiter_alpha: float = 0.0

    #: How the temperature floor is enforced after the cooling update.
    #:
    #: ``False`` (default, historical): a cell whose update would take it below
    #: ``floor_temperature`` keeps its ORIGINAL temperature -- the whole update
    #: is discarded. ``True``: it is clamped ONTO the floor instead (cells that
    #: START below it are left alone either way).
    #:
    #: Clamping is the more defensible numerics -- reverting makes cooling
    #: non-monotone in dt, and a cell at 1e5 K that wants to cross the floor
    #: stays at 1e5 K rather than reaching 1e4 K. It is also what makes the
    #: EXPLICIT path usable at all: with the revert, a stiff forward step always
    #: overshoots the floor and is always discarded, so ``EXPLICIT_COOLING``
    #: silently applies NO cooling (the 256^3 Cas A run reproduced the ADIABATIC
    #: solution to four significant figures).
    #:
    #: It defaults to False anyway, because the revert turns out to be
    #: LOAD-BEARING for stability: it is a de-facto crush guard that suppresses
    #: cooling exactly in the unresolved cells that would otherwise run away.
    #: Switching it on makes the Cas A piston runs abort (with the wide LLF
    #: blend) or blow up silently (with the per-step cap). Turn it on only
    #: together with a crush control that has been shown to hold.
    clamp_to_floor: bool = False

    #: Cap on the FRACTIONAL temperature drop applied in one hydro step
    #: (0 = off; 0.3 means a cell may lose at most 30% of its temperature per
    #: step). Backward Euler is unconditionally stable for the cooling ODE, but
    #: the operator SPLITTING is not: a cell taken from its post-shock
    #: temperature to the floor in a single step loses its pressure support
    #: before the hydro ever sees an intermediate state, and the neighbours ram
    #: into it -- the crush runaway that aborts the Cas A piston runs. Capping
    #: the per-step drop reaches the SAME equilibrium (over several steps
    #: instead of one) and, unlike a cooling CFL, is purely local: one crushing
    #: cell does not throttle the global time step.
    max_cooling_fraction: float = 0.0

    #: Constant volumetric ISM heating (0 = off), as the EFFECTIVE rescaled-
    #: temperature rate: dT~/dt = (gamma - 1) * heating_rate. Physically this
    #: represents Gamma per particle (e.g. photoelectric heating,
    #: de/dt = n * Gamma), whose T~ rate is density-independent; the
    #: conversion from a cgs Gamma [erg/s] belongs in the setup helper (see
    #: the showcase ``ism_ti_cooling_setup``). Together with a cooling curve
    #: that extends below 1e4 K (``athenak_ism_cooling``) this enables the
    #: classic two-phase thermal-instability equilibrium (Lambda = Gamma).
    heating_rate: float = 0.0

    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()
