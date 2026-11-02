"""
Configuration and parameter containers for radiative cooling.

Defines the integer tags that select a cooling-curve type and a cooling method
(explicit / implicit), together with the NamedTuples that carry the parameters
of each cooling curve and the overall cooling configuration.
"""

# typing
from typing import (
    NamedTuple,
    Union,
)
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

    #: Value of Lambda at the reference temperature.
    factor: float = 1.0

    #: Power-law exponent of Lambda(T) = factor * (T / T_ref)^exponent.
    exponent: float = 1.0

    #: Reference temperature T_ref (rescaled units).
    reference_temperature: float = 1e8


class PiecewisePowerLawParams(NamedTuple):
    """
    Tabulated parameters of a piecewise power-law cooling curve.

    The tables hold, per temperature bin, the curve value and slope plus the
    Townsend temporal-evolution coefficients (``Y_table``).
    """

    #: log10 of the (rescaled) bin-edge temperatures T_k, ascending.
    log10_T_table: jnp.ndarray = jnp.array([])

    #: log10 of the (rescaled) cooling rate Lambda_k at the bin edges.
    log10_Lambda_table: jnp.ndarray = jnp.array([])

    #: Power-law slope alpha_k of each bin,
    #: Lambda(T) = Lambda_k * (T / T_k)^alpha_k for T_k <= T < T_{k+1}.
    alpha_table: jnp.ndarray = jnp.array([])

    #: Townsend (2009) temporal-evolution coefficients Y_k at the bin edges.
    Y_table: jnp.ndarray = jnp.array([])

    #: Reference temperature of the temporal evolution function (the top of
    #: the table).
    reference_temperature: float = 1e8


class CoolingNetConfig(NamedTuple):
    """Static configuration of a neural-network cooling curve."""

    #: The static part of the (equinox) network, i.e. its architecture.
    network_static: Union[PyTree, NoneType] = None


class CoolingNetParams(NamedTuple):
    """Trainable parameters of a neural-network cooling curve."""

    #: The trainable part of the (equinox) network, i.e. its weights.
    network_params: Union[PyTree, NoneType] = None


class MixingCoolingParams(NamedTuple):
    """Parameters of the simple mixing-layer cooling model (Lancaster 2026)."""

    #: Ratio of the shear time to the minimum cooling time, t_sh / t_coolmin.
    xi: float = 0.5

    #: Shear Mach number v_rel / c_s with respect to the hot medium.
    mach_number: float = 0.5

    #: Density contrast chi between the cold and the hot phase (the hot to
    #: cold temperature ratio).
    density_contrast: float = 10.0


# Union of every cooling-curve parameter container; the active variant is
# selected by the cooling-curve type tag in CoolingCurveConfig.
COOLING_CURVE_TYPE = Union[
    SimplePowerLawParams,
    PiecewisePowerLawParams,
    CoolingNetParams,
    MixingCoolingParams,
]


class CoolingCurveConfig(NamedTuple):
    """Static configuration selecting the cooling-curve model."""

    #: The cooling-curve type tag (``SIMPLE_POWER_LAW``,
    #: ``PIECEWISE_POWER_LAW``, ``NEURAL_NET_COOLING``,
    #: ``NEURAL_NET_COOLING_WITH_DENSITY`` or ``SIMPLE_MIXING_LAYER_COOLING``).
    cooling_curve_type: int = SIMPLE_POWER_LAW

    #: The network architecture of a neural-network cooling curve.
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()


class CoolingConfig(NamedTuple):
    """Top-level cooling configuration (activation, method and curve)."""

    #: Switch for radiative cooling.
    cooling: bool = False

    #: Time integration of the temperature update, ``EXPLICIT_COOLING``
    #: (forward Euler) or ``IMPLICIT_COOLING`` (backward Euler, solved by a
    #: safeguarded Newton iteration).
    cooling_method: int = IMPLICIT_COOLING

    #: The cooling-curve model.
    cooling_curve_config: CoolingCurveConfig = CoolingCurveConfig()


class CoolingParams(NamedTuple):
    """
    Runtime cooling parameters: composition, temperature floor, the
    cooling-resolution limiter, the floor handling, the per-step cap, the
    heating and the cooling curve.
    """

    #: Hydrogen mass fraction X.
    hydrogen_mass_fraction: float = 0.76

    #: Metal mass fraction Z.
    metal_mass_fraction: float = 0.02

    #: Temperature floor of the cooling update (rescaled units); see
    #: ``clamp_to_floor`` for how it is enforced.
    floor_temperature: float = 1e4

    #: Cooling-resolution limiter: the minimum number of grid cells over which
    #: the cooling length ``l_cool = c_s * t_cool`` must be resolved (0 = off).
    #: Where ``l_cool`` falls below this many cells, the cooling rate is
    #: suppressed by ``min(1, (l_cool / (resolution_limiter_alpha * dx))^2)``.
    #: An unresolved radiative shock otherwise collapses into a cell-scale cold
    #: dense layer with no pressure support, which the ram pressure then
    #: crushes without bound; suppressing the unrepresentable cooling keeps
    #: such layers at the resolved adiabatic solution, while resolved cooling
    #: regions are untouched. Applied by both the finite-difference and the
    #: finite-volume solver (not by the mixing-layer cooling model).
    resolution_limiter_alpha: float = 0.0

    #: How the temperature floor is enforced after the cooling update.
    #:
    #: ``False`` (default): a cell whose update would take it below
    #: ``floor_temperature`` keeps its original temperature, i.e. the whole
    #: update is discarded. ``True``: it is clamped onto the floor instead.
    #: Cells that start below the floor are left alone either way.
    #:
    #: Clamping is the more defensible numerics: reverting makes cooling
    #: non-monotone in dt, and a cell at 1e5 K that would cross the floor stays
    #: at 1e5 K rather than reaching 1e4 K. Clamping is also what makes the
    #: explicit update usable for stiff cooling: with the revert, a stiff
    #: forward step always overshoots the floor and is discarded, so
    #: ``EXPLICIT_COOLING`` applies no cooling at all in those cells.
    #:
    #: NOTE: The revert stays the default because it suppresses cooling
    #: exactly in the unresolved cells that would otherwise collapse, which
    #: stabilises unresolved radiative shocks under ram pressure. Enable
    #: clamping only together with a crush control (e.g. the
    #: cooling-resolution limiter or the per-step cap) that keeps such layers
    #: stable.
    clamp_to_floor: bool = False

    #: Cap on the fractional temperature drop applied in one hydro step
    #: (0 = off; 0.3 means a cell may lose at most 30% of its temperature per
    #: step). Backward Euler is unconditionally stable for the cooling ODE, but
    #: the operator splitting is not: a cell taken from its post-shock
    #: temperature to the floor in a single step loses its pressure support
    #: before the hydrodynamics ever sees an intermediate state, and the
    #: neighbours ram into it. Capping the per-step drop reaches the same
    #: equilibrium (over several steps instead of one) and, unlike a cooling
    #: CFL condition, is purely local: one crushing cell does not throttle the
    #: global time step.
    max_cooling_fraction: float = 0.0

    #: Constant volumetric ISM heating (0 = off), as the effective
    #: rescaled-temperature rate: dT~/dt = (gamma - 1) * heating_rate.
    #: Physically this represents a heating rate Gamma per particle (e.g.
    #: photoelectric heating, de/dt = n * Gamma), whose T~ rate is
    #: density-independent; the conversion from a cgs Gamma [erg/s] belongs in
    #: the problem setup. Together with a cooling curve that extends below
    #: 1e4 K (``athenak_ism_cooling``) this enables the classic two-phase
    #: thermal-instability equilibrium (Lambda = Gamma).
    heating_rate: float = 0.0

    #: Optional temperature dependence of the heating, Gamma(T) =
    #: heating_rate * f(T~), tabulated as log10 of the rescaled temperature
    #: (ascending) and the dimensionless factor f. Empty tables (default) mean
    #: f = 1, i.e. the old constant heating. Needed for TIGRESS-style
    #: photoelectric heating that switches off in ionised gas: without the
    #: cut-off, constant heating spuriously heats a hot rarefied wind
    #: (n_H Gamma ~ n_H^2 Lambda already at n_H ~ 1e-3 cm^-3, T ~ 1e6-1e7 K).
    heating_log10_T_table: jnp.ndarray = jnp.array([])

    #: The heating factor f at ``heating_log10_T_table`` (linear in log10 T~).
    heating_factor_table: jnp.ndarray = jnp.array([])

    #: The parameters of the cooling curve selected by
    #: ``CoolingCurveConfig.cooling_curve_type``.
    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()
