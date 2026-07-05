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

    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()
