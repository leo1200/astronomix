from types import NoneType
from typing import NamedTuple
from typing import Union
import jax.numpy as jnp
from jaxtyping import PyTree

SIMPLE_POWER_LAW = 1
PIECEWISE_POWER_LAW = 2
NEURAL_NET_COOLING = 3
NEURAL_NET_COOLING_WITH_DENSITY = 4

EXPLICIT_COOLING = 1
IMPLICIT_COOLING = 2


class SimplePowerLawParams(NamedTuple):
    factor: float = 1.0
    exponent: float = 1.0
    reference_temperature: float = 1e8

class PiecewisePowerLawParams(NamedTuple):
    log10_T_table: jnp.ndarray = jnp.array([])
    log10_Lambda_table: jnp.ndarray = jnp.array([])
    alpha_table: jnp.ndarray = jnp.array([])
    Y_table: jnp.ndarray = jnp.array([])
    reference_temperature: float = 1e8

class CoolingNetConfig(NamedTuple):
    network_static: Union[PyTree, NoneType] = None

class CoolingNetParams(NamedTuple):
    network_params: Union[PyTree, NoneType] = None

COOLING_CURVE_TYPE = Union[SimplePowerLawParams, PiecewisePowerLawParams, CoolingNetParams]

class CoolingCurveConfig(NamedTuple):
    cooling_curve_type: int = SIMPLE_POWER_LAW
    
    #: In case of neural the cooling the network architecture
    cooling_net_config: CoolingNetConfig = CoolingNetConfig()


class CoolingConfig(NamedTuple):
    cooling: bool = False
    cooling_method: int = IMPLICIT_COOLING
    cooling_curve_config: CoolingCurveConfig = CoolingCurveConfig()

class CoolingParams(NamedTuple):
    # NOTE: CURRENTLY ONLY POWER LAW COOLING
    hydrogen_mass_fraction: float = 0.76
    metal_mass_fraction: float = 0.02

    floor_temperature: float = 1e4

    cooling_curve_params: COOLING_CURVE_TYPE = SimplePowerLawParams()