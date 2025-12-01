from typing import NamedTuple

class TurbulentForcingConfig(NamedTuple):
    turbulent_forcing: bool = False

class TurbulentForcingParams(NamedTuple):
    energy_injection_rate: float = 2.0