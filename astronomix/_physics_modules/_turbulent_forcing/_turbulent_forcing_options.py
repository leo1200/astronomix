from typing import NamedTuple

class TurbulentForcingConfig(NamedTuple):
    vacuum_protection: bool = False
    turbulent_forcing: bool = False

class TurbulentForcingParams(NamedTuple):
    protection_density_threshold: float = 0.02
    protection_max_velocity: float = 50.0
    energy_injection_rate: float = 2.0