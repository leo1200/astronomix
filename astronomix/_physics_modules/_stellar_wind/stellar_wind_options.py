from typing import NamedTuple

# wind injection schemes
MEO = 0 # momentum and energy overwrite
EI = 1 # thermal energy injection
MEI = 2 # momentum and energy injection

class WindConfig(NamedTuple):
    stellar_wind: bool = False
    num_injection_cells: int = 10
    wind_injection_scheme: int = EI
    trace_wind_density: bool = False

class WindParams(NamedTuple):
    wind_mass_loss_rate: float = 0.0
    wind_final_velocity: float = 0.0

    # only necesarry for the MEO injection scheme
    pressure_floor: float = 100000.0