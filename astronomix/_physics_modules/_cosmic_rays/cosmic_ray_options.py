from typing import NamedTuple

class CosmicRayConfig(NamedTuple):

    #: main switch for cosmic rays
    cosmic_rays: bool = False

    #: turn on injection of CRs at shocks
    diffusive_shock_acceleration: bool = False

class CosmicRayParams(NamedTuple):

    #: starting time of diffusive shock acceleration
    diffusive_shock_acceleration_start_time: float = 0.0

    #: efficiency of diffusive shock acceleration
    diffusive_shock_acceleration_efficiency: float = 0.1