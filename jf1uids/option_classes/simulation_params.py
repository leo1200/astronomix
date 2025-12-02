from typing import NamedTuple
import jax.numpy as jnp

from jf1uids._physics_modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
)
from jf1uids._physics_modules._cooling.cooling_options import CoolingParams
from jf1uids._physics_modules._cosmic_rays.cosmic_ray_options import CosmicRayParams
from jf1uids._physics_modules._neural_net_force._neural_net_force_options import (
    NeuralNetForceParams,
)
from jf1uids._physics_modules._stellar_wind.stellar_wind_options import WindParams


class SimulationParams(NamedTuple):
    """
    Different from the simulation configuration, the simulation parameters
    do not require recompilation when changed. The simulation can be
    differentiated with respect to them.
    """

    #: The Courant-Friedrichs-Lewy number, a factor
    #: in the time step calculation.
    C_cfl: float = 0.4

    #: Gravitational constant.
    gravitational_constant: float = 1.0

    #: The adiabatic index of the gas.
    gamma: float = 5 / 3

    #: Minimum allowed density.
    #: NOTE: CURRENTLY ONLY USED IN
    #: FINITE DIFFERENCE MODE IF
    #: config.enforce_positivity IS TRUE.
    minimum_density: float = 1e-14

    #: Minimum allowed pressure.
    #: NOTE: CURRENTLY ONLY USED IN
    #: FINITE DIFFERENCE MODE IF
    #: config.enforce_positivity IS TRUE.
    minimum_pressure: float = 1e-14

    #: The maximum time step.
    dt_max: float = 0.001

    #: The final time of the simulation.
    t_end: float = 0.2

    #: Snapshot timepoints
    snapshot_timepoints: jnp.ndarray = jnp.array([0.0])

    # parameters of physics modules

    #: The parameters of the stellar wind module.
    wind_params: WindParams = WindParams()

    #: Cosmic ray parameters
    cosmic_ray_params: CosmicRayParams = CosmicRayParams()

    #: The parameters of the cooling module.
    cooling_params: CoolingParams = CoolingParams()

    #: The parameters of the neural network force module.
    neural_net_force_params: NeuralNetForceParams = NeuralNetForceParams()

    #: The parameters of the CNN MHD corrector module.
    cnn_mhd_corrector_params: CNNMHDconfig = CNNMHDconfig()

