# TODO: fix units

from autocvd import autocvd

from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
autocvd(num_gpus = 1)

from astronomix._physics_modules._cooling._cooling import get_pressure_from_temperature, get_temperature_from_pressure
from astronomix._physics_modules._cooling._cooling_tables import schure_cooling

import jax.numpy as jnp

# timing
from timeit import default_timer as timer

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# fluids
from astronomix import WindParams
from astronomix import SimulationConfig
from astronomix import SimulationParams
from astronomix.option_classes import WindConfig
from astronomix._physics_modules._cooling.cooling_options import NEURAL_NET_COOLING, PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, CoolingConfig, CoolingCurveConfig, CoolingNetConfig, CoolingNetParams, CoolingParams, PiecewisePowerLawParams, SimplePowerLawParams

from astronomix import get_helper_data
from astronomix._fluid_equations._equations import conserved_state_from_primitive, primitive_state_from_conserved
from astronomix import get_registered_variables
from astronomix.option_classes.simulation_config import BACKWARDS, finalize_config

import pickle

from astronomix import time_integration

# astronomix constants
from astronomix.option_classes.simulation_config import OPEN_BOUNDARY, REFLECTIVE_BOUNDARY, SPHERICAL

# units
from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# wind-specific
from astronomix._physics_modules._stellar_wind.weaver import Weaver


import equinox as eqx
import jax
import optax
from matplotlib.gridspec import GridSpec

# code units
code_length = 10e18 * u.cm
code_mass = 1e-3 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# general cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
# without a floor temperature, the simulations crash
floor_temperature = (1e2 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value


class CoolingNet(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(
        self,
        hidden_features: int,
        *,
        key
    ):
        self.mlp = eqx.nn.MLP(
            in_size=1,       # input: log10_T
            out_size=1,      # output: log10_Lambda
            width_size=hidden_features,
            depth=3,
            key=key,
        )
        
    def __call__(self, x):
        return self.mlp(x)
        

# initialize cooling corrector network
key = jax.random.PRNGKey(42)
cooling_corrector_network = CoolingNet(
    hidden_features = 256,
    key = key
)
cooling_corrector_params, cooling_corrector_static = eqx.partition(cooling_corrector_network, eqx.is_array)

# pre-train the network to log10_Lambda_table(log10_T_table)
# from the schure cooling table
schure_cooling_params = schure_cooling(code_units)
log10_T_table = schure_cooling_params.log10_T_table
log10_Lambda_table = schure_cooling_params.log10_Lambda_table
log10_T_table = log10_T_table[:, None]
log10_Lambda_table = log10_Lambda_table[:, None]

def temp_unitful(log10_T_code):
    return (((10 ** log10_T_code) * code_units.code_energy / code_units.code_mass * (c.m_p / c.k_B)).to(u.K)).value

def lambda_unitful(log10_lambda_code):
    # revert    Lambda = jnp.array(
    #     (
    #         Lambda * u.erg * u.cm ** 3 / u.s / c.m_p ** 2
    #     ).to(
    #         code_units.code_energy * code_units.code_length ** 3 / (code_units.code_time * code_units.code_mass ** 2)
    # ).value)

    # log10_Lambda = jnp.log10(Lambda)

    return ((10 ** log10_lambda_code * code_units.code_energy * code_units.code_length ** 3 / (code_units.code_time * code_units.code_mass ** 2)) * c.m_p ** 2).to(u.erg * u.cm ** 3 / u.s).value



corrector_resolutions = [500, 1000, 2000]
reference_resolutions = [10000, 10000, 10000]

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(temp_unitful(log10_T_table), lambda_unitful(log10_Lambda_table), label="Schure et al., 2009", color="blue")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("temperature in K")
ax.set_ylabel("cooling function Λ(T)\nin erg cm³ / s")
ax.set_title("cooling function comparison")

for low_res, high_res in zip(corrector_resolutions, reference_resolutions):
    with open("models/cooling_corrector" + str(high_res) + "_" + str(low_res) +  ".pkl", "rb") as f:
        cooling_corrector_params = pickle.load(f)
    cooling_corrector_network = eqx.combine(cooling_corrector_params, cooling_corrector_static)
    Lambda_net_after_training = jax.vmap(cooling_corrector_network)(log10_T_table)
    ax.plot(temp_unitful(log10_T_table), lambda_unitful(Lambda_net_after_training), label="learned cooling function, N = " + str(low_res), linestyle="--")

ax.legend()
plt.tight_layout()
plt.savefig("figures/cooling_function_comparison.svg")