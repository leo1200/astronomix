from autocvd import autocvd
autocvd(num_gpus = 1)

from astronomix._physics_modules._cooling.cooling_options import PIECEWISE_POWER_LAW, SIMPLE_POWER_LAW, SimplePowerLawParams


import jax.numpy as jnp

from astronomix._physics_modules._cooling._cooling import _cooling_rate, _evaluate_piecewise_power_law, _piecewise_power_law_temporal_evolution_function, _piecewise_power_law_temporal_evolution_function_inverse, _temporal_evolution_function, _temporal_evolution_function, _temporal_evolution_function_inverse

from astronomix import CodeUnits
from astropy import units as u
import astropy.constants as c
from astropy.constants import m_p

# code units
code_length = 3 * u.parsec
code_mass = 1e-3 * u.M_sun
code_velocity = 1 * u.km / u.s
code_units = CodeUnits(code_length, code_mass, code_velocity)

# cooling params
hydrogen_mass_fraction = 0.76
metal_mass_fraction = 0.02
reference_temperature = (1e8 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
# without a floor temperature, the simulations crash
floor_temperature = (2e4 * u.K * c.k_B / c.m_p).to(code_units.code_energy / code_units.code_mass).value
factor = (1e-23 * u.erg * u.cm ** 3 / u.s / c.m_p ** 2).to(code_units.code_energy * code_units.code_length ** 3 / (code_units.code_time * code_units.code_mass ** 2)).value
exponent = -0.7

cooling_curve_paramsA = SimplePowerLawParams(
    factor = factor,
    exponent = exponent,
    reference_temperature = reference_temperature
)

from astronomix._physics_modules._cooling._cooling_tables import schure_cooling

cooling_curve_paramsB = schure_cooling(code_units)

cooling_curve_params = cooling_curve_paramsB

cooling_curve_type = PIECEWISE_POWER_LAW

# print("reference temperature", reference_temperature)
# print("log10_T", log10_T)
# print("log10_Lambda", log10_Lambda)
# print("alpha", alpha)
# print("Y_table", Y_table)

import matplotlib.pyplot as plt

# plot the cooling curve
T_values = jnp.logspace(cooling_curve_paramsB.log10_T_table[0], cooling_curve_paramsB.log10_T_table[-1], 500)
Lambda_values = _cooling_rate(T_values, cooling_curve_type, cooling_curve_params)
print("Lambda_values", Lambda_values)
y_values = _temporal_evolution_function(T_values, cooling_curve_type, cooling_curve_params)
y_inv_values = _temporal_evolution_function_inverse(y_values, cooling_curve_type, cooling_curve_params)
print(y_values)
print(y_inv_values)

# plot the cooling curve and the temporal evolution function, including the inverse
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.loglog(T_values, Lambda_values)
ax1.set_xlabel("Temperature")
ax1.set_ylabel(r"Cooling rate $\Lambda$")
ax1.set_title("Cooling curve")

ax2.plot(T_values, y_values, label="y(T)")
ax2.plot(y_inv_values, y_values, "--", label="T(y) inverse")
ax2.set_xlabel("Temperature")
ax2.set_ylabel(r"Temporal evolution function $y(T)$")
ax2.set_title("Temporal evolution function and its inverse")
ax2.legend()

plt.tight_layout()
plt.savefig("cooling_curve_and_temporal_evolution_function.svg")