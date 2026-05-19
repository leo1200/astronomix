# This notebook is for testing the shock finder on a simple 1D shock tube problem.
# data from https://astronomix-mhd.web.app/notebooks/hydrodynamics/simple_example.html

# %%
import jax.numpy as jnp

# constants
from astronomix import SPHERICAL, CARTESIAN

# astronomix option structures
from astronomix import SimulationConfig
from astronomix import SimulationParams

# simulation setup
from astronomix import get_helper_data
from astronomix import finalize_config
from astronomix import get_registered_variables
from astronomix import construct_primitive_state
from astronomix.shock_finder.shock_finder_copy import (
    find_shock_zone,
    shock_criteria,
    shock_sensor,
)
# time integration, core function
from astronomix import time_integration
from astronomix.option_classes.simulation_config import DOUBLE_MINMOD, HLL, HLLC, MINMOD, OSHER, SUPERBEE


# plotting
import matplotlib.pyplot as plt

# %%
limiter = MINMOD
num_cells = 501
box_size = 1.0
config = SimulationConfig(
    geometry = CARTESIAN,
    first_order_fallback = False,
    riemann_solver = HLLC,
    limiter = limiter,
    box_size = box_size,
    num_cells = num_cells,
)
params = SimulationParams(
    t_end = 0.2, # the typical value for a shock test
)
box_size = 1.0
helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)
# setup the shock initial fluid state in terms of rho, u, p
shock_pos = 0.5
r = helper_data.geometric_centers
rho = jnp.where(r < shock_pos, 1.0, 0.125)
u = jnp.zeros_like(r)
p = jnp.where(r < shock_pos, 1.0, 0.1)

# get initial state
initial_state = construct_primitive_state(
    config = config,
    registered_variables = registered_variables,
    density = rho,
    velocity_x = u,
    gas_pressure = p,
)
config = finalize_config(config, initial_state.shape)


#%%
final_state = time_integration(initial_state, config, params, registered_variables)
rho_final = final_state[registered_variables.density_index]
u_final = final_state[registered_variables.velocity_index]
p_final = final_state[registered_variables.pressure_index]


#%%
# ── 4. helper_data ─────────────────────────────────────
helper_data = get_helper_data(config)

# %%
# ── 5. run find_shock_zone ─────────────────────────────
max_idx, left_idx, right_idx = find_shock_zone(
    final_state,
    config,
    registered_variables,
    helper_data,
)

pressure = final_state[registered_variables.pressure_index]
shock_crit = shock_criteria(
    final_state,
    config,
    registered_variables,
    helper_data,
)
sensor = shock_sensor(pressure)

print("Shock center index:", max_idx)
print("Shock region indices:", left_idx, right_idx)
print("Shock center x:", r[max_idx])
print("Shock region x:", r[left_idx], r[right_idx])
print("Number of shock-criteria cells:", shock_crit.sum())
print("Max shock sensor:", sensor.max())

plt.figure(figsize=(8, 5))
plt.plot(r, pressure, label="total pressure")
plt.plot(r, sensor / sensor.max() * pressure.max(), label="scaled shock sensor")
plt.axvline(r[max_idx], linestyle="--", label="shock center")
plt.axvline(r[left_idx], linestyle=":", label="left boundary")
plt.axvline(r[right_idx], linestyle=":", label="right boundary")
plt.xlabel("x")
plt.ylabel("pressure / scaled sensor")
plt.legend()
plt.tight_layout()
#plt.savefig("figures/run_shock_finder.svg")

# %%
