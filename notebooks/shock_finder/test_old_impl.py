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
from astronomix.shock_finder.shock_finder_adjust_equation import (
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
from astronomix.shock_finder.shock_finder_adjust_equation import (
    shock_criteria,
    shock_sensor,
    find_shock_zone,
    _calculate_1d_divergence,
)

# extract fields
r = helper_data.geometric_centers
pressure = final_state[registered_variables.pressure_index]
density = final_state[registered_variables.density_index]
velocity = final_state[registered_variables.velocity_index]

# ── old criteria broken out individually ───────────────
# criterion 1: ∇·v < 0
div_v = _calculate_1d_divergence(velocity, config, r)
c1_old = div_v < 0

# criterion 2: ∇T·∇ρ > 0
pseudo_temperature = pressure / density
div_T = jnp.zeros_like(pseudo_temperature)
div_T = div_T.at[1:-1].set((pseudo_temperature[2:] - pseudo_temperature[:-2]) / 2)
div_rho = jnp.zeros_like(density)
div_rho = div_rho.at[1:-1].set((density[2:] - density[:-2]) / 2)
c2_old = div_T * div_rho > 0

# criterion 3: combined shock_criteria (all three AND'd)
shock_crit_old = shock_criteria(final_state, config, registered_variables, helper_data)
sensor = shock_sensor(pressure)

# old shock zone
max_idx, left_idx, right_idx = find_shock_zone(final_state, config, registered_variables, helper_data)

# ── plot ───────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

axes[0].plot(r, pressure, label="pressure")
axes[0].plot(r, sensor / sensor.max() * pressure.max(), label="scaled shock sensor", alpha=0.6)
axes[0].axvline(r[max_idx], linestyle="--", color="black", label="shock center")
axes[0].axvline(r[left_idx], linestyle=":", color="gray", label="left/right boundary")
axes[0].axvline(r[right_idx], linestyle=":", color="gray")
axes[0].set_ylabel("pressure")
axes[0].legend()

axes[1].plot(r, c1_old.astype(float), color="tab:orange", label="criterion 1: ∇·v < 0")
axes[1].set_ylabel("True/False")
axes[1].legend()

axes[2].plot(r, c2_old.astype(float), color="tab:green", label="criterion 2: ∇T·∇ρ > 0")
axes[2].set_ylabel("True/False")
axes[2].legend()

axes[3].plot(r, shock_crit_old.astype(float), color="tab:red", label="criterion 3 (all AND'd)")
axes[3].set_ylabel("True/False")
axes[3].legend()

plt.xlabel("x")
plt.tight_layout()
plt.show()