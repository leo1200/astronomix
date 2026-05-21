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
from astronomix.shock_finder.shock_finder_own import find_shocks_pfrommer
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
    t_end = 0.2,
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

final_state = time_integration(initial_state, config, params, registered_variables)
rho_final = final_state[registered_variables.density_index]
u_final   = final_state[registered_variables.velocity_index]
p_final   = final_state[registered_variables.pressure_index]
pressure  = final_state[registered_variables.pressure_index]

# %%
# start finding shocks — API unchanged, shock_direction computed internally
new_result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

# %%
print("final_state:", type(final_state))
print("config:", type(config))
print("registered_variables:", type(registered_variables))
print("helper_data:", type(helper_data))
print(new_result.shock_surface_cells.dtype)   # should be bool
print(new_result.shock_zones.dtype)           # should be bool
print(new_result.shock_ids.dtype)             # should be int32
print(new_result.shock_zone_ids.dtype)        # should be int32
print(new_result.shock_direction.dtype)       # should be float32
print(new_result.mach_numbers.dtype)          # should be float32

# %%
new_surface_idx = jnp.where(new_result.shock_surface_cells)[0]
new_zone_idx    = jnp.where(new_result.shock_zones)[0]

print("New shock surface indices:", new_surface_idx)
print("New shock zone cell count:", new_zone_idx.size)
print("New shock surface count:",   jnp.sum(new_result.shock_surface_cells))
print("New Mach numbers at surface:", new_result.mach_numbers[new_result.shock_surface_cells])

plt.figure(figsize=(10, 5))
plt.plot(r, pressure, label="total pressure")

# only show shock direction where it is meaningful (inside shock zones)
shock_dir_masked = jnp.where(new_result.shock_zones, new_result.shock_direction, jnp.nan)
plt.plot(r, shock_dir_masked, label="shock direction")

if new_surface_idx.size > 0:
    for idx in new_surface_idx:
        plt.axvline(
            r[idx], linestyle="-.", color="green",
            label="new shock surface" if idx == new_surface_idx[0] else None
        )

if new_zone_idx.size > 0:
    plt.axvspan(r[new_zone_idx[0]], r[new_zone_idx[-1]], alpha=0.12, color="green", label="new shock zone")

plt.xlabel("x")
plt.ylabel("pressure / scaled sensor")
plt.legend()
plt.tight_layout()

# %%
from astronomix.shock_finder.shock_finder_own import (
    _calculate_shock_direction,
    _shock_zone_criterion_converging_flow,
    _shock_zone_criterion_aligned_gradients,
    _shock_zone_criterion_minimum_mach,
)

# extract fields
r        = helper_data.geometric_centers
pressure = final_state[registered_variables.pressure_index]
density  = final_state[registered_variables.density_index]
velocity = final_state[registered_variables.velocity_index]

# shock_direction must be computed before criterion 3
shock_direction = _calculate_shock_direction(pressure, density, config)

c1 = _shock_zone_criterion_converging_flow(velocity, config)
c2 = _shock_zone_criterion_aligned_gradients(pressure, density, config)
c3 = _shock_zone_criterion_minimum_mach(     # ← shock_direction now required
    final_state,
    config,
    registered_variables,
    helper_data,
    shock_direction,                          # ← new argument
)

fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
axes[0].plot(r, pressure, label="pressure")
axes[0].set_ylabel("pressure")
axes[0].legend()
axes[1].plot(r, c1.astype(float), color="tab:orange", label="criterion 1: ∇·v < 0")
axes[1].set_ylabel("True/False")
axes[1].legend()
axes[2].plot(r, c2.astype(float), color="tab:green", label="criterion 2: ∇T·∇ρ > 0")
axes[2].set_ylabel("True/False")
axes[2].legend()
axes[3].plot(r, c3.astype(float), color="tab:red", label="criterion 3: M > 1.3")
axes[3].set_ylabel("True/False")
axes[3].legend()
plt.xlabel("x")
plt.tight_layout()
plt.show()

# %%
# shock stats
print("Shock surface x position:", r[new_surface_idx])
print("Mach number at shock:",     new_result.mach_numbers[new_result.shock_surface_cells])
print("Number of shock zone cells:", new_zone_idx.size)

# %%
# 4-panel fluid plot
entropy = p_final / rho_final ** (5/3)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(r, rho_final)
axes[0, 0].set_title("density")
axes[0, 0].set_ylabel("ρ")

axes[0, 1].plot(r, u_final)
axes[0, 1].set_title("velocity")
axes[0, 1].set_ylabel("v_x")

axes[1, 0].plot(r, entropy)
axes[1, 0].set_title("entropy")
axes[1, 0].set_ylabel("P/ρ^γ")

axes[1, 1].plot(r, p_final)
axes[1, 1].set_title("pressure")
axes[1, 1].set_ylabel("P")

# mark shock surface on all panels
for ax in axes.flat:
    for idx in new_surface_idx:
        ax.axvline(r[idx], linestyle="--", color="red", label="shock")
    ax.set_xlabel("x")

axes[0, 0].legend()
plt.tight_layout()
plt.show()

# %%