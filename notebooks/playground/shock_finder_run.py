# %%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from astronomix.option_classes.simulation_config import (
    SimulationConfig,
    BoundarySettings1D,
    OPEN_BOUNDARY,
    CARTESIAN,
    FINITE_VOLUME,
    HLL,
    MINMOD,
    finalize_config,
)
from astronomix.shock_finder.shock_finder import (
    find_shock_zone,
    shock_criteria,
    shock_sensor,
)
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.shock_finder.shock_finder import find_shock_zone



# %%
config = SimulationConfig(
    dimensionality=1,
    geometry=CARTESIAN,
    num_cells=400,
    box_size=1.0,
    solver_mode=FINITE_VOLUME,
    riemann_solver=HLL,
    limiter=MINMOD,
    boundary_settings=BoundarySettings1D(
        left_boundary=OPEN_BOUNDARY,
        right_boundary=OPEN_BOUNDARY,
    ),
)

# ── 2. registered_variables ────────────────────────────
registered_variables = get_registered_variables(config)
# 1D finite volume gives:
# density_index  = 0
# velocity_index = 1
# pressure_index = 2
# num_vars       = 3

# ── 3. build state with a shock at cell 200 ────────────
num_cells = 400
num_vars  = registered_variables.num_vars
dummy_state = jnp.zeros((num_vars, num_cells))
config = finalize_config(config, dummy_state.shape)

# ── more interesting state ─────────────────────────────
x_np = np.linspace(0, 1, num_cells)

# smooth shock using tanh (more realistic than hard step)
shock_pos   = 0.5
shock_width = 0.015

def smooth_step(x, pos, width):
    return 0.5 * (1 - np.tanh((x - pos) / width))

np.random.seed(42)
noise = np.random.randn(num_cells)

# pre-shock: high density/pressure, post-shock: low
# add turbulent noise on the left (pre-shock) side
left_mask = x_np < shock_pos

density_np  = 1.0 + 2.5 * smooth_step(x_np, shock_pos, shock_width)
density_np += 0.08 * noise * left_mask           # turbulence pre-shock

velocity_np  = 2.0 * smooth_step(x_np, shock_pos, shock_width)
velocity_np += 0.05 * noise * left_mask

pressure_np  = 1.0 + 4.0 * smooth_step(x_np, shock_pos, shock_width)
pressure_np += 0.1 * noise * left_mask

state = dummy_state
state = state.at[registered_variables.density_index].set(jnp.array(density_np))
state = state.at[registered_variables.velocity_index].set(jnp.array(velocity_np))
state = state.at[registered_variables.pressure_index].set(jnp.array(pressure_np))

# ── 4. helper_data ─────────────────────────────────────
helper_data = get_helper_data(config)
# for 1D Cartesian this builds:
# geometric_centers = cell center positions from 0 to 1
# cell_volumes      = grid_spacing per cell
# inner/outer_cell_boundaries

# %%
# ── 5. run find_shock_zone ─────────────────────────────
max_idx, left_idx, right_idx = find_shock_zone(
    state,
    config,
    registered_variables,
    helper_data,
)

# %%
print(f"Shock center : cell {max_idx}")
print(f"Shock zone   : cells {left_idx} to {right_idx}")

# %%
x = jnp.linspace(0, config.box_size.x, num_cells)

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
fig.suptitle("Shock Finder Result", fontsize=14)

axes[0].plot(x, state[registered_variables.density_index],  color="steelblue",  lw=1.5)
axes[1].plot(x, state[registered_variables.velocity_index], color="darkorange", lw=1.5)
axes[2].plot(x, state[registered_variables.pressure_index], color="seagreen",   lw=1.5)

axes[0].set_ylabel("Density")
axes[1].set_ylabel("Velocity")
axes[2].set_ylabel("Pressure")
axes[2].set_xlabel("x")

# ── mark shock from find_shock_zone output ────────────
for ax in axes:
    ax.axvspan(x[left_idx], x[right_idx], alpha=0.2, color="red")
    ax.axvline(x[max_idx], color="red", lw=1.5, ls="--")

zone_patch  = mpatches.Patch(color="red", alpha=0.2, label="shock zone")
center_line = plt.Line2D([0], [0], color="red", lw=1.5, ls="--", label="shock center")
axes[0].legend(handles=[zone_patch, center_line], loc="upper right")

plt.tight_layout()
plt.show()
# %%
