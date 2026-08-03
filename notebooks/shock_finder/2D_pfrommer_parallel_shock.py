# ============================================================================
# 2D Shock Finder Test — Two Parallel Rotated Shocks (blast wave in 1D)
# ============================================================================
# High pressure in the middle region, low pressure on both sides.
# This drives two shocks propagating outward in opposite directions
# along the shock normal — like a 1D blast wave, rotated by SHOCK_ANGLE.
#
# Initial conditions (three regions along the normal direction):
#   left region  (d < -1/6): rho=0.125, p=0.1   (low pressure)
#   mid  region  (-1/6<d<1/6): rho=1.0, p=1.0   (high pressure — the driver)
#   right region (d >  1/6): rho=0.125, p=0.1   (low pressure)
#
# Ground truth:
#   - two distinct shock surfaces, each perpendicular to the shock normal
#   - the shock normal has angle SHOCK_ANGLE from the x-axis
#   - shocks propagate in opposite directions → d_s points outward on each
#   - ds_x mean ≈ 0 (left shock cancels right shock), but per-shock ≈ ±cos θ
#   - Mach numbers should be roughly symmetric (same jump ratio on both sides)
# ============================================================================

#%%
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from astronomix import CARTESIAN, SimulationConfig, SimulationParams
from astronomix import get_helper_data, finalize_config
from astronomix import get_registered_variables, construct_primitive_state
from astronomix import time_integration
from astronomix.option_classes.simulation_config import HLLC, MINMOD
from astronomix._physics_modules._shock_finder.pfrommer_shock_finder import find_shocks_pfrommer

from astronomix._physics_modules._shock_finder.plot_helper import plot_shock_diagnostics_2d

# ============================================================================
# CONFIGURATION
# ============================================================================

num_cells = 128
box_size  = 1.0

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=2,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
)
params = SimulationParams(t_end=0.15)   # slightly shorter — keeps shocks separated

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, 2)  — last axis is (x, y)
geometric_centers = helper_data.geometric_centers

# helper_data.geometric_centers is a grid of nx × ny cells, where each cell contains its (x, y) coordinates.
geometry_x = geometric_centers[..., 0] # (nx, ny)
geometry_y = geometric_centers[..., 1] # (nx, ny)

# ============================================================================
# INITIAL CONDITIONS — double Sod (two outward-propagating shocks)
#
# A high-pressure driver region occupies the middle third of the domain
# along n̂ = (cos θ, sin θ) at θ = FRONT_NORMAL_ANGLE degrees:
#
#   left region  (d < -1/6): p=0.1, ρ=0.125  (ambient)
#   middle region(|d| < 1/6): p=1.0, ρ=1.0   (driver)
#   right region (d > +1/6): p=0.1, ρ=0.125  (ambient)
#
# The driver launches two shocks propagating in opposite directions along n̂.
# Both shocks should have the same Mach number and |normal| = FRONT_NORMAL_ANGLE.
# No initial velocity anywhere.
# ============================================================================
FRONT_NORMAL_ANGLE = 30.0        # angle of the vector perpendicular to the pressure discontinuity line
target_theta_rad = jnp.deg2rad(FRONT_NORMAL_ANGLE)
target_nx_hat    = jnp.cos(target_theta_rad)   # x-component of shock normal
target_ny_hat    = jnp.sin(target_theta_rad)   # y-component of shock normal

target_signed_dist = (geometry_x - 0.5) * target_nx_hat + (geometry_y - 0.5) * target_ny_hat

# three regions
in_left  = target_signed_dist < -1/6
in_right = target_signed_dist >  1/6
in_mid   = ~in_left & ~in_right

# high pressure driver in the middle → two shocks propagate outward
rho = jnp.where(in_mid, 1.0,   0.125)
p   = jnp.where(in_mid, 1.0,   0.1  )
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    gas_pressure=p,
)
config = finalize_config(config, initial_state.shape)

#%%
# RUN SIMULATION
final_state = time_integration(initial_state, config, params, registered_variables)
rho_final = final_state[registered_variables.density_index]
p_final   = final_state[registered_variables.pressure_index]


#%%
# RUN SHOCK FINDER

result = find_shocks_pfrommer(
    final_state,
    config,
    registered_variables,
    helper_data,
)

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)


# ============================================================================
# DIAGNOSTICS
# ============================================================================

print(f"=== Two Parallel Shocks ({FRONT_NORMAL_ANGLE}°) ===")
print(f"Expected shock normal direction : ({float(target_nx_hat):.3f}, {float(target_ny_hat):.3f})")
print(f"Expected shock_dir_y / shock_dir_x     : {float(jnp.tan(target_theta_rad)):.3f}")

surface_mask = result.shock_surface_cells
surface_mach = result.mach_numbers[surface_mask]

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found — shock may have left domain or is too weak")


#%%
# PLOTS

fig, axes = plot_shock_diagnostics_2d(
    p_final, rho_final, result,
    geometry_x, geometry_y,
    box_size=box_size,
    mach_vmin=1.0,
    mach_vmax=2.0,
    suptitle=f"Two Parallel Rotated Shocks ({FRONT_NORMAL_ANGLE}°) — Shock Finder Validation",
)

# ----------------------------------------------------------------------
# 6. Problem-specific panel: slice along shock normal through center
# ----------------------------------------------------------------------
t_vals   = np.linspace(-0.5, 0.5, 300)
x_sample = np.clip(0.5 + t_vals * float(target_nx_hat), geometry_x_np.min(), geometry_x_np.max())
y_sample = np.clip(0.5 + t_vals * float(target_ny_hat), geometry_y_np.min(), geometry_y_np.max())

# nearest-cell lookup using geometry
geometry_x_nearest_np = np.argmin(np.abs(geometry_x_np[:, 0:1] - x_sample[np.newaxis, :]), axis=0)
geometry_y_nearest_np = np.argmin(np.abs(geometry_y_np[0:1, :] - y_sample[:, np.newaxis]), axis=1)

p_arr    = np.array(p_final)
surf_arr = np.array(result.shock_surface_cells)
zone_arr = np.array(result.shock_zones)

p_nearest    = p_arr[geometry_x_nearest_np, geometry_y_nearest_np]
surf_nearest = surf_arr[geometry_x_nearest_np, geometry_y_nearest_np]
zone_nearest = zone_arr[geometry_x_nearest_np, geometry_y_nearest_np]

ax6 = axes[1, 2]
ax6.plot(t_vals, p_nearest, label="pressure")
ax6.fill_between(t_vals, 0, 1, where=zone_nearest,
                  alpha=0.2, color="green", label="shock zone")

first = True
for ti in t_vals[surf_nearest]:
    ax6.axvline(
        ti,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="shock surface" if first else None
    )
    first = False

ax6.set_title(f"Slice along normal (θ={FRONT_NORMAL_ANGLE}°)\nexpect 2 red lines + 2 green zones")
ax6.set_xlabel("distance along normal")
ax6.set_ylabel("P")
ax6.set_box_aspect(1)
ax6.legend(fontsize=8)

plt.show()
