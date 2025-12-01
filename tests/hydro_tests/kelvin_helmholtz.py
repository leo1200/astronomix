# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

# numerics
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# astronomix
from astronomix import SimulationConfig, get_helper_data, SimulationParams, time_integration
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix import get_registered_variables
import os
from astronomix.option_classes.simulation_config import (
    HLL, HLLC, FORWARDS, MINMOD, PERIODIC_BOUNDARY,
    BoundarySettings, BoundarySettings1D, finalize_config
)

# def run_simulation(solver_name, solver_type, num_cells):
#     """
#     Runs a single simulation for a given solver and resolution,
#     and returns the final density field.
#     """
#     print(f"🚀 Starting simulation: Solver={solver_name}, Resolution={num_cells}x{num_cells}")

#     # simulation settings
#     box_size = 1.0
    
#     # setup simulation config
#     config = SimulationConfig(
#         runtime_debugging=False,
#         first_order_fallback=False,
#         riemann_solver=solver_type,
#         progress_bar=True,
#         dimensionality=2,
#         box_size=box_size,
#         num_cells=num_cells, # Use the passed resolution
#         fixed_timestep=False,
#         differentiation_mode=FORWARDS,
#         num_timesteps = 2000,
#         boundary_settings=BoundarySettings(
#             x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
#             y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
#         ),
#         limiter=MINMOD,
#         return_snapshots = False,
#         exact_end_time=True,
#     )

#     helper_data = get_helper_data(config)
#     params = SimulationParams(t_end=2.0, C_cfl=0.4)
#     registered_variables = get_registered_variables(config)

#     # --- Initial Conditions ---
#     grid_spacing = config.box_size / config.num_cells
#     x = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)
#     y = jnp.linspace(grid_spacing / 2, config.box_size - grid_spacing / 2, config.num_cells)
#     X, Y = jnp.meshgrid(x, y, indexing="ij")

#     rho = jnp.ones_like(X)
#     u_x = 0.5 * jnp.ones_like(X)
#     u_y = 0.01 * jnp.sin(2 * jnp.pi * X)
    
#     mask = (Y > 0.25) & (Y < 0.75)
#     u_x = jnp.where(mask, -0.5, u_x)
#     rho = jnp.where(mask, 2.0, rho)
    
#     p = jnp.ones((num_cells, num_cells)) * 2.5

#     initial_state = construct_primitive_state(
#         config=config,
#         registered_variables=registered_variables,
#         density=rho,
#         velocity_x=u_x,
#         velocity_y=u_y,
#         gas_pressure=p
#     )
#     config = finalize_config(config, initial_state.shape)
    
#     # --- Run Time Integration ---
#     result = time_integration(initial_state, config, params, registered_variables)
    
#     # Return only the final density field (index 0)
#     final_density = result[0, :, :]
#     print(f"✅ Finished simulation: Solver={solver_name}, Resolution={num_cells}x{num_cells}")
#     return final_density


# =============================================================================
# Main script execution
# =============================================================================

# Define the parameters for the comparison
# Create a dictionary to map string names to the solver constants
solver_config = {
    "HLL": HLL,
    "HLLC": HLLC
}
resolutions = [256, 512, 1024, 2048]
box_size = 1.0

# Store results in a dictionary
results = {}

# # Run all simulations
# for solver_name, solver_const in solver_config.items():
#     for res in resolutions:
#         # Create a unique key using the string name and resolution
#         key = (solver_name, res)
#         results[key] = run_simulation(
#             solver_name=solver_name,
#             solver_type=solver_const,
#             num_cells=res
#         )

print("\n📊 All simulations complete. Generating plot...")

# Save the results to a file using jnp.savez

# os.makedirs("results", exist_ok=True)
save_path = "results/kelvin_helmholtz_results.npz"

# # Convert keys to strings for saving
# save_dict = {f"{solver}_{res}": jnp.array(density) for (solver, res), density in results.items()}
# jnp.savez(save_path, **save_dict)

# To reload the results later:
loaded = jnp.load(save_path)
results = {}
for key in loaded.files:
    solver, res = key.split("_")
    results[(solver, int(res))] = loaded[key]


# =============================================================================
# Plotting
# =============================================================================

# Determine global min and max for consistent color scaling
global_min = float('inf')
global_max = float('-inf')
for final_density in results.values():
    global_min = min(global_min, jnp.min(final_density))
    global_max = max(global_max, jnp.max(final_density))

# Create a LogNorm for color scaling
norm = LogNorm(vmin=global_min, vmax=global_max)

# Create the plot grid
num_rows = len(solver_config)
num_cols = len(resolutions)
fig, axes = plt.subplots(
    num_rows,
    num_cols,
    figsize=(12, 6),
    sharex=True,
    sharey=True,
    gridspec_kw={'wspace': 0.05, 'hspace': 0.2}
)

fig.suptitle('Comparison of HLL and HLLC Solvers at Different Resolutions', fontsize=16)

# Populate the grid with plots
# Iterate through the items in our solver_config dictionary
for i, (solver_name, solver_const) in enumerate(solver_config.items()):
    for j, res in enumerate(resolutions):
        ax = axes[i, j]
        # Use the correct key to retrieve the result
        key = (solver_name, res)
        final_density = results[key]

        # Use imshow to plot the density
        im = ax.imshow(
            final_density.T,
            norm=norm,
            cmap="jet",
            origin="lower",
            extent=[0, box_size, 0, box_size],
            rasterized=True
        )
        ax.set_aspect('equal', 'box')

        # Set titles for columns (only on the top row)
        if i == 0:
            ax.set_title(f"{res}x{res}", fontsize=12)

        # Set labels for rows (only on the first column) using the solver_name
        if j == 0:
            ax.set_ylabel(solver_name, fontsize=14, fontweight='bold', labelpad=15)
        
        # Turn off tick labels for a cleaner look
        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

# Add a single colorbar for the entire figure
fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.05, pad=0.02, label='Density (Log Scale)')

plt.tight_layout()
plt.savefig("figures/kelvin_helmholtz.pdf", bbox_inches='tight', dpi=300)