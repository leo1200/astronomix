"""
3D Isothermal Magnetohydrodynamical turbulence can be 
characterized by the two following dimensionless numbers

- the Mach number M_s = v_rms / c_s, where c_s is the constant 
  isothermal sound speed and v_rms is the root mean square velocity
- the Alfvén Mach number M_A = v_rms / c_A, with the Alfvén 
  speed c_A = ||B|| / sqrt(\mu_0 \rho), here \mu_0 = 1

As typical values we might use
# TODO

For a turbulent simulation with constant driving amplitude
and initial density \rho = 1, these two dimensionless numbers
are practically controlled by

- the isothermal sound speed c_s
- the initial magnetic field magnitude B_0

A simple practical way to vary M_s and M_A in a controlled fashion
is to tune the driving amplitude of the simulation such 
that v_rms = 1.0. We can then approximately set M_s 
based on

- c_s = 1/M_s

and M_A based on

- B_0 = 1 / M_A(t = 0)

The actual M_s = v_rms / c_s and M_A = v_rms / c_A_rms
will differ but out main focus is covering an interesting
dynamic space.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

# numerics
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# astronomix
from astronomix._fluid_equations._equations import get_absolute_velocity
from astronomix._finite_difference._magnetic_update._constrained_transport import initialize_interface_fields
from astronomix._physics_modules._turbulent_forcing._turbulent_forcing_options import TurbulentForcingConfig, TurbulentForcingParams
from astronomix.initial_condition_generation.construct_primitive_state import construct_primitive_state
from astronomix.option_classes.simulation_config import (
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    ISOTHERMAL,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    SimulationConfig,
    finalize_config
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.time_stepping import time_integration
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.analysis_helpers.energy_spectrum import get_kinetic_energy_spectrum, get_magnetic_energy_spectrum
from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# set the sound speed based on a target Mach number
M_s_aim = 0.5
sound_speed = 1.0 / M_s_aim

# set the initial magnetic field based on a target Alfvén Mach number
M_A_aim = 10.0
B_0 = 1.0 / M_A_aim

# alternatively one might want to set the initial magnetic field
# based on a target plasma beta
# beta = P_gas / P_magnetic = (c_s^2 * rho) / (B^2 / 2) = 2 * M_A^2 / M_s^2
# -> B = sqrt(2 * rho * c_s^2 / beta) = sqrt(2 * M_s^2 / beta)
# plasma_beta_aim = 0.1
# B_0 = jnp.sqrt(2 * M_s_aim**2 / plasma_beta_aim)

# Configure the simulation
config = SimulationConfig(
    solver_mode = FINITE_DIFFERENCE,
    equation_of_state = ISOTHERMAL, # IDEAL_GAS / ISOTHERMAL
    memory_analysis = True,
    progress_bar = True,
    dimensionality = 3,
    num_cells = 256,
    enforce_positivity = False, # only necessary for high Mach turbulence
    box_size = 1.0,
    mhd = True,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
        BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        ),
    ),
    turbulent_forcing_config = TurbulentForcingConfig(
        turbulent_forcing = True,
        vacuum_protection = True, # this is only necessary for high Mach turbulence
    ),
)

# Set the simulation parameters
params = SimulationParams(
    C_cfl = 1.5,
    isothermal_sound_speed = sound_speed, # only active when equation_of_state is isothermal
    t_end = 4.0,
    turbulent_forcing_params = TurbulentForcingParams(
        energy_injection_rate = 1.65,
        protection_density_threshold = 0.02,
        protection_max_velocity = 50.0,
    ),
    minimum_density = 0.02,
)

# Initialize the registered variables
registered_variables = get_registered_variables(config)

# Initialize the simulation state
density = jnp.ones((config.num_cells, config.num_cells, config.num_cells), dtype=jnp.float32)
velocity_x = jnp.zeros_like(density)
velocity_y = jnp.zeros_like(density)
velocity_z = jnp.zeros_like(density)
magnetic_field_x = jnp.zeros_like(density)
magnetic_field_y = jnp.zeros_like(density)
magnetic_field_z = jnp.ones_like(density) * B_0
bxb, byb, bzb = initialize_interface_fields(magnetic_field_x, magnetic_field_y, magnetic_field_z)
initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=density,
    velocity_x=velocity_x,
    velocity_y=velocity_y,
    velocity_z=velocity_z,
    magnetic_field_x=magnetic_field_x,
    magnetic_field_y=magnetic_field_y,
    magnetic_field_z=magnetic_field_z,
    interface_magnetic_field_x=bxb,
    interface_magnetic_field_y=byb,
    interface_magnetic_field_z=bzb,
)

# Finalize the config
config = finalize_config(config, initial_state.shape)

# Run the simulation
final_state = time_integration(initial_state, config, params, registered_variables)

# Calculate the rms velocity and the turbulent crossing time
v = get_absolute_velocity(final_state, config, registered_variables)
v_rms = jnp.sqrt(jnp.mean(v**2))
t_cross = config.box_size.x / v_rms
print(f"RMS velocity: {v_rms:.3f}")
print(f"Turbulent crossing time: {t_cross:.3f}")

# Calculate the final Mach number
final_density = final_state[registered_variables.density_index]
if config.equation_of_state == IDEAL_GAS:
    final_pressure = final_state[registered_variables.pressure_index]
    final_sound_speed = jnp.sqrt(params.gamma * final_pressure / final_density)
    print(f"Final sound speed: {jnp.mean(final_sound_speed):.3f}")
    final_mach_number = v_rms / jnp.mean(final_sound_speed)
else:
    final_sound_speed = params.isothermal_sound_speed
    final_mach_number = v_rms / final_sound_speed
print(f"Final Mach number: {final_mach_number:.3f}")

# Print the minimum and maximum density
print(f"Final density: min={jnp.min(final_density):.3f}, max={jnp.max(final_density):.3f}")

# Calculate the final Alfvén Mach number
final_magnetic_field_x = final_state[registered_variables.magnetic_index.x]
final_magnetic_field_y = final_state[registered_variables.magnetic_index.y]
final_magnetic_field_z = final_state[registered_variables.magnetic_index.z]
final_magnetic_energy_density = 0.5 * (final_magnetic_field_x**2 + final_magnetic_field_y**2 + final_magnetic_field_z**2)
final_alfven_speed = jnp.sqrt(2 * final_magnetic_energy_density / final_density)
final_alfven_mach_number = v_rms / jnp.mean(final_alfven_speed)
print(f"Final Alfvén Mach number: {final_alfven_mach_number:.3f}")

# Plot the final density, velocity magnitude, and magnetic field magnitude slices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

z_level = config.num_cells.x // 2
# take the z_level where the maximum density is located
# z_level = jnp.argmax(jnp.max(final_density, axis=(0, 1)))
print(f"Plotting z-level: {z_level}")

# Get final velocity components
final_velocity_x = final_state[registered_variables.velocity_index.x]
final_velocity_y = final_state[registered_variables.velocity_index.y]
final_velocity_z = final_state[registered_variables.velocity_index.z]
velocity_magnitude = jnp.sqrt(final_velocity_x**2 + final_velocity_y**2 + final_velocity_z**2)

# Get magnetic field magnitude
magnetic_magnitude = jnp.sqrt(final_magnetic_field_x**2 + final_magnetic_field_y**2 + final_magnetic_field_z**2)

# Plot density
im0 = axes[0].imshow(final_density[:, :, z_level], origin="lower", cmap="viridis")
axes[0].set_title("Final Density Slice")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
divider0 = make_axes_locatable(axes[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im0, cax=cax0, label="Density")

# Plot velocity magnitude
im1 = axes[1].imshow(velocity_magnitude[:, :, z_level], origin="lower", cmap="viridis")
axes[1].set_title("Final Velocity Magnitude Slice")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
divider1 = make_axes_locatable(axes[1])
cax1 = divider1.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im1, cax=cax1, label="Velocity")

# Plot magnetic field magnitude
im2 = axes[2].imshow(magnetic_magnitude[:, :, z_level], origin="lower", cmap="viridis")
axes[2].set_title("Final Magnetic Field Magnitude Slice")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
divider2 = make_axes_locatable(axes[2])
cax2 = divider2.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im2, cax=cax2, label="Magnetic Field")

plt.tight_layout()
plt.savefig("final_state_slices.png", dpi=300)

# also plot the final kinetic and magnetic power spectra
k, kinetic_spectrum = get_kinetic_energy_spectrum(
    final_velocity_x, final_velocity_y, final_velocity_z, final_density
)
_, magnetic_spectrum = get_magnetic_energy_spectrum(
    final_magnetic_field_x, final_magnetic_field_y, final_magnetic_field_z
)
fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(k, kinetic_spectrum, label="Kinetic Energy Spectrum")
ax.loglog(k, magnetic_spectrum, label="Magnetic Energy Spectrum")
ax.set_xlabel("Wavenumber k")
ax.set_ylabel("Energy Spectrum E(k)")
ax.set_title("Final Energy Spectra")
ax.set_xlim(left = 30)
ax.set_ylim(bottom = 1e-10, top = 1e-1)
add_power_law_indicators(
    ax,
    anchor=(50, 1e-5),
    exponents=[-5/3,],
    scales=[0.5, 0.5],
    labels=[r' $k^{-5/3}$', r' $k^{-2}$'],
)
ax.legend()
plt.tight_layout()
plt.savefig("final_energy_spectra.png", dpi=300)