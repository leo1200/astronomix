# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import jax
import numpy as np
from astronomix._modules._cooling._simple_mixing_cooling import _cooling_rate
from trml import P0, T_cold, T_hot, registered_variables, mixing_cooling_params, params, t_coolmin, rho_cold, rho_hot, initial_state, config
from astronomix._geometry.boundaries import _boundary_handler
from astronomix.time_stepping._utils import _pad

import matplotlib.pyplot as plt

# load trml_final_state.npz
data = np.load("data/trml_final_state.npz")
final_state = data["final_state"]

# pad the final state and apply boundary handler
# final_state = _pad(final_state, config)
# final_state = _boundary_handler(final_state, config, registered_variables, params)

# retrieve the final temperature
final_temperature = (
    final_state[registered_variables.pressure_index] / final_state[registered_variables.density_index]
)
final_density = final_state[registered_variables.density_index]

# retrieve the initial temperature
initial_temperature = (
    initial_state[registered_variables.pressure_index] / initial_state[registered_variables.density_index]
)
initial_density = initial_state[registered_variables.density_index]

# plot the temperature PDF
fig_pdf, ax_pdf = plt.subplots(figsize=(6, 4))
temp_flat = final_temperature.flatten()
rho_flat = final_density.flatten()
num_bins = 400

# just changes where the bins are placed not the 
# the bin widths will still be in linear space
temperature_bins = np.logspace(np.log10(T_cold), np.log10(T_hot), num_bins)

# produces dV/dT
ax_pdf.hist(
    temp_flat,
    bins=temperature_bins,
    # weights=rho_flat,
    density=True,
    log=True,
    histtype=u'step',
    linewidth=2,
    label=f"astronomix ({final_state.shape[1]}^3)",
)

# also plot Lachlans PDF
(
    log10_T_T_hot,
    mean_diff,
    mean_comp,
    mean_cool,
    sigm_diff,
    sigm_comp,
    sigm_cool,
    dV_dlogT
) = np.load("data/lancaster/phase_N64_M1_2_chi1e+02_xi3e+00.npy").T

dV_dT = dV_dlogT / (10**log10_T_T_hot * T_hot * np.log(10))
dV_dT = dV_dT / np.trapezoid(dV_dT, 10**log10_T_T_hot * T_hot)  # normalize the PDF

ax_pdf.plot(
    10**log10_T_T_hot * T_hot,
    dV_dT,
    label="Lancaster (64^3)",
    linestyle="--",
)

(
    log10_T_T_hot,
    mean_diff,
    mean_comp,
    mean_cool,
    sigm_diff,
    sigm_comp,
    sigm_cool,
    dV_dlogT
) = np.load("data/lancaster/phase_N512_M1_2_chi1e+02_xi3e+00.npy").T

dV_dT = dV_dlogT / (10**log10_T_T_hot * T_hot * np.log(10))
dV_dT = dV_dT / np.trapezoid(dV_dT, 10**log10_T_T_hot * T_hot)  # normalize the PDF

ax_pdf.plot(
    10**log10_T_T_hot * T_hot,
    dV_dT,
    label="Lancaster (512^3)",
    linestyle="--",
)

ax_pdf.legend()

ax_pdf.set_xlabel("Temperature")
ax_pdf.set_xscale("log")
ax_pdf.set_xlim(T_cold, T_hot)
ax_pdf.set_ylim(bottom = 1e-1)
ax_pdf.set_ylabel(r"$dV/dT$")
ax_pdf.set_title("Temperature PDF at Final State")
fig_pdf.tight_layout()
fig_pdf.savefig("figures/trml_temperature_pdf.svg")

# Plot the cooling function over the temperature range
# T = P / rho
isobaric_density_bins = P0 / temperature_bins
cooling_rate = _cooling_rate(temperature_bins, isobaric_density_bins, mixing_cooling_params, params.gamma)
fig_cooling, ax_cooling = plt.subplots(figsize=(6, 4))
ax_cooling.plot(
    temperature_bins,
    -cooling_rate * isobaric_density_bins * t_coolmin,
)
ax_cooling.set_xlabel("Temperature")
ax_cooling.set_xscale("log")
ax_cooling.set_yscale("log")
ax_cooling.set_xlim(T_cold, T_hot)
ax_cooling.set_ylim(bottom=1e-3)
ax_cooling.set_ylabel(r'$(\dot{\epsilon}_{\mathrm{cool}} - \dot{\epsilon}_{\mathrm{heat}})\, t_{\mathrm{cool,min}}$')
ax_cooling.set_title("Cooling Rate vs Temperature")
fig_cooling.tight_layout()
fig_cooling.savefig("figures/trml_cooling_rate.svg")

# Plot initial (top row) and final (bottom row) temperature, velocity and density
# 2D slices along the z-axis at y = 0.5.
fig_slices, axes = plt.subplots(2, 3, figsize=(12, 8))
y_slice_index_initial = initial_state.shape[2] // 2
y_slice_index_final = final_state.shape[2] // 2

im0 = axes[0, 0].imshow(initial_temperature[:, y_slice_index_initial, :].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", norm = LogNorm(vmin = T_cold, vmax = T_hot))
axes[0, 0].set_title("Initial Temperature Slice at y = 0.5")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("z")
cax = make_axes_locatable(axes[0, 0]).append_axes("right", size="5%", pad=0.1)
fig_slices.colorbar(im0, cax=cax, label="Temperature")

im1 = axes[0, 1].imshow(initial_state[registered_variables.velocity_index.z, :, y_slice_index_initial, :].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
axes[0, 1].set_title("Initial V_z Slice at y = 0.5")
axes[0, 1].set_xlabel("x")
axes[0, 1].set_ylabel("z")
cax = make_axes_locatable(axes[0, 1]).append_axes("right", size="5%", pad=0.1)
fig_slices.colorbar(im1, cax=cax, label="Velocity")

im2 = axes[0, 2].imshow(initial_density[:, y_slice_index_initial, :].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", norm = LogNorm(vmin = rho_hot, vmax = rho_cold))
axes[0, 2].set_title("Initial Density Slice at y = 0.5")
axes[0, 2].set_xlabel("x")
axes[0, 2].set_ylabel("z")
cax = make_axes_locatable(axes[0, 2]).append_axes("right", size="5%", pad=0.1)
fig_slices.colorbar(im2, cax=cax, label="Density")

im3 = axes[1, 0].imshow(final_temperature[:, y_slice_index_final, :].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", norm = LogNorm(vmin = T_cold, vmax = T_hot))
axes[1, 0].set_title("Final Temperature Slice at y = 0.5")
axes[1, 0].set_xlabel("x")
axes[1, 0].set_ylabel("z")
cax = make_axes_locatable(axes[1, 0]).append_axes("right", size="5%", pad=0.1)
fig_slices.colorbar(im3, cax=cax, label="Temperature")

im4 = axes[1, 1].imshow(final_state[registered_variables.velocity_index.z, :, y_slice_index_final, :].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto")
axes[1, 1].set_title("Final V_z Slice at y = 0.5")
axes[1, 1].set_xlabel("x")
axes[1, 1].set_ylabel("z")
cax = make_axes_locatable(axes[1, 1]).append_axes("right", size="5%", pad=0.1)
fig_slices.colorbar(im4, cax=cax, label="Velocity")

im5 = axes[1, 2].imshow(final_density[:, y_slice_index_final, :].T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", norm = LogNorm(vmin = rho_hot, vmax = rho_cold))
axes[1, 2].set_title("Final Density Slice at y = 0.5")
axes[1, 2].set_xlabel("x")
axes[1, 2].set_ylabel("z")
cax = make_axes_locatable(axes[1, 2]).append_axes("right", size="5%", pad=0.1)
fig_slices.colorbar(im5, cax=cax, label="Density")
fig_slices.tight_layout()

fig_slices.savefig("figures/trml_final_slices.png", dpi=500)

# plot of the averages along the z-axis, averaging over x and y
fig_avg, axes_avg = plt.subplots(1, 3, figsize=(12, 4))
z_initial = np.linspace(0, 1.5, initial_state.shape[3])
z_final = np.linspace(0, 1.5, final_state.shape[3])
axes_avg[0].plot(z_initial, initial_temperature.mean(axis=(0, 1)), label="Initial")
axes_avg[0].plot(z_final, final_temperature.mean(axis=(0, 1)), label="Final")
axes_avg[0].set_xlabel("z")
axes_avg[0].set_ylabel("Average Temperature")
axes_avg[0].set_title("Average Temperature vs z")
axes_avg[0].set_yscale("log")
axes_avg[0].legend()

axes_avg[1].plot(z_initial, initial_state[registered_variables.velocity_index.z, :, :, :].mean(axis=(0, 1)), label="Initial")
axes_avg[1].plot(z_final, final_state[registered_variables.velocity_index.z, :, :, :].mean(axis=(0, 1)), label="Final")
axes_avg[1].set_xlabel("z")
axes_avg[1].set_ylabel("Average V_z")
axes_avg[1].set_title("Average V_z vs z")
axes_avg[1].legend()

axes_avg[2].plot(z_initial, initial_density.mean(axis=(0, 1)), label="Initial")
axes_avg[2].plot(z_final, final_density.mean(axis=(0, 1)), label="Final")
axes_avg[2].set_xlabel("z")
axes_avg[2].set_ylabel("Average Density")
axes_avg[2].set_title("Average Density vs z")
axes_avg[2].set_yscale("log")
axes_avg[2].legend()
fig_avg.tight_layout()
fig_avg.savefig("figures/trml_final_averages.svg")