import numpy as np
import matplotlib.pyplot as plt

resolutions = [128, 256]
configuration_names = ["fv_mhd_lax_mid", "fv_mhd_hll_mid", "fv_mhd_hll_eul", "fd_mhd"]

base_size = 3
fig, axs = plt.subplots(len(resolutions), len(configuration_names), figsize=(base_size * len(configuration_names), base_size * len(resolutions)))

min_density = 0.0
max_density = 1.5

for i, res in enumerate(resolutions):
    test_name = f"mhd_blast_test1_{res}cells"
    for j, config in enumerate(configuration_names):
        data = np.load(f"results/{config}/data/{test_name}.npz")
        final_state = data['final_state']
        density_slice = final_state[0, :, :, res // 2]
        im = axs[i, j].imshow(density_slice, vmin=min_density, vmax=max_density, cmap='jet')
        if i == 0:
            axs[i, j].set_title(f"{config}")
        # no axis ticks
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])

axs[0,0].set_ylabel(f"resolution: {resolutions[0]}³ cells", fontsize=12)
axs[1,0].set_ylabel(f"resolution: {resolutions[1]}³ cells", fontsize=12)

# add a common horizontal colorbar below all subplots
# plt.tight_layout()
# make room at the bottom for the colorbar and its label; reduce horizontal spacing
fig.subplots_adjust(bottom=0.18)
# create an axes for the colorbar in figure coordinates: [left, bottom, width, height]
# (raise it slightly and make a touch taller so label fits)
cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.035])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
# place a horizontal label just below the colorbar and ensure it's drawn at the bottom
cbar.set_label('density', rotation=0, labelpad=8, fontsize=11)
cbar.ax.xaxis.set_label_position('bottom')
plt.savefig("fv_oscillations_comparison.png", dpi=800, bbox_inches='tight')