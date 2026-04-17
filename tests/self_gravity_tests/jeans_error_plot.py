import numpy as np
import matplotlib.pyplot as plt

from astronomix.plotting_helpers.power_law_indicators import add_power_law_indicators

# =======================
# Example usage (your case)
# =======================

data = np.load("jeans_waves_errors.npz")
resolutions = data["resolutions"]
errors_fd = data["errors_fd"]
errors_fv = data["errors_fv"]

fig_error, ax_error = plt.subplots(figsize=(8, 6))

ax_error.loglog(resolutions, errors_fd, marker='o', label='Finite Difference')
ax_error.loglog(resolutions, errors_fv, marker='o', label='Finite Volume')

# Choose anchor in lower-left region
anchor = (20, 1e-9)

add_power_law_indicators(
    ax=ax_error,
    anchor=anchor,
    exponents=[-2, -5],
    x_span=2.0,
    scales=[1.0, 1.0],
    x_label='N'
)

ax_error.set_xlabel('Resolution (Number of Cells)')
ax_error.set_ylabel('L1 Density Error')
ax_error.set_title('L1 Density Error vs Resolution for Jeans Waves')
ax_error.legend()

fig_error.tight_layout()
fig_error.savefig("figures/jeans_waves_error_comparison.svg")