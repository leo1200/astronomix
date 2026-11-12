"""
Turbulent radiative mixing layer.

Setup based on Lancaster 2026 (unpublished):

- box with dimensions (Lx, Ly, Lz) = (L_box, L_box, 1.5 * L_box), L_box = 1.0
- boundaries
    - x, y: periodic
    - z: bottom outflow, top fixed to the hot inflow state
- z > z_center is the hot state, z < z_center is the cold state, joined by a
  tanh interface of thickness ~ grid_spacing / 2 at z_center = L_z / 2
- the two phases are
    - U_hot  = (rho, v_x, v_y, v_z, P) = (rho_0,       +v_rel/2, 0, 0, P_0)
    - U_cold = (rho, v_x, v_y, v_z, P) = (chi * rho_0, -v_rel/2, 0, 0, P_0)
  with rho_0 = P_0 = 1 and the shear velocity set by the Mach number,
  M = v_rel / c_hot, c_hot = sqrt(gamma * P_0 / rho_0).
- a Kelvin-Helmholtz seed perturbation is injected in the interface tails so
  the shear layer rolls up and, together with the mixing-layer cooling, forms
  a developed turbulent radiative mixing layer.

This is an inviscid setup (numerical dissipation only), matching the reference
that produced the paper figure.

The default resolution here (num_cells_x = 128) is chosen so the layer clearly
develops while the run finishes in a few minutes on a single GPU. The paper
figure uses higher resolution (num_cells_x = 256 / 512) and a longer horizon.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import os
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

# astronomix constants
from astronomix import (
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
)
from astronomix.option_classes.simulation_config import FIXED_BOUNDARY_OPEN_MOMENTUM
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    SIMPLE_MIXING_LAYER_COOLING,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    BackendConfig,
    FDSolverConfig,
)
from astronomix.option_classes.simulation_params import (
    FixedBoundaryState,
    FixedBoundaryState1D,
)
from astronomix.data_classes.simulation_helper_data import HelperDataRequirements
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    MixingCoolingParams,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    time_integration,
    construct_primitive_state,
    get_registered_variables,
    finalize_config,
)

# frame tracking (frame_tracking.py next to this script)
from frame_tracking import frame_tracking_update


# figures are written to the local figures/ directory
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# =============== ↓ Box setup ↓ ===============================
# -------------------------------------------------------------
# num_cells_x in the ~96-160 range develops a visible mixing layer in a few
# minutes on one GPU; raise to 256 / 512 to match the paper figure.
num_cells_x = int(os.environ.get("NUM_CELLS_X", 128))
num_cells_y = num_cells_x
num_cells_z = int(1.5 * num_cells_x)
box_size = 1.0
grid_spacing = box_size / num_cells_x
L_x, L_y, L_z = box_size, box_size, 1.5 * box_size


# -------------------------------------------------------------
# =============== ↑ Box setup ↑ ===============================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Physics constants ↓ =======================
# -------------------------------------------------------------
# Long enough for the KH-seeded shear layer to roll up and mix; the reference
# runs to 30 t_sh, 20 t_sh already yields a well-developed layer here.
t_end_in_t_sh = float(os.environ.get("T_END_IN_T_SH", 20.0))
density_contrast = 100.0
# cooling timescale ratio; higher xi (stronger cooling) needs the
# positivity-preserving WENO enabled in the config below to stay stable
xi = 3.0
mach_number = 0.5
gamma = 5 / 3
P0 = 1.0

rho_hot = 1.0
rho_cold = density_contrast * rho_hot
T_hot = P0 / rho_hot
T_cold = P0 / rho_cold
c_hot = (gamma * P0 / rho_hot) ** 0.5
v_rel = mach_number * c_hot
t_sh = L_x / v_rel


# -------------------------------------------------------------
# =============== ↑ Physics constants ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Config ↓ ==================================
# -------------------------------------------------------------
config = SimulationConfig(
    solver_config=FDSolverConfig(
        # positivity-preserving WENO keeps the strong density contrast plus
        # cooling stable
        weno_positivity_preserving=True,
    ),
    backend_config=BackendConfig(
        pallas_block_shape=(4, 4, 4),
    ),
    progress_bar=True,
    dimensionality=3,
    box_size=StaticFloatVector(L_x, L_y, L_z),
    num_cells=StaticIntVector(num_cells_x, num_cells_y, num_cells_z),
    boundary_settings=BoundarySettings(
        x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        z=BoundarySettings1D(OPEN_BOUNDARY, FIXED_BOUNDARY_OPEN_MOMENTUM),
    ),
    cooling_config=CoolingConfig(
        cooling=True,
        cooling_method=IMPLICIT_COOLING,
        cooling_curve_config=CoolingCurveConfig(
            cooling_curve_type=SIMPLE_MIXING_LAYER_COOLING,
        ),
    ),
)

registered_variables = get_registered_variables(config)


# -------------------------------------------------------------
# =============== ↑ Config ↑ ==================================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Initial two-phase state ↓ =================
# -------------------------------------------------------------
def single_interface(lower_value, upper_value, coordinate, interface_center, smoothing_length):
    """
    Blend two constant values across a single smooth (tanh) interface.

    Args:
        lower_value: The value taken far below the interface.
        upper_value: The value taken far above the interface.
        coordinate: The coordinate along which the interface varies.
        interface_center: The coordinate of the interface midpoint.
        smoothing_length: The width over which the transition is smoothed.

    Returns:
        The field transitioning smoothly from lower_value to upper_value
        across the interface.
    """
    return 0.5 * (
        lower_value * (1 - jnp.tanh((coordinate - interface_center) / smoothing_length))
        + upper_value * (1 + jnp.tanh((coordinate - interface_center) / smoothing_length))
    )

helper_data = get_helper_data(config)
cell_centers = helper_data.geometric_centers
X = cell_centers[:, :, :, 0]
Y = cell_centers[:, :, :, 1]
Z = cell_centers[:, :, :, 2]
z_center = L_z / 2
smoothing_length = grid_spacing / 2

density = single_interface(rho_cold, rho_hot, Z, z_center, smoothing_length)
pressure = P0 * jnp.ones_like(density)
velocity_x = single_interface(-v_rel / 2, v_rel / 2, Z, z_center, smoothing_length)
velocity_y = jnp.zeros_like(density)
velocity_z = jnp.zeros_like(density)

# --- Kelvin-Helmholtz seed perturbation ------------------------------------
# Envelope peaks in the tanh tails (~2 * smoothing_length from z_center) and
# vanishes in the bulk phases and at the sharpest point of the interface (where
# WENO's shock sensor would damp the seed most). This tail-localized envelope is
# what lets the shear layer roll up into a proper turbulent mixing layer.
dz = Z - z_center
envelope = jnp.exp(-((jnp.abs(dz) - 2 * smoothing_length) / (3 * grid_spacing)) ** 2)
amp = 0.03 * v_rel  # 3% of the shear velocity

# Deterministic low-k KH modes (well resolved) plus low-amplitude broadband noise.
mode_numbers = jnp.array([2, 4, 6])
kx = 2 * jnp.pi * mode_numbers / L_x
ky = 2 * jnp.pi * mode_numbers / L_y

key = jax.random.PRNGKey(42)
key_ph, key_n = jax.random.split(key, 2)
phases = jax.random.uniform(key_ph, (3, 3), minval=0.0, maxval=2 * jnp.pi)

modes = jnp.zeros_like(density)
for i in range(3):
    for j in range(3):
        modes = modes + jnp.sin(kx[i] * X + ky[j] * Y + phases[i, j])
modes = modes / 9.0  # normalize to O(1)

key_nx, key_ny, key_nz = jax.random.split(key_n, 3)
noise_x = jax.random.normal(key_nx, density.shape)
noise_y = jax.random.normal(key_ny, density.shape)
noise_z = jax.random.normal(key_nz, density.shape)

# v_z gets the full deterministic + noise treatment (the KH growth direction);
# v_x and v_y get noise only so they do not fight the mean shear.
velocity_x = velocity_x + amp * envelope * 0.3 * noise_x
velocity_y = velocity_y + amp * envelope * 0.3 * noise_y
velocity_z = velocity_z + amp * envelope * (modes + 0.3 * noise_z)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=density,
    velocity_x=velocity_x,
    velocity_y=velocity_y,
    velocity_z=velocity_z,
    gas_pressure=pressure,
)

mixing_cooling_params = MixingCoolingParams(
    xi=xi,
    mach_number=mach_number,
    density_contrast=density_contrast,
)

params = SimulationParams(
    t_end=t_end_in_t_sh * t_sh,
    C_cfl=1.5,
    gamma=gamma,
    minimum_density=rho_cold / 100,
    minimum_pressure=P0 / 100,
    fixed_boundary_state=FixedBoundaryState(
        z=FixedBoundaryState1D(
            right_state=jnp.array([rho_hot, v_rel / 2, 0.0, 0.0, P0])
        )
    ),
    cooling_params=CoolingParams(
        cooling_curve_params=mixing_cooling_params,
        floor_temperature=T_cold,
    ),
)

config = finalize_config(config, initial_state.shape)

# frame tracking: keep the mixing layer centred in z, applied once per time
# step through the per-step update hook; it needs the z cell centres with
# the ghost cells included
cell_centers_z = get_helper_data(
    config,
    padded=True,
    requirements=HelperDataRequirements(needs_cell_centers_z=True),
).cell_centers_z
config = config._replace(per_step_update=frame_tracking_update(cell_centers_z))


# -------------------------------------------------------------
# =============== ↑ Initial two-phase state ↑ =================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Run ↓ =====================================
# -------------------------------------------------------------
final_state = time_integration(initial_state, config, params, registered_variables)


# -------------------------------------------------------------
# =============== ↑ Run ↑ =====================================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Plot temperature slices ↓ =================
# -------------------------------------------------------------
y_index = num_cells_y // 2
initial_temperature = (
    initial_state[registered_variables.pressure_index]
    / initial_state[registered_variables.density_index]
)
final_temperature = (
    final_state[registered_variables.pressure_index]
    / final_state[registered_variables.density_index]
)

fig, (ax_i, ax_f) = plt.subplots(1, 2, figsize=(9, 5))
for ax, field, title in (
    (ax_i, initial_temperature, "Initial temperature"),
    (ax_f, final_temperature, "Final temperature"),
):
    slice_2d = field[:, y_index, :]
    print(f"{title} range: min={float(jnp.min(slice_2d)):.3e} max={float(jnp.max(slice_2d)):.3e}")
    # explicit positive vmin/vmax so LogNorm/colorbar stay valid even if the
    # field dips to zero or below in a floored cell
    vmax = float(jnp.max(slice_2d))
    positive = slice_2d[slice_2d > 0]
    vmin = float(jnp.min(positive)) if positive.size else vmax * 1e-6
    if not (vmax > vmin > 0):
        vmin, vmax = T_cold, T_hot
    im = ax.imshow(
        slice_2d.T,
        origin="lower",
        extent=[0, L_x, 0, L_z],
        norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label="Temperature")

fig.tight_layout()
fig.savefig(figures_dir / "turbulent_mixing_layer_temperature.png", dpi=200)

# -------------------------------------------------------------
# =============== ↑ Plot temperature slices ↑ =================
# -------------------------------------------------------------
