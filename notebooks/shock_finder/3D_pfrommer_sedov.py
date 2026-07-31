# ============================================================================
# 3D Shock Finder Test — Point Explosion (Sedov-Taylor blast wave)
# ============================================================================
# A single point-like energy injection at the domain center drives one
# outward-propagating spherical shock (Sedov-Taylor blast wave).
#
# Same setup pattern as the 2D test, generalized to 3D:
#   - injection sphere of radius r_explosion at domain center
#   - pressure inside sphere set so that integrating p/(gamma-1) over the
#     sphere volume reproduces E_explosion
#   - ambient density/pressure elsewhere, zero initial velocity
#
# Diagnostics:
#   - three orthogonal mid-plane projections (xy @ z=0.5, xz @ y=0.5,
#     yz @ x=0.5) showing density/pressure with shock zones, shock
#     surface, and shock direction arrows overlaid
#   - 3D average shock radius, computed as the mean of
#     sqrt(x^2 + y^2 + z^2) (measured from the explosion center) over all
#     shock-surface cells, compared against the analytic Sedov-Taylor
#     radius R(t) = xi_0 * (E * t^2 / rho_ambient)^(1/5)
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

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

#%%
# CONFIGURATION

num_cells = 64          # per-axis resolution (3D is expensive: 64^3 cells)
box_size  = 1.0

config = SimulationConfig(
    geometry=CARTESIAN,
    dimensionality=3,
    riemann_solver=HLLC,
    limiter=MINMOD,
    box_size=box_size,
    num_cells=num_cells,
)
params = SimulationParams(t_end=0.15)

helper_data          = get_helper_data(config)
registered_variables = get_registered_variables(config)

# geometric_centers shape: (nx, ny, nz, 3) — last axis is (x, y, z)
geometric_centers = helper_data.geometric_centers

geometry_x = geometric_centers[..., 0]  # (nx, ny, nz)
geometry_y = geometric_centers[..., 1]  # (nx, ny, nz)
geometry_z = geometric_centers[..., 2]  # (nx, ny, nz)


# ============================================================================
# INITIAL CONDITIONS — point explosion (single outward-propagating shock)
# ============================================================================

TARGET_CENTER = (0.5, 0.5, 0.5)
center_x, center_y, center_z = TARGET_CENTER

E_explosion = 1.0

rho_ambient = 1.0
p_ambient   = 1e-4

r_explosion = 0.08   # a bit larger than the 2D case since 3D volume shrinks faster

dx_from_center = geometry_x - center_x
dy_from_center = geometry_y - center_y
dz_from_center = geometry_z - center_z

r = jnp.sqrt(dx_from_center**2 + dy_from_center**2 + dz_from_center**2)

# injection volume (3D analog of the 2D injection_area)
injection_volume = (4.0 / 3.0) * jnp.pi * r_explosion**3

gamma_gas = params.gamma

# E = p * V / (gamma - 1)  =>  p = E * (gamma - 1) / V
p_explosion_gas = E_explosion * (gamma_gas - 1) / injection_volume

p   = jnp.where(r < r_explosion, p_explosion_gas, p_ambient)
rho = jnp.ones_like(geometry_x) * rho_ambient
u_x = jnp.zeros_like(geometry_x)
u_y = jnp.zeros_like(geometry_y)
u_z = jnp.zeros_like(geometry_z)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u_x,
    velocity_y=u_y,
    velocity_z=u_z,
    gas_pressure=p,
)
config = finalize_config(config, initial_state.shape)


# ============================================================================
# RUN SIMULATION
# ============================================================================

#%%
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

shock_dir_x = result.shock_direction[0]   # (nx, ny, nz)
shock_dir_y = result.shock_direction[1]
shock_dir_z = result.shock_direction[2]

surface_mask = np.array(result.shock_surface_cells).astype(bool)


#%%
# DIAGNOSTICS

print(f"=== 3D Point Explosion at center {TARGET_CENTER} ===")
print(f"num_shocks (surface cells): {result.num_shocks}")

if result.num_shocks == 0:
    print("WARNING: no shock surface cells found")
else:
    surface_mach = np.array(result.mach_numbers)[surface_mask]
    print(f"Mach at surface: min={surface_mach.min():.3f}  max={surface_mach.max():.3f}  mean={surface_mach.mean():.3f}")


# ============================================================================
# 3D average shock radius vs analytic Sedov-Taylor radius
# ============================================================================

geometry_x_np = np.array(geometry_x)
geometry_y_np = np.array(geometry_y)
geometry_z_np = np.array(geometry_z)

# distances of surface cells from explosion center: sqrt(x^2 + y^2 + z^2)
dx_surf = geometry_x_np[surface_mask] - center_x
dy_surf = geometry_y_np[surface_mask] - center_y
dz_surf = geometry_z_np[surface_mask] - center_z

r_surface = np.sqrt(dx_surf**2 + dy_surf**2 + dz_surf**2)

if len(r_surface) > 0:
    r_measured_mean   = r_surface.mean()
    r_measured_std    = r_surface.std()
    r_measured_median = np.median(r_surface)
else:
    r_measured_mean = r_measured_std = r_measured_median = np.nan

# Analytic Sedov-Taylor similarity solution:
#   R(t) = xi_0 * (E * t^2 / rho_ambient)^(1/5)
# xi_0 depends on gamma; for gamma = 5/3 (monatomic ideal gas), xi_0 ≈ 1.15
# (Sedov's tabulated constant). For other gamma this constant shifts, but
# gamma = 5/3 is the astronomix default unless overridden.
xi_0 = 1.15  # valid for gamma = 5/3; adjust if params.gamma differs

t_end = params.t_end
r_analytic = xi_0 * (E_explosion * t_end**2 / rho_ambient) ** (1.0 / 5.0)

print("\n=== Shock radius: measured vs analytic (Sedov-Taylor) ===")
print(f"  gamma used in sim:      {gamma_gas:.4f}  (xi_0={xi_0} assumes gamma=5/3)")
print(f"  t_end:                  {t_end}")
print(f"  measured mean radius:   {r_measured_mean:.4f}  (std={r_measured_std:.4f}, median={r_measured_median:.4f})")
print(f"  analytic Sedov radius:  {r_analytic:.4f}")
if not np.isnan(r_measured_mean):
    rel_err = 100.0 * (r_measured_mean - r_analytic) / r_analytic
    print(f"  relative error:         {rel_err:+.2f} %")


#%%
# PLOTS — three orthogonal mid-plane projections
# ============================================================================
# For each projection plane, take the mid-plane slice index along the
# orthogonal axis, then overlay: background field, shock zone, shock
# surface, and shock direction arrows (projected onto that plane).
# ============================================================================

def mid_index(coord_1d, target):
    return int(np.argmin(np.abs(coord_1d - target)))

# 1D coordinate arrays along each axis (assumes a regular grid)
x_1d = geometry_x_np[:, 0, 0]
y_1d = geometry_y_np[0, :, 0]
z_1d = geometry_z_np[0, 0, :]

ix_mid = mid_index(x_1d, center_x)
iy_mid = mid_index(y_1d, center_y)
iz_mid = mid_index(z_1d, center_z)

rho_final_np = np.array(rho_final)
zones_np     = np.array(result.shock_zones).astype(bool)

shock_dir_x_np = np.array(shock_dir_x)
shock_dir_y_np = np.array(shock_dir_y)
shock_dir_z_np = np.array(shock_dir_z)


def plot_projection(ax, plane, slice_idx, axis0_label, axis1_label):
    """
    plane: one of 'xy', 'xz', 'yz'
    slice_idx: index along the orthogonal (sliced-out) axis
    """
    if plane == "xy":
        A = geometry_x_np[:, :, slice_idx]
        B = geometry_y_np[:, :, slice_idx]
        field = rho_final_np[:, :, slice_idx]
        zone_slice = zones_np[:, :, slice_idx]
        surf_slice = surface_mask[:, :, slice_idx]
        da = shock_dir_x_np[:, :, slice_idx]
        db = shock_dir_y_np[:, :, slice_idx]
        c0, c1 = center_x, center_y
    elif plane == "xz":
        A = geometry_x_np[:, slice_idx, :]
        B = geometry_z_np[:, slice_idx, :]
        field = rho_final_np[:, slice_idx, :]
        zone_slice = zones_np[:, slice_idx, :]
        surf_slice = surface_mask[:, slice_idx, :]
        da = shock_dir_x_np[:, slice_idx, :]
        db = shock_dir_z_np[:, slice_idx, :]
        c0, c1 = center_x, center_z
    elif plane == "yz":
        A = geometry_y_np[slice_idx, :, :]
        B = geometry_z_np[slice_idx, :, :]
        field = rho_final_np[slice_idx, :, :]
        zone_slice = zones_np[slice_idx, :, :]
        surf_slice = surface_mask[slice_idx, :, :]
        da = shock_dir_y_np[slice_idx, :, :]
        db = shock_dir_z_np[slice_idx, :, :]
        c0, c1 = center_y, center_z
    else:
        raise ValueError(plane)

    ax.pcolormesh(A, B, field, cmap="plasma", shading="auto", alpha=0.85)

    ax.contourf(
        A, B, zone_slice.astype(float),
        levels=[0.5, 1.5], colors=["green"], alpha=0.22,
    )

    ax.contour(
        A, B, surf_slice.astype(float),
        levels=[0.5], colors="red", linewidths=1.5,
    )

    # shock direction arrows at surface cells in this slice
    Af = A[surf_slice]
    Bf = B[surf_slice]
    daf = da[surf_slice]
    dbf = db[surf_slice]

    mag = np.sqrt(daf**2 + dbf**2)
    valid = mag > 0
    Af, Bf, daf, dbf = Af[valid], Bf[valid], daf[valid], dbf[valid]
    mag = mag[valid]

    if len(Af) > 0:
        daf_u = daf / mag
        dbf_u = dbf / mag

        n_arrows = 60
        if len(Af) > n_arrows:
            idx = np.linspace(0, len(Af) - 1, n_arrows).astype(int)
            Af, Bf, daf_u, dbf_u = Af[idx], Bf[idx], daf_u[idx], dbf_u[idx]

        ax.quiver(
            Af, Bf, daf_u, dbf_u,
            angles="xy", scale_units="xy", scale=20,
            color="white", width=0.004, headwidth=4, headlength=5,
            pivot="middle", zorder=20,
        )

    # mark explosion center
    ax.plot(c0, c1, marker="+", color="cyan", markersize=10, mew=2, zorder=25)

    ax.set_xlabel(axis0_label)
    ax.set_ylabel(axis1_label)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)


fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

fig.suptitle(
    f"3D Point Explosion (Sedov-Taylor) — mid-plane projections at t={params.t_end}\n"
    f"measured shock radius = {r_measured_mean:.4f}  |  analytic = {r_analytic:.4f}  "
    f"({100*(r_measured_mean - r_analytic)/r_analytic:+.2f}%)"
    if not np.isnan(r_measured_mean) else
    f"3D Point Explosion (Sedov-Taylor) — mid-plane projections at t={params.t_end}",
    fontsize=12,
)

plot_projection(axes[0], "xy", iz_mid, "x", "y")
axes[0].set_title(f"xy plane (z ≈ {z_1d[iz_mid]:.3f})")

plot_projection(axes[1], "xz", iy_mid, "x", "z")
axes[1].set_title(f"xz plane (y ≈ {y_1d[iy_mid]:.3f})")

plot_projection(axes[2], "yz", ix_mid, "y", "z")
axes[2].set_title(f"yz plane (x ≈ {x_1d[ix_mid]:.3f})")

axes[0].legend(
    handles=[
        Patch(facecolor="green", edgecolor="green", alpha=0.22, label="shock zone"),
        Line2D([0], [0], color="red", lw=1.5, label="shock surface"),
        Line2D([0], [0], color="white", lw=0, marker=r"$\rightarrow$",
               markersize=12, label="shock direction"),
    ],
    loc="upper right", fontsize=8,
)

plt.show()

# %%