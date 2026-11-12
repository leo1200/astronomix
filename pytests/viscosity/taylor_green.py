"""
3D Taylor-Green vortex: viscous kinetic-energy decay.

Initializes the classic Taylor-Green vortex on a periodic cubic box at low Mach
number, integrates it with the viscous finite-difference solver, and compares the
kinetic-energy time series against the incompressible reference decay
E_k(t) = E_k(0) exp(-6 nu t). The comparison plot is written to ``figures/``.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# The energy decays by orders of magnitude over the run, so we integrate in
# double precision to keep finite precision from distorting the time series.
jax.config.update("jax_enable_x64", True)

# plotting
import matplotlib.pyplot as plt

# astronomix constants
from astronomix import (
    FORWARDS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
    BoundarySettings,
    BoundarySettings1D,
    SnapshotSettings,
    DiffusionConfig,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
    construct_primitive_state,
    initialize_interface_fields,
    finalize_config,
)
from astronomix._fluid_equations.total_quantities import calculate_kinetic_energy


# figures are written to the local figures/ directory
from pathlib import Path
figures_dir = Path(__file__).resolve().parent / "figures"
figures_dir.mkdir(exist_ok=True)

# -------------------------------------------------------------
# =============== ↓ Parameters ↓ ==============================
# -------------------------------------------------------------
gamma = 5.0 / 3.0
box_size = 2.0 * jnp.pi
num_cells = 32
Re = 100.0            # Reynolds number, Re = rho0 V0 / (mu k0), k0 = 1
mach = 0.1            # keep << 1 so the incompressible reference holds
t_end = 10.0

# Derived (dynamic) viscosity from the Reynolds number.
rho0 = 1.0
V0 = mach                 # c_s = 1 for the pressure below
p0 = rho0 / gamma         # => c_s = sqrt(gamma p0 / rho0) = 1
mu = rho0 * V0 / Re       # dynamic viscosity
nu = mu / rho0            # kinematic viscosity
# -------------------------------------------------------------
# =============== ↑ Parameters ↑ ==============================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Configuration ↓ ===========================
# -------------------------------------------------------------
config = SimulationConfig(
    progress_bar=True,
    mhd=False,
    diffusion_config=DiffusionConfig(viscosity=True),
    dimensionality=3,
    num_cells=num_cells,
    box_size=float(box_size),
    differentiation_mode=FORWARDS,
    boundary_settings=BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    return_snapshots=True,
    num_snapshots=100,
    snapshot_settings=SnapshotSettings(
        return_final_state=True,
        return_kinetic_energy=True,
    ),
)

params = SimulationParams(
    t_end=t_end,
    C_cfl=0.4,
    dt_max=jnp.inf,
    minimum_density=1e-8,
    minimum_pressure=1e-10,
    viscosity=mu,
)

helper_data = get_helper_data(config)
registered_variables = get_registered_variables(config)
# -------------------------------------------------------------
# =============== ↑ Configuration ↑ ===========================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Initial Taylor-Green state ↓ ==============
# -------------------------------------------------------------
dx = config.box_size / num_cells
coords = jnp.arange(num_cells) * dx + 0.5 * dx
x, y, z = jnp.meshgrid(coords, coords, coords, indexing="ij")

rho = rho0 * jnp.ones_like(x)
v_x = V0 * jnp.sin(x) * jnp.cos(y) * jnp.cos(z)
v_y = -V0 * jnp.cos(x) * jnp.sin(y) * jnp.cos(z)
v_z = jnp.zeros_like(x)
p = (
    p0
    + (rho0 * V0**2 / 16.0)
    * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y))
    * (jnp.cos(2.0 * z) + 2.0)
)

# The vortex is purely hydrodynamic, but the constrained-transport machinery
# still wants zero interface magnetic fields to be present.
B_x = jnp.zeros_like(rho)
B_y = jnp.zeros_like(rho)
B_z = jnp.zeros_like(rho)
bxb, byb, bzb = initialize_interface_fields(B_x, B_y, B_z)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=v_x,
    velocity_y=v_y,
    velocity_z=v_z,
    gas_pressure=p,
    magnetic_field_x=B_x,
    magnetic_field_y=B_y,
    magnetic_field_z=B_z,
    interface_magnetic_field_x=bxb,
    interface_magnetic_field_y=byb,
    interface_magnetic_field_z=bzb,
)

config = finalize_config(config, initial_state.shape)

# Analytical reference: E_k(t) = E_k(0) exp(-6 nu t) for k0 = 1 at low Mach.
Ek0 = calculate_kinetic_energy(initial_state, helper_data, config, registered_variables)
# -------------------------------------------------------------
# =============== ↑ Initial Taylor-Green state ↑ ==============
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Run ↓ =====================================
# -------------------------------------------------------------
snapshots = jax.block_until_ready(
    time_integration(initial_state, config, params, registered_variables)
)

t = snapshots.time_points
Ek_sim = snapshots.kinetic_energy
Ek_exact = Ek0 * jnp.exp(-6.0 * nu * t)
# -------------------------------------------------------------
# =============== ↑ Run ↑ =====================================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Plot ↓ ====================================
# -------------------------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
ax.plot(t, Ek_sim, label="Simulation", linewidth=2)
ax.plot(t, Ek_exact, "--", label=r"$E_k(0)\,e^{-6\nu t}$", linewidth=2)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$E_k$")
ax.set_title(f"3D Taylor-Green vortex  (N={num_cells}, Re={Re:.0f}, Ma={mach})")
ax.legend()
fig.tight_layout()
fig.savefig(figures_dir / "taylor_green_energy_decay.png", dpi=200)
# -------------------------------------------------------------
# =============== ↑ Plot ↑ ====================================
# -------------------------------------------------------------
