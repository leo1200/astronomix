"""Decompose the internal-energy rate of the cells that go negative in the FCT run.

For one saved Evrard snapshot, evaluate per cell: the hydrodynamic rate (PP-WENO
flux divergence of E minus the kinetic part), and the gravitational rates of the
high-order, low-order (donor) and flux-corrected couplings.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/evrard_rates.py out/evrard_pp_fct_n32_x32_fourth_none.npz 11
"""
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    GravityConfig,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_registered_variables,
    get_helper_data,
    FDSolverConfig,
)
from astronomix.option_classes.simulation_config import FOURTH_ORDER_CONSERVATIVE
from astronomix._fluid_equations._equations import conserved_state_from_primitive
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_native_for_axis
from astronomix._modules._gravity import _gravity as gravity_module
from astronomix._stencil_operations._stencil_operations import _shift

data = np.load(sys.argv[1])
snap = int(sys.argv[2])
state = jnp.asarray(data["states"][snap], dtype=jnp.float32)
n = state.shape[-1]
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)


def make_config(fct):
    config = SimulationConfig(
        gravity_config=GravityConfig(self_gravity=True, self_gravity_version=FOURTH_ORDER_CONSERVATIVE,
                                     poisson_manual_open_boundaries=True, work_flux_correction=fct),
        dimensionality=3,
        box_size=4.0,
        num_cells=n,
        backend_config=BackendConfig(backend=NATIVE_JAX),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        solver_config=FDSolverConfig(weno_positivity_preserving=True),
    )
    return finalize_config(config, state.shape)


params = SimulationParams(minimum_density=1e-5, minimum_pressure=3e-6)
config = make_config(False)
rv = get_registered_variables(config)
q = conserved_state_from_primitive(state, params.gamma, config, rv)
fluxes = [_weno_flux_native_for_axis(a)(q, params, config, rv) for a in range(3)]
dx = config.grid_spacing
divergence = sum((f - _shift(f, 1, axis=a + 1)) / dx for a, f in enumerate(fluxes))
dt = 1.0  # rates: the sources scale linearly with dt
drho = -divergence[rv.density_index] * dt
density_fluxes = [f[rv.density_index] for f in fluxes]
rho, v, p = state[0], state[1:4], state[4]
kinetic_hydro = sum(v[a] * (-divergence[1 + a]) for a in range(3)) - 0.5 * (v**2).sum(0) * (-divergence[0])
hydro_internal = -divergence[rv.energy_index] - kinetic_hydro

S_high = gravity_module._fd_gravity_source(state, density_fluxes, drho, dt, config, params, rv)
S_fct = gravity_module._fd_gravity_source(state, density_fluxes, drho, dt, make_config(True), params, rv)
kinetic_gravity = sum(v[a] * S_high[1 + a] for a in range(3))
grav_high = S_high[rv.energy_index] - kinetic_gravity
grav_fct = S_fct[rv.energy_index] - kinetic_gravity

e = p / (params.gamma - 1.0)
nxt = data["states"][snap + 1][4]
bad = np.argwhere(np.asarray(nxt) < 0)
print(f"snapshot {snap}: {len(bad)} cells negative at the next snapshot")
for idx in bad[:8]:
    i = tuple(idx)
    print(f"cell {i}: rho={float(rho[i]):.2e} e={float(e[i]):.2e} |v|={float(jnp.sqrt((v[:, i[0], i[1], i[2]]**2).sum())):.2f} "
          f"rates: hydro={float(hydro_internal[i]):+.2e} grav_high={float(grav_high[i]):+.2e} "
          f"grav_fct={float(grav_fct[i]):+.2e}  e*(|v|+c)/dx={float(e[i]*(jnp.sqrt((v[:, i[0], i[1], i[2]]**2).sum()) + jnp.sqrt(params.gamma*p[i]/rho[i]))/dx):.2e}")

# geometry of the first failing cell
i, j, k = (int(x) for x in bad[0])
centres = (np.arange(n) + 0.5) * dx - 2.0
position = np.array([centres[i], centres[j], centres[k]])
radial = position / np.linalg.norm(position)
print("cell", (i, j, k), "r =", np.linalg.norm(position), " v =", np.asarray(v[:, i, j, k]), " v_r =", float(np.dot(np.asarray(v[:, i, j, k]), radial)))
phi = gravity_module._compute_total_potential(rho, dx, config, params, rv, params.gravitational_constant)
for a in range(3):
    offset = [0, 0, 0]; offset[a] = 1
    up = (i + offset[0], j + offset[1], k + offset[2]); down = (i - offset[0], j - offset[1], k - offset[2])
    F_right = float(density_fluxes[a][i, j, k]); F_left = float(density_fluxes[a][down])
    print(f" axis {a}: m={float(rho[i,j,k]*v[a,i,j,k]):+.2e}  F_left={F_left:+.2e} F_right={F_right:+.2e}  "
          f"phi(down,i,up)=({float(phi[down]):.4f},{float(phi[i,j,k]):.4f},{float(phi[up]):.4f})  "
          f"rho(down,i,up)=({float(rho[down]):.1e},{float(rho[i,j,k]):.1e},{float(rho[up]):.1e})")
