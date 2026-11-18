"""Where does the pressure go negative in adiabatic MHD turbulence?

For a saved turb.py snapshot (primitive state + interface B) evaluate, in float64:

1. Admissibility of the Lax-Friedrichs split states q +- F/alpha at the cell's
   own fast speed alpha = |v_n| + c_f. Zhang-Shu positivity rests on these being
   admissible. For hydro they are; for multi-dimensional MHD they need not be
   (Wu 2018, SINUM 56: the "LF splitting property" fails for MHD unless a
   discrete div(B) condition holds), and the theta-scaling cannot repair an
   inadmissible base (theta = 0 leaves it as is).
2. One forward-Euler step with the PP-WENO fluxes at Courant number C_FE (the
   code's sum-of-axes CFL). The theory covers C_FE <= 1/2, i.e. C_cfl <= 0.754
   for the SSPRK(5,4) step; turb.py runs C_cfl = 1.5, i.e. C_FE ~ 1.0.
   The cell-centred B is advanced by the WENO B flux here; the code then swaps
   in the CT field with E += 0.5 (B_ct^2 - B_weno^2), which holds p fixed, so
   the pressure of this update is the pressure the code carries.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/mhd_pressure_diag.py SNAP.npy
"""
import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

# ruff: noqa: E402
import numpy as np
import jax.numpy as jnp

from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    NATIVE_JAX,
    PERIODIC_BOUNDARY,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
    finalize_config,
    get_registered_variables,
)
from astronomix._finite_difference._interface_fluxes._weno import _weno_flux_native_for_axis
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.option_classes.simulation_config import IDEAL_GAS

parser = argparse.ArgumentParser()
parser.add_argument("snapshot")
parser.add_argument("--courant", default="0.25,0.5,0.75,1.0")
args = parser.parse_args()

GAMMA = 5.0 / 3.0
loaded = np.load(args.snapshot)
primitive = jnp.asarray(loaded["state"] if args.snapshot.endswith(".npz") else loaded, dtype=jnp.float64)
n = primitive.shape[-1]
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
config = SimulationConfig(
    equation_of_state=IDEAL_GAS,
    dimensionality=3,
    num_cells=n,
    box_size=1.0,
    mhd=True,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False),
    solver_config=FDSolverConfig(weno_positivity_preserving=True),
)
config = finalize_config(config, primitive.shape)
rv = get_registered_variables(config)
params = SimulationParams(gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)

q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
rho = q[rv.density_index]
MOMENTUM = list(rv.velocity_index)
MAGNETIC = list(rv.magnetic_index)
ENERGY = rv.pressure_index
m = q[jnp.array(MOMENTUM)]
B = q[jnp.array(MAGNETIC)]
E = q[ENERGY]
v = m / rho
p = primitive[rv.pressure_index]
dx = config.grid_spacing


def gas_pressure(state):
    kinetic = 0.5 * jnp.sum(state[jnp.array(MOMENTUM)] ** 2, axis=0) / state[0]
    magnetic = 0.5 * jnp.sum(state[jnp.array(MAGNETIC)] ** 2, axis=0)
    return (GAMMA - 1.0) * (state[ENERGY] - kinetic - magnetic)


def physical_flux(axis):
    """Ideal-MHD flux along ``axis`` in the registered slot layout."""
    total_pressure = p + 0.5 * jnp.sum(B**2, axis=0)
    v_dot_B = jnp.sum(v * B, axis=0)
    flux = jnp.zeros_like(q)
    flux = flux.at[0].set(m[axis])
    for k in range(3):
        flux = flux.at[MOMENTUM[k]].set(m[k] * v[axis] - B[axis] * B[k] + (total_pressure if k == axis else 0.0))
        flux = flux.at[MAGNETIC[k]].set(v[axis] * B[k] - B[axis] * v[k])
    flux = flux.at[ENERGY].set((E + total_pressure) * v[axis] - B[axis] * v_dot_B)
    return flux


def fast_speed(axis):
    a2 = GAMMA * p / rho
    b2 = jnp.sum(B**2, axis=0) / rho
    bn2 = B[axis] ** 2 / rho
    disc = jnp.sqrt(jnp.maximum((a2 + b2) ** 2 - 4.0 * a2 * bn2, 0.0))
    return jnp.sqrt(0.5 * (a2 + b2 + disc))


print(f"snapshot {os.path.basename(args.snapshot)}: N={n} min rho={float(rho.min()):.3e} "
      f"min p={float(p.min()):.3e} min beta={float((2 * p / jnp.sum(B**2, 0)).min()):.2e} "
      f"cells p<0: {int((p < 0).sum())}")

# 1. Lax-Friedrichs split states at the cell's own speed (the hardest admissible alpha)
for axis in range(3):
    alpha = jnp.abs(v[axis]) + fast_speed(axis)
    flux = physical_flux(axis)
    for sign, name in ((1.0, "q+F/a"), (-1.0, "q-F/a")):
        split = q + sign * flux / alpha[None]
        split_p = gas_pressure(split)
        bad = (split_p < 0) | (split[0] < 0)
        ratio = jnp.where(p > 0, split_p / p, jnp.inf)
        print(f"  axis {axis} {name}: inadmissible cells {int(bad.sum()):7d}  "
              f"min p_split/p = {float(ratio.min()):+.3e}")

# 1b. the same at the code's splitting speed (stencil maximum of |v_n| + c_f over
# cells i-2..i+3): a face whose upwind base state q_i + F_i/alpha or
# q_{i+1} - F_{i+1}/alpha is inadmissible gets theta = 0, i.e. first-order Rusanov
beta = 2.0 * p / jnp.sum(B**2, axis=0)
for axis in range(3):
    spatial = axis
    radius = jnp.abs(v[axis]) + fast_speed(axis)
    alpha = jnp.max(jnp.stack([jnp.roll(radius, -offset, axis=spatial) for offset in (-2, -1, 0, 1, 2, 3)]), axis=0)
    flux = physical_flux(axis)
    plus_base = q + flux / alpha[None]
    minus_base = jnp.roll(q, -1, axis=spatial + 1) - jnp.roll(flux, -1, axis=spatial + 1) / alpha[None]
    first_order = (gas_pressure(plus_base) <= 0) | (gas_pressure(minus_base) <= 0)
    low_beta = beta < 0.1
    print(f"  axis {axis}: faces forced to first order {float(first_order.mean()) * 100:6.2f} % "
          f"(cells with beta<0.1: {float(low_beta.mean()) * 100:5.1f} %, of those forced "
          f"{float((first_order & low_beta).sum() / jnp.maximum(low_beta.sum(), 1)) * 100:5.1f} %)")

# 2. one forward-Euler step with the PP-WENO fluxes
fluxes = [_weno_flux_native_for_axis(axis)(q, params, config, rv) for axis in range(3)]
divergence = sum((f - _shift(f, 1, axis=a + 1)) / dx for a, f in enumerate(fluxes))
speed_sum = sum(float(jnp.max(jnp.abs(v[a]) + fast_speed(a))) for a in range(3))
for courant in (float(c) for c in args.courant.split(",")):
    dt = courant * dx / speed_sum
    updated = q - dt * divergence
    updated_p = gas_pressure(updated)
    print(f"  FE step C_FE={courant:.2f} (C_cfl={courant * 1.508:.2f}): cells p<0 {int((updated_p < 0).sum()):7d} "
          f"rho<0 {int((updated[0] < 0).sum())}  min p={float(updated_p.min()):+.3e}")
