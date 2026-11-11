"""Why does the paired PP-WENO still produce p < 0 in 3D adiabatic MHD turbulence?

Phase ``step`` (GPU, the run's own Pallas / fp32 numerics): from the last
positive state saved by ``turb.py --stop-on-negative-p``, take full SSPRK(5,4)
+ CT steps (no forcing) until the pressure first goes negative; save that
step's input state, dt and the failing cell.

Phase ``patch`` (CPU, x64, native kernel): on a periodic patch centred on the
failing cell (large enough that the centre is exact for one step), compare at
the centre cell:

* the full step;
* single forward-Euler steps at C_FE = 0.125 ... 1 (the proof covers <= 1/2);
* the same forward-Euler steps with the Godunov-Powell source
  -(div B) (0, B, v, v.B), div B the central difference of the cell-centred
  field. That source removes the internal-energy term -(v.B)(div B) that the
  conservative MHD system carries wherever the divergence seen by the fluid
  update is not zero;
* the face-local inflow-pair bases per axis, and their axis sum.

    WENO_VARIANT=pp python examples/scripts/validation/weno_stability/negp_forensics.py step LASTPOSITIVE.npy OUT.npz
    WENO_VARIANT=pp PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/negp_forensics.py patch OUT.npz
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["step", "patch"])
parser.add_argument("inputs", nargs="+")
parser.add_argument("--mturb", type=float, default=20.0)
parser.add_argument("--cfl", type=float, default=1.5)
parser.add_argument("--max-steps", type=int, default=200)
parser.add_argument("--patch", type=int, default=49)
args = parser.parse_args()

if args.phase == "patch":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "1"
elif os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs

from astronomix import (
    BackendConfig, BoundarySettings, BoundarySettings1D, NATIVE_JAX, PALLAS, PERIODIC_BOUNDARY,
    PositivityConfig, SimulationConfig, SimulationParams, finalize_config, get_helper_data,
    get_registered_variables,
)
from astronomix._fluid_equations._equations_mhd import (
    conserved_state_from_primitive_mhd, primitive_state_from_conserved_mhd,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import update_cell_center_fields
from astronomix._finite_difference._time_integrators import _ssprk as ssprk_module
from astronomix._finite_difference._timestep_estimation._timestep_estimator import _cfl_time_step_fd
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 5.0 / 3.0
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)


def make_config(shape, backend):
    config = SimulationConfig(
        equation_of_state=IDEAL_GAS, dimensionality=3, num_cells=shape[-1], box_size=shape[-1] / 256.0,
        mhd=True, backend_config=BackendConfig(backend=backend),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        positivity_config=PositivityConfig(clamp_in_estimates=False), **weno_variant_kwargs(),
    )
    config = finalize_config(config, shape)
    params = SimulationParams(C_cfl=args.cfl, gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)
    return config, params, get_registered_variables(config)


def gas_pressure(q, rv):
    kinetic = 0.5 * sum(q[k] ** 2 for k in rv.momentum_index) / q[rv.density_index]
    magnetic = 0.5 * sum(q[k] ** 2 for k in rv.magnetic_index)
    return (GAMMA - 1.0) * (q[rv.energy_index] - kinetic - magnetic)


def one_step(primitive, dt, config, params, rv, helper_data):
    q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
    faces = [jnp.array(primitive[k], copy=True) for k in (-3, -2, -1)]
    new_q, bx, by, bz = ssprk_module._ssprk4_with_ct(
        jnp.array(q, copy=True), *faces, GAMMA, config.grid_spacing, dt, params, helper_data, config, rv)
    new_primitive = jnp.concatenate([
        primitive_state_from_conserved_mhd(new_q, 1e-30, 1e-30, GAMMA, config, rv),
        bx[None], by[None], bz[None]], axis=0)
    return new_q, new_primitive


if args.phase == "step":
    primitive = jnp.asarray(np.load(args.inputs[0]))
    config, params, rv = make_config(primitive.shape, PALLAS)
    helper_data = get_helper_data(config)
    for step in range(args.max_steps):
        dt = float(_cfl_time_step_fd(primitive, config.grid_spacing, jnp.inf, GAMMA, config, params, rv, args.cfl))
        new_q, new_primitive = one_step(primitive, dt, config, params, rv, helper_data)
        p_new = gas_pressure(new_q, rv)
        p_min = float(p_new.min())
        print(f"step {step}: dt={dt:.3e} min p in={float(primitive[rv.pressure_index].min()):+.3e} out={p_min:+.3e}",
              flush=True)
        if p_min < 0:
            cell = np.unravel_index(int(jnp.argmin(p_new)), p_new.shape)
            np.savez(args.inputs[1], state=np.asarray(primitive), dt=dt, cell=np.array(cell),
                     p_out=np.asarray(p_new[cell]))
            print(f"FIRST NEGATIVE at step {step}, cell {cell}, p={p_min:+.3e}; saved {args.inputs[1]}", flush=True)
            break
        primitive = new_primitive
    sys.exit(0)

# ↓ ———————————————————————————————————————————————————————————————— ↓
# patch phase (CPU, x64)
# ↑ ———————————————————————————————————————————————————————————————— ↑
data = np.load(args.inputs[0])
cell = tuple(int(c) for c in data["cell"])
dt = float(data["dt"])
half = args.patch // 2
full = data["state"].astype(np.float64)
patch = np.roll(full, shift=[0] + [half - c for c in cell], axis=(0, 1, 2, 3))[:, :args.patch, :args.patch, :args.patch]
centre = (half, half, half)
primitive = jnp.asarray(patch)
config, params, rv = make_config(primitive.shape, NATIVE_JAX)
helper_data = get_helper_data(config)
dx = config.grid_spacing

q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
q = update_cell_center_fields(q, primitive[-3], primitive[-2], primitive[-1], config, rv)
rho = q[rv.density_index]
B = [q[k] for k in rv.magnetic_index]
v = [q[k] / rho for k in rv.momentum_index]
div_central = sum((jnp.roll(B[a], -1, a) - jnp.roll(B[a], 1, a)) / (2 * dx) for a in range(3))
p0 = float(gas_pressure(q, rv)[centre])
v_dot_b = float(sum(v[a][centre] * B[a][centre] for a in range(3)))
b_mag = float(jnp.sqrt(sum(B[a][centre] ** 2 for a in range(3))))
print(f"cell {cell}: p={p0:.3e} rho={float(rho[centre]):.3e} |B|={b_mag:.3e} beta={2*p0/b_mag**2:.2e} "
      f"v.B={v_dot_b:+.3e} central div B * dx/|B| = {float(div_central[centre]) * dx / b_mag:+.3e}")
print(f"dt = {dt:.3e};  (gamma-1) * dt * (v.B)(div B) = {(GAMMA - 1) * dt * v_dot_b * float(div_central[centre]):+.3e}  "
      f"(compare p = {p0:.3e})")

new_q, _ = one_step(primitive, dt, config, params, rv, helper_data)
print(f"full SSPRK step (x64, native): p = {float(gas_pressure(new_q, rv)[centre]):+.3e}  "
      f"(fp32 Pallas run: {float(data['p_out']):+.3e})")


def forward_euler(primitive, dt_fe):
    """One forward-Euler step of the production stage machinery."""
    original = ssprk_module.ssprk4

    def fe(u0, dt_, *, rhs, pre_stage=lambda u: u, post_stage=lambda u: u, finalize=lambda u: u):
        u = pre_stage(u0)
        du = rhs(u, dt_)
        return finalize(post_stage(jax.tree_util.tree_map(lambda a, b: a + b, u, du)))

    ssprk_module.ssprk4 = fe
    try:
        new_q, _ = one_step(primitive, dt_fe, config, params, rv, helper_data)
    finally:
        ssprk_module.ssprk4 = original
    return new_q


dt_unit = dt / args.cfl  # dt at C_cfl = 1
for c_fe in (0.125, 0.25, 0.5, 1.0):
    dt_fe = c_fe * dt_unit
    fe_q = forward_euler(primitive, dt_fe)
    source = -dt_fe * div_central
    powell_q = fe_q
    for a, k in enumerate(rv.momentum_index):
        powell_q = powell_q.at[k].add(source * B[a])
    for a, k in enumerate(rv.magnetic_index):
        powell_q = powell_q.at[k].add(source * v[a])
    powell_q = powell_q.at[rv.energy_index].add(source * sum(v[a] * B[a] for a in range(3)))
    print(f"FE C_FE={c_fe:5.3f}: p = {float(gas_pressure(fe_q, rv)[centre]):+.3e}   "
          f"with Godunov-Powell source: p = {float(gas_pressure(powell_q, rv)[centre]):+.3e}", flush=True)

# ↓ ———————————————————————————————————————————————————————————————— ↓
# first-order inflow-pair bases at the centre cell (theta = 0 parts)
# ↑ ———————————————————————————————————————————————————————————————— ↑
from astronomix._finite_difference._interface_fluxes._weno_positivity import mhd_physical_flux


def fast_speed(state, axis):
    density = state[rv.density_index]
    field = [state[k] for k in rv.magnetic_index]
    a2 = GAMMA * gas_pressure(state, rv) / density
    b2 = sum(f * f for f in field) / density
    bn2 = field[axis] ** 2 / density
    return jnp.sqrt(0.5 * (a2 + b2 + jnp.sqrt(jnp.maximum((a2 + b2) ** 2 - 4.0 * a2 * bn2, 0.0))))


def inflow_base(state, axis):
    """(alpha_L w+_{i-1} + alpha_R w-_{i+1}) / (alpha_L + alpha_R) at every cell, with
    alpha the stencil maximum of |v_n| + c_f (faces i -+ 1/2)."""
    velocity = state[rv.momentum_index[axis]] / state[rv.density_index]
    radius = jnp.abs(velocity) + fast_speed(state, axis)
    alpha_right = jnp.max(jnp.stack([jnp.roll(radius, -o, axis) for o in (-2, -1, 0, 1, 2, 3)]), axis=0)
    alpha_left = jnp.roll(alpha_right, 1, axis)
    flux = mhd_physical_flux(state, GAMMA, rv, axis)
    # w+_{i-1} at face i-1/2 and w-_{i+1} at face i+1/2
    plus_left = jnp.roll(state, 1, axis + 1) + jnp.roll(flux, 1, axis + 1) / alpha_left[None]
    minus_right = jnp.roll(state, -1, axis + 1) - jnp.roll(flux, -1, axis + 1) / alpha_right[None]
    total = alpha_left + alpha_right
    return (alpha_left[None] * plus_left + alpha_right[None] * minus_right) / total[None], total


print("inflow-pair bases at the centre (theta = 0):")
weighted_sum = 0.0
weight_total = 0.0
for axis in range(3):
    base, total = inflow_base(q, axis)
    p_base = float(gas_pressure(base, rv)[centre])
    # B_n of the two neighbours set to the centre's (pressure of each neighbour held)
    k = rv.magnetic_index[axis]
    equalised = q
    for shift in (1, -1):
        mask = jnp.zeros_like(q[k], dtype=bool).at[tuple(c - shift if a == axis else c for a, c in enumerate(centre))].set(True)
        new_bn = jnp.where(mask, q[k][centre], q[k])
        equalised = equalised.at[rv.energy_index].add(jnp.where(mask, 0.5 * (new_bn**2 - q[k]**2), 0.0))
        equalised = equalised.at[k].set(jnp.where(mask, new_bn, equalised[k]))
    base_eq, _ = inflow_base(equalised, axis)
    p_eq = float(gas_pressure(base_eq, rv)[centre])
    bn = [float(q[k][tuple(c + s if a == axis else c for a, c in enumerate(centre))]) for s in (-1, 0, 1)]
    print(f"  axis {axis}: B_n(i-1,i,i+1)=({bn[0]:+.4f},{bn[1]:+.4f},{bn[2]:+.4f})  p(base)={p_base:+.3e}  "
          f"p(base, equal B_n)={p_eq:+.3e}")
    weighted_sum = weighted_sum + float(total[centre]) * base[:, centre[0], centre[1], centre[2]]
    weight_total += float(total[centre])
summed = weighted_sum / weight_total
print(f"  axis-weighted sum of the bases: p = {float(gas_pressure(summed[:, None, None, None], rv)[0, 0, 0]):+.3e}")
