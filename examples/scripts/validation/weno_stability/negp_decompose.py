"""Rebuild one forward-Euler update at the failing cell from the PP decomposition.

Captures the code's own inputs to the paired recombination (state, cell flux,
splitting speed, split face fluxes) for the three sweeps, recomputes the
thetas with the code's ``_paired_scalings``, and writes the update of the
centre cell as

    q^{n+1} = (1 - lambda sum_d S_d) q + sum_d (lambda S_d / 2) (own_d + in_d).

It then reports which of the states is not admissible.

    WENO_VARIANT=pp PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/negp_decompose.py FAILSTEP.npz --cfl 0.375 --cfe 0.25
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_ENABLE_X64"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("failstep")
parser.add_argument("--cfl", type=float, default=0.375)
parser.add_argument("--cfe", type=float, default=0.25)
parser.add_argument("--patch", type=int, default=25)
args = parser.parse_args()

# ruff: noqa: E402
import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs

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
from astronomix._fluid_equations._equations_mhd import conserved_state_from_primitive_mhd
from astronomix._finite_difference._interface_fluxes import _weno as weno_module
from astronomix._finite_difference._interface_fluxes import _weno_positivity as pp
from astronomix._finite_difference._magnetic_update._constrained_transport import update_cell_center_fields
from astronomix._finite_difference._timestep import _cfl_time_step_fd
from astronomix.option_classes.simulation_config import IDEAL_GAS

GAMMA = 5.0 / 3.0
data = np.load(args.failstep)
cell = tuple(int(c) for c in data["cell"])
half = args.patch // 2
full = data["state"].astype(np.float64)
patch = np.roll(full, shift=[0] + [half - c for c in cell], axis=(0, 1, 2, 3))[:, :args.patch, :args.patch, :args.patch]
c = half
periodic = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)
config = SimulationConfig(
    equation_of_state=IDEAL_GAS,
    dimensionality=3,
    num_cells=args.patch,
    box_size=args.patch / 256.0,
    mhd=True,
    backend_config=BackendConfig(backend=NATIVE_JAX),
    boundary_settings=BoundarySettings(periodic, periodic, periodic),
    positivity_config=PositivityConfig(clamp_in_estimates=False),
    solver_config=FDSolverConfig(**weno_variant_kwargs()),
)
config = finalize_config(config, patch.shape)
rv = get_registered_variables(config)
params = SimulationParams(C_cfl=args.cfl, gamma=GAMMA, minimum_density=1e-30, minimum_pressure=1e-30)
primitive = jnp.asarray(patch)
q = conserved_state_from_primitive_mhd(primitive[:-3], GAMMA, rv)
q = update_cell_center_fields(q, primitive[-3], primitive[-2], primitive[-1], config, rv)
dx = config.grid_spacing
# the full-domain step's dt (C_cfl = args.cfl), scaled to the forward-Euler Courant number
dt_fe = float(data["dt"]) / args.cfl * args.cfe
lam = dt_fe / dx


def pressure(state):
    return pp._gas_pressure(state, GAMMA, config, rv)


captured = []
original = weno_module.positivity_preserving_interface_flux


def capture(conserved_state, cell_flux, common_speed, plus_face_flux, minus_face_flux, plus_shift, minus_shift,
            params_, config_, rv_, axis=0):
    captured.append(tuple(np.asarray(a) for a in (conserved_state, cell_flux, common_speed, plus_face_flux,
                                                  minus_face_flux)))
    return original(conserved_state, cell_flux, common_speed, plus_face_flux, minus_face_flux, plus_shift,
                    minus_shift, params_, config_, rv_, axis)


weno_module.positivity_preserving_interface_flux = capture
with jax.disable_jit():
    fluxes = [weno_module._weno_flux_native_for_axis(a)(q, params, config, rv) for a in range(3)]
assert len(captured) == 3, len(captured)

total_update = q - lam * sum(f - jnp.roll(f, 1, axis=a + 1) for a, f in enumerate(fluxes))
print(f"p(q) = {float(pressure(q)[c, c, c]):+.3e}; forward Euler (lambda from C_FE={args.cfe}) p = "
      f"{float(pressure(total_update)[c, c, c]):+.3e}")

rebuilt = None
weight_sum = 0.0
own_terms, inflow_terms = [], []
for axis, (state, cell_flux, speed, plus_face, minus_face) in enumerate(captured):
    # sweep-local layout: the active axis leads; the centre is the same index
    state, cell_flux, speed = jnp.asarray(state), jnp.asarray(cell_flux), jnp.asarray(speed)
    alpha = jnp.maximum(speed, 1e-30)[None]
    plus_w, minus_w = pp._upwind_split_states(state, cell_flux, speed)
    plus_step = 2.0 * jnp.asarray(plus_face) / alpha - plus_w
    minus_step = -2.0 * jnp.asarray(minus_face) / alpha - minus_w
    plus_theta, minus_theta = pp._paired_scalings(state, speed, plus_w, minus_w, plus_step, minus_step,
                                                  params, config, rv)
    a_right, a_left = float(speed[c, c, c]), float(speed[c - 1, c, c])
    total = a_right + a_left

    def at(field, i):
        return field[:, i, c, c]

    t_plus_r, t_minus_r = float(plus_theta[c, c, c]), float(minus_theta[c, c, c])
    t_plus_l, t_minus_l = float(plus_theta[c - 1, c, c]), float(minus_theta[c - 1, c, c])
    mirror_plus = at(plus_w, c) - t_plus_r * at(plus_step, c)          # face i+1/2, cell i
    mirror_minus = at(minus_w, c - 1) - t_minus_l * at(minus_step, c - 1)  # face i-1/2, cell i
    own = (a_right * mirror_plus + a_left * mirror_minus) / total
    face_plus_left = at(plus_w, c - 1) + t_plus_l * at(plus_step, c - 1)  # W+ at i-1/2
    face_minus_right = at(minus_w, c) + t_minus_r * at(minus_step, c)     # W- at i+1/2
    inflow = (a_left * face_plus_left + a_right * face_minus_right) / total
    inflow_base = (a_left * at(plus_w, c - 1) + a_right * at(minus_w, c)) / total

    def p1(v):
        return float(pressure(v[:, None, None, None])[0, 0, 0])

    print(f"axis {axis}: alpha_L={a_left:.3f} alpha_R={a_right:.3f} thetas(+L,-L,+R,-R)=({t_plus_l:.2f},{t_minus_l:.2f},"
          f"{t_plus_r:.2f},{t_minus_r:.2f})  p(own)={p1(own):+.3e} p(inflow)={p1(inflow):+.3e} "
          f"p(inflow base)={p1(inflow_base):+.3e}")
    # back to the untransposed component order (the y / z sweeps swap x <-> y / z)
    if axis > 0:
        swap = {1: "y", 2: "z"}[axis]
        for index in (rv.momentum_index, rv.magnetic_index):
            i_x, i_o = index.x, getattr(index, swap)
            own = own.at[jnp.array([i_x, i_o])].set(own[jnp.array([i_o, i_x])])
            inflow = inflow.at[jnp.array([i_x, i_o])].set(inflow[jnp.array([i_o, i_x])])
    own_terms.append((total, own))
    inflow_terms.append((total, inflow))
    weight_sum += total

centre_q = q[:, c, c, c]
rebuilt = (1.0 - lam * weight_sum) * centre_q
for (total, own), (_, inflow) in zip(own_terms, inflow_terms):
    rebuilt = rebuilt + 0.5 * lam * total * (own + inflow)
inflow_sum = sum(t * v for t, v in inflow_terms) / weight_sum
own_sum = sum(t * v for t, v in own_terms) / weight_sum


def p1(v):
    return float(pressure(v[:, None, None, None])[0, 0, 0])


print(f"rebuilt update p = {p1(rebuilt):+.3e} (max |rebuilt - forward Euler| = "
      f"{float(jnp.max(jnp.abs(rebuilt - total_update[:, c, c, c]))):.1e}); lambda*sum S = {lam * weight_sum:.3f}")
print(f"axis-weighted own sum p = {p1(own_sum):+.3e}; axis-weighted inflow sum p = {p1(inflow_sum):+.3e}")
