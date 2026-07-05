"""
External-potential test (no self-gravity).

A uniform medium (constant rho, p, zero velocity) sits in a *linear* external
gravitational potential  phi(x) = g0 * x.  The acceleration is then constant,
a = -dphi/dx = -g0, everywhere in the interior.  Because the state is uniform
there is no pressure force, so the gas undergoes exact uniform free fall:

    v(t) = -g0 * t,   rho(t) = rho0,   p(t) = p0.

This exercises the external-potential path end to end:
    * config.external_potential = True, config.self_gravity = False
      -> finalize_config sets config.gravity = True, so the gravity source-term
         machinery runs purely from the external potential.
    * _compute_total_potential / _pad_external_potential bring the bare-grid
      potential onto the padded state layout and the source term differentiates
      it for the acceleration.
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from astronomix import CARTESIAN
from astronomix.data_classes.simulation_helper_data import get_helper_data
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    OPEN_BOUNDARY,
    BoundarySettings1D,
    GravityConfig,
    SimulationConfig,
    finalize_config,
)
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import get_registered_variables
from astronomix.time_stepping.time_integration import time_integration

# ---------------------------------------------------------------------------
# problem constants
# ---------------------------------------------------------------------------
NUM_CELLS = 64
BOX_SIZE = 1.0
G0 = 1.0            # potential slope -> acceleration a = -G0
RHO0 = 1.0
P0 = 1.0
GAMMA = 5.0 / 3.0
T_END = 0.05
A_EXPECTED = -G0    # interior acceleration

# ---------------------------------------------------------------------------
# configuration: finite volume, external potential only (no self-gravity)
# ---------------------------------------------------------------------------
config = SimulationConfig(
    dimensionality=1,
    geometry=CARTESIAN,
    box_size=BOX_SIZE,
    num_cells=NUM_CELLS,
    gravity_config=GravityConfig(external_potential=True),
    boundary_settings=BoundarySettings1D(
        left_boundary=OPEN_BOUNDARY,
        right_boundary=OPEN_BOUNDARY,
    ),
)

registered_variables = get_registered_variables(config)
helper_data = get_helper_data(config)
params = SimulationParams(t_end=T_END, gamma=GAMMA)

x = helper_data.geometric_centers           # cell centers on the bare grid
rho = jnp.full_like(x, RHO0)
u = jnp.zeros_like(x)
p = jnp.full_like(x, P0)

state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=rho,
    velocity_x=u,
    gas_pressure=p,
)

# external potential on the *bare* grid (same shape as a state field, no ghosts)
external_phi = G0 * x
params = params._replace(gravitational_potential=external_phi)

config = finalize_config(config, state.shape)
print(f"config.gravity (should be True): {config.gravity}")
assert config.gravity, "finalize_config did not turn on the master gravity flag"

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
final_state = time_integration(state, config, params, registered_variables)

# ---------------------------------------------------------------------------
# check (interior cells only; one-sided stencils at the very edges differ)
# ---------------------------------------------------------------------------
v_final = final_state[registered_variables.velocity_index]
rho_final = final_state[registered_variables.density_index]
p_final = final_state[registered_variables.pressure_index]

interior = slice(8, NUM_CELLS - 8)
v_expected = A_EXPECTED * T_END

v_mean = float(jnp.mean(v_final[interior]))
v_max_err = float(jnp.max(jnp.abs(v_final[interior] - v_expected)))
rho_err = float(jnp.max(jnp.abs(rho_final[interior] - RHO0)))
p_err = float(jnp.max(jnp.abs(p_final[interior] - P0)))

print(f"expected interior velocity v = a*t = {v_expected:.6f}")
print(f"measured  interior velocity (mean) = {v_mean:.6f}")
print(f"max |v - v_expected| (interior)    = {v_max_err:.3e}")
print(f"max |rho - rho0|     (interior)    = {rho_err:.3e}")
print(f"max |p - p0|         (interior)    = {p_err:.3e}")

tol = 1e-4
ok = (v_max_err < tol) and (rho_err < tol) and (p_err < tol)
print("\nRESULT:", "PASS" if ok else "FAIL")
assert ok, "external-potential free-fall test failed"
