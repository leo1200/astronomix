"""2D smooth-wave accuracy: entropy wave and vortical (shear) wave, advected in x.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/smooth2d.py shear 0.1
"""

# general
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

# ruff: noqa: E402
import numpy as np

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weno_variant import weno_variant_kwargs, weno_variant_name
import jax.numpy as jnp

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    ISOTHERMAL,
    IDEAL_GAS,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)

wave = sys.argv[1]
speed = float(sys.argv[2])
eos = sys.argv[3] if len(sys.argv) > 3 else "ideal"
tag = os.environ.get("LAB_TAG", "run")
periodic = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
errors = []
for n in [16, 32, 64]:
    config = SimulationConfig(
        dimensionality=2,
        num_cells=n,
        equation_of_state=ISOTHERMAL if eos == "iso" else IDEAL_GAS,
        boundary_settings=BoundarySettings(periodic, periodic),
        **weno_variant_kwargs(),
    )
    params = SimulationParams(t_end=1.0 / speed, gamma=5.0 / 3.0, C_cfl=0.8, isothermal_sound_speed=1.0)
    registered_variables = get_registered_variables(config)
    helper = get_helper_data(config)
    x = helper.geometric_centers[..., 0]
    sine = jnp.sin(2 * jnp.pi * x)
    rho = 1.0 + 0.2 * sine if wave == "entropy" else jnp.ones_like(x)
    vy = 0.1 * sine if wave == "shear" else jnp.zeros_like(x)
    kwargs = dict(gas_pressure=jnp.ones_like(x)) if eos != "iso" else {}
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=jnp.full_like(x, speed),
        velocity_y=vy,
        **kwargs,
    )
    config = finalize_config(config, state.shape)
    final = np.asarray(time_integration(state, config, params, registered_variables))
    field = final[0] if wave == "entropy" else final[registered_variables.velocity_index.y]
    reference = np.asarray(rho if wave == "entropy" else vy)
    errors.append(float(np.mean(np.abs(field - reference))))
orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
print(f"{tag:>12s} {eos:5s} {wave:8s} u={speed:4.2f} L1: " + " ".join(f"{e:.2e}" for e in errors)
      + "  orders: " + " ".join(f"{o:.2f}" for o in orders), flush=True)
