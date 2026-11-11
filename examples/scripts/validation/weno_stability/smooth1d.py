"""Smooth-wave accuracy: does the scheme change cost accuracy in smooth flow?

Advected entropy (density) wave in 1D, rho = 1 + 0.2 sin(2 pi x), u = U, p = 1,
one period on a periodic box. The exact solution is the initial state. A
low-Mach advection speed is the worst case for a single (scalar) splitting
speed, because the entropy field's own speed |u| is then much smaller than
|u| + c.

    PYTHONPATH=. JAX_PLATFORMS=cpu python examples/scripts/validation/weno_stability/smooth1d.py
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
    PERIODIC_BOUNDARY,
    PERIODIC_ROLL,
    BoundarySettings1D,
)

tag = os.environ.get("LAB_TAG", "run")
speeds = [float(s) for s in (sys.argv[1:] or ["1.0", "0.1"])]
for speed in speeds:
    errors = []
    for n in [16, 32, 64, 128]:
        config = SimulationConfig(
            dimensionality=1,
            num_cells=n,
            **weno_variant_kwargs(),
            boundary_settings=BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        )
        params = SimulationParams(t_end=1.0 / speed, gamma=5.0 / 3.0, C_cfl=0.4)
        registered_variables = get_registered_variables(config)
        x = get_helper_data(config).geometric_centers
        rho = 1.0 + 0.2 * jnp.sin(2 * jnp.pi * x)
        state = construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=rho,
            velocity_x=jnp.full_like(x, speed),
            gas_pressure=jnp.ones_like(x),
        )
        config = finalize_config(config, state.shape)
        final = np.asarray(time_integration(state, config, params, registered_variables))
        errors.append(float(np.mean(np.abs(final[0] - np.asarray(rho)))))
    orders = [np.log2(errors[k] / errors[k + 1]) for k in range(len(errors) - 1)]
    print(f"{tag:>10s} u={speed:4.2f} (Mach {speed / np.sqrt(5 / 3):.2f}) L1: "
          + " ".join(f"{e:.2e}" for e in errors)
          + "  orders: " + " ".join(f"{o:.2f}" for o in orders), flush=True)
