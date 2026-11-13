"""
3D self-gravitating slab advection correctness pytest (fast).

Advects a self-gravitating density slab across a periodic cubic box at a single
low resolution and checks the final state against the analytic solution for the
three finite-difference self-gravity treatments (simple source, flux-based
source, corrected flux-based source). This is the fast correctness check; the
resolution-sweep convergence figure lives in
``examples/scripts/forward/self_gravity/slab_convergence.py``.
"""

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix import (
    GravityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix.option_classes.simulation_config import (
    StaticFloatVector,
    StaticIntVector,
)

# astronomix functions
from astronomix import (
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.test_setups.self_gravity.slab_advection import (
    setup_slab_advection,
    slab_advection_solution,
)

# Self-gravity source terms are compared against the analytic slab solution, so
# run in double precision to keep round-off well below the tolerance.
pytestmark = pytest.mark.x64


# Cubic box of side length 3 pi (one wavelength along each axis).
BOX_LENGTH = float(3.0 * jnp.pi)
BOX = StaticFloatVector(BOX_LENGTH, BOX_LENGTH, BOX_LENGTH)

# One base configuration per self-gravity treatment.
CONFIG_LIST = [
    SimulationConfig(
        box_size=BOX,
        mhd=False,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=version,
        ),
        dimensionality=3,
        progress_bar=False,
    )
    for version in (SIMPLE_SOURCE, SECOND_ORDER_CONSERVATIVE, FOURTH_ORDER_CONSERVATIVE)
]


def test_slab_advection(N=16, tol=5e-2):
    """Advect the self-gravitating slab at ``N``^3 and check every scheme.

    Each self-gravity treatment is integrated at a single low resolution and its
    final state compared against the analytic slab solution; the mean L1 error
    over the five primitive variables must stay below ``tol``.

    Args:
        N: The per-dimension resolution of the cubic grid.
        tol: The maximum allowed mean L1 error per scheme.
    """
    for base_config in CONFIG_LIST:
        config = base_config._replace(num_cells=StaticIntVector(N, N, N))

        initial_state, config, params = setup_slab_advection(
            config,
            SimulationParams(C_cfl=1.5),
        )

        registered_variables = get_registered_variables(config)
        helper_data = get_helper_data(config)

        final_state = time_integration(
            initial_state, config, params, registered_variables
        )
        true_final_state = slab_advection_solution(
            config, registered_variables, params, helper_data
        )

        indices = (
            registered_variables.density_index,
            registered_variables.velocity_index.x,
            registered_variables.velocity_index.y,
            registered_variables.velocity_index.z,
            registered_variables.pressure_index,
        )
        l1 = jnp.mean(
            jnp.stack([
                jnp.mean(jnp.abs(final_state[i] - true_final_state[i]))
                for i in indices
            ])
        )
        version = base_config.gravity_config.self_gravity_version
        assert l1 < tol, f"slab advection (gravity v{version}) L1 {l1:.3e} exceeds {tol}"


if __name__ == "__main__":
    with jax.enable_x64(True):
        test_slab_advection()
