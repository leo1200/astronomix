"""
Step retry pytest (``PositivityConfig.step_retry``).

A near-vacuum double rarefaction run at a time step far beyond what the
positivity-preserving WENO tolerates loses positivity and turns non-finite
without the retry, and stays finite and positive with it; at an admissible
time step the retry never triggers and leaves the run unchanged. The switch is
rejected where there is no step size to reduce.

The case where the retry matters in practice (a near-vacuum cell of
weak-field Mach-10 turbulence that accelerates within a single step) needs a
3D turbulent run; see ``examples/scripts/validation/ct_upwind_dissipation``.

    JAX_PLATFORMS=cpu python -m pytest pytests/hydrodynamics/test_step_retry.py
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import OPEN_BOUNDARY

# astronomix containers
from astronomix import (
    BoundarySettings1D,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

jax.config.update("jax_enable_x64", True)


def _run_double_rarefaction(C_cfl, step_retry):
    """
    Run the double rarefaction u = -+3.7 (vacuum at u = -+3.74, gamma 1.4) to
    t = 0.1 on 400 cells with the positivity-preserving WENO.

    Args:
        C_cfl: The CFL number.
        step_retry: Whether steps are retried.

    Returns:
        The final primitive state and the registered variables.
    """
    config = SimulationConfig(
        dimensionality=1,
        num_cells=400,
        boundary_settings=BoundarySettings1D(
            left_boundary=OPEN_BOUNDARY,
            right_boundary=OPEN_BOUNDARY,
        ),
        weno_positivity_preserving=True,
        positivity_config=PositivityConfig(step_retry=step_retry),
    )
    params = SimulationParams(t_end=0.1, gamma=1.4, C_cfl=C_cfl)
    registered_variables = get_registered_variables(config)
    x = get_helper_data(config).geometric_centers
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.ones_like(x),
        velocity_x=jnp.where(x < 0.5, -3.7, 3.7),
        gas_pressure=jnp.full_like(x, 0.4),
    )
    config = finalize_config(config, state.shape)
    final = np.asarray(time_integration(state, config, params, registered_variables))
    return final, registered_variables


def test_retry_rescues_oversized_steps():
    """At C_cfl 2.5 the plain run turns non-finite, the retried one stays positive."""
    plain, _ = _run_double_rarefaction(2.5, step_retry=False)
    assert not np.all(np.isfinite(plain))
    retried, registered_variables = _run_double_rarefaction(2.5, step_retry=True)
    assert np.all(np.isfinite(retried))
    assert retried[registered_variables.density_index].min() > 0.0
    assert retried[registered_variables.pressure_index].min() > 0.0


def test_retry_inactive_at_admissible_steps():
    """At C_cfl 0.4 no step is retried: the run is unchanged."""
    plain, _ = _run_double_rarefaction(0.4, step_retry=False)
    retried, _ = _run_double_rarefaction(0.4, step_retry=True)
    np.testing.assert_allclose(retried, plain, rtol=1e-12, atol=1e-14)


def test_retry_rejected_for_fixed_steps():
    """A fixed time step leaves nothing to reduce."""
    config = SimulationConfig(
        dimensionality=1,
        num_cells=16,
        fixed_timestep=True,
        positivity_config=PositivityConfig(step_retry=True),
    )
    registered_variables = get_registered_variables(config)
    with pytest.raises(ValueError, match="step_retry"):
        finalize_config(config, (registered_variables.num_vars, 16))


if __name__ == "__main__":
    test_retry_rescues_oversized_steps()
    test_retry_inactive_at_admissible_steps()
    test_retry_rejected_for_fixed_steps()
