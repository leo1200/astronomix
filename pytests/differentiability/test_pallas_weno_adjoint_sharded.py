"""
The sharded Pallas adjoint of the hydro WENO flux equals the single-device one.

On a mesh the adjoint runs per x-slab with a 5-cell halo along the flux axis
(``_weno_flux_hydro_pallas_vjp``). Checked on 4 fake CPU devices in Pallas
interpret mode (float64), for every flux axis, with and without dual energy,
with positivity preservation and the frozen linearisation. The conftest splits
the CPU into four devices when JAX runs on the CPU::

    JAX_PLATFORMS=cpu python -m pytest pytests/differentiability/test_pallas_weno_adjoint_sharded.py
"""

# jax
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

# numerics
import numpy as np

# testing
import pytest

# astronomix functions
from astronomix._pallas_helpers import pallas_mesh_context
from astronomix._finite_difference._interface_fluxes._weno_pallas import _weno_flux_hydro_pallas_vjp

# the shared setup of the single-device adjoint test (a sibling module)
import test_pallas_weno_adjoint as single

pytestmark = [pytest.mark.x64, pytest.mark.slow]

NUM_CELLS = 24


@pytest.fixture
def without_shardy():
    """The GSPMD partitioner for the duration of a test; the previous partitioner afterwards."""
    previous = jax.config.jax_use_shardy_partitioner
    jax.config.update("jax_use_shardy_partitioner", False)
    yield
    jax.config.update("jax_use_shardy_partitioner", previous)


@pytest.mark.devices(4)
@pytest.mark.parametrize("dual_energy", [False, True])
def test_sharded_adjoint_equals_single_device(dual_energy, monkeypatch, without_shardy):
    monkeypatch.setattr(single, "NUM_CELLS", NUM_CELLS)
    config, registered_variables = single.flux_config(True, dual_energy, True)
    # x-blocks of 2 cells, so the block-rounded halo (6) fits the 6-cell shards
    config = config._replace(
        backend_config=config.backend_config._replace(pallas_block_shape=(2, 4, 4))
    )
    rng = np.random.default_rng(3)
    state, internal_energy, _ = single.conserved_state(rng, cold_fraction=0.2)
    cotangent = jnp.asarray(rng.standard_normal(state.shape))
    mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Auto,), devices=jax.devices()[:4])
    state_spec = P(None, "x")
    for axis in range(3):

        def adjoint(q, c, g):
            return _weno_flux_hydro_pallas_vjp(
                q, c, single.PARAMS, config, registered_variables, axis=axis,
                internal_energy_density=g)

        reference = adjoint(state, cotangent, internal_energy if dual_energy else None)
        with mesh, pallas_mesh_context(mesh, state_spec):
            state_sharding = NamedSharding(mesh, state_spec)
            sharded = jax.jit(adjoint)(
                jax.device_put(state, state_sharding),
                jax.device_put(cotangent, state_sharding),
                jax.device_put(internal_energy, NamedSharding(mesh, P("x"))) if dual_energy else None,
            )
        references = reference if dual_energy else (reference,)
        results = sharded if dual_energy else (sharded,)
        for result, expected in zip(results, references):
            error = float(jnp.max(jnp.abs(np.asarray(result) - np.asarray(expected))) / jnp.max(jnp.abs(expected)))
            assert error < 1e-9, (axis, error)
