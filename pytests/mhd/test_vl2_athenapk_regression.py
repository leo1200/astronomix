"""
Regression test of the VL2 finite-volume scheme against AthenaPK.

Small runs of AthenaPK (VL2 + PLM + HLLD / HLLE / HLLC with GLM divergence
cleaning, first-order flux correction, periodic and outflow boundaries in 1D, 2D
and 3D) are stored in ``pytests/mhd/data/athenapk_vl2`` (written by
``examples/scripts/validation/athenapk_vl2/make_regression_data.py``). Every case
is re-run here from AthenaPK's initial state, and astronomix has to take the same
number of cycles and agree with AthenaPK's final state to round-off.

The tolerance (relative L1 difference per variable below 1e-9) is several orders
of magnitude above the round-off differences actually observed (1e-15 to 1e-12)
and several orders below the effect of any change to the discretization.
"""

# general
import os

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import NATIVE_JAX, PALLAS

# astronomix containers
from astronomix import SnapshotSettings

# astronomix functions
from astronomix import time_integration
from astronomix.test_setups.mhd.athenapk_vl2_cases import (
    REGRESSION_CASES,
    astronomix_setup,
)

pytestmark = pytest.mark.x64

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "athenapk_vl2")

#: Largest accepted relative L1 difference to AthenaPK, per variable.
RELATIVE_TOLERANCE = 1e-9


def _available_backends():
    """The native backend always; Pallas as well when a GPU is present."""
    backends = [NATIVE_JAX]
    if jax.default_backend() == "gpu":
        backends.append(PALLAS)
    return backends


@pytest.mark.parametrize("backend", _available_backends(), ids=lambda backend: "pallas" if backend == PALLAS else "native")
@pytest.mark.parametrize("case", REGRESSION_CASES, ids=lambda case: case.name)
def test_vl2_matches_athenapk(case, backend):
    reference = np.load(os.path.join(DATA_DIRECTORY, f"{case.name}.npz"))
    initial_state = jnp.asarray(reference["initial_primitive_state"])

    config, params, registered_variables = astronomix_setup(case, initial_state.shape, backend)
    config = config._replace(
        return_snapshots=True,
        num_snapshots=1,
        snapshot_settings=SnapshotSettings(return_final_state=True),
    )
    result = time_integration(initial_state, config, params, registered_variables)

    final_state = np.asarray(result.final_state)
    reference_state = reference["final_primitive_state"]
    axes = tuple(range(1, reference_state.ndim))
    relative_l1 = np.sum(np.abs(final_state - reference_state), axis=axes) / np.maximum(
        np.sum(np.abs(reference_state), axis=axes), 1e-300
    )

    assert int(result.num_iterations) == int(reference["cycles"])
    assert np.all(relative_l1 < RELATIVE_TOLERANCE), relative_l1
