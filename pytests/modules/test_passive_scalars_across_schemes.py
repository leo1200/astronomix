"""
Passive scalars in every scheme family: a top hat advected once around a periodic box.

The gas moves at a uniform speed, so after one crossing the scalar is back where it started; the L1 distance from
the initial top hat is the error of the (shared, operator-split) advection and the same in every scheme. Checked
for the five hydrodynamic families and for the MHD layouts in 1D (the scalars sit behind the field rows, and the
MHD layouts carry three velocity components in one dimension), with one tolerance: an L1 error below 0.06 at 64
cells and the scalar inside [0, 1] up to 2e-3.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_passive_scalars_across_schemes.py
"""

# numerics
import jax.numpy as jnp
import numpy as np
import pytest

# astronomix containers
from astronomix import PassiveScalarConfig, SimulationParams

# astronomix functions
from astronomix import construct_primitive_state, get_registered_variables

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, MHD_SCHEMES, cell_centres, make_config, run

pytestmark = pytest.mark.x64

CELLS = 64
L1_TOLERANCE = 0.06
BOUND_TOLERANCE = 2e-3


def _advected_top_hat(solver_mode, solver_config, mhd):
    """The top hat after one crossing of the box, and the initial one."""
    config = make_config(
        solver_mode, solver_config, (CELLS,), mhd=mhd, passive_scalar_config=PassiveScalarConfig(num_scalars=1)
    )
    registered_variables = get_registered_variables(config)
    position = cell_centres(CELLS, 1.0)
    top_hat = ((position > 0.25) & (position < 0.5)).astype(float)
    ones = jnp.ones(CELLS)
    fields = dict(density=ones, velocity_x=ones, gas_pressure=ones, passive_scalars=jnp.asarray(top_hat)[None])
    if mhd:
        zeros = jnp.zeros(CELLS)
        fields.update(
            velocity_y=zeros, velocity_z=zeros, magnetic_field_x=zeros, magnetic_field_y=zeros, magnetic_field_z=zeros
        )
        if not isinstance(registered_variables.interface_magnetic_field_index, int):
            fields.update(interface_magnetic_field_x=zeros, interface_magnetic_field_y=zeros,
                          interface_magnetic_field_z=zeros)
    initial_state = construct_primitive_state(config, registered_variables, **fields)
    final_state, _ = run(config, np.asarray(initial_state), SimulationParams(t_end=1.0, C_cfl=0.4))
    return final_state[registered_variables.passive_scalar_index], top_hat


@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_top_hat_hydrodynamics(name):
    scalar, top_hat = _advected_top_hat(*HYDRO_SCHEMES[name], mhd=False)
    error = np.mean(np.abs(scalar - top_hat))
    print(f"{name}: L1 error {error:.3e}, range [{scalar.min():.2e}, {scalar.max():.4f}]")
    assert error < L1_TOLERANCE
    assert scalar.min() > -BOUND_TOLERANCE and scalar.max() < 1.0 + BOUND_TOLERANCE


@pytest.mark.parametrize("name", [name for name, (*_, dimensionality) in MHD_SCHEMES.items() if dimensionality == 1])
def test_top_hat_mhd_layouts(name):
    solver_mode, solver_config, _ = MHD_SCHEMES[name]
    scalar, top_hat = _advected_top_hat(solver_mode, solver_config, mhd=True)
    error = np.mean(np.abs(scalar - top_hat))
    print(f"{name}: L1 error {error:.3e}, range [{scalar.min():.2e}, {scalar.max():.4f}]")
    assert error < L1_TOLERANCE
    assert scalar.min() > -BOUND_TOLERANCE and scalar.max() < 1.0 + BOUND_TOLERANCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
