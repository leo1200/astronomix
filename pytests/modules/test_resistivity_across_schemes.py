"""
Resistivity in every MHD scheme family: the resistive decay of a force-free field.

The field ``B = B_0 (0, sin(k x), cos(k x))`` has a uniform magnitude and ``J = k B``, so it exerts no force; under
an ohmic diffusivity ``eta`` it decays as ``exp(-eta k^2 t)`` and the gas stays at rest. The schemes damp the
field by their own numerical dissipation as well (the HLLD solver of VL2 by 0.9 % here), so each family's run
with ``eta`` is compared with the same run without it: the ratio of the amplitudes matches the exponential to
2e-3 in every family at 32 cells (VL2 reaches 1.4e-3, 2.8e-4 and 3.3e-5 at 32, 64 and 128 cells: its numerical
dissipation, not the resistive term; the other families 7e-6 and below). For the ideal gas the field energy the
resistivity takes heats the gas (ohmic heating in the production finite-difference scheme, the resistive
Poynting flux elsewhere): the thermal energy gained equals the field energy lost to 1e-4. The resistive field
enters the edge electromotive force of every constrained transport (FD production, FD flux form, VL2 with
``CONSTRAINED_TRANSPORT``) and the field rows of the cell-centred schemes (VL2 with GLM, the split scheme with
the induction source).

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_resistivity_across_schemes.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import jax.numpy as jnp
import numpy as np
import pytest

# astronomix constants
from astronomix.option_classes.simulation_config import IDEAL_GAS, ISOTHERMAL

# astronomix containers
from astronomix import DiffusionConfig, SimulationParams

# astronomix functions
from astronomix import construct_primitive_state, get_registered_variables

# the scheme families of the module tests
from _schemes import MHD_SCHEMES, cell_centres, make_config, mode_amplitude, run

GAMMA = 5.0 / 3.0
RESISTIVITY = 0.01
FIELD_STRENGTH = 0.1
WAVENUMBER = 2.0 * np.pi
T_END = 0.5
CELLS = 32
RATIO_TOLERANCE = 2e-3
ENERGY_TOLERANCE = 1e-4

CASES = [(name, IDEAL_GAS) for name in MHD_SCHEMES] + [
    (name, ISOTHERMAL) for name, (*_, dimensionality) in MHD_SCHEMES.items() if dimensionality == 1
]


def _final_state(config, registered_variables, initial_state, resistivity):
    params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=GAMMA, resistivity=resistivity)
    final_state, _ = run(config, initial_state, params)
    return final_state


@pytest.mark.parametrize("name, equation_of_state", CASES, ids=[f"{name}, eos {eos}" for name, eos in CASES])
def test_force_free_decay(name, equation_of_state):
    solver_mode, solver_config, dimensionality = MHD_SCHEMES[name]
    shape = (CELLS,) + (4,) * (dimensionality - 1)
    config = make_config(
        solver_mode,
        solver_config,
        shape,
        mhd=True,
        equation_of_state=equation_of_state,
        diffusion_config=DiffusionConfig(resistivity=True),
    )
    registered_variables = get_registered_variables(config)
    position = cell_centres(CELLS, 1.0)
    phase = np.broadcast_to(WAVENUMBER * position.reshape((CELLS,) + (1,) * (dimensionality - 1)), shape)
    field_y = jnp.asarray(FIELD_STRENGTH * np.sin(phase))
    field_z = jnp.asarray(FIELD_STRENGTH * np.cos(phase))
    zeros = jnp.zeros(shape)
    fields = dict(
        density=jnp.ones(shape),
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        magnetic_field_x=zeros,
        magnetic_field_y=field_y,
        magnetic_field_z=field_z,
    )
    if equation_of_state == IDEAL_GAS:
        fields["gas_pressure"] = jnp.ones(shape)
    if not isinstance(registered_variables.interface_magnetic_field_index, int):
        # The field varies along x only: the x-faces carry zero, the other faces the cell values.
        fields.update(interface_magnetic_field_x=zeros, interface_magnetic_field_y=field_y,
                      interface_magnetic_field_z=field_z)
    initial_state = np.asarray(construct_primitive_state(config, registered_variables, **fields))

    resistive_state = _final_state(config, registered_variables, initial_state, RESISTIVITY)
    ideal_state = _final_state(config, registered_variables, initial_state, 0.0)

    magnetic_index = registered_variables.magnetic_index
    expected_ratio = np.exp(-RESISTIVITY * WAVENUMBER**2 * T_END)
    for component, phase_shift in ((magnetic_index.y, 0.0), (magnetic_index.z, 0.5 * np.pi)):
        ratio = mode_amplitude(resistive_state[component], position, WAVENUMBER, phase_shift) / mode_amplitude(
            ideal_state[component], position, WAVENUMBER, phase_shift
        )
        deviation = ratio / expected_ratio - 1.0
        print(f"{name}, eos {equation_of_state}: amplitude ratio off the exponential by {deviation:+.2e}")
        assert abs(deviation) < RATIO_TOLERANCE

    if equation_of_state == IDEAL_GAS:
        thermal_gain = (resistive_state[registered_variables.pressure_index].mean() - 1.0) / (GAMMA - 1.0)
        field_energy = 0.5 * sum(resistive_state[index] ** 2 for index in tuple(magnetic_index)).mean()
        field_loss = 0.5 * FIELD_STRENGTH**2 - field_energy
        print(f"{name}: heat gained over field energy lost {thermal_gain / field_loss - 1:+.2e}")
        assert abs(thermal_gain / field_loss - 1.0) < ENERGY_TOLERANCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
