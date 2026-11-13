"""
Self-gravity in every scheme family: a Jeans wave below the Jeans length.

A standing adiabatic density wave ``rho = rho_0 (1 + eps cos(k x))`` at rest oscillates with
``omega^2 = c_s^2 k^2 - 4 pi G rho_0``; here gravity lowers the frequency of the sound wave by a factor sqrt(2).
After 0.6 periods the mode amplitude is ``eps cos(omega t) = -0.81 eps`` (without gravity it would be +0.58 eps).
Gravity enters the stages of every scheme but VL2 through the explicit sources (``SIMPLE_SOURCE`` for the
finite-volume ones, the default coupling of the finite-difference ones); VL2 keeps Athena++'s coupling. The MHD
layouts (without field) of the families in 1D run the same wave. Every family reaches the linear amplitude to
2e-2 of ``eps`` at 64 cells (the finite-volume ones are second order, and
RK2_SSP and MUSCL_HANCOCK run with the minmod limiter that ``finalize_config`` sets for gravity). The periodic
finite-volume boxes of RK2_SSP and MUSCL_HANCOCK have ghost cells: the Poisson equation is solved on the domain
and the potential wrapped into them.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/modules/test_gravity_across_schemes.py
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

# astronomix containers
from astronomix import GravityConfig, SimulationParams

# astronomix functions
from astronomix import construct_primitive_state, get_registered_variables

# the scheme families of the module tests
from _schemes import HYDRO_SCHEMES, MHD_SCHEMES, cell_centres, make_config, mode_amplitude, run

GAMMA = 5.0 / 3.0
EPSILON = 1e-4
WAVENUMBER = 2.0 * np.pi
SOUND_SPEED_SQUARED = GAMMA
GRAVITATIONAL_CONSTANT = 0.5 * SOUND_SPEED_SQUARED * WAVENUMBER**2 / (4.0 * np.pi)
FREQUENCY = np.sqrt(0.5 * SOUND_SPEED_SQUARED) * WAVENUMBER
T_END = 0.6 * 2.0 * np.pi / FREQUENCY
CELLS = 64
TOLERANCE = 2e-2


def _deviation(solver_mode, solver_config, mhd):
    """The amplitude of the wave at the end time off linear theory, in units of ``eps``."""
    config = make_config(
        solver_mode, solver_config, (CELLS,), mhd=mhd, gravity_config=GravityConfig(self_gravity=True)
    )
    registered_variables = get_registered_variables(config)
    position = cell_centres(CELLS, 1.0)
    zeros = jnp.zeros(CELLS)
    fields = dict(
        density=jnp.asarray(1.0 + EPSILON * np.cos(WAVENUMBER * position)),
        velocity_x=zeros,
        gas_pressure=jnp.asarray(1.0 + GAMMA * EPSILON * np.cos(WAVENUMBER * position)),
    )
    if mhd:
        fields.update(
            velocity_y=zeros, velocity_z=zeros, magnetic_field_x=zeros, magnetic_field_y=zeros, magnetic_field_z=zeros
        )
        if not isinstance(registered_variables.interface_magnetic_field_index, int):
            fields.update(
                interface_magnetic_field_x=zeros, interface_magnetic_field_y=zeros, interface_magnetic_field_z=zeros
            )
    initial_state = np.asarray(construct_primitive_state(config, registered_variables, **fields))
    params = SimulationParams(t_end=T_END, C_cfl=0.4, gamma=GAMMA, gravitational_constant=GRAVITATIONAL_CONSTANT)
    final_state, _ = run(config, initial_state, params)
    amplitude = mode_amplitude(final_state[registered_variables.density_index], position, WAVENUMBER, 0.5 * np.pi)
    return (amplitude - EPSILON * np.cos(FREQUENCY * T_END)) / EPSILON


@pytest.mark.parametrize("name", list(HYDRO_SCHEMES))
def test_jeans_wave(name):
    deviation = _deviation(*HYDRO_SCHEMES[name], mhd=False)
    print(f"{name}: off linear theory by {deviation:+.2e} eps")
    assert abs(deviation) < TOLERANCE


@pytest.mark.parametrize("name", [name for name, (*_, dimensionality) in MHD_SCHEMES.items() if dimensionality == 1])
def test_jeans_wave_mhd_layouts(name):
    solver_mode, solver_config, _ = MHD_SCHEMES[name]
    deviation = _deviation(solver_mode, solver_config, mhd=True)
    print(f"{name} (MHD layout): off linear theory by {deviation:+.2e} eps")
    assert abs(deviation) < TOLERANCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
