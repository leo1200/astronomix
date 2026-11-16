"""
The eigenvectors of the MHD flux Jacobian at and near the triple point B_t = 0, v_A = a, where the
fast, the Alfven and the slow speed coincide: the rows L and the columns R of the characteristic
basis stay inverse to each other.

Checks max |L R - I| of the basis of the isothermal and of the ideal gas, in float32 and float64,
in the array form (``_riemann_flux._characteristic_basis``) and in the form of the kernels (the
functions of ``_riemann_flux_pallas`` on plain arrays), on interface states with the normal field
B_x = a sqrt(rho) (1 + d) and the transverse field |B_t| = a sqrt(rho) t. L R = I needs mode
weights with a_f^2 + a_s^2 = 1 (``_eigen_mhd._mode_weightings``).

    JAX_PLATFORMS=cpu python -m pytest pytests/mhd/test_eigenvectors_at_the_triple_point.py
"""

# general
import itertools

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import FINITE_DIFFERENCE, IDEAL_GAS, ISOTHERMAL

# astronomix containers
from astronomix.option_classes.presets import FD_RIEMANN_IDEAL_GAS, FD_RIEMANN_ISOTHERMAL
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix._finite_difference._interface_fluxes import _riemann_flux as array_form
from astronomix._finite_difference._interface_fluxes import _riemann_flux_pallas as kernel_form
from astronomix.variable_registry.registered_variables import get_registered_variables

GAMMA = 5.0 / 3.0
#: (density, sound speed) of the gases.
GASES = ((1.0, 1.0), (2.5, 0.1))
#: The detunings d of the normal field, B_x = a sqrt(rho) (1 + d).
DETUNINGS = (0.0, 1e-7, -1e-7, 1e-5, -1e-5, 1e-3, -1e-3)
#: The transverse fields t, |B_t| = a sqrt(rho) t.
TRANSVERSE = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)
#: (B_y, B_z) / |B_t|: both components, and y alone.
DIRECTIONS = ((0.6, 0.8), (1.0, 0.0))
#: The velocity in units of the sound speed.
VELOCITY = (0.3, -0.2, 0.1)

BOTH_GASES = pytest.mark.parametrize("equation_of_state", [ISOTHERMAL, IDEAL_GAS], ids=["isothermal", "ideal_gas"])
BOTH_PRECISIONS = pytest.mark.parametrize("x64", [False, True], ids=["float32", "float64"])
BOTH_FORMS = pytest.mark.parametrize("form", ["array_form", "kernels"])

# Bounds of max |L R - I|. Measured on the states of this test and, for the bound over all states, on a finer
# grid of 1062 states per gas (d = 0 and +-10^-8 to +-10^-1, t = 0 and 10^-8 to 10^-1); the two forms agree to 1 %:
#
#                     at the triple point    over all states    finer grid
#   isothermal float32      1.2e-7               1.4e-5           8.9e-5
#   isothermal float64      2.2e-16              2.0e-11          5.0e-9
#   ideal gas  float32      1.2e-7               2.0e-4           2.5e-4
#   ideal gas  float64      2.4e-14              1.0e-8           1.2e-8
#
# At the triple point the basis is exact to rounding. Next to it the defect is the rounding of the
# discriminant of the wave speeds, (b^2 + a^2)^2 - 4 b_x^2 a^2, which leaves the weighted sum
# a_f^2 c_f^2 + a_s^2 c_s^2 off a^2 by up to the square root of the rounding unit; it depends on the rounding of
# single operations, so the bounds over all states are four times the values of the finer grid. Mode weights
# that are not normalised give 1 in float32 (at the triple point and for d = 1e-5) and 2e-4 in float64
# (d = -1e-7).
TRIPLE_POINT_BOUND = {False: 5e-7, True: 1e-13}
DEFECT_BOUND = {
    (ISOTHERMAL, False): 4e-4,
    (ISOTHERMAL, True): 2e-8,
    (IDEAL_GAS, False): 1e-3,
    (IDEAL_GAS, True): 5e-8,
}


def _config_and_params(equation_of_state, sound_speed):
    """A 3D finite-difference MHD configuration with the Riemann-flux preset of the gas, and its parameters."""
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        dimensionality=3,
        mhd=True,
        equation_of_state=equation_of_state,
        num_cells=16,
        box_size=1.0,
        solver_config=FD_RIEMANN_IDEAL_GAS if equation_of_state == IDEAL_GAS else FD_RIEMANN_ISOTHERMAL,
        progress_bar=False,
    )
    if equation_of_state == IDEAL_GAS:
        return config, SimulationParams(gamma=GAMMA, minimum_density=1e-10, minimum_pressure=1e-12)
    return config, SimulationParams(minimum_density=1e-10, isothermal_sound_speed=sound_speed)


def _interface_states(density, sound_speed, dtype):
    """
    The interface states of the grid of detunings, transverse fields and directions.

    Returns:
        The arrays (rho, v_x, v_y, v_z, B_x, B_y, B_z, a^2), one entry per state, and the mask of the states
        at the triple point itself.
    """
    grid = list(itertools.product(DETUNINGS, TRANSVERSE, DIRECTIONS))
    detuning = np.array([d for d, _, _ in grid], dtype=np.float64)
    transverse = np.array([t for _, t, _ in grid], dtype=np.float64)
    cosine, sine = (np.array([direction[axis] for _, _, direction in grid], dtype=np.float64) for axis in (0, 1))
    ones = np.ones(len(grid), dtype=dtype)
    sound_speed = np.asarray(sound_speed, dtype=dtype)
    unit_field = sound_speed * np.sqrt(np.asarray(density, dtype=dtype))
    fields = (
        density * ones,
        *(sound_speed * component * ones for component in VELOCITY),
        unit_field * (1.0 + detuning).astype(dtype),
        unit_field * (transverse * cosine).astype(dtype),
        unit_field * (transverse * sine).astype(dtype),
        sound_speed * sound_speed * ones,
    )
    at_triple_point = (detuning == 0.0) & (transverse == 0.0)
    return tuple(jnp.asarray(field, dtype=dtype) for field in fields), at_triple_point


def _array_form_defect(equation_of_state, sound_speed, fields):
    """max |L R - I| per state of the array form; the product is formed in float64 from the entries."""
    config, params = _config_and_params(equation_of_state, sound_speed)
    registered_variables = get_registered_variables(config)
    # the cell-centred rows: density, momentum, field and (ideal gas) energy
    num_rows = 8 if equation_of_state == IDEAL_GAS else 7
    conserved = jnp.zeros((num_rows,) + fields[0].shape, dtype=fields[0].dtype)
    setup = array_form._scheme_setup(conserved, params, config, registered_variables)
    bar = array_form.InterfaceState(*fields)
    rows, columns = array_form._characteristic_basis(conserved, bar, setup, registered_variables)
    rows = np.stack([np.asarray(row, dtype=np.float64) for row in rows])
    columns = np.stack([np.asarray(column, dtype=np.float64) for column in columns])
    product = np.einsum("svn,tvn->stn", rows, columns)
    return np.max(np.abs(product - np.eye(len(rows))[:, :, None]), axis=(0, 1))


def _kernel_form_defect(equation_of_state, sound_speed, fields):
    """max |L R - I| per state of the functions of the kernels: unit characteristic vectors there and back."""
    config, params = _config_and_params(equation_of_state, sound_speed)
    rho, _, _, _, B_x, B_y, B_z, a2 = fields
    setup = kernel_form._kernel_setup(kernel_form._kernel_scalars(params, config, rho.dtype), config)
    basis = kernel_form._characteristic_basis(kernel_form.InterfaceState(rho, B_x, B_y, B_z, a2), setup)
    num_waves = 7 if equation_of_state == IDEAL_GAS else 6
    zero, one = jnp.zeros_like(rho), jnp.ones_like(rho)
    defect = np.zeros(rho.shape)
    for wave in range(num_waves):
        unit = tuple(one if index == wave else zero for index in range(num_waves))
        primitives = kernel_form._primitives_from_characteristic(unit, basis, setup)
        back = kernel_form._characteristic_variables(primitives, basis, setup)
        for index in range(num_waves):
            entry = np.asarray(back[index], dtype=np.float64) - float(index == wave)
            defect = np.maximum(defect, np.abs(entry))
    return defect


@BOTH_FORMS
@BOTH_PRECISIONS
@BOTH_GASES
def test_rows_and_columns_are_inverse_at_the_triple_point(equation_of_state, x64, form):
    """L R = I at the triple point to rounding, and near it to the rounding of the wave speeds."""
    defect_of_form = _array_form_defect if form == "array_form" else _kernel_form_defect
    with jax.enable_x64(x64):
        dtype = np.float64 if x64 else np.float32
        for density, sound_speed in GASES:
            fields, at_triple_point = _interface_states(density, sound_speed, dtype)
            defect = defect_of_form(equation_of_state, sound_speed, fields)
            print(
                f"basis eos {equation_of_state} {form} {dtype.__name__} rho {density} a {sound_speed}:"
                f" max |L R - I| {np.max(defect[at_triple_point]):.3e} at the triple point,"
                f" {np.max(defect):.3e} over all states"
            )
            assert np.all(np.isfinite(defect))
            assert np.max(defect[at_triple_point]) < TRIPLE_POINT_BOUND[x64]
            assert np.max(defect) < DEFECT_BOUND[equation_of_state, x64]
