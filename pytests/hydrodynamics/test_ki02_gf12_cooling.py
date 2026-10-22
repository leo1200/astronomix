"""
KI02 + Gnat & Ferland (2012) cooling in TIGRESS conventions (CPU, seconds).

Guards ``ki02_gf12_cooling`` and the temperature-dependent heating factor
(``CoolingParams.heating_log10_T_table`` / ``heating_factor_table``):

1. Normalisation and units, against independent physical formulas: the
   kernel's volumetric cooling must equal n_H^2 * KI02(T) (corrected constant
   1.184e5) in cold gas and n_H^2 * Lambda_GF12(T) in hot gas, with
   n_H = rho / (1.4271 m_H) and the physical T(T1) of the mu(T) mapping.
2. The heating factor: f = 1 in neutral gas, f -> 0 in ionised gas, and empty
   tables reproduce the old constant heating exactly (backwards compatible).
3. Thermal equilibrium at n_H = 1 cm^-3, heat_ratio = 1: T = 5010 K.
4. Isochoric cooling from 1e7 K through ``update_pressure_by_cooling``:
   crossing time and end state against an independent adaptive reference.

Runs in float32, the precision of the existing cooling tests.
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# units and constants
import astropy.units as u

# astronomix constants
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    PIECEWISE_POWER_LAW,
)
from astronomix._modules._cooling._cooling_tables import (
    _GF12_TABLE_PATH,
    _HYDROGEN_MASS_CGS,
    TIGRESS_MU_H,
)

# astronomix containers
from astronomix import (
    SimulationConfig,
    SimulationParams,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)
from astronomix.units.unit_helpers import CodeUnits

# astronomix functions
from astronomix import get_registered_variables
from astronomix._modules._cooling._cooling import (
    dtemperature_dt,
    get_effective_molecular_weights,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
    update_pressure_by_cooling,
)
from astronomix._modules._cooling._cooling_tables import (
    ki02_gf12_cooling,
    ki02_gf12_temperature,
)

GAMMA = 5.0 / 3.0
X, Z = 0.7, 0.02
CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
MYR = 3.156e13

SETUP = ki02_gf12_cooling(CODE_UNITS, X, Z)
MU, _, _ = get_effective_molecular_weights(X, Z)
RHO_U = (1 * CODE_UNITS.code_density).to(u.g / u.cm**3).value
P_U = (1 * CODE_UNITS.code_pressure).to(u.erg / u.cm**3).value
T_U = (1 * CODE_UNITS.code_time).to(u.s).value

CURVE_CONFIG = CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW)


def _as_float32(tree):
    """
    Cast the array leaves of a pytree to float32.

    Other test modules switch x64 on globally when they are imported, which makes
    the tables float64 while the test states stay float32; the implicit solver
    cannot mix the two, so the tables are pinned to the precision of the states.
    """
    return jax.tree_util.tree_map(
        lambda leaf: leaf.astype(jnp.float32) if isinstance(leaf, jax.Array) else leaf,
        tree,
    )


def _cooling_params(heating=True):
    """CoolingParams of the KI02 + GF12 setup, optionally without heating."""
    return CoolingParams(
        hydrogen_mass_fraction=X,
        metal_mass_fraction=Z,
        floor_temperature=SETUP.floor_temperature,
        clamp_to_floor=True,
        heating_rate=SETUP.heating_rate if heating else 0.0,
        heating_log10_T_table=_as_float32(SETUP.heating_log10_T_table),
        heating_factor_table=_as_float32(SETUP.heating_factor_table),
        cooling_curve_params=_as_float32(SETUP.cooling_curve_params),
    )


def _physical_T_to_code(T_phys):
    """Code temperature T~ of gas with physical temperature T (inverts T(T1))."""
    T1_grid = np.logspace(0.9, 9.0, 20001)
    T_grid = ki02_gf12_temperature(T1_grid)
    T1 = 10 ** np.interp(np.log10(T_phys), np.log10(T_grid), np.log10(T1_grid))
    return SETUP.T1_to_code * T1


def _edot_cgs(n_H, T_phys, params):
    """Kernel d e / dt in erg cm^-3 s^-1 for cells at (n_H, physical T)."""
    rho_code = n_H * TIGRESS_MU_H * _HYDROGEN_MASS_CGS / RHO_U
    T_code = _physical_T_to_code(T_phys)
    rate = dtemperature_dt(
        jnp.asarray(rho_code, jnp.float32),
        jnp.asarray(T_code, jnp.float32),
        X,
        Z,
        GAMMA,
        CURVE_CONFIG,
        params.cooling_curve_params,
        heating_rate=params.heating_rate,
        heating_log10_T_table=params.heating_log10_T_table,
        heating_factor_table=params.heating_factor_table,
    )
    return np.asarray(rho_code / MU * rate / (GAMMA - 1)) * P_U / T_U


def _ki02_corrected(T):
    """Corrected KI02 cooling, erg cm^3 / s per n_H^2 (independent of the model)."""
    return 2e-26 * (
        1e7 * np.exp(-1.184e5 / (T + 1000)) + 1.4e-2 * np.sqrt(T) * np.exp(-92 / T)
    )


def _gf12(T):
    """Gnat & Ferland (2012) CIE cooling, per n_H^2, interpolated from the table."""
    d = np.load(_GF12_TABLE_PATH)
    lam = d["Lambda_H"] + d["Lambda_He"] + d["Lambda_metal"]
    return 10 ** np.interp(np.log10(T), np.log10(d["T"]), np.log10(lam))


@pytest.mark.parametrize("n_H", [1e-2, 1.0, 1e2])
def test_cold_gas_is_corrected_ki02_with_tigress_n_H(n_H):
    """n_H^2 KI02(T) below 7000 K (catches the 1.148e5 typo and n_p vs n_H)."""
    T = np.logspace(np.log10(30.0), np.log10(7000.0), 25)
    cooling = -_edot_cgs(np.full_like(T, n_H), T, _cooling_params(heating=False))
    np.testing.assert_allclose(cooling, n_H**2 * _ki02_corrected(T), rtol=0.03)


@pytest.mark.parametrize("n_H", [1e-3, 1.0])
def test_hot_gas_is_gnat_ferland(n_H):
    """n_H^2 Lambda_GF12(T) between 1e5 and 1e7 K."""
    T = np.logspace(5.0, 7.0, 21)
    cooling = -_edot_cgs(np.full_like(T, n_H), T, _cooling_params(heating=False))
    np.testing.assert_allclose(cooling, n_H**2 * _gf12(T), rtol=0.02)


def test_heating_factor_cuts_off_in_ionised_gas():
    """Full photoelectric heating in neutral gas, none once H is ionised."""
    params = _cooling_params()
    # Add back the (tiny) cooling at n_H = 1, leaving the heating alone.
    neutral = _edot_cgs(np.array([1.0]), np.array([30.0]), params)[0] + _ki02_corrected(
        30.0
    )
    np.testing.assert_allclose(neutral, 2e-26, rtol=0.02)
    f = np.interp(
        np.log10(_physical_T_to_code(np.array([8e3, 1e5]))),
        np.asarray(SETUP.heating_log10_T_table),
        np.asarray(SETUP.heating_factor_table),
    )
    assert f[0] > 0.99 and f[1] < 1e-3


def test_empty_heating_tables_keep_constant_heating():
    """Default (empty) tables must reproduce the old constant heating."""
    rho = jnp.asarray([1e-2, 1.0, 1e2], jnp.float32)
    T = jnp.asarray(_physical_T_to_code(np.array([1e2, 1e5, 1e7])), jnp.float32)
    common = (X, Z, GAMMA, CURVE_CONFIG, SETUP.cooling_curve_params)
    old = dtemperature_dt(rho, T, *common, heating_rate=SETUP.heating_rate)
    new = dtemperature_dt(
        rho,
        T,
        *common,
        heating_rate=SETUP.heating_rate,
        heating_log10_T_table=jnp.array([]),
        heating_factor_table=jnp.array([]),
    )
    np.testing.assert_array_equal(np.asarray(old), np.asarray(new))


def test_thermal_equilibrium_at_unit_density():
    """Gamma = n_H Lambda at n_H = 1, heat_ratio = 1: the WNM branch at 5010 K."""
    T = np.logspace(np.log10(2e3), np.log10(9e3), 4001)
    net = _edot_cgs(np.ones_like(T), T, _cooling_params())
    T_eq = T[np.where(np.diff(np.sign(net)))[0]]
    assert len(T_eq) == 1
    np.testing.assert_allclose(T_eq[0], 5010.0, rtol=0.01)


def test_isochoric_cooling_from_1e7_K():
    """
    Isochoric cooling at n_H = 1 over 10 Myr in 100 implicit steps.

    Reference (adaptive RK2 on the same physics, frac = 0.01): crosses 2e4 K at
    3.875 Myr, T(10 Myr) = 5035 K.
    """
    registered_variables = get_registered_variables(SimulationConfig(dimensionality=1))
    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=IMPLICIT_COOLING,
        cooling_curve_config=CURVE_CONFIG,
    )
    simulation_params = SimulationParams(gamma=GAMMA, cooling_params=_cooling_params())
    rho = jnp.asarray([TIGRESS_MU_H * _HYDROGEN_MASS_CGS / RHO_U], jnp.float32)
    T0 = jnp.asarray(_physical_T_to_code(np.array([1e7])), jnp.float32)
    state = jnp.zeros((registered_variables.num_vars, 1), jnp.float32)
    state = state.at[registered_variables.density_index].set(rho)
    state = state.at[registered_variables.pressure_index].set(
        get_pressure_from_temperature(rho, T0, X, Z)
    )

    # A Python float rather than numpy's float64, so that the time step does not
    # promote the float32 state when x64 is switched on.
    time_step = float(0.1 * MYR / T_U)
    step = jax.jit(
        lambda s: update_pressure_by_cooling(
            s,
            registered_variables,
            cooling_config,
            simulation_params,
            time_step,
        )
    )
    T_phys = []
    for _ in range(100):
        state = step(state)
        T_code = get_temperature_from_pressure(
            state[registered_variables.density_index],
            state[registered_variables.pressure_index],
            X,
            Z,
        )
        T_phys.append(
            float(ki02_gf12_temperature(np.asarray(T_code) / SETUP.T1_to_code)[0])
        )
    T_phys = np.array(T_phys)
    t_cross = 0.1 * (np.argmax(T_phys < 2e4) + 1)
    np.testing.assert_allclose(t_cross, 3.875, rtol=0.05)
    np.testing.assert_allclose(T_phys[-1], 5035.0, rtol=0.02)


def test_townsend_coefficients_are_finite():
    """The Townsend (2009) Y_k coefficients of the table are all finite."""
    Y = np.asarray(SETUP.cooling_curve_params.Y_table)
    assert np.all(np.isfinite(Y))
