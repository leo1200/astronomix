"""
Implicit radiative-cooling solver pytest (CPU, seconds).

Guards the backward-Euler solve in ``update_temperature_implicit`` against the
two ways it has actually failed in this repository:

1. **The monotonicity bound.** With the heating off, cooling is a pure sink, so
   the backward-Euler root satisfies ``T_new <= T_old`` exactly. A plain Newton
   iteration does NOT respect this: ``Lambda(T)`` is non-monotone, the Jacobian
   ``1 - dt * d(rate)/dT`` passes through zero on the falling branch of the
   curve, and the unguarded step then jumps the wrong way. Measured on the
   Schure curve at float32 that produced ``T_new`` up to 240x ``T_old``, which
   is a direct route to a CFL collapse (it aborted the 256^3 Cas A runs as soon
   as the dense pistons met the radiative shell).

2. **Actually solving the equation.** A fixed-point sweep
   ``T <- T_old + dt * rate(T)`` diverges once the step is stiff and, after its
   iteration cap, simply returns ``~T_old`` -- i.e. it silently applies NO
   cooling in exactly the cells that most need it. The residual check below
   fails such a solver.

Both run at float32, which is the production precision here and the precision
in which both failures appeared.
"""

# ==== GPU selection ====
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")  # pointwise solve; no GPU needed
# ruff: noqa: E402
# =======================

import numpy as np
import pytest

import jax.numpy as jnp

from astronomix._modules._cooling._cooling import (
    dtemperature_dt,
    update_temperature_implicit,
)
from astronomix._modules._cooling.cooling_options import (
    CoolingCurveConfig,
    PIECEWISE_POWER_LAW,
    PiecewisePowerLawParams,
)

GAMMA = 5.0 / 3.0
X_H, Z_METAL = 0.70, 0.02

#: dt values spanning the resolved case and the deeply stiff one. The Cas A
#: 256^3 production runs sit at ~1.7e-4 in these units.
TIME_STEPS = [1e-6, 1e-5, 1e-4, 1e-3]


def _curve():
    """A non-monotone piecewise power law: a rising branch, a peak, a fall.

    The falling branch is the part that breaks an unguarded Newton step, so a
    monotone test curve would not exercise the bug at all.

    Everything here is in the kernel's own RESCALED units (``T~ = p * mu / rho``
    and a Lambda rescaled to match), not Kelvin and cgs -- that is what the
    solver actually sees, and it is what puts the peak of the curve at a cooling
    time comparable to the time step, i.e. in the stiff regime the tests are
    about. A cgs table with T in Kelvin gives dt*|rate|/T ~ 1e-20 and would
    exercise nothing. The shape mirrors an ISM curve: a steep rise out of 1e4 K,
    a peak, then the slow bremsstrahlung fall.
    """
    log10_T = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.0, 0.0, 1.0])
    log10_L = np.array([-3.0, -1.5, -1.0, -1.3, -1.8, -2.3, -2.4, -2.1])
    alpha = np.diff(log10_L) / np.diff(log10_T)
    alpha = np.append(alpha, alpha[-1])
    return CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW), (
        PiecewisePowerLawParams(
            log10_T_table=jnp.asarray(log10_T, jnp.float32),
            log10_Lambda_table=jnp.asarray(log10_L, jnp.float32),
            alpha_table=jnp.asarray(alpha, jnp.float32),
            Y_table=jnp.zeros(len(log10_T), jnp.float32),
            reference_temperature=10.0,
        )
    )


def _grid():
    """Temperatures across the whole curve x densities across a piston contrast."""
    T = np.geomspace(1.01e-4, 5.0, 240)
    rho = np.geomspace(1e-3, 1e3, 40)  # Orlando knots run to chi_n = 100
    TT, RR = np.meshgrid(T, rho, indexing="ij")
    return jnp.asarray(TT, jnp.float32), jnp.asarray(RR, jnp.float32)


@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_cooling_cannot_heat(time_step):
    """Pure sink => T_new <= T_old, everywhere, exactly."""
    curve_config, curve_params = _curve()
    temperature, density = _grid()

    new_temperature = update_temperature_implicit(
        density, temperature, time_step, X_H, Z_METAL, GAMMA,
        curve_config, curve_params, heating_rate=0.0,
    )

    ratio = np.asarray(new_temperature) / np.asarray(temperature)
    worst = float(ratio.max())
    assert worst <= 1.0 + 1e-5, (
        f"cooling RAISED the temperature by up to {worst:.3e}x at "
        f"dt = {time_step:.0e}; backward Euler on a pure sink cannot do this, "
        "so the implicit solver left the physical bracket"
    )


@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_implicit_solve_has_small_residual(time_step):
    """The returned T must actually solve T - T_old - dt*rate(T) = 0.

    Asserted over the RESOLVED cells (stiffness < 1) only. Deep in the stiff
    regime this equation genuinely acquires several roots and any bracketing
    solver may return a different one than a reference bisection, so a
    tight residual bound there would encode the reference's arbitrary choice
    rather than a correctness requirement. Where the step is resolved there is
    no such ambiguity, and this is a real floor: the geometric-bisection variant
    tried during development underflowed to zero in float32 and showed up here
    with a residual of 1e30.
    """
    curve_config, curve_params = _curve()
    temperature, density = _grid()

    new_temperature = update_temperature_implicit(
        density, temperature, time_step, X_H, Z_METAL, GAMMA,
        curve_config, curve_params, heating_rate=0.0,
    )
    rate_new = dtemperature_dt(
        density, new_temperature, X_H, Z_METAL, GAMMA,
        curve_config, curve_params, heating_rate=0.0,
    )
    rate_old = dtemperature_dt(
        density, temperature, X_H, Z_METAL, GAMMA,
        curve_config, curve_params, heating_rate=0.0,
    )

    residual = np.asarray(new_temperature - temperature - time_step * rate_new)
    relative = np.abs(residual) / np.maximum(np.abs(np.asarray(new_temperature)), 1e-30)
    # Exclude cells that cooled onto the bottom edge of the table. Lambda drops
    # discontinuously to zero below it, so F has a second root there (T = T_old,
    # where nothing cools at all) and the residual of the physically correct
    # answer -- "cooled as far as the tabulated curve allows" -- is not small.
    # Production never sees this: floor_temperature sits at that edge and
    # update_pressure_by_cooling clamps any cell that would cross it.
    T_table_min = float(10 ** np.asarray(curve_params.log10_T_table)[0])
    resolved = (
        time_step * np.abs(np.asarray(rate_old)) / np.asarray(temperature) < 1.0
    ) & (np.asarray(new_temperature) > 1.001 * T_table_min)
    worst = float(relative[resolved].max())
    assert worst < 1e-3, (
        f"backward-Euler residual up to {worst:.3e} in RESOLVED cells at "
        f"dt = {time_step:.0e}: the implicit equation is not being solved"
    )


def test_cooling_is_applied_where_it_is_stiff():
    """In stiff cells the solve must move T substantially, not return T_old.

    This is the regression for the silent no-op: the old fixed-point sweep
    returned an unchanged temperature for 100% of stiff cells at the Cas A
    production time step, so every "cooling on" run was effectively adiabatic
    exactly where the radiative shell and the piston knots live.
    """
    curve_config, curve_params = _curve()
    temperature, density = _grid()
    time_step = 1.7e-4  # the Cas A 256^3 production dt

    rate = dtemperature_dt(
        density, temperature, X_H, Z_METAL, GAMMA,
        curve_config, curve_params, heating_rate=0.0,
    )
    stiffness = time_step * np.abs(np.asarray(rate)) / np.asarray(temperature)
    stiff = stiffness > 1.0
    assert stiff.sum() > 100, "test grid does not contain a stiff regime"

    new_temperature = update_temperature_implicit(
        density, temperature, time_step, X_H, Z_METAL, GAMMA,
        curve_config, curve_params, heating_rate=0.0,
    )
    ratio = (np.asarray(new_temperature) / np.asarray(temperature))[stiff]
    untouched = float(np.mean(ratio > 0.99))
    assert untouched < 0.05, (
        f"{100 * untouched:.1f}% of stiff cells came back within 1% of their "
        "original temperature: the implicit solver is silently skipping cooling"
    )


# =============================================================================
# ==== ↓ The full update: floor handling and the per-step cap ↓ ===============
# =============================================================================
def _pressure_update(explicit, max_cooling_fraction, T_tilde, density,
                     time_step=1.7e-4, clamp_to_floor=True):
    """Run ``update_pressure_by_cooling`` on a handful of cells.

    The resolution limiter is switched OFF here so the floor and cap behaviour
    is isolated -- with it on it suppresses precisely the stiff cells these
    cases are about, and every ratio comes back 1.0 for the wrong reason.
    """
    from astronomix import SimulationParams
    from astronomix.variable_registry.registered_variables import RegisteredVariables
    from astronomix._modules._cooling._cooling import (
        get_pressure_from_temperature, get_temperature_from_pressure,
        update_pressure_by_cooling,
    )
    from astronomix._modules._cooling.cooling_options import (
        CoolingConfig, CoolingParams, EXPLICIT_COOLING, IMPLICIT_COOLING,
    )

    curve_config, curve_params = _curve()
    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=EXPLICIT_COOLING if explicit else IMPLICIT_COOLING,
        cooling_curve_config=curve_config,
    )
    cooling_params = CoolingParams(
        hydrogen_mass_fraction=X_H, metal_mass_fraction=Z_METAL,
        floor_temperature=FLOOR_TILDE,
        resolution_limiter_alpha=0.0,
        max_cooling_fraction=max_cooling_fraction,
        clamp_to_floor=clamp_to_floor,
        cooling_curve_params=curve_params,
    )

    registered = RegisteredVariables(num_vars=3, density_index=0)._replace(
        pressure_index=2)
    T = jnp.asarray(T_tilde, jnp.float32)
    rho = jnp.asarray(density, jnp.float32)
    pressure = get_pressure_from_temperature(rho, T, X_H, Z_METAL)
    state = jnp.stack([rho, jnp.zeros_like(rho), pressure])

    out = update_pressure_by_cooling(
        state, registered, cooling_config,
        SimulationParams(gamma=GAMMA, cooling_params=cooling_params),
        time_step, grid_spacing=0.0273,
    )
    T_new = get_temperature_from_pressure(out[0], out[2], X_H, Z_METAL)
    return np.asarray(T_new) / np.asarray(T)


#: Bottom of the test table, i.e. the temperature floor these cases use.
FLOOR_TILDE = 1e-4

#: A stiff dense cell, a stiffer one, a hot cell, and one already below the
#: floor -- the four cases the floor logic has to get right at once.
_T_CELLS = np.array([3.0e-4, 1.0e-3, 1.0e0, 5.0e-5])
_RHO_CELLS = np.array([5.0e2, 5.0e2, 1.0e0, 1.0e0])


@pytest.mark.parametrize("explicit", [False, True])
def test_stiff_cells_actually_cool(explicit):
    """Neither path may leave a stiff cell untouched.

    The explicit path used to REVERT the whole update when the forward step
    would cross the floor, which meant a stiff cell never cooled at all: the
    256^3 Cas A run with ``--explicit-cooling`` reproduced the adiabatic
    solution to four significant figures while appearing perfectly healthy.
    """
    ratio = _pressure_update(explicit, 0.0, _T_CELLS, _RHO_CELLS)
    assert ratio[0] < 0.95 and ratio[1] < 0.95, (
        f"stiff cells came back at {ratio[0]:.4f} / {ratio[1]:.4f} of their "
        "original temperature: the cooling update is being discarded"
    )


@pytest.mark.parametrize("explicit", [False, True])
def test_floor_never_heats_already_cold_gas(explicit):
    """A cell starting below the floor must be left alone, not clamped UP to it.

    The cold unshocked ejecta sits far below the floor; clamping it would be a
    spurious heat source.
    """
    ratio = _pressure_update(explicit, 0.0, _T_CELLS, _RHO_CELLS)
    assert ratio[3] == pytest.approx(1.0, abs=1e-6), (
        f"a cell below the floor was moved to {ratio[3]:.4f} of its temperature"
    )
    assert ratio.min() >= 0.0, "cooling produced a negative temperature"


def test_per_step_cooling_cap_is_respected():
    """``max_cooling_fraction`` bounds the drop applied in a single step."""
    uncapped = _pressure_update(False, 0.0, _T_CELLS, _RHO_CELLS)
    capped = _pressure_update(False, 0.3, _T_CELLS, _RHO_CELLS)
    # the cap only means something if the unrestricted solve cools past it
    assert uncapped[:2].max() < 0.7, "test cells do not cool enough for the cap to bite"
    assert capped[:2] == pytest.approx(0.7, abs=1e-3), (
        f"cap of 0.3 gave {capped[:2]} rather than 0.7"
    )
    # the cap must not manufacture cooling where there was none
    assert capped[2] == pytest.approx(1.0, abs=1e-6)
    assert capped[3] == pytest.approx(1.0, abs=1e-6)


def test_default_floor_revert_suppresses_stiff_cooling():
    """Pin down what the DEFAULT floor handling actually does.

    Not an endorsement -- a documented weakness. With ``clamp_to_floor=False``
    (the default) a stiff cell whose update would cross the floor keeps its
    original temperature, so the EXPLICIT path applies no cooling to it at all.
    That is why ``--explicit-cooling`` reproduced the adiabatic Cas A solution.
    The revert is kept as the default only because it doubles as crush
    protection: switching it off makes the piston runs abort or blow up.

    If a future change makes the default clamp instead, this test fails and the
    Cas A stability work has to be revisited at the same time.
    """
    reverted = _pressure_update(True, 0.0, _T_CELLS, _RHO_CELLS,
                                clamp_to_floor=False)
    clamped = _pressure_update(True, 0.0, _T_CELLS, _RHO_CELLS,
                               clamp_to_floor=True)
    assert reverted[:2] == pytest.approx(1.0, abs=1e-6), (
        "the default is no longer the revert; if that is deliberate, the Cas A "
        "crush protection must be re-established before relying on it"
    )
    assert clamped[:2].max() < 0.95, "clamping should let the same cells cool"
