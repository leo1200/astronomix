"""
Implicit radiative-cooling solver pytest (CPU, seconds).

Guards the backward-Euler solve in ``update_temperature_implicit`` against its
two failure modes:

1. **The monotonicity bound.** With the heating off, cooling is a pure sink, so
   the backward-Euler root satisfies ``T_new <= T_old`` exactly. A plain Newton
   iteration does NOT respect this: ``Lambda(T)`` is non-monotone, the Jacobian
   ``1 - dt * d(rate)/dT`` passes through zero on the falling branch of the
   curve, and the unguarded step then jumps the wrong way. On a non-monotone
   curve in float32 this gives ``T_new`` of up to 240x ``T_old``, and a
   temperature jump of that size collapses the CFL time step.

2. **Actually solving the equation.** A fixed-point sweep
   ``T <- T_old + dt * rate(T)`` diverges once the step is stiff and, after its
   iteration cap, simply returns ``~T_old`` -- i.e. it silently applies NO
   cooling in exactly the cells that most need it. The residual check below
   fails such a solver.

Both run in float32, the precision in which both failures appear.

The second half of the file runs the full ``update_pressure_by_cooling`` update
and checks the temperature-floor handling and the per-step cooling cap.
"""

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# testing
import pytest

# astronomix constants
from astronomix._modules._cooling.cooling_options import (
    EXPLICIT_COOLING,
    IMPLICIT_COOLING,
    PIECEWISE_POWER_LAW,
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
    PiecewisePowerLawParams,
)

# astronomix functions
from astronomix import get_registered_variables
from astronomix._modules._cooling._cooling import (
    dtemperature_dt,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
    update_pressure_by_cooling,
    update_temperature_implicit,
)

GAMMA = 5.0 / 3.0
HYDROGEN_MASS_FRACTION = 0.70
METAL_MASS_FRACTION = 0.02

#: Time steps spanning the resolved and the deeply stiff regime: on the test
#: grid dt * |rate| / T reaches from well below one to far above it.
TIME_STEPS = [1e-6, 1e-5, 1e-4, 1e-3]

#: A time step deep in the stiff regime, used by the single-step tests.
STIFF_TIME_STEP = 1.7e-4

#: Bottom of the test table, i.e. the temperature floor of the full-update cases.
FLOOR_TEMPERATURE = 1e-4

#: A stiff dense cell, a stiffer one, a hot cell, and one already below the
#: floor -- the four cases the floor logic has to get right at once.
FLOOR_CASE_TEMPERATURES = np.array([3.0e-4, 1.0e-3, 1.0e0, 5.0e-5])
FLOOR_CASE_DENSITIES = np.array([5.0e2, 5.0e2, 1.0e0, 1.0e0])


def _curve():
    """
    A non-monotone piecewise power law: a rising branch, a peak, a fall.

    The falling branch is the part that breaks an unguarded Newton step, so a
    monotone test curve would not exercise the bug at all.

    Everything here is in the kernel's own RESCALED units (``T~ = p * mu / rho``
    and a Lambda rescaled to match), not Kelvin and cgs -- that is what the
    solver actually sees, and it is what puts the peak of the curve at a cooling
    time comparable to the time step, i.e. in the stiff regime the tests are
    about. A cgs table with T in Kelvin gives dt*|rate|/T ~ 1e-20 and would
    exercise nothing. The shape mirrors an ISM curve: a steep rise out of 1e4 K,
    a peak, then the slow bremsstrahlung fall.

    Returns:
        The cooling-curve configuration and the matching curve parameters.
    """
    log10_temperature = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.0, 0.0, 1.0])
    log10_cooling_function = np.array([-3.0, -1.5, -1.0, -1.3, -1.8, -2.3, -2.4, -2.1])
    slopes = np.diff(log10_cooling_function) / np.diff(log10_temperature)
    slopes = np.append(slopes, slopes[-1])
    curve_config = CoolingCurveConfig(cooling_curve_type=PIECEWISE_POWER_LAW)
    curve_params = PiecewisePowerLawParams(
        log10_T_table=jnp.asarray(log10_temperature, jnp.float32),
        log10_Lambda_table=jnp.asarray(log10_cooling_function, jnp.float32),
        alpha_table=jnp.asarray(slopes, jnp.float32),
        Y_table=jnp.zeros(len(log10_temperature), jnp.float32),
        reference_temperature=10.0,
    )
    return curve_config, curve_params


def _grid():
    """
    Temperatures across the whole curve times densities across a 1e6 contrast.

    Returns:
        The temperature and density grids (float32, shape (240, 40)).
    """
    temperatures = np.geomspace(1.01e-4, 5.0, 240)
    densities = np.geomspace(1e-3, 1e3, 40)
    temperature_grid, density_grid = np.meshgrid(temperatures, densities, indexing="ij")
    return jnp.asarray(temperature_grid, jnp.float32), jnp.asarray(density_grid, jnp.float32)


def _implicit_update(density, temperature, time_step, curve_config, curve_params):
    """Advance the temperature by one backward-Euler cooling step (heating off)."""
    return update_temperature_implicit(
        density,
        temperature,
        time_step,
        HYDROGEN_MASS_FRACTION,
        METAL_MASS_FRACTION,
        GAMMA,
        curve_config,
        curve_params,
        heating_rate=0.0,
    )


def _temperature_rate(density, temperature, curve_config, curve_params):
    """The cooling rate dT/dt at the given state (heating off)."""
    return dtemperature_dt(
        density,
        temperature,
        HYDROGEN_MASS_FRACTION,
        METAL_MASS_FRACTION,
        GAMMA,
        curve_config,
        curve_params,
        heating_rate=0.0,
    )


@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_cooling_cannot_heat(time_step):
    """Pure sink => T_new <= T_old, everywhere, exactly."""
    curve_config, curve_params = _curve()
    temperature, density = _grid()

    new_temperature = _implicit_update(density, temperature, time_step, curve_config, curve_params)

    ratio = np.asarray(new_temperature) / np.asarray(temperature)
    worst = float(ratio.max())
    assert worst <= 1.0 + 1e-5, (
        f"cooling RAISED the temperature by up to {worst:.3e}x at "
        f"dt = {time_step:.0e}; backward Euler on a pure sink cannot do this, "
        "so the implicit solver left the physical bracket"
    )


@pytest.mark.parametrize("time_step", TIME_STEPS)
def test_implicit_solve_has_small_residual(time_step):
    """
    The returned T must actually solve T - T_old - dt*rate(T) = 0.

    Asserted over the RESOLVED cells (stiffness < 1) only. Deep in the stiff
    regime this equation genuinely acquires several roots and any bracketing
    solver may return a different one than a reference bisection, so a
    tight residual bound there would encode the reference's arbitrary choice
    rather than a correctness requirement. Where the step is resolved there is
    no such ambiguity, and this is a real floor: a geometric bisection that
    underflows to zero in float32 shows up here with a residual of 1e30.
    """
    curve_config, curve_params = _curve()
    temperature, density = _grid()

    new_temperature = _implicit_update(density, temperature, time_step, curve_config, curve_params)
    rate_new = _temperature_rate(density, new_temperature, curve_config, curve_params)
    rate_old = _temperature_rate(density, temperature, curve_config, curve_params)

    residual = np.asarray(new_temperature - temperature - time_step * rate_new)
    relative = np.abs(residual) / np.maximum(np.abs(np.asarray(new_temperature)), 1e-30)
    # Exclude cells that cooled onto the bottom edge of the table. Lambda drops
    # discontinuously to zero below it, so F has a second root there (T = T_old,
    # where nothing cools at all) and the residual of the physically correct
    # answer -- "cooled as far as the tabulated curve allows" -- is not small.
    # A simulation never sees this: floor_temperature sits at that edge and
    # update_pressure_by_cooling clamps any cell that would cross it.
    table_minimum_temperature = float(10 ** np.asarray(curve_params.log10_T_table)[0])
    stiffness = time_step * np.abs(np.asarray(rate_old)) / np.asarray(temperature)
    resolved = (stiffness < 1.0) & (np.asarray(new_temperature) > 1.001 * table_minimum_temperature)
    worst = float(relative[resolved].max())
    assert worst < 1e-3, (
        f"backward-Euler residual up to {worst:.3e} in RESOLVED cells at "
        f"dt = {time_step:.0e}: the implicit equation is not being solved"
    )


def test_cooling_is_applied_where_it_is_stiff():
    """
    In stiff cells the solve must move T substantially, not return T_old.

    This is the regression for the silent no-op: a fixed-point sweep returns an
    unchanged temperature for every stiff cell at this time step, so a run with
    cooling switched on would stay adiabatic exactly where cooling dominates.
    """
    curve_config, curve_params = _curve()
    temperature, density = _grid()
    time_step = STIFF_TIME_STEP

    rate = _temperature_rate(density, temperature, curve_config, curve_params)
    stiffness = time_step * np.abs(np.asarray(rate)) / np.asarray(temperature)
    stiff = stiffness > 1.0
    assert stiff.sum() > 100, "test grid does not contain a stiff regime"

    new_temperature = _implicit_update(density, temperature, time_step, curve_config, curve_params)
    ratio = (np.asarray(new_temperature) / np.asarray(temperature))[stiff]
    untouched = float(np.mean(ratio > 0.99))
    assert untouched < 0.05, (
        f"{100 * untouched:.1f}% of stiff cells came back within 1% of their "
        "original temperature: the implicit solver is silently skipping cooling"
    )


# =============================================================================
# ==== ↓ The full update: floor handling and the per-step cap ↓ ===============
# =============================================================================


def _pressure_update(
    use_explicit_cooling,
    max_cooling_fraction,
    temperature,
    density,
    time_step=STIFF_TIME_STEP,
    clamp_to_floor=True,
):
    """
    Run ``update_pressure_by_cooling`` on a handful of cells.

    The resolution limiter is switched OFF here so the floor and cap behaviour
    is isolated -- with it on it suppresses precisely the stiff cells these
    cases are about, and every ratio comes back 1.0 for the wrong reason.

    Args:
        use_explicit_cooling: Use the explicit instead of the implicit update.
        max_cooling_fraction: The per-step cooling cap (0 disables it).
        temperature: The rescaled cell temperatures.
        density: The cell densities.
        time_step: The time step.
        clamp_to_floor: Clamp cells that would cross the floor to it (instead of
            reverting their update).

    Returns:
        The ratio T_new / T_old per cell.
    """
    curve_config, curve_params = _curve()
    cooling_config = CoolingConfig(
        cooling=True,
        cooling_method=EXPLICIT_COOLING if use_explicit_cooling else IMPLICIT_COOLING,
        cooling_curve_config=curve_config,
    )
    cooling_params = CoolingParams(
        hydrogen_mass_fraction=HYDROGEN_MASS_FRACTION,
        metal_mass_fraction=METAL_MASS_FRACTION,
        floor_temperature=FLOOR_TEMPERATURE,
        resolution_limiter_alpha=0.0,
        max_cooling_fraction=max_cooling_fraction,
        clamp_to_floor=clamp_to_floor,
        cooling_curve_params=curve_params,
    )

    registered_variables = get_registered_variables(SimulationConfig(dimensionality=1))
    temperature = jnp.asarray(temperature, jnp.float32)
    density = jnp.asarray(density, jnp.float32)
    pressure = get_pressure_from_temperature(
        density,
        temperature,
        HYDROGEN_MASS_FRACTION,
        METAL_MASS_FRACTION,
    )
    state = jnp.zeros((registered_variables.num_vars,) + density.shape, jnp.float32)
    state = state.at[registered_variables.density_index].set(density)
    state = state.at[registered_variables.pressure_index].set(pressure)

    # A positive grid spacing runs the resolution-limiter code path, so
    # resolution_limiter_alpha = 0 also checks that the limiter is really off.
    cooled_state = update_pressure_by_cooling(
        state,
        registered_variables,
        cooling_config,
        SimulationParams(gamma=GAMMA, cooling_params=cooling_params),
        time_step,
        grid_spacing=0.0273,
    )
    new_temperature = get_temperature_from_pressure(
        cooled_state[registered_variables.density_index],
        cooled_state[registered_variables.pressure_index],
        HYDROGEN_MASS_FRACTION,
        METAL_MASS_FRACTION,
    )
    return np.asarray(new_temperature) / np.asarray(temperature)


@pytest.mark.parametrize("use_explicit_cooling", [False, True])
def test_stiff_cells_actually_cool(use_explicit_cooling):
    """
    Neither path may leave a stiff cell untouched.

    An explicit update that REVERTS the whole step whenever the forward step
    would cross the floor never cools a stiff cell at all, and the run then
    reproduces the adiabatic solution while appearing perfectly healthy.
    """
    ratio = _pressure_update(
        use_explicit_cooling,
        0.0,
        FLOOR_CASE_TEMPERATURES,
        FLOOR_CASE_DENSITIES,
    )
    assert ratio[0] < 0.95 and ratio[1] < 0.95, (
        f"stiff cells came back at {ratio[0]:.4f} / {ratio[1]:.4f} of their "
        "original temperature: the cooling update is being discarded"
    )


@pytest.mark.parametrize("use_explicit_cooling", [False, True])
def test_floor_never_heats_already_cold_gas(use_explicit_cooling):
    """
    A cell starting below the floor must be left alone, not clamped UP to it.

    Cold gas can legitimately sit far below the floor (e.g. after adiabatic
    expansion); clamping it would be a spurious heat source.
    """
    ratio = _pressure_update(
        use_explicit_cooling,
        0.0,
        FLOOR_CASE_TEMPERATURES,
        FLOOR_CASE_DENSITIES,
    )
    assert ratio[3] == pytest.approx(1.0, abs=1e-6), (
        f"a cell below the floor was moved to {ratio[3]:.4f} of its temperature"
    )
    assert ratio.min() >= 0.0, "cooling produced a negative temperature"


def test_per_step_cooling_cap_is_respected():
    """``max_cooling_fraction`` bounds the drop applied in a single step."""
    uncapped = _pressure_update(False, 0.0, FLOOR_CASE_TEMPERATURES, FLOOR_CASE_DENSITIES)
    capped = _pressure_update(False, 0.3, FLOOR_CASE_TEMPERATURES, FLOOR_CASE_DENSITIES)
    # The cap only means something if the unrestricted solve cools past it.
    assert uncapped[:2].max() < 0.7, "test cells do not cool enough for the cap to bite"
    assert capped[:2] == pytest.approx(0.7, abs=1e-3), (
        f"cap of 0.3 gave {capped[:2]} rather than 0.7"
    )
    # The cap must not manufacture cooling where there was none.
    assert capped[2] == pytest.approx(1.0, abs=1e-6)
    assert capped[3] == pytest.approx(1.0, abs=1e-6)


def test_default_floor_revert_suppresses_stiff_cooling():
    """
    Pin down what the DEFAULT floor handling actually does.

    Not an endorsement -- a documented weakness. With ``clamp_to_floor=False``
    (the default) a stiff cell whose update would cross the floor keeps its
    original temperature, so the EXPLICIT path applies no cooling to it at all.
    The revert is kept as the default only because it doubles as crush
    protection for radiatively cooled shocks: without it, runs in which dense
    gas is crushed by ram pressure while cooling strongly abort or blow up.

    If a future change makes the default clamp instead, this test fails, and
    the stability of strongly cooling runs has to be re-checked with it.
    """
    reverted = _pressure_update(
        True,
        0.0,
        FLOOR_CASE_TEMPERATURES,
        FLOOR_CASE_DENSITIES,
        clamp_to_floor=False,
    )
    clamped = _pressure_update(
        True,
        0.0,
        FLOOR_CASE_TEMPERATURES,
        FLOOR_CASE_DENSITIES,
        clamp_to_floor=True,
    )
    assert reverted[:2] == pytest.approx(1.0, abs=1e-6), (
        "the default is no longer the revert; if that is deliberate, the crush "
        "protection of strongly cooling runs must be re-established"
    )
    assert clamped[:2].max() < 0.95, "clamping should let the same cells cool"

# =============================================================================
# ==== ↑ The full update: floor handling and the per-step cap ↑ ===============
# =============================================================================
