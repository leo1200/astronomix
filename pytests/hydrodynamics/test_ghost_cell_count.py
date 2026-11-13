"""
Ghost-cell count pytest: the number of layers ``finalize_config`` sets is enough.

All stencils wrap around the padded array, so with too few ghost-cell layers the cells next to a boundary read the
opposite edge of the grid. A run with the number of layers of ``finalize_config`` must therefore give the same
result as a run with two more layers. Open boundaries hide a missing layer (the wrapped values are close to the
extrapolated ones), so the cases are a rough state (independent values per cell) between reflective walls. Checked
for the scheme families that run with ghost cells:

* the finite-volume integrators (two layers; one layer gives a different result);
* the finite-difference scheme for hydrodynamics with SSPRK(5,4): plain, with an external potential, viscosity,
  fourth-order conduction and passive scalars (the counts are pinned: both integrators fill the boundary before
  every right-hand side, so the count is the reach of one right-hand side);
* the finite-difference scheme for MHD with SSPRK(5,4) and LSRK4 (9 layers in 2D);
* a number of layers set by the user is raised to what the scheme reads and never lowered.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_ghost_cell_count.py
"""

# ==== GPU selection ====
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# numerics
import numpy as np

# testing
import pytest

# jax
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# astronomix constants
from astronomix import (
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
    LSRK4,
    MUSCL_HANCOCK,
    REFLECTIVE_BOUNDARY,
    RK2_SSP,
    SSPRK54,
    VL2,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    DiffusionConfig,
    FDSolverConfig,
    FVSolverConfig,
    GravityConfig,
    PassiveScalarConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    time_integration,
)

GAMMA = 5.0 / 3.0
NUM_CELLS = 32
NUM_STEPS = 3
TIME_STEP = 0.02 / NUM_CELLS  # well inside the stability limit of every scheme for the rough state
WALLS = BoundarySettings1D(REFLECTIVE_BOUNDARY, REFLECTIVE_BOUNDARY)

# options of the finite-difference scheme for hydrodynamics that read further than the plain flux difference, with
# the count in 1D: the reach of one right-hand side (WENO5 3, fourth-order work of the potential 4, viscous flux 6,
# fourth-order conduction 4) or of the three right-hand sides of the scalar sub-step without a fill (9)
FD_HYDRO_OPTIONS = {
    "plain": (dict(), 3),
    "external potential": (dict(external_potential=True), 4),
    "viscosity": (dict(diffusion_config=DiffusionConfig(viscosity=True)), 6),
    "conduction": (dict(diffusion_config=DiffusionConfig(thermal_conduction=True, conduction_order=4)), 4),
    "passive scalars": (dict(num_scalars=2), 9),
}


def _rough_case(solver_config, dimensionality=1, mhd=False, num_scalars=0, external_potential=False, **options):
    """Finalized configuration, parameters, registry and state: a rough state between reflective walls."""
    config = SimulationConfig(
        solver_mode=FINITE_DIFFERENCE if isinstance(solver_config, FDSolverConfig) else FINITE_VOLUME,
        dimensionality=dimensionality,
        mhd=mhd,
        num_cells=NUM_CELLS,
        box_size=1.0,
        boundary_settings=WALLS if dimensionality == 1 else BoundarySettings(WALLS, WALLS, WALLS),
        solver_config=solver_config,
        passive_scalar_config=PassiveScalarConfig(num_scalars=num_scalars),
        gravity_config=GravityConfig(external_potential=external_potential),
        fixed_timestep=True,
        num_timesteps=NUM_STEPS,
        progress_bar=False,
        **options,
    )
    registered_variables = get_registered_variables(config)

    # independent values per cell: density and pressure over a decade, subsonic to transonic velocities
    rng = np.random.default_rng(3)
    shape = (NUM_CELLS,) * dimensionality
    fields = dict(density=10.0 ** rng.uniform(-1.0, 0.0, shape), gas_pressure=10.0 ** rng.uniform(-0.5, 0.5, shape))
    for axis, name in enumerate("xyz" if mhd else "xyz"[:dimensionality]):
        fields[f"velocity_{name}"] = 0.3 * rng.standard_normal(shape)
        if mhd:
            fields[f"magnetic_field_{name}"] = 0.25 * rng.standard_normal(shape) + (0.4 if axis == 0 else 0.0)
    if num_scalars > 0:
        fields["passive_scalars"] = rng.uniform(0.0, 1.0, (num_scalars,) + shape)
    state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        **{name: jnp.asarray(values) for name, values in fields.items()},
    )

    params = SimulationParams(t_end=NUM_STEPS * TIME_STEP, gamma=GAMMA, viscosity=0.01, thermal_conductivity=0.02)
    if external_potential:
        x = (np.arange(NUM_CELLS) + 0.5) / NUM_CELLS
        grid = np.meshgrid(*([x] * dimensionality), indexing="ij")
        params = params._replace(gravitational_potential=jnp.asarray(sum(0.5 * (a - 0.4) ** 2 for a in grid)))
    return finalize_config(config, state.shape), params, registered_variables, state


def _final_state(case, extra_layers=0):
    """The final state of a case with ``extra_layers`` ghost-cell layers more than ``finalize_config`` set."""
    config, params, registered_variables, state = case
    config = config._replace(num_ghost_cells=config.num_ghost_cells + extra_layers)
    return np.asarray(time_integration(jnp.array(state), config, params, registered_variables))


def _assert_two_more_layers_change_nothing(case):
    derived, wider = _final_state(case), _final_state(case, extra_layers=2)
    assert np.all(np.isfinite(derived))
    different = derived != wider
    print(f"{case[0].num_ghost_cells} layers: {int(different.sum())} of {different.size} entries differ,"
          f" largest difference {np.max(np.abs(derived - wider)):.1e}")
    # rounding only (the compiled step depends on the padded shape); a missing layer shows at 1e-6 and above
    np.testing.assert_allclose(derived, wider, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("time_integrator", [RK2_SSP, MUSCL_HANCOCK, VL2])
def test_fv_more_ghost_cells_give_the_same_result(time_integrator):
    """Finite volume: the boundary is filled after every stage and one stage reads two layers."""
    case = _rough_case(FVSolverConfig(time_integrator=time_integrator))
    assert case[0].num_ghost_cells == 2
    _assert_two_more_layers_change_nothing(case)
    # ... and not fewer: with one layer the limited slope in the cell at the wall reads the opposite edge
    assert np.max(np.abs(_final_state(case, extra_layers=-1) - _final_state(case))) > 1e-6


@pytest.mark.parametrize("option", list(FD_HYDRO_OPTIONS))
def test_fd_hydro_more_ghost_cells_give_the_same_result(option):
    """Finite-difference hydrodynamics with SSPRK(5,4), which fills the boundary before each of its five stages."""
    options, layers = FD_HYDRO_OPTIONS[option]
    case = _rough_case(FDSolverConfig(time_integrator=SSPRK54), **options)
    assert case[0].num_ghost_cells == layers
    _assert_two_more_layers_change_nothing(case)


@pytest.mark.parametrize("time_integrator", [SSPRK54, LSRK4])
def test_fd_mhd_more_ghost_cells_give_the_same_result(time_integrator):
    """Finite-difference MHD in 2D: the face-field update reads 9 layers (6 for the electric field, 3 more for
    the cell-centred field rebuilt from the faces), further than the flux difference."""
    case = _rough_case(FDSolverConfig(time_integrator=time_integrator), dimensionality=2, mhd=True)
    assert case[0].num_ghost_cells == 9
    _assert_two_more_layers_change_nothing(case)


def test_the_user_count_is_raised_and_never_lowered():
    """``finalize_config`` keeps a larger number of layers and raises a smaller one to what the scheme reads."""
    for solver_config, dimensionality in ((FVSolverConfig(time_integrator=VL2), 1), (FDSolverConfig(), 2)):
        assert _rough_case(solver_config, dimensionality, num_ghost_cells=20)[0].num_ghost_cells == 20
    assert _rough_case(FVSolverConfig(time_integrator=RK2_SSP), num_ghost_cells=1)[0].num_ghost_cells == 2
