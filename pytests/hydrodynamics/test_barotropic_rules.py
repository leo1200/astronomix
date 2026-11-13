"""
What ``finalize_config`` rejects for a barotropic gas (``ISOTHERMAL``, ``POLYTROPIC``).

A barotropic gas has no energy equation. The finite-volume framework runs it with the Riemann solvers that need
no energy jump (``HLL``, ``LAX_FRIEDRICHS``), with gravity and viscosity; everything that acts on the energy equation
is rejected in every scheme with a message that names what to use instead. One line per rule.

Run on the CPU::

    JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest pytests/hydrodynamics/test_barotropic_rules.py
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# testing
import pytest

# numerics
import jax

jax.config.update("jax_enable_x64", True)

# astronomix constants
from astronomix import FINITE_DIFFERENCE, FINITE_VOLUME, PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    DONOR_CELL,
    HLL,
    HLLC,
    HLLD,
    IDEAL_GAS,
    ISOTHERMAL,
    LAX_FRIEDRICHS,
    MUSCL_HANCOCK,
    PIECEWISE_LINEAR,
    POLYTROPIC,
    RK2_SSP,
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
    SimulationConfig,
    SnapshotSettings,
)
from astronomix._modules._cooling.cooling_options import CoolingConfig
from astronomix._modules._stellar_wind.stellar_wind_options import WindConfig

# astronomix functions
from astronomix import finalize_config

CELLS = 16
PERIODIC = BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY)


def _finalized(equation_of_state=ISOTHERMAL, solver_overrides=None, **overrides):
    """Finalise a periodic 3D configuration; a barotropic hydrodynamic state has four rows, an MHD state seven."""
    settings = dict(
        dimensionality=3,
        box_size=1.0,
        num_cells=CELLS,
        boundary_settings=BoundarySettings(PERIODIC, PERIODIC, PERIODIC),
        solver_mode=FINITE_VOLUME,
        equation_of_state=equation_of_state,
        solver_config=FVSolverConfig(
            **{**dict(time_integrator=RK2_SSP, riemann_solver=HLL), **(solver_overrides or {})}
        ),
        mhd=False,
        progress_bar=False,
    )
    settings.update(overrides)
    num_variables = (7 if settings["mhd"] else 4) + (equation_of_state == IDEAL_GAS)
    return finalize_config(SimulationConfig(**settings), (num_variables, CELLS, CELLS, CELLS))


@pytest.mark.parametrize(
    "overrides, message",
    [
        # The finite-difference scheme knows the isothermal gas only.
        (
            dict(equation_of_state=POLYTROPIC, solver_mode=FINITE_DIFFERENCE, solver_config=FDSolverConfig()),
            "solver_mode=FINITE_VOLUME, or equation_of_state=ISOTHERMAL",
        ),
        # The solvers that restore the contact need the energy jump.
        (dict(solver_overrides=dict(riemann_solver=HLLC)), "use riemann_solver=HLL"),
        (dict(equation_of_state=POLYTROPIC, solver_overrides=dict(riemann_solver=HLLC)), "use riemann_solver=HLL"),
        (dict(solver_overrides=dict(time_integrator=MUSCL_HANCOCK, riemann_solver=HLLC)), "use riemann_solver=HLL"),
        (dict(solver_overrides=dict(time_integrator=VL2, riemann_solver=HLLC)), "use riemann_solver=HLL"),
        (dict(mhd=True, solver_overrides=dict(time_integrator=VL2, riemann_solver=HLLD)), "use riemann_solver=HLL"),
        # What acts on the energy equation.
        (dict(cooling_config=CoolingConfig(cooling=True)), "cooling_config.cooling acts on the energy equation"),
        (
            dict(diffusion_config=DiffusionConfig(thermal_conduction=True)),
            "diffusion_config.thermal_conduction acts on the energy equation",
        ),
        (
            dict(gravity_config=GravityConfig(self_gravity=True, time_centred_energy_source=True)),
            "gravity_config.time_centred_energy_source acts on the energy equation",
        ),
        # The stellar wind injects thermal energy; its tracer follows the wind.
        (dict(wind_config=WindConfig(stellar_wind=True)), "wind_config.stellar_wind acts on the energy equation"),
        (
            dict(
                wind_config=WindConfig(trace_wind_density=True),
                solver_overrides=dict(time_integrator=MUSCL_HANCOCK),
            ),
            "wind_config.trace_wind_density acts on the energy equation",
        ),
        # The same in the finite-difference schemes.
        (
            dict(
                solver_mode=FINITE_DIFFERENCE,
                solver_config=FDSolverConfig(),
                cooling_config=CoolingConfig(cooling=True),
            ),
            "cooling_config.cooling acts on the energy equation",
        ),
        # A thermal diagnostic reads the pressure.
        (
            dict(snapshot_settings=SnapshotSettings(return_temperature_pdf=True)),
            "snapshot_settings.return_temperature_pdf reads the pressure",
        ),
    ],
)
def test_rejected_for_a_barotropic_gas(overrides, message):
    with pytest.raises(ValueError, match=message) as error:
        _finalized(**overrides)
    # Every message names what to use instead.
    assert "use " in str(error.value) or "switch it off" in str(error.value)


def test_accepted_for_a_barotropic_gas():
    """The combinations the framework carries pass, and the finite-difference isothermal run is not touched."""
    for equation_of_state in (ISOTHERMAL, POLYTROPIC):
        for time_integrator in (RK2_SSP, MUSCL_HANCOCK, VL2):
            # (The VL2 scheme has the Lax-Friedrichs solver with donor cells only.)
            for riemann_solver, reconstruction in ((HLL, PIECEWISE_LINEAR), (LAX_FRIEDRICHS, DONOR_CELL)):
                config = _finalized(
                    equation_of_state,
                    solver_overrides=dict(
                        time_integrator=time_integrator, riemann_solver=riemann_solver, reconstruction=reconstruction
                    ),
                    gravity_config=GravityConfig(self_gravity=True),
                    diffusion_config=DiffusionConfig(viscosity=True),
                )
                assert config.solver_config.riemann_solver == riemann_solver
                assert config.solver_config.time_integrator == time_integrator
    _finalized(ISOTHERMAL, solver_mode=FINITE_DIFFERENCE, solver_config=FDSolverConfig(), mhd=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x"]))
