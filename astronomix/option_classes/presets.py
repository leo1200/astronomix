"""
Presets: named solver settings.

A preset is a complete solver container, an :class:`FVSolverConfig` or an
:class:`FDSolverConfig`, for ``SimulationConfig.solver_config``. A script names
the solver mode and the preset and changes single settings with ``_replace``::

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        solver_config=ATHENAPK_VL2._replace(riemann_solver=HLLC),
        ...
    )

The settings outside the container that belong to a scheme (the CFL number,
the equation of state, ``mhd``) stay with the script.
"""

from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD,
    FDSolverConfig,
    FVSolverConfig,
    GLM,
    HLL5R,
    HLLD,
    INDUCTION_SOURCE,
    MUSCL_HANCOCK,
    PIECEWISE_LINEAR,
    RIEMANN_FLUX,
    SSPRK33,
    SSPRK54,
    VAN_LEER,
    VL2,
)

#: The scheme of AthenaPK (finite volume, ``solver_mode=FINITE_VOLUME``): the
#: VL2 predictor-corrector with van Leer-limited piecewise-linear
#: reconstruction, HLLD, GLM divergence cleaning and the first-order flux
#: correction. For hydrodynamics set ``riemann_solver=HLLC``.
ATHENAPK_VL2 = FVSolverConfig(
    time_integrator=VL2,
    reconstruction=PIECEWISE_LINEAR,
    limiter=VAN_LEER,
    riemann_solver=HLLD,
    divergence_control=GLM,
    first_order_flux_correction=True,
)

#: The scheme of the 10080^3 turbulence run of Beattie et al. (2025), as published by Waagan, Federrath &
#: Klingenberg (2011) (finite volume, ``solver_mode=FINITE_VOLUME``): dimensionally split MUSCL-Hancock on
#: primitive variables with monotonised-central slopes, the HLL5R relaxation solver, Powell's term in the
#: induction equation only and parabolic divergence cleaning; cell-centred field. Use with ``mhd=True``,
#: ``equation_of_state=ISOTHERMAL`` (the ideal gas runs as well) and a CFL number of 0.8.
BEATTIE_REFERENCE = FVSolverConfig(
    time_integrator=MUSCL_HANCOCK,
    reconstruction=PIECEWISE_LINEAR,
    limiter=DOUBLE_MINMOD,
    riemann_solver=HLL5R,
    divergence_control=INDUCTION_SOURCE,
    parabolic_cleaning=True,
)

#: The production scheme of the paper (finite difference,
#: ``solver_mode=FINITE_DIFFERENCE``): WENO-Z flux splitting with SSPRK(5,4)
#: and the upwind dissipation of the constrained-transport EMFs. Runs at Mach
#: numbers of two and above add ``weno_positivity_preserving=True``.
FD_PAPER = FDSolverConfig(
    time_integrator=SSPRK54,
    weno_z=True,
    ct_upwind_dissipation=True,
    ct_upwind_strength=0.03,
)

#: The finite-difference scheme with a Riemann flux for isothermal MHD: HLLEM_T
#: flux, flux-form constrained transport, SSPRK(3,3). Use with ``mhd=True``,
#: ``equation_of_state=ISOTHERMAL`` and a CFL number of 1.0.
FD_RIEMANN_ISOTHERMAL = FDSolverConfig(scheme=RIEMANN_FLUX, time_integrator=SSPRK33)

#: The finite-difference scheme with a Riemann flux for ideal-gas MHD:
#: HLLD_BOUNDED flux, flux-form constrained transport with the energy held in
#: the field rebuild, SSPRK(5,4). Use with ``mhd=True``,
#: ``equation_of_state=IDEAL_GAS`` and a CFL number of 1.0.
FD_RIEMANN_IDEAL_GAS = FDSolverConfig(scheme=RIEMANN_FLUX, time_integrator=SSPRK54)
