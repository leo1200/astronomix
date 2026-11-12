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
    FDSolverConfig,
    FVSolverConfig,
    GLM,
    HLLD,
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
