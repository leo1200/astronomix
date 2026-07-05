"""
Exposed right-hand side of the fluid equations for analysis.

Recomputes the semi-discrete RHS ``dU/dt = R(U)`` of the finite-difference
hydrodynamics solver as a standalone, differentiable function of the conserved
state. This is what the Jacobian helpers linearise; it is deliberately kept
outside the time-stepping machinery so it can be fed to ``jax.jvp`` directly.
"""

# general
from functools import partial

# jax
import jax

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSERVATIVE_GAS_STATE,
    FINITE_DIFFERENCE,
    FINITE_VOLUME,
)

# astronomix functions
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)
from astronomix._fluid_equations._equations import primitive_state_from_conserved
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._modules._viscosity._viscosity import fd_viscosity_source
from astronomix._stencil_operations._stencil_operations import _shift
from astronomix.time_stepping._utils import _pad, _unpad


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def _exposed_rhs(conserved_state, params, config, registered_variables):
    """
    Compute the right-hand side ``R(U)`` of the fluid equations.

    Only the finite-difference hydrodynamics path is implemented: the
    conserved state is padded and given boundary conditions, the WENO interface
    fluxes are differenced to form the spatial RHS, and any active source terms
    (currently viscosity) are added before the ghost cells are stripped again.

    Args:
        conserved_state: The (unpadded) conserved fluid state ``U``.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The (unpadded) right-hand side ``R(U)`` of the conserved variables.

    TODO: Refactor to share the RHS with the simulator instead of recomputing it.
    """

    # Pad the state with ghost cells so the boundary handler and stencils have
    # the halo they need.
    conserved_state = _pad(conserved_state, config)

    # Fill the ghost cells according to the configured boundary conditions.
    conserved_state = _boundary_handler(
        conserved_state, config, registered_variables, params, CONSERVATIVE_GAS_STATE
    )

    # The WENO fluxes and source terms work on primitive variables, so recover
    # them from the conserved state.
    primitive_state = primitive_state_from_conserved(
        conserved_state,
        params.gamma,
        config,
        registered_variables,
    )

    if config.solver_mode == FINITE_DIFFERENCE:
        if not config.mhd:
            dF_x = _weno_flux_x(conserved_state, params, config, registered_variables)

            if config.dimensionality >= 2:
                dF_y = _weno_flux_y(conserved_state, params, config, registered_variables)

            if config.dimensionality == 3:
                dF_z = _weno_flux_z(conserved_state, params, config, registered_variables)

            # The spatial RHS is the negative divergence of the interface
            # fluxes: in each direction the flux at the right face minus the
            # flux at the left face (recovered by shifting the interface array),
            # divided by the grid spacing.
            if config.dimensionality == 1:
                rhs_q = -1/config.grid_spacing * (
                    (dF_x - _shift(dF_x, 1, axis=1))
                )
            elif config.dimensionality == 2:
                rhs_q = -1/config.grid_spacing * (
                    (dF_x - _shift(dF_x, 1, axis=1))
                    + (dF_y - _shift(dF_y, 1, axis=2))
                )
            elif config.dimensionality == 3:
                rhs_q = -1/config.grid_spacing * (
                    (dF_x - _shift(dF_x, 1, axis=1))
                    + (dF_y - _shift(dF_y, 1, axis=2))
                    + (dF_z - _shift(dF_z, 1, axis=3))
                )
        else:
            raise NotImplementedError(
                "Extracted RHS currently only implemented for hydrodynamics."
            )

        # Add the active source terms to the RHS. Only viscosity is wired up
        # here; the other physics modules are not yet exposed in this RHS.
        if config.diffusion:
            rhs_q = fd_viscosity_source(
                primitive_state, params, config, registered_variables
            )

    elif config.solver_mode == FINITE_VOLUME:
        raise NotImplementedError(
            "Extracted RHS currently only implemented for finite difference solver."
        )

    # Strip the ghost cells so the returned RHS matches the unpadded input.
    rhs_q = _unpad(rhs_q, config)

    return rhs_q