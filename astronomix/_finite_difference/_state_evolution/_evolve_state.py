"""
One finite-difference time step on the primitive state.

Converts the primitive state to conserved variables, dispatches to the
requested Runge-Kutta integrator (SSPRK4 or low-storage RK4, with constrained
transport for MHD), converts back to primitives, and re-fills ghost cells when
ghost-cell boundaries are in use.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import Array, Float
from beartype import beartype as typechecker

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    GHOST_CELLS,
    IDEAL_GAS,
    ISOTHERMAL,
    RK4_LSRK,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._equations import (
    conserved_state_from_primitive,
    primitive_state_from_conserved,
)
from astronomix._fluid_equations._equations_mhd import (
    conserved_state_from_primitive_isothermal,
    conserved_state_from_primitive_mhd,
    primitive_state_from_conserved_isothermal,
    primitive_state_from_conserved_mhd,
)
from astronomix._fluid_equations._dual_energy import advect_internal_energy
from astronomix._fluid_equations._passive_scalars import (
    _fill_scalar_ghost_cells,
    advect_passive_scalars,
    update_shock_history,
)
from astronomix.variable_registry.registered_variables import NUM_SHOCK_HISTORY_SCALARS
from astronomix._finite_difference._magnetic_update._constrained_transport import update_cell_center_fields
from astronomix._finite_difference._time_integrators._ssprk import (
    _lsrk4_hydro,
    _lsrk4_with_ct,
    _ssprk4_hydro,
    _ssprk4_with_ct,
)
from astronomix._geometry.boundaries import _boundary_handler


@partial(jax.jit, static_argnames=["config", "registered_variables"], donate_argnames=["primitive_state"])
def _evolve_state_fd(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Advance the primitive state by one finite-difference time step.

    Args:
        primitive_state: The primitive state array.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data.
        registered_variables: The registered variables.

    Returns:
        The primitive state after one time step.
    """

    # Passive scalars occupy a contiguous block at the very END of the state
    # (after the dual-energy ``g``, hence after the MHD interface field). Split
    # them off FIRST so every downstream slicing convention sees exactly the
    # state it was written for; they are advected operator-split with the
    # pre-step flow and reattached at the end.
    _scalars = registered_variables.passive_scalars_active
    _shock_history = registered_variables.shock_history_active
    passive_scalars = None
    if _scalars:
        _sidx = registered_variables.passive_scalar_index
        passive_scalars = primitive_state[_sidx:]
        primitive_state = primitive_state[:_sidx]
        registered_variables = registered_variables._replace(
            num_vars=_sidx, passive_scalar_index=-1, num_passive_scalars=0,
            passive_scalars_active=False, shock_history_active=False,
        )
        # advect with the PRE-step flow (operator splitting); the hydro CFL is
        # set by |u| + c and is therefore always stricter than |u|, so a single
        # SSP-RK3 step over dt is stable without substepping
        passive_scalars = advect_passive_scalars(
            passive_scalars, primitive_state, dt, config.grid_spacing,
            config, registered_variables,
        )

    # Dual-energy formalism: the internal-energy density ``g`` is carried as the
    # LAST variable of the state. Split it off so the hydro/MHD machinery sees
    # the standard state (for MHD: interface B as the last three vars), advect
    # it with the pre-step flow, and feed it into the coupled WENO pressure
    # recovery; it is re-synced from the recovered pressure and reattached at
    # the end.
    _dual = registered_variables.internal_energy_active
    internal_energy_density = None
    if _dual:
        _gidx = registered_variables.internal_energy_index
        internal_energy_density = primitive_state[_gidx]
        primitive_state = primitive_state[:_gidx]
        registered_variables = registered_variables._replace(
            num_vars=_gidx, internal_energy_index=-1, internal_energy_active=False,
        )
        if config.mhd:
            _g_cons = conserved_state_from_primitive_mhd(
                primitive_state[:-3], gamma, registered_variables
            )
        else:
            _g_cons = conserved_state_from_primitive(
                primitive_state, gamma, config, registered_variables
            )
        internal_energy_density = advect_internal_energy(
            internal_energy_density, _g_cons,
            primitive_state[registered_variables.pressure_index],
            dt, config.grid_spacing, config, registered_variables,
        )
        del _g_cons
        # g is an internal-energy density: the explicit div(gv) + p div(v)
        # update can undershoot zero in strongly-expanding (homologous-ejecta)
        # cells, and a negative g would feed a negative switched pressure into
        # the WENO flux. Hold it to the same floor as the pressure.
        internal_energy_density = jnp.maximum(
            internal_energy_density, params.minimum_pressure / (gamma - 1.0)
        )

    if config.mhd:
        # NOTE: here we assume the magnetic field at interfaces
        # is stored in the last three indices of the state array
        if config.equation_of_state == IDEAL_GAS:
            conserved_state = conserved_state_from_primitive_mhd(
                primitive_state[:-3], gamma, registered_variables
            )
        elif config.equation_of_state == ISOTHERMAL:
            conserved_state = conserved_state_from_primitive_isothermal(
                primitive_state[:-3], config, registered_variables
            )

        # extract interface magnetic fields
        bxb = primitive_state[registered_variables.interface_magnetic_field_index.x]
        byb = primitive_state[registered_variables.interface_magnetic_field_index.y]
        bzb = primitive_state[registered_variables.interface_magnetic_field_index.z]

        # update conserved state and interface magnetic fields — RK4_LSRK
        # selects the 2N-storage Carpenter-Kennedy LSRK4 variant (saves one
        # conserved + three interface-B carry registers vs SSPRK4).
        if config.time_integrator == RK4_LSRK:
            mhd_integrator = _lsrk4_with_ct
        else:
            mhd_integrator = _ssprk4_with_ct

        conserved_state, bxb, byb, bzb = mhd_integrator(
            conserved_state,
            bxb,
            byb,
            bzb,
            gamma,
            config.grid_spacing,
            dt,
            params,
            helper_data,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )

        # back to primitive state
        if config.equation_of_state == IDEAL_GAS:
            primitive_state = primitive_state_from_conserved_mhd(
                conserved_state, params.minimum_density, params.minimum_pressure, gamma, config, registered_variables,
                internal_energy_density=internal_energy_density,
            )
        elif config.equation_of_state == ISOTHERMAL:
            primitive_state = primitive_state_from_conserved_isothermal(
                conserved_state, params.minimum_density, config, registered_variables
            )

        # append updated interface magnetic fields
        # NOTE: same assumption as above
        primitive_state = jnp.concatenate(
            [primitive_state, bxb[None, :], byb[None, :], bzb[None, :]], axis=0
        )
    else:
        conserved_state = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )

        # Dispatch to the requested time integrator.  RK4_LSRK is the
        # Carpenter-Kennedy 2N-storage low-storage RK4 (one fewer full-state
        # register than SSPRK4, at the cost of a smaller stability CFL).
        if int(config.time_integrator) == RK4_LSRK:
            integrator = _lsrk4_hydro
        else:
            integrator = _ssprk4_hydro

        conserved_state = integrator(
            conserved_state,
            gamma,
            config.grid_spacing,
            dt,
            params,
            helper_data,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )

        primitive_state = primitive_state_from_conserved(
            conserved_state,
            gamma,
            config,
            registered_variables,
            internal_energy_density=internal_energy_density,
        )

    # When ghost-cell boundaries are in use, refill the ghost zones from the
    # updated interior so the next step sees a consistent boundary state.
    if config.boundary_handling == GHOST_CELLS:
        primitive_state = _boundary_handler(
            primitive_state, config, registered_variables, params
        )

    # dual-energy: re-sync g from the recovered (switched) pressure and reattach
    # it as the last variable so the carried state stays self-consistent. The
    # recovered pressure is floored here too — the coupled recovery itself does
    # not floor, and a switch-active cell must never hand a sub-floor g to the
    # next step.
    if _dual:
        g_new = jnp.maximum(
            primitive_state[registered_variables.pressure_index],
            params.minimum_pressure,
        ) / (gamma - 1.0)
        primitive_state = jnp.concatenate(
            [primitive_state, g_new[None, :]], axis=0
        )

    # Passive scalars: apply the shock bookkeeping against the UPDATED state
    # (the entropy comparison needs the post-shock entropy), fill their ghost
    # cells, and reattach the block at the end of the state.
    if _scalars:
        if _shock_history:
            n_hist = NUM_SHOCK_HISTORY_SCALARS
            passive_scalars = jnp.concatenate(
                [passive_scalars[:-n_hist],
                 update_shock_history(
                     passive_scalars[-n_hist:], primitive_state, dt, gamma,
                     config.shock_entropy_jump, config, registered_variables)],
                axis=0,
            )
        if config.boundary_handling == GHOST_CELLS:
            passive_scalars = _fill_scalar_ghost_cells(passive_scalars, config)
        primitive_state = jnp.concatenate([primitive_state, passive_scalars], axis=0)

    return primitive_state