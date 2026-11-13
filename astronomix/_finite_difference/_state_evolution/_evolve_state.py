"""
One finite-difference time step on the primitive state.

Converts the primitive state to conserved variables, dispatches to the
requested Runge-Kutta integrator (SSPRK4 or low-storage RK4, with constrained
transport for MHD), converts back to primitives, and re-fills ghost cells when
ghost-cell boundaries are in use.

The rows the finite-difference solver appends behind the gas / field variables
are handled around that update by operator splitting: the passive-scalar block
and the dual-energy internal-energy density ``g`` are split off first and
advected with the pre-step flow (``g`` also enters the coupled pressure
recovery), then ``g`` is re-synced from the recovered pressure and both are
reattached.
"""

# general
from functools import partial

# typing
from typing import Union
from jaxtyping import (
    Array,
    Float,
)

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    BACKWARDS,
    GHOST_CELLS,
    IDEAL_GAS,
    LSRK4,
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
)
from astronomix._finite_difference._time_integrators._ssprk import (
    _lsrk4_hydro,
    _lsrk4_with_ct,
    _ssprk4_hydro,
    _ssprk4_with_ct,
)
from astronomix._geometry.boundaries import _boundary_handler


@partial(
    jax.jit,
    static_argnames=["config", "registered_variables"],
    donate_argnames=["primitive_state"],
)
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

    The passive-scalar block and the dual-energy ``g`` are split off first and
    advected operator-split with the pre-step flow, so the Runge-Kutta update
    sees the plain gas / field state (``g`` enters it only through the coupled
    pressure recovery). Afterwards ``g`` is re-synced from the recovered
    pressure and both are reattached in the registered order.

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

    # -------------------------------------------------------------
    # ============= ↓ Split off and advect passive scalars ↓ =======
    # -------------------------------------------------------------

    # The passive scalars occupy a contiguous block at the very end of the
    # state (behind the dual-energy ``g``, hence behind the MHD interface
    # field). They are split off FIRST so every downstream slicing convention
    # sees exactly the state it was written for; they are advected
    # operator-split with the pre-step flow and reattached at the end.
    passive_scalars_active = registered_variables.passive_scalars_active
    passive_scalars = None
    if passive_scalars_active:
        passive_scalar_index = registered_variables.passive_scalar_index
        passive_scalars = primitive_state[passive_scalar_index:]
        primitive_state = primitive_state[:passive_scalar_index]
        registered_variables = registered_variables._replace(
            num_vars=passive_scalar_index,
            passive_scalar_index=-1,
            num_passive_scalars=0,
            passive_scalars_active=False,
        )
        # Advect with the pre-step flow (operator splitting);
        # ``advect_passive_scalars`` sub-cycles internally when the flow
        # requires it.
        passive_scalars = advect_passive_scalars(
            passive_scalars,
            primitive_state,
            dt,
            config.grid_spacing,
            config,
            registered_variables,
        )
        if config.differentiation_mode == BACKWARDS:
            # Reverse-mode memory: the scalars never feed back on the hydro, so
            # the backward pass of their advection is independent of the hydro
            # backward pass and XLA is free to schedule the two together,
            # adding the scalar sub-steps' transient memory to the hydro peak.
            # Tying the two through a barrier (a numerical no-op, whose
            # transpose is a barrier on the cotangents) makes the scalar
            # backward pass wait for the hydro one. Only on the reverse-mode
            # (masked) path, so the FORWARDS program is untouched.
            passive_scalars, primitive_state = jax.lax.optimization_barrier(
                (passive_scalars, primitive_state)
            )

    # -------------------------------------------------------------
    # ============= ↑ Split off and advect passive scalars ↑ =======
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ============ ↓ Split off and advect the dual energy ↓ ========
    # -------------------------------------------------------------

    # Dual-energy formalism: once the scalars are split off, the internal-energy
    # density ``g`` is the last row of the state. Split it off so the hydro /
    # MHD machinery sees the standard state (for MHD: the interface field as the
    # last three rows), advect it with the pre-step flow, and feed it into the
    # coupled WENO pressure recovery; it is re-synced from the recovered
    # pressure and reattached at the end.
    dual_energy_active = registered_variables.internal_energy_active
    internal_energy_density = None
    if dual_energy_active:
        internal_energy_index = registered_variables.internal_energy_index
        internal_energy_density = primitive_state[internal_energy_index]
        primitive_state = primitive_state[:internal_energy_index]
        registered_variables = registered_variables._replace(
            num_vars=internal_energy_index,
            internal_energy_index=-1,
            internal_energy_active=False,
        )
        if config.mhd:
            conserved_state_pre_step = conserved_state_from_primitive_mhd(
                primitive_state[:registered_variables.interface_magnetic_field_index.x],
                gamma,
                registered_variables,
            )
        else:
            conserved_state_pre_step = conserved_state_from_primitive(
                primitive_state,
                gamma,
                config,
                registered_variables,
            )
        internal_energy_density = advect_internal_energy(
            internal_energy_density,
            conserved_state_pre_step,
            primitive_state[registered_variables.pressure_index],
            dt,
            config.grid_spacing,
            config,
            registered_variables,
        )
        del conserved_state_pre_step
        # g is an internal-energy density: the explicit div(g v) + p div(v)
        # update can undershoot zero in strongly expanding cells, and a negative
        # g would feed a negative switched pressure into the WENO flux. Hold it
        # to the same floor as the pressure.
        internal_energy_density = jnp.maximum(
            internal_energy_density, params.minimum_pressure / (gamma - 1.0)
        )

    # -------------------------------------------------------------
    # ============ ↑ Split off and advect the dual energy ↑ ========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =============== ↓ Hydro / MHD Runge-Kutta update ↓ ===========
    # -------------------------------------------------------------

    if config.mhd:
        # The interface magnetic field occupies the last three rows of the
        # (scalar- and g-free) state; the conserved state is built from the
        # cell-centred rows in front of it.
        interface_field_start = registered_variables.interface_magnetic_field_index.x
        if config.equation_of_state == IDEAL_GAS:
            conserved_state = conserved_state_from_primitive_mhd(
                primitive_state[:interface_field_start], gamma, registered_variables
            )
        else:
            # a barotropic gas: density and momentum only
            conserved_state = conserved_state_from_primitive_isothermal(
                primitive_state[:interface_field_start], config, registered_variables
            )

        # extract interface magnetic fields
        interface_field_x = primitive_state[registered_variables.interface_magnetic_field_index.x]
        interface_field_y = primitive_state[registered_variables.interface_magnetic_field_index.y]
        interface_field_z = primitive_state[registered_variables.interface_magnetic_field_index.z]

        # update conserved state and interface magnetic fields — LSRK4
        # selects the 2N-storage Carpenter-Kennedy LSRK4 variant (saves one
        # conserved + three interface-B carry registers vs SSPRK4).
        if config.solver_config.time_integrator == LSRK4:
            mhd_integrator = _lsrk4_with_ct
        else:
            mhd_integrator = _ssprk4_with_ct

        conserved_state, interface_field_x, interface_field_y, interface_field_z = mhd_integrator(
            conserved_state,
            interface_field_x,
            interface_field_y,
            interface_field_z,
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
            # This state is carried to the next step, so the estimate-only
            # clamps (``clamp_in_estimates``) must not touch it.
            primitive_state = primitive_state_from_conserved_mhd(
                conserved_state,
                params.minimum_density,
                params.minimum_pressure,
                gamma,
                config,
                registered_variables,
                internal_energy_density=internal_energy_density,
                clamp=False,
            )
        else:
            primitive_state = primitive_state_from_conserved_isothermal(
                conserved_state,
                params.minimum_density,
                config,
                registered_variables,
                clamp=False,
            )

        # Append the updated interface magnetic fields as the last three rows.
        primitive_state = jnp.concatenate(
            [
                primitive_state,
                interface_field_x[None, :],
                interface_field_y[None, :],
                interface_field_z[None, :],
            ],
            axis=0,
        )
    else:
        conserved_state = conserved_state_from_primitive(
            primitive_state, gamma, config, registered_variables
        )

        # Dispatch to the requested time integrator.  LSRK4 is the
        # Carpenter-Kennedy 2N-storage low-storage RK4 (one fewer full-state
        # register than SSPRK4, at the cost of a smaller stability CFL).
        if int(config.solver_config.time_integrator) == LSRK4:
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

    # -------------------------------------------------------------
    # =============== ↑ Hydro / MHD Runge-Kutta update ↑ ===========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ Re-sync and reattach the g row ↓ ==========
    # -------------------------------------------------------------

    # Re-sync g from the recovered (switched) pressure and reattach it behind
    # the gas / field rows, so the carried state stays self-consistent. The
    # recovered pressure is floored here too: the coupled recovery itself does
    # not floor, and a switch-active cell must never hand a sub-floor g to the
    # next step.
    if dual_energy_active:
        resynced_internal_energy_density = jnp.maximum(
            primitive_state[registered_variables.pressure_index],
            params.minimum_pressure,
        ) / (gamma - 1.0)
        primitive_state = jnp.concatenate(
            [primitive_state, resynced_internal_energy_density[None, :]], axis=0
        )

    # -------------------------------------------------------------
    # ================ ↑ Re-sync and reattach the g row ↑ ==========
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # =================== ↓ Reattaching the scalars ↓ ==============
    # -------------------------------------------------------------

    # Fill the scalars' ghost cells and reattach the block at the end of the
    # state.
    if passive_scalars_active:
        if config.boundary_handling == GHOST_CELLS:
            passive_scalars = _fill_scalar_ghost_cells(passive_scalars, config)
        primitive_state = jnp.concatenate([primitive_state, passive_scalars], axis=0)

    # -------------------------------------------------------------
    # =================== ↑ Reattaching the scalars ↑ ==============
    # -------------------------------------------------------------

    return primitive_state
