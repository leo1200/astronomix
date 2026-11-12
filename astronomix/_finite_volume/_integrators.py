"""
Time integrators of the finite-volume framework: an integrator is a sequence
of stages (``_stage.py``).

``vl2_step`` is the VL2 finite-volume scheme of AthenaPK, ``rk2_ssp_step`` the
unsplit second-order strong-stability-preserving Runge-Kutta scheme
(hydrodynamics): two forward Euler stages and the average with the initial
state::

    U^(1)    = U^n   - dt L(W^n)
    U^(2)    = U^(1) - dt L(W^(1))
    U^{n+1}  = (U^n + U^(2)) / 2

with the reconstruction, the limiter and the Riemann solver of the
configuration. Gravity couples to it as a source around the step
(``evolve_state.py``).

``vl2_step``: the VL2 finite-volume scheme of AthenaPK.

A re-implementation of AthenaPK's default second-order scheme: the van Leer
predictor-corrector ("VL2", Stone & Gardiner 2009) with a donor-cell predictor
half step and a piecewise-linear (harmonic van Leer slope) corrector step,
AthenaPK's HLLD / HLLE / HLLC / LLF interface solvers, and — for MHD — the
cell-centred GLM-MHD system of Dedner et al. (2002) with the cleaning scalar psi
carried as a ninth variable. AthenaPK's first-order flux correction and its
density / pressure floors are available as options. Self-gravity (and an
external potential) couple as in Athena++, whose VL2 integrator this is: every
stage adds the source of ``_add_athena_gravity_source`` built from the
potential of the stage's initial state and the stage's Riemann mass fluxes,
optionally with the time-centred energy source of Mullen, Hanawa & Gammie
(2021). The gravity source runs on the native backend only. With
``divergence_control=CONSTRAINED_TRANSPORT`` the same step keeps the magnetic
field divergence-free with the face-centred field of
``_divergence/_constrained_transport.py`` instead of the cleaning scalar.

One step of size ``dt`` from the primitive state ``W^n`` (conserved form
``U^n``) reads, exactly as AthenaPK's task list::

    c_h      = max over cells and axes of |v_d| + c_fast,d     (GLM-MHD only)
    U^{n+½}  = U^n - dt/2 L_DC(W^n)       then  psi *= exp(-α c_h (dt/2) / dx)
    W^{n+½}  = W(U^{n+½})
    U^{n+1}  = U^n - dt   L_PLM(W^{n+½})  then  psi *= exp(-α c_h dt / dx)
    W^{n+1}  = W(U^{n+1})

with the flux divergence ``L(W)_i = (1/dx) Σ_d (F_{i+½} - F_{i-½})``. With
gravity each stage adds ``beta dt S(rho_stage, F_stage, phi_stage)`` to its
update, where ``phi_stage`` is the potential of the stage's initial density
(two Poisson solves per step, as in Athena++; three with the time-centred
energy source).

The per-face physics (reconstruction, Riemann solvers, conversions) is written as
elementwise functions that the Pallas kernels in ``_stage_pallas.py`` call as
well, so both backends evaluate the same expressions. The scheme reproduces
AthenaPK to round-off on its standard test problems
(``pytests/mhd/vl2_athenapk_regression.py``; the full comparison is documented in
``examples/scripts/validation/athenapk_vl2/README.md``).
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

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSTRAINED_TRANSPORT,
    DONOR_CELL,
    GLM,
    STATE_TYPE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_volume._divergence._constrained_transport import (
    _ct_layout_from_state,
    _state_from_ct_layout,
)
from astronomix._finite_volume._divergence._glm import _cleaning_speed_and_damping
from astronomix._finite_volume._gravity_coupling import _vl2_gravitational_potential
from astronomix._finite_volume._stage import (
    _conserved_from_primitive,
    _primitive_from_conserved,
    _update_ghost_cells,
    fv_stage,
)
from astronomix._finite_volume._stage_pallas import (
    _fv_stage_pallas,
    _fv_stage_pallas_supported,
)


# -------------------------------------------------------------
# ===================== ↓ VL2 time step ↓ =====================
# -------------------------------------------------------------


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def vl2_step(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Advance the primitive state by one VL2 step: a donor-cell predictor to the
    half step, then the corrector with the van Leer limited PLM, or with donor
    cell when ``reconstruction=DONOR_CELL`` is set (``finalize_config`` sets the
    ``VAN_LEER`` limiter). Runs on the Pallas backend when it supports the configuration and
    natively otherwise; with gravity or constrained transport always natively.

    Args:
        primitive_state: The primitive state ``W^n``. With constrained
            transport its layout is ``(rho, v_x, v_y, v_z, p, B_x, B_y, B_z,
            b_x, b_y, b_z)``, the face rows holding the value at ``i + 1/2``.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data (unused, kept for the common signature).
        registered_variables: The registered variables.

    Returns:
        The primitive state ``W^{n+1}``.
    """
    # Constrained transport works on a layout of its own (three spatial axes,
    # faces stored at their lower index); the step converts on entry and exit.
    if config.solver_config.divergence_control == CONSTRAINED_TRANSPORT:
        primitive_state = _ct_layout_from_state(primitive_state, config, registered_variables)

    # Updates between steps (turbulent forcing, cooling, ...) may have changed
    # the physical cells without touching the halo.
    primitive_state = _update_ghost_cells(primitive_state, config, params, registered_variables)

    # The cleaning speed and the psi damping are quantities of the GLM
    # cleaning; constrained transport has none.
    cleaning_speed, damping_factors = None, (None, None)
    if config.solver_config.divergence_control == GLM:
        cleaning_speed, damping_factors = _cleaning_speed_and_damping(
            primitive_state,
            dt,
            config,
            params,
            registered_variables,
        )
    predictor_damping_factor, corrector_damping_factor = damping_factors

    # The gravity source needs every face's mass flux, which the fused Pallas
    # stage kernels never write out (they form the flux divergence in
    # registers), so with gravity the stages run natively.
    if not config.gravity_config.gravity and _fv_stage_pallas_supported(primitive_state, config):

        # --------------- ↓ Pallas stages ↓ ----------------

        # The Pallas stages take the base register in primitive form and
        # recompute U^n in-kernel, which saves a state-sized buffer. In the
        # predictor the base state is the stage state itself.
        half_step_primitive_state = _fv_stage_pallas(
            primitive_state,
            base_primitive_state=None,
            stage_time_step=0.5 * dt,
            reconstruction=DONOR_CELL,
            cleaning_speed=cleaning_speed,
            damping_factor=predictor_damping_factor,
            gamma=gamma,
            config=config,
            params=params,
            registered_variables=registered_variables,
        )
        half_step_primitive_state = _update_ghost_cells(
            half_step_primitive_state,
            config,
            params,
            registered_variables,
        )
        new_primitive_state = _fv_stage_pallas(
            half_step_primitive_state,
            base_primitive_state=primitive_state,
            stage_time_step=dt,
            reconstruction=config.solver_config.reconstruction,
            cleaning_speed=cleaning_speed,
            damping_factor=corrector_damping_factor,
            gamma=gamma,
            config=config,
            params=params,
            registered_variables=registered_variables,
        )

        # --------------- ↑ Pallas stages ↑ ----------------

        return _update_ghost_cells(new_primitive_state, config, params, registered_variables)

    # --------------- ↓ Native stages ↓ ----------------

    # The potential of each stage's initial state: Athena++ solves the
    # Poisson equation after every stage, so the predictor sees phi(rho^n) and
    # the corrector phi(rho^{n+1/2}).
    initial_potential = None
    if config.gravity_config.gravity:
        initial_potential = _vl2_gravitational_potential(
            primitive_state[registered_variables.density_index],
            config,
            params,
            registered_variables,
        )

    initial_conserved_state = _conserved_from_primitive(
        primitive_state,
        gamma,
        config,
        registered_variables,
    )
    half_step_primitive_state, half_step_end_potential = fv_stage(
        primitive_state,
        initial_conserved_state,
        stage_time_step=0.5 * dt,
        reconstruction=DONOR_CELL,
        cleaning_speed=cleaning_speed,
        damping_factor=predictor_damping_factor,
        gamma=gamma,
        config=config,
        params=params,
        registered_variables=registered_variables,
        stage_potential=initial_potential,
        initial_potential=initial_potential,
    )
    half_step_primitive_state = _update_ghost_cells(
        half_step_primitive_state,
        config,
        params,
        registered_variables,
    )

    # The time-centred energy source has already solved for the potential of
    # the half-step density (before the floors, which only matters where they
    # act), so it is reused rather than solved again.
    half_step_potential = half_step_end_potential
    if config.gravity_config.gravity and half_step_potential is None:
        half_step_potential = _vl2_gravitational_potential(
            half_step_primitive_state[registered_variables.density_index],
            config,
            params,
            registered_variables,
        )

    new_primitive_state, _ = fv_stage(
        half_step_primitive_state,
        initial_conserved_state,
        stage_time_step=dt,
        reconstruction=config.solver_config.reconstruction,
        cleaning_speed=cleaning_speed,
        damping_factor=corrector_damping_factor,
        gamma=gamma,
        config=config,
        params=params,
        registered_variables=registered_variables,
        stage_potential=half_step_potential,
        initial_potential=initial_potential,
    )

    # --------------- ↑ Native stages ↑ ----------------

    new_primitive_state = _update_ghost_cells(new_primitive_state, config, params, registered_variables)
    if config.solver_config.divergence_control == CONSTRAINED_TRANSPORT:
        new_primitive_state = _state_from_ct_layout(new_primitive_state, config, registered_variables)
    return new_primitive_state


# -------------------------------------------------------------
# ===================== ↑ VL2 time step ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ RK2 SSP time step ↓ ===================
# -------------------------------------------------------------


def _forward_euler_stage(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    One forward Euler stage ``U - dt L(W)`` from the state itself, on the
    Pallas backend when it supports the configuration and natively otherwise.

    Args:
        primitive_state: The primitive state ``W`` (ghost cells updated).
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        registered_variables: The registered variables.

    Returns:
        The primitive state after the stage, ghost cells updated.
    """
    # Hydrodynamics: a zero cleaning speed and no damping.
    cleaning_speed, (_, damping_factor) = _cleaning_speed_and_damping(
        primitive_state,
        dt,
        config,
        params,
        registered_variables,
    )
    stage_arguments = dict(
        stage_time_step=dt,
        reconstruction=config.solver_config.reconstruction,
        cleaning_speed=cleaning_speed,
        damping_factor=damping_factor,
        gamma=gamma,
        config=config,
        params=params,
        registered_variables=registered_variables,
    )
    if _fv_stage_pallas_supported(primitive_state, config):
        new_primitive_state = _fv_stage_pallas(primitive_state, base_primitive_state=None, **stage_arguments)
    else:
        conserved_state = _conserved_from_primitive(primitive_state, gamma, config, registered_variables)
        new_primitive_state, _ = fv_stage(primitive_state, conserved_state, **stage_arguments)
    return _update_ghost_cells(new_primitive_state, config, params, registered_variables)


@partial(jax.jit, static_argnames=["config", "registered_variables"])
def rk2_ssp_step(
    primitive_state: STATE_TYPE,
    dt: Float[Array, ""],
    gamma: Union[float, Float[Array, ""]],
    config: SimulationConfig,
    params: SimulationParams,
    helper_data: HelperData,
    registered_variables: RegisteredVariables,
) -> STATE_TYPE:
    """
    Advance the primitive state by one step of the second-order
    strong-stability-preserving Runge-Kutta scheme: two forward Euler stages
    and the average with the initial state in conserved variables.

    Args:
        primitive_state: The primitive state ``W^n``.
        dt: The time step.
        gamma: The adiabatic index.
        config: The simulation configuration.
        params: The simulation parameters.
        helper_data: The helper data (unused, kept for the common signature).
        registered_variables: The registered variables.

    Returns:
        The primitive state ``W^{n+1}``.
    """
    # Updates between steps (turbulent forcing, cooling, ...) may have changed
    # the physical cells without touching the halo.
    primitive_state = _update_ghost_cells(primitive_state, config, params, registered_variables)

    first_stage_state = _forward_euler_stage(primitive_state, dt, gamma, config, params, registered_variables)
    second_stage_state = _forward_euler_stage(first_stage_state, dt, gamma, config, params, registered_variables)

    initial_conserved_state = _conserved_from_primitive(primitive_state, gamma, config, registered_variables)
    second_conserved_state = _conserved_from_primitive(second_stage_state, gamma, config, registered_variables)
    new_primitive_state = _primitive_from_conserved(
        0.5 * (initial_conserved_state + second_conserved_state),
        gamma,
        config,
        params,
        registered_variables,
    )
    return _update_ghost_cells(new_primitive_state, config, params, registered_variables)


# -------------------------------------------------------------
# =================== ↑ RK2 SSP time step ↑ ===================
# -------------------------------------------------------------
