"""
Strong Stability Preserving Runge-Kutta (SSPRK) time integrator.

See _magnetic_update/_constrained_transport.py for more details on the
Constrained Transport (CT) implementation following (Seo & Ryu 2023,
https://arxiv.org/abs/2304.04360).
"""

# general
from functools import partial

# typing
from typing import Union

# jax
import jax
import jax.numpy as jnp

# astronomix constants
from astronomix.option_classes.simulation_config import (
    CONSERVATIVE_GAS_STATE,
    GHOST_CELLS,
    MAGNETIC_FIELD_ONLY,
    SIMPLE_SOURCE,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._fluid_equations._enforce_positivity import (
    _enforce_positivity,
    _apply_stage_positivity,
)
from astronomix._finite_difference._interface_fluxes._weno import (
    _hydro_pallas_flux_supported,
    _weno_flux_x,
    _weno_flux_y,
    _weno_flux_z,
)
from astronomix._finite_difference._interface_fluxes._flux_blending import (
    _blend_interface_flux,
)
from astronomix._finite_difference._time_integrators._ssprk_pallas import (
    _div_axis_pallas_shape_ok,
    _hydro_flux_div_axis_pallas,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    _constrained_transport_rhs_from_slices,
    update_cell_center_fields,
)
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._integrators._explicit_rk import lsrk4, ssprk4
from astronomix._modules._time_integrator_sources import _time_integrator_sources
from astronomix._pallas_helpers import _backend_is_pallas, pl
from astronomix._stencil_operations._stencil_operations import _shift


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"])
def _ssprk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    Integrates the MHD equations for one time step using a 5-stage, 4th-order
    Strong Stability Preserving Runge-Kutta (SSPRK) method
    with Constrained Transport (CT).
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    # For the MHD/CT path the WENO kernel itself is still native (the Pallas
    # MHD WENO kernel is not yet written — see guide §4.1).  We can still pick
    # up the per-axis divergence accumulator + ``input_output_aliases``
    # memory win whenever the user selected the Pallas backend, since that
    # kernel is mhd-agnostic and just operates on whatever flux tensor it is
    # handed.
    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

    def rhs(u, dt_tilde):
        """
        Computes the right-hand side (RHS) of the MHD equations for a given stage.
        ``dt_tilde`` is the stage-effective step (``k * dt``); the state pytree
        ``u`` is the ``(q, bx, by, bz)`` tuple.
        """

        current_q, bx, by, bz = u

        current_q = update_cell_center_fields(
            current_q, bx, by, bz, config, registered_variables
        )

        # in the future we might support
        # different grid spacings in each direction
        dtdx = dt_tilde / grid_spacing
        dtdy = dt_tilde / grid_spacing
        dtdz = dt_tilde / grid_spacing

        # Axis-incremental flow: build each axis's full dF, extract the
        # two magnetic-flux slices CT needs (plus the density-flux slice
        # for any physics modules that consume it), consume dF for the
        # divergence step, then free dF.  CT runs at the end on the six
        # small single-channel slices instead of the three full 8-var dF
        # arrays — saves ~7/8 × 3 = 2.6× state-shape buffers at peak.
        my = registered_variables.magnetic_index.y
        mz = registered_variables.magnetic_index.z
        di = registered_variables.density_index

        # Unified flux blending (deep-void density ramp and/or FCT positivity):
        # apply to the full interface flux BEFORE the transverse magnetic-flux
        # slices are extracted, so CT consumes the blended (locally-diffusive)
        # induction flux. CT stays div(B)=0 by construction (single-valued edge
        # EMFs from consistent face fluxes).
        blend = config.positivity_config.deepvoid_blend or config.positivity_config.preserving_flux

        # x-axis
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        if blend:
            dF_x = _blend_interface_flux(dF_x, current_q, 0, dtdx, params, config, registered_variables)
        By_flux_x = dF_x[my]
        Bz_flux_x = dF_x[mz]
        density_flux_x = dF_x[di]
        if use_pallas_div:
            rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        else:
            rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x

        # y-axis
        if config.dimensionality >= 2:
            mx = registered_variables.magnetic_index.x
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            if blend:
                dF_y = _blend_interface_flux(dF_y, current_q, 1, dtdy, params, config, registered_variables)
            Bx_flux_y = dF_y[mx]
            Bz_flux_y = dF_y[mz]
            density_flux_y = dF_y[di]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
                )
            else:
                rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0

        # z-axis
        if config.dimensionality == 3:
            mx = registered_variables.magnetic_index.x
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            if blend:
                dF_z = _blend_interface_flux(dF_z, current_q, 2, dtdz, params, config, registered_variables)
            Bx_flux_z = dF_z[mx]
            By_flux_z = dF_z[my]
            density_flux_z = dF_z[di]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
                )
            else:
                rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0

        # CT now runs on the six single-channel B-flux slices only — the
        # three 8-var dF arrays have all been freed by this point.
        rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
            current_q,
            By_flux_x, Bz_flux_x,
            Bx_flux_y, Bz_flux_y,
            Bx_flux_z, By_flux_z,
            dtdx, dtdy, dtdz,
            config, registered_variables,
        )

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)


        # Add physics source terms
        rhs_q += _time_integrator_sources(
            current_q,
            density_fluxes,
            rhs_q[registered_variables.density_index], # drho
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
        )

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    def pre_stage(u):
        q, bx, by, bz = u
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config, registered_variables, params, MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]
        return (q, bx, by, bz)

    def finalize(u):
        q, bx, by, bz = u
        # Update the cell-centered magnetic fields in the conserved state array
        # from the final interface magnetic fields.
        q = update_cell_center_fields(
            q, bx, by, bz, config, registered_variables
        )
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        return (q, bx, by, bz)

    return ssprk4(
        (conserved_state, bx_interface, by_interface, bz_interface),
        dt, rhs=rhs, pre_stage=pre_stage, finalize=finalize,
    )


def _hydro_density_fluxes_needed(config) -> bool:
    """Whether any FD physics module actually consumes the per-axis density
    flux slices.  Only self-gravity variants other than SIMPLE_SOURCE do,
    so for typical setups (hydrodynamics only / wind / cooling without
    flux-coupled gravity) the standalone density flux arrays can be skipped
    and the fused Pallas WENO+divergence path is safe."""
    return config.gravity_config.gravity and (
        config.gravity_config.self_gravity_version != SIMPLE_SOURCE
    )


def _hydro_step_rhs(
    current_q,
    dt_tilde,
    *,
    params,
    config,
    registered_variables,
    gamma,
    grid_spacing,
    helper_data,
    density_fluxes_needed: bool,
):
    """RHS for one hydro WENO time-step stage (excluding RK coefficient logic).

    ``dt_tilde`` is the stage-effective step (``k * dt``).  Returns
    ``rhs_q = -dt_tilde * div(F(current_q)) + dt_tilde * S(current_q)``.

    Shared by the SSPRK4 and LSRK4 (low-storage) integrators below; the only
    integrator-specific code is the way ``dt_tilde`` is built and how each
    stage's update accumulates ``rhs_q`` back into the running state.
    """
    dtdx = dt_tilde / grid_spacing
    dtdy = dt_tilde / grid_spacing
    dtdz = dt_tilde / grid_spacing

    # Fused WENO + axis-flux-divergence: each axis is built and consumed one
    # at a time, so the full-state-sized ``dF_x/y/z`` temporaries that used
    # to dominate the peak memory footprint never coexist.  Falls back to the
    # explicit flux + divergence path when (a) Pallas is unavailable /
    # unsupported or (b) a physics module needs the standalone density flux
    # slices.
    # The flux blending (deep-void density ramp and/or FCT positivity) post-
    # processes each assembled WENO interface flux before the divergence, so it
    # needs the standalone per-axis flux array — it is incompatible with the
    # fused WENO+divergence Pallas kernel. Fall back to the explicit
    # flux+divergence path when either blending path is enabled.
    use_fused_pallas = (
        _hydro_pallas_flux_supported(current_q, config)
        and not density_fluxes_needed
        and not config.positivity_config.deepvoid_blend
        and not config.positivity_config.preserving_flux
    )

    if use_fused_pallas:
        # Compute each axis flux with the standard (1-flux-per-cell) WENO
        # kernel, then immediately consume it via a per-axis divergence
        # kernel that accumulates into ``rhs_q`` in place (via the kernel's
        # ``input_output_aliases``).  This keeps WENO compute unchanged
        # relative to the original Pallas path while ensuring all three
        # ``dF`` temporaries never coexist and the rhs lives in a single
        # physical buffer across axes.
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
            )
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
            )
            del dF_z

        density_fluxes = None
    else:
        # Per-axis flux + divergence path.  Accumulate axis-by-axis rather
        # than holding all three flux arrays live simultaneously, so XLA
        # can reuse buffers between axes.
        blend = config.positivity_config.deepvoid_blend or config.positivity_config.preserving_flux
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        if blend:
            dF_x = _blend_interface_flux(dF_x, current_q, 0, dtdx, params, config, registered_variables)
        rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        if density_fluxes_needed:
            density_fluxes = [dF_x[registered_variables.density_index]]
        else:
            density_fluxes = None
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            if blend:
                dF_y = _blend_interface_flux(dF_y, current_q, 1, dtdy, params, config, registered_variables)
            rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            if density_fluxes_needed:
                density_fluxes.append(dF_y[registered_variables.density_index])
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            if blend:
                dF_z = _blend_interface_flux(dF_z, current_q, 2, dtdz, params, config, registered_variables)
            rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            if density_fluxes_needed:
                density_fluxes.append(dF_z[registered_variables.density_index])
            del dF_z

        if density_fluxes_needed:
            density_fluxes = tuple(density_fluxes)

    # Add physics source terms
    rhs_q += _time_integrator_sources(
        current_q,
        density_fluxes,
        rhs_q[registered_variables.density_index],  # drho
        dt_tilde,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
    )

    return rhs_q


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state"])
def _ssprk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params, # Assuming SimulationParams type
    helper_data, # Assuming HelperData type
    config, # Assuming SimulationConfig type
    registered_variables: RegisteredVariables,
):
    """
    Integrates the Euler (hydrodynamics) equations for one time step using a
    5-stage, 4th-order Strong Stability Preserving Runge-Kutta (SSPRK) method.

    Three-register Spiteri-Ruuth scheme: needs ``q0``, ``q_curr`` and
    ``q_final`` simultaneously.  For storage-constrained runs, the
    ``_lsrk4_hydro`` 2-register Carpenter-Kennedy LSRK4 is available below
    via ``time_integrator=RK4_LSRK``.
    """

    # for procceses with similar or smaller time scales as the hydrodynamics,
    # they should be included as source terms in the RK stages, otherwise
    # they could be handled outside

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    def pre_stage(q):
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
        return q

    def rhs(q, dt_stage):
        return _hydro_step_rhs(
            q,
            dt_stage,
            params=params,
            config=config,
            registered_variables=registered_variables,
            gamma=gamma,
            grid_spacing=grid_spacing,
            helper_data=helper_data,
            density_fluxes_needed=density_fluxes_needed,
        )

    def finalize(q):
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        return q

    return ssprk4(conserved_state, dt, rhs=rhs, pre_stage=pre_stage, finalize=finalize)


@partial(jax.jit, static_argnames=["registered_variables", "config"], donate_argnames=["conserved_state"])
def _lsrk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params,
    helper_data,
    config,
    registered_variables: RegisteredVariables,
):
    """Carpenter-Kennedy 2N-storage, 5-stage, 4th-order low-storage RK4.

    The integrator carries two full-state registers (``q`` and ``dq``)
    instead of the three (``q0``, ``q_curr``, ``q_final``) required by the
    SSPRK4 Spiteri-Ruuth scheme above.  That saves one full conserved-state
    buffer at peak, which on the 128^3 Sedov benchmark cuts the per-device
    temp footprint by another ~50 MB on top of the WENO/divergence Pallas
    improvements.

    The trade-off is a smaller linear-stability CFL than SSPRK4 (the user
    should expect roughly half of the 1.5 that SSPRK4 tolerates with the
    5th-order WENO scheme); LSRK4 has no SSP property either, so very strong
    shocks may need a slightly tighter limiter / floor than SSPRK4 to avoid
    sporadic non-monotone overshoots.
    """

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    dtdx = dt / grid_spacing
    dtdy = dt / grid_spacing
    dtdz = dt / grid_spacing

    def pre_stage(q):
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
        return q

    def lsrk_increment(q, dq, a_coef, dt_step):
        # Fused path: write the LSRK4 ``dq_new = A[i] * dq + dt * L(q)``
        # update directly into the ``dq`` buffer using the per-axis
        # divergence kernel's ``input_output_aliases``.  The
        # rhs/``L(q)``-sized scratch register is never materialised, which is
        # what gets us below the 3-buffer floor of the explicit
        # rhs-then-update path.
        use_fused_pallas = (
            _hydro_pallas_flux_supported(q, config)
            and not density_fluxes_needed
        )

        if use_fused_pallas:
            dF_x = _weno_flux_x(q, params, config, registered_variables)
            dq = _hydro_flux_div_axis_pallas(
                dF_x, dtdx, config, axis=0,
                rhs_accumulator=dq, scale_in=a_coef,
            )
            del dF_x

            if config.dimensionality >= 2:
                dF_y = _weno_flux_y(q, params, config, registered_variables)
                dq = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1,
                    rhs_accumulator=dq, scale_in=1.0,
                )
                del dF_y

            if config.dimensionality == 3:
                dF_z = _weno_flux_z(q, params, config, registered_variables)
                dq = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2,
                    rhs_accumulator=dq, scale_in=1.0,
                )
                del dF_z

            # Physics source terms.  Sedov-style hydro with no active modules
            # makes this a no-op (``_time_integrator_sources`` returns zeros); for
            # active modules the dt-scaled source is added on top of the
            # already-scaled ``A[i] * dq + dt * L(q)`` value in ``dq``.
            sources = _time_integrator_sources(
                q,
                None,
                dq[registered_variables.density_index],
                dt_step,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            if sources is not None:
                dq = dq + sources
            return dq

        # Fallback: explicit ``rhs = dt * L(q)`` then ``dq = A * dq + rhs``.
        rhs = _hydro_step_rhs(
            q,
            dt_step,
            params=params,
            config=config,
            registered_variables=registered_variables,
            gamma=gamma,
            grid_spacing=grid_spacing,
            helper_data=helper_data,
            density_fluxes_needed=density_fluxes_needed,
        )
        return a_coef * dq + rhs

    def finalize(q):
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        return q

    return lsrk4(
        conserved_state, dt,
        pre_stage=pre_stage, finalize=finalize, lsrk_increment=lsrk_increment,
    )


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"],
)
def _lsrk4_with_ct(
    conserved_state,
    bx_interface,
    by_interface,
    bz_interface,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """Carpenter-Kennedy 2N-storage 5-stage 4th-order LSRK4 for MHD-CT.

    Mirrors ``_lsrk4_hydro`` but carries the four MHD register pairs that
    ``_ssprk4_with_ct``'s Spiteri-Ruuth scheme used as three-register triples:

      * ``(q, dq)`` for the conserved state (8 vars),
      * ``(bx, dbx)``, ``(by, dby)``, ``(bz, dbz)`` for the three interface
        magnetic-field components.

    Compared to the SSPRK4 carry ``(q0, q_curr, q_final)`` plus the three
    ``(bx0, bx_curr, bx_final)`` triples this saves one full conserved
    register plus three interface-B registers.

    Trade-off: linear-stability CFL drops from SSPRK4's ~1.5 to roughly 1.4
    and LSRK4 has no SSP property — same caveats as ``_lsrk4_hydro``.
    Selected via ``config.time_integrator == RK4_LSRK``.
    """

    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

    dtdx = dt / grid_spacing
    dtdy = dt / grid_spacing
    dtdz = dt / grid_spacing

    def compute_lqs(current_q, bx, by, bz, dq, a_coef):
        """Compute ``dq_new = a_coef * dq + dt * L_q`` (in-place via the
        Pallas div-axis accumulator when available) and the three
        interface-B ``dt * L_b{x,y,z}`` increments.

        The fused conserved-state path matches ``_lsrk4_hydro``: each
        axis's divergence kernel folds ``a_coef * dq + (-dt/dx) * div``
        directly into the ``dq`` register, so the LSRK4 update never
        materialises a separate ``rhs_q``.  When Pallas is unavailable
        we fall back to the explicit
        ``rhs_q`` → ``dq = a_coef * dq + rhs_q`` pattern.
        """
        current_q = update_cell_center_fields(
            current_q, bx, by, bz, config, registered_variables
        )

        # Axis-incremental flow — see the matching SSPRK4-with-CT path
        # above for the rationale.  Each axis's full dF is built, the
        # two magnetic-flux slices CT needs are extracted, dF is consumed
        # for the divergence step (folding ``a_coef * dq`` in for the
        # first axis), then freed.  CT runs on the six small slices only.
        my = registered_variables.magnetic_index.y
        mz = registered_variables.magnetic_index.z
        di = registered_variables.density_index

        # x-axis: fold the LSRK4 ``a_coef * dq + ...`` step into the
        # first axis's div kernel via ``scale_in`` so ``rhs_q`` is never
        # materialised; subsequent axes accumulate (scale_in = 1.0).  The
        # native fallback path keeps the explicit ``rhs_q`` register.
        dF_x = _weno_flux_x(current_q, params, config, registered_variables)
        By_flux_x = dF_x[my]
        Bz_flux_x = dF_x[mz]
        density_flux_x = dF_x[di]
        if use_pallas_div:
            dq = _hydro_flux_div_axis_pallas(
                dF_x, dtdx, config, axis=0,
                rhs_accumulator=dq, scale_in=a_coef,
            )
            rhs_q_for_phys = None
        else:
            rhs_q_for_phys = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x

        if config.dimensionality >= 2:
            mx = registered_variables.magnetic_index.x
            dF_y = _weno_flux_y(current_q, params, config, registered_variables)
            Bx_flux_y = dF_y[mx]
            Bz_flux_y = dF_y[mz]
            density_flux_y = dF_y[di]
            if use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_y, dtdy, config, axis=1, rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0

        if config.dimensionality == 3:
            mx = registered_variables.magnetic_index.x
            dF_z = _weno_flux_z(current_q, params, config, registered_variables)
            Bx_flux_z = dF_z[mx]
            By_flux_z = dF_z[my]
            density_flux_z = dF_z[di]
            if use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_z, dtdz, config, axis=2, rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0

        rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
            current_q,
            By_flux_x, Bz_flux_x,
            Bx_flux_y, Bz_flux_y,
            Bx_flux_z, By_flux_z,
            dtdx, dtdy, dtdz,
            config, registered_variables,
        )

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)

        # Physics source terms.  On the Pallas-fused path the divergence
        # has already been folded into ``dq``; we add ``dt * S`` on top.
        # On the native fallback we still have a standalone ``rhs_q_for_phys``
        # and fold the full LSRK4 update at the end.
        if use_pallas_div:
            sources = _time_integrator_sources(
                current_q,
                density_fluxes,
                dq[registered_variables.density_index],
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            if sources is not None:
                dq = dq + sources
        else:
            rhs_q_for_phys += _time_integrator_sources(
                current_q,
                density_fluxes,
                rhs_q_for_phys[registered_variables.density_index],
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
            )
            dq = a_coef * dq + rhs_q_for_phys

        return dq, rhs_bx, rhs_by, rhs_bz

    def pre_stage(u):
        q, bx, by, bz = u
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE,
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config, registered_variables, params, MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]
        return (q, bx, by, bz)

    def lsrk_increment(u, du, a_coef, _dt_step):
        # ``compute_lqs`` returns the new ``dq`` already in LSRK4 form
        # (``a_coef * dq_old + dt * L_q``), folding the accumulate into the
        # divergence kernel when Pallas is available.  The interface-B deltas
        # use the explicit ``a_coef * db + dt * L_b`` low-storage update.
        q, bx, by, bz = u
        dq, dbx, dby, dbz = du
        dq, rhs_bx, rhs_by, rhs_bz = compute_lqs(q, bx, by, bz, dq, a_coef)
        dbx = a_coef * dbx + rhs_bx
        dby = a_coef * dby + rhs_by
        dbz = a_coef * dbz + rhs_bz
        return (dq, dbx, dby, dbz)

    def finalize(u):
        q, bx, by, bz = u
        q = update_cell_center_fields(
            q, bx, by, bz, config, registered_variables,
        )
        q = _apply_stage_positivity(
            q, config.positivity_config.per_stage_mode, config, gamma,
            params.minimum_density, params.minimum_pressure,
            params.positivity_max_velocity, registered_variables,
        )
        return (q, bx, by, bz)

    return lsrk4(
        (conserved_state, bx_interface, by_interface, bz_interface),
        dt, pre_stage=pre_stage, finalize=finalize, lsrk_increment=lsrk_increment,
    )
