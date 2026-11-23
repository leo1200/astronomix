"""
Runge-Kutta drivers of the finite-difference solver.

One time step of the hydrodynamics and of the MHD equations with Constrained
Transport (CT), either with the 5-stage, 4th-order Strong Stability Preserving
Runge-Kutta scheme (SSPRK4, Spiteri & Ruuth 2002) or with the 2N-storage
low-storage RK4 (LSRK4, Carpenter & Kennedy 1994). The drivers build the
stage right-hand sides (WENO interface fluxes, flux divergence, physics
sources) and hand them to the generic schemes in
``_integrators/_explicit_rk.py``; ``_stage_remat`` wraps the stages for
reverse-mode rematerialisation (``SimulationConfig.rematerialize_stages``).

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
    IDEAL_GAS,
    MAGNETIC_FIELD_ONLY,
    RIEMANN_FLUX,
    SIMPLE_SOURCE,
    SSPRK33,
)

# astronomix containers
from astronomix.data_classes.simulation_helper_data import HelperData
from astronomix.option_classes.simulation_config import SimulationConfig
from astronomix.option_classes.simulation_params import SimulationParams
from astronomix.variable_registry.registered_variables import RegisteredVariables

# astronomix functions
from astronomix._finite_difference._interface_fluxes._riemann_flux import (
    interface_flux as riemann_interface_flux,
    maximum_signal_speed,
    transverse_compression,
)
from astronomix._finite_difference._interface_fluxes._riemann_flux_pallas import _riemann_flux_pallas_supported
from astronomix._finite_difference._interface_fluxes._weno import (
    _weno_flux,
)
from astronomix._finite_difference._interface_fluxes._weno_pallas import (
    _hydro_pallas_flux_supported,
    _mhd_pallas_flux_supported,
    _update_cell_center_and_weno_flux_mhd_pallas_keep_halo_x_with_ct_mod,
    _weno_flux_mhd_pallas_keep_halo_x,
)
from astronomix._finite_difference._interface_fluxes._weno_positivity_pallas import (
    mhd_inflow_reference_dispatch,
)
from astronomix._finite_difference._time_integrators._padded_stage import (
    padded_stage_supported,
    riemann_flux_stage_on_padded_shards,
)
from astronomix._finite_difference._time_integrators._ssprk_pallas import (
    _div_axis_pallas_shape_ok,
    _hydro_flux_div_axis_native,
    _hydro_flux_div_axis_native_from_kept_halo_sharded,
    _hydro_flux_div_axis_pallas,
)
from astronomix._finite_difference._magnetic_update._constrained_transport import (
    _constrained_transport_rhs_from_slices,
    ct_upwind_dissipation_rhs,
    update_cell_center_fields,
)
from astronomix._finite_difference._magnetic_update._constrained_transport_pallas import (
    _ct_rhs_pallas_supported,
    _ct_rhs_pallas_x_precomputed,
)
from astronomix._finite_difference._magnetic_update._energy_flux import magnetic_energy_corrected_flux
from astronomix._finite_difference._magnetic_update._flux_form_ct import (
    flux_form_ct_rhs,
    flux_form_ct_rhs_with_energy,
)
from astronomix._finite_difference._ratio_rules import _floored_ratio
from astronomix._geometry.boundaries import _boundary_handler
from astronomix._integrators._explicit_rk import lsrk4, ssprk33, ssprk4
from astronomix._modules._explicit_sources import explicit_sources
from astronomix._modules._resistivity._resistivity import resistive_face_increment
from astronomix._pallas_helpers import (
    _backend_is_pallas,
    _pallas_mesh_splits_axis,
    diffable_pallas_call_n,
    pl,
)
from astronomix._stencil_operations._stencil_operations import _shift


def _stage_remat(fn, config):
    """
    Wrap a Runge-Kutta stage function in ``jax.checkpoint`` when reverse-mode
    rematerialisation is requested (``config.rematerialize_stages``).

    Applied to the right-hand side and to the per-stage boundary and
    cell-centred-field hooks. The backward pass then stores only their inputs
    and recomputes the internals (WENO reconstruction, divergence, sources)
    when it needs them: about one extra forward evaluation per stage in
    exchange for not holding every stage's residuals across the whole step.
    Under a plain forward evaluation, or forward-mode AD, ``jax.checkpoint``
    is inlined and changes nothing.

    Args:
        fn: The stage function.
        config: The simulation configuration.

    Returns:
        ``fn`` itself, or its checkpointed version.
    """
    if not config.rematerialize_stages:
        return fn
    return jax.checkpoint(fn)


def _riemann_flux_increment(
    conserved_state,
    face_fields,
    dt_over_dx,
    params: SimulationParams,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
):
    """
    The fluid increment of a stage under ``scheme = RIEMANN_FLUX``:
    -dt/h sum_d delta_d(F_d) with the interface flux of ``_riemann_flux.py``.
    For ideal-gas MHD the energy row carries the fluid half of the
    Poynting-consistent energy flux (``_energy_flux.py``); the CT half is added
    with the constrained transport.

    Args:
        conserved_state: The conserved state (MHD: with the cell-centred field rebuilt).
        face_fields: The interface fields (bx, by, bz); None in the hydrodynamic form.
        dt_over_dx: The (stage) time step over the cell size.
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The increment of the conserved state and the density fluxes of the axes.
    """
    energy_flux_channels = None
    if config.mhd and config.equation_of_state == IDEAL_GAS:
        magnetic_index = registered_variables.magnetic_index
        energy_flux_channels = (
            registered_variables.energy_index,
            (magnetic_index.x, magnetic_index.y, magnetic_index.z),
        )
    use_kernels = _riemann_flux_pallas_supported(conserved_state, config)
    compression = transverse_compression(conserved_state, params, config, registered_variables)
    increment, density_fluxes = None, []
    for axis in range(config.dimensionality):
        # The reference step of the guard: one cell over the largest signal speed of the axis.
        reference_step = _floored_ratio(
            1.0, maximum_signal_speed(conserved_state, params, config, registered_variables, axis)
        )
        flux = riemann_interface_flux(
            conserved_state, None if face_fields is None else face_fields[axis], reference_step,
            compression[axis], params, config, registered_variables, axis,
        )
        density_fluxes.append(flux[registered_variables.density_index])
        if use_kernels:
            # the same difference (and energy row) as below, in one kernel
            increment = _hydro_flux_div_axis_pallas(
                flux, dt_over_dx, config, axis=axis, rhs_accumulator=increment,
                magnetic_state=conserved_state if energy_flux_channels is not None else None,
                energy_flux_channels=energy_flux_channels,
            )
            continue
        if energy_flux_channels is not None:
            flux = magnetic_energy_corrected_flux(flux, conserved_state, energy_flux_channels, axis)
        divergence = -dt_over_dx * (flux - _shift(flux, 1, axis=axis + 1))
        increment = divergence if increment is None else increment + divergence
    return increment, tuple(density_fluxes)


def _riemann_flux_rhs_on_padded_shards(
    u,
    dt_tilde,
    gamma,
    grid_spacing,
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    sink_potential=None,
):
    """
    The right-hand side of a stage under ``scheme = RIEMANN_FLUX`` on a device
    mesh, with one halo exchange per split axis (``_padded_stage.py``): the
    increments of the stage functions of ``_ssprk_with_ct`` (``u`` the tuple
    of the conserved state and the interface fields) and of ``_hydro_step_rhs``
    (``u`` the conserved state). A derivative takes the array form, the steps
    of those stage functions.
    """
    face_fields = tuple(u[1:]) if config.mhd else None
    conserved_state = u[0] if config.mhd else u

    def array_form(state, fields, time_step, spacing, parameters):
        if fields is not None:
            state = update_cell_center_fields(state, *fields, config, registered_variables, parameters)
        increment, density_fluxes = _riemann_flux_increment(
            state, fields, time_step / spacing, parameters, config, registered_variables
        )
        if fields is None:
            return state, increment, None, density_fluxes
        if config.equation_of_state == IDEAL_GAS:
            *field_increments, energy = flux_form_ct_rhs_with_energy(
                state, *fields, time_step, spacing, parameters, config, registered_variables
            )
            increment = increment.at[registered_variables.energy_index].add(energy)
        else:
            field_increments = flux_form_ct_rhs(
                state, *fields, time_step, spacing, parameters, config, registered_variables
            )
        return state, increment, tuple(field_increments), density_fluxes

    def on_padded_shards(state, fields, time_step, spacing, parameters):
        return riemann_flux_stage_on_padded_shards(
            state, fields, time_step, spacing, parameters, config, registered_variables
        )

    state, increment, field_increments, density_fluxes = diffable_pallas_call_n(
        (conserved_state, face_fields, dt_tilde, grid_spacing, params),
        pallas_branch=on_padded_shards,
        native_branch=array_form,
    )
    increment += explicit_sources(
        state,
        dt_tilde,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
        density_fluxes=density_fluxes,
        density_increment=increment[registered_variables.density_index],
        sink_potential=sink_potential,
    )
    if face_fields is None:
        return increment
    if config.diffusion_config.resistivity:
        resistive_increments = resistive_face_increment(
            *face_fields, params.resistivity, dt_tilde, grid_spacing, config.dimensionality
        )
        field_increments = tuple(
            field_increment + resistive_increment
            for field_increment, resistive_increment in zip(field_increments, resistive_increments)
        )
    return (increment, *field_increments)


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state", "bx_interface", "by_interface", "bz_interface"],
)
def _ssprk_with_ct(
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
    internal_energy_density=None,
    sink_potential=None,
):
    """
    Integrates the MHD equations for one time step using a 5-stage, 4th-order
    Strong Stability Preserving Runge-Kutta (SSPRK) method, or SSPRK(3,3) with
    ``time_integrator=SSPRK33`` (``scheme = RIEMANN_FLUX``), with Constrained
    Transport (CT).

    Args:
        conserved_state: The conserved state (cell-centred fields included).
        bx_interface: The interface magnetic field B_x.
        by_interface: The interface magnetic field B_y.
        bz_interface: The interface magnetic field B_z.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.
        sink_potential: The gravitational potential of the sink particles,
            which acts on the gas together with its own gravity, or ``None``.

    Returns:
        The updated ``(conserved_state, bx_interface, by_interface,
        bz_interface)``.
    """

    # Processes with time scales similar to or shorter than the hydrodynamics
    # should be included as source terms in the RK stages; slower ones could be
    # handled outside.

    # The per-axis divergence kernel (with its ``input_output_aliases`` memory
    # saving) only operates on whatever flux tensor it is handed, so it serves
    # the MHD equations as well whenever the Pallas backend is selected.
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
            current_q, bx, by, bz, config, registered_variables, params
        )

        # For ideal MHD with the positivity-preserving WENO, compute each
        # cell's axis-summed first-order inflow, so that the inflow faces of a
        # cell can be limited jointly (see _weno_positivity.py).
        inflow_reference = None
        if (
            config.solver_config.weno_positivity_preserving
            and config.equation_of_state == IDEAL_GAS
            and internal_energy_density is None
        ):
            inflow_reference = mhd_inflow_reference_dispatch(
                current_q,
                params,
                config,
                registered_variables,
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
        magnetic_index = registered_variables.magnetic_index
        density_index = registered_variables.density_index

        if config.solver_config.scheme == RIEMANN_FLUX:
            rhs_q, (density_flux_x, density_flux_y, density_flux_z) = _riemann_flux_increment(
                current_q, (bx, by, bz), dtdx, params, config, registered_variables
            )
        else:
            # --------------- ↓ x-axis ↓ ----------------
            dF_x = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=0,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            By_flux_x = dF_x[magnetic_index.y]
            Bz_flux_x = dF_x[magnetic_index.z]
            density_flux_x = dF_x[density_index]
            if use_pallas_div:
                rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
            else:
                rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
            del dF_x
            # --------------- ↑ x-axis ↑ ----------------

            # --------------- ↓ y-axis ↓ ----------------
            if config.dimensionality >= 2:
                dF_y = _weno_flux(
                    current_q,
                    params,
                    config,
                    registered_variables,
                    axis=1,
                    internal_energy_density=internal_energy_density,
                    inflow_reference=inflow_reference,
                )
                Bx_flux_y = dF_y[magnetic_index.x]
                Bz_flux_y = dF_y[magnetic_index.z]
                density_flux_y = dF_y[density_index]
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
            # --------------- ↑ y-axis ↑ ----------------

            # --------------- ↓ z-axis ↓ ----------------
            if config.dimensionality == 3:
                dF_z = _weno_flux(
                    current_q,
                    params,
                    config,
                    registered_variables,
                    axis=2,
                    internal_energy_density=internal_energy_density,
                    inflow_reference=inflow_reference,
                )
                Bx_flux_z = dF_z[magnetic_index.x]
                By_flux_z = dF_z[magnetic_index.y]
                density_flux_z = dF_z[density_index]
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
            # --------------- ↑ z-axis ↑ ----------------

        # CT now runs on the six single-channel B-flux slices only — the
        # three 8-var dF arrays have all been freed by this point.
        if config.solver_config.scheme == RIEMANN_FLUX:
            if config.equation_of_state == IDEAL_GAS:
                rhs_bx, rhs_by, rhs_bz, ct_energy = flux_form_ct_rhs_with_energy(
                    current_q, bx, by, bz, dt_tilde, grid_spacing, params, config, registered_variables
                )
                rhs_q = rhs_q.at[registered_variables.energy_index].add(ct_energy)
            else:
                rhs_bx, rhs_by, rhs_bz = flux_form_ct_rhs(
                    current_q, bx, by, bz, dt_tilde, grid_spacing, params, config, registered_variables
                )
        else:
            rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
                current_q,
                By_flux_x,
                Bz_flux_x,
                Bx_flux_y,
                Bz_flux_y,
                Bx_flux_z,
                By_flux_z,
                dtdx,
                dtdy,
                dtdz,
                config,
                registered_variables,
            )
        # Upwind dissipation of the edge EMFs on the interface fields, which
        # the central CT interpolations leave undamped near the grid scale
        # (see ct_upwind_dissipation_rhs); also a curl, so div(B) = 0 is kept.
        if config.solver_config.ct_upwind_dissipation:
            upwind_rhs_bx, upwind_rhs_by, upwind_rhs_bz = ct_upwind_dissipation_rhs(
                current_q,
                bx,
                by,
                bz,
                dt_tilde,
                grid_spacing,
                gamma,
                params,
                config,
                registered_variables,
            )
            rhs_bx = rhs_bx + upwind_rhs_bx
            rhs_by = rhs_by + upwind_rhs_by
            rhs_bz = rhs_bz + upwind_rhs_bz
        # Explicit ohmic resistivity: a further curl of an edge EMF on the
        # interface fields, so div(B) = 0 is kept exactly (see _resistivity).
        if config.diffusion_config.resistivity:
            resistive_rhs_bx, resistive_rhs_by, resistive_rhs_bz = resistive_face_increment(
                bx,
                by,
                bz,
                params.resistivity,
                dt_tilde,
                grid_spacing,
                config.dimensionality,
            )
            rhs_bx = rhs_bx + resistive_rhs_bx
            rhs_by = rhs_by + resistive_rhs_by
            rhs_bz = rhs_bz + resistive_rhs_bz

        if config.dimensionality == 1:
            density_fluxes = (density_flux_x,)
        elif config.dimensionality == 2:
            density_fluxes = (density_flux_x, density_flux_y)
        else:
            density_fluxes = (density_flux_x, density_flux_y, density_flux_z)

        # Add the physics source terms; the density increment is handed in
        # separately for the modules that need it.
        density_increment = rhs_q[registered_variables.density_index]
        rhs_q += explicit_sources(
            current_q,
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
            density_fluxes=density_fluxes,
            density_increment=density_increment,
            sink_potential=sink_potential,
        )

        return rhs_q, rhs_bx, rhs_by, rhs_bz

    # On a device mesh: the stage on padded shards, one halo exchange per split axis.
    if padded_stage_supported(conserved_state, config):
        rhs = partial(
            _riemann_flux_rhs_on_padded_shards, gamma=gamma, grid_spacing=grid_spacing, params=params,
            helper_data=helper_data, config=config, registered_variables=registered_variables,
            sink_potential=sink_potential,
        )
    rhs = _stage_remat(rhs, config)

    def pre_stage(u):
        q, bx, by, bz = u
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config,
                registered_variables,
                params,
                MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]
        return (q, bx, by, bz)

    def finalize(u):
        q, bx, by, bz = u
        # Update the cell-centered magnetic fields in the conserved state array
        # from the final interface magnetic fields.
        q = update_cell_center_fields(
            q, bx, by, bz, config, registered_variables, params
        )
        return (q, bx, by, bz)

    def post_stage(u):
        # The increment of a stage is evaluated at the state with its
        # cell-centred B rebuilt from the faces (start of ``rhs``) but is
        # added to the stored state, whose B and E are the previous stage's
        # WENO update. The two have the same pressure, yet the sum moves it by
        # (gamma - 1) lambda dF_B . (B_stored - B_faces), of no definite sign
        # and large at low beta. Rebuilding the stored stage state (pressure
        # held) makes every increment start from the state it was evaluated
        # at, which the positivity-preserving WENO proof needs.
        q, bx, by, bz = u
        q = update_cell_center_fields(q, bx, by, bz, config, registered_variables, params)
        return (q, bx, by, bz)

    # Only the positivity-preserving ideal-MHD scheme and the ideal-gas scheme
    # with a Riemann flux (whose rebuild holds the energy) need the stage states
    # resynchronised; everywhere else the Runge-Kutta loop keeps its identity hook.
    stage_hooks = {"pre_stage": _stage_remat(pre_stage, config)}
    rebuilds_stage_states = (
        config.solver_config.weno_positivity_preserving or config.solver_config.scheme == RIEMANN_FLUX
    )
    if rebuilds_stage_states and config.equation_of_state == IDEAL_GAS:
        stage_hooks["post_stage"] = _stage_remat(post_stage, config)
    runge_kutta = ssprk33 if config.solver_config.time_integrator == SSPRK33 else ssprk4
    return runge_kutta(
        (conserved_state, bx_interface, by_interface, bz_interface),
        dt,
        rhs=rhs,
        **stage_hooks,
        finalize=_stage_remat(finalize, config),
    )


def _hydro_density_fluxes_needed(config) -> bool:
    """
    Whether any FD physics module actually consumes the per-axis density
    flux slices.  Only self-gravity variants other than SIMPLE_SOURCE do,
    so for typical setups (hydrodynamics only / wind / cooling without
    flux-coupled gravity) the standalone density flux arrays can be skipped
    and the fused Pallas WENO+divergence path is safe.

    Args:
        config: The simulation configuration.

    Returns:
        True if the density fluxes must be kept.
    """
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
    internal_energy_density=None,
    sink_potential=None,
):
    """
    RHS for one hydro time-step stage (excluding RK coefficient logic),
    ``rhs_q = -dt_tilde * div(F(current_q)) + dt_tilde * S(current_q)``,
    with the WENO flux splitting or (``scheme = RIEMANN_FLUX``) the flux of
    ``_riemann_flux.py``.

    Shared by the SSPRK4 and LSRK4 (low-storage) integrators below; the only
    integrator-specific code is the way ``dt_tilde`` is built and how each
    stage's update accumulates ``rhs_q`` back into the running state.

    Args:
        current_q: The conserved state of the stage.
        dt_tilde: The stage-effective step (``k * dt``).
        params: The simulation parameters.
        config: The simulation configuration.
        registered_variables: The registered variables.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        helper_data: The helper data.
        density_fluxes_needed: Whether a physics module consumes the per-axis
            density fluxes (see ``_hydro_density_fluxes_needed``).
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.
        sink_potential: The gravitational potential of the sink particles,
            which acts on the gas together with its own gravity, or ``None``.

    Returns:
        The stage increment ``rhs_q``.
    """
    dtdx = dt_tilde / grid_spacing
    dtdy = dt_tilde / grid_spacing
    dtdz = dt_tilde / grid_spacing

    if config.solver_config.scheme == RIEMANN_FLUX:
        rhs_q, density_fluxes = _riemann_flux_increment(
            current_q, None, dtdx, params, config, registered_variables
        )
        return rhs_q + explicit_sources(
            current_q,
            dt_tilde,
            gamma,
            config,
            params,
            helper_data,
            registered_variables,
            density_fluxes=density_fluxes,
            density_increment=rhs_q[registered_variables.density_index],
            sink_potential=sink_potential,
        )

    # Fused WENO + axis-flux-divergence: each axis is built and consumed one
    # at a time, so the full-state-sized ``dF_x/y/z`` temporaries never
    # coexist.  Falls back to the explicit flux + divergence path when (a)
    # Pallas is unavailable / unsupported or (b) a physics module needs the
    # standalone density flux slices.
    use_fused_pallas = (
        _hydro_pallas_flux_supported(current_q, config)
        and not density_fluxes_needed
    )

    if use_fused_pallas:
        # Compute each axis flux with the standard (1-flux-per-cell) WENO
        # kernel, then immediately consume it via a per-axis divergence
        # kernel that accumulates into ``rhs_q`` in place (via the kernel's
        # ``input_output_aliases``).  This keeps WENO compute unchanged
        # relative to the original Pallas path while ensuring all three
        # ``dF`` temporaries never coexist and the rhs lives in a single
        # physical buffer across axes.
        dF_x = _weno_flux(
            current_q,
            params,
            config,
            registered_variables,
            axis=0,
            internal_energy_density=internal_energy_density,
        )
        rhs_q = _hydro_flux_div_axis_pallas(dF_x, dtdx, config, axis=0)
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=1,
                internal_energy_density=internal_energy_density,
            )
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_y, dtdy, config, axis=1, rhs_accumulator=rhs_q
            )
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=2,
                internal_energy_density=internal_energy_density,
            )
            rhs_q = _hydro_flux_div_axis_pallas(
                dF_z, dtdz, config, axis=2, rhs_accumulator=rhs_q
            )
            del dF_z

        density_fluxes = None
    else:
        # Per-axis flux + divergence path.  Accumulate axis-by-axis rather
        # than holding all three flux arrays live simultaneously, so XLA
        # can reuse buffers between axes.
        dF_x = _weno_flux(
            current_q,
            params,
            config,
            registered_variables,
            axis=0,
            internal_energy_density=internal_energy_density,
        )
        rhs_q = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        if density_fluxes_needed:
            density_fluxes = [dF_x[registered_variables.density_index]]
        else:
            density_fluxes = None
        del dF_x

        if config.dimensionality >= 2:
            dF_y = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=1,
                internal_energy_density=internal_energy_density,
            )
            rhs_q = rhs_q - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            if density_fluxes_needed:
                density_fluxes.append(dF_y[registered_variables.density_index])
            del dF_y

        if config.dimensionality == 3:
            dF_z = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=2,
                internal_energy_density=internal_energy_density,
            )
            rhs_q = rhs_q - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            if density_fluxes_needed:
                density_fluxes.append(dF_z[registered_variables.density_index])
            del dF_z

        if density_fluxes_needed:
            density_fluxes = tuple(density_fluxes)

    # Add the physics source terms; the density increment is handed in
    # separately for the modules that need it.
    density_increment = rhs_q[registered_variables.density_index]
    rhs_q += explicit_sources(
        current_q,
        dt_tilde,
        gamma,
        config,
        params,
        helper_data,
        registered_variables,
        density_fluxes=density_fluxes,
        density_increment=density_increment,
        sink_potential=sink_potential,
    )

    return rhs_q


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state"],
)
def _ssprk_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    sink_potential=None,
):
    """
    Integrates the Euler (hydrodynamics) equations for one time step using a
    5-stage, 4th-order Strong Stability Preserving Runge-Kutta (SSPRK) method,
    or SSPRK(3,3) with ``time_integrator=SSPRK33`` (``scheme = RIEMANN_FLUX``).

    Three-register Spiteri-Ruuth scheme: needs ``q0``, ``q_curr`` and
    ``q_final`` simultaneously.  For storage-constrained runs, the
    ``_lsrk4_hydro`` 2-register Carpenter-Kennedy LSRK4 is available below
    via ``time_integrator=LSRK4``.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.
        sink_potential: The gravitational potential of the sink particles,
            which acts on the gas together with its own gravity, or ``None``.

    Returns:
        The updated conserved state.
    """

    # Processes with time scales similar to or shorter than the hydrodynamics
    # should be included as source terms in the RK stages; slower ones could be
    # handled outside.

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    def pre_stage(q):
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
            internal_energy_density=internal_energy_density,
            sink_potential=sink_potential,
        )

    # On a device mesh: the stage on padded shards, one halo exchange per split axis.
    if padded_stage_supported(conserved_state, config):
        rhs = partial(
            _riemann_flux_rhs_on_padded_shards, gamma=gamma, grid_spacing=grid_spacing, params=params,
            helper_data=helper_data, config=config, registered_variables=registered_variables,
            sink_potential=sink_potential,
        )
    rhs = _stage_remat(rhs, config)

    runge_kutta = ssprk33 if config.solver_config.time_integrator == SSPRK33 else ssprk4
    return runge_kutta(
        conserved_state,
        dt,
        rhs=rhs,
        pre_stage=_stage_remat(pre_stage, config),
    )


@partial(
    jax.jit,
    static_argnames=["registered_variables", "config"],
    donate_argnames=["conserved_state"],
)
def _lsrk4_hydro(
    conserved_state,
    gamma: Union[float, jnp.ndarray],
    grid_spacing: Union[float, jnp.ndarray],
    dt: Union[float, jnp.ndarray],
    params: SimulationParams,
    helper_data: HelperData,
    config: SimulationConfig,
    registered_variables: RegisteredVariables,
    internal_energy_density=None,
    sink_potential=None,
):
    """
    Carpenter-Kennedy 2N-storage, 5-stage, 4th-order low-storage RK4.

    The integrator carries two full-state registers (``q`` and ``dq``)
    instead of the three (``q0``, ``q_curr``, ``q_final``) required by the
    SSPRK4 Spiteri-Ruuth scheme above, which saves one full conserved-state
    buffer at peak on top of the memory savings of the fused WENO /
    divergence Pallas kernels.

    The trade-off is a smaller linear-stability CFL than SSPRK4 (the user
    should expect roughly half of the 1.5 that SSPRK4 tolerates with the
    5th-order WENO scheme); LSRK4 has no SSP property either, so very strong
    shocks may need a slightly tighter limiter / floor than SSPRK4 to avoid
    sporadic non-monotone overshoots.

    Args:
        conserved_state: The conserved state.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.
        sink_potential: The gravitational potential of the sink particles,
            which acts on the gas together with its own gravity, or ``None``.

    Returns:
        The updated conserved state.
    """

    density_fluxes_needed = _hydro_density_fluxes_needed(config)

    dtdx = dt / grid_spacing
    dtdy = dt / grid_spacing
    dtdz = dt / grid_spacing

    def pre_stage(q):
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
            dF_x = _weno_flux(
                q,
                params,
                config,
                registered_variables,
                axis=0,
                internal_energy_density=internal_energy_density,
            )
            dq = _hydro_flux_div_axis_pallas(
                dF_x,
                dtdx,
                config,
                axis=0,
                rhs_accumulator=dq,
                scale_in=a_coef,
            )
            del dF_x

            if config.dimensionality >= 2:
                dF_y = _weno_flux(
                    q,
                    params,
                    config,
                    registered_variables,
                    axis=1,
                    internal_energy_density=internal_energy_density,
                )
                dq = _hydro_flux_div_axis_pallas(
                    dF_y,
                    dtdy,
                    config,
                    axis=1,
                    rhs_accumulator=dq,
                    scale_in=1.0,
                )
                del dF_y

            if config.dimensionality == 3:
                dF_z = _weno_flux(
                    q,
                    params,
                    config,
                    registered_variables,
                    axis=2,
                    internal_energy_density=internal_energy_density,
                )
                dq = _hydro_flux_div_axis_pallas(
                    dF_z,
                    dtdz,
                    config,
                    axis=2,
                    rhs_accumulator=dq,
                    scale_in=1.0,
                )
                del dF_z

            # Physics source terms.  Sedov-style hydro with no active modules
            # makes this a no-op (``explicit_sources`` returns zeros); for
            # active modules the dt-scaled source is added on top of the
            # already-scaled ``A[i] * dq + dt * L(q)`` value in ``dq``.
            sources = explicit_sources(
                q,
                dt_step,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
                density_increment=dq[registered_variables.density_index],
                sink_potential=sink_potential,
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
            internal_energy_density=internal_energy_density,
            sink_potential=sink_potential,
        )
        return a_coef * dq + rhs

    lsrk_increment = _stage_remat(lsrk_increment, config)

    return lsrk4(
        conserved_state,
        dt,
        pre_stage=_stage_remat(pre_stage, config),
        lsrk_increment=lsrk_increment,
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
    internal_energy_density=None,
    sink_potential=None,
):
    """
    Carpenter-Kennedy 2N-storage 5-stage 4th-order LSRK4 for MHD-CT.

    Mirrors ``_lsrk4_hydro`` but carries the four MHD register pairs that
    ``_ssprk_with_ct``'s Spiteri-Ruuth scheme used as three-register triples:

      * ``(q, dq)`` for the conserved state (8 vars),
      * ``(bx, dbx)``, ``(by, dby)``, ``(bz, dbz)`` for the three interface
        magnetic-field components.

    Compared to the SSPRK4 carry ``(q0, q_curr, q_final)`` plus the three
    ``(bx0, bx_curr, bx_final)`` triples this saves one full conserved
    register plus three interface-B registers.

    Trade-off: linear-stability CFL drops from SSPRK4's ~1.5 to roughly 1.4
    and LSRK4 has no SSP property — same caveats as ``_lsrk4_hydro``.
    Selected via ``config.solver_config.time_integrator == LSRK4``.

    Args:
        conserved_state: The conserved state (cell-centred fields included).
        bx_interface: The interface magnetic field B_x.
        by_interface: The interface magnetic field B_y.
        bz_interface: The interface magnetic field B_z.
        gamma: The adiabatic index.
        grid_spacing: The cell size.
        dt: The time step.
        params: The simulation parameters.
        helper_data: The helper data.
        config: The simulation configuration.
        registered_variables: The registered variables.
        internal_energy_density: The dual-energy ``g`` with shape
            ``(x, y, z)`` (used in the pressure recovery of the fluxes), or
            None.
        sink_potential: The gravitational potential of the sink particles,
            which acts on the gas together with its own gravity, or ``None``.

    Returns:
        The updated ``(conserved_state, bx_interface, by_interface,
        bz_interface)``.
    """

    use_pallas_div = (
        _backend_is_pallas(config) and pl is not None
        and _div_axis_pallas_shape_ok(conserved_state, config)
    )

    # Multi-GPU fast path for x split over devices. The x-WENO kernel keeps one
    # x-halo cell of its flux, so the x divergence needs no second halo
    # exchange; the x parts of the constrained-transport modified fluxes are
    # computed in the same shard_map, and with the Pallas CT kernels the
    # cell-centred field update is fused into it as well. The fused kernels
    # implement the plain WENO flux only, so positivity-preserving WENO and
    # dual energy take the standard path.
    use_kept_x_flux_halo = (
        use_pallas_div
        and int(config.dimensionality) == 3
        and _pallas_mesh_splits_axis(conserved_state, 0)
        and _mhd_pallas_flux_supported(conserved_state, config)
        and not config.solver_config.weno_positivity_preserving
        and internal_energy_density is None
    )
    fuse_cell_center_update = (
        use_kept_x_flux_halo and _ct_rhs_pallas_supported(conserved_state, config)
    )

    # With y and z unsplit the y and z divergences are purely shard-local and
    # cheaper in plain JAX than as separate Pallas calls.
    yz_unsplit = not (
        _pallas_mesh_splits_axis(conserved_state, 1)
        or _pallas_mesh_splits_axis(conserved_state, 2)
    )

    def stage_rhs(
        kept_halo_path,
        current_q,
        bx,
        by,
        bz,
        dq,
        a_coef,
        dt,
        gamma,
        grid_spacing,
        params,
        helper_data,
        sink_potential,
    ):
        """Compute ``dq_new = a_coef * dq + dt * L_q`` (in-place via the
        Pallas div-axis accumulator when available) and the three
        interface-B ``dt * L_b{x,y,z}`` increments.

        The fused conserved-state path matches ``_lsrk4_hydro``: each
        axis's divergence kernel folds ``a_coef * dq + (-dt/dx) * div``
        directly into the ``dq`` register, so the LSRK4 update never
        materialises a separate ``rhs_q``.  When Pallas is unavailable
        we fall back to the explicit
        ``rhs_q`` → ``dq = a_coef * dq + rhs_q`` pattern.

        ``kept_halo_path`` (static) selects the multi-GPU fast path described
        above; both paths compute the same stage. The fast path runs inside
        the ``jax.custom_jvp`` of ``diffable_pallas_call_n``, which cannot
        differentiate through values read from the enclosing scope, so every
        traced input, ``sink_potential`` included, is an argument.
        """
        dtdx = dt / grid_spacing
        dtdy = dt / grid_spacing
        dtdz = dt / grid_spacing
        fuse_update = kept_halo_path and fuse_cell_center_update

        if not fuse_update:
            current_q = update_cell_center_fields(
                current_q, bx, by, bz, config, registered_variables
            )

        # For ideal MHD with the positivity-preserving WENO, compute each
        # cell's axis-summed first-order inflow, so that the inflow faces of a
        # cell can be limited jointly (see _weno_positivity.py).
        inflow_reference = None
        if (
            config.solver_config.weno_positivity_preserving
            and config.equation_of_state == IDEAL_GAS
            and internal_energy_density is None
        ):
            inflow_reference = mhd_inflow_reference_dispatch(
                current_q,
                params,
                config,
                registered_variables,
            )

        # Axis-incremental flow — see the matching SSPRK4-with-CT path
        # above for the rationale.  Each axis's full dF is built, the
        # two magnetic-flux slices CT needs are extracted, dF is consumed
        # for the divergence step (folding ``a_coef * dq`` in for the
        # first axis), then freed.  CT runs on the six small slices only.
        magnetic_index = registered_variables.magnetic_index
        density_index = registered_variables.density_index

        # -------------------------------------------------------------
        # ======================= ↓ x-axis ↓ ==========================
        # -------------------------------------------------------------

        # Fold the LSRK4 ``a_coef * dq + ...`` step into the first axis's
        # divergence kernel via ``scale_in`` so ``rhs_q`` is never
        # materialised; subsequent axes accumulate (scale_in = 1.0).  The
        # native fallback path keeps the explicit ``rhs_q`` register.
        flux_x_modified = None
        flux_x_with_halo = None
        if fuse_update:
            current_q, flux_x_with_halo, dF_x, flux_x_modified = (
                _update_cell_center_and_weno_flux_mhd_pallas_keep_halo_x_with_ct_mod(
                    current_q,
                    bx,
                    by,
                    bz,
                    params,
                    config,
                    registered_variables,
                )
            )
        elif kept_halo_path:
            flux_x_with_halo, dF_x = _weno_flux_mhd_pallas_keep_halo_x(
                current_q,
                params,
                config,
                registered_variables,
            )
        else:
            dF_x = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=0,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
        By_flux_x = dF_x[magnetic_index.y]
        Bz_flux_x = dF_x[magnetic_index.z]
        density_flux_x = dF_x[density_index]
        if kept_halo_path:
            dq = _hydro_flux_div_axis_native_from_kept_halo_sharded(
                flux_x_with_halo,
                dtdx,
                config,
                axis=0,
                rhs_accumulator=dq,
                scale_in=a_coef,
                kept_halo=1,
            )
            rhs_q_for_phys = None
        elif use_pallas_div:
            dq = _hydro_flux_div_axis_pallas(
                dF_x,
                dtdx,
                config,
                axis=0,
                rhs_accumulator=dq,
                scale_in=a_coef,
            )
            rhs_q_for_phys = None
        else:
            rhs_q_for_phys = -dtdx * (dF_x - _shift(dF_x, 1, axis=1))
        del dF_x, flux_x_with_halo

        # Serialize the per-axis flux passes.  ``dF_{x,y,z}`` each depend only
        # on ``current_q`` (not on ``dq`` or each other), so without an explicit
        # dependency XLA is free to schedule all three axis reconstructions —
        # and their halo-padded copies of the 8-var flux and the full state —
        # concurrently, tripling the transient peak.  Routing ``current_q`` into
        # the next axis *through* the just-updated accumulator forces the x-axis
        # flux buffer to be freed before the y-axis one is built, holding a
        # single axis live at a time.  ``optimization_barrier`` is a numerical
        # no-op (bit-identical output); at production grid sizes each axis kernel
        # already saturates the GPU, so the lost cross-axis overlap costs little
        # throughput (a few per cent in multi-GPU runs that are not memory
        # bound) while cutting the RHS peak footprint ~3x.
        if use_pallas_div:
            current_q, dq = jax.lax.optimization_barrier((current_q, dq))
        else:
            current_q, rhs_q_for_phys = jax.lax.optimization_barrier(
                (current_q, rhs_q_for_phys)
            )

        # -------------------------------------------------------------
        # ======================= ↑ x-axis ↑ ==========================
        # -------------------------------------------------------------

        # On the fast path with y and z unsplit, the y and z divergences are
        # shard-local plain-JAX updates of the accumulator.
        native_yz_divergence = kept_halo_path and yz_unsplit

        # -------------------------------------------------------------
        # ======================= ↓ y-axis ↓ ==========================
        # -------------------------------------------------------------

        if config.dimensionality >= 2:
            dF_y = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=1,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            Bx_flux_y = dF_y[magnetic_index.x]
            Bz_flux_y = dF_y[magnetic_index.z]
            density_flux_y = dF_y[density_index]
            if native_yz_divergence:
                dq = _hydro_flux_div_axis_native(
                    dF_y,
                    dtdy,
                    axis=1,
                    rhs_accumulator=dq,
                )
            elif use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_y,
                    dtdy,
                    config,
                    axis=1,
                    rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdy * (dF_y - _shift(dF_y, 1, axis=2))
            del dF_y

            # Same serialization barrier between the y and z axes (see above).
            if use_pallas_div:
                current_q, dq = jax.lax.optimization_barrier((current_q, dq))
            else:
                current_q, rhs_q_for_phys = jax.lax.optimization_barrier(
                    (current_q, rhs_q_for_phys)
                )
        else:
            Bx_flux_y = 0.0
            Bz_flux_y = 0.0

        # -------------------------------------------------------------
        # ======================= ↑ y-axis ↑ ==========================
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # ======================= ↓ z-axis ↓ ==========================
        # -------------------------------------------------------------

        if config.dimensionality == 3:
            dF_z = _weno_flux(
                current_q,
                params,
                config,
                registered_variables,
                axis=2,
                internal_energy_density=internal_energy_density,
                inflow_reference=inflow_reference,
            )
            Bx_flux_z = dF_z[magnetic_index.x]
            By_flux_z = dF_z[magnetic_index.y]
            density_flux_z = dF_z[density_index]
            if native_yz_divergence:
                dq = _hydro_flux_div_axis_native(
                    dF_z,
                    dtdz,
                    axis=2,
                    rhs_accumulator=dq,
                )
            elif use_pallas_div:
                dq = _hydro_flux_div_axis_pallas(
                    dF_z,
                    dtdz,
                    config,
                    axis=2,
                    rhs_accumulator=dq,
                )
            else:
                rhs_q_for_phys = rhs_q_for_phys - dtdz * (dF_z - _shift(dF_z, 1, axis=3))
            del dF_z
        else:
            Bx_flux_z = 0.0
            By_flux_z = 0.0

        # -------------------------------------------------------------
        # ======================= ↑ z-axis ↑ ==========================
        # -------------------------------------------------------------

        # -------------------------------------------------------------
        # ============ ↓ Constrained transport and sources ↓ ==========
        # -------------------------------------------------------------

        if fuse_update:
            rhs_bx, rhs_by, rhs_bz = _ct_rhs_pallas_x_precomputed(
                current_q,
                flux_x_modified[0],
                flux_x_modified[1],
                Bx_flux_y,
                Bz_flux_y,
                Bx_flux_z,
                By_flux_z,
                dtdx,
                dtdy,
                dtdz,
                config,
                registered_variables,
            )
        else:
            rhs_bx, rhs_by, rhs_bz = _constrained_transport_rhs_from_slices(
                current_q,
                By_flux_x,
                Bz_flux_x,
                Bx_flux_y,
                Bz_flux_y,
                Bx_flux_z,
                By_flux_z,
                dtdx,
                dtdy,
                dtdz,
                config,
                registered_variables,
            )
        del flux_x_modified

        # Upwind dissipation of the edge EMFs on the interface fields (see
        # ct_upwind_dissipation_rhs); a curl, so div(B) = 0 is kept.
        if config.solver_config.ct_upwind_dissipation:
            upwind_rhs_bx, upwind_rhs_by, upwind_rhs_bz = ct_upwind_dissipation_rhs(
                current_q,
                bx,
                by,
                bz,
                dt,
                grid_spacing,
                gamma,
                params,
                config,
                registered_variables,
            )
            rhs_bx = rhs_bx + upwind_rhs_bx
            rhs_by = rhs_by + upwind_rhs_by
            rhs_bz = rhs_bz + upwind_rhs_bz
        # Explicit ohmic resistivity: a further curl of an edge EMF on the
        # interface fields, so div(B) = 0 is kept exactly (see _resistivity).
        if config.diffusion_config.resistivity:
            resistive_rhs_bx, resistive_rhs_by, resistive_rhs_bz = resistive_face_increment(
                bx,
                by,
                bz,
                params.resistivity,
                dt,
                grid_spacing,
                config.dimensionality,
            )
            rhs_bx = rhs_bx + resistive_rhs_bx
            rhs_by = rhs_by + resistive_rhs_by
            rhs_bz = rhs_bz + resistive_rhs_bz

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
            sources = explicit_sources(
                current_q,
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
                density_fluxes=density_fluxes,
                density_increment=dq[registered_variables.density_index],
                sink_potential=sink_potential,
            )
            if sources is not None:
                dq = dq + sources
        else:
            rhs_q_for_phys += explicit_sources(
                current_q,
                dt,
                gamma,
                config,
                params,
                helper_data,
                registered_variables,
                density_fluxes=density_fluxes,
                density_increment=rhs_q_for_phys[registered_variables.density_index],
                sink_potential=sink_potential,
            )
            dq = a_coef * dq + rhs_q_for_phys

        # -------------------------------------------------------------
        # ============ ↑ Constrained transport and sources ↑ ==========
        # -------------------------------------------------------------

        return dq, rhs_bx, rhs_by, rhs_bz

    def compute_lqs(current_q, bx, by, bz, dq, a_coef):
        """The stage increments of ``stage_rhs``, on the fast path if it applies."""
        primals = (
            current_q,
            bx,
            by,
            bz,
            dq,
            a_coef,
            dt,
            gamma,
            grid_spacing,
            params,
            helper_data,
            sink_potential,
        )
        if not use_kept_x_flux_halo:
            return stage_rhs(False, *primals)
        # The kept-halo kernels call Pallas outside the differentiable
        # wrappers, so derivatives are taken through the standard path, which
        # computes the same stage.
        return diffable_pallas_call_n(
            primals,
            pallas_branch=partial(stage_rhs, True),
            native_branch=partial(stage_rhs, False),
        )

    def pre_stage(u):
        q, bx, by, bz = u
        if config.boundary_handling == GHOST_CELLS:
            q = _boundary_handler(
                q, config, registered_variables, params, CONSERVATIVE_GAS_STATE,
            )
            b_curr = _boundary_handler(
                jnp.stack([bx, by, bz], axis=0),
                config,
                registered_variables,
                params,
                MAGNETIC_FIELD_ONLY,
            )
            bx, by, bz = b_curr[0], b_curr[1], b_curr[2]
        return (q, bx, by, bz)

    def lsrk_increment(u, du, a_coef, dt_step):
        # ``compute_lqs`` returns the new ``dq`` already in LSRK4 form
        # (``a_coef * dq_old + dt * L_q``), folding the accumulate into the
        # divergence kernel when Pallas is available.  The interface-B deltas
        # use the explicit ``a_coef * db + dt * L_b`` low-storage update.
        # ``lsrk4`` hands every stage the full step ``dt_step == dt``, which
        # ``compute_lqs`` already closes over.
        q, bx, by, bz = u
        dq, dbx, dby, dbz = du
        dq, rhs_bx, rhs_by, rhs_bz = compute_lqs(q, bx, by, bz, dq, a_coef)
        dbx = a_coef * dbx + rhs_bx
        dby = a_coef * dby + rhs_by
        dbz = a_coef * dbz + rhs_bz
        return (dq, dbx, dby, dbz)

    lsrk_increment = _stage_remat(lsrk_increment, config)

    def finalize(u):
        q, bx, by, bz = u
        q = update_cell_center_fields(
            q, bx, by, bz, config, registered_variables,
        )
        return (q, bx, by, bz)

    return lsrk4(
        (conserved_state, bx_interface, by_interface, bz_interface),
        dt,
        pre_stage=_stage_remat(pre_stage, config),
        finalize=_stage_remat(finalize, config),
        lsrk_increment=lsrk_increment,
    )
