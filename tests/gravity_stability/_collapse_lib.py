"""Self-contained Evrard-collapse runner for the gravity-stability experiments.

Mirrors ``paper_plots/gravity/_collapse.py`` but accepts arbitrary extra
``SimulationConfig`` keyword arguments (``extra_config``) so the new
energy-source-distribution knobs can be swept without touching the paper
scripts. Reports per-snapshot energy diagnostics for stability analysis.
"""

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    SimulationConfig,
    SimulationParams,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.initial_condition_generation.construct_primitive_state import (
    construct_primitive_state,
)
from astronomix.option_classes.simulation_config import (
    NATIVE_JAX,
    PALLAS,
    FORWARDS,
    BoundarySettings,
    BoundarySettings1D,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    GravityConfig,
    PositivityConfig,
    POSITIVITY_NONE,
    SnapshotSettings,
    finalize_config,
)

GAMMA = 5 / 3
BOX_SIZE = 4.0
NUM_SNAPSHOTS = 60
PALLAS_BLOCK_SHAPE = (4, 4, 8)


def _backend_kwargs(backend):
    if backend == PALLAS:
        return dict(
            backend=PALLAS,
            pallas_block_shape=PALLAS_BLOCK_SHAPE,
            pallas_use_triton=True,
            pallas_interpret=False,
        )
    return dict(backend=NATIVE_JAX)


def collapse_config(num_cells, self_gravity_version, backend=PALLAS,
                    want_states=False, pp_flux=False, protect=False,
                    per_stage_mode=POSITIVITY_NONE, positivity_overrides=None):
    # By default NO floor (default_positivity_protection=False) so the energy
    # conservation measure is clean; pp_flux / protect / per_stage_mode opt in.
    pos = dict(
        default_positivity_protection=bool(protect),
        per_stage_mode=per_stage_mode,
        preserving_flux=bool(pp_flux),
    )
    if positivity_overrides:
        pos.update(positivity_overrides)

    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        runtime_debugging=False,
        progress_bar=False,
        mhd=False,
        dimensionality=3,
        box_size=BOX_SIZE,
        num_cells=num_cells,
        differentiation_mode=FORWARDS,
        gravity_config=GravityConfig(
            self_gravity=True,
            self_gravity_version=self_gravity_version,
            poisson_manual_open_boundaries=True,
        ),
        positivity_config=PositivityConfig(**pos),
        boundary_settings=BoundarySettings(
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        ),
        return_snapshots=True,
        snapshot_settings=SnapshotSettings(
            return_states=want_states,
            return_final_state=True,
            return_total_energy=True,
            return_internal_energy=True,
            return_kinetic_energy=True,
            return_gravitational_energy=True,
        ),
        num_snapshots=NUM_SNAPSHOTS,
        **_backend_kwargs(backend),
    )


def run_collapse(
    num_cells, self_gravity_version, t_end, backend=PALLAS,
    initial_energy=0.05, want_states=False, pp_flux=False, protect=False,
    per_stage_mode=POSITIVITY_NONE,
):
    config = collapse_config(
        num_cells, self_gravity_version, backend=backend,
        want_states=want_states, pp_flux=pp_flux, protect=protect,
        per_stage_mode=per_stage_mode,
    )
    params = SimulationParams(
        t_end=t_end,
        C_cfl=0.4,
        dt_max=jnp.inf,
        minimum_density=1e-5,
        minimum_pressure=3e-6,
    )

    helper_data = get_helper_data(config)
    registered_variables = get_registered_variables(config)

    R = 1.0
    M = 1.0
    dx = config.box_size / (config.num_cells - 1)

    rho = jnp.where(
        helper_data.r <= R, M / (2 * jnp.pi * R**2 * helper_data.r), 1e-4
    )
    v = jnp.zeros_like(rho)
    e = initial_energy
    p = (GAMMA - 1) * rho * e
    p = jnp.where(p < params.minimum_pressure, params.minimum_pressure, p)

    initial_state = construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=v,
        velocity_y=v,
        velocity_z=v,
        gas_pressure=p,
    )
    config = finalize_config(config, initial_state.shape)

    snapshots = jax.block_until_ready(
        time_integration(initial_state, config, params, registered_variables)
    )
    return snapshots, helper_data, registered_variables


def diagnose(snapshots):
    """Return a dict of stability/conservation diagnostics for a run.

    A crash leaves the remaining snapshot-buffer slots bit-exactly zero (and a
    final NaN slot). Real collapse total energy is never bit-exactly 0, so a
    snapshot is "good" only if its total energy is finite AND nonzero; the run
    crashed if any later slot is dead (==0) or non-finite.
    """
    t = np.asarray(snapshots.time_points)
    total = np.asarray(snapshots.total_energy)
    good = np.isfinite(total) & (total != 0.0)
    n_good = int(good.sum())
    # first dead/non-finite slot after the run started
    bad = ~good
    bad[0] = False  # snapshot 0 is always the (good) IC
    crash_idx = int(np.argmax(bad)) if bad.any() else -1
    crashed = bool(bad.any())
    if n_good > 0:
        e0 = total[0]
        rel_err = np.abs(total[good] - e0) / np.abs(e0)
        max_rel_err = float(np.nanmax(rel_err))
        final_rel_err = float(rel_err[-1])
        t_final = float(t[good][-1])
    else:
        max_rel_err = final_rel_err = t_final = float("nan")
    return dict(
        t_final=t_final,
        t_target=float(t[-1]),
        n_finite=n_good,
        n_total=len(t),
        crashed=crashed,
        crash_t=float(t[crash_idx]) if crash_idx >= 0 else None,
        max_rel_err=max_rel_err,
        final_rel_err=final_rel_err,
    )
