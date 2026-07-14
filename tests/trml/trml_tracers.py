"""
Turbulent radiative mixing layer (TRML) with Lagrangian tracer particles.

Same hydro setup as ``trml.py`` (Lancaster 2026), but additionally seeds
mass-weighted tracer particles and records their temperature (and position)
along their Lagrangian trajectories at every snapshot. This is the data the
Fokker-Planck self-consistency test (``fokker_planck_test.py``) consumes:

  * tracer temperature time series T_p(t)  -> transition statistics A(T), D(T)
  * mass-weighted Eulerian temperature PDF -> the marginal P(T) to match
  * Lagrangian tracer temperature marginal -> validation of mass weighting

Set ``QUICK = True`` for a cheap pipeline shakedown (small N, short run).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

import os

import jax
import jax.numpy as jnp
import numpy as np

from astronomix import (
    SimulationConfig,
    get_helper_data,
    SimulationParams,
    time_integration,
    construct_primitive_state,
    get_registered_variables,
)
from astronomix.data_classes.simulation_state_struct import StateStruct
from astronomix.option_classes.simulation_params import FixedBoundaryState, FixedBoundaryState1D
from astronomix.option_classes.simulation_config import (
    FIXED_BOUNDARY_OPEN_MOMENTUM,
    OPEN_BOUNDARY,
    PALLAS,
    PositivityConfig,
    SnapshotSettings,
    StaticFloatVector,
    StaticIntVector,
    finalize_config,
    FINITE_DIFFERENCE,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
)
from astronomix._modules._tracers._tracer_options import TracerConfig, MASS_WEIGHTED
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    SIMPLE_MIXING_LAYER_COOLING,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
    MixingCoolingParams,
)

# ---- run size --------------------------------------------------------------
# TRML_PRESET = quick (cheap shakedown) | long64 (steady-state N=64) | fiducial (N=128)
#
# Two-phase strategy (per the FP study findings):
#   Phase 1 spins up to equilibrium with NO tracers / NO snapshots.
#   Phase 2 then seeds tracers mass-weighted from the EQUILIBRIUM density (so the
#   Lagrangian marginal equals the Eulerian mass PDF by construction -> Check 0
#   passes) and records densely over a SHORT window, giving a much finer lag
#   spacing dt_min = t_record / num_snapshots to resolve the diffusive window.
PRESET = os.environ.get("TRML_PRESET", "long64")
# TRML_SUFFIX overrides the output suffix (used by the xi scan to keep per-xi
# data/figures separate without adding a preset for every value).
SUFFIX = os.environ.get(
    "TRML_SUFFIX", {"quick": "_quick", "long64": "_long64", "fiducial": ""}[PRESET]
)

if PRESET == "quick":
    num_cells_x = 64
    t_spinup_in_t_sh = 3.0      # phase-1 spin-up (no tracers)
    t_record_in_t_sh = 3.0      # phase-2 recording window
    num_snapshots = 600         # dense recording -> dt_min = t_record/num_snapshots
    num_tracers = 5000
    settle_t_sh = 0.3           # analysis: skip this much of the recording window
elif PRESET == "long64":
    num_cells_x = 64
    t_spinup_in_t_sh = 12.0
    t_record_in_t_sh = 8.0
    num_snapshots = 2000
    num_tracers = 20000
    settle_t_sh = 0.5
else:  # fiducial
    num_cells_x = 128
    t_spinup_in_t_sh = 15.0
    t_record_in_t_sh = 10.0
    num_snapshots = 2500
    num_tracers = 20000
    settle_t_sh = 0.5

# ---- box setup -------------------------------------------------------------
num_cells_y = num_cells_x
num_cells_z = int(1.5 * num_cells_x)
box_size = 1.0
grid_spacing = box_size / num_cells_x
L_x = box_size
L_y = box_size
L_z = 1.5 * box_size

# ---- mixing settings -------------------------------------------------------
density_contrast = 100.0
xi = float(os.environ.get("TRML_XI", "100.0"))   # Damkohler proxy = t_sh / t_coolmin
mach_number = 0.5
gamma = 5 / 3
P0 = 1.0

rho_hot = 1.0
rho_cold = density_contrast * rho_hot
T_hot = P0 / rho_hot
T_cold = P0 / rho_cold
c_hot = (gamma * P0 / rho_hot) ** 0.5
v_rel = mach_number * c_hot
t_sh = L_x / v_rel
t_coolmin = t_sh / xi

# ---- simulation config -----------------------------------------------------
config = SimulationConfig(
    solver_mode=FINITE_DIFFERENCE,
    backend=PALLAS,
    pallas_block_shape=(4, 4, 8),
    pallas_use_triton=True,
    pallas_interpret=False,
    # progress_bar forces a host sync every step (jax.debug.callback) which
    # dominates wall-clock on these cheap steps; keep it off for throughput.
    memory_analysis=False,
    print_elapsed_time=True,
    progress_bar=False,
    dimensionality=3,
    box_size=StaticFloatVector(L_x, L_y, L_z),
    num_cells=StaticIntVector(num_cells_x, num_cells_y, num_cells_z),
    boundary_settings=BoundarySettings(
        x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        z=BoundarySettings1D(OPEN_BOUNDARY, FIXED_BOUNDARY_OPEN_MOMENTUM),
    ),
    # positivity protection replaces the old enforce_positivity flag; keep it on
    # so the strong density contrast plus cooling stays stable.
    positivity_config=PositivityConfig(default_positivity_protection=True),
    cooling_config=CoolingConfig(
        cooling=True,
        cooling_method=IMPLICIT_COOLING,
        cooling_curve_config=CoolingCurveConfig(
            cooling_curve_type=SIMPLE_MIXING_LAYER_COOLING,
        ),
    ),
    frame_tracking=True,
    # ---- tracer particles ----
    # This is the PHASE-2 (recording) config. Tracers are seeded mass-weighted
    # from the equilibrium density (passed explicitly), so the Lagrangian marginal
    # matches the Eulerian mass PDF without recycling: recycle=False. reinject
    # still relocates the rare boundary-crosser to the inflow.
    # Regeneration thermostat keeps the Lagrangian marginal equal to the
    # instantaneous mass PDF (the thin mixing layer turns over faster than a
    # fixed tracer set, fed by un-tracered inflow, can follow). The timescale
    # (in t_sh) is well above the diffusive-window lags so trajectory segments
    # stay long enough to measure A(T), D(T). Tunable via TRML_REGEN_TSH.
    state_struct=True,
    tracer_config=TracerConfig(
        tracers=True,
        num_tracers=num_tracers,
        seed_mode=MASS_WEIGHTED,
        record_positions=True,
        recycle=False,
        reinject=True,
        regenerate=True,
        regenerate_timescale=float(os.environ.get("TRML_REGEN_TSH", "0.2")) * t_sh,
    ),
    return_snapshots=True,
    num_snapshots=num_snapshots,
    snapshot_settings=SnapshotSettings(
        return_final_state=True,
        return_states=False,
        return_temperature_pdf=True,
        return_mass_weighted_temperature_pdf=True,
        # The cold phase compresses (rho up to ~1.6 rho_cold) and cools BELOW the
        # nominal T_cold, with a tail to ~1e-3; the hot phase slightly overshoots
        # T_hot. Use a range that brackets the actual gas so no mass is dropped
        # (a [T_cold, T_hot] range cuts through the middle of the cold peak).
        num_temperature_bins=150,
        temperature_pdf_min=1e-3,
        temperature_pdf_max=1.5,
    ),
)


def single_interface(f_l, f_u, Z, z_center, smoothing_length):
    return 0.5 * (
        f_l * (1 - jnp.tanh((Z - z_center) / smoothing_length))
        + f_u * (1 + jnp.tanh((Z - z_center) / smoothing_length))
    )


registered_variables = get_registered_variables(config)

helper_data = get_helper_data(config)
cell_centers = helper_data.geometric_centers
X = cell_centers[:, :, :, 0]
Y = cell_centers[:, :, :, 1]
Z = cell_centers[:, :, :, 2]
z_center = L_z / 2
smoothing_length = grid_spacing / 2
density = single_interface(rho_cold, rho_hot, Z, z_center, smoothing_length)
pressure = P0 * jnp.ones_like(density)
velocity_x = single_interface(-v_rel / 2, v_rel / 2, Z, z_center, smoothing_length)
velocity_y = jnp.zeros_like(density)
velocity_z = jnp.zeros_like(density)

# --- perturbation (identical recipe to trml.py) -----------------------------
dz = Z - z_center
envelope = jnp.exp(-((jnp.abs(dz) - 2 * smoothing_length) / (3 * grid_spacing)) ** 2)
amp = 0.03 * v_rel
mode_numbers = jnp.array([2, 4, 6])
kx = 2 * jnp.pi * mode_numbers / L_x
ky = 2 * jnp.pi * mode_numbers / L_y
key = jax.random.PRNGKey(42)
key_ph, key_n = jax.random.split(key, 2)
phases = jax.random.uniform(key_ph, (3, 3), minval=0.0, maxval=2 * jnp.pi)
modes = jnp.zeros_like(density)
for i in range(3):
    for j in range(3):
        modes = modes + jnp.sin(kx[i] * X + ky[j] * Y + phases[i, j])
modes = modes / 9.0
key_nx, key_ny, key_nz = jax.random.split(key_n, 3)
noise_x = jax.random.normal(key_nx, density.shape)
noise_y = jax.random.normal(key_ny, density.shape)
noise_z = jax.random.normal(key_nz, density.shape)
velocity_x = velocity_x + amp * envelope * 0.3 * noise_x
velocity_y = velocity_y + amp * envelope * 0.3 * noise_y
velocity_z = velocity_z + amp * envelope * (modes + 0.3 * noise_z)

initial_state = construct_primitive_state(
    config=config,
    registered_variables=registered_variables,
    density=density,
    velocity_x=velocity_x,
    velocity_y=velocity_y,
    velocity_z=velocity_z,
    gas_pressure=pressure,
)

mixing_cooling_params = MixingCoolingParams(
    xi=xi,
    mach_number=mach_number,
    density_contrast=density_contrast,
)

base_params_kwargs = dict(
    C_cfl=1.5,
    gamma=gamma,
    minimum_density=rho_cold / 100,
    minimum_pressure=P0 / 100,
    fixed_boundary_state=FixedBoundaryState(
        z=FixedBoundaryState1D(
            right_state=jnp.array([rho_hot, v_rel / 2, 0.0, 0.0, P0])
        )
    ),
    cooling_params=CoolingParams(
        cooling_curve_params=mixing_cooling_params,
        floor_temperature=T_cold,
    ),
)

# Phase-1 (spin-up) config: identical hydro, but NO tracers and NO snapshots,
# returning the bare final (equilibrium) state.
spinup_config = config._replace(
    state_struct=False,
    tracer_config=TracerConfig(tracers=False),
    return_snapshots=False,
    snapshot_settings=config.snapshot_settings._replace(return_final_state=False),
)
spinup_config = finalize_config(spinup_config, initial_state.shape)

# Phase-2 (recording) config: tracers on, dense snapshots over the short window.
record_config = finalize_config(config, initial_state.shape)

spinup_params = SimulationParams(t_end=t_spinup_in_t_sh * t_sh, **base_params_kwargs)
record_params = SimulationParams(t_end=t_record_in_t_sh * t_sh, **base_params_kwargs)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    eq_path = f"data/trml_equilibrium{SUFFIX}.npz"
    if os.environ.get("TRML_LOAD_EQUILIBRIUM", "0") == "1" and os.path.exists(eq_path):
        print(f"loading equilibrium state from {eq_path}")
        equilibrium_state = jnp.asarray(np.load(eq_path)["equilibrium_state"])
    else:
        print(f"=== Phase 1: spin-up to equilibrium ({t_spinup_in_t_sh} t_sh) ===")
        equilibrium_state = time_integration(
            initial_state, spinup_config, spinup_params, registered_variables
        )
        np.savez(eq_path, equilibrium_state=np.asarray(equilibrium_state))
        print(f"saved equilibrium state to {eq_path}")

    print(f"=== Phase 2: record with tracers ({t_record_in_t_sh} t_sh, "
          f"{num_snapshots} snapshots, dt_min={t_record_in_t_sh/num_snapshots:.4f} t_sh) ===")
    # Tracers auto-seed mass-weighted from the EQUILIBRIUM density (tracers=None).
    result = time_integration(
        StateStruct(primitive_state=equilibrium_state),
        record_config,
        record_params,
        registered_variables,
    )

    # log-spaced temperature bin edges shared by the Eulerian PDFs — MUST match
    # the histogram range actually used when recording (temperature_pdf_min/max),
    # not the nominal T_cold/T_hot, or the analysis bins the tracers on a
    # different grid than the recorded Eulerian PDFs.
    logT_bin_edges = np.linspace(
        np.log10(record_config.snapshot_settings.temperature_pdf_min),
        np.log10(record_config.snapshot_settings.temperature_pdf_max),
        record_config.snapshot_settings.num_temperature_bins + 1,
    )

    out_path = f"data/trml_tracers{SUFFIX}.npz"
    tracer_generation = (
        np.asarray(result.tracer_generation)
        if result.tracer_generation is not None
        else np.zeros_like(np.asarray(result.tracer_temperature), dtype=np.int32)
    )
    np.savez(
        out_path,
        time_points=np.asarray(result.time_points),
        tracer_temperature=np.asarray(result.tracer_temperature),
        tracer_position=np.asarray(result.tracer_position),
        tracer_generation=tracer_generation,
        temperature_pdf=np.asarray(result.temperature_pdf),
        mass_temperature_pdf=np.asarray(result.mass_temperature_pdf),
        logT_bin_edges=logT_bin_edges,
        # scalars / metadata for the analysis
        T_cold=T_cold,
        T_hot=T_hot,
        t_sh=t_sh,
        t_coolmin=t_coolmin,
        L_x=L_x,
        L_y=L_y,
        L_z=L_z,
        grid_spacing=grid_spacing,
        # the recording window is already equilibrium; skip only a short settle
        steady_state_t_sh=settle_t_sh,
        xi=xi,
        mach_number=mach_number,
        density_contrast=density_contrast,
    )
    print(f"saved tracer data to {out_path}")
    print(f"  tracer_temperature shape: {result.tracer_temperature.shape}")
    print(f"  tracer_position shape:    {result.tracer_position.shape}")
    print(f"  num_iterations:           {result.num_iterations}")
