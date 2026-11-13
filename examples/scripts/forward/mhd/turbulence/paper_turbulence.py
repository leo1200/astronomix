"""
Driven isothermal-MHD turbulence after HOW-MHD (Seo & Ryu 2023,
arXiv:2304.04360, Secs. 3.9 / 3.10), with Ornstein-Uhlenbeck (temporally
correlated) driving instead of the paper's white-in-time driving. Produces the
data of the paper's ``turbulence_slices`` and ``turbulence_spectra`` figures
(``make_fig14.py`` / ``make_fig15.py``). It runs the TurbGen driving now
(solenoidal, parabola over 1 <= k <= 2 with the peak at 1.5 in units of
2 pi / L, amplitude adjusted to the target dispersion); the paper's figures
were made with the earlier forcing module (tag ``pre-streamline``).

Cases (one per invocation):
  ISM : isothermal, M_turb ~ 10,  beta_p = 0.1, run to  5 t_cross
  ICM : isothermal, M_turb ~ 0.5, beta_p = 1e6, run to 30 t_cross
  CMP : M_turb ~ 1, beta_p = 1, isothermal and adiabatic, run to 4 t_cross

M_turb = v_rms / a, beta_p = P_thermal / (B0^2 / 2). The default normalisation
keeps the driven v_rms ~ 1 and sets a = 1 / M_turb, B0 = sqrt(2 P_thermal /
beta) with P_thermal = a^2 rho0 (isothermal) or P0 (adiabatic,
a = sqrt(gamma P0 / rho0)). It is physically identical to the paper's a = 1,
v_rms = M_turb normalisation for every dimensionless diagnostic.

The driving peaks where the paper's spectrum |dv_k|^2 ~ k^6 exp(-8 k / k_exp),
k_exp = 4 pi / L0, does: at k_f = 0.75 k_exp = 3 pi, i.e. 1.5 in units of
2 pi / L (``--kdrive``).

The run is split into ``--nsnap`` equal intervals. Each interval is one
``time_integration`` call in the disk-checkpointing (``TO_DISK``) mode, which
writes the loop carry (state, driving state) as an Orbax checkpoint to
``<ckpt-root>/<tag>/<interval>``; the next interval continues from that
checkpoint, so a run resumed in a new Slurm job takes exactly the path of an
uninterrupted one. The state at the end of every interval is reduced on the
GPU (scalars, density / kinetic / magnetic spectra, mid-plane slices) and only
the reduced arrays reach the host, so the memory cost does not grow with the
number of snapshots. The accumulated diagnostics are saved next to the
checkpoints after every interval; re-running the same command resumes from the
last finished interval. Once the last interval is done, the reduced data is
written to ``data/paper_<tag>.npz``.

    python paper_turbulence.py --mturb 10 --beta 0.1 --tcross 5 --N 256 --tag ISM_N256
"""

# ==== GPU selection ====
import os
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    from autocvd import autocvd
    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import json
import shutil
import sys
import time as walltime
from pathlib import Path

# typing
from typing import Any, NamedTuple

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import PERIODIC_BOUNDARY
from astronomix.option_classes.simulation_config import (
    IDEAL_GAS,
    ISOTHERMAL,
    ON_DEVICE,
    POSITIVITY_HARD_FLOOR,
    POSITIVITY_NONE,
    TO_DISK,
)

# astronomix containers
from astronomix import (
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix._modules._turbulence_driving import TurbulenceDrivingConfig, TurbulenceDrivingParams
from astronomix.time_stepping.time_integration import LoopState

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)
from astronomix._snapshotting._orbax_storage import (
    latest_step,
    load_loop_checkpoint,
)
from astronomix.analysis_helpers.energy_spectrum import (
    get_kinetic_energy_spectrum,
    get_magnetic_energy_spectrum,
    vector_field_energy_spectrum,
)


#: Script-relative data directory, shared with make_fig14.py / make_fig15.py.
DATA_DIR = Path(__file__).resolve().parent / "data"

#: Default root of the per-run checkpoint directories (transient scratch).
CHECKPOINT_ROOT = Path("/e/scratch/astronomix/revision/turbulence")

#: Accumulated diagnostics, saved next to the Orbax checkpoints of a run.
DIAGNOSTICS_FILE = "diagnostics.npz"

#: A snapshot counts as alive (a usable slice) while v_rms exceeds this; a run
#: that collapsed to rest or to NaN keeps the slice of its last alive snapshot.
ALIVE_VRMS = 0.1

#: The scalar diagnostics recorded per snapshot, in the order the reducer
#: returns them (volume averages over the box unless noted).
SCALAR_NAMES = (
    "vrms_t",        # sqrt(<v^2>)
    "drho_t",        # sqrt(<(rho - rho0)^2>)
    "EK_t",          # <rho v^2 / 2>
    "EB_t",          # <B^2 / 2>
    "rho_min_t",     # minimum density
    "rho_max_t",     # maximum density
    "vmax_t",        # maximum |v|
    "sound_t",       # sound speed (mean of sqrt(gamma P / rho) when adiabatic)
    "nonfinite_t",   # 1 when any state value is NaN or infinite
)

#: The spectra recorded per snapshot (the npz keys make_fig15.py reads).
SPECTRUM_NAMES = ("spec_rho", "spec_EK", "spec_EB")

#: The mid-plane slices kept from the last alive snapshot.
SLICE_NAMES = ("EK_slice", "EB_slice", "rho_slice")


class TurbulenceSetup(NamedTuple):
    """Everything the run needs besides the command-line arguments.

    Attributes:
        config: The finalized simulation configuration (TO_DISK, one interval
            per ``time_integration`` call).
        params: The simulation parameters (t_start / t_end are set per interval).
        registered_variables: The registered variables.
        initial_pressure: The initial gas pressure (adiabatic runs; ``None``
            otherwise).
        sound_speed: The (initial) sound speed a.
        magnetic_field: The initial guide field B0 along z.
        crossing_time: t_cross = L_inj / v_rms aimed at.
        end_time: The end time, ``tcross * crossing_time``.
        positivity_preserving: Whether positivity-preserving WENO is on.
        checkpoint_directory: The Orbax checkpoint directory of this run.
    """
    config: Any
    params: Any
    registered_variables: Any
    initial_pressure: Any
    sound_speed: float
    magnetic_field: float
    crossing_time: float
    end_time: float
    positivity_preserving: bool
    checkpoint_directory: Path


def parse_arguments():
    """Parse the command line (see the module docstring for the cases)."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mturb", type=float, required=True, help="target turbulent Mach number")
    parser.add_argument("--beta", type=float, required=True, help="initial plasma beta")
    parser.add_argument("--eos", choices=["iso", "adiabatic"], default="iso")
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--N", type=int, default=128, help="cells per dimension")
    parser.add_argument("--tcross", type=float, default=5.0, help="t_end in crossing times")
    parser.add_argument(
        "--kdrive",
        type=float,
        default=1.5,
        help="peak of the driving in units of 2 pi / L (= 0.75 k_exp); driven band kdrive -+ 0.5",
    )
    parser.add_argument(
        "--amplitude-factor",
        type=float,
        default=1.0,
        help="TurbGen amplitude factor at the start (adjusted automatically towards the target dispersion)",
    )
    parser.add_argument(
        "--nsnap",
        type=int,
        default=60,
        help="number of equal intervals; a snapshot is reduced at t = 0 and at the end "
             "of every interval, and every interval end is a checkpoint",
    )
    parser.add_argument("--cfl", type=float, default=1.5)
    parser.add_argument(
        "--rhomin",
        type=float,
        default=1e-10,
        help="minimum_density: the floor of the positivity-preserving WENO scalings "
             "(a larger value would limit genuine voids)",
    )
    parser.add_argument(
        "--diag",
        type=int,
        default=0,
        help="blow-up diagnosis instead of the production run: per-snapshot scalars "
             "(min rho, max|v|, max|B|, max b^2/rho, NaN flag) through the snapshot "
             "callback, with a progress bar, to pinpoint when and how a run fails. "
             "Use a high --nsnap. No checkpoints.",
    )
    parser.add_argument("--outdir", type=str, default=str(DATA_DIR))
    parser.add_argument(
        "--norm",
        choices=["vrms1", "paper"],
        default="vrms1",
        help="vrms1: a = 1/M_turb, drive v_rms ~ 1. paper: a = 1, drive v_rms ~ M_turb, "
             "B0 = sqrt(2/beta), the normalisation of HOW-MHD's Fig. 14 log E_B",
    )
    parser.add_argument(
        "--mhd",
        type=int,
        default=1,
        help="1 = MHD (default), 0 = pure hydro (no magnetic field)",
    )
    parser.add_argument(
        "--weno-z",
        type=int,
        default=1,
        help="1 = WENO-Z nonlinear weights (the default scheme), 0 = Jiang-Shu",
    )
    parser.add_argument(
        "--pp",
        choices=["auto", "1", "0"],
        default="auto",
        help="positivity-preserving WENO: 'auto' switches it on for supersonic driving "
             "(M_turb >= 2), where deep voids form, and leaves the subsonic runs on the "
             "plain scheme, whose low-Mach accuracy it would cost",
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default=str(CHECKPOINT_ROOT),
        help="root of the checkpoint directories; this run uses <root>/<tag>",
    )
    parser.add_argument(
        "--keep-checkpoints",
        type=int,
        default=2,
        help="number of most recent interval checkpoints kept on disk (at least 1)",
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=0,
        help="stop after this many intervals in this job (0 = no limit); the next "
             "invocation resumes",
    )
    parser.add_argument(
        "--stop-after-hours",
        type=float,
        default=0.0,
        help="do not start an interval that would end later than this many hours after "
             "the job start (0 = no limit), so a Slurm job ends cleanly before its walltime",
    )
    parser.add_argument(
        "--floor",
        type=int,
        default=0,
        help="1 = AthenaPK's density floor at --rhomin once per step (momentum kept), which "
             "bounds the Alfven speed of near-vacuum, strongly magnetised cells; --rhomin is "
             "then both the floor and the target of the positivity-preserving WENO",
    )
    parser.add_argument(
        "--alfven-ceiling",
        type=float,
        default=0.0,
        help="with --floor 1: also AthenaPK's Alfven-speed ceiling, rho >= |B|^2 / v^2 for this v "
             "(0 = off), which keeps bounding the time step when the field in evacuated cells grows",
    )
    parser.add_argument(
        "--ct-upwind",
        type=float,
        nargs="?",
        const=0.03,
        default=0.0,
        help="upwind dissipation of the constrained-transport edge EMFs "
             "(FDSolverConfig.ct_upwind_dissipation) at strength kappa (ct_upwind_strength; "
             "the bare flag is the default 0.03, 0 = off, 1 = the full upwind form), which damps "
             "the grid-scale modes of the interface fields that the central CT interpolations "
             "leave undamped",
    )
    parser.add_argument(
        "--step-retry",
        action="store_true",
        help="redo a step at half the step size when it leaves a non-positive density "
             "(PositivityConfig.step_retry); not part of the run signature, so a run can be "
             "resumed with it",
    )
    parser.add_argument(
        "--branch-from",
        type=str,
        default="",
        help="TAG:STEP -- start this run from snapshot STEP of run TAG (same --ckpt-root), "
             "e.g. to continue it with --floor; only --rhomin and --floor may differ",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete this run's checkpoints and start from t = 0",
    )
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


# -------------------------------------------------------------
# ================== ↓ Problem setup ↓ ========================
# -------------------------------------------------------------
def build_initial_state(args, config, registered_variables, magnetic_field, initial_pressure):
    """
    The state at t = 0: uniform gas at rest, threaded (MHD) by the uniform
    field B0 along z.

    Args:
        args: The parsed command-line arguments.
        config: The simulation configuration.
        registered_variables: The registered variables.
        magnetic_field: The guide field B0.
        initial_pressure: The gas pressure (adiabatic runs; ``None`` otherwise).

    Returns:
        The primitive state.
    """
    rho0 = 1.0
    shape = (args.N, args.N, args.N)
    density = jnp.full(shape, rho0, dtype=jnp.float32)
    zero = jnp.zeros_like(density)
    initial_fields = dict(
        config=config,
        registered_variables=registered_variables,
        density=density,
        velocity_x=zero,
        velocity_y=zero,
        velocity_z=zero,
    )
    if args.mhd:
        magnetic_field_z = jnp.full(shape, magnetic_field, dtype=jnp.float32)
        interface_x, interface_y, interface_z = initialize_interface_fields(
            zero,
            zero,
            magnetic_field_z,
        )
        initial_fields.update(
            magnetic_field_x=zero,
            magnetic_field_y=zero,
            magnetic_field_z=magnetic_field_z,
            interface_magnetic_field_x=interface_x,
            interface_magnetic_field_y=interface_y,
            interface_magnetic_field_z=interface_z,
        )
    if args.eos == "adiabatic":
        initial_fields["gas_pressure"] = jnp.full(shape, initial_pressure, dtype=jnp.float32)
    return construct_primitive_state(**initial_fields)


def build_setup(args):
    """
    Build the configuration, parameters and initial state of a driven
    turbulence run: uniform gas at rest threaded by a uniform field B0 along z.

    Args:
        args: The parsed command-line arguments.

    Returns:
        A :class:`TurbulenceSetup`.
    """
    rho0 = 1.0
    if args.norm == "paper":
        # Sound speed fixed; the driving reaches v_rms ~ M_turb.
        sound_speed = 1.0
        vrms_target = args.mturb
    else:
        sound_speed = 1.0 / args.mturb
        vrms_target = 1.0
    adiabatic = args.eos == "adiabatic"
    if adiabatic:
        # P0 such that sqrt(gamma P0 / rho0) = a.
        initial_pressure = rho0 * sound_speed ** 2 / args.gamma
        thermal_pressure = initial_pressure
    else:
        initial_pressure = None
        thermal_pressure = sound_speed ** 2 * rho0
    magnetic_field = float(np.sqrt(2.0 * thermal_pressure / args.beta))
    if args.pp == "auto":
        positivity_preserving = args.mturb >= 2.0
    else:
        positivity_preserving = args.pp == "1"
    checkpoint_directory = Path(args.ckpt_root) / args.tag

    periodic = BoundarySettings1D(
        left_boundary=PERIODIC_BOUNDARY,
        right_boundary=PERIODIC_BOUNDARY,
    )
    config = SimulationConfig(
        equation_of_state=IDEAL_GAS if adiabatic else ISOTHERMAL,
        memory_analysis=False,
        # Progress is reported per interval by this script.
        progress_bar=False,
        dimensionality=3,
        num_cells=args.N,
        box_size=1.0,
        mhd=bool(args.mhd),
        boundary_settings=BoundarySettings(periodic, periodic, periodic),
        turbulence_driving_config=TurbulenceDrivingConfig(
            turbulence_driving=True,
            k_min=args.kdrive - 0.5,
            k_max=args.kdrive + 0.5,
            amplitude_auto_adjust=True,
        ),
        solver_config=FDSolverConfig(
            weno_z=bool(args.weno_z),
            # Positivity-preserving WENO keeps the deep voids of the supersonic
            # runs positive without floors, vacuum handling or flux blending.
            weno_positivity_preserving=positivity_preserving,
            ct_upwind_dissipation=bool(args.mhd) and args.ct_upwind > 0.0,
            ct_upwind_strength=args.ct_upwind if args.ct_upwind > 0.0 else 0.03,
        ),
        positivity_config=PositivityConfig(
            per_step_mode=POSITIVITY_HARD_FLOOR if args.floor else POSITIVITY_NONE,
            step_retry=args.step_retry,
        ),
        # One interval per time_integration call: TO_DISK with a single
        # segment integrates [t_start, t_end] and checkpoints the loop carry
        # (state, driving state) at its end, which is what lets the
        # next interval -- in this job or the next one -- continue the same
        # stochastic realisation.
        snapshot_storage_mode=TO_DISK,
        snapshot_storage_path=str(checkpoint_directory),
        num_snapshots=1,
    )

    injection_length = config.box_size / 2.0
    crossing_time = injection_length / vrms_target
    end_time = args.tcross * crossing_time

    params = SimulationParams(
        C_cfl=args.cfl,
        gamma=args.gamma,
        isothermal_sound_speed=sound_speed,
        t_end=end_time,
        turbulence_driving_params=TurbulenceDrivingParams(
            velocity=vrms_target,
            k_driving=args.kdrive,
            solenoidal_weight=1.0,
            amplitude_factor=args.amplitude_factor,
        ),
        minimum_density=args.rhomin,
        minimum_pressure=1e-6,
        maximum_alfven_speed=args.alfven_ceiling if args.alfven_ceiling > 0.0 else float("inf"),
    )

    registered_variables = get_registered_variables(config)
    # The state is built here only for its shape; the run builds it again when
    # it starts from t = 0, so no copy of it stays on the GPU for the whole run.
    initial_state = build_initial_state(
        args,
        config,
        registered_variables,
        magnetic_field,
        initial_pressure,
    )
    config = finalize_config(config, initial_state.shape)
    del initial_state

    print(f"[{args.tag}] eos={args.eos} M_turb_aim={args.mturb} beta={args.beta:g} "
          f"a={sound_speed:.3f} P0={initial_pressure} B0={magnetic_field:.4g} "
          f"kdrive={args.kdrive} amplitude_factor={args.amplitude_factor} cfl={args.cfl} rhomin={args.rhomin} "
          f"weno_z={bool(args.weno_z)} positivity_preserving={positivity_preserving} "
          f"t_end={end_time:.4f} ({args.tcross} t_cross) intervals={args.nsnap}", flush=True)

    return TurbulenceSetup(
        config=config,
        params=params,
        registered_variables=registered_variables,
        initial_pressure=initial_pressure,
        sound_speed=sound_speed,
        magnetic_field=magnetic_field,
        crossing_time=crossing_time,
        end_time=end_time,
        positivity_preserving=positivity_preserving,
        checkpoint_directory=checkpoint_directory,
    )


def run_signature(args, setup):
    """The parameters a resumed run must share with the checkpoints it finds (JSON)."""
    # ``floor`` only enters when it is on, so runs made before the option
    # existed still resume.
    floor_entry = dict(floor=args.floor) if args.floor else {}
    if args.alfven_ceiling > 0.0:
        floor_entry["alfven_ceiling"] = args.alfven_ceiling
    # Likewise the upwind CT dissipation, which changes the scheme.
    if args.ct_upwind:
        floor_entry["ct_upwind"] = args.ct_upwind
    return json.dumps(
        dict(
            **floor_entry,
            mturb=args.mturb,
            beta=args.beta,
            eos=args.eos,
            gamma=args.gamma,
            N=args.N,
            tcross=args.tcross,
            driving="TurbGen",
            kdrive=args.kdrive,
            amplitude_factor=args.amplitude_factor,
            nsnap=args.nsnap,
            cfl=args.cfl,
            rhomin=args.rhomin,
            norm=args.norm,
            mhd=args.mhd,
            weno_z=bool(args.weno_z),
            positivity_preserving=setup.positivity_preserving,
        ),
        sort_keys=True,
    )
# -------------------------------------------------------------
# ================== ↑ Problem setup ↑ ========================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============ ↓ In-flight snapshot reduction ↓ ===============
# -------------------------------------------------------------
def build_snapshot_reducer(args, setup):
    """
    Build the jitted reduction of one (unpadded) state to the per-snapshot
    diagnostics: the scalars of ``SCALAR_NAMES``, the spectra of
    ``SPECTRUM_NAMES`` and the mid-plane slices of ``SLICE_NAMES``. Only these
    small arrays leave the GPU.

    The definitions are those of the earlier post-hoc analysis of stored
    snapshots: volume means of the NaN-cleaned fields, and the spectra of the
    NaN-cleaned density fluctuation, sqrt(rho)-weighted velocity and magnetic
    field, shell-summed on the physical-wavenumber bins.

    Args:
        args: The parsed command-line arguments.
        setup: The :class:`TurbulenceSetup`.

    Returns:
        ``reduce(primitive_state) -> (scalars, spectra, slices, wavenumbers)``.
    """
    registered_variables = setup.registered_variables
    density_index = registered_variables.density_index
    velocity_indices = (
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
    )
    if args.mhd:
        magnetic_indices = (
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        )
    adiabatic = args.eos == "adiabatic"
    rho0 = 1.0
    mid_plane = args.N // 2

    def reduce_snapshot(primitive_state):
        density = primitive_state[density_index]
        velocity_x, velocity_y, velocity_z = (primitive_state[i] for i in velocity_indices)
        if args.mhd:
            magnetic_x, magnetic_y, magnetic_z = (primitive_state[i] for i in magnetic_indices)
        else:
            magnetic_x = magnetic_y = magnetic_z = jnp.zeros_like(density)

        velocity_squared = velocity_x ** 2 + velocity_y ** 2 + velocity_z ** 2
        kinetic_energy_density = 0.5 * density * velocity_squared
        magnetic_energy_density = 0.5 * (magnetic_x ** 2 + magnetic_y ** 2 + magnetic_z ** 2)

        if adiabatic:
            positive_density = jnp.maximum(density, 1e-12)
            pressure = jnp.maximum(primitive_state[registered_variables.pressure_index], 1e-12)
            sound_speed = jnp.mean(jnp.sqrt(args.gamma * pressure / positive_density))
        else:
            sound_speed = jnp.asarray(setup.sound_speed, dtype=density.dtype)

        scalars = jnp.stack([
            jnp.sqrt(jnp.mean(jnp.nan_to_num(velocity_squared))),
            jnp.sqrt(jnp.mean(jnp.nan_to_num((density - rho0) ** 2))),
            jnp.mean(jnp.nan_to_num(kinetic_energy_density)),
            jnp.mean(jnp.nan_to_num(magnetic_energy_density)),
            jnp.min(density),
            jnp.max(density),
            jnp.sqrt(jnp.max(velocity_squared)),
            sound_speed,
            jnp.any(~jnp.isfinite(primitive_state)).astype(density.dtype),
        ])

        # Spectra of the NaN-cleaned fields, so a failing snapshot still gives
        # finite arrays rather than poisoning a time average.
        clean_density = jnp.nan_to_num(density)
        clean_velocity = [jnp.nan_to_num(v) for v in (velocity_x, velocity_y, velocity_z)]
        clean_magnetic = [jnp.nan_to_num(b) for b in (magnetic_x, magnetic_y, magnetic_z)]
        wavenumbers, density_spectrum = vector_field_energy_spectrum(
            clean_density - jnp.mean(clean_density),
            jnp.zeros_like(clean_density),
            jnp.zeros_like(clean_density),
        )
        _, kinetic_spectrum = get_kinetic_energy_spectrum(
            *clean_velocity,
            jnp.maximum(clean_density, 0.0),
        )
        _, magnetic_spectrum = get_magnetic_energy_spectrum(*clean_magnetic)
        spectra = jnp.stack([density_spectrum, kinetic_spectrum, magnetic_spectrum])

        slices = jnp.stack([
            kinetic_energy_density[:, :, mid_plane],
            magnetic_energy_density[:, :, mid_plane],
            density[:, :, mid_plane],
        ])
        return scalars, spectra, slices, wavenumbers

    return jax.jit(reduce_snapshot)


class DiagnosticsLog:
    """
    The per-snapshot diagnostics accumulated over a run, saved next to the
    checkpoints after every interval so a resumed job continues the record.

    Attributes:
        signature: The run parameters (JSON) the record belongs to.
        times: The snapshot times.
        scalars: Per snapshot, the scalars of ``SCALAR_NAMES``.
        spectra: Per snapshot, the spectra of ``SPECTRUM_NAMES``.
        wavenumbers: The physical-wavenumber bin centres of the spectra.
        slices: The mid-plane slices of the last alive snapshot.
        slice_index: The snapshot the slices were taken from (-1: none yet).
        interval_steps: Per finished interval, the number of time steps.
        interval_wall: Per finished interval, the wall time [s] of the
            integration (including compilation in a job's first interval and
            the checkpoint write), the checkpoint read and the reduction.
        jobs: The number of jobs (invocations) that advanced the run.
    """

    def __init__(self, signature):
        self.signature = signature
        self.times = []
        self.scalars = []
        self.spectra = []
        self.wavenumbers = None
        self.slices = None
        self.slice_index = -1
        self.interval_steps = []
        self.interval_wall = []
        self.jobs = 0
        self.branched_from = ""

    @property
    def num_snapshots(self):
        """The number of snapshots recorded so far (t = 0 included)."""
        return len(self.times)

    @property
    def first_nonfinite(self):
        """The first snapshot holding a NaN or infinite value, or -1."""
        flags = np.asarray([row[SCALAR_NAMES.index("nonfinite_t")] for row in self.scalars])
        bad = np.flatnonzero(flags > 0)
        return int(bad[0]) if len(bad) else -1

    def append(self, time, scalars, spectra, slices, wavenumbers):
        """Record one reduced snapshot; keep its slices if it is alive."""
        self.times.append(float(time))
        self.scalars.append(np.asarray(scalars, dtype=np.float64))
        self.spectra.append(np.asarray(spectra))
        self.wavenumbers = np.asarray(wavenumbers)
        vrms = float(scalars[SCALAR_NAMES.index("vrms_t")])
        if vrms > ALIVE_VRMS:
            self.slices = np.asarray(slices)
            self.slice_index = self.num_snapshots - 1

    def save(self, directory):
        """Write the record atomically (a killed job never leaves half a file)."""
        path = Path(directory) / DIAGNOSTICS_FILE
        temporary_path = Path(directory) / f"partial_{DIAGNOSTICS_FILE}"
        np.savez(
            temporary_path,
            signature=self.signature,
            times=np.asarray(self.times),
            scalars=np.asarray(self.scalars),
            spectra=np.asarray(self.spectra),
            wavenumbers=self.wavenumbers,
            slices=self.slices if self.slices is not None else np.zeros((0,)),
            slice_index=self.slice_index,
            interval_steps=np.asarray(self.interval_steps, dtype=np.int64),
            interval_wall=np.asarray(self.interval_wall, dtype=np.float64),
            jobs=self.jobs,
            branched_from=self.branched_from,
        )
        os.replace(temporary_path, path)

    @classmethod
    def load(cls, directory):
        """Read a saved record, or return ``None`` if there is none."""
        path = Path(directory) / DIAGNOSTICS_FILE
        if not path.exists():
            return None
        stored = np.load(path)
        log = cls(str(stored["signature"]))
        log.times = [float(t) for t in stored["times"]]
        log.scalars = list(stored["scalars"])
        log.spectra = list(stored["spectra"])
        log.wavenumbers = stored["wavenumbers"]
        log.slice_index = int(stored["slice_index"])
        log.slices = stored["slices"] if log.slice_index >= 0 else None
        log.interval_steps = [int(n) for n in stored["interval_steps"]]
        log.interval_wall = [float(w) for w in stored["interval_wall"]]
        log.jobs = int(stored["jobs"])
        log.branched_from = str(stored["branched_from"]) if "branched_from" in stored else ""
        return log
# -------------------------------------------------------------
# ============ ↑ In-flight snapshot reduction ↑ ===============
# -------------------------------------------------------------


def print_snapshot_line(args, setup, log):
    """One progress line for the latest snapshot of ``log``."""
    named = dict(zip(SCALAR_NAMES, log.scalars[-1]))
    snapshot = log.num_snapshots - 1
    line = (f"[{args.tag}] snap {snapshot:3d}/{args.nsnap} "
            f"t/tc={log.times[-1] / setup.crossing_time:6.3f} "
            f"M={named['vrms_t'] / named['sound_t']:.3f} "
            f"M_mass={np.sqrt(2.0 * named['EK_t']) / named['sound_t']:.3f} v_rms={named['vrms_t']:.4f} "
            f"E_K={named['EK_t']:.4e} E_B={named['EB_t']:.4e} "
            f"rho=[{named['rho_min_t']:.3e}, {named['rho_max_t']:.3e}] "
            f"max|v|={named['vmax_t']:.3f} nonfinite={int(named['nonfinite_t'])}")
    if snapshot > 0 and len(log.interval_steps) == snapshot:
        line += f" steps={log.interval_steps[-1]} wall={log.interval_wall[-1]:.1f}s"
    print(line, flush=True)


def prune_checkpoints(directory, keep):
    """Delete all but the ``keep`` (at least one) most recent interval checkpoints."""
    steps = sorted(
        int(child.name)
        for child in Path(directory).iterdir()
        if child.is_dir() and child.name.isdigit()
    )
    for step in steps[:-max(1, keep)]:
        shutil.rmtree(Path(directory) / str(step), ignore_errors=True)


def branch_from_run(args, signature, checkpoint_directory):
    """
    Seed this run with snapshot STEP of another run (``--branch-from TAG:STEP``):
    its checkpoint STEP and its diagnostics up to that snapshot, re-signed with
    this run's parameters. Only the floor settings (``rhomin``, ``floor``,
    ``alfven_ceiling``) may differ, so the branch continues the same flow with
    a changed floor.

    Args:
        args: The parsed command-line arguments.
        signature: This run's signature (JSON).
        checkpoint_directory: This run's (empty) checkpoint directory.
    """
    source_tag, step = args.branch_from.rsplit(":", 1)
    step = int(step)
    source_directory = Path(args.ckpt_root) / source_tag
    source_log = DiagnosticsLog.load(source_directory)
    if source_log is None or not (source_directory / str(step)).is_dir():
        raise SystemExit(f"[{args.tag}] --branch-from: no checkpoint {step} with diagnostics "
                         f"in {source_directory}")
    floor_keys = ("rhomin", "floor", "alfven_ceiling")
    source_parameters = {k: v for k, v in json.loads(source_log.signature).items() if k not in floor_keys}
    parameters = {k: v for k, v in json.loads(signature).items() if k not in floor_keys}
    if source_parameters != parameters:
        raise SystemExit(f"[{args.tag}] --branch-from: {source_tag} differs in more than the floor")
    shutil.copytree(source_directory / str(step), checkpoint_directory / str(step))
    branched = DiagnosticsLog(signature)
    branched.times = source_log.times[:step + 1]
    branched.scalars = source_log.scalars[:step + 1]
    branched.spectra = source_log.spectra[:step + 1]
    branched.wavenumbers = source_log.wavenumbers
    branched.interval_steps = source_log.interval_steps[:step]
    branched.interval_wall = source_log.interval_wall[:step]
    branched.jobs = source_log.jobs
    branched.branched_from = args.branch_from
    # The kept slice must come from a snapshot this branch still holds.
    if 0 <= source_log.slice_index <= step:
        branched.slices = source_log.slices
        branched.slice_index = source_log.slice_index
    branched.save(checkpoint_directory)
    print(f"[{args.tag}] branched from {source_tag} at snapshot {step}", flush=True)


# -------------------------------------------------------------
# ============ ↓ Segmented, restartable run ↓ =================
# -------------------------------------------------------------
def run_segmented(args, setup):
    """
    Integrate the run interval by interval, reducing each interval's final
    state on the GPU and checkpointing after every interval; resume from the
    checkpoints when they exist.

    Every interval after the first continues from the Orbax checkpoint the
    previous interval wrote (state, driving state), and its start and
    end times come from the fixed interval grid, not from the stored clock. An
    uninterrupted run and one resumed in a new job therefore execute identical
    integrations.

    Args:
        args: The parsed command-line arguments.
        setup: The :class:`TurbulenceSetup`.

    Returns:
        The :class:`DiagnosticsLog`, complete when every interval is done.
    """
    job_start = walltime.time()
    checkpoint_directory = setup.checkpoint_directory
    if args.fresh and checkpoint_directory.exists():
        shutil.rmtree(checkpoint_directory)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)

    signature = run_signature(args, setup)
    reduce_snapshot = build_snapshot_reducer(args, setup)
    interval_times = [setup.end_time * i / args.nsnap for i in range(args.nsnap + 1)]

    # -------------------------------------------------------------
    # ================ ↓ Fresh start or resume ↓ ==================
    # -------------------------------------------------------------
    if args.branch_from and latest_step(checkpoint_directory) is None:
        branch_from_run(args, signature, checkpoint_directory)
    completed_intervals = latest_step(checkpoint_directory) or 0
    if completed_intervals == 0:
        log = DiagnosticsLog(signature)
        state = build_initial_state(
            args,
            setup.config,
            setup.registered_variables,
            setup.magnetic_field,
            setup.initial_pressure,
        )
        driving_state = None
        log.append(0.0, *jax.device_get(reduce_snapshot(state)))
        print_snapshot_line(args, setup, log)
    else:
        log = DiagnosticsLog.load(checkpoint_directory)
        if log is None or log.signature != signature:
            raise SystemExit(
                f"[{args.tag}] the checkpoints in {checkpoint_directory} belong to a "
                "different run (other parameters, or no diagnostics); use another "
                "--tag or --fresh"
            )
        checkpoint = load_loop_checkpoint(checkpoint_directory, completed_intervals)
        state, driving_state = checkpoint.primitive_state, checkpoint.driving_state
        # The diagnostics are saved right after each checkpoint, so a job
        # killed in between leaves them exactly one snapshot behind.
        if log.num_snapshots == completed_intervals:
            log.append(checkpoint.time, *jax.device_get(reduce_snapshot(state)))
            log.interval_steps.append(int(checkpoint.num_iterations))
            log.interval_wall.append(float("nan"))
            log.save(checkpoint_directory)
        if log.num_snapshots != completed_intervals + 1:
            raise SystemExit(
                f"[{args.tag}] the diagnostics hold {log.num_snapshots} snapshots but "
                f"{completed_intervals} intervals are checkpointed"
            )
        del checkpoint
        print(f"[{args.tag}] RESUME after interval {completed_intervals}/{args.nsnap} "
              f"at t/tc = {interval_times[completed_intervals] / setup.crossing_time:.3f}",
              flush=True)
    # -------------------------------------------------------------
    # ================ ↑ Fresh start or resume ↑ ==================
    # -------------------------------------------------------------

    if log.first_nonfinite >= 0:
        print(f"[{args.tag}] the run went non-finite at snapshot {log.first_nonfinite}; "
              "not continuing it", flush=True)
        return log

    # -------------------------------------------------------------
    # ================== ↓ Interval loop ↓ ========================
    # -------------------------------------------------------------
    wall_this_job = []
    for interval in range(completed_intervals, args.nsnap):
        # Stop cleanly before the walltime: an interval is only started when
        # the slowest one of this job (the first includes the compilation)
        # would still end within the budget.
        if args.stop_after and len(wall_this_job) >= args.stop_after:
            break
        if args.stop_after_hours > 0.0 and wall_this_job:
            elapsed = walltime.time() - job_start
            if elapsed + 1.2 * max(wall_this_job) > 3600.0 * args.stop_after_hours:
                break
        if not wall_this_job:
            log.jobs += 1

        interval_start = walltime.time()
        interval_params = setup.params._replace(
            t_start=interval_times[interval],
            t_end=interval_times[interval + 1],
        )
        # The first interval of a fresh run seeds the driving from its random
        # seed; every later one continues the stored driving state.
        if driving_state is None:
            restart_state = None
        else:
            restart_state = LoopState(primitive_state=None, driving_state=driving_state)
        final_state = time_integration(
            state,
            setup.config,
            interval_params,
            setup.registered_variables,
            restart_state=restart_state,
        )
        del final_state, state

        # Continue from what was written to disk rather than from the returned
        # array: the two are bit-identical, and this makes the in-job path the
        # very path a resumed job takes.
        checkpoint = load_loop_checkpoint(checkpoint_directory, interval + 1)
        state, driving_state = checkpoint.primitive_state, checkpoint.driving_state
        log.append(checkpoint.time, *jax.device_get(reduce_snapshot(state)))
        log.interval_steps.append(int(checkpoint.num_iterations))
        del checkpoint
        wall_this_job.append(walltime.time() - interval_start)
        log.interval_wall.append(wall_this_job[-1])
        log.save(checkpoint_directory)
        prune_checkpoints(checkpoint_directory, args.keep_checkpoints)
        print_snapshot_line(args, setup, log)

        if log.first_nonfinite >= 0:
            print(f"[{args.tag}] NON-FINITE state at snapshot {log.first_nonfinite} "
                  f"(t/tc = {log.times[-1] / setup.crossing_time:.3f}); stopping", flush=True)
            break
    # -------------------------------------------------------------
    # ================== ↑ Interval loop ↑ ========================
    # -------------------------------------------------------------

    return log
# -------------------------------------------------------------
# ============ ↑ Segmented, restartable run ↑ =================
# -------------------------------------------------------------


def write_paper_data(args, setup, log):
    """
    Write ``paper_<tag>.npz`` (the input of make_fig14.py / make_fig15.py) and
    print the stationary-phase summary (the last third of the snapshots).

    Args:
        args: The parsed command-line arguments.
        setup: The :class:`TurbulenceSetup`.
        log: The complete (or failed) :class:`DiagnosticsLog`.
    """
    scalars = np.asarray(log.scalars)
    named = {name: scalars[:, i] for i, name in enumerate(SCALAR_NAMES)}
    spectra = np.asarray(log.spectra)
    time_points = np.asarray(log.times)
    t_over_tc = time_points / setup.crossing_time
    mach = named["vrms_t"] / named["sound_t"]
    # The mass-weighted Mach number, sqrt(2 E_K / <rho>) / a with <rho> = rho0
    # (the mass is conserved up to the floor's additions, which stay far below
    # 1e-4): unlike the volume-weighted v_rms it is not dominated by the fast,
    # near-vacuum cells of the supersonic runs.
    mach_mass = np.sqrt(2.0 * named["EK_t"]) / named["sound_t"]
    first_bad = log.first_nonfinite
    if log.slices is not None:
        slices = log.slices
    else:
        slices = np.zeros((len(SLICE_NAMES), args.N, args.N), dtype=np.float32)
    total_steps = int(np.sum(log.interval_steps))
    wall_time = float(np.nansum(log.interval_wall))

    stationary = max(1, log.num_snapshots // 3)
    print(f"[{args.tag}] stationary (last third): M_turb={np.mean(mach[-stationary:]):.3f} "
          f"M_turb(mass-weighted)={np.mean(mach_mass[-stationary:]):.3f} "
          f"v_rms={np.mean(named['vrms_t'][-stationary:]):.4f} "
          f"(rho-rho0)_rms={np.mean(named['drho_t'][-stationary:]):.4f} "
          f"E_K={np.mean(named['EK_t'][-stationary:]):.4e} "
          f"E_B={np.mean(named['EB_t'][-stationary:]):.4e} "
          f"E_B/E_B0={np.mean(named['EB_t'][-stationary:]) / (0.5 * setup.magnetic_field ** 2):.4g} "
          f"rho_max/rho_min={np.mean(named['rho_max_t'][-stationary:] / named['rho_min_t'][-stationary:]):.4g} "
          f"rho_min(run)={np.min(named['rho_min_t']):.3e} "
          f"rho_max(run)={np.max(named['rho_max_t']):.3e} "
          f"first_bad_snap={first_bad}", flush=True)
    print(f"[{args.tag}] M_turb(t/tc): "
          + " ".join(f"{tc:.2f}:{m:.2f}" for tc, m in zip(t_over_tc, mach)), flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    out = Path(args.outdir) / f"paper_{args.tag}.npz"
    np.savez(
        out,
        tag=args.tag,
        eos=args.eos,
        norm=args.norm,
        mturb_aim=args.mturb,
        beta=args.beta,
        gamma=args.gamma,
        N=args.N,
        a=setup.sound_speed,
        B0=setup.magnetic_field,
        kdrive=args.kdrive,
        amplitude_factor=args.amplitude_factor,
        tcross=args.tcross,
        t_cross=setup.crossing_time,
        nsnap=args.nsnap,
        rhomin=args.rhomin,
        time_points=time_points,
        t_over_tc=t_over_tc,
        Ms_t=mach,
        Ms_mass_t=mach_mass,
        alfven_ceiling=args.alfven_ceiling,
        **named,
        k=log.wavenumbers,
        **{name: spectra[:, i] for i, name in enumerate(SPECTRUM_NAMES)},
        **{name: slices[i] for i, name in enumerate(SLICE_NAMES)},
        slice_index=log.slice_index,
        EB0=float(0.5 * setup.magnetic_field ** 2),
        first_bad=first_bad,
        weno_z=bool(args.weno_z),
        positivity_preserving=setup.positivity_preserving,
        cfl=args.cfl,
        wall_time=wall_time,
        interval_wall=np.asarray(log.interval_wall),
        interval_steps=np.asarray(log.interval_steps),
        num_steps=total_steps,
        jobs=log.jobs,
        floor=bool(args.floor),
        branched_from=log.branched_from,
    )
    print(f"[{args.tag}] wrote {out} (wall {wall_time:.1f} s over {log.jobs} job(s), "
          f"{total_steps} steps)", flush=True)


# -------------------------------------------------------------
# ============== ↓ Blow-up diagnosis mode ↓ ===================
# -------------------------------------------------------------
def run_blowup_diagnosis(args, setup):
    """
    Integrate the whole run in one call with per-snapshot scalar diagnostics
    through the snapshot callback (no states stored, no checkpoints), printing
    one line per snapshot, to pinpoint when and how a run fails.

    Args:
        args: The parsed command-line arguments.
        setup: The :class:`TurbulenceSetup`.
    """
    registered_variables = setup.registered_variables
    density_index = registered_variables.density_index
    velocity_indices = (
        registered_variables.velocity_index.x,
        registered_variables.velocity_index.y,
        registered_variables.velocity_index.z,
    )
    if args.mhd:
        magnetic_indices = (
            registered_variables.magnetic_index.x,
            registered_variables.magnetic_index.y,
            registered_variables.magnetic_index.z,
        )
    diagnosis_config = setup.config._replace(
        snapshot_storage_mode=ON_DEVICE,
        snapshot_storage_path=None,
        return_snapshots=False,
        activate_snapshot_callback=True,
        num_snapshots=args.nsnap,
        progress_bar=True,
    )
    records = []

    def record_on_host(stats):
        time, rho_min, v_max, b_max, b2_over_rho_max, nonfinite = (float(x) for x in np.asarray(stats))
        records.append((time, rho_min, v_max, b_max, b2_over_rho_max, nonfinite))
        print(f"[diag {args.tag}] t={time:.4f} t/tc={time / setup.crossing_time:.3f} "
              f"min_rho={rho_min:.3e} max|v|={v_max:.3f} max|B|={b_max:.3f} "
              f"max(b2/rho)={b2_over_rho_max:.3e} NaN={int(nonfinite)}", flush=True)

    def snapshot_callback(time, state, registered_variables):
        density = state[density_index]
        speed = jnp.sqrt(sum(state[i] ** 2 for i in velocity_indices))
        if args.mhd:
            magnetic_squared = sum(state[i] ** 2 for i in magnetic_indices)
        else:
            magnetic_squared = jnp.zeros_like(density)
        stats = jnp.stack([
            time,
            jnp.min(density),
            jnp.max(speed),
            jnp.sqrt(jnp.max(magnetic_squared)),
            jnp.max(magnetic_squared / jnp.maximum(density, 1e-30)),
            jnp.any(~jnp.isfinite(state)).astype(jnp.float32),
        ])
        jax.debug.callback(record_on_host, stats)

    start = walltime.time()
    final_state = time_integration(
        build_initial_state(
            args,
            setup.config,
            registered_variables,
            setup.magnetic_field,
            setup.initial_pressure,
        ),
        diagnosis_config,
        setup.params,
        registered_variables,
        snapshot_callback,
    )
    jax.block_until_ready(final_state)
    # Every queued host callback has landed before the table is written.
    jax.effects_barrier()
    print(f"[diag {args.tag}] wall {walltime.time() - start:.1f}s")

    table = np.array(sorted(records), dtype=float).reshape(-1, 6)
    os.makedirs(args.outdir, exist_ok=True)
    np.savetxt(
        os.path.join(args.outdir, f"diag_{args.tag}.txt"),
        table,
        header="t min_rho max_v max_B max_b2_over_rho nan",
    )
    nonfinite_rows = table[table[:, 5] > 0]
    if len(nonfinite_rows):
        first_time = nonfinite_rows[0, 0]
        print(f"[diag {args.tag}] FIRST NaN at t={first_time:.4f} "
              f"(t/tc={first_time / setup.crossing_time:.3f})")
    else:
        print(f"[diag {args.tag}] no NaN -- all finite over the run")
# -------------------------------------------------------------
# ============== ↑ Blow-up diagnosis mode ↑ ===================
# -------------------------------------------------------------


def main():
    """Run (or resume) one driven-turbulence case and write its reduced data."""
    args = parse_arguments()
    setup = build_setup(args)

    if args.diag:
        run_blowup_diagnosis(args, setup)
        return

    log = run_segmented(args, setup)
    finished = log.num_snapshots == args.nsnap + 1
    failed = log.first_nonfinite >= 0
    if finished or failed:
        write_paper_data(args, setup, log)
    if failed:
        sys.exit(3)
    if finished:
        print(f"[{args.tag}] DONE", flush=True)
    else:
        print(f"[{args.tag}] PAUSED after interval {log.num_snapshots - 1}/{args.nsnap}; "
              "run the same command again to continue", flush=True)


if __name__ == "__main__":
    main()
