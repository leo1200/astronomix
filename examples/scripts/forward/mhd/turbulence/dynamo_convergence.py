"""Driven subsonic MHD turbulence in astronomix: the small-scale dynamo, with
resolution, in the finite-difference and the finite-volume scheme.

This script drives the ICM case of ``paper_turbulence.py`` (isothermal,
``M_turb ~ 0.5``, initial ``beta_p = 1e6``, i.e. a seed field so weak that
everything the magnetic field does afterwards is dynamo) with either of
astronomix's two MHD schemes:

    --scheme fd   5th-order finite-difference WENO (WENO-Z weights by default,
                  Jiang-Shu with --weno-js), characteristic-wise flux
                  splitting, SSP-RK4, constrained transport, isothermal EOS,
                  C_cfl = 1.5 (0.75 with --pp, the positivity-preserving WENO)
    --scheme vl2  2nd-order finite-volume VL2 + PLM + HLLD with GLM divergence
                  cleaning (AthenaPK's scheme), C_cfl = 0.3; the isothermal box
                  is emulated with gamma = 1.0001 and p0 = rho0 a^2 / gamma, as
                  AthenaPK does it (``athenapk_turb.py``)

Both run on the same grid, from the same initial state, with the same driving
(the TurbGen driving with the same seed: its sequence of patterns follows the
time, not the steps, so the two schemes are driven by the same acceleration),
and both are reduced in flight through the same
estimator as the AthenaPK runs of ``athenapk_turb.py``
(``_mhd_spectral.snapshot_spectra``), so the numerical viscosity and
resistivity of the two schemes are measured by one definition.

Physical setup: unit periodic box, ``rho0 = 1``, isothermal sound speed
``a = 1 / M_turb = 2``, ``B0 = sqrt(2 a^2 rho0 / beta)``, solenoidal TurbGen
driving to the target dispersion ``v_rms = 1`` (amplitude adjusted
automatically), run for 40 crossing times ``t_cross = L/2 / v_rms``. The
earlier results of this study were made with the earlier forcing module (tag
``pre-streamline``); the script runs the TurbGen driving now.

Snapshots are reduced *in flight* (spectra and scalars computed on the GPU, only
the reduced arrays crossing to the host), so the memory cost is independent of
the number of snapshots. The scalars (energies, Mach numbers, signal speed) are
recorded ``--history-every`` times more often than the spectra, which gives the
growth-rate fit a densely sampled ``E_B(t)`` -- the counterpart of AthenaPK's
``.hst`` history -- at no measurable cost.

    PYTHONPATH=$(git rev-parse --show-toplevel) \\
        python examples/scripts/forward/mhd/turbulence/dynamo_convergence.py \\
        --scheme vl2 --n 128 --seed-field sin --transfer --x64
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
import sys
import time as walltime
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import (
    FINITE_VOLUME,
    HLLD,
    NATIVE_JAX,
    PALLAS,
    PERIODIC_BOUNDARY,
    VL2,
)
from astronomix.option_classes.simulation_config import (
    CARTESIAN,
    FINITE_DIFFERENCE,
    IDEAL_GAS,
    ISOTHERMAL,
    KINEMATIC_VISCOSITY,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    DiffusionConfig,
    FDSolverConfig,
    FVSolverConfig,
    PositivityConfig,
    SimulationConfig,
    SimulationParams,
)
from astronomix._modules._turbulence_driving import TurbulenceDrivingConfig, TurbulenceDrivingParams

# astronomix functions
from astronomix._spatial_operators._differencing import _interface_field_divergence
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_registered_variables,
    initialize_interface_fields,
    time_integration,
)

# the shared estimator of the study
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mhd_spectral import (
    GLM_TRANSFER_NAME,
    SCALAR_NAMES,
    SPECTRUM_NAMES,
    glm_cleaning_transfer,
    shell_numbers,
    snapshot_spectra,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Shared physical setup -- athenapk_turb.py must stay in sync with these.
BOX_SIZE = 1.0          # unit periodic box
RHO0 = 1.0              # uniform initial density
L_INJ = 0.5             # injection scale, used for t_cross = L_INJ / v_rms
VRMS_TARGET = 1.0       # the normalisation drives v_rms ~ 1, so a = 1 / M_turb

#: Default C_cfl per scheme: astronomix's FD number (sum-of-speeds convention,
#: an effective 0.5 per axis), the bound under which the positivity-preserving
#: WENO is provable with SSP-RK4, and AthenaPK's VL2 number in 3D.
DEFAULT_CFL = {"fd": 1.5, "fd_pp": 0.75, "vl2": 0.3}

#: Fraction of the scalar-snapshot spacing added to the snapshot time before it
#: is converted to a snapshot number. A snapshot is taken at the first step at or
#: after its nominal time, so its time lies in [k spacing, k spacing + dt); the
#: margin keeps the floor on k when round-off puts the time a hair below it.
SNAPSHOT_NUMBER_MARGIN = 1e-3


def scheme_description(args):
    """``(label, scheme, scheme_key)`` of a run, as stored in its result file.

    The label names the discretisation as the figures and tables print it; the
    key is the short identifier the analysis scripts index runs by.

    Args:
        args: The parsed command line.

    Returns:
        The label, the scheme string and the scheme key.
    """
    if args.scheme == "vl2":
        return "astronomix FV VL2+GLM", "PLM / VL2 / HLLD / GLM", "vl2"
    weights = "WENO5" if args.weno_js else "WENO-Z"
    weights_key = "weno5" if args.weno_js else "wenoz"
    # The upwind EMF dissipation (FDSolverConfig.ct_upwind_dissipation)
    # changes the CT, so it gets its own label and key.
    if args.ct_upwind:
        strength = "" if args.ct_upwind == 1.0 else f" x{args.ct_upwind:g}"
        ct_label = f"CT+upwind EMF{strength}"
        ct_scheme = f"CT (upwind EMF{strength})"
        ct_key = "_uct" if args.ct_upwind == 1.0 else f"_uct{args.ct_upwind:g}"
    else:
        ct_label, ct_scheme, ct_key = "CT", "CT", ""
    if args.pp:
        return (f"astronomix FD {weights}+{ct_label}+PP",
                f"{weights} (positivity preserving) / SSP-RK4 / {ct_scheme}",
                f"{weights_key}_pp{ct_key}")
    return f"astronomix FD {weights}+{ct_label}", f"{weights} / SSP-RK4 / {ct_scheme}", f"{weights_key}{ct_key}"


# -------------------------------------------------------------
# ============ ↓ Solver configuration ↓ =======================
# -------------------------------------------------------------
def build_config(args, num_snapshots):
    """The simulation configuration of the requested scheme.

    Args:
        args: The parsed command line.
        num_snapshots: The number of (scalar) snapshots over the run.

    Returns:
        The (not yet finalised) simulation configuration.
    """
    periodic = BoundarySettings(
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    )
    # Temporally correlated, like AthenaPK; --forcing athenapk drives the band
    # 1 <= k <= 3 with the parabola peaked at AthenaPK's kpeak = 2 (all lattice
    # modes of the band, not AthenaPK's 30).
    k_band = (1.0, 3.0) if args.forcing == "athenapk" else (args.kdrive - 0.5, args.kdrive + 0.5)
    driving_config = TurbulenceDrivingConfig(
        turbulence_driving=True,
        k_min=k_band[0],
        k_max=k_band[1],
        random_seed=args.seed,
        amplitude_auto_adjust=True,
    )
    backend = PALLAS if args.backend == "pallas" else NATIVE_JAX
    shared = dict(
        dimensionality=3,
        geometry=CARTESIAN,
        mhd=True,
        box_size=BOX_SIZE,
        num_cells=args.n,
        boundary_settings=periodic,
        turbulence_driving_config=driving_config,
        backend_config=BackendConfig(backend=backend),
        # In-flight reduction: no snapshot buffer, so the memory cost does not
        # grow with num_snapshots (see module docstring).
        return_snapshots=False,
        activate_snapshot_callback=True,
        num_snapshots=num_snapshots,
        # Our own per-snapshot progress line replaces the progress bar.
        progress_bar=False,
    )

    if args.scheme == "vl2":
        return SimulationConfig(
            solver_mode=FINITE_VOLUME,
            solver_config=FVSolverConfig(time_integrator=VL2, riemann_solver=HLLD),
            # VL2 has no isothermal EOS; gamma -> 1 emulates it (see params).
            equation_of_state=IDEAL_GAS,
            # Subsonic turbulence never approaches vacuum: no floors, as in
            # AthenaPK's turbulence deck.
            positivity_config=PositivityConfig(),
            **shared,
        )

    return SimulationConfig(
        solver_mode=FINITE_DIFFERENCE,
        equation_of_state=ISOTHERMAL,
        # Subsonic turbulence never approaches vacuum, so the scheme runs
        # without the positivity-preserving WENO by default: it splits every
        # mass-carrying field with the common (fastest) speed, which costs
        # low-Mach accuracy. --pp switches it on to measure that cost.
        positivity_config=PositivityConfig(step_retry=args.step_retry),
        solver_config=FDSolverConfig(
            # WENO-Z weights (Borges et al.) are the baseline; the Jiang-Shu
            # weights with an absolute epsilon are fully nonlinear on O(1)
            # turbulent fields and drop the stencil to low order.
            weno_z=not args.weno_js,
            weno_positivity_preserving=args.pp,
            ct_upwind_dissipation=args.ct_upwind > 0.0,
            ct_upwind_strength=args.ct_upwind if args.ct_upwind > 0.0 else 0.03,
        ),
        diffusion_config=DiffusionConfig(
            # Explicit Laplacian coefficients for the calibration ladder (the same
            # test athenapk_turb.py --mom-diff/--ohm-diff runs on AthenaPK).
            viscosity=args.mom_diff > 0.0,
            viscosity_type=KINEMATIC_VISCOSITY,
            resistivity=args.ohm_diff > 0.0,
        ),
        **shared,
    )


def build_params(args, sound_speed, t_end):
    """The simulation parameters of the requested scheme.

    Args:
        args: The parsed command line.
        sound_speed: The isothermal sound speed ``a``.
        t_end: The end time of the run.

    Returns:
        The simulation parameters.
    """
    # The FD scheme is isothermal and keeps the library's default gamma, which
    # it does not read; the VL2 scheme gets its near-isothermal gamma.
    gamma = args.gamma if args.scheme == "vl2" else SimulationParams().gamma
    return SimulationParams(
        C_cfl=args.cfl,
        isothermal_sound_speed=sound_speed,
        gamma=gamma,
        glm_alpha=args.glm_alpha,
        t_end=t_end,
        turbulence_driving_params=TurbulenceDrivingParams(
            velocity=VRMS_TARGET,
            k_driving=args.kdrive,
            solenoidal_weight=1.0,
            amplitude_factor=args.amplitude_factor,
        ),
        minimum_density=args.rhomin,
        minimum_pressure=1e-6,
        viscosity=args.mom_diff,
        resistivity=args.ohm_diff,
    )


def initial_state(args, config, registered_variables, sound_speed, seed_field_strength):
    """The fluid at rest in the uniform box, threaded by the seed field.

    Args:
        args: The parsed command line.
        config: The simulation configuration.
        registered_variables: The registered variables.
        sound_speed: The isothermal sound speed ``a``.
        seed_field_strength: ``B0``.

    Returns:
        The primitive state.
    """
    # Let the array dtype follow the x64 flag rather than pinning float32, so
    # --x64 really runs in double precision end to end.
    shape = (args.n,) * 3
    zeros = jnp.zeros(shape)
    if args.seed_field == "uniform":
        magnetic_field_x = zeros
        magnetic_field_y = zeros
        magnetic_field_z = jnp.full(shape, seed_field_strength)
    else:
        # B_x = sqrt(2) B0 sin(2 pi z / L): zero net flux, mean energy 0.5 B0^2
        # (the sqrt(2) is what keeps the energy equal to the uniform case), and
        # divergence-free for free since it varies only along z. Cell-centred z.
        z = (jnp.arange(args.n) + 0.5) * (BOX_SIZE / args.n)
        profile = jnp.sqrt(2.0) * seed_field_strength * jnp.sin(2.0 * jnp.pi * z / BOX_SIZE)
        magnetic_field_x = jnp.broadcast_to(profile[None, None, :], shape).astype(zeros.dtype)
        magnetic_field_y = zeros
        magnetic_field_z = zeros

    if args.scheme == "vl2":
        # AthenaPK's isothermal emulation: p0 = rho0 a^2 / gamma, so that the
        # adiabatic sound speed sqrt(gamma p0 / rho0) is exactly a.
        return construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=jnp.full(shape, RHO0),
            velocity_x=zeros,
            velocity_y=zeros,
            velocity_z=zeros,
            gas_pressure=jnp.full(shape, RHO0 * sound_speed ** 2 / args.gamma),
            magnetic_field_x=magnetic_field_x,
            magnetic_field_y=magnetic_field_y,
            magnetic_field_z=magnetic_field_z,
        )

    interface_field_x, interface_field_y, interface_field_z = initialize_interface_fields(
        magnetic_field_x,
        magnetic_field_y,
        magnetic_field_z,
    )
    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=jnp.full(shape, RHO0),
        velocity_x=zeros,
        velocity_y=zeros,
        velocity_z=zeros,
        magnetic_field_x=magnetic_field_x,
        magnetic_field_y=magnetic_field_y,
        magnetic_field_z=magnetic_field_z,
        interface_magnetic_field_x=interface_field_x,
        interface_magnetic_field_y=interface_field_y,
        interface_magnetic_field_z=interface_field_z,
    )
# -------------------------------------------------------------
# ============ ↑ Solver configuration ↑ =======================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ========= ↓ In-flight snapshot reduction (callback) ↓ ========
# -------------------------------------------------------------
def make_snapshot_reducer(args, config, registered_variables, sound_speed,
                          snapshot_spacing, t_cross, records):
    """The snapshot callback: scalars at every snapshot, spectra at every
    ``args.history_every``-th, both reduced on the device.

    Every scheme goes through ``snapshot_spectra``. What differs is only what
    the state holds: the FD scheme stores point values, a staggered field next
    to the cell-centred one and no pressure (isothermal); the VL2 scheme stores
    cell averages, a cell-centred field, the pressure of its near-isothermal
    ideal gas and the GLM scalar psi. Its spectra are therefore stored twice,
    raw and with the cell-average transfer function divided out of the energy
    spectra -- the convention of the AthenaPK runs of the earlier study, whose
    dissipation budget uses the raw rows and whose spectral metrics use the
    deconvolved ones.

    Args:
        args: The parsed command line.
        config: The finalised simulation configuration.
        registered_variables: The registered variables.
        sound_speed: The isothermal sound speed ``a``.
        snapshot_spacing: Time between two scalar snapshots.
        t_cross: The crossing time, for the progress line.
        records: Host-side dict the callback appends to; its ``keep`` entry
            switches recording off for the warm-up call.

    Returns:
        The callback ``(time, state, registered_variables) -> None``.
    """
    is_vl2 = args.scheme == "vl2"
    grid_spacing = BOX_SIZE / args.n
    num_ghost_cells = config.num_ghost_cells
    interior = (slice(num_ghost_cells, -num_ghost_cells) if num_ghost_cells else slice(None),) * 3
    density_index = registered_variables.density_index
    velocity_indices = tuple(registered_variables.velocity_index)
    magnetic_indices = tuple(registered_variables.magnetic_index)
    if is_vl2:
        pressure_index = registered_variables.pressure_index
        psi_index = registered_variables.magnetic_psi_index
    else:
        interface_indices = tuple(registered_variables.interface_magnetic_field_index)

    # Rows per spectrum set: the three energy spectra, the two ideal transfer
    # spectra with --transfer, and for VL2 the GLM cleaning transfer after
    # them. VL2 stores two sets (raw and deconvolved), FD one.
    num_rows = 3 + (2 if args.transfer else 0) + (1 if is_vl2 and args.transfer else 0)
    spectra_shape = (2 if is_vl2 else 1, num_rows, args.n // 2 + 1)

    def reduce_spectra(density, velocity, magnetic_field, pressure, psi):
        """Raw (and for VL2 deconvolved) spectra of one snapshot, stacked."""
        if args.scalars_only:
            return jnp.zeros(spectra_shape, dtype=density.dtype)
        _, raw_spectra = snapshot_spectra(
            density,
            *velocity,
            *magnetic_field,
            sound_speed,
            grid_spacing,
            transfer=args.transfer,
            dealias=args.dealias,
            cell_average_input=args.cell_average,
            pressure=pressure,
            gamma=args.gamma if is_vl2 else None,
        )
        if not is_vl2:
            return raw_spectra[None]

        if args.transfer:
            # The GLM cleaning coupling, so the budget can say how much of the
            # measured numerical resistivity is the divergence cleaning.
            cleaning_transfer = glm_cleaning_transfer(*magnetic_field, psi, grid_spacing)
            raw_spectra = jnp.concatenate([raw_spectra, cleaning_transfer[None]])
        # The cell-average deconvolution applies to the energy spectra only; the
        # transfer rows are carried raw, exactly as athenapk_turb.py stores them.
        _, deconvolved_energy_spectra = snapshot_spectra(
            density,
            *velocity,
            *magnetic_field,
            sound_speed,
            grid_spacing,
            deconvolve_cell_average=True,
        )
        deconvolved_spectra = jnp.concatenate([deconvolved_energy_spectra, raw_spectra[3:]])
        return jnp.stack([raw_spectra, deconvolved_spectra])

    def signal_speed_for_time_step(density, velocity, magnetic_field, squared_sound_speed):
        """The signal speed the scheme's CFL step is set by.

        The FD scheme sets ``dt = C_cfl dx / (lam_x + lam_y + lam_z)``, the VL2
        scheme ``dt = C_cfl dx / max(lam_x, lam_y, lam_z)``, with ``lam_d`` the
        largest ``|v_d| + c_fast,d``. Recording it lets the step count be
        reconstructed on the host without a second run.
        """
        safe_density = jnp.maximum(density, 1e-12)
        squared_field_over_density = sum(component ** 2 for component in magnetic_field) / safe_density
        axis_speeds = []
        for velocity_component, normal_field in zip(velocity, magnetic_field):
            normal_alfven_squared = normal_field ** 2 / safe_density
            fast_sum = squared_sound_speed + squared_field_over_density
            fast_speed = jnp.sqrt(0.5 * (fast_sum + jnp.sqrt(jnp.maximum(
                fast_sum ** 2 - 4.0 * squared_sound_speed * normal_alfven_squared, 0.0))))
            axis_speeds.append(jnp.max(jnp.abs(velocity_component) + fast_speed))
        if is_vl2:
            return jnp.max(jnp.stack(axis_speeds))
        return sum(axis_speeds)

    def record_on_host(time, is_spectral, scalars, spectra, signal_speed,
                       divergence_face, psi_rms, magnetic_energy_slice):
        """Host side of the callback: append the reduced arrays."""
        if not records["keep"]:
            return
        time = float(time)
        scalars = np.asarray(scalars)
        records["history"].append((time, scalars, float(signal_speed), float(psi_rms)))
        if not bool(is_spectral):
            return
        records["spectral"].append(dict(
            time=time,
            scalars=scalars,
            spectra=np.asarray(spectra),
            signal_speed=float(signal_speed),
            divergence_face=float(divergence_face),
            psi_rms=float(psi_rms),
            magnetic_energy_slice=np.asarray(magnetic_energy_slice, dtype=np.float32),
        ))
        named = dict(zip(SCALAR_NAMES, scalars))
        print(f"  t/tc={time / t_cross:6.2f}  M={named['mach']:.3f}  "
              f"E_K={named['E_K']:.4f}  E_B={named['E_B']:.3e}  "
              f"<|B|>={named['mean_absB']:.4e}  "
              f"E_B/E_K={named['E_B'] / max(named['E_K'], 1e-30):.3e}", flush=True)

    def reduce_snapshot(time, state, registered_variables):
        density = state[density_index][interior]
        velocity = tuple(state[index][interior] for index in velocity_indices)
        magnetic_field = tuple(state[index][interior] for index in magnetic_indices)
        if is_vl2:
            pressure = state[pressure_index][interior]
            psi = state[psi_index][interior]
            squared_sound_speed = args.gamma * pressure / jnp.maximum(density, 1e-12)
            psi_rms = jnp.sqrt(jnp.mean(psi ** 2))
        else:
            pressure = None
            psi = None
            squared_sound_speed = sound_speed ** 2
            psi_rms = jnp.zeros((), dtype=density.dtype)

        scalars, _ = snapshot_spectra(
            density,
            *velocity,
            *magnetic_field,
            sound_speed,
            grid_spacing,
            spectra=False,
        )

        # The spectra (and the transfer, which dominates the cost) only every
        # history_every-th snapshot, on the same time grid an unthinned run
        # with --nsnap snapshots would use.
        snapshot_number = jnp.floor(
            (time + SNAPSHOT_NUMBER_MARGIN * snapshot_spacing) / snapshot_spacing
        ).astype(jnp.int32)
        is_spectral = (snapshot_number % args.history_every) == 0
        spectra = jax.lax.cond(
            is_spectral,
            lambda: reduce_spectra(density, velocity, magnetic_field, pressure, psi),
            lambda: jnp.zeros(spectra_shape, dtype=density.dtype),
        )

        # The constraint constrained transport actually preserves is the
        # 6th-order interface divergence of the *staggered* field, not the
        # centred difference of the cell-centred field that the shared
        # `rel_divB` measures. Both are reported for FD, with the same
        # normalisation so they sit on one axis; the VL2 scheme has no
        # staggered field and only the cell-centred measure.
        if is_vl2:
            divergence_face = jnp.full((), jnp.nan, dtype=density.dtype)
        else:
            interface_field = tuple(state[index][interior] for index in interface_indices)
            face_divergence = _interface_field_divergence(*interface_field, grid_spacing)
            field_magnitude = jnp.sqrt(sum(component ** 2 for component in magnetic_field))
            divergence_face = jnp.mean(jnp.where(
                field_magnitude > 0.0,
                0.5 * jnp.sqrt(3.0) * grid_spacing * jnp.abs(face_divergence)
                / jnp.maximum(field_magnitude, 1e-300),
                0.0,
            ))

        # One mid-plane magnetic-energy slice per spectral snapshot, for the
        # animation, carried in the reduced file rather than in raw dumps.
        if args.slice_series:
            z_mid = density.shape[2] // 2
            magnetic_energy_slice = 0.5 * sum(component[:, :, z_mid] ** 2
                                              for component in magnetic_field)
        else:
            magnetic_energy_slice = jnp.zeros((1, 1), dtype=density.dtype)

        jax.debug.callback(
            record_on_host,
            time,
            is_spectral,
            scalars,
            spectra,
            signal_speed_for_time_step(density, velocity, magnetic_field, squared_sound_speed),
            divergence_face,
            psi_rms,
            magnetic_energy_slice,
        )

    return reduce_snapshot
# -------------------------------------------------------------
# ========= ↑ In-flight snapshot reduction (callback) ↑ ========
# -------------------------------------------------------------


def parse_arguments():
    """The command line of the runner."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scheme", choices=("fd", "vl2"), default="fd",
                        help="fd: WENO + constrained transport (finite difference); "
                             "vl2: VL2 + PLM + HLLD + GLM (finite volume)")
    parser.add_argument("--n", type=int, default=128, help="cells per dimension")
    parser.add_argument("--mturb", type=float, default=0.5, help="target turbulent Mach number")
    parser.add_argument("--beta", type=float, default=1e6, help="initial plasma beta")
    parser.add_argument("--seed-field", choices=("uniform", "sin"), default="uniform",
                        help="uniform: B0 along z, the HOW-MHD ICM configuration. Its "
                             "mean is exactly conserved in a periodic box, so the "
                             "early growth is tangling of a mean field rather than a "
                             "dynamo eigenmode -- see the README. sin: "
                             "B_x = sqrt(2) B0 sin(2 pi z / L), zero net flux at the "
                             "same magnetic energy, matching AthenaPK's b_config = 2.")
    parser.add_argument("--tcross", type=float, default=40.0, help="run length in crossing times")
    parser.add_argument("--amplitude-factor", type=float, default=1.0,
                        help="TurbGen amplitude factor at the start (adjusted "
                             "automatically towards the target dispersion)")
    parser.add_argument("--kdrive", type=float, default=None,
                        help="peak of the driving in units of 2 pi / L. Default 1.5 "
                             "(the k^6 exp(-8k/kpk) spectrum of the paper, band "
                             "kdrive -+ 0.5), 2 (AthenaPK's kpeak) for --forcing athenapk")
    parser.add_argument("--forcing", choices=("smooth", "athenapk"), default="smooth",
                        help="smooth: TurbGen parabola over kdrive -+ 0.5. athenapk: "
                             "TurbGen parabola over 1 <= k <= 3, peak at 2")
    parser.add_argument("--weno-js", action="store_true",
                        help="FD only: Jiang-Shu nonlinear weights instead of WENO-Z")
    parser.add_argument("--weno-z", action="store_true",
                        help="FD only: WENO-Z weights, the default (accepted so the "
                             "earlier run scripts still parse)")
    parser.add_argument("--pp", action="store_true",
                        help="FD only: positivity-preserving WENO (Zhang & Shu); "
                             "the default C_cfl becomes 0.75")
    parser.add_argument("--ct-upwind", type=float, nargs="?", const=0.03, default=0.0,
                        help="FD only: upwind dissipation of the CT edge EMFs "
                             "(FDSolverConfig.ct_upwind_dissipation) at strength kappa "
                             "(ct_upwind_strength; the bare flag is the default 0.03, 0 = off)")
    parser.add_argument("--rhomin", type=float, default=1e-4,
                        help="minimum_density, the target of the positivity-preserving WENO; "
                             "supersonic runs (--mturb >~ 3) need it far below their void "
                             "densities, e.g. 1e-10 as in paper_turbulence.py")
    parser.add_argument("--step-retry", action="store_true",
                        help="FD: redo a step at half the step size when it leaves a "
                             "non-positive density (PositivityConfig.step_retry)")
    parser.add_argument("--gamma", type=float, default=1.0001,
                        help="VL2 only: adiabatic index of the near-isothermal gas")
    parser.add_argument("--glm-alpha", type=float, default=0.1,
                        help="VL2 only: GLM damping (AthenaPK glmmhd_alpha)")
    parser.add_argument("--backend", choices=("pallas", "native"), default="pallas",
                        help="compute backend of the solver")
    parser.add_argument("--mom-diff", type=float, default=0.0,
                        help="FD only: explicit kinematic viscosity in code units "
                             "(calibration runs only; 0 disables)")
    parser.add_argument("--ohm-diff", type=float, default=0.0,
                        help="FD only: explicit ohmic resistivity in code units "
                             "(calibration runs only; 0 disables)")
    parser.add_argument("--cfl", type=float, default=None,
                        help="C_cfl; default 1.5 (fd), 0.75 (fd --pp), 0.3 (vl2)")
    parser.add_argument("--nsnap", type=int, default=81,
                        help="spectral snapshots over the whole run. The in-flight "
                             "spectra are not free (see the README's runtime "
                             "section), so timing runs use a small number")
    parser.add_argument("--history-every", type=int, default=5,
                        help="scalar snapshots per spectral snapshot interval: the "
                             "scalar time series (E_B(t) for the growth-rate fit) "
                             "is this many times denser than the spectra")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--x64", action="store_true",
                        help="run in double precision (the earlier FD ladder ran "
                             "in x32)")
    parser.add_argument("--dealias", action="store_true",
                        help="form the transfer spectra on a 3/2-refined grid, so "
                             "the quadratic terms carry no aliases into the kept "
                             "shells (Orszag). Control for the aliasing systematic.")
    parser.add_argument("--cell-average", action="store_true",
                        help="FD only: box-filter the state before reducing it, i.e. "
                             "store what a finite-volume code would have stored. "
                             "Control for the FD-vs-FV representation difference.")
    parser.add_argument("--transfer", action="store_true",
                        help="also record the ideal spectral transfer, which turns "
                             "the spectra into a measurement of the scheme's "
                             "scale-dependent numerical diffusivity (see "
                             "make_dissipation_figure.py). Roughly doubles the "
                             "per-snapshot reduction cost.")
    parser.add_argument("--scalars-only", action="store_true",
                        help="skip the spectra and record only the scalar "
                             "diagnostics (zero-filled spectra)")
    parser.add_argument("--slice-series", action="store_true",
                        help="keep the mid-plane magnetic-energy slice at every "
                             "spectral snapshot, not just the last one")
    parser.add_argument("--save-slices", action="store_true",
                        help="also store mid-plane slices of the last snapshot")
    parser.add_argument("--tag", type=str, default="", help="suffix for the output file name")
    parser.add_argument("--outdir", type=str, default=str(DATA_DIR))
    args = parser.parse_args()

    if args.scheme == "vl2":
        if args.weno_js or args.pp or args.mom_diff > 0.0 or args.ohm_diff > 0.0:
            parser.error("--weno-js, --pp, --mom-diff and --ohm-diff are FD options")
        if args.cell_average:
            parser.error("--cell-average is the FD representation control; the "
                         "VL2 state already holds cell averages")
    if args.weno_js and args.weno_z:
        parser.error("--weno-js and --weno-z exclude each other")
    if args.cfl is None:
        args.cfl = DEFAULT_CFL["vl2" if args.scheme == "vl2" else ("fd_pp" if args.pp else "fd")]
    if args.kdrive is None:
        args.kdrive = 2.0 if args.forcing == "athenapk" else 1.5
    return args


def main():
    args = parse_arguments()
    if args.x64:
        jax.config.update("jax_enable_x64", True)

    sound_speed = 1.0 / args.mturb
    thermal_pressure = sound_speed ** 2 * RHO0
    seed_field_strength = float(np.sqrt(2.0 * thermal_pressure / args.beta))
    t_cross = L_INJ / VRMS_TARGET
    t_end = args.tcross * t_cross
    num_scalar_snapshots = args.nsnap * args.history_every
    snapshot_spacing = t_end / num_scalar_snapshots
    label, scheme, scheme_key = scheme_description(args)

    config = build_config(args, num_scalar_snapshots)
    registered_variables = get_registered_variables(config)
    params = build_params(args, sound_speed, t_end)
    state = initial_state(args, config, registered_variables, sound_speed, seed_field_strength)
    config = finalize_config(config, state.shape)

    print(f"[{label} N={args.n}] {scheme} backend={args.backend} "
          f"x64={args.x64} a={sound_speed} "
          f"{'gamma=' + str(args.gamma) + ' glm_alpha=' + str(args.glm_alpha) + ' ' if args.scheme == 'vl2' else ''}"
          f"beta0={args.beta:g} B0={seed_field_strength:.4g} seed={args.seed_field} "
          f"forcing={args.forcing} kdrive={args.kdrive} amplitude_factor={args.amplitude_factor} "
          f"nu={args.mom_diff:g} eta={args.ohm_diff:g} "
          f"C_cfl={args.cfl} t_end={t_end} ({args.tcross} t_cross)", flush=True)

    records = {"history": [], "spectral": [], "keep": True}
    reduce_snapshot = make_snapshot_reducer(
        args,
        config,
        registered_variables,
        sound_speed,
        snapshot_spacing,
        t_cross,
        records,
    )

    # Warm-up call: same config (so the same compiled executable is reused) with
    # a t_end of a few steps, purely to move JIT compilation out of the timed
    # region. Its callbacks are discarded.
    records["keep"] = False
    start = walltime.time()
    warm_params = params._replace(t_end=1e-4)
    jax.block_until_ready(time_integration(state, config, warm_params, registered_variables,
                                           reduce_snapshot))
    # block_until_ready waits for the *output*, not for the host callbacks the
    # snapshot recorder queues, so without this barrier a warm-up callback could
    # still land after `keep` is flipped back on and inject a spurious record.
    jax.effects_barrier()
    t_compile = walltime.time() - start
    records["keep"] = True
    print(f"[{label} N={args.n}] compile + warm-up: {t_compile:.1f} s", flush=True)

    start = walltime.time()
    final_state = time_integration(state, config, params, registered_variables, reduce_snapshot)
    jax.block_until_ready(final_state)
    t_run = walltime.time() - start
    # Every snapshot record has landed on the host after this barrier.
    jax.effects_barrier()
    print(f"[{label} N={args.n}] run: {t_run:.1f} s", flush=True)

    # -------------------------------------------------------------
    # ================ ↓ Assemble the result file ↓ ===============
    # -------------------------------------------------------------
    history = sorted(records["history"], key=lambda record: record[0])
    history_times = np.array([record[0] for record in history])
    history_scalars = np.array([record[1] for record in history])
    history_signal_speed = np.array([record[2] for record in history])
    history_psi_rms = np.array([record[3] for record in history])
    spectral = sorted(records["spectral"], key=lambda record: record["time"])
    times = np.array([record["time"] for record in spectral])
    scalars = np.array([record["scalars"] for record in spectral])
    spectra_versions = np.array([record["spectra"] for record in spectral])

    # Step count from the recorded signal speeds, n_steps = int dt' / dt(t'),
    # over the dense scalar history. Accurate to its sampling of the speed.
    grid_spacing = BOX_SIZE / args.n
    inverse_time_step = history_signal_speed / (args.cfl * grid_spacing)
    n_steps = (float(np.trapezoid(inverse_time_step, history_times))
               if len(history_times) > 1 else float("nan"))

    spectrum_names = list(SPECTRUM_NAMES[:spectra_versions.shape[2]])
    if spectra_versions.shape[2] > len(SPECTRUM_NAMES):
        spectrum_names.append(GLM_TRANSFER_NAME)

    scalar_index = {name: index for index, name in enumerate(SCALAR_NAMES)}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{scheme_key}_N{args.n}"
    path = out / f"astronomix_{tag}.npz"
    payload = dict(
        code="astronomix", label=label, scheme=scheme, scheme_key=scheme_key,
        scheme_family=args.scheme, finite_volume=args.scheme == "vl2",
        weno_z=args.scheme == "fd" and not args.weno_js,
        positivity_preserving=args.pp, backend=args.backend,
        forcing=args.forcing, mom_diff=args.mom_diff, ohm_diff=args.ohm_diff,
        N=args.n, tag=tag, t_wall=t_run, t_compile=t_compile,
        n_steps_estimated=n_steps, zone_updates_per_s=args.n ** 3 * n_steps / t_run,
        times=times, t_over_tc=times / t_cross, t_cross=t_cross,
        n_shell=shell_numbers(args.n), spectra=spectra_versions[:, 0], x64=args.x64,
        signal_speed=np.array([record["signal_speed"] for record in spectral]),
        cfl=args.cfl, a=sound_speed, B0=seed_field_strength, beta0=args.beta,
        mturb=args.mturb, rho0=RHO0, gamma=args.gamma if args.scheme == "vl2" else 1.0,
        glm_alpha=args.glm_alpha if args.scheme == "vl2" else np.nan,
        seed_field=args.seed_field, kdrive=args.kdrive, amplitude_factor=args.amplitude_factor, seed=args.seed,
        history_every=args.history_every,
        scalar_names=np.array(SCALAR_NAMES), spectrum_names=np.array(spectrum_names),
        # The dense scalar history, under the names AthenaPK's .hst series is
        # stored with, so the growth-rate fits read both codes alike.
        hst_time=history_times,
        hst_KE=history_scalars[:, scalar_index["E_K"]],
        hst_ME=history_scalars[:, scalar_index["E_B"]],
        hst_mach=history_scalars[:, scalar_index["mach"]],
        hst_v_rms=history_scalars[:, scalar_index["v_rms"]],
        hst_signal_speed=history_signal_speed,
        **{name: scalars[:, index] for index, name in enumerate(SCALAR_NAMES)},
    )
    if args.scheme == "vl2":
        payload["spectra_deconv"] = spectra_versions[:, 1]
        payload["psi_rms"] = np.array([record["psi_rms"] for record in spectral])
        payload["hst_psi_rms"] = history_psi_rms
    else:
        payload["rel_divB_face"] = np.array([record["divergence_face"] for record in spectral])
    if args.slice_series:
        payload["EB_slice_series"] = np.array([record["magnetic_energy_slice"]
                                               for record in spectral])
    if args.save_slices:
        # time_integration returns the *unpadded* state (unlike the padded one
        # the snapshot callback sees), so no ghost slicing here.
        z_mid = args.n // 2
        final_state = np.asarray(final_state)
        density_slice = final_state[registered_variables.density_index][:, :, z_mid]
        payload["rho_slice"] = density_slice
        payload["EB_slice"] = 0.5 * sum(final_state[index][:, :, z_mid] ** 2
                                        for index in registered_variables.magnetic_index)
        payload["EK_slice"] = 0.5 * density_slice * sum(
            final_state[index][:, :, z_mid] ** 2 for index in registered_variables.velocity_index)
    np.savez_compressed(path, **payload)
    print(f"[{label} N={args.n}] wrote {path}  "
          f"(t_wall={t_run:.1f} s, ~{n_steps:.0f} steps)", flush=True)
    # -------------------------------------------------------------
    # ================ ↑ Assemble the result file ↑ ===============
    # -------------------------------------------------------------


if __name__ == "__main__":
    main()
