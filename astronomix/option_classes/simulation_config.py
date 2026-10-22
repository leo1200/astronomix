"""
Static simulation configuration.

Defines :class:`SimulationConfig` — the bundle of options that, unlike the
simulation parameters, necessitate recompilation when changed — together with
the integer-coded enumerations they reference (backends, solver/boundary/Riemann
modes, positivity modes, ...), the small geometry vector helpers, the sub-configs
for gravity and positivity, and the ``finalize_config`` pass that fills in
derived fields and validates the configuration.
"""

# general
import math
import subprocess

# typing
from types import NoneType
from typing import NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Float

# jax
import jax

# astronomix containers
from astronomix._modules._cnn_mhd_corrector._cnn_mhd_corrector_options import (
    CNNMHDconfig,
)
from astronomix._modules._cooling.cooling_options import CoolingConfig
from astronomix._modules._cosmic_rays.cosmic_ray_options import CosmicRayConfig
from astronomix._modules._neural_net_force._neural_net_force_options import (
    NeuralNetForceConfig,
)
from astronomix._modules._sink_particles._sink_particle_options import SinkParticleConfig
from astronomix._modules._stellar_wind.stellar_wind_options import WindConfig
from astronomix._modules._turbulent_forcing._turbulent_forcing_options import (
    TurbulentForcingConfig,
)

# -------------------------------------------------------------
# ================= ↓ Constant definitions ↓ ==================
# -------------------------------------------------------------

# backends (very limited support currently)
NATIVE_JAX = 0
PALLAS = 1
#: OPTIMAL_BACKEND is not a backend of its own: it is a request to pick the
#: fastest available one at ``finalize_config`` time.  It resolves to PALLAS
#: when JAX's default backend is a GPU new enough to run the Triton kernels
#: (compute capability >= 8.0, read from ``jax.devices()``) and falls back to
#: NATIVE_JAX everywhere else (older GPUs, CPU -- including ``JAX_PLATFORMS=cpu``
#: on a GPU node).
OPTIMAL_BACKEND = 2

# Per-step state floor (``PositivityConfig.per_step_mode``). Positivity of the
# finite-difference scheme itself comes from ``weno_positivity_preserving``; the
# floor remains for the finite-volume solver and as the temperature floor of
# radiatively cooled runs (``per_step_specific_floor``). HARD_FLOOR clamps
# density (and, for an ideal gas, pressure) pointwise; it is not conservative.
POSITIVITY_NONE = 0
POSITIVITY_HARD_FLOOR = 1

#: Attempts of a step after the first under ``PositivityConfig.step_retry``,
#: each at half the step size of the one before (the last at 1/64).
MAXIMUM_STEP_RETRIES = 6

# solver modes
FINITE_VOLUME = 0
FINITE_DIFFERENCE = 1

# differentiation modes
FORWARDS = 0
BACKWARDS = 1

# Passive-scalar sub-cycling loop (``SimulationConfig.passive_scalar_substep_loop``).
#   SUBSTEPS_AUTO     - DYNAMIC under FORWARDS, MASKED under BACKWARDS (default)
#   SUBSTEPS_DYNAMIC  - a traced trip count (lowers to a while loop): exactly the
#                       flow-derived number of sub-steps runs; forward-mode AD
#                       only (reverse mode cannot differentiate a while loop)
#   SUBSTEPS_MASKED   - a static loop over ``max_passive_scalar_substeps`` in
#                       which sub-steps beyond the flow-derived count are masked
#                       out (``lax.cond``): reverse-mode differentiable, same
#                       primal
SUBSTEPS_AUTO = 0
SUBSTEPS_DYNAMIC = 1
SUBSTEPS_MASKED = 2

# Rematerialisation in reverse mode (``SimulationConfig.ad_remat``). Unlike the
# integer enumerations, the modes are strings, so ``ad_remat="stage"`` works
# without importing the constants.
#   "none"  - store every residual of the step (fastest backward, most memory)
#   "stage" - ``jax.checkpoint`` around every Runge-Kutta-stage RHS (hydro WENO
#             fluxes + blends + divergence + sources, and the passive-scalar
#             advection RHS): the backward keeps only the stage inputs and
#             recomputes each stage's internals (about one extra forward)
#   "axis"  - "stage" plus nested checkpoints: each axis' flux + blend +
#             divergence inside the hydro RHS (non-fused flux path), and in the
#             passive-scalar RHS each axis and each scalar separately, so the
#             backward holds one axis' (one scalar's) internals at a time
AD_REMAT_NONE = "none"
AD_REMAT_STAGE = "stage"
AD_REMAT_AXIS = "axis"
AD_REMAT_MODES = (AD_REMAT_NONE, AD_REMAT_STAGE, AD_REMAT_AXIS)

# limiter types
MINMOD = 0
OSHER = 1
DOUBLE_MINMOD = 2
SUPERBEE = 3
VAN_ALBADA = 4
VAN_ALBADA_PP = 5
#: The harmonic-mean (van Leer) slope of AthenaPK's / Athena++'s piecewise
#: linear reconstruction. This is the slope of the VL2 scheme, which always
#: uses it and does not read ``limiter`` (its choice between piecewise-linear
#: and donor-cell reconstruction is ``first_order_fallback``); ``finalize_config``
#: sets ``limiter = VAN_LEER`` for VL2 only so that the configuration reports
#: the reconstruction in use. The classic finite-volume scheme does not
#: implement it.
VAN_LEER = 6

# splitting modes
UNSPLIT = 0
SPLIT = 1

# Riemann solvers
HLL = 0
HLLC = 1
HLLC_LM = 2
LAX_FRIEDRICHS = 3
HYBRID_HLLC = 4
AM_HLLC = 5
#: The HLLD solver of Miyoshi & Kusano (2005) for ideal MHD, in AthenaPK's
#: GLM form. Only implemented in the VL2 scheme (which also maps HLL, HLLC and
#: LAX_FRIEDRICHS onto AthenaPK's HLLE, HLLC and LLF solvers); ``finalize_config``
#: rejects it for the other finite-volume time integrators.
HLLD = 6

# time integrators
# currently only for finite volume
RK2_SSP = 0
MUSCL = 1
# currently only for finite difference
RK4_SSP = 2
RK4_LSRK = 3
# finite volume only
#: The second-order van Leer predictor-corrector of AthenaPK / Athena++
#: (Stone & Gardiner 2009): a donor-cell half step followed by a full step
#: with van Leer-limited piecewise-linear reconstruction (donor cell in both
#: steps with ``first_order_fallback``). Selecting it switches the
#: finite-volume solver to the AthenaPK-equivalent scheme, which for MHD
#: carries the cell-centred field together with the GLM cleaning scalar psi
#: (Dedner et al. 2002). The scheme is Cartesian. Self-gravity and an external
#: potential enter through Athena++'s coupling (``ATHENA_SECOND_ORDER``); the
#: scheme has no stellar-wind-tracer or cosmic-ray terms, and
#: ``finalize_config`` rejects those combinations (spherical geometry, as for
#: every spherical configuration, is switched to MUSCL instead).
VL2 = 4

# boundary conditions
OPEN_BOUNDARY = 0
REFLECTIVE_BOUNDARY = 1
PERIODIC_BOUNDARY = 2
FIXED_BOUNDARY = 3
MHD_JET_BOUNDARY = 4
FIXED_BOUNDARY_OPEN_MOMENTUM = 5

PRIMITIVE_GAS_STATE = 0
CONSERVATIVE_GAS_STATE = 1
VELOCITY_ONLY = 2
MAGNETIC_FIELD_ONLY = 3

# geometry types
CARTESIAN = 0
CYLINDRICAL = 1
SPHERICAL = 2

# axes
VARAXIS = 0
XAXIS = 1
YAXIS = 2
ZAXIS = 3

# boundary handling modes
GHOST_CELLS = 0
PERIODIC_ROLL = 1
# OPEN_SHIFT = 2

# self-gravity coupling schemes (FD):
#   SIMPLE_SOURCE              - rho * v * a energy source (non-conservative)
#   SECOND_ORDER_CONSERVATIVE  - flux-based energy source (2nd-order accurate)
#   FOURTH_ORDER_CONSERVATIVE  - corrected flux-based energy source (4th-order,
#                                the energy-conserving high-order scheme)
# and of the finite-volume VL2 scheme:
#   ATHENA_SECOND_ORDER        - Athena++'s coupling (second order): cell-centred
#                                rho grad(phi) momentum source and the energy
#                                source from the stage's Riemann mass fluxes
SIMPLE_SOURCE = 0
SECOND_ORDER_CONSERVATIVE = 1
FOURTH_ORDER_CONSERVATIVE = 2
ATHENA_SECOND_ORDER = 3

# Magnetic part integrators for split MHD
IMPLICIT_MIDPOINT = 0
IMPLICIT_EULER = 1

# Numerical precision
SINGLE_PRECISION = 0
DOUBLE_PRECISION = 1

# Viscosity types
KINEMATIC_VISCOSITY = 0
DYNAMIC_VISCOSITY = 1

# Equation of state
IDEAL_GAS = 0
ISOTHERMAL = 1

# Snapshot storage modes
ON_DEVICE = 0
TO_DISK = 1

# -------------------------------------------------------------
# ================= ↑ Constant definitions ↑ ==================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =================== ↓ Type definitions ↓ ====================
# -------------------------------------------------------------


class StaticIntVector(NamedTuple):
    """A static (compile-time) per-axis integer triple (e.g. cells per axis)."""

    x: int = -1
    y: int = -1
    z: int = -1


class StaticFloatVector(NamedTuple):
    """A static (compile-time) per-axis float triple (e.g. box size per axis)."""

    x: float = -1.0
    y: float = -1.0
    z: float = -1.0

    def __truediv__(self, other: StaticIntVector) -> "StaticFloatVector":
        """Divide component-wise by a :class:`StaticIntVector` (e.g. box / cells)."""
        if not isinstance(other, StaticIntVector):
            return NotImplemented
        return StaticFloatVector(
            x=self.x / other.x,
            y=self.y / other.y,
            z=self.z / other.z,
        )


STATE_TYPE = Union[
    Float[Array, "num_vars num_cells_x"],
    Float[Array, "num_vars num_cells_x num_cells_y"],
    Float[Array, "num_vars num_cells_x num_cells_y num_cells_z"],
]

STATE_TYPE_ALTERED = Union[
    Float[Array, "num_vars num_cells_a"],
    Float[Array, "num_vars num_cells_a num_cells_b"],
    Float[Array, "num_vars num_cells_a num_cells_b num_cells_c"],
]

FIELD_TYPE = Union[
    Float[Array, "num_cells_x"],
    Float[Array, "num_cells_x num_cells_y"],
    Float[Array, "num_cells_x num_cells_y num_cells_z"],
]

# -------------------------------------------------------------
# =================== ↑ Type definitions ↑ ====================
# -------------------------------------------------------------


class SnapshotSettings(NamedTuple):
    """Settings for the snapshot output of the simulation."""

    #: Whether to record the full primitive state at every checkpoint.
    #: This is the single biggest snapshot allocation
    #: (``num_snapshots × num_vars × num_cells^d``); it is **opt-in**.
    #: Set to ``True`` if you actually need the per-snapshot states; for
    #: the common case of only wanting a final state plus integrated
    #: diagnostics (energies, total mass, runtime, num_iterations), the
    #: default ``False`` skips the per-snapshot state allocation entirely.
    return_states: bool = False

    #: Whether to return the final state of the simulation.
    return_final_state: bool = False

    #: Whether to return the total mass at the times the snapshots were taken.
    return_total_mass: bool = False

    #: Whether to return the total energy at the times the snapshots were taken.
    return_total_energy: bool = False

    #: Whether to return internal energy
    return_internal_energy: bool = False

    #: Whether to return kinetic energy
    return_kinetic_energy: bool = False

    #: Whether to return gravitational energy
    return_gravitational_energy: bool = False

    #: Whether to return radial momentum
    return_radial_momentum: bool = False

    #: Whether to return the kinetic energy spectrum
    return_kinetic_energy_spectrum: bool = False

    #: Whether to return the magnetic energy spectrum
    return_magnetic_energy_spectrum: bool = False

    #: Whether to return the helicity spectrum
    return_helicity_spectrum: bool = False

    #: Whether to return the magnetic field divergence
    #: NOTE: currently only implemented for finite difference MHD
    return_magnetic_divergence: bool = False

    #: Whether to return the temperature PDF (dV/dlogT)
    return_temperature_pdf: bool = False
    num_temperature_bins: int = 100
    temperature_pdf_min: float = 1e-10
    temperature_pdf_max: float = 1e10


class BoundarySettings1D(NamedTuple):
    """The boundary-condition type at the left and right end of a single axis."""

    left_boundary: int = OPEN_BOUNDARY
    right_boundary: int = OPEN_BOUNDARY


class BoundarySettings(NamedTuple):
    """Per-axis boundary settings for the simulation."""

    x: BoundarySettings1D = BoundarySettings1D()
    y: BoundarySettings1D = BoundarySettings1D()
    z: BoundarySettings1D = BoundarySettings1D()


class GravityConfig(NamedTuple):
    """Self-gravity and external-potential configuration."""

    #: Self-gravity switch (currently only for periodic / manual-open boundaries).
    self_gravity: bool = False

    #: Coupling of the self-gravity source to the hydrodynamics. One of
    #: ``SIMPLE_SOURCE`` / ``SECOND_ORDER_CONSERVATIVE`` /
    #: ``FOURTH_ORDER_CONSERVATIVE`` (finite difference) or
    #: ``ATHENA_SECOND_ORDER`` (the finite-volume VL2 scheme, which
    #: ``finalize_config`` switches to it).
    self_gravity_version: int = FOURTH_ORDER_CONSERVATIVE

    #: Enable an external, static gravitational potential provided via
    #: ``params.gravitational_potential``. It is added to the self-gravity
    #: potential (if any) in ``_compute_total_potential``.
    external_potential: bool = False

    #: Manual open boundary conditions in the Poisson solver.
    poisson_manual_open_boundaries: bool = False

    #: Flux-corrected gravitational work (finite difference,
    #: SECOND/FOURTH_ORDER_CONSERVATIVE). Every conservative energy coupling
    #: is a choice of the potential-energy flux q at each face,
    #: S_E,i = -(1/dx) sum [(q - F phi_i)_{i+1/2} - (q - F phi_i)_{i-1/2}],
    #: and conserves total energy for ANY q. The scheme's high-order q charges
    #: half the climb of mass crossing a face to each side, so a cold or
    #: tenuous receiver can be driven to negative pressure; the low-order
    #: q = F phi_downwind charges the whole climb to the donor (the cell the
    #: mass leaves). This option blends them face by face,
    #: q = q_low + psi (q_high - q_low), with psi in [0, 1] chosen by a
    #: one-sided Zalesak limiter so that every cell's internal-energy loss
    #: rate stays within half its internal energy per wave-crossing time (rate
    #: budgets, so psi does not depend on dt; the per-cell scaling is
    #: sufficient, not maximal, and the bound is not guaranteed where the
    #: low-order coupling alone exceeds it). Exactly conservative; high order
    #: wherever psi = 1.
    work_flux_correction: bool = False

    #: Periodic Poisson solve with the Green's function of the second-order
    #: (3-, 5- or 7-point) discrete Laplacian, -4 pi G / sum_d [4 sin^2(k_d dx
    #: / 2) / dx^2], instead of the spectral -4 pi G / k^2. This is the kernel
    #: of Athena++'s FFT solver (and the operator its multigrid solver
    #: inverts). With it the density is exactly the discrete Laplacian of the
    #: potential, so the cell-centred momentum source of ``ATHENA_SECOND_ORDER``
    #: is the divergence of a discrete gravitational stress tensor (Mullen,
    #: Hanawa & Gammie 2021): momentum is conserved locally, not only in total.
    #: The potential is second-order accurate, so the finite-difference
    #: couplings keep the spectral kernel. The open-boundary (Hockney-Eastwood)
    #: solve always uses the point-mass kernel.
    poisson_discrete_laplacian: bool = False

    #: ``ATHENA_SECOND_ORDER`` only: the time-centred energy source of Mullen,
    #: Hanawa & Gammie (2021). Each VL2 stage, which updates from ``U^n``,
    #: charges its mass fluxes with the average of the potential at ``t^n``
    #: and of the potential of the stage's new density (one more Poisson solve
    #: per step). The energy change of every stage then equals minus the change
    #: of the gravitational energy ``(1/2) sum rho phi`` exactly, so total
    #: energy is conserved to round-off for any time step (with a symmetric
    #: Green's function and no floor acting). Athena++ itself charges the
    #: potential of the stage's initial state (second-order energy error).
    time_centred_energy_source: bool = False

    #: Master gravity switch. Set automatically in ``finalize_config`` to
    #: ``self_gravity or external_potential``; gates the gravity source-term
    #: machinery so an external potential works without self-gravity. Not set
    #: by the user directly.
    gravity: bool = False


class PositivityConfig(NamedTuple):
    """
    State floors and estimate clamps.

    The finite-difference scheme keeps density and pressure positive through
    ``SimulationConfig.weno_positivity_preserving`` (every stage a convex
    combination of admissible states). What is left here is not a positivity
    patch: the per-step floor serves the finite-volume solver and the
    temperature floor of radiatively cooled runs, the clamps keep wave-speed
    and time-step estimates finite, and the cold-crush blend damps the
    runaway compression of radiatively cooled shells.
    """

    #: State floor applied once per step before the evolve (on the primitive
    #: state): ``POSITIVITY_NONE`` or ``POSITIVITY_HARD_FLOOR``. With
    #: ``time_integrator=VL2`` the hard floor also switches on AthenaPK's
    #: density / pressure floors inside every stage's primitive recovery.
    per_step_mode: int = POSITIVITY_NONE

    #: Upgrade the per-step HARD_FLOOR pressure clamp to the density-scaled
    #: temperature floor ``p >= max(minimum_pressure,
    #: rho * params.minimum_specific_pressure)`` (Athena-style temperature
    #: floor). Meant for radiatively cooled runs: a radiatively cooled shock
    #: layer compresses to the isothermal jump, and without isothermal pressure
    #: support (p ∝ rho) the constant floor leaves it effectively pressureless,
    #: so it is crushed by the ram pressure without bound. With cooling active
    #: the floor's energy input is radiated away (the isothermal balance).
    #: No-op when ``params.minimum_specific_pressure == 0``.
    per_step_specific_floor: bool = False

    #: Read-only density/pressure clamp in the flux / eigenvalue / timestep
    #: estimates. It never modifies the evolved state (the step-end primitive
    #: recovery is unclamped).
    clamp_in_estimates: bool = True

    #: Redo a time step at half the step size, from the same state, forcing
    #: and PRNG key, when it leaves a non-finite value or a non-positive
    #: density (or pressure, for an ideal gas), up to ``MAXIMUM_STEP_RETRIES``
    #: times. The positivity-preserving WENO keeps every stage positive only
    #: while no stage's splitting speeds exceed those the step was sized with;
    #: a near-vacuum cell that accelerates within a single step (its mass
    #: drains faster than its momentum) breaks that, and the step is then
    #: repeated instead of carrying a negative density into the next one.
    #: Adaptive time stepping and forward-mode runs only.
    step_retry: bool = False

    #: Cold-crush first-order flux blending (the FD counterpart of Athena's
    #: first-order flux correction for radiatively cooled gas): blend the WENO
    #: interface flux toward LLF at interfaces with a COLD side under
    #: COMPRESSION. The weight is a temperature ramp on the COLDER adjacent
    #: cell's recovered ``p/rho`` (1 at the effective temperature floor
    #: ``params.minimum_specific_pressure``, 0 at ``coldcrush_blend_factor``
    #: times it) times a compressive-velocity gate, so freely expanding cold
    #: gas and a static ambient medium never activate it. It catches both
    #: cold-cold isothermal collapse and the crushing of a cold dense clump by
    #: hot surroundings; the trade is locally first-order shock fronts into
    #: cold gas (classic flux-correction behaviour). Radiatively cooled cells
    #: crushed by ram pressure otherwise collapse without bound once the grid
    #: resolves the cooling layer: the local first-order diffusion saturates
    #: the collapse the way coarse-grid numerical diffusion does at lower
    #: resolution. Inert unless ``params.minimum_specific_pressure > 0``.
    coldcrush_blend: bool = False

    #: Upper end of the temperature ramp of ``coldcrush_blend``, in units of
    #: ``params.minimum_specific_pressure``: interfaces whose colder side has
    #: ``p/rho`` above this multiple of the floor are not blended.
    coldcrush_blend_factor: float = 8.0


class BackendConfig(NamedTuple):
    """Compute-backend configuration: which backend runs the kernels and the
    Pallas/Triton knobs that shape them.

    Grouped as its own sub-config (like ``PositivityConfig`` / ``GravityConfig``)
    so backend concerns stay together.  Construct nested, e.g.
    ``SimulationConfig(backend_config=BackendConfig(backend=NATIVE_JAX))``.
    """

    #: Backend. Defaults to OPTIMAL_BACKEND, which ``finalize_config`` resolves
    #: to PALLAS on compute-capability >= 8.0 GPUs and NATIVE_JAX otherwise.
    backend: int = OPTIMAL_BACKEND
    #: Pallas kernel block shape ``(bx, by, bz)``; ``None`` picks the tuned
    #: default, 128 cells along the contiguous last axis ((1, 1, 128) in 3D,
    #: (1, 128) in 2D; see ``_default_pallas_block_shape``). Every block size
    #: is reduced to the largest power of two that divides the grid extent.
    #: With the default ``pallas_num_warps=4`` keep blocks at 128 cells (one
    #: element per thread) — larger blocks register-spill, smaller ones idle
    #: threads — and their rows long along the last (contiguous) axis.
    pallas_block_shape: Optional[Tuple[int, int, int]] = None
    pallas_use_triton: bool = True
    pallas_interpret: bool = False
    pallas_num_warps: int = 4
    #: Toggle for the Pallas constrained-transport helpers
    #: (``update_cell_center_fields``, ``constrained_transport_rhs``).
    #: Disabled by default: the staged Pallas-CT pipeline saves a large share
    #: of the temporary memory on small grids, but only a few percent on
    #: production-size grids, while adding a noticeable one-time compile cost.
    #: Flip to True if the small-grid memory profile matters; the rest of the
    #: Pallas backend stays on regardless.
    pallas_ct: bool = False
    #: Replace the IEEE ``sqrt`` in the MHD WENO kernel with the refined
    #: approximate ``rsqrt`` path (``x * jax.lax.rsqrt(x)`` -> ``rsqrt.approx.f64``,
    #: still ~1 ULP). On GPUs where the double-precision ``sqrt`` dominates the
    #: kernel this markedly speeds up the double-precision step (fewer register
    #: spills) without changing the convergence behaviour (A100; on a GH200 it
    #: makes no measurable difference).
    #: NOTE: only the *forward* Pallas kernel is switched; derivatives go through
    #: the native tangent, which uses IEEE ``sqrt``, so with this on reverse-mode
    #: AD is ~1 ULP inconsistent with the forward (``finalize_config`` warns).
    #: There is no equivalent fast f64 path for division, so only ``sqrt`` changes.
    use_approximate_rsqrt: bool = False
    #: How the multi-device Pallas kernels exchange their halos along a split
    #: axis. ``"ppermute"``: two collective permutes over the whole periodic
    #: ring of devices. ``"pair_all_to_all"``: two phases of all-to-alls over
    #: disjoint neighbour pairs (even-odd, then odd-even including the
    #: wraparound pair), so every collective spans two devices only and its
    #: communicator does not grow with the device count. Both move the same
    #: halo data and give bit-identical results; an axis split over an odd
    #: number of devices always uses the ring. The pairwise exchange is ~5%
    #: faster at 8-32 GPUs and 5-13% at 128 GPUs on JUPITER (GH200); its
    #: per-pair NCCL communicators need device memory beyond the XLA pool, so
    #: multi-node runs should leave ~10% of the GPU memory to NCCL.
    halo_exchange: str = "pair_all_to_all"
    #: Reverse mode (``differentiation_mode == BACKWARDS``) of the ideal-gas
    #: hydro and ideal-MHD WENO fluxes through a Pallas adjoint kernel
    #: (``jax.vjp`` of the forward kernel's window function, inside the kernel;
    #: the positivity-preserving recombination of MHD in native JAX) instead of
    #: the transpose of the native-JAX tangent, which also re-runs the native
    #: forward for its linearisation (the hydro gradient is ~4x faster at
    #: 128^3-256^3). Opt-in, because it differentiates with respect to the state
    #: (and the dual-energy ``g`` and the MHD inflow reference) only: the
    #: physical constants in ``params`` get a ZERO cotangent through the flux,
    #: and forward-mode AD (``jax.jvp``) through a BACKWARDS configuration then
    #: raises. A sharded run needs at least the block-rounded 5-cell halo per
    #: shard along every flux axis that is split.
    pallas_adjoint: bool = False


class SimulationConfig(NamedTuple):
    """
    Configuration object for the simulation.
    The simulation configuration are parameters defining
    the simulation where changes necessitate recompilation.
    """

    # Static simulation parameters

    #: Compute-backend configuration (see :class:`BackendConfig`): backend
    #: choice + Pallas/Triton kernel knobs.  Accessed as
    #: ``config.backend_config.backend`` / ``.pallas_block_shape`` / etc.
    backend_config: BackendConfig = BackendConfig()

    #: Basic solver mode, either finite volume or finite difference.
    #: Defaults to the finite-difference HOW-MHD scheme (Jeongbhin Seo,
    #: Dongsu Ryu, 2023), which is the recommended solver.
    solver_mode: int = FINITE_DIFFERENCE

    #: Precision mode.
    numerical_precision: int = SINGLE_PRECISION

    #: Debug runtime errors, throws exceptions
    #: on e.g. negative pressure or density.
    #: Significantly reduces performance.
    runtime_debugging: bool = False

    #: Donate the state arrays to the time integration function
    #: to reduce memory allocations. If activated, the
    #: initial state arrays will be invalid after
    #: the simulation.
    donate_state: bool = False

    #: Memory analysis of the main time integration
    #: function
    memory_analysis: bool = False

    #: Build the simulation helper data on the host (CPU) and
    #: only move the fields that are actually needed by the
    #: enabled subsystems onto the accelerator. Useful in
    #: production runs where a large meshgrid like
    #: ``geometric_centers`` is not required on device and the
    #: per-field memory footprint matters.
    host_helper_data: bool = False

    #: Print the elapsed time of the simulation
    print_elapsed_time: bool = False

    #: Activate progress bar
    progress_bar: bool = False

    #: The number of dimensions of the simulation.
    dimensionality: int = 1

    #: Use a struct for the state.
    state_struct: bool = False

    #: The geometry of the simulation.
    geometry: int = CARTESIAN

    #: The random seed for any stochastic processes
    #: in the simulation, e.g. turbulent forcing.
    random_seed: int = 42

    #: The equation of state for the simulation.
    #: NOTE: CURRENTLY ONLY IMPLEMENTED FOR 
    #: FINITE DIFFERENCE MODE.
    equation_of_state: int = IDEAL_GAS

    #: Magnetohydrodynamics switch.
    mhd: bool = False

    #: Integrator used for the magnetic part in the FV MHD scheme.
    fv_magnetic_integrator: int = IMPLICIT_MIDPOINT

    #: VL2 scheme only: add the non-conservative Dedner et al. (2002) sources,
    #: ``-(div B) B`` in the momentum and ``-B . grad(psi)`` in the energy
    #: equation, to the parabolic damping of psi (AthenaPK
    #: ``glmmhd_source = dedner_extended``). The default is the plain damping
    #: only (AthenaPK's default ``dedner_plain``).
    glm_extended_source: bool = False

    #: VL2 scheme only: AthenaPK's first-order flux correction. After each
    #: stage, every cell whose update would produce a non-positive density or
    #: pressure has all its face fluxes replaced by first-order local
    #: Lax-Friedrichs fluxes (repeated up to four times, exactly as
    #: AthenaPK's ``first_order_flux_correct``).
    first_order_flux_correction: bool = False

    #: State floors, estimate clamps and the cold-crush flux blend (see
    #: PositivityConfig). Positivity of the finite-difference scheme itself
    #: comes from ``weno_positivity_preserving``.
    positivity_config: PositivityConfig = PositivityConfig()

    #: Dual-energy formalism (Bryan et al. 1995 switch) for adiabatic FD
    #: hydro and MHD. When True a separately-advected internal-energy density
    #: ``g`` is carried through the time loop and used in the WENO pressure
    #: recovery wherever ``e_E/E < dual_energy_eta``, so high-Mach / low-beta
    #: float32 cancellation of ``e_int = E - KE - ME`` does not corrupt the
    #: pressure.
    dual_energy: bool = False

    #: Switch threshold for the dual-energy formalism: the fraction of total
    #: energy below which the total-energy internal energy is deemed
    #: cancellation-unreliable and the advected ``g`` is used instead.
    dual_energy_eta: float = 1e-3

    #: Number of user-defined passive scalars: per-parcel labels advected with
    #: the flow (``dC/dt + v.grad C = 0``) that do not act back on it —
    #: composition mass fractions, an ejecta/circumstellar discriminator, and so
    #: on. Finite-difference solver only; they are carried as the last variables
    #: of the state and advected operator-split (WENO5 + SSP-RK3) in
    #: ``_passive_scalars.py``.
    num_passive_scalars: int = 0

    #: Physical bounds for the user's passive scalars, one ``(lo, hi)`` pair per
    #: scalar (``None`` for an unbounded one). Declaring them is strongly
    #: recommended for anything that is a mass fraction: the recovered label is
    #: a ratio ``s / rho~``, and in the near-vacuum interior of a blast wave the
    #: denominator can collapse and the ratio run away (for example to values
    #: far outside [0, 1] where radiative cooling and a fast piston compress the
    #: same cells).
    #:
    #: Note this is NOT the same as clipping a scalar to its own current range,
    #: which is destructive: that clips smooth extrema every step and degrades
    #: the convergence order. A *physical* bound never activates on smooth data
    #: that respects it, so it is free.
    passive_scalar_bounds: Tuple[Optional[Tuple[float, float]], ...] = ()

    #: CFL number for the passive-scalar sub-steps. The scalar advection is
    #: operator-split, so it does not inherit the hydro timestep's safety: the
    #: hydro CFL is set by ``|u| + c``, while what constrains this is ``|u|`` and
    #: — the binding one — positivity of the companion density under
    #: ``h |div v| <= 0.5``. The number of sub-steps is derived from the flow at
    #: every step, so a benign step takes exactly one and costs nothing.
    passive_scalar_cfl: float = 0.4

    #: Cap on the derived sub-step count, so a single pathological cell cannot
    #: stall a run. Hitting the cap means the scalars are being integrated
    #: outside their stability limit in some cells; the physical bounds in
    #: ``passive_scalar_bounds`` are the backstop for that.
    max_passive_scalar_substeps: int = 8

    #: How the sub-cycling loop is built (``SUBSTEPS_AUTO`` / ``_DYNAMIC`` /
    #: ``_MASKED``, see the constants). The flow-derived count is a traced
    #: integer, and a loop with a traced trip count is a while loop, which
    #: reverse-mode AD cannot differentiate. The MASKED loop runs to the static
    #: cap ``max_passive_scalar_substeps`` and skips (``lax.cond``) the sub-steps
    #: beyond the flow-derived count, so its primal is the same computation. The
    #: default AUTO keeps the dynamic loop under ``differentiation_mode ==
    #: FORWARDS`` (forward results unchanged, bit for bit) and switches to the
    #: masked one under BACKWARDS. In the masked loop every sub-step (the first
    #: included) is one iteration of a scan whose body is ``jax.checkpoint``-ed
    #: (its RHS too, whatever ``ad_remat`` says), so the backward stores one
    #: scalar-stack copy per iteration and recomputes the rest: reverse-mode
    #: memory grows only mildly with the cap, compute by one extra advection
    #: evaluation. A cap of 1 is a straight-line single sub-step (no loop).
    #: Lower ``max_passive_scalar_substeps`` in reverse-mode runs whose flow
    #: never needs more than one or two sub-steps.
    passive_scalar_substep_loop: int = SUBSTEPS_AUTO

    #: Track when each parcel was shocked, adding three further library-managed
    #: scalars (``entropy_initial``, ``time_since_shock``, ``density_time``)
    #: after the user's. ``density_time`` is the ionization age up to a unit
    #: conversion, and ``time_since_shock`` drives electron/ion temperature
    #: relaxation — the Dwarkadas, Dewey & Bauer (2010) proxy that makes
    #: non-equilibrium-ionization spectral synthesis possible without an
    #: ionization network.
    track_shock_history: bool = False

    #: Entropy rise (in nats, above the parcel's own ``t = 0`` value) that counts
    #: as having been shocked. Because the Rankine-Hugoniot jump fixes the
    #: entropy rise as a function of Mach number alone, this threshold is
    #: equivalent to a minimum shock strength; for ``gamma = 5/3``:
    #:
    #: ===========  ==========
    #: Mach number  rise [nats]
    #: ===========  ==========
    #: 2            0.18
    #: 3            0.57
    #: **3.3**      **0.69**
    #: 5            1.31
    #: 10           2.57
    #: 100          7.12
    #: ===========  ==========
    #:
    #: The default ``log(2)`` therefore flags everything above Mach ~3.3 —
    #: comfortably below a supernova remnant's Mach ~100 forward and reverse
    #: shocks, while ignoring the weak compressions and sound waves that carry
    #: no ionization. Lower it if weak shocks matter for the problem at hand.
    shock_entropy_jump: float = 0.6931471805599453

    #: Give the shock-history latch a derivative (straight-through estimator).
    #: The primal is UNCHANGED: a parcel is still flagged by the boolean test
    #: ``entropy rise > shock_entropy_jump AND div v < 0``. With the switch off
    #: that boolean has zero derivative, so d(shocked_fraction, time_since_shock,
    #: density_time)/d(state) only transports the existing history, and an
    #: adjoint cannot see that moving a shock changes which parcels are freshly
    #: shocked. With it on, the tangent (and cotangent) of the latch is taken
    #: through the smooth surrogate
    #:
    #:     soft = sigmoid((ds - shock_entropy_jump) / ad_shock_latch_entropy_width)
    #:          * sigmoid(-div_v / (ad_shock_latch_compression_width * c_s))
    #:
    #: (the same two criteria; ``div_v`` is the undivided central difference
    #: the latch uses and ``c_s`` the local sound speed, held constant), merged
    #: with the carried fraction as a probabilistic OR:
    #: ``d sf = (1 - latch) d sf_old + (1 - sf_old) d soft``. Away from the
    #: threshold soft saturates and the derivative reduces to transport -- but
    #: only if the entropy sigmoid is narrow enough: quiescent, never-shocked
    #: gas (entropy rise 0, div_v = 0) sits at the steepest point of the
    #: compression sigmoid, so its leak is set by sigmoid(-jump / width) alone,
    #: and it ACCUMULATES step after step in the carried fraction. With width
    #: 0.25 about 22 % of the at-threshold response leaks into every ambient
    #: cell per step; with width 0.1 (the default) about 0.4 %. This is a
    #: surrogate gradient, not the derivative of the primal (which is zero
    #: almost everywhere); check it with a finite-difference Taylor test at the
    #: amplitudes the optimiser uses.
    ad_smooth_shock_latch: bool = False

    #: Width in nats of the entropy-rise sigmoid of the smooth latch. Keep it
    #: well below ``shock_entropy_jump`` (see the leak above).
    ad_shock_latch_entropy_width: float = 0.1

    #: Width of the compression sigmoid of the smooth latch, in units of the
    #: local sound speed (``div_v`` is an undivided difference, i.e. a velocity).
    ad_shock_latch_compression_width: float = 0.05

    #: Give the shock-history ACCUMULATORS the derivative of the shock arrival
    #: time (tangent only; the primal is unchanged bit for bit). A parcel's
    #: ``time_since_shock`` is t - t_s and its ``density_time`` the integral of rho
    #: since t_s; with the boolean latch their derivative is transport only, so an
    #: adjoint cannot see that moving a shock changes WHEN gas is shocked -- the
    #: ionization age that thermal X-ray emission depends on. With this on, a
    #: cell whose entropy rise crosses ``shock_entropy_jump`` during a step (under
    #: compression) gets the derivative of f dt (resp. rho f dt), f = a_now /
    #: (a_now - a_prev) the fraction of the step after the linearly interpolated
    #: crossing (a = rise - jump, before and after the step): a shock arriving
    #: delta_t earlier gives d(f dt) = delta_t exactly, for any shock strength
    #: and any number of steps per crossing. Nothing is added outside the
    #: crossing step, so -- unlike ``ad_smooth_shock_latch``, whose small per-step
    #: leak in quiescent gas accumulates into the accumulators over many steps and
    #: is carried into the gas when it is shocked later -- nothing leaks or
    #: accumulates. The shocked fraction's own tangent stays transport. Mutually
    #: exclusive with ``ad_smooth_shock_latch``.
    ad_shock_arrival: bool = False

    #: Whether the accumulators' per-step increments (``shocked_fraction * dt``,
    #: ``shocked_fraction * rho * dt``) carry the tangent of the advected shocked
    #: fraction (default, the exact derivative of the update). False: the
    #: increments see the shocked fraction as a constant (tangent only; the primal
    #: is unchanged bit for bit), so the accumulators change only through their
    #: own transport, the density and the arrival term. In the primal the edge of
    #: the shocked region is set by the shock criterion, not by advection, so the
    #: transport tangent of the latch field at that edge is not a derivative the
    #: primal follows; over thousands of steps it is integrated into the
    #: accumulators of every cell the front passes (2026-10-13, under test).
    ad_shock_latch_transport: bool = True

    #: Damping of the arrival term (``ad_shock_arrival``) for crossings whose
    #: entropy rise barely changes within the step: the term is weighted by
    #: d^2 / (d^2 + w^2), d = rise_now - rise_previous (nats), w this width
    #: (a constant of the step, tangent only). A slow crossing turns a modest
    #: tangent of the rise into a large shift of the interpolated arrival; 0
    #: (default): undamped.
    ad_shock_arrival_damping: float = 0.0

    #: Self-gravity / external-potential configuration (see GravityConfig).
    gravity_config: GravityConfig = GravityConfig()

    #: Explicit viscous diffusion term with the coefficient
    #: ``params.viscosity`` (see ``viscosity_type``), in both the
    #: finite-difference and the finite-volume solver.
    diffusion: bool = False

    #: Viscosity type - either kinematic or dynamic viscosity.
    viscosity_type: int = DYNAMIC_VISCOSITY

    #: Switch for explicit ohmic resistivity in the induction equation. The
    #: switch pairs with the coefficient of the same name,
    #: ``params.resistivity`` (eta). It is applied to the interface fields as
    #: the curl of an edge EMF; 3D finite-difference CT MHD with the isothermal
    #: EOS only, since no ohmic heating term is added to the energy equation.
    resistivity: bool = False

    #: Upwind dissipation of the constrained-transport edge EMFs on the
    #: interface fields (finite-difference MHD), in the form of the upwind CT of
    #: Londrillo & Del Zanna (2004): every edge EMF gets the jumps of the two
    #: interface-field components meeting there, reconstructed to the edge from
    #: both sides at fifth order, times half the local fast signal speed across
    #: the jump. The central high-order CT interpolations leave the grid-scale
    #: modes of the interface fields almost undamped, so that without it a field
    #: oblique to the grid lets grid-scale perturbations grow; the added term is
    #: O(dx^5) on smooth fields and keeps div B = 0 exactly (it is differenced by
    #: the CT's own discrete curl). Its strength is ``ct_upwind_strength``.
    #: Finite-difference MHD only.
    ct_upwind_dissipation: bool = False

    #: Strength kappa of ``ct_upwind_dissipation`` (1: the full upwind form,
    #: half the signal speed times the jump). The grid-scale instability is
    #: removed down to kappa = 0.01, while the added resistivity grows with
    #: kappa (subsonic dynamo: +2-5 % at 0.03, about x2.3 at 1; at Mach 10 none
    #: measurable below 1); 0.03 keeps a factor three of margin.
    ct_upwind_strength: float = 0.03

    #: Explicit thermal conduction term div(kappa grad T) in the energy
    #: equation (constant conductivity ``params.thermal_conductivity``, or
    #: ``kappa = rho alpha`` with ``conduction_density_weighted``; explicit
    #: integration). Currently only for finite difference mode.
    thermal_conduction: bool = False

    #: Interpret ``params.thermal_conductivity`` as the Athena-style
    #: DIFFUSIVITY ``alpha`` instead of a constant conductivity, i.e. use
    #: ``kappa = rho * alpha`` so the heat flux is ``-rho alpha grad T``
    #: (conservative face-flux form). This keeps the TEMPERATURE diffusivity
    #: ``chi = (gamma - 1) alpha`` density-INDEPENDENT — the convention used
    #: for thermal-instability / Field-length studies (AthenaK
    #: ``<hydro> alpha_iso``), where the constant-kappa form would instead
    #: suppress conduction exactly inside the cold dense clumps.
    conduction_density_weighted: bool = False

    #: Formal order of the conduction discretisation: 2 (default) or 4; any
    #: other value is rejected by ``finalize_config``. In a FINITE-DIFFERENCE
    #: scheme the state IS the pointwise value, so evaluating ``T = p/rho``
    #: (and ``kappa = rho*alpha``) pointwise is already exact -- the order is
    #: set purely by the derivative stencils. Order 4 uses the 4th-order central
    #: first derivative for the pointwise heat flux and the 4th-order
    #: conservative face interpolation
    #: ``(-F_{i-1} + 7F_i + 7F_{i+1} - F_{i+2})/12`` for its divergence (the
    #: same linear flux the WENO kernel uses), so it is consistent with the
    #: 5th-order hydro rather than reducing it to 2nd order.
    conduction_order: int = 2

    #: The size of the simulation box.
    box_size: Union[float, StaticFloatVector] = 1.0

    #: The number of cells in the simulation.
    num_cells: Union[int, StaticIntVector] = 400

    #: The reconstruction order is the number of
    #: cells on each side of the cell of interest
    #: used to calculate the gradients for the
    #: reconstruction at the interfaces.
    reconstruction_order: int = 1

    #: The limiter for the reconstruction of the classic finite-volume
    #: scheme. The VL2 scheme does not read it (see ``VAN_LEER``).
    limiter: int = MINMOD

    #: The Riemann solver used. Only for finite volume mode; ``HLLD`` is only
    #: available with the VL2 time integrator.
    riemann_solver: int = HLL

    #: Dimensional splitting / unsplit mode.
    #: Note that the UNSPLIT scheme currently
    #: interferes with energy conservation in settings
    #: with self-gravity.
    split: int = UNSPLIT

    #: Time integration method: RK2_SSP, MUSCL or VL2 for the finite-volume
    #: solver, RK4_SSP or RK4_LSRK for the finite-difference solver (which
    #: falls back to RK4_SSP for any other value).
    time_integrator: int = RK2_SSP

    # Explanation of the ghost cells
    #                                |---------|
    #                           |---------|
    # stencil              |---------|
    # cells            || 1g | 2g | 3c | 4g | 5g ||
    # reconstructions        |L  R|L  R|L  R|    |
    # fluxes                     -->  -->
    # update                      | 3c'|
    # --> all others are ghost cells

    #: The number of ghost cells.
    num_ghost_cells: int = reconstruction_order + 1

    #: Grid spacing.
    grid_spacing: float = box_size / num_cells

    #: Explicit boundary handling mode.
    boundary_handling: int = GHOST_CELLS

    #: Boundary settings for the simulation.
    boundary_settings: Union[NoneType, BoundarySettings1D, BoundarySettings] = None

    #: Enables a fixed timestep for the simulation
    #: based on the specified number of timesteps.
    fixed_timestep: bool = False

    #: Exactly reach the end time. In adaptive timestepping,
    #: one might otherwise overshoot.
    exact_end_time: bool = True

    #: Adds the sources with the current timestep to
    #: a hypothetical state to estimate the actual timestep.
    #: Useful for time-dependent sources, but additional
    #: computational overhead.
    source_term_aware_timestep: bool = False

    #: The number of timesteps for the fixed timestep mode.
    num_timesteps: int = 1000

    #: Use a maximum timestep in adaptive timestep mode.
    use_max_adaptive_timestep: bool = True

    #: The differentiation mode one whats to use
    #: the solver in (forwards or backwards).
    differentiation_mode: int = FORWARDS

    #: The number of checkpoints used in the setup
    #: with backwards differetiability and adaptive
    #: time stepping.
    num_checkpoints: int = 100

    #: Rematerialisation for reverse-mode AD: ``"none"`` (default), ``"stage"``
    #: or ``"axis"`` (see ``AD_REMAT_*``). Changes only what the backward pass
    #: stores versus recomputes; the primal is the same computation, and under
    #: forward-only evaluation or forward-mode AD ``jax.checkpoint`` is inlined
    #: (no cost). For a finite-difference WENO step, "stage" roughly halves and
    #: "axis" roughly quarters the temporary memory of the backward pass, at
    #: the cost of about one ("stage") or two ("axis") extra forward
    #: evaluations of the rematerialised pieces.
    ad_remat: str = AD_REMAT_NONE

    #: ``ad_remat == "axis"`` only: split each axis' flux + blend + divergence
    #: into this many slabs along a PERPENDICULAR axis (z for the x / y fluxes,
    #: y for the z flux; never the first spatial axis, the one a multi-GPU run
    #: splits), a ``lax.map`` with one ``jax.checkpoint`` per slab. The
    #: backward then holds one slab's WENO / blend internals at a time. Exact
    #: (every piece is a stencil along the flux axis only); 1 = off; a count
    #: that does not divide the slab axis falls back to 1.
    ad_remat_chunks: int = 1

    #: Reverse-mode memory of the passive-scalar block (``ad_remat != "none"``
    #: only; default False = unchanged). With many scalars, the backward of the
    #: scalar advection dominates the memory of a reverse-mode gradient: the
    #: masked sub-step scan's stack of ``max_passive_scalar_substeps``
    #: scalar-stack copies and the tie-preserving clips' mask residuals of the
    #: ratio recovery and of the shock history. With True: the sub-steps run in
    #: an equinox checkpointed while loop over the flow-derived count (two
    #: scalar-stack checkpoints instead of the cap's), and the ratio recovery /
    #: bounds and the shock-history update are ``jax.checkpoint``-ed (their
    #: masks are recomputed, not stored). Same arithmetic in the primal;
    #: derivatives equal up to rounding (a different XLA program).
    ad_scalar_lean: bool = False

    #: Return intermediate snapshots of the time evolution
    #: instead of only the final fluid state.
    return_snapshots: bool = False

    #: Snapshot settings
    snapshot_settings: SnapshotSettings = SnapshotSettings()

    #: Where the snapshots are stored. ``ON_DEVICE`` (default) keeps the
    #: snapshot diagnostics in preallocated device buffers and returns them
    #: at the end (the classic behaviour). ``TO_DISK`` instead streams each
    #: snapshot to disk via Orbax: the run is split into segments between the
    #: snapshot times, and the loop carry (primitive state, PRNG key, OU
    #: forcing field) plus the time is written to ``snapshot_storage_path``
    #: after each segment. Each device writes its own shard, so this scales
    #: to multiple devices / nodes. TO_DISK is forward-mode only.
    snapshot_storage_mode: int = ON_DEVICE

    #: Directory the Orbax checkpoints are written to / read from when
    #: ``snapshot_storage_mode == TO_DISK``. Required in that mode.
    snapshot_storage_path: Union[str, NoneType] = None

    #: Call a user given function on the snapshot data,
    #: e.g. for saving or plotting. Must have signature
    #: callback(time, state, registered_variables).
    activate_snapshot_callback: bool = False

    #: Return snapshots at specific time points.
    use_specific_snapshot_timepoints: bool = False

    #: The number of snapshots to return.
    num_snapshots: int = 10

    #: Fallback to the first order Godunov scheme.
    first_order_fallback: bool = False

    #: WENO nonlinear weights: ``False`` = classic Jiang-Shu
    #: ``alpha_k = c_k/(eps+IS_k)^2``, ``True`` = WENO-Z (Borges et al. 2008)
    #: ``alpha_k = c_k (1 + tau_5/(eps+IS_k))``. WENO-Z is scale invariant (the
    #: nonlinearity is a ratio of smoothness indicators, not a comparison
    #: against an absolute epsilon) and keeps the optimal linear weights at
    #: smooth extrema, where JS drops order and damps small-amplitude features.
    #: Implemented in the native and the Pallas kernels; derivatives of the
    #: Pallas backend go through the native tangent, which uses the same weights.
    weno_z: bool = False

    #: Absolute floor in the WENO smoothness denominators (JS and Z).
    weno_epsilon: float = 1e-7

    #: Evaluate the characteristic basis of the WENO projection at an
    #: ADMISSIBLE interface state: the interface sound speed comes from the
    #: averaged pressure, c^2 = gamma <p> / <rho>, and the enthalpy is rebuilt
    #: from it. ``False`` uses c^2 = (gamma - 1)(<h> - v^2/2) from an
    #: UNWEIGHTED enthalpy mean and a MASS-WEIGHTED velocity; that combination
    #: is not the state of any gas, is not Galilean invariant, and at a density
    #: jump (ratio >~ 10) carrying a velocity jump of a few sound speeds its
    #: c^2 is negative -- the clamp then zeroes the acoustic upwind correction
    #: exactly at the strongest jumps, so a cold dense slab driven at high Mach
    #: number into tenuous gas blows up within a few steps. Smooth-flow results
    #: of the two bases agree to the WENO dissipation level (the basis moves by
    #: O(dx^2)). Ideal gas only (the isothermal basis has a fixed sound speed).
    #: Native and Pallas.
    weno_admissible_face_state: bool = True

    #: Positivity-preserving WENO (Zhang & Shu 2012, J. Comput. Phys. 231,
    #: 2245), inside the reconstruction. With alpha the splitting speed of an
    #: interface, each split flux is f^+- = +-(alpha/2) w^+- with
    #: w^+- = q +- F/alpha - sum_s (1 - alpha_s/alpha) R_s L_s q. Every field
    #: that carries mass is split with the common alpha (the stencil's spectral
    #: radius), which makes the frozen-basis splitting monotone for every
    #: stencil cell; mass-free fields (hydro shear, isothermal-MHD Alfven) keep
    #: their own speed alpha_s. Each WENO face value is then pulled toward its
    #: upwind split state by the largest theta in [0, 1] keeping it, and its
    #: mirror about q +- F/alpha, admissible (positive density; positive
    #: pressure for an ideal gas; closed form, as pressure is concave). theta = 1
    #: in smooth flow. For ideal MHD a single q +- F/alpha is often NOT
    #: admissible at the fast speed when beta is low (Wu 2018, SIAM J. Numer.
    #: Anal. 56, 2124), so theta acts on the weighted parts of each cell's
    #: update instead: its own two mirror states per axis (the cell's flux
    #: cancels; base q_i) and its inflow states of ALL axes jointly (base: the
    #: axis-summed first-order inflow, in which the neighbours' magnetic-tension
    #: terms cancel up to the discrete div B; per axis they do not). The SSPRK stages then
    #: rebuild the cell-centred B from the faces, pressure held, so each
    #: increment starts from the state it was evaluated at. Each forward-Euler
    #: stage is positivity preserving for C_cfl <= 1/2 (sum-of-speeds CFL),
    #: i.e. 0.75 with SSPRK4 (ideal MHD: up to the rare inadmissible inflow
    #: base, where theta = 0). Implies ``weno_admissible_face_state``. Native
    #: and Pallas (not the fused WENO+divergence kernel, which is then skipped).
    weno_positivity_preserving: bool = False

    #: If > 0, ADD a relative contribution ``weno_epsilon_relative * (amx*|q|)^2``
    #: to the WENO epsilon, where ``q`` is the local characteristic variable and
    #: ``amx`` the family's dissipation coefficient — i.e. compare the
    #: smoothness indicators against the local DATA SCALE instead of an
    #: absolute constant. With ``weno_epsilon`` alone the weights are fully
    #: nonlinear whenever the variables are O(10) or larger, regardless of how
    #: smooth the solution is. NATIVE backend only (``finalize_config`` rejects
    #: it with the Pallas backend).
    weno_epsilon_relative: float = 0.0

    #: Differentiate through the WENO reconstruction with its nonlinear
    #: weights, characteristic eigenvectors and Lax-Friedrichs splitting speed
    #: FROZEN (``stop_gradient``): the tangent / adjoint is then that of the
    #: linear scheme the primal step actually used, the usual linearisation for
    #: WENO adjoints. The primal is unchanged. Needed for forward-mode
    #: derivatives through long runs with cold, near-uniform gas: there
    #: IS_k << weno_epsilon, d alpha / d IS ~ 2 / epsilon^3 ~ 1e21, and the
    #: float32 tangent overflows long before the end of such a run.
    weno_ad_frozen_weights: bool = False

    #: If > 0: on interfaces whose colder side is below this factor times
    #: ``params.minimum_specific_pressure``, take the DERIVATIVE of the flux
    #: through the monotone Lax-Friedrichs flux; the primal value is
    #: unchanged. In cold, kinetic-dominated gas (freely expanding ejecta, far
    #: below the temperature floor) the linearised WENO flux -- with or without
    #: frozen weights -- advances the tangent with a linear high-order scheme
    #: that is not stable at sharp density edges, and the tangent grows by
    #: orders of magnitude per year while the true perturbation does not. A
    #: tangent-only linearisation, so ``jax.jvp`` / ``jax.grad`` are those of a
    #: locally first-order scheme there. Lives in the cold-crush blend, so it
    #: requires ``positivity_config.coldcrush_blend`` (ideal gas).
    ad_tangent_llf_cold_factor: float = 0.0

    # physical modules

    #: Turbulent forcing configuration (see TurbulentForcingConfig).
    turbulent_forcing_config: TurbulentForcingConfig = TurbulentForcingConfig()

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()

    #: Cosmic rays
    cosmic_ray_config: CosmicRayConfig = CosmicRayConfig()

    #: The configuration for the cooling module.
    cooling_config: CoolingConfig = CoolingConfig()

    #: Frame tracking in z-direction
    #: shifting the frame to follow a
    #: turbulent radiative mixing layer
    frame_tracking: bool = False

    #: Configuration of the neural network force module.
    neural_net_force_config: NeuralNetForceConfig = NeuralNetForceConfig()

    #: Configuration of the CNN MHD corrector module.
    cnn_mhd_corrector_config: CNNMHDconfig = CNNMHDconfig()

    #: Configuration of the sink particle formation module.
    sink_particle_config: SinkParticleConfig = SinkParticleConfig()


def _parse_compute_capability(compute_capability):
    """
    Parse a compute capability such as ``"8.0"`` into the tuple ``(8, 0)``.
    Anything that is not of the form ``major.minor`` gives ``None``.
    """
    try:
        major, minor = str(compute_capability).strip().split(".")[:2]
        return int(major), int(minor)
    except (ValueError, TypeError):
        return None


def gpu_compute_capability_at_least_80() -> bool:
    """Return whether JAX runs on GPUs that all have compute capability >= 8.0.

    Compute capability 8.0 (Ampere) is the floor for the Triton kernels the
    Pallas backend compiles to, so this is the predicate that decides whether
    OPTIMAL_BACKEND resolves to PALLAS.

    The question is asked of JAX, not of the machine: what matters is the
    platform the kernels will be compiled for. ``JAX_PLATFORMS=cpu`` on a GPU
    node therefore answers False (the kernels would run on XLA:CPU, where
    Pallas only has an interpreter), which ``nvidia-smi`` could not tell.
    The capability is read from ``jax.devices()``; ``nvidia-smi`` is only a
    fallback for GPU devices that do not expose it. Any failure is treated as
    "not capable" so the safe NATIVE_JAX fallback is chosen.

    Returns:
        True if JAX's default backend is a GPU backend and every one of its
        devices reports compute capability >= 8.0, False otherwise.
    """
    try:
        devices = jax.devices()
    except RuntimeError:
        return False
    if not devices or any(device.platform != "gpu" for device in devices):
        return False
    capabilities = [
        _parse_compute_capability(getattr(device, "compute_capability", None))
        for device in devices
    ]
    if all(capability is not None for capability in capabilities):
        return all(capability >= (8, 0) for capability in capabilities)
    return _nvidia_smi_compute_capability_at_least_80()


def _nvidia_smi_compute_capability_at_least_80() -> bool:
    """
    Fallback of :func:`gpu_compute_capability_at_least_80`: ask ``nvidia-smi``
    whether every visible NVIDIA GPU has compute capability >= 8.0 (False if
    it cannot be queried).
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    compute_caps = []
    for line in output.strip().splitlines():
        major, minor = map(int, line.strip().split("."))
        compute_caps.append((major, minor))

    if not compute_caps:
        return False

    return all(compute_cap >= (8, 0) for compute_cap in compute_caps)


def finalize_config(config: SimulationConfig, state_shape) -> SimulationConfig:
    """Fill in derived configuration fields and validate the configuration.

    Resolves the values that depend on the actual state shape or on
    cross-field consistency: the positivity-protection defaults, the number
    of cells per axis, the grid spacing, the geometry- and solver-specific
    overrides, the master gravity switch, the boundary defaults, and the
    disk-snapshot and sink particle requirements. Combinations the solvers
    do not implement (e.g. cosmic rays with the finite-difference solver,
    self-gravity or non-Cartesian geometry with VL2, HLLD outside VL2) raise
    a ``ValueError``.

    Args:
        config: The user-supplied simulation configuration.
        state_shape: The shape of the (unpadded) primitive state array, used
            to derive ``num_cells`` per axis.

    Returns:
        The finalized simulation configuration.
    """

    # The positivity-preserving reconstruction evaluates its characteristic
    # basis at the admissible interface state, so it switches that on too.
    if config.weno_positivity_preserving and not config.weno_admissible_face_state:
        config = config._replace(weno_admissible_face_state=True)

    # Resolve the OPTIMAL_BACKEND request into a concrete backend before any
    # downstream code inspects ``config.backend_config.backend``. PALLAS needs
    # an Ampere-class (compute capability >= 8.0) GPU for its Triton kernels;
    # anywhere else we fall back to the portable NATIVE_JAX backend.
    if config.backend_config.backend == OPTIMAL_BACKEND:
        if gpu_compute_capability_at_least_80():
            print("OPTIMAL_BACKEND: using the PALLAS backend (GPU compute capability >= 8.0).")
            config = config._replace(
                backend_config=config.backend_config._replace(backend=PALLAS)
            )
        else:
            print(
                "OPTIMAL_BACKEND: using the NATIVE_JAX backend (JAX is not running on a "
                "compute capability >= 8.0 GPU)."
            )
            config = config._replace(
                backend_config=config.backend_config._replace(backend=NATIVE_JAX)
            )

    # Cosmic rays exist only in the finite-volume solver. Under the
    # finite-difference solver the registry does not add the CR variable, so
    # the CR code paths that still run (DSA injection, energy totals) index
    # variable -1 -- the pressure in 1D, the last passive scalar when scalars
    # are carried -- and the run silently returns a CR-free (and corrupted)
    # solution. Refuse instead.
    if config.cosmic_ray_config.cosmic_rays and config.solver_mode == FINITE_DIFFERENCE:
        raise ValueError(
            "cosmic_ray_config.cosmic_rays is not implemented for the "
            "finite-difference solver (the default solver_mode): the CR "
            "configuration would be silently ignored and the CR modules would "
            "write into variable -1. Use solver_mode=FINITE_VOLUME for cosmic "
            "rays."
        )

    # Reverse-mode / AD options.
    if config.ad_remat not in AD_REMAT_MODES:
        raise ValueError(
            f"ad_remat must be one of {AD_REMAT_MODES}, got {config.ad_remat!r}."
        )
    if int(config.ad_remat_chunks) < 1:
        raise ValueError(f"ad_remat_chunks must be >= 1, got {config.ad_remat_chunks!r}.")
    if config.ad_tangent_llf_cold_factor > 0.0 and not (
        config.positivity_config.coldcrush_blend and config.equation_of_state == IDEAL_GAS
    ):
        raise ValueError(
            "ad_tangent_llf_cold_factor acts inside the cold-crush flux blend; it "
            "needs positivity_config.coldcrush_blend=True and an ideal gas."
        )
    if config.passive_scalar_substep_loop not in (SUBSTEPS_AUTO, SUBSTEPS_DYNAMIC, SUBSTEPS_MASKED):
        raise ValueError(
            "passive_scalar_substep_loop must be SUBSTEPS_AUTO, SUBSTEPS_DYNAMIC "
            f"or SUBSTEPS_MASKED, got {config.passive_scalar_substep_loop!r}."
        )
    if int(config.max_passive_scalar_substeps) < 1:
        raise ValueError("max_passive_scalar_substeps must be >= 1.")
    if (config.passive_scalar_substep_loop == SUBSTEPS_DYNAMIC
            and config.differentiation_mode == BACKWARDS
            and (config.num_passive_scalars > 0 or config.track_shock_history)):
        print(
            "NOTE: passive_scalar_substep_loop = SUBSTEPS_DYNAMIC uses a traced "
            "trip count (a while loop); reverse-mode AD through the passive "
            "scalars will fail. Use SUBSTEPS_AUTO or SUBSTEPS_MASKED."
        )
    if config.ad_smooth_shock_latch and not config.track_shock_history:
        print("NOTE: ad_smooth_shock_latch has no effect without track_shock_history.")
    if config.ad_shock_arrival and config.ad_smooth_shock_latch:
        raise ValueError(
            "ad_shock_arrival and ad_smooth_shock_latch are alternative tangents of the "
            "shock history; switch on at most one."
        )
    if config.ad_shock_arrival and not config.track_shock_history:
        print("NOTE: ad_shock_arrival has no effect without track_shock_history.")

    # The conduction source only distinguishes order 4 from everything else,
    # so any other value would silently run the second-order discretisation.
    if config.conduction_order not in (2, 4):
        raise ValueError(f"conduction_order must be 2 or 4, got {config.conduction_order!r}.")

    # weno_epsilon_relative is native-only.
    if config.weno_epsilon_relative > 0.0 and config.backend_config.backend == PALLAS:
        raise ValueError(
            "weno_epsilon_relative is implemented for the NATIVE_JAX backend "
            "only; pass BackendConfig(backend=NATIVE_JAX)."
        )

    # Approximate-rsqrt fast path: the forward MHD WENO kernel is switched to the
    # refined approximate rsqrt, but the native tangent behind the Pallas
    # derivatives still uses IEEE sqrt.  Purely forward runs (convergence, runtime) are unaffected; warn
    # only so a reverse-mode/AD user knows the forward and backward differ by the
    # rsqrt's ~1 ULP.
    if config.backend_config.use_approximate_rsqrt:
        print(
            "NOTE: backend_config.use_approximate_rsqrt is ON — the MHD WENO "
            "forward kernel uses approximate rsqrt while its tangent uses IEEE "
            "sqrt, so reverse-mode gradients are ~1 ULP inconsistent with the "
            "forward. Fine for forward-only runs; turn off for exact AD."
        )

    if jax.config.jax_enable_x64:
        config = config._replace(numerical_precision=DOUBLE_PRECISION)
    else:
        config = config._replace(numerical_precision=SINGLE_PRECISION)

    # set the number of cells
    if config.dimensionality == 1:
        num_cells_x = state_shape[-1]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, -1, -1))
    if config.dimensionality == 2:
        num_cells_x, num_cells_y = state_shape[-2:]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, num_cells_y, -1))
    elif config.dimensionality == 3:
        num_cells_x, num_cells_y, num_cells_z = state_shape[-3:]
        config = config._replace(num_cells=StaticIntVector(num_cells_x, num_cells_y, num_cells_z))

    if isinstance(config.box_size, float):
        config = config._replace(
            box_size=StaticFloatVector(
                config.box_size,
                config.box_size,
                config.box_size
            )
        )

    # For now we assume the grid spacing is the same in all dimensions, so the
    # scalar ``grid_spacing`` is taken from the x-axis and the other axes are
    # only checked for consistency below. This restriction can be lifted once
    # the solver accepts a per-axis grid-spacing vector.
    grid_spacing_vec = config.box_size / config.num_cells

    if config.dimensionality == 1:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
    elif config.dimensionality == 2:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not math.isclose(grid_spacing_vec.x, grid_spacing_vec.y):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )
    elif config.dimensionality == 3:
        config = config._replace(grid_spacing=grid_spacing_vec.x)
        if not (
            math.isclose(grid_spacing_vec.x, grid_spacing_vec.y)
            and math.isclose(grid_spacing_vec.x, grid_spacing_vec.z)
        ):
            raise ValueError(
                "For now, we assume the grid spacing is the same in all dimensions. "
                f"Got grid spacing {grid_spacing_vec}."
            )

    if config.geometry == SPHERICAL:
        print(
            "For spherical geometry, only HLL is currently supported. Also, only the "
            "unsplit mode has been tested."
        )
        # SPHERICAL is intrinsically 1D in this code; pick the x component
        # so grid_spacing stays a scalar (otherwise CFL divisions blow up
        # because StaticFloatVector can't be divided by a scalar wave speed).
        config = config._replace(grid_spacing=(config.box_size / config.num_cells).x)

        if config.riemann_solver != HLL:
            print("Setting HLL Riemann solver for spherical geometry.")
            config = config._replace(riemann_solver=HLL)

        if config.split != SPLIT:
            print("Setting unsplit mode for spherical geometry")
            config = config._replace(split=SPLIT)

        if config.limiter == VAN_ALBADA or config.limiter == VAN_ALBADA_PP:
            print("Setting minmod limiter for spherical geometry")
            config = config._replace(limiter=MINMOD)

        if config.time_integrator != MUSCL:
            print("Setting MUSCL time integrator for spherical geometry")
            config = config._replace(time_integrator=MUSCL)

    # master gravity switch: active if self-gravity and/or an external
    # potential is used. This gates the (shared) gravity source-term machinery.
    config = config._replace(gravity_config=config.gravity_config._replace(
        gravity=config.gravity_config.self_gravity
        or config.gravity_config.external_potential
    ))

    uses_vl2 = config.solver_mode == FINITE_VOLUME and config.time_integrator == VL2

    if config.gravity_config.gravity and (config.limiter != MINMOD) and not uses_vl2:
        print(
            "Curiously, in self-gravitating systems, the VAN_ALBADA limiters seem to cause crashes."
        )
        print("Setting MINMOD limiter for gravity.")
        config = config._replace(limiter=MINMOD)

    # Athena++'s gravity coupling needs the per-stage Riemann mass fluxes of
    # the VL2 scheme; the other couplings are those of the finite-difference
    # solver (the classic finite-volume schemes ignore the choice).
    if (
        config.gravity_config.gravity
        and config.gravity_config.self_gravity_version == ATHENA_SECOND_ORDER
        and not uses_vl2
    ):
        raise ValueError(
            "The ATHENA_SECOND_ORDER self-gravity coupling is only available "
            "with the VL2 finite-volume scheme (time_integrator=VL2)."
        )

    # The AthenaPK-equivalent VL2 finite-volume scheme. Its piecewise-linear
    # stencil needs two ghost cells; with periodic boundaries on every axis the
    # wrap-around is done by rolling the arrays instead (as in the FD solver).
    if uses_vl2:

        # The VL2 update replaces the whole finite-volume step: it has no
        # geometric source terms and only fluxes for the gas and field
        # variables, so these combinations would be silently ignored
        # (geometry) or fail on the extra state variables. Spherical geometry
        # was already switched to MUSCL above. Gravity enters through
        # Athena++'s coupling, the only one built on the VL2 stages.
        if (
            config.gravity_config.gravity
            and config.gravity_config.self_gravity_version != ATHENA_SECOND_ORDER
        ):
            print("Setting the Athena++ (ATHENA_SECOND_ORDER) gravity coupling for the VL2 scheme.")
            config = config._replace(
                gravity_config=config.gravity_config._replace(
                    self_gravity_version=ATHENA_SECOND_ORDER
                )
            )
        if config.gravity_config.gravity and config.gravity_config.work_flux_correction:
            raise ValueError(
                "The flux-corrected gravitational work (work_flux_correction) "
                "is implemented for the finite-difference couplings only."
            )
        if config.geometry != CARTESIAN:
            raise ValueError("The VL2 scheme is implemented for Cartesian geometry only.")
        if config.wind_config.trace_wind_density or config.cosmic_ray_config.cosmic_rays:
            raise ValueError(
                "The VL2 scheme does not evolve the stellar-wind tracer "
                "(wind_config.trace_wind_density) or the cosmic-ray variable "
                "(cosmic_ray_config.cosmic_rays)."
            )

        if config.split != UNSPLIT:
            print("Setting unsplit mode for the VL2 scheme.")
            config = config._replace(split=UNSPLIT)
        # The VL2 scheme does not read ``limiter``: its corrector always uses
        # the van Leer slope, or donor cell with ``first_order_fallback``.
        # Setting VAN_LEER only makes the configuration report the
        # reconstruction actually in use.
        if config.limiter != VAN_LEER and not config.first_order_fallback:
            print("Setting the VAN_LEER (AthenaPK PLM) limiter for the VL2 scheme.")
            config = config._replace(limiter=VAN_LEER)
        if config.mhd and config.riemann_solver not in (HLLD, HLL, LAX_FRIEDRICHS):
            print("Setting the HLLD Riemann solver for VL2 MHD.")
            config = config._replace(riemann_solver=HLLD)
        if not config.mhd and config.riemann_solver not in (HLLC, HLL, LAX_FRIEDRICHS):
            print("Setting the HLLC Riemann solver for VL2 hydrodynamics.")
            config = config._replace(riemann_solver=HLLC)
        if config.riemann_solver == LAX_FRIEDRICHS and not config.first_order_fallback:
            raise ValueError(
                "As in AthenaPK, the LAX_FRIEDRICHS solver of the VL2 scheme is "
                "only available with donor-cell reconstruction "
                "(first_order_fallback=True)."
            )
        periodic_1d = BoundarySettings1D(
            left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
        )
        if config.dimensionality == 1:
            fully_periodic = config.boundary_settings == periodic_1d
        else:
            fully_periodic = config.boundary_settings is not None and all(
                axis_settings == periodic_1d
                for axis_settings in tuple(config.boundary_settings)[: config.dimensionality]
            )
        if fully_periodic:
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=2)

        # The Poisson solve takes the density array as it is, so with ghost
        # cells it would treat the halo as part of the domain. Isolated
        # self-gravity in a periodic hydrodynamic box is set by
        # ``poisson_manual_open_boundaries``.
        if config.gravity_config.self_gravity and config.boundary_handling == GHOST_CELLS:
            raise ValueError(
                "Self-gravity with the VL2 scheme requires periodic hydrodynamic "
                "boundaries on every axis (use poisson_manual_open_boundaries "
                "for an isolated potential)."
            )

    # HLLD is implemented only inside the VL2 scheme; the classic
    # finite-volume Riemann-solver dispatch does not know it.
    elif config.solver_mode == FINITE_VOLUME and config.riemann_solver == HLLD:
        raise ValueError(
            "The HLLD Riemann solver is only available with the VL2 time "
            "integrator (time_integrator=VL2)."
        )

    # The upwind EMF dissipation acts on the constrained-transport interface
    # fields, which only the finite-difference MHD scheme carries.
    if config.ct_upwind_dissipation and config.solver_mode != FINITE_DIFFERENCE:
        raise ValueError("ct_upwind_dissipation requires finite-difference MHD")

    # A retried step needs a step size to reduce, and its data-dependent loop
    # has no reverse-mode derivative.
    if config.positivity_config.step_retry and (
        config.fixed_timestep or config.differentiation_mode != FORWARDS
    ):
        raise ValueError(
            "positivity_config.step_retry requires adaptive time stepping and "
            "differentiation_mode=FORWARDS"
        )

    # Finite-difference-specific checks.
    if config.solver_mode == FINITE_DIFFERENCE:

        # The WENO5 stencil reaches three cells beyond each interface, so the
        # default ghost-cell count (reconstruction_order + 1 = 2) is too few:
        # 1D runs then read wrapped / stale values at the edges, and smooth
        # periodic problems converged at first order. Same rule as 2D/3D.
        if config.dimensionality == 1:
            if config.boundary_settings == BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ):
                config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
            elif config.boundary_handling == GHOST_CELLS:
                config = config._replace(num_ghost_cells=max(config.num_ghost_cells, 4))

        if config.dimensionality == 3 and config.boundary_settings == BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ):
            # Fully periodic boundaries are enforced more cheaply by rolling the
            # arrays (PERIODIC_ROLL) than by maintaining explicit ghost cells.
            print(
                "For 3D simulations with periodic boundaries, setting boundary handling to "
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 3:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        if config.dimensionality == 2 and config.boundary_settings == BoundarySettings(
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
            BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ),
        ):
            # Fully periodic boundaries are enforced more cheaply by rolling the
            # arrays (PERIODIC_ROLL) than by maintaining explicit ghost cells.
            print(
                "For 2D simulations with periodic boundaries, setting boundary handling to "
                "PERIODIC_ROLL and num_ghost_cells to 0 for better performance."
            )
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)
        else:
            if config.dimensionality == 2:
                config = config._replace(boundary_handling=GHOST_CELLS, num_ghost_cells=4)

        # The FD scheme has two supported time integrators: the SSPRK4
        # Spiteri-Ruuth 3-register scheme (default) and the Carpenter-Kennedy
        # 2N-storage LSRK4 ("RK4_LSRK") which trades CFL margin for one fewer
        # full-state buffer.  Anything else falls back to SSPRK4.
        if config.time_integrator not in (RK4_SSP, RK4_LSRK):
            print(
                "Setting time integrator to RK4_SSP for finite difference solver mode."
            )
            config = config._replace(time_integrator=RK4_SSP)

        if config.boundary_handling == PERIODIC_ROLL:
            config = config._replace(num_ghost_cells=0)

        if config.boundary_handling == GHOST_CELLS and (
            config.diffusion or config.thermal_conduction or config.resistivity
        ):
            config = config._replace(num_ghost_cells=max(config.num_ghost_cells, 6))

        if config.ct_upwind_dissipation and not config.mhd:
            raise ValueError("ct_upwind_dissipation requires finite-difference MHD")

        if config.resistivity:
            if not config.mhd or config.dimensionality != 3:
                raise ValueError("resistivity requires 3D finite-difference MHD")
            if config.equation_of_state != ISOTHERMAL:
                raise ValueError(
                    "resistivity is implemented for the ISOTHERMAL EOS only: the "
                    "ohmic heating eta J^2 is not added to the energy equation"
                )

    # Pick sensible default boundary conditions when the user left them unset.
    if config.boundary_settings is None:
        if config.geometry == CARTESIAN:
            print("Automatically setting open boundaries for Cartesian geometry.")
            if config.dimensionality == 1:
                config = config._replace(
                    boundary_settings=BoundarySettings1D(
                        left_boundary=OPEN_BOUNDARY, right_boundary=OPEN_BOUNDARY
                    )
                )
            else:
                config = config._replace(boundary_settings=BoundarySettings())
        elif config.geometry == SPHERICAL and config.dimensionality == 1:
            print(
                "Automatically setting reflective left and open right boundary for "
                "spherical geometry."
            )
            config = config._replace(
                boundary_settings=BoundarySettings1D(
                    left_boundary=REFLECTIVE_BOUNDARY, right_boundary=OPEN_BOUNDARY
                )
            )

    if config.wind_config.stellar_wind:
        print(
            "For stellar wind simulations, we need source term aware timesteps, turning on."
        )
        config = config._replace(source_term_aware_timestep=True)

    # Disk-snapshot (Orbax) mode requirements.
    if config.snapshot_storage_mode == TO_DISK:
        if not config.snapshot_storage_path:
            raise ValueError(
                "snapshot_storage_mode == TO_DISK requires a non-empty "
                "snapshot_storage_path (the directory the Orbax checkpoints "
                "are written to)."
            )
        if config.differentiation_mode != FORWARDS:
            raise ValueError(
                "snapshot_storage_mode == TO_DISK is forward-mode only; "
                "set differentiation_mode = FORWARDS."
            )

    # Sink particle formation requirements. The formation checks of Federrath
    # et al. (2010) are formulated in 3D and need the self-gravity potential.
    if config.sink_particle_config.sink_particles:
        if config.dimensionality != 3:
            raise ValueError("Sink particles are only supported in 3D.")
        if not config.gravity_config.self_gravity:
            raise ValueError("Sink particles require self_gravity = True.")
        if not config.state_struct:
            raise ValueError(
                "Sink particles require state_struct = True; the sinks are "
                "passed in and returned in the StateStruct."
            )

    return config


def riemann_solver_to_string(riemann_solver: int) -> str:
    """Return the human-readable name of a Riemann-solver constant."""
    if riemann_solver == HLL:
        return "HLL"
    elif riemann_solver == HLLC:
        return "HLLC"
    elif riemann_solver == HLLC_LM:
        return "HLLC_LM"
    elif riemann_solver == LAX_FRIEDRICHS:
        return "Lax-Friedrichs"
    elif riemann_solver == HYBRID_HLLC:
        return "Hybrid HLLC"
    elif riemann_solver == AM_HLLC:
        return "AM HLLC"
    elif riemann_solver == HLLD:
        return "HLLD"


def limiter_to_string(limiter: int) -> str:
    """Return the human-readable name of a slope-limiter constant."""
    if limiter == MINMOD:
        return "Minmod"
    elif limiter == SUPERBEE:
        return "Superbee"
    elif limiter == OSHER:
        return "Osher"
    elif limiter == DOUBLE_MINMOD:
        return "Double Minmod"
    elif limiter == VAN_ALBADA:
        return "Van Albada"
    elif limiter == VAN_ALBADA_PP:
        return "Van Albada PP"
    elif limiter == VAN_LEER:
        return "Van Leer"


def solver_mode_to_string(solver_mode: int) -> str:
    """Return the short label (``"FV"`` / ``"FD"``) of a solver-mode constant."""
    if solver_mode == FINITE_VOLUME:
        return "FV"
    elif solver_mode == FINITE_DIFFERENCE:
        return "FD"


def config_to_string(config: SimulationConfig) -> str:
    """Return a compact one-line description of the solver configuration."""
    if config.solver_mode == FINITE_VOLUME:
        return (
            f"FV, {riemann_solver_to_string(config.riemann_solver)}, "
            f"{limiter_to_string(config.limiter)}, {config.num_cells.x} cells"
        )
    elif config.solver_mode == FINITE_DIFFERENCE:
        return f"FD, {config.num_cells.x} cells"
