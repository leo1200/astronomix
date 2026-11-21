"""
Static simulation configuration.

Defines :class:`SimulationConfig` — the bundle of options that, unlike the
simulation parameters, necessitate recompilation when changed — together with
the integer-coded enumerations they reference (backends, solver/boundary/Riemann
modes, positivity modes, ...), the small geometry vector helpers, the containers
of the solver settings (:class:`FVSolverConfig`, :class:`FDSolverConfig`) and of
the physics modules, and the ``finalize_config`` pass that fills in derived
fields and validates the configuration.
"""

# general
import math
import subprocess

# typing
from types import NoneType
from typing import Callable, NamedTuple, Optional, Tuple, Union
from jaxtyping import Array, Float

# jax
import jax

# astronomix containers
from astronomix._modules._cooling.cooling_options import CoolingConfig
from astronomix._modules._relaxation_zone._relaxation_zone_options import RelaxationZoneConfig
from astronomix._modules._stellar_wind.stellar_wind_options import WindConfig
from astronomix._modules._turbulence_driving._turbulence_driving_options import TurbulenceDrivingConfig

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
# finite-difference scheme itself comes from
# ``FDSolverConfig.weno_positivity_preserving``; the
# floor remains for the finite-volume solver and for near-vacuum, strongly
# magnetised cells (``maximum_alfven_speed``). HARD_FLOOR clamps density (and,
# for an ideal gas, pressure) pointwise; it is not conservative.
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

# Time integrators of the finite-volume solver
# (``FVSolverConfig.time_integrator``).
#: Unsplit scheme: two forward-Euler stages (SSP-RK2, Heun).
RK2_SSP = 0
#: Dimensionally split scheme: a Strang sequence of sweeps along the axes,
#: each with the Hancock half-step predictor on the face states. The scheme of
#: the spherical 1D grid and of the wind density tracer.
MUSCL_HANCOCK = 1
#: The second-order van Leer predictor-corrector of AthenaPK / Athena++
#: (Stone & Gardiner 2009): a donor-cell half step followed by a full step
#: with van Leer-limited piecewise-linear reconstruction (donor cell in both
#: steps with ``reconstruction=DONOR_CELL``). Selecting it switches the
#: finite-volume solver to the AthenaPK-equivalent scheme, which for MHD
#: carries the cell-centred field together with the GLM cleaning scalar psi
#: (Dedner et al. 2002). The scheme is Cartesian. Self-gravity and an external
#: potential enter through Athena++'s coupling (``ATHENA_SECOND_ORDER``); the
#: scheme has no stellar-wind-tracer term, and
#: ``finalize_config`` rejects that combination (spherical geometry, as for
#: every spherical configuration, is switched to MUSCL_HANCOCK instead).
VL2 = 2

# Time integrators of the finite-difference solver
# (``FDSolverConfig.time_integrator``). Their values are disjoint from the
# finite-volume group, so that ``finalize_config`` recognises an integrator of
# the other solver.
#: The five-stage, fourth-order SSP Runge-Kutta scheme of Spiteri & Ruuth
#: (three registers).
SSPRK54 = 10
#: The three-stage, third-order SSP Runge-Kutta scheme of Shu & Osher (two
#: registers); with ``scheme = RIEMANN_FLUX`` only.
SSPRK33 = 11
#: The 2N-storage fourth-order Runge-Kutta scheme of Carpenter & Kennedy: one
#: full-state buffer fewer than SSPRK54, at a smaller CFL margin; with
#: ``scheme = WENO_FLUX_SPLITTING`` only.
LSRK4 = 12

# Schemes of the finite-difference solver (``FDSolverConfig.scheme``).
#: WENO reconstruction of the split characteristic fluxes, HOW-MHD constrained
#: transport: the production scheme.
WENO_FLUX_SPLITTING = 0
#: WENO face states of the characteristic variables, a Riemann flux with a
#: high-order correction and a guard towards the first-order Lax-Friedrichs
#: flux, flux-form constrained transport (``_riemann_flux.py``). MHD in three
#: Cartesian dimensions, hydrodynamics in one to three.
RIEMANN_FLUX = 1

# Riemann solvers of ``scheme = RIEMANN_FLUX`` (``FDSolverConfig.riemann_solver``)
# next to ``HLL`` and ``HLLC``; their numbers continue the group of Riemann solvers below.
#: HLLEM_T for a barotropic gas, HLLD_BOUNDED for ideal-gas MHD, HLLC for
#: ideal-gas hydrodynamics.
RIEMANN_BY_EOS = -1
#: HLL with the anti-diffusion of the Alfven and slow waves limited by the
#: tangential field (barotropic gas); without a field, of the shear waves.
HLLEM_T = 5
#: HLLD with bounded outer speeds (ideal gas).
HLLD_BOUNDED = 6

# Face states of the finite-volume solver (``FVSolverConfig.reconstruction``).
DONOR_CELL = 0
PIECEWISE_LINEAR = 1

# Slope limiters of the piecewise-linear reconstruction
# (``FVSolverConfig.limiter``).
MINMOD = 0
#: The harmonic-mean (van Leer) slope of AthenaPK's / Athena++'s piecewise
#: linear reconstruction. This is the slope of the VL2 scheme:
#: ``finalize_config`` sets ``limiter = VAN_LEER`` for VL2 (its choice between
#: piecewise-linear and donor-cell reconstruction is ``reconstruction``).
#: ``RK2_SSP`` and the split scheme (``MUSCL_HANCOCK``) offer it as well.
VAN_LEER = 1
SUPERBEE = 2
DOUBLE_MINMOD = 3

# Riemann solvers (``FVSolverConfig.riemann_solver``)
HLL = 0
HLLC = 1
LAX_FRIEDRICHS = 2
#: The HLLD solver of Miyoshi & Kusano (2005) for ideal MHD, in AthenaPK's
#: GLM form. Only implemented in the VL2 scheme (which also maps HLL, HLLC and
#: LAX_FRIEDRICHS onto AthenaPK's HLLE, HLLC and LLF solvers); ``finalize_config``
#: rejects it for the other finite-volume time integrators.
HLLD = 3
#: The five-wave relaxation solver of Bouchut, Klingenberg & Waagan (2007, 2010), with two fluxes of the magnetic
#: field where the normal field jumps. The solver of the split scheme with
#: ``divergence_control = INDUCTION_SOURCE``; ideal gas and isothermal gas.
HLL5R = 4

# Control of the magnetic divergence in the finite-volume solver
# (``FVSolverConfig.divergence_control``).
#: Cell-centred field with the cleaning scalar psi (Dedner et al. 2002).
GLM = 0
#: Face-centred field, updated with edge electromotive forces.
CONSTRAINED_TRANSPORT = 1
#: Cell-centred field without a cleaning scalar: Powell's term ``u div B`` in the induction equation only
#: (Waagan 2009), as the pair of magnetic fluxes of ``HLL5R`` and a source per cell. The divergence is not kept
#: at zero: it moves with the gas, and ``parabolic_cleaning`` diffuses it. The MHD form of the split scheme
#: (``MUSCL_HANCOCK``), with full sweeps x, y, z whose order is reversed every other step.
INDUCTION_SOURCE = 2

# Halo exchange of the multi-device Pallas kernels
# (``BackendConfig.halo_exchange``).
#: Two collective permutes over the whole periodic ring of devices.
PPERMUTE = 0
#: Two phases of all-to-alls over disjoint neighbour pairs.
PAIR_ALL_TO_ALL = 1


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

# Viscosity types
KINEMATIC_VISCOSITY = 0
DYNAMIC_VISCOSITY = 1

# Equation of state
IDEAL_GAS = 0
ISOTHERMAL = 1
# p = K rho^Gamma, no energy equation (finite volume only so far)
POLYTROPIC = 2

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
    ``FDSolverConfig.weno_positivity_preserving`` (every stage a convex
    combination of admissible states). What is left here is not a positivity
    patch: the per-step floor serves the finite-volume solver, and the clamps
    keep wave-speed and time-step estimates finite.
    """

    #: State floor applied once per step before the evolve (on the primitive
    #: state): ``POSITIVITY_NONE`` or ``POSITIVITY_HARD_FLOOR``. With
    #: ``time_integrator=VL2`` the hard floor also switches on AthenaPK's
    #: density / pressure floors inside every stage's primitive recovery.
    per_step_mode: int = POSITIVITY_NONE

    #: Read-only density/pressure clamp in the flux / eigenvalue / timestep
    #: estimates. It never modifies the evolved state (the step-end primitive
    #: recovery is unclamped).
    clamp_in_estimates: bool = True

    #: Redo a time step at half the step size, from the same state and
    #: driving state, when it leaves a non-finite value or a non-positive
    #: density (or pressure, for an ideal gas), up to ``MAXIMUM_STEP_RETRIES``
    #: times. The positivity-preserving WENO keeps every stage positive only
    #: while no stage's splitting speeds exceed those the step was sized with;
    #: a near-vacuum cell that accelerates within a single step (its mass
    #: drains faster than its momentum) breaks that, and the step is then
    #: repeated instead of carrying a negative density into the next one.
    #: Adaptive time stepping and forward-mode runs only.
    step_retry: bool = False


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
    #: axis. ``PPERMUTE``: two collective permutes over the whole periodic
    #: ring of devices. ``PAIR_ALL_TO_ALL``: two phases of all-to-alls over
    #: disjoint neighbour pairs (even-odd, then odd-even including the
    #: wraparound pair), so every collective spans two devices only and its
    #: communicator does not grow with the device count. Both move the same
    #: halo data and give bit-identical results; an axis split over an odd
    #: number of devices always uses the ring. The pairwise exchange is ~5%
    #: faster at 8-32 GPUs and 5-13% at 128 GPUs on JUPITER (GH200); its
    #: per-pair NCCL communicators need device memory beyond the XLA pool, so
    #: multi-node runs should leave ~10% of the GPU memory to NCCL.
    halo_exchange: int = PAIR_ALL_TO_ALL
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


class FVSolverConfig(NamedTuple):
    """Settings of the finite-volume solver (``solver_mode = FINITE_VOLUME``)."""

    #: Time integration method: ``RK2_SSP`` (unsplit), ``MUSCL_HANCOCK``
    #: (dimensionally split) or ``VL2`` (the AthenaPK-equivalent scheme).
    #: Note that the unsplit RK2_SSP scheme currently interferes with energy
    #: conservation in settings with self-gravity.
    time_integrator: int = RK2_SSP

    #: Face states: ``PIECEWISE_LINEAR``, or ``DONOR_CELL`` for the first-order
    #: Godunov scheme.
    reconstruction: int = PIECEWISE_LINEAR

    #: The limiter of the piecewise-linear reconstruction (``MINMOD``,
    #: ``SUPERBEE``, ``DOUBLE_MINMOD`` or ``VAN_LEER``). The VL2 scheme always
    #: uses ``VAN_LEER`` (see there).
    limiter: int = MINMOD

    #: The Riemann solver used: ``HLL``, ``HLLC`` or ``LAX_FRIEDRICHS``;
    #: ``HLLD`` is only available with the VL2 time integrator, ``HLL5R``
    #: only with ``divergence_control = INDUCTION_SOURCE``.
    riemann_solver: int = HLL

    #: VL2 scheme only (MHD): ``GLM`` cleaning or ``CONSTRAINED_TRANSPORT``,
    #: as in Athena's VL2 + CT integrator (Stone & Gardiner 2009, New
    #: Astronomy 14, 139). The face-centred field is the primary variable
    #: (state rows ``interface_magnetic_field_index``; index ``i`` holds the
    #: value at ``i + 1/2``, as in the finite-difference layout), the
    #: cell-centred field is the average of the two faces of a cell, the edge
    #: electromotive forces come from the upwind formula of Gardiner & Stone
    #: (2005, 2008), and there is no cleaning scalar psi. The face divergence
    #: stays at rounding. Periodic boundaries on every axis, no gravity, native
    #: JAX only (see ``_finite_volume/_divergence/_constrained_transport.py``).
    #: ``INDUCTION_SOURCE`` is the MHD form of the split scheme (see the constant).
    divergence_control: int = GLM

    #: With ``divergence_control = INDUCTION_SOURCE``: once per step the field gets ``dt eta grad(div B)`` with
    #: ``eta = lambda h / (2 d)`` in ``d`` dimensions and ``lambda`` the largest signal speed of the step, the
    #: diffusive cleaning of the eight-wave unit of FLASH.
    parabolic_cleaning: bool = False

    #: VL2 scheme only: add the non-conservative Dedner et al. (2002) sources,
    #: ``-(div B) B`` in the momentum and ``-B . grad(psi)`` in the energy
    #: equation, to the parabolic damping of psi (AthenaPK
    #: ``glmmhd_source = dedner_extended``). The default is the plain damping
    #: only (AthenaPK's default ``dedner_plain``).
    glm_extended_source: bool = False

    #: AthenaPK's first-order flux correction. After each stage (each sweep
    #: of the split scheme), every cell whose update would produce a
    #: non-positive density or pressure has all its face fluxes replaced by
    #: first-order local Lax-Friedrichs fluxes (repeated up to four times,
    #: exactly as AthenaPK's ``first_order_flux_correct``). Cartesian
    #: geometry only.
    first_order_flux_correction: bool = False


class FDSolverConfig(NamedTuple):
    """Settings of the finite-difference solver (``solver_mode = FINITE_DIFFERENCE``)."""

    #: The scheme: ``WENO_FLUX_SPLITTING`` (characteristic flux splitting,
    #: HOW-MHD constrained transport) or ``RIEMANN_FLUX`` (WENO face states,
    #: Riemann flux with high-order correction and guard, flux-form constrained
    #: transport). Under ``RIEMANN_FLUX`` the fields ``weno_z``,
    #: ``weno_admissible_face_state``, ``weno_positivity_preserving``,
    #: ``ct_upwind_dissipation``, ``ct_upwind_strength`` and ``dual_energy``
    #: must keep their defaults.
    scheme: int = WENO_FLUX_SPLITTING

    #: Time integration method: ``SSPRK54``, ``SSPRK33`` (``RIEMANN_FLUX``
    #: only) or ``LSRK4`` (``WENO_FLUX_SPLITTING`` only).
    time_integrator: int = SSPRK54

    #: WENO nonlinear weights: ``False`` = classic Jiang-Shu
    #: ``alpha_k = c_k/(eps+IS_k)^2``, ``True`` = WENO-Z (Borges et al. 2008)
    #: ``alpha_k = c_k (1 + tau_5/(eps+IS_k))``. WENO-Z is scale invariant (the
    #: nonlinearity is a ratio of smoothness indicators, not a comparison
    #: against an absolute epsilon) and keeps the optimal linear weights at
    #: smooth extrema, where JS drops order and damps small-amplitude features.
    #: Implemented in the native and the Pallas kernels; derivatives of the
    #: Pallas backend go through the native tangent, which uses the same weights.
    weno_z: bool = False

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
    #: stencil cell; for an isothermal gas the mass-free fields (shear waves,
    #: MHD Alfven waves) keep their own speed alpha_s. Each WENO face value is then pulled toward its
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

    #: Differentiate through the WENO reconstruction with its nonlinear
    #: weights, characteristic eigenvectors and Lax-Friedrichs splitting speed
    #: FROZEN (``stop_gradient``): the tangent / adjoint is then that of the
    #: linear scheme the primal step actually used, the usual linearisation for
    #: WENO adjoints. The primal is unchanged. Needed for forward-mode
    #: derivatives through long runs with cold, near-uniform gas: there
    #: IS_k << WENO_EPSILON, d alpha / d IS ~ 2 / epsilon^3 ~ 1e21, and the
    #: float32 tangent overflows long before the end of such a run.
    weno_ad_frozen_weights: bool = False

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

    #: Dual-energy formalism (Bryan et al. 1995 switch) for adiabatic FD
    #: hydro and MHD. When True a separately-advected internal-energy density
    #: ``g`` is carried through the time loop and used in the WENO pressure
    #: recovery wherever ``e_E/E < DUAL_ENERGY_ETA``, so high-Mach / low-beta
    #: float32 cancellation of ``e_int = E - KE - ME`` does not corrupt the
    #: pressure.
    dual_energy: bool = False

    #: Riemann solver of ``scheme = RIEMANN_FLUX``: ``RIEMANN_BY_EOS``
    #: (``HLLEM_T`` for a barotropic gas; for the ideal gas ``HLLD_BOUNDED``
    #: in MHD, ``HLLC`` in hydrodynamics), one of these or ``HLL``.
    riemann_solver: int = RIEMANN_BY_EOS


class PassiveScalarConfig(NamedTuple):
    """Passive scalars of the finite-difference solver, advected operator-split
    once per step."""

    #: Number of user-defined passive scalars: per-parcel labels advected with
    #: the flow (``dC/dt + v.grad C = 0``) that do not act back on it —
    #: composition mass fractions, an ejecta/circumstellar discriminator, and so
    #: on. Finite-difference solver only; they are carried as the last variables
    #: of the state and advected operator-split (WENO5 + SSP-RK3) in
    #: ``_passive_scalars.py``.
    num_scalars: int = 0

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
    bounds: Tuple[Optional[Tuple[float, float]], ...] = ()

    #: CFL number for the passive-scalar sub-steps. The scalar advection is
    #: operator-split, so it does not inherit the hydro timestep's safety: the
    #: hydro CFL is set by ``|u| + c``, while what constrains this is ``|u|`` and
    #: — the binding one — positivity of the companion density under
    #: ``h |div v| <= 0.5``. The number of sub-steps is derived from the flow at
    #: every step, so a benign step takes exactly one and costs nothing.
    cfl: float = 0.4

    #: Cap on the derived sub-step count, so a single pathological cell cannot
    #: stall a run. Hitting the cap means the scalars are being integrated
    #: outside their stability limit in some cells; the physical bounds in
    #: ``bounds`` are the backstop for that. Under
    #: ``differentiation_mode == BACKWARDS`` the loop runs to this static cap
    #: and skips the sub-steps beyond the derived count (same primal), so lower
    #: it in reverse-mode runs whose flow never needs more than one or two.
    max_substeps: int = 8


class DiffusionConfig(NamedTuple):
    """Explicit diffusive physics; the coefficients are in ``SimulationParams``."""

    #: Explicit viscous diffusion term with the coefficient
    #: ``params.viscosity`` (see ``viscosity_type``), in both the
    #: finite-difference and the finite-volume solver.
    viscosity: bool = False

    #: Viscosity type - either kinematic or dynamic viscosity.
    viscosity_type: int = DYNAMIC_VISCOSITY

    #: Switch for explicit ohmic resistivity in the induction equation. The
    #: switch pairs with the coefficient of the same name,
    #: ``params.resistivity`` (eta). It is applied to the interface fields as
    #: the curl of an edge EMF; 3D finite-difference CT MHD with the isothermal
    #: EOS only, since no ohmic heating term is added to the energy equation.
    resistivity: bool = False

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


class SimulationConfig(NamedTuple):
    """
    Configuration object for the simulation.
    The simulation configuration are parameters defining
    the simulation where changes necessitate recompilation.
    """

    # ---------------------------- problem ----------------------------

    #: The number of dimensions of the simulation.
    dimensionality: int = 1

    #: The geometry of the simulation.
    geometry: int = CARTESIAN

    #: The size of the simulation box.
    box_size: Union[float, StaticFloatVector] = 1.0

    #: The number of cells in the simulation.
    num_cells: Union[int, StaticIntVector] = 400

    #: Boundary settings for the simulation.
    boundary_settings: Union[NoneType, BoundarySettings1D, BoundarySettings] = None

    #: The equation of state for the simulation: ``IDEAL_GAS``, ``ISOTHERMAL``
    #: (p = c_s^2 rho, ``params.isothermal_sound_speed``) or ``POLYTROPIC``
    #: (p = K rho^Gamma, ``params.polytropic_constant`` and
    #: ``params.polytropic_exponent``). The last two are barotropic: no energy
    #: equation and no pressure slot in the state.
    equation_of_state: int = IDEAL_GAS

    #: Magnetohydrodynamics switch.
    mhd: bool = False

    # ---------------------------- solver -----------------------------

    #: Basic solver mode, either finite volume or finite difference.
    #: Defaults to the finite-difference HOW-MHD scheme (Jeongbhin Seo,
    #: Dongsu Ryu, 2023), which is the recommended solver.
    solver_mode: int = FINITE_DIFFERENCE

    #: The settings of the solver: an :class:`FVSolverConfig` for
    #: ``FINITE_VOLUME``, an :class:`FDSolverConfig` for ``FINITE_DIFFERENCE``.
    #: ``None`` stands for the default container of the mode, which
    #: ``finalize_config`` fills in.
    solver_config: Union[NoneType, FVSolverConfig, FDSolverConfig] = None

    #: State floors and estimate clamps (see PositivityConfig). Positivity of
    #: the finite-difference scheme itself comes from
    #: ``FDSolverConfig.weno_positivity_preserving``.
    positivity_config: PositivityConfig = PositivityConfig()

    #: Compute-backend configuration (see :class:`BackendConfig`): backend
    #: choice + Pallas/Triton kernel knobs.  Accessed as
    #: ``config.backend_config.backend`` / ``.pallas_block_shape`` / etc.
    backend_config: BackendConfig = BackendConfig()

    # ------------------------ physics modules ------------------------

    #: Self-gravity / external-potential configuration (see GravityConfig).
    gravity_config: GravityConfig = GravityConfig()

    #: Viscosity, resistivity and thermal conduction (see DiffusionConfig).
    diffusion_config: DiffusionConfig = DiffusionConfig()

    #: Passive scalars (see PassiveScalarConfig).
    passive_scalar_config: PassiveScalarConfig = PassiveScalarConfig()

    #: The configuration for the stellar wind module.
    wind_config: WindConfig = WindConfig()

    #: The configuration for the cooling module.
    cooling_config: CoolingConfig = CoolingConfig()

    #: The configuration of the turbulence driving (TurbGen port, see
    #: TurbulenceDrivingConfig); its parameters are
    #: ``SimulationParams.turbulence_driving_params``.
    turbulence_driving_config: TurbulenceDrivingConfig = TurbulenceDrivingConfig()

    #: The relaxation zone (see RelaxationZoneConfig): a region pulled towards
    #: a target state once per step; its target and rate are
    #: ``SimulationParams.relaxation_zone_params``.
    relaxation_zone_config: RelaxationZoneConfig = RelaxationZoneConfig()

    #: User-supplied update of the primitive state, called once per time
    #: step before the hydro update (after the stellar wind and cooling
    #: updates, before viscosity):
    #: ``per_step_update(primitive_state, dt, current_time, config, params,
    #: helper_data, registered_variables)`` returns the new primitive state
    #: (same shape, ghost cells included). ``helper_data`` only holds the
    #: fields the built-in modules need. Differentiable inputs (for example
    #: network weights) go into ``params.per_step_update_params``.
    per_step_update: Callable | None = None

    # ---------------- time stepping and differentiation ----------------

    #: Enables a fixed timestep for the simulation
    #: based on the specified number of timesteps.
    fixed_timestep: bool = False

    #: Exactly reach the end time. In adaptive timestepping,
    #: one might otherwise overshoot.
    exact_end_time: bool = True

    #: The number of timesteps for the fixed timestep mode.
    num_timesteps: int = 1000

    #: The differentiation mode one whats to use
    #: the solver in (forwards or backwards).
    differentiation_mode: int = FORWARDS

    #: Reverse-mode memory saving of the finite-difference solver: wrap every
    #: Runge-Kutta-stage right-hand side (and that of the passive-scalar
    #: advection) in ``jax.checkpoint``. The backward pass then keeps only the
    #: stage inputs and recomputes each stage's internals, about one extra
    #: forward evaluation for roughly half the temporary memory. The primal is
    #: the same computation; without reverse mode the wrapper is inlined.
    rematerialize_stages: bool = False

    #: The number of checkpoints used in the setup
    #: with backwards differetiability and adaptive
    #: time stepping.
    num_checkpoints: int = 100

    # ----------------------- output and runtime ------------------------

    #: Return intermediate snapshots of the time evolution
    #: instead of only the final fluid state.
    return_snapshots: bool = False

    #: Snapshot settings
    snapshot_settings: SnapshotSettings = SnapshotSettings()

    #: Where the snapshots are stored. ``ON_DEVICE`` (default) keeps the
    #: snapshot diagnostics in preallocated device buffers and returns them
    #: at the end (the classic behaviour). ``TO_DISK`` instead streams each
    #: snapshot to disk via Orbax: the run is split into segments between the
    #: snapshot times, and the loop carry (primitive state and the state of
    #: the turbulence driving) plus the time is written to ``snapshot_storage_path``
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

    #: Memory analysis of the main time integration
    #: function
    memory_analysis: bool = False

    #: Print the elapsed time of the simulation
    print_elapsed_time: bool = False

    #: Activate progress bar
    progress_bar: bool = False

    #: Debug runtime errors, throws exceptions
    #: on e.g. negative pressure or density.
    #: Significantly reduces performance.
    runtime_debugging: bool = False

    #: Donate the state arrays to the time integration function
    #: to reduce memory allocations. If activated, the
    #: initial state arrays will be invalid after
    #: the simulation.
    donate_state: bool = False

    # ------- derived by ``finalize_config``, not set by the user -------

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
    num_ghost_cells: int = 2

    #: Grid spacing.
    grid_spacing: float = box_size / num_cells

    #: Explicit boundary handling mode.
    boundary_handling: int = GHOST_CELLS

    #: Adds the sources with the current timestep to
    #: a hypothetical state to estimate the actual timestep.
    #: Useful for time-dependent sources, but additional
    #: computational overhead.
    source_term_aware_timestep: bool = False


#: The time integrators of each solver (``time_integrator`` of its container).
FV_TIME_INTEGRATORS = (RK2_SSP, MUSCL_HANCOCK, VL2)
FD_TIME_INTEGRATORS = (SSPRK54, SSPRK33, LSRK4)


def get_solver_config(config: SimulationConfig) -> Union[FVSolverConfig, FDSolverConfig]:
    """Return the solver settings of a configuration.

    This is ``config.solver_config``, or the default container of
    ``config.solver_mode`` when that field is ``None``. A container or a time
    integrator that belongs to the other solver raises a ``ValueError``.
    ``finalize_config`` stores the result, so a finalized configuration always
    carries its container; code that runs before it (the variable registry,
    the construction of the initial state) calls this function instead.
    """
    solver_config = config.solver_config
    if config.solver_mode == FINITE_VOLUME:
        if solver_config is None:
            return FVSolverConfig()
        if not isinstance(solver_config, FVSolverConfig):
            raise ValueError(
                "solver_mode=FINITE_VOLUME needs an FVSolverConfig as solver_config, "
                f"got {type(solver_config).__name__}."
            )
        if solver_config.time_integrator not in FV_TIME_INTEGRATORS:
            raise ValueError(
                "FVSolverConfig.time_integrator must be RK2_SSP, MUSCL_HANCOCK or VL2, "
                f"got {solver_config.time_integrator!r}."
            )
    elif config.solver_mode == FINITE_DIFFERENCE:
        if solver_config is None:
            return FDSolverConfig()
        if not isinstance(solver_config, FDSolverConfig):
            raise ValueError(
                "solver_mode=FINITE_DIFFERENCE needs an FDSolverConfig as solver_config, "
                f"got {type(solver_config).__name__}."
            )
        if solver_config.time_integrator not in FD_TIME_INTEGRATORS:
            raise ValueError(
                "FDSolverConfig.time_integrator must be SSPRK54, SSPRK33 or LSRK4, "
                f"got {solver_config.time_integrator!r}."
            )
    return solver_config


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


def _rounded_for_kernels(layers: int, config: SimulationConfig) -> int:
    """
    Kernels (hydrodynamics in 2D and 3D, MHD in 3D): the block is the largest power of two per axis that divides
    the padded grid, and with too few cells in it the run falls back to the native path without a message. A
    multiple of 4 keeps the padded extents multiples of 8 where the cell counts are, as with the old count 4: a
    full block of 128 cells in 3D, 64 cells in 2D (6 layers: 64 cells in 3D, native in 2D; 7: no block at all).
    """
    if config.dimensionality == 3 or (config.dimensionality == 2 and not config.mhd):
        layers += -layers % 4
    return layers


def _module_reach(config: SimulationConfig) -> int:
    """
    Cells to either side of a cell that the modules read between two boundary fills of the finite-volume
    schemes and of the scheme with a Riemann flux: the explicit sources of a stage (``_explicit_sources.py``)
    and the advection of the passive scalars.
    """
    reach = 0
    diffusion = config.diffusion_config
    if diffusion.thermal_conduction:
        # The fourth-order flux: a fourth-order slope (2) interpolated onto the faces (2); the second-order
        # forms difference the neighbours (1).
        reach = 4 if diffusion.conduction_order == 4 else 1
    if diffusion.viscosity or diffusion.resistivity:
        # A sixth-order derivative (3) of sixth-order derivatives (3): the viscous stress, the curl of eta J on
        # a cell-centred field and the divergence of the resistive Poynting flux.
        reach = max(reach, 6)
    gravity = config.gravity_config
    if gravity.gravity and config.solver_mode == FINITE_VOLUME and config.solver_config.time_integrator != VL2:
        # The sixth-order slope of the potential (3) of the SIMPLE_SOURCE coupling (VL2 couples gravity itself).
        reach = max(reach, 3)
    if gravity.gravity and gravity.self_gravity_version == FOURTH_ORDER_CONSERVATIVE:
        # The sixth-order slope of the potential (3), averaged onto the faces (1).
        reach = max(reach, 4)
    if config.passive_scalar_config.num_scalars > 0:
        # One SSPRK3 sub-step per step without a boundary fill inside: three right-hand sides with the WENO5
        # reach 3. (A sub-cycled step would need 9 per sub-step.)
        reach = max(reach, 9)
    return reach


def _required_ghost_cells(config: SimulationConfig) -> int:
    """
    The smallest number of ghost-cell layers with which every stencil read between two boundary fills sees
    filled cells, so that the result does not change when layers are added. All stencils wrap around the padded
    array: with fewer layers the cells next to a boundary read the opposite edge.
    """
    if config.solver_mode == FINITE_VOLUME:
        # Every stage ends with a boundary fill. The flux at a face reads the limited slopes of its two cells,
        # and a slope reads both neighbours of its cell: 2 (first-order flux correction and GLM included). The
        # explicit sources of a stage read the stage's state, filled before the stage.
        return max(2, _module_reach(config))

    solver = config.solver_config
    if solver.scheme == RIEMANN_FLUX:
        # Every gas, both integrators (each fills the boundary before every right-hand side). One right-hand
        # side reads four layers: the guarded flux at a face reads the cells from three below to four above it,
        # the update of a cell therefore four cells to either side. In MHD each of those cells holds a field
        # rebuilt from the faces three below to two above it (seven layers); the constrained transport and the
        # energy flux read six, the resistive electromotive force (sixth-order difference of edge values 2.5,
        # interpolation to the edge 1.5, sixth-order slope 3) seven. MHD is 3D only, so its count is 7 rounded
        # up for the kernels: 8. The explicit sources are part of the right-hand side.
        return _rounded_for_kernels(max(7 if config.mhd else 4, _module_reach(config)), config)

    # WENO_FLUX_SPLITTING. Both integrators fill the boundary before every right-hand side, so the count is
    # the reach of one right-hand side. Cells of the state it reads to either side of the cell it updates:
    # the WENO5 flux difference 3, the fourth-order conduction flux 4, the viscous flux (a sixth-order
    # derivative of sixth-order derivatives) 6, the fourth-order gravitational work 4 (the sixth-order slope
    # of the potential, averaged onto the faces).
    layers = 3
    if config.diffusion_config.thermal_conduction and config.diffusion_config.conduction_order == 4:
        layers = 4
    gravity = config.gravity_config
    if gravity.gravity and gravity.self_gravity_version == FOURTH_ORDER_CONSERVATIVE:
        layers = 4
    if config.diffusion_config.viscosity:
        layers = 6
    if solver.dual_energy:
        # The internal energy of the dual-energy formalism is advected once per step from one cell to either side
        # and not filled afterwards, and the WENO flux recovers the pressure from it in every cell it reads: one
        # layer more than the flux differences reach.
        layers = max(layers, 4)
    if config.mhd:
        # Cells that the update of a transverse face field reads along one axis: the sixth-order difference of the
        # edge-averaged electric field reads edge values 3.5 cells away, and an edge value the fluxes of the
        # cells up to 2.5 further: 6. The positivity-preserving flux of an ideal gas reads one cell further
        # along its own axis (7) and, where the inflow of a cell is summed over the axes (not in 1D, not with
        # the dual energy), three cells across it, interpolated to the edge from 1.5 cells away (8). The
        # cell-centred field in these cells is rebuilt from the normal face field 2.5 cells further: 3 more
        # layers (9, 10, 11). The Ohmic term reads the transverse face field 7 cells away (sixth-order
        # difference of edge values 2.5, interpolation to the edge 1.5, sixth-order slope 3): within these.
        field_reach = 6
        if solver.weno_positivity_preserving and config.equation_of_state == IDEAL_GAS:
            field_reach = 8 if config.dimensionality > 1 and not solver.dual_energy else 7
        layers = max(layers, 3 + field_reach)
    if config.passive_scalar_config.num_scalars > 0:
        # The scalar advection is one SSPRK3 sub-step per step without a boundary fill inside: three
        # right-hand sides with the WENO5 reach 3. (A sub-cycled step would need 9 per sub-step.)
        layers = max(layers, 9)
    return _rounded_for_kernels(layers, config)


def finalize_config(config: SimulationConfig, state_shape) -> SimulationConfig:
    """Fill in derived configuration fields and validate the configuration.

    Resolves the values that depend on the actual state shape or on
    cross-field consistency: the solver settings (the default container of
    the solver mode when ``solver_config`` is ``None``), the
    positivity-protection defaults, the number
    of cells per axis, the grid spacing, the box of the turbulence driving,
    the geometry- and solver-specific overrides, the master gravity switch, the boundary defaults, and the
    disk-snapshot requirements. Combinations the solvers do not implement
    (e.g. self-gravity or non-Cartesian geometry with VL2, HLLD outside VL2,
    a solver container or a time integrator of the other solver) raise a
    ``ValueError``.

    Args:
        config: The user-supplied simulation configuration.
        state_shape: The shape of the (unpadded) primitive state array, used
            to derive ``num_cells`` per axis.

    Returns:
        The finalized simulation configuration.
    """

    # The solver settings: the default container of the mode when none was
    # given; a container or a time integrator of the other solver raises.
    config = config._replace(solver_config=get_solver_config(config))
    finite_volume = config.solver_mode == FINITE_VOLUME
    finite_difference = config.solver_mode == FINITE_DIFFERENCE

    # The positivity-preserving reconstruction evaluates its characteristic
    # basis at the admissible interface state, so it switches that on too.
    if (
        finite_difference
        and config.solver_config.weno_positivity_preserving
        and not config.solver_config.weno_admissible_face_state
    ):
        config = config._replace(
            solver_config=config.solver_config._replace(weno_admissible_face_state=True)
        )

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

    if int(config.passive_scalar_config.max_substeps) < 1:
        raise ValueError("passive_scalar_config.max_substeps must be >= 1.")

    # The conduction source only distinguishes order 4 from everything else,
    # so any other value would silently run the second-order discretisation.
    if config.diffusion_config.conduction_order not in (2, 4):
        raise ValueError(
            "diffusion_config.conduction_order must be 2 or 4, "
            f"got {config.diffusion_config.conduction_order!r}."
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

    # The modes of the turbulence driving are those of the box.
    if config.turbulence_driving_config.turbulence_driving:
        if config.geometry != CARTESIAN:
            raise ValueError("turbulence_driving_config.turbulence_driving requires geometry=CARTESIAN.")
        config = config._replace(
            turbulence_driving_config=config.turbulence_driving_config._replace(
                dimensionality=config.dimensionality,
                box_size=tuple(config.box_size)[: config.dimensionality],
            )
        )

    if config.geometry == SPHERICAL:
        print(
            "For spherical geometry, only HLL and the dimensionally split "
            "MUSCL_HANCOCK integrator are currently supported."
        )
        # SPHERICAL is intrinsically 1D in this code; pick the x component
        # so grid_spacing stays a scalar (otherwise CFL divisions blow up
        # because StaticFloatVector can't be divided by a scalar wave speed).
        config = config._replace(grid_spacing=(config.box_size / config.num_cells).x)

        if finite_volume:
            if config.solver_config.riemann_solver != HLL:
                print("Setting HLL Riemann solver for spherical geometry.")
                config = config._replace(
                    solver_config=config.solver_config._replace(riemann_solver=HLL)
                )

            if config.solver_config.time_integrator != MUSCL_HANCOCK:
                print("Setting the MUSCL_HANCOCK time integrator for spherical geometry")
                config = config._replace(
                    solver_config=config.solver_config._replace(
                        time_integrator=MUSCL_HANCOCK
                    )
                )

        # A finite-difference configuration with spherical geometry has
        # always run SSPRK54, whatever integrator was asked for (before the
        # solver containers the spherical override set the split
        # finite-volume integrator, which the finite-difference block then
        # replaced by its default). Kept so that no result changes.
        elif finite_difference and config.solver_config.time_integrator != SSPRK54:
            print("Setting the SSPRK54 time integrator for spherical geometry")
            config = config._replace(
                solver_config=config.solver_config._replace(time_integrator=SSPRK54)
            )

    # master gravity switch: active if self-gravity and/or an external
    # potential is used. This gates the (shared) gravity source-term machinery.
    config = config._replace(gravity_config=config.gravity_config._replace(
        gravity=config.gravity_config.self_gravity
        or config.gravity_config.external_potential
    ))

    uses_vl2 = finite_volume and config.solver_config.time_integrator == VL2

    # The trial updates of the first-order flux correction use the Cartesian
    # flux divergence.
    if (
        finite_volume
        and config.geometry != CARTESIAN
        and config.solver_config.first_order_flux_correction
    ):
        raise ValueError(
            "The first-order flux correction (first_order_flux_correction) is "
            "implemented for Cartesian geometry only."
        )

    # The MHD form of the split scheme (preset BEATTIE_REFERENCE). The HLL5R solver and the induction source are
    # the two halves of one discretisation: the solver returns the pair of magnetic fluxes that the source
    # completes.
    induction_source = finite_volume and config.solver_config.divergence_control == INDUCTION_SOURCE
    if induction_source and config.geometry != CARTESIAN:
        raise ValueError("divergence_control=INDUCTION_SOURCE is implemented for Cartesian geometry only.")
    if finite_volume and (config.solver_config.riemann_solver == HLL5R) != induction_source:
        raise ValueError(
            "riemann_solver=HLL5R and divergence_control=INDUCTION_SOURCE belong together: set both (preset "
            "BEATTIE_REFERENCE) or neither."
        )
    if finite_volume and config.solver_config.parabolic_cleaning and not induction_source:
        raise ValueError("parabolic_cleaning is the divergence cleaning of divergence_control=INDUCTION_SOURCE.")
    if induction_source:
        if not config.mhd:
            raise ValueError("divergence_control=INDUCTION_SOURCE requires mhd=True.")
        if config.solver_config.time_integrator != MUSCL_HANCOCK:
            raise ValueError(
                "divergence_control=INDUCTION_SOURCE is the MHD form of the split scheme: use "
                "time_integrator=MUSCL_HANCOCK."
            )
        if config.equation_of_state == POLYTROPIC:
            raise ValueError(
                "divergence_control=INDUCTION_SOURCE is tested for the ideal gas and the isothermal gas only: use "
                "equation_of_state=IDEAL_GAS or ISOTHERMAL."
            )
        if config.solver_config.first_order_flux_correction:
            raise ValueError(
                "The first-order flux correction replaces fluxes by Lax-Friedrichs fluxes of the GLM form, which "
                "have no counterpart of the induction source: switch first_order_flux_correction off."
            )
        # The modules enter through the explicit sources around the sweeps; the MHD layout has no tracer row.
        if config.wind_config.trace_wind_density:
            raise ValueError(
                "wind_config.trace_wind_density is not implemented with divergence_control=INDUCTION_SOURCE (the "
                "MHD layout of the split scheme has no tracer row): switch it off."
            )
        # A fully periodic box runs on the periodic-roll layout, which the kernels and a device mesh need; other
        # boxes keep their ghost cells.
        periodic_sides = BoundarySettings1D(left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY)
        if config.dimensionality == 1:
            sides_per_axis = (config.boundary_settings,)
        else:
            sides_per_axis = tuple(config.boundary_settings or ())[: config.dimensionality]
        if sides_per_axis and all(sides == periodic_sides for sides in sides_per_axis):
            config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)

    if finite_volume and config.mhd and not (uses_vl2 or induction_source):
        raise ValueError(
            "Finite-volume MHD is available with the VL2 scheme (time_integrator=VL2) and with the split scheme "
            "of the preset BEATTIE_REFERENCE (time_integrator=MUSCL_HANCOCK, divergence_control=INDUCTION_SOURCE)."
        )

    if (
        finite_volume
        and config.gravity_config.gravity
        and config.solver_config.limiter != MINMOD
        and not uses_vl2
    ):
        print("Setting MINMOD limiter for gravity.")
        config = config._replace(solver_config=config.solver_config._replace(limiter=MINMOD))

    # Athena++'s gravity coupling needs the per-stage Riemann mass fluxes of
    # the VL2 scheme; the other couplings are those of the finite-difference
    # solver.
    if (
        config.gravity_config.gravity
        and config.gravity_config.self_gravity_version == ATHENA_SECOND_ORDER
        and not uses_vl2
    ):
        raise ValueError(
            "The ATHENA_SECOND_ORDER self-gravity coupling is only available "
            "with the VL2 finite-volume scheme (time_integrator=VL2)."
        )
    # RK2_SSP and MUSCL_HANCOCK take gravity from the explicit sources, added
    # after a stage without its mass fluxes, which the conservative couplings
    # read.
    if (
        finite_volume
        and not uses_vl2
        and config.gravity_config.gravity
        and config.gravity_config.self_gravity_version != SIMPLE_SOURCE
    ):
        print("Setting the SIMPLE_SOURCE gravity coupling for the RK2_SSP and MUSCL_HANCOCK schemes.")
        config = config._replace(
            gravity_config=config.gravity_config._replace(self_gravity_version=SIMPLE_SOURCE)
        )

    constrained_transport = (
        finite_volume and config.solver_config.divergence_control == CONSTRAINED_TRANSPORT
    )

    if constrained_transport and not uses_vl2:
        raise ValueError(
            "divergence_control=CONSTRAINED_TRANSPORT is an option of the VL2 "
            "finite-volume scheme (time_integrator=VL2)."
        )

    # The AthenaPK-equivalent VL2 finite-volume scheme. Its piecewise-linear
    # stencil needs two ghost cells; with periodic boundaries on every axis the
    # wrap-around is done by rolling the arrays instead (as in the FD solver).
    if uses_vl2:

        # The VL2 update replaces the whole finite-volume step: it has no
        # geometric source terms and only fluxes for the gas and field
        # variables, so these combinations would be silently ignored
        # (geometry) or fail on the extra state variables. Spherical geometry
        # was already switched to MUSCL_HANCOCK above. Gravity enters through
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
        if config.wind_config.trace_wind_density:
            raise ValueError(
                "The VL2 scheme does not evolve the stellar-wind tracer "
                "(wind_config.trace_wind_density)."
            )

        donor_cell = config.solver_config.reconstruction == DONOR_CELL
        # The VL2 scheme is AthenaPK's: its corrector always uses the van Leer
        # slope, or donor cell with ``reconstruction=DONOR_CELL``. The stage
        # reads ``limiter``, so VAN_LEER is set here.
        if config.solver_config.limiter != VAN_LEER and not donor_cell:
            print("Setting the VAN_LEER (AthenaPK PLM) limiter for the VL2 scheme.")
            config = config._replace(
                solver_config=config.solver_config._replace(limiter=VAN_LEER)
            )
        if config.mhd and config.solver_config.riemann_solver not in (
            HLLD, HLL, LAX_FRIEDRICHS
        ):
            print("Setting the HLLD Riemann solver for VL2 MHD.")
            config = config._replace(
                solver_config=config.solver_config._replace(riemann_solver=HLLD)
            )
        if not config.mhd and config.solver_config.riemann_solver not in (
            HLLC, HLL, LAX_FRIEDRICHS
        ):
            print("Setting the HLLC Riemann solver for VL2 hydrodynamics.")
            config = config._replace(
                solver_config=config.solver_config._replace(riemann_solver=HLLC)
            )
        if config.solver_config.riemann_solver == LAX_FRIEDRICHS and not donor_cell:
            raise ValueError(
                "As in AthenaPK, the LAX_FRIEDRICHS solver of the VL2 scheme is "
                "only available with donor-cell reconstruction "
                "(reconstruction=DONOR_CELL)."
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
            config = config._replace(boundary_handling=GHOST_CELLS)

        # VL2 with constrained transport: the staggered update is written for
        # the periodic-roll layout only and has no counterpart of the GLM
        # sources or of the gravity coupling.
        if constrained_transport:
            if not config.mhd:
                raise ValueError("divergence_control=CONSTRAINED_TRANSPORT requires mhd=True.")
            if not fully_periodic:
                raise ValueError(
                    "VL2 with constrained transport (divergence_control="
                    "CONSTRAINED_TRANSPORT) is implemented for periodic boundaries "
                    "on every axis only."
                )
            if config.gravity_config.gravity:
                raise ValueError(
                    "VL2 with constrained transport (divergence_control="
                    "CONSTRAINED_TRANSPORT) has no gravity coupling."
                )
            if config.solver_config.glm_extended_source:
                raise ValueError(
                    "glm_extended_source belongs to the GLM cleaning, which "
                    "divergence_control=CONSTRAINED_TRANSPORT replaces."
                )

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

    # HLLD is an MHD solver, and finite-volume MHD runs under VL2 only.
    elif finite_volume and config.solver_config.riemann_solver == HLLD:
        raise ValueError(
            "The HLLD Riemann solver is only available with the VL2 time "
            "integrator (time_integrator=VL2)."
        )

    # A barotropic gas (ISOTHERMAL, POLYTROPIC) has no energy equation. The finite-volume framework carries it
    # with the two solvers that need no energy jump and with gravity; both finite-difference schemes carry it.
    if finite_volume and config.equation_of_state != IDEAL_GAS:
        if config.solver_config.riemann_solver not in (HLL, LAX_FRIEDRICHS, HLL5R):
            raise ValueError(
                "HLLC and HLLD are Riemann solvers of the ideal gas: use riemann_solver=HLL (or LAX_FRIEDRICHS) "
                "with a barotropic equation of state (ISOTHERMAL, POLYTROPIC)."
            )
    if config.equation_of_state != IDEAL_GAS:
        # The stellar wind injects thermal energy (its tracer follows the wind).
        acts_on_the_energy_equation = (
            ("cooling_config.cooling", config.cooling_config.cooling),
            ("diffusion_config.thermal_conduction", config.diffusion_config.thermal_conduction),
            ("gravity_config.time_centred_energy_source", config.gravity_config.time_centred_energy_source),
            ("wind_config.stellar_wind", config.wind_config.stellar_wind),
            ("wind_config.trace_wind_density", config.wind_config.trace_wind_density),
        )
        for option, is_set in acts_on_the_energy_equation:
            if is_set:
                raise ValueError(
                    f"{option} acts on the energy equation, which a barotropic gas (ISOTHERMAL, POLYTROPIC) does "
                    "not have: switch it off, or use equation_of_state=IDEAL_GAS."
                )
        snapshots = config.snapshot_settings
        thermal_diagnostics = (
            ("snapshot_settings.return_internal_energy", snapshots.return_internal_energy),
            ("snapshot_settings.return_temperature_pdf", snapshots.return_temperature_pdf),
        )
        for option, is_set in thermal_diagnostics:
            if is_set:
                raise ValueError(
                    f"{option} reads the pressure of an ideal gas, which a barotropic gas (ISOTHERMAL, "
                    "POLYTROPIC) does not carry: switch it off, or use equation_of_state=IDEAL_GAS."
                )

    # The physics modules (``_modules/``) in every scheme: what each needs.
    if config.diffusion_config.resistivity and not config.mhd:
        raise ValueError("diffusion_config.resistivity acts on the magnetic field: use mhd=True.")
    if config.geometry != CARTESIAN:
        # The stencils of the diffusive sources and of the scalar advection are Cartesian.
        cartesian_only = (
            ("diffusion_config.viscosity", config.diffusion_config.viscosity),
            ("diffusion_config.resistivity", config.diffusion_config.resistivity),
            ("diffusion_config.thermal_conduction", config.diffusion_config.thermal_conduction),
            ("passive_scalar_config.num_scalars > 0", config.passive_scalar_config.num_scalars > 0),
        )
        for option, is_set in cartesian_only:
            if is_set:
                raise ValueError(f"{option} is implemented for geometry=CARTESIAN only: switch it off.")
    if config.relaxation_zone_config.relaxation_zone and config.mhd:
        # Relaxing the magnetic field cell by cell towards a target would
        # break div B = 0.
        raise ValueError("relaxation_zone_config.relaxation_zone is not implemented for MHD: switch it off.")
    if config.wind_config.stellar_wind and config.dimensionality != 3:
        # The 3D wind is a source of the stages of every scheme; the 1D wind is the injection of the spherical
        # grid, which the split finite-volume scheme carries.
        if not (config.dimensionality == 1 and config.geometry == SPHERICAL and finite_volume):
            raise ValueError(
                "wind_config.stellar_wind is implemented in 3D (every scheme) and on the spherical 1D grid "
                "(solver_mode=FINITE_VOLUME, geometry=SPHERICAL): use one of them."
            )

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
    if finite_difference:

        # Layout: fully periodic boundaries are served by rolling the arrays, every other case by ghost cells,
        # whose number of layers is set below (``_required_ghost_cells``).
        if config.dimensionality == 1:
            if config.boundary_settings == BoundarySettings1D(
                left_boundary=PERIODIC_BOUNDARY, right_boundary=PERIODIC_BOUNDARY
            ):
                config = config._replace(boundary_handling=PERIODIC_ROLL, num_ghost_cells=0)

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
                config = config._replace(boundary_handling=GHOST_CELLS)

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
                config = config._replace(boundary_handling=GHOST_CELLS)

        if config.boundary_handling == PERIODIC_ROLL:
            config = config._replace(num_ghost_cells=0)

        # The upwind EMF dissipation acts on the constrained-transport
        # interface fields, which only the MHD scheme carries.
        if config.solver_config.ct_upwind_dissipation and not config.mhd:
            raise ValueError("ct_upwind_dissipation requires finite-difference MHD")

        # The scheme and what goes with it.
        fd_config = config.solver_config
        if fd_config.scheme == WENO_FLUX_SPLITTING:
            if fd_config.time_integrator == SSPRK33:
                raise ValueError("time_integrator=SSPRK33 requires scheme=RIEMANN_FLUX.")
            if fd_config.riemann_solver != RIEMANN_BY_EOS:
                raise ValueError("FDSolverConfig.riemann_solver is read with scheme=RIEMANN_FLUX only.")
        elif fd_config.scheme == RIEMANN_FLUX:
            # What the scheme serves.
            if config.geometry != CARTESIAN or (config.mhd and config.dimensionality != 3):
                raise ValueError(
                    "scheme=RIEMANN_FLUX serves MHD in three and hydrodynamics in one to three Cartesian dimensions."
                )
            if fd_config.time_integrator == LSRK4:
                raise ValueError("time_integrator=LSRK4 requires scheme=WENO_FLUX_SPLITTING.")
            solver_of_the_gas = HLLEM_T
            if config.equation_of_state == IDEAL_GAS:
                solver_of_the_gas = HLLD_BOUNDED if config.mhd else HLLC
            if fd_config.riemann_solver not in (RIEMANN_BY_EOS, HLL, solver_of_the_gas):
                raise ValueError(
                    "scheme=RIEMANN_FLUX: riemann_solver must be RIEMANN_BY_EOS, HLL, HLLEM_T "
                    "(barotropic gas), HLLD_BOUNDED (ideal-gas MHD) or HLLC (ideal-gas hydrodynamics)."
                )
            # The options of the other scheme are not read: they keep their defaults.
            for name in (
                "weno_z",
                "weno_admissible_face_state",
                "weno_positivity_preserving",
                "ct_upwind_dissipation",
                "ct_upwind_strength",
                "dual_energy",
            ):
                if getattr(fd_config, name) != FDSolverConfig._field_defaults[name]:
                    raise ValueError(f"FDSolverConfig.{name} is read with scheme=WENO_FLUX_SPLITTING only.")
            # What has not been checked with the scheme.
            not_checked = {}
            if config.boundary_settings is not None:
                axes = config.boundary_settings
                if config.dimensionality == 1:
                    axes = (axes,)
                boundary_types = {side for axis in axes[: config.dimensionality] for side in axis}
                served = {PERIODIC_BOUNDARY, OPEN_BOUNDARY} if config.mhd else {
                    PERIODIC_BOUNDARY,
                    OPEN_BOUNDARY,
                    REFLECTIVE_BOUNDARY,
                }
                not_checked["boundaries other than periodic and open (and reflective without a field)"] = not (
                    boundary_types <= served
                )
            for name, requested in not_checked.items():
                if requested:
                    raise ValueError(f"scheme=RIEMANN_FLUX does not serve {name} yet.")
            # The constrained-transport kernels of the scheme follow the
            # backend: ``pallas_ct`` is derived here and what a user set is
            # replaced (the kernels of the scheme are one set, selected by
            # ``backend`` alone; the field is the switch of the other scheme).
            config = config._replace(
                backend_config=config.backend_config._replace(pallas_ct=config.backend_config.backend == PALLAS)
            )
        else:
            raise ValueError(
                f"FDSolverConfig.scheme must be WENO_FLUX_SPLITTING or RIEMANN_FLUX, got {fd_config.scheme!r}."
            )

    # Ghost-cell layers: the user's number, raised to what the scheme reads between two boundary fills.
    if config.boundary_handling == GHOST_CELLS:
        config = config._replace(num_ghost_cells=max(config.num_ghost_cells, _required_ghost_cells(config)))

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

    return config


def riemann_solver_to_string(riemann_solver: int) -> str:
    """Return the human-readable name of a Riemann-solver constant."""
    if riemann_solver == HLL:
        return "HLL"
    elif riemann_solver == HLLC:
        return "HLLC"
    elif riemann_solver == LAX_FRIEDRICHS:
        return "Lax-Friedrichs"
    elif riemann_solver == HLLD:
        return "HLLD"
    elif riemann_solver == HLL5R:
        return "HLL5R"


def limiter_to_string(limiter: int) -> str:
    """Return the human-readable name of a slope-limiter constant."""
    if limiter == MINMOD:
        return "Minmod"
    elif limiter == SUPERBEE:
        return "Superbee"
    elif limiter == DOUBLE_MINMOD:
        return "Double Minmod"
    elif limiter == VAN_LEER:
        return "Van Leer"


def config_to_string(config: SimulationConfig) -> str:
    """Return a compact one-line description of the solver configuration."""
    if config.solver_mode == FINITE_VOLUME:
        solver_config = get_solver_config(config)
        return (
            f"FV, {riemann_solver_to_string(solver_config.riemann_solver)}, "
            f"{limiter_to_string(solver_config.limiter)}, {config.num_cells.x} cells"
        )
    elif config.solver_mode == FINITE_DIFFERENCE:
        return f"FD, {config.num_cells.x} cells"
