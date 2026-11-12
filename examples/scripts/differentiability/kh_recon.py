r"""
Single vs multiple shooting for Kelvin-Helmholtz initial-condition reconstruction.

A twin experiment on a 2D Kelvin-Helmholtz (KH) shear layer: a fixed tanh base
flow carries an unknown transverse-velocity seed, parametrised by a control
``p`` in R^{5x2} (cos/sin amplitudes of the streamwise wavenumbers kx = 2..6
under a Gaussian envelope on the layer). The forward model integrates the
inviscid Euler equations with the finite-difference WENO-Z solver (Pallas
primal; derivatives through the native-JAX tangent) to a horizon T and
observes the terminal velocity field ``y``. The task is to recover ``p`` from
``y`` (noiseless).

Every method solves the *same* discrete problem. The forward map over T is the
composition of the M segment maps ``Phi_h`` (h = T/M), and the observation is
generated with that composition; single shooting differentiates through it,
multiple shooting lifts its interior states. (Integrating once over T instead
gives a slightly different time-step sequence; ``diagnose`` measures that
mismatch.)

Methods (``run --method``):

* ``single``: Adam on the reduced loss J(p) = 1/2 ||H(Phi_T(u0(p))) - y||^2,
  cosine-decay or constant learning rate.
* ``coarse_to_fine``: single shooting with a continuation in the misfit: the
  residual is low-passed along x (|kx| <= K) and K grows in stages up to the
  full field. This is the terminal-only analogue of a horizon continuation
  (which would need observations at intermediate times).
* ``multiple``: multiple shooting with M segments, the interior states lifted
  and continuity imposed by an augmented Lagrangian (inner Adam sweeps at fixed
  multipliers and penalty, outer multiplier update, penalty growth when the
  defect stalls).
* ``gauss_newton``: single-shooting Levenberg-Marquardt; with ten controls the
  Jacobian costs ten forward-mode tangents.
* ``gauss_newton_multiple``: Bock's condensed Gauss-Newton multiple shooting.
* ``gauss_newton_penalty``: multiple shooting with continuity as a penalty
  whose weight grows over the iterations (Gauss-Newton; each node moves in
  the subspace of Bock's condensed step, with its own coefficients), then
  single-shooting Gauss-Newton from the result.

Starts (``--control-start``, ``--node-start``): the cold control, and for
multiple shooting the interior nodes on its trajectory; or estimates from the
terminal observation by time reversal (``reversal_node_estimates``): the
control as the projection of the reversed t = 0 estimate, the interior nodes
at t >= ``reversal_fraction`` T as the reversed estimates there (accurate in
the nonlinear phase, unusable in the linear one). Single shooting can use the
reversal only through its control; multiple shooting places it at the nodes.

The cost unit is one full-horizon gradient evaluation: one Adam step of single
shooting, and one of multiple shooting (all M segments of length T/M).

Metrics, fixed before any campaign data: the *feasible* terminal loss of the
final control (its own forward integration over the full T, also for multiple
shooting, whose lifted loss is evaluated on a discontinuous trajectory),
relative to the loss of the zero control; the IC error
||u0(p) - u0(p*)|| / ||u0(p*)||; success = IC error < 0.1. ``summarize``
reports medians, interquartile ranges, success rates and paired comparisons
and marks, per method and horizon, the configuration with the lowest median
feasible loss (the tuning rule; it needs no truth). ``plot`` draws median
curves with IQR bands against the cost, ``recon`` shows the median-IC-error
run of each method at the same horizon.

Seeds: a truth index t draws the true control (``PRNGKey(t)``), an init index
i the cold start (``1e-2 * N(0, 1)`` from ``PRNGKey(100 + i)``); truth 0 with
inits 0..15 is the original campaign. The (truth, init) pairs of a job are
optimised in batches of independent runs, which share only the GPU (a single
256^2 integration leaves it mostly idle), and a batch checkpoints its
optimiser state so a chain of Slurm jobs can finish it
(``kh_shooting_campaign.sh``).

Usage (GPU subcommands pick a free GPU with autocvd unless
CUDA_VISIBLE_DEVICES is set; cell files go to ``--output-directory``)::

    python kh_recon.py time --horizons 20 60 --batch-sizes 1 4 8
    python kh_recon.py diagnose --horizon 60
    python kh_recon.py run --method multiple --horizon 60 --truths 3 4 \
        --inits 0 1 2 3 --output-directory <dir>
    python kh_recon.py summarize --output-directory <dir> --baseline single:<tag>
    python kh_recon.py plot --output-directory <dir> --tags single:<tag> \
        multiple:<tag> --output figures/<name>.png
    python kh_recon.py recon --output-directory <dir> --tags single:<tag> \
        multiple:<tag> --output figures/<name>.png
"""

# general
import argparse
import glob
import json
import math
import os
import time

# typing
from typing import Callable, NamedTuple, Optional, Tuple

# jax
import jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import (
    BACKWARDS,
    FORWARDS,
    NATIVE_JAX,
    OPEN_BOUNDARY,
    PALLAS,
    PERIODIC_BOUNDARY,
)

# astronomix containers
from astronomix import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FDSolverConfig,
    SimulationConfig,
    SimulationParams,
)

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
REDUCED_DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "data")

NUM_SEGMENTS = 8
INIT_SCALE = 1e-2
SUCCESS_THRESHOLD = 0.1

# The original campaign's settings; the defaults of ``run``.
DEFAULT_NUM_STEPS = 1980
DEFAULT_LEARNING_RATE = 3e-3
DEFAULT_FINAL_LEARNING_RATE_FRACTION = 0.05
DEFAULT_INNER_STEPS = 180
DEFAULT_OUTER_ITERATIONS = 11
DEFAULT_PENALTY_INITIAL = 1.0
DEFAULT_PENALTY_GROWTH = 1.6
DEFAULT_RECORD_EVERY = 10

METHOD_COLORS = {
    "single": "#1f77b4",
    "coarse_to_fine": "#2ca02c",
    "multiple": "#d62728",
    "gauss_newton": "#9467bd",
    "gauss_newton_multiple": "#8c564b",
    "gauss_newton_penalty": "#e377c2",
}
METHOD_LABELS = {
    "single": "single shooting",
    "coarse_to_fine": "single shooting, coarse-to-fine",
    "multiple": "multiple shooting",
    "gauss_newton": "single shooting, Gauss-Newton",
    "gauss_newton_multiple": "multiple shooting, Gauss-Newton",
    "gauss_newton_penalty": "multiple shooting, penalty Gauss-Newton",
}


# ============================================================================
#  The KH problem
# ============================================================================

class KHParams(NamedTuple):
    """Physical and numerical parameters of the KH twin experiment."""

    num_cells: int = 256
    box_size: float = 1.0
    #: Shear-layer centre and thickness.
    layer_center: float = 0.5
    layer_thickness: float = 0.02
    #: Velocity jump across the layer.
    velocity_jump: float = 1.0
    density: float = 1.0
    #: Mach number of the velocity jump; low, so the flow is nearly
    #: incompressible.
    mach_number: float = 0.3
    gamma: float = 5.0 / 3.0
    c_cfl: float = 1.0
    #: Checkpoints of the adaptive reverse-mode loop per segment; None sizes
    #: them to the expected number of time steps (see ``build_problem``).
    num_checkpoints: Optional[int] = None
    #: Reverse-mode rematerialisation of the Runge-Kutta stages. It is both
    #: lighter and faster here (the backward is memory-bound), and the
    #: computed gradient is the same.
    rematerialize_stages: bool = True
    #: Differentiate the WENO reconstruction with its nonlinear weights frozen
    #: (``weno_ad_frozen_weights``), the usual WENO linearisation. The exact
    #: weight derivative is ~ 2 / epsilon^3 where the gas is near-uniform (far
    #: from the layer), and over a long horizon the float32 forward-mode
    #: tangent overflows; the primal is unchanged.
    frozen_weno_weights: bool = False
    #: Reverse mode through the Pallas adjoint of the WENO flux
    #: (``BackendConfig.pallas_adjoint``, ~4x faster) instead of the transpose
    #: of the native tangent; only read by BACKWARDS configurations.
    pallas_adjoint: bool = False
    #: Streamwise wavenumbers of the control.
    wavenumber_min: int = 2
    wavenumber_max: int = 6
    #: rms of the true seed relative to the velocity jump.
    seed_amplitude: float = 1e-2
    #: Gaussian envelope half-width of the seed around the layer.
    envelope_width: float = 0.04

    @property
    def pressure(self):
        sound_speed = self.velocity_jump / self.mach_number
        return sound_speed ** 2 * self.density / self.gamma

    @property
    def growth_time(self):
        return self.layer_thickness / self.velocity_jump

    @property
    def num_modes(self):
        return self.wavenumber_max - self.wavenumber_min + 1


class ReconstructionProblem(NamedTuple):
    """The discrete forward model shared by all methods.

    Every callable takes the observation ``y`` (or the truth) as an argument
    instead of closing over it, so one compilation serves every truth seed.
    """

    kh_params: KHParams
    #: Horizon in KH growth times.
    horizon: float
    num_segments: int
    #: Control -> transverse-velocity seed field, (2, n, n).
    seed_field: Callable
    #: Control -> primitive initial state.
    initial_state: Callable
    #: Primitive state -> primitive state one segment (T/M) later.
    segment_map: Callable
    #: Primitive state -> observed velocity field, (2, n, n).
    observe: Callable
    #: Control -> observed terminal velocity (all segments in sequence).
    terminal_prediction: Callable
    #: Observed velocity (2, n, n) -> a full primitive state: the base density
    #: and the pressure of the incompressible limit (see ``build_problem``).
    state_from_velocity: Callable
    #: Primitive state -> the same state with its velocity reversed.
    reverse_velocity: Callable
    #: Cell-centre coordinates.
    grid_x: jnp.ndarray
    grid_y: jnp.ndarray


def build_simulation(
    kh_params: KHParams,
    segment_time: float,
    backend: int,
    differentiation_mode: int = BACKWARDS,
    num_checkpoints: int = 100,
):
    """
    Configure the inviscid finite-difference solver for one segment.

    Args:
        kh_params: The KH parameters.
        segment_time: The physical integration time of one segment.
        backend: PALLAS or NATIVE_JAX.
        differentiation_mode: BACKWARDS (checkpointed adaptive loop, reverse
            mode only) or FORWARDS (plain adaptive loop, forward mode only).
        num_checkpoints: Checkpoints of the reverse-mode loop.

    Returns:
        The (unfinalized) configuration, the simulation parameters and the
        registered variables.
    """
    config = SimulationConfig(
        dimensionality=2,
        num_cells=kh_params.num_cells,
        box_size=kh_params.box_size,
        progress_bar=False,
        solver_config=FDSolverConfig(
            weno_z=True,
            weno_ad_frozen_weights=kh_params.frozen_weno_weights,
        ),
        differentiation_mode=differentiation_mode,
        num_checkpoints=num_checkpoints,
        rematerialize_stages=kh_params.rematerialize_stages,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        backend_config=BackendConfig(backend=backend, pallas_adjoint=kh_params.pallas_adjoint),
    )
    params = SimulationParams(
        t_end=segment_time,
        C_cfl=kh_params.c_cfl,
        gamma=kh_params.gamma,
    )
    registered_variables = get_registered_variables(config)
    return config, params, registered_variables


def build_problem(
    kh_params: KHParams,
    horizon: float,
    num_segments: int = NUM_SEGMENTS,
    backend: int = PALLAS,
    differentiation_mode: int = BACKWARDS,
) -> ReconstructionProblem:
    """
    Build the forward model of the reconstruction problem.

    Args:
        kh_params: The KH parameters.
        horizon: The horizon T in KH growth times.
        num_segments: The number M of segments the horizon is split into.
        backend: PALLAS or NATIVE_JAX.
        differentiation_mode: BACKWARDS for gradients (reverse mode), FORWARDS
            for Jacobians (forward mode); the primal is the same.

    Returns:
        The reconstruction problem.
    """
    segment_time = horizon * kh_params.growth_time / num_segments

    # The checkpointed reverse-mode loop allocates all its checkpoint slots up
    # front (per segment and run), so they are sized to the expected number of
    # steps of a segment plus a margin, capped at 128: fewer slots than steps
    # only cost recomputation of primal steps, which run on the fast Pallas
    # path. The step estimate is the CFL step of the unperturbed layer (sum
    # over the two axes of the signal speeds).
    num_checkpoints = kh_params.num_checkpoints
    if num_checkpoints is None:
        sound_speed = kh_params.velocity_jump / kh_params.mach_number
        cell_size = kh_params.box_size / kh_params.num_cells
        estimated_time_step = kh_params.c_cfl * cell_size / (0.5 * kh_params.velocity_jump + 2 * sound_speed)
        num_checkpoints = min(int(math.ceil(1.25 * segment_time / estimated_time_step)) + 8, 128)

    config, params, registered_variables = build_simulation(
        kh_params,
        segment_time,
        backend,
        differentiation_mode,
        num_checkpoints,
    )
    centers = get_helper_data(config).geometric_centers
    grid_x, grid_y = centers[..., 0], centers[..., 1]

    wavenumbers = jnp.arange(kh_params.wavenumber_min, kh_params.wavenumber_max + 1)
    envelope = jnp.exp(-(((grid_y - kh_params.layer_center) / kh_params.envelope_width) ** 2))
    cosine_modes = jnp.cos(2 * jnp.pi * wavenumbers[None, None, :] * grid_x[..., None])
    sine_modes = jnp.sin(2 * jnp.pi * wavenumbers[None, None, :] * grid_x[..., None])

    def seed_field(control):
        transverse_velocity = envelope * jnp.sum(
            control[:, 0] * cosine_modes + control[:, 1] * sine_modes, axis=-1
        )
        return jnp.stack([jnp.zeros_like(transverse_velocity), transverse_velocity])

    def base_state_with_seed(seed):
        base_velocity_x = 0.5 * kh_params.velocity_jump * jnp.tanh(
            (grid_y - kh_params.layer_center) / kh_params.layer_thickness
        )
        return construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=kh_params.density * jnp.ones_like(base_velocity_x),
            velocity_x=base_velocity_x + seed[0],
            velocity_y=seed[1],
            gas_pressure=kh_params.pressure * jnp.ones_like(base_velocity_x),
        )

    state_shape = base_state_with_seed(jnp.zeros((2,) + grid_x.shape)).shape
    config = finalize_config(config, state_shape)

    def initial_state(control):
        return base_state_with_seed(seed_field(control))

    def segment_map(state):
        return time_integration(state, config, params, registered_variables)

    def observe(state):
        return jnp.stack([
            state[registered_variables.velocity_index.x],
            state[registered_variables.velocity_index.y],
        ])

    cell_size = kh_params.box_size / kh_params.num_cells

    def incompressible_pressure(velocity):
        # The pressure of the low-Mach limit, lap p = -rho d_i v_j d_j v_i,
        # with centred differences, solved spectrally. The y direction is
        # treated as periodic too: the source vanishes away from the layer,
        # where the base flow is uniform. The modes the centred Laplacian does
        # not see (the mean and the Nyquist modes) are set to zero.
        def d_dx(field):
            return (jnp.roll(field, -1, 0) - jnp.roll(field, 1, 0)) / (2 * cell_size)

        def d_dy(field):
            return (jnp.roll(field, -1, 1) - jnp.roll(field, 1, 1)) / (2 * cell_size)

        velocity_x, velocity_y = velocity
        source = -kh_params.density * (
            d_dx(velocity_x) ** 2 + 2 * d_dy(velocity_x) * d_dx(velocity_y) + d_dy(velocity_y) ** 2
        )
        wavenumbers = 2 * jnp.pi * jnp.fft.fftfreq(kh_params.num_cells, d=cell_size)
        wavenumbers_x, wavenumbers_y = jnp.meshgrid(wavenumbers, wavenumbers, indexing="ij")
        laplacian = -(jnp.sin(wavenumbers_x * cell_size) ** 2 + jnp.sin(wavenumbers_y * cell_size) ** 2) / cell_size ** 2
        singular = jnp.abs(laplacian) * cell_size ** 2 < 1e-8
        pressure_hat = jnp.where(singular, 0.0, jnp.fft.fft2(source) / jnp.where(singular, 1.0, laplacian))
        return jnp.real(jnp.fft.ifft2(pressure_hat))

    def state_from_velocity(velocity):
        return construct_primitive_state(
            config=config,
            registered_variables=registered_variables,
            density=kh_params.density * jnp.ones_like(velocity[0]),
            velocity_x=velocity[0],
            velocity_y=velocity[1],
            gas_pressure=kh_params.pressure + incompressible_pressure(velocity),
        )

    def reverse_velocity(state):
        return (
            state.at[registered_variables.velocity_index.x].multiply(-1.0)
            .at[registered_variables.velocity_index.y].multiply(-1.0)
        )

    # The segments run in sequence through a scan, so the segment map is
    # traced and compiled once rather than once per segment.
    def terminal_prediction(control):
        def advance_one_segment(state, _):
            return segment_map(state), None

        final_state, _ = jax.lax.scan(
            advance_one_segment,
            initial_state(control),
            None,
            length=num_segments,
        )
        return observe(final_state)

    return ReconstructionProblem(
        kh_params=kh_params,
        horizon=horizon,
        num_segments=num_segments,
        seed_field=seed_field,
        initial_state=initial_state,
        segment_map=segment_map,
        observe=observe,
        terminal_prediction=terminal_prediction,
        state_from_velocity=state_from_velocity,
        reverse_velocity=reverse_velocity,
        grid_x=grid_x,
        grid_y=grid_y,
    )


def truth_control(problem: ReconstructionProblem, truth_index: int) -> jnp.ndarray:
    """
    The true control of a truth seed: Gaussian mode amplitudes with uniform
    phases, scaled so the seed field has rms ``seed_amplitude * velocity_jump``.

    Args:
        problem: The reconstruction problem.
        truth_index: The truth seed.

    Returns:
        The true control, (num_modes, 2).
    """
    kh_params = problem.kh_params
    amplitude_key, phase_key = jax.random.split(jax.random.PRNGKey(truth_index))
    amplitudes = jax.random.normal(amplitude_key, (kh_params.num_modes,))
    phases = jax.random.uniform(phase_key, (kh_params.num_modes,), minval=0.0, maxval=2 * jnp.pi)
    unscaled_control = jnp.stack([amplitudes * jnp.cos(phases), -amplitudes * jnp.sin(phases)], axis=1)
    unscaled_rms = jnp.sqrt(jnp.mean(problem.seed_field(unscaled_control)[1] ** 2))
    return kh_params.seed_amplitude * kh_params.velocity_jump * unscaled_control / unscaled_rms


def cold_start_control(problem: ReconstructionProblem, init_index: int) -> jnp.ndarray:
    """Return the cold-start control of an init seed, ``1e-2 * N(0, 1)``."""
    return INIT_SCALE * jax.random.normal(
        jax.random.PRNGKey(100 + init_index),
        (problem.kh_params.num_modes, 2),
    )


def initial_condition_error(problem, control, true_control):
    """Return ||u0(p) - u0(p*)|| / ||u0(p*)|| on the seed fields."""
    true_seed = problem.seed_field(true_control)
    return jnp.linalg.norm(problem.seed_field(control) - true_seed) / jnp.linalg.norm(true_seed)


def lowpass_along_x(field, wavenumber_cutoff):
    """
    Keep the streamwise Fourier modes |kx| <= cutoff of a (2, nx, ny) field.

    Args:
        field: The field, x along axis 1 (periodic).
        wavenumber_cutoff: The largest kept |kx| (box units); None keeps all.

    Returns:
        The filtered field.
    """
    if wavenumber_cutoff is None:
        return field
    num_cells_x = field.shape[1]
    wavenumbers = jnp.abs(jnp.fft.fftfreq(num_cells_x, d=1.0 / num_cells_x))
    keep = (wavenumbers <= wavenumber_cutoff)[None, :, None]
    return jnp.real(jnp.fft.ifft(jnp.fft.fft(field, axis=1) * keep, axis=1))


def terminal_loss(problem, control, observation, wavenumber_cutoff=None):
    """Return the (optionally x-low-passed) terminal loss 1/2 ||H(Phi_T(u0(p))) - y||^2."""
    residual = problem.terminal_prediction(control) - observation
    return 0.5 * jnp.sum(lowpass_along_x(residual, wavenumber_cutoff) ** 2)


def reversal_node_estimates(problem, segment_map, observation):
    """
    Estimate the states at the node times t_k = k T / M (k = 0..M) from the
    terminal observation alone, by time reversal: the inviscid Euler equations
    are invariant under (rho, v, p)(t) -> (rho, -v, p)(T - t), so the terminal
    state with its velocity reversed, integrated forward over k segments and
    reversed again, approximates the state at T - k T / M. The unobserved
    density and pressure at T are those of the incompressible limit (base
    density, Poisson pressure).

    The estimate is good where the flow is nonlinear and large (the late
    nodes), and poor in the linear phase: the modes that decay forward grow
    backward, from the numerical dissipation and from the missing terminal
    density and pressure, while the true perturbation shrinks to the seed.

    Args:
        problem: The reconstruction problem.
        segment_map: The (jitted) segment map.
        observation: The terminal velocity observation, (2, n, n).

    Returns:
        The M + 1 estimated states, ordered by node time.
    """
    state = problem.reverse_velocity(problem.state_from_velocity(observation))
    backward_states = [state]
    for _ in range(problem.num_segments):
        state = segment_map(state)
        backward_states.append(state)
    return [problem.reverse_velocity(state) for state in backward_states[::-1]]


def project_onto_controls(problem, transverse_velocity):
    """
    Return the control whose seed field best fits a transverse-velocity field
    (least squares over the seed basis).

    Args:
        problem: The reconstruction problem.
        transverse_velocity: The v_y field, (n, n).

    Returns:
        The control, (num_modes, 2).
    """
    num_modes = problem.kh_params.num_modes
    unit_controls = jnp.eye(2 * num_modes).reshape(2 * num_modes, num_modes, 2)
    basis = jax.vmap(lambda control: problem.seed_field(control)[1].ravel())(unit_controls).T
    coefficients, *_ = jnp.linalg.lstsq(basis, transverse_velocity.ravel())
    return coefficients.reshape(num_modes, 2)


# ============================================================================
#  Adam (as optax.adam: bias-corrected moments, epsilon outside the root)
# ============================================================================

class AdamState(NamedTuple):
    step: jnp.ndarray
    first_moment: dict
    second_moment: dict


def adam_init(variables) -> AdamState:
    """Return a zeroed Adam state for a pytree of variables."""
    zeros = jax.tree_util.tree_map(jnp.zeros_like, variables)
    return AdamState(step=jnp.zeros((), jnp.int32), first_moment=zeros, second_moment=zeros)


def adam_update(
    variables,
    gradient,
    state: AdamState,
    learning_rate,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    """
    One Adam step.

    Args:
        variables: The current variables (pytree).
        gradient: Their gradient (same pytree).
        state: The Adam state.
        learning_rate: The step size of this step: a scalar, or one per run
            of a batch (leading axis of every leaf).

    Returns:
        The updated variables and Adam state.
    """

    def leaf_learning_rate(leaf):
        if jnp.ndim(learning_rate) == 0:
            return learning_rate
        return jnp.reshape(learning_rate, (-1,) + (1,) * (leaf.ndim - 1))

    step = state.step + 1
    first_moment = jax.tree_util.tree_map(
        lambda moment, grad: beta1 * moment + (1 - beta1) * grad, state.first_moment, gradient
    )
    second_moment = jax.tree_util.tree_map(
        lambda moment, grad: beta2 * moment + (1 - beta2) * grad ** 2, state.second_moment, gradient
    )
    first_correction = 1 - beta1 ** step.astype(jnp.float32)
    second_correction = 1 - beta2 ** step.astype(jnp.float32)
    new_variables = jax.tree_util.tree_map(
        lambda value, first, second: value - leaf_learning_rate(value) * (first / first_correction)
        / (jnp.sqrt(second / second_correction) + epsilon),
        variables,
        first_moment,
        second_moment,
    )
    return new_variables, AdamState(step=step, first_moment=first_moment, second_moment=second_moment)


def learning_rate_at(step, num_steps, learning_rate, schedule, final_fraction):
    """Return the learning rate of step ``step``: constant, or cosine decay to ``final_fraction``."""
    if schedule == "constant":
        return learning_rate
    progress = min(step, num_steps) / num_steps
    return learning_rate * ((1 - final_fraction) * 0.5 * (1 + math.cos(math.pi * progress)) + final_fraction)


# ============================================================================
#  The methods
# ============================================================================

class MethodSettings(NamedTuple):
    """Hyperparameters of all methods; each method reads its own."""

    #: Adam steps (single shooting, coarse-to-fine).
    num_steps: int = DEFAULT_NUM_STEPS
    learning_rate: float = DEFAULT_LEARNING_RATE
    #: "cosine" or "constant" (single shooting, coarse-to-fine).
    schedule: str = "cosine"
    final_learning_rate_fraction: float = DEFAULT_FINAL_LEARNING_RATE_FRACTION
    #: Streamwise cutoffs of the coarse-to-fine stages (equal step shares);
    #: a last stage with the full misfit follows.
    cutoff_stages: Tuple[int, ...] = (2, 4, 8, 16)
    #: Augmented Lagrangian (multiple shooting).
    inner_steps: int = DEFAULT_INNER_STEPS
    outer_iterations: int = DEFAULT_OUTER_ITERATIONS
    penalty_initial: float = DEFAULT_PENALTY_INITIAL
    penalty_growth: float = DEFAULT_PENALTY_GROWTH
    #: The penalty grows when an outer iteration reduces the total defect
    #: by less than this factor.
    defect_decrease_threshold: float = 0.7
    penalty_cap: float = 1e6
    #: Levenberg-Marquardt (Gauss-Newton).
    num_iterations: int = 60
    damping_initial: float = 1e-2
    #: Weight of the continuity defects in the Gauss-Newton multiple-shooting
    #: merit.
    continuity_weight: float = 1.0
    #: The start control: "cold", "reversal" (the projection of the
    #: time-reversed observation at t = 0, see ``reversal_node_estimates``) or
    #: "reversal_node" (the cold control fitted by Gauss-Newton to the
    #: time-reversed observation at the first reversal node).
    control_start: str = "cold"
    #: The start of the multiple-shooting interior nodes: "forward" (the start
    #: control's trajectory) or "reversal" (the time-reversed observation at
    #: the nodes t >= reversal_fraction T, the start control's trajectory
    #: before).
    node_start: str = "forward"
    reversal_fraction: float = 0.5
    #: Bock's relaxation of the multiple-shooting matching conditions,
    #: r_j - (1 - kappa) r_j^0 = 0 with the start defects r_j^0: kappa starts
    #: here and grows by ``relaxation_increment`` after every accepted step up
    #: to 1 (the plain conditions). 1 = no relaxation.
    relaxation_initial: float = 1.0
    relaxation_increment: float = 0.1
    #: Anchors of the reversal nodes (``node_start`` "reversal"): the merit
    #: gets anchor_weight/2 sum_j ||v(s_j) - v(b_j)||^2 over the reversal
    #: nodes b_j (the velocity, as observed), and the weight is multiplied by
    #: ``anchor_decay`` after every accepted step: the nodes are first held at
    #: the time-reversed estimates while the control fits them, then released.
    #: 0 = no anchors.
    anchor_weight: float = 0.0
    anchor_decay: float = 0.7
    #: Re-simulate the nodes before the first reversal node from the control
    #: after every accepted step (the early horizon as one single-shooting
    #: segment).
    early_restore: bool = False
    #: Penalty multiple shooting (``gauss_newton_penalty``): the continuity
    #: weight starts at ``continuity_weight``, is multiplied by
    #: ``continuity_growth`` after every accepted step up to
    #: ``continuity_max``, and the lifted phase ends with the first accepted
    #: step at that weight; ``polish_iterations`` single-shooting
    #: Gauss-Newton iterations follow.
    continuity_growth: float = 3.0
    continuity_max: float = 1e4
    polish_iterations: int = 60
    record_every: int = DEFAULT_RECORD_EVERY


class MethodKernels(NamedTuple):
    """The jitted functions of the methods, built once per problem.

    The Adam methods optimise a batch of independent runs at once (leading
    axis of controls, observations and truths): the runs share nothing but the
    GPU, which a single 256^2 integration leaves mostly idle. The observation
    and the truth are arguments, so every batch of a job reuses the same
    compilations.
    """

    feasible_loss: Callable
    ic_error: Callable
    single_loss_and_gradient: Callable
    filtered_loss_and_gradient: Callable
    adam_step: Callable
    multiple_adam_step: Callable
    lifted_loss_and_defects: Callable
    batched_segment_map: Callable
    residual_and_jacobian: Callable
    segment_map: Callable
    segment_tangent: Callable
    segment_tangents: Callable
    initial_state_tangents: Callable


def build_kernels(problem: ReconstructionProblem) -> MethodKernels:
    """
    Build the jitted kernels of all methods (compiled lazily on first use).

    Args:
        problem: The reconstruction problem.

    Returns:
        The method kernels; all but ``residual_and_jacobian`` act on batches.
    """
    segments_in_parallel = jax.vmap(problem.segment_map)

    def run_loss(control, observation):
        return terminal_loss(problem, control, observation)

    def filtered_run_loss(control, observation, cutoff):
        return terminal_loss(problem, control, observation, cutoff)

    def run_ic_error(control, true_control):
        return initial_condition_error(problem, control, true_control)

    def multiple_shooting_residuals(variables, observation):
        segment_starts = jnp.concatenate(
            [problem.initial_state(variables["control"])[None], variables["interior_states"]],
            axis=0,
        )
        segment_ends = segments_in_parallel(segment_starts)
        defects = segment_ends[:-1] - segment_starts[1:]
        data_residual = problem.observe(segment_ends[-1]) - observation
        return defects, data_residual

    def augmented_lagrangian(variables, observation, multipliers, penalty):
        defects, data_residual = multiple_shooting_residuals(variables, observation)
        return (
            0.5 * jnp.sum(data_residual ** 2)
            + jnp.sum(multipliers * defects)
            + 0.5 * penalty * jnp.sum(defects ** 2)
        )

    def multiple_adam_step(variables, state, observations, multipliers, penalties, learning_rate):
        gradient = jax.vmap(jax.grad(augmented_lagrangian))(variables, observations, multipliers, penalties)
        new_variables, new_state = adam_update(variables, gradient, state, learning_rate)
        control_gradient_norm = jnp.sqrt(jnp.sum(gradient["control"] ** 2, axis=(1, 2)))
        return new_variables, new_state, control_gradient_norm

    def lifted_loss_and_defects(variables, observation):
        defects, data_residual = multiple_shooting_residuals(variables, observation)
        return 0.5 * jnp.sum(data_residual ** 2), jnp.sum(defects ** 2), defects

    def residual_and_jacobian(control, observation):
        def flat_residual(flat_control):
            prediction = problem.terminal_prediction(flat_control.reshape(control.shape))
            return (prediction - observation).ravel()

        flat_control = control.ravel()
        return flat_residual(flat_control), jax.jacfwd(flat_residual)(flat_control)

    return MethodKernels(
        feasible_loss=jax.jit(jax.vmap(run_loss)),
        ic_error=jax.jit(jax.vmap(run_ic_error)),
        single_loss_and_gradient=jax.jit(jax.vmap(jax.value_and_grad(run_loss))),
        filtered_loss_and_gradient=jax.jit(
            jax.vmap(jax.value_and_grad(filtered_run_loss), in_axes=(0, 0, None))
        ),
        adam_step=jax.jit(adam_update),
        multiple_adam_step=jax.jit(multiple_adam_step),
        lifted_loss_and_defects=jax.jit(jax.vmap(lifted_loss_and_defects)),
        batched_segment_map=jax.jit(jax.vmap(problem.segment_map)),
        residual_and_jacobian=jax.jit(residual_and_jacobian),
        segment_map=jax.jit(problem.segment_map),
        segment_tangent=jax.jit(lambda state, tangent: jax.jvp(problem.segment_map, (state,), (tangent,))),
        segment_tangents=jax.jit(
            jax.vmap(
                lambda state, tangent: jax.jvp(problem.segment_map, (state,), (tangent,))[1],
                in_axes=(None, 0),
            )
        ),
        initial_state_tangents=jax.jit(
            jax.vmap(
                lambda control, tangent: jax.jvp(problem.initial_state, (control,), (tangent,))[1],
                in_axes=(None, 0),
            )
        ),
    )


class Recorder:
    """Collects the convergence traces of a batch of optimisations.

    Every recorded value is a scalar shared by the batch or an array with one
    entry per run. ``optimizer_time`` accumulates only the blocking
    optimisation steps, so the metric evaluations of the trace (heavier for
    multiple shooting, which integrates its control separately) do not count
    against a method. The first call of each function (its compilation) is
    left out as well.
    """

    def __init__(self):
        self.columns = {}
        self.start_time = time.time()
        self.optimizer_time = 0.0
        self.called_functions = set()

    def timed(self, function, *arguments):
        """Call ``function`` to completion and add its time to the optimizer time."""
        start_time = time.time()
        result = jax.block_until_ready(function(*arguments))
        if id(function) in self.called_functions:
            self.optimizer_time += time.time() - start_time
        self.called_functions.add(id(function))
        return result

    def record(self, **values):
        values.setdefault("wall_time", time.time() - self.start_time)
        values.setdefault("optimizer_time", self.optimizer_time)
        for name, value in values.items():
            self.columns.setdefault(name, []).append(np.asarray(value, dtype=np.float64))

    def run_arrays(self, run_index, prefix="trace_"):
        """Return the traces of one run of the batch."""
        return {
            prefix + name: np.array([entry if entry.ndim == 0 else entry[run_index] for entry in column])
            for name, column in self.columns.items()
        }


class Checkpointer:
    """Saves and restores the optimiser state of a batch.

    A batch of long runs outlasts one Slurm job; the next job of a chain
    resumes from the last checkpoint (written atomically every few minutes).
    """

    def __init__(self, path, interval_seconds):
        self.path = path
        self.interval_seconds = interval_seconds
        self.last_save_time = time.time()

    def load(self, recorder):
        """
        Restore a saved checkpoint, if any.

        Args:
            recorder: The recorder, whose traces and clocks are restored.

        Returns:
            The step to continue from and the saved state arrays (None
            without a checkpoint).
        """
        if self.path is None or not os.path.exists(self.path):
            return 0, None
        with np.load(self.path) as data:
            contents = {name: data[name] for name in data.files}
        recorder.optimizer_time = float(contents["optimizer_time"])
        recorder.start_time = time.time() - float(contents["elapsed_time"])
        recorder.columns = {
            name[len("column__"):]: list(values)
            for name, values in contents.items()
            if name.startswith("column__")
        }
        state = {
            name[len("state__"):]: values
            for name, values in contents.items()
            if name.startswith("state__")
        }
        print(f"  resumed from {self.path} at step {int(contents['step'])}", flush=True)
        return int(contents["step"]), state

    def maybe_save(self, step, state, recorder):
        """Save the state if the interval has passed since the last save."""
        if self.path is None or time.time() - self.last_save_time < self.interval_seconds:
            return
        payload = {
            "step": step,
            "optimizer_time": recorder.optimizer_time,
            "elapsed_time": time.time() - recorder.start_time,
        }
        for name, column in recorder.columns.items():
            payload["column__" + name] = np.stack(column)
        for name, value in state.items():
            payload["state__" + name] = np.asarray(value)
        temporary_path = self.path + ".partial.npz"
        np.savez(temporary_path, **payload)
        os.replace(temporary_path, self.path)
        self.last_save_time = time.time()


def run_single_shooting(
    problem,
    kernels,
    observations,
    start_controls,
    true_controls,
    settings,
    checkpointer,
    coarse_to_fine=False,
):
    """
    Single shooting with Adam on a batch of runs, optionally with the
    coarse-to-fine continuation of the misfit.

    Args:
        problem: The reconstruction problem.
        kernels: The method kernels of the problem.
        observations: The terminal observations y, (B, 2, n, n).
        start_controls: The cold starts, (B, num_modes, 2).
        true_controls: The truths (only for the IC-error trace).
        settings: The method settings.
        checkpointer: Saves and restores the optimiser state.
        coarse_to_fine: Low-pass the misfit along x in stages.

    Returns:
        The final controls and the recorder.
    """
    num_steps = settings.num_steps
    if coarse_to_fine:
        # The last stage keeps every mode (cutoff n/2); the cutoff is traced,
        # so the stages share one compilation.
        stage_cutoffs = list(settings.cutoff_stages) + [problem.kh_params.num_cells // 2]
    else:
        stage_cutoffs = [None]
    steps_per_stage = int(math.ceil(num_steps / len(stage_cutoffs)))

    controls = start_controls
    adam_state = adam_init(controls)
    recorder = Recorder()
    first_step, saved_state = checkpointer.load(recorder)
    if saved_state is not None:
        controls = jnp.asarray(saved_state["controls"])
        adam_state = AdamState(
            step=jnp.asarray(saved_state["adam_step"]),
            first_moment=jnp.asarray(saved_state["adam_first_moment"]),
            second_moment=jnp.asarray(saved_state["adam_second_moment"]),
        )
    for step in range(first_step, num_steps + 1):
        cutoff = stage_cutoffs[min(step // steps_per_stage, len(stage_cutoffs) - 1)]
        if coarse_to_fine:
            losses, gradients = recorder.timed(
                kernels.filtered_loss_and_gradient,
                controls,
                observations,
                jnp.float32(cutoff),
            )
        else:
            losses, gradients = recorder.timed(kernels.single_loss_and_gradient, controls, observations)
        if step % settings.record_every == 0 or step == num_steps:
            # The plain single-shooting loss is the feasible loss itself.
            full_losses = kernels.feasible_loss(controls, observations) if coarse_to_fine else losses
            recorder.record(
                step=step,
                cost=step,
                objective=losses,
                feasible_loss=full_losses,
                ic_error=kernels.ic_error(controls, true_controls),
                gradient_norm=jnp.sqrt(jnp.sum(gradients ** 2, axis=(1, 2))),
                cutoff=-1 if cutoff is None else cutoff,
            )
        if step < num_steps:
            learning_rate = learning_rate_at(
                step,
                num_steps,
                settings.learning_rate,
                settings.schedule,
                settings.final_learning_rate_fraction,
            )
            controls, adam_state = recorder.timed(
                kernels.adam_step,
                controls,
                gradients,
                adam_state,
                learning_rate,
            )
            checkpointer.maybe_save(
                step + 1,
                {
                    "controls": controls,
                    "adam_step": adam_state.step,
                    "adam_first_moment": adam_state.first_moment,
                    "adam_second_moment": adam_state.second_moment,
                },
                recorder,
            )
    return controls, recorder


def run_coarse_to_fine(problem, kernels, observations, start_controls, true_controls, settings, checkpointer):
    """Single shooting with the coarse-to-fine misfit continuation (see ``run_single_shooting``)."""
    return run_single_shooting(
        problem,
        kernels,
        observations,
        start_controls,
        true_controls,
        settings,
        checkpointer,
        coarse_to_fine=True,
    )


def run_multiple_shooting(
    problem,
    kernels,
    observations,
    start_controls,
    true_controls,
    settings,
    checkpointer,
):
    """
    Multiple shooting with an augmented Lagrangian and Adam inner sweeps, on a
    batch of runs.

    The optimisation variables of a run are the control p and the interior
    segment-start states s_1..s_{M-1}; the merit is
    1/2||H(Phi_h(s_{M-1})) - y||^2 + sum_j <lambda_j, r_j> + rho/2 ||r_j||^2 with
    the defects r_j = Phi_h(s_j) - s_{j+1}, s_0 = u0(p). One Adam step
    integrates all M segments (in parallel), i.e. costs one full-horizon
    gradient evaluation. Multipliers and penalty are per run.

    Args:
        problem: The reconstruction problem.
        kernels: The method kernels of the problem.
        observations: The terminal observations y, (B, 2, n, n).
        start_controls: The cold starts; the interiors start on their
            trajectories.
        true_controls: The truths (only for the IC-error trace).
        settings: The method settings.
        checkpointer: Saves and restores the optimiser state.

    Returns:
        The final controls and the recorder.
    """
    num_runs = start_controls.shape[0]
    num_steps = settings.inner_steps * settings.outer_iterations
    recorder = Recorder()
    step, saved_state = checkpointer.load(recorder)

    # -------------------------------------------------------------
    # ============== ↓ initial or restored iterate ↓ ==============
    # -------------------------------------------------------------
    if saved_state is None:
        # Start on the cold controls' own trajectories, so the defects start
        # at zero and the first iterate is the single-shooting starting point.
        interior_states = []
        states = jax.vmap(problem.initial_state)(start_controls)
        for _ in range(problem.num_segments - 1):
            states = kernels.batched_segment_map(states)
            interior_states.append(states)
        variables = {"control": start_controls, "interior_states": jnp.stack(interior_states, axis=1)}
        multipliers = jnp.zeros_like(variables["interior_states"])
        penalties = np.full(num_runs, settings.penalty_initial)
        previous_defects = None
        adam_state = adam_init(variables)
    else:
        variables = {
            "control": jnp.asarray(saved_state["control"]),
            "interior_states": jnp.asarray(saved_state["interior_states"]),
        }
        multipliers = jnp.asarray(saved_state["multipliers"])
        penalties = np.asarray(saved_state["penalties"])
        previous_defects = np.asarray(saved_state["previous_defects"])
        if np.all(np.isnan(previous_defects)):
            previous_defects = None
        adam_state = AdamState(
            step=jnp.asarray(saved_state["adam_step"]),
            first_moment={
                "control": jnp.asarray(saved_state["adam_first_moment_control"]),
                "interior_states": jnp.asarray(saved_state["adam_first_moment_interior_states"]),
            },
            second_moment={
                "control": jnp.asarray(saved_state["adam_second_moment_control"]),
                "interior_states": jnp.asarray(saved_state["adam_second_moment_interior_states"]),
            },
        )
    # -------------------------------------------------------------
    # ============== ↑ initial or restored iterate ↑ ==============
    # -------------------------------------------------------------

    def record(step, gradient_norms):
        lifted_losses, total_defects, _ = kernels.lifted_loss_and_defects(variables, observations)
        recorder.record(
            step=step,
            cost=step,
            objective=lifted_losses,
            feasible_loss=kernels.feasible_loss(variables["control"], observations),
            ic_error=kernels.ic_error(variables["control"], true_controls),
            gradient_norm=gradient_norms,
            defect=total_defects,
            penalty=penalties,
        )

    gradient_norms = np.full(num_runs, np.nan)
    if saved_state is None:
        record(0, gradient_norms)
    while step < num_steps:
        variables, adam_state, gradient_norms = recorder.timed(
            kernels.multiple_adam_step,
            variables,
            adam_state,
            observations,
            multipliers,
            jnp.asarray(penalties, dtype=jnp.float32),
            settings.learning_rate,
        )
        step += 1
        if step % settings.record_every == 0:
            record(step, gradient_norms)

        # ---------------- ↓ outer augmented-Lagrangian update ↓ ----------------
        if step % settings.inner_steps == 0:
            _, total_defects, defects = kernels.lifted_loss_and_defects(variables, observations)
            run_penalties = jnp.asarray(penalties, dtype=jnp.float32)[:, None, None, None, None]
            multipliers = multipliers + run_penalties * defects
            total_defects = np.asarray(total_defects, dtype=np.float64)
            if previous_defects is not None:
                stalled = total_defects > settings.defect_decrease_threshold * previous_defects
                penalties = np.where(
                    stalled,
                    np.minimum(penalties * settings.penalty_growth, settings.penalty_cap),
                    penalties,
                )
            previous_defects = total_defects
        # ---------------- ↑ outer augmented-Lagrangian update ↑ ----------------

        saved_previous_defects = np.full(num_runs, np.nan) if previous_defects is None else previous_defects
        checkpointer.maybe_save(
            step,
            {
                "control": variables["control"],
                "interior_states": variables["interior_states"],
                "multipliers": multipliers,
                "penalties": penalties,
                "previous_defects": saved_previous_defects,
                "adam_step": adam_state.step,
                "adam_first_moment_control": adam_state.first_moment["control"],
                "adam_first_moment_interior_states": adam_state.first_moment["interior_states"],
                "adam_second_moment_control": adam_state.second_moment["control"],
                "adam_second_moment_interior_states": adam_state.second_moment["interior_states"],
            },
            recorder,
        )
    if step % settings.record_every != 0:
        record(step, gradient_norms)
    return variables["control"], recorder


def run_gauss_newton(problem, kernels, observations, start_controls, true_controls, settings, checkpointer):
    """
    Single-shooting Levenberg-Marquardt, one run (batch of one).

    The 10-column Jacobian of the terminal prediction comes from forward-mode
    tangents (themselves batched over the ten directions); the damped normal
    equations use Marquardt's diagonal scaling. A rejected step raises the
    damping and retries without a new Jacobian; the run stops early when ten
    increasingly damped steps all fail.

    Args:
        problem: The reconstruction problem (forward-mode differentiable).
        kernels: The method kernels of the problem.
        observations: The terminal observation y, (1, 2, n, n).
        start_controls: The cold start, (1, num_modes, 2).
        true_controls: The truth (only for the IC-error trace).
        settings: The method settings.
        checkpointer: Unused; a Gauss-Newton run fits in one job.

    Returns:
        The final control (1, num_modes, 2) and the recorder.
    """
    num_controls = start_controls[0].size
    controls = start_controls
    damping = settings.damping_initial
    recorder = Recorder()
    cost = 0.0
    for iteration in range(settings.num_iterations + 1):
        residual, jacobian = recorder.timed(kernels.residual_and_jacobian, controls[0], observations[0])
        loss = 0.5 * float(jnp.sum(residual ** 2))
        normal_matrix = np.asarray(jacobian.T @ jacobian, dtype=np.float64)
        gradient = np.asarray(jacobian.T @ residual, dtype=np.float64)
        recorder.record(
            step=iteration,
            cost=cost,
            objective=loss,
            feasible_loss=loss,
            ic_error=kernels.ic_error(controls, true_controls)[0],
            gradient_norm=np.linalg.norm(gradient),
            damping=damping,
        )
        if iteration == settings.num_iterations:
            break
        # A Jacobian (one primal, ten tangents) is counted as ten full-horizon
        # gradient evaluations and a trial forward run as one third of one;
        # the recorded optimizer time is the measured comparison.
        cost += num_controls
        scaling = np.diag(np.diag(normal_matrix))
        accepted = False
        for _ in range(10):
            step = np.linalg.solve(normal_matrix + damping * scaling, -gradient)
            trial_controls = controls + jnp.asarray(step.reshape(controls.shape), dtype=controls.dtype)
            trial_loss = float(recorder.timed(kernels.feasible_loss, trial_controls, observations)[0])
            cost += 1.0 / 3.0
            if trial_loss < loss:
                controls = trial_controls
                damping = max(damping / 3.0, 1e-8)
                accepted = True
                break
            damping = min(4.0 * damping, 1e12)
        # No damped step lowers the loss: the run has converged to the
        # round-off floor (or is stuck), and further Jacobians would only
        # repeat the same rejected steps.
        if not accepted:
            break
    return controls, recorder


def run_gauss_newton_multiple(problem, kernels, observations, start_controls, true_controls, settings, checkpointer):
    """
    Constrained Gauss-Newton multiple shooting (Bock), condensed, one run.

    The variables are the control p and the interior states s_1..s_{M-1}
    (started on the start control's trajectory, or with the late nodes from
    the time-reversed observation, ``settings.node_start``). Each iteration linearises the
    continuity constraints Phi_h(s_j) - s_{j+1} = 0 and the data residual,
    condenses the interior updates (ds_0 = B dp, ds_{j+1} = G_j ds_j + r_j) so
    the Levenberg-Marquardt system is in p alone, and back-substitutes the
    interiors. All Jacobian products are forward-mode tangents through single
    segments of length T/M. The merit is 1/2||data residual||^2 +
    rho/2 sum_j ||r_j||^2. With ``settings.relaxation_initial`` < 1 the
    matching conditions are Bock's relaxed ones, r_j - (1 - kappa) r_j^0 = 0
    (r_j^0 the start defects), in the condensing, the back-substitution and
    the merit, with kappa growing to 1 over the accepted steps: a large start
    defect (the reversal nodes) is then closed gradually by the control
    instead of by one linear step that moves the nodes onto the start
    control's trajectory. The damping follows ``run_gauss_newton``, so the
    two differ only in the formulation, except for a backtracking line search
    on the whole step (the damping cannot shorten the interior update). The
    recorded feasible loss integrates p alone over the full horizon.

    Args:
        problem: The reconstruction problem (forward-mode differentiable).
        kernels: The method kernels of the problem.
        observations: The terminal observation y, (1, 2, n, n).
        start_controls: The cold start, (1, num_modes, 2).
        true_controls: The truth (only for the IC-error trace).
        settings: The method settings.
        checkpointer: Unused; a run fits in one job.

    Returns:
        The final control (1, num_modes, 2) and the recorder.
    """
    observation = observations[0]
    control = start_controls[0]
    num_controls = control.size
    basis = jnp.eye(num_controls, dtype=control.dtype).reshape((num_controls,) + control.shape)
    recorder = Recorder()

    # ---------------- ↓ lifted residuals and merit ↓ ----------------
    def lifted_residuals(control, interior_states):
        segment_starts = [problem.initial_state(control)] + interior_states
        segment_ends = [kernels.segment_map(state) for state in segment_starts]
        defects = [segment_ends[j] - segment_starts[j + 1] for j in range(problem.num_segments - 1)]
        data_residual = problem.observe(segment_ends[-1]) - observation
        return segment_starts, defects, data_residual

    def relaxed(defects):
        # Bock's relaxed matching conditions: the share (1 - kappa) of the
        # start defects is tolerated (``settings.relaxation_initial``).
        return [defect - (1.0 - relaxation) * start for defect, start in zip(defects, start_defects)]

    def anchor_loss(interior_states):
        return 0.5 * sum(
            float(jnp.sum((problem.observe(interior_states[node - 1]) - target) ** 2))
            for node, target in anchor_targets.items()
        )

    def merit_of(defects, data_residual, interior_states):
        data_loss = 0.5 * float(jnp.sum(data_residual ** 2))
        total_defect = float(sum(jnp.sum(defect ** 2) for defect in defects))
        relaxed_defect = float(sum(jnp.sum(defect ** 2) for defect in relaxed(defects)))
        merit = data_loss + 0.5 * settings.continuity_weight * relaxed_defect
        if anchor_weight > 0.0:
            merit += anchor_weight * anchor_loss(interior_states)
        return merit, data_loss, total_defect

    def restore_early_nodes(control, interior_states):
        # The nodes before the first reversal node, re-simulated from the
        # control: the early horizon is then one single-shooting segment.
        state = problem.initial_state(control)
        restored = list(interior_states)
        for node in range(1, first_reversal_node):
            state = kernels.segment_map(state)
            restored[node - 1] = state
        return restored
    # ---------------- ↑ lifted residuals and merit ↑ ----------------

    # Feasible interiors: the start control's own trajectory (zero defects);
    # with the reversal start the late nodes are replaced by the time-reversed
    # observation (``reversal_node_estimates``).
    interior_states = []
    state = problem.initial_state(control)
    for _ in range(problem.num_segments - 1):
        state = kernels.segment_map(state)
        interior_states.append(state)
    first_reversal_node = problem.num_segments
    anchor_targets = {}
    if settings.node_start == "reversal":
        estimates = reversal_node_estimates(problem, kernels.segment_map, observation)
        first_reversal_node = int(math.ceil(settings.reversal_fraction * problem.num_segments))
        for node in range(max(first_reversal_node, 1), problem.num_segments):
            interior_states[node - 1] = estimates[node]
            if settings.anchor_weight > 0.0:
                anchor_targets[node] = problem.observe(estimates[node])
    anchor_weight = settings.anchor_weight
    segment_starts, defects, data_residual = lifted_residuals(control, interior_states)
    start_defects = list(defects)
    relaxation = min(settings.relaxation_initial, 1.0)
    merit, data_loss, total_defect = merit_of(defects, data_residual, interior_states)

    damping = settings.damping_initial
    cost = 0.0
    for iteration in range(settings.num_iterations + 1):
        recorder.record(
            step=iteration,
            cost=cost,
            objective=data_loss,
            feasible_loss=kernels.feasible_loss(control[None], observations)[0],
            ic_error=kernels.ic_error(control[None], true_controls)[0],
            gradient_norm=float("nan"),
            defect=total_defect,
            damping=damping,
            relaxation=relaxation,
            anchor_weight=anchor_weight,
        )
        if iteration == settings.num_iterations:
            break

        # ---------------- ↓ condensed sensitivities ↓ ----------------
        start_time = time.time()
        # Columns of d(data residual)/dp: B e_k carried through all segments.
        tangents = kernels.initial_state_tangents(control, basis)
        anchor_sensitivities = {}
        for j, segment_start in enumerate(segment_starts):
            tangents = kernels.segment_tangents(segment_start, tangents)
            if j + 1 in anchor_targets:
                anchor_sensitivities[j + 1] = np.asarray(
                    jax.vmap(problem.observe)(tangents), dtype=np.float64
                ).reshape(num_controls, -1).T
        sensitivity = np.asarray(jax.vmap(problem.observe)(tangents), dtype=np.float64)
        sensitivity = sensitivity.reshape(num_controls, -1).T
        # The (relaxed) defects carried to the end: u_{j+1} = G_j u_j + r_j,
        # u_0 = 0.
        relaxed_defects = relaxed(defects)
        carried = jnp.zeros_like(segment_starts[0])
        anchor_residuals = {}
        for j in range(problem.num_segments - 1):
            carried = kernels.segment_tangent(segment_starts[j], carried)[1] + relaxed_defects[j]
            if j + 1 in anchor_targets:
                # the anchor misfit the linearised step starts from: the node
                # moved by its carried defects
                anchor_residuals[j + 1] = np.asarray(
                    problem.observe(interior_states[j] + carried) - anchor_targets[j + 1], dtype=np.float64
                ).ravel()
        carried_end = kernels.segment_tangent(segment_starts[-1], carried)[1]
        condensed_residual = np.asarray(data_residual + problem.observe(carried_end), dtype=np.float64).ravel()
        recorder.optimizer_time += time.time() - start_time
        cost += num_controls
        # ---------------- ↑ condensed sensitivities ↑ ----------------

        normal_matrix = sensitivity.T @ sensitivity
        gradient = sensitivity.T @ condensed_residual
        for node, anchor_sensitivity in anchor_sensitivities.items():
            normal_matrix = normal_matrix + anchor_weight * anchor_sensitivity.T @ anchor_sensitivity
            gradient = gradient + anchor_weight * anchor_sensitivity.T @ anchor_residuals[node]
        scaling = np.diag(np.diag(normal_matrix))
        accepted = False
        for _ in range(10):
            start_time = time.time()
            step = np.linalg.solve(normal_matrix + damping * scaling, -gradient)
            control_step = jnp.asarray(step.reshape(control.shape), dtype=control.dtype)
            # Back-substitute the interiors through the linearised segments.
            interior_step = kernels.initial_state_tangents(control, control_step[None])[0]
            interior_steps = []
            for j in range(problem.num_segments - 1):
                interior_step = kernels.segment_tangent(segment_starts[j], interior_step)[1] + relaxed_defects[j]
                interior_steps.append(interior_step)
            recorder.optimizer_time += time.time() - start_time
            # The damping only shortens the control step; the interior update
            # always takes the full Newton step on the defects. A backtracking
            # line search on the whole (control, interior) step globalises it.
            for step_length in (1.0, 0.5, 0.25, 0.125):
                start_time = time.time()
                new_control = control + step_length * control_step
                new_interior_states = [
                    interior_states[j] + step_length * interior_steps[j] for j in range(problem.num_segments - 1)
                ]
                trial = lifted_residuals(new_control, new_interior_states)
                trial_merit, trial_data_loss, trial_defect = merit_of(trial[1], trial[2], new_interior_states)
                recorder.optimizer_time += time.time() - start_time
                cost += 1.0 / 3.0
                if trial_merit < merit:
                    control = new_control
                    interior_states = new_interior_states
                    segment_starts, defects, data_residual = trial
                    merit, data_loss, total_defect = trial_merit, trial_data_loss, trial_defect
                    accepted = True
                    break
            if accepted:
                damping = max(damping / 3.0, 1e-8)
                break
            damping = min(4.0 * damping, 1e12)
        if not accepted:
            break
        if settings.early_restore and first_reversal_node > 1:
            interior_states = restore_early_nodes(control, interior_states)
            segment_starts, defects, data_residual = lifted_residuals(control, interior_states)
        if relaxation < 1.0:
            # Tighten the relaxed matching conditions.
            relaxation = min(1.0, relaxation + settings.relaxation_increment)
        # Release the anchors.
        anchor_weight *= settings.anchor_decay
        # The merit of the new relaxation, anchor weight and early nodes.
        merit, data_loss, total_defect = merit_of(defects, data_residual, interior_states)
    return control[None], recorder


def run_gauss_newton_penalty(problem, kernels, observations, start_controls, true_controls, settings, checkpointer):
    """
    Penalty multiple shooting, Gauss-Newton, one run, followed by single-
    shooting Gauss-Newton from its control.

    The variables are the control p and the interior states s_1..s_{M-1},
    started as in ``run_gauss_newton_multiple`` (``settings.node_start``). The
    merit is 1/2 ||data residual||^2 + w/2 sum_j ||Phi_h(s_j) - s_{j+1}||^2:
    continuity is a misfit, not a constraint, so a defect can be closed by
    either side of it. The weight w starts at ``continuity_weight`` and grows
    by ``continuity_growth`` per accepted step to ``continuity_max`` (a
    continuation on the slack of the matching conditions): the lifted problem
    is first solved loosely and then tightened onto a trajectory.

    Each node moves in the subspace of Bock's condensed step, node j by
    T_j a_j + c_j beta_j, with T_j the control tangents carried along the
    current nodes (T_0 = d s_0 / d p, T_{j+1} = G_j T_j) and c_j the carried
    defects (c_1 = r_0, c_{j+1} = G_j c_j + r_j), but with its own
    coefficients a_j, beta_j (a_0 is the control step). Bock's step is the
    member a_j = a_0, beta_j = 1: it satisfies the linearised matching
    conditions exactly, so the step tends to Bock's as w grows. The
    Gauss-Newton system in (a_0..a_{M-1}, beta_1..beta_{M-1}) costs Bock's
    forward-mode sweep (num_controls + 1 tangents per segment); damping, line
    search and stopping rule follow ``run_gauss_newton_multiple``. The lifted
    phase ends with the first accepted step at ``continuity_max`` (or when no
    damped step is accepted, or after ``num_iterations``), and
    ``polish_iterations`` iterations of ``run_gauss_newton`` follow; the trace
    column ``phase`` tells the two apart. With KH_PENALTY_CHECK=1 the first
    step prints its node coefficients and the merit change along it against
    the Gauss-Newton model (an implementation check).

    Args:
        problem: The reconstruction problem (forward-mode differentiable).
        kernels: The method kernels of the problem.
        observations: The terminal observation y, (1, 2, n, n).
        start_controls: The cold start, (1, num_modes, 2).
        true_controls: The truth (only for the IC-error trace).
        settings: The method settings.
        checkpointer: Unused; a run fits in one job.

    Returns:
        The final control (1, num_modes, 2) and the recorder.
    """
    observation = observations[0]
    control = start_controls[0]
    num_controls = control.size
    num_segments = problem.num_segments
    num_variables = num_segments * num_controls + num_segments - 1
    basis = jnp.eye(num_controls, dtype=control.dtype).reshape((num_controls,) + control.shape)
    observe_all = jax.vmap(problem.observe)
    check_linearisation = os.environ.get("KH_PENALTY_CHECK") == "1"
    recorder = Recorder()

    # ---------------- ↓ lifted residuals, merit, variable layout ↓ ----------------
    def lifted_residuals(control, interior_states):
        segment_starts = [problem.initial_state(control)] + interior_states
        segment_ends = [kernels.segment_map(state) for state in segment_starts]
        defects = [segment_ends[j] - segment_starts[j + 1] for j in range(num_segments - 1)]
        data_residual = problem.observe(segment_ends[-1]) - observation
        return segment_starts, defects, data_residual

    def merit_of(defects, data_residual, weight):
        data_loss = 0.5 * float(jnp.sum(data_residual ** 2))
        total_defect = float(sum(jnp.sum(defect ** 2) for defect in defects))
        return data_loss + 0.5 * weight * total_defect, data_loss, total_defect

    def control_columns(node):
        return slice(node * num_controls, (node + 1) * num_controls)

    def beta_column(node):
        return num_segments * num_controls + node - 1

    def host_rows(vectors):
        return np.asarray(vectors, dtype=np.float64).reshape(vectors.shape[0], -1)
    # ---------------- ↑ lifted residuals, merit, variable layout ↑ ----------------

    interior_states = []
    state = problem.initial_state(control)
    for _ in range(num_segments - 1):
        state = kernels.segment_map(state)
        interior_states.append(state)
    if settings.node_start == "reversal":
        estimates = reversal_node_estimates(problem, kernels.segment_map, observation)
        first_reversal_node = int(math.ceil(settings.reversal_fraction * num_segments))
        for node in range(max(first_reversal_node, 1), num_segments):
            interior_states[node - 1] = estimates[node]
    weight = settings.continuity_weight
    segment_starts, defects, data_residual = lifted_residuals(control, interior_states)
    merit, data_loss, total_defect = merit_of(defects, data_residual, weight)

    damping = settings.damping_initial
    cost = 0.0
    lifted_done = False
    for iteration in range(settings.num_iterations + 1):
        recorder.record(
            step=iteration,
            cost=cost,
            objective=data_loss,
            feasible_loss=kernels.feasible_loss(control[None], observations)[0],
            ic_error=kernels.ic_error(control[None], true_controls)[0],
            gradient_norm=float("nan"),
            defect=total_defect,
            damping=damping,
            continuity_weight=weight,
            phase=1,
        )
        if lifted_done or iteration == settings.num_iterations:
            break

        # ---------------- ↓ Bock's sweep: carried tangents and defects ↓ ----------------
        node_tangents = [recorder.timed(kernels.initial_state_tangents, control, basis)]
        carried = [jnp.zeros_like(segment_starts[0])]
        propagated_carried = []
        for j in range(num_segments):
            directions = jnp.concatenate([node_tangents[j], carried[j][None]], axis=0)
            propagated = recorder.timed(kernels.segment_tangents, segment_starts[j], directions)
            node_tangents.append(propagated[:num_controls])
            propagated_carried.append(propagated[num_controls])
            if j < num_segments - 1:
                carried.append(propagated[num_controls] + defects[j])
        cost += num_controls
        # ---------------- ↑ Bock's sweep: carried tangents and defects ↑ ----------------

        # ---------------- ↓ Gauss-Newton system of the penalty merit ↓ ----------------
        # Every residual block is linearised as constant + sum_i u_i (S z)_i
        # with a few state-sized vectors u_i and a selection S of the
        # variables z, so the normal matrix is assembled from small Gram
        # matrices.
        start_time = time.time()
        normal_matrix = np.zeros((num_variables, num_variables))
        gradient = np.zeros(num_variables)

        def accumulate(rows, constant, selection, scale):
            gram = rows @ rows.T
            normal_matrix[:] += scale * selection.T @ gram @ selection
            gradient[:] += scale * selection.T @ (rows @ constant)

        identity = np.eye(num_controls)
        for j in range(num_segments - 1):
            # defect j: r_j + T_{j+1} (a_j - a_{j+1}) + (G_j c_j) beta_j - c_{j+1} beta_{j+1}
            rows = np.concatenate([
                host_rows(node_tangents[j + 1]),
                host_rows(propagated_carried[j][None]),
                host_rows(carried[j + 1][None]),
            ])
            selection = np.zeros((num_controls + 2, num_variables))
            selection[:num_controls, control_columns(j)] = identity
            selection[:num_controls, control_columns(j + 1)] = -identity
            if j >= 1:
                selection[num_controls, beta_column(j)] = 1.0
            selection[num_controls + 1, beta_column(j + 1)] = -1.0
            accumulate(rows, host_rows(defects[j][None])[0], selection, weight)
        # data: r + H T_M a_{M-1} + H (G_{M-1} c_{M-1}) beta_{M-1}
        rows = np.concatenate([
            host_rows(observe_all(node_tangents[num_segments])),
            host_rows(problem.observe(propagated_carried[num_segments - 1])[None]),
        ])
        selection = np.zeros((num_controls + 1, num_variables))
        selection[:num_controls, control_columns(num_segments - 1)] = identity
        selection[num_controls, beta_column(num_segments - 1)] = 1.0
        accumulate(rows, host_rows(data_residual[None])[0], selection, 1.0)
        # Variables the merit does not see (beta of a node without carried
        # defect) are held.
        diagonal = np.diag(normal_matrix).copy()
        active = diagonal > 1e-12 * diagonal.max()
        active_matrix = normal_matrix[np.ix_(active, active)]
        active_gradient = gradient[active]
        scaling = np.diag(diagonal[active])
        recorder.optimizer_time += time.time() - start_time
        # ---------------- ↑ Gauss-Newton system of the penalty merit ↑ ----------------

        accepted = False
        for trial_index in range(10):
            start_time = time.time()
            step = np.zeros(num_variables)
            step[active] = np.linalg.solve(active_matrix + damping * scaling, -active_gradient)
            control_step = jnp.asarray(step[control_columns(0)].reshape(control.shape), dtype=control.dtype)
            interior_steps = [
                jnp.tensordot(jnp.asarray(step[control_columns(node)], dtype=control.dtype), node_tangents[node], axes=1)
                + float(step[beta_column(node)]) * carried[node]
                for node in range(1, num_segments)
            ]
            recorder.optimizer_time += time.time() - start_time
            if check_linearisation and iteration == 0 and trial_index == 0:
                coefficients = step[:num_segments * num_controls].reshape(num_segments, num_controls)
                spread = np.linalg.norm(coefficients - coefficients[0], axis=1) / max(np.linalg.norm(coefficients[0]), 1e-30)
                print(f"  [check] w {weight:g}: |a_j - a_0| / |a_0| = {np.array2string(spread, precision=3)}", flush=True)
                print(f"  [check] beta_j = {np.array2string(step[num_segments * num_controls:], precision=4)}"
                      f" (held: {np.flatnonzero(~active[num_segments * num_controls:]) + 1})", flush=True)
                # The coefficients are ill-determined where the control
                # tangents are nearly dependent; compare with Bock's own
                # (undamped, condensed) step in control and node space.
                sensitivity = host_rows(observe_all(node_tangents[num_segments])).T
                condensed_residual = host_rows(
                    (data_residual + problem.observe(propagated_carried[num_segments - 1]))[None]
                )[0]
                bock_control_step = np.linalg.lstsq(sensitivity, -condensed_residual, rcond=None)[0]
                print(f"  [check] |a_0 - Bock| / |Bock| = "
                      f"{np.linalg.norm(coefficients[0] - bock_control_step) / np.linalg.norm(bock_control_step):.3e}",
                      flush=True)
                node_mismatch = []
                for node in range(1, num_segments):
                    bock_node_step = jnp.tensordot(
                        jnp.asarray(bock_control_step, dtype=control.dtype), node_tangents[node], axes=1
                    ) + carried[node]
                    node_mismatch.append(float(
                        jnp.linalg.norm(interior_steps[node - 1] - bock_node_step) / jnp.linalg.norm(bock_node_step)
                    ))
                print(f"  [check] node steps vs Bock: {np.array2string(np.array(node_mismatch), precision=3)}",
                      flush=True)
                for step_length in (1e-1, 1e-2):
                    trial = lifted_residuals(
                        control + step_length * control_step,
                        [interior_states[j] + step_length * interior_steps[j] for j in range(num_segments - 1)],
                    )
                    actual = merit_of(trial[1], trial[2], weight)[0] - merit
                    model = step_length * gradient @ step + 0.5 * step_length ** 2 * step @ normal_matrix @ step
                    print(f"  [check] step length {step_length:g}: merit change {actual:.6e}, "
                          f"Gauss-Newton model {model:.6e}, ratio {actual / model:.4f}", flush=True)
            for step_length in (1.0, 0.5, 0.25, 0.125):
                start_time = time.time()
                new_control = control + step_length * control_step
                new_interior_states = [
                    interior_states[j] + step_length * interior_steps[j] for j in range(num_segments - 1)
                ]
                trial = lifted_residuals(new_control, new_interior_states)
                trial_merit, trial_data_loss, trial_defect = merit_of(trial[1], trial[2], weight)
                recorder.optimizer_time += time.time() - start_time
                cost += 1.0 / 3.0
                if trial_merit < merit:
                    control = new_control
                    interior_states = new_interior_states
                    segment_starts, defects, data_residual = trial
                    merit, data_loss, total_defect = trial_merit, trial_data_loss, trial_defect
                    accepted = True
                    break
            if accepted:
                damping = max(damping / 3.0, 1e-8)
                break
            damping = min(4.0 * damping, 1e12)
        if not accepted or weight >= settings.continuity_max:
            lifted_done = True
        else:
            # Tighten the matching conditions; the merit of the new weight.
            weight = min(weight * settings.continuity_growth, settings.continuity_max)
            merit, data_loss, total_defect = merit_of(defects, data_residual, weight)

    if settings.polish_iterations > 0:
        # Phase 2: single-shooting Gauss-Newton from the lifted control,
        # appended to the trace.
        polish_start = time.time() - recorder.start_time
        lifted_steps = recorder.columns["step"][-1]
        lifted_cost = recorder.columns["cost"][-1]
        lifted_optimizer_time = recorder.optimizer_time
        final_controls, polish = run_gauss_newton(
            problem,
            kernels,
            observations,
            control[None],
            true_controls,
            settings._replace(num_iterations=settings.polish_iterations),
            None,
        )
        for row in range(len(polish.columns["step"])):
            recorder.record(
                step=lifted_steps + 1 + polish.columns["step"][row],
                cost=lifted_cost + polish.columns["cost"][row],
                objective=polish.columns["objective"][row],
                feasible_loss=polish.columns["feasible_loss"][row],
                ic_error=polish.columns["ic_error"][row],
                gradient_norm=polish.columns["gradient_norm"][row],
                defect=0.0,
                damping=polish.columns["damping"][row],
                continuity_weight=float("nan"),
                phase=2,
                wall_time=polish_start + polish.columns["wall_time"][row],
                optimizer_time=lifted_optimizer_time + polish.columns["optimizer_time"][row],
            )
        recorder.optimizer_time = lifted_optimizer_time + polish.optimizer_time
        return final_controls, recorder
    return control[None], recorder


METHOD_RUNNERS = {
    "single": run_single_shooting,
    "coarse_to_fine": run_coarse_to_fine,
    "multiple": run_multiple_shooting,
    "gauss_newton": run_gauss_newton,
    "gauss_newton_multiple": run_gauss_newton_multiple,
    "gauss_newton_penalty": run_gauss_newton_penalty,
}


# ============================================================================
#  Run cells
# ============================================================================

def cell_path(output_directory, method, tag, horizon, truth_index, init_index):
    """Return the file of one (method, hyperparameter tag, horizon, truth, init) cell."""
    return os.path.join(
        output_directory,
        f"{method}__{tag}__T{horizon:g}__t{truth_index}_i{init_index}.npz",
    )


def run_cells(
    method,
    horizon,
    runs,
    settings,
    output_directory,
    backend=PALLAS,
    kh_params=KHParams(),
    num_segments=NUM_SEGMENTS,
    batch_size=8,
    checkpoint_minutes=20.0,
):
    """
    Run one method on a list of (tag, learning rate, truth, init) runs at one
    horizon.

    The runs are optimised in batches of independent runs (Gauss-Newton one
    at a time), each with its own learning rate, so a learning-rate grid
    shares the GPU; one compilation serves all batches of the job. A cell
    whose file exists is skipped.

    Args:
        method: One of METHOD_RUNNERS.
        horizon: The horizon in KH growth times.
        runs: (tag, learning_rate, truth_index, init_index) tuples; the tag
            names the cell files of the run's configuration.
        settings: The method settings (the learning rate is per run).
        output_directory: Where the cell files go.
        backend: PALLAS or NATIVE_JAX.
        kh_params: The KH parameters (grid, modes, derivative options).
        num_segments: The number M of segments.
        batch_size: Runs optimised together.
        checkpoint_minutes: Interval of the optimiser-state checkpoints.
    """
    os.makedirs(output_directory, exist_ok=True)
    # Gauss-Newton builds Jacobians by forward-mode tangents, which the
    # checkpointed reverse-mode loop does not support.
    gauss_newton_method = method in ("gauss_newton", "gauss_newton_multiple", "gauss_newton_penalty")
    differentiation_mode = FORWARDS if gauss_newton_method else BACKWARDS
    if gauss_newton_method:
        batch_size = 1
    problem = build_problem(
        kh_params,
        horizon,
        num_segments=num_segments,
        backend=backend,
        differentiation_mode=differentiation_mode,
    )
    kernels = build_kernels(problem)
    terminal_prediction = jax.jit(problem.terminal_prediction)

    pending = [
        run for run in runs
        if not os.path.exists(cell_path(output_directory, method, run[0], horizon, run[2], run[3]))
    ]
    observation_of_truth = {}
    reversal_control_of_truth = {}
    # The first reversal node and the sub-problem up to it (two-stage start).
    node_index = int(math.ceil(settings.reversal_fraction * num_segments))
    node_fraction = node_index / num_segments
    sub_problem = sub_kernels = None
    for batch_start in range(0, len(pending), batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        # A short last batch is padded with copies of its last run (discarded
        # afterwards), so it reuses the compilation of the full batches.
        num_padding = batch_size - len(batch) if len(pending) > batch_size else 0
        padded_batch = batch + [batch[-1]] * num_padding
        for _, _, truth_index, _ in batch:
            if truth_index not in observation_of_truth:
                observation_of_truth[truth_index] = terminal_prediction(truth_control(problem, truth_index))
        true_controls = jnp.stack([truth_control(problem, run[2]) for run in padded_batch])
        observations = jnp.stack([observation_of_truth[run[2]] for run in padded_batch])
        if settings.control_start == "reversal":
            # The projection of the time-reversed observation at t = 0; it
            # depends on the truth only, not on the init seed.
            for _, _, truth_index, _ in batch:
                if truth_index not in reversal_control_of_truth:
                    initial_estimate = reversal_node_estimates(
                        problem, kernels.segment_map, observation_of_truth[truth_index]
                    )[0]
                    reversal_control_of_truth[truth_index] = project_onto_controls(
                        problem, problem.observe(initial_estimate)[1]
                    )
            start_controls = jnp.stack([reversal_control_of_truth[run[2]] for run in padded_batch])
        else:
            start_controls = jnp.stack([cold_start_control(problem, run[3]) for run in padded_batch])
        start_optimizer_time = 0.0
        if settings.control_start == "reversal_node":
            # Two-stage single shooting (an ablation of the reversal-node
            # multiple shooting): Gauss-Newton fits the cold control to the
            # time-reversed velocity at the first reversal node, over that
            # shorter horizon (the same segment length, so the same segment
            # map), and the result starts the full-horizon run.
            if sub_problem is None:
                sub_problem = build_problem(
                    kh_params,
                    horizon * node_fraction,
                    num_segments=node_index,
                    backend=backend,
                    differentiation_mode=differentiation_mode,
                )
                sub_kernels = build_kernels(sub_problem)
            fitted_controls = []
            for run_index, (_, _, truth_index, _) in enumerate(padded_batch):
                node_target = problem.observe(
                    reversal_node_estimates(problem, kernels.segment_map, observation_of_truth[truth_index])[node_index]
                )
                fitted, stage_recorder = METHOD_RUNNERS["gauss_newton"](
                    sub_problem,
                    sub_kernels,
                    node_target[None],
                    start_controls[run_index][None],
                    true_controls[run_index][None],
                    settings,
                    None,
                )
                fitted_controls.append(fitted[0])
                start_optimizer_time += stage_recorder.optimizer_time
            start_controls = jnp.stack(fitted_controls)
        batch_settings = settings._replace(
            learning_rate=jnp.asarray([run[1] for run in padded_batch], dtype=jnp.float32)
        )
        reference_losses = np.asarray(kernels.feasible_loss(jnp.zeros_like(true_controls), observations))

        checkpoint_directory = os.path.join(output_directory, "checkpoints")
        os.makedirs(checkpoint_directory, exist_ok=True)
        # The name holds every run's configuration tag, so batches of different
        # configurations (or problems) never share a checkpoint.
        batch_tags = "+".join(sorted({run[0] for run in batch}))
        batch_name = "_".join(f"l{rate:g}t{truth}i{init}" for _, rate, truth, init in batch)
        checkpoint_path = os.path.join(
            checkpoint_directory,
            f"{method}__{batch_tags}__T{horizon:g}__{batch_name}.npz",
        )
        checkpointer = Checkpointer(checkpoint_path, 60.0 * checkpoint_minutes)

        start_time = time.time()
        final_controls, recorder = METHOD_RUNNERS[method](
            problem,
            kernels,
            observations,
            start_controls,
            true_controls,
            batch_settings,
            checkpointer,
        )
        wall_time = time.time() - start_time
        final_losses = np.asarray(kernels.feasible_loss(final_controls, observations))
        final_ic_errors = np.asarray(kernels.ic_error(final_controls, true_controls))

        for run_index, (tag, learning_rate, truth_index, init_index) in enumerate(batch):
            trace = recorder.run_arrays(run_index)
            path = cell_path(output_directory, method, tag, horizon, truth_index, init_index)
            np.savez(
                path,
                method=method,
                tag=tag,
                horizon=horizon,
                truth_index=truth_index,
                init_index=init_index,
                num_cells=problem.kh_params.num_cells,
                num_segments=problem.num_segments,
                wavenumber_max=problem.kh_params.wavenumber_max,
                backend=backend,
                rematerialize_stages=problem.kh_params.rematerialize_stages,
                frozen_weno_weights=problem.kh_params.frozen_weno_weights,
                settings_json=json.dumps(settings._replace(learning_rate=learning_rate)._asdict()),
                reference_loss=reference_losses[run_index],
                start_control=np.asarray(start_controls[run_index]),
                true_control=np.asarray(true_controls[run_index]),
                final_control=np.asarray(final_controls[run_index]),
                final_feasible_loss=final_losses[run_index],
                final_relative_loss=final_losses[run_index] / reference_losses[run_index],
                final_ic_error=final_ic_errors[run_index],
                batch_size=len(padded_batch),
                batch_wall_time=wall_time,
                batch_optimizer_time=recorder.optimizer_time,
                start_optimizer_time=start_optimizer_time,
                **trace,
            )
            print(
                f"{method} {tag} T={horizon:g} truth {truth_index} init {init_index}: "
                f"ic {trace['trace_ic_error'][0]:.3f} -> {final_ic_errors[run_index]:.4f}, "
                f"J/J(0) {trace['trace_feasible_loss'][0] / reference_losses[run_index]:.2e} -> "
                f"{final_losses[run_index] / reference_losses[run_index]:.2e} "
                f"[batch of {len(padded_batch)}: {recorder.optimizer_time:.0f} s optimizer, "
                f"{wall_time:.0f} s wall]",
                flush=True,
            )
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


# ============================================================================
#  Timing and diagnostics
# ============================================================================

def time_steps(
    horizons,
    backends,
    batch_sizes,
    num_repeats,
    output_path,
    kh_params=KHParams(),
    skip_jacobian=False,
):
    """
    Time one optimisation step of every method at N = 256 (compile excluded),
    for batches of independent runs.

    Args:
        horizons: The horizons in KH growth times.
        backends: The backends to time (PALLAS, NATIVE_JAX).
        batch_sizes: The batch sizes of the Adam methods.
        num_repeats: Timed repetitions per step.
        output_path: The JSON file the timings are written to.
        kh_params: The KH parameters (grid, rematerialisation).
        skip_jacobian: Leave out the Gauss-Newton Jacobian.
    """

    def timed(function, *arguments):
        start_time = time.time()
        jax.block_until_ready(function(*arguments))
        compile_and_first = time.time() - start_time
        start_time = time.time()
        for _ in range(num_repeats):
            jax.block_until_ready(function(*arguments))
        return compile_and_first, (time.time() - start_time) / num_repeats

    results = []
    for backend in backends:
        backend_name = "pallas" if backend == PALLAS else "native"
        for horizon in horizons:
            problem = build_problem(kh_params, horizon, backend=backend)
            kernels = build_kernels(problem)
            observation = jax.jit(problem.terminal_prediction)(truth_control(problem, 0))
            for batch_size in batch_sizes:
                controls = jnp.stack([
                    cold_start_control(problem, init_index) for init_index in range(batch_size)
                ])
                observations = jnp.stack([observation] * batch_size)
                initial_states = jax.vmap(problem.initial_state)(controls)
                variables = {
                    "control": controls,
                    "interior_states": jnp.stack([initial_states] * (problem.num_segments - 1), axis=1),
                }
                multipliers = jnp.zeros_like(variables["interior_states"])
                penalties = jnp.ones(batch_size)
                row = {
                    "backend": backend_name,
                    "horizon": horizon,
                    "batch_size": batch_size,
                    "rematerialize_stages": kh_params.rematerialize_stages,
                    "xla_flags": os.environ.get("XLA_FLAGS", ""),
                }
                for name, function, arguments in (
                    ("forward", kernels.feasible_loss, (controls, observations)),
                    ("single_gradient", kernels.single_loss_and_gradient, (controls, observations)),
                    (
                        "multiple_step",
                        kernels.multiple_adam_step,
                        (variables, adam_init(variables), observations, multipliers, penalties, 1e-3),
                    ),
                ):
                    # A batch too large for the GPU is recorded, not fatal.
                    try:
                        first, steady = timed(function, *arguments)
                    except jax.errors.JaxRuntimeError as error:
                        row[name + "_s"] = None
                        print(
                            f"{backend_name} T={horizon:g} B={batch_size} {name}: {str(error)[:80]}",
                            flush=True,
                        )
                        continue
                    row[name + "_first_s"] = first
                    row[name + "_s"] = steady
                    print(
                        f"{backend_name} T={horizon:g} B={batch_size} {name}: first {first:.1f} s, "
                        f"step {steady:.3f} s ({steady / batch_size:.3f} s per run)",
                        flush=True,
                    )
                results.append(row)

            if skip_jacobian:
                continue
            forward_mode_problem = build_problem(
                kh_params,
                horizon,
                backend=backend,
                differentiation_mode=FORWARDS,
            )
            first, steady = timed(
                build_kernels(forward_mode_problem).residual_and_jacobian,
                cold_start_control(problem, 0),
                observation,
            )
            results.append({
                "backend": backend_name,
                "horizon": horizon,
                "batch_size": 1,
                "gauss_newton_jacobian_first_s": first,
                "gauss_newton_jacobian_s": steady,
            })
            print(
                f"{backend_name} T={horizon:g} Gauss-Newton Jacobian: first {first:.1f} s, step {steady:.3f} s",
                flush=True,
            )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(results, file, indent=1)
    print(f"-> {output_path}")


def diagnose(
    horizon,
    truth_index,
    num_slice_points,
    slice_extent,
    num_power_iterations,
    output_path,
    num_cells=256,
    backend=PALLAS,
    frozen_weno_weights=False,
):
    """
    Diagnostics of the forward map at the truth.

    * Model mismatch of the original campaign: its single-shooting loss
      integrated once over T while the observation came from M segments; the
      loss of the truth under that one-shot map is reported against J_ref.
    * Singular values of the Jacobian of the terminal velocity with respect to
      the control at the truth, relative to those of the seed map (the
      control-restricted tangent growth) and its condition number.
    * The largest singular value of the full-state tangent map
      delta u_0 -> delta H(u_T) at the truth (power iteration on J^T J with
      forward- and reverse-mode products), the unrestricted tangent growth.
    * Loss slices J(p* + a d) along the direction to a cold start and along
      two random directions (|d| = |p*|), for multimodality.

    Args:
        horizon: The horizon in KH growth times.
        truth_index: The truth seed.
        num_slice_points: Points per slice.
        slice_extent: The slices span a in [-extent, extent].
        num_power_iterations: Power iterations for the full-state growth.
        output_path: The npz file the diagnostics are written to.
        num_cells: The grid size per axis.
        backend: PALLAS or NATIVE_JAX.
        frozen_weno_weights: Differentiate with frozen WENO weights.
    """
    kh_params = KHParams(num_cells=num_cells, frozen_weno_weights=frozen_weno_weights)
    # Forward evaluations and forward-mode tangents use the plain adaptive
    # loop; the reverse-mode products of the power iteration the checkpointed one.
    problem = build_problem(
        kh_params,
        horizon,
        backend=backend,
        differentiation_mode=FORWARDS,
    )
    reverse_mode_problem = build_problem(kh_params, horizon, backend=backend)
    true_control = truth_control(problem, truth_index)
    terminal_prediction = jax.jit(problem.terminal_prediction)
    observation = terminal_prediction(true_control)
    loss = jax.jit(lambda control: terminal_loss(problem, control, observation))
    reference_loss = float(loss(jnp.zeros_like(true_control)))

    # ---------------- ↓ one-shot vs segmented forward map ↓ ----------------
    full_config, full_params, registered_variables = build_simulation(
        problem.kh_params,
        horizon * problem.kh_params.growth_time,
        backend,
        FORWARDS,
    )
    initial_state = problem.initial_state(true_control)
    full_config = finalize_config(full_config, initial_state.shape)
    one_shot_final = jax.jit(
        lambda state: time_integration(state, full_config, full_params, registered_variables)
    )(initial_state)
    one_shot_loss = 0.5 * float(jnp.sum((problem.observe(one_shot_final) - observation) ** 2))
    print(
        f"T={horizon:g}: J(0) {reference_loss:.4e}; "
        f"one-shot map at the truth J/J(0) {one_shot_loss / reference_loss:.3e}"
    )
    # ---------------- ↑ one-shot vs segmented forward map ↑ ----------------

    # ---------------- ↓ control-restricted tangent growth ↓ ----------------
    flat_shape = (-1, true_control.size)
    terminal_jacobian = jax.jit(jax.jacfwd(problem.terminal_prediction))(true_control).reshape(flat_shape)
    seed_jacobian = jax.jacfwd(problem.seed_field)(true_control).reshape(flat_shape)
    terminal_singular_values = np.linalg.svd(np.asarray(terminal_jacobian, np.float64), compute_uv=False)
    seed_singular_values = np.linalg.svd(np.asarray(seed_jacobian, np.float64), compute_uv=False)
    tangent_growth = terminal_singular_values / seed_singular_values.max()
    print(
        f"T={horizon:g}: control-restricted tangent growth sigma/sigma_seed = "
        f"{np.array2string(tangent_growth, precision=2)}"
    )
    print(f"T={horizon:g}: condition number {terminal_singular_values[0] / terminal_singular_values[-1]:.1f}")
    # ---------------- ↑ control-restricted tangent growth ↑ ----------------

    # ---------------- ↓ full-state tangent growth ↓ ----------------
    def terminal_velocity_of_state(state, segment_map):
        def advance_one_segment(state, _):
            return segment_map(state), None

        final_state, _ = jax.lax.scan(advance_one_segment, state, None, length=problem.num_segments)
        return problem.observe(final_state)

    tangent_product = jax.jit(
        lambda state, direction: jax.jvp(
            lambda state: terminal_velocity_of_state(state, problem.segment_map),
            (state,),
            (direction,),
        )[1]
    )
    adjoint_product = jax.jit(
        lambda state, cotangent: jax.vjp(
            lambda state: terminal_velocity_of_state(state, reverse_mode_problem.segment_map),
            state,
        )[1](cotangent)[0]
    )
    power_direction = jax.random.normal(jax.random.PRNGKey(2000), initial_state.shape)
    power_direction = power_direction / jnp.linalg.norm(power_direction)
    full_state_growth = []
    for _ in range(num_power_iterations):
        power_image = tangent_product(initial_state, power_direction)
        full_state_growth.append(float(jnp.linalg.norm(power_image)))
        power_direction = adjoint_product(initial_state, power_image)
        power_direction = power_direction / jnp.linalg.norm(power_direction)
    print(
        f"T={horizon:g}: full-state tangent growth (power iteration) "
        f"{np.array2string(np.array(full_state_growth), precision=2)}"
    )
    # ---------------- ↑ full-state tangent growth ↑ ----------------

    # ---------------- ↓ loss slices through the truth ↓ ----------------
    norm_truth = jnp.linalg.norm(true_control)
    directions = [cold_start_control(problem, 0) - true_control]
    for key_index in range(2):
        direction = jax.random.normal(jax.random.PRNGKey(1000 + key_index), true_control.shape)
        directions.append(direction)
    directions = [direction * norm_truth / jnp.linalg.norm(direction) for direction in directions]
    offsets = np.linspace(-slice_extent, slice_extent, num_slice_points)
    slices = np.array([
        [float(loss(true_control + offset * direction)) / reference_loss for offset in offsets]
        for direction in directions
    ])
    # ---------------- ↑ loss slices through the truth ↑ ----------------

    np.savez(
        output_path,
        horizon=horizon,
        truth_index=truth_index,
        reference_loss=reference_loss,
        one_shot_relative_loss=one_shot_loss / reference_loss,
        terminal_singular_values=terminal_singular_values,
        seed_singular_values=seed_singular_values,
        tangent_growth=tangent_growth,
        full_state_growth=np.array(full_state_growth),
        full_state_direction=np.asarray(power_direction),
        full_state_image=np.asarray(tangent_product(initial_state, power_direction)),
        slice_offsets=offsets,
        slice_relative_losses=slices,
        slice_directions=np.asarray(jnp.stack(directions)),
    )
    print(f"-> {output_path}")


# ============================================================================
#  Reduction and figures (CPU)
# ============================================================================

def load_cells(output_directory, method=None, tag=None, horizon=None, truths=None):
    """
    Load the cell files of a campaign, optionally filtered.

    Args:
        output_directory: The directory of the cell files.
        method: Keep only this method.
        tag: Keep only this hyperparameter tag.
        horizon: Keep only this horizon.
        truths: Keep only these truth seeds.

    Returns:
        A list of dicts, one per cell.
    """
    cells = []
    for path in sorted(glob.glob(os.path.join(output_directory, "*__*__T*__t*_i*.npz"))):
        with np.load(path, allow_pickle=False) as data:
            cell = {name: data[name] for name in data.files}
        cell = {name: (value.item() if value.ndim == 0 else value) for name, value in cell.items()}
        if method is not None and cell["method"] != method:
            continue
        if tag is not None and cell["tag"] != tag:
            continue
        if horizon is not None and float(cell["horizon"]) != float(horizon):
            continue
        if truths is not None and int(cell["truth_index"]) not in truths:
            continue
        cells.append(cell)
    return cells


def wilson_interval(successes, total, z=1.0):
    """Return the Wilson score interval (fraction, low, high); z = 1 is 68 %."""
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    fraction = successes / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / denominator
    return fraction, max(0.0, center - half_width), min(1.0, center + half_width)


def summarize(output_directory, truths, baseline, summary_path):
    """
    Print and store the final metrics of every (method, tag, horizon) group:
    median and IQR of the feasible relative loss and the IC error, the success
    rate, the wall clock, and the paired win fractions against a baseline on
    the same (truth, init) pairs. Per method and horizon the tag with the
    lowest median feasible loss is marked: the tuning rule, which needs no
    truth.

    Args:
        output_directory: The directory of the cell files.
        truths: The truth seeds to include (None: all).
        baseline: (method, tag) of the paired baseline, or None.
        summary_path: The JSON file the summary is written to.
    """
    cells = load_cells(output_directory, truths=truths)
    groups = {}
    for cell in cells:
        groups.setdefault((cell["method"], cell["tag"], float(cell["horizon"])), []).append(cell)

    best_tag = {}
    for (method, tag, horizon), group in groups.items():
        median_loss = float(np.median([cell["final_relative_loss"] for cell in group]))
        if (method, horizon) not in best_tag or median_loss < best_tag[(method, horizon)][1]:
            best_tag[(method, horizon)] = (tag, median_loss)

    summary = []
    def group_order(item):
        (method, tag, horizon), _ = item
        return horizon, method, tag

    for (method, tag, horizon), group in sorted(groups.items(), key=group_order):
        relative_losses = np.array([cell["final_relative_loss"] for cell in group])
        ic_errors = np.array([cell["final_ic_error"] for cell in group])
        # Optimizer seconds per run: a batch's time shared by its runs.
        wall_times = np.array([
            cell.get("batch_optimizer_time", cell.get("optimizer_time", np.nan)) / cell.get("batch_size", 1)
            for cell in group
        ])
        successes = int(np.sum(ic_errors < SUCCESS_THRESHOLD))
        fraction, low, high = wilson_interval(successes, len(group))
        # Optimiser steps (iterations for Gauss-Newton) until the IC error
        # first falls below the threshold, over the successful runs.
        steps_to_success = [
            float(cell["trace_step"][np.argmax(cell["trace_ic_error"] < SUCCESS_THRESHOLD)])
            for cell in group
            if np.any(cell["trace_ic_error"] < SUCCESS_THRESHOLD)
        ]
        row = {
            "method": method,
            "tag": tag,
            "horizon": horizon,
            "num_runs": len(group),
            "selected": best_tag[(method, horizon)][0] == tag,
            "relative_loss_quartiles": np.percentile(relative_losses, [25, 50, 75]).tolist(),
            "ic_error_quartiles": np.percentile(ic_errors, [25, 50, 75]).tolist(),
            "success": [successes, fraction, low, high],
            "steps_to_success_quartiles": (
                np.percentile(steps_to_success, [25, 50, 75]).tolist() if steps_to_success else None
            ),
            "optimizer_time_median_s": float(np.median(wall_times)),
        }

        paired_text = ""
        if baseline is not None and (method, tag) != tuple(baseline):
            baseline_cells = {
                (cell["truth_index"], cell["init_index"]): cell
                for cell in groups.get((baseline[0], baseline[1], horizon), [])
            }
            pairs = [
                (cell, baseline_cells[(cell["truth_index"], cell["init_index"])])
                for cell in group
                if (cell["truth_index"], cell["init_index"]) in baseline_cells
            ]
            if pairs:
                loss_wins = sum(
                    cell["final_relative_loss"] < other["final_relative_loss"] for cell, other in pairs
                )
                ic_wins = sum(cell["final_ic_error"] < other["final_ic_error"] for cell, other in pairs)
                row["paired_wins_vs_baseline"] = {
                    "num_pairs": len(pairs),
                    "loss": loss_wins,
                    "ic_error": ic_wins,
                }
                paired_text = f"  beats baseline: J {loss_wins}/{len(pairs)}, ic {ic_wins}/{len(pairs)}"
        summary.append(row)
        print(
            f"T={horizon:g} {method:15s} {tag:24s}{'*' if row['selected'] else ' '} n={len(group):2d}  "
            f"J/J_ref {row['relative_loss_quartiles'][1]:.2e} "
            f"[{row['relative_loss_quartiles'][0]:.1e}, {row['relative_loss_quartiles'][2]:.1e}]  "
            f"ic {row['ic_error_quartiles'][1]:.3f} "
            f"[{row['ic_error_quartiles'][0]:.3f}, {row['ic_error_quartiles'][2]:.3f}]  "
            f"success {successes}/{len(group)} [{low:.2f}, {high:.2f}]  "
            f"steps to success {np.median(steps_to_success) if steps_to_success else float('nan'):.0f}  "
            f"{row['optimizer_time_median_s'] / 3600:.2f} h" + paired_text
        )
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as file:
        json.dump(summary, file, indent=1)
    print(f"-> {summary_path}")


def paired_decision(output_directory, horizon, multiple, single, num_starts, truths, inits, required_wins=13):
    """
    The pre-registered decision rule (``kh_shooting_notes.md``): multiple
    shooting is demonstrated if it beats single shooting with multi-start
    parity in at least ``required_wins`` paired runs on both feasible metrics.

    Single shooting's result for run (t, i) is, among its starts
    i, i + n, ..., i + (K - 1) n (n = number of reporting inits, K =
    ``num_starts``), the one with the lowest feasible misfit; no truth is used.

    Args:
        output_directory: The directory of the cell files.
        horizon: The decision horizon.
        multiple: (method, tag) of the multiple-shooting configuration.
        single: (method, tag) of the single-shooting configuration.
        num_starts: Starts per run for single shooting (multi-start parity).
        truths: The reporting truths.
        inits: The reporting inits.
        required_wins: Paired wins needed on both metrics.

    Returns:
        True if multiple shooting is demonstrated.
    """

    def cells_by_run(method, tag):
        return {
            (cell["truth_index"], cell["init_index"]): cell
            for cell in load_cells(output_directory, method=method, tag=tag, horizon=horizon)
        }

    def run_time(cell):
        return cell["batch_optimizer_time"] / cell["batch_size"]

    multiple_cells = cells_by_run(*multiple)
    single_cells = cells_by_run(*single)
    loss_wins = ic_wins = num_pairs = 0
    multiple_times, single_times = [], []
    for truth_index in truths:
        for init_index in inits:
            starts = [
                single_cells.get((truth_index, init_index + start * len(inits)))
                for start in range(num_starts)
            ]
            if (truth_index, init_index) not in multiple_cells or any(cell is None for cell in starts):
                print(f"  t{truth_index} i{init_index}: missing cells")
                continue
            multiple_cell = multiple_cells[(truth_index, init_index)]
            single_cell = min(starts, key=lambda cell: cell["final_relative_loss"])
            num_pairs += 1
            loss_win = multiple_cell["final_relative_loss"] < single_cell["final_relative_loss"]
            ic_win = multiple_cell["final_ic_error"] < single_cell["final_ic_error"]
            loss_wins += int(loss_win)
            ic_wins += int(ic_win)
            multiple_times.append(run_time(multiple_cell))
            single_times.append(sum(run_time(cell) for cell in starts))
            print(
                f"  t{truth_index} i{init_index}: MS J/J(0) {multiple_cell['final_relative_loss']:.2e} "
                f"ic {multiple_cell['final_ic_error']:.3f} | SS best of {num_starts} J/J(0) "
                f"{single_cell['final_relative_loss']:.2e} ic {single_cell['final_ic_error']:.3f}"
                f"{'  MS wins both' if loss_win and ic_win else ''}"
            )
    demonstrated = num_pairs > 0 and loss_wins >= required_wins and ic_wins >= required_wins
    print(
        f"T={horizon:g}: MS beats SS (best of {num_starts}) on J in {loss_wins}/{num_pairs}, "
        f"on IC error in {ic_wins}/{num_pairs}; median time per run MS "
        f"{np.median(multiple_times) / 60 if multiple_times else float('nan'):.1f} min, SS "
        f"{np.median(single_times) / 60 if single_times else float('nan'):.1f} min "
        f"-> {'DEMONSTRATED' if demonstrated else 'not demonstrated'}"
    )
    return demonstrated


def trace_column(cell, column):
    """Return a trace column; losses relative to the zero-control loss J(0)."""
    values = cell["trace_" + column]
    if column in ("feasible_loss", "objective"):
        return values / cell["reference_loss"]
    return values


def median_curve(cells, column):
    """Return the common cost axis and the 25/50/75 % curves of a trace column."""
    length = min(len(cell["trace_" + column]) for cell in cells)
    cost = cells[0]["trace_cost"][:length]
    values = np.stack([trace_column(cell, column)[:length] for cell in cells])
    return cost, np.percentile(values, [25, 50, 75], axis=0)


def make_convergence_figure(output_directory, selections, horizons, truths, output_path):
    """
    Convergence figure: rows = horizons; columns = feasible relative terminal
    loss, IC error and the multiple-shooting defect against the cost in
    full-horizon gradient evaluations; median line, IQR band, thin per-run
    lines. The lifted multiple-shooting loss is dashed.

    Args:
        output_directory: The directory of the cell files.
        selections: (method, tag) pairs to show.
        horizons: The horizons (rows).
        truths: The truth seeds to include (None: all).
        output_path: The figure file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(horizons), 3, figsize=(15, 4.2 * len(horizons)), squeeze=False)
    for row_index, horizon in enumerate(horizons):
        for method, tag in selections:
            cells = load_cells(output_directory, method=method, tag=tag, horizon=horizon, truths=truths)
            if not cells:
                continue
            color = METHOD_COLORS[method]
            columns = [("feasible_loss", 0), ("ic_error", 1)]
            if method == "multiple":
                columns.append(("defect", 2))
            for column, axis_index in columns:
                axis = axes[row_index][axis_index]
                for cell in cells:
                    axis.plot(cell["trace_cost"], trace_column(cell, column), color=color, alpha=0.15, lw=0.6)
                cost, (lower, median, upper) = median_curve(cells, column)
                axis.fill_between(cost, lower, upper, color=color, alpha=0.25, lw=0)
                axis.plot(
                    cost,
                    median,
                    color=color,
                    lw=2.2,
                    label=f"{METHOD_LABELS[method]} (n={len(cells)})",
                )
            if method == "multiple":
                cost, (_, median, _) = median_curve(cells, "objective")
                axes[row_index][0].plot(
                    cost,
                    median,
                    color=color,
                    lw=1.4,
                    ls="--",
                    label="multiple shooting, lifted loss (infeasible)",
                )
        labels = [
            r"feasible terminal loss $J(p)/J(0)$",
            r"IC error $\|u_0(p)-u_0(p^\star)\|/\|u_0(p^\star)\|$",
            r"continuity defect $\sum_j\|r_j\|^2$",
        ]
        for axis_index, label in enumerate(labels):
            axis = axes[row_index][axis_index]
            axis.set_yscale("log")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if row_index == len(horizons) - 1:
                axis.set_xlabel("cost [full-horizon gradient evaluations]")
        axes[row_index][1].axhline(SUCCESS_THRESHOLD, color="0.4", lw=0.8, ls=":")
        axes[row_index][0].text(
            -0.3,
            0.5,
            f"$T={horizon:g}\\,t_g$",
            transform=axes[row_index][0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
        )
        axes[row_index][0].legend(fontsize=8, loc="best")
    figure.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    print(f"-> {output_path}")


def vorticity(velocity, cell_size):
    """Return omega_z = d v_y/dx - d v_x/dy (central differences) of a (2, nx, ny) field."""
    dvy_dx = (np.roll(velocity[1], -1, 0) - np.roll(velocity[1], 1, 0)) / (2 * cell_size)
    dvx_dy = (np.roll(velocity[0], -1, 1) - np.roll(velocity[0], 1, 1)) / (2 * cell_size)
    return dvy_dx - dvx_dy


def median_cell(cells):
    """Return the cell with the median final IC error (the lower median for even counts)."""
    order = np.argsort([cell["final_ic_error"] for cell in cells])
    return cells[order[(len(cells) - 1) // 2]]


def make_reconstruction_figure(output_directory, selections, horizons, truths, output_path, cache_path):
    """
    Representative reconstructions: for every horizon, the median-IC-error run
    of each selected method. Columns: truth, then one per method; per horizon
    an initial-state row (v_y') and a final-state row (omega_z of the forward
    integration of that initial state). Needs a GPU unless cached.

    Args:
        output_directory: The directory of the cell files.
        selections: (method, tag) pairs, one column each.
        horizons: The horizons.
        truths: The truth seeds to include (None: all).
        output_path: The figure file (a .svg twin is written too).
        cache_path: The npz cache of the forward-integrated fields.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=True) as data:
            panels = list(data["panels"])
    else:
        panels = []
        for horizon in horizons:
            problem = build_problem(KHParams(), horizon)
            terminal_prediction = jax.jit(problem.terminal_prediction)
            chosen = [
                median_cell(load_cells(output_directory, method=method, tag=tag, horizon=horizon, truths=truths))
                for method, tag in selections
            ]
            cell_size = problem.kh_params.box_size / problem.kh_params.num_cells
            for method_index, cell in enumerate(chosen):
                true_control = jnp.asarray(cell["true_control"])
                final_control = jnp.asarray(cell["final_control"])
                panels.append({
                    "horizon": horizon,
                    "method": selections[method_index][0],
                    "truth_index": int(cell["truth_index"]),
                    "init_index": int(cell["init_index"]),
                    "ic_error": float(cell["final_ic_error"]),
                    "relative_loss": float(cell["final_relative_loss"]),
                    "seed_true": np.asarray(problem.seed_field(true_control)[1]),
                    "seed_final": np.asarray(problem.seed_field(final_control)[1]),
                    "vorticity_true": vorticity(np.asarray(terminal_prediction(true_control)), cell_size),
                    "vorticity_final": vorticity(np.asarray(terminal_prediction(final_control)), cell_size),
                })
        np.savez(cache_path, panels=np.array(panels, dtype=object))

    # Zoom onto the shear layer.
    lower_y, upper_y = 0.28, 0.72
    num_columns = 1 + len(selections)
    num_rows = 2 * len(horizons)
    figure, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(3.6 * num_columns, 2.6 * num_rows),
        constrained_layout=True,
        squeeze=False,
    )
    for horizon_index, horizon in enumerate(horizons):
        horizon_panels = [panel for panel in panels if float(panel["horizon"]) == float(horizon)]
        for kind_index, (true_key, final_key, label) in enumerate((
            ("seed_true", "seed_final", r"$v_y'(t=0)$"),
            ("vorticity_true", "vorticity_final", r"$\omega_z(T)$"),
        )):
            row_index = 2 * horizon_index + kind_index
            fields = [horizon_panels[0][true_key]] + [panel[final_key] for panel in horizon_panels]
            num_cells = fields[0].shape[1]
            cropped = [field[:, int(lower_y * num_cells):int(upper_y * num_cells)] for field in fields]
            limit = float(np.percentile(np.abs(np.stack(cropped)), 99.5))
            titles = ["truth"] + [
                f"{METHOD_LABELS[panel['method']]}\n"
                f"IC err {panel['ic_error']:.3f}, J/J(0) {panel['relative_loss']:.1e}"
                for panel in horizon_panels
            ]
            for column_index, field in enumerate(cropped):
                axis = axes[row_index][column_index]
                image = axis.imshow(
                    field.T,
                    origin="lower",
                    extent=[0, 1, lower_y, upper_y],
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    aspect="auto",
                    rasterized=True,
                )
                if kind_index == 0:
                    axis.set_title(titles[column_index], fontsize=9)
                axis.set_xticks([])
                axis.set_yticks([])
            axes[row_index][0].set_ylabel(f"$T={horizon:g}\\,t_g$\n{label}", fontsize=10)
            figure.colorbar(image, ax=axes[row_index][:], shrink=0.85, location="right", pad=0.01)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    figure.savefig(os.path.splitext(output_path)[0] + ".svg", dpi=160, bbox_inches="tight")
    print(f"-> {output_path}")


# ============================================================================
#  Command line
# ============================================================================

def parse_selections(tags):
    """Parse ``method:tag`` strings into (method, tag) pairs."""
    return [tuple(selection.split(":", 1)) for selection in tags]


def default_tag(method, settings):
    """Return a file-name tag of the hyperparameters a method reads."""
    if method == "single":
        return f"lr{settings.learning_rate:g}_{settings.schedule}_s{settings.num_steps}"
    if method == "coarse_to_fine":
        stages = "-".join(str(cutoff) for cutoff in settings.cutoff_stages)
        return f"lr{settings.learning_rate:g}_{settings.schedule}_k{stages}_s{settings.num_steps}"
    if method == "multiple":
        return (
            f"lr{settings.learning_rate:g}_rho{settings.penalty_initial:g}x{settings.penalty_growth:g}"
            f"_s{settings.inner_steps}x{settings.outer_iterations}"
        )
    if method == "gauss_newton_multiple":
        tag = f"mu{settings.damping_initial:g}_rho{settings.continuity_weight:g}_it{settings.num_iterations}"
        if settings.node_start == "reversal":
            tag += f"_nrev{settings.reversal_fraction:g}"
        if settings.relaxation_initial < 1.0:
            tag += f"_kappa{settings.relaxation_initial:g}+{settings.relaxation_increment:g}"
        if settings.anchor_weight > 0.0:
            tag += f"_anchor{settings.anchor_weight:g}x{settings.anchor_decay:g}"
        if settings.early_restore:
            tag += "_early"
    elif method == "gauss_newton_penalty":
        tag = (
            f"mu{settings.damping_initial:g}_w{settings.continuity_weight:g}x{settings.continuity_growth:g}"
            f"to{settings.continuity_max:g}_it{settings.num_iterations}_pol{settings.polish_iterations}"
        )
        if settings.node_start == "reversal":
            tag += f"_nrev{settings.reversal_fraction:g}"
    else:
        tag = f"mu{settings.damping_initial:g}_it{settings.num_iterations}"
    if settings.control_start == "reversal":
        tag += "_crev"
    elif settings.control_start == "reversal_node":
        tag += f"_cnode{settings.reversal_fraction:g}"
    return tag


def select_gpu():
    """Pick a free GPU with autocvd unless one is already assigned."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        from autocvd import autocvd

        autocvd(num_gpus=1)


def main():
    parser = argparse.ArgumentParser(
        description="Single vs multiple shooting for KH initial-condition reconstruction."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="run one method on (truth, init) pairs at one horizon")
    run_parser.add_argument("--method", choices=sorted(METHOD_RUNNERS), required=True)
    run_parser.add_argument("--horizon", type=float, required=True, help="horizon in KH growth times")
    run_parser.add_argument("--truths", type=int, nargs="+", default=[0])
    run_parser.add_argument("--inits", type=int, nargs="+", default=[0])
    run_parser.add_argument("--tag", default=None, help="hyperparameter tag of the file names (default: derived)")
    run_parser.add_argument("--output-directory", required=True)
    run_parser.add_argument("--native", action="store_true", help="native-JAX backend instead of Pallas")
    run_parser.add_argument("--num-cells", type=int, default=256)
    run_parser.add_argument("--batch-size", type=int, default=8, help="runs optimised together")
    run_parser.add_argument("--checkpoint-minutes", type=float, default=20.0)
    run_parser.add_argument("--rematerialize-stages", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--frozen-weno-weights", action="store_true", help="differentiate with frozen WENO weights")
    run_parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=None,
        help="a learning-rate grid optimised in one batch (overrides --learning-rate)",
    )
    run_parser.add_argument("--wavenumber-max", type=int, default=KHParams().wavenumber_max)
    run_parser.add_argument("--num-segments", type=int, default=NUM_SEGMENTS)
    for field, default in MethodSettings()._asdict().items():
        if field == "cutoff_stages":
            run_parser.add_argument("--cutoff-stages", type=int, nargs="+", default=list(default))
        elif isinstance(default, bool):
            run_parser.add_argument("--" + field.replace("_", "-"), action="store_true")
        else:
            run_parser.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)

    time_parser = commands.add_parser("time", help="time one step of every method")
    time_parser.add_argument("--horizons", type=float, nargs="+", default=[20.0, 60.0])
    time_parser.add_argument("--native", action="store_true", help="also time the native-JAX backend")
    time_parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8])
    time_parser.add_argument("--rematerialize-stages", action=argparse.BooleanOptionalAction, default=True)
    time_parser.add_argument("--skip-jacobian", action="store_true")
    time_parser.add_argument("--num-cells", type=int, default=256)
    time_parser.add_argument("--repeats", type=int, default=3)
    time_parser.add_argument(
        "--output",
        default=os.path.join(REDUCED_DATA_DIRECTORY, "kh_shooting_timing.json"),
    )

    diagnose_parser = commands.add_parser("diagnose", help="forward-map diagnostics at the truth")
    diagnose_parser.add_argument("--horizon", type=float, required=True)
    diagnose_parser.add_argument("--truth", type=int, default=0)
    diagnose_parser.add_argument("--slice-points", type=int, default=41)
    diagnose_parser.add_argument("--slice-extent", type=float, default=1.5)
    diagnose_parser.add_argument("--power-iterations", type=int, default=8)
    diagnose_parser.add_argument("--num-cells", type=int, default=256)
    diagnose_parser.add_argument("--native", action="store_true", help="native-JAX backend instead of Pallas")
    diagnose_parser.add_argument("--frozen-weno-weights", action="store_true")
    diagnose_parser.add_argument("--output", default=None)

    summarize_parser = commands.add_parser("summarize", help="final metrics per (method, tag, horizon)")
    summarize_parser.add_argument("--output-directory", required=True)
    summarize_parser.add_argument("--truths", type=int, nargs="+", default=None)
    summarize_parser.add_argument("--baseline", default=None, help="method:tag of the paired baseline")
    summarize_parser.add_argument(
        "--summary",
        default=os.path.join(REDUCED_DATA_DIRECTORY, "kh_shooting_summary.json"),
    )

    decide_parser = commands.add_parser("decide", help="the pre-registered paired decision rule")
    decide_parser.add_argument("--output-directory", required=True)
    decide_parser.add_argument("--horizon", type=float, required=True)
    decide_parser.add_argument("--multiple", required=True, help="method:tag of multiple shooting")
    decide_parser.add_argument("--single", required=True, help="method:tag of single shooting")
    decide_parser.add_argument("--starts", type=int, default=1, help="single-shooting starts per run")
    decide_parser.add_argument("--truths", type=int, nargs="+", default=[3, 4, 5, 6])
    decide_parser.add_argument("--inits", type=int, nargs="+", default=[0, 1, 2, 3])

    figure_commands = (
        ("plot", "convergence figure"),
        ("recon", "median reconstructions (GPU unless cached)"),
    )
    for name, help_text in figure_commands:
        figure_parser = commands.add_parser(name, help=help_text)
        figure_parser.add_argument("--output-directory", required=True)
        figure_parser.add_argument("--tags", nargs="+", required=True, help="method:tag pairs")
        figure_parser.add_argument("--horizons", type=float, nargs="+", default=[20.0, 60.0])
        figure_parser.add_argument("--truths", type=int, nargs="+", default=None)
        figure_parser.add_argument("--output", required=True)

    arguments = parser.parse_args()
    if arguments.command == "run":
        select_gpu()
        settings = MethodSettings(**{field: getattr(arguments, field) for field in MethodSettings._fields})
        settings = settings._replace(cutoff_stages=tuple(settings.cutoff_stages))
        runs = []
        for learning_rate in arguments.learning_rates or [settings.learning_rate]:
            tag = arguments.tag or default_tag(arguments.method, settings._replace(learning_rate=learning_rate))
            if not arguments.tag:
                if arguments.wavenumber_max != KHParams().wavenumber_max:
                    tag += f"_kmax{arguments.wavenumber_max}"
                if arguments.num_segments != NUM_SEGMENTS:
                    tag += f"_M{arguments.num_segments}"
                if arguments.frozen_weno_weights:
                    tag += "_frozen"
            runs += [
                (tag, learning_rate, truth_index, init_index)
                for truth_index in arguments.truths
                for init_index in arguments.inits
            ]
        run_cells(
            arguments.method,
            arguments.horizon,
            runs,
            settings,
            arguments.output_directory,
            backend=NATIVE_JAX if arguments.native else PALLAS,
            kh_params=KHParams(
                num_cells=arguments.num_cells,
                wavenumber_max=arguments.wavenumber_max,
                rematerialize_stages=arguments.rematerialize_stages,
                frozen_weno_weights=arguments.frozen_weno_weights,
            ),
            num_segments=arguments.num_segments,
            batch_size=arguments.batch_size,
            checkpoint_minutes=arguments.checkpoint_minutes,
        )
    elif arguments.command == "time":
        select_gpu()
        backends = [PALLAS, NATIVE_JAX] if arguments.native else [PALLAS]
        time_steps(
            arguments.horizons,
            backends,
            arguments.batch_sizes,
            arguments.repeats,
            arguments.output,
            kh_params=KHParams(
                num_cells=arguments.num_cells,
                rematerialize_stages=arguments.rematerialize_stages,
            ),
            skip_jacobian=arguments.skip_jacobian,
        )
    elif arguments.command == "diagnose":
        select_gpu()
        default_output = os.path.join(REDUCED_DATA_DIRECTORY, f"kh_shooting_diagnose_T{arguments.horizon:g}.npz")
        output = arguments.output or default_output
        diagnose(
            arguments.horizon,
            arguments.truth,
            arguments.slice_points,
            arguments.slice_extent,
            arguments.power_iterations,
            output,
            num_cells=arguments.num_cells,
            backend=NATIVE_JAX if arguments.native else PALLAS,
            frozen_weno_weights=arguments.frozen_weno_weights,
        )
    elif arguments.command == "summarize":
        baseline = parse_selections([arguments.baseline])[0] if arguments.baseline else None
        summarize(arguments.output_directory, arguments.truths, baseline, arguments.summary)
    elif arguments.command == "decide":
        paired_decision(
            arguments.output_directory,
            arguments.horizon,
            parse_selections([arguments.multiple])[0],
            parse_selections([arguments.single])[0],
            arguments.starts,
            arguments.truths,
            arguments.inits,
        )
    elif arguments.command == "plot":
        make_convergence_figure(
            arguments.output_directory,
            parse_selections(arguments.tags),
            arguments.horizons,
            arguments.truths,
            arguments.output,
        )
    elif arguments.command == "recon":
        select_gpu()
        cache = os.path.splitext(arguments.output)[0] + "_fields.npz"
        make_reconstruction_figure(
            arguments.output_directory,
            parse_selections(arguments.tags),
            arguments.horizons,
            arguments.truths,
            arguments.output,
            cache,
        )


if __name__ == "__main__":
    main()
