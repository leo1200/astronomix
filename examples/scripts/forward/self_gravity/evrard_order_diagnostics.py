"""Why the energy error of the positivity-preserving scheme converges below RK4 order.

Companion to ``evrard_timestep_convergence.py`` (Fig. 16 of the methods paper).
The conservative gravity couplings conserve total energy exactly in space, so
their energy error is the time integrator's. With positivity-preserving WENO
(PP-WENO) the face value is pulled toward its first-order upwind state by
``theta = min(1, largest admissible fraction)``: a function of the state only,
not of the time step, but only Lipschitz where it switches between 1 and the
admissible fraction. This script tests whether that switching is what limits
the order, without touching the library: the WENO module's reference to
``positivity_preserving_interface_flux`` is replaced, for this process only, by
one of these variants:

* ``library``: the library flux, unchanged; with ``--record-theta`` the scalings
  are recomputed alongside and summarised per call through an ordered host
  callback (number of limited faces, their radii, the faces that switched since
  the previous step), so the run itself is bit-for-bit the library's;
* ``unit``: ``theta = 1`` everywhere (the PP splitting speeds without the
  limiter; not positivity preserving);
* ``smooth``: ``theta = g(f_face) g(f_mirror)`` with the admissible fractions
  ``f`` computed up to ``SMOOTH_REACH`` instead of 1 and
  ``g(f) = 1 - (1 - f / SMOOTH_REACH)^5`` (``g = 1`` beyond): ``g(f) <= f``
  keeps every face admissible, and ``g`` is C^4 where the limiter switches off.

``--precise-rk`` replaces the SSPRK(5,4) coefficients by the 15-digit
Shu-Osher table of Gottlieb, Ketcheson & Shu (2009), rewritten in the
three-register form, with the float64 weights of every stage summing to
exactly one. The stored runs of the figure predate the same change in the
library: its earlier coefficients satisfied the order conditions only to
~1e-10 and their stage weights summed to ``1 - 1.7e-17`` in float64, so every
step scaled the state and the total energy ``E = H + G`` (``G`` quadratic in
the density) drifted by ``delta (H + 2 G)`` per step, a round-off floor that
grew with the number of steps (``ROUND_OFF_DRIFT_PER_STEP``, removed in the
figure). With the current library ``--precise-rk`` changes nothing.

Every run integrates the Evrard collapse at 32^3 in float64 with a fixed number
of time steps and stores the energy history; the results go to
``data/diagnostics/<tag>.npz``.

``--plot`` draws ``figures/evrard_order_diagnostics.{svg,pdf,png}`` from the
stored runs (the tags listed in ``COLD_SWEEPS`` / ``MILD_SWEEPS`` and the
theta / jitter runs below) and the caches of ``evrard_timestep_convergence.py``.

Run from the directory of this script (the runs of the figure):

    python evrard_order_diagnostics.py --e0 0.05 --steps 1000,4000 --record-theta --tag cold_pp_z_theta
    python evrard_order_diagnostics.py --e0 0.2 --steps 1000,4000 --record-theta --tag mild_pp_z_theta
    python evrard_order_diagnostics.py --e0 0.05 --steps 960,980,1000,1020,1040,3840,3920,4000,4080,4160 --num-snapshots 13 --tag cold_pp_z_jitter
    python evrard_order_diagnostics.py --e0 0.2 --steps 960,980,1000,1020,1040,3840,3920,4000,4080,4160 --num-snapshots 13 --tag mild_pp_z_jitter
    python evrard_order_diagnostics.py --e0 0.05 --weno-js --tag cold_pp_js
    python evrard_order_diagnostics.py --e0 0.05 --variant smooth --tag cold_smooth_z
    python evrard_order_diagnostics.py --e0 0.05 --floors 1e-12 1e-14 --tag cold_pp_z_lowfloor
    python evrard_order_diagnostics.py --e0 0.05 --precise-rk --steps 250,500,1000,2000,4000,8000,16000,32000 --tag cold_pp_z_rkexact
    python evrard_order_diagnostics.py --e0 0.2 --precise-rk --steps 250,500,1000,2000,4000,8000,16000,32000 --tag mild_pp_z_rkexact
    python evrard_order_diagnostics.py --e0 0.2 --no-pp --tag mild_face_z
    python evrard_order_diagnostics.py --e0 0.2 --no-pp --weno-js --tag mild_face_js
    python evrard_order_diagnostics.py --e0 0.2 --weno-js --tag mild_pp_js
    python evrard_order_diagnostics.py --e0 0.2 --variant unit --tag mild_unit_z
    python evrard_order_diagnostics.py --e0 0.2 --variant smooth --tag mild_smooth_z
    python evrard_order_diagnostics.py --plot

(with ``PYTHONPATH`` set to the repository root).
"""

# ==== GPU selection ====
import os

if os.environ.get("CUDA_VISIBLE_DEVICES") is None and os.environ.get("JAX_PLATFORMS") != "cpu":
    from autocvd import autocvd

    autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# general
import argparse
import time
from fractions import Fraction

# jax
import jax

# The order study needs float64 (the energy errors reach 1e-13); x64 must be
# set before any array is created.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix import NATIVE_JAX
from astronomix.option_classes.simulation_config import (
    FOURTH_ORDER_CONSERVATIVE,
    SECOND_ORDER_CONSERVATIVE,
    SIMPLE_SOURCE,
)

# astronomix functions
from astronomix._integrators import _explicit_rk as explicit_rk_module
from astronomix._finite_difference._interface_fluxes import _weno as weno_module
from astronomix._finite_difference._interface_fluxes._weno_positivity import (
    DIVISION_FLOOR,
    _admissible_fraction,
    _gas_pressure,
    _upwind_split_states,
)

# figure helpers
from _collapse import (
    BOX_SIZE,
    MINIMUM_DENSITY,
    MINIMUM_PRESSURE,
    CollapseScheme,
    run_collapse,
)
from _common import (
    DATA_DIR,
    observed_orders,
)

NUM_CELLS = 32
T_END = 1.2
DIAGNOSTICS_DIR = DATA_DIR / "diagnostics"

#: The admissible fraction at which the smooth scaling reaches 1.
SMOOTH_REACH = 5.0

#: theta below this counts as limited (the library's theta is exactly 1 in
#: unlimited faces; the margin only ignores round-off in the recomputation).
LIMITED_THRESHOLD = 1.0 - 1e-12

#: Radial bins of the limited-face histogram (the sphere has radius 1, the
#: box half-width is 2).
RADIAL_BIN_EDGES = np.linspace(0.0, 2.4, 25)

GRAVITY_VERSIONS = {
    "simple": SIMPLE_SOURCE,
    "second": SECOND_ORDER_CONSERVATIVE,
    "fourth": FOURTH_ORDER_CONSERVATIVE,
}

#: Calls of the interface flux per time step: five SSPRK4 stages, three axes.
CALLS_PER_STEP = 15

#: The library's flux, kept before any variant replaces the module reference.
LIBRARY_FLUX = weno_module.positivity_preserving_interface_flux

# The SSPRK(5,4) Shu-Osher table of Gottlieb, Ketcheson & Shu (2009), 15 digits:
# u1 = u0 + b10 L0, u2 = a20 u0 + a21 u1 + b21 L1, u3 = a30 u0 + a32 u2 + b32 L2,
# u4 = a40 u0 + a43 u3 + b43 L3,
# u5 = a52 u2 + a53 u3 + b53 L3 + a54 u4 + b54 L4.
SSPRK54_TABLE = dict(
    b10=0.391752226571890,
    a20=0.444370493651235,
    a21=0.555629506348765,
    b21=0.368410593050371,
    a30=0.620101851488403,
    a32=0.379898148511597,
    b32=0.251891774271694,
    a40=0.178079954393132,
    a43=0.821920045606868,
    b43=0.544974750228521,
    a52=0.517231671970585,
    a53=0.096059710526147,
    b53=0.063692468666290,
    a54=0.386708617503269,
    b54=0.226007483236906,
)


# -------------------------------------------------------------
# =========== ↓ Scalings of the positivity limiter ↓ ===========
# -------------------------------------------------------------


def _split_steps(conserved_state, cell_flux, common_speed, plus_face_flux, minus_face_flux, plus_shift, minus_shift):
    """The upwind split states and the face steps ``d = w_hat - w`` of both
    split fluxes, as in ``positivity_preserving_interface_flux``.

    Args:
        conserved_state: The conserved state (variables first).
        cell_flux: The physical flux at the cell centres.
        common_speed: The splitting speed of each interface.
        plus_face_flux: The reconstructed plus split flux.
        minus_face_flux: The reconstructed minus split flux.
        plus_shift: The plus split-state shift of the mass-free fields.
        minus_shift: The minus split-state shift.

    Returns:
        ``(alpha, plus_unshifted, minus_unshifted, plus_owner, minus_owner,
        plus_step, minus_step)``.
    """
    alpha = jnp.maximum(common_speed, DIVISION_FLOOR)[None]
    plus_unshifted, minus_unshifted = _upwind_split_states(
        conserved_state,
        cell_flux,
        common_speed,
        0,
    )
    plus_owner = plus_unshifted + plus_shift
    minus_owner = minus_unshifted + minus_shift
    plus_step = 2.0 * plus_face_flux / alpha - plus_owner
    minus_step = -2.0 * minus_face_flux / alpha - minus_owner
    return alpha, plus_unshifted, minus_unshifted, plus_owner, minus_owner, plus_step, minus_step


def _fraction_up_to(base, step, reach, params, config, registered_variables):
    """Largest ``t`` in ``[0, reach]`` with ``base + t step`` admissible.

    The library's ``_admissible_fraction`` with the upper end 1 replaced by
    ``reach`` (same floors, same one-step chord root for the pressure).

    Args:
        base: The starting state.
        step: The direction of the segment.
        reach: The largest fraction considered.
        params: The simulation parameters (floors, gamma).
        config: The simulation configuration.
        registered_variables: The registered variables.

    Returns:
        The fraction per interface.
    """
    density_index = registered_variables.density_index
    base_density = base[density_index]
    density_floor = jnp.minimum(params.minimum_density, 0.5 * base_density)
    density_step = step[density_index]
    fraction = jnp.where(
        density_step < 0.0,
        (base_density - density_floor) / jnp.maximum(-density_step, DIVISION_FLOOR),
        reach,
    )
    fraction = jnp.clip(jnp.where(base_density > 0.0, fraction, 0.0), 0.0, reach)

    base_pressure = _gas_pressure(base, params.gamma, config, registered_variables)
    pressure_floor = jnp.minimum(params.minimum_pressure, 0.5 * jnp.maximum(base_pressure, 0.0))
    base_margin = base_pressure - pressure_floor
    end_pressure = _gas_pressure(base + fraction[None] * step, params.gamma, config, registered_variables)
    end_margin = end_pressure - pressure_floor
    chord_root = fraction * base_margin / jnp.maximum(base_margin - end_margin, DIVISION_FLOOR)
    fraction = jnp.where(end_margin >= 0.0, fraction, chord_root)
    return jnp.where(base_margin > 0.0, fraction, 0.0)


def _smooth_scaling(fraction):
    """``g(f) = 1 - (1 - f / R)^5`` below the reach ``R``, 1 above: C^4 at ``R``
    and ``g(f) <= f``.

    Args:
        fraction: The admissible fraction (at most ``SMOOTH_REACH``).

    Returns:
        The smooth scaling in [0, 1].
    """
    return 1.0 - (1.0 - jnp.minimum(fraction, SMOOTH_REACH) / SMOOTH_REACH) ** 5


def _library_thetas(owner, unshifted, step, neighbour_shift, params, config, registered_variables):
    """The library's scaling of one split flux: the smaller admissible fraction
    of the face value (with the shift jump of the neighbour's update, see
    ``_weno_positivity.py``) and its mirror."""
    return jnp.minimum(
        _admissible_fraction(owner - neighbour_shift, step, params, config, registered_variables),
        _admissible_fraction(unshifted, -step, params, config, registered_variables),
    )


def _smooth_thetas(owner, unshifted, step, neighbour_shift, params, config, registered_variables):
    """The smooth scaling of one split flux: the product of the smoothed face
    and mirror fractions (a product of scalings in [0, 1] is below either)."""
    face_fraction = _fraction_up_to(
        owner - neighbour_shift, step, SMOOTH_REACH, params, config, registered_variables
    )
    mirror_fraction = _fraction_up_to(unshifted, -step, SMOOTH_REACH, params, config, registered_variables)
    return _smooth_scaling(face_fraction) * _smooth_scaling(mirror_fraction)


# -------------------------------------------------------------
# =========== ↑ Scalings of the positivity limiter ↑ ===========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================ ↓ Host-side theta recorder ↓ ================
# -------------------------------------------------------------


class ThetaRecorder:
    """Collects the per-call limiter statistics sent by the ordered callback.

    The calls arrive in program order: per time step five stages, each with the
    x, y and z sweep. The packed limited-face mask of each (stage, axis) is
    compared with the same call of the previous step to count the faces that
    switched between limited and unlimited.
    """

    def __init__(self):
        self.limited_counts = []
        self.deficits = []
        self.minimum_thetas = []
        self.radial_histograms = []
        self.switched_counts = []
        self.previous_masks = {}
        self.num_calls = 0

    def __call__(self, limited_count, deficit, minimum_theta, radial_histogram, packed_mask):
        packed_mask = np.asarray(packed_mask)
        call_key = self.num_calls % CALLS_PER_STEP
        previous_mask = self.previous_masks.get(call_key)
        if previous_mask is None:
            switched = 0
        else:
            switched = int(np.unpackbits(np.bitwise_xor(previous_mask, packed_mask)).sum())
        self.previous_masks[call_key] = packed_mask
        self.limited_counts.append(int(limited_count))
        self.deficits.append(float(deficit))
        self.minimum_thetas.append(float(minimum_theta))
        self.radial_histograms.append(np.asarray(radial_histogram))
        self.switched_counts.append(switched)
        self.num_calls += 1

    def arrays(self):
        """The statistics reshaped to (step, stage, axis)."""
        num_steps = self.num_calls // CALLS_PER_STEP
        shape = (num_steps, 5, 3)
        used = num_steps * CALLS_PER_STEP
        return {
            "limited_count": np.asarray(self.limited_counts[:used]).reshape(shape),
            "deficit": np.asarray(self.deficits[:used]).reshape(shape),
            "minimum_theta": np.asarray(self.minimum_thetas[:used]).reshape(shape),
            "radial_histogram": np.asarray(self.radial_histograms[:used]).reshape(shape + (-1,)),
            "switched_count": np.asarray(self.switched_counts[:used]).reshape(shape),
        }


def _face_radius(num_cells):
    """Distance of the face i + 1/2 from the centre, in any sweep layout (the
    grid is symmetric under the axis permutations of the sweeps)."""
    cell_size = BOX_SIZE / num_cells
    face_positions = (np.arange(num_cells) + 1.0) * cell_size - 0.5 * BOX_SIZE
    centre_positions = (np.arange(num_cells) + 0.5) * cell_size - 0.5 * BOX_SIZE
    sweep, first, second = np.meshgrid(face_positions, centre_positions, centre_positions, indexing="ij")
    return np.sqrt(sweep**2 + first**2 + second**2)


# -------------------------------------------------------------
# ================ ↑ Host-side theta recorder ↑ ================
# -------------------------------------------------------------


def _exactly_complementary(weights):
    """Recompute the smallest weight so that the float64 weights sum to
    exactly one (checked in exact rational arithmetic).

    Args:
        weights: The convex weights of the states of one stage.

    Returns:
        The weights with the smallest one replaced.
    """
    weights = list(weights)
    slack = int(np.argmin(np.abs(weights)))
    others = sum(Fraction(weight) for index, weight in enumerate(weights) if index != slack)
    weights[slack] = float(1 - others)
    if sum(Fraction(weight) for weight in weights) != 1:
        raise ValueError("the weights cannot be made to sum to one exactly")
    return tuple(weights)


def install_precise_runge_kutta(exact_weights=True):
    """Replace the library's SSPRK(5,4) coefficients by ``SSPRK54_TABLE``.

    The library's three-register form eliminates ``b53 L3`` through
    ``u4 = a40 u0 + a43 u3 + b43 L3``, so with ``r = b53 / b43`` the final stage
    is ``-r a40 u0 + a52 u2 + (a53 - r a43) u3 + (a54 + r) u4 + b54 L4``. With
    ``exact_weights`` the weight of ``u0`` of every stage is adjusted by units
    in the last place so that the float64 weights of each stage sum to exactly
    one (the 15-digit last row sums to 1 + 1e-15); otherwise the scaled state
    drifts by the weights' excess every step.

    Args:
        exact_weights: Whether to make the stage weights sum to one exactly.
    """
    table = SSPRK54_TABLE
    ratio = table["b53"] / table["b43"]
    stage_weights = [
        (table["a20"], table["a21"]),
        (table["a30"], table["a32"]),
        (table["a40"], table["a43"]),
    ]
    final_weights = (
        -ratio * table["a40"],
        table["a52"],
        table["a53"] - ratio * table["a43"],
        table["a54"] + ratio,
    )
    if exact_weights:
        stage_weights = [_exactly_complementary(weights) for weights in stage_weights]
        final_weights = _exactly_complementary(final_weights)
    explicit_rk_module._SSPRK4_K0 = (
        1.0,
        stage_weights[0][0],
        stage_weights[1][0],
        stage_weights[2][0],
        final_weights[0],
    )
    explicit_rk_module._SSPRK4_KRHS = (
        table["b10"],
        table["b21"],
        table["b32"],
        table["b43"],
        table["b54"],
    )
    explicit_rk_module._SSPRK4_KCURR = (
        0.0,
        stage_weights[0][1],
        stage_weights[1][1],
        stage_weights[2][1],
        final_weights[3],
    )
    explicit_rk_module._SSPRK4_FINAL = (
        final_weights[0],
        0.0,
        final_weights[1],
        final_weights[2],
        final_weights[3],
    )
    explicit_rk_module._SSPRK4_FINAL_RAW = (0.0, 0.0, 0.0, ratio)


def install_variant(variant, recorder=None, num_cells=NUM_CELLS):
    """Replace the WENO module's positivity-preserving interface flux.

    Args:
        variant: ``library``, ``unit`` or ``smooth`` (see the module docstring).
        recorder: A ``ThetaRecorder`` that receives the library scalings of
            every call, or None.
        num_cells: The grid size (for the face radii of the histogram).
    """
    radial_bin = np.clip(
        np.digitize(_face_radius(num_cells), RADIAL_BIN_EDGES) - 1,
        0,
        len(RADIAL_BIN_EDGES) - 2,
    )
    radial_bin = jnp.asarray(radial_bin)

    def record(plus_theta, minus_theta):
        limited = jnp.stack([plus_theta < LIMITED_THRESHOLD, minus_theta < LIMITED_THRESHOLD])
        limited_any = limited[0] | limited[1]
        radial_histogram = jnp.zeros(len(RADIAL_BIN_EDGES) - 1).at[radial_bin].add(
            limited_any.astype(jnp.float64)
        )
        jax.debug.callback(
            recorder,
            jnp.sum(limited),
            jnp.sum(1.0 - plus_theta) + jnp.sum(1.0 - minus_theta),
            jnp.minimum(jnp.min(plus_theta), jnp.min(minus_theta)),
            radial_histogram,
            jnp.packbits(limited.ravel()),
            ordered=True,
        )

    def patched_flux(
        conserved_state,
        cell_flux,
        common_speed,
        plus_face_flux,
        minus_face_flux,
        plus_shift,
        minus_shift,
        params,
        config,
        registered_variables,
        axis=0,
        inflow_reference=None,
    ):
        (
            alpha,
            plus_unshifted,
            minus_unshifted,
            plus_owner,
            minus_owner,
            plus_step,
            minus_step,
        ) = _split_steps(
            conserved_state,
            cell_flux,
            common_speed,
            plus_face_flux,
            minus_face_flux,
            plus_shift,
            minus_shift,
        )
        if recorder is not None:
            record(
                _library_thetas(
                    plus_owner, plus_unshifted, plus_step, minus_shift, params, config, registered_variables
                ),
                _library_thetas(
                    minus_owner, minus_unshifted, minus_step, plus_shift, params, config, registered_variables
                ),
            )

        if variant == "library":
            return LIBRARY_FLUX(
                conserved_state,
                cell_flux,
                common_speed,
                plus_face_flux,
                minus_face_flux,
                plus_shift,
                minus_shift,
                params,
                config,
                registered_variables,
                axis=axis,
                inflow_reference=inflow_reference,
            )
        if variant == "unit":
            plus_theta = jnp.ones_like(common_speed)
            minus_theta = jnp.ones_like(common_speed)
        elif variant == "smooth":
            plus_theta = _smooth_thetas(
                plus_owner, plus_unshifted, plus_step, minus_shift, params, config, registered_variables
            )
            minus_theta = _smooth_thetas(
                minus_owner, minus_unshifted, minus_step, plus_shift, params, config, registered_variables
            )
        else:
            raise ValueError(f"unknown variant {variant!r}")
        plus_flux = 0.5 * alpha * (plus_owner + plus_theta[None] * plus_step)
        minus_flux = -0.5 * alpha * (minus_owner + minus_theta[None] * minus_step)
        return plus_flux + minus_flux

    weno_module.positivity_preserving_interface_flux = patched_flux


# -------------------------------------------------------------
# ================== ↓ Diagnostics figure ↓ ===================
# -------------------------------------------------------------

#: Relative energy drift per step of the float64 round-off (measured from the
#: finest steps of ``evrard_timestep_convergence.py``; same for all couplings).
ROUND_OFF_DRIFT_PER_STEP = {"cold": 4.5e-17, "mild": 5.6e-17}

#: The step sweeps of the figure: (tag, label, colour, marker). The runs with
#: the exactly normalised 15-digit coefficients (``_rkexact``) have no drift
#: to remove.
COLD_SWEEPS = [
    ("cold_pp_js", "PP-WENO, Jiang-Shu weights", "C1", "s"),
    ("cold_smooth_z", r"PP-WENO, smooth $\theta$", "C2", "^"),
    ("cold_pp_z_lowfloor", "PP-WENO, floors $10^{-12}$", "C4", "D"),
    ("cold_pp_z_rkexact", "PP-WENO, exact SSPRK(5,4) weights", "C3", "v"),
]
MILD_SWEEPS = [
    ("mild_face_z", "no PP (face state)", "k", "o"),
    ("mild_face_js", "no PP, Jiang-Shu weights", "0.5", "o"),
    ("mild_pp_js", "PP-WENO, Jiang-Shu weights", "C1", "s"),
    ("mild_unit_z", r"PP splitting, $\theta \equiv 1$", "C0", "x"),
    ("mild_smooth_z", r"PP-WENO, smooth $\theta$", "C2", "^"),
    ("mild_pp_z_rkexact", "PP-WENO, exact SSPRK(5,4) weights", "C3", "v"),
]


def _drift_to_remove(tag):
    """The round-off drift per step to remove from a sweep (none for the
    exactly normalised coefficients)."""
    if tag.endswith("_rkexact"):
        return 0.0
    return ROUND_OFF_DRIFT_PER_STEP[tag.split("_")[0]]


def _signed_final_errors(data):
    """Steps and signed final relative energy errors of one diagnostics file."""
    steps = data["steps"]
    errors = np.array(
        [
            (data[f"total_energy_{n}"][-1] - data[f"total_energy_{n}"][0])
            / abs(data[f"total_energy_{n}"][0])
            for n in steps
        ]
    )
    return steps, errors


def _faces_per_radial_bin(num_cells):
    """Number of faces of one sweep in every radial bin."""
    counts, _ = np.histogram(_face_radius(num_cells), bins=RADIAL_BIN_EDGES)
    return counts


def plot_diagnostics():
    """Draw ``figures/evrard_order_diagnostics.{svg,pdf,png}`` from the cached runs.

    (a, b) Fraction of limited faces against radius and time (cold, mild).
    (c) Signed energy-error history, scaled by ``(n / 1000)^4``: the curves of
    a fourth-order error coincide. (d) Signed final error at step counts within
    +-4 % of 1000 and 4000. (e, f) Final error with the round-off drift
    removed against the time step, for the variants of the scheme.
    """
    import matplotlib.pyplot as plt

    from _common import (
        apply_paper_style,
        save_figure,
    )

    apply_paper_style()
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # --------------- ↓ Where the limiter acts ↓ ----------------
    faces_per_bin = 3 * _faces_per_radial_bin(NUM_CELLS)
    for ax, tag, title in (
        (axes[0, 0], "cold_pp_z_theta", "cold: faces with $\\theta < 1$"),
        (axes[1, 0], "mild_pp_z_theta", "mild: faces with $\\theta < 1$"),
    ):
        data = np.load(DIAGNOSTICS_DIR / f"{tag}.npz")
        num_steps = 4000
        histogram = data[f"theta_radial_histogram_{num_steps}"][:, 0].sum(axis=1)
        fraction = histogram / np.maximum(faces_per_bin, 1)[None]
        times = np.arange(num_steps) * T_END / num_steps
        image = ax.pcolormesh(
            np.linspace(0.0, T_END, num_steps + 1),
            RADIAL_BIN_EDGES,
            fraction.T,
            cmap="magma_r",
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )
        switched = data[f"theta_switched_count_{num_steps}"][:, 0].sum(axis=1)
        twin = ax.twinx()
        window = num_steps // 60
        twin.plot(
            times[: len(switched) // window * window : window],
            switched[: len(switched) // window * window].reshape(-1, window).mean(axis=1),
            color="C0",
            linewidth=1.2,
        )
        twin.set_ylabel("faces switching per step", color="C0")
        ax.set_ylim(0.0, 2.2)
        ax.set_xlabel("time")
        ax.set_ylabel("face radius")
        ax.set_title(title + " (4000 steps)")
        fig.colorbar(image, ax=twin, pad=0.2, label="limited fraction of the faces")
    # --------------- ↑ Where the limiter acts ↑ ----------------

    # --------------- ↓ Energy-error histories ↓ ----------------
    ax = axes[0, 1]
    for tag, linestyle in (("cold_pp_z_theta", "-"), ("mild_pp_z_theta", "--")):
        data = np.load(DIAGNOSTICS_DIR / f"{tag}.npz")
        for num_steps, color in ((1000, "C0"), (4000, "C3")):
            energy = data[f"total_energy_{num_steps}"]
            scaled_error = (energy - energy[0]) / abs(energy[0]) * (num_steps / 1000) ** 4
            ax.plot(
                data[f"time_{num_steps}"],
                scaled_error,
                color=color,
                linestyle=linestyle,
                label=f"{tag.split('_')[0]}, {num_steps} steps",
            )
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$(E - E_0)/|E_0| \times (n / 1000)^4$")
    ax.set_title("signed energy error, scaled for fourth order")
    ax.legend(fontsize=9)
    # --------------- ↑ Energy-error histories ↑ ----------------

    # --------------- ↓ Step jitter ↓ ----------------
    ax = axes[1, 1]
    for tag, color in (("cold_pp_z_jitter", "C3"), ("mild_pp_z_jitter", "C0")):
        path = DIAGNOSTICS_DIR / f"{tag}.npz"
        if not path.exists():
            continue
        steps, errors = _signed_final_errors(np.load(path))
        case = tag.split("_")[0]
        errors = errors - _drift_to_remove(tag) * steps
        for centre, marker in ((1000, "o"), (4000, "s")):
            near = np.abs(steps / centre - 1) < 0.05
            ax.plot(
                steps[near] / centre,
                errors[near] * (centre / 1000) ** 4,
                color=color,
                marker=marker,
                linestyle="-",
                label=f"{case}, $n \\approx {centre}$",
            )
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.set_xlabel("number of steps / nominal")
    ax.set_ylabel(r"$(E - E_0)/|E_0| \times (n / 1000)^4$ (drift removed)")
    ax.set_title("final error at step counts within 4 %")
    ax.legend(fontsize=9)
    # --------------- ↑ Step jitter ↑ ----------------

    # --------------- ↓ Variants of the scheme ↓ ----------------
    for ax, case, sweeps, reference_tag in (
        (axes[0, 2], "cold", COLD_SWEEPS, "cold"),
        (axes[1, 2], "mild", MILD_SWEEPS, "mild"),
    ):
        for tag, label, color, marker in sweeps:
            path = DIAGNOSTICS_DIR / f"{tag}.npz"
            if not path.exists():
                continue
            steps, errors = _signed_final_errors(np.load(path))
            corrected = np.abs(errors - _drift_to_remove(tag) * steps)
            ax.plot(T_END / steps, corrected, color=color, marker=marker, label=label)
        reference = np.load(DATA_DIR / f"evrard_timestep_convergence_fp64_{reference_tag}.npz")
        total_energy = reference["total_energy_2"]
        errors = (total_energy[:, -1] - total_energy[:, 0]) / np.abs(total_energy[:, 0])
        steps = reference["num_timesteps"]
        corrected = np.abs(errors - ROUND_OFF_DRIFT_PER_STEP[case] * steps)
        ax.plot(T_END / steps[:-1], corrected[:-1], color="C2", linewidth=2.5, label="baseline (PP-WENO, WENO-Z)")
        guide_dt = T_END / np.array([250.0, 8000.0])
        ax.plot(guide_dt, 2e-9 * (guide_dt / guide_dt[0]) ** 4, color="0.4", linestyle="--", label=r"$\propto \Delta t^4$")
        ax.plot(guide_dt, 2e-9 * (guide_dt / guide_dt[0]) ** 3, color="0.4", linestyle=":", label=r"$\propto \Delta t^3$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 1e-7)
        ax.set_xlabel(r"time step $\Delta t$")
        ax.set_ylabel(r"$|\Delta E|/|E_0|$ (library weights: drift removed)")
        ax.set_title(f"{case} collapse: variants of the scheme")
        ax.legend(fontsize=8)
    # --------------- ↑ Variants of the scheme ↑ ----------------

    fig.tight_layout()
    print(f"Saved -> {save_figure(fig, 'evrard_order_diagnostics', extensions=('svg', 'pdf', 'png'))}")


# -------------------------------------------------------------
# ================== ↑ Diagnostics figure ↑ ===================
# -------------------------------------------------------------


def main():
    """Parse the arguments, run the step sweep and store the diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e0", type=float, default=0.05, help="initial thermal energy per mass")
    parser.add_argument("--steps", default="250,500,1000,2000,4000,8000,16000")
    parser.add_argument("--gravity", default="fourth", choices=list(GRAVITY_VERSIONS))
    parser.add_argument("--variant", default="library", choices=["library", "unit", "smooth"])
    parser.add_argument("--weno-js", action="store_true", help="Jiang-Shu instead of WENO-Z weights")
    parser.add_argument("--no-pp", action="store_true", help="switch positivity-preserving WENO off")
    parser.add_argument("--fct", action="store_true", help="flux-corrected gravitational work")
    parser.add_argument("--record-theta", action="store_true", help="record the limiter statistics")
    parser.add_argument(
        "--precise-rk",
        action="store_true",
        help="15-digit SSPRK(5,4) coefficients instead of the library's",
    )
    parser.add_argument(
        "--inexact-weights",
        action="store_true",
        help="with --precise-rk: keep the table's stage weights (sum 1 + 1e-15)",
    )
    parser.add_argument("--num-snapshots", type=int, default=121)
    parser.add_argument("--num-cells", type=int, default=NUM_CELLS)
    parser.add_argument("--t-end", type=float, default=T_END)
    parser.add_argument(
        "--floors",
        type=float,
        nargs=2,
        default=(MINIMUM_DENSITY, MINIMUM_PRESSURE),
        metavar=("RHO_MIN", "P_MIN"),
        help="density and pressure floors of the admissibility test and wave speeds",
    )
    parser.add_argument("--tag", help="name of the diagnostics file")
    parser.add_argument("--plot", action="store_true", help="only draw the diagnostics figure")
    args = parser.parse_args()
    if args.plot:
        plot_diagnostics()
        return

    scheme = CollapseScheme(
        self_gravity_version=GRAVITY_VERSIONS[args.gravity],
        weno_z=not args.weno_js,
        positivity_preserving=not args.no_pp,
        work_flux_correction=args.fct,
    )
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    step_counts = [int(steps) for steps in args.steps.split(",")]
    if args.precise_rk:
        install_precise_runge_kutta(exact_weights=not args.inexact_weights)

    results = {
        "steps": np.array(step_counts),
        "dt": args.t_end / np.array(step_counts, dtype=float),
        "initial_energy": np.array(args.e0),
    }
    for num_steps in step_counts:
        recorder = ThetaRecorder() if args.record_theta else None
        # One variant per process: the patched flux is looked up when the
        # time integration is first traced.
        if not args.no_pp:
            install_variant(args.variant, recorder, args.num_cells)
        start = time.time()
        snapshots, _, registered_variables = run_collapse(
            args.num_cells,
            scheme,
            args.t_end,
            initial_energy=args.e0,
            backend=NATIVE_JAX,
            num_snapshots=args.num_snapshots,
            fixed_num_timesteps=num_steps,
            minimum_density=args.floors[0],
            minimum_pressure=args.floors[1],
        )
        wall_time = time.time() - start
        total_energy = np.asarray(snapshots.total_energy)
        final_state = np.asarray(snapshots.final_state)
        signed_energy_error = (total_energy[-1] - total_energy[0]) / abs(total_energy[0])
        energy_error = abs(signed_energy_error)
        minimum_pressure = float(final_state[registered_variables.pressure_index].min())
        finite = bool(np.all(np.isfinite(final_state)))
        print(
            f"[{args.tag}] steps={num_steps:6d} dE/E={signed_energy_error:+.4e} "
            f"min p={minimum_pressure:.3e} finite={finite} wall={wall_time:.0f}s",
            flush=True,
        )
        results[f"time_{num_steps}"] = np.asarray(snapshots.time_points)
        results[f"total_energy_{num_steps}"] = total_energy
        results[f"internal_energy_{num_steps}"] = np.asarray(snapshots.internal_energy)
        results[f"kinetic_energy_{num_steps}"] = np.asarray(snapshots.kinetic_energy)
        results[f"gravitational_energy_{num_steps}"] = np.asarray(snapshots.gravitational_energy)
        results[f"minimum_pressure_{num_steps}"] = np.array(minimum_pressure)
        results[f"wall_time_{num_steps}"] = np.array(wall_time)
        if recorder is not None:
            jax.effects_barrier()
            for name, values in recorder.arrays().items():
                results[f"theta_{name}_{num_steps}"] = values

    errors = [
        abs(results[f"total_energy_{n}"][-1] - results[f"total_energy_{n}"][0])
        / abs(results[f"total_energy_{n}"][0])
        for n in step_counts
    ]
    if len(step_counts) > 1:
        print(
            f"[{args.tag}] observed orders: "
            + " ".join(f"{order:.2f}" for order in observed_orders(results["dt"], errors)),
            flush=True,
        )
    results["radial_bin_edges"] = RADIAL_BIN_EDGES
    np.savez(DIAGNOSTICS_DIR / f"{args.tag}.npz", **results)
    print(f"Saved -> {DIAGNOSTICS_DIR / f'{args.tag}.npz'}")


if __name__ == "__main__":
    main()
