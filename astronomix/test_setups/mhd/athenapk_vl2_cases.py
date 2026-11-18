"""
# AthenaPK VL2 validation cases

The validation cases of the VL2 finite-volume scheme against AthenaPK. Each
case fixes one AthenaPK configuration (problem, mesh, boundaries, Riemann
solver, reconstruction, GLM and positivity options) and the matching astronomix
configuration. Cases with an AthenaPK problem generator start from AthenaPK's
own initial condition; cases with an ``initial_condition`` start both codes from
the same primitive state built here (AthenaPK via a restart file).

Together they exercise every code path of the scheme: HLLD (all its wave
branches and the degenerate case), HLLE and LLF for GLM-MHD, HLLC and HLLE for
hydrodynamics, PLM and donor-cell reconstruction, the plain and the extended
Dedner source, the first-order flux correction, the floors, and periodic and
outflow boundaries in 1D, 2D and 3D.

``REGRESSION_CASES`` are small versions of representative cases, whose AthenaPK
results are stored in ``pytests/mhd/data/athenapk_vl2`` and checked by
``pytests/mhd/test_vl2_athenapk_regression.py``. The full comparison (which runs
AthenaPK itself) lives in ``examples/scripts/validation/athenapk_vl2``.

## References

- Grete, P. et al. 2023, IEEE TPDS 34, 85 (Parthenon / AthenaPK).
- Stone, J. M. et al. 2008, ApJS 178, 137 (Athena: linear waves, field loop).
- Orszag, S. A. & Tang, C.-M. 1979, J. Fluid Mech. 90, 129.
- Balsara, D. S. & Spicer, D. S. 1999, J. Comput. Phys. 149, 270 (MHD rotor).
- Brio, M. & Wu, C. C. 1988, J. Comput. Phys. 75, 400.
- Einfeldt, B. et al. 1991, J. Comput. Phys. 92, 273.
"""

# typing
from typing import (
    Callable,
    NamedTuple,
    Optional,
)

# jax
import jax.numpy as jnp

# numerics
import numpy as np

# astronomix constants
from astronomix.option_classes.simulation_config import (
    DONOR_CELL,
    FINITE_VOLUME,
    HLL,
    HLLC,
    HLLD,
    LAX_FRIEDRICHS,
    OPEN_BOUNDARY,
    PERIODIC_BOUNDARY,
    PIECEWISE_LINEAR,
    POSITIVITY_HARD_FLOOR,
    VAN_LEER,
    VL2,
)

# astronomix containers
from astronomix.option_classes.simulation_config import (
    BackendConfig,
    BoundarySettings,
    BoundarySettings1D,
    FVSolverConfig,
    PositivityConfig,
    SimulationConfig,
    StaticFloatVector,
)
from astronomix.option_classes.simulation_params import SimulationParams

# astronomix functions
from astronomix.option_classes.simulation_config import finalize_config
from astronomix.variable_registry.registered_variables import get_registered_variables

#: The adiabatic index as written in AthenaPK's input files.
FIVE_THIRDS = 1.666666666666667


class ValidationCase(NamedTuple):
    """One AthenaPK configuration and its astronomix counterpart."""

    #: Name of the case (and of its stored regression result).
    name: str
    #: AthenaPK problem generator (``problem/generator``).
    problem_generator: str
    dimensionality: int
    num_cells: tuple
    lower_corner: tuple
    upper_corner: tuple
    #: ``"periodic"`` or ``"outflow"`` on every active axis.
    boundary: str
    mhd: bool
    #: AthenaPK's Riemann solver name: ``"hlld"``, ``"hlle"``, ``"hllc"`` or ``"llf"``.
    riemann_solver: str
    cfl: float
    gamma: float
    end_time: float
    #: AthenaPK's reconstruction: ``"plm"`` or ``"dc"`` (donor cell).
    reconstruction: str = "plm"
    glm_alpha: float = 0.1
    glm_extended_source: bool = False
    first_order_flux_correction: bool = False
    #: ``(density_floor, pressure_floor)``, or ``None`` without floors.
    floors: Optional[tuple] = None
    #: Extra AthenaPK ``problem`` parameters.
    problem_parameters: dict = {}
    #: Builds the primitive state of both codes; ``None`` uses AthenaPK's own.
    initial_condition: Optional[Callable] = None


# -------------------------------------------------------------
# =================== ↓ Initial conditions ↓ ==================
# -------------------------------------------------------------


def _cell_centers(case: ValidationCase):
    """The cell-centre coordinates of the case's grid (one array per axis)."""
    axes = [
        case.lower_corner[axis]
        + (np.arange(case.num_cells[axis]) + 0.5)
        * (case.upper_corner[axis] - case.lower_corner[axis])
        / case.num_cells[axis]
        for axis in range(case.dimensionality)
    ]
    return np.meshgrid(*axes, indexing="ij")


def _glm_mhd_primitive_state(density, velocity, pressure, magnetic_field):
    """Stack a primitive GLM-MHD state in the VL2 layout (psi starts at zero)."""
    return np.stack([density, *velocity, pressure, *magnetic_field, np.zeros_like(density)])


def magnetized_blast(case: ValidationCase):
    """A strong blast (pressure ratio 100) in a plasma of beta 0.2 along (1,1,1)."""
    x, y, z = _cell_centers(case)
    radius = np.sqrt(x**2 + y**2 + z**2)
    density = np.ones_like(x)
    pressure = np.where(radius < 0.1, 10.0, 0.1)
    field_component = np.full_like(x, 1.0 / np.sqrt(3.0))
    return _glm_mhd_primitive_state(density, [0 * x] * 3, pressure, [field_component] * 3)


def mhd_rotor(case: ValidationCase):
    """The MHD rotor of Balsara & Spicer (1999) (first version, with taper)."""
    x, y = _cell_centers(case)
    radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    inner_radius, outer_radius, rotation_speed = 0.1, 0.115, 2.0
    taper = (outer_radius - radius) / (outer_radius - inner_radius)
    inside = radius <= inner_radius
    in_taper = (radius > inner_radius) & (radius < outer_radius)
    density = np.where(inside, 10.0, np.where(in_taper, 1.0 + 9.0 * taper, 1.0))
    angular_factor = np.where(
        inside,
        rotation_speed / inner_radius,
        np.where(in_taper, taper * rotation_speed / np.maximum(radius, 1e-300), 0.0),
    )
    velocity = [-angular_factor * (y - 0.5), angular_factor * (x - 0.5), 0 * x]
    field = [np.full_like(x, 5.0 / np.sqrt(4.0 * np.pi)), 0 * x, 0 * x]
    return _glm_mhd_primitive_state(density, velocity, np.ones_like(x), field)


def brio_wu(case: ValidationCase):
    """The Brio & Wu (1988) shock tube (compound wave, HLLD degeneracies)."""
    (x,) = _cell_centers(case)
    left = x < 0.5
    density = np.where(left, 1.0, 0.125)
    pressure = np.where(left, 1.0, 0.1)
    field = [np.full_like(x, 0.75), np.where(left, 1.0, -1.0), 0 * x]
    return _glm_mhd_primitive_state(density, [0 * x] * 3, pressure, field)


def einfeldt_rarefaction(case: ValidationCase):
    """
    Two magnetized streams receding at Mach 24 (Einfeldt et al. 1991): a
    near-vacuum rarefaction.
    """
    (x,) = _cell_centers(case)
    velocity = [np.where(x < 0.5, -4.0, 4.0), 0 * x, 0 * x]
    field = [np.full_like(x, 0.5), 0 * x, 0 * x]
    return _glm_mhd_primitive_state(np.ones_like(x), velocity, np.full_like(x, 0.01), field)


def colliding_flows(case: ValidationCase):
    """Two cold (Mach ~250), perturbed, magnetized streams colliding head on."""
    x, y = _cell_centers(case)
    stream_velocity = np.where(x < 0.0, 10.0, -10.0) * (1.0 + 0.1 * np.sin(2.0 * np.pi * y))
    velocity = [stream_velocity, 0 * x, 0 * x]
    field = [0 * x, np.full_like(x, 0.3), 0 * x]
    return _glm_mhd_primitive_state(np.ones_like(x), velocity, np.full_like(x, 1e-3), field)


def low_beta_blast(case: ValidationCase):
    """A blast (pressure ratio 10^4) in a plasma of beta 0.002."""
    x, y, z = _cell_centers(case)
    radius = np.sqrt(x**2 + y**2 + z**2)
    field_component = np.full_like(x, 1.0 / np.sqrt(2.0))
    return _glm_mhd_primitive_state(
        np.ones_like(x),
        [0 * x] * 3,
        np.where(radius < 0.1, 10.0, 1e-3),
        [field_component, field_component, 0 * x],
    )


# -------------------------------------------------------------
# =================== ↑ Initial conditions ↑ ==================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ====================== ↓ Case table ↓ =======================
# -------------------------------------------------------------

_CP_ALFVEN_PARAMETERS = dict(
    compute_error="false",
    b_par=1.0,
    b_perp=0.1,
    pres=0.1,
    v_par=0.0,
    dir=1,
)
_ALFVEN_BOX = dict(
    dimensionality=3,
    lower_corner=(0.0, 0.0, 0.0),
    upper_corner=(3.0, 1.5, 1.5),
    boundary="periodic",
)
_ORSZAG_TANG = dict(
    problem_generator="orszag_tang",
    dimensionality=2,
    num_cells=(128, 128),
    lower_corner=(-0.5, -0.5),
    upper_corner=(0.5, 0.5),
    boundary="periodic",
    mhd=True,
    riemann_solver="hlld",
    cfl=0.4,
    gamma=FIVE_THIRDS,
    end_time=0.5,
)

CASES = [
    ValidationCase(
        name="cp_alfven_3d",
        problem_generator="cpaw",
        num_cells=(32, 16, 16),
        mhd=True,
        riemann_solver="hlld",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=5.0,
        problem_parameters=_CP_ALFVEN_PARAMETERS,
        **_ALFVEN_BOX,
    ),
    ValidationCase(
        name="fast_wave_3d",
        problem_generator="linear_wave",
        num_cells=(32, 16, 16),
        mhd=True,
        riemann_solver="hlld",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=1.0,
        problem_parameters=dict(wave_flag=0, amp=1e-4, vflow=0.0),
        **_ALFVEN_BOX,
    ),
    ValidationCase(
        name="slow_wave_3d",
        problem_generator="linear_wave",
        num_cells=(32, 16, 16),
        mhd=True,
        riemann_solver="hlld",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=1.0,
        problem_parameters=dict(wave_flag=2, amp=1e-4, vflow=0.0),
        **_ALFVEN_BOX,
    ),
    ValidationCase(
        name="entropy_wave_3d_advected",
        problem_generator="linear_wave",
        num_cells=(32, 16, 16),
        mhd=True,
        riemann_solver="hlld",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=1.0,
        problem_parameters=dict(wave_flag=3, amp=1e-4, vflow=1.0),
        **_ALFVEN_BOX,
    ),
    ValidationCase(name="orszag_tang", **_ORSZAG_TANG),
    ValidationCase(name="orszag_tang_fofc", first_order_flux_correction=True, **_ORSZAG_TANG),
    ValidationCase(name="orszag_tang_extended_glm", glm_extended_source=True, **_ORSZAG_TANG),
    ValidationCase(name="orszag_tang_donor_cell", reconstruction="dc", **_ORSZAG_TANG),
    ValidationCase(
        name="field_loop_hlle",
        problem_generator="field_loop",
        dimensionality=2,
        num_cells=(128, 64),
        lower_corner=(-1.0, -0.5),
        upper_corner=(1.0, 0.5),
        boundary="periodic",
        mhd=True,
        riemann_solver="hlle",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=1.0,
        glm_alpha=0.4,
        problem_parameters=dict(rad=0.3, amp=1e-3, vflow=1.0, iprob=1),
    ),
    ValidationCase(
        name="magnetized_blast_3d_fofc",
        problem_generator="blast",
        dimensionality=3,
        num_cells=(64, 64, 64),
        lower_corner=(-0.5, -0.5, -0.5),
        upper_corner=(0.5, 0.5, 0.5),
        boundary="periodic",
        mhd=True,
        riemann_solver="hlld",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=0.1,
        first_order_flux_correction=True,
        problem_parameters=dict(radius_outer=0.1, pressure_ratio=100.0),
        initial_condition=magnetized_blast,
    ),
    ValidationCase(
        name="mhd_rotor_outflow",
        problem_generator="orszag_tang",
        dimensionality=2,
        num_cells=(128, 128),
        lower_corner=(0.0, 0.0),
        upper_corner=(1.0, 1.0),
        boundary="outflow",
        mhd=True,
        riemann_solver="hlld",
        cfl=0.4,
        gamma=1.4,
        end_time=0.15,
        initial_condition=mhd_rotor,
    ),
    ValidationCase(
        name="brio_wu_outflow",
        problem_generator="sod",
        dimensionality=1,
        num_cells=(512,),
        lower_corner=(0.0,),
        upper_corner=(1.0,),
        boundary="outflow",
        mhd=True,
        riemann_solver="hlld",
        cfl=0.5,
        gamma=2.0,
        end_time=0.1,
        initial_condition=brio_wu,
    ),
    ValidationCase(
        name="sod_hllc",
        problem_generator="sod",
        dimensionality=1,
        num_cells=(256,),
        lower_corner=(0.0,),
        upper_corner=(1.0,),
        boundary="outflow",
        mhd=False,
        riemann_solver="hllc",
        cfl=0.5,
        gamma=1.4,
        end_time=0.2,
    ),
    ValidationCase(
        name="sod_hlle",
        problem_generator="sod",
        dimensionality=1,
        num_cells=(256,),
        lower_corner=(0.0,),
        upper_corner=(1.0,),
        boundary="outflow",
        mhd=False,
        riemann_solver="hlle",
        cfl=0.5,
        gamma=1.4,
        end_time=0.2,
    ),
    ValidationCase(
        name="sound_wave_3d_hlle",
        problem_generator="linear_wave",
        num_cells=(32, 16, 16),
        mhd=False,
        riemann_solver="hlle",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=1.0,
        problem_parameters=dict(wave_flag=0, amp=1e-4, vflow=0.0),
        **_ALFVEN_BOX,
    ),
    ValidationCase(
        name="sedov_3d_hllc_fofc_floors",
        problem_generator="blast",
        dimensionality=3,
        num_cells=(64, 64, 64),
        lower_corner=(-0.5, -0.5, -0.5),
        upper_corner=(0.5, 0.5, 0.5),
        boundary="periodic",
        mhd=False,
        riemann_solver="hllc",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=0.02,
        first_order_flux_correction=True,
        floors=(1e-8, 1e-8),
        problem_parameters=dict(radius_outer=0.1, pressure_ambient=0.001, pressure_ratio=1.0e5),
    ),
    # Cases in which AthenaPK aborts with a negative pressure unless the
    # first-order flux correction (and, for the low-beta blast, the floors) is on.
    ValidationCase(
        name="einfeldt_mhd_fofc",
        problem_generator="sod",
        dimensionality=1,
        num_cells=(256,),
        lower_corner=(0.0,),
        upper_corner=(1.0,),
        boundary="outflow",
        mhd=True,
        riemann_solver="hlld",
        cfl=0.5,
        gamma=FIVE_THIRDS,
        end_time=0.08,
        first_order_flux_correction=True,
        initial_condition=einfeldt_rarefaction,
    ),
    ValidationCase(
        name="colliding_flows_mhd_fofc",
        problem_generator="orszag_tang",
        dimensionality=2,
        num_cells=(64, 64),
        lower_corner=(-0.5, -0.5),
        upper_corner=(0.5, 0.5),
        boundary="outflow",
        mhd=True,
        riemann_solver="hlld",
        cfl=0.4,
        gamma=FIVE_THIRDS,
        end_time=0.05,
        first_order_flux_correction=True,
        initial_condition=colliding_flows,
    ),
    ValidationCase(
        name="low_beta_blast_fofc_floors",
        problem_generator="blast",
        dimensionality=3,
        num_cells=(32, 32, 32),
        lower_corner=(-0.5, -0.5, -0.5),
        upper_corner=(0.5, 0.5, 0.5),
        boundary="periodic",
        mhd=True,
        riemann_solver="hlld",
        cfl=0.3,
        gamma=FIVE_THIRDS,
        end_time=0.05,
        first_order_flux_correction=True,
        floors=(1e-6, 1e-6),
        problem_parameters=dict(radius_outer=0.1, pressure_ratio=1.0e4),
        initial_condition=low_beta_blast,
    ),
]

CASES_BY_NAME = {case.name: case for case in CASES}

#: Small versions of representative cases for the fast regression test
#: (``pytests/mhd/test_vl2_athenapk_regression.py``), whose AthenaPK results are
#: stored in ``pytests/mhd/data/athenapk_vl2`` by
#: ``examples/scripts/validation/athenapk_vl2/make_regression_data.py``.
REGRESSION_CASES = [
    CASES_BY_NAME["cp_alfven_3d"]._replace(
        name="cp_alfven_3d_small",
        num_cells=(16, 8, 8),
        end_time=1.0,
    ),
    CASES_BY_NAME["orszag_tang"]._replace(
        name="orszag_tang_small",
        num_cells=(32, 32),
        end_time=0.25,
    ),
    CASES_BY_NAME["field_loop_hlle"]._replace(
        name="field_loop_hlle_small",
        num_cells=(32, 16),
        end_time=0.25,
    ),
    CASES_BY_NAME["magnetized_blast_3d_fofc"]._replace(
        name="magnetized_blast_fofc_small",
        num_cells=(16, 16, 16),
        end_time=0.03,
    ),
    CASES_BY_NAME["brio_wu_outflow"]._replace(
        name="brio_wu_small",
        num_cells=(128,),
    ),
    CASES_BY_NAME["sod_hllc"]._replace(
        name="sod_hllc_small",
        num_cells=(128,),
    ),
    CASES_BY_NAME["einfeldt_mhd_fofc"]._replace(
        name="einfeldt_mhd_fofc_small",
        num_cells=(128,),
    ),
    CASES_BY_NAME["colliding_flows_mhd_fofc"]._replace(
        name="colliding_flows_mhd_fofc_small",
        num_cells=(32, 32),
        end_time=0.03,
    ),
]

# -------------------------------------------------------------
# ====================== ↑ Case table ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================= ↓ astronomix configuration ↓ ==============
# -------------------------------------------------------------

#: AthenaPK's Riemann solver names mapped to the astronomix solvers.
_RIEMANN_SOLVERS = {"hlld": HLLD, "hlle": HLL, "hllc": HLLC, "llf": LAX_FRIEDRICHS}

#: AthenaPK's boundary names mapped to the astronomix boundaries.
_BOUNDARIES = {"periodic": PERIODIC_BOUNDARY, "outflow": OPEN_BOUNDARY}


def astronomix_setup(case: ValidationCase, state_shape, backend: int, block_shape=None):
    """
    The astronomix configuration, parameters and variable registry of a case.

    Args:
        case: The validation case.
        state_shape: The shape of the (unpadded) primitive state.
        backend: ``NATIVE_JAX`` or ``PALLAS``.
        block_shape: Optional Pallas block shape.

    Returns:
        ``(config, params, registered_variables)``.
    """
    axis_boundary = BoundarySettings1D(_BOUNDARIES[case.boundary], _BOUNDARIES[case.boundary])
    if case.dimensionality == 1:
        boundary_settings = axis_boundary
    else:
        inactive_axes = 3 - case.dimensionality
        boundary_settings = BoundarySettings(
            *([axis_boundary] * case.dimensionality + [BoundarySettings1D()] * inactive_axes)
        )
    box_size = [
        case.upper_corner[axis] - case.lower_corner[axis] for axis in range(case.dimensionality)
    ]

    # AthenaPK's floors correspond to the per-step hard floor, which also
    # switches on the floors inside every VL2 stage.
    if case.floors:
        positivity_config = PositivityConfig(per_step_mode=POSITIVITY_HARD_FLOOR)
    else:
        positivity_config = PositivityConfig()

    config = SimulationConfig(
        solver_mode=FINITE_VOLUME,
        solver_config=FVSolverConfig(
            time_integrator=VL2,
            reconstruction=DONOR_CELL if case.reconstruction == "dc" else PIECEWISE_LINEAR,
            limiter=VAN_LEER,
            riemann_solver=_RIEMANN_SOLVERS[case.riemann_solver],
            glm_extended_source=case.glm_extended_source,
            first_order_flux_correction=case.first_order_flux_correction,
        ),
        mhd=case.mhd,
        dimensionality=case.dimensionality,
        box_size=StaticFloatVector(*(box_size + [1.0] * (3 - case.dimensionality))),
        boundary_settings=boundary_settings,
        positivity_config=positivity_config,
        backend_config=BackendConfig(backend=backend, pallas_block_shape=block_shape),
    )
    params = SimulationParams(
        C_cfl=case.cfl,
        gamma=case.gamma,
        t_end=case.end_time,
        glm_alpha=case.glm_alpha,
        minimum_density=case.floors[0] if case.floors else 1e-14,
        minimum_pressure=case.floors[1] if case.floors else 1e-14,
    )
    registered_variables = get_registered_variables(config)
    config = finalize_config(config, state_shape)
    return config, params, registered_variables


def initial_primitive_state(case: ValidationCase):
    """The case's own initial primitive state (``None`` for problem-generator cases)."""
    if case.initial_condition is None:
        return None
    return jnp.asarray(case.initial_condition(case))


# -------------------------------------------------------------
# ================= ↑ astronomix configuration ↑ ==============
# -------------------------------------------------------------
