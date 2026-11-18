"""
Configuration container for sink particles.

Sink particles are created following the checks of Federrath et al. (2010),
ApJ 713, 269, Section 2.2, with a safety density threshold above which the
checks that judge by gas motion are skipped, and they accrete the gas above
the density threshold within their accretion radius (Section 2.3). The density
threshold is derived from the Jeans length resolution (Eq. 32 of the paper).
They move with a kick-drift-kick leapfrog, once per hydro step, and all their
gravity goes through the grid (Gong & Ostriker 2013, ApJS 204, 8). All options
here are static: changing them triggers recompilation.
"""

# typing
from typing import NamedTuple


class SinkParticleConfig(NamedTuple):
    """
    Static configuration of the sink particles: their formation, their
    accretion of gas and their motion.

    A sink moves with a kick-drift-kick leapfrog, once per hydro step, under
    the gravity of the gas, of the other sinks and of the external potential.
    All of it goes through the grid: the sink's mass is spread over the 3×3×3
    cells around it with triangular-shaped-cloud weights and added to the gas
    density of the Poisson solve, and its acceleration is read back from the
    same cells with the same weights, so a sink feels no force from itself
    and the forces between two masses are equal and opposite. Adaptive steps
    keep every sink from moving more than half a cell per step.

    Along a non-periodic axis a sink can leave the box. It keeps its slot, a
    warning is printed, and its gravity fades out between one cell inside and
    one cell outside the edge; beyond that it moves in a straight line. Where
    the Poisson solve treats the edge of the box differently from the gas
    boundaries (periodic gas with ``poisson_manual_open_boundaries``, mixed
    boundaries), a sink within 4 cells of the edge (2 with VL2) feels a force
    from itself, and the forces between masses are not exactly equal and
    opposite.
    """

    #: Switch sink particles on or off.
    sink_particles: bool = False

    #: Size of the sink particle arrays. JIT requires fixed array sizes, so
    #: this many slots are allocated up front. When a new sink would need
    #: more slots, the run stops with an error. Sinks passed into a run, or
    #: restored from a checkpoint, with fewer slots get empty slots appended;
    #: with more slots, the run does not start.
    max_num_sinks: int = 256

    #: Maximum number of cells that can be examined per time step with the
    #: control-volume checks (Jeans instability, bound state, proximity).
    #: Cells beyond this number are ignored for that step (with a warning).
    max_num_candidates: int = 128

    #: Accretion radius r_acc in units of the grid spacing. It sets both the
    #: radius of the control volume and the density threshold. Federrath et
    #: al. (2010), Section 4.2.1, use r_acc = 2.5 grid cells. With ghost cells,
    #: its integer part must not exceed num_ghost_cells.
    accretion_radius_in_cells: float = 2.5

    #: Jeans length N, in units of the grid spacing, that defines the safety
    #: density threshold ρ_safe = π c_s² / (G N² Δx²), the density at which
    #: the Jeans length is N cells. Relative to the density of Eq. 32,
    #: ρ_res = π c_s² / (4 G r_acc²), at which the Jeans length is 2 r_acc,
    #: ρ_safe / ρ_res = (2 r_acc / (N Δx))², which is (5 / N)² for
    #: r_acc = 2.5 Δx. accretion_radius_in_cells changes ρ_res but not
    #: ρ_safe.
    #: Above ρ_safe, the checks that judge by gas motion are skipped: the
    #: converging flow and bound state checks of the creation, and the bound
    #: and radial velocity checks of the accretion. On a uniform grid these
    #: checks can keep rejecting collapsing gas, e.g. a dense layer fed by fast
    #: inflow, until the Jeans length is far below a cell. The default puts
    #: ρ_safe at the Larson–Penston density at half a cell, the creation
    #: threshold of Gong & Ostriker (2013), ApJS 204, 8, so that
    #: ρ_safe ≈ 22 ρ_res for r_acc = 2.5 Δx. The safety threshold is not part
    #: of the method of Federrath et al. (2010).
    safety_jeans_length_in_cells: float = 1.06
