from typing import NamedTuple

# tracer seeding modes
MASS_WEIGHTED = 0  # sample initial positions with probability proportional to density
UNIFORM = 1        # sample initial positions uniformly in volume

# tracer position integrators (in space, using the current velocity field)
EULER = 0  # x <- x + dt * v(x)
RK2 = 1    # midpoint: k1 = v(x); k2 = v(x + dt/2 * k1); x <- x + dt * k2


class TracerConfig(NamedTuple):
    """Configuration of the Lagrangian tracer-particle module.

    Tracer particles are massless points advected by the (interpolated) fluid
    velocity. They carry no back-reaction on the flow; their purpose is to
    sample fluid quantities (here: temperature) along Lagrangian trajectories,
    which is what the Fokker-Planck / stochastic-temperature analysis needs.

    The module is single-GPU only for now: the velocity interpolation gathers
    from the full field, so a sharded field would need a halo/all-gather that is
    not yet implemented. Leave ``tracers=False`` (the default) for multi-device
    runs.

    Attributes:
        tracers: Master switch for the tracer module.
        num_tracers: Number of tracer particles.
        seed_mode: How initial positions are drawn when not supplied explicitly
            (``MASS_WEIGHTED`` ~ rho, or ``UNIFORM`` in volume).
        record_positions: Also store tracer positions at each snapshot (needed
            to mask particles that have drifted to a non-periodic boundary).
        integrator: Spatial integrator for the position update (``EULER`` / ``RK2``).
    """

    tracers: bool = False
    num_tracers: int = 10000
    seed_mode: int = MASS_WEIGHTED
    record_positions: bool = True
    integrator: int = RK2

    #: Re-inject tracers that exit a non-periodic boundary at the high end of
    #: that axis (the inflow side), instead of clamping them there. For a TRML
    #: with cold outflow at the bottom and hot inflow at the top this keeps the
    #: tracer population stationary and mass-representative; clamped tracers
    #: would otherwise pile up at the outflow boundary and bias the cold tail.
    #: A re-injection teleports the particle, so increments spanning one must be
    #: discarded in the analysis (detected from the recorded positions).
    reinject: bool = True

    #: Flux-matched boundary recycling (3D, inflow along +z). Each step a number
    #: of tracers matched to the top-boundary inflow mass flux is relocated to
    #: the hot inflow layer (random in-plane, top cell in z). This represents
    #: the inflowing mass that a fixed tracer set would otherwise never sample,
    #: and the removed (randomly chosen) tracers represent the cold gas cycling
    #: out — so the Lagrangian ensemble tracks the steady-state mass PDF rather
    #: than its initial seeding. Like re-injection, a recycle teleports the
    #: particle; increments spanning one are dropped in the analysis.
    recycle: bool = True

    #: Regeneration thermostat. Each step a fraction ``dt / regenerate_timescale``
    #: of tracers is re-drawn ∝ the *current* density, keeping the Lagrangian
    #: marginal equal to the instantaneous mass distribution even where the flow
    #: turns over faster than a fixed tracer set can follow (the thin, fast-
    #: cooling mixing layer fed by un-tracered inflow). The timescale must stay
    #: well above the lags used to measure A(T), D(T) so trajectory segments
    #: between regenerations are long enough; a regenerated tracer teleports, so
    #: increments spanning one are dropped in the analysis (via the recorded
    #: positions). Off by default (timescale ``inf``).
    regenerate: bool = False
    regenerate_timescale: float = float("inf")
