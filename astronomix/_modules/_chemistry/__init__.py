"""
Advected chemical species with a pluggable reaction source term.

astronomix can carry chemical-species number densities as extra scalar fields in
the state array; in finite-volume mode they advect with the flow for free (like
the cosmic-ray / wind-density tracers). Once per hydro step an optional,
user-supplied source term is applied to them (and, if it chooses, the energy
field) as an operator-split update. astronomix provides only the mechanism —
registration, advection and the hook — so no particular chemistry engine is
baked into the core.
"""
