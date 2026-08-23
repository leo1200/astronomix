"""
Astrochemical reaction-network coupling.

Advects chemical species as scalar fields on the fluid grid and, once per hydro
step, reacts them per cell using a carbox reaction network integrated with a
stiff Diffrax solver. See ``_chemistry.py`` for the driver and
``chemistry_options.py`` for the configuration / parameter containers.

LICENSE: unlike the rest of astronomix (MIT), this package is GPL-3.0 — the
thermochemistry is derived from KROME (GPL) and the coupling depends on carbox
(GPL). See ``LICENSE.md`` in this directory.
"""
