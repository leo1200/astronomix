"""
Astrochemical reaction-network coupling.

Advects chemical species as scalar fields on the fluid grid and, once per hydro
step, reacts them per cell using a carbox reaction network integrated with a
stiff Diffrax solver. See ``_chemistry.py`` for the driver and
``chemistry_options.py`` for the configuration / parameter containers.
"""
