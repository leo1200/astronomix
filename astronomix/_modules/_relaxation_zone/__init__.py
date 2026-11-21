"""
The relaxation zone: pull the state towards a target state, e.g. an absorbing or reservoir layer.

Only the option containers are exported here: the configuration and parameter classes import them, so importing the
update (which needs those classes) from this package would be circular. ``per_step_updates`` imports
``apply_relaxation_zone`` from ``_relaxation_zone.py`` directly.
"""

from astronomix._modules._relaxation_zone._relaxation_zone_options import (
    RelaxationZoneConfig,
    RelaxationZoneParams,
)
