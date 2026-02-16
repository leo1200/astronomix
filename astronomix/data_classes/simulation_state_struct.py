from types import NoneType
from typing import NamedTuple, Union

from astronomix.option_classes.simulation_config import STATE_TYPE


class StateStruct(NamedTuple):
    """
    Struct for the simulation state.
    """

    #: The fluid state.
    primitive_state: Union[STATE_TYPE, NoneType] = None

    # here you might add more fields like the 
    # positions of star particles, etc.
