"""
Configuration and parameter containers for the CNN-based MHD corrector.

These hold the static network architecture (in the config) and the trainable
network weights (in the params), kept separate so the architecture stays
static under jit while the weights can be differentiated through.
"""

# typing
from typing import NamedTuple, Union
from types import NoneType
from jaxtyping import PyTree


class CNNMHDconfig(NamedTuple):
    """Static configuration of the CNN MHD corrector.

    Attributes:
        cnn_mhd_corrector: Whether the CNN MHD corrector is active.
        network_static: The static (non-trainable) part of the equinox network,
            split off via ``eqx.partition`` so it can be carried as a static
            jit argument. ``None`` when the corrector is inactive.
    """

    cnn_mhd_corrector: bool = False
    network_static: Union[PyTree, NoneType] = None


class CNNMHDParams(NamedTuple):
    """Trainable parameters of the CNN MHD corrector.

    Attributes:
        network_params: The trainable part of the equinox network (the leaves
            differentiated against). ``None`` when the corrector is inactive.
    """

    network_params: Union[PyTree, NoneType] = None
