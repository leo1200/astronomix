from types import NoneType
from typing import NamedTuple, Union

from jaxtyping import PyTree

class CNNMHDconfig(NamedTuple):
    cnn_mhd_corrector: bool = False
    network_static: Union[PyTree, NoneType] = None

class CNNMHDParams(NamedTuple):
    network_params: Union[PyTree, NoneType] = None