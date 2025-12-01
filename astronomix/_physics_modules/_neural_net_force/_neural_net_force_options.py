from types import NoneType
from typing import NamedTuple, Union

from jaxtyping import PyTree

class NeuralNetForceConfig(NamedTuple):
    neural_net_force: bool = False
    network_static: Union[PyTree, NoneType] = None

class NeuralNetForceParams(NamedTuple):
    network_params: Union[PyTree, NoneType] = None