"""
Configuration and parameter containers for the neural-network body force.

The trainable weights live in ``NeuralNetForceParams`` (a differentiable leaf),
while the static network structure is kept in ``NeuralNetForceConfig`` so it can
be treated as a static argument by the JIT machinery.
"""

# typing
from typing import NamedTuple, Union
from types import NoneType
from jaxtyping import PyTree


class NeuralNetForceConfig(NamedTuple):
    neural_net_force: bool = False
    network_static: Union[PyTree, NoneType] = None


class NeuralNetForceParams(NamedTuple):
    network_params: Union[PyTree, NoneType] = None
