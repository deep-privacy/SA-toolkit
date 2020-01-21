from .xvector import Xtractor
from .brij_xvector import BrijSpeakerXvector
from .gradient_reverse import grad_reverse

"""
``damped.nets `` provides a sets of building block to create custom neural net
architecture.
"""

# if somebody does "from somepackage import *", this is what they will
# be able to access:
_all__ = [
    "Xtractor",
    "BrijSpeakerXvector",
    "grad_reverse",
]
