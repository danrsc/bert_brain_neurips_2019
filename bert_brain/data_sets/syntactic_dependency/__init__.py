from . import conll_reader
from . import morphological_paradigm
from . import tree_module

from .conll_reader import *
from .morphological_paradigm import *
from .tree_module import *

__all__ = ['conll_reader', 'morphological_paradigm', 'tree_module']
__all__.extend(conll_reader.__all__)
__all__.extend(morphological_paradigm.__all__)
__all__.extend(tree_module.__all__)
