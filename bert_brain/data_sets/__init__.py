from . import corpus_base
from . import corpus_loader
from . import data_preparer
from . import dataset
from . import fmri_example_builders
from . import harry_potter
from . import input_features
from . import preprocessors
from . import spacy_token_meta

from .corpus_base import *
from .corpus_loader import *
from .data_preparer import *
from .dataset import *
from .fmri_example_builders import *
from .harry_potter import *
from .input_features import *
from .preprocessors import *
from .spacy_token_meta import *

from dataclasses import dataclass
from typing import Union

__all__ = [
    'corpus_base', 'corpus_loader', 'data_preparer', 'dataset', 'fmri_example_builders',
    'harry_potter', 'input_features', 'preprocessors']
__all__.extend(corpus_base.__all__)
__all__.extend(corpus_loader.__all__)
__all__.extend(data_preparer.__all__)
__all__.extend(dataset.__all__)
__all__.extend(fmri_example_builders.__all__)
__all__.extend(harry_potter.__all__)
__all__.extend(input_features.__all__)
__all__.extend(preprocessors.__all__)
__all__.extend(spacy_token_meta.__all__)


@dataclass(frozen=True)
class _CorpusConstants:
    HarryPotterCorpus: Union[type, str] = HarryPotterCorpus


def _corpus_subclasses_recursive():
    def sub(c, result):
        result.append(c)
        for sc in c.__subclasses__():
            sub(sc, result)
    corpus_types = list()
    for cb in CorpusBase.__subclasses__():
        sub(cb, corpus_types)
    return corpus_types


CorpusTypes = _CorpusConstants()
CorpusKeys = _CorpusConstants(**dict((t.__name__, t.__name__) for t in _corpus_subclasses_recursive()))


__all__.append('CorpusTypes')
__all__.append('CorpusKeys')
