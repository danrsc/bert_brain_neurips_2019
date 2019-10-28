from . import syntactic_dependency
from . import boolean_questions
from . import choice_of_plausible_alternatives
from . import colorless_green
from . import commitment_bank
from . import corpus_base
from . import corpus_loader
from . import data_preparer
from . import dataset
from . import dataset_one_task_at_a_time
from . import fmri_example_builders
from . import harry_potter
from . import input_features
from . import multi_sentence_reading_comprehension
from . import natural_stories
from . import preprocessors
from . import reading_comprehension_with_common_sense_reasoning
from . import recognizing_textual_entailment
from . import spacy_token_meta
from . import stanford_sentiment_treebank
from . import university_college_london_corpus
from . import winograd_schema_challenge
from . import word_in_context

from .syntactic_dependency import *
from .boolean_questions import *
from .choice_of_plausible_alternatives import *
from .colorless_green import *
from .commitment_bank import *
from .corpus_base import *
from .corpus_loader import *
from .data_preparer import *
from .dataset import *
from .dataset_one_task_at_a_time import *
from .fmri_example_builders import *
from .harry_potter import *
from .input_features import *
from .multi_sentence_reading_comprehension import *
from .natural_stories import *
from .preprocessors import *
from .reading_comprehension_with_common_sense_reasoning import *
from .recognizing_textual_entailment import *
from .spacy_token_meta import *
from .stanford_sentiment_treebank import *
from .university_college_london_corpus import *
from .winograd_schema_challenge import *
from .word_in_context import *

from dataclasses import dataclass
from typing import Union

__all__ = [
    'syntactic_dependency', 'boolean_questions', 'choice_of_plausible_alternatives', 'colorless_green',
    'commitment_bank', 'corpus_base', 'corpus_loader', 'data_preparer', 'dataset', 'fmri_example_builders',
    'harry_potter', 'input_features', 'multi_sentence_reading_comprehension', 'natural_stories',
    'preprocessors', 'reading_comprehension_with_common_sense_reasoning', 'recognizing_textual_entailment',
    'spacy_token_meta', 'stanford_sentiment_treebank', 'university_college_london_corpus',
    'winograd_schema_challenge', 'word_in_context']
__all__.extend(syntactic_dependency.__all__)
__all__.extend(boolean_questions.__all__)
__all__.extend(choice_of_plausible_alternatives.__all__)
__all__.extend(colorless_green.__all__)
__all__.extend(commitment_bank.__all__)
__all__.extend(corpus_base.__all__)
__all__.extend(corpus_loader.__all__)
__all__.extend(data_preparer.__all__)
__all__.extend(dataset.__all__)
__all__.extend(dataset_one_task_at_a_time.__all__)
__all__.extend(fmri_example_builders.__all__)
__all__.extend(harry_potter.__all__)
__all__.extend(input_features.__all__)
__all__.extend(multi_sentence_reading_comprehension.__all__)
__all__.extend(natural_stories.__all__)
__all__.extend(preprocessors.__all__)
__all__.extend(reading_comprehension_with_common_sense_reasoning.__all__)
__all__.extend(recognizing_textual_entailment.__all__)
__all__.extend(spacy_token_meta.__all__)
__all__.extend(stanford_sentiment_treebank.__all__)
__all__.extend(university_college_london_corpus.__all__)
__all__.extend(winograd_schema_challenge.__all__)
__all__.extend(word_in_context.__all__)


@dataclass(frozen=True)
class _CorpusConstants:
    BooleanQuestions: Union[type, str] = BooleanQuestions
    ChoiceOfPlausibleAlternatives: Union[type, str] = ChoiceOfPlausibleAlternatives
    ColorlessGreenCorpus: Union[type, str] = ColorlessGreenCorpus
    CommitmentBank: Union[type, str] = CommitmentBank
    HarryPotterCorpus: Union[type, str] = HarryPotterCorpus
    LinzenAgreementCorpus: Union[type, str] = LinzenAgreementCorpus
    MultiSentenceReadingComprehension: Union[type, str] = MultiSentenceReadingComprehension
    NaturalStoriesCorpus: Union[type, str] = NaturalStoriesCorpus
    ReadingComprehensionWithCommonSenseReasoning: Union[type, str] = ReadingComprehensionWithCommonSenseReasoning
    RecognizingTextualEntailment: Union[type, str] = RecognizingTextualEntailment
    StanfordSentimentTreebank: Union[type, str] = StanfordSentimentTreebank
    UclCorpus: Union[type, str] = UclCorpus
    WinogradSchemaChallenge: Union[type, str] = WinogradSchemaChallenge
    WordInContext: Union[type, str] = WordInContext


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
