import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..common import zip_equal
from .syntactic_dependency import preprocess_english_morphology, collect_paradigms, extract_dependency_patterns, \
    generate_morph_pattern_test, DependencyTree, universal_dependency_reader, make_token_to_paradigms, \
    make_ltm_to_word, GeneratedExample

from .input_features import RawData, FieldSpec, KindData, ResponseKind
from .corpus_base import CorpusBase, CorpusExampleUnifier


__all__ = ['ColorlessGreenCorpus', 'LinzenAgreementCorpus']


def _iterate_delimited(path, field_delimiter='\t', pattern_field_delimiter='!', pattern_context_delimiter='_'):
    with open(path, 'rt') as file:
        for line in file:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:
                continue
            yield GeneratedExample.from_delimited(
                line, field_delimiter, pattern_field_delimiter, pattern_context_delimiter)


@dataclass
class _LinzenExample:
    words: Tuple[str, ...]
    index_target: int
    correct_form: str
    incorrect_form: str
    num_attractors: int

    @property
    def agreement_tuple(self):
        return self.words, self.correct_form, self.incorrect_form, self.index_target


def _iterate_linzen(directory_path):
    with open(os.path.join(directory_path, 'subj_agr_filtered.text'), 'rt') as sentence_file:
        with open(os.path.join(directory_path, 'subj_agr_filtered.gold'), 'rt') as gold_file:
            for sentence, gold in zip_equal(sentence_file, gold_file):
                index_target, correct_form, incorrect_form, num_attractors = gold.split('\t')
                yield _LinzenExample(
                    sentence.split()[:-1],  # remove <eos>
                    int(index_target),
                    correct_form,
                    incorrect_form,
                    int(num_attractors))


def generate_examples(english_web_path, bert_tokenizer):

    conll_reader = universal_dependency_reader

    if isinstance(english_web_path, str):
        english_web_path = [english_web_path]

    paradigms = collect_paradigms(english_web_path, morphology_preprocess_fn=preprocess_english_morphology)

    trees = [
        DependencyTree.from_conll_rows(sentence_rows, conll_reader.root_index, conll_reader.offset, text)
        for sentence_rows, text in conll_reader.iterate_sentences_chain_streams(
            english_web_path,
            morphology_preprocess_fn=preprocess_english_morphology)]

    syntax_patterns = extract_dependency_patterns(trees, freq_threshold=5, feature_keys={'Number'})

    paradigms = make_token_to_paradigms(paradigms)

    ltm_paradigms = make_ltm_to_word(paradigms)

    examples = list()
    for pattern in syntax_patterns:
        examples.extend(generate_morph_pattern_test(trees, pattern, ltm_paradigms, paradigms, bert_tokenizer))

    return examples


def _agreement_data(example_manager: CorpusExampleUnifier, examples, data_key):
    class_correct = 1
    class_incorrect = 0
    classes = list()

    for example_id, example in enumerate(examples):
        words, correct_form, incorrect_form, index_target = example.agreement_tuple
        words = list(words)

        # the generated example actually doesn't use the test item (the form field); it is a different random word
        # until we put the test item in there
        words[index_target] = correct_form

        data_ids = -1 * np.ones(len(words), dtype=np.int64)
        data_ids[index_target] = len(classes)
        example_manager.add_example(example_id, words, [example_id] * len(words), data_key, data_ids)
        classes.append(class_correct)

        # switch to the wrong number-agreement
        words[index_target] = incorrect_form

        data_ids = np.copy(data_ids)
        data_ids[index_target] = len(classes)
        example_manager.add_example(example_id, words, [example_id] * len(words), data_key, data_ids)
        classes.append(class_incorrect)

    return classes


class ColorlessGreenCorpus(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='english_web_universal_dependencies_v_2_3_path')

    def __init__(self, path=None):
        self.path = path

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        english_web_paths = [
            os.path.join(self.path, 'en_ewt-ud-train.conllu'),
            os.path.join(self.path, 'en_ewt-ud-dev.conllu'),
            os.path.join(self.path, 'en_ewt-ud-test.conllu')]

        classes = _agreement_data(
            example_manager, generate_examples(english_web_paths, example_manager.bert_tokenizer), 'colorless')

        def _readonly(arr):
            arr.setflags(write=False)
            return arr

        classes = {'colorless': KindData(ResponseKind.generic, _readonly(np.array(classes, dtype=np.float64)))}

        return RawData(
            list(example_manager.iterate_examples(fill_data_keys=True)), classes,
            validation_proportion_of_train=0.1, field_specs={'colorless': FieldSpec(is_sequence=False)})


class LinzenAgreementCorpus(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='linzen_agreement_path')

    def __init__(self, path=None):
        self.path = path

    def _load(self, run_info, example_manager: CorpusExampleUnifier):

        classes = _agreement_data(example_manager, _iterate_linzen(self.path), 'linzen_agree')

        def _readonly(arr):
            arr.setflags(write=False)
            return arr

        classes = {'linzen_agree': KindData(ResponseKind.generic, _readonly(np.array(classes, dtype=np.float)))}

        return RawData(
            list(example_manager.iterate_examples(fill_data_keys=True)), classes,
            validation_proportion_of_train=0.1, field_specs={'linzen_agree': FieldSpec(is_sequence=False)})
