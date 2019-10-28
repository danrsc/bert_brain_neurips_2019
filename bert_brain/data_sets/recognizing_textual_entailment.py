import os
import json

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier


__all__ = ['RecognizingTextualEntailment']


class RecognizingTextualEntailment(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='rte_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()
        classes = {
            'not_entailment': 0,
            'entailment': 1
        }
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                premise = fields['premise'].split()
                hypothesis = fields['hypothesis'].split()
                if fields['label'] not in classes:
                    raise ValueError('Unknown label: {}'.format(fields['label']))
                label = classes[fields['label']]
                data_ids = -1 * np.ones(len(premise) + len(hypothesis), dtype=np.int64)
                # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                data_ids[0] = len(labels)
                examples.append(example_manager.add_example(
                    example_key=None,
                    words=premise + hypothesis,
                    sentence_ids=[0] * len(premise) + [1] * len(hypothesis),
                    data_key='rte',
                    data_ids=data_ids,
                    start=0,
                    stop=len(premise),
                    start_sequence_2=len(premise),
                    stop_sequence_2=len(premise) + len(hypothesis)))
                labels.append(label)
        return examples

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        train = RecognizingTextualEntailment._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        validation = RecognizingTextualEntailment._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = RecognizingTextualEntailment._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={'rte': KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={'rte': FieldSpec(is_sequence=False)})
