import os
import json

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier


__all__ = ['MultiSentenceReadingComprehension']


class MultiSentenceReadingComprehension(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='multi_sentence_reading_comprehension_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                passage = fields['passage']['text']
                passage = passage.split()
                questions = fields['passage']['questions']
                for question in questions:
                    q = question['question'].split()
                    for answer in question['answers']:
                        a = answer['text'].split()
                        label = answer['label']
                        data_ids = -1 * np.ones(len(passage) + len(q) + len(answer), dtype=np.int64)
                        # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                        data_ids[0] = len(labels)
                        examples.append(example_manager.add_example(
                            example_key=None,
                            words=passage + q + a,
                            sentence_ids=[0] * len(passage) + [1] * (len(question) + len(answer)),
                            data_key='multi_rc',
                            data_ids=data_ids,
                            start=0,
                            stop=len(passage),
                            start_sequence_2=len(passage),
                            stop_sequence_2=len(passage) + len(q) + len(a)))
                        labels.append(label)
        return examples

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        train = MultiSentenceReadingComprehension._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        validation = MultiSentenceReadingComprehension._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = MultiSentenceReadingComprehension._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={'multi_rc': KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={'multi_rc': FieldSpec(is_sequence=False)})
