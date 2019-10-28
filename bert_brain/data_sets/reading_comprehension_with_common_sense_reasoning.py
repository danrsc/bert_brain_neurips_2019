import os
from collections import OrderedDict
import json

import numpy as np

from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier


__all__ = ['ReadingComprehensionWithCommonSenseReasoning']


class ReadingComprehensionWithCommonSenseReasoning(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='reading_comprehension_with_common_sense_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                passage = fields['passage']['text']
                entities = OrderedDict()
                for entity_indices in fields['passage']['entities']:
                    start, end = entity_indices['start'], entity_indices['end']
                    entities[(start, end)] = len(entities), passage[start:(end + 1)]
                for question_answer in fields['qas']:
                    multipart_id = len(example_manager)
                    question_template = question_answer['query']
                    answer_ids = set()
                    for answer_indices in question_answer['answers']:
                        entity_id, _ = entities[(answer_indices['start'], answer_indices['end'])]
                        answer_ids.add(entity_id)
                    for k in entities:
                        entity_id, entity = entities[k]
                        label = 1 if entity_id in answer_ids else 0
                        question = question_template.replace('@placeholder', entity)
                        data_ids = -1 * np.ones(len(passage) + len(question), dtype=np.int64)
                        # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                        data_ids[0] = len(labels)
                        examples.append(example_manager.add_example(
                            example_key=None,
                            words=passage + question,
                            sentence_ids=[0] * len(passage) + [1] * len(question),
                            data_key='record',
                            data_ids=data_ids,
                            start=0,
                            stop=len(passage),
                            start_sequence_2=len(passage),
                            stop_sequence_2=len(passage) + len(question),
                            multipart_id=multipart_id))
                        labels.append(label)
        return examples

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        train = ReadingComprehensionWithCommonSenseReasoning._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels)
        validation = ReadingComprehensionWithCommonSenseReasoning._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels)
        test = ReadingComprehensionWithCommonSenseReasoning._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={'record': KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={'record': FieldSpec(is_sequence=False)})
