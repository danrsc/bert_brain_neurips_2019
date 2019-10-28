import os
import json

import numpy as np

from ..common import NamedSpanEncoder
from .input_features import RawData, KindData, ResponseKind, FieldSpec
from .corpus_base import CorpusBase, CorpusExampleUnifier


__all__ = ['WinogradSchemaChallenge']


class WinogradSchemaChallenge(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='winograd_schema_challenge_path')

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def _read_examples(path, example_manager: CorpusExampleUnifier, labels, named_span_encoder):
        examples = list()
        with open(path, 'rt') as f:
            for line in f:
                fields = json.loads(line.strip('\n'))
                text = fields['text'].split()
                span_ids = [0] * len(text)
                target = fields['target']
                span_id = 1
                while True:
                    span_index_field = 'span{}_index'.format(span_id)
                    span_text_field = 'span{}_text'.format(span_id)
                    if span_index_field not in target:
                        break
                    span_index = target[span_index_field]
                    span_text = target[span_text_field].split()
                    encoded_span = named_span_encoder.encode('span_{}'.format(span_id))
                    for i in range(len(span_text)):
                        if span_text[i] != text[i + span_index]:
                            raise ValueError('Mismatched span')
                        span_ids[i + span_index] += encoded_span
                    span_id += 1
                label = fields['label']
                data_ids = -1 * np.ones(len(text), dtype=np.int64)
                # doesn't matter which word we attach the label to since we specify below that is_sequence=False
                data_ids[0] = len(labels)
                examples.append(example_manager.add_example(
                    example_key=None,
                    words=text,
                    sentence_ids=[0] * len(text),
                    data_key='wsc',
                    data_ids=data_ids,
                    span_ids=span_ids,
                    start=0,
                    stop=len(text)))
                labels.append(label)
        return examples

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        labels = list()
        named_span_encoder = NamedSpanEncoder()
        train = WinogradSchemaChallenge._read_examples(
            os.path.join(self.path, 'train.jsonl'), example_manager, labels, named_span_encoder)
        validation = WinogradSchemaChallenge._read_examples(
            os.path.join(self.path, 'val.jsonl'), example_manager, labels, named_span_encoder)
        test = WinogradSchemaChallenge._read_examples(
            os.path.join(self.path, 'test.jsonl'), example_manager, labels, named_span_encoder)
        labels = np.array(labels, dtype=np.float64)
        labels.setflags(write=False)
        return RawData(
            input_examples=train,
            validation_input_examples=validation,
            test_input_examples=test,
            response_data={'wsc': KindData(ResponseKind.generic, labels)},
            is_pre_split=True,
            field_specs={'wsc': FieldSpec(is_sequence=False)})
