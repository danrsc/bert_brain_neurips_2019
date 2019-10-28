import os
from inspect import signature
import hashlib
from collections import OrderedDict
import itertools
import dataclasses
from typing import Sequence, Union, Optional, Hashable, Mapping

import numpy as np
import torch

from spacy.language import Language as SpacyLanguage
from pytorch_pretrained_bert import BertTokenizer

from .spacy_token_meta import bert_tokenize_with_spacy_meta
from .input_features import InputFeatures, FieldSpec
from .corpus_cache import save_to_cache, load_from_cache


__all__ = ['CorpusBase', 'CorpusExampleUnifier']


class CorpusExampleUnifier:

    def __init__(self, spacy_tokenize_model: SpacyLanguage, bert_tokenizer: BertTokenizer):
        self.spacy_tokenize_model = spacy_tokenize_model
        self.bert_tokenizer = bert_tokenizer
        self._examples = OrderedDict()
        self._seen_data_keys = OrderedDict()

    def add_example(
            self,
            example_key: Optional[Hashable],
            words: Sequence[str],
            sentence_ids: Sequence[int],
            data_key: Optional[Union[str, Sequence[str]]],
            data_ids: Optional[Sequence[int]],
            start: int = 0,
            stop: Optional[int] = None,
            start_sequence_2: Optional[int] = None,
            stop_sequence_2: Optional[int] = None,
            start_sequence_3: Optional[int] = None,
            stop_sequence_3: Optional[int] = None,
            is_apply_data_id_to_entire_group: bool = False,
            multipart_id: Optional[int] = None,
            span_ids: Optional[Sequence[int]] = None,
            allow_new_examples: bool = True) -> Optional[InputFeatures]:
        """
        Adds an example for the current data loader to return later. Simplifies the process of merging examples
        across different response measures. For example MEG and fMRI
        Args:
            example_key: For instance, the position of the example within a story. If this is set to None, then the
                tokens will be used as the example_key. However, this may be undesirable since in a given passage,
                sentences, especially short sentences, can be repeated.
            words: The words in the example
            sentence_ids: For each word, identifies which sentence the word belongs to. Used to compute
                index_of_word_in_sentence in the resulting InputFeatures
            data_key: A key (or multiple keys) to designate which response data set(s) data_ids references
            data_ids: indices into the response data, one for each token
            start: Offset where the actual input features should start. It is best to compute spacy meta on full
                sentences, then slice the resulting tokens. start and stop are used to slice words, sentence_ids,
                data_ids and type_ids
            stop: Exclusive end point for the actual input features. If None, the full length is used
            is_apply_data_id_to_entire_group: If a word is broken into multiple tokens, generally a single token is
                heuristically chosen as the 'main' token corresponding to that word. The data_id it is assigned is given
                by data offset, while all the tokens that are not the main token in the group are assigned -1. If this
                parameter is set to True, then all of the multiple tokens corresponding to a word are assigned the same
                data_id, and none are set to -1. This can be a better option for fMRI where the predictions are not at
                the word level, but rather at the level of an image containing multiple words.
            start_sequence_2: Used for bert to combine multiple sequences as a single input. Generally this is used for
                tasks like question answering where type_id=0 is the question and type_id=1 is the answer.
                If not specified, type_id=0 is used for every token
            stop_sequence_2: Used for bert to combine multiple sequences as a single input. Generally this is used for
                tasks like question answering where type_id=0 is the question and type_id=1 is the answer.
            start_sequence_3: Used for bert to combine 3 sequences as a single input. Generally this is used for tasks
                like question answering with a context. type_id=0 is the context and type_id=1 is the question and
                answer
            stop_sequence_3: Used for bert to combine 3 sequences as a single input. Generally this is used for tasks
                like question answering with a context. type_id=0 is the context and type_id=1 is the question and
                answer
            multipart_id: Used to express that this example needs to be in the same batch as other examples sharing the
                same multipart_id to be evaluated
            span_ids: Bit-encoded span identifiers which indicate which spans each word belongs to when spans are
                labeled in the input. If not given, no span ids will be set on the returned InputFeatures instance.
            allow_new_examples: If False, then if the example does not already exist in this instance, it will not
                be added. Only new data_ids will be added to existing examples. Returns None when the example does
                not exist.
        Returns:
            The InputFeatures instance associated with the example
        """
        input_features = bert_tokenize_with_spacy_meta(
            self.spacy_tokenize_model, self.bert_tokenizer,
            len(self._examples), words, sentence_ids, data_key, data_ids,
            start, stop,
            start_sequence_2, stop_sequence_2,
            start_sequence_3, stop_sequence_3,
            multipart_id,
            span_ids,
            is_apply_data_id_to_entire_group)

        if example_key is None:
            example_key = tuple(input_features.token_ids)

        if example_key not in self._examples:
            if allow_new_examples:
                self._examples[example_key] = input_features
            else:
                return None
        else:
            current = dataclasses.asdict(input_features)
            have = dataclasses.asdict(self._examples[example_key])
            assert(len(have) == len(current))
            for k in have:
                assert(k in current)
                if k == 'unique_id' or k == 'data_ids':
                    continue
                else:
                    # handles NaN, whereas np.array_equal does not
                    np.testing.assert_array_equal(have[k], current[k])
            if data_key is not None:
                if isinstance(data_key, str):
                    data_key = [data_key]
                for k in data_key:
                    self._seen_data_keys[k] = True
                    self._examples[example_key].data_ids[k] = input_features.data_ids[k]

        return self._examples[example_key]

    def iterate_examples(self, fill_data_keys=False):
        for k in self._examples:
            if fill_data_keys:
                for data_key in self._seen_data_keys:
                    if data_key not in self._examples[k].data_ids:
                        self._examples[k].data_ids[data_key] = -1 * np.ones(
                            len(self._examples[k].token_ids), dtype=np.int64)
            yield self._examples[k]

    def remove_data_keys(self, data_keys):
        if isinstance(data_keys, str):
            data_keys = [data_keys]
        for ex in self.iterate_examples():
            for data_key in data_keys:
                if data_key in ex.data_ids:
                    del ex.data_ids[data_key]
        for data_key in data_keys:
            if data_key in self._seen_data_keys:
                del self._seen_data_keys[data_key]

    def __len__(self):
        return len(self._examples)


class CorpusBase:

    @classmethod
    def _path_attributes(cls) -> Optional[Mapping[str, str]]:
        """
        A corpus declares a mapping from the paths object to its own path attributes
        by defining this function. E.g.:
            def _path_attributes(cls):
                return dict(path='harry_potter_path')
        """
        raise NotImplementedError('{} does not implement _path_attributes'.format(cls))

    @classmethod
    def _hash_arguments(cls, kwargs):
        hash_ = hashlib.sha256()
        for key in kwargs:
            s = '{}={}'.format(key, kwargs[key])
            hash_.update(s.encode())
        return hash_.hexdigest()

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        sig = signature(cls.__init__)
        bound_arguments = sig.bind_partial(*args, **kwargs)
        bound_arguments.apply_defaults()
        obj._bound_arguments = bound_arguments.arguments
        obj._argument_hash = cls._hash_arguments(obj._bound_arguments)
        obj._cache_base_path = None
        return obj

    @property
    def argument_hash(self):
        return self._argument_hash

    def set_paths_from_path_object(self, path_obj):
        #   A corpus declares a mapping from the paths object to its own path attributes
        #   by defining this function. E.g.:
        #       def _path_attributes(cls):
        #           return dict(path='harry_potter_path')
        path_attribute_mapping = type(self)._path_attributes()
        for path_attribute in path_attribute_mapping:
            current_value = getattr(self, path_attribute)
            if current_value is None:
                if not hasattr(path_obj, path_attribute_mapping[path_attribute]):
                    raise ValueError(
                        'Paths instance has no attribute {}'.format(path_attribute_mapping[path_attribute]))
                paths_value = getattr(path_obj, path_attribute_mapping[path_attribute])
                setattr(self, path_attribute, paths_value)
        # special case for cache_path
        if self._cache_base_path is None:
            # noinspection PyAttributeOutsideInit
            self._cache_base_path = os.path.join(path_obj.cache_path, type(self).__name__)

    def cache_path(self, run_info):
        arg_hash = type(self)._hash_arguments({'argument_hash': self._argument_hash, 'run_info': run_info})
        return os.path.join(self._cache_base_path, '{}.npz'.format(arg_hash))

    def check_paths(self):
        path_attribute_mapping = type(self)._path_attributes()
        for path_attribute in path_attribute_mapping:
            current_value = getattr(self, path_attribute)
            if current_value is None:
                raise ValueError('{} is not populated. Either call load with a Paths instance or set the path manually '
                                 'before calling load'.format(path_attribute))

    @staticmethod
    def _populate_default_field_specs(raw_data):
        x, y, z = raw_data.input_examples, raw_data.validation_input_examples, raw_data.test_input_examples
        if x is None:
            x = []
        if y is None:
            y = []
        if z is None:
            z = []
        all_fields = set()
        for ex in itertools.chain(x, y, z):
            all_fields.update([field.name for field in dataclasses.fields(ex)])

        default_field_specs = {
            'unique_id': FieldSpec(tensor_dtype=torch.long, is_sequence=False),
            'tokens': FieldSpec(fill_value='[PAD]', tensor_dtype=str),
            'token_ids': FieldSpec(tensor_dtype=torch.long),
            'mask': FieldSpec(tensor_dtype=torch.uint8),
            'is_stop': FieldSpec(fill_value=1, tensor_dtype=torch.uint8),
            'is_begin_word_pieces': FieldSpec(tensor_dtype=torch.uint8),
            'token_lengths': FieldSpec(tensor_dtype=torch.long),
            'token_probabilities': FieldSpec(fill_value=-20.),
            'head_location': FieldSpec(fill_value=np.nan),
            'head_tokens': FieldSpec(fill_value='[PAD]', tensor_dtype=str),
            'head_token_ids': FieldSpec(tensor_dtype=torch.long),
            'type_ids': FieldSpec(tensor_dtype=torch.long),
            'data_ids': FieldSpec(fill_value=-1, tensor_dtype=torch.long),
            'span_ids': FieldSpec(fill_value=0, tensor_dtype=torch.long),
            'index_word_in_example': FieldSpec(fill_value=-1, tensor_dtype=torch.long),
            'index_token_in_sentence': FieldSpec(fill_value=0, tensor_dtype=torch.long),
            'multipart_id': FieldSpec(tensor_dtype=torch.long, is_sequence=False)
        }

        if raw_data.field_specs is None:
            raw_data.field_specs = {}
        for field in all_fields:
            if field not in raw_data.field_specs and field in default_field_specs:
                raw_data.field_specs[field] = default_field_specs[field]

    def load(
            self,
            index_run,
            spacy_tokenizer_model: SpacyLanguage,
            bert_tokenizer: BertTokenizer,
            paths_obj=None,
            force_cache_miss=False):

        run_info = self._run_info(index_run)

        if paths_obj is not None:
            self.set_paths_from_path_object(paths_obj)
        else:
            self.check_paths()

        result = load_from_cache(self.cache_path(run_info), run_info, self._bound_arguments, force_cache_miss)
        if result is not None:
            return result

        example_manager = CorpusExampleUnifier(spacy_tokenizer_model, bert_tokenizer)
        result = self._load(run_info, example_manager)
        CorpusBase._populate_default_field_specs(result)
        save_to_cache(self.cache_path(run_info), result, run_info, self._bound_arguments)
        return result

    def _run_info(self, index_run):
        return -1

    def _load(self, run_load_info, example_manager: CorpusExampleUnifier):
        raise NotImplementedError('{} does not implement _load'.format(type(self)))
