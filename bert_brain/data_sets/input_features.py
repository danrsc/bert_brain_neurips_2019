from dataclasses import dataclass
import dataclasses
from typing import Sequence, Optional, Mapping, Any, Union

import numpy as np
import torch


__all__ = ['InputFeatures', 'RawData', 'FieldSpec', 'KindData', 'ResponseKind', 'split_data']


@dataclass
class FieldSpec:
    fill_value: Any = None
    tensor_dtype: Any = torch.float
    is_sequence: bool = True

    def __post_init__(self):
        if self.fill_value is None:
            if self.tensor_dtype.is_floating_point:
                self.fill_value = np.nan
            else:
                self.fill_value = 0

    def __eq__(self, other):
        return np.isclose(self.fill_value, other.fill_value, rtol=0, atol=0, equal_nan=True) \
               and self.tensor_dtype == other.tensor_dtype \
               and self.is_sequence == other.is_sequence


@dataclass
class InputFeatures:
    unique_id: int
    tokens: Sequence[str]
    token_ids: Sequence[int]
    mask: Sequence[int]
    is_stop: Sequence[int]
    is_begin_word_pieces: Sequence[int]
    token_lengths: Sequence[int]
    token_probabilities: Sequence[float]
    type_ids: Sequence[int]
    head_location: Sequence[int]
    head_tokens: Sequence[str]
    head_token_ids: Sequence[int]
    index_word_in_example: Sequence[int]  # useful for grouping tokens together in the model
    index_token_in_sentence: Sequence[int]  # useful for positional embedding
    data_ids: Union[Mapping[str, Sequence[int]], Sequence[int]]
    multipart_id: Optional[int] = None
    span_ids: Optional[Sequence[int]] = None


@dataclass
class KindData:
    kind: str
    data: np.array


@dataclass(frozen=True)
class _ResponseKind:
    hp_fmri: str
    hp_meg: str
    ucl_erp: str
    ucl_eye: str
    ucl_self_paced: str
    ns_reaction_times: str
    ns_froi: str
    generic: str


ResponseKind = _ResponseKind(**dict((f.name, f.name) for f in dataclasses.fields(_ResponseKind)))


@dataclass
class RawData:
    input_examples: Sequence[InputFeatures]
    response_data: Mapping[str, KindData]
    test_input_examples: Optional[Sequence[InputFeatures]] = None
    validation_input_examples: Optional[Sequence[InputFeatures]] = None
    is_pre_split: bool = False
    test_proportion: float = 0.0
    validation_proportion_of_train: float = 0.1
    field_specs: Optional[Mapping[str, FieldSpec]] = None
    metadata: Optional[Mapping[str, np.array]] = None


def split_data(to_split, test_proportion, validation_of_train_proportion, shuffle=True, random_state=None):
    from sklearn.model_selection import train_test_split

    if test_proportion > 0:
        idx_train, idx_test = train_test_split(
            np.arange(len(to_split)), test_size=test_proportion, shuffle=shuffle, random_state=random_state)
    else:
        idx_train = np.arange(len(to_split))
        idx_test = []

    if validation_of_train_proportion > 0:
        idx_train, idx_validation = train_test_split(
            idx_train, test_size=validation_of_train_proportion, shuffle=shuffle, random_state=random_state)
    else:
        idx_validation = []

    train = [to_split[i] for i in idx_train]
    validation = [to_split[i] for i in idx_validation]
    test = [to_split[i] for i in idx_test]
    return train, validation, test
