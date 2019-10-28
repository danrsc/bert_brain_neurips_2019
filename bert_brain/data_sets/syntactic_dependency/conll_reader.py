# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# https://github.com/facebookresearch/colorlessgreenRNNs
# /blob/8d41f2a2301d612ce25be90dfc1e96f828f77c85/src/syntactic_testsets/conll_reader.py

# !/usr/bin/env python


import sys

from dataclasses import dataclass
from typing import Optional


__all__ = ['ConllRow', 'ConllReader', 'universal_dependency_reader', 'universal_dependency_fine_part_of_speech_reader',
           'conll_09_reader', 'zgen_reader', 'arcs_conll_reader', 'dependency_label_types', 'get_reader', 'all_readers']


@dataclass
class ConllRow:
    index: Optional[str] = None
    word: str = ''
    lemma: str = ''
    head_id: Optional[str] = None
    pos: str = ''
    dependency_label: str = ''
    morph: Optional[str] = None


class ConllReader:

    def __init__(
            self,
            name: str,
            index_column: Optional[int] = 0,
            word_column: int = 1,
            lemma_column: int = 2,
            part_of_speech_column: int = 3,
            morphology_column: Optional[int] = 5,
            head_index_column: int = 6,
            dependency_label_column: int = 7,
            offset: int = 1,
            root_index: int = 0,
            num_columns: int = 10):
        self.name = name
        self.index_column = index_column
        self.word_column = word_column
        self.lemma_column = lemma_column
        self.part_of_speech_column = part_of_speech_column
        self.morphology_column = morphology_column
        self.head_index_column = head_index_column
        self.dependency_label_column = dependency_label_column
        self.offset = offset
        self.root_index = root_index
        self.num_columns = num_columns

    @staticmethod
    def _text_comment_or_none(line):
        if not line.startswith('#'):
            return None
        idx_equals = line.find('=')
        if idx_equals < 0:
            return None
        prefix = line[1:idx_equals].strip()
        if prefix == 'text':
            return line[idx_equals + 1:].strip()
        return None

    def _read_row(self, fields, morphology_preprocess_fn):
        if self.morphology_column is not None:
            morph = fields[self.morphology_column]
            if morphology_preprocess_fn is not None:
                morph = morphology_preprocess_fn(morph)
        else:
            morph = None

        return ConllRow(
            index=fields[self.index_column],
            word=fields[self.word_column],
            lemma=fields[self.lemma_column],
            head_id=fields[self.head_index_column],
            pos=fields[self.part_of_speech_column],
            dependency_label=fields[self.dependency_label_column],
            morph=morph)

    @staticmethod
    def _iterate_blank_line_blocks(stream):
        block = list()
        text = None
        for line in stream:
            line = line.strip()
            if len(line) == 0:
                if len(block) > 0 or text is not None:
                    yield block, text
                block = list()
                text = None
            else:
                text_candidate = ConllReader._text_comment_or_none(line)
                if text_candidate is not None:
                    if text is not None:
                        raise ValueError('two text comments in the same sentence')
                    text = text_candidate
                elif not line.startswith('#'):
                    fields = line.split('\t')
                    block.append(fields)

        if len(block) > 0 or text is not None:
            yield block, text

    def iterate_sentences_chain_streams(self, streams_or_paths, morphology_preprocess_fn=None):
        for stream_or_path in streams_or_paths:
            for result in self.iterate_sentences(stream_or_path, morphology_preprocess_fn=morphology_preprocess_fn):
                yield result

    def iterate_sentences(self, stream_or_path, morphology_preprocess_fn=None):
        if isinstance(stream_or_path, str):
            with open(stream_or_path, 'rt') as stream:
                for result in self.iterate_sentences(stream, morphology_preprocess_fn):
                    yield result
        else:
            for block, text in ConllReader._iterate_blank_line_blocks(stream_or_path):
                is_valid = True
                # Check that the grid is consistent.

                rows = list()
                for row in block:
                    if len(row) != len(block[0]):
                        print(block)
                        # raise ValueError('Inconsistent number of columns:\n%s'% block)
                        sys.stderr.write('Inconsistent number of columns', block)
                        is_valid = False
                        break
                    rows.append(self._read_row(row, morphology_preprocess_fn))

                if is_valid:
                    yield rows, text


universal_dependency_reader = ConllReader('universal_dependency_reader')
universal_dependency_fine_part_of_speech_reader = ConllReader(
    'universal_dependency_fine_part_of_speech_reader', part_of_speech_column=4)
conll_09_reader = ConllReader(
    'conll_09_reader',
    part_of_speech_column=4, morphology_column=6, head_index_column=8, dependency_label_column=10, num_columns=12)
zgen_reader = ConllReader(
    'zgen_reader', index_column=None, word_column=0, lemma_column=0, part_of_speech_column=1, morphology_column=None,
    head_index_column=2, dependency_label_column=3, offset=0, root_index=-1, num_columns=4)
arcs_conll_reader = ConllReader(
    'arcs_conll_reader', lemma_column=1, part_of_speech_column=2, morphology_column=None, head_index_column=3,
    dependency_label_column=6, num_columns=7)

all_readers = dict((r.name, r) for r in [
    universal_dependency_reader, universal_dependency_fine_part_of_speech_reader, conll_09_reader, zgen_reader,
    arcs_conll_reader])

dependency_label_types = {
    "core": "ccomp csubj csubjpass dobj iobj nsubj nsubjpass xcomp".split(),
    "non_core": """acl discourse nmod advcl dislocated nummod advmod expl parataxis amod foreign remnant appos
          goeswith reparandum compound list root -NONE- conj mwe vocative dep name""".split(),
    "func": "aux auxpass case cc cop det mark neg".split(),
    "other": "punct".split()}


def get_reader(name):
    if name.lower() == 'ud':
        return universal_dependency_reader
    elif name.lower() == 'zgen':
        return zgen_reader
    elif name.lower() == 'conll09':
        return conll_09_reader
    elif name.lower() == 'ud_fine_pos':
        return universal_dependency_fine_part_of_speech_reader
    elif name.lower() in all_readers:
        return all_readers[name.lower()]
    else:
        raise ValueError('Unknown reader: {}'.format(name))
