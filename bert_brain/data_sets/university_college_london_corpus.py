import csv
import os
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat

from .corpus_base import CorpusBase, CorpusExampleUnifier
from .input_features import RawData, KindData, ResponseKind


__all__ = ['UclCorpus']


class UclCorpus(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(
            frank_2013_eye_path='frank_2013_eye_path',
            frank_2015_erp_path='frank_2015_erp_path')

    def __init__(
            self, frank_2013_eye_path=None, frank_2015_erp_path=None,
            subtract_erp_baseline=False, include_erp=True, include_eye=True, self_paced_inclusion='all'):
        self.frank_2013_eye_path = frank_2013_eye_path
        self.frank_2015_erp_path = frank_2015_erp_path
        self.subtract_erp_baseline = subtract_erp_baseline
        self.include_erp = include_erp
        self.include_eye = include_eye
        self.self_paced_inclusion = self_paced_inclusion

    def _load(self, run_info, example_manager: CorpusExampleUnifier):
        data = OrderedDict()
        if self.include_erp:
            erp = self._read_erp(example_manager)
            for k in erp:
                data[k] = KindData(ResponseKind.ucl_erp, erp[k])
        # erp / eye have the same sentences, so if self_paced_inclusion is eye, we can also use erp
        if self.include_eye or (self.self_paced_inclusion == 'eye' and not self.include_erp):
            eye_subjects = self._read_eye_subject_ids()
            eye_keys = OrderedDict([
                ('first_fixation', 'RTfirstfix'),
                ('first_pass', 'RTfirstpass'),
                ('right_bounded', 'RTrightbound'),
                ('go_past', 'RTgopast')])
            eye = self._read_2013_data('eyetracking.RT.txt', example_manager, eye_subjects, eye_keys)
            if self.include_eye:
                for k in eye:
                    data[k] = KindData(ResponseKind.ucl_eye, eye[k])
            else:
                example_manager.remove_data_keys(eye_keys)
        if self.self_paced_inclusion == 'all' or self.self_paced_inclusion == 'eye':
            self_paced_subjects = self._read_self_paced_subject_ids()
            self_paced_keys = OrderedDict([('reading_time', 'RT')])
            self_paced = self._read_2013_data(
                'selfpacedreading.RT.txt', example_manager, self_paced_subjects, self_paced_keys,
                self.self_paced_inclusion == 'all')
            for k in self_paced:
                data[k] = KindData(ResponseKind.ucl_self_paced, self_paced[k])
        elif self.self_paced_inclusion != 'none':
            raise ValueError('Unexpected value for self_paced_inclusion: {}'.format(self.self_paced_inclusion))

        examples = list(example_manager.iterate_examples(fill_data_keys=True))

        for k in data:
            data[k].data.setflags(write=False)

        return RawData(examples, data, test_proportion=0., validation_proportion_of_train=0.1)

    def _read_erp(self, example_manager: CorpusExampleUnifier):
        data = loadmat(self.frank_2015_erp_path)
        sentences = data['sentences'].squeeze(axis=1)
        erp = data['ERP'].squeeze(axis=1)
        erp_base = data['ERPbase'].squeeze(axis=1) if self.subtract_erp_baseline else None

        rows = list()
        base_rows = list()

        iterable = zip(sentences, erp, erp_base) if erp_base is not None else zip(sentences, erp)

        erp_names = ['elan', 'lan', 'n400', 'epnp', 'p600', 'pnp']

        for i, item in enumerate(iterable):
            if erp_base is not None:
                s, e, e_base = item
            else:
                # noinspection PyTupleAssignmentBalance
                s, e = item
                e_base = None
            s = s.squeeze(axis=0)

            example_manager.add_example(
                example_key=None,
                words=[str(w[0]) for w in s],
                sentence_ids=[i] * len(s),
                data_key=erp_names,
                data_ids=len(rows) + np.arange(len(s)))

            for w_e in e:
                rows.append(np.expand_dims(np.asarray(w_e, dtype=np.float32), 0))

            if e_base is not None:
                for w_base in e_base:
                    base_rows.append(np.expand_dims(np.asarray(w_base, dtype=np.float32), 0))

        erp = np.concatenate(rows, axis=0)
        if self.subtract_erp_baseline:
            erp_base = np.concatenate(base_rows, axis=0)
            erp = erp - erp_base
        assert(len(erp_names) == erp.shape[2])
        return OrderedDict(
            [(n, np.squeeze(v, axis=2)) for n, v in zip(erp_names, np.split(erp, erp.shape[2], axis=2))])

    def _read_eye_subject_ids(self):
        subject_ids = set()
        with open(os.path.join(self.frank_2013_eye_path, 'eyetracking.subj.txt'), 'rt', newline='') as subject_file:
            for record in csv.DictReader(subject_file, delimiter='\t'):
                if record['age'] == 'NA':
                    continue
                is_monolingual = int(record['monoling']) == 1
                if not is_monolingual:
                    continue
                if int(record['age_en']) > 0:
                    continue
                subject_ids.add(int(record['subj_nr']))
        return subject_ids

    def _read_self_paced_subject_ids(self):
        subject_ids = set()
        with open(
                os.path.join(self.frank_2013_eye_path, 'selfpacedreading.subj.txt'), 'rt', newline='') as subject_file:
            for record in csv.DictReader(subject_file, delimiter='\t'):
                if int(record['age_en']) > 0:
                    continue
                subject_ids.add(int(record['subj_nr']))
        return subject_ids

    def _read_2013_data(
            self, file_name, example_manager: CorpusExampleUnifier, subject_ids, data_keys, allow_new_examples=True):
        sentence_words = dict()
        data = dict()
        for k in data_keys:
            data[k] = dict()
        seen_subject_ids = set()

        with open(os.path.join(self.frank_2013_eye_path, file_name), 'rt', newline='') as file:
            for record in csv.DictReader(file, delimiter='\t'):
                subject_id = int(record['subj_nr'])
                sentence_id = int(record['sent_nr'])
                word_position = int(record['word_pos'])
                word = record['word']

                if subject_id not in subject_ids:
                    continue
                seen_subject_ids.add(subject_id)
                if sentence_id not in sentence_words:
                    sentence_words[sentence_id] = dict()
                    for k in data_keys:
                        data[k][sentence_id] = dict()
                if word_position not in sentence_words[sentence_id]:
                    sentence_words[sentence_id][word_position] = word
                    for k in data_keys:
                        data[k][sentence_id][word_position] = dict()
                for k in data_keys:
                    data[k][sentence_id][word_position][subject_id] = float(record[data_keys[k]])

        seen_subject_ids = list(sorted(seen_subject_ids))
        num_words = sum(len(sentence_words[sentence_id]) for sentence_id in sentence_words)

        final_data = OrderedDict((k, np.full((num_words, len(seen_subject_ids)), np.nan)) for k in data_keys)

        data_offset = 0
        for sentence_id in sorted(sentence_words):
            sorted_pos = sorted(sentence_words[sentence_id])
            words = [sentence_words[sentence_id][p] for p in sorted_pos]
            example_manager.add_example(
                example_key=None,
                words=words,
                sentence_ids=[sentence_id] * len(words),
                data_key=[k for k in data_keys],
                data_ids=np.arange(len(words)) + data_offset,
                allow_new_examples=allow_new_examples)
            for k in data_keys:
                for i, p in enumerate(sorted_pos):
                    current_data = data[k][sentence_id][p]
                    for j, subject_id in enumerate(seen_subject_ids):
                        if subject_id in current_data:
                            final_data[k][i + data_offset, j] = current_data[subject_id]
            data_offset += len(sentence_words[sentence_id])

        return final_data
