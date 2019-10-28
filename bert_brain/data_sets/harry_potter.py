import os
from collections import OrderedDict
from itertools import combinations
from dataclasses import dataclass, replace as dataclass_replace
from typing import Mapping, Sequence, Optional, Union
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from scipy.ndimage.filters import gaussian_filter

import nibabel
import cortex

from ..common import MultiReplace
from .corpus_base import CorpusBase, CorpusExampleUnifier
from .fmri_example_builders import FMRICombinedSentenceExamples, FMRIExample, PairFMRIExample
from .input_features import RawData, KindData, ResponseKind


__all__ = ['HarryPotterCorpus', 'read_harry_potter_story_features', 'harry_potter_leave_out_fmri_run',
           'HarryPotterMakeLeaveOutFmriRun', 'get_indices_from_normalized_coordinates', 'get_mask_for_subject']


@dataclass
class _HarryPotterWordFMRI:
    word: str
    index_in_all_words: int
    index_in_sentence: int
    sentence_id: int
    time: float
    run: int
    story_features: Mapping


@dataclass(frozen=True)
class _MEGKindProperties:
    file_name: str
    is_preprocessed: bool = False


class HarryPotterCorpus(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='harry_potter_path')

    # we need to use these so we can have run information even when we
    # don't read in fmri data; i.e. as a way to do train-test splits the
    # same for MEG as for fMRI. We will assert that the run lengths we
    # get are equal to these when we read fMRI
    static_run_lengths = (340, 352, 279, 380)

    def __init__(
            self,
            path: Optional[str] = None,
            meg_subjects: Optional[Sequence[str]] = None,
            meg_kind: str = 'pca',
            separate_meg_axes: Optional[Union[str, Sequence[str]]] = None,
            group_meg_sentences_like_fmri: bool = False,
            fmri_subjects: Optional[Sequence[str]] = None,
            fmri_smooth_factor: Optional[float] = 1.,
            fmri_skip_start_trs: int = 20,
            fmri_skip_end_trs: int = 15,
            fmri_window_duration: float = 8.,
            fmri_minimum_duration_required: float = 7.8,
            fmri_sentence_mode: str = 'multiple'):
        """
        Loader for Harry Potter data
        Args:
            path: The path to the directory where the data is stored
            meg_subjects: Which subjects' data to load for MEG. None will cause all subjects' data to load. An
                empty list can be provided to cause MEG loading to be skipped.
            meg_kind: One of ('pca_label', 'mean_label', 'pca_sensor')
                pca_label is source localized and uses the pca within an ROI label for the label with
                    100ms slices of time
                mean_label is similar to pca_label, but using the mean within an ROI label
                pca_sensor uses a PCA on the sensor space to produce 10 latent components and time is averaged over
                the whole word
            separate_meg_axes: None, True, or one or more of ('subject', 'roi', 'time'). If not provided (the default),
                then when MEG data has not been preprocessed, a dictionary with a single key ('hp_meg') that maps to
                an array of shape (word, subject, roi, 100ms slice) [Note that this shape may be modified by
                preprocessing] is returned. 'subject', 'roi', and 'time' only apply to non-preprocessed data.
                If 'subject' is provided, then the dictionary is keyed by 'hp_meg.<subject-id>', e.g. 'hp_meg.A",
                and the shape of each value is (word, roi, 100ms slice). Similarly each separate axis that is
                provided causes the data array to be further split and each resulting data array can be found under a
                more complex key. The keys are generated in the order <subject-id>.<roi>.<time> all of which are
                optional. When time is in the key, it is a multiple of 100 giving the ms start time of the window. If
                the meg_kind is a preprocessed_kind, then only True is supported for this argument. If True, the
                data is split such that each component gets a separate key. For example, 'hp_meg_A.0', 'hp_meg_A.1', ...
                which is a scalar value for each word.
            group_meg_sentences_like_fmri: If False, examples for MEG are one sentence each. If True, then examples
                are created as they would be for fMRI, i.e. including sentences as required by the
                fmri_window_size_features parameter
            fmri_subjects: Which subjects' data to load for fMRI. None will cause all subjects' data to load. An
                empty list can be provided to cause fMRI loading to be skipped.
            fmri_smooth_factor: The sigma parameter of the gaussian blur function
                applied to blur the fMRI data spatially, or None to skip blurring.
            fmri_skip_start_trs: The number of TRs to remove from the beginning of each fMRI run, since the first few
                TRs can be problematic
            fmri_skip_end_trs: The number of TRs to remove from the end of each fMRI run, since the last few TRs can be
                problematic
            fmri_window_duration: The duration of the window of time preceding a TR from which to
                choose the words that will be involved in predicting that TR. For example, if this is 8, then all words
                which occurred with tr_time > word_time >= tr_time - 8 will be used to build the example for the TR.
            fmri_minimum_duration_required: The minimum duration of the time between the earliest word used to
                predict a TR and the occurrence of the TR. This much time is required for the TR to be a legitimate
                target. For example, if this is set to 7.5, then letting the time of the earliest word occurring in the
                window_duration before the TR be min_word_time, if tr_time - min_word_time <
                minimum_duration_required, the TR is not is not used to build any examples.
            fmri_sentence_mode: One of ['multiple', 'single', 'ignore']. When 'multiple', an example consists of the
                combination of sentences as described above. If 'single', changes the behavior of the function so that
                the feature window is truncated by the start of a sentence, thus resulting in examples with one
                sentence at a time. If 'ignore', then each example consists of exactly the words in the feature window
                without consideration of the sentence boundaries
        """
        self.path = path
        self.fmri_subjects = fmri_subjects
        self.meg_subjects = meg_subjects
        self.meg_kind = meg_kind
        self.separate_meg_axes = separate_meg_axes
        self.group_meg_sentences_like_fmri = group_meg_sentences_like_fmri
        self.fmri_smooth_factor = fmri_smooth_factor
        self.fmri_skip_start_trs = fmri_skip_start_trs
        self.fmri_skip_end_trs = fmri_skip_end_trs
        self.fmri_example_builder = FMRICombinedSentenceExamples(
            window_duration=fmri_window_duration,
            minimum_duration_required=fmri_minimum_duration_required,
            use_word_unit_durations=False,  # since the word-spacing is constant in Harry Potter, not needed
            sentence_mode=fmri_sentence_mode)

    @staticmethod
    def _add_fmri_example(
            example, example_manager: CorpusExampleUnifier, data_keys=None, data_ids=None,
            is_apply_data_id_to_entire_group=False, allow_new_examples=True, words_override=None):
        key = tuple(w.index_in_all_words for w in example.words)
        if isinstance(example, PairFMRIExample):
            len_1 = example.len_1
            start_2 = example.second_offset
            stop_2 = len(example.words) - len_1 + start_2
        else:
            len_1 = len(example.words)
            start_2 = None
            stop_2 = None
        features = example_manager.add_example(
            key,
            words_override if words_override is not None else [w.word for w in example.full_sentences],
            [w.sentence_id for w in example.full_sentences],
            data_keys,
            data_ids,
            start=example.offset,
            stop=example.offset + len_1,
            start_sequence_2=start_2,
            stop_sequence_2=stop_2,
            is_apply_data_id_to_entire_group=is_apply_data_id_to_entire_group,
            allow_new_examples=allow_new_examples)

        assert (all(w.run == example.words[0].run for w in example.words[1:]))
        return features

    def story_features_per_fmri_example(self, paths_obj):
        if paths_obj is not None:
            self.set_paths_from_path_object(path_obj=paths_obj)
        else:
            self.check_paths()
        fmri_examples = self._compute_examples_for_fmri()
        unique_ids = dict()
        words = dict()
        for example in fmri_examples:
            key = tuple(w.index_in_all_words for w in example.words)
            if key not in unique_ids:
                unique_ids[key] = len(unique_ids)
            words[unique_ids[key]] = example.words
        return words

    @staticmethod
    def _meg_kind_properties(meg_kind):
        kind_properties = {
            'pca_label': _MEGKindProperties('harry_potter_meg_100ms_pca.npz'),
            'mean_label': _MEGKindProperties('harry_potter_meg_100ms_mean_flip.npz'),
            'pca_sensor': _MEGKindProperties('harry_potter_meg_sensor_pca_35_word_mean.npz'),
            'pca_sensor_full': _MEGKindProperties('harry_potter_meg_sensor_pca_35_word_full.npz'),
            'ica_sensor_full': _MEGKindProperties('harry_potter_meg_sensor_ica_35_word_full.npz'),
            'leila': _MEGKindProperties('harry_potter_meg_sensor_25ms_leila.npz'),
            'rank_clustered_kmeans_L2_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_kmeans_L2_A.npz', is_preprocessed=True),
            'rank_clustered_kmeans': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_kmeans_L2.npz', is_preprocessed=True),
            'rank_clustered_L2': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_L2.npz', is_preprocessed=True),
            'rank_clustered_median': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_median.npz', is_preprocessed=True),
            'rank_clustered_mean': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_mean.npz', is_preprocessed=True),
            'rank_clustered_rms': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_rms.npz', is_preprocessed=True),
            'rank_clustered_counts': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_counts.npz', is_preprocessed=True),
            'rank_clustered_mean_time_slice_ms_100_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_mean_time_slice_ms_100_A.npz', is_preprocessed=True),
            'rank_clustered_mean_whole_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_mean_whole_A.npz', is_preprocessed=True),
            'rank_clustered_sum_time_slice_ms_100_A': _MEGKindProperties(
                'harry_potter_meg_rank_clustered_sum_time_slice_ms_100_A.npz', is_preprocessed=True),
            'direct_rank_clustered_sum_25_ms': _MEGKindProperties(
                'harry_potter_meg_direct_rank_clustered_sum_25.npz', is_preprocessed=True),
        }

        if meg_kind not in kind_properties:
            raise ValueError('Unknown meg_kind: {}'.format(meg_kind))

        return kind_properties[meg_kind]

    @property
    def meg_path(self):
        return os.path.join(self.path, HarryPotterCorpus._meg_kind_properties(self.meg_kind).file_name)

    def _run_info(self, index_run):
        if self.meg_kind in ('rank_clustered',):
            return index_run % 4
        return -1

    def _load(self, run_info, example_manager: CorpusExampleUnifier):

        data = OrderedDict()

        run_at_unique_id = list()

        fmri_examples = None
        if (((self.meg_subjects is None or len(self.meg_subjects) > 0) and self.group_meg_sentences_like_fmri)
                or self.fmri_subjects is None  # None means all subjects
                or len(self.fmri_subjects)) > 0:
            fmri_examples = self._compute_examples_for_fmri()
            for example in fmri_examples:
                features = HarryPotterCorpus._add_fmri_example(example, example_manager)
                assert(features.unique_id == len(run_at_unique_id))
                run_at_unique_id.append(example.words[0].run)
        meg_examples = None
        if self.meg_subjects is None or len(self.meg_subjects) > 0:
            if not self.group_meg_sentences_like_fmri:
                # add all of the sentences first to guarantee consistent example ids
                sentences, _ = self._harry_potter_fmri_word_info(HarryPotterCorpus.static_run_lengths)
                meg_examples = list()
                for sentence_id, sentence in enumerate(sentences):
                    key = tuple(w.index_in_all_words for w in sentence)
                    features = example_manager.add_example(
                        key,
                        [w.word for w in sentence],
                        [w.sentence_id for w in sentence],
                        None,
                        None)
                    assert(all(w.run == sentence[0].run for w in sentence[1:]))
                    assert(features.unique_id <= len(run_at_unique_id))
                    if features.unique_id < len(run_at_unique_id):
                        assert(run_at_unique_id[features.unique_id] == sentence[0].run)
                    else:
                        run_at_unique_id.append(sentence[0].run)
                    # tr_target is not going to be used, so we don't actually build it out
                    meg_examples.append(FMRIExample(sentence, [sentence_id] * len(sentence), [], sentence, 0))
            else:
                meg_examples = fmri_examples

        run_at_unique_id = np.array(run_at_unique_id)
        run_at_unique_id.setflags(write=False)

        metadata = dict(fmri_runs=run_at_unique_id)

        indicator_validation = None
        if self.meg_subjects is None or len(self.meg_subjects) > 0:
            meg, block_metadata, indicator_validation = self._read_meg(run_info, example_manager, meg_examples)
            for k in meg:
                data[k] = KindData(ResponseKind.hp_meg, meg[k])
            if block_metadata is not None:
                metadata['meg_blocks'] = block_metadata
        if self.fmri_subjects is None or len(self.fmri_subjects) > 0:
            fmri = self._read_fmri(run_info, example_manager, fmri_examples)
            for k in fmri:
                data[k] = KindData(ResponseKind.hp_fmri, fmri[k])

        for k in data:
            data[k].data.setflags(write=False)

        input_examples = list(example_manager.iterate_examples(fill_data_keys=True))

        if indicator_validation is not None:
            train = list()
            validation = list()
            for is_validation, ex in zip(indicator_validation, input_examples):
                if is_validation:
                    validation.append(ex)
                else:
                    train.append(ex)
            return RawData(
                input_examples=train,
                validation_input_examples=validation, response_data=data, is_pre_split=True, metadata=metadata)

        return RawData(
            input_examples,
            response_data=data, metadata=metadata, validation_proportion_of_train=0.1, test_proportion=0.)

    def _read_preprocessed_meg(self, run_info, example_manager: CorpusExampleUnifier, examples):
        # see make_harry_potter.ipynb for how these are constructed

        with np.load(self.meg_path, allow_pickle=True) as loaded:
            blocks = loaded['blocks']
            stimuli = loaded['stimuli']
            assert (stimuli[2364] == '..."')
            stimuli[2364] = '...."'  # this was an ellipsis followed by a ., but the period got dropped somehow

            held_out_block = np.unique(blocks)[run_info]
            not_fixation = np.logical_not(stimuli == '+')
            new_indices = np.full(len(not_fixation), -1, dtype=np.int64)
            new_indices[not_fixation] = np.arange(np.count_nonzero(not_fixation))
            stimuli = stimuli[not_fixation]
            blocks = blocks[not_fixation]

            subjects = loaded['subjects']

            if self.meg_subjects is not None:
                indicator_subjects = np.array([s in self.meg_subjects for s in subjects])
                # noinspection PyTypeChecker
                subjects = subjects[indicator_subjects]

            data = OrderedDict()
            for subject in subjects:
                data['hp_meg_{}'.format(subject)] = \
                    loaded['data_{}_hold_out_{}'.format(subject, held_out_block)][not_fixation]

            if 'data_multi_subject_hold_out_{}'.format(held_out_block) in loaded \
                    and (self.meg_subjects is None or 'multi_subject' in self.meg_subjects):
                data['hp_meg_multi_subject'] = \
                    loaded['data_multi_subject_hold_out_{}'.format(held_out_block)][not_fixation]

        if self.separate_meg_axes is not None:
            if not isinstance(self.separate_meg_axes, bool):
                raise ValueError('Only boolean values are supported for \'separate_meg_axes\' '
                                 'when the meg_kind is preprocessed')
            if self.separate_meg_axes:
                separated_data = OrderedDict()
                for k in data:
                    assert(len(data[k].shape) == 2)
                    for index_task, item in enumerate(np.split(data[k], data[k].shape[1], axis=1)):
                        separated_data['{}.{}'.format(k, index_task)] = item
                data = separated_data

        block_metadata = np.full(len(example_manager), -1, dtype=np.int64)

        for example in examples:
            indices = np.array([w.index_in_all_words for w in example.full_sentences])
            # use these instead of the words given in example to ensure our indexing is not off
            # we will fail an assert below if we try to add something messed up
            example_stimuli = stimuli[new_indices[indices]]

            features = HarryPotterCorpus._add_fmri_example(
                example,
                example_manager,
                words_override=[_clean_word(w) for w in example_stimuli],
                data_keys=[k for k in data],
                data_ids=new_indices[indices],
                allow_new_examples=False)
            assert (features is not None)

            example_blocks = blocks[new_indices[indices]]
            assert (np.all(example_blocks == example_blocks[0]))
            block_metadata[features.unique_id] = example_blocks[0]

        return data, block_metadata, block_metadata == held_out_block

    def _read_meg(self, run_info, example_manager: CorpusExampleUnifier, examples):

        if HarryPotterCorpus._meg_kind_properties(self.meg_kind).is_preprocessed:
            return self._read_preprocessed_meg(run_info, example_manager, examples)

        # separate_task_axes should be a tuple of strings in 'roi', 'subject', 'time'
        separate_task_axes = self.separate_meg_axes
        if separate_task_axes is None:
            separate_task_axes = []
        elif isinstance(separate_task_axes, str):
            separate_task_axes = [separate_task_axes]
        for axis in separate_task_axes:
            if axis not in ['roi', 'subject', 'time']:
                raise ValueError('Unknown separate_task_axis: {}'.format(axis))

        with np.load(self.meg_path, allow_pickle=True) as loaded:

            stimuli = loaded['stimuli']

            assert(stimuli[2364] == '..."')
            stimuli[2364] = '...."'  # this was an elipsis followed by a ., but the period got dropped somehow

            # blocks should be int, but is stored as float
            blocks = loaded['blocks'] if 'blocks' in loaded else None
            blocks = np.round(blocks).astype(np.int64) if blocks is not None else None
            # (subjects, words, rois, 100ms_slices)
            data = loaded['data']
            rois = loaded['rois'] if 'rois' in loaded else None
            subjects = loaded['subjects']

        if self.meg_subjects is not None:
            indicator_subjects = np.array([s in self.meg_subjects for s in subjects])
            # noinspection PyTypeChecker
            subjects = subjects[indicator_subjects]
            # noinspection PyTypeChecker
            data = data[indicator_subjects]

        times = None
        if len(data.shape) == 4:
            if data.shape[-1] == 500:
                data = np.reshape(data, data.shape[:-1] + (data.shape[-1] // 100, 100))
                data = np.mean(data, axis=-1)
            elif data.shape[-1] == 5:
                times = np.arange(data.shape[-1]) * 100
            elif data.shape[-1] == 20:
                times = np.arange(data.shape[-1]) * 25

        if len(data.shape) == 4:
            # -> (words, subjects, rois, 100ms_slices)
            data = np.transpose(data, axes=(1, 0, 2, 3))
        else:
            # -> (words, subjects, space)
            data = np.transpose(data, axes=(1, 0, 2))

        if rois is not None:
            assert (len(rois) == data.shape[2])

        indicator_plus = stimuli == '+'

        # I used to make longer passages that went between pluses. Leaving the code here
        # as a comment in case we want to use that again in the future
        # if examples_between_plus:
        #     indicator_block = np.concatenate([np.full(1, True), np.diff(blocks) > 0])
        #     example_id_words = np.zeros(len(stimuli), dtype=np.int64)
        #     indices_plus = np.where(np.logical_or(indicator_plus, indicator_block))[0]
        #     if indices_plus[0] == 0:
        #         indices_plus = indices_plus[1:]
        #     last = 0
        #     current_passage = 0
        #     for i in indices_plus:
        #         example_id_words[last:i] = current_passage
        #         last = i
        #         current_passage += 1
        #     example_id_words[last:] = current_passage
        # else:  # use sentences as examples

        new_indices = -1 * np.ones(len(data), dtype=np.int64)

        not_plus = np.logical_not(indicator_plus)
        data = data[not_plus]
        stimuli = stimuli[not_plus]
        if blocks is not None:
            blocks = blocks[not_plus]

        new_indices[not_plus] = np.arange(len(data))

        # example_id_words = example_id_words[not_plus]

        # data is (words, subjects, rois, 100ms_slices)
        data = OrderedDict([('hp_meg', data)])

        subject_axis = 1
        roi_axis = 2
        time_axis = 3
        if 'subject' in separate_task_axes:
            separated_data = OrderedDict()
            for k in data:
                assert (len(subjects) == data[k].shape[subject_axis])
                for idx_subject, subject in enumerate(subjects):
                    k_new = k + '.{}'.format(subject)
                    separated_data[k_new] = np.take(data[k], idx_subject, axis=subject_axis)
            data = separated_data
            roi_axis -= 1
            time_axis -= 1
        if 'roi' in separate_task_axes:
            if rois is None:
                raise ValueError('Cannot separate roi axis. rois does not exist.')
            separated_data = OrderedDict()
            for k in data:
                assert (len(rois) == data[k].shape[roi_axis])
                for idx_roi, roi in enumerate(rois):
                    k_new = k + '.{}'.format(roi)
                    separated_data[k_new] = np.take(data[k], idx_roi, axis=roi_axis)
            data = separated_data
            time_axis -= 1
        if 'time' in separate_task_axes:
            if times is None:
                raise ValueError('Cannot separate time axis. times does not exist')
            separated_data = OrderedDict()
            for k in data:
                assert (len(times) == data[k].shape[time_axis])
                for idx_time, window in enumerate(times):
                    k_new = k + '.{}'.format(window)
                    separated_data[k_new] = np.take(data[k], idx_time, axis=time_axis)
            data = separated_data

        block_metadata = np.full(len(example_manager), -1, dtype=np.int64) if blocks is not None else None

        for example in examples:
            indices = np.array([w.index_in_all_words for w in example.full_sentences])
            # use these instead of the words given in example to ensure our indexing is not off
            # we will fail an assert below if we try to add something messed up
            example_stimuli = stimuli[new_indices[indices]]

            features = HarryPotterCorpus._add_fmri_example(
                example,
                example_manager,
                words_override=[_clean_word(w) for w in example_stimuli],
                data_keys=[k for k in data],
                data_ids=new_indices[indices],
                allow_new_examples=False)
            assert (features is not None)

            if blocks is not None:
                example_blocks = blocks[new_indices[indices]]
                assert(np.all(example_blocks == example_blocks[0]))
                block_metadata[features.unique_id] = example_blocks[0]

        return data, block_metadata, None

    def _read_harry_potter_fmri_files(self):
        # noinspection PyPep8
        subject_runs = dict(
            F=[4, 5, 6, 7],
            G=[3, 4, 5, 6],
            H=[3, 4, 9, 10],
            I=[7, 8, 9, 10],
            J=[7, 8, 9, 10],
            K=[7, 8, 9, 10],
            L=[7, 8, 9, 10],
            M=[7, 8, 9, 10],
            N=[7, 8, 9, 10])

        subjects = self.fmri_subjects
        if subjects is None:
            subjects = list(subject_runs.keys())

        if isinstance(subjects, str):
            subjects = [subjects]

        path_fmt = os.path.join(self.path, 'fmri', '{subject}', 'funct', '{run}', 'ars{run:03}a001.hdr')

        all_subject_data = OrderedDict()
        masks = OrderedDict()

        for subject in subjects:
            if subject not in subject_runs:
                raise ValueError('Unknown subject: {}. Known values are: {}'.format(subject, list(subject_runs.keys())))
            subject_data = list()
            for run in subject_runs[subject]:
                functional_file = path_fmt.format(subject=subject, run=run)
                data = nibabel.load(functional_file).get_data()
                subject_data.append(np.transpose(data))

            masks[subject] = get_mask_for_subject(subject)
            all_subject_data[subject] = subject_data

        return all_subject_data, masks

    def _harry_potter_fmri_word_info(self, run_lengths):

        time_images = np.arange(1351) * 2
        words = np.load(os.path.join(self.path, 'words_fmri.npy'), allow_pickle=True)
        words = [w.item() for w in words]
        time_words = np.load(os.path.join(self.path, 'time_words_fmri.npy'), allow_pickle=True)
        assert (len(words) == len(time_words))

        # searchsorted returns first location such that time_words[i] < time_images[word_images[i]]
        word_images = np.searchsorted(time_images, time_words, side='right') - 1

        story_features = read_harry_potter_story_features(os.path.join(self.path, 'story_features.mat'))
        indices_in_sentences = story_features[('Word_Num', 'sentence_length')]
        assert (len(indices_in_sentences) == len(words))

        run_ids = np.concatenate([[idx] * run_lengths[idx] for idx in range(len(run_lengths))])
        assert (len(run_ids) == len(time_images))

        words = [
            _HarryPotterWordFMRI(
                word=_clean_word(words[i]),
                index_in_all_words=i,
                index_in_sentence=indices_in_sentences[i],
                sentence_id=-1,  # we will fix this below
                time=time_words[i],
                run=run_ids[word_images[i]],
                story_features=dict((k, story_features[k][i]) for k in story_features)) for i in range(len(words))]

        sentences = list()
        for sentence_id, sentence in enumerate(_group_sentences(words, index_fn=lambda w: w.index_in_sentence)):
            has_plus = False
            for idx_w, word in enumerate(sentence):
                word.sentence_id = sentence_id
                if word.word == '+':
                    assert (idx_w == 0)  # assert no natural pluses
                    has_plus = True
            if has_plus:
                sentence = sentence[1:]
            if len(sentence) > 0:
                sentences.append(sentence)

        # split this by run so its easier for the caller to work with
        time_images = np.split(time_images, np.cumsum(run_lengths)[:-1])

        assert(all([len(time_images[i]) == run_lengths[i] for i in range(len(run_lengths))]))

        return sentences, time_images

    def _compute_examples_for_fmri(self):
        # get the words, image indices, and story features per sentence
        sentences, time_images_per_run = self._harry_potter_fmri_word_info(HarryPotterCorpus.static_run_lengths)

        # split up the sentences by run
        run_words = OrderedDict()  # use ordered dict to keep the runs in order
        for sentence_id, sentence in enumerate(sentences):
            assert (all(w.run == sentence[0].run for w in sentence))  # assert no sentence spans more than 1 run
            if sentence[0].run not in run_words:
                run_words[sentence[0].run] = list()
            run_words[sentence[0].run].extend(sentence)

        # get the tuple of sentences required by the window for each TR
        examples = list()
        tr_offset = 0
        for run, time_images in zip(run_words, time_images_per_run):
            # drop the first 20 and last 15 images of each run since these can have issues
            assert (len(time_images) > self.fmri_skip_start_trs + self.fmri_skip_end_trs)
            offset_increment = len(time_images)
            time_images = time_images[self.fmri_skip_start_trs:(len(time_images) - self.fmri_skip_end_trs)]
            run_examples = self.fmri_example_builder(
                run_words[run],
                [w.time for w in run_words[run]],
                [w.sentence_id for w in run_words[run]],
                time_images,
                tr_offset=tr_offset + self.fmri_skip_start_trs)

            examples.extend(run_examples)
            tr_offset += offset_increment
        return examples

    def _read_fmri(self, run_info, example_manager: CorpusExampleUnifier, fmri_examples):

        data, spatial_masks = self._read_harry_potter_fmri_files()

        # we assume that the runs are the same across subjects below. assert it here
        run_lengths = None
        for subject in data:
            if run_lengths is None:
                run_lengths = [len(r) for r in data[subject]]
            else:
                assert (np.array_equal([len(r) for r in data[subject]], run_lengths))

        assert(np.array_equal(run_lengths, HarryPotterCorpus.static_run_lengths))

        # apply spatial smoothing before masking
        for subject in data:
            subject_data = data[subject]
            if self.fmri_smooth_factor is not None:
                for idx in range(len(subject_data)):
                    # for each example, apply a gaussian filter spatially
                    for ax_idx in range(len(subject_data[idx])):
                        subject_data[idx][ax_idx] = gaussian_filter(
                            subject_data[idx][ax_idx],
                            sigma=self.fmri_smooth_factor, order=0, mode='reflect', truncate=4.0)
            # apply spatial mask
            data[subject] = [d[:, spatial_masks[subject]] for d in subject_data]

        active_image_indices = list()

        def _replace_tr(ex_, active_list_):
            data_ids = [-1] * len(ex_.full_sentences)
            for i, t in enumerate(ex_.tr_target):
                if t is not None:
                    # keep only the first target if there are multiple
                    data_ids[i + ex_.offset] = len(active_list_)
                    active_list_.append(t[0])
            return dataclass_replace(ex_, tr_target=data_ids)

        # filter unused images
        local_examples = [_replace_tr(ex, active_image_indices) for ex in fmri_examples]

        active_image_indices = np.array(active_image_indices)

        masked_data = OrderedDict()
        for subject in data:
            masked_data['hp_fmri_{}'.format(subject)] = np.concatenate(data[subject])[active_image_indices]
        data = masked_data

        for subject in data:
            # add a subject axis as axis 1 since downstream preprocessors expect it (they handle multi-subject data)
            data[subject] = np.expand_dims(data[subject], axis=1)

        for example in local_examples:
            features = HarryPotterCorpus._add_fmri_example(
                example,
                example_manager,
                data_keys=[k for k in data],
                data_ids=example.tr_target,
                is_apply_data_id_to_entire_group=True,
                allow_new_examples=False)

            assert(features is not None)

        return data


def read_harry_potter_story_features(path):
    mat = loadmat(path)
    features = np.squeeze(mat['features'], axis=0)
    result = dict()
    for index_feature_type, feature_type in enumerate(features['type']):
        feature_type = np.squeeze(feature_type, axis=0)
        if feature_type.dtype.type != np.str_:
            assert(len(feature_type) == 0)
            feature_type = None
        else:
            feature_type = str(feature_type)

        if feature_type is None:
            continue  # there are some empty ones which also have no values; not sure why

        feature_names = np.squeeze(features['names'][index_feature_type], axis=0)
        if feature_names.dtype.type == np.object_:
            feature_names = [str(np.squeeze(s, axis=0)) for s in feature_names]
        elif feature_names.dtype.type != np.str_:
            assert(len(feature_names) == 0)
            feature_names = None
        else:
            feature_names = [str(feature_names)]

        feature_values = features['values'][index_feature_type]
        if feature_names is not None:
            if len(feature_names) != feature_values.shape[1]:
                if feature_names[0] == 'word_length':
                    feature_names = feature_names[:-1]
            assert(len(feature_names) == feature_values.shape[1])
            feature_values = [np.squeeze(x, axis=1) for x in np.split(feature_values, len(feature_names), axis=1)]
            for feature_name, feature_value in zip(feature_names, feature_values):
                result[(feature_type, feature_name)] = feature_value
        else:
            result[feature_type] = feature_values
    return result


def _group_sentences(to_group, index_fn=None):

    if index_fn is None:
        def _identity(x):
            return x
        index_fn = _identity

    current = list()
    for item in to_group:
        index_in_sentence = index_fn(item)
        if len(current) > 0:
            if index_in_sentence <= index_fn(current[-1]):
                yield current
                current = list()
        current.append(item)
    if len(current) > 0:
        yield current


_replacer = MultiReplace({
        '@': '',
        '\\': '',
        '…': '...',
        '‘': '\'',
        '—': '--',
        '^': ''  # this just seems like spurious character in context; maybe it was supposed to be @
    })


def _clean_word(w):
    return _replacer.replace(w)


class HarryPotterMakeLeaveOutFmriRun:

    def __init__(self, shuffle=True, make_test=False):
        self.shuffle = shuffle
        self.make_test = make_test

    def __call__(self, index_variation_run):
        return partial(
            harry_potter_leave_out_fmri_run,
            index_variation_run=index_variation_run,
            shuffle=self.shuffle,
            make_test=self.make_test)


def harry_potter_leave_out_fmri_run(raw_data, index_variation_run, random_state=None, shuffle=True, make_test=False):
    if raw_data.is_pre_split:
        raise ValueError(
            'Misconfiguration. The data has already been split, but harry_potter_leave_out_fmri_run is active')

    runs = raw_data.metadata['fmri_runs']
    unique_runs = np.unique(runs)
    if make_test:
        folds = list(combinations(unique_runs, 2))
        index_validation, index_test = folds[index_variation_run % len(folds)]
        if index_variation_run % (len(folds) * 2) >= len(folds):
            index_test, index_validation = index_validation, index_test
        validation_run = unique_runs[index_validation]
        test_run = unique_runs[index_test]
    else:
        test_run = None
        index_validation = index_variation_run % len(unique_runs)
        validation_run = unique_runs[index_validation]

    train_examples = list()
    validation_examples = list()
    test_examples = list()

    for example in raw_data.input_examples:
        if runs[example.unique_id] == validation_run:
            validation_examples.append(example)
        elif runs[example.unique_id] == test_run:
            test_examples.append(example)
        else:
            train_examples.append(example)
    if shuffle:
        if random_state is not None:
            random_state.shuffle(train_examples)
            random_state.shuffle(validation_examples)
            random_state.shuffle(test_examples)
        else:
            np.random.shuffle(train_examples)
            np.random.shuffle(validation_examples)
            np.random.shuffle(test_examples)
    return train_examples, validation_examples, test_examples


def get_mask_for_subject(subject):
    if subject in ['H', 'L', 'K']:
        return cortex.db.get_mask('fMRI_story_{}'.format(subject), '{}_ars_auto2'.format(subject), 'thick')
    return cortex.db.get_mask('fMRI_story_{}'.format(subject), '{}_ars'.format(subject), 'thick')


def get_indices_from_normalized_coordinates(subject, x, y, z, closest_k=None, distance=None):

    mask = get_mask_for_subject(subject)

    max_z, max_y, max_x = mask.shape

    is_single = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)

    x, y, z = np.reshape(np.asarray(x), (-1, 1)), np.reshape(np.asarray(y), (-1, 1)), np.reshape(np.asarray(z), (-1, 1))

    # transform from normalized coordinates to voxel space
    slice_coord = np.concatenate([x * max_x, y * max_y, z * max_z], axis=1)

    mask_z, mask_y, mask_x = np.where(mask)
    mask_coord = np.concatenate(
        [np.expand_dims(mask_x, 1), np.expand_dims(mask_y, 1), np.expand_dims(mask_z, 1)], axis=1)

    distances = cdist(slice_coord, mask_coord)

    if distance is not None:
        distances[distances > distance] = np.nan
    if closest_k is None:
        closest_k = distances.shape[1]
    max_valid = np.max(np.sum(np.logical_not(np.isnan(distances)), axis=1))
    closest_k = min(max_valid, closest_k)

    # nan sorts last
    closest_indices = np.take(np.argsort(distances, axis=1), np.arange(closest_k), axis=1)
    axis_0_indices = np.tile(np.reshape(np.arange(len(closest_indices)), (-1, 1, 1)), (1, closest_indices.shape[1], 1))
    axis_1_compressed_indices = np.tile(np.reshape(np.arange(closest_k), (1, -1, 1)), (closest_indices.shape[0], 1, 1))
    closest_indices = np.reshape(
        np.concatenate([axis_0_indices, np.expand_dims(closest_indices, 2), axis_1_compressed_indices], axis=2),
        (-1, 3))
    axis_0, indices, axis_1 = np.split(closest_indices, 3, axis=1)

    compressed_distances = np.full((distances.shape[0], closest_k), np.nan)
    compressed_indices = np.full((distances.shape[0], closest_k), -1)

    compressed_distances[axis_0, axis_1] = distances[axis_0, indices]
    compressed_indices[axis_0, axis_1] = indices

    compressed_indices[np.isnan(compressed_distances)] = -1

    if is_single:
        compressed_indices = np.squeeze(compressed_indices, axis=0)
        compressed_distances = np.squeeze(compressed_distances, axis=0)

    return compressed_indices, compressed_distances, mask_coord.shape[0]
