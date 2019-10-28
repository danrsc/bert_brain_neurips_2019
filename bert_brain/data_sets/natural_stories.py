import logging
import os
import itertools
import csv
from dataclasses import dataclass
from collections import OrderedDict
import dataclasses
from functools import partial
from typing import Sequence, Optional

import numpy as np
from scipy.io import loadmat

from .corpus_base import CorpusBase, CorpusExampleUnifier
from .fmri_example_builders import FMRICombinedSentenceExamples, FMRIExample
from .spacy_token_meta import make_tokenizer_model
from .input_features import RawData, KindData, ResponseKind
from .praat_textgrid import TextGrid


__all__ = ['read_natural_story_codings', 'NaturalStoriesCorpus', 'natural_stories_leave_stories_out',
           'natural_stories_make_leave_stories_out']


logger = logging.getLogger(__name__)


@dataclass
class _BatchRecord:
    worker_id: str
    work_time_in_seconds: int
    correct: int
    item: int
    zone: int
    reaction_time: int


@dataclass
class _WordRecord:
    word: str
    item: int
    zone: int
    sentence: int


def _read_batch(path):
    with open(path, 'rt', newline='') as f:
        for record in csv.DictReader(f):
            yield _BatchRecord(
                worker_id=record['WorkerId'],
                work_time_in_seconds=int(record['WorkTimeInSeconds']),
                correct=int(record['correct']),
                item=int(record['item']),
                zone=int(record['zone']) - 2,  # subtract 2 to match with zone in stories
                reaction_time=int(record['RT']))


def _read_sentence_ids(directory_path):
    sentence_id = 0
    result = dict()
    with open(os.path.join(directory_path, 'stories-aligned.conllx'), 'rt') as conllx_file:
        for line in conllx_file:
            # ID:     Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens;
            #         may be a decimal number for empty nodes
            #         (decimal numbers can be lower than 1 but must be greater than 0).
            # FORM:   Word form or punctuation symbol.
            # LEMMA:  Lemma or stem of word form.
            # UPOS:   Universal part-of-speech tag.
            # XPOS:   Language-specific part-of-speech tag; underscore if not available.
            # FEATS:  List of morphological features from the universal feature inventory or from a defined
            #         language-specific extension; underscore if not available.
            # HEAD:   Head of the current word, which is either a value of ID or zero (0).
            # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a
            #         defined language-specific subtype of one.
            # DEPS:   Enhanced dependency graph in the form of a list of head-deprel pairs.
            # MISC:   Any other annotation.
            line = line.strip()
            if line.startswith('#'):
                continue
            if len(line) == 0:
                sentence_id += 1
                continue
            id_, form, lemma, upos, xpos, feats, head, deprel, deps, misc = line.split('\t')
            token_id = [int(p) for p in misc[len('TokenId='):].split('.')]
            if len(token_id) > 2:
                item, zone, part = token_id
            else:
                item, zone = token_id
            result[(item, zone)] = sentence_id
    return result


def _sentence_iterator(word_records):
    sentence = list()
    for record in word_records:
        if len(sentence) > 0 and sentence[-1].sentence != record.sentence:
            yield sentence
            sentence = list()
        sentence.append(record)
    if len(sentence) > 0:
        yield sentence


def _read_story_sentences(directory_path):
    sentence_ids = _read_sentence_ids(directory_path)
    story_records = dict()
    with open(os.path.join(directory_path, 'all_stories.tok'), 'rt', newline='') as all_stories_file:
        for record in csv.DictReader(all_stories_file, delimiter='\t'):
            item = int(record['item'])
            zone = int(record['zone'])
            record = _WordRecord(record['word'], item, zone, sentence_ids[(item, zone)])
            if record.item not in story_records:
                story_records[record.item] = list()
            story_records[record.item].append(record)
    for item in sorted(story_records):  # sort so we get consistent unique ids
        story = sorted(story_records[item], key=lambda r: r.zone)
        for sentence in _sentence_iterator(story):
            yield sentence


@dataclass
class _CodingRecord:
    # categorized into groups as in http://www.lrec-conf.org/proceedings/lrec2018/pdf/337.pdf
    # see that link for examples of each

    story_name: str
    sentence_number: int
    sentence: str

    # conjunction
    local_verb_phrase_conjunction: int
    non_local_verb_phrase_conjunction: int
    local_noun_phrase_conjunction: int
    non_local_noun_phrase_conjunction: int
    sentential_co_ordination: int
    complementizer_phrase_conjunctions: int
    adjective_conjunction: int

    # relative clauses
    subject_relative_clause_restrictive: int
    subject_relative_clause_non_restrictive: int
    object_relative_clause_restrictive: int
    object_relative_clause_non_restrictive: int
    object_relative_clause_non_canonical: int
    no_relativizer_object_relative_clause: int
    adverbial_relative_clause: int
    free_relative_clause: int

    # ambiguity
    noun_phrase_sentence_ambiguity: int
    main_verb_reduced_relative_ambiguity_easier: int
    main_verb_reduced_relative_ambiguity_hard: int
    prepositional_phrase_attachment_ambiguity: int

    # displacement
    tough_movement: int
    parenthetical: int
    topicalization: int
    question_wh_subject: int
    question_wh_other: int

    # miscellaneous
    non_local_subject_verb: int
    non_local_verb_direct_object: int
    gerund_modifier: int
    sentential_subject: int
    post_nominal_adjective: int
    idiom: int
    quotation: int
    it_cleft: int
    even_than_construction: int
    if_then_construction: int
    as_as_construction: int
    so_that_construction: int
    question_yn: int
    infinitive_verb_phrase_subject: int
    inanimate_subjects: int
    animacy_hierarchy_violation: int


def _read_coding(path, story_name):

    field_name_map = {
        'sentence_number': 'SentNum',
        'sentence': 'Sentence',

        # conjunction
        'local_verb_phrase_conjunction': 'local VP conjunction',
        'non_local_verb_phrase_conjunction': 'non-local VP conjunction',
        'local_noun_phrase_conjunction': 'local NP conjunction',
        'non_local_noun_phrase_conjunction': 'non-local NP conjunction',
        'sentential_co_ordination': 'S co-ordination',
        'complementizer_phrase_conjunctions': 'CP conjunctions',
        'adjective_conjunction': 'adj conjunction',

        # relative clauses
        'subject_relative_clause_restrictive': 'SRC restr',
        'subject_relative_clause_non_restrictive': 'SRC non-restr',
        'object_relative_clause_restrictive': 'ORC restr',
        'object_relative_clause_non_restrictive': 'ORC non-restr',
        'object_relative_clause_non_canonical': 'ORC non-canon',
        'no_relativizer_object_relative_clause': 'no-relativizer ORC',
        'adverbial_relative_clause': 'adverbial RC',
        'free_relative_clause': 'free relative',

        # ambiguity
        'noun_phrase_sentence_ambiguity': 'NP/S ambig',
        'main_verb_reduced_relative_ambiguity_easier': 'MV/RR ambig EASIER',
        'main_verb_reduced_relative_ambiguity_hard': 'MV/RR ambig HARD',
        'prepositional_phrase_attachment_ambiguity': 'attachment ambig',

        # displacement
        'tough_movement': 'tough mvt',
        'parenthetical': 'parenthetical',
        'topicalization': 'topicalization',
        'question_wh_subject': 'question_wh_subj',
        'question_wh_other': 'question_wh_other',

        # miscellaneous
        'non_local_subject_verb': 'non-local SV',
        'non_local_verb_direct_object': 'non-local verb-DO',
        'gerund_modifier': 'gerund modifier',
        'sentential_subject': 'sent subj',
        'post_nominal_adjective': 'post-nominal adj',
        'idiom': 'idiom',
        'quotation': 'quote',
        'it_cleft': 'it-cleft',
        'even_than_construction': 'even…than',
        'if_then_construction': 'if...then constr',
        'as_as_construction': 'as…as constr',
        'so_that_construction': 'so…that constr',
        'question_yn': 'question_YN',
        'infinitive_verb_phrase_subject': 'inf VP subject',
        'inanimate_subjects': 'inanimate subjects',
        'animacy_hierarchy_violation': 'animacy hierarchy viol'
    }

    fields = dataclasses.fields(_CodingRecord)

    for field in fields:
        if field.name == 'story_name':
            continue
        if field.name not in field_name_map:
            raise ValueError('missing field in field_name_map: {}'.format(field.name))

    records = list()
    with open(path, 'rt', newline='') as coding_file:
        for record in csv.DictReader(coding_file, delimiter='\t'):
            values = dict()
            for field in fields:
                if field.name == 'story_name':
                    values[field.name] = story_name
                    continue
                record_name = field_name_map[field.name]
                str_value = '' if record_name not in record else record[record_name].strip()
                if len(str_value) == 0 and field.type == int:
                    values[field.name] = 0
                else:
                    values[field.name] = field.type(record[record_name])
            records.append(_CodingRecord(**values))
    return records


def _read_codings(directory_path):

    result = dict()
    for coding_file_name in [
            'aqua.txt', 'boar.txt', 'elvis.txt', 'high_school.txt', 'king_of_birds.txt', 'matchstick.txt',
            'mr_sticky.txt', 'roswell.txt', 'tourette.txt', 'tulips.txt']:
        story_name = os.path.splitext(coding_file_name)[0]
        for record in _read_coding(os.path.join(directory_path, coding_file_name), story_name):
            result[record.sentence] = record
    return result


class NaturalStoriesCorpus(CorpusBase):

    @classmethod
    def _path_attributes(cls):
        return dict(path='natural_stories_path')

    def __init__(
            self,
            path: Optional[str] = None,
            include_reaction_times: bool = True,
            group_reaction_time_sentences_like_froi: bool = False,
            froi_subjects: Optional[Sequence[str]] = None,
            froi_skip_start_trs: int = 0,
            froi_skip_end_trs: int = 0,
            froi_window_duration: float = 8.,
            froi_minimum_duration_required: float = 7.8,
            froi_use_word_unit_durations: bool = False,
            froi_sentence_mode: str = 'multiple',
            froi_minimum_story_count: int = 1):
        """

        Args:
            path: The path to the directory where the data is stored
            include_reaction_times: Whether to include self-paced reading times
            group_reaction_time_sentences_like_froi: If False, examples for reaction times are one sentence each.
                If True, then examples are created as they would be for fROIs, i.e. including sentences as required by
                the froi_window_size_features parameter
            froi_subjects: Which subjects' data to load for fROIs. None will cause all subjects' data to load. An
                empty list can be provided to cause fROI loading to be skipped.
            froi_skip_start_trs: The number of TRs to remove from the beginning of each fMRI run, since the first few
                TRs can be problematic
            froi_skip_end_trs: The number of TRs to remove from the end of each fMRI run, since the last few TRs can be
                problematic
            froi_window_duration: The duration of the window of time preceding a TR from which to
                choose the words that will be involved in predicting that TR. For example, if this is 8, then all words
                which occurred with tr_time > word_time >= tr_time - 8 will be used to build the example for the TR.
            froi_minimum_duration_required: The minimum duration of the time between the earliest word used to
                predict a TR and the occurrence of the TR. This much time is required for the TR to be a legitimate
                target. For example, if this is set to 7.5, then letting the time of the earliest word occurring in the
                window_duration before the TR be min_word_time, if tr_time - min_word_time <
                minimum_duration_required, the TR is not is not used to build any examples.
            froi_use_word_unit_durations: If True, then window_duration and minimum_duration_required are in number
                of words rather than time_units. window_duration = 8. would select the 8 previous words.
            froi_sentence_mode: One of ['multiple', 'single', 'ignore']. When 'multiple', an example consists of the
                combination of sentences as described above. If 'single', changes the behavior of the function so that
                the feature window is truncated by the start of a sentence, thus resulting in examples with one
                sentence at a time. If 'ignore', then each example consists of exactly the words in the feature window
                without consideration of the sentence boundaries
            froi_minimum_story_count: A participant must have heard at least this many stories in the scanner to be
                included
        """
        self.path = path
        self.include_reaction_times = include_reaction_times
        self.group_reaction_time_sentences_like_froi = group_reaction_time_sentences_like_froi
        self.froi_subjects = froi_subjects
        self.froi_skip_start_trs = froi_skip_start_trs
        self.froi_skip_end_trs = froi_skip_end_trs
        self.froi_example_builder = FMRICombinedSentenceExamples(
            window_duration=froi_window_duration,
            minimum_duration_required=froi_minimum_duration_required,
            use_word_unit_durations=froi_use_word_unit_durations,
            sentence_mode=froi_sentence_mode)
        self.froi_minimum_story_count = froi_minimum_story_count

    def _load(self, run_info, example_manager: CorpusExampleUnifier):

        data = OrderedDict()
        story_ids = list()

        froi_examples = None
        froi_subject_stories_subjects = None
        froi_subject_stories_stories = None
        if ((self.include_reaction_times and self.group_reaction_time_sentences_like_froi)
                or self.froi_subjects is None  # None means all subjects
                or len(self.froi_subjects)) > 0:

            froi_response_data = None
            item_to_num_images = None
            if self.froi_subjects is None or len(self.froi_subjects) > 0:
                froi_response_data, item_to_num_images, _, froi_subject_stories = self._read_froi_response_data()
                froi_subject_stories_subjects, froi_subject_stories_stories = froi_subject_stories
                for k in froi_response_data:
                    data[k] = KindData(ResponseKind.ns_froi, froi_response_data[k])

            froi_examples = self._compute_examples_froi_audio(item_to_num_images)
            for example in froi_examples:
                key = tuple((w.item, w.zone) for w in example.words)
                features = example_manager.add_example(
                    key,
                    [w.word for w in example.full_sentences],
                    [w.sentence for w in example.full_sentences],
                    None,
                    None,
                    start=example.offset,
                    stop=example.offset + len(example.words))
                assert (features.unique_id == len(story_ids))
                story_ids.append(example.words[0].item)

            if froi_response_data is not None:
                self._add_froi_targets(example_manager, froi_examples, froi_response_data, item_to_num_images)

        if self.include_reaction_times:
            if not self.group_reaction_time_sentences_like_froi:
                # add all of the sentences first to guarantee consistent example ids
                sentences = list(_read_story_sentences(self.path))
                reaction_time_examples = list()
                for sentence in sentences:
                    key = tuple((w.item, w.zone) for w in sentence)
                    features = example_manager.add_example(
                        key,
                        [w.word for w in sentence],
                        [w.sentence for w in sentence],
                        None,
                        None)
                    assert (features.unique_id <= len(story_ids))
                    if features.unique_id < len(story_ids):
                        assert (story_ids[features.unique_id] == sentence[0].item)
                    else:
                        story_ids.append(sentence[0].item)
                    # tr_target is not going to be used, so we don't build it
                    reaction_time_examples.append(
                        FMRIExample(sentence, [w.sentence for w in sentence], [], sentence, 0))
            else:
                reaction_time_examples = froi_examples

            reaction_times = self._read_reaction_times(example_manager, reaction_time_examples)
            for k in reaction_times:
                data[k] = KindData(ResponseKind.ns_reaction_times, reaction_times[k])

        examples = list(example_manager.iterate_examples(fill_data_keys=True))

        story_ids = np.array(story_ids)
        story_ids.setflags(write=False)
        froi_subject_stories_subjects.setflags(write=False)
        froi_subject_stories_stories.setflags(write=False)

        for k in data:
            data[k].data.setflags(write=False)

        return RawData(examples, data, test_proportion=0., validation_proportion_of_train=0.1,
                       metadata=dict(
                           story_ids=story_ids,
                           froi_subject_stories_subjects=froi_subject_stories_subjects,
                           froi_subject_stories_stories=froi_subject_stories_stories))

    def _read_reaction_time_batches(self):
        groups = dict()
        all_worker_ids = set()
        for record in itertools.chain(
                _read_batch(os.path.join(self.path, 'batch1_pro.csv')),
                _read_batch(os.path.join(self.path, 'batch2_pro.csv'))):
            if record.correct < 5:  # rater had poor comprehension
                continue
            all_worker_ids.add(record.worker_id)
            key = (record.item, record.zone)
            if key not in groups:
                groups[key] = dict()
            groups[key][record.worker_id] = record.reaction_time

        reaction_times = np.full((len(groups), len(all_worker_ids)), np.nan)
        sorted_keys = list(sorted(groups))
        all_worker_ids = list(sorted(all_worker_ids))
        for idx_key, key in enumerate(sorted_keys):
            for idx_worker, worker in enumerate(all_worker_ids):
                if worker in groups[key]:
                    reaction_times[idx_key, idx_worker] = groups[key][worker]

        return reaction_times, sorted_keys, all_worker_ids

    def _read_reaction_times(self, example_manager: CorpusExampleUnifier, examples):
        reaction_times, keys, _ = self._read_reaction_time_batches()
        key_to_row = dict((k, i) for i, k in enumerate(keys))
        for example in examples:
            features = example_manager.add_example(
                tuple((w.item, w.zone) for w in example.words),
                [w.word for w in example.full_sentences],
                [w.sentence for w in example.full_sentences],
                'ns_spr',
                [key_to_row[(w.item, w.zone)] for w in example.full_sentences],
                start=example.offset,
                stop=example.offset + len(example.words),
                allow_new_examples=False)
            assert(features is not None)

        return {'ns_spr': reaction_times}

    def _read_audio_times(self):

        result = dict()
        mins = dict()
        maxes = dict()

        path = os.path.join(self.path, 'audio_alignments')

        for file in os.listdir(path):
            if os.path.splitext(file)[1] != '.TextGrid':
                continue
            text_grid = TextGrid.from_file(os.path.join(path, file))
            for tier in text_grid:
                if tier.name != 'words':
                    continue
                for word in tier:
                    if len(word.text) == 0 or word.text == '</s>' or word.text == '<s>':
                        continue
                    # ignore everything in the text except the token id,
                    # use the canonical form from _read_story_sentences
                    index_last_slash = word.text.rfind('/')
                    if index_last_slash < 0:
                        continue
                    token_id = word.text[(index_last_slash + 1):]
                    # item is the story identifier
                    item, zone = [int(p) for p in token_id.split('.')[:2]]
                    if item not in mins:
                        mins[item] = tier.xmin
                        maxes[item] = tier.xmax
                        result[item] = list()
                    result[item].append((zone, word.xmin, word.xmax))

        return OrderedDict((key, result[key]) for key in sorted(result)), mins, maxes

    def _compute_examples_froi_audio(self, item_to_num_images=None):
        audio_times, mins, maxes = self._read_audio_times()

        item_words = OrderedDict()
        for sentence_word_records in _read_story_sentences(self.path):
            if sentence_word_records[0].item not in item_words:
                item_words[sentence_word_records[0].item] = list()
            item_words[sentence_word_records[0].item].extend(sentence_word_records)

        item_tr_offsets = OrderedDict()
        tr_offset = 0
        if item_to_num_images is not None:
            for item in item_to_num_images:
                item_tr_offsets[item] = tr_offset
                tr_offset += item_to_num_images[item]
        for item in item_words:
            if item not in item_tr_offsets:
                item_tr_offsets[item] = tr_offset

        examples = list()
        for item in item_words:

            word_times = dict()
            for zone, min_time, max_time in audio_times[item]:
                # we can get duplicate entries for compound words like long-bearded
                if zone in word_times:
                    word_times[zone] = min(word_times[zone], min_time)
                else:
                    word_times[zone] = min_time

            word_times = list(word_times[z] for z in sorted(word_times))

            if item_to_num_images is not None and item in item_to_num_images:
                num_images = item_to_num_images[item]
            else:
                max_word_time = int(np.ceil(np.max(word_times)))
                # doesn't really matter if this goes past what is required
                num_images = int(np.ceil(max_word_time / 2.)) + 1

            time_images = np.arange(num_images, dtype=np.float) * 2.
            assert (len(time_images) > self.froi_skip_start_trs + self.froi_skip_end_trs)
            time_images = time_images[self.froi_skip_start_trs:(len(time_images) - self.froi_skip_end_trs)]

            examples.extend(self.froi_example_builder(
                words=item_words[item],
                word_times=word_times,
                word_sentence_ids=[w.sentence for w in item_words[item]],
                tr_times=time_images,
                tr_offset=item_tr_offsets[item] + self.froi_skip_start_trs))

        return examples

    def _read_froi_response_data(self):
        froi_path = os.path.join(self.path, 'fROI')
        suffix = '_language_fROIs.mat'
        froi_subjects = self.froi_subjects
        if froi_subjects is None:
            froi_subjects = list()
            for name in os.listdir(froi_path):
                if name.endswith(suffix):
                    froi_subjects.append(name[:-len(suffix)])
        elif isinstance(froi_subjects, str):
            froi_subjects = [froi_subjects]

        story_to_item = dict(
            (s, i + 1) for i, s in enumerate(
                ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis',
                 'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourette']))

        collated_data = dict()
        item_to_num_images = dict()
        num_rois = None

        for subject in froi_subjects:
            subject_path = os.path.join(froi_path, subject + suffix)
            data = loadmat(subject_path)
            data = data['data']['stories'][0, 0]
            stories = data.dtype.fields.keys()
            # Tree is not part of the natural stories corpus
            stories = [s for s in stories if s != 'Tree']
            if len(stories) < self.froi_minimum_story_count:
                continue
            collated_data[subject] = dict()
            for story in stories:
                # (roi, image)
                # Note that these time-series begin with 16s (8 time points) of fixation,
                # and end with either 16 or 32s (8 or 16 time points) of fixation
                story_data = data[story][0, 0].T  # after transpose, this is (images, rois)
                if num_rois is None:
                    num_rois = story_data.shape[1]
                elif num_rois != story_data.shape[1]:
                    raise ValueError('Inconsistent number of rois')
                item = story_to_item[story]
                start_image_padding = 8
                collated_data[subject][item] = story_data[start_image_padding:]
                if item in item_to_num_images:
                    item_to_num_images[item] = min(item_to_num_images[item], collated_data[subject][item].shape[0])
                    for s in collated_data:
                        if item in collated_data[s]:
                            collated_data[s][item] = collated_data[s][item][:item_to_num_images[item]]
                else:
                    item_to_num_images[item] = collated_data[subject][item].shape[0]

        # transform collated data into an (image, subject, roi) array
        data = np.full((sum(item_to_num_images[i] for i in item_to_num_images), len(collated_data), num_rois), np.nan)
        subject_stories = (list(), list())
        subjects = sorted(collated_data)
        item_to_num_images = OrderedDict((k, item_to_num_images[k]) for k in sorted(item_to_num_images))
        data_offset = 0
        for index_item, item in enumerate(item_to_num_images):
            for index_subject, subject in enumerate(subjects):
                if item in collated_data[subject]:
                    # keep track of which subjects have read which stories, so we can use this in
                    # the train/test split scheme later
                    subject_stories[0].append(subject)
                    subject_stories[1].append(item)

                    # set the data
                    data[data_offset:(data_offset + item_to_num_images[item]), index_subject] = collated_data[subject][
                        item]

            data_offset += item_to_num_images[item]

        subject_stories = (np.array(subject_stories[0]), np.array(subject_stories[1]))

        # get rois in order; tuples are abbreviations, followed by full name
        hemisphere_rois = [
            ('pt', 'posterior temporal'),
            ('at', 'anterior temporal'),
            ('ifg', 'inferior frontal gyrus'),
            ('ifgpo', 'inferior frontal gyrus, pars orbitalis'),  # note that these are purposefully a single entry
            ('mfg', 'middle frontal gyrus'),
            ('ag', 'angular gyrus')]
        short_rois = ['lh_' + roi[0] for roi in hemisphere_rois] + ['rh_' + roi[0] for roi in hemisphere_rois]
        # long_rois = ['left ' + roi[1] for roi in hemisphere_rois] + ['right ' + roi[1] for roi in hemisphere_rois]

        if len(short_rois) != num_rois:
            raise ValueError('Unexpected number of rois')

        data_keys = ['ns_{}'.format(roi) for roi in short_rois]

        data = OrderedDict(
            (n, np.squeeze(d, axis=2))
            for n, d in zip(data_keys, np.split(data, data.shape[2], axis=2)))

        return data, item_to_num_images, subjects, subject_stories

    def _add_froi_targets(
            self, example_manager: CorpusExampleUnifier, examples, froi_response_data, item_to_num_images):

        data_keys = [k for k in froi_response_data]

        for example in examples:
            # this means we don't actually have images for this story
            if example.words[0].item not in item_to_num_images:
                continue

            images = list()
            for target_tr in example.tr_target:
                if target_tr is None:
                    images.append(-1)
                else:
                    images.append(target_tr[0])  # keep only the first target if there are multiple

            pad_end = len(example.full_sentences) - len(example.words) - example.offset
            features = example_manager.add_example(
                tuple((w.item, w.zone) for w in example.words),
                [w.word for w in example.full_sentences],
                [w.sentence for w in example.full_sentences],
                data_keys,
                [-1] * example.offset + images + [-1] * pad_end,
                start=example.offset,
                stop=example.offset + len(example.words),
                is_apply_data_id_to_entire_group=True,
                allow_new_examples=False)
            assert(features is not None)


def read_natural_story_codings(directory_path, corpus_loader):

    bert_tokenizer = corpus_loader.make_bert_tokenizer()
    spacy_tokenize_model = make_tokenizer_model()

    example_manager = CorpusExampleUnifier(spacy_tokenize_model, bert_tokenizer)

    codings = _read_codings(directory_path)
    result = dict()
    for unique_id, sentence_word_records in enumerate(_read_story_sentences(directory_path)):
        words = [wr.word for wr in sentence_word_records]
        text = ' '.join(words)
        input_features = example_manager.add_example(
            tuple((w.item, w.zone) for w in sentence_word_records),
            [w.word for w in sentence_word_records],
            [w.sentence for w in sentence_word_records],
            None,
            None)
        token_key = ' '.join(input_features.tokens)
        if text not in codings:
            raise ValueError('Unable to find codings for sentence: {}'.format(text))
        result[token_key] = codings[text]
    return result


def natural_stories_make_leave_stories_out(index_variation_run):
    return partial(natural_stories_leave_stories_out, index_variation_run=index_variation_run)


def natural_stories_leave_stories_out(raw_data, index_variation_run, random_state=None, shuffle=True):
    story_ids = raw_data.metadata['story_ids']
    stories_with_froi = set()
    stories_without_froi = set()
    froi_response_keys = set(
        k for k in raw_data.response_data if raw_data.response_data[k].kind == ResponseKind.ns_froi)
    for example in raw_data.input_examples:
        for k in raw_data.response_data:
            if np.any(example.data_ids[k] >= 0):
                if k in froi_response_keys:
                    stories_with_froi.add(story_ids[example.unique_id])
                else:
                    stories_without_froi.add(story_ids[example.unique_id])

    stories_without_froi = list(sorted(stories_without_froi))
    stories_with_froi = list(sorted(stories_with_froi))

    item_to_story = dict(
        (i + 1, s) for i, s in enumerate(
            ['Boar', 'Aqua', 'MatchstickSeller', 'KingOfBirds', 'Elvis',
             'MrSticky', 'HighSchool', 'Roswell', 'Tulips', 'Tourette']))

    if len(stories_with_froi) == 0:
        folds = stories_without_froi
    elif len(stories_without_froi) == 0:
        folds = stories_with_froi
    else:
        folds = list(itertools.product(stories_without_froi, stories_with_froi))

    validation_stories = folds[index_variation_run % len(folds)]
    if np.isscalar(validation_stories):
        validation_stories = {validation_stories}
    else:
        validation_stories = set(validation_stories)

    # noinspection PyTypeChecker
    validation_story_names = [item_to_story[s] for s in sorted(validation_stories)]
    logger.info('Validation stories: {}'.format(validation_story_names))

    train_examples = list()
    validation_examples = list()
    for example in raw_data.input_examples:
        if story_ids[example.unique_id] in validation_stories:
            validation_examples.append(example)
        else:
            train_examples.append(example)

    if shuffle:
        if random_state is not None:
            random_state.shuffle(train_examples)
            random_state.shuffle(validation_examples)
        else:
            np.random.shuffle(train_examples)
            np.random.shuffle(validation_examples)

    test_examples = list()
    return train_examples, validation_examples, test_examples


# library(plyr)
# library(dplyr)
# library(ggplot2)
#
# #read in RT data from 2 separate files
# b2 <- read.csv('batch2_pro.csv')
# b1 <- read.csv('batch1_pro.csv')
# d <- rbind(b1, b2)
#
# ##subtract 2 from zone to properly align region...should confirm with Hal that this is correct,
# ## but the RTs seem to line up correctly in plots
# d$zone <- d$zone - 2
#
# #read in story words and region
# #item is story (1-10), zone is RT region
# word.df <- read.csv('all_stories.tok', sep = '\t')
# d <- merge(d, word.df, by= c('item', 'zone'), all.x = T, all.y = T)
#
# #remove regions that do not have words
# d <- filter(d, !is.na(word))
#
# #exclude stories where subject does not get more than 4/6 correct
# unfiltered <- d
# d <- filter(d, correct > 4)
#
# #exclude data points less than 50 ms, greater than 3000 ms
# d <- d[d$RT > 100 & d$RT < 3000, ]
# d$l <- nchar(as.character(d$word))
#
#
# #calculate by-word statistics
#
# gmean <- function(x) exp(mean(log(x)))
# gsd   <- function(x) exp(sd(log(x)))
#
# word.info = d %>%
#   group_by(word, zone, item) %>%
#     summarise(nItem=length(RT),
#               meanItemRT=mean(RT),
# 	      sdItemRT=sd(RT),
# 	      gmeanItemRT=sd(RT),
# 	      gsdItemRT=gsd(RT))
#
# d <- inner_join(d, word.info, by=c("word", "zone", "item"))
#
# #write processed output, by word, overall
# #write.table(word.info, 'processed_wordinfo.tsv', quote = F, row.names=F, sep="\t")
# #write.table(d, 'processed_RTs.tsv', quote=F, row.names=F, sep="\t")
#
# ggplot(d, aes(RT)) + facet_grid( . ~ WorkerId) + geom_histogram()
#
#
# ##make plot
# make.story.plot <- function(item, group)
# {
# return (
#   ggplot(word.info[word.info$item == item & word.info$group == group, ],
#   aes(x = zone, y = meanItemRT, group = group)) +
#   geom_line(colour = 'grey') +
#   geom_text(aes( x = zone, y= meanItemRT, label = word), size = 2) + facet_grid(group~ .) +
#   theme_bw() + coord_cartesian(ylim = c(min(word.info$meanItemRT), 550)))
# }
#
# word.info$group <- cut(word.info$zone, 20, labels = F)
#
# pdf('practice_RTplot.pdf')
# for (i in seq(1:20))
# print(make.story.plot(1, i))
# dev.off()
