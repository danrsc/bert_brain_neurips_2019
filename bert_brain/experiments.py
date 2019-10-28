import hashlib
import itertools
import random

import numpy as np
import torch
from scipy import signal

from .common import SwitchRemember
from .data_sets import ResponseKind, CorpusTypes, PreprocessSoSFilter, PreprocessDetrend, PreprocessStandardize, \
    PreprocessKMeans, PreprocessRandomPair, PreprocessMakeBinary, preprocess_fork_no_cluster_to_disk, \
    PreprocessFeatureNormalize
from .modeling import KeyedLinear, CriticKeys, KeyedCombinedLinear, KeyedGroupConcatLinear
from .settings import TrainingVariation, LoadFrom, Settings, OptimizationSettings, PredictionHeadSettings, \
    CriticSettings


__all__ = ['task_hash', 'set_random_seeds', 'iterate_powerset', 'named_variations', 'match_variation']


def _internal_hash_update(hash_, loss_tasks):
    if isinstance(loss_tasks, TrainingVariation):
        for loss_task in sorted(loss_tasks.loss_tasks):
            hash_.update(loss_task.encode())
        if loss_tasks.load_from is not None:
            hash_.update(loss_tasks.load_from.variation_name.encode())
            _internal_hash_update(hash_, loss_tasks.load_from.loss_tasks)
    else:
        for loss_task in sorted(loss_tasks):
            hash_.update(loss_task.encode())


def task_hash(loss_tasks):
    hash_ = hashlib.sha256()
    _internal_hash_update(hash_, loss_tasks)
    return hash_.hexdigest()


def set_random_seeds(seed, index_run, n_gpu):
    hash_ = hashlib.sha256('{}'.format(seed).encode())
    hash_.update('{}'.format(index_run).encode())
    seed = np.frombuffer(hash_.digest(), dtype='uint32')
    random_state = np.random.RandomState(seed)
    np.random.set_state(random_state.get_state())
    seed = np.random.randint(low=0, high=np.iinfo('uint32').max)
    random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    return seed


def rank_space(data):
    from scipy.stats.mstats import rankdata
    return rankdata(data, axis=0)


def iterate_powerset(items):
    for sub_set in itertools.chain.from_iterable(
            itertools.combinations(items, num) for num in range(1, len(items) + 1)):
        yield sub_set


def named_variations(name):

    erp_tasks = ('epnp', 'pnp', 'elan', 'lan', 'n400', 'p600')
    ns_froi_tasks = ('ns_lh_pt', 'ns_lh_at', 'ns_lh_ifg', 'ns_lh_ifgpo', 'ns_lh_mfg', 'ns_lh_ag',
                     'ns_rh_pt', 'ns_rh_at', 'ns_rh_ifg', 'ns_rh_ifgpo', 'ns_rh_mfg', 'ns_rh_ag')

    # noinspection PyPep8Naming
    load_from_I = LoadFrom('hp_fmri_20', ('hp_fmri_I',), map_run=lambda r: r % 4)

    name = SwitchRemember(name)
    auxiliary_loss_tasks = set()

    if name == 'erp':
        training_variations = list(iterate_powerset(erp_tasks))
        settings = Settings(corpora=(CorpusTypes.UclCorpus(),))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'erp_joint':
        training_variations = [erp_tasks]
        settings = Settings(corpora=(CorpusTypes.UclCorpus(),))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories':
        training_variations = [('ns_spr',),
                               erp_tasks + ('ns_spr',),
                               ns_froi_tasks + ('ns_spr',),
                               erp_tasks + ns_froi_tasks,
                               erp_tasks + ns_froi_tasks + ('ns_spr',)]
        settings = Settings(
            corpora=(
                CorpusTypes.NaturalStoriesCorpus(
                    froi_window_duration=10.,
                    froi_minimum_duration_required=9.5,
                    froi_use_word_unit_durations=False,
                    froi_sentence_mode='ignore'),
                CorpusTypes.UclCorpus()),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        settings.prediction_heads[ResponseKind.ns_froi] = PredictionHeadSettings(
            ResponseKind.ns_froi, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    elif name == 'ns_froi':
        training_variations = [ns_froi_tasks]
        settings = Settings(
            corpora=(
                CorpusTypes.NaturalStoriesCorpus(
                    froi_sentence_mode='ignore',
                    froi_window_duration=10.,
                    froi_minimum_duration_required=9.5,
                    froi_use_word_unit_durations=False,
                    froi_minimum_story_count=2,
                    include_reaction_times=False),),
            optimization_settings=OptimizationSettings(num_train_epochs=3))
        settings.prediction_heads[ResponseKind.ns_froi] = PredictionHeadSettings(
            ResponseKind.ns_froi, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 7  # there are 7 stories that have been recorded in froi
        min_memory = 4 * 1024 ** 3
    elif name == 'ns_hp':
        training_variations = [('hp_fmri_I',),
                               ns_froi_tasks,
                               ns_froi_tasks + ('hp_fmri_I',)]
        settings = Settings(
            corpora=(
                CorpusTypes.NaturalStoriesCorpus(
                    froi_sentence_mode='ignore',
                    froi_window_duration=10.1,
                    froi_minimum_duration_required=9.6,
                    froi_use_word_unit_durations=False,
                    froi_minimum_story_count=2,
                    include_reaction_times=False),
                CorpusTypes.HarryPotterCorpus(
                    fmri_subjects='I',
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    meg_subjects=[])),
            optimization_settings=OptimizationSettings(num_train_epochs=30, num_epochs_train_prediction_heads_only=10))
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        settings.prediction_heads[ResponseKind.ns_froi] = PredictionHeadSettings(
            ResponseKind.ns_froi, KeyedLinear, dict(is_sequence='naked_pooled'))
        # settings.split_functions[CorpusKeys.harry_potter] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessSoSFilter(signal.butter(10, 0.1, 'hp', fs=0.5, output='sos')),
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_se_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #         critic_type=CriticKeys.single_k_least_se,
        #         critic_kwargs=dict(
        #             k_fn=KLeastSEHalvingEpochs(
        #                 0.5, delay_in_epochs=9, minimum_k=5000),
        #             moving_average_decay=0.999))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'nat_stories_head_loc':
        training_variations = [('ns_spr',), erp_tasks + ('ns_spr',), erp_tasks]
        settings = Settings(corpora=(CorpusTypes.NaturalStoriesCorpus(), CorpusTypes.UclCorpus()))
        auxiliary_loss_tasks = {'input_head_location'}
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'number_agreement':
        agr = ('colorless', 'linzen_agree')
        training_variations = [agr, erp_tasks + agr, erp_tasks]
        settings = Settings(
            corpora=(CorpusTypes.ColorlessGreenCorpus(), CorpusTypes.LinzenAgreementCorpus(), CorpusTypes.UclCorpus()),
            optimization_settings=OptimizationSettings(num_train_epochs=50))
        num_runs = 10
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri':
        training_variations = [('hp_fmri_I',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus()),
            optimization_settings=OptimizationSettings(num_train_epochs=10))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_20':
        training_variations = [('hp_fmri_I',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=['I'],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=False,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=2,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_meg':
        training_variations = [
            TrainingVariation(('hp_meg',), load_from=load_from_I),
            ('hp_meg',),
            TrainingVariation(('hp_meg', 'hp_fmri_I'), load_from=load_from_I),
            ('hp_meg', 'hp_fmri_I'),
            ('hp_fmri_I',)]
        # training_variations = [
        #     ('hp_fmri_I', 'hp_meg'), ('hp_meg',), ('hp_fmri_I',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects='I',
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        final_linear_start = \
            settings.optimization_settings.num_train_epochs \
            - settings.optimization_settings.num_final_epochs_train_prediction_heads_only
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5,
        #             delay_in_epochs=settings.optimization_settings.num_epochs_train_prediction_heads_only,
        #             minimum_k=100,
        #             final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics['hp_fmri_I'] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5, delay_in_epochs=2, minimum_k=20000, final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.pearson, critic_kwargs=dict(should_penalize_scale=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_meg_joint_45':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        fmri_tasks_ = tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)
        training_variations = [
            fmri_tasks_ + ('hp_meg',)]
        # ('hp_meg',),
        # fmri_tasks_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=45,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', force_cpu=True))
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True, force_cpu=True))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5,
        #             delay_in_epochs=settings.optimization_settings.num_epochs_train_prediction_heads_only,
        #             minimum_k=100,
        #             final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics['hp_fmri_I'] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5, delay_in_epochs=2, minimum_k=20000, final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.pearson, critic_kwargs=dict(should_penalize_scale=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_meg_joint':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        fmri_tasks_ = tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)
        training_variations = [
            fmri_tasks_ + ('hp_meg',)]
            # ('hp_meg',),
            # fmri_tasks_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=60,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', force_cpu=True))
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True, force_cpu=True))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5,
        #             delay_in_epochs=settings.optimization_settings.num_epochs_train_prediction_heads_only,
        #             minimum_k=100,
        #             final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics['hp_fmri_I'] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5, delay_in_epochs=2, minimum_k=20000, final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.pearson, critic_kwargs=dict(should_penalize_scale=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_meg_independent':
        fmri_subjects_ = ['I']
        # fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = [('hp_fmri_{}'.format(s),) for s in fmri_subjects_]
        training_variations += [('hp_fmri_{}'.format(s), 'hp_meg') for s in fmri_subjects_]
        training_variations += [('hp_meg',)]
        # ('hp_meg',),
        # fmri_tasks_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=60,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', force_cpu=True))
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True, force_cpu=True))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5,
        #             delay_in_epochs=settings.optimization_settings.num_epochs_train_prediction_heads_only,
        #             minimum_k=100,
        #             final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics['hp_fmri_I'] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_se,
        #     critic_kwargs=dict(
        #         k_fn=KLeastSEHalvingEpochs(
        #             0.5, delay_in_epochs=2, minimum_k=20000, final_full_epochs_start=final_linear_start),
        #         moving_average_decay=0.999))
        # settings.critics[ResponseKind.hp_meg] = CriticSettings(
        #     critic_type=CriticKeys.pearson, critic_kwargs=dict(should_penalize_scale=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_erp':
        training_variations = [
            TrainingVariation(erp_tasks, load_from=load_from_I),
            erp_tasks,
            TrainingVariation(erp_tasks + ('hp_fmri_I',), load_from=load_from_I),
            erp_tasks + ('hp_fmri_I',)]
        # training_variations = [
        #     ('hp_fmri_I', 'hp_meg'), ('hp_meg',), ('hp_fmri_I',)]
        settings = Settings(
            corpora=(
                CorpusTypes.UclCorpus(),
                CorpusTypes.HarryPotterCorpus(
                    fmri_subjects='I',
                    fmri_sentence_mode='ignore',
                    fmri_window_duration=10.1,
                    fmri_minimum_duration_required=9.6,
                    group_meg_sentences_like_fmri=False,
                    meg_kind='leila',
                    meg_subjects=[])),
            optimization_settings=OptimizationSettings(
                num_train_epochs=12,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_HIKL_independent':
        subjects_ = ['H', 'I', 'K', 'L']
        training_variations = [('hp_fmri_{}'.format(s),) for s in subjects_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=False,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        # final_linear_start = \
        #     settings.optimization_settings.num_train_epochs \
        #     - settings.optimization_settings.num_final_epochs_train_prediction_heads_only
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        # for subject in subjects_:
        #     settings.critics['hp_fmri_{}'.format(subject)] = CriticSettings(
        #         critic_type=CriticKeys.single_k_least_se,
        #         critic_kwargs=dict(
        #             k_fn=KLeastSEHalvingEpochs(
        #                 0.5, delay_in_epochs=2, minimum_k=5000, final_full_epochs_start=final_linear_start),
        #             moving_average_decay=0.999))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_HIKL_joint':
        subjects_ = ['H', 'I', 'K', 'L']
        joint = tuple('hp_fmri_{}'.format(s) for s in subjects_)
        training_variations = [joint]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=False,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=40,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        # final_linear_start = \
        #     settings.optimization_settings.num_train_epochs \
        #     - settings.optimization_settings.num_final_epochs_train_prediction_heads_only
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        # for subject in subjects_:
        #     settings.critics['hp_fmri_{}'.format(subject)] = CriticSettings(
        #         critic_type=CriticKeys.single_k_least_se,
        #         critic_kwargs=dict(
        #             k_fn=KLeastSEHalvingEpochs(
        #                 0.5, delay_in_epochs=2, minimum_k=5000, final_full_epochs_start=final_linear_start),
        #             moving_average_decay=0.999))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg':
        training_variations = [('hp_meg',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_combined':
        training_variations = [
            TrainingVariation(('hp_meg',), load_from=load_from_I),
            ('hp_meg',),
            TrainingVariation(('hp_meg', 'hp_fmri_I'), load_from=load_from_I),
            ('hp_meg', 'hp_fmri_I'),
            ('hp_fmri_I',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects='I',
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedCombinedLinear, kwargs=dict(naked_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_linear':
        training_variations = [('hp_meg',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=-1,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_erp':
        training_variations = [
            TrainingVariation(
                erp_tasks, load_from=LoadFrom('hp_fmri_meg', loss_tasks=('hp_meg',), map_run=lambda i: i % 4)),
            erp_tasks]
        settings = Settings(
            corpora=(CorpusTypes.UclCorpus(),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=22,
                num_epochs_train_prediction_heads_only=20,
                num_final_epochs_train_prediction_heads_only=0))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'erp_hp_meg':
        training_variations = [
            TrainingVariation(
                ('hp_meg',), load_from=LoadFrom('hp_meg_erp', loss_tasks=erp_tasks)),
            TrainingVariation(
                ('hp_meg',), load_from=LoadFrom(
                    'hp_meg_erp',
                    TrainingVariation(erp_tasks, load_from=LoadFrom('hp_fmri_meg', loss_tasks=('hp_meg',))))),
            ('hp_meg',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=None),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessDetrend(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, head_type=KeyedLinear, kwargs=dict(is_sequence=True))
        num_runs = 12
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_fmri_linear':
        fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = list()
        for subject in fmri_subjects_:
            training_variations.append(TrainingVariation(
                ('hp_fmri_{}'.format(subject),), load_from=LoadFrom(
                    'erp_hp_meg',
                    TrainingVariation(
                        ('hp_meg',), load_from=LoadFrom(
                            'hp_meg_erp',
                            TrainingVariation(erp_tasks, load_from=LoadFrom(
                                'hp_fmri_meg',
                                loss_tasks=('hp_meg',))))))))
            training_variations.append(('hp_fmri_{}'.format(subject),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=15,
                num_epochs_train_prediction_heads_only=-1,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_fmri':
        fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = list()
        for subject in fmri_subjects_:
            training_variations.append(TrainingVariation(
                ('hp_fmri_{}'.format(subject),), load_from=LoadFrom(
                    'erp_hp_meg',
                    TrainingVariation(
                        ('hp_meg',), load_from=LoadFrom(
                            'hp_meg_erp',
                            TrainingVariation(erp_tasks, load_from=LoadFrom(
                                'hp_fmri_meg',
                                loss_tasks=('hp_meg',))))))))
            training_variations.append(('hp_fmri_{}'.format(subject),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_simple_fmri':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = list()
        for subject in fmri_subjects_:
            training_variations.append(TrainingVariation(
                ('hp_fmri_{}'.format(subject),), load_from=LoadFrom(
                    'hp_meg',
                    loss_tasks=('hp_meg',))))
            training_variations.append(('hp_fmri_{}'.format(subject),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_simple_fmri_linear':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = list()
        for subject in fmri_subjects_:
            training_variations.append(TrainingVariation(
                ('hp_fmri_{}'.format(subject),), load_from=LoadFrom(
                    'hp_fmri_meg',
                    loss_tasks=('hp_meg',))))
            training_variations.append(('hp_fmri_{}'.format(subject),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=-1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'erp_hp_fmri':
        fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = list()
        for subject in fmri_subjects_:
            training_variations.append(TrainingVariation(
                ('hp_fmri_{}'.format(subject),), load_from=LoadFrom(
                    'hp_meg_erp',
                    loss_tasks=erp_tasks)))
            training_variations.append(('hp_fmri_{}'.format(subject),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),  # None means everyone
            optimization_settings=OptimizationSettings(
                num_train_epochs=30,
                num_epochs_train_prediction_heads_only=10,
                num_final_epochs_train_prediction_heads_only=0))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_20_linear':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        training_variations = [('hp_fmri_{}'.format(s),) for s in fmri_subjects_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=-1),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_20_linear':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        training_variations = [('hp_fmri_{}'.format(s),) for s in fmri_subjects_]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=-1),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'sst':
        training_variations = [('sentiment',)]
        settings = Settings(corpora=(CorpusTypes.StanfordSentimentTreebank(),))
        num_runs = 1
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_HKL_from_I':
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        training_variations = list()
        for s in fmri_subjects_:
            training_variations.append(TrainingVariation(('hp_fmri_{}'.format(s),), load_from=load_from_I))
            training_variations.append(('hp_fmri_{}'.format(s),))
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=-1),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence=False))
        num_runs = 100
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_HKL_from_I_fine_tune':
        # noinspection PyPep8Naming
        load_from_I = LoadFrom('hp_fmri_20', ('hp_fmri_I',))
        training_variations = [
            TrainingVariation(('hp_fmri_H',), load_from=load_from_I),
            ('hp_fmri_H',),
            TrainingVariation(('hp_fmri_K',), load_from=load_from_I),
            ('hp_fmri_K',),
            TrainingVariation(('hp_fmri_L',), load_from=load_from_I),
            ('hp_fmri_L',)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=['H', 'K', 'L'],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=20,
                num_epochs_train_prediction_heads_only=10))
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled'))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_diff_cluster':
        # fmri_subjects_ = ['I']
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = [tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessKMeans(num_clusters=100, transform_fn=rank_space, n_init=100),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.preprocess_fork_fn = preprocess_fork_no_cluster_to_disk
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_diff_cluster_2':
        fmri_subjects_ = ['I']
        # fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = [tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessKMeans(num_clusters=100, transform_fn=rank_space, n_init=100),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.preprocess_fork_fn = preprocess_fork_no_cluster_to_disk
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_diff_2':
        fmri_subjects_ = ['I']
        # fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = [tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_fmri_meg_diff_cluster':
        # fmri_subjects_ = ['I']
        fmri_subjects_ = ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N']
        # fmri_subjects_ = ['H', 'I', 'K', 'L']
        training_variations = [tuple('hp_fmri_{}'.format(s) for s in fmri_subjects_)]
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=fmri_subjects_,
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='leila',
                meg_subjects=[]),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_fmri] = [
            PreprocessDetrend(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessKMeans(num_clusters=100, transform_fn=rank_space),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.preprocess_fork_fn = preprocess_fork_no_cluster_to_disk
        settings.prediction_heads[ResponseKind.hp_fmri] = PredictionHeadSettings(
            ResponseKind.hp_fmri, KeyedLinear, dict(is_sequence='naked_pooled', hidden_sizes=[20]))
        settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_L2':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)

        training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_L2',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_median':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)

        training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_median',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_cluster_median':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_median',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=10,
                num_epochs_train_prediction_heads_only=5,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessStandardize(
                stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None),
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(
                stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, average_axis=None)]
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_counts':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)

        training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_counts',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_rms':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)

        # just do joint for now
        training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_rms',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_mean':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)
        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_mean',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end)),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_mean_100ms':
        meg_subjects_ = ['A']  # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        # training_variations.append(all_subj)

        # training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_mean_time_slice_ms_100_A',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=[10], hidden_activation=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_mean_whole':
        meg_subjects_ = ['A']  # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        # training_variations.append(all_subj)

        # training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_mean_whole_A',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=[20], hidden_activation=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_cluster_sum_100ms':
        meg_subjects_ = ['A']  # ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        # training_variations.append(all_subj)

        # training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='rank_clustered_sum_time_slice_ms_100_A',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessFeatureNormalize(),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=None, hidden_activation=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    elif name == 'hp_meg_diff_drc_25':
        meg_subjects_ = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']  # , 'multi_subject']
        training_variations = list()
        all_subj = ()
        for subject in meg_subjects_:
            # training_variations.append(tuple('hp_meg_{}.{}'.format(subject, idx) for idx in range(300)))
            training_variations.append(('hp_meg_{}'.format(subject),))
            if subject != 'multi_subject':
                all_subj += training_variations[-1]
        training_variations.append(all_subj)

        training_variations = [training_variations[-1]]

        settings = Settings(
            corpora=(CorpusTypes.HarryPotterCorpus(
                fmri_subjects=[],
                fmri_sentence_mode='ignore',
                fmri_window_duration=10.1,
                fmri_minimum_duration_required=9.6,
                group_meg_sentences_like_fmri=True,
                meg_kind='direct_rank_clustered_sum_25_ms',
                meg_subjects=meg_subjects_),),
            optimization_settings=OptimizationSettings(
                num_train_epochs=3,
                num_epochs_train_prediction_heads_only=1,
                num_final_epochs_train_prediction_heads_only=0),
            filter_when_not_in_loss_keys=(ResponseKind.hp_fmri, ResponseKind.hp_meg))
        # settings.split_functions[CorpusKeys.HarryPotterCorpus] = HarryPotterMakeLeaveOutFmriRun(make_test=True)
        # settings.preprocessors[ResponseKind.hp_fmri] = [
        #     PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(
        #         stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True, use_absolute=True)]
        settings.preprocessors[ResponseKind.hp_meg] = [
            PreprocessFeatureNormalize(),
            ('diff', PreprocessRandomPair(
                num_samples_per_group=5000,
                metadata_example_group_by='fmri_runs',
                data_id_pair_fn_map=PreprocessRandomPair.pair_from_end,
                emit_both=True,
                stop_mode='content')),
            PreprocessMakeBinary(threshold=0)]
        settings.critics[ResponseKind.hp_meg] = CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(critic_type=CriticKeys.single_mae)
        # settings.critics[ResponseKind.hp_fmri] = CriticSettings(
        #     critic_type=CriticKeys.single_k_least_ae_on_eval,
        #     critic_kwargs=dict(k_fn=KLeastSEHalvingEpochs(0.5, delay_in_epochs=9, minimum_k=5000)))
        settings.data_id_in_batch_keys += (ResponseKind.hp_meg,)
        settings.prediction_heads[ResponseKind.hp_meg] = PredictionHeadSettings(
            ResponseKind.hp_meg, KeyedGroupConcatLinear,
            dict(num_per_data_id=2, hidden_sizes=None, hidden_activation=None, include_pooled=True))
        num_runs = 4
        min_memory = 4 * 1024 ** 3
    else:
        raise ValueError('Unknown name: {}. Valid choices are: \n{}'.format(name.var, '\n'.join(name.tests)))

    return training_variations, settings, num_runs, min_memory, auxiliary_loss_tasks


def match_variation(variation_set_name, training_variation):
    """
    Given a variation_set_name (the --name argument in run_variations.py) and a training_variation which can be
    a string, a TrainingVariation instance, or a tuple, finds the matching canonical training variation and returns
    it.
    Notes:
        We need to simplify the training variation specification so this function is not necessary
    Args:
        variation_set_name: The variation set to look in, e.g. 'hp_fmri'
        training_variation: The variation to match, e.g. ('hp_fmri_I',)

    Returns:
        The canonical form of the training variation.
    """
    training_variations, _, _, _, _ = named_variations(variation_set_name)
    if isinstance(training_variation, str):
        training_variation_name = training_variation
    elif isinstance(training_variation, TrainingVariation):
        training_variation_name = training_variation.name
    else:
        training_variation_name = str(tuple(training_variation))
    for t in training_variations:
        t_name = t.name if isinstance(t, TrainingVariation) else str(tuple(t))
        if training_variation_name == t_name:
            return t
    raise ValueError('Unable to match training_variation: {}'.format(training_variation))
