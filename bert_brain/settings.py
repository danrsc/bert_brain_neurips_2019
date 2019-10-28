from dataclasses import dataclass, field
from typing import Sequence, Callable, MutableMapping, Mapping, Optional, Union, Tuple

import numpy as np
from .data_sets import PreprocessStandardize, PreprocessLog, \
    PreprocessPCA, PreprocessClip, PreprocessDetrend, HarryPotterMakeLeaveOutFmriRun, PreparedDataView, \
    ResponseKind, InputFeatures, RawData, natural_stories_make_leave_stories_out, KindData, CorpusKeys, CorpusBase, \
    UclCorpus
from .modeling import CriticKeys, FMRIConvConvWithDilationHead


__all__ = ['OptimizationSettings', 'PredictionHeadSettings', 'CriticSettings', 'TrainingVariation', 'LoadFrom',
           'Settings']


@dataclass
class OptimizationSettings:
    # Total number of training epochs to perform.
    num_train_epochs: int = 3
    # initial learning rate for Adam
    learning_rate: float = 5e-5
    # Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.
    warmup_proportion: float = 0.1
    train_batch_size: int = 32
    predict_batch_size: int = 8
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride: int = 128
    # Whether to perform optimization and keep the optimizer averages on CPU
    optimize_on_cpu: bool = False
    # Whether to use 16-bit float precision instead of 32-bit
    fp16: bool = False
    # Loss scaling, positive power of 2 values can improve fp16 convergence.
    loss_scale: float = 128
    # Number of updates steps to accumulate before performing a backward/update pass.
    gradient_accumulation_steps: int = 1
    # local_rank for distributed training on gpus; probably don't need this
    local_rank: int = -1
    # During the first num_epochs_train_prediction_heads_only, only the prediction heads will be trained
    num_epochs_train_prediction_heads_only: int = 0
    # During the last num_final_prediction_head_only_epochs, only the prediction heads will be trained
    num_final_epochs_train_prediction_heads_only: int = 0


@dataclass
class PredictionHeadSettings:
    key: str
    head_type: type
    kwargs: dict


@dataclass
class CriticSettings:
    critic_type: str
    critic_kwargs: Optional[Mapping] = None


def _default_split_functions():

    return {
        CorpusKeys.HarryPotterCorpus: HarryPotterMakeLeaveOutFmriRun(),
        CorpusKeys.NaturalStoriesCorpus: natural_stories_make_leave_stories_out
    }


def _default_preprocessors():

    preprocess_standardize = PreprocessStandardize(stop_mode='content')

    # ResponseKind.colorless defaults to None
    # ResponseKind.linzen_agree defaults to None

    # alternate hp_meg for soft_label_cross_entropy
    #     [
    #         #  preprocess_detrend,
    #         #  partial(preprocess_standardize, average_axis=None),
    #         PreprocessDiscretize(bins=np.exp(np.linspace(-0.2, 1., 5))),  # bins=np.arange(6) - 2.5
    #         PreprocessNanMean())]

    return {
        ResponseKind.hp_fmri: [
            PreprocessDetrend(stop_mode=None, metadata_example_group_by='fmri_runs', train_on_all=True),
            PreprocessStandardize(stop_mode=None)],
        ResponseKind.hp_meg: [
            PreprocessStandardize(average_axis=None, stop_mode='content'),
            PreprocessPCA(stop_mode='content'),
            PreprocessStandardize(average_axis=None, stop_mode='content')],
        # ResponseKind.hp_meg: [
        #     PreprocessDetrend(stop_mode='content', metadata_example_group_by='fmri_runs', train_on_all=True),
        #     PreprocessStandardize(stop_mode='content', average_axis=None)],

        ResponseKind.ucl_erp: preprocess_standardize,
        ResponseKind.ucl_eye: [PreprocessLog(), preprocess_standardize],
        ResponseKind.ucl_self_paced: [PreprocessLog(), preprocess_standardize],
        ResponseKind.ns_reaction_times: [
            PreprocessClip(maximum=3000, value_beyond_max=np.nan), PreprocessLog(), preprocess_standardize],
        ResponseKind.ns_froi: PreprocessStandardize(stop_mode=None),
    }


def _default_supplemental_fields():
    return {'token_lengths', 'token_probabilities'}


def _default_prediction_heads():

    return {
        ResponseKind.hp_fmri: PredictionHeadSettings(
            ResponseKind.hp_fmri, FMRIConvConvWithDilationHead, dict(
                hidden_channels=10,
                hidden_kernel_size=5,
                out_kernel_size=5,
                out_dilation=5,
                memory_efficient=False)),
        ResponseKind.ns_froi: PredictionHeadSettings(
            ResponseKind.ns_froi, FMRIConvConvWithDilationHead, dict(
                hidden_channels=10,
                hidden_kernel_size=5,
                out_kernel_size=5,
                out_dilation=5,
                memory_efficient=False))
    }


def _default_critics():

    return {
        CorpusKeys.UclCorpus: CriticSettings(critic_type=CriticKeys.mse),
        ResponseKind.ns_froi: CriticSettings(critic_type=CriticKeys.single_mse),
        CorpusKeys.NaturalStoriesCorpus: CriticSettings(critic_type=CriticKeys.mse),
        ResponseKind.hp_fmri: CriticSettings(critic_type=CriticKeys.single_mse),
        CorpusKeys.HarryPotterCorpus: CriticSettings(critic_type=CriticKeys.mse),
        CorpusKeys.ColorlessGreenCorpus: CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy),
        CorpusKeys.LinzenAgreementCorpus: CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy),
        CorpusKeys.StanfordSentimentTreebank: CriticSettings(critic_type=CriticKeys.single_binary_cross_entropy)
    }


@dataclass
class LoadFrom:
    variation_name: str
    loss_tasks: Union[Sequence[str], 'TrainingVariation']
    map_run: Optional[Callable[[int], int]] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            self.name = '{}:{}'.format(
                self.variation_name,
                self.loss_tasks.name if isinstance(self.loss_tasks, TrainingVariation) else tuple(self.loss_tasks))


@dataclass
class TrainingVariation:
    loss_tasks: Sequence[str]
    # If specified, then this causes a partially fine-tuned model to be loaded instead of the base pre-trained model.
    load_from: Optional[LoadFrom] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.name is None:
            suffix = '.' + self.load_from.name if self.load_from is not None else ''
            self.name = '{}{}'.format(tuple(self.loss_tasks), suffix)


_PreprocessorTypeUnion = Union[
    Callable[[PreparedDataView, Optional[Mapping[str, np.ndarray]]], PreparedDataView],
    Tuple[
        str,
        Callable[
            [PreparedDataView, Optional[Mapping[str, np.ndarray]]],
            Tuple[PreparedDataView, Optional[Mapping[str, np.ndarray]]]]],
    str]
PreprocessorTypeUnion = Union[_PreprocessorTypeUnion, Sequence[_PreprocessorTypeUnion]]


@dataclass
class Settings:
    # which data to load
    corpora: Optional[Sequence[CorpusBase]] = (UclCorpus(),)

    # maps from a corpus key to a function which takes the index of the current variation and returns another
    # function which will do the splitting. The returned function should take a RawData instance and a RandomState
    # instance and return a tuple of (train, validation, test) examples
    # if not specified, the data will be randomly split according to the way the fields of the RawData instance are set.
    # I.e. if the RawData instance is pre split, that is respected. Otherwise test_proportion and
    # validation_proportion_of_train fields in the RawData instance govern the split
    split_functions: MutableMapping[
        str,
        Callable[[int], Callable[
            [RawData, np.random.RandomState],
            Tuple[
                Optional[Sequence[InputFeatures]],
                Optional[Sequence[InputFeatures]],
                Optional[Sequence[InputFeatures]]]]]] = field(default_factory=_default_split_functions)

    # Mapping from [response_key, kind, or corpus_key] to a preprocessor. Lookups fall back in that
    # order. This determines how the data will be processed. If not specified, no preprocessing is applied
    preprocessors: MutableMapping[str, PreprocessorTypeUnion] = field(default_factory=_default_preprocessors)

    preprocess_fork_fn: Callable[
        [str, KindData, PreprocessorTypeUnion], Tuple[Optional[str], Optional[PreprocessorTypeUnion]]] = None

    bert_model: str = 'bert-base-uncased'
    optimization_settings: OptimizationSettings = OptimizationSettings()

    # fields which should be concatenated with the output of BERT before the prediction heads are applied
    supplemental_fields: set = field(default_factory=_default_supplemental_fields)

    # Mapping from [response_key, kind, or corpus_key] to a prediction head. Lookups fall back in that order.
    # This determines how the output from BERT is used to make predictions for each target. Note that whenever two
    # fields map to the same PredictionHeadSettings instance or to two PredictionHeadSettings instances with the same
    # key, the predictions for those fields will be made by a single instance of the prediction head. If two fields map
    # to PredictionHeadSettings instances that have different keys, then the predictions for those fields will be made
    # by two different instances of the prediction head even if the head has the same type. This enables different kinds
    # of parameter-sharing between fields.
    prediction_heads: MutableMapping[str, PredictionHeadSettings] = field(default_factory=_default_prediction_heads)

    # Sequence of [response_key or kind]. Data corresponding to fields specified here will not be put into a
    # batch directly; This allows the system to save significant resources by not padding a tensor for a full
    # sequence of entries when the number of real entries in the tensor is sparse. Mostly, this is for fMRI where
    # each image is huge, and there are relatively few target images relative to the number of tokens in the sequence.
    # Entries here must be paired with an appropriate prediction head, for example FMRIHead or KeyedGroupPooledLinear,
    # which puts into its prediction output dictionary a (field, data_ids) key with the data_ids in the order of
    # the predictions it made for that field. The system will then go fetch the data corresponding to those data_ids
    # and put them into the batch before the losses are computed.
    data_id_in_batch_keys: Sequence[str] = (ResponseKind.ns_froi, ResponseKind.hp_fmri)

    # Sequence of [response_key or kind]. Data corresponding to fields specified here will not be put into the dataset
    # unless those fields are in the loss
    filter_when_not_in_loss_keys: Optional[Sequence[str]] = None

    # mapping from [response_key, kind, or corpus_key] to critic settings; lookups fall back in that order
    critics: MutableMapping[str, Union[CriticSettings, str]] = field(default_factory=_default_critics)

    # fields which should be used to evaluate the loss. All critics specified by the critics setting will be invoked,
    # but only the fields listed here will be considered part of the loss for the purpose of optimization and
    # only those fields will be reported in train_loss / eval_loss. Other critic output is available as metrics for
    # reporting, but is otherwise ignored by training.
    loss_tasks: Union[set, TrainingVariation] = field(default_factory=set)

    # fields which are not in the response_data part of the RawData structure, but which should nevertheless be used
    # as output targets of the model. If a field is already in loss_tasks then it does not need to also be specified
    # here, but this allows non-loss fields to be output/evaluated. This is useful when we want to predict something
    # like syntax that is on the input features rather than in the response data. The use-case for this is a model
    # has been pre-trained to make a prediction on a non-response field, and we want to track how those predictions
    # are changing as we target a different optimization metric.
    non_response_outputs: set = field(default_factory=set)

    # TODO: not currently used. Should revive from ulmfit code
    # can use an extension to render to file, e.g. 'png', or 'show' to plot
    visualize_mode: str = None

    seed: int = 42

    # turn on tqdm at more granular levels
    show_epoch_progress: bool = False
    show_step_progress: bool = False

    # Whether not to use CUDA when available
    no_cuda: bool = False

    def get_split_functions(self, index_run):
        result = dict()
        for key in self.split_functions:
            if self.split_functions[key] is not None:
                result[key] = self.split_functions[key](index_run)
        if len(result) == 0:
            return None
        return result

    def _lookup_critic(self, field_name, data_set):
        if field_name in self.critics:
            return self.critics[field_name]
        if data_set.is_response_data(field_name):
            kind = data_set.response_data_kind(field_name)
            if kind in self.critics:
                return self.critics[kind]
            data_key = data_set.data_set_key_for_field(field_name)
            if data_key in self.critics:
                return self.critics[data_key]
        return CriticKeys.mse

    def get_critic(self, field_name, data_set):
        critic = self._lookup_critic(field_name, data_set)
        if isinstance(critic, str):
            return CriticSettings(critic)
        return critic
