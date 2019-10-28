import fnmatch
import inspect
import os
import warnings
from collections import OrderedDict
import dataclasses
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from .experiments import named_variations, task_hash, match_variation
from .settings import TrainingVariation
from .modeling import CriticMapping
from .result_output import read_predictions

__all__ = [
    'Aggregator',
    'read_variation_results',
    'nan_pearson',
    'regression_handler',
    'class_handler',
    'bincount_axis',
    'make_prediction_handler',
    'ResultQuery',
    'query_results',
    'get_field_predictions',
    'k_vs_k']


class Aggregator:
    def __init__(self):
        """
        Helper class to aggregate metrics over runs etc.
        """
        self._field_values = None
        self._counts = None

    def update(self, result, is_sequence):
        if self._field_values is None:
            self._field_values = OrderedDict()
            self._counts = OrderedDict()
            if dataclasses.is_dataclass(result):
                for field in dataclasses.fields(result):
                    self._field_values[field.name] = list()
                    self._counts[field.name] = list()
            else:
                for field in result:
                    self._field_values[field] = list()
                    self._counts[field] = list()

        if dataclasses.is_dataclass(result):
            result = dataclasses.asdict(result)
        for field in result:
            if field not in self._field_values:
                raise ValueError('Unexpected field in result: {}'.format(field))
            if result[field] is None:
                self._counts[field].append(0)
            elif np.isscalar(result[field]):
                self._field_values[field].append(result[field])
                self._counts[field].append(1)
            elif is_sequence:
                self._field_values[field].extend(result[field])
                self._counts[field].append(len(result[field]))
            else:
                self._field_values[field].append(result[field])
                self._counts[field].append(1)

    def __contains__(self, item):
        return item in self._field_values

    def __iter__(self):
        for k in self._field_values:
            yield k

    def __getitem__(self, item):
        return self._field_values[item]

    def value_dict(self, names=None, fn=None, value_on_key_error=None):
        if names is None:
            if fn is None:
                return OrderedDict(self._field_values)
            return OrderedDict((k, fn(self._field_values[k])) for k in self._field_values)
        if isinstance(names, str):
            names = [names]
        result = OrderedDict()
        for name in names:
            if value_on_key_error is not None and name not in self._field_values:
                result[name] = value_on_key_error
            else:
                result[name] = fn(self._field_values[name]) if fn is not None else self._field_values[name]
        return result

    def values(self, name, fn=None):
        if fn is None:
            return self._field_values[name]
        return fn(self._field_values[name])

    def counts(self, name):
        return self._counts[name]


def read_no_cluster_data(path):
    with np.load(path) as loaded:
        unique_ids = loaded['unique_ids']
        lengths = loaded['lengths']
        data_ids = loaded['data_ids']
        splits = np.cumsum(lengths)[:-1]
        data_ids = np.split(data_ids, splits)
        return unique_ids, data_ids, loaded['data']


def expand_predictions(prediction, cluster_ids):
    is_prediction_1d = len(prediction.shape) == 1
    if is_prediction_1d:
        prediction = np.expand_dims(prediction, 0)
    expanded = np.zeros((prediction.shape[0], np.prod(cluster_ids.shape)), prediction.dtype)
    for idx, c in enumerate(np.unique(cluster_ids)):
        indicator = cluster_ids == c
        expanded[:, indicator] = prediction[:, idx]
    if is_prediction_1d:
        return np.reshape(expanded, cluster_ids.shape)
    else:
        return np.reshape(expanded, (prediction.shape[0],) + cluster_ids.shape)


def _read_variation_parallel_helper(item):
    (result_path, model_path, variation_set_name, variation_hash, index_run, aux_loss,
     compute_scalar, k_vs_k_feature_axes, loss_handler_kwargs) = item
    training_variations, _, _, _, _ = named_variations(variation_set_name)
    training_variation = None
    for v in training_variations:
        if task_hash(v) == variation_hash:
            training_variation = v
            break
    if training_variation is None:
        raise RuntimeError('Bad variation hash')
    output_dir = os.path.join(result_path, variation_set_name, variation_hash)
    model_dir = os.path.join(model_path, variation_set_name, variation_hash, 'run_{}'.format(index_run))
    validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
    if not os.path.exists(validation_npz_path):
        return index_run, None
    output_results_by_name = read_predictions(validation_npz_path)
    run_results = dict()
    for name in output_results_by_name:
        if isinstance(training_variation, TrainingVariation):
            in_training_variation = name in training_variation.loss_tasks
        else:
            in_training_variation = name in training_variation
        if not in_training_variation and name not in aux_loss:
            continue
        no_cluster_path = os.path.join(model_dir, '{}_no_cluster_to_disk.npz'.format(name))
        cluster_id_path = os.path.join(model_dir, 'kmeans_clusters_{}.npy'.format(name))
        cluster_ids = None
        no_cluster_unique_ids = None
        no_cluster_data_ids = None
        no_cluster_data = None
        if os.path.exists(cluster_id_path) and os.path.exists(no_cluster_path):
            cluster_ids = np.load(cluster_id_path)
            no_cluster_unique_ids, no_cluster_data_ids, no_cluster_data = read_no_cluster_data(no_cluster_path)
        output_results = output_results_by_name[name]
        run_aggregated = Aggregator()
        loss = None
        for output_result in output_results:
            if loss is None:
                loss = output_result.critic_type
            else:
                assert (loss == output_result.critic_type)
            if cluster_ids is not None:
                output_result.prediction = expand_predictions(output_result.prediction, cluster_ids)
                output_result.mask = expand_predictions(output_result.mask, cluster_ids)
                index_unique_id = np.where(output_result.unique_id == no_cluster_unique_ids)[0]
                assert(len(index_unique_id) == 1)
                index_unique_id = index_unique_id[0]
                data_ids = no_cluster_data_ids[index_unique_id]
                data_ids = data_ids[data_ids >= 0]
                seen = set()
                unique_data_ids = list()
                for d in data_ids:
                    if d not in seen:
                        unique_data_ids.append(d)
                        seen.add(d)
                assert(len(unique_data_ids) == output_result.target.shape[0])
                output_result.target = np.array(list([no_cluster_data[d] for d in unique_data_ids]))
            run_aggregated.update(output_result, is_sequence=output_result.sequence_type != 'single')

        loss_handler_kwargs = dict(loss_handler_kwargs)
        if isinstance(k_vs_k_feature_axes, dict):
            if name in k_vs_k_feature_axes:
                loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes[name]
            else:
                loss_handler_kwargs['k_vs_k_feature_axes'] = -1
        else:
            loss_handler_kwargs['k_vs_k_feature_axes'] = k_vs_k_feature_axes
        handler = make_prediction_handler(loss, loss_handler_kwargs)
        result_dict = handler(run_aggregated)
        if compute_scalar:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                result_dict = dict((k, np.nanmean(result_dict[k])) for k in result_dict)
        run_results[name] = result_dict
    return index_run, run_results


def read_variation_results(paths, variation_set_name, training_variation, aux_loss, num_runs,
                           compute_scalar=True, k_vs_k_feature_axes=-1, **loss_handler_kwargs):

    task_arguments = [(paths.result_path, paths.model_path, variation_set_name, task_hash(training_variation), i,
                       aux_loss, compute_scalar, k_vs_k_feature_axes, loss_handler_kwargs) for i in range(num_runs)]

    with ThreadPoolExecutor() as ex:
        mapped = ex.map(_read_variation_parallel_helper, task_arguments)
    # mapped = map(_read_variation_parallel_helper, task_arguments)

    has_warned = False
    count_runs = 0
    aggregated = dict()
    for index_run, run_results in mapped:
        if run_results is None:
            if not has_warned:
                print('Warning: results incomplete. Some output files not found')
            has_warned = True
            continue

        count_runs += 1
        for name in run_results:
            if name not in aggregated:
                aggregated[name] = Aggregator()
            aggregated[name].update(run_results[name], is_sequence=False)

    return aggregated, count_runs


def nan_pearson(x, y, axis=0, keepdims=False):
    if not np.array_equal(x.shape, y.shape):
        raise ValueError('x and y must be the same shape')
    if np.isscalar(x):
        raise ValueError('x and y must not be scalar')
    if np.prod(x.shape) == 0:
        result = np.full_like(x, np.nan)
        if x.shape[axis] < 1:
            print(x.shape)
            raise ValueError('x and y must have at least 2 values')
        result = np.take(result, [0], axis=axis)
        if not keepdims:
            result = np.squeeze(result, axis=axis)
        return result
    with warnings.catch_warnings():
        # suppress ddof < 1 for slice
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        x = x - np.nanmean(x, axis=axis, keepdims=True)
        y = y - np.nanmean(y, axis=axis, keepdims=True)
        std_x = np.nanstd(x, axis=axis, keepdims=True, ddof=1)
        std_y = np.nanstd(y, axis=axis, keepdims=True, ddof=1)
    total = np.nansum(
        np.divide(x, std_x, where=std_x != 0) * np.divide(y, std_y, where=std_y != 0),
        axis=axis, keepdims=True)
    counts = np.sum(np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y))), axis=axis, keepdims=True)
    result = np.divide(total, (counts - 1), where=counts > 1)
    result = np.where(np.logical_and(np.logical_and(std_x != 0, std_y != 0), counts > 1),
                      result,
                      np.full_like(result, np.nan))
    if not keepdims:
        result = np.squeeze(result, axis)
    return result


def aggregator_regression_handler(aggregator, k_vs_k_num_samples=0, k_vs_k_k=20, k_vs_k_feature_axes=-1):
    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))

    target_counts = np.array(aggregator.counts('target'))
    prediction_counts = np.array(aggregator.counts('prediction'))
    assert(np.array_equal(target_counts, prediction_counts))

    splits = None
    if np.any(target_counts > 1):
        splits = np.cumsum(target_counts)[:-1]

    return regression_handler(
        predictions, target, mask, k_vs_k_num_samples, k_vs_k_k, k_vs_k_feature_axes, splits, is_single_example=False)


def regression_handler(
        predictions, target, mask,
        k_vs_k_num_samples=0, k_vs_k_k=20, k_vs_k_feature_axes=-1,
        splits=None, is_single_example=False):

    if is_single_example and len(target) > 1:
        seq_r = nan_pearson(predictions, target)
    elif splits is not None:
        seq_r = list()
        for seq_predictions, seq_target in zip(np.split(predictions, splits), np.split(target, splits)):
            seq_r.append(nan_pearson(seq_predictions, seq_target))
        seq_r = np.array(seq_r)
        with warnings.catch_warnings():
            # filter mean of empty slice
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            seq_r = np.nanmean(seq_r, axis=0)
    else:
        seq_r = np.nan

    if len(mask) > 0:
        assert(len(mask) == len(target))
        masked_target = np.where(mask, target, np.nan)
    else:
        masked_target = target

    variance = np.nanvar(masked_target, axis=0)

    mu = np.nanmean(masked_target, axis=0)
    mean_abs_deviation = np.nanmean(np.abs(masked_target - mu))

    mse = np.nanmean(np.square(predictions - masked_target), axis=0)
    mae = np.nanmean(np.abs(predictions - masked_target), axis=0)

    variance = np.where(variance < 1e-8, np.nan, variance)

    result = dict(
        mse=mse,
        mae=mae,
        pove=1 - (mse / variance),
        povu=(mse / variance),
        pode=1 - (mse / mean_abs_deviation),
        podu=(mae / mean_abs_deviation),
        variance=variance,
        mad=mean_abs_deviation,
        r_seq=seq_r)

    if k_vs_k_num_samples > 0:
        k_vs_k_mask = np.reshape(mask, (mask.shape[0], -1))
        # TODO: hack because the mask is sometimes wrong
        k_vs_k_mask = np.full_like(k_vs_k_mask, True)
        if not np.all(k_vs_k_mask == k_vs_k_mask[:, 0:1]):
            raise ValueError('For k_vs_k, the mask must be the same for all features')
        k_vs_k_mask = k_vs_k_mask[:, 0]
        accuracy = k_vs_k(
            predictions[k_vs_k_mask], target[k_vs_k_mask], k=k_vs_k_k, num_samples=k_vs_k_num_samples,
            feature_axes=k_vs_k_feature_axes)
        result['{0}_vs_{0}'.format(k_vs_k_k)] = np.mean(accuracy, axis=0)

    return result


def bincount_axis(x, weights=None, minlength=None, axis=-1):
    """
    Similar to np.bincount, but applied along an axis. By using weights, this function can do sums along contiguous
    segments of an array with variable numbers of elements (in which case x is essentially the label for a segment
    we are summing over). Without weights, this can be used to count the number of elements within a segment.
    See the documentation for np.bincount
    Args:
        x: Input array
        weights: Weights array, same shape as x.
        minlength: A minimum number of bins for the output array, defaults to np.max(x) + 1
        axis: Which axis to apply the bincount over

    Returns:
        out: The result of binning the input array
    """

    if minlength is None:
        minlength = np.max(x) + 1

    if axis < 0:
        axis += len(x.shape)
    transpose_axes = list(range(len(x.shape)))
    transpose_axes = transpose_axes[:axis] + transpose_axes[axis+1:] + [axis]
    x = np.transpose(x, transpose_axes)
    shape = x.shape
    x = np.reshape(x, (-1, x.shape[-1]))
    x += np.expand_dims(minlength * np.arange(x.shape[0]), 1)
    num_bins = minlength * x.shape[0]
    x = np.reshape(x, (-1,))
    if weights is not None:
        weights = np.transpose(weights, transpose_axes)
        weights = np.reshape(weights, (-1,))

    if weights is not None and np.iscomplexobj(weights):
        x_real = np.bincount(x, np.real(weights), num_bins)
        x_imag = np.bincount(x, np.imag(weights), num_bins)
        x = x_real + 1j * x_imag
    else:
        x = np.bincount(x, weights, num_bins)
    x = np.reshape(x, shape[:-1] + (minlength,))
    transpose_axes = list(range(len(x.shape)))
    transpose_axes = transpose_axes[:axis] + transpose_axes[-1:] + transpose_axes[axis:-1]
    return np.transpose(x, transpose_axes)


def aggregator_class_handler(aggregator, pos_weight=None, is_binary=False):

    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))

    return class_handler(predictions, target, mask, pos_weight, is_binary)


def class_handler(predictions, target, mask, pos_weight=None, is_binary=False, is_single_example=False):
    # is_single_example is not currently used; it is there so the caller can pass it without knowing
    # what the loss handler is

    if len(mask) != 0:
        assert(len(mask) == len(target))
        target = np.where(mask, target, np.nan)

    if is_binary:
        max_val = np.maximum(-predictions, 0)
        log_weight = 1
        if pos_weight is not None:
            log_weight = (pos_weight - 1) * target + 1
        cross_entropy = \
            predictions - predictions * target + max_val \
            + np.log(log_weight * (np.exp(-max_val) + np.exp(-predictions - max_val)))

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            cross_entropy = np.nanmean(cross_entropy, axis=0)

        indicator_valid = np.logical_not(np.isnan(target))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            target_positive = np.logical_and(np.greater(target, 0), indicator_valid)
            target_negative = np.logical_and(np.equal(target, 0), indicator_valid)
            predictions_positive = np.logical_and(np.greater_equal(predictions, 0), indicator_valid)
            predictions_negative = np.logical_and(np.less(predictions, 0), indicator_valid)

        true_positive = np.logical_and(predictions_positive, target_positive)
        true_negative = np.logical_and(predictions_negative, target_negative)
        false_positive = np.logical_and(predictions_positive, target_negative)
        false_negative = np.logical_and(predictions_negative, target_positive)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            precision = np.sum(true_positive, axis=0) / (np.sum(true_positive, axis=0) + np.sum(false_positive, axis=0))
        # nothing was classified as positive, define this to be precision 0.
        # where does something weird to scalar values...so we handle it separately
        if np.isscalar(precision):
            if np.isnan(precision):
                precision = np.array([0.])[0]
        else:
            precision = np.where(np.sum(true_positive, axis=0) + np.sum(false_positive, axis=0) == 0, 0., precision)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            recall = np.sum(true_positive, axis=0) / (np.sum(true_positive, axis=0) + np.sum(false_negative, axis=0))
        # either there are no real positive examples (define this to be recall 1),
        # or the predictions are nan (define this to be recall 0).
        if np.isscalar(recall):
            if np.isnan(recall):
                if np.sum(predictions_positive, axis=0) + np.sum(predictions_negative, axis=0) == 0:
                    recall = np.array([0.])[0]
                else:
                    recall = np.array([1.])[0]
        else:
            recall = np.where(np.sum(true_positive, axis=0) + np.sum(false_negative, axis=0) == 0, 1., recall)
            nan_prediction = np.sum(predictions_positive, axis=0) + np.sum(predictions_negative, axis=0) == 0
            recall = np.where(nan_prediction, 0., recall)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            valid_counts = np.sum(indicator_valid, axis=0)
            accuracy = np.divide(
                np.sum(true_positive, axis=0) + np.sum(true_negative, axis=0), valid_counts, where=valid_counts > 0)
            pos_acc = np.divide(np.sum(target_positive, axis=0), valid_counts, where=valid_counts > 0)
            neg_acc = np.divide(np.sum(target_negative, axis=0), valid_counts, where=valid_counts > 0)
        positive_better = np.sum(np.greater_equal(pos_acc, neg_acc)) > np.sum(np.less(pos_acc, neg_acc))
        mode_accuracy = pos_acc if positive_better else neg_acc
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            f1 = 2 * precision * recall / (precision + recall)
        if np.isscalar(f1):
            if precision + recall == 0:
                f1 = 0
        else:
            f1 = np.where(precision + recall == 0, 0, f1)

        poma = np.divide(accuracy, mode_accuracy, where=mode_accuracy != 0)

        return dict(
            xent=cross_entropy,
            acc=accuracy,
            macc=mode_accuracy,
            poma=poma,
            prec=precision,
            rec=recall,
            f1=f1)
    else:
        is_hard_label = len(predictions.shape) > len(target.shape) or target.shape[-1] == 1
        log_sum = logsumexp(predictions, axis=-1)
        # noinspection PyUnresolvedReferences
        log_sum = np.reshape(log_sum, log_sum.shape + (1,))
        log_softmax = predictions - log_sum
        predicted_class = np.argmax(predictions, axis=-1)
        if is_hard_label:
            log_softmax = np.reshape(log_softmax, (-1, log_softmax.shape[-1]))
            indices = np.reshape(target, -1)
            cross_entropy = np.zeros(indices.shape, log_softmax.dtype)
            for result_index, (index, current) in enumerate(zip(indices, log_softmax)):
                if index < 0:
                    cross_entropy[result_index] = np.nan
                else:
                    cross_entropy[result_index] = log_softmax[index]
            cross_entropy = np.reshape(cross_entropy, target.shape)
            cross_entropy = np.nanmean(cross_entropy, axis=0)
            accuracy = np.sum(np.equal(predicted_class, target), axis=0) / np.sum(np.greater_equal(target, 0), axis=0)
            modes = np.argmax(bincount_axis(target, axis=0), axis=0)
            mode_accuracy = np.sum(np.equal(modes, target), axis=0) / np.sum(np.greater_equal(target, 0), axis=0)
        else:
            # soft class labels
            cross_entropy = np.nanmean(np.sum(-log_softmax * target, axis=-1), axis=0)
            max_values = np.max(target, axis=-1, keepdims=True)
            indicator_max = np.isclose(target, max_values)
            count_max = np.sum(indicator_max, axis=-1, keepdims=True)

            partial_credit = np.where(
                indicator_max,
                np.divide(1., count_max, where=count_max > 0),
                np.zeros(indicator_max.shape, target.dtype))

            partial_credit = np.where(count_max == 0, np.nan, partial_credit)
            constant_accuracy = np.nanmean(partial_credit, axis=0)
            mode_accuracy = np.max(constant_accuracy, axis=-1)

            partial_credit = np.reshape(partial_credit, (-1, partial_credit.shape[-1]))
            predicted_class = np.reshape(predicted_class, -1)
            partial_credit = np.array([c[p] for c, p in zip(partial_credit, predicted_class)])
            partial_credit = np.reshape(partial_credit, target.shape[:-1])
            accuracy = np.nanmean(partial_credit, axis=0)

        poma = accuracy / mode_accuracy

        return dict(xent=cross_entropy, acc=accuracy, macc=mode_accuracy, poma=poma)


_prediction_handlers = dataclasses.asdict(CriticMapping(
    mse=aggregator_regression_handler,
    mae=aggregator_regression_handler,
    k_least_se=aggregator_regression_handler,
    k_least_se_on_eval=aggregator_regression_handler,
    k_least_ae=aggregator_regression_handler,
    k_least_ae_on_eval=aggregator_regression_handler,
    pearson=aggregator_regression_handler,
    cross_entropy=aggregator_class_handler,
    binary_cross_entropy=(aggregator_class_handler, dict(is_binary=True)),
    soft_label_cross_entropy=aggregator_class_handler,
    single_mse=aggregator_regression_handler,
    single_mae=aggregator_regression_handler,
    single_k_least_se=aggregator_regression_handler,
    single_k_least_se_on_eval=aggregator_regression_handler,
    single_k_least_ae=aggregator_regression_handler,
    single_k_least_ae_on_eval=aggregator_regression_handler,
    single_pearson=aggregator_regression_handler,
    single_cross_entropy=aggregator_class_handler,
    single_binary_cross_entropy=(aggregator_class_handler, dict(is_binary=True)),
    single_soft_label_cross_entropy=aggregator_class_handler), dict_factory=OrderedDict)


_no_aggregator_prediction_handlers = dataclasses.asdict(CriticMapping(
    mse=regression_handler,
    mae=regression_handler,
    k_least_se=regression_handler,
    k_least_se_on_eval=regression_handler,
    k_least_ae=regression_handler,
    k_least_ae_on_eval=regression_handler,
    pearson=regression_handler,
    cross_entropy=class_handler,
    binary_cross_entropy=(class_handler, dict(is_binary=True)),
    soft_label_cross_entropy=class_handler,
    single_mse=regression_handler,
    single_mae=regression_handler,
    single_k_least_se=regression_handler,
    single_k_least_se_on_eval=regression_handler,
    single_k_least_ae=regression_handler,
    single_k_least_ae_on_eval=regression_handler,
    single_pearson=regression_handler,
    single_cross_entropy=class_handler,
    single_binary_cross_entropy=(class_handler, dict(is_binary=True)),
    single_soft_label_cross_entropy=class_handler), dict_factory=OrderedDict)


def make_prediction_handler(which_loss, loss_kwargs=None, using_aggregator=True):
    handler_map = _prediction_handlers if using_aggregator else _no_aggregator_prediction_handlers
    if which_loss not in handler_map:
        raise ValueError('Unknown value for which_loss. Known values are: {}'.format(handler_map.keys()))
    factory = handler_map[which_loss]
    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
    if isinstance(factory, tuple):
        factory, factory_kwargs = factory
        loss_kwargs.update(factory_kwargs)
    factory_signature = inspect.signature(factory)
    bad_keys = [k for k in loss_kwargs if k not in factory_signature.parameters]
    for k in bad_keys:
        del loss_kwargs[k]
    return partial(factory, **loss_kwargs)


def k_vs_k(predictions, target, k=20, num_samples=1000, pair_examples=None, feature_axes=-1):
    """
    Estimates how accurate a classifier would be if the classifier chose between
    1) the concatenated predictions of (e.g.) brain activity for the k examples corresponding to the true k examples
    and
    2) the concatenated predictions of k distractor examples
    by looking to see whether the vector formed by the k true or k distractor example predictions is closer to the
    vector formed by the k true target examples
    Args:
        predictions: The predictions output by a model. Should have shape (examples, ..., features) where examples is
            typically 1 per word, and features is typically voxels in fMRI or sensors in MEG. Accuracy is scored
            separately for each feature on the feature axis.
        target:  The true values for the features. Must have the same shape as predictions
        k: How many examples to combine together for the classifier
        num_samples: The number of samples of k-concatenations to use for estimating accuracy
        pair_examples: If present, must be a 1-D array with len(pair_examples) == len(predictions). pair_examples[i]
            gives the id of a group, or a negative value indicates that the word at index i should never be used. When
            a group id is given, the distractor will contain a word with the same group id at the position in the
            concatenated vector where word i is. Letting distractor_word_indices be the set of indices used in the
            distractor, and true_word_indices be the indices used in the true k examples, then:
            pair_examples[true_word_indices[j]] == pair_examples[distractor_word_indices[j]]
        feature_axes: An int or tuple indicating which axes to compute accuracy for. Axes which are neither the example
            axis (axis 0) or feature_axes will be used to make joint predictions.

    Returns:
        An accuracy array of shape
        (num_samples, predictions.shape[feature_axis_0], predictions.shape[feature_axis_1], ...)
    """

    if not np.array_equal(predictions.shape, target.shape):
        raise ValueError('predictions and target must have the same shape')

    if np.isscalar(feature_axes):
        feature_axes = [feature_axes]
    feature_axes = list(sorted([f if f > 0 else len(predictions.shape) + f for f in feature_axes]))
    transpose_axes = [i for i in range(len(predictions.shape)) if i not in feature_axes] + feature_axes
    if np.array_equal(transpose_axes, np.arange(len(predictions.shape))):
        transpose_axes = None
    if transpose_axes is not None:
        predictions = np.transpose(predictions, transpose_axes)
        target = np.transpose(target, transpose_axes)

    value_shape = predictions.shape[-len(feature_axes):]

    predictions = np.reshape(
        predictions, (predictions.shape[0], int(np.prod(predictions.shape[1:-len(feature_axes)])), -1,))
    target = np.reshape(
        target, (target.shape[0], int(np.prod(target.shape[1:-len(feature_axes)])), -1))

    # predictions, target data with the same shape: (words, ..., features)
    # k = how many words to classify at once
    # num_samples = how many words to classify
    accuracy = np.full((num_samples, target.shape[-1]), np.nan)

    if pair_examples is not None and len(pair_examples) > 0:
        if len(pair_examples) != len(predictions):
            raise ValueError('When specified, pair_examples must have 1 value per example')
        predictions = predictions[pair_examples >= 0]
        target = target[pair_examples >= 0]
        pair_examples = pair_examples[pair_examples >= 0]

    for index_sample in range(num_samples):
        indices_true = np.random.choice(len(target), k)
        sample_target = target[indices_true]
        sample_predictions_correct = predictions[indices_true]
        if pair_examples is not None and len(pair_examples) > 0:
            indices_distractor = _find_restricted_distractor_indices(indices_true, pair_examples)
        else:
            indices_distractor = np.random.choice(len(target), k)
        sample_predictions_incorrect = predictions[indices_distractor]

        sample_target = np.reshape(sample_target, (-1, sample_target.shape[-1]))
        sample_predictions_correct = np.reshape(
            sample_predictions_correct, (-1, sample_predictions_correct.shape[-1]))
        sample_predictions_incorrect = np.reshape(
            sample_predictions_incorrect, (-1, sample_predictions_incorrect.shape[-1]))

        distance_correct = np.sum((sample_target - sample_predictions_correct) ** 2, axis=0)
        distance_incorrect = np.sum((sample_target - sample_predictions_incorrect) ** 2, axis=0)
        accuracy[index_sample] = \
            (distance_correct < distance_incorrect) * 1.0 + (distance_correct == distance_incorrect) * 0.5

    return np.reshape(accuracy, (accuracy.shape[0],) + value_shape)


def _find_restricted_distractor_indices(indices_true, pair_examples):
    indices_distractor = np.zeros_like(indices_true)
    for i, w in enumerate(indices_true):
        id_group = pair_examples[w]
        other_words = np.where(pair_examples == id_group)[0]
        assert len(other_words) > 1
        indices_distractor[i] = np.random.permutation(np.setdiff1d(other_words, np.array(w)))[0]
    return indices_distractor


@dataclasses.dataclass
class ResultQuery:
    variation_set_name: str
    metric: str
    key: str
    training_variation: Optional[str] = None
    second_variation_set_name: Optional[str] = None
    second_training_variation: Optional[str] = None

    def as_dict_with_combined_second(self, sep=':', key_shorten_fn=None):
        result = dataclasses.asdict(self, dict_factory=OrderedDict)
        if key_shorten_fn is not None:
            result['key'] = key_shorten_fn(result['key'])
            result['training_variation'] = tuple(key_shorten_fn(x) for x in result['training_variation'])
            if result['second_training_variation'] is not None:
                result['second_training_variation'] = tuple(
                    key_shorten_fn(x) for x in result['second_training_variation'])
        result['combined_variation_set_name'] = result['variation_set_name']
        result['combined_training_variation'] = result['training_variation']
        if self.second_variation_set_name is not None and self.second_variation_set_name != self.variation_set_name:
            result['combined_variation_set_name'] = \
                result['variation_set_name'] + sep + result['second_variation_set_name']
        if self.second_variation_set_name is not None and self.second_training_variation != self.training_variation:
            result['combined_training_variation'] = \
                str(result['training_variation']) + sep + str(result['second_training_variation'])
        return result


def is_metric_k_vs_k(metric):
    split_metric = metric.split('_')
    if len(split_metric) == 3 and split_metric[1] == 'vs' and split_metric[0] == split_metric[2]:
        return True


def _run_query(item):
    paths, variation_set_name, variation_hash, needs_k_vs_k, compute_scalar, loss_handler_kwargs = item
    variations, _, num_runs, _, aux_loss = named_variations(variation_set_name)
    variation = None
    for v in variations:
        if task_hash(v) == variation_hash:
            variation = v
            break
    if variation is None:
        raise RuntimeError('Bad hash: unable to find match')
    current_loss_handler_kwargs = loss_handler_kwargs
    if needs_k_vs_k:
        if 'k_vs_k_num_samples' not in current_loss_handler_kwargs:
            current_loss_handler_kwargs = dict(current_loss_handler_kwargs)
            current_loss_handler_kwargs['k_vs_k_num_samples'] = 1000
    query_key = variation_set_name, variation_hash
    result = read_variation_results(
        paths, variation_set_name, variation, aux_loss, num_runs, compute_scalar=compute_scalar,
        **current_loss_handler_kwargs)
    return query_key, result


def query_results(paths, result_queries, compute_scalar=False, **loss_handler_kwargs):
    needs_k_vs_k = set()
    query_set = set()
    for result_query in result_queries:
        training_variations, _, num_runs, _, aux_loss = named_variations(result_query.variation_set_name)
        query_training_variation_hash = task_hash(result_query.training_variation) \
            if result_query.training_variation is not None else None
        for training_variation in training_variations:
            training_variation_hash = task_hash(training_variation)
            if query_training_variation_hash is None or query_training_variation_hash == training_variation_hash:
                result_key = result_query.variation_set_name, training_variation_hash
                query_set.add(result_key)
                if result_query.metric == 'k_vs_k':
                    needs_k_vs_k.add(result_key)
                if result_query.second_variation_set_name is not None:
                    second_variation = result_query.second_training_variation \
                        if result_query.second_training_variation is not None else training_variation
                    second_key = result_query.second_variation_set_name, task_hash(second_variation)
                    query_set.add(second_key)
                    if result_query.metric == 'k_vs_k':
                        needs_k_vs_k.add(second_key)

    query_items = list()
    for query_key in query_set:
        variation_set_name, variation_hash = query_key
        query_items.append(
            (paths, variation_set_name, variation_hash, query_key in needs_k_vs_k, compute_scalar, loss_handler_kwargs))

    cache = dict()
    result = list()
    with ProcessPoolExecutor() as ex:
        q_results = ex.map(_run_query, query_items)
    for query_key, query_result in q_results:
        cache[query_key] = query_result

    for result_query in result_queries:
        matched = False
        training_variations, _, num_runs, _, aux_loss = named_variations(result_query.variation_set_name)
        query_training_variation_hash = task_hash(result_query.training_variation) \
            if result_query.training_variation is not None else None
        for training_variation in training_variations:
            training_variation_hash = task_hash(training_variation)
            if query_training_variation_hash is None or query_training_variation_hash == training_variation_hash:
                cache_key = result_query.variation_set_name, training_variation_hash
                aggregated, count_runs = cache[cache_key]
                for aggregated_key in aggregated:
                    if fnmatch.fnmatch(aggregated_key, result_query.key):
                        result_query = dataclasses.replace(result_query, key=aggregated_key)
                        if result_query.metric == 'k_vs_k':
                            values = None
                            for metric in aggregated[result_query.key]:
                                if is_metric_k_vs_k(metric):
                                    values = aggregated[result_query.key].values(metric)
                                    break
                            if values is None:
                                raise ValueError('k_vs_k not found')
                        else:
                            values = aggregated[result_query.key].values(result_query.metric)
                        if result_query.training_variation is None:
                            result_query = dataclasses.replace(result_query, training_variation=training_variation)
                        matched = True
                        result.append((result_query, np.asarray(values)))
        if not matched:
            raise ValueError('No match for query: {}'.format(result_query))
    for idx_result in range(len(result)):
        if result[idx_result][0].second_variation_set_name is not None:
            result_query, first_values = result[idx_result]
            training_variations, _, num_runs, _, aux_loss = named_variations(result_query.second_variation_set_name)
            if result_query.second_training_variation is None:
                result_query = dataclasses.replace(
                    result_query, second_training_variation=result_query.training_variation)
            query_training_variation_hash = task_hash(result_query.second_training_variation)
            for training_variation in training_variations:
                training_variation_hash = task_hash(training_variation)
                if query_training_variation_hash == training_variation_hash:
                    cache_key = result_query.second_variation_set_name, training_variation_hash
                    aggregated, count_runs = cache[cache_key]
                    if result_query.metric == 'k_vs_k':
                        second_values = None
                        for metric in aggregated[result_query.key]:
                            if is_metric_k_vs_k(metric):
                                second_values = aggregated[result_query.key].values(metric)
                                break
                        if second_values is None:
                            raise ValueError('k_vs_k not found')
                    else:
                        second_values = aggregated[result_query.key].values(result_query.metric)
                    result[idx_result] = (result_query, first_values, np.asarray(second_values))
                    break  # break out of looping over training variations
            if len(result[idx_result]) == 2:
                raise ValueError('No match for query: {}'.format(result_query))
    return result


def get_field_predictions(
        paths_obj, variation_set_name, training_variation, field_name, index_run=None, pre_matched=False):
    num_runs = None
    if index_run is None:
        _, _, num_runs, _, _ = named_variations(variation_set_name)
    if not pre_matched:
        training_variation = match_variation(variation_set_name, training_variation)
    output_dir = os.path.join(paths_obj.result_path, variation_set_name, task_hash(training_variation))
    aggregator = None
    run_iterable = (index_run,) if index_run is not None else range(num_runs)
    for index_run in run_iterable:
        validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
        if not os.path.exists(validation_npz_path):
            raise ValueError('Path does not exist: {}'.format(validation_npz_path))
        output_results = read_predictions(validation_npz_path)
        field_results = output_results[field_name]
        aggregator = Aggregator()
        for result in field_results:
            aggregator.update(result, is_sequence=result.sequence_type != 'single')

    target = np.array(aggregator.values('target'))
    predictions = np.array(aggregator.values('prediction'))
    mask = np.array(aggregator.values('mask'))
    ids = np.array(aggregator.values('unique_id'))
    masked_target = np.where(mask, target, np.nan)
    return predictions, masked_target, ids
