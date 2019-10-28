import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from dataclasses import replace, asdict
from typing import Optional
import numpy as np
from scipy.stats import boxcox
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import sosfilt
from sklearn.decomposition import PCA

from .input_features import InputFeatures, KindData

__all__ = [
    'PreprocessBoxcox',
    'PreprocessLog',
    'PreprocessDetrend',
    'PreprocessDiscretize',
    'PreprocessBaseline',
    'PreprocessFeatureStandardize',
    'PreprocessFeatureNormalize',
    'PreprocessSequenceStandardize',
    'PreprocessDiff',
    'PreprocessStandardize',
    'PreprocessMakeBinary',
    'PreprocessNanMean',
    'PreprocessPCA',
    'PreprocessNanGeometricMean',
    'PreprocessClip',
    'PreprocessGaussianBlur',
    'PreprocessCompress',
    'PreprocessSoSFilter',
    'PreprocessSqueeze',
    'PreprocessKMeans',
    'PreprocessMiniBatchKMeans',
    'PreprocessRandomPair',
    'preprocess_fork_no_cluster_to_disk']


def _fit_boxcox(item):
    data, indicator_train = item
    if indicator_train is not None:
        data = data[indicator_train]
    data = data[np.logical_not(np.isnan(data))]
    _, transform_lambda = boxcox(data)
    return transform_lambda


def _apply_boxcox(data, transform_lambdas):
    data = np.copy(data)
    for idx, transform_lambda in enumerate(transform_lambdas):
        indicator_valid = np.logical_not(np.isnan(data[:, idx]))
        data[indicator_valid, idx] = boxcox(data[indicator_valid, idx], transform_lambda)
    return data


def _indicator_from_examples(data_size, examples, stop_mode=None):
    result = np.full(data_size, False)
    for ex in examples:
        data_ids = ex.data_ids
        if stop_mode is not None:
            if stop_mode == 'content':
                data_ids = np.where(ex.is_stop, -1, data_ids)
            elif stop_mode == 'stop':
                data_ids = np.where(ex.is_stop, data_ids, -1)
            else:
                raise ValueError('Unknown value for stop_mode: {}'.format(stop_mode))
        data_ids = data_ids[data_ids >= 0]
        result[data_ids] = True
    return result


def _unsorted_group_by(items, group_by_fn):
    groups = dict()
    for item in items:
        key = group_by_fn(item)
        if key not in groups:
            groups[key] = list()
        groups[key].append(item)
    for key in sorted(groups):
        yield key, groups[key]


def _parallel_column_map(fit_fn, apply_fn, data, indicator_fit=None):
    shape = data.shape
    data = np.reshape(data, (data.shape[0], -1))
    with ProcessPoolExecutor() as ex:
        fit_result = list(ex.map(fit_fn, [(data[:, i], indicator_fit) for i in range(data.shape[1])]))
    assert(len(fit_result) == data.shape[1])
    data = apply_fn(data, fit_result)
    return np.reshape(data, shape)


class PreprocessBoxcox:

    def __call__(self, loaded_data_tuple, metadata, random_state):
        data = _parallel_column_map(_fit_boxcox, _apply_boxcox, loaded_data_tuple.data)
        return replace(loaded_data_tuple, data=data)


class PreprocessLog:

    def __init__(self, min_value: float = -20.):
        self.min_value = min_value

    def __call__(self, loaded_data_tuple, metadata, random_state):
        isnan = np.isnan(loaded_data_tuple.data)
        data = np.where(isnan, 1, loaded_data_tuple.data)
        if np.any(np.less(data, 0)):
            raise ValueError('Values must be >= 0')
        data = np.log(np.maximum(data, np.power(np.e, self.min_value)))
        return replace(loaded_data_tuple, data=np.where(isnan, np.nan, data))


def _lin_regress(item):
    from scipy.stats import linregress
    y, indicator_train = item
    x = np.arange(len(y))

    if indicator_train is not None:
        x = x[indicator_train]
        y = y[indicator_train]

    indicator_valid = np.logical_not(np.isnan(y))
    x = x[indicator_valid]
    y = y[indicator_valid]

    if len(x) == 0:
        return 0, 0

    m, b, _, _, _ = linregress(x, y)
    return m, b


def _remove_lin_regress(data, p):
    p = np.concatenate(list(np.expand_dims(p_col, 1) for p_col in p), axis=1)
    #      (1, num_columns)            (num_rows, 1)
    lines = np.reshape(p[0], (1, -1)) * np.reshape(np.arange(len(data)), (-1, 1)) + np.reshape(p[1], (1, -1))
    lines = np.reshape(lines, data.shape)
    return data - lines


class PreprocessDetrend:

    def __init__(
            self,
            stop_mode: Optional[str] = None,
            metadata_example_group_by: str = None,
            train_on_all: bool = False):
        self.stop_mode = stop_mode
        self.metadata_example_group_by = metadata_example_group_by
        self.train_on_all = train_on_all

    @staticmethod
    def _detrend(arr, indicator_train):
        return _parallel_column_map(_lin_regress, _remove_lin_regress, arr, indicator_train)

    def __call__(self, loaded_data_tuple, metadata, random_state):
        train_examples = loaded_data_tuple.train
        if self.train_on_all:
            train_examples = chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test)
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), train_examples, self.stop_mode)

        if self.metadata_example_group_by is not None:
            if metadata is None or self.metadata_example_group_by not in metadata:
                raise ValueError('metadata_example_group_by {} not found in metadata'.format(
                    self.metadata_example_group_by))
            data = np.copy(loaded_data_tuple.data)
            grouped_examples = _unsorted_group_by(
                chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test),
                lambda ex: metadata[self.metadata_example_group_by][ex.unique_id])
            for group, group_examples in grouped_examples:
                indicator_group = _indicator_from_examples(len(data), group_examples)
                group_data = data[indicator_group]
                group_indicator_train = indicator_train[indicator_group] if indicator_train is not None else None
                group_data = PreprocessDetrend._detrend(group_data, group_indicator_train)
                data[indicator_group] = group_data
        else:
            data = PreprocessDetrend._detrend(loaded_data_tuple.data, indicator_train)

        return replace(loaded_data_tuple, data=data)


class PreprocessKMeans:

    def __init__(self, num_clusters, stop_mode=None, transform_fn=None, n_init=10):
        self.num_clusters = num_clusters
        self.n_init = n_init
        self.stop_mode = stop_mode
        self.transform_fn = transform_fn
        self.output_model_path = None
        self.data_key = None

    def set_model_path(self, output_model_path, data_key):
        self.output_model_path = output_model_path
        self.data_key = data_key

    def __call__(self, loaded_data_tuple, metadata, random_state):
        from sklearn.cluster import KMeans
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), loaded_data_tuple.train, self.stop_mode)
        valid_train_values = loaded_data_tuple.data[indicator_train]
        if self.transform_fn is not None:
            valid_train_values = self.transform_fn(valid_train_values)
        k_means = KMeans(self.num_clusters, n_init=self.n_init, random_state=random_state)
        print('clustering...', end='', flush=True)
        clusters = k_means.fit_predict(
            np.transpose(np.reshape(valid_train_values, (valid_train_values.shape[0], -1))))
        cluster_means = np.full((loaded_data_tuple.data.shape[0], self.num_clusters), np.nan)
        data = np.reshape(loaded_data_tuple.data, (len(loaded_data_tuple.data), -1))
        for index_cluster, cluster in enumerate(np.unique(clusters)):
            indicator_cluster = clusters == cluster
            cluster_means[:, index_cluster] = np.mean(data[:, indicator_cluster], axis=1)

        clusters = np.reshape(clusters, valid_train_values.shape[1:])
        if not os.path.exists(self.output_model_path):
            os.makedirs(self.output_model_path)
        np.save(os.path.join(self.output_model_path, 'kmeans_clusters_{}.npy'.format(self.data_key)), clusters)
        print('done')

        return replace(loaded_data_tuple, data=cluster_means)


class PreprocessMiniBatchKMeans:

    def __init__(self, num_clusters, stop_mode=None, transform_fn=None, n_init=10, batch_size=100):
        self.num_clusters = num_clusters
        self.n_init = n_init
        self.batch_size = batch_size
        self.stop_mode = stop_mode
        self.transform_fn = transform_fn
        self.output_model_path = None
        self.data_key = None

    def set_model_path(self, output_model_path, data_key):
        self.output_model_path = output_model_path
        self.data_key = data_key

    def __call__(self, loaded_data_tuple, metadata, random_state):
        from sklearn.cluster import MiniBatchKMeans
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), loaded_data_tuple.train, self.stop_mode)
        valid_train_values = loaded_data_tuple.data[indicator_train]
        if self.transform_fn is not None:
            valid_train_values = self.transform_fn(valid_train_values)
        k_means = MiniBatchKMeans(
            self.num_clusters, n_init=self.n_init, batch_size=self.batch_size, random_state=random_state)
        print('clustering...', end='', flush=True)
        clusters = k_means.fit_predict(
            np.transpose(np.reshape(valid_train_values, (valid_train_values.shape[0], -1))))
        cluster_means = np.full((loaded_data_tuple.data.shape[0], self.num_clusters), np.nan)
        data = np.reshape(loaded_data_tuple.data, (len(loaded_data_tuple.data), -1))
        for index_cluster, cluster in enumerate(np.unique(clusters)):
            indicator_cluster = clusters == cluster
            cluster_means[:, index_cluster] = np.mean(data[:, indicator_cluster], axis=1)

        clusters = np.reshape(clusters, valid_train_values.shape[1:])
        if not os.path.exists(self.output_model_path):
            os.makedirs(self.output_model_path)
        np.save(os.path.join(self.output_model_path, 'kmeans_clusters_{}.npy'.format(self.data_key)), clusters)
        print('done')

        return replace(loaded_data_tuple, data=cluster_means)


class PreprocessDiscretize:

    # noinspection PyShadowingBuiltins
    def __init__(self, bins=10, range=None, use_one_hot=True):
        self.bins = bins
        self.range = range
        self.use_one_hot = use_one_hot

    # noinspection PyShadowingBuiltins
    def __call__(self, loaded_data_tuple, metadata, random_state):
        bin_edges = np.histogram_bin_edges(loaded_data_tuple.data, self.bins, range)
        if np.isscalar(self.bins):
            bin_edges = bin_edges[1:]
        data = np.digitize(loaded_data_tuple.data, bin_edges, right=True)
        if self.use_one_hot:
            one_hot = np.zeros(data.shape + (len(bin_edges) + 1,), data.dtype)
            one_hot = np.reshape(one_hot, (-1, one_hot.shape[-1]))
            for idx, bin in enumerate(np.reshape(data, -1)):
                one_hot[idx, bin] = 1
            data = np.reshape(one_hot, data.shape + (one_hot.shape[-1],))
        return replace(loaded_data_tuple, data=data)


class PreprocessBaseline:

    def __init__(self, num_baseline):
        """
        Computes a running mean using a window of num_baseline values and subtracts this running mean
        from the data. This completely ignores example boundaries. Validation/test examples are removed if the baselines
        from those examples would overlap with train examples
        """
        self.num_baseline = num_baseline

    def _find_max_mins(self, examples, keep_if_greater_than=None):
        if examples is None:
            return None, None, examples
        data_ids = np.concatenate([ex.data_ids for ex in examples])
        data_ids = data_ids[data_ids >= 0]
        max_id = np.max(data_ids)
        min_id = np.min(data_ids)
        if keep_if_greater_than is None:
            return max_id, min_id, examples
        clean = list()
        for ex in examples:
            ex_min = np.min(ex.data_idx[ex.data_ids >= 0])
            if ex_min - self.num_baseline > keep_if_greater_than:
                clean.append(ex)
        return max_id, min_id, clean

    def _compute_baseline(self, arr):
        indicator_nan = np.isnan(arr)
        result = np.cumsum(np.where(indicator_nan, 0, arr), axis=0)
        if len(result) > self.num_baseline:
            result[self.num_baseline:] = result[self.num_baseline:] - result[:-self.num_baseline]
        counts = np.cumsum(np.logical_not(indicator_nan))
        if len(counts) > self.num_baseline:
            counts[self.num_baseline:] = counts[self.num_baseline:] - counts[:-self.num_baseline]
        return result / counts

    def _subtract_baseline(self, examples, data):
        if examples is None:
            return
        data_ids = np.concatenate([ex.data_ids for ex in examples])
        data_ids = data_ids[data_ids >= 0]
        baseline = self._compute_baseline(data[data_ids])
        data[data_ids] = data[data_ids] - baseline

    def __call__(self, loaded_data_tuple, metadata, random_state):
        max_train, _, _ = self._find_max_mins(loaded_data_tuple.train)
        _, _, clean_validation = self._find_max_mins(loaded_data_tuple.validation, keep_if_greater_than=max_train)
        loaded_data_tuple = replace(loaded_data_tuple, validation=clean_validation)
        _, _, clean_test = self._find_max_mins(loaded_data_tuple.test, keep_if_greater_than=max_train)
        loaded_data_tuple = replace(loaded_data_tuple, test=clean_test)

        data = np.copy(loaded_data_tuple.data)

        self._subtract_baseline(loaded_data_tuple.train, data)
        self._subtract_baseline(loaded_data_tuple.validation, data)
        self._subtract_baseline(loaded_data_tuple.test, data)

        return replace(loaded_data_tuple, data=data)


class PreprocessFeatureStandardize:

    def __init__(self):
        pass

    def __call__(self, loaded_data_tuple, metadata, random_state):
        d = np.reshape(loaded_data_tuple.data, (loaded_data_tuple.data.shape[0], -1))
        d = (d - np.nanmean(d, axis=1, keepdims=True)) / np.nanstd(d, axis=1, keepdims=True)
        return replace(loaded_data_tuple, data=np.reshape(d, loaded_data_tuple.data.shape))


class PreprocessFeatureNormalize:

    def __init__(self):
        pass

    def __call__(self, loaded_data_tuple, metadata, random_state):
        d = np.reshape(loaded_data_tuple.data, (loaded_data_tuple.data.shape[0], -1))
        n = np.nansum(np.abs(d), axis=1, keepdims=True)
        d = np.divide(d, n, where=n != 0)
        return replace(loaded_data_tuple, data=np.reshape(d, loaded_data_tuple.data.shape))


class PreprocessSequenceStandardize:

    def __init__(self, stop_mode):
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple, metadata, random_state):

        data = np.copy(loaded_data_tuple.data)
        for ex in chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test):

            data_indices = ex.data_ids
            compute_indices = data_indices
            if self.stop_mode == 'content':
                compute_indices = np.where(ex.is_stop, -1, compute_indices)
            elif self.stop_mode == 'stop':
                compute_indices = np.where(ex.is_stop, compute_indices, -1)
            elif self.stop_mode is not None:
                raise ValueError('Unable to understand stop_mode: {}'.format(self.stop_mode))

            data_indices = data_indices[data_indices >= 0]
            compute_indices = compute_indices[compute_indices >= 0]

            compute = data[compute_indices]
            mean = np.mean(compute, axis=0, keepdims=True)
            std = np.std(compute, axis=0, keepdims=True)

            data[data_indices] = (data[data_indices] - mean) / std

        return replace(loaded_data_tuple, data=data)


class PreprocessDiff:

    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def __call__(self, loaded_data_tuple, metadata, random_state):
        data = loaded_data_tuple.data
        padding = np.full((1,) + data.shape[1:], self.fill_value, data.dtype)
        return replace(loaded_data_tuple, data=np.concatenate([padding, np.diff(data, axis=0)]))


class PreprocessPCA:

    def __init__(
            self,
            feature_axis: int = 1,  # features with respect to PCA, e.g. subjects
            stop_mode: Optional[str] = None):
        self.feature_axis = feature_axis
        self.stop_mode = stop_mode

    def __call__(self, loaded_data_tuple, metadata, random_state):

        indicator_train = _indicator_from_examples(len(loaded_data_tuple.data), loaded_data_tuple.train, self.stop_mode)

        all_values = loaded_data_tuple.data
        # -> (samples, ..., features)
        all_values = np.moveaxis(all_values, self.feature_axis, -1)
        result_shape = all_values.shape[:-1]
        # -> (samples, task, features)
        all_values = np.reshape(
            all_values,
            (all_values.shape[0], int(np.prod(all_values.shape[1:-1])), all_values.shape[-1]))
        # -> (task, samples, features)
        all_values = np.transpose(all_values, (1, 0, 2))
        result = list()
        for current in all_values:
            pca = PCA(n_components=1)
            train_values = current[indicator_train]
            pca.fit(train_values)
            result.append(pca.transform(current))

        # -> (task, samples, 1)
        result = np.array(result)
        # -> (samples, task, 1)
        result = np.transpose(result, (1, 0, 2))
        # -> (samples, ...)
        result = np.reshape(result, result_shape)

        return replace(loaded_data_tuple, data=result)


class PreprocessStandardize:

    def __init__(
            self,
            average_axis: Optional[int] = 1,
            stop_mode: Optional[str] = None,
            metadata_example_group_by: Optional[str] = None,
            train_on_all: Optional[bool] = False,
            use_absolute: Optional[bool] = False):
        self.stop_mode = stop_mode
        self.average_axis = average_axis
        self.metadata_example_group_by = metadata_example_group_by
        self.train_on_all = train_on_all
        self.use_absolute = use_absolute

    def _standardize(self, data, indicator_train):

        valid_train_values = data
        if indicator_train is not None:
            valid_train_values = data[indicator_train]

        if len(valid_train_values) == 0:
            raise ValueError('No training values')

        with warnings.catch_warnings():
            # catch 'Mean of emtpy slice' warning here; for example if a participant has no data
            # within the training examples
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            pre_average_mean = np.nanmean(valid_train_values, axis=0, keepdims=True)
            if self.use_absolute:
                pre_average_std = np.nanmean(np.abs(valid_train_values - pre_average_mean), axis=0, keepdims=True)
            else:
                pre_average_std = np.nanstd(valid_train_values, axis=0, keepdims=True)

        transformed_data = np.divide(data - pre_average_mean, pre_average_std, where=pre_average_std != 0)

        if self.average_axis is not None:
            standardized_train_values = transformed_data
            if indicator_train is not None:
                standardized_train_values = standardized_train_values[indicator_train]
            with warnings.catch_warnings():
                # catch 'Mean of emtpy slice' warning here
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                standardized_train_values = np.nanmean(standardized_train_values, axis=self.average_axis)
                transformed_data = np.nanmean(transformed_data, axis=self.average_axis)
            post_average_mean = np.nanmean(standardized_train_values, axis=0, keepdims=True)
            if self.use_absolute:
                post_average_std = np.nanmean(
                    np.abs(standardized_train_values - post_average_mean), axis=0, keepdims=True)
            else:
                post_average_std = np.nanstd(standardized_train_values, axis=0, keepdims=True)
            transformed_data = np.divide(
                transformed_data - post_average_mean, post_average_std, where=post_average_std != 0)

        return transformed_data

    def __call__(self, loaded_data_tuple, metadata, random_state):

        train_examples = loaded_data_tuple.train
        if self.train_on_all:
            train_examples = chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test)
        indicator_train = _indicator_from_examples(
            len(loaded_data_tuple.data), train_examples, self.stop_mode)

        if self.metadata_example_group_by is not None:
            if metadata is None or self.metadata_example_group_by not in metadata:
                raise ValueError('metadata_example_group_by not found: {}'.format(self.metadata_example_group_by))
            if self.average_axis is None:   # we're going to keep the shape
                data = np.full(loaded_data_tuple.data.shape, np.nan)
            else:
                data = np.full(
                    loaded_data_tuple.data.shape[:self.average_axis]
                    + loaded_data_tuple.data.shape[(self.average_axis + 1):], np.nan)
            grouped_examples = _unsorted_group_by(
                chain(loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test),
                lambda ex: metadata[self.metadata_example_group_by][ex.unique_id])
            for group, group_examples in grouped_examples:
                indicator_group = _indicator_from_examples(len(data), group_examples)
                group_data = loaded_data_tuple.data[indicator_group]
                if len(group_data) == 0:
                    continue
                group_indicator_train = indicator_train[indicator_group] if indicator_train is not None else None
                group_data = self._standardize(group_data, group_indicator_train)
                data[indicator_group] = group_data
        else:
            data = self._standardize(loaded_data_tuple.data, indicator_train)

        return replace(loaded_data_tuple, data=data)


class PreprocessMakeBinary:

    def __init__(self, threshold, dtype=None, policy_equal='random', policy_nan='propagate'):
        self.threshold = threshold
        if dtype is None:
            if policy_nan == 'propagate':
                self.dtype = np.float32
            else:
                self.dtype = np.int32
        else:
            self.dtype = dtype
        self.policy_equal = policy_equal
        self.policy_nan = policy_nan

    def __call__(self, loaded_data_tuple, metadata, random_state):
        data = loaded_data_tuple.data
        indicator_nan = np.isnan(data)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if self.policy_equal == 'greater':
                indicator = data >= self.threshold
            elif self.policy_equal == 'less':
                indicator = data > self.threshold
            elif self.policy_equal == 'random':
                indicator = np.logical_or(
                    data > self.threshold,
                    np.logical_and(data == self.threshold, random_state.randint(0, 2, data.shape) == 1))
            else:
                raise ValueError('Unrecognized policy_equal: {}'.format(self.policy_equal))

        if self.policy_nan == 'greater':
            indicator = np.where(indicator_nan, True, indicator)
        elif self.policy_nan == 'less':
            indicator = np.where(indicator_nan, False, indicator)
        elif self.policy_nan == 'random':
            indicator = np.where(indicator_nan, random_state.randint(0, 2, data.shape) == 1, indicator)
        elif self.policy_nan == 'propagate':
            indicator = np.where(indicator_nan, np.nan, indicator.astype(np.float32))
        else:
            raise ValueError('Unrecognized policy_nan: {}'.format(self.policy_nan))

        return replace(loaded_data_tuple, data=indicator.astype(self.dtype))


class PreprocessNanMean:

    def __init__(self, axis: int = 1):
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata, random_state):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = np.nanmean(loaded_data_tuple.data, self.axis)

        return replace(loaded_data_tuple, data=data)


class PreprocessSqueeze:

    def __init__(self, axis: int = 1):
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata, random_state):
        return replace(loaded_data_tuple, data=np.squeeze(loaded_data_tuple.data, self.axis))


class PreprocessClip:

    def __init__(
            self, minimum=None, maximum=None, value_beyond_min=None, value_beyond_max=None):
        self.value_beyond_min = value_beyond_min
        self.value_beyond_max = value_beyond_max
        self.min = minimum
        self.max = maximum

    def __call__(self, loaded_data_tuple, metadata, random_state):
        data = loaded_data_tuple.data
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            if self.min is not None:
                if self.value_beyond_min is not None:
                    data = np.where(data < self.min, self.value_beyond_min, data)
                else:
                    data = np.maximum(self.min, data)
            if self.max is not None:
                if self.value_beyond_max is not None:
                    data = np.where(data > self.max, self.value_beyond_max, data)
                else:
                    data = np.minimum(self.max, data)
        return replace(loaded_data_tuple, data=data)


class PreprocessNanGeometricMean:

    def __init__(self, axis=1):
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata, random_state):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            data = np.exp(np.nanmean(np.log(loaded_data_tuple.data), axis=self.axis))

        return replace(loaded_data_tuple, data=data)


class PreprocessGaussianBlur:

    def __init__(self, sigma=1, axis=1, order=0, mode='reflect', cval=0.0, truncate=4.0):
        """
        This is meant to blue over a non-example axis, e.g. spatially in fMRI
        """
        self.sigma, self.axis, self.order, self.mode, self.cval, self.truncate = \
            sigma, axis, order, mode, cval, truncate

    def __call__(self, loaded_data_tuple, metadata, random_state):
        data = gaussian_filter1d(
            loaded_data_tuple.data,
            sigma=self.sigma, axis=self.axis, order=self.order, mode=self.mode, truncate=self.truncate)

        return replace(loaded_data_tuple, data=data)


class PreprocessCompress:

    def __init__(self, metadata_condition_name, compress_axis=1):
        self.metadata_condition_name, self.compress_axis = metadata_condition_name, compress_axis

    def __call__(self, loaded_data_tuple, metadata, random_state):
        if metadata is None or self.metadata_condition_name not in metadata:
            raise ValueError('Unable to find metadata_condition_name: {}'.format(self.metadata_condition_name))
        condition = metadata[self.metadata_condition_name]
        data = np.compress(condition, loaded_data_tuple.data, axis=self.compress_axis)
        return replace(loaded_data_tuple, data=data)


class PreprocessSoSFilter:

    def __init__(self, sos, axis=0):
        """
        Apply scipy.signal.sosfilt to data
        Args:
            sos: iirfilter created with output='sos',
                e.g.
                    A high-pass butterworth filter for a sampling rate of 0.5 Hz
                    and cutoff 0.2 Hz
                signal.butter(10, 0.2, 'hp', fs=0.5, output='sos')
            axis: Which axis to apply along
        """
        self.sos = sos
        self.axis = axis

    def __call__(self, loaded_data_tuple, metadata, random_state):
        return replace(loaded_data_tuple, data=sosfilt(self.sos, loaded_data_tuple.data, axis=self.axis))


class PreprocessRandomPair:

    def __init__(
            self,
            num_samples_per_group,
            metadata_example_group_by,
            data_id_pair_fn_map,
            combine_fn=None,
            emit_both=False,
            stop_mode=None):
        self.num_samples_per_group = num_samples_per_group
        self.metadata_example_group_by = metadata_example_group_by
        self.data_id_pair_fn_per_response_data = data_id_pair_fn_map
        self.combine_fn = combine_fn
        self.emit_both = emit_both
        self.stop_mode = stop_mode

    @staticmethod
    def pair_from_end(data_ids1, data_ids2, is_stop1, is_stop2, random_state, emit_both, stop_mode):
        skip = len(data_ids2) - (len(data_ids1) - 1)
        skip1, skip2 = 0, 0
        if skip < 0:  # seq 2 is shorter, skip some of seq 1
            start = len(data_ids1)
            skip1 = -skip
        else:
            start = len(data_ids1) + skip
            skip2 = skip
        paired = list()
        for i in range(len(data_ids1) + len(data_ids2)):
            i1 = i - start + 1 + skip1
            i2 = i - start + skip2
            if i >= start:
                meets_stop_requirement = True
                if stop_mode is not None:
                    if stop_mode == 'content':
                        if is_stop1[i1] or is_stop2[i2]:
                            meets_stop_requirement = False
                    elif stop_mode == 'stop':
                        if not is_stop1[i1] or not is_stop2[i2]:
                            meets_stop_requirement = False
                    elif stop_mode == 'matched':
                        if is_stop1[i1] != is_stop2[i2]:
                            meets_stop_requirement = False
                    else:
                        raise ValueError('Unknown value for stop_mode: {}'.format(stop_mode))
                if meets_stop_requirement:
                    paired.append((data_ids1[i1], data_ids2[i2]))
                    if emit_both:
                        paired[i1] = paired[i]
                else:
                    paired.append(None)
            else:
                paired.append(None)
        return paired

    @staticmethod
    def pair_from_start(data_ids1, data_ids2, is_stop1, is_stop2, random_state, emit_both, stop_mode):
        end = len(data_ids1) + min(len(data_ids1) - 1, len(data_ids2))
        paired = list()
        for i in range(len(data_ids1) + len(data_ids2)):
            i1 = i - len(data_ids1) + 1
            i2 = i - len(data_ids1)
            if len(data_ids1) <= i < end:
                meets_stop_requirement = True
                if stop_mode is not None:
                    if stop_mode == 'content':
                        if is_stop1[i1] or is_stop2[i2]:
                            meets_stop_requirement = False
                    elif stop_mode == 'stop':
                        if not is_stop1[i1] or not is_stop2[i2]:
                            meets_stop_requirement = False
                    elif stop_mode == 'matched':
                        if is_stop1[i1] != is_stop2[i2]:
                            meets_stop_requirement = False
                    else:
                        raise ValueError('Unknown value for stop_mode: {}'.format(stop_mode))
                if meets_stop_requirement:
                    paired.append((data_ids1[i1], data_ids2[i2]))
                    if emit_both:
                        paired[i1] = paired[i]
                else:
                    paired.append(None)
            else:
                paired.append(None)
        return paired

    @staticmethod
    def pair_random(data_ids1, data_ids2, is_stop1, is_stop2, random_state, emit_both, stop_mode):
        idx1 = np.arange(len(data_ids1) - 1) + 1  # skip [CLS]
        idx2 = np.arange(len(data_ids2))

        indicator_valid1 = data_ids1[1:] >= 0
        indicator_valid2 = data_ids2 >= 0
        if stop_mode is not None:
            if stop_mode == 'content':
                indicator_valid1 = np.logical_and(indicator_valid1, np.logical_not(is_stop1[1:]))
                indicator_valid2 = np.logical_and(indicator_valid2, np.logical_not(is_stop2))
            elif stop_mode == 'stop':
                indicator_valid1 = np.logical_and(indicator_valid1, is_stop1[1:])
                indicator_valid2 = np.logical_and(indicator_valid2, is_stop2)
            elif stop_mode == 'matched':
                pass
            else:
                raise ValueError('Unknown value for stop_mode: {}'.format(stop_mode))

        idx1 = random_state.permutation(idx1[indicator_valid1])
        idx2 = idx2[indicator_valid2]

        if stop_mode == 'matched':
            indicator_content1 = is_stop1[idx1]
            indicator_content2 = is_stop2[idx2]
            idx1_content = idx1[indicator_content1]
            idx1_stop = idx1[np.logical_not(indicator_content1)]
            idx2_content = idx2[indicator_content2]
            idx2_stop = idx2[np.logical_not(indicator_content2)]
            idx1_content = idx1_content[:min(len(idx1_content), len(idx2_content))]
            idx2_content = idx2_content[:len(idx1_content)]
            idx1_stop = idx1_stop[:min(len(idx1_stop), len(idx2_stop))]
            idx2_stop = idx2_stop[:len(idx1_stop)]
            idx1 = np.concatenate(idx1_content, idx1_stop)
            idx2 = np.concatenate(idx2_content, idx2_stop)
        else:
            idx1 = idx1[:min(len(idx1), len(idx2))]
            idx2 = idx2[:len(idx1)]

        idx_i = 0
        paired = list()
        for i in range(len(data_ids1) + len(data_ids2)):
            i2 = i - len(data_ids1)
            if idx_i < len(idx2) and i2 == idx2[idx_i]:
                i1 = idx1[idx_i]
                paired.append((data_ids1[i1], data_ids2[i2]))
                if emit_both:
                    paired[i1] = paired[i]
                idx_i += 1
            else:
                paired.append(None)
        return paired

    def _get_data_id_pair_fn(self, response_key, kind):
        if callable(self.data_id_pair_fn_per_response_data):
            return self.data_id_pair_fn_per_response_data
        if response_key in self.data_id_pair_fn_per_response_data:
            return self.data_id_pair_fn_per_response_data[response_key]
        elif kind in self.data_id_pair_fn_per_response_data:
            return self.data_id_pair_fn_per_response_data[kind]
        else:
            raise ValueError('Unspecified data_id_pair_fn')

    def __call__(self, loaded_data_tuple, metadata, random_state):
        if metadata is None or self.metadata_example_group_by not in metadata:
            raise ValueError('metadata_example_group_by {} not found in metadata'.format(
                self.metadata_example_group_by))
        combined = dict()
        old_to_new = dict()
        metadata_indices = list()
        unique_id = 0
        for split_name in ['train', 'validation', 'test']:
            split = getattr(loaded_data_tuple, split_name)
            grouped_examples = _unsorted_group_by(
                split, lambda ex: metadata[self.metadata_example_group_by][ex.unique_id])
            paired = list()
            len1 = list()
            for idx_group, (group, group_examples) in enumerate(grouped_examples):
                picked = set()
                while len(paired) < self.num_samples_per_group * (idx_group + 1):
                    while True:
                        i1 = random_state.choice(len(group_examples))
                        while True:
                            i2 = random_state.choice(len(group_examples))
                            if i2 != i1:
                                break
                        if (i1, i2) not in picked:
                            picked.add((i1, i2))
                            break

                    ex1 = asdict(group_examples[i1])
                    ex2 = asdict(group_examples[i2])

                    assert(ex2['tokens'][0] == '[CLS]')  # skip token 0 on ex2

                    pair = dict()
                    for key in ex1:
                        if key == 'unique_id':
                            pair[key] = unique_id
                            unique_id += 1
                        elif key == 'data_ids':
                            pair[key] = type(ex1[key])()
                            for k in ex1[key]:
                                assert(isinstance(ex1[key][k], np.ndarray))
                                pair[key][k] = np.concatenate([ex1[key][k], ex2[key][k][1:]])
                        elif key == 'index_word_in_example':
                            pair[key] = np.concatenate([ex1[key], ex2[key][1:] - 1 + ex1[key][-1]])
                        elif key == 'index_token_in_sentence':
                            pair[key] = np.concatenate([ex1[key], ex2[key][1:] - 1 + ex1[key][-1]])
                        elif key == 'multipart_id':
                            if ex1[key] is None and ex2[key] is None:
                                pair[key] = None
                            if ex1[key] != ex2[key]:
                                raise ValueError('Cannot pair examples with different multipart ids')
                            pair[key] = ex1[key]
                        elif ex1[key] is None or ex2[key] is None:
                            if ex1[key] is not None or ex2[key] is not None:
                                raise ValueError('Cannot pair examples where one is None '
                                                 'and the other is not on key: {}'.format(key))
                            pair[key] = None
                        elif isinstance(ex1[key], tuple):
                            pair[key] = ex1[key] + ex2[key][1:]
                        else:
                            if not isinstance(ex1[key], np.ndarray):
                                print(key, type(ex1[key]))
                            assert(isinstance(ex1[key], np.ndarray))
                            pair[key] = np.concatenate([ex1[key], ex2[key][1:]])

                    paired.append(InputFeatures(**pair))
                    len1.append(len(ex1['tokens']))
                    metadata_indices.append(ex1['unique_id'])

            for response_k in loaded_data_tuple.data:
                if response_k not in combined:
                    combined[response_k] = list()
                    old_to_new[response_k] = dict()
                data_id_pair_fn = self._get_data_id_pair_fn(response_k, loaded_data_tuple.data[response_k].kind)
                data = loaded_data_tuple.data[response_k].data
                for ex1_len, pair in zip(len1, paired):
                    data_id_pairs = data_id_pair_fn(
                        pair.data_ids[response_k][:ex1_len],
                        pair.data_ids[response_k][ex1_len:],
                        pair.is_stop[:ex1_len],
                        pair.is_stop[ex1_len:],
                        random_state,
                        self.emit_both,
                        self.stop_mode)
                    assert(len(data_id_pairs) == len(pair.data_ids[response_k]))
                    new_data_ids = list()
                    for data_id_pair in data_id_pairs:
                        if isinstance(data_id_pair, tuple):
                            id1, id2 = data_id_pair
                            if id1 < 0 or id2 < 0:
                                new_data_ids.append(-1)
                            else:
                                if (id1, id2) in old_to_new[response_k]:
                                    new_data_ids.append(old_to_new[response_k][(id1, id2)])
                                else:
                                    old_to_new[response_k][(id1, id2)] = len(combined[response_k])
                                    new_data_ids.append(len(combined[response_k]))
                                    if self.combine_fn is None:
                                        combined[response_k].append(data[id2] - data[id1])
                                    else:
                                        combined[response_k].append(self.combine_fn(data[id1], data[id2]))
                        elif isinstance(data_id_pair, int):
                            if data_id_pair >= 0:
                                raise ValueError('Invalid data_id_pair: {}'.format(data_id_pair))
                            else:
                                new_data_ids.append(-1)
                        elif data_id_pair is None:
                            new_data_ids.append(-1)
                        else:
                            raise ValueError('Invalid data_id_pair: {}'.format(data_id_pair))
                    assert(len(new_data_ids) == len(pair.data_ids[response_k]))
                    pair.data_ids[response_k] = np.array(new_data_ids)
            loaded_data_tuple = replace(loaded_data_tuple, **{split_name: paired})
        loaded_data_tuple = replace(loaded_data_tuple, data=type(loaded_data_tuple.data)(
            (k, KindData(loaded_data_tuple.data[k].kind, np.array(combined[k]))) for k in loaded_data_tuple.data))
        metadata = type(metadata)((k, metadata[k][metadata_indices]) for k in metadata)
        return loaded_data_tuple, metadata


class PreprocessToDisk:

    def __init__(self, delete=True):
        self.output_model_path = None
        self.data_key = None
        self.delete = delete

    def set_model_path(self, output_model_path, data_key):
        self.output_model_path = output_model_path
        self.data_key = data_key

    def __call__(self, loaded_data_tuple, metadata, random_state):
        if not os.path.exists(self.output_model_path):
            os.makedirs(self.output_model_path)
        print('saving {} to disk...'.format(self.data_key), end='', flush=True)
        unique_ids = list()
        lengths = list()
        data_ids = list()
        for ex in chain(
                loaded_data_tuple.train, loaded_data_tuple.validation, loaded_data_tuple.test):
            unique_ids.append(ex.unique_id)
            lengths.append(len(ex.data_ids))
            data_ids.extend(ex.data_ids)
        np.savez(
            os.path.join(self.output_model_path, '{}.npz'.format(self.data_key)),
            unique_ids=np.array(unique_ids),
            lengths=np.array(lengths),
            data_ids=np.array(data_ids),
            data=loaded_data_tuple.data)
        if self.delete:
            return replace(loaded_data_tuple, data=None)
        print('done')
        return loaded_data_tuple


def preprocess_fork_no_cluster_to_disk(name, kind, preprocessor):
    if preprocessor is None or isinstance(preprocessor, str):
        return None, None
    if callable(preprocessor):
        if isinstance(preprocessor, PreprocessKMeans):
            return name + '_no_cluster_to_disk', PreprocessToDisk()
        else:
            return None, None
    else:
        new_preprocess = list()
        has_kmeans = False
        for step in preprocessor:
            if isinstance(step, PreprocessKMeans):
                has_kmeans = True
            else:
                new_preprocess.append(step)
        if not has_kmeans:
            return None, None
        new_preprocess.append(PreprocessToDisk())
        return name + '_no_cluster_to_disk', new_preprocess
