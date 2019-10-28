import warnings
import numpy as np
from scipy.stats import ttest_1samp, ttest_rel

from tqdm import trange

from .experiments import named_variations, match_variation
from .aggregate_metrics import get_field_predictions


__all__ = [
    'paired_squared_error',
    'sorted_cumulative_mean_diff',
    'PermutationTestResult',
    'sample_differences',
    'one_sample_permutation_test',
    'two_sample_permutation_test',
    'get_k_vs_k_paired',
    'get_mse_paired',
    'wilcoxon_axis',
    'ResultPValues']


def paired_squared_error(
        paths_obj,
        variation_set_name_a, training_variation_a,
        variation_set_name_b, training_variation_b,
        field_name,
        num_contiguous):
    predictions_a, target_a, ids_a = get_field_predictions(
        paths_obj, variation_set_name_a, training_variation_a, field_name)
    predictions_b, target_b, ids_b = get_field_predictions(
        paths_obj, variation_set_name_b, training_variation_b, field_name)

    def _sort(p, t, i):
        sort_order = np.argsort(i)
        return p[sort_order], t[sort_order], i[sort_order]

    predictions_a, target_a, ids_a = _sort(predictions_a, target_a, ids_a)
    predictions_b, target_b, ids_b = _sort(predictions_b, target_b, ids_b)

    err_a = np.square(predictions_a - target_a)
    err_b = np.square(predictions_b - target_b)

    if not np.array_equal(ids_a, ids_b):
        raise ValueError('Mismatched example ids')

    def _mean_contiguous(e):
        return np.array(
            list(np.nanmean(item, axis=0) for item in np.array_split(e, int(np.ceil(len(e) / num_contiguous)))))

    mse_a = np.nanmean(err_a, axis=0)
    mse_b = np.nanmean(err_b, axis=0)
    pove_a = 1 - mse_a / np.nanvar(target_a, axis=0)
    pove_b = 1 - mse_b / np.nanvar(target_b, axis=0)
    err_a = _mean_contiguous(err_a)
    err_b = _mean_contiguous(err_b)
    return err_a - err_b, pove_a, pove_b


def _k_vs_k_accuracy(predictions, target, indices_true, indices_distractor):
    sample_target = target[indices_true]
    sample_distractor = predictions[indices_distractor]
    sample_predictions = predictions[indices_true]

    sample_target = np.reshape(sample_target, (-1, sample_target.shape[-1]))
    sample_distractor = np.reshape(sample_distractor, (-1, sample_distractor.shape[-1]))
    sample_predictions = np.reshape(sample_predictions, (-1, sample_predictions.shape[-1]))

    distance_correct = np.sum((sample_target - sample_predictions) ** 2, axis=0)
    distance_incorrect = np.sum((sample_target - sample_distractor) ** 2, axis=0)
    return (distance_incorrect > distance_correct) * 1.0 + (distance_incorrect == distance_correct) * 0.5


class PermutationTestResult:
    def __init__(self, true_values, permutation_values, p_values):
        self.true_values = true_values
        self.permutation_values = permutation_values
        self.p_values = p_values


def paired_k_vs_k_permutation(
        paths_obj,
        variation_set_name_a, training_variation_a, variation_set_name_b, training_variation_b, field_name,
        k=20, num_k_vs_k_samples=100, num_permutations=1000):
    _, _, num_runs_a, _, _ = named_variations(variation_set_name_a)
    _, _, num_runs_b, _, _ = named_variations(variation_set_name_b)
    assert (num_runs_a == num_runs_b)
    training_variation_a = match_variation(variation_set_name_a, training_variation_a)
    training_variation_b = match_variation(variation_set_name_b, training_variation_b)

    results_a = list()
    results_b = list()
    results_sum = list()
    results_diff = list()

    def block_permute_indices(count, block_size):
        permute_indices_ = np.random.permutation(int(np.ceil(count / block_size)))
        permute_indices_ = np.reshape(
            permute_indices_ * block_size, (-1, 1)) + np.reshape(np.arange(block_size), (1, -1))
        permute_indices_ = np.reshape(permute_indices_, -1)
        return permute_indices_[permute_indices_ < count]

    def read_results(variation_name, training_variation, idx_run):
        p, t, ids = get_field_predictions(
            paths_obj, variation_name, training_variation, field_name, idx_run, pre_matched=True)
        sort_order = np.argsort(ids)
        return p[sort_order], t[sort_order]

    for index_run in range(num_runs_a):
        predictions_a, target_a = read_results(variation_set_name_a, training_variation_a, index_run)
        predictions_b, target_b = read_results(variation_set_name_b, training_variation_b, index_run)
        assert (len(target_a) == len(target_b))

        for index_permutation in trange(num_permutations + 1):

            if index_permutation == 0:
                permuted_target_a = target_a
                permuted_target_b = target_b
            else:
                # permute in blocks of 10
                permute_indices = block_permute_indices(len(target_a), block_size=10)
                permuted_target_a = target_a[permute_indices]
                permuted_target_b = target_b[permute_indices]

            accuracy_a = None
            accuracy_b = None
            for index_k_vs_k_sample in range(num_k_vs_k_samples):
                if index_k_vs_k_sample == 0:
                    accuracy_a = np.full((num_k_vs_k_samples, target_a.shape[-1]), np.nan)
                    accuracy_b = np.full((num_k_vs_k_samples, target_b.shape[-1]), np.nan)
                indices_true = np.random.choice(len(target_a), k)
                indices_distractor = np.random.choice(len(target_a), k)
                accuracy_a[index_k_vs_k_sample] = _k_vs_k_accuracy(
                    predictions_a, permuted_target_a, indices_true, indices_distractor)
                accuracy_b[index_k_vs_k_sample] = _k_vs_k_accuracy(
                    predictions_b, permuted_target_b, indices_true, indices_distractor)

            mean_a = np.mean(accuracy_a, axis=0)
            mean_b = np.mean(accuracy_b, axis=0)

            for vals, results in [
                    (mean_a, results_a),
                    (mean_b, results_b),
                    (mean_a + mean_b, results_sum),
                    (mean_a - mean_b, results_diff)]:
                if index_permutation == 0:
                    results.append(
                        PermutationTestResult(
                            vals,
                            np.full((num_permutations, target_a.shape[-1]), np.nan),
                            None))
                else:
                    results[-1].permutation_values[index_permutation - 1] = vals

        for results, side in [(results_a, 'ge'), (results_b, 'ge'), (results_sum, 'ge'), (results_diff, 'both')]:
            if side == 'both':
                indicator_ge = np.abs(results[-1].permutation_values) >= np.abs(results[-1].true_values)
            else:
                indicator_ge = results[-1].permutation_values >= results[-1].true_values
            results[-1].p_values = np.mean(indicator_ge, axis=0)

    return results_a, results_b, results_sum, results_diff


def get_k_vs_k_paired(
        paths_obj,
        variation_set_name_a, training_variation_a, variation_set_name_b, training_variation_b, field_name,
        k=20, num_samples=1000, mean_within_run=False):
    _, _, num_runs_a, _, _ = named_variations(variation_set_name_a)
    _, _, num_runs_b, _, _ = named_variations(variation_set_name_b)
    assert (num_runs_a == num_runs_b)
    training_variation_a = match_variation(variation_set_name_a, training_variation_a)
    training_variation_b = match_variation(variation_set_name_b, training_variation_b)
    index_sample = 0
    accuracy_a = None
    accuracy_b = None
    sample_accuracy_a = None
    sample_accuracy_b = None

    def read_results(variation_name, training_variation, idx_run):
        p, t, ids = get_field_predictions(
            paths_obj, variation_name, training_variation, field_name, idx_run, pre_matched=True)
        sort_order = np.argsort(ids)
        return p[sort_order], t[sort_order]

    for index_run in range(num_runs_a):
        predictions_a, target_a = read_results(variation_set_name_a, training_variation_a, index_run)
        predictions_b, target_b = read_results(variation_set_name_b, training_variation_b, index_run)
        assert (len(target_a) == len(target_b))
        if index_run == 0:
            if mean_within_run:
                sample_accuracy_a = np.full((num_samples, target_a.shape[-1]), np.nan)
                sample_accuracy_b = np.full((num_samples, target_b.shape[-1]), np.nan)
                accuracy_a = np.full((num_runs_a, target_a.shape[-1]), np.nan)
                accuracy_b = np.full((num_runs_b, target_b.shape[-1]), np.nan)
            else:
                accuracy_a = np.full((num_samples * num_runs_a, target_a.shape[-1]), np.nan)
                accuracy_b = np.full((num_samples * num_runs_a, target_b.shape[-1]), np.nan)
                sample_accuracy_a = accuracy_a
                sample_accuracy_b = accuracy_b
        if mean_within_run:
            index_sample = 0
        for _ in range(num_samples):
            indices_true = np.random.choice(len(target_a), k)
            indices_distractor = np.random.choice(len(target_a), k)
            sample_accuracy_a[index_sample] = _k_vs_k_accuracy(
                predictions_a, target_a, indices_true, indices_distractor)
            sample_accuracy_b[index_sample] = _k_vs_k_accuracy(
                predictions_b, target_b, indices_true, indices_distractor)
            index_sample += 1
        if mean_within_run:
            accuracy_a[index_run] = np.mean(sample_accuracy_a, axis=0)
            accuracy_b[index_run] = np.mean(sample_accuracy_b, axis=0)
    return accuracy_a, accuracy_b


def get_mse_paired(
        paths_obj,
        variation_set_name_a, training_variation_a, variation_set_name_b, training_variation_b, field_name):
    _, _, num_runs_a, _, _ = named_variations(variation_set_name_a)
    _, _, num_runs_b, _, _ = named_variations(variation_set_name_b)
    assert (num_runs_a == num_runs_b)
    training_variation_a = match_variation(variation_set_name_a, training_variation_a)
    training_variation_b = match_variation(variation_set_name_b, training_variation_b)

    def read_results(variation_name, training_variation, idx_run):
        p, t, ids = get_field_predictions(
            paths_obj, variation_name, training_variation, field_name, idx_run, pre_matched=True)
        sort_order = np.argsort(ids)
        return p[sort_order], t[sort_order], ids[sort_order]

    mse_a = list()
    mse_b = list()
    pove_a = list()
    pove_b = list()
    for index_run in range(num_runs_a):
        predictions_a, target_a, ids_a = read_results(variation_set_name_a, training_variation_a, index_run)
        predictions_b, target_b, ids_b = read_results(variation_set_name_b, training_variation_b, index_run)
        if not np.array_equal(ids_a, ids_b):
            raise ValueError('Mismatched ids')
        err_a = np.square(predictions_a - target_a)
        err_b = np.square(predictions_b - target_b)
        mse_a.append(np.nanmean(err_a, axis=0))
        mse_b.append(np.nanmean(err_b, axis=0))
        var_a = np.nanvar(target_a, axis=0)
        var_b = np.nanvar(target_b, axis=0)
        pove_a.append(np.where(var_a > 0, np.divide(1 - mse_a[-1], var_a, where=var_a > 0), np.nan))
        pove_b.append(np.where(var_b > 0, np.divide(1 - mse_b[-1], var_b, where=var_b > 0), np.nan))

    return np.array(mse_a), np.array(pove_a), np.array(mse_b), np.array(pove_b)


def sample_differences(
        sample_a_values,
        sample_b_values,
        num_contiguous_examples=10,
        unique_ids_a=None,
        unique_ids_b=None):

    def _sort(values, ids):
        if ids is None:
            return values, ids
        ids = np.asarray(ids)
        sort_order = np.argsort(ids)
        return values[sort_order], ids[sort_order]

    sample_a_values, unique_ids_a = _sort(sample_a_values, unique_ids_a)
    sample_b_values, unique_ids_b = _sort(sample_b_values, unique_ids_b)

    if unique_ids_a is not None and unique_ids_b is not None:
        if not np.array_equal(unique_ids_a, unique_ids_b):
            raise ValueError('Ids do not match between unique_ids_a and unique_ids_b')

    def _contiguous_values(s):
        fill_length = int(np.ceil(len(s) / num_contiguous_examples)) * num_contiguous_examples
        temp = np.full((fill_length,) + s.shape[1:], np.nan)
        temp[:len(s)] = s
        temp = np.reshape(temp, (len(temp) // num_contiguous_examples, num_contiguous_examples) + temp.shape[1:])
        return np.nanmean(temp, axis=1)

    sample_a_values = _contiguous_values(sample_a_values)
    sample_b_values = _contiguous_values(sample_b_values)

    return sample_a_values - sample_b_values


def sorted_cumulative_mean_diff(a, b):
    indices_sorted = np.argsort(
        -np.maximum(np.where(np.isnan(a), -np.inf, a), np.where(np.isnan(b), -np.inf, b)), axis=-1)
    diff_values = a - b
    diff_values = np.take_along_axis(diff_values, indices_sorted, axis=-1)
    counts = np.cumsum(np.where(np.isnan(diff_values), 0, 1), axis=-1)
    diff_values = np.nancumsum(diff_values, axis=-1)
    diff_values = diff_values / counts
    return np.nanmean(diff_values, axis=0), np.nanstd(diff_values, axis=0)


def one_sample_permutation_test(
        sample_predictions,
        sample_target,
        value_fn,
        num_contiguous_examples=10,
        num_permutation_samples=1000,
        unique_ids=None,
        side='both'):

    if side not in ['both', 'less', 'greater']:
        raise ValueError('side must be one of: \'both\', \'less\', \'greater\'')

    def _sort(predictions, targets, ids):
        if ids is None:
            return predictions, targets, ids
        ids = np.asarray(ids)
        sort_order = np.argsort(ids)
        return predictions[sort_order], targets[sort_order], ids[sort_order]

    sample_predictions, sample_target, unique_ids = _sort(sample_predictions, sample_target, unique_ids)

    keep_length = int(np.ceil(len(sample_predictions) / num_contiguous_examples)) * num_contiguous_examples
    sample_predictions = sample_predictions[:keep_length]
    sample_target = sample_target[:keep_length]
    true_values = value_fn(sample_predictions, sample_target)
    abs_true_values = np.abs(true_values) if side == 'both' else None

    sample_target = np.reshape(
        sample_target,
        (sample_target.shape[0] // num_contiguous_examples, num_contiguous_examples) + sample_target.shape[1:])

    count_as_extreme = np.zeros(abs_true_values.shape, np.int64)
    for _ in trange(num_permutation_samples, desc='Permutation'):
        indices_target = np.random.permutation(len(sample_target))
        permuted_target = np.reshape(
            sample_target[indices_target], (sample_predictions.shape[0],) + sample_target.shape[2:])
        permuted_values = value_fn(sample_predictions, permuted_target)
        if side == 'less':
            as_extreme = np.where(np.less_equal(permuted_values, true_values), 1, 0)
        elif side == 'greater':
            as_extreme = np.where(np.greater_equal(permuted_values, true_values), 1, 0)
        else:
            assert(side == 'both')
            as_extreme = np.where(np.greater_equal(permuted_values, abs_true_values), 1, 0)
        count_as_extreme += as_extreme

    p_values = count_as_extreme / num_permutation_samples
    return p_values, true_values


def two_sample_permutation_test(
        sample_a_values,
        sample_b_values,
        num_contiguous_examples=10,
        num_permutation_samples=1000,
        unique_ids_a=None,
        unique_ids_b=None):

    def _sort(values, ids):
        if ids is None:
            return values, ids
        ids = np.asarray(ids)
        sort_order = np.argsort(ids)
        return values[sort_order], ids[sort_order]

    sample_a_values, unique_ids_a = _sort(sample_a_values, unique_ids_a)
    sample_b_values, unique_ids_b = _sort(sample_b_values, unique_ids_b)

    if unique_ids_a is not None and unique_ids_b is not None:
        if not np.array_equal(unique_ids_a, unique_ids_b):
            raise ValueError('Ids do not match between unique_ids_a and unique_ids_b')

    def _contiguous_values(s):
        fill_length = int(np.ceil(len(s) / num_contiguous_examples)) * num_contiguous_examples
        temp = np.full((fill_length,) + s.shape[1:], np.nan)
        temp[:len(s)] = s
        temp = np.reshape(temp, (len(temp) // num_contiguous_examples, num_contiguous_examples) + temp.shape[1:])
        return np.nanmean(temp, axis=1)

    sample_a_values = _contiguous_values(sample_a_values)
    sample_b_values = _contiguous_values(sample_b_values)

    true_difference = np.mean(sample_a_values - sample_b_values, axis=0)
    abs_true_difference = np.abs(true_difference)

    all_values = np.concatenate([sample_a_values, sample_b_values], axis=0)

    count_greater_equal = np.zeros(true_difference.shape, np.int64)
    for _ in trange(num_permutation_samples, desc='Permutation'):
        permuted = np.random.permutation(all_values)
        first, second = np.split(permuted, 2)
        permutation_difference = np.abs(np.mean(first - second, axis=0))
        greater_equal = np.where(np.greater_equal(permutation_difference, abs_true_difference), 1, 0)
        count_greater_equal += greater_equal
    p_values = count_greater_equal / num_permutation_samples
    return p_values, true_difference


def wilcoxon_axis(x, y=None, zero_method="wilcox", correction=False):
    # copied from scipy.stats with adjustments so we can apply it along axis=0
    from scipy.stats import distributions
    from scipy.stats.mstats import rankdata
    """
    Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but spliting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.

    Returns
    -------
    T : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    p-value : float
        The two-sided p-value for the test.

    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' \
                          or 'pratt' or 'zsplit'")

    if y is None:
        d = x
    else:
        x, y = map(np.asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x-y

    d_shape = d.shape
    d = np.reshape(d, (d.shape[0], -1))

    if zero_method == "wilcox":
        d = np.where(np.not_equal(d, 0), d, np.nan)  # Keep all non-zero differences

    count = np.sum(np.logical_not(np.isnan(d)), axis=0)
    if np.any(count < 10):
        warnings.warn("Warning: sample size too small for normal approximation.")
    ranked = rankdata(np.ma.masked_invalid(np.abs(d[:, count > 0])), axis=0)
    r = np.full(d.shape, np.nan, ranked.dtype)
    r[:, count > 0] = ranked
    r = np.where(r == 0, np.nan, r)
    r_plus = np.nansum((d > 0) * r, axis=0)
    r_minus = np.nansum((d < 0) * r, axis=0)
    if zero_method == "zsplit":
        r_zero = np.nansum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    # noinspection PyPep8Naming
    T = np.minimum(r_plus, r_minus)
    mn = count*(count + 1.) * 0.25
    se = count*(count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = np.where(d == 0, np.nan, r)

    flat_r = np.reshape(r, (r.shape[0], -1, 1))
    column_id = np.tile(np.reshape(np.arange(flat_r.shape[1]), (1, -1, 1)), (r.shape[0], 1, 1))
    flat_r_with_column = np.reshape(np.concatenate([column_id, flat_r], axis=2), (-1, 2))

    repeats_with_column, repnum = np.unique(flat_r_with_column, return_counts=True, axis=0)
    repeats_with_column = repeats_with_column[repnum > 1]
    repnum = repnum[repnum > 1]
    if len(repnum) != 0:
        column_id = np.asarray(np.round(repeats_with_column[:, 0]), dtype=np.int64)
        weights = repnum * (repnum * repnum - 1)
        weights = np.asarray(weights, dtype=np.float64)
        repeat_correction = 0.5 * np.bincount(column_id, weights=weights)
        column_repeat_correction = np.zeros(flat_r.shape[1], se.dtype)
        column_repeat_correction[:len(repeat_correction)] += repeat_correction
        column_repeat_correction = np.reshape(column_repeat_correction, se.shape)
        # Correction for repeated elements.
        se -= column_repeat_correction

    se = np.sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = 2. * distributions.norm.sf(np.abs(z))
    return np.reshape(T, d_shape[1:]), np.reshape(prob, d_shape[1:])


class ResultPValues:
    def __init__(self, label, subject, a_values, b_values=None, ttest_1_sample_pop_mean=0.5):
        self.label = label
        self.subject = subject
        self.a_mean = np.nanmean(a_values, axis=0)
        self.a_std = np.nanstd(a_values, axis=0)
        self.a_ttest_1samp_p_values = None
        self.b_ttest_1samp_p_values = None
        self.ttest_rel_p_values = None
        self.wilcoxon_p_values = None

        enough_a = np.max(np.count_nonzero(np.logical_not(np.isnan(a_values)), axis=0)) >= 10
        if enough_a:
            _, self.a_ttest_1samp_p_values = ttest_1samp(
                a_values, ttest_1_sample_pop_mean * np.ones(a_values.shape[1:], a_values.dtype))
        if b_values is not None:
            self.b_mean = np.nanmean(b_values, axis=0)
            self.b_std = np.nanstd(b_values, axis=0)
            self.ab_sorted_diff_mean, self.ab_sorted_diff_std = sorted_cumulative_mean_diff(a_values, b_values)
            if np.max(np.count_nonzero(np.logical_not(np.isnan(b_values)), axis=0)) >= 10:
                _, self.b_ttest_1samp_p_values = ttest_1samp(
                    b_values, ttest_1_sample_pop_mean * np.ones(b_values.shape[1:], b_values.dtype))
                if enough_a:
                    _, self.ttest_rel_p_values = ttest_rel(a_values, b_values)
                    _, self.wilcoxon_p_values = wilcoxon_axis(a_values, b_values)
