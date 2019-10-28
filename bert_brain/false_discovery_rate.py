import numpy as np


__all__ = ['fdr_correction']


def fdr_correction(p_values, alpha=0.05, method='by', axis=None):
    """
    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests.
    Modified from the code at https://www.statsmodels.org/dev/_modules/statsmodels/stats/multitest.html

    Args:
        p_values: The p_values to correct.
        alpha: The error rate to correct the p-values with.
        method: one of by (for Benjamini/Yekutieli) or bh for Benjamini/Hochberg
        axis: Which axis of p_values to apply the correction along. If None, p_values is flattened.

    Returns:
        indicator_alternative: An boolean array with the same shape as p_values_corrected that is True where
            the null hypothesis should be rejected
        p_values_corrected: The p_values corrected for FDR. Same shape as p_values
    """
    p_values = np.asarray(p_values)

    shape = p_values.shape
    if axis is None:
        p_values = np.reshape(p_values, -1)
        axis = 0

    indices_sorted = np.argsort(p_values, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)

    correction_factor = np.arange(1, p_values.shape[axis] + 1) / p_values.shape[axis]
    if method == 'bh':
        pass
    elif method == 'by':
        c_m = np.sum(1 / np.arange(1, p_values.shape[axis] + 1), axis=axis, keepdims=True)
        correction_factor = correction_factor / c_m
    else:
        raise ValueError('Unrecognized method: {}'.format(method))

    # set everything left of the maximum qualifying p-value
    indicator_alternative = p_values <= correction_factor * alpha
    indices_all = np.reshape(
        np.arange(indicator_alternative.shape[axis]),
        (1,) * axis + (indicator_alternative.shape[axis],) + (1,) * (len(indicator_alternative.shape) - 1 - axis))
    indices_max = np.nanmax(np.where(indicator_alternative, indices_all, np.nan), axis=axis, keepdims=True).astype(int)
    indicator_alternative = indices_all <= indices_max
    del indices_all

    p_values = np.clip(
        np.take(
            np.minimum.accumulate(
                np.take(p_values / correction_factor, np.arange(p_values.shape[axis] - 1, -1, -1), axis=axis),
                axis=axis),
            np.arange(p_values.shape[axis] - 1, -1, -1),
            axis=axis),
        a_min=0,
        a_max=1)

    indices_sorted = np.argsort(indices_sorted, axis=axis)
    p_values = np.take_along_axis(p_values, indices_sorted, axis=axis)
    indicator_alternative = np.take_along_axis(indicator_alternative, indices_sorted, axis=axis)

    return np.reshape(indicator_alternative, shape), np.reshape(p_values, shape)
