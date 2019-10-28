from dataclasses import dataclass
import dataclasses
from typing import Optional, Any
from collections import OrderedDict

import numpy as np
import torch


__all__ = [
    'NoValidInputs',
    'logical_not',
    'masked_squared_error',
    'masked_absolute_error',
    'masked_pearsons_distance',
    'masked_cross_entropy',
    'masked_binary_cross_entropy_with_logits',
    'masked_soft_label_cross_entropy',
    'stop_word_and_target_not_nan_mask',
    'NamedTargetStopWordAwareMSE',
    'NamedTargetStopWordAwareMAE',
    'NamedTargetStopWordAwareKLeastSE',
    'NamedTargetStopWordAwareKLeastSEEvalUpdate',
    'NamedTargetStopWordAwareKLeastAE',
    'NamedTargetStopWordAwareKLeastAEEvalUpdate',
    'NamedTargetStopWordAwarePearsonDistance',
    'NamedTargetStopWordAwareBinaryCrossEntropyWithLogits',
    'NamedTargetStopWordAwareCrossEntropy',
    'NamedTargetStopWordAwareSoftLabelCrossEntropy',
    'NamedTargetSingleMSE',
    'NamedTargetSingleMAE',
    'NamedTargetSingleKLeastSE',
    'NamedTargetSingleKLeastSEEvalUpdate',
    'NamedTargetSingleKLeastAE',
    'NamedTargetSingleKLeastAEEvalUpdate',
    'NamedTargetSinglePearsonDistance',
    'NamedTargetSingleBinaryCrossEntropyWithLogits',
    'NamedTargetSingleCrossEntropy',
    'NamedTargetSingleSoftLabelCrossEntropy',
    'CriticMapping',
    'CriticKeys',
    'make_loss_handler',
    'KLeastSEHalvingEpochs',
    'DetailedResult']


class NoValidInputs(Exception):

    def __init__(self):
        super().__init__()


def logical_not(t):
    # use xor with 1 to give a logical not
    return t ^ 1


def masked_squared_error(mask, predictions, target):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
    result = torch.zeros_like(target)
    result.masked_scatter_(mask, sq_err)
    return result, valid_count


def masked_absolute_error(mask, predictions, target):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    abs_err = torch.abs(torch.masked_select(predictions, mask) - torch.masked_select(target, mask))
    result = torch.zeros_like(target)
    result.masked_scatter_(mask, abs_err)
    return result, valid_count


def _values_or_zeros(mask, source):
    # this seems inefficient, but other ways I've tried mess up the gradient
    result = torch.zeros_like(source)
    result.masked_scatter_(mask, torch.masked_select(source, mask))
    return result


def masked_pearsons_distance(mask, predictions, target, sequence_axis=1):
    valid_counts_per_example = mask.sum(dim=sequence_axis, keepdim=True)
    # wherever valid_counts_per_example is less than 2, we need to set the mask to False
    indicator_valid_example = valid_counts_per_example > 1
    mask = mask & indicator_valid_example
    valid_counts_per_example = mask.sum(dim=sequence_axis, keepdim=True)

    # ignore the values where valid_counts_per_example == 0, distance will already be 0 at these locations
    valid_count = (valid_counts_per_example > 0).sum().item()

    if valid_count == 0:
        raise NoValidInputs()

    # convert type on counts for computations
    valid_counts_per_example = valid_counts_per_example.type(target.dtype)

    # this way of computing is more numerically stable than some alternatives

    # replace masked values with zero
    predictions = _values_or_zeros(mask, predictions)
    target = _values_or_zeros(mask, target)

    # compute the mean
    mean_predictions = predictions.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example
    mean_target = target.sum(dim=sequence_axis, keepdim=True) / valid_counts_per_example

    # remove the mean, and re-mask
    predictions = _values_or_zeros(mask, predictions - mean_predictions)
    target = _values_or_zeros(mask, target - mean_target)

    # compute the variance
    var_predictions = (predictions ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)
    var_target = (target ** 2).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # min_value is an epsilon to avoid divide-by-zero, and to prevent sqrt from blowing up numerically
    min_value = torch.zeros((), dtype=var_predictions.dtype, device=var_predictions.device) + 1e-8
    safe_var_predictions = torch.max(var_predictions, min_value)
    safe_var_target = torch.max(var_target, min_value)

    # scale by the std
    predictions = predictions / torch.sqrt(safe_var_predictions)
    target = target / torch.sqrt(safe_var_target)

    # now r is straightforward to compute
    r = (predictions * target).sum(dim=sequence_axis, keepdim=True) / (valid_counts_per_example - 1)

    # convert to distance
    distance = 1 - r

    # final masking to get rid of numerically unstable values
    distance = _values_or_zeros(indicator_valid_example, distance)

    return distance, valid_count, var_predictions, var_target, mean_predictions, mean_target


def masked_cross_entropy(mask, predictions, target):

    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    predictions = predictions.view(np.prod(predictions.size()[:-1]), predictions.size()[-1])
    target = target.view(-1)
    flat_mask = mask.view(-1)
    valid_indices = torch.nonzero(flat_mask)
    predictions = predictions[valid_indices]
    target = target[valid_indices]
    loss = torch.nn.functional.cross_entropy(predictions, target, reduction='none')
    result = torch.zeros(mask.size(), dtype=loss.dtype, device=loss.device)
    result.masked_scatter_(mask, loss)
    return result, valid_count


def masked_binary_cross_entropy_with_logits(mask, predictions, target, pos_weight=None):
    valid_count = mask.sum().item()
    if valid_count == 0:
        raise NoValidInputs()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.masked_select(predictions, mask),
        torch.masked_select(target, mask),
        reduction='none',
        pos_weight=pos_weight)
    result = torch.zeros(mask.size(), dtype=loss.dtype, device=loss.device)
    result.masked_scatter_(mask, loss)
    return result, valid_count


def masked_soft_label_cross_entropy(mask, predictions, target):
    # note we just assume that the target values sum to 1 along axis=-1
    if mask is not None:
        valid_count = mask.sum().item()
        if valid_count == 0:
            raise NoValidInputs()
    else:
        valid_count = None

    # set up 1s in the prediction where the mask is False;
    # this will mean that log_softmax does not give an nan in case the predictions are
    # strange where they are meaningless
    if mask is not None:
        safer_input = torch.ones_like(predictions)
        safer_input.masked_scatter_(mask.view(mask.size() + (1,)), predictions)
    else:
        safer_input = predictions

    softmax = torch.nn.functional.log_softmax(safer_input, dim=-1)
    terms = -softmax * target
    cross_entropy = terms.sum(dim=-1)
    if mask is not None:
        cross_entropy = _values_or_zeros(mask, cross_entropy)
    else:
        valid_count = np.prod(cross_entropy.size())
    return cross_entropy, valid_count


def stop_word_and_target_not_nan_mask(keep_content, target, is_stop, is_begin_word_pieces):
    if is_stop is not None:
        if len(is_stop.size()) < len(target.size()):
            is_stop = is_stop.view(is_stop.size() + (1,) * (len(target.size()) - len(is_stop.size())))
        is_keep = logical_not(is_stop) if keep_content else is_stop
        if is_begin_word_pieces is not None:
            if len(is_begin_word_pieces.size()) < len(target.size()):
                is_begin_word_pieces = is_begin_word_pieces.view(
                    is_begin_word_pieces.size() + (1,) * (len(target.size()) - len(is_begin_word_pieces.size())))
            return is_keep & logical_not(torch.isnan(target)) & is_begin_word_pieces
        else:
            return is_keep & logical_not(torch.isnan(target))
    else:
        if is_begin_word_pieces is not None:
            if len(is_begin_word_pieces.size()) < len(target.size()):
                is_begin_word_pieces = is_begin_word_pieces.view(
                    is_begin_word_pieces.size() + (1,) * (len(target.size()) - len(is_begin_word_pieces.size())))
            return logical_not(torch.isnan(target)) & is_begin_word_pieces
        else:
            return logical_not(torch.isnan(target))


def k_least_squared_error(
        is_eval, is_sequence, k, mask, predictions, target, accumulator, active_mask, moving_average_decay,
        use_abs=False):

    if is_sequence:
        flat_shape = (target.size()[0] * target.size()[1], int(np.prod(target.size()[2:])))
    else:
        flat_shape = (target.size()[0], int(np.prod(target.size()[1:])))

    flat_mask = torch.reshape(mask, flat_shape)

    if not is_eval:
        if use_abs:
            sq_err = torch.abs(torch.masked_select(predictions, mask) - torch.masked_select(target, mask))
        else:
            sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
        sq_err_or_zeros = torch.zeros_like(target)
        sq_err_or_zeros.masked_scatter_(mask, sq_err)
        sq_err_or_zeros = sq_err_or_zeros.detach()

        sq_err_or_zeros = torch.reshape(sq_err_or_zeros, flat_shape)

        flat_mask_float = flat_mask.type(sq_err_or_zeros.dtype)
        indices_terms = (torch.cumsum(flat_mask.type(torch.long), dim=0) - 1) * flat_mask.type(torch.long)
        num_terms = flat_mask.sum(dim=0, keepdim=True)

        # alpha^(num_terms - 1) * x_0 + \sum_i alpha^(num_terms - i - 1) (1 - alpha) x_i

        one_minus_alpha_coeff = (1 - moving_average_decay) * flat_mask_float

        if accumulator is None:
            assert(active_mask is None)
            first_terms = (indices_terms == 0) & flat_mask
            one_minus_alpha_coeff.masked_scatter_(first_terms, torch.ones_like(one_minus_alpha_coeff))
            indices_terms = indices_terms - 1
            num_terms = num_terms - 1
            alpha_coeff = torch.pow(
                moving_average_decay,
                num_terms.type(flat_mask_float.dtype) - indices_terms.type(flat_mask_float.dtype) - 1) * flat_mask_float
            alpha_coeff.masked_scatter_(first_terms, torch.ones_like(alpha_coeff))

            accumulator = torch.sum(alpha_coeff * one_minus_alpha_coeff * sq_err_or_zeros, dim=0, keepdim=True)
            active_mask = torch.sum(flat_mask, dim=0, keepdim=True) > 0
        else:
            alpha_coeff = torch.pow(
                moving_average_decay,
                num_terms.type(flat_mask_float.dtype) - indices_terms.type(flat_mask_float.dtype) - 1) * flat_mask_float

            accumulator = accumulator + torch.sum(
                alpha_coeff * one_minus_alpha_coeff * sq_err_or_zeros, dim=0, keepdim=True)
            current_active = torch.sum(flat_mask, dim=0, keepdim=True) > 0
            active_mask = active_mask | current_active
    elif accumulator is None:
        raise RuntimeError('Cannot call k_least_squared_error with is_eval=True and accumulator=None')

    if mask.size()[0] > 0:  # only do this if there are items in the batch for this loss
        # set scores to a value greater than everything in the accumulator
        scores = torch.zeros_like(accumulator) + accumulator.max() + 1
        # set scores to accumulator where it is valid
        scores.masked_scatter_(active_mask, accumulator)

        _, top_k = torch.topk(scores, k, len(scores.size()) - 1, largest=False, sorted=False)

        top_k_mask = torch.zeros(scores.size(), dtype=mask.dtype, device=mask.device)
        top_k_mask.scatter_(
            len(top_k_mask.size()) - 1, top_k, torch.ones(top_k.size(), dtype=mask.dtype, device=mask.device))

        top_k_mask = top_k_mask.repeat((flat_shape[0],) + (1,) * (len(top_k_mask.size()) - 1))
        top_k_mask = torch.reshape(top_k_mask, mask.size())
        mask = mask & top_k_mask

    err = masked_absolute_error(mask, predictions, target) \
        if use_abs else masked_squared_error(mask, predictions, target)
    return accumulator, active_mask, err


def update_k_least(accumulator, counts, k):
    safe_divisor = torch.max(torch.ones_like(counts), counts)
    mse = accumulator / safe_divisor.type(accumulator.dtype)
    # set scores to a value greater than everything in the accumulator
    scores = torch.zeros_like(mse) + mse.max() + 1
    # set scores to mse where it is valid
    scores.masked_scatter_(counts > 0, mse)

    _, top_k = torch.topk(scores, k, len(scores.size()) - 1, largest=False, sorted=False)

    top_k_mask = torch.zeros(scores.size(), dtype=torch.uint8, device=accumulator.device)
    top_k_mask.scatter_(
        len(top_k_mask.size()) - 1, top_k, torch.ones(top_k.size(), dtype=torch.uint8, device=accumulator.device))

    return top_k_mask


def k_least_squared_error_update_on_eval(
        is_eval, is_sequence, mask, predictions, target, top_k_mask, next_accumulator, next_counts, use_abs=False):

    if is_sequence:
        flat_shape = (target.size()[0] * target.size()[1], int(np.prod(target.size()[2:])))
    else:
        flat_shape = (target.size()[0], int(np.prod(target.size()[1:])))

    flat_mask = torch.reshape(mask, flat_shape)

    if is_eval:
        if use_abs:
            sq_err = torch.abs(torch.masked_select(predictions, mask) - torch.masked_select(target, mask))
        else:
            sq_err = (torch.masked_select(predictions, mask) - torch.masked_select(target, mask)) ** 2
        sq_err_or_zeros = torch.zeros_like(target)
        sq_err_or_zeros.masked_scatter_(mask, sq_err)
        sq_err_or_zeros = sq_err_or_zeros.detach()

        sq_err_or_zeros = torch.reshape(sq_err_or_zeros, flat_shape)

        if next_accumulator is None:
            assert(next_counts is None)
            next_accumulator = torch.sum(sq_err_or_zeros, dim=0, keepdim=True)
            next_counts = torch.sum(flat_mask, dim=0, keepdim=True)
        else:
            next_accumulator = next_accumulator + torch.sum(sq_err_or_zeros, dim=0, keepdim=True)
            next_counts = next_counts + torch.sum(flat_mask, dim=0, keepdim=True)

    if mask.size()[0] > 0 and top_k_mask is not None:
        top_k_mask = top_k_mask.repeat((flat_shape[0],) + (1,) * (len(top_k_mask.size()) - 1))
        top_k_mask = torch.reshape(top_k_mask, mask.size())
        mask = mask & top_k_mask

    err = masked_absolute_error(mask, predictions, target) \
        if use_abs else masked_squared_error(mask, predictions, target)
    return next_accumulator, next_counts, err


@dataclass
class DetailedResult:
    mask: Optional[np.ndarray]
    prediction: np.ndarray
    target: np.ndarray
    sequence_type: str
    data_set_id: Optional[int] = None
    unique_id: Optional[int] = None


def _masked_reduce(loss, valid_count, reduction, as_numpy):
    if as_numpy:
        loss = loss.detach().cpu().numpy()
    if reduction == 'mean' or reduction == 'sum':
        loss = loss.sum()
        if as_numpy:
            loss = loss.item()
        if reduction == 'mean':
            return loss / valid_count
        return loss
    if reduction != 'none':
        raise ValueError('Unknown value for reduction: {}'.format(reduction))
    return loss, valid_count


class _NamedTargetMaskedLoss:

    def __init__(self, field, weight=1.):
        self.field, self.weight = field, weight

    def apply_weight(self, result):
        is_tuple = isinstance(result, tuple)
        if is_tuple:
            loss = result[0]
        else:
            loss = result
        if isinstance(loss, str):
            assert(loss == 'no_valid_inputs')
        else:
            loss = self.weight * loss
        if is_tuple:
            return (loss,) + result[1:]
        return loss

    def __call__(
            self,
            is_eval,
            epoch,
            global_step,
            batch,
            prediction_dict,
            return_detailed=False,
            reduction='mean',
            as_numpy=False,
            apply_weight=True):

        if self.field not in batch:
            if reduction == 'mean' or reduction == 'sum':
                result = 'no_valid_inputs'
            else:
                result = 'no_valid_inputs', 0
            if return_detailed:
                detailed_result = list()
                return result, detailed_result
            else:
                return result

        predictions = prediction_dict[self.field]
        target = batch[self.field]
        target = target.to(predictions.device)
        mask = self._get_mask(is_eval, epoch, global_step, batch, predictions, target)

        try:
            result, valid_count = self._masked_loss(is_eval, epoch, global_step, mask, predictions, target)
            result = _masked_reduce(result, valid_count, reduction, as_numpy)
        except NoValidInputs:
            if reduction == 'mean' or reduction == 'sum':
                result = 'no_valid_inputs'
            else:
                result = 'no_valid_inputs', 0

        if apply_weight:
            result = self.apply_weight(result)

        if return_detailed:

            example_indices = None
            group_prediction_key = (self.field, 'example_ids')
            if group_prediction_key in prediction_dict:
                example_indices = prediction_dict[group_prediction_key].detach().cpu().numpy()

            batch_mask = mask.detach().cpu().numpy()
            batch_predictions = predictions.detach().cpu().numpy()
            batch_target = target.detach().cpu().numpy()
            batch_mask = np.split(batch_mask, len(batch_mask)) if len(batch_mask) > 0 else batch_mask
            batch_predictions = np.split(batch_predictions, len(batch_predictions)) \
                if len(batch_predictions) > 0 else batch_predictions
            batch_target = np.split(batch_target, len(batch_target)) if len(batch_target) > 0 else batch_target

            sequence_type = 'sequence' if self._is_sequence_loss() else 'single'

            if example_indices is not None:  # group by the example indices
                sequence_type = 'grouped'
                grouped = dict()
                for m, p, t, ex in zip(batch_mask, batch_predictions, batch_target, example_indices):
                    if ex not in grouped:
                        grouped[ex] = (list(), list(), list())
                    grouped[ex][0].append(np.expand_dims(m, 1))
                    grouped[ex][1].append(np.expand_dims(p, 1))
                    grouped[ex][2].append(np.expand_dims(t, 1))
                batch_mask = list()
                batch_predictions = list()
                batch_target = list()
                example_indices = [ex for ex in sorted(grouped)]
                for ex in example_indices:
                    batch_mask.append(np.concatenate(grouped[ex][0], axis=1))
                    batch_predictions.append(np.concatenate(grouped[ex][1], axis=1))
                    batch_target.append(np.concatenate(grouped[ex][2], axis=1))

            detailed_result = list()
            for idx, (example_mask, example_predictions, example_targets) in enumerate(zip(
                    batch_mask, batch_predictions, batch_target)):

                if example_indices is not None:
                    idx = example_indices[idx]

                data_set_id = batch['data_set_id'].detach().cpu().numpy()[idx] \
                    if 'data_set_id' in batch else None
                unique_id = batch['unique_id'].detach().cpu().numpy()[idx] \
                    if 'unique_id' in batch else None
                detailed_result.append(
                    DetailedResult(
                        mask=np.abs(np.squeeze(example_mask, axis=0) - 1) < 1e-4,  # convert to bool
                        prediction=np.squeeze(example_predictions, axis=0),
                        target=np.squeeze(example_targets, axis=0),
                        sequence_type=sequence_type,
                        data_set_id=data_set_id,
                        unique_id=unique_id))
            return result, detailed_result
        return result

    def _get_mask(self, is_eval, epoch, global_step, batch, predictions, target):
        return logical_not(torch.isnan(target))

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))

    @classmethod
    def _is_sequence_loss(cls):
        return False

    def shape_adjust(self, shape):
        return shape


class _NamedTargetStopWordAwareLoss(_NamedTargetMaskedLoss):

    def __init__(self, field, keep_content=True, weight=1.):
        super().__init__(field, weight)
        self.keep_content = keep_content

    def _get_mask(self, is_eval, epoch, global_step, batch, predictions, target):
        return stop_word_and_target_not_nan_mask(
            self.keep_content,
            target, batch['is_stop'].to(target.device), batch['is_begin_word_pieces'].to(target.device))

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        raise NotImplementedError('{} does not implement _masked_loss'.format(type(self)))

    @classmethod
    def _is_sequence_loss(cls):
        return True


class NamedTargetStopWordAwareMSE(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_squared_error(mask, predictions, target)


class NamedTargetStopWordAwareMAE(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_absolute_error(mask, predictions, target)


class NamedTargetStopWordAwareKLeastSE(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, k_fn, moving_average_decay=0.98, keep_content=True, weight=1.):
        super().__init__(field, keep_content, weight)
        self.k_fn = k_fn
        self.moving_average_decay = moving_average_decay
        self._accumulator = None
        self._active_mask = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[2:]))
        k = self.k_fn(epoch, global_step, num_features)
        self._accumulator, self._active_mask, result = k_least_squared_error(
            is_eval, is_sequence=True, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator, active_mask=self._active_mask,
            moving_average_decay=self.moving_average_decay)
        return result


class NamedTargetStopWordAwareKLeastSEEvalUpdate(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, k_fn, keep_content=True, weight=1.):
        super().__init__(field, keep_content, weight)
        self.k_fn = k_fn
        self._accumulator = None
        self._counts = None
        self._top_k_mask = None
        self._num_features = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features is None:
            self._num_features = int(np.prod(target.size()[2:]))
        self._accumulator, self._counts, result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=True, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask, next_accumulator=self._accumulator, next_counts=self._counts)
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_fn(epoch, global_step, self._num_features)
        self._top_k_mask = update_k_least(self._accumulator, self._counts, k)
        self._accumulator = None
        self._counts = None


class NamedTargetStopWordAwareKLeastAE(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, k_fn, moving_average_decay=0.98, keep_content=True, weight=1.):
        super().__init__(field, keep_content, weight)
        self.k_fn = k_fn
        self.moving_average_decay = moving_average_decay
        self._accumulator = None
        self._active_mask = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[2:]))
        k = self.k_fn(epoch, global_step, num_features)
        self._accumulator, self._active_mask, result = k_least_squared_error(
            is_eval, is_sequence=True, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator, active_mask=self._active_mask,
            moving_average_decay=self.moving_average_decay, use_abs=True)
        return result


class NamedTargetStopWordAwareKLeastAEEvalUpdate(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, k_fn, keep_content=True, weight=1.):
        super().__init__(field, keep_content, weight)
        self.k_fn = k_fn
        self._accumulator = None
        self._counts = None
        self._top_k_mask = None
        self._num_features = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features is None:
            self._num_features = int(np.prod(target.size()[2:]))
        self._accumulator, self._counts, result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=True, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask, next_accumulator=self._accumulator, next_counts=self._counts, use_abs=True)
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_fn(epoch, global_step, self._num_features)
        self._top_k_mask = update_k_least(self._accumulator, self._counts, k)
        self._accumulator = None
        self._counts = None


class NamedTargetStopWordAwarePearsonDistance(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, keep_content=True, should_penalize_scale=False, weight=1., axis=1):
        super().__init__(field, keep_content, weight)
        self.should_penalize_scale = should_penalize_scale
        self.axis = axis

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        distance, valid_count, var_input, var_target, mean_input, mean_target = masked_pearsons_distance(
            mask, predictions, target, sequence_axis=self.axis)
        loss = distance
        if self.should_penalize_scale:
            loss = loss + (var_input - var_target) ** 2
        return loss, valid_count


class NamedTargetStopWordAwareCrossEntropy(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, num_classes, keep_content=True, weight=1.):
        self.num_classes = num_classes
        super().__init__(field, keep_content, weight)

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_cross_entropy(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)


class NamedTargetStopWordAwareBinaryCrossEntropyWithLogits(_NamedTargetStopWordAwareLoss):

    def __init__(self, field, keep_content=True, weight=1., pos_weight=None):
        super().__init__(field, keep_content, weight)
        self.pos_weight = pos_weight

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_binary_cross_entropy_with_logits(mask, predictions, target, self.pos_weight)


class NamedTargetStopWordAwareSoftLabelCrossEntropy(_NamedTargetStopWordAwareLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_soft_label_cross_entropy(mask, predictions, target)


class NamedTargetSingleMSE(_NamedTargetMaskedLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_squared_error(mask, predictions, target)


class NamedTargetSingleMAE(_NamedTargetMaskedLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_absolute_error(mask, predictions, target)


class KLeastSEHalvingEpochs:
    def __init__(self, half_life_in_epochs, delay_in_epochs=0, minimum_k=100, final_full_epochs_start=None):
        self.half_life_in_epochs = half_life_in_epochs
        self.delay_in_epochs = delay_in_epochs
        self.minimum_k = minimum_k
        self.final_full_epochs_start = final_full_epochs_start

    def __call__(self, epoch, global_step, num_features):
        if self.final_full_epochs_start is not None and epoch >= self.final_full_epochs_start:
            return num_features
        epoch = max(0, epoch - self.delay_in_epochs)
        k = int(np.round(np.power(2., -epoch / self.half_life_in_epochs) * num_features))
        return max(k, min(self.minimum_k, num_features))


class NamedTargetSingleKLeastSE(_NamedTargetMaskedLoss):

    def __init__(self, field, k_fn, moving_average_decay=0.98, weight=1.):
        super().__init__(field, weight)
        self.k_fn = k_fn
        self.moving_average_decay = moving_average_decay
        self._accumulator = None
        self._active_mask = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[1:]))
        k = self.k_fn(epoch, global_step, num_features)
        self._accumulator, self._active_mask, result = k_least_squared_error(
            is_eval, is_sequence=False, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator, active_mask=self._active_mask,
            moving_average_decay=self.moving_average_decay)
        return result


class NamedTargetSingleKLeastSEEvalUpdate(_NamedTargetMaskedLoss):

    def __init__(self, field, k_fn, weight=1.):
        super().__init__(field, weight)
        self.k_fn = k_fn
        self._accumulator = None
        self._counts = None
        self._top_k_mask = None
        self._num_features = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features is None:
            self._num_features = int(np.prod(target.size()[1:]))
        self._accumulator, self._counts, result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=False, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask, next_accumulator=self._accumulator, next_counts=self._counts)
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_fn(epoch, global_step, self._num_features)
        self._top_k_mask = update_k_least(self._accumulator, self._counts, k)
        self._accumulator = None
        self._counts = None


class NamedTargetSingleKLeastAE(_NamedTargetMaskedLoss):

    def __init__(self, field, k_fn, moving_average_decay=0.98, weight=1.):
        super().__init__(field, weight)
        self.k_fn = k_fn
        self.moving_average_decay = moving_average_decay
        self._accumulator = None
        self._active_mask = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        num_features = int(np.prod(target.size()[1:]))
        k = self.k_fn(epoch, global_step, num_features)
        self._accumulator, self._active_mask, result = k_least_squared_error(
            is_eval, is_sequence=False, k=k, mask=mask, predictions=predictions, target=target,
            accumulator=self._accumulator, active_mask=self._active_mask,
            moving_average_decay=self.moving_average_decay, use_abs=True)
        return result


class NamedTargetSingleKLeastAEEvalUpdate(_NamedTargetMaskedLoss):

    def __init__(self, field, k_fn, weight=1.):
        super().__init__(field, weight)
        self.k_fn = k_fn
        self._accumulator = None
        self._counts = None
        self._top_k_mask = None
        self._num_features = None

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        if self._num_features is None:
            self._num_features = int(np.prod(target.size()[1:]))
        self._accumulator, self._counts, result = k_least_squared_error_update_on_eval(
            is_eval, is_sequence=False, mask=mask, predictions=predictions, target=target,
            top_k_mask=self._top_k_mask, next_accumulator=self._accumulator, next_counts=self._counts, use_abs=True)
        return result

    def after_eval_batches(self, epoch, global_step):
        k = self.k_fn(epoch, global_step, self._num_features)
        self._top_k_mask = update_k_least(self._accumulator, self._counts, k)
        self._accumulator = None
        self._counts = None


class NamedTargetSinglePearsonDistance(_NamedTargetMaskedLoss):

    def __init__(self, field, should_penalize_scale=False, weight=1., axis=0):
        super().__init__(field, weight)
        self.should_penalize_scale = should_penalize_scale
        self.axis = axis

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        distance, valid_count, var_input, var_target, mean_input, mean_target = masked_pearsons_distance(
            mask, predictions, target, sequence_axis=self.axis)
        loss = distance
        if self.should_penalize_scale:
            loss = loss + (var_input - var_target) ** 2
        return loss, valid_count


class NamedTargetSingleCrossEntropy(_NamedTargetMaskedLoss):

    def __init__(self, field, num_classes, weight=1.):
        self.num_classes = num_classes
        super().__init__(field, weight)

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_cross_entropy(mask, predictions, target)

    def shape_adjust(self, shape):
        return shape + (self.num_classes,)


class NamedTargetSingleSoftLabelCrossEntropy(_NamedTargetMaskedLoss):

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_soft_label_cross_entropy(mask, predictions, target)


class NamedTargetSingleBinaryCrossEntropyWithLogits(_NamedTargetMaskedLoss):

    def __init__(self, field, weight=1., pos_weight=None):
        super().__init__(field, weight)
        self.pos_weight = pos_weight

    def _masked_loss(self, is_eval, epoch, global_step, mask, predictions, target):
        return masked_binary_cross_entropy_with_logits(mask, predictions, target, self.pos_weight)


@dataclass(frozen=True)
class CriticMapping:
    # this metadata trick allows us to give the canonical value along with the field definition
    # while not specifying a default (so we force all versions of the mapping to instantiate all the fields)
    mse: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareMSE))
    mae: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareMAE))
    k_least_se: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareKLeastSE))
    k_least_se_on_eval: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareKLeastSEEvalUpdate))
    k_least_ae: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareKLeastAE))
    k_least_ae_on_eval: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareKLeastAEEvalUpdate))
    pearson: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwarePearsonDistance))
    cross_entropy: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetStopWordAwareCrossEntropy))
    binary_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetStopWordAwareBinaryCrossEntropyWithLogits))
    soft_label_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetStopWordAwareSoftLabelCrossEntropy))
    single_mse: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleMSE))
    single_mae: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleMAE))
    single_k_least_se: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleKLeastSE))
    single_k_least_se_on_eval: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleKLeastSEEvalUpdate))
    single_k_least_ae: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleKLeastAE))
    single_k_least_ae_on_eval: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSingleKLeastAEEvalUpdate))
    single_pearson: Any = dataclasses.field(metadata=dict(hidden_value=NamedTargetSinglePearsonDistance))
    single_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetSingleCrossEntropy))
    single_binary_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetSingleBinaryCrossEntropyWithLogits))
    single_soft_label_cross_entropy: Any = dataclasses.field(
        metadata=dict(hidden_value=NamedTargetSingleSoftLabelCrossEntropy))


CriticKeys = CriticMapping(**dict((f.name, f.name) for f in dataclasses.fields(CriticMapping)))
_critic_type_dict = OrderedDict((f.name, f.metadata['hidden_value']) for f in dataclasses.fields(CriticMapping))


def make_loss_handler(field, which_loss, loss_kwargs=None):
    if which_loss not in _critic_type_dict:
        raise ValueError('Unknown value for which_loss. Known values are: {}'.format(_critic_type_dict.keys()))
    factory = _critic_type_dict[which_loss]
    if loss_kwargs is None:
        loss_kwargs = {}
    return factory(field, **loss_kwargs)
