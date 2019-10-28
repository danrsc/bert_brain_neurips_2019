from collections import OrderedDict
import dataclasses
from typing import Mapping, Any, Sequence
import logging

import numpy as np

"""
Low-level functions for reading raw results from experiment output files
"""


__all__ = ['OutputResult', 'write_predictions', 'read_predictions', 'write_loss_curve', 'read_loss_curve']


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class OutputResult:
    name: str
    critic_type: str
    critic_kwargs: Mapping[str, Any]
    unique_id: int
    data_key: str
    tokens: Sequence[str]
    mask: Sequence[bool]
    prediction: Sequence[float]
    target: Sequence[float]
    sequence_type: str


def _num_tokens(tokens):
    for idx, token in enumerate(tokens):
        if token == '[PAD]':
            return idx
    return len(tokens)


def write_predictions(output_path, all_results, data_set, settings):

    """Write final predictions to an output file."""
    logger.info("Writing predictions to: %s" % output_path)

    output_dict = dict()
    for key in all_results:

        if len(all_results[key]) == 0:
            continue

        critic_settings = settings.get_critic(key, data_set)

        predictions = list()
        targets = list()
        masks = list()
        lengths = list()
        target_lengths = list()
        data_keys = list()
        unique_ids = list()
        tokens = list()

        sequence_type = None
        for detailed_result in all_results[key]:
            if sequence_type is None:
                sequence_type = detailed_result.sequence_type
            else:
                assert(sequence_type == detailed_result.sequence_type)
            current_tokens = data_set.get_tokens(detailed_result.data_set_id, detailed_result.unique_id)
            num_tokens = _num_tokens(current_tokens)
            tokens.extend(current_tokens[:num_tokens])
            unique_ids.append(detailed_result.unique_id)
            data_keys.append(data_set.data_set_key_for_id(detailed_result.data_set_id))
            lengths.append(num_tokens)
            if sequence_type == 'sequence':
                predictions.append(detailed_result.prediction[:num_tokens])
                targets.append(detailed_result.target[:num_tokens])
                if detailed_result.mask is not None:
                    masks.append(detailed_result.mask[:num_tokens])
                else:
                    masks.append(None)
                target_lengths.append(num_tokens)
            elif sequence_type == 'single':
                predictions.append(np.expand_dims(detailed_result.prediction, 0))
                targets.append(np.expand_dims(detailed_result.target, 0))
                masks.append(np.expand_dims(detailed_result.mask, 0) if detailed_result.mask is not None else None)
            elif sequence_type == 'grouped':
                predictions.append(detailed_result.prediction)
                targets.append(detailed_result.target)
                masks.append(detailed_result.mask)
                target_lengths.append(len(detailed_result.target))

        if any(m is None for m in masks) and any(m is not None for m in masks):
            raise ValueError('Unable to write a mixture of None and non-None masks')

        output_dict['predictions_{}'.format(key)] = np.concatenate(predictions)
        output_dict['target_{}'.format(key)] = np.concatenate(targets)
        output_dict['masks_{}'.format(key)] = np.concatenate(masks) if masks[0] is not None else None
        output_dict['lengths_{}'.format(key)] = np.array(lengths)
        output_dict['target_lengths_{}'.format(key)] = np.array(target_lengths)
        output_dict['data_keys_{}'.format(key)] = np.array(data_keys)
        output_dict['unique_ids_{}'.format(key)] = np.array(unique_ids)
        output_dict['tokens_{}'.format(key)] = np.array(tokens)
        output_dict['critic_{}'.format(key)] = critic_settings.critic_type
        output_dict['sequence_type_{}'.format(key)] = sequence_type
        if critic_settings.critic_kwargs is not None:
            for critic_key in critic_settings.critic_kwargs:
                output_dict['critic_kwarg_{}_{}'.format(key, critic_key)] = critic_settings.critic_kwargs[critic_key]

    np.savez(output_path, keys=np.array([k for k in all_results if len(all_results[k]) > 0]), **output_dict)


def read_predictions(output_path):
    with np.load(output_path, allow_pickle=True) as npz:
        keys = [k.item() for k in npz['keys']]

        result = OrderedDict()
        for key in keys:
            predictions = npz['predictions_{}'.format(key)]
            target = npz['target_{}'.format(key)]
            masks = npz['masks_{}'.format(key)]
            lengths = npz['lengths_{}'.format(key)]
            target_lengths = npz['target_lengths_{}'.format(key)]
            data_keys = npz['data_keys_{}'.format(key)]
            unique_ids = npz['unique_ids_{}'.format(key)]
            tokens = npz['tokens_{}'.format(key)]
            critic_type = npz['critic_{}'.format(key)].item()
            sequence_type = npz['sequence_type_{}'.format(key)].item()
            critic_kwarg_prefix = 'critic_kwarg_{}'.format(key)
            critic_kwargs = dict()
            for npz_key in npz.keys():
                if npz_key.startswith(critic_kwarg_prefix):
                    critic_kwargs[npz_key[len(critic_kwarg_prefix):]] = npz[npz_key].item()
            if len(critic_kwargs) == 0:
                critic_kwargs = None

            splits = np.cumsum(lengths)[:-1]
            if sequence_type == 'sequence':
                target_splits = splits
            elif sequence_type == 'grouped':
                target_splits = np.cumsum(target_lengths)[:-1]
            else:
                target_splits = None
            if target_splits is not None:
                predictions = np.split(predictions, target_splits)
                target = np.split(target, target_splits)
                if masks is not None:
                    # noinspection PyTypeChecker
                    masks = np.split(masks, target_splits)
            data_keys = [k.item() for k in data_keys]
            unique_ids = [u.item() for u in unique_ids]
            tokens = np.split(tokens, splits)
            tokens = [[t.item() for t in s] for s in tokens]

            results = list()
            for idx in range(len(tokens)):
                results.append(OutputResult(
                    key, critic_type, critic_kwargs,
                    unique_ids[idx], data_keys[idx], tokens[idx], masks[idx], predictions[idx], target[idx],
                    sequence_type))

            result[key] = results

    return result


def write_loss_curve(output_path, task_results):
    output_dict = dict()
    keys = [k for k in task_results.results]
    output_dict['__keys__'] = keys
    for key in keys:
        output_dict['epochs_{}'.format(key)] = np.array([tr.epoch for tr in task_results.results[key]])
        output_dict['steps_{}'.format(key)] = np.array([tr.step for tr in task_results.results[key]])
        output_dict['values_{}'.format(key)] = np.array([tr.value for tr in task_results.results[key]])
    np.savez(output_path, **output_dict)


def read_loss_curve(output_path):
    npz = np.load(output_path, allow_pickle=True)
    keys = npz['__keys__']
    result = OrderedDict()
    for key in keys:
        result[key] = (npz['epochs_{}'.format(key)], npz['steps_{}'.format(key)], npz['values_{}'.format(key)])
    return result
