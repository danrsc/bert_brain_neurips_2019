import dataclasses
import os
from typing import Tuple

import numpy as np

from .experiments import named_variations, task_hash
from .result_output import read_loss_curve


__all__ = [
    'average_unique_epochs_within_loss_curves',
    'average_unique_steps_within_loss_curves',
    'LossCurve',
    'loss_curves_for_variation']


def average_unique_steps_within_loss_curves(curves):
    for curve in curves:
        unique_steps = np.unique(curve.steps)
        step_values = list()
        step_epochs = list()
        for step in unique_steps:
            step_values.append(np.nanmean(curve.values[curve.steps == step]))
            step_epochs.append(curve.epochs[curve.steps == step][0])
        curve.steps = unique_steps
        curve.epochs = np.array(step_epochs)
        curve.values = np.array(step_values)


def average_unique_epochs_within_loss_curves(curves):
    for curve in curves:
        unique_epochs = np.unique(curve.epochs)
        epoch_values = list()
        for epoch in unique_epochs:
            epoch_values.append(np.nanmean(curve.values[curve.epochs == epoch]))
        curve.steps = unique_epochs
        curve.epochs = unique_epochs
        curve.values = np.array(epoch_values)


@dataclasses.dataclass
class LossCurve:
    training_variation: Tuple[str, ...]
    train_eval_kind: str
    index_run: int
    key: str
    epochs: np.ndarray
    steps: np.ndarray
    values: np.ndarray


def loss_curves_for_variation(paths, variation_set_name):
    training_variations, _, num_runs, _, _ = named_variations(variation_set_name)

    def read_curve(kind, training_variation_, index_run_):
        file_name = 'train_curve.npz' if kind == 'train' else 'validation_curve.npz'
        output_dir = os.path.join(paths.result_path, variation_set_name, task_hash(training_variation_))
        curve_path = os.path.join(output_dir, 'run_{}'.format(index_run_), file_name)
        result_ = list()
        if os.path.exists(curve_path):
            curve = read_loss_curve(curve_path)
            for key in curve:
                result_.append(
                    LossCurve(training_variation_, kind, index_run_, key, curve[key][0], curve[key][1], curve[key][2]))
        return result_

    result = list()
    for training_variation in training_variations:
        for index_run in range(num_runs):
            result.extend(read_curve('train', training_variation, index_run))
            result.extend(read_curve('validation', training_variation, index_run))

    return result
