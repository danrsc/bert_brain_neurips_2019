import os
from .experiments import named_variations, match_variation, task_hash
from .result_output import read_predictions


__all__ = ['sentence_predictions']


def sentence_predictions(paths, variation_set_name, training_variation, key):
    _, _, num_runs, _, _ = named_variations(variation_set_name)
    training_variation = match_variation(variation_set_name, training_variation)
    output_dir = os.path.join(paths.result_path, variation_set_name, task_hash(training_variation))
    result = dict()
    has_warned = False
    for index_run in range(num_runs):
        validation_npz_path = os.path.join(output_dir, 'run_{}'.format(index_run), 'output_validation.npz')
        if not os.path.exists(validation_npz_path):
            if not has_warned:
                print('warning: results are incomplete. Some runs not found')
            has_warned = True
            continue
        output_results = read_predictions(validation_npz_path)
        for output_result in output_results[key]:
            if output_result.unique_id not in result:
                result[output_result.unique_id] = list()
            result[output_result.unique_id].append(output_result)
    return result
