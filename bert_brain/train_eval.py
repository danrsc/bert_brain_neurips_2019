import os
import gc
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence, Union, Optional, Mapping
import logging

import numpy as np
import torch
from pytorch_pretrained_bert import BertAdam
from torch.utils.data import SequentialSampler, DistributedSampler, DataLoader as TorchDataLoader, RandomSampler
from tqdm import tqdm, trange

from .data_sets import collate_fn, max_example_sequence_length, PreparedDataDataset, PreparedData
from .modeling import make_loss_handler, BertMultiPredictionHead, KeyedLinear
from .settings import Settings, PredictionHeadSettings
from .result_output import write_predictions, write_loss_curve


logger = logging.getLogger(__name__)


__all__ = ['evaluate', 'train', 'TaskResult', 'TaskResults', 'make_datasets', 'setup_prediction_heads_and_losses']


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def evaluate(
        settings: Settings,
        model,
        loss_handlers,
        device,
        epoch,
        global_step,
        eval_results,
        eval_data_set,
        return_detailed=False):

    if settings.optimization_settings.local_rank == -1:
        eval_sampler = SequentialSampler(eval_data_set)
    else:
        eval_sampler = DistributedSampler(eval_data_set)
    eval_data_loader = TorchDataLoader(
        eval_data_set,
        sampler=eval_sampler, batch_size=settings.optimization_settings.predict_batch_size, collate_fn=collate_fn)

    model.eval()
    all_results = OrderedDict()
    logger.info("Start evaluating")

    if settings.show_step_progress:
        batch_iterator = tqdm(eval_data_loader, desc="Evaluating")
    else:
        batch_iterator = eval_data_loader

    total_loss = 0
    total_count = 0
    losses_to_write = OrderedDict()
    losses_to_write_counts = OrderedDict()
    for batch in batch_iterator:
        # if len(all_results) % 1000 == 0:
        #     logger.info("Processing example: %d" % (len(all_results)))
        for k in batch:
            batch[k] = batch[k].to(device)
        with torch.no_grad():
            predictions = model(batch, eval_data_set)
            loss_result = OrderedDict(
                (h.field,
                 (h.weight,
                  h(True, epoch, global_step, batch, predictions,
                    return_detailed=return_detailed, apply_weight=False, as_numpy=True, reduction='none')))
                for h in loss_handlers)
            if return_detailed:
                loss_dict = OrderedDict()
                for k in loss_result:
                    weight, (summary, detailed) = loss_result[k]
                    loss_dict[k] = weight, summary
                    if k not in all_results:
                        all_results[k] = list()
                    all_results[k].extend(detailed)
            else:
                loss_dict = loss_result
            for data_key in loss_dict:
                weight, (data_loss, data_valid_count) = loss_dict[data_key]
                if data_key not in losses_to_write:
                    losses_to_write[data_key] = 0
                    losses_to_write_counts[data_key] = 0
                if data_valid_count == 0:
                    current = np.nan
                else:
                    current = np.sum(data_loss)
                    losses_to_write[data_key] += current
                    losses_to_write_counts[data_key] += data_valid_count

                if data_valid_count > 0:
                    kind = eval_data_set.response_data_kind(data_key)
                    if data_key in settings.loss_tasks or kind in settings.loss_tasks:
                        total_loss += current
                        total_count += data_valid_count

    for h in loss_handlers:
        if hasattr(h, 'after_eval_batches'):
            h.after_eval_batches(epoch, global_step)

    for k in losses_to_write:
        if losses_to_write_counts[k] == 0:
            losses_to_write[k] = np.nan
        else:
            losses_to_write[k] /= losses_to_write_counts[k]
        eval_results.add_result(k, epoch, global_step, losses_to_write[k])

    if total_count > 0:
        if len(losses_to_write) < 4:
            logger.info('eval:  {:<#8.6}, '.format(total_loss / total_count) + ', '.join(
                ['{}: {:<#8.6}'.format(k, losses_to_write[k]) for k in losses_to_write]))
        else:
            logger.info('eval:  {}'.format(total_loss / total_count))
    else:
        if len(losses_to_write) < 4:
            logger.info('eval:  {:<#8.6}, '.format(total_loss / total_count) + ', '.join(
                ['{}: {:<#8.6}'.format(k, losses_to_write[k]) for k in losses_to_write]))
        else:
            logger.info('eval:  {}'.format(np.nan))

    if return_detailed:
        return all_results


def _loss_weights(loss_count_dict):
    keys = [k for k in loss_count_dict]
    counts = np.array([loss_count_dict[k] for k in keys])
    loss_weights = 1. / (np.sum(1. / counts) * counts)
    return dict(zip(keys, [w.item() for w in loss_weights]))


def make_datasets(
        data: Mapping[str, PreparedData],
        loss_tasks,
        which: Optional[Union[str, Sequence[str]]] = None,
        data_id_in_batch_keys: Optional[Sequence[str]] = None,
        filter_when_not_in_loss_keys: Optional[Sequence[str]] = None):
    if which is None:
        which = ['train', 'validation', 'test']
    max_sequence_length = max_example_sequence_length(data)
    is_single = isinstance(which, str)
    if is_single:
        which = [which]
    result = [
        PreparedDataDataset(
            max_sequence_length, data, loss_tasks, which=w,
            data_id_in_batch_keys=data_id_in_batch_keys, filter_when_not_in_loss_keys=filter_when_not_in_loss_keys)
        for w in which]
    if is_single:
        result = result[0]
    return result


def _raise_if_head_settings_inconsistent(head_a, head_b):
    if head_a.head_type != head_b.head_type:
        raise ValueError('Inconsistent types in prediction head settings with same key: {}'.format(head_b.key))
    if len(head_a.kwargs) != len(head_b.kwargs):
        raise ValueError('Inconsistent kwargs in prediction head settings with same key: {}'.format(head_b.key))
    for kwarg in head_a.kwargs:
        if kwarg not in head_b.kwargs \
                or head_a.kwargs[kwarg] != head_b.kwargs[kwarg]:
            raise ValueError('Inconsistent kwargs in prediction head settings with same key: {}'.format(head_b.key))


def setup_prediction_heads_and_losses(settings, data_set):

    loss_example_counts = dict()
    loss_handlers = list()

    all_kinds = set([data_set.response_data_kind(k) for k in data_set.fields
                     if data_set.response_data_kind(k) is not None])

    for k in settings.loss_tasks:
        if k not in all_kinds and k not in data_set.fields:
            raise ValueError('loss_task is not present as a field: {}'.format(k))

    prediction_heads = dict()
    for k in data_set.fields:
        kind = data_set.response_data_kind(k) if data_set.is_response_data(k) else None
        corpus_key = data_set.data_set_key_for_field(k)
        if k in settings.loss_tasks or k in settings.non_response_outputs or data_set.is_response_data(k):
            if k in settings.loss_tasks or kind in settings.loss_tasks:
                loss_example_counts[k] = data_set.num_examples_for_field(k)
            critic_settings = settings.get_critic(k, data_set)
            handler = make_loss_handler(k, critic_settings.critic_type, critic_settings.critic_kwargs)
            loss_handlers.append(handler)

            prediction_shape = handler.shape_adjust(data_set.value_shape(k))
            prediction_head_settings = None
            if k in settings.prediction_heads:
                prediction_head_settings = settings.prediction_heads[k]
            elif kind in settings.prediction_heads:
                prediction_head_settings = settings.prediction_heads[kind]
            elif corpus_key in settings.prediction_heads:
                prediction_head_settings = settings.prediction_heads[corpus_key]

            if prediction_head_settings is None:
                if data_set.is_sequence(k):
                    prediction_head_settings = PredictionHeadSettings(
                        '__default_sequence__', KeyedLinear, dict(is_sequence=True))
                else:
                    prediction_head_settings = PredictionHeadSettings(
                        '__default_pooled__', KeyedLinear, dict(is_sequence=False))

            if prediction_head_settings.key not in prediction_heads:
                prediction_heads[prediction_head_settings.key] = (prediction_head_settings, OrderedDict())
            else:
                _raise_if_head_settings_inconsistent(
                    prediction_heads[prediction_head_settings.key][0], prediction_head_settings)
            prediction_heads[prediction_head_settings.key][1][k] = prediction_shape

    prediction_heads = [prediction_heads[k] for k in prediction_heads]

    loss_weights = _loss_weights(loss_example_counts)
    for loss_handler in loss_handlers:
        if loss_handler.field in loss_weights:
            loss_handler.weight = loss_weights[loss_handler.field]

    token_supplemental_key_to_shape = OrderedDict()
    pooled_supplemental_key_to_shape = OrderedDict()

    for k in data_set.fields:
        if k in settings.supplemental_fields:
            if data_set.is_sequence(k):
                token_supplemental_key_to_shape[k] = data_set.value_shape(k)
            else:
                pooled_supplemental_key_to_shape[k] = data_set.value_shape(k)

    return prediction_heads, token_supplemental_key_to_shape, pooled_supplemental_key_to_shape, loss_handlers


def train(
        settings: Settings,
        output_validation_path: str,
        output_test_path: str,
        output_model_path: str,
        train_data_set: PreparedDataDataset,
        validation_data_set: PreparedDataDataset,
        test_data_set: Optional[PreparedDataDataset],
        n_gpu: int,
        device,
        load_from_path: str = None):

    output_train_curve_path = os.path.join(os.path.split(output_validation_path)[0], 'train_curve.npz')
    output_validation_curve_path = os.path.join(os.path.split(output_validation_path)[0], 'validation_curve.npz')

    num_train_steps = int(
        len(train_data_set) /
        settings.optimization_settings.train_batch_size /
        settings.optimization_settings.gradient_accumulation_steps * settings.optimization_settings.num_train_epochs)

    num_epochs_prediction_head_only_train = settings.optimization_settings.num_epochs_train_prediction_heads_only
    if num_epochs_prediction_head_only_train < 0:
        num_epochs_prediction_head_only_train = settings.optimization_settings.num_train_epochs
    start_final_epochs_prediction_head_only_train = int(
        settings.optimization_settings.num_train_epochs
        - settings.optimization_settings.num_final_epochs_train_prediction_heads_only)

    prediction_heads, token_supplemental_key_to_shape, pooled_supplemental_key_to_shape, loss_handlers = \
        setup_prediction_heads_and_losses(settings, train_data_set)

    # Prepare model
    model_loader = BertMultiPredictionHead.from_fine_tuned \
        if load_from_path is not None else BertMultiPredictionHead.from_pretrained
    model = model_loader(
        load_from_path if load_from_path is not None else settings.bert_model,
        map_location=lambda storage, loc: None if loc == 'cpu' else storage.cuda(device.index),
        prediction_head_settings=prediction_heads,
        token_supplemental_key_to_shape=token_supplemental_key_to_shape,
        pooled_supplemental_key_to_shape=pooled_supplemental_key_to_shape)

    if settings.optimization_settings.fp16:
        model.half()
    model.to(device)
    if settings.optimization_settings.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[settings.optimization_settings.local_rank],
            output_device=settings.optimization_settings.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if settings.optimization_settings.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_())
                           for n, param in model.named_parameters()]
    elif settings.optimization_settings.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_())
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']

    non_prediction_head_parameters = None
    if num_epochs_prediction_head_only_train > 0 or start_final_epochs_prediction_head_only_train:
        non_prediction_head_parameters = [p for n, p in param_optimizer if not n.startswith('prediction_head.')]
        for p in non_prediction_head_parameters:
            p.requires_grad = False

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=settings.optimization_settings.learning_rate,
                         warmup=settings.optimization_settings.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0
    train_results = TaskResults()
    validation_results = TaskResults()
    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_data_set))
    # for now we set max_sequence_length so these are never split
    logger.info("  Num split examples = %d", len(train_data_set))
    logger.info("  Batch size = %d", settings.optimization_settings.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    if settings.optimization_settings.local_rank == -1:
        train_sampler = RandomSampler(train_data_set)
    else:
        train_sampler = DistributedSampler(train_data_set)
    train_data_loader = TorchDataLoader(
        train_data_set,
        sampler=train_sampler, batch_size=settings.optimization_settings.train_batch_size, collate_fn=collate_fn)

    if settings.show_epoch_progress:
        epoch_range = trange(int(settings.optimization_settings.num_train_epochs), desc="Epoch")
    else:
        epoch_range = range(int(settings.optimization_settings.num_train_epochs))

    for index_epoch in epoch_range:

        logger.info('Starting epoch {}'.format(index_epoch))

        model.train()

        if index_epoch == start_final_epochs_prediction_head_only_train:
            for p in non_prediction_head_parameters:
                p.requires_grad = False
        elif index_epoch == num_epochs_prediction_head_only_train:
            for p in non_prediction_head_parameters:
                p.requires_grad = True

        if settings.show_step_progress:
            batch_iterator = tqdm(train_data_loader, desc="Iteration")
        else:
            batch_iterator = train_data_loader

        for step, batch in enumerate(batch_iterator):
            if n_gpu == 1:
                for k in batch:
                    batch[k] = batch[k].to(device)
            predictions = model(batch, train_data_set)
            loss_dict = OrderedDict(
                (h.field,
                 (h.weight,
                  h(False, index_epoch, global_step, batch, predictions, apply_weight=False))) for h in loss_handlers)

            # free up memory
            del predictions
            del batch

            loss = None
            losses_to_write = OrderedDict()
            for data_key in loss_dict:
                weight, data_loss = loss_dict[data_key]
                no_valid_inputs = isinstance(data_loss, str) and data_loss == 'no_valid_inputs'
                kind = train_data_set.response_data_kind(data_key)
                if (data_key in settings.loss_tasks or kind in settings.loss_tasks) and not no_valid_inputs:
                    current = weight * data_loss
                    losses_to_write[data_key] = np.nan if no_valid_inputs else data_loss.detach().cpu().numpy().item()
                    if loss is None:
                        loss = current
                    else:
                        loss += current
                train_result = np.nan if no_valid_inputs else data_loss.detach().cpu().numpy().item()
                train_results.add_result(
                    data_key,
                    index_epoch,
                    global_step,
                    train_result)

            del loss_dict

            if loss is not None:
                if len(losses_to_write) < 4:
                    logger.info('train: {:<#8.6}, '.format(loss.item()) + ', '.join(
                        ['{}: {:<#8.6}'.format(k, losses_to_write[k]) for k in losses_to_write]))
                else:
                    logger.info('train: {}'.format(loss.item()))
                if n_gpu > 1:  # hmm - not sure how this is supposed to work
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if settings.optimization_settings.fp16 and settings.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * settings.loss_scale
                if settings.optimization_settings.gradient_accumulation_steps > 1:
                    loss = loss / settings.optimization_settings.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % settings.optimization_settings.gradient_accumulation_steps == 0:
                if settings.optimization_settings.fp16 or settings.optimization_settings.optimize_on_cpu:
                    if settings.optimization_settings.fp16 and settings.loss_scale != 1.0:
                        # scale down gradients for fp16 training
                        for param in model.parameters():
                            param.grad.data = param.grad.data / settings.loss_scale
                    is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                    if is_nan:
                        logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                        settings.loss_scale = settings.loss_scale / 2
                        model.zero_grad()
                        continue
                    optimizer.step()
                    copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                else:
                    optimizer.step()
                model.zero_grad()
                global_step += 1

            # we're being super aggressive about releasing memory here because
            # we're right on the edge of fitting in gpu
            del loss
            gc.collect()
            torch.cuda.empty_cache()

        write_loss_curve(output_train_curve_path, train_results)
        if len(validation_data_set) > 0:
            evaluate(settings, model, loss_handlers, device,
                     index_epoch, global_step, validation_results, validation_data_set)
            write_loss_curve(output_validation_curve_path, validation_results)

    logger.info("***** Running predictions *****")
    logger.info("  Num orig examples = %d", len(validation_data_set))
    logger.info("  Num split examples = %d", len(validation_data_set))
    logger.info("  Batch size = %d", settings.optimization_settings.predict_batch_size)

    if len(validation_data_set) > 0:
        all_validation = evaluate(
            settings, model, loss_handlers, device, settings.optimization_settings.num_train_epochs - 1,
            global_step, TaskResults(), validation_data_set,
            return_detailed=True)
    else:
        all_validation = {}

    if len(test_data_set) > 0:
        all_test = evaluate(
            settings, model, loss_handlers, device, settings.optimization_settings.num_train_epochs - 1,
            global_step, TaskResults(), test_data_set, return_detailed=True)
    else:
        all_test = {}

    write_predictions(output_validation_path, all_validation, validation_data_set, settings)
    write_predictions(output_test_path, all_test, test_data_set, settings)

    # Save a trained model and the associated configuration
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    model.save(output_model_path)

    # clean up after we're done to try to release CUDA resources to other people when there are no more tasks
    gc.collect()
    torch.cuda.empty_cache()


@dataclass
class TaskResult:
    epoch: int
    step: int
    value: float


class TaskResults:

    def __init__(self):
        self.results = OrderedDict()

    def add_result(self, name, epoch, step, value):
        if name not in self.results:
            self.results[name] = list()
        self.results[name].append(TaskResult(epoch, step, value))
