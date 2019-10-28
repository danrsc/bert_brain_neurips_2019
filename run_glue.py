# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import argparse
import csv
import logging
import dataclasses
import os
import gc
from itertools import groupby

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from bert_brain import cuda_most_free_device, task_hash, set_random_seeds, named_variations, Settings, TrainingVariation
from tqdm_logging import replace_root_logger_handler
from bert_brain_paths import Paths


replace_root_logger_handler()
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")),
            "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [None]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), 
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    # noinspection PyMethodMayBeStatic
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def from_fine_tuned(model_path, map_location='default_map_location', *inputs, **kwargs):
    config = BertConfig(os.path.join(model_path, CONFIG_NAME))
    model = BertForSequenceClassification(config, *inputs, **kwargs)
    if map_location == 'default_map_location':
        map_location = 'cpu' if not torch.cuda.is_available() else None
    state_dict = torch.load(os.path.join(model_path, WEIGHTS_NAME), map_location=map_location)
    # Load from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if 'gamma' in key:
            new_key = key.replace('gamma', 'weight')
        if 'beta' in key:
            new_key = key.replace('beta', 'bias')
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        # noinspection PyProtectedMember
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        # noinspection PyProtectedMember
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    start_prefix = ''
    if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
        start_prefix = 'bert.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
            model.__class__.__name__, "\n\t".join(error_msgs)))
    return model


def _run_glue_for_variation(
        model_path,
        result_path,
        task_name,
        train_features,
        eval_features,
        processor,
        output_mode,
        settings: Settings,
        index_run: int,
        index_sub_run: int,
        device,
        n_gpu,
        mnli_mm_eval_features=None):

    run_model_path = os.path.join(model_path, 'run_{}'.format(index_run))
    run_result_path = os.path.join(result_path, 'run_{}'.format(index_run))

    if not os.path.exists(run_model_path):
        os.makedirs(run_model_path)
    if not os.path.exists(run_result_path):
        os.makedirs(run_result_path)

    # noinspection PyTypeChecker
    model = from_fine_tuned(
        run_model_path,
        map_location=lambda storage, loc: None if loc == 'cpu' else storage.cuda(device.index),
        num_labels=len(processor.get_labels()))

    if settings.optimization_settings.fp16:
        model.half()
    model.to(device)
    if settings.optimization_settings.local_rank != -1:
        try:
            # noinspection PyPep8Naming
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from "
                              "https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    num_train_optimization_steps = None
    optimizer = None
    if train_features is not None:

        num_train_optimization_steps = (
                int(
                    len(train_features)
                    / settings.optimization_settings.train_batch_size
                    / settings.optimization_settings.gradient_accumulation_steps)
                * settings.optimization_settings.num_train_epochs)
        if settings.optimization_settings.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':
                [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        if settings.optimization_settings.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from "
                                  "https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=settings.optimization_settings.learning_rate,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if settings.optimization_settings.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=settings.optimization_settings.loss_scale)
            # warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
            #                                      t_total=num_train_optimization_steps)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=settings.optimization_settings.learning_rate,
                                 warmup=settings.optimization_settings.warmup_proportion,
                                 t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    if train_features is not None:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", settings.optimization_settings.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)

        all_label_ids = None
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        if settings.optimization_settings.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=settings.optimization_settings.train_batch_size)

        model.train()
        loss = None
        for _ in trange(int(settings.optimization_settings.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, len(processor.get_labels())), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if settings.optimization_settings.gradient_accumulation_steps > 1:
                    loss = loss / settings.optimization_settings.gradient_accumulation_steps

                if settings.optimization_settings.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % settings.optimization_settings.gradient_accumulation_steps == 0:
                    if settings.optimization_settings.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = settings.optimization_settings.learning_rate \
                                       * warmup_linear.get_lr(
                                            global_step, settings.optimization_settings.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

    task_model_path = os.path.join(run_model_path, task_name, '{:.0e}_{}'.format(
        settings.optimization_settings.learning_rate, index_sub_run))

    if train_features is not None \
            and (settings.optimization_settings.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        if not os.path.exists(task_model_path):
            os.makedirs(task_model_path)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(task_model_path, WEIGHTS_NAME)
        output_config_file = os.path.join(task_model_path, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

        # Load a trained model and vocabulary that you have fine-tuned
        model = BertForSequenceClassification.from_pretrained(task_model_path, num_labels=len(processor.get_labels()))
    else:
        model = BertForSequenceClassification.from_pretrained(
            task_model_path, num_labels=len(processor.get_labels()))
    model.to(device)

    if eval_features is not None \
            and (settings.optimization_settings.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_features))
        logger.info("  Batch size = %d", settings.optimization_settings.predict_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        elif output_mode == "regression":
            all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=settings.optimization_settings.predict_batch_size)

        model.eval()
        eval_loss = 0
        nb_eval_steps = 0
        preds = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, len(processor.get_labels())), label_ids.view(-1))
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            else:
                raise ValueError('Unexpected output_mode: {}'.format(output_mode))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss / global_step if train_features is not None else None

        result['eval_loss'] = eval_loss
        result['global_step'] = global_step
        result['loss'] = loss

        if not os.path.exists(run_result_path):
            os.makedirs(run_result_path)

        output_eval_file = os.path.join(run_result_path, "{}_{:.0e}_{}_results.txt".format(
            task_name, settings.optimization_settings.learning_rate, index_sub_run))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # hack for MNLI-MM
        if task_name == "mnli":
            task_name = "mnli-mm"

            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(mnli_mm_eval_features))
            logger.info("  Batch size = %d", settings.optimization_settings.predict_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in mnli_mm_eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in mnli_mm_eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in mnli_mm_eval_features], dtype=torch.long)
            all_label_ids = torch.tensor([f.label_id for f in mnli_mm_eval_features], dtype=torch.long)

            eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(
                eval_data, sampler=eval_sampler, batch_size=settings.optimization_settings.predict_batch_size)

            model.eval()
            eval_loss = 0
            nb_eval_steps = 0
            preds = []

            for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask, labels=None)

                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, len(processor.get_labels())), label_ids.view(-1))

                eval_loss += tmp_eval_loss.mean().item()
                nb_eval_steps += 1
                if len(preds) == 0:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds[0] = np.append(
                        preds[0], logits.detach().cpu().numpy(), axis=0)

            eval_loss = eval_loss / nb_eval_steps
            preds = preds[0]
            preds = np.argmax(preds, axis=1)
            result = compute_metrics(task_name, preds, all_label_ids.numpy())
            loss = tr_loss / global_step if train_features is not None else None

            result['eval_loss'] = eval_loss
            result['global_step'] = global_step
            result['loss'] = loss

            output_eval_file = os.path.join(run_result_path, "{}_{:.0e}_{}_results.txt".format(
                task_name, settings.optimization_settings.learning_rate, index_sub_run))
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))


@dataclasses.dataclass
class Result:
    task_name: str
    training_variation_name: str
    index_run: int
    index_sub_run: int
    learning_rate: float
    metric: str
    value: float


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    # hard coded for now
    # parser.add_argument('--name', action='store', required=False, default='erp', help='Which set to run')
    parser.add_argument('--log_level', action='store', required=False, default='WARNING',
                        help='Sets the log-level. Defaults to WARNING')

    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.log_level.upper())

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst-2": Sst2Processor,
        "sts-b": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor,
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst-2": "classification",
        "sts-b": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification",
    }

    num_runs_per_input_run = {
        "cola": 1,
        "mnli": 1,
        "mrpc": 10,
        "sst-2": 1,
        "sts-b": 1,
        "qqp": 1,
        "qnli": 1,
        "rte": 10,
        "wnli": 10,
    }

    learning_rates = {
        "cola": 2e-5,
        "mnli": 5e-5,
        "mrpc": 2e-5,
        "sst-2": 5e-5,
        "sts-b": 3e-5,
        "qqp": 5e-5,
        "qnli": 3e-5,
        "rte": 2e-5,
        "wnli": 5e-5,
    }

    setattr(args, 'name', 'hp_meg_simple_fmri')

    training_variations, settings, num_runs, min_memory, aux_loss_tasks = named_variations(args.name)
    settings.optimization_settings.num_epochs_train_prediction_heads_only = 0
    settings.optimization_settings.num_final_epochs_train_prediction_heads_only = 0
    settings.optimization_settings.num_train_epochs = args.num_train_epochs

    # chosen_variation = None
    # for training_variation in training_variations:
    #     if isinstance(training_variation, TrainingVariation) and training_variation.loss_tasks == ('hp_fmri_I',):
    #         if chosen_variation is not None:
    #             raise ValueError('2 matches to hard coded training variation')
    #         chosen_variation = training_variation
    # if chosen_variation is None:
    #     raise ValueError('Unable to find variation which matches chosen variation')
    # training_variations = [chosen_variation]
    training_variations = [('hp_fmri_I',)]
    num_runs = 4

    if settings.optimization_settings.local_rank == -1 or settings.no_cuda:
        if not torch.cuda.is_available or settings.no_cuda:
            device = torch.device('cpu')
        else:
            device_id, free = cuda_most_free_device()
            torch.cuda.set_device(device_id)
            logger.info('binding to device {} with {} memory free'.format(device_id, free))
            device = torch.device('cuda:{}'.format(device_id))
        n_gpu = 1  # torch.cuda.device_count()
    else:
        torch.cuda.set_device(settings.optimization_settings.local_rank)
        device = torch.device("cuda", settings.optimization_settings.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(settings.optimization_settings.local_rank != -1), settings.optimization_settings.fp16))

    if settings.optimization_settings.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            settings.optimization_settings.gradient_accumulation_steps))

    settings.optimization_settings.train_batch_size = \
        settings.optimization_settings.train_batch_size \
        // settings.optimization_settings.gradient_accumulation_steps

    # if not args.do_train and not args.do_eval:
    #     raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    task_names = args.task_name.lower()
    if task_names == 'all':
        task_names = [k for k in learning_rates]
    elif task_names not in processors:
        raise ValueError("Task not found: {}".format(task_names))
    else:
        task_names = [task_names]

    def io_setup(training_variation_):
        temp_paths = Paths()
        hash_ = task_hash(training_variation_)
        model_path_ = os.path.join(temp_paths.model_path, args.name, hash_)
        result_path_ = os.path.join(temp_paths.result_path, args.name, hash_)

        if not os.path.exists(model_path_):
            os.makedirs(model_path_)
        if not os.path.exists(result_path_):
            os.makedirs(result_path_)

        return result_path_, model_path_

    tokenizer = BertTokenizer.from_pretrained(settings.bert_model, do_lower_case=True)

    results = list()

    for task_name in task_names:

        print(task_name)

        processor = processors[task_name]()
        output_mode = output_modes[task_name]

        train_features = None
        paths = Paths()
        if args.do_train:
            train_features = convert_examples_to_features(
                processor.get_train_examples(os.path.join(paths.glue_path, task_name.upper())),
                processor.get_labels(), args.max_seq_length, tokenizer, output_mode)

        eval_features = None
        mnli_mm_eval_features = None
        if args.do_eval:
            eval_features = convert_examples_to_features(
                processor.get_dev_examples(os.path.join(paths.glue_path, task_name.upper())),
                processor.get_labels(), args.max_seq_length, tokenizer, output_mode)
            if task_name == 'mnli':
                mnli_mm_processor = processors['mnli-mm']()
                mnli_mm_eval_features = convert_examples_to_features(
                    mnli_mm_processor.get_dev_examples(os.path.join(paths.glue_path, task_name.upper())),
                    processor.get_labels(), args.max_seq_length, tokenizer, output_mode)

        if args.do_train or args.do_eval:
            for training_variation in training_variations:
                result_path, model_path = io_setup(training_variation)
                for index_run in range(num_runs):
                    for index_sub_run in range(num_runs_per_input_run[task_name]):
                        settings.optimization_settings.learning_rate = learning_rates[task_name]
                        set_random_seeds(
                            settings.seed, index_run * num_runs_per_input_run[task_name] + index_sub_run, n_gpu)
                        _run_glue_for_variation(
                            model_path, result_path, task_name, train_features, eval_features, processor, output_mode,
                            settings, index_run, index_sub_run, device, n_gpu, mnli_mm_eval_features)
                        gc.collect()
                        torch.cuda.empty_cache()

        for training_variation in training_variations:
            result_path, model_path = io_setup(training_variation)
            for index_run in range(num_runs):
                for index_sub_run in range(num_runs_per_input_run[task_name]):
                    output_eval_file = os.path.join(
                        result_path,
                        'run_{}'.format(index_run),
                        '{}_{:.0e}_{}_results.txt'.format(task_name, learning_rates[task_name], index_sub_run))

                    training_variation_name = training_variation.name \
                        if isinstance(training_variation, TrainingVariation) else str(training_variation)

                    def _read_eval_file(eval_file_path, task_name_):
                        with open(eval_file_path, 'rt') as eval_file:
                            for line in eval_file:
                                line = line.strip()
                                if len(line) == 0:
                                    continue
                                index_equals = line.index(' = ')
                                metric = line[:index_equals].strip()
                                if metric == 'global_step' or metric == 'loss' or metric == 'eval_loss':
                                    continue
                                value_str = line[index_equals + len(' = '):].strip()
                                if value_str == 'None':
                                    continue
                                value = float(value_str)
                                results.append(Result(
                                    task_name_, training_variation_name, index_run, index_sub_run,
                                    learning_rates[task_name], metric, value))

                    _read_eval_file(output_eval_file, task_name)

                    if task_name == 'mnli':
                        second_task_name = 'mnli-mm'
                        output_eval_file_2 = os.path.join(
                            result_path,
                            'run_{}'.format(index_run),
                            '{}_{:.0e}_{}_results.txt'.format(
                                second_task_name, learning_rates[task_name], index_sub_run))
                        _read_eval_file(output_eval_file_2, second_task_name)

    results = sorted(
        results, key=lambda r: dataclasses.astuple(dataclasses.replace(r, value=0., index_sub_run=-1)))

    aggregated = list()
    for group_key, group in groupby(
            results, key=lambda r: dataclasses.astuple(dataclasses.replace(r, value=0., index_sub_run=-1))):
        group = list(group)
        total = sum(r.value for r in group)
        aggregated.append(dataclasses.replace(group[0], value=total/len(group), index_sub_run=-1))

    paths = Paths()
    summary_path = os.path.join(paths.result_path, args.name, 'glue_summary.txt')

    with open(summary_path, 'wt') as summary_file:
        for result in aggregated:
            summary_file.write(str(result))
            summary_file.write('\n')


if __name__ == "__main__":
    main()
