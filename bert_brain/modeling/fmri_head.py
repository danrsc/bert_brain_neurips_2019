from collections import OrderedDict

import numpy as np
import torch

from .utility_modules import Conv1DCausal


__all__ = ['FMRIConvConvWithDilationHead']


class FMRIConvConvWithDilationHead(torch.nn.Module):

    def __init__(
            self,
            in_sequence_channels,
            in_pooled_channels,
            prediction_key_to_shape,
            hidden_channels=10,
            hidden_kernel_size=5,
            out_kernel_size=5,
            out_dilation=5,
            memory_efficient=False,
            index_layer=-1):
        super().__init__()
        # the total number of tokens giving information to the output layer token is going to be:
        # out_dilation * (out_kernel_size - 1) + (hidden_kernel_size - 1)
        # so with the default settings of 5, 5, 5 we would see 24 tokens. Bearing in mind that some of these
        # will be punctuation, we are hoping to get roughly 16 words, i.e. 8 seconds of time
        self.prediction_key_to_shape = OrderedDict(prediction_key_to_shape)
        self.splits = [int(np.prod(self.prediction_key_to_shape[k])) for k in self.prediction_key_to_shape]
        self.index_layer = index_layer
        out_channels = sum(self.splits)
        self.memory_efficient = memory_efficient
        self.out_kernel_size = out_kernel_size
        self.out_dilation = out_dilation
        self.conv1d_hidden = Conv1DCausal(
            in_sequence_channels,
            hidden_channels,
            hidden_kernel_size,
            transpose_axes=(0, 2, 1),
            should_transpose_output=self.memory_efficient)
        if self.memory_efficient:
            # since the number of targets is small compared to the size of the sequence,
            # we can manually concatenate the inputs to a linear layer and only make
            # predictions for the targets instead of applying a true convolution if we want to
            # save memory. This will be slower
            self.output = torch.nn.Linear(hidden_channels * out_kernel_size, out_channels)
        else:
            self.output = Conv1DCausal(
                hidden_channels,
                out_channels,
                out_kernel_size,
                dilation=out_dilation,
                transpose_axes=(0, 2, 1),
                should_transpose_input=False)

    def forward(self, sequence_output, pooled_output, batch):
        all_data_ids = [batch[(k, 'data_ids')] for k in self.prediction_key_to_shape]
        for idx in range(1, len(all_data_ids)):
            if not torch.equal(all_data_ids[0], all_data_ids[idx]):
                raise ValueError('Inconsistent data_ids cannot be used within the same instance of FMRIHead')
        data_ids = all_data_ids[0]

        hidden = self.conv1d_hidden(sequence_output[self.index_layer])
        if self.memory_efficient:

            # pad the sequence so we don't index out of bounds or into other examples in the batch
            hidden = torch.nn.functional.pad(
                hidden, (0, self.out_dilation * (self.out_kernel_size - 1), 0), mode='constant', value=0)
            data_ids = torch.reshape(data_ids, (data_ids.size()[0], data_ids.size()[1]))  # ensure data_ids is 2D
            data_ids = torch.nn.functional.pad(
                data_ids, (0, self.out_dilation * (self.out_kernel_size - 1)), mode='constant', value=-1)
            example_ids = torch.arange(len(data_ids), device=hidden.device).view((-1, 1)).repeat((1, data_ids.size(1)))

            # flatten to (batch * sequence, channels)
            hidden = torch.reshape(hidden, (hidden.size()[0] * hidden.size()[1], hidden.size(2)))
            data_ids = torch.reshape(data_ids, (data_ids.size()[0] * data_ids.size()[1],))
            example_ids = torch.reshape(example_ids, (example_ids.size()[0] * example_ids.size()[1],))

            # find positive data_ids
            indices_keep = torch.nonzero(data_ids >= 0)

            # remember what the data_ids are
            data_ids = data_ids[indices_keep]
            example_ids = example_ids[indices_keep]

            # now get the indices for the kernel, tiliing the indices on axis=1 and then subtracting the dilation
            # offsets
            indices_keep = indices_keep.repeat((1, self.out_kernel_size - 1))
            offsets = \
                torch.arange(start=-(self.out_kernel_size - 1), end=1, step=1, device=indices_keep.device) \
                * self.out_dilation
            indices_keep = indices_keep + offsets.view((offsets.size()[0], 1))

            # flatten the kernel indices to 1D
            indices_keep = indices_keep.view(-1)

            # select the hidden data according to the indices we computed
            hidden = hidden[indices_keep]

            # reshape to get (batch, kernel * channels)
            hidden = torch.reshape(
                hidden, (hidden.size()[0] / self.out_kernel_size, self.out_kernel_size * hidden.size()[1]))
            # make predictions
            predictions = self.output(hidden)
        else:
            predictions = self.output(hidden)
            example_ids = torch.reshape(torch.arange(len(data_ids), device=hidden.device), (-1, 1))
            example_ids = example_ids.repeat((1, data_ids.size(1)))

            predictions = torch.reshape(
                predictions, (predictions.size()[0] * predictions.size()[1],) + predictions.size()[2:])
            data_ids = torch.reshape(data_ids, (data_ids.size()[0] * data_ids.size()[1],))
            example_ids = torch.reshape(example_ids, (example_ids.size()[0] * example_ids.size()[1],))

            indicator_keep = data_ids >= 0
            predictions = predictions[indicator_keep]
            data_ids = data_ids[indicator_keep]
            example_ids = example_ids[indicator_keep]

        predictions = torch.split(predictions, self.splits, dim=-1)
        result = OrderedDict()
        assert (len(self.prediction_key_to_shape) == len(predictions))
        for k, p in zip(self.prediction_key_to_shape, predictions):
            p = p.view(p.size()[:1] + self.prediction_key_to_shape[k])
            result[k] = p
            result[(k, 'data_ids')] = data_ids
            result[(k, 'example_ids')] = example_ids
        return result
