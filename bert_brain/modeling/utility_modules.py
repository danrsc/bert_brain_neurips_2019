import numpy as np
import torch


__all__ = ['GroupPool', 'GroupConcat', 'Conv1DCausal', 'at_most_one_data_id', 'k_data_ids']


def at_most_one_data_id(data_ids, return_first_index=False, return_last_index=False):

    if len(data_ids.size()) != 2:
        raise ValueError('data_ids must be 2D')

    maxes, _ = torch.max(data_ids, dim=1)
    repeated_maxes = torch.reshape(maxes, (-1, 1)).repeat((1, data_ids.size()[1]))
    mins, _ = torch.min(torch.where(data_ids < 0, repeated_maxes, data_ids), dim=1)

    if torch.sum(maxes != mins) > 0:
        raise ValueError('More than one data_id exists for some examples')

    if return_first_index or return_last_index:
        index_array = torch.arange(data_ids.size()[1], device=data_ids.device).view(
            (1, data_ids.size()[1])).repeat((data_ids.size()[0], 1))
        indicator_valid = data_ids >= 0
        first_index = None
        if return_first_index:
            first_index_array = torch.where(
                indicator_valid, index_array, torch.full_like(index_array, data_ids.size()[1] + 1))
            first_index, _ = torch.min(first_index_array, dim=1)
        last_index = None
        if return_last_index:
            last_index_array = torch.where(indicator_valid, index_array, torch.full_like(index_array, -1))
            last_index, _ = torch.max(last_index_array, dim=1)
        if return_first_index and return_last_index:
            return maxes, first_index, last_index
        if return_first_index:
            return maxes, first_index
        if return_last_index:
            return maxes, last_index

    return maxes


def k_data_ids(k, data_ids, return_indices=False, check_unique=False):

    if len(data_ids.size()) != 2:
        raise ValueError('data_ids must be 2D')

    indicator_valid = data_ids >= 0
    count_valid = torch.sum(indicator_valid, dim=1)
    if torch.max(count_valid) != k or torch.min(count_valid) != k:
        print(count_valid)
        raise ValueError('Incorrect number of data_ids. Expected {}'.format(k))

    data_ids = torch.masked_select(data_ids, indicator_valid)
    data_ids = torch.reshape(data_ids, (data_ids.size()[0], k))

    if check_unique:
        mins, _ = torch.min(data_ids, dim=1)
        maxes, _ = torch.max(data_ids, dim=1)
        if torch.sum(maxes != mins) > 0:
            raise ValueError('More than one data_id exists for some examples')

    if return_indices:
        index_array = torch.arange(data_ids.size()[1], device=data_ids.device).view(
            (1, data_ids.size()[1])).repeat((data_ids.size()[0], 1))
        indices = torch.masked_select(index_array, indicator_valid)
        indices = torch.reshape(indices, (indicator_valid.size()[0], k))
        return data_ids, indices

    return data_ids


class GroupConcat(torch.nn.Module):

    def __init__(self, num_per_group):
        super().__init__()
        self.num_per_group = num_per_group

    # noinspection PyMethodMayBeStatic
    def forward(self, x, groupby):

        # first attach an example_id to the groups to ensure that we don't concat across examples in the batch

        # array of shape (batch, sequence, 1) which identifies example
        example_ids = torch.arange(
            groupby.size()[0], device=x.device).view((groupby.size()[0], 1, 1)).repeat((1, groupby.size()[1], 1))

        # indices to ensure stable sort, and to give us indices_sort
        indices = torch.arange(groupby.size()[0] * groupby.size()[1], device=x.device).view(groupby.size() + (1,))

        # -> (batch, sequence, 3): attach example_id to each group and add indices to guarantee stable sort
        groupby = torch.cat((example_ids, groupby.view(groupby.size() + (1,)), indices), dim=2)

        # -> (batch * sequence, 3)
        groupby = groupby.view((groupby.size()[0] * groupby.size()[1], groupby.size()[2]))

        # filter out the bogus groupby
        groupby = groupby[groupby[:, 1] >= 0]

        # this allows us to sort the 3 dimensions together
        groups = torch.unique(groupby, sorted=True, dim=0)

        _, counts = torch.unique_consecutive(groups[:, :2], return_counts=True, dim=0)

        # check that the input is what we expected
        if torch.min(counts) != self.num_per_group or torch.max(counts) != self.num_per_group:
            raise ValueError('Expected exactly {} per unique groupby. min count: {}, max count: {}'.format(
                self.num_per_group, torch.min(counts), torch.max(counts)))

        # get the true groups and example_ids
        example_ids = groups[:, 0]
        indices_sort = groups[:, 2]
        groups = groups[:, 1]

        # -> (batch * sequence, n, m, ..., k)
        x = x.view((x.size()[0] * x.size()[1],) + x.size()[2:])

        # sort x so that grouped items are together
        x = x[indices_sort]

        x = x.view((x.size()[0] // self.num_per_group, self.num_per_group) + x.size()[1:])
        groups = groups.view((groups.size()[0] // self.num_per_group, self.num_per_group))
        example_ids = example_ids.view((example_ids.size()[0] // self.num_per_group, self.num_per_group))

        # all of these are the same on axis=1, so take the first
        groups = groups[:, 0]
        example_ids = example_ids[:, 0]

        return x, groups, example_ids


class GroupPool(torch.nn.Module):

    # noinspection PyMethodMayBeStatic
    def forward(self, x, groupby):

        # first attach an example_id to the groups to ensure that we don't pool across examples in the batch

        # array of shape (batch, sequence, 1) which identifies example
        example_ids = torch.arange(
            groupby.size()[0], device=x.device).view((groupby.size()[0], 1, 1)).repeat((1, groupby.size()[1], 1))
        # -> (batch, sequence, 2): attach example_id to each group
        groupby = torch.cat((example_ids, groupby.view(groupby.size() + (1,))), dim=2)

        # -> (batch * sequence, 2)
        groupby = groupby.view((groupby.size()[0] * groupby.size()[1], groupby.size()[2]))

        # each group is a (example_id, group) tuple
        groups, group_indices = torch.unique(groupby, sorted=True, return_inverse=True, dim=0)

        # split the groups into the true groups and example_ids
        example_ids = groups[:, 0]
        groups = groups[:, 1]

        # -> (batch * sequence, 1, 1, ..., 1)
        group_indices = group_indices.view((x.size()[0] * x.size()[1],) + (1,) * (len(x.size()) - 2))

        # -> (batch * sequence, n, m, ..., k)
        group_indices = group_indices.repeat((1,) + x.size()[2:])

        # -> (batch * sequence, n, m, ..., k)
        x = x.view((x.size()[0] * x.size()[1],) + x.size()[2:])

        pooled = torch.zeros((groups.size()[0],) + x.size()[1:], dtype=x.dtype, device=x.device)
        pooled.scatter_add_(dim=0, index=group_indices, src=x)

        # -> (batch * sequence)
        group_indices = group_indices[:, 0]
        counts = torch.zeros(groups.size()[0], dtype=x.dtype, device=x.device)
        counts.scatter_add_(
            dim=0, index=group_indices, src=torch.ones(len(group_indices), dtype=x.dtype, device=x.device))
        counts = counts.view(counts.size() + (1,) * len(pooled.size()[1:]))
        pooled = pooled / counts

        # filter out groups < 0
        indicator_valid = groups >= 0
        pooled = pooled[indicator_valid]
        groups = groups[indicator_valid]
        example_ids = example_ids[indicator_valid]

        return pooled, groups, example_ids


class Conv1DCausal(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True,
                 transpose_axes=None, should_transpose_input=True, should_transpose_output=True):
        super().__init__()
        self.transpose_axes = transpose_axes
        self.should_transpose_input = should_transpose_input
        self.should_transpose_output = should_transpose_output
        padding = dilation * (kernel_size - 1)
        self.conv1d = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        if self.transpose_axes is not None and self.should_transpose_input:
            x = x.permute(*self.transpose_axes)
        result = self.conv1d(x)
        # remove the element from the right padding
        result = result[:, :, :-self.conv1d.padding[0]]
        if self.transpose_axes is not None and self.should_transpose_output:
            result = result.permute(*np.argsort(self.transpose_axes))
        return result
