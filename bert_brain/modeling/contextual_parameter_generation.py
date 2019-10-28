from collections import OrderedDict

import numpy as np
import torch
from torch import nn


class LinearContextualParameterGeneration(torch.nn.Module):

    def __init__(self, embedding_size, inner_module: torch.nn.Module, whitelist=None, blacklist=None):
        super().__init__()

        def gather_parameters(result, module_, prefix_=''):
            for name_, tensor in module_.named_parameters(prefix_[:-1], False):
                result[name_] = (module_, tensor.size())

            for name_, child in module_.named_children():
                if child is not None:
                    gather_parameters(result, child, prefix_ + name_ + '.')

        self._generated_parameters = OrderedDict()
        gather_parameters(self._generated_parameters, inner_module)

        self._splits = [int(np.prod(self._generated_parameters[k][1])) for k in self._generated_parameters]

        self.generator = torch.nn.Linear(embedding_size, sum(self._splits))
        # replace all the parameters with variables now so they don't show up in parameters
        for name in self._generated_parameters:
            module, _ = self._generated_parameters[name]
            current = getattr(module, name)
            setattr(module, name, current.data)
        self.inner_module = inner_module

    def forward(self, task_embedding, *args, **kwargs):
        parameters = self.generator(task_embedding)
        parameters = torch.split(parameters, self._splits, dim=-1)
        assert(len(parameters) == len(self._generated_parameters))
        for name, parameter in zip(self._generated_parameters, parameters):
            module, shape = self._generated_parameters[name]
            setattr(module, name, parameter.view(shape))
        return self.inner_module(*args, **kwargs)
