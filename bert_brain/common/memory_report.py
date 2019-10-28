import gc
import torch
import numpy as np


__all__ = ['print_tensor_sizes']


def print_tensor_sizes(gpu=True, cpu=True, count_only=False):
    obj_count = 0
    total_size = 0
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            is_cpu = str(obj.device) == 'cpu'
            if not is_cpu and not gpu:
                continue
            if is_cpu and not cpu:
                continue
            obj_count += 1
            total_size += int(np.prod(obj.size()))
            if not count_only:
                print(obj.type(), obj.size(), obj.device)
    print('count: {}, size: {}'.format(obj_count, total_size))
