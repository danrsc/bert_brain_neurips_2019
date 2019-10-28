import time
import itertools
import gc
from functools import partial
from contextlib import contextmanager
import queue
from tqdm import tqdm
from threading import Thread
from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch


__all__ = ['cuda_map_unordered',
           'cuda_pool_executor',
           'cuda_most_free_device',
           'cuda_memory_info',
           'DeviceMemoryInfo',
           'ProgressContext',
           'cuda_auto_empty_cache_context']


@contextmanager
def cuda_auto_empty_cache_context(device):

    if not isinstance(device, torch.device):
        device = torch.device(device)

    if device.type == 'cpu':
        yield device
    else:
        with torch.cuda.device(device) as device_context:
            yield device_context
            gc.collect()
            torch.cuda.empty_cache()


def _set_device_id_and_initialize(device_queue, no_available_queue, initializer, initargs):

    try:
        device_id = device_queue.get(timeout=5)
    except queue.Empty:
        no_available_queue.put(1)
        device_id = device_queue.get()

    if device_id < 0:
        return

    print('binding to device {}'.format(device_id))

    torch.cuda.set_device(device_id)

    if initializer is not None:
        if initargs is None:
            initargs = ()
        initializer(*initargs)


def _monitor_devices(max_workers, min_memory, starting_ids, device_queue, no_available_queue):

    current_use = dict()
    for device_id in starting_ids:
        if device_id not in current_use:
            current_use[device_id] = min_memory
        else:
            current_use[device_id] += min_memory

    needed_count = 0
    while True:
        try:
            need = no_available_queue.get(timeout=100)
            if need == 2:  # signals shutdown
                for _ in range(max_workers):  # signal workers waiting on a device to exit
                    device_queue.put(-1)
                return
            else:
                needed_count += 1
        except queue.Empty:
            pass

        if needed_count > 0:
            memory_info = cuda_memory_info()
            selected_free = None
            selected_device = None
            for device_id, device_memory in enumerate(memory_info):
                projected_use = current_use[device_id] if device_id in current_use else 0
                device_free = device_memory.free + torch.cuda.memory_allocated(device_id) - projected_use
                if device_free > min_memory and (selected_free is None or device_free > selected_free):
                    selected_free = device_free
                    selected_device = device_id

            if selected_device is not None:
                if selected_device not in current_use:
                    current_use[selected_device] = min_memory
                else:
                    current_use[selected_device] += min_memory
                device_queue.put(selected_device)
                needed_count -= 1


def _cuda_memory_retry_wrap(retry_item):
    try:
        args = () if retry_item.args is None else retry_item.args
        retry_item.result = retry_item.func(*args)
        retry_item.exception = None
    except RuntimeError as e:
        if str(e) != 'CUDA error: out of memory':
            raise
        retry_item.result = None
        retry_item.exception = e

    return retry_item


class OutOfMemoryRetry(object):

    def __init__(self, func, args):
        self.func = func
        self.args = args
        self.num_tries = 0
        self.result = None
        self.exception = None


_progress_stop_sentinel = 'kill_progress_monitor'


def _monitor_progress(progress_t, progress_queue):
    while True:
        p = progress_queue.get()
        if p == _progress_stop_sentinel:
            return
        if p < 0:
            progress_t.n = progress_t.n - p
            progress_t.refresh()
        else:
            progress_t.update(p)


class ProgressContext(object):

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs
        self._mp_context = None
        self._progress_queue = None
        self._progress_bar = None

    @property
    def mp_context(self):
        return self._mp_context

    @property
    def progress_queue(self):
        return self._progress_queue

    @property
    def progress_bar(self):
        return self._progress_bar

    def __enter__(self):
        self._mp_context = get_context('spawn')
        self._progress_queue = self.mp_context.Queue()
        self._progress_bar = tqdm(*self._args, **self._kwargs)
        self._args = None
        self._kwargs = None
        progress_monitor = Thread(target=_monitor_progress, args=(self.progress_bar, self.progress_queue), daemon=True)
        progress_monitor.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.close()
        self.progress_queue.put(_progress_stop_sentinel)


def cuda_map_unordered(
        min_memory, func, iterables,
        max_workers=None, mp_context=None, initializer=None, initargs=None,
        num_cuda_memory_retries=0, chunksize=1):

    items = iterables

    if num_cuda_memory_retries > 0:
        items = map(lambda args: OutOfMemoryRetry(func, args), zip(*items))

    finished = False

    while not finished:
        retries = list()

        with cuda_pool_executor(
                min_memory, max_workers, mp_context=mp_context, initializer=initializer, initargs=initargs) as ex:

            if num_cuda_memory_retries > 0:
                result = ex.map(_cuda_memory_retry_wrap, items, chunksize=chunksize)
                for item in result:
                    if item.exception is not None:
                        if item.num_tries < num_cuda_memory_retries:
                            item.num_tries += 1
                            item.exception = None
                            item.result = None
                            retries.append(item)
                        else:
                            raise item.exception
                    else:
                        yield item.result
            else:
                for item in ex.map(func, *items, chunksize=chunksize):
                    yield item

            finished = len(retries) == 0
            items = retries


def _get_chunks(*iterables, chunksize):
    """ Iterates over zip()ed iterables in chunks. """
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(fn, chunk):
    """ Processes a chunk of an iterable passed to map.
    Runs the function passed to map() on a chunk of the
    iterable passed to map.
    This function is run in a separate process.
    """
    return [fn(*args) for args in chunk]


def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


class _RetryItem:

    def __init__(self, should_retry_fn, fn, args, kwargs):
        self.should_retry_fn = should_retry_fn
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class CudaOutOfMemoryShouldRetry(object):

    def __init__(self, num_retries):
        self._num_retries = num_retries

    def __call__(self, ex):
        if str(ex) != 'CUDA error: out of memory':
            return False
        if self._num_retries > 0:
            self._num_retries -= 1
            return True
        return False


class CudaPoolExecutor(object):

    @staticmethod
    def _create_process_pool_executor(min_memory, max_workers, mp_context, initializer, initargs):
        memory_info = cuda_memory_info()
        if max_workers is None:
            max_workers = len(memory_info)
        if mp_context is None:
            mp_context = get_context('spawn')

        memory_info = [
            (device_id, device_info.free) for device_id, device_info in enumerate(memory_info)]

        memory_info = sorted(memory_info, key=lambda id_free_total: (-id_free_total[1], id_free_total[0]))

        device_ids = list()
        for i in range(max_workers):
            use, current = divmod(i, len(memory_info))
            free = memory_info[current][1] - use * min_memory
            if free >= min_memory:
                device_ids.append(memory_info[current][0])
            elif current == 0:
                break  # the most free one is all used up, so the others must be too

        if len(device_ids) == 0:
            raise ValueError('No devices with enough memory available')

        device_queue = mp_context.Queue()
        no_available_queue = mp_context.Queue()
        for device_id in device_ids:
            device_queue.put(device_id)

        device_monitor = Thread(target=_monitor_devices, args=(
            max_workers, min_memory, device_ids, device_queue, no_available_queue), daemon=True)
        device_monitor.start()

        process_pool_executor = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp_context,
            initializer=_set_device_id_and_initialize,
            initargs=(device_queue, no_available_queue, initializer, initargs))

        return process_pool_executor, no_available_queue

    def __init__(self, min_memory, max_workers=None, mp_context=None, initializer=None, initargs=()):
        self._process_pool_executor, self._no_available_queue = CudaPoolExecutor._create_process_pool_executor(
            min_memory, max_workers, mp_context, initializer, initargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown(wait=True)

    def _map(self, fn, *iterables, timeout=None, num_cuda_memory_retries=0):
        """Returns an iterator equivalent to map(fn, iter).
        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.
        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        if timeout is not None:
            end_time = timeout + time.monotonic()

        retry_items = None
        if num_cuda_memory_retries > 0:
            retry_items = list()
            fs = list()
            for args in zip(*iterables):
                fs.append(self._process_pool_executor.submit(fn, *args))
                retry_items.append(_RetryItem(CudaOutOfMemoryShouldRetry(num_cuda_memory_retries), fn, args, {}))
        else:
            fs = [self._process_pool_executor.submit(fn, *args) for args in zip(*iterables)]

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator():
            try:
                # reverse to keep finishing order
                fs.reverse()
                if retry_items is not None:
                    retry_items.reverse()
                while fs:
                    # Careful not to keep a reference to the popped future
                    if retry_items:
                        retry_item = retry_items.pop()
                    else:
                        retry_item = None

                    while True:
                        try:
                            if timeout is None:
                                yield fs.pop().result()
                                retry_item = None
                            else:
                                yield fs.pop().result(end_time - time.monotonic())
                                retry_item = None
                        except BaseException as ex:
                            if retry_item is None or not retry_item.should_retry_fn(ex):
                                raise
                            else:
                                fs.append(self._process_pool_executor.submit(retry_item.fn, *retry_item.args))
            finally:
                for future in fs:
                    future.cancel()

        return result_iterator()

    def map_unordered(self, fn, *iterables, timeout=None, num_cuda_memory_retries=0):
        """Returns an iterator equivalent to map(fn, iter).
        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            num_cuda_memory_retries: The number of times to retry if we get a CudaOutOfMemory exception. Retries
                are counted separately on each item
        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.
        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """

        if timeout is not None:
            end_time = timeout + time.monotonic()

        fs = dict()
        for args in zip(*iterables):
            retry_item = None
            if num_cuda_memory_retries > 0:
                retry_item = _RetryItem(CudaOutOfMemoryShouldRetry(num_cuda_memory_retries), fn, args, {})
            fs[self._process_pool_executor.submit(fn, *args)] = retry_item

        # Yield must be hidden in closure so that the futures are submitted
        # before the first iterator value is required.
        def result_iterator(futures):
            futures = dict(futures)
            next_fs = dict()
            try:
                while len(futures) > 0:
                    next_fs = dict()
                    if timeout is None:
                        iterable = as_completed(futures)
                    else:
                        iterable = as_completed(futures, end_time - time.monotonic())
                    for future in iterable:
                        retry_item_ = futures[future]
                        del futures[future]
                        try:
                            yield future.result()
                            retry_item_ = None
                        except BaseException as ex:
                            if retry_item_ is None or not retry_item_.should_retry_fn(ex):
                                raise
                            next_fs[self._process_pool_executor.submit(retry_item_.fn, *retry_item_.args)] = retry_item_
                    futures = next_fs
            finally:
                for future in futures:
                    future.cancel()
                for future in next_fs:
                    future.cancel()

        return result_iterator(fs)

    def map(self, func, *iterables, timeout=None, chunksize=1, num_cuda_memory_retries=0):

        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = self._map(
            partial(_process_chunk, func),
            _get_chunks(*iterables, chunksize=chunksize),
            timeout=timeout,
            num_cuda_memory_retries=num_cuda_memory_retries)
        return _chain_from_iterable_of_lists(results)

    def submit(self, fn, *args, **kwargs):
        return self._process_pool_executor.submit(fn, *args, **kwargs)

    def shutdown(self, wait=True):
        self._no_available_queue.put(2)
        self._process_pool_executor.shutdown(wait)


@contextmanager
def cuda_pool_executor(min_memory, max_workers=None, mp_context=None, initializer=None, initargs=()):
    memory_info = cuda_memory_info()
    if max_workers is None:
        max_workers = len(memory_info)
    if mp_context is None:
        mp_context = get_context('spawn')

    memory_info = [
        (device_id, device_info.free) for device_id, device_info in enumerate(memory_info)]

    memory_info = sorted(memory_info, key=lambda id_free_total: (-id_free_total[1], id_free_total[0]))

    device_ids = list()
    for i in range(max_workers):
        use, current = divmod(i, len(memory_info))
        free = memory_info[current][1] - use * min_memory
        if free >= min_memory:
            device_ids.append(memory_info[current][0])
        elif current == 0:
            break  # the most free one is all used up, so the others must be too

    if len(device_ids) == 0:
        raise ValueError('No devices with enough memory available')

    device_queue = mp_context.Queue()
    no_available_queue = mp_context.Queue()
    for device_id in device_ids:
        device_queue.put(device_id)

    device_monitor = Thread(target=_monitor_devices, args=(
        max_workers, min_memory, device_ids, device_queue, no_available_queue), daemon=True)
    device_monitor.start()

    yield ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
        initializer=_set_device_id_and_initialize,
        initargs=(device_queue, no_available_queue, initializer, initargs))

    no_available_queue.put(2)


def _local_thread_cuda_memory_info():
    import numba.cuda
    try:
        result = list()
        for i in range(len(numba.cuda.gpus)):
            try:
                numba.cuda.select_device(i)
                context = numba.cuda.current_context()
                free, total = context.get_memory_info()
            except numba.cuda.cudadrv.driver.CudaAPIError:
                # assume this is an out of memory error
                free = 0
                total = float('nan')
            result.append(DeviceMemoryInfo(free, total))
            numba.cuda.close()
        return result
    finally:
        numba.cuda.close()


def _cuda_memory_info(q):
    # execute this in a separate process so numba.cuda.close() does not mess up any current device usage
    q.put(_local_thread_cuda_memory_info())


class DeviceMemoryInfo(object):

    def __init__(self, free, total):
        self.free, self.total = free, total

    def __str__(self):
        return '{}(free={}, total={})'.format(type(self), self.free, self.total)


def cuda_memory_info():
    ctx = get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=_cuda_memory_info, args=(q,))
    p.start()
    result = q.get()
    return result


def cuda_most_free_device():
    device_id = None
    free = None
    for i, memory_info in enumerate(cuda_memory_info()):
        if free is None or memory_info.free > free:
            free = memory_info.free
            device_id = i
    return device_id, free
