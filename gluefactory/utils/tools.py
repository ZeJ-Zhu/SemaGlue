"""
Various handy Python and PyTorch utils.

Author: Paul-Edouard Sarlin (skydes)
"""

import os
import random
import time
from collections.abc import Iterable
from contextlib import contextmanager

import numpy as np
import torch


class AverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0

    def update(self, tensor):
        assert tensor.dim() == 1
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        if self._num_examples == 0:
            return np.nan
        else:
            return self._sum / self._num_examples


# same as AverageMetric, but tracks all elements
class FAverageMetric:
    def __init__(self):
        self._sum = 0
        self._num_examples = 0
        self._elements = []

    def update(self, tensor):
        self._elements += tensor.cpu().numpy().tolist()
        assert tensor.dim() == 1
        tensor = tensor[~torch.isnan(tensor)]
        self._sum += tensor.sum().item()
        self._num_examples += len(tensor)

    def compute(self):
        if self._num_examples == 0:
            return np.nan
        else:
            return self._sum / self._num_examples


class MedianMetric:
    def __init__(self):
        self._elements = []

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return np.nanmedian(self._elements)


class PRMetric:
    def __init__(self):
        self.labels = []
        self.predictions = []

    @torch.no_grad()
    def update(self, labels, predictions, mask=None):
        assert labels.shape == predictions.shape
        self.labels += (
            (labels[mask] if mask is not None else labels).cpu().numpy().tolist()
        )
        self.predictions += (
            (predictions[mask] if mask is not None else predictions)
            .cpu()
            .numpy()
            .tolist()
        )

    @torch.no_grad()
    def compute(self):
        return np.array(self.labels), np.array(self.predictions)

    def reset(self):
        self.labels = []
        self.predictions = []


class QuantileMetric:
    def __init__(self, q=0.05):
        self._elements = []
        self.q = q

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return np.nanquantile(self._elements, self.q)


class RecallMetric:
    def __init__(self, ths, elements=[]):
        self._elements = elements
        self.ths = ths

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if isinstance(self.ths, Iterable):
            return [self.compute_(th) for th in self.ths]
        else:
            return self.compute_(self.ths[0])

    def compute_(self, th):
        if len(self._elements) == 0:
            return np.nan
        else:
            s = (np.array(self._elements) < th).sum()
            return s / len(self._elements)


def cal_error_auc(errors, thresholds):
    #对错误率进行排序
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    #计算召回率 (i+1)/n数组
    recall = (np.arange(len(errors)) + 1) / len(errors)
    #开头插入一个零
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    #二分查找
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.round((np.trapz(r, x=e) / t), 4))
    return aucs


class AUCMetric:
    def __init__(self, thresholds, elements=None):
        self._elements = elements
        self.thresholds = thresholds
        if not isinstance(thresholds, list):
            self.thresholds = [thresholds]

    def update(self, tensor):
        assert tensor.dim() == 1
        self._elements += tensor.cpu().numpy().tolist()

    def compute(self):
        if len(self._elements) == 0:
            return np.nan
        else:
            return cal_error_auc(self._elements, self.thresholds)


class Timer(object):
    """A simpler timer context object.
    Usage:
    ```
    > with Timer('mytimer'):
    >   # some computations
    [mytimer] Elapsed: X
    ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.duration = time.time() - self.tstart
        if self.name is not None:
            print("[%s] Elapsed: %s" % (self.name, self.duration))


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """#通过模块路径和基类信息，获取定义在该模块中的继承自基类的类对象。在这里，主要用于获取模型类
    import inspect

    mod = __import__(mod_path, fromlist=[""])#fromlist参数指定导入模块时，需要包含的子模块或对象的名称，这里只导入模块本身，不导入子模块或对象
    classes = inspect.getmembers(mod, inspect.isclass)#inspect.getmembers获取模块中的所有类对象，inspect.isclass函数用于过滤出类对象。
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]#过滤出定义在指定模块中的类对象，确保它们的__module__属性与给定的模块路径相匹配
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]#再次过滤，只保留继承自指定基类“BaseClass"的类对象
    assert len(classes) == 1, classes#确保最终结果只有一个类对象。
    return classes[0][1]


def set_num_threads(nt):#用于设置全局线程数，强制限制 NumPy 等库使用的线程数。这对于确保训练的可重复性很有用
    """Force numpy and other libraries to use a limited number of threads."""
    try:
        import mkl
    except ImportError:
        pass
    else:
        mkl.set_num_threads(nt)
    torch.set_num_threads(1)
    os.environ["IPC_ENABLE"] = "1"
    for o in [
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[o] = str(nt)


def set_seed(seed):#可复现性
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_random_state(with_cuda):
    #获取pytorch随机数生成器的状态
    pth_state = torch.get_rng_state()
    #获取Numpy随机数生成器的状态
    np_state = np.random.get_state()
    #获取python内置的random模块的状态
    py_state = random.getstate()
    #获取cuda设备上的随机数生成器的状态
    if torch.cuda.is_available() and with_cuda:
        cuda_state = torch.cuda.get_rng_state_all()
    else:
        cuda_state = None
    return pth_state, np_state, py_state, cuda_state


def set_random_state(state):
    pth_state, np_state, py_state, cuda_state = state
    torch.set_rng_state(pth_state)
    np.random.set_state(np_state)
    random.setstate(py_state)
    if (
        cuda_state is not None
        and torch.cuda.is_available()
        and len(cuda_state) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda_state)


@contextmanager
def fork_rng(seed=None, with_cuda=True):#语句块中，可以执行一些可能影响随机状态的操作，而不会影响到其他部分。在结束语句块后，会恢复到之前的随机状态。这对于在实验中进行一些具有随机性的操作并确保实验可复现性很有用
    state = get_random_state(with_cuda)
    if seed is not None:
        set_seed(seed)
    try:
        yield
    finally:
        set_random_state(state)
