import string

import h5py
import torch
from pathlib import Path
from ..datasets.base_dataset import collate
from ..settings import DATA_PATH
from ..utils.tensor import batch_to_device
from .base_model import BaseModel
from .utils.misc import pad_to_length

#定义了一个名为 pad_local_features 的函数，用于填充局部特征数据，使其达到指定的序列长度 seq_l。
#具体来说，它通过调用 pad_to_length 函数对输入字典 pred 中的多个键对应的值进行填充操作
def pad_local_features(pred: dict, seq_l: int):
    #对关键点进行填充，使用 pad_to_length 函数，填充值为 -2，填充模式为 "random_c"
    pred["keypoints"] = pad_to_length(
        pred["keypoints"],
        seq_l,
        -2,
        mode="random_c",
    )
    if "keypoint_scores" in pred.keys():
        pred["keypoint_scores"] = pad_to_length(
            pred["keypoint_scores"], seq_l, -1, mode="zeros"
        )
    if "descriptors" in pred.keys():
        pred["descriptors"] = pad_to_length(
            pred["descriptors"], seq_l, -2, mode="random"
        )
    if "scales" in pred.keys():
        pred["scales"] = pad_to_length(pred["scales"], seq_l, -1, mode="zeros")
    if "oris" in pred.keys():
        pred["oris"] = pad_to_length(pred["oris"], seq_l, -1, mode="zeros")

    if "depth_keypoints" in pred.keys():
        pred["depth_keypoints"] = pad_to_length(
            pred["depth_keypoints"], seq_l, -1, mode="zeros"
        )
    if "valid_depth_keypoints" in pred.keys():
        pred["valid_depth_keypoints"] = pad_to_length(
            pred["valid_depth_keypoints"], seq_l, -1, mode="zeros"
        )
    return pred


def pad_line_features(pred, seq_l: int = None):
    raise NotImplementedError


def recursive_load(grp, pkeys):
    return {
        k: torch.from_numpy(grp[k].__array__())
        if isinstance(grp[k], h5py.Dataset)
        else recursive_load(grp[k], list(grp.keys()))
        for k in pkeys
    }


class CacheLoader(BaseModel):
    default_conf = {
        "path": "???",  # can be a format string like exports/{scene}/
        "data_keys": None,  # load all keys
        "device": None,  # load to same device as data
        "trainable": False,
        "add_data_path": True,
        "collate": True,
        "scale": ["keypoints", "lines", "orig_lines"],
        "padding_fn": None,
        "padding_length": None,  # required for batching!
        "numeric_type": "float32",  # [None, "float16", "float32", "float64"]
    }

    required_data_keys = ["name"]  # we need an identifier

    #初始化方法，用于设置类的属性。在这里，初始化了缓存文件对象 (hfiles) 和填充函数 (padding_fn)
    #并将数值类型 (numeric_dtype) 转换为 PyTorch 的数据类型
    def _init(self, conf):
        self.hfiles = {}
        self.padding_fn = conf.padding_fn
        if self.padding_fn is not None:
            self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    #缓存文件中的数据加载到 PyTorch 张量中，并在加载过程中进行一些处理，以适应模型的需求
    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:#检查配置中是否指定了设备
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        #解析路径模板中的变量名var_names
        var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        #针对每个数据样本，根据路径模板和样本信息构建缓存文件路径。
        #如果配置中要求添加数据路径，则将数据路径添加到缓存文件路径中
        for i, name in enumerate(data["name"]):
            fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
            if self.conf.add_data_path:
                fpath = Path(DATA_PATH) / fpath
            #使用h5py库打开hdf5文件，根据样本名称获取相应的数据组（'grp')
            hfile = h5py.File(str(fpath), "r")
            grp = hfile[name]
            pkeys = (
                self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
            )
            #使用递归加载函数recursive_load从hdf5数据组中加载指定键("pkeys")的数据
            pred = recursive_load(grp, pkeys)
            #如果数值类型numeric_dtype指定了，则将加载的张量转换为相应的pytorch数据类型
            if self.numeric_dtype is not None:
                pred = {
                    k: v
                    if not isinstance(v, torch.Tensor) or not torch.is_floating_point(v)
                    else v.to(dtype=self.numeric_dtype)
                    for k, v in pred.items()
                }
            #将加载的数据转移到指定的设备，并根据配置中的缩放要求对特定键的数据进行缩放
            pred = batch_to_device(pred, device)
            for k, v in pred.items():
                for pattern in self.conf.scale:
                    if k.startswith(pattern):
                        #根据视图索引字符串选择相应的缩放系数。如果视图索引字符串为空，则使用 data["scales"]
                        #否则使用 data[f"view{view_idx}"]["scales"]
                        view_idx = k.replace(pattern, "")
                        scales = (
                            data["scales"]
                            if len(view_idx) == 0
                            else data[f"view{view_idx}"]["scales"]
                        )
                        pred[k] = pred[k] * scales[i]
            # use this function to fix number of keypoints etc.
            #如果指定了padding_fn，则使用该函数对加载的数据进行填充，确保数据符合指定的填充长度
            if self.padding_fn is not None:
                pred = self.padding_fn(pred, self.conf.padding_length)
            preds.append(pred)
            hfile.close()
        #根据配置中的 collate 参数，决定是否对 preds 列表中的结果进行合并。如果需要合并，则使用 collate 函数。
        #批处理操作
        if self.conf.collate:
            return batch_to_device(collate(preds), device)
        else:
            assert len(preds) == 1
            return batch_to_device(preds[0], device)

    def loss(self, pred, data):
        raise NotImplementedError
