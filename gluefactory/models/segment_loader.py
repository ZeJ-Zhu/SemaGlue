import string
import h5py
import torch
from pathlib import Path
from ..datasets.base_dataset import collate
from ..settings import DATA_PATH, SEGMENT_PATH, SEGMENT_F2_PATH, SEGMENT_F2_PATH_C,SEGMENT_F2_MEGA_PATH_TRAIN,SEGMENT_F2_MEGA_PATH_VAL, DATA_DINO_PATH
from ..utils.tensor import batch_to_device
from .base_model import BaseModel
from .utils.misc import pad_to_length


def pad_line_features(pred, seq_l: int = None):
    raise NotImplementedError

def recursive_load(grp, pkeys):
    return {
        #将 h5py.Dataset 中的数据转换为 NumPy 数组，然后 torch.from_numpy 用于将 NumPy 数组转换为 PyTorch 张量
        k: torch.from_numpy(grp[k].__array__())
        #检查 grp[k] 是否是 h5py.Dataset 类型的对象。
        #如果是，就将该对象的数据转换为 PyTorch 的张量 (torch.Tensor)，然后存储在字典中的键 k 下；如果不是，则递归调用 recursive_load 函数处理该对象
        if isinstance(grp[k], h5py.Dataset)
        else recursive_load(grp[k], list(grp.keys()))
        for k in pkeys
    }

class Mega_DinoLoader(BaseModel):
    default_conf = {
        "path": "exports/megadepth-undist-depth-r1024_dino/{scene}.h5",
        "device": None,
        "add_data_path": True,
        "collate": True,
        "data_keys": ["features"],
        "numeric_type": "float32"
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        for i,name in enumerate(data["name"]):
            fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
            if self.conf.add_data_path:
                fpath = Path(DATA_DINO_PATH) / Path(fpath)
            hfile = h5py.File(str(fpath), "r")
            grp = hfile[name]
            pkeys = (
                self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
            )
            pred = recursive_load(grp, pkeys)
            if self.numeric_dtype is not None:
                pred = {
                    k: (
                        v
                        if not isinstance(v, torch.Tensor)
                        or not torch.is_floating_point(v)
                        else v.to(dtype=self.numeric_dtype)
                    )
                    for k, v in pred.items()
                }
            pred = batch_to_device(pred, device)
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
    
class Mega_Train_SegmentLoader(BaseModel):
    default_conf = {
        "path": "exports/megadepth-undist-depth-r1024_Train_F2_Seg/{scene}.h5",
        "device": None,
        "add_data_path": True,
        "collate": True,
        "data_keys": ["feature_map"],
        #"padding_fn": None,
        #"padding_length": None,
        "numeric_type": "float32"
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        # self.padding_fn = conf.padding_fn
        # if self.padding_fn is not None:
        #     self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        for i,name in enumerate(data["name"]):
            fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
            if self.conf.add_data_path:
                fpath = Path(SEGMENT_F2_MEGA_PATH_TRAIN) / Path(fpath)
            hfile = h5py.File(str(fpath), "r")
            grp = hfile[name]
            pkeys = (
                self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
            )
            pred = recursive_load(grp, pkeys)
            if self.numeric_dtype is not None:
                pred = {
                    k: (
                        v
                        if not isinstance(v, torch.Tensor)
                        or not torch.is_floating_point(v)
                        else v.to(dtype=self.numeric_dtype)
                    )
                    for k, v in pred.items()
                }
            pred = batch_to_device(pred, device)
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

class Mega_Val_SegmentLoader(BaseModel):
    default_conf = {
        "path": "exports/megadepth-undist-depth-r1024_Val_F2_Seg/{scene}.h5",
        "device": None,
        "add_data_path": True,
        "collate": True,
        "data_keys": ["feature_map"],
        #"padding_fn": None,
        #"padding_length": None,
        "numeric_type": "float32"
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        # self.padding_fn = conf.padding_fn
        # if self.padding_fn is not None:
        #     self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        for i,name in enumerate(data["name"]):
            fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
            if self.conf.add_data_path:
                fpath = Path(SEGMENT_F2_MEGA_PATH_VAL) / Path(fpath)
            hfile = h5py.File(str(fpath), "r")
            grp = hfile[name]
            pkeys = (
                self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
            )
            pred = recursive_load(grp, pkeys)
            if self.numeric_dtype is not None:
                pred = {
                    k: (
                        v
                        if not isinstance(v, torch.Tensor)
                        or not torch.is_floating_point(v)
                        else v.to(dtype=self.numeric_dtype)
                    )
                    for k, v in pred.items()
                }
            pred = batch_to_device(pred, device)
            # for k, v in pred.items():
            #     for pattern in self.conf.scale:
            #         if k.startswith(pattern):
            #             view_idx = k.replace(pattern, "")
            #             scales = (
            #                 data["scales"]
            #                 if len(view_idx) == 0
            #                 else data[f"view{view_idx}"]["scales"]
            #             )
            #             pred[k] = pred[k] * scales[i]
            # # use this function to fix number of keypoints etc.
            # #如果指定了padding_fn，则使用该函数对加载的数据进行填充，确保数据符合指定的填充长度
            # if self.padding_fn is not None:
            #     pred = self.padding_fn(pred, self.conf.padding_length)
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
    
class Homo_Train_SegmentLoader(BaseModel):
    default_conf = {
        "path": "???",
        "device": None,
        "add_data_path": True,
        "collate": True,
        "data_keys": ["feature_map0", "feature_map1"],
        #"padding_fn": None,
        #"padding_length": None,
        "numeric_type": "float32"
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        # self.padding_fn = conf.padding_fn
        # if self.padding_fn is not None:
        #     self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:#cpu
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        #var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        #for i,name in enumerate(data["name"]):
        name = data['name']
        name_str = '/'.join(name)
            #fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
        fpath = data['idx'][0]
        if self.conf.add_data_path:
            fpath = Path(SEGMENT_F2_PATH_C) / Path(self.conf.path) / f"{fpath}.h5"
        hfile = h5py.File(str(fpath), "r")
        grp = hfile[name_str]
        pkeys = (
            self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
        )
        pred = recursive_load(grp, pkeys)
        if self.numeric_dtype is not None:
            pred = {
                k: (
                    v
                    if not isinstance(v, torch.Tensor)
                    or not torch.is_floating_point(v)
                    else v.to(dtype=self.numeric_dtype)
                )
                for k, v in pred.items()
            }
        pred = batch_to_device(pred, device)
            # for k, v in pred.items():
            #     for pattern in self.conf.scale:
            #         if k.startswith(pattern):
            #             view_idx = k.replace(pattern, "")
            #             scales = (
            #                 data["scales"]
            #                 if len(view_idx) == 0
            #                 else data[f"view{view_idx}"]["scales"]
            #             )
            #             pred[k] = pred[k] * scales[i]
            # # use this function to fix number of keypoints etc.
            # #如果指定了padding_fn，则使用该函数对加载的数据进行填充，确保数据符合指定的填充长度
            # if self.padding_fn is not None:
            #     pred = self.padding_fn(pred, self.conf.padding_length)
        preds.append(pred)#[{'feature_map0': tensor([[[6.8569e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [5.9068e-05, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [3.0088e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  ...,
        #  [1.7726e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [2.7466e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[1.2131e-03, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   1.6272e-05, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 1.5697e-03,
        #   1.5364e-03, 7.1144e-04],
        #  [0.0000e+00, 0.0000e+00, 8.0526e-05,  ..., 2.1477e-03,
        #   1.8768e-03, 9.8228e-04],
        #  ...,
        #  [0.0000e+00, 0.0000e+00, 7.2527e-04,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 6.3467e-04,  ..., 3.6240e-04,
        #   0.0000e+00, 6.3896e-05],
        #  [0.0000e+00, 0.0000e+00, 5.4312e-04,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[5.1842e-03, 5.4398e-03, 5.2948e-03,  ..., 4.9210e-03,
        #   4.9057e-03, 4.4250e-03],
        #  [3.9749e-03, 4.5700e-03, 4.2572e-03,  ..., 4.0398e-03,
        #   4.1809e-03, 5.8479e-03],
        #  [4.0855e-03, 5.1498e-03, 5.5161e-03,  ..., 5.0621e-03,
        #   5.0735e-03, 7.6981e-03],
        #  ...,
        #  [4.3297e-03, 5.6267e-03, 6.7825e-03,  ..., 7.9117e-03,
        #   8.2855e-03, 1.0452e-02],
        #  [4.3411e-03, 5.7335e-03, 6.7940e-03,  ..., 8.5220e-03,
        #   8.1406e-03, 1.0208e-02],
        #  [4.7874e-03, 5.7564e-03, 6.7291e-03,  ..., 1.0567e-02,
        #   1.0056e-02, 1.1177e-02]],

        # ...,

        # [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  ...,
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  ...,
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[4.8981e-03, 3.6526e-03, 4.0359e-03,  ..., 4.7874e-03,
        #   4.8790e-03, 5.9052e-03],
        #  [6.7406e-03, 5.5237e-03, 5.7602e-03,  ..., 6.8703e-03,
        #   7.1182e-03, 7.9117e-03],
        #  [7.4234e-03, 6.1569e-03, 7.0953e-03,  ..., 7.8964e-03,
        #   7.9346e-03, 8.4152e-03],
        #  ...,
        #  [8.0414e-03, 6.7444e-03, 6.8779e-03,  ..., 1.0956e-02,
        #   1.0750e-02, 1.1360e-02],
        #  [8.1329e-03, 6.7673e-03, 6.9656e-03,  ..., 1.0506e-02,
        #   9.7351e-03, 1.2444e-02],
        #  [8.4381e-03, 7.9041e-03, 8.4229e-03,  ..., 1.1055e-02,
        #   1.1436e-02, 1.2352e-02]]]), 'feature_map1': tensor([[[6.9141e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [6.2585e-05, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [3.0684e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  ...,
        #  [1.8179e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [2.7704e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[1.2197e-03, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   2.7597e-05, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 1.5802e-03,
        #   1.5459e-03, 7.2050e-04],
        #  [0.0000e+00, 0.0000e+00, 9.1910e-05,  ..., 2.1572e-03,
        #   1.8835e-03, 9.9277e-04],
        #  ...,
        #  [0.0000e+00, 0.0000e+00, 7.3242e-04,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 6.3753e-04,  ..., 3.6430e-04,
        #   0.0000e+00, 6.3956e-05],
        #  [0.0000e+00, 0.0000e+00, 5.3596e-04,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[5.1804e-03, 5.4283e-03, 5.2910e-03,  ..., 4.9133e-03,
        #   4.8981e-03, 4.4136e-03],
        #  [3.9749e-03, 4.5738e-03, 4.2534e-03,  ..., 4.0321e-03,
        #   4.1771e-03, 5.8365e-03],
        #  [4.0817e-03, 5.1460e-03, 5.5122e-03,  ..., 5.0621e-03,
        #   5.0735e-03, 7.6904e-03],
        #  ...,
        #  [4.3259e-03, 5.6305e-03, 6.7787e-03,  ..., 7.9117e-03,
        #   8.2855e-03, 1.0452e-02],
        #  [4.3373e-03, 5.7335e-03, 6.7940e-03,  ..., 8.5220e-03,
        #   8.1406e-03, 1.0208e-02],
        #  [4.8027e-03, 5.7678e-03, 6.7368e-03,  ..., 1.0567e-02,
        #   1.0048e-02, 1.1177e-02]],

        # ...,

        # [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  ...,
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  ...,
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00],
        #  [0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
        #   0.0000e+00, 0.0000e+00]],

        # [[4.8866e-03, 3.6449e-03, 4.0321e-03,  ..., 4.7836e-03,
        #   4.8790e-03, 5.9013e-03],
        #  [6.7368e-03, 5.5237e-03, 5.7640e-03,  ..., 6.8741e-03,
        #   7.1220e-03, 7.9193e-03],
        #  [7.4158e-03, 6.1607e-03, 7.0992e-03,  ..., 7.8964e-03,
        #   7.9346e-03, 8.4152e-03],
        #  ...,
        #  [8.0338e-03, 6.7368e-03, 6.8779e-03,  ..., 1.0956e-02,
        #   1.0742e-02, 1.1360e-02],
        #  [8.1253e-03, 6.7635e-03, 6.9695e-03,  ..., 1.0513e-02,
        #   9.7351e-03, 1.2444e-02],
        #  [8.4305e-03, 7.9041e-03, 8.4305e-03,  ..., 1.1063e-02,
        #   1.1444e-02, 1.2352e-02]]])}]
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

class Homo_Val_SegmentLoader(BaseModel):
    default_conf = {
        "path": "???",
        "device": None,
        "add_data_path": True,
        "collate": True,
        "data_keys": ["feature_map0", "feature_map1"],
        #"padding_fn": None,
        #"padding_length": None,
        "numeric_type": "float32"
    }

    required_data_keys = ["name"]  # we need an identifier

    def _init(self, conf):
        self.hfiles = {}
        # self.padding_fn = conf.padding_fn
        # if self.padding_fn is not None:
        #     self.padding_fn = eval(self.padding_fn)
        self.numeric_dtype = {
            None: None,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        }[conf.numeric_type]

    def _forward(self, data):
        preds = []
        device = self.conf.device
        if not device:
            devices = set(
                [v.device for v in data.values() if isinstance(v, torch.Tensor)]
            )
            if len(devices) == 0:#cpu
                device = "cpu"
            else:
                assert len(devices) == 1
                device = devices.pop()

        #var_names = [x[1] for x in string.Formatter().parse(self.conf.path) if x[1]]
        #for i,name in enumerate(data["name"]):
        #for i, name in enumerate(data["name"]):
        name = data['name']
        name_str = '/'.join(name)
            #fpath = self.conf.path.format(**{k: data[k][i] for k in var_names})
        fpath = data['idx'][0]
        if self.conf.add_data_path:
            # 去掉self.conf.path最后的一个/
            pa = self.conf.path.rstrip('/')
            # 拼接成新的路径
            fp = f"{SEGMENT_F2_PATH_C}{pa}_val/"
            fpath = f"{fp}{fpath}.h5"
            #fpath = Path(SEGMENT_PATH) / Path(self.conf.path)+'val' /f"{fpath}.h5"
        hfile = h5py.File(str(fpath), "r")
        grp = hfile[name_str]
        pkeys = (
            self.conf.data_keys if self.conf.data_keys is not None else grp.keys()
        )
        pred = recursive_load(grp, pkeys)
        if self.numeric_dtype is not None:
            pred = {
                k: (
                    v
                    if not isinstance(v, torch.Tensor)
                    or not torch.is_floating_point(v)
                    else v.to(dtype=self.numeric_dtype)
                )
                for k, v in pred.items()
            }
        pred = batch_to_device(pred, device)
            # for k, v in pred.items():
            #     for pattern in self.conf.scale:
            #         if k.startswith(pattern):
            #             view_idx = k.replace(pattern, "")
            #             scales = (
            #                 data["scales"]
            #                 if len(view_idx) == 0
            #                 else data[f"view{view_idx}"]["scales"]
            #             )
            #             pred[k] = pred[k] * scales[i]
            # # use this function to fix number of keypoints etc.
            # #如果指定了padding_fn，则使用该函数对加载的数据进行填充，确保数据符合指定的填充长度
            # if self.padding_fn is not None:
            #     pred = self.padding_fn(pred, self.conf.padding_length)
        preds.append(pred)#[{'feature_map0': tensor([[[6.8569e-04, 0.0000e+00, 0.0000e+00,  ..., 0.0000e+00,
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