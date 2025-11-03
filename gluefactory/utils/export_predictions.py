"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .tensor import batch_to_device


@torch.no_grad()
#model
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,#半精度浮点数
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)#先将输出文件的目录创建好，然后创建一个HDF5文件对象
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    #遍历数据加载器产生的每个批次数据
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        # renormalization
        #对于一些特殊的键，归一化操作，根据视图缩放因子进行缩放
        for k in pred.keys():
            if k.startswith("keypoints"):
                idx = k.replace("keypoints", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("lines"):
                idx = k.replace("lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]
            if k.startswith("orig_lines"):
                idx = k.replace("orig_lines", "")
                scales = 1.0 / (
                    data["scales"] if len(idx) == 0 else data[f"view{idx}"]["scales"]
                )
                pred[k] = pred[k] * scales[None]

        #将pytorch张量转换为numpy数组
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
        # pred = {k: v[0].cpu().numpy() if isinstance(v[0], torch.Tensor) else v[0] for k, v in pred.items()}
      
        #根据需要将数据类型转换为半精度浮点数
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        #创建 HDF5 文件中的组（group）和数据集（dataset），将处理好的预测结果保存到 HDF5 文件中
        try:
            name = data["name"][0]
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except RuntimeError:
            continue

        del pred
    hfile.close()
    return output_file

def export_seg_mega_predictions(
    loader,
    model,
    output_file,
    as_half=False,#半精度浮点数
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)#先将输出文件的目录创建好，然后创建一个HDF5文件对象
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    #遍历数据加载器产生的每个批次数据
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        pred = model(data)
        #{'name': ['2774598813_5e2757b1ea_o.jpg'], 'scene': ['0004'], 'T_w2cam': Pose: torch.Size([1]) torch.float32 cuda:0, 'depth': tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      ...,
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.]]], device='cuda:0'), 'camera': Camera torch.Size([1]) torch.float32 cuda:0, 'scales': tensor([[0.6577, 0.6570]], device='cuda:0'), 'image_size': tensor([[1024,  680]], device='cuda:0'), 'transform': tensor([[[0.6577, 0.0000, 0.0000],
    #      [0.0000, 0.6570, 0.0000],
    #      [0.0000, 0.0000, 1.0000]]], device='cuda:0', dtype=torch.float64), 'original_image_size': tensor([[1557, 1035]], device='cuda:0'), 'image': tensor([[[[0.2549, 0.2602, 0.2566,  ..., 0.3344, 0.3337, 0.3349],
    #       [0.2480, 0.2559, 0.2565,  ..., 0.3257, 0.3329, 0.3276],
    #       [0.2555, 0.2504, 0.2528,  ..., 0.3317, 0.3309, 0.3329],
    #       ...,
    #       [0.1559, 0.1633, 0.1642,  ..., 0.0441, 0.0131, 0.0091],
    #       [0.1523, 0.1567, 0.1662,  ..., 0.0666, 0.0066, 0.0070],
    #       [0.1567, 0.1541, 0.1648,  ..., 0.0834, 0.0199, 0.0039]],

    #      [[0.4235, 0.4227, 0.4194,  ..., 0.5030, 0.5023, 0.4978],
    #       [0.4197, 0.4215, 0.4207,  ..., 0.5004, 0.5064, 0.4904],
    #       [0.4257, 0.4182, 0.4199,  ..., 0.5058, 0.5038, 0.4958],
    #       ...,
    #       [0.1873, 0.1916, 0.1940,  ..., 0.0758, 0.0372, 0.0279],
    #       [0.1837, 0.1850, 0.1939,  ..., 0.1084, 0.0324, 0.0237],
    #       [0.1910, 0.1853, 0.1981,  ..., 0.1302, 0.0492, 0.0243]],

    #      [[0.6000, 0.6023, 0.5995,  ..., 0.6795, 0.6788, 0.6771],
    #       [0.5961, 0.6058, 0.6095,  ..., 0.6677, 0.6743, 0.6636],
    #       [0.6071, 0.6110, 0.6147,  ..., 0.6717, 0.6727, 0.6679],
    #       ...,
    #       [0.2618, 0.2640, 0.2635,  ..., 0.1228, 0.0555, 0.0303],
    #       [0.2582, 0.2595, 0.2684,  ..., 0.1540, 0.0620, 0.0324],
    #       [0.2597, 0.2586, 0.2758,  ..., 0.1860, 0.0885, 0.0406]]]],
    #    device='cuda:0'), 'idx': tensor([996], device='cuda:0')}
        if callback_fn is not None:
            pred = {**callback_fn(pred, data), **pred}
        #指定了要选择的键 (keys)，则仅保留预测结果中包含在这些键中的项
        #如果没有指定特定的键（即 keys="*"），则保留所有项
        if keys != "*":
            if len(set(keys) - set(pred.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
            pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
        assert len(pred) > 0

        #将pytorch张量转换为numpy数组
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        #根据需要将数据类型转换为半精度浮点数
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        #创建 HDF5 文件中的组（group）和数据集（dataset），将处理好的预测结果保存到 HDF5 文件中
        try:
            name = data["name"][0]#'525156938_8c76a2f265_o.jpg'
            grp = hfile.create_group(name)
            for k, v in pred.items():
                grp.create_dataset(k, data=v)
        except RuntimeError:
            continue

        del pred
    hfile.close()
    return output_file

def export_seg_homo_predictions(
    loader,
    model,
    out_file,
    as_half=False,#半精度浮点数
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    #遍历数据加载器产生的每个批次数据
    for data_ in tqdm(loader):
        data = batch_to_device(data_, device, non_blocking=True)
        idx = data_['idx'].item()
        output_file = out_file / f"{idx}.h5"
        Path(output_file).parent.mkdir(exist_ok=True, parents=True)#先将输出文件的目录创建好，然后创建一个HDF5文件对象
        hfile = h5py.File(str(output_file), "w")
        data0 = data['view0']
        data1 = data['view1']
        #{'name': ['7ee/7eef1faf134f0b62d5eaf6999f37.jpg'], 'original_image_size': tensor([[ 768, 1024]]), 'H_0to1': tensor([[[ 1.0000e+00, -7.1964e-18,  1.1163e-15],
        #  [-0.0000e+00,  1.0000e+00,  3.4614e-14],
        #  [ 3.3735e-20,  3.9019e-20,  1.0000e+00]]]), 'idx': tensor([0]), 'view0': {'image': tensor([[[[0.1569, 0.1647, 0.1647,  ..., 0.1020, 0.1020, 0.1059],
        #   [0.1569, 0.1608, 0.1608,  ..., 0.1137, 0.1137, 0.1176],
        #   [0.1569, 0.1608, 0.1608,  ..., 0.1216, 0.1216, 0.1216],
        #   ...,
        #   [0.1882, 0.1843, 0.1765,  ..., 0.6824, 0.6863, 0.6824],
        #   [0.1765, 0.1725, 0.1804,  ..., 0.6863, 0.6863, 0.6863],
        #   [0.1882, 0.1804, 0.1686,  ..., 0.6941, 0.7020, 0.7020]],

        #  [[0.1294, 0.1412, 0.1412,  ..., 0.0627, 0.0627, 0.0627],
        #   [0.1294, 0.1451, 0.1451,  ..., 0.0706, 0.0706, 0.0706],
        #   [0.1294, 0.1451, 0.1451,  ..., 0.0745, 0.0745, 0.0745],
        #   ...,
        #   [0.1804, 0.1804, 0.1725,  ..., 0.6667, 0.6706, 0.6667],
        #   [0.1686, 0.1725, 0.1647,  ..., 0.6706, 0.6706, 0.6706],
        #   [0.1725, 0.1725, 0.1765,  ..., 0.6784, 0.6863, 0.6863]],

        #  [[0.0941, 0.1098, 0.1098,  ..., 0.0588, 0.0588, 0.0627],
        #   [0.0941, 0.1098, 0.1098,  ..., 0.0667, 0.0667, 0.0706],
        #   [0.0980, 0.1098, 0.1098,  ..., 0.0745, 0.0745, 0.0745],
        #   ...,
        #   [0.1608, 0.1608, 0.1569,  ..., 0.5647, 0.5647, 0.5569],
        #   [0.1490, 0.1490, 0.1529,  ..., 0.5647, 0.5647, 0.5608],
        #   [0.1608, 0.1569, 0.1490,  ..., 0.5804, 0.5843, 0.5882]]]]), 'H_': tensor([[[ 2.0768e+00, -2.0390e-01, -4.2298e+01],
        #  [-7.0438e-01,  2.4968e+00, -7.1647e+02],
        #  [-2.1520e-03,  2.5600e-03,  1.0000e+00]]]), 'coords': tensor([[[  -42.2979,  -716.4684],
        #  [  -69.3328,   508.1560],
        #  [  682.6311,   659.9773],
        #  [-2378.6924,  1926.3489]]]), 'image_size': tensor([[640., 480.]])}, 'view1': {'image': tensor([[[[0.1569, 0.1647, 0.1647,  ..., 0.1020, 0.1020, 0.1059],
        #   [0.1569, 0.1608, 0.1608,  ..., 0.1137, 0.1137, 0.1176],
        #   [0.1569, 0.1608, 0.1608,  ..., 0.1216, 0.1216, 0.1216],
        #   ...,
        #   [0.1882, 0.1843, 0.1765,  ..., 0.6824, 0.6863, 0.6824],
        #   [0.1765, 0.1725, 0.1804,  ..., 0.6863, 0.6863, 0.6863],
        #   [0.1882, 0.1804, 0.1686,  ..., 0.6941, 0.7020, 0.7020]],

        #  [[0.1294, 0.1412, 0.1412,  ..., 0.0627, 0.0627, 0.0627],
        #   [0.1294, 0.1451, 0.1451,  ..., 0.0706, 0.0706, 0.0706],
        #   [0.1294, 0.1451, 0.1451,  ..., 0.0745, 0.0745, 0.0745],
        #   ...,
        #   [0.1804, 0.1804, 0.1725,  ..., 0.6667, 0.6706, 0.6667],
        #   [0.1686, 0.1725, 0.1647,  ..., 0.6706, 0.6706, 0.6706],
        #   [0.1725, 0.1725, 0.1765,  ..., 0.6784, 0.6863, 0.6863]],

        #  [[0.0941, 0.1098, 0.1098,  ..., 0.0588, 0.0588, 0.0627],
        #   [0.0941, 0.1098, 0.1098,  ..., 0.0667, 0.0667, 0.0706],
        #   [0.0980, 0.1098, 0.1098,  ..., 0.0745, 0.0745, 0.0745],
        #   ...,
        #   [0.1608, 0.1608, 0.1569,  ..., 0.5647, 0.5647, 0.5569],
        #   [0.1490, 0.1490, 0.1529,  ..., 0.5647, 0.5647, 0.5608],
        #   [0.1608, 0.1569, 0.1490,  ..., 0.5804, 0.5843, 0.5882]]]]), 'H_': tensor([[[ 2.0768e+00, -2.0390e-01, -4.2298e+01],
        #  [-7.0438e-01,  2.4968e+00, -7.1647e+02],
        #  [-2.1520e-03,  2.5600e-03,  1.0000e+00]]]), 'coords': tensor([[[  -42.2979,  -716.4684],
        #  [  -69.3328,   508.1560],
        #  [  682.6311,   659.9773],
        #  [-2378.6924,  1926.3489]]]), 'image_size': tensor([[640., 480.]])}}
        
        #{'name': ['2774598813_5e2757b1ea_o.jpg'], 'scene': ['0004'], 'T_w2cam': Pose: torch.Size([1]) torch.float32 cuda:0, 'depth': tensor([[[0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      ...,
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.],
    #      [0., 0., 0.,  ..., 0., 0., 0.]]], device='cuda:0'), 'camera': Camera torch.Size([1]) torch.float32 cuda:0, 'scales': tensor([[0.6577, 0.6570]], device='cuda:0'), 'image_size': tensor([[1024,  680]], device='cuda:0'), 'transform': tensor([[[0.6577, 0.0000, 0.0000],
    #      [0.0000, 0.6570, 0.0000],
    #      [0.0000, 0.0000, 1.0000]]], device='cuda:0', dtype=torch.float64), 'original_image_size': tensor([[1557, 1035]], device='cuda:0'), 'image': tensor([[[[0.2549, 0.2602, 0.2566,  ..., 0.3344, 0.3337, 0.3349],
    #       [0.2480, 0.2559, 0.2565,  ..., 0.3257, 0.3329, 0.3276],
    #       [0.2555, 0.2504, 0.2528,  ..., 0.3317, 0.3309, 0.3329],
    #       ...,
    #       [0.1559, 0.1633, 0.1642,  ..., 0.0441, 0.0131, 0.0091],
    #       [0.1523, 0.1567, 0.1662,  ..., 0.0666, 0.0066, 0.0070],
    #       [0.1567, 0.1541, 0.1648,  ..., 0.0834, 0.0199, 0.0039]],

    #      [[0.4235, 0.4227, 0.4194,  ..., 0.5030, 0.5023, 0.4978],
    #       [0.4197, 0.4215, 0.4207,  ..., 0.5004, 0.5064, 0.4904],
    #       [0.4257, 0.4182, 0.4199,  ..., 0.5058, 0.5038, 0.4958],
    #       ...,
    #       [0.1873, 0.1916, 0.1940,  ..., 0.0758, 0.0372, 0.0279],
    #       [0.1837, 0.1850, 0.1939,  ..., 0.1084, 0.0324, 0.0237],
    #       [0.1910, 0.1853, 0.1981,  ..., 0.1302, 0.0492, 0.0243]],

    #      [[0.6000, 0.6023, 0.5995,  ..., 0.6795, 0.6788, 0.6771],
    #       [0.5961, 0.6058, 0.6095,  ..., 0.6677, 0.6743, 0.6636],
    #       [0.6071, 0.6110, 0.6147,  ..., 0.6717, 0.6727, 0.6679],
    #       ...,
    #       [0.2618, 0.2640, 0.2635,  ..., 0.1228, 0.0555, 0.0303],
    #       [0.2582, 0.2595, 0.2684,  ..., 0.1540, 0.0620, 0.0324],
    #       [0.2597, 0.2586, 0.2758,  ..., 0.1860, 0.0885, 0.0406]]]],
    #    device='cuda:0'), 'idx': tensor([996], device='cuda:0')}
        pred0 = model(data0)
        pred1 = model(data1)
        if callback_fn is not None:
            pred0 = {**callback_fn(pred0, data), **pred0}
            pred1 = {**callback_fn(pred1, data), **pred1}
        #指定了要选择的键 (keys)，则仅保留预测结果中包含在这些键中的项
        #如果没有指定特定的键（即 keys="*"），则保留所有项
        if keys != "*":
            if len(set(keys) - set(pred0.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred0.keys())}")
            pred0 = {k: v for k, v in pred0.items() if k in keys + optional_keys}
        assert len(pred0) > 0
        if keys != "*":
            if len(set(keys) - set(pred1.keys())) > 0:
                raise ValueError(f"Missing key {set(keys) - set(pred1.keys())}")
            pred1 = {k: v for k, v in pred1.items() if k in keys + optional_keys}
        assert len(pred1) > 0
        pred = {#添加后缀”0“和”1“，并合并到一个字典pred中
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }
        #将pytorch张量转换为numpy数组
        pred = {k: v[0].detach().cpu().numpy() for k, v in pred.items()}
        #根据需要将数据类型转换为半精度浮点数
        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)
        #创建 HDF5 文件中的组（group）和数据集（dataset），将处理好的预测结果保存到 HDF5 文件中
        name = data["name"][0]#7ee/7eef1faf134f0b62d5eaf6999f37.jpg
        grp = hfile.create_group(name)
        for k, v in pred.items():
            grp.create_dataset(k, data=v)
        del pred
        hfile.close()
    return output_file