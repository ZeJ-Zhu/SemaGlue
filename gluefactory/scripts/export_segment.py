import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..datasets import get_dataset
from ..geometry.depth import sample_depth
from ..models import get_model
from ..settings import DATA_PATH, SEGMENT_PATH, SEGMENT_F2_PATH, SEGMENT_F2_PATH_C,SEGMENT_F2_MEGA_PATH_TRAIN,SEGMENT_F2_MEGA_PATH_VAL
from ..utils.export_predictions import export_seg_mega_predictions, export_seg_homo_predictions

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# import debugpy
# debugpy.listen(('127.0.13.25', 8001))
# debugpy.wait_for_client()


#python -m gluefactory.scripts.export_segment --method segment --num_workers 8


resize = 1024
n_kpts = 2048
#包括 SuperPoint、SIFT、KeyNetAffNetHardNet 等多种方法的配置。
#每个配置包含了特征名称、要提取的特征类型、灰度图处理等信息
configs = {
    "segment":{
        "name": f"r{resize}_F2_Seg",
        #"keys": ["feature_map", "segment_map"],
        "keys": ["feature_map"],
        "conf": {
            "name": "gluefactory.models.segments.segment",
            "num_classes": 150,
            "embed_dims": [32,64,160,256],
            "norm_typ": 'SyncBN',
            #Hamburger Parameters
            "ham_channels": 256, #ham_channels输入hamburger的通道
            "put_cheese": True,

            "DUAL": False,
            "SPATIAL": True,
            "RAND_INIT": True,

            "TRAIN_STEPS": 6,
            "EVAL_STEPS": 6,

            "MD_S": 1, 
            "MD_D": 512,
            "MD_R": 64,
            "INV_T": 1,
            "BETA": 0.1,
            "Eta": 0.9
        }
    },
    "sp": {
        "name": f"r{resize}_SP-k{n_kpts}-nms3",
        "keys": ["keypoints", "descriptors", "keypoint_scores"],
        "gray": True,
        "conf": {
            "name": "gluefactory_nonfree.superpoint",
            "nms_radius": 3,
            "max_num_keypoints": n_kpts,
            "detection_threshold": 0.000,
        },
    },
}


#用于从预测的关键点中采样深度信息
def get_kp_depth(pred, data):
    d, valid = sample_depth(pred["keypoints"], data["depth"])
    return {"depth_keypoints": d, "valid_depth_keypoints": valid}


#用于运行特征导出操作。该函数接收特征文件路径、场景名称和命令行参数作为输入，根据配置获取数据集、模型等信息
#然后执行特征导出操作，并将结果保存到 HDF5 文件中。
def run_export_megadepth(feature_file, scene, args):
    conf = {
        "data": {
            "name": "megadepth",
            "views": 1,
            #"grayscale": configs[args.method]["gray"],
            "preprocessing": {
                "resize": resize,
                "side": "long",
            },
            "batch_size": 1,
            "num_workers": args.num_workers,
            #"read_depth": True,
            "train_split": [scene],
            "train_num_per_scene": None,
        },
        "split": "val",
        "model": configs[args.method]["conf"],
    }
    #将配置字典转换为OmegaConf对象
    conf = OmegaConf.create(conf)
    #获取特征的键
    keys = configs[args.method]["keys"]
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(conf.split or "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #根据配置中的模型信息，使用 get_model 获取模型对象，并将其设置为推断模式，并移动到可用的设备上（GPU 或 CPU）
    model = get_model(conf.model.name)(conf.model).eval().to(device)

    # #该函数用于存储每个关键点的深度信息，并更新 keys 列表以包括深度相关的键
    # if args.export_sparse_depth:
    #     callback_fn = get_kp_depth  # use this to store the depth of each keypoint
    #     keys = keys + ["depth_keypoints", "valid_depth_keypoints"]
    # else:
    #     callback_fn = None
    export_seg_mega_predictions(
        loader, model, feature_file, as_half=True, keys=keys, callback_fn=None
    )


def run_export_seg_withdataset_homography(feature_file, args):
    conf = {
        # "data": {
        #     "name": "homography",
        #     "views": 1,
        #     #"grayscale": configs[args.method]["gray"],
        #     "preprocessing": {
        #         "resize": resize,
        #         "side": "long",
        #     },
        #     "batch_size": 1,
        #     "num_workers": args.num_workers,
        #     #"read_depth": True,
        #     "train_split": [scene],
        #     "train_num_per_scene": None,
        # },
        "data": {
            "name": "homographies",
            "data_dir": "revisitop1m",
            "batch_size": 1,
            "train_size": 150000,
            "val_size": 2000,
            "num_workers": 14,  #14
            "homography": {
                "difficulty": 0.7,
                "max_angle": 45,
            },
            "photometric":{
                    "name": "lg",
            },
        },
        "split": "val",#val
        "model": configs[args.method]["conf"],
    }
    #将配置字典转换为OmegaConf对象
    conf = OmegaConf.create(conf)
    #获取特征的键
    keys = configs[args.method]["keys"]
    dataset = get_dataset(conf.data.name)(conf.data)
    loader = dataset.get_data_loader(conf.split)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #根据配置中的模型信息，使用 get_model 获取模型对象，并将其设置为推断模式，并移动到可用的设备上（GPU 或 CPU）
    model = get_model(conf.model.name)(conf.model).eval().to(device)

    # #该函数用于存储每个关键点的深度信息，并更新 keys 列表以包括深度相关的键
    # if args.export_sparse_depth:
    #     callback_fn = get_kp_depth  # use this to store the depth of each keypoint
    #     keys = keys + ["depth_keypoints", "valid_depth_keypoints"]
    # else:
    #     callback_fn = None
    export_seg_homo_predictions(
        loader, model, feature_file, as_half=True, keys=keys, callback_fn=None
    )

def run_export_seg_homography(dataloader):
    conf = {
        # "data": {
        #     "name": "homography",
        #     "views": 1,
        #     #"grayscale": configs[args.method]["gray"],
        #     "preprocessing": {
        #         "resize": resize,
        #         "side": "long",
        #     },
        #     "batch_size": 1,
        #     "num_workers": args.num_workers,
        #     #"read_depth": True,
        #     "train_split": [scene],
        #     "train_num_per_scene": None,
        # },
        #"split": "train",
        "split": "train",
        "model": configs["segment"]["conf"],
    }
    export_root = Path(SEGMENT_PATH, "exports", "homography_seg_val")
    export_root.mkdir(parents=True, exist_ok=True)
    #print("idex:", idex)
    #idx = data_['idx'].item()
    feature_file = export_root
    #feature_file = export_root / f"{idx}.h5"
    #feature_file = export_root / f"{idex}.h5"
    # #检查导出文件是否已经存在，如果存在则跳过当前场景的导出。这里的 and False 表示始终跳过导出。
    # if feature_file.exists() and False:
    #     continue
    #调用 run_export 函数，导出模型在当前场景上的本地特征
    #logging.info(f"Export local features for idex {idex}")
   
    #将配置字典转换为OmegaConf对象
    conf = OmegaConf.create(conf)
    #获取特征的键
    keys = configs["segment"]["keys"]
    # dataset = get_dataset(conf.data.name)(conf.data)
    # loader = dataset.get_data_loader(conf.split or "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #根据配置中的模型信息，使用 get_model 获取模型对象，并将其设置为推断模式，并移动到可用的设备上（GPU 或 CPU）
    model = get_model(conf.model.name)(conf.model).eval().to(device)

    # #该函数用于存储每个关键点的深度信息，并更新 keys 列表以包括深度相关的键
    # if args.export_sparse_depth:
    #     callback_fn = get_kp_depth  # use this to store the depth of each keypoint
    #     keys = keys + ["depth_keypoints", "valid_depth_keypoints"]
    # else:
    #     callback_fn = None
    export_seg_homo_predictions(
        dataloader, model, feature_file, as_half=True, keys=keys, callback_fn=None
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_prefix", type=str, default="")
    parser.add_argument("--method", type=str, default="segment")
    parser.add_argument("--scenes", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--export_sparse_depth", action="store_true")
    args = parser.parse_args()
    #第一阶段
    export_root = Path(SEGMENT_F2_PATH, "exports", "homography_seg_val_EncoderCovRelu")
    export_root.mkdir(parents=True, exist_ok=True)
    run_export_seg_withdataset_homography(export_root, args)
    
    #第二阶段
    # export_name = configs[args.method]["name"]

    # #data_root包含了原始数据集的图像，深度图等信息
    # data_root = Path(DATA_PATH, "megadepth/Undistorted_SfM")
    # #export_root存储导出的特征或深度信息
    # export_root = Path(SEGMENT_F2_MEGA_PATH_TRAIN, "exports", "megadepth-undist-depth-" + export_name)
    # export_root.mkdir(parents=True, exist_ok=True)

    # #如果没有提供特定场景的列表 (--scenes)，则读取数据集根目录下所有的场景目录名。
    # #否则，从指定文件中读取场景列表
    # if args.scenes is None:
    #     scenes = [p.name for p in data_root.iterdir() if p.is_dir()]
    # else:
    #     with open(DATA_PATH / "megadepth" / args.scenes, "r") as f:
    #         scenes = f.read().split()
    # for i, scene in enumerate(scenes):
    #     print(f"{i} / {len(scenes)}", scene)
    #     feature_file = export_root / (scene + ".h5")
    #     #检查导出文件是否已经存在，如果存在则跳过当前场景的导出。这里的 and False 表示始终跳过导出。
    #     if feature_file.exists() and False:
    #         continue
    #     #如果场景的图像文件夹不存在，将打印跳过信息并继续下一个场景
    #     if not (data_root / scene / "images").exists():
    #         logging.info("Skip " + scene)
    #         continue
    #     #调用 run_export 函数，导出模型在当前场景上的本地特征
    #     logging.info(f"Export local features for scene {scene}")
    #     run_export_megadepth(feature_file, scene, args)
