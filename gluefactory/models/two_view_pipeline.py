"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:很多个子模型
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from omegaconf import OmegaConf
import torch
import os
from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "segment": {"name": None},
        "matcher": {"name": None},
        "filter": {"name": None},
        "solver": {"name": None},
        "extractor_dino": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "segment",
        "matcher",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.segment.name:
            self.segment = get_model(conf.segment.name)(to_ctr(conf.segment))

        if conf.matcher.name:
            self.matcher = get_model(conf.matcher.name)(to_ctr(conf.matcher))

        if conf.extractor_dino.name:
            self.extractor_dino = get_model(conf.extractor_dino.name)(to_ctr(conf.extractor_dino))
            
        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def extract_view(self, data, i):#提取视图特征
        data_i = data[f"view{i}"]
        #检索了当前视图的缓存预测，如果没有缓存的预测，则初始化为空字典
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract#self.conf.allow_no_extract：False
        if self.conf.extractor.name and not skip_extract:#不应跳过提取
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def extract_segment(self, data ,i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("segment", {})
        #pred_i = data['segment']['feature_map'+str(i)]
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.segment.name and not skip_extract:
            pred_i = {**pred_i, **self.segment(data_i)}
        elif self.conf.segment.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.segment({**data_i, **pred_i})}
        return pred_i
    
    def extract_di(self, data, i):
        data_i = data[f'view{i}']
        pred_i = data_i.get("dino", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract#self.conf.allow_no_extract：False
        if self.conf.extractor_dino.name and not skip_extract:#不应跳过提取
            pred_i = {**pred_i, **self.extractor_dino(data_i)}
        elif self.conf.extractor_dino.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor_dino({**data_i, **pred_i})}
        return pred_i
        
    def _forward(self, data):#按照模型的处理流程，依次调用提取器、匹配器、过滤器和解算器，并将它们的输出合并到一个字典中
        #print(data)
        #name:包含图像名称的列表
        #original_image_size:包含原始图像大小的张量
        #H_0to1:从视图0到视图1的变换矩阵
        #idx：包含索引信息的张量
        #view0：包含视图0数据的字典：其中image，H
        #view1，包含视图1数据的字典，其中image,H
        #coords：包含坐标信息的张量
        #image_size：包含图像大小的张量
        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred = {#添加后缀”0“和”1“，并合并到一个字典pred中
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
        }

        if self.conf.extractor_dino.name:
            segment0 = self.extract_di(data, "0")
            segment1 = self.extract_di(data, "1")
            pred["feature_map0"] = segment0["features"]
            pred["feature_map1"] = segment1["features"]
            
        if self.conf.segment.name:  
            # with torch.no_grad():
            segment0 = self.extract_segment(data, "0")
            segment1 = self.extract_segment(data, "1")
            pred["feature_map0"] = segment0["feature_map"]
            pred["feature_map1"] = segment1["feature_map"]

        if self.conf.matcher.name:#如果定义了matcher，则调用匹配器的前向传播方法，并将其输出合并到pred中
            pred = {**pred, **self.matcher({**data, **pred})}#将一个字典的内容解包到另一个字典中

        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        if self.conf.ground_truth.name and self.conf.run_gt_in_forward:#如果定义了地面真实值模型，并且配置要求在前向传播中运行地面真实值模型
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})
        return pred

    def loss(self, pred, data):
        losses = {}#初始化
        metrics = {}
        total = 0

        # get labels
        if self.conf.ground_truth.name and not self.conf.run_gt_in_forward:
            gt_pred = self.ground_truth({**data, **pred})
            pred.update({f"gt_{k}": v for k, v in gt_pred.items()})

        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():#检查是否配置了apply_loss
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, {**pred, **data})
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}
                total = losses_["total"] + total
        return {**losses, "total": total}, metrics
