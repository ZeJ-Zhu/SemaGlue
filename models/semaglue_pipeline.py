from omegaconf import OmegaConf

from .semaglue.base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict

import torch
import importlib.util

from .semaglue.base_model import BaseModel
from .segments.segnext import SegNext
from .semaglue.semaglue import SemaGlue


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    import inspect

    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def get_model(name):
    import_paths = [
        "." + name,
    ]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseModel)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_model__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'Model {name} not found in any of [{" ".join(import_paths)}]')


class SemaGluePipeline(BaseModel):
    default_conf = {
        "segment":{
            "name": "segments.segment",
            "embed_dims": [32,64,160,256],
            "norm_typ": 'SyncBN', 
        },
        "matcher": {
            "name": "semaglue.semaglue",
            "features": "superpoint",
            "segment_dim": 480, #segnext dimension
            "flash": False,
            "checkpointed": True,
            "n_layers": 9,
            "scale": 2,
        },
    }
    required_data_keys = ['image0', 'image1']
    strict_conf = False  # need to pass new confs to children models
    components = [
        "segment",
        "matcher",
    ]

    def _init(self, conf):
        if conf.segment.name:
            self.segment = SegNext(to_ctr(conf.segment))
        if conf.matcher.name:
            self.matcher = SemaGlue(to_ctr(conf.matcher))

    def extract_segment(self, data ,i):
        data_i = data[f"image{i}"]
        pred_i = {**self.segment(data_i)}
        return pred_i
    
    def _forward(self, data):
        pred = {}
        if self.conf.segment.name:  
            with torch.no_grad():
                segment0 = self.extract_segment(data, "0")
                segment1 = self.extract_segment(data, "1")
                pred["feature_map0"] = segment0["feature_map"]
                pred["feature_map1"] = segment1["feature_map"]
        if self.conf.matcher.name:
            pred = {**pred, **self.matcher({**data, **pred})}
        return pred