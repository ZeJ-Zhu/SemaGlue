#from lightglue import LightGlue as LightGlue_
from .lightglue_official import LightGlue as LightGlue_
from omegaconf import OmegaConf
from pathlib import Path
import torch
from ..base_model import BaseModel


class LightGlue(BaseModel):
    default_conf = {"features": "superpoint", **LightGlue_.default_conf}
    required_data_keys = [
        "view0",
        "keypoints0",
        "descriptors0",
        "view1",
        "keypoints1",
        "descriptors1",
    ]

    def _init(self, conf):
        dconf = OmegaConf.to_container(conf)
        self.net = LightGlue_(dconf.pop("features"), **dconf)
        
        ckpt = Path("/data/zzj/glue-factory-main1/glue-factory-main/outputs/training/sp+lg_megadepth_noFlash_paper_main/checkpoint_best.tar")
        ckpt = torch.load(str(ckpt), map_location="cpu")
        state_dict = ckpt["model"]
        dict_params = set(state_dict.keys())
        model_params = set(map(lambda n: n[0], self.net.named_parameters()))
        diff = model_params - dict_params
        if len(diff) > 0:
            state_dict = {k.replace('matcher.', ''): v for k, v in state_dict.items()}
        self.net.load_state_dict(state_dict, strict=False)
        
        self.set_initialized()

    def _forward(self, data):
        required_keys = ["keypoints", "descriptors", "scales", "oris"]
        # view0 = {
        #     **data["view0"],
        #     **{k: data[k + "0"] for k in required_keys if (k + "0") in data},
        # }
        # view1 = {
        #     **data["view1"],
        #     **{k: data[k + "1"] for k in required_keys if (k + "1") in data},
        # }
        return self.net(data)
        #return self.net({"image0": view0, "image1": view1})

    def loss(pred, data):
        raise NotImplementedError
