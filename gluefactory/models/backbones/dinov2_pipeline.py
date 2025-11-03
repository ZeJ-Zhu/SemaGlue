import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from .dinov2 import dino #DinoVisionTransformer

class DinoV2(BaseModel):
    default_conf = {"weights": "dinov2_vitb14", "allow_resize": False}
    required_data_keys = ["image"]

    def _init(self, conf):
        # segment_path="/data/zzj/glue-factory-main1/glue-factory-main/gluefactory/models/matchers/omniglue/weights/dinov2_vitb14_pretrain.pth"
        # state_dict = torch.load(segment_path, map_location=torch.device('cpu'))
        self.net = dino.vit_base()
        # self.net.load_state_dict(state_dict, strict=False)
        
        self.set_initialized()

    def _forward(self, data):
        img = data["image"]
        height, width = img.shape[2:]
        img = F.interpolate(img, size=(int(height/1.15), int(width/1.15)), mode='bilinear', align_corners=True)
        if self.conf.allow_resize:
            img = F.upsample(img, [int(x // 14 * 14) for x in img.shape[-2:]])
        desc, cls_token = self.net.get_intermediate_layers(
            img, n=1, return_class_token=True, reshape=True
        )[0]

        return {
            "features": desc,#b c h w
            # "global_descriptor": cls_token,
            # "descriptors": desc.flatten(-2).transpose(-2, -1),#b n c
        }

    def loss(self, pred, data):
        raise NotImplementedError
