from pathlib import Path
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from ..semaglue.base_model import BaseModel

from .mscan import MSCAN

def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):

    return F.interpolate(input, size, scale_factor, mode, align_corners)

class SegNext(BaseModel):
    default_conf={
        "in_channels" : 3,
        "embed_dims"  : [32,64,160,256], 
        "ffn_ratios"  : [8,8,4,4], 
        "depths"       : [3,3,5,2],
        "num_stages"  : 4,
        "dropout"     : 0.1 ,
        "drop_path"   : 0.0, 
    }

    required_data_keys = []
    
    def _init(self, conf):
        norm_cfg_dict = dict(type=conf.norm_typ)
        self.backbone = MSCAN(in_chans=conf.in_channels, embed_dims=conf.embed_dims,
                            mlp_ratios=conf.ffn_ratios, depths=conf.depths, num_stages=conf.num_stages,
                            drop_rate=conf.drop_path, drop_path_rate=conf.dropout, norm_cfg=norm_cfg_dict)
        
        segment_path= Path(__file__).parent.parent / 'weights/segnext_tiny_512x512_ade_160k.pth'
        state_dict = torch.load(segment_path, map_location=torch.device('cpu'))
        
        backbone_state_dict = {k: v for k, v in state_dict['state_dict'].items() if k.startswith('backbone')}

        self.load_state_dict(backbone_state_dict, strict=True)

    def _forward(self, image):
        pred = {}
        if image.shape[1] == 1:
            image = image.repeat(1,3,1,1)
        
        enc_feats = self.backbone(image)
        features = enc_feats[1:] 
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        dec_out = torch.cat(features, dim=1)

        pred = {
            "feature_map": dec_out,
        }
        return pred
    
    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError