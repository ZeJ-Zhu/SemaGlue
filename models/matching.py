import torch

from .superpoint import SuperPoint
from .semaglue_pipeline import SemaGluePipeline

from pathlib import Path
from omegaconf import OmegaConf

class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + SemaGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))

        default_conf = OmegaConf.create(SemaGluePipeline.default_conf)
        self.semaglue = SemaGluePipeline(default_conf).eval().cuda()  # load the matcher

        print('Loaded SemaGlue model')
        exper = Path("./models/weights/checkpoint_best.tar")
        ckpt = exper
        ckpt = torch.load(str(ckpt), map_location="cpu")

        state_dict = ckpt["model"]

        matcher_state_dict = {k.split('matcher.', 1)[1]: v for k, v in state_dict.items() if k.startswith('matcher.')}
        self.semaglue.matcher.load_state_dict(matcher_state_dict, strict=True)


    def forward(self, data):
        """ Run SuperPoint and SemaGlue """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**self.semaglue(data)}
        pred = {**data, **pred}

        return pred
