import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint

root = Path(__file__).parent.parent  # top-level directory

from .seg_attn import f_LayerNorm, Seg_FeatureProcessor, CDA
    
from einops import rearrange

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True

@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])

class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0

class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softma04x(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)

class SAF(nn.ModuleList):
    def __init__(self, dim, num_heads=2,flash: bool = False, bias=True, LayerNorm_type="WithBias"):
        super(SAF, self).__init__()
        self.s_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.f_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.dim = dim
        self.num_heads = num_heads
        self.proj = nn.Linear(dim, dim)
        self.Wq = nn.Linear(dim, dim, bias=bias)
        self.Wkv = nn.Linear(dim, 2*dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2*dim, 2*dim),
            nn.LayerNorm(2*dim,elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2*dim,dim)
        )
    
    def forward(
        self,
        desc: torch.Tensor,
        feature: torch.Tensor,
        mask: Optional[torch.Tensor] =None,
    ) ->torch.Tensor:
        feature = self.s_fnorm1(feature).transpose(1,2)
        desc = self.f_fnorm1(desc).transpose(1,2)
        feature = self.proj(feature)
        q = self.Wq(desc)
        kv = self.Wkv(feature)
        q = q.unflatten(-1, (self.num_heads, -1)).transpose(1,2)  
        kv = kv.unflatten(-1, (self.num_heads, -1, 2)).transpose(1,2) 
        k, v = kv[...,0],kv[...,1]
        context = self.inner_attn(q, k, v, mask=mask)
        context = context.permute(0,2,1,3)
        message = rearrange(context, 'b m head c -> b m (head c)', head = self.num_heads)
        message = self.out_proj(message)
        x = desc+ self.ffn(torch.cat([desc,message], -1))
        return x

class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
           
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
      
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
    
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class Seg_Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads, flash, bias)
        self.cross_attn = CrossBlock(embed_dim, num_heads, flash, bias)
        
        self.segment_attention = CDA(embed_dim, num_heads)
  
        self.cross_sd_attn = SAF(embed_dim, num_heads, flash, bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
    
    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        feature0,
        feature1,
    ):
        original_desc0 = desc0.clone()
        original_desc1 = desc1.clone()
        feature0 = self.segment_attention(desc0, feature0)
        feature1 = self.segment_attention(desc1, feature1)
    
        desc0 = self.cross_sd_attn(desc0, feature0)
        desc1 = self.cross_sd_attn(desc1, feature1)
        
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        desc0 = desc0 + self.ffn(torch.cat([desc0, original_desc0], -1))
        desc1 = desc1 + self.ffn(torch.cat([desc1, original_desc1], -1))
        desc0, desc1 = self.cross_attn(desc0, desc1)
        return desc0, desc1
    
def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    #z0 b m 1 
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores

class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)

def filter_matches(scores: torch.Tensor, th: float):
    """从对数分配矩阵中提取匹配对 [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices

    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]

    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)

    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)

    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)

    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)

    return m0, m1, mscores0, mscores1      

class SemaGlue(nn.Module):
    default_conf = {
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "segmenti_dim": 480,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False, 
        "filter_threshold": 0.1, 
        "checkpointed": False,
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = (conf.descriptor_dim) // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d= conf.num_heads, conf.n_layers, conf.descriptor_dim
        ds = conf.segment_dim
     
        self.seg_featureprocessor = Seg_FeatureProcessor(ds, d, h)
        
        self.seg_tranformers = nn.ModuleList(
            [Seg_Transformer(d, h, conf.flash) for _ in range(n)]
        )
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n-1)]
        )
    
    def forward(self, data: dict) -> dict:
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        data0, data1 = data["image0"], data["image1"]
        _, m, _ = kpts0.shape
        _, n, _ = kpts1.shape
        size0, size1 = data0.shape[2:], data1.shape[2:]
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()
        desc0 = data["descriptors0"].detach().transpose(-1,-2).contiguous()
        desc1 = data["descriptors1"].detach().transpose(-1,-2).contiguous()
        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim  
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        feature_map0 = data['feature_map0']
        feature_map1 = data['feature_map1']
        global_feature0 = self.seg_featureprocessor(desc0, feature_map0)
        global_feature1 = self.seg_featureprocessor(desc1, feature_map1)
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.seg_tranformers[i], desc0, desc1, encoding0, encoding1, global_feature0, global_feature1
                )
            else:
                desc0, desc1 = self.seg_tranformers[i](
                    desc0, desc1, encoding0, encoding1, global_feature0, global_feature1)

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "desc0": desc0,
            "desc1": desc1,
        }

        return pred

__main_model__ = SemaGlue
