import warnings
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange

from ...settings import DATA_PATH
from ..utils.losses import NLLLoss
from ..utils.metrics import matcher_metrics
from ..aff_net.fusion import AFF,iAFF,MS_CAM,Aff
from .seg_attn import Seg_TransformerBlock9,Seg_TransformerBlock11,Seg_TransformerBlock25,Seg_TransformerBlock12,Seg_TransformerBlock21,Seg_TransformerBlock19,Seg_TransformerBlock18,Seg_TransformerBlock17,Seg_TransformerBlock16,Seg_TransformerBlock15,Seg_TransformerBlock14,Seg_TransformerBlock13,Seg_TransformerBlock8,f_LayerNorm,Seg_TransformerBlock6,Seg_TransformerBlock3,Seg_TransformerBlock4,Seg_TransformerBlock5,Seg_TransformerBlock7,Seg_TransformerBlock10
from .seg_attn import Seg_FeatureProcessor1
from .multimodel_pathway import reparameterize_aux_into_target_model
from .utils import SELayer,SELayer1

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
    # print("freqs[0]",freqs[0].shape)
    # print("rotate_half(t)",rotate_half(t).shape)
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])                                                                                               

def normalize_complex_vector(complex_vector):
    magnitude = torch.abs(complex_vector)
    phase = torch.angle(complex_vector)
    normalized_vector = magnitude * torch.exp(1j * torch.zeros_like(phase))
    return normalized_vector

class LearnableFourierPositionalEncoding(nn.Module):
    #M:位置编码的维度（即傅里叶频率的数量），dim:输出的位置编码向量的维度，F_dim：用于傅里叶变换的中间表示的维度，如果未提供和dim相同
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)#使用线性层self.Wr将输入位置x投影到维度F_dim//2
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
            #detach()返回一个新的 Tensor，这个 Tensor 和原来的 Tensor 共享相同的内存空间，但是不会被计算图所追踪，也就是说它不会参与反向传播
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
        if allow_flash and not FLASH_AVAILABLE:#是否在可能的情况下使用FlashAttention
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask: Optional[torch.Tensor] = None) -> torch.Tensor:#mask可选的
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
        else:#如果FlashAttention和标准实现都不可用，则使用标准点积和softmax计算注意力
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            attn = F.softma04x(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)
    
class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True#使用偏置
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0#检查embed_dim是否可以均匀地被“num_heads“整除
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)#线性层，将输入嵌入维度映射到三倍的嵌入维度
        self.inner_attn = Attention(flash)#自定义的注意力机制模块
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)#映射回
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
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
        self.scale = dim_head**-0.5#self.scale是dim_head的倒数的平方根
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

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):#接受一个func函数，应用于x0和x1
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(#将qk和v张量进行形状变换，以适应多头注意力机制的计算。将最后一个维度划分为多个维度，然后进行维度交换
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(
                qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5#乘上self.scale的平方根
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)#计算注意力矩阵，通过使用torch.einsum进行批量矩阵乘法，得到形状为（batch_size,num_heads,seq_len,seq_len)的注意力矩阵
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))#将掩码应用到注意力矩阵，将不需要关注的位置的注意力分数设为负无穷。这可以确保在 softmax 操作时，这些位置的注意力分数趋近于零
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            #使用 torch.einsum 将注意力矩阵与值张量相乘，得到加权的输出 m0 和 m1
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            #如果存在掩码 (mask)，则对输出进行 nan-to-num 操作，将 NaN 替换为零
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        #m0 和 m1 被映射（map_）到新的形状：通过将每个张量进行转置，交换其第一和第二维，然后使用 flatten 函数在 start_dim=-2 的维度上进行展平。
        #这通常用于将形状为 (batch_size, num_heads, seq_len, dim) 的张量变形为 (batch_size, num_heads * seq_len, dim)。
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1

class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, mask0, mask1)
        else:
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)
            return self.cross_attn(desc0, desc1)

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        desc0 = self.self_attn(desc0, encoding0, mask0)
        desc1 = self.self_attn(desc1, encoding1, mask1)
        return self.cross_attn(desc0, desc1, mask)

class Seg_Transformer3(nn.Module):#divide_21_ffn_nfnorm_c_pool_newSEncoder_loadfalse_lr20/mega_divide_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr20
    def __init__(self, embed_dim: int, seg_embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True):
        super().__init__()
        self.self_attn = SelfBlock(embed_dim, num_heads, flash, bias)
 
        self.cross_attn = CrossBlock(embed_dim, num_heads, flash, bias)
        
        self.segment_attention = Seg_TransformerBlock20(embed_dim, num_heads)
   
        self.cross_sd_attn = TransformerBlock_4(embed_dim, num_heads, flash, bias)

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
        global_feature0,
        global_feature1,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, global_feature0, global_feature1,mask0, mask1)
        else:
            original_desc0 = desc0.clone()
            original_desc1 = desc1.clone()

            global_feature0 = self.segment_attention(desc0, global_feature0)
            global_feature1 = self.segment_attention(desc1, global_feature1)

            desc0 = self.cross_sd_attn(desc0, global_feature0)
            desc1 = self.cross_sd_attn(desc1, global_feature1)
            
            desc0 = self.self_attn(desc0, encoding0)
            desc1 = self.self_attn(desc1, encoding1)

            desc0 = desc0 + self.ffn(torch.cat([desc0, original_desc0], -1))
            desc1 = desc1 + self.ffn(torch.cat([desc1, original_desc1], -1))

            desc0, desc1 = self.cross_attn(desc0, desc1)
            return desc0, desc1, global_feature0, global_feature1

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, global_feature0, global_feature1, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        original_desc0 = desc0.clone()
        original_desc1 = desc1.clone()
        global_feature0 = self.segment_attention(desc0, global_feature0)
        global_feature1 = self.segment_attention(desc1, global_feature1)
        # feature0 = self.segment_attention(desc0, feature_map0)
        # feature1 = self.segment_attention(desc1, feature_map1)
        desc0 = self.cross_sd_attn(desc0, global_feature0)
        desc1 = self.cross_sd_attn(desc1, global_feature1)
        
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        desc0 = desc0 + self.ffn(torch.cat([desc0, original_desc0], -1))
        desc1 = desc1 + self.ffn(torch.cat([desc1, original_desc1], -1))
        desc0, desc1 = self.cross_attn(desc0, desc1)
        return desc0, desc1, global_feature0, global_feature1

def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    #z0 b m 1 
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2) #b m m
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)#scores全零 b m+1 n+1
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
        #计算相似矩阵
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)
    
def filter_matches(scores: torch.Tensor, th: float):
    """从对数分配矩阵中提取匹配对 [Bx M+1 x N+1]"""
    # 计算对数分配矩阵在第二维（列）和第三维（行）上的最大值及其索引
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices

    # 创建索引范围 [0, M] 和 [0, N]
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]

    # 判断是否存在互相匹配
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)

    # 计算有效匹配的分数
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)

    # 根据互相匹配和分数是否超过阈值来判断匹配是否有效
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)

    # 将无效匹配的索引设置为-1
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)

    # 返回有效匹配的索引和分数
    return m0, m1, mscores0, mscores1

class LightGlue(nn.Module):
    default_conf = {
        "name": "lightglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "segment_dim": 256,#equal to segment的维度
        "num_classes": 144,#
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "depth_confidence": -1,  # early stopping, disable with -1
        "width_confidence": -1,  # point pruning, disable with -1
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "weights": None,  # either a path or the name of pretrained weights (disk, ...)
        "weights_from_version": "v0.1_arxiv",
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]#期望输入

    #url = "https://github.com/cvg/LightGlue/releases/download/{}/{}_lightglue.pth"

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        # 根据配置参数设置输入投影（Linear 或 Identity）
        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        # 设置位置编码
        head_dim = (conf.descriptor_dim) // conf.num_heads
        # head_dim = (2* conf.descriptor_dim) // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        # 设置多层 Transformer 模块
        h, n, d= conf.num_heads, conf.n_layers, conf.descriptor_dim
        ds = conf.segment_dim

        self.seg_featureprocessor = Seg_FeatureProcessor1(ds, d, h)
        # self.seg_featureprocessor = Seg_FeatureProcessor4(ds, d, h) #npre
        # self.seg_featureprocessor = Seg_FeatureProcessor6(ds, d) #ncc
        
        self.seg_tranformers = nn.ModuleList(
            [Seg_Transformer3(d, ds, h, conf.flash) for _ in range(n)]
        )

        # 设置匹配分配模块和令牌置信度模块
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])

        self.token_confidence = nn.ModuleList(
            [TokenConfidence(d) for _ in range(n-1)]
        )
        
        # 设置损失函数
        self.loss_fn = NLLLoss(conf.loss)

        # 加载预训练权重（如果提供了权重路径）
        state_dict = None
        if conf.weights is not None:
            if Path(conf.weights).exists():
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                state_dict = torch.load(str(DATA_PATH / conf.weights), map_location="cpu")
            else:
                fname = (
                        f"{conf.weights}_{conf.weights_from_version}".replace(".", "-")
                        + ".pth"
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    self.url.format(conf.weights_from_version, conf.weights),
                    file_name=fname,
                )

        # 如果存在预训练权重，加载到模型中（并进行一些重命名处理）
        if state_dict:
            for i in range(self.conf.n_layers):
                pattern = f"self_attn.{i}", f"seg_tranformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"seg_tranformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"segment_attention.{i}", f"seg_tranformers.{i}.segment_attention"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_sd_attn.{i}", f"seg_tranformers.{i}.cross_sd_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=False)

    #选择禁用以确保编译的稳定性和正确性。
    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )
        
        for i in range(self.conf.n_layers):
            self.seg_tranformers[i] = torch.compile(
                self.seg_tranformers[i], mode = mode, fullgraph=True
            )
    
    def forward(self, data: dict) -> dict:#输入数据被假定为一个字典，该方法返回一个字典类型的对象
        #检查输入数据是否包含必需的键
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"

        #获取输入数据中的关键信息，如关键点，描述子等
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape   #b=3 m=512  _=2
        #print("kpts0.shape"+kpts0.shape)
        b, n, _ = kpts1.shape   #b=3 n=512  _=2device
        device = kpts0.device
        #根据输入数据中是否包含视图信息，获取图像大小
        if "view0" in data.keys() and "view1" in data.keys():
            size0 = data["view0"].get("image_size")
            size1 = data["view1"].get("image_size")

        #对关键点进行归一化
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        #如果配置要求添加尺度和方向信息，则将其添加到关键点
        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        #获取输入数据的描述子
        desc0 = data["descriptors0"].contiguous()
        desc1 = data["descriptors1"].contiguous()

        #确保描述子的最后一维与模型配置一致
        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        #如果启用了自动混合精度，将描述子转换为半精度
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        #对描述子进行投影
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        # GNN + final_proj + assignment
        do_early_stop = self.conf.depth_confidence > 0 and not self.training
        do_point_pruning = self.conf.width_confidence > 0 and not self.training

        all_desc0, all_desc1 = [], []

        if do_point_pruning:
            ind0 = torch.arange(0, m, device=device)[None]
            ind1 = torch.arange(0, n, device=device)[None]
            # We store the index of the layer at which pruning is detected.
            prune0 = torch.ones_like(ind0)
            prune1 = torch.ones_like(ind1)
        token0, token1 = None, None

        feature_map0 = data['feature_map0']
        feature_map1 = data['feature_map1']

        global_feature0 = self.seg_featureprocessor(desc0, feature_map0)
        global_feature1 = self.seg_featureprocessor(desc1, feature_map1)

        # global_feature0 = self.seg_featureprocessor(feature_map0)
        # global_feature1 = self.seg_featureprocessor(feature_map1)
        
        #循环执行GNN操作
        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                # desc0, desc1 = checkpoint(#checkpoint优化内存占用
                #     self.transformers[i], desc0, desc1, encoding0, encoding1
                # )
                
                desc0, desc1, local_feature0, local_feature1 = checkpoint(
                    self.seg_tranformers[i], desc0, desc1, encoding0, encoding1, global_feature0, global_feature1
                )
            else:
                # desc0, desc1, global_feature0, global_feature1, local_feature0, local_feature1 = self.seg_tranformers[i](
                #     desc0, desc1, encoding0, encoding1, global_feature0, global_feature1, local_feature0, local_feature1)
                
                desc0, desc1, local_feature0, local_feature1 = self.seg_tranformers[i](
                    desc0, desc1, encoding0, encoding1, global_feature0, global_feature1)

            if self.training or i == self.conf.n_layers - 1:#如果处于训练模式，并且是最后一层
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

            # only for eval
            # if do_early_stop:
            #     assert b == 1
            #     token0, token1 = self.token_confidence[i](desc0, desc1)
            #     if self.check_if_stop(token0[..., :m, :], token1[..., :n, :], i, m + n):
            #         break
            # if do_point_pruning:
            #     assert b == 1
            #     scores0 = self.log_assignment[i].get_matchability(desc0)#一个线性层加softmax
            #     prunemask0 = self.get_pruning_mask(token0, scores0, i)#得到被mask的desc
            #     keep0 = torch.where(prunemask0)[1]
            #     ind0 = ind0.index_select(1, keep0)
            #     desc0 = desc0.index_select(1, keep0)
            #     encoding0 = encoding0.index_select(-2, keep0)
            #     prune0[:, ind0] += 1
            #     scores1 = self.log_assignment[i].get_matchability(desc1)
            #     prunemask1 = self.get_pruning_mask(token1, scores1, i)
            #     keep1 = torch.where(prunemask1)[1]
            #     ind1 = ind1.index_select(1, keep1)
            #     desc1 = desc1.index_select(1, keep1)
            #     encoding1 = encoding1.index_select(-2, keep1)
            #     prune1[:, ind1] += 1

        #选择最终的描述子和匹配信息
        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)

        if do_point_pruning:
            m0_ = torch.full((b, m), -1, device=m0.device, dtype=m0.dtype)
            m1_ = torch.full((b, n), -1, device=m1.device, dtype=m1.dtype)
            m0_[:, ind0] = torch.where(m0 == -1, -1, ind1.gather(1, m0.clamp(min=0)))
            m1_[:, ind1] = torch.where(m1 == -1, -1, ind0.gather(1, m1.clamp(min=0)))
            mscores0_ = torch.zeros((b, m), device=mscores0.device)
            mscores1_ = torch.zeros((b, n), device=mscores1.device)
            mscores0_[:, ind0] = mscores0
            mscores1_[:, ind1] = mscores1
            m0, m1, mscores0, mscores1 = m0_, m1_, mscores0_, mscores1_
        else:
            prune0 = torch.ones_like(mscores0) * self.conf.n_layers
            prune1 = torch.ones_like(mscores1) * self.conf.n_layers

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "ref_descriptors0": torch.stack(all_desc0, 1),
            "ref_descriptors1": torch.stack(all_desc1, 1),
            # "ref_seg0": feature0,
            # "ref_seg1": feature1,
            "log_assignment": scores,
            "prune0": prune0,
            "prune1": prune1,
        }

        return pred
  
    def confidence_thresholds(self, layer_index: int) -> float:
        """scaled confidence threshold"""
        threshold = 0.8 + 0.1 * np.exp(-4.0 * layer_index / self.conf.n_layers)
        return np.clip(threshold, 0, 1)

    def get_pruning_mask(
        self, confidences: torch.Tensor, scores: torch.Tensor, layer_index: int
    ) -> torch.Tensor:
        """mask points which should be removed"""
        #保留分数高于 (1 - self.conf.width_confidence) 阈值的点
        keep = scores > (1 - self.conf.width_confidence)
        if confidences is not None:  # Low-confidence points are never pruned.
            #按位或
            keep |= confidences <= self.confidence_thresholds[layer_index]
        #得到是mask
        return keep

    def check_if_stop(
        self,
        confidences0: torch.Tensor,
        confidences1: torch.Tensor,
        layer_index: int,
        num_points: int,
    ) -> torch.Tensor:
        """evaluate stopping condition"""
        confidences = torch.cat([confidences0, confidences1], -1)
        threshold = self.confidence_thresholds[layer_index]
        ratio_confident = 1.0 - (confidences < threshold).float().sum() / num_points
        return ratio_confident > self.conf.depth_confidence

    def pruning_min_kpts(self, device: torch.device):
        if self.conf.flash and FLASH_AVAILABLE and device.type == "cuda":
            return self.pruning_keypoint_thresholds["flash"]
        else:
            return self.pruning_keypoint_thresholds[device.type]

    def loss(self, pred, data):
        def loss_params(pred, i):
            # pred['ref_descriptors0'] b 9 m c
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]#, pred["ref_seg0"], pred["ref_seg1"]
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)# nll损失
        N = pred["ref_descriptors0"].shape[1]
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        # B = pred['log_assignment'].shape[0]
        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)  #b
        for i in range(N - 1):
            params_i = loss_params(pred, i)
            nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            if self.conf.loss.gamma > 0.0:
                weight = self.conf.loss.gamma ** (N - i - 1)
            else:
                weight = i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll * weight

            losses["confidence"] += self.token_confidence[i].loss(
                pred["ref_descriptors0"][:, i],
                pred["ref_descriptors1"][:, i],
                params_i["log_assignment"],
                pred["log_assignment"],
                # pred["ref_seg0"],
                # pred["ref_seg1"]
            ) / (N - 1)   #训练置信器，pointout的

            del params_i
        losses["total"] /= sum_weights

        # confidences
        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = LightGlue
