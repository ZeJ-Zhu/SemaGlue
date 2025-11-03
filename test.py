# import debugpy
# debugpy.listen(('127.0.13.25', 8001))
# debugpy.wait_for_client()
import torch
import time

def z_order_flatten(image_tensor):
    # 获取图像的高度、宽度和通道数
    h, w, c = image_tensor.size()
    
    # 初始化结果张量
    result = torch.zeros((h * w, c), dtype=image_tensor.dtype, device=image_tensor.device)
    
    # 对图像进行Z字形拉长
    for i in range(h):
        if i % 2 == 0:
            # 偶数行：从左到右
            result[i*w:(i+1)*w, :] = image_tensor[i, :, :]
        else:
            # 奇数行：从右到左
            result[i*w:(i+1)*w, :] = image_tensor[i, torch.arange(w-1, -1, -1), :]
    
    return result

def z_order_flatten_batch(image_tensor):
    # 获取图像张量的形状信息
    b, c, h, w = image_tensor.size()
    
    # 对图像张量进行Z字形拉长
    result = []
    for i in range(h):
        print(image_tensor[:,:,i,:].shape)
        if i % 2 == 0:
            result.append(image_tensor[:, :, i, :])
        else:
            result.append(torch.flip(image_tensor[:, :, i, :], dims=[2]))
    
    # 沿高度维度拼接结果张量
    result = torch.cat(result, dim=2)
    
    return result

def z_order_flatten(image_tensor):
    b, c, h, w = image_tensor.size()
    # 将图像的高度和宽度拉平成一个一维序列
    flattened = image_tensor.view(b, c, -1)

    # 创建一个掩码来指定z字形拉长的顺序
    mask = torch.arange(h * w).view(h, w)
    mask[1::2, :] = mask[1::2, :].flip(1)
    mask = mask.view(-1)

    # 使用掩码对拉平后的张量重新排序
    flattened_z_order = flattened[:, :, mask]

    return flattened_z_order

# 示例用法
# 创建一个包含两张3x3大小的示例图像张量
# image_batch = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                               [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                               [[19, 20, 21], [22, 23, 24], [25, 26, 27]]],
#                              [[[28, 29, 30], [31, 32, 33], [34, 35, 36]],
#                               [[37, 38, 39], [40, 41, 42], [43, 44, 45]],
#                               [[46, 47, 48], [49, 50, 51], [52, 53, 54]]]], dtype=torch.float32)

# print(image_batch.shape)
# # 调用函数进行Z字形拉长
# start_time = time.time()
# flattened_image_batch, mask = z_order_flatten(image_batch)
# old_time = time.time() - start_time

# # 打印结果
# print(f"z_order_flatten time: {old_time:.6f} seconds")
# print(flattened_image_batch)
# print(flattened_image_batch.shape)
# print(mask)

#--------------------------------------------
import torch
import torch.nn as nn
import time
from gluefactory.models.matchers.utils import ChannelAttention
class Get_Si(nn.Module):
    def __init__(self,stride=1):
        super(Get_Si, self).__init__()
        self.stride = stride
        ##self.weight_matrix = nn.Parameter(torch.randn(9))  # 定义可学习的权重矩阵
        #self.Conv = nn.Conv2d(9,1,kernel_size=1,stride=1)
        #self.linear = nn.Linear(seg_dim, 1,bias=True)
        
    def expand_dimension(self, wx, c):
        wx = torch.unsqueeze(wx, dim=2)
        wx = wx.repeat(1,1,c)
        return wx
    
    def get_Si(self, original_kpts,segment):
        batchsize,c,h,w = segment.shape
        _, m, _ = original_kpts.shape   #b=3 m=512  _=2

        x = original_kpts[:, :, 0].permute(1, 0)
        y = original_kpts[:, :, 1].permute(1, 0)

        x = x * h  # -> b m
        y = y * w  # -> b m
        #  stack -> b m 2
        #  repeat -> b c m 2
        #  segment[x, y]
        x = x.floor().to(torch.int64)  # m b
        y = y.floor().to(torch.int64)

        #扩展维度c
        x = self.expand_dimension(x, c).transpose(1,2)
        y = self.expand_dimension(y, c).transpose(1,2) #(m,c,b)

        # 将位置点坐标转换成一维索引
        # 注意：这里假设特征图的布局是 (height, width, channels, batchsize)
        # 如果是其他布局，需要调整下标的顺序
        indices = x * w + y
        #indices = indices.floor().to(torch.int64)
        indices = torch.clamp(indices, min = 0, max = h*w)
        #print(indices.shape)
        # 获取像素值
        fsegment = segment.permute(2, 3, 1, 0).view(-1, c, batchsize)# h*w c b
        #print(segment.shape)
        # 使用gather函数获取对应的值
        result = torch.gather(fsegment, 0, indices)
        #print(result.shape)
        result = result.permute(2,0,1)
        return result  # 将结果调整为 (b, m, c)

# class NewGet_Si(nn.Module):
#     def __init__(self,stride=1):
#         super(NewGet_Si, self).__init__()
#         self.stride = stride
#         ##self.weight_matrix = nn.Parameter(torch.randn(9))  # 定义可学习的权重矩阵
#         #self.Conv = nn.Conv2d(9,1,kernel_size=1,stride=1)
#         #self.linear = nn.Linear(seg_dim, 1,bias=True)
        
#     def expand_dimension(self, wx, c):
#         wx = torch.unsqueeze(wx, dim=1)
#         wx = wx.repeat(1,c,1,1)
#         return wx
    
#     def get_Si(self, original_kpts,segment):
#         batchsize,c,h,w = segment.shape
#         _, m, _ = original_kpts.shape   #b=3 m=512  _=2

#         x = original_kpts[:, :, 0]
#         y = original_kpts[:, :, 1]

#         x = x * h  # -> b m
#         y = y * w  # -> b m
#         #  stack -> b m 2
#         #  repeat -> b c m 2
#         #  segment[x, y]
#         x = x.floor().to(torch.int64)  # b m
#         y = y.floor().to(torch.int64)
        
#         indices = torch.stack((x, y), dim=2)
#         indices = self.expand_dimension(indices, c)
        
#         result = segment[indices]

#         return result  # 将结果调整为 (b, m, c)
class NewGet_Si(nn.Module):
    def __init__(self, stride=1):
        super(NewGet_Si, self).__init__()
        self.stride = stride

    def expand_dimension(self, wx, c):
        wx = torch.unsqueeze(wx, dim=1)
        wx = wx.repeat(1, c, 1, 1)
        return wx
    
    def get_Si(self, original_kpts, segment):
        batchsize, c, h, w = segment.shape
        _, m, _ = original_kpts.shape   # b=3, m=512, _=2

        x = original_kpts[:, :, 0] * w  # b, m
        y = original_kpts[:, :, 1] * h  # b, m
        
        x = x.floor().to(torch.int64)  # b, m
        y = y.floor().to(torch.int64)  # b, m
        
        x = x.unsqueeze(1).repeat(1, c, 1)  # b, c, m
        y = y.unsqueeze(1).repeat(1, c, 1)  # b, c, m
        
        # Make sure indices are within the image bounds
        x = torch.clamp(x, 0, h-1)
        y = torch.clamp(y, 0, w-1)
        
        # Gather values from segment using the indices
        indices = x * w + y
        segment_flat = segment.view(batchsize, c, -1)
        result = torch.gather(segment_flat, 2, indices)
        
        return result.permute(0,2,1)  # 返回结果，形状为 (b, c, m)
    
class OldPatchExtractor(nn.Module):
    def __init__(self,seg_dim,stride=1):
        super(OldPatchExtractor, self).__init__()
        self.stride = stride
        self.weight_matrix = nn.Parameter(torch.randn(9))  # 定义可学习的权重矩阵
        self.pool = ChannelAttention(seg_dim, reduction=1)
        self.Conv = nn.Conv2d(9,1,kernel_size=1,stride=1)
        self.linear = nn.Linear(seg_dim, 1,bias=True)

    def expand_dimension(self,wx, c,dim=2):
        wx = torch.unsqueeze(wx, dim)
        wx = wx.repeat(1,1,c,1)
        return wx
    
    def get_Patchsi(self, original_kpts, segment):
        batchsize, c, h, w = segment.shape
        _, m, _ = original_kpts.shape
        x = original_kpts[:, :, 0].permute(1, 0)
        y = original_kpts[:, :, 1].permute(1, 0)

        x = x * h
        y = y * w
        x = x.floor().to(torch.int64)
        y = y.floor().to(torch.int64)
        # x = self.expand_dimension(x, c).transpose(1,2)
        # y = self.expand_dimension(y, c).transpose(1,2) #(m,c,b)
        # 定义周围像素的相对位置
        offsets = torch.tensor([-1*self.stride, 0, self.stride], dtype=torch.int64, device=segment.device)
        surrounding_indices = [(x + dx, y + dy) for dx in offsets for dy in offsets]
        # 转换为一维索引
        indices = [x * w + y for x, y in surrounding_indices]
        surrounding_indices_stack = torch.stack(indices, dim=0)
        surrounding_indices_stack = self.expand_dimension(surrounding_indices_stack, c,2)# 9 m c b
        surrounding_indices_stack = torch.clamp(surrounding_indices_stack, min = 0, max = h*w-1)
        # for i in range(len(indices)):
        #     indices[i] = self.expand_dimension(indices[i], c,1)# m c b
        # 获取像素值
        fsegment = segment.permute(2, 3, 1, 0).view(-1, c, batchsize)# h*w c b
        #
        surrounding_pixels = []
        for i in range(surrounding_indices_stack.shape[0]):
            idx = surrounding_indices_stack[i]
            gathered_pixels = torch.gather(fsegment, 0, idx)
            surrounding_pixels.append(gathered_pixels)
        surrounding_pixels = torch.stack(surrounding_pixels, dim=0)
        pool = self.Conv(surrounding_pixels.permute(1,0,2,3)).squeeze(1).permute(2,0,1)
        s = self.pool(segment).view(batchsize,c,-1).permute(0,2,1)#b 1 c
        surrounding_s =self.linear(surrounding_pixels[4].permute(2,0,1))
        confidence = torch.nn.functional.normalize(surrounding_s,dim=1)#b m 1
        result = (1-confidence)*pool + surrounding_pixels[4].permute(2,0,1)+s
        return result
    
class PatchExtractor(nn.Module):
    def __init__(self, seg_dim, stride=1):
        super(PatchExtractor, self).__init__()
        self.stride = stride
        self.weight_matrix = nn.Parameter(torch.randn(9))  # 定义可学习的权重矩阵
        self.pool = ChannelAttention(seg_dim, reduction=1)
        self.Conv = nn.Conv2d(9,1,kernel_size=1,stride=1)
        self.linear = nn.Linear(seg_dim, 1,bias=True)

    def expand_dimension(self, tensor, c, dim):
        # This method should be defined as per your specific needs
        # Placeholder function for expanding the tensor dimension
        return tensor.unsqueeze(dim).expand(-1,-1, c, -1)

    def get_Patchsi(self, original_kpts, segment):
        batchsize, c, h, w = segment.shape
        _, m, _ = original_kpts.shape

        x = original_kpts[:, :, 0].permute(1, 0) * h
        y = original_kpts[:, :, 1].permute(1, 0) * w

        x = x.floor().to(torch.int64)
        y = y.floor().to(torch.int64)

        # 定义周围像素的相对位置
        offsets = torch.tensor([-1*self.stride, 0, self.stride], dtype=torch.int64, device=segment.device)
        dx, dy = torch.meshgrid(offsets, offsets, indexing='ij')
        dx, dy = dx.reshape(-1), dy.reshape(-1)

        # 计算所有相对位置的索引
        x_offsets = (x.unsqueeze(-1) + dx).clamp(0, h - 1)#m b 9
        y_offsets = (y.unsqueeze(-1) + dy).clamp(0, w - 1)
        surrounding_indices = (x_offsets * w + y_offsets).permute(0,2,1).reshape(-1, batchsize)#m*9 b

        # 获取像素值
        fsegment = segment.permute(2, 3, 1, 0).reshape(-1, c, batchsize)#hw c b
        surrounding_pixels = fsegment.gather(0, surrounding_indices.unsqueeze(1).expand(-1, c, -1))#m*9 c b

        surrounding_pixels = surrounding_pixels.reshape(9, m, batchsize, c)
        surrounding_pixels = surrounding_pixels.permute(0, 3, 2, 1)#9 c b m

        pool = self.Conv(surrounding_pixels.permute(2, 0, 3, 1)).squeeze(1)
        s = self.pool(segment).view(batchsize, c, -1).permute(0, 2, 1)

        confidence = self.linear(surrounding_pixels[4].permute(1, 2, 0))
        # confidence = torch.nn.functional.normalize(surrounding_s, dim=1)
        result = (1 - confidence) * pool + surrounding_pixels[4].permute(1, 2, 0) + s

        return result


# batchsize, c, h, w = 64, 256, 64, 64
# m = 256  # number of keypoints
# original_kpts = torch.rand(batchsize, m, 2)
# original_kpts[:,:,0] = torch.clamp(original_kpts[:,:,0], min = 0, max = h)
# original_kpts[:,:,1] = torch.clamp(original_kpts[:,:,1], min = 0, max = w)
# segment = torch.rand(batchsize, c, h, w)
# original_kpts = torch.tensor([[[0.3, 0.4], [0.4, 0.4], [0.7, 0.8], [0.8, 0.9]],
#                              [[0.81, 0.91], [0.1, 0.32], [0.42, 0.92], [0.0, 0.83]]], dtype=torch.float32)
# segment = torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
#                               [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                               [[19, 20, 21], [22, 23, 24], [25, 26, 27]]],
#                              [[[28, 29, 30], [31, 32, 33], [34, 35, 36]],
#                               [[37, 38, 39], [40, 41, 42], [43, 44, 45]],
#                               [[46, 47, 48], [49, 50, 51], [52, 53, 54]]]], dtype=torch.float32)

# extractor = Get_Si()
# newextactor = NewGet_Si()
# # old_extractor = OldPatchExtractor(seg_dim=c)
# # new_extractor = PatchExtractor(seg_dim=c)
# # 测试旧代码的时间
# start_time = time.time()
# old_result = extractor.get_Si(original_kpts, segment)
# old_time = time.time() - start_time
# print(old_time)

# new_start_time = time.time()
# result = newextactor.get_Si(original_kpts, segment)
# newtime = time.time() - new_start_time
# print(newtime)
# print(old_result.equal(result))
#print(old_result)

# # 测试新代码的时间
# start_time = time.time()
# new_result = new_extractor.get_Patchsi(original_kpts, segment)
# print("new_result:", new_result)
# new_time = time.time() - start_time

# # 打印结果
# print(f"Old code execution time: {old_time:.6f} seconds")
# print(f"New code execution time: {new_time:.6f} seconds")

# # 检查结果是否一致
# if torch.allclose(old_result, new_result, atol=1e-6):
#     print("The results are consistent between the old and new code.")
# else:
#     print("The results are not consistent between the old and new code.")






#----------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional

class CrossModalReparamLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 origin_layer=None,
                 aux_weight=None,
                 is_aux_trainable=True):
        super().__init__(in_features, out_features, bias)
        self.cross_modal_scale = nn.Parameter(torch.zeros(1))
        assert self.weight.size() == aux_weight.size(), 'Target weight and aux weight must have the same shape'
        self.aux_weight = aux_weight
        self.aux_weight.requires_grad_(is_aux_trainable)
        if origin_layer is not None:
            with torch.no_grad():
                self.weight.copy_(origin_layer.weight)
                self.bias.copy_(origin_layer.bias)

    def forward(self, input):
        weight = self.weight + self.cross_modal_scale * self.aux_weight
        return F.linear(input, weight, self.bias)


def build_cross_modal_reparam_linear(origin_layer, aux_layer):
    assert origin_layer.weight.size() == aux_layer.weight.size()
    return CrossModalReparamLinear(in_features=origin_layer.in_features, out_features=origin_layer.out_features, origin_layer=origin_layer,
                                   bias=origin_layer.bias is not None,
                                   aux_weight=aux_layer.weight)


def _get_attr_by_name(obj, attr_name):
    attrs = attr_name.split('.')
    for a in attrs:
        obj = obj.__getattr__(a)
    return obj

def _set_attr_by_name(obj, attr_name, attr_value):
    owner = obj
    attr_names = attr_name.split('.')
    if len(attr_names) > 1:
        for a in attr_names[:-1]:
            owner = owner.__getattr__(a)
    owner.__setattr__(attr_names[-1], attr_value)

def change_original_linear_to_reparam(target_module, aux_module, layer_name):
    origin_linear_layer = _get_attr_by_name(target_module, layer_name)
    aux_linear_layer = _get_attr_by_name(aux_module, layer_name)
    reparam_layer = build_cross_modal_reparam_linear(origin_linear_layer, aux_linear_layer)
    _set_attr_by_name(target_module, layer_name, reparam_layer)


def reparameterize_aux_into_target_model(target_model, aux_model,
                               layer_names=('Wqkv', 'out_proj'), main_body_name='blocks'):
    target_transformer_blocks = _get_attr_by_name(target_model, main_body_name)
    aux_transformer_blocks = _get_attr_by_name(aux_model, main_body_name)
    for target_block, aux_block in zip(target_transformer_blocks, aux_transformer_blocks):
        for layer_name in layer_names:
            change_original_linear_to_reparam(target_block, aux_block, layer_name)

def reparameterize_selfblock_layers(target_model, aux_model):
    for target_block, aux_block in zip(target_model, aux_model):
        target_block.Wqkv = CrossModalReparamLinear(
            target_block.Wqkv.in_features,
            target_block.Wqkv.out_features,
            origin_layer=target_block.Wqkv,
            aux_weight=aux_block.Wqkv.weight
        )
        target_block.out_proj = CrossModalReparamLinear(
            target_block.out_proj.in_features,
            target_block.out_proj.out_features,
            origin_layer=target_block.out_proj,
            aux_weight=aux_block.out_proj.weight
        )

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

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
        elif FLASH_AVAILABLE:#如果未启用FlashAttention或设备不是cuda，回退到标准的Pytorch实现
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
        #encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        # q = apply_cached_rotary_emb(encoding, q)
        # k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))

# batchsize, c, m = 8, 256, 512
# n = 9 
# h = 4

# segment = torch.rand(batchsize, c, m)
# desc = torch.rand(batchsize, c, m)

# seg_tranformers = nn.ModuleList(
#             [SelfBlock(c, h) for _ in range(n)]
#         )
# desc_tranformers = nn.ModuleList(
#             [SelfBlock(c, h) for _ in range(n)]
#         )
# reparameterize_selfblock_layers(desc_tranformers, seg_tranformers)

# for i in range(n):
#     desc = desc_tranformers[i](desc)

# print("outputs", desc)


#---------------------------
# import json

# # 原始字符串
# cfg_str = '{"type": "GN", "num_groups": 32}'

# # 将字符串转换为字典
# cfg = json.loads(cfg_str)

# # 检查类型
# print(isinstance(cfg_str, dict))
# print(isinstance(cfg, dict))  # 这应该输出 True
# print(cfg)  # 这应该输出 {'type': 'GN', 'num_groups': 32}

#--------------------------------
import torch

# # 定义预训练权重文件的路径
# segment_path = "/data/zzj/glue-factory-main1/pretrained/segnext_tiny_512x512_ade_160k.pth"

# # 加载预训练权重文件
# pretrained_weights = torch.load(segment_path, map_location=torch.device('cpu'))

# # 查看权重文件中的内容
# # print(pretrained_weights.keys())
# # print(pretrained_weights['meta'])
# # print(pretrained_weights['state_dict'])
# # print(pretrained_weights['optimizer'])
# # 获取模型的状态字典
# state_dict = pretrained_weights['state_dict']

# # 打印权重参数的变量名字
# for name, param in state_dict.items():
#     print(name)

#---------------------------------------------------
# import torch

# # 假设 x 是一个形状为 (3, 4) 的二维张量
# x = torch.tensor([[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]])

# # 计算沿着列方向的均值
# mu = x.mean(-1, keepdim=True)

# print(mu)

# import mmseg
# print(mmseg.__version__)

#-------------------------------------------------
# import torch
# torch.manual_seed(0)

# x= torch.tensor([1., 2.], requires_grad=True)
# clone_x = x.clone() 
# detach_x = x.detach()
# clone_detach_x = x.clone().detach() 

# f = torch.nn.Linear(2, 1)
# y = f(x)
# y.backward()

# print(x.grad)
# print(clone_x.requires_grad)
# print(clone_x.grad)
# print(detach_x.requires_grad)
# print(clone_detach_x.requires_grad)

#------------------------------------------------
# target output size of 5
# m = nn.AdaptiveAvgPool1d(1)
# input = torch.randn(1, 64, 8)
# output = m(input)
# print(output.shape)

#-----------------------------------------------
import torch

# def get_covariance_matrix(x, eye=None):
#     eps = 1e-8
#     B, C, H, W = x.shape  # 获取输入的形状
#     HW = H * W  # 计算每个特征图的总元素数

#     # 如果未提供 eye 矩阵，创建一个 C x C 的单位矩阵
#     if eye is None:
#         eye = torch.eye(C, device=x.device, dtype=x.dtype) * eps

#     # 调整输入的形状为 B x C x (H * W)
#     x = x.contiguous().view(B, C, -1)

#     # 计算协方差矩阵，并添加一个小的单位矩阵以增强数值稳定性
#     x_cor = torch.bmm(x, x.transpose(1, 2)).div(HW - 1) + eye

#     return x_cor

# def zca_whitening(inputs):
#     B, C, H, W = inputs.shape
#     cor = get_covariance_matrix(inputs)
#     inputs = inputs.contiguous().view(B, C, -1)
    
#     U, S, V = torch.linalg.svd(cor)
    
#     inputs_dtype = inputs.dtype
#     S = S.to(inputs_dtype)
#     U = U.to(inputs_dtype)
    
#     epsilon = 0.1
#     S = 1.0 / torch.sqrt(S + epsilon)
#     SS = torch.diag_embed(S)
    
#     ZCAMatrix = torch.matmul(torch.matmul(U, SS), torch.transpose(U, dim0=1, dim1=2))
#     result = torch.matmul(ZCAMatrix, inputs)
#     result = result.contiguous().view(B, C, H, W)
    
#     return result

# 测试示例
# inputs = torch.randn(16, 3, 32, 32).cuda()  # 示例输入
# outputs = zca_whitening(inputs)
# print(outputs.shape)  # 输出形状应与输入形状相同

#-----------------------------------------------------------------------
#import tarfile

# # 定义tar文件的路径
# tar_path = "/data/zzj/glue-factory-main1/glue-factory-main/outputs/training/divide_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr20/checkpoint_best.tar"

# # 列出tar文件的内容
# def list_tar_contents(tar_path):
#     with tarfile.open(tar_path, 'r') as tar:
#         contents = tar.getnames()
#     return contents

# # 列出tar文件的内容
# try:
#     tar_contents = list_tar_contents(tar_path)
#     print(tar_contents)
# except FileNotFoundError as e:
#     print(f"文件未找到: {e}")

#--------------------------------------------------------------
import time
import torch
import torch.nn.functional as F

# def new_get_covariance_matrix(x, eye=None, eps=1e-8):
#     B, C, HW = x.shape
#     if eye is None:
#         eye = torch.eye(C, device=x.device, dtype=x.dtype) * eps
#     # x = x.view(B, C, -1)  # B X C X (H X W)
#     # 使用 torch.bmm 进行批量矩阵乘法，提高效率
#     x_cor = torch.bmm(x, x.transpose(1, 2)) / (HW - 1) + eye
#     return x_cor

# def new_zca_whitening(inputs):
#     B, C, HW = inputs.shape
#     cor = new_get_covariance_matrix(inputs)
    
#     U, S, V = torch.linalg.svd(cor)
    
#     inputs_dtype = inputs.dtype
#     S = S.to(inputs_dtype)
#     U = U.to(inputs_dtype)
    
#     epsilon = 0.1
#     S = 1.0 / torch.sqrt(S + epsilon)
#     SS = torch.diag_embed(S)
    
#     ZCAMatrix = torch.matmul(torch.matmul(U, SS), torch.transpose(U, dim0=1, dim1=2))
#     result = torch.matmul(ZCAMatrix, inputs)
    
#     return result

import torch
import torch.nn.functional as F

def get_covariance_matrix(x, eye=None, eps=1e-8):
    B, C, H, W = x.shape
    HW = H * W
    if eye is None:
        eye = torch.eye(C, device=x.device, dtype=x.dtype) * eps
    x = x.view(B, C, -1)  # B X C X (H X W)
    # 使用 torch.bmm 进行批量矩阵乘法，提高效率
    x_cor = torch.bmm(x, x.transpose(1, 2)) / (HW - 1) + eye
    return x_cor

def zca_whitening(inputs):
    B, C, H, W = inputs.shape
    cor = get_covariance_matrix(inputs)
    inputs = inputs.contiguous().view(B, C, -1)
    
    U, S, V = torch.linalg.svd(cor)
    
    inputs_dtype = inputs.dtype
    S = S.to(inputs_dtype)
    U = U.to(inputs_dtype)
    
    epsilon = 0.1
    S = 1.0 / torch.sqrt(S + epsilon)
    SS = torch.diag_embed(S)
    
    ZCAMatrix = torch.matmul(torch.matmul(U, SS), V.transpose(1, 2))
    result = torch.matmul(ZCAMatrix, inputs)
    result = result.contiguous().view(B, C, H, W)
    
    return result

def randomized_svd(A, n_components, n_iter=3):
    B, C, _ = A.shape
    P = torch.randn(B, C, n_components, device=A.device, dtype=A.dtype)  # 生成随机投影矩阵
    Z = torch.bmm(A, P)  # 随机投影
    for _ in range(n_iter):
        Z = torch.bmm(A, torch.bmm(A.transpose(1, 2), Z))  # 子空间迭代
    Q, _ = torch.linalg.qr(Z, mode='reduced')  # 使用新的QR分解函数
    B = torch.bmm(Q.transpose(1, 2), A)
    U_hat, S, V = torch.svd(B)  # 低维SVD
    U = torch.bmm(Q, U_hat)  # 映射回原始空间
    return U, S, V

def new_zca_whitening(inputs, n_components=None, n_iter=5):
    with torch.no_grad():  # 禁用梯度计算
        B, C, H, W = inputs.shape
        cor = get_covariance_matrix(inputs)
        inputs = inputs.view(B, C, -1)
        
        # 使用随机SVD
        if n_components is None:
            n_components = C  # 默认使用全部成分
        U, S, V = randomized_svd(cor, n_components=n_components, n_iter=n_iter)
        
        inputs_dtype = inputs.dtype
        S = S.to(inputs_dtype)
        U = U.to(inputs_dtype)
        
        epsilon = 0.1
        S = 1.0 / torch.sqrt(S + epsilon)
        SS = torch.diag_embed(S)
        
        ZCAMatrix = torch.matmul(torch.matmul(U, SS), V.transpose(1, 2))
        
        result = torch.matmul(ZCAMatrix, inputs)
        result = result.view(B, C, H, W)
        
    return result

def lowrank_zca_whitening(inputs, q):
    B, C, H, W = inputs.shape
    cor = get_covariance_matrix(inputs)
    inputs = inputs.contiguous().view(B, C, -1)
    
    # 使用 torch.svd_lowrank 计算 SVD
    U, S, V = torch.svd_lowrank(cor, q=C)  # q 参数设置为特征维度 C
    
    inputs_dtype = inputs.dtype
    S = S.to(inputs_dtype)
    U = U.to(inputs_dtype)
    
    epsilon = 0.1
    S = 1.0 / torch.sqrt(S + epsilon)
    SS = torch.diag_embed(S)
    
    ZCAMatrix = torch.matmul(torch.matmul(U, SS), V.transpose(1, 2))  # 注意这里使用 V 而不是 V^T
    result = torch.matmul(ZCAMatrix, inputs)
    result = result.contiguous().view(B, C, H, W)
    
    return result

# 确保你的数据在 GPU 上
# 确保数据在 GPU 上
# inputs = torch.randn(1, 480, 64, 64).cuda()

import time
import torch

# 预热
# for _ in range(5):
#     zca_whitening(inputs)
#     lowrank_zca_whitening(inputs=inputs, q=50)
#     new_zca_whitening(inputs, n_components=150)
    
# # 测试 new_zca_whitening
# new_times = []
# old_times = []
# lowrank_times = []
# for _ in range(10):  # 重复测试10次
#     inputs = torch.randn(1, 480, 64, 64).cuda()
#     torch.cuda.synchronize()
#     start_time = time.time()
#     result_old = zca_whitening(inputs)
#     torch.cuda.synchronize()
#     old_times.append(time.time() - start_time)
    
#     torch.cuda.synchronize()
#     start_time = time.time()
#     result_new = new_zca_whitening(inputs, n_components=150, n_iter=5)
#     torch.cuda.synchronize()
#     new_times.append(time.time() - start_time)
    
#     torch.cuda.synchronize()
#     start_time = time.time()
#     lowrank_result = lowrank_zca_whitening(inputs, q=480)
#     torch.cuda.synchronize()
#     lowrank_times.append(time.time() - start_time)

# 计算平均时间
# avg_new_time = sum(new_times) / len(new_times)
# avg_old_time = sum(old_times) / len(old_times)
# avg_lowrank_time = sum(lowrank_times) / len(lowrank_times)

# print(f"Average time for new_zca_whitening: {avg_new_time:.4f} seconds")
# print(f"Average time for zca_whitening: {avg_old_time:.4f} seconds")
# print(f"Average time for lowrank_zca_whitening: {avg_lowrank_time:.4f} seconds")

# # print("result_old", result_old)
# # print("result_new", result_new)

# # 检查结果是否一致
# if torch.allclose(result_new, result_old, atol=1):
#     print("new_Results are the same.")
# else:
#     print("new_Results differ.")
    
# if torch.allclose(lowrank_result, result_old, atol=1e-3):
#     print("new_Results are the same.")
# else:
#     print("new_Results differ.")

#--------------------------------------------------------------
import cv2
import numpy as np

# # 定义源图像和目标图像中的对应点
# src_points = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
# dst_points = np.array([[150, 150], [250, 150], [250, 250], [150, 250]], dtype=np.float32)

# # 计算单映射矩阵
# H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

# # 应用单映射矩阵进行透视变换
# img = cv2.imread('"/data/zzj/JSeg-main/image/69522173_2519789305.jpg"')
# height, width = img.shape[:2]
# warped_img = cv2.warpPerspective(img, H, (width, height))

# # 显示结果
# cv2.imshow('Warped Image', warped_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#----------------------------------
# import numpy as np
# import cv2
# # 准备数据
# points1 = np.array([[54.88135039, 71.51893664], [60.27633761, 54.4883183 ], [42.36547993, 64.58941131], [43.75872113, 89.17730008], [96.36627605, 38.34415188], [79.17250381, 52.88949198], [56.80445611, 92.55966383], [ 7.10360582, 8.71292997]], dtype=np.float32)
# points2 = np.array([[ 2.02183974, 83.26198455], [77.81567509, 87.00121482], [97.86183422, 79.91585642], [46.14793623, 78.05291763], [11.82744259, 63.99210213], [14.33532874, 94.4668917 ], [52.18483218, 41.466194 ], [26.45556121, 77.42336894]], dtype=np.float32)
# # 调用cv2.findFundamentalMat函数
# F, mask = cv2.findFundamentalMat(points1, points2, method=cv2.FM_8POINT)
# print("Fundamental Matrix F:")
# print(F)
# print("Mask:")
# print(mask)

#-------------------------------
import numpy as np
from PIL import Image

# def get_semantic_labels(image_path, mask_path, coords):
#     """
#     mask_path (str): 掩模图像路径。
#     coords (np.ndarray): 图像中的坐标数组，形状为 (m, 2)。

#     返回:
#     np.ndarray: 对应坐标的语义类别标签数组，形状为 (m, )。
#     """
#     # 加载掩模图像并转换为 NumPy 数组
#     mask = Image.open(mask_path).convert("L")
#     mask_np = np.array(mask)
    
#     # 检查坐标范围是否在图像尺寸内
#     h, w = mask_np.shape
#     valid_coords = (coords[:, 0] < w) & (coords[:, 1] < h) & (coords[:, 0] >= 0) & (coords[:, 1] >= 0)
    
#     if not np.all(valid_coords):
#         raise ValueError("一些坐标超出了图像的范围")
    
#     # 获取每个坐标的语义标签
#     labels = mask_np[coords[:, 1], coords[:, 0]]
    
#     return labels

# # 示例使用
# image_path = "/data/zzj/JSeg-main/image/fffff44226f3a3674a81d45b2ecce50.jpg"
# mask_path = "/data/zzj/JSeg-main/image/fffff44226f3a3674a81d45b2ecce50.jpg"
# coords = (100, 150)  # 示例坐标

# semantic_label = get_semantic_label(image_path, mask_path, coords)
# print(f"坐标 {coords} 的语义标签是: {semantic_label}")

        # cov_S = self.conv1(input_S)
        # input_R = input_R.permute(0,2,1)
        # cov_S = self.norm1(cov_S)#b c h w
        # kv = self.kv(x)
        # k, v = kv.chunk(2, dim=1)
        # q = self.q_dwconv(self.q(y))# b c h w
        
        # q = self.z_order_flatten(q)# b c hw
        # q = F.interpolate(q, size = m, mode="linear", align_corners=False)
        
        # q = rearrange(q, 'b (head c) m -> b head c m', head = self.num_heads)
        # k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        # v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)

        # out = (attn @ v)
        # out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        # out = self.project_out(out)
        
        
        # norm_S = self.S_norm1(input_S)
        # norm_R = self.R_fnorm1(input_R)
        # kv = self.kv(x)
        # k, v = kv.chunk(2, dim=1)
        # q = self.q(y)# b c m
        # q = rearrange(q, 'b (head c) n -> b head c n', head = self.num_heads)
        # k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        # v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        # out = (attn @ v)

        # out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        # out = self.project_out(out)

#---------------------------------------------------------
import numpy as np

# # 假设有以下数据
# # unique_labels 是一个 3xM 的数组，每列是一个三通道标签
# unique_labels = np.array([
#     [120, 120, 120],
#     [255, 0, 0],
#     [0, 255, 0],
#     [0, 0, 255]
# ])

# # 创建 label_to_int 字典
# label_to_int = {tuple(label): i for i, label in enumerate(unique_labels)}

# # 假设 labels0 是一个 3xN 的标签矩阵
# labels0 = np.array([
#     [120, 120, 120, 255, 0, 0],
#     [255, 0, 0, 0, 255, 0],
#     [0, 0, 255, 0, 0, 255]
# ])

# # 转置 labels0 使其每列表示一个三通道标签
# labels0 = labels0.T

# # 使用 label_to_int 字典映射三通道标签到整数标签
# labels0_to_int = np.array([label_to_int.get(tuple(label), -1) for label in labels0])

# print(labels0_to_int)

#--------------------------------------------------------
# import matplotlib.pyplot as plt
# import numpy as np

# def draw_snowflake():
#     fig, ax = plt.subplots()

#     # Define the points for the snowflake
#     angles = np.linspace(0, 2 * np.pi, 7)
#     x = np.cos(angles)
#     y = np.sin(angles)

#     # Draw the outer hexagon
#     ax.plot(x, y, color='blue', linewidth=3)

#     # Draw the inner lines
#     for i in range(6):
#         ax.plot([0, x[i]], [0, y[i]], color='blue', linewidth=3)

#     # Draw the inner branches
#     for i in range(6):
#         ax.plot([x[i] * 0.5, x[i] * 0.87], [y[i] * 0.5, y[i] * 0.87], color='blue', linewidth=3)
#         ax.plot([x[i] * 0.5, x[i] * 0.5 * np.cos(np.pi / 3) - y[i] * 0.5 * np.sin(np.pi / 3)],
#                 [y[i] * 0.5, x[i] * 0.5 * np.sin(np.pi / 3) + y[i] * 0.5 * np.cos(np.pi / 3)],
#                 color='blue', linewidth=3)

#     # Set the aspect of the plot to be equal
#     ax.set_aspect('equal')

#     # Remove the axes
#     ax.axis('off')
    
#     plt.savefig('snow.png')
#     # Show the plot
#     plt.show()

# # Call the function to draw the snowflake
# draw_snowflake()

#------------------------------------------------------------------
from PIL import Image
import numpy as np

def load_image(image_path):
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        return image_array
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# 示例路径
image_path = "/data/zzj/SuperGluePretrainedNetwork/results/yfcc_SP_DivideSGFM_plot_train_umap_n8/output_17298.png"
image_data = load_image(image_path)

if image_data is not None:
    print(f"Image loaded successfully with shape {image_data.shape}")
else:
    print("Failed to load image")
