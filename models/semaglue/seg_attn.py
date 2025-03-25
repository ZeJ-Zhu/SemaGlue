import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numbers
from einops import rearrange

## Layer Norm

def to_3d(x):
    b, c, h, w = x.shape
    result = x.view(b, c, -1)
    result = result.permute(0, 2, 1)
    return result

def to_4d(x,h,w):
    b, m, c = x.shape
    x = x.permute(0, 2, 1)
    assert m == h*w
    result = x.view(b, c, h, w)
    return result

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        result = (x-mu) / torch.sqrt(sigma+1e-5)
        result = result* self.weight + self.bias
        return result

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class f_LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(f_LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        x = self.body(x)
        x = x.permute(0,2,1)
        return x 
    
class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def s_order_flatten(self, image_tensor):
        b, c, h, w = image_tensor.size()
        flattened = image_tensor.view(b, c, -1)

        mask = torch.arange(h * w).view(h, w)
        mask[1::2, :] = mask[1::2, :].flip(1)
        mask = mask.view(-1)

        flattened_z_order = flattened[:, :, mask]

        return flattened_z_order
    
    def forward(self, x, y):
        _, _, m = x.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))

        q = self.s_order_flatten(q)
        q = F.interpolate(q, size = m, mode="linear", align_corners=False)
        
        q = rearrange(q, 'b (head c) m -> b head c m', head = self.num_heads)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)

        out = self.project_out(out)
        return out
    
class Seg_FeatureProcessor(nn.Module):
    def __init__(self, seg_dim, dim, num_heads=2, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention1(dim, num_heads, bias)
    
    def forward(self, input_R, input_S):
        cov_S = self.conv1(input_S)
        norm_R = self.fnorm1(input_R)
        cov_S = self.norm1(cov_S)

        global_S = self.attn(norm_R, cov_S).permute(0,2,1)
        
        return global_S
    
class CDA_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CDA_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        q = self.q(y)# b c m
        q = rearrange(q, 'b (head c) n -> b head c n', head = self.num_heads)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)

        out = self.project_out(out)
        return out
    
class CDA(nn.Module):
    def __init__(self, dim, num_heads=2, bias=True, LayerNorm_type='WithBias'):
        super(CDA, self).__init__()
        self.S_norm1 = f_LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = CDA_Attention(dim, num_heads, bias)

    def forward(self, input_R, input_S):
        norm_S = self.S_norm1(input_S)
        norm_R = self.R_fnorm1(input_R)
        
        message_S = self.attn(norm_R, norm_S)
        message_S = message_S.permute(0,2,1)
    
        return message_S