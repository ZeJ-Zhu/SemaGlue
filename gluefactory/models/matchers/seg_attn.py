import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import numbers
from einops import rearrange
from .utils import SpatialGate,SE_Block,CBAM,ChannelAttention
## Layer Norm

def to_3d(x):
    b, c, h, w = x.shape
    result = x.view(b, c, -1)
    result = result.permute(0, 2, 1)
    return result
    #return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x,h,w):
    b, m, c = x.shape
    x = x.permute(0, 2, 1)
    assert m == h*w
    result = x.view(b, c, h, w)
    return result
    #return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        #�������ͽ���ת��Ϊ����һ��Ԫ�ص�Ԫ�� (normalized_shape,)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)#���һ��ά�ȵķ���
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
        #return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


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
        #x b m c
        #x = x.permute(0,2,1) 
        x = self.body(x)# b m c
        x = x.permute(0,2,1)
        return x  #b,c,m

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.ffn = nn.Sequential(#�����������Բ��ǰ�������磬���м侭��LayerNorm��GELU�����
            nn.Linear(dim , 2 * hidden_features),
            nn.LayerNorm(2 * hidden_features, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * hidden_features, dim),
        )

        # self.project_in = nn.Conv1d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # self.dwconv = nn.Conv1d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
        #                         groups=hidden_features * 2, bias=bias)

        # self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):#b c m 
        x = x.permute(0,2,1)
        x = self.ffn(x)
        # x = self.project_in(x) # b c*5 m
        # x1, x2 = self.dwconv(x).chunk(2, dim=1)
        # x = F.gelu(x1) * x2
        # x = self.project_out(x)
        x = x.permute(0,2,1)
        return x

class ln(nn.Module):
    def __init__(self, h, w, desclength=512) -> None:
        super(ln, self).__init__()
        self.linear = nn.Linear(h*w, desclength, bias= True)
    
    def forward(self, q):
        return self.linear(q)
##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
        
class Attention1(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention1, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, h, w = y.shape

        kv = self.kv_dwconv(self.kv(x))
        #chunk����������kv��ά��dim=1�Ϸָ����������k��v
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        l = ln(h, w, m).to(q.device)
        q = l(q)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)
        #k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class Seg_TransformerBlock(nn.Module):
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention1(dim, num_heads, bias)
        self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_R:b m c  
         # Ŀ���СΪԭ�����ķ�֮һ
        target_size = (input_S.size(2) // 4, input_S.size(3) // 4)
        # ʹ�� interpolate �������н��߶Ȳ���
        input_S = F.interpolate(input_S, size=target_size, mode='bilinear', align_corners=False)
        #input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.fnorm1(input_R)# input b m c  out: b c m
        input_S = self.norm1(input_S)#b c h w
        # Rlength = input_R.shape[2]
        # Slength = input_S.shape[2]*input_S.shape[3]
        input_R = input_R + self.attn(input_R, input_S)#b c m
        input_R = input_R.permute(0,2,1)#b m c
        input_R_ffn = self.ffn(self.norm2(input_R))# b c m
        input_R = input_R + input_R_ffn.transpose(1,2)

        return input_R#b m c
    
##########################################################################
class TransformerBlock_1(nn.Module):
    def __init__(self, dim_2, dim, dim_in, num_heads=2, ffn_expansion_factor=1, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock_1, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim_in, (1, 1))
        self.conv2 = nn.Conv2d(dim, dim_in, (1, 1))
        self.conv3 = nn.Conv2d(dim_in, dim, (1, 1))
        self.norm1 = LayerNorm(dim_in, LayerNorm_type)
        self.attn = Attention1(dim_in, num_heads, bias)
        self.norm2 = LayerNorm(dim_in, LayerNorm_type)
        self.ffn = FeedForward(dim_in, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_ch = input_R.size()[1]
        input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        input_R = self.conv2(input_R)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.norm1(input_R)
        input_S = self.norm1(input_S)
        input_R = input_R + self.attn(input_R, input_S)
        input_R = input_R + self.ffn(self.norm2(input_R))
        input_R = self.conv3(input_R)

        return input_R
    
class TransformerBlock_2(nn.Module):
    #dim_2 seg输入channel dim cat之后维度 
    def __init__(self, dim_2, dim, dim_in, num_heads=2, ffn_expansion_factor=1, bias=False, LayerNorm_type="WithBias"):
        super(TransformerBlock_2, self).__init__()
        
        self.linear1 = nn.Linear(dim+dim_2, dim_in)
        self.linear2 = nn.Linear(dim_in, dim+dim_2)
        self.conv1 = nn.Conv2d(dim_2, dim_in, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim_in, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim_in, LayerNorm_type)
        self.attn = Attention1(dim_in, num_heads, bias)
        #self.norm2 = f_LayerNorm(dim_in, LayerNorm_type)
        #self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm3 = f_LayerNorm(dim, LayerNorm_type)
        self.norm4 = f_LayerNorm(dim_2, LayerNorm_type)
        self.ffn_R = FeedForward(dim, ffn_expansion_factor, bias)
        self.ffn_Si = FeedForward(dim_2, ffn_expansion_factor, bias)
    # input_R:b m c
    def forward(self, input_R, input_S, input_Si):
        c1 = input_R.shape[2]
        c2 = input_Si.shape[2]
        input_R = torch.cat((input_R, input_Si), 2)
        input_R = self.linear1(input_R)
         # 目标大小为原来的四分之一
        target_size = (input_S.size(2) // 4, input_S.size(3) // 4)
        # 使用 interpolate 函数进行降尺度采样
        input_S = F.interpolate(input_S, size=target_size, mode='bilinear', align_corners=False)
        #input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.fnorm1(input_R)# input b m c  out: b c m
        input_S = self.norm1(input_S)#b c h w
        # Rlength = input_R.shape[2]
        # Slength = input_S.shape[2]*input_S.shape[3]
        input_R = input_R + self.attn(input_R, input_S)#b c m
        input_R = input_R.permute(0,2,1)#b m c
        input_R = self.linear2(input_R)
        out_R, out_Si = torch.split(input_R,[c1,c2],2)
        out_R_ffn = self.ffn_R(self.norm3(out_R))
        out_Si_ffn = self.ffn_Si(self.norm4(out_Si))
        out_R = out_R + out_R_ffn.transpose(1,2)
        out_Si = out_Si + out_Si_ffn.transpose(1,2)
        #input_R_ffn = self.ffn(self.norm2(input_R))# b c m
        #input_R = input_R + input_R_ffn.transpose(1,2)
        return out_R, out_Si
        #return input_R#b m c

class Seg_TransformerBlock1(nn.Module):#待测试
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock1, self).__init__()

        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention2(dim, num_heads, bias)
        self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_R:b m c  
         # Ŀ���СΪԭ�����ķ�֮һ
        #target_size = input_R.shape[1] **0.5
        original_size = input_R.shape[1]
        target_size = (input_S.shape[2]//4) * (input_S.shape[3]//4)
        S_target_size = (input_S.shape[2]//4,input_S.shape[3]//4)
        R_permute = input_R.permute(0,2,1)
        #target_size = (input_S.size(2) // 4, input_S.size(3) // 4)
        # ʹ�� interpolate �������н��߶Ȳ���
        input_R = F.interpolate(R_permute, size = target_size, mode="linear", align_corners=False)
        input_R = input_R.permute(0,2,1)
        #print(input_R.shape)
        input_S = F.interpolate(input_S, size = S_target_size, mode="bilinear",align_corners=False)
        #input_S = F.interpolate(input_S, size=target_size, mode='bilinear', align_corners=False)
        #input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.fnorm1(input_R)# input b m c  out: b c m
        input_S = self.norm1(input_S)#b c h w
        # Rlength = input_R.shape[2]
        # Slength = input_S.shape[2]*input_S.shape[3]
        message = self.attn(input_R, input_S)
        input_R = input_R + message#b c m
        input_R = F.interpolate(input_R, size = original_size, mode="linear",align_corners=False)
        input_R = input_R.permute(0,2,1)#b m c
        input_R_ffn = self.ffn(self.norm2(input_R))# b c m
        input_R = input_R + input_R_ffn.transpose(1,2)

        return input_R#b m c
class Attention3(nn.Module):#待测试
    def __init__(self, dim, max_num_keypoints, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention3, self).__init__()
        self.max_num_keypoints = max_num_keypoints
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, h, w = y.shape

        kv = self.kv_dwconv(self.kv(x))
        #chunk����������kv��ά��dim=1�Ϸָ����������k��v
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w

        #q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b c h w -> b c (h w)')
        q = F.interpolate(q, size = self.max_num_keypoints, mode="linear", align_corners=False)
        q = rearrange(q, 'b (head c) m -> b head c m', head = self.num_heads)
        # l = ln(h, w, m).to(q.device)
        # q = l(q)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)
        #k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Seg_TransformerBlock3(nn.Module):
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, max_num_keypoints, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock3, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention3(dim, max_num_keypoints, num_heads, bias)
        self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_R:b m c  
         # Ŀ���СΪԭ�����ķ�֮һ
        target_size = (input_S.size(2) // 4, input_S.size(3) // 4)
        # ʹ�� interpolate �������н��߶Ȳ���
        input_S = F.interpolate(input_S, size=target_size, mode='bilinear', align_corners=False)
        #input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.fnorm1(input_R)# input b m c  out: b c m
        input_S = self.norm1(input_S)#b c h w
        # Rlength = input_R.shape[2]
        # Slength = input_S.shape[2]*input_S.shape[3]
        input_R = input_R + self.attn(input_R, input_S)#b c m
        input_R = input_R.permute(0,2,1)#b m c
        input_R_ffn = self.ffn(self.norm2(input_R))# b c m
        input_R = input_R + input_R_ffn.transpose(1,2)

        return input_R#b m c
class Seg_TransformerBlock4(nn.Module):
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, max_num_keypoints, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock4, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention3(dim, max_num_keypoints, num_heads, bias)
        #self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        #self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, input_R, input_S):
        # input_R:b m c  
         # Ŀ���СΪԭ�����ķ�֮һ
        # target_size = (input_S.size(2) // 4, input_S.size(3) // 4)
        # # ʹ�� interpolate �������н��߶Ȳ���
        # input_S = F.interpolate(input_S, size=target_size, mode='bilinear', align_corners=False)
        #input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
        input_S = self.conv1(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.fnorm1(input_R)# input b m c  out: b c m
        input_S = self.norm1(input_S)#b c h w
        # Rlength = input_R.shape[2]
        # Slength = input_S.shape[2]*input_S.shape[3]
        input_S = self.attn(input_R,input_S)#b c m
        
        # input_R = input_R + self.attn(input_R, input_S)#b c m
        # input_R = input_R.permute(0,2,1)#b m c
        # input_R_ffn = self.ffn(self.norm2(input_R))# b c m
        # input_R = input_R + input_R_ffn.transpose(1,2)

        return input_S.permute(0,2,1)#b m c
    
# class Seg_TransformerBlock5(nn.Module):#withnews_22
#     #dim_2 seg����channel   dim:�м�ͳһchannel
#     def __init__(self, dim_2, dim, max_num_keypoints, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
#         super(Seg_TransformerBlock5, self).__init__()
#         self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
#         # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
#         self.norm1 = LayerNorm(dim, LayerNorm_type)
#         self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
#         self.attn = Attention3(dim, max_num_keypoints, num_heads, bias)
#         self.norm2 = f_LayerNorm(dim, LayerNorm_type)
#         #self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

#     def forward(self, input_R, input_S):
#         # input_R:b m c  
#          # Ŀ���СΪԭ�����ķ�֮һ
#         target_size = (input_S.size(2) // 4, input_S.size(3) // 4)
#         # ʹ�� interpolate �������н��߶Ȳ���
#         input_S = F.interpolate(input_S, size=target_size, mode='bilinear', align_corners=False)
#         #input_S = F.interpolate(input_S, [input_R.shape[2], input_R.shape[3]])
#         input_S = self.conv1(input_S)
#         # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
#         input_R = self.fnorm1(input_R)# input b m c  out: b c m
#         input_S = self.norm1(input_S)#b c h w
#         # Rlength = input_R.shape[2]
#         # Slength = input_S.shape[2]*input_S.shape[3]
#         input_S = self.attn(input_R,input_S)#b c m
        
#         # input_R = input_R + self.attn(input_R, input_S)#b c m
#         # input_R = input_R.permute(0,2,1)#b m c
#         # input_R_ffn = self.ffn(self.norm2(input_R))# b c m
#         # input_R = input_R + input_R_ffn.transpose(1,2)

#         return input_S.permute(0,2,1)#b m c
class Attention4(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention4, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, n = y.shape

        kv = self.kv_dwconv(self.kv(x))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))# b c m
        q = rearrange(q, 'b (head c) n -> b head c n', head = self.num_heads)
        # l = ln(h, w, m).to(q.device)
        # q = l(q)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)
        #k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Seg_TransformerBlock5(nn.Module):#withnews_21
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock5, self).__init__()
        #self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.fnorm2 = f_LayerNorm(dim_2,LayerNorm_type)
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        #self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention4(dim, num_heads, bias)
        #self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.fnorm2(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        input_R = self.fnorm1(input_R)# input b m c  out: b c m
        message_R = self.attn(input_R,input_S)#b c m
        input_R = input_R.permute(0,2,1)
        message_R = message_R.permute(0,2,1)
        R = input_R + self.ffn(torch.cat([input_R, message_R], -1))

        return R#b m c
    
class Seg_TransformerBlock8(nn.Module):#withnews_27
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock8, self).__init__()
        #self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.fnorm2 = f_LayerNorm(dim,LayerNorm_type)
        # self.conv2 = nn.Conv2d(dim, dim_2, (1, 1))
        #self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention4(dim, num_heads, bias)
        #self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        norm_S = self.fnorm2(input_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        message_R = self.attn(norm_S,norm_R)#b c m
        #input_R = input_R.permute(0,2,1)
        message_R = message_R.permute(0,2,1)
        R = input_R + self.ffn(torch.cat([input_R, message_R], -1))

        return R#b m c
    
class Attention5(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention5, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        #self.kv_dwconv = nn.Conv1d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        #self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, desc, seg):
        b, c, m = desc.shape# b c m
        b, c, m = seg.shape

        kv = self.kv(desc)
        k, v = kv.chunk(2, dim=1)
        q = self.q(seg)
        
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
    
class GetSurroundingRegion1(nn.Module):
    def __init__(self,seg_dim,strd=9):
        super(GetSurroundingRegion1, self).__init__()
        self.stride = strd
        self.weight_matrix = nn.Parameter(torch.randn(9))  # 定义可学习的权重矩阵
        self.pool = ChannelAttention(seg_dim)
        self.Conv = nn.Conv2d(9,1,kernel_size=1,stride=1)
        self.linear = nn.Linear(seg_dim, 1,bias=True)
        
    def expand_dimension(self,wx, c,dim=2):
        wx = torch.unsqueeze(wx, dim)
        wx = wx.repeat(1,1,c,1)
        return wx
    
    def forward(self, original_kpts, segment):
        batchsize, c, h, w = segment.shape
        _, m, _ = original_kpts.shape
        x = original_kpts[:, :, 0].permute(1, 0)
        y = original_kpts[:, :, 1].permute(1, 0)

        x = x * h
        y = y * w
        x = x.floor().to(torch.int64)
        y = y.floor().to(torch.int64)
        # 定义周围像素的相对位置
        offsets = torch.tensor([-1*self.stride, 0, self.stride], dtype=torch.int64, device=segment.device)
        surrounding_indices = [(x + dx, y + dy) for dx in offsets for dy in offsets]
        # 转换为一维索引
        indices = [x * w + y for x, y in surrounding_indices]
        surrounding_indices_stack = torch.stack(indices, dim=0)
        surrounding_indices_stack = self.expand_dimension(surrounding_indices_stack, c,2)# 9 m c b
        surrounding_indices_stack = torch.clamp(surrounding_indices_stack, min = 0, max = h*w-1)
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
        #s = self.expand_dimension1(s,m)
        surrounding_s =self.linear(surrounding_pixels[4].permute(2,0,1))
        confidence = torch.nn.functional.normalize(surrounding_s,dim=1)#b m 1
        result = (1-confidence)*pool + surrounding_pixels[4].permute(2,0,1)+ s
        return result
    
class Seg_TransformerBlock6(nn.Module):
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock6, self).__init__()
        self.patch = GetSurroundingRegion1(seg_dim)
        self.linear = nn.Linear(seg_dim,dim,bias)
        self.fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention5(dim, num_heads, bias)
        self.attn1 = Attention5(dim, num_heads, bias)
        #self.norm2 = f_LayerNorm(dim, LayerNorm_type)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S,keypoints):
        feature_S = self.patch(keypoints,input_S)#b m c
        feature_S = self.linear(feature_S) 
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        norm_S = self.fnorm2(feature_S)#b c m 
        message_S = self.attn(norm_R, norm_S)#b c m
        message_R = self.attn1(message_S,norm_R)
        message_R = message_R.permute(0,2,1)#b m c
        input_R_ffn = self.ffn(torch.cat([input_R, message_R], -1))
        input_R = input_R + input_R_ffn

        return input_R#b m c
    
class Seg_TransformerBlock7(nn.Module):
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, seg_dim, dim, max_num_keypoints,num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock7, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention4(dim, num_heads, bias)
        self.attn1 = Attention3(dim,max_num_keypoints,num_heads,bias)
        self.attn2 = Attention4(dim,num_heads,bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        S =self.conv1(input_S)
        S = self.norm1(S)
        R = self.fnorm1(input_R)#bcm
        message_S = self.attn1(R,S)
        message_S1 = self.attn(R,message_S)
        # input_S = F.interpolate(input_S, size=input_size, mode='bilinear', align_corners=True)
        message_R = self.attn2(message_S1,R)
        message_R = message_R.permute(0,2,1)
        R = input_R + self.ffn(torch.cat([input_R, message_R], -1))
        return R#b m c


#----------------------------------------------------
class NZ_OrderAttn(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(NZ_OrderAttn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
    
    def forward(self, x, y):
        b, c, m = x.shape
        b, c, h, w = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w
        
        q = rearrange(q, 'b c h w -> b c (h w)')
        #q = self.z_order_flatten(q)# b c hw
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

class ZOrder_DConv_attn(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(ZOrder_DConv_attn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.v_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def z_order_flatten(self, image_tensor):
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
    
    def forward(self, x, y):
        b, c, m = x.shape
        b, c, h, w = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        k = self.k_dwconv(k)
        v = self.v_dwconv(v)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w

        #q = rearrange(q, 'b c h w -> b c (h w)')
        q = self.z_order_flatten(q)# b c hw
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
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Attention7(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention7, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def z_order_flatten(self, image_tensor):
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
    
    def forward(self, x, y):
        b, c, m = x.shape
        b, c, h, w = y.shape

        kv = self.kv(x)
        # if self.training:
        #     for parm in self.kv.parameters():
        #         print(parm)
        #     print(self.training)
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w
        #q = self.q(y)
        
        #q = rearrange(q, 'b c h w -> b c (h w)')
        q = self.z_order_flatten(q)# b c hw
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

class DConv_attn(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(DConv_attn, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.k_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.v_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1,groups=dim, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, n = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        k = self.k_dwconv(k)
        v = self.v_dwconv(v)
        
        q = self.q_dwconv(self.q(y))# b c m
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
    
class Attention8(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention8, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, n = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(y))# b c m
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
    
class Seg_TransformerBlock9(nn.Module):#withnews_30
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock9, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
        
        norm3_S = self.S_fnorm3(S).permute(0,2,1)
        #norm_R = norm_R.permute(0,2,1)
        #R = norm_R + self.ffn1(torch.cat([norm_R,norm3_S], -1))
        return norm3_S#b m c
    
class Seg_TransformerBlock11(nn.Module):#withnews_33
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock11, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        #S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
        
        norm3_S = self.S_fnorm3(message2_S).permute(0,2,1)
        norm_R = norm_R.permute(0,2,1)
        R = norm_R + self.ffn1(torch.cat([norm_R,norm3_S], -1))
        return R

class Seg_TransformerBlock13(nn.Module):#21_ffn_nfnorm
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock13, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        #self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
    
        norm_R = norm_R.permute(0,2,1)
        R = norm_R + self.ffn1(torch.cat([norm_R,S], -1))
        return R
    
class Seg_TransformerBlock14(nn.Module):#21_ffn_nfnorm_nzorder
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock14, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = NZ_OrderAttn(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        #self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
    
        norm_R = norm_R.permute(0,2,1)
        R = norm_R + self.ffn1(torch.cat([norm_R,S], -1))
        return R

class Seg_TransformerBlock15(nn.Module):#21_ffn_nfnorm_DConv
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock15, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = ZOrder_DConv_attn(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        #self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = DConv_attn(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
    
        norm_R = norm_R.permute(0,2,1)
        R = norm_R + self.ffn1(torch.cat([norm_R,S], -1))
        return R

class Seg_TransformerBlock16(nn.Module):#21_ffn_nfnorm_cposextract/21_ffn_nfnorm_c_lr15/21_ffn_nfnorm_c_lr15_factor0.9_EncoderCovRelu_loadfalse
    #21_ffn_nfnorm_c_lr15_factor0.9_DecoderCovRelu_loadfalse/21_ffn_nfnorm_c_lr15_factor0.9_b64_newSEncoder_loadfalse/2divide_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr20
    #21_ffn_nfnorm_c_lr15_factor0.8_end30_DecoderCovRelu_loadfalse
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock16, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        #self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
    
        return S
       
class Seg_TransformerBlock12(nn.Module):#withnews_34  在30上的变体取消ffn
    def __init__(self, dim_2, dim, num_heads=4, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock12, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)

    def forward(self, input_R, input_S):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        
        return message2_S
   
#-----------------------------------------------position extract 
class Attention9(nn.Module):#withnews_29 
    def __init__(self, dim, num_heads, bias, strd=1):#, seglength=256, desclength=512):
        super(Attention9, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        #--------------
        self.stride = strd
        self.weight_matrix = nn.Parameter(torch.randn(9))  # 定义可学习的权重矩阵
        self.pool = ChannelAttention(dim)
        self.Conv = nn.Conv2d(9,1,kernel_size=1,stride=1)
        self.linear = nn.Linear(dim, 1,bias=True)
    
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
        # 定义周围像素的相对位置
        offsets = torch.tensor([-1*self.stride, 0, self.stride], dtype=torch.int64, device=segment.device)
        surrounding_indices = [(x + dx, y + dy) for dx in offsets for dy in offsets]
        # 转换为一维索引
        indices = [x * w + y for x, y in surrounding_indices]
        surrounding_indices_stack = torch.stack(indices, dim=0)
        surrounding_indices_stack = self.expand_dimension(surrounding_indices_stack, c,2)# 9 m c b
        surrounding_indices_stack = torch.clamp(surrounding_indices_stack, min = 0, max = h*w-1)
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
        #s = self.expand_dimension1(s,m)
        surrounding_s =self.linear(surrounding_pixels[4].permute(2,0,1))
        confidence = torch.nn.functional.normalize(surrounding_s,dim=1)#b m 1
        result = (1-confidence)*pool + surrounding_pixels[4].permute(2,0,1)+ s
        return result
    
    def forward(self, x, y, original_kpts):
        b, c, m = x.shape
        b, c, h, w = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w

        #q = rearrange(q, 'b c h w -> b c (h w)')
        q = self.get_Patchsi(original_kpts, q).permute(0,2,1)# b c m
        
        q = rearrange(q, 'b (head c) m -> b head c m', head = self.num_heads)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Seg_TransformerBlock10(nn.Module):#withnews_29
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock10, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention9(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        self.ffn1 = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, input_R, input_S, original_kpts):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S, original_kpts)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        message2_S = self.attn1(norm_R,norm2_S)#b c m
        message2_S = message2_S.permute(0,2,1)
        S = message_S + self.ffn(torch.cat([message_S, message2_S], -1))
        
        norm3_S = self.S_fnorm3(S).permute(0,2,1)
        norm_R = norm_R.permute(0,2,1)
        R = norm_R + self.ffn1(torch.cat([norm_R,norm3_S], -1))
        return R#b m c
    
class Seg_TransformerBlock17(nn.Module):#posextract_channel/posextract_channel_withencoding
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock17, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
    
    def expand_dimension(self, wx, c):
        wx = torch.unsqueeze(wx, dim=2)
        wx = wx.repeat(1,1,c)
        return wx
    
    def get_Si(self, original_kpts,segment):
        batchsize,c,h,w = segment.shape
        _, m, _ = original_kpts.shape   #b=3 m=512  _=2

        x = original_kpts[:, :, 0].permute(1, 0)
        y = original_kpts[:, :, 1].permute(1, 0)

        x = x * h
        y = y * w
        x = x.floor().to(torch.int64)
        y = y.floor().to(torch.int64)

        #扩展维度c
        x = self.expand_dimension(x, c).transpose(1,2)
        y = self.expand_dimension(y, c).transpose(1,2) #(m,c,b)

        # 将位置点坐标转换成一维索引
        # 注意：这里假设特征图的布局是 (height, width, channels, batchsize)
        # 如果是其他布局，需要调整下标的顺序
        indices = x * w + y
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

    def forward(self, input_R, input_S, original_kpts):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        pose_S = self.get_Si(original_kpts, input_S).permute(0,2,1)
        message2_S = self.attn1(norm2_S, pose_S)#b c m
        message2_S = message2_S.permute(0,2,1)
    
        # norm3_S = self.S_fnorm3(message2_S).permute(0,2,1)
        # norm_R = norm_R.permute(0,2,1)
        # R = norm_R + self.ffn(torch.cat([norm_R,norm3_S], -1))
        return message2_S
    
class Seg_TransformerBlock19(nn.Module):#posextract_channel_withencoding_newSEncoder_lr20
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock19, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        #self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        #self.linear = nn.Linear(dim_2, dim)
        self.attn1 = Attention8(dim, num_heads, bias)
        # self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
        #     nn.Linear(2 * dim, 2 * dim),
        #     nn.LayerNorm(2 * dim, elementwise_affine=True),
        #     nn.GELU(),
        #     nn.Linear(2 * dim, dim),
        # )
    
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

    def forward(self, input_R, input_S, original_kpts):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        pose_S = self.get_Si(original_kpts, input_S).permute(0,2,1)
        message2_S = self.attn1(norm2_S, pose_S)#b c m
        message2_S = message2_S.permute(0,2,1)
    
        return message2_S
    
class Seg_TransformerBlock18(nn.Module):#posextract_channel_cffn_EncoderCovRelu_lr15
    #dim_2 seg����channel   dim:�м�ͳһchannel
    def __init__(self, dim_2, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock18, self).__init__()
        self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.S_fnorm2 = f_LayerNorm(dim, LayerNorm_type)
        self.S_fnorm3 = f_LayerNorm(dim, LayerNorm_type)
        self.attn1 = Attention8(dim, num_heads, bias)
        self.ffn = nn.Sequential(#包含两个线性层的前馈神经网络，其中间经过LayerNorm和GELU激活函数
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
    
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

    def forward(self, input_R, input_S, original_kpts):
        input_S = self.conv1(input_S)
        norm_S = self.norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        message_S = self.attn(norm_R,norm_S)#b c m
        #input_R = input_R.permute(0,2,1)
        message_S = message_S.permute(0,2,1)#b m c
        
        norm2_S = self.S_fnorm2(message_S)
        pose_S = self.get_Si(original_kpts, input_S).permute(0,2,1)
        message2_S = self.attn1(norm2_S, pose_S)#b c m
        message2_S = message2_S.permute(0,2,1)
    
        # norm3_S = self.S_fnorm3(message2_S).permute(0,2,1)
        norm_R = norm_R.permute(0,2,1)
        R = norm_R + self.ffn(torch.cat([norm_R,message2_S], -1))
        return R

#-------------------------------------------------------------
#单独的模块
class Attention10(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention10, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        #self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, n = y.shape

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
        #out = self.inner_attn(q, k, v)
        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        
        out = self.project_out(out)
        return out
    
class Seg_TransformerBlock20(nn.Module):#testb16_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr20
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock20, self).__init__()
        #self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.S_norm1 = f_LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention10(dim, num_heads, bias)

    def forward(self, input_R, input_S):
        norm_S = self.S_norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        
        message_S = self.attn(norm_R, norm_S)#b c m
        message_S = message_S.permute(0,2,1)
    
        return message_S

class Seg_TransformerBlock25(nn.Module):#ccfuse
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock25, self).__init__()
        self.S_norm1 = f_LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention10(dim, num_heads, bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )
        
    def forward(self, input_R, input_S):
        norm_S = self.S_norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        
        message_R = self.attn(norm_S, norm_R)#b c m
        message_R = message_R.permute(0,2,1)
        R = input_R + self.ffn(torch.cat([input_R, message_R], -1))
        return R
#----------------------------------------------------flash attention
from typing import Optional

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

class Attention13(nn.Module):#divide_21_ffn_nfnorm_npre_flash_NewSEncoder_loadfalse_lr20
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention13, self).__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        #self.q_dwconv = nn.Conv1d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        self.inner_attn = Attention(allow_flash=False)

    def forward(self, x, y):
        b, c, m = x.shape# b c m
        b, c, n = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        q = self.q(y)# b c m
        q = rearrange(q, 'b (head c) n -> b head c n', head = self.num_heads)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)
        out = self.inner_attn(q, k, v)
        # out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)

        out = self.project_out(out)
        return out

class Seg_TransformerBlock21(nn.Module):#divide_21_ffn_nfnorm_npre_flash_newSEncoder_loadfalse_lr20
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_TransformerBlock21, self).__init__()
        #self.conv1 = nn.Conv2d(dim_2, dim, (1, 1))
        self.S_norm1 = f_LayerNorm(dim, LayerNorm_type)
        self.R_fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention13(dim, num_heads, bias)

    def forward(self, input_R, input_S):
        norm_S = self.S_norm1(input_S)
        norm_R = self.R_fnorm1(input_R)# input b m c  out: b c m
        
        message_S = self.attn(norm_R, norm_S)#b c m
        message_S = message_S.permute(0,2,1)
    
        return message_S

#_-------------------------------------------------------
class Posextract_Channel(nn.Module):#testb16_posextract_channel_withencoding_newSEncoder_loadfalse_lr20
    def __init__(self, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Posextract_Channel, self).__init__()
        # self.linear = nn.Linear(seg_embed_dim, dim, bias=bias)
        self.S_norm1 = f_LayerNorm(dim, LayerNorm_type)
        self.kS_norm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention10(dim, num_heads, bias)
    
    def forward(self, local_S, global_S):
        norm_S = self.S_norm1(global_S)
        norm_kS = self.kS_norm1(local_S)# input b m c  out: b c m
        
        message_S = self.attn(norm_S, norm_kS)#b c m
        message_S = message_S.permute(0,2,1)
    
        return message_S#b m c

#-------------------------------------------Seg_FeatureProcessor
class Seg_FeatureProcessor(nn.Module):#testb16_posextract_channel_withencoding_newSEncoder_loadfalse_lr20/divide_posextract_channel_withencoding_newSEncoder_loadfalse_lr20/divide_posextract_channel_cat_newSEncoder_loadfalse_lr20
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        self.linear = nn.Linear(seg_dim, dim, bias=bias)
        
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
        
        return result.permute(0,2,1)  # 返回结果，形状为 (b, m, c)
    
    def forward(self, input_R, input_S, original_kpts):
        cov_S = self.conv1(input_S)
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        cov_S = self.norm1(cov_S)#b c h w

        global_S = self.attn(norm_R, cov_S).permute(0,2,1)#b m c
        local_S = self.get_Si(original_kpts, input_S)#b m c 
        local_S = self.linear(local_S)
        
        return global_S, local_S

class Seg_FeatureProcessor2(nn.Module):#divide_21_ffn_nfnorm_c_fuseselfcross_newSEncoder_loadfalse_lr20
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor2, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        
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
        
        return result.permute(0,2,1)  # 返回结果，形状为 (b, m, c)
    
    def forward(self, input_R, input_S, original_kpts):
        cov_S = self.conv1(input_S)
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        cov_S = self.norm1(cov_S)#b c h w

        global_S = self.attn(norm_R, cov_S).permute(0,2,1)#b m c
        local_S = self.get_Si(original_kpts, cov_S)#b m c 
        
        return global_S, local_S

#---------------------------------------------------------
class Attention14(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention14, self).__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        #self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)
        #self.inner_attn = Attention(allow_flash = False)

    def z_order_flatten(self, image_tensor):
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
    
    def forward(self, x, y):
        b, c, m = x.shape
        b, c, h, w = y.shape

        #kv = self.kv(x)
        #k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w
        #q = self.q(y)

        #q = rearrange(q, 'b c h w -> b c (h w)')
        q = self.z_order_flatten(q)# b c hw
        q = F.interpolate(q, size = m, mode="linear", align_corners=False)

        out = self.project_out(q)
        return out
    
class Seg_FeatureProcessor1(nn.Module):#testb16_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr20/divide_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr20/divide_21_ffn_nfnorm_c_pool_newSEncoder_loadfalse_lr20/divide_21_ffn_nfnorm_c_poolinput_newSEncoder_loadfalse_lr20
    #divide_21_ffn_nfnorm_c_newSEncoder_loadfalse_lr15/divide_21_ffn_nfnorm_c_newSEncoder_loadfalse_b64_lr20
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor1, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention7(dim, num_heads, bias)
        #self.attn = Attention14(dim, num_heads, bias)
    
    def forward(self, input_R, input_S):
        cov_S = self.conv1(input_S)
        # if self.training:
        #     for parm in self.conv1.parameters():
        #         print(parm)
        #     print(self.training)
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        cov_S = self.norm1(cov_S)#b c h w

        global_S = self.attn(norm_R, cov_S).permute(0,2,1)#b m c

        return global_S

class Seg_FeatureProcessor5(nn.Module):
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor5, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        #self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        #self.attn = Attention7(dim, num_heads, bias)
        self.attn = Attention14(dim, num_heads, bias)
    
    def forward(self, input_R, input_S):
        cov_S = self.conv1(input_S)
        #norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        input_R = input_R.permute(0,2,1)
        cov_S = self.norm1(cov_S)#b c h w

        global_S = self.attn(input_R, cov_S).permute(0,2,1)#b m c
        
        return global_S

#-----------------------
class Seg_FeatureProcessor6(nn.Module):#divide_21_ffn_nfnorm_ncc_all_NewSEncoder_loadfalse_lr20
    def __init__(self, seg_dim, dim):
        super(Seg_FeatureProcessor6, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
    
    def forward(self, input_S):
        cov_S = self.conv1(input_S)
        b, c, _, _ = cov_S.shape
        #一阶段
        # cov_S = F.interpolate(cov_S, size=(cov_S.shape[2] // 8, cov_S.shape[3] // 8), mode='bilinear', align_corners=False)
        #二阶段
        cov_S = F.interpolate(cov_S, size=(cov_S.shape[2] // 16, cov_S.shape[3] // 16), mode='bilinear', align_corners=False)
        cov_S = cov_S.view(b,c,-1).permute(0,2,1)
        return cov_S
#--------------------------------------------------
def get_covariance_matrix(x, eye=None):
    eps = 1e-8
    B, C, H, W = x.shape
    HW = H * W
    if eye is None:
        eye = torch.eye(C, device=x.device, dtype=x.dtype) * eps
    x = x.view(B, C, -1)
    x_cor = torch.bmm(x, x.transpose(1, 2)).div(HW - 1) + eye
    return x_cor

def zca_whitening(inputs):
    with torch.no_grad():  # 禁用梯度计算
        B, C, H, W = inputs.shape
        cor = get_covariance_matrix(inputs)
        inputs = inputs.view(B, C, -1)
        
        S, U = torch.linalg.eigh(cor)
        
        inputs_dtype = inputs.dtype
        S = S.to(inputs_dtype)
        U = U.to(inputs_dtype)
        
        epsilon = 0.1
        S = 1.0 / torch.sqrt(S + epsilon)
        SS = torch.diag_embed(S)
        
        ZCAMatrix = torch.matmul(torch.matmul(U, SS), U.transpose(1, 2))
        
        result = torch.matmul(ZCAMatrix, inputs)
        result = result.view(B, C, H, W)
        
    return result

class Attention11(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention11, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def z_order_flatten(self, image_tensor):
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
    
    def forward(self, x, y):
        b, c, m = x.shape
        b, c, h, w = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w

        #q = rearrange(q, 'b c h w -> b c (h w)')
        q = self.z_order_flatten(q)# b c hw
        q = F.interpolate(q, size = m, mode="linear", align_corners=False)
        q = zca_whitening(q)
        
        q = rearrange(q, 'b (head c) m -> b head c m', head = self.num_heads)
        k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class Seg_FeatureProcessor3(nn.Module):#divide_21_ffn_nfnorm_c_zcawhiteningpro_newSEncoder_loadfalse_lr20
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor3, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention11(dim, num_heads, bias)
    
    def forward(self, input_R, input_S):
        cov_S = self.conv1(input_S)
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        cov_S = self.norm1(cov_S)#b c h w

        global_S = self.attn(norm_R, cov_S).permute(0,2,1)#b m c
        
        return global_S

#-------------------------------------------------------------------
class Attention12(nn.Module):
    def __init__(self, dim, num_heads, bias):#, seglength=256, desclength=512):
        super(Attention12, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.linear = nn.Linear(seglength, desclength, bias = True)
        self.kv = nn.Conv1d(dim, dim * 2, kernel_size=1, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv1d(dim, dim, kernel_size=1, bias=bias)

    def z_order_flatten(self, image_tensor):
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
    
    def forward(self, x, y):
        b, c, m = x.shape
        b, c, h, w = y.shape

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=1)
        #y = self.linear(y)
        q = self.q_dwconv(self.q(y))# b c h w

        #q = rearrange(q, 'b c h w -> b c (h w)')
        q = self.z_order_flatten(q)# b c hw
        q = F.interpolate(q, size = m, mode="linear", align_corners=False)
        
        # q = rearrange(q, 'b (head c) m -> b head c m', head = self.num_heads)
        # k = rearrange(k, 'b (head c) m -> b head c m', head = self.num_heads)
        # v = rearrange(v, 'b (head c) m -> b head c m', head = self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)

        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        # attn = attn.softmax(dim=-1)

        # out = (attn @ v)

        # out = rearrange(out, 'b head c m -> b (head c) m', head = self.num_heads)
        # #out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        # out = self.project_out(out)
        return q
    
class Seg_FeatureProcessor4(nn.Module):#divide_21_ffn_nfnorm_npre_flash_newSEncoder_loadfalse_lr20
    def __init__(self, seg_dim, dim, num_heads=2, ffn_expansion_factor=1, bias=True, LayerNorm_type='WithBias'):
        super(Seg_FeatureProcessor4, self).__init__()
        self.conv1 = nn.Conv2d(seg_dim, dim, (1, 1))
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.fnorm1 = f_LayerNorm(dim, LayerNorm_type)
        self.attn = Attention12(dim, num_heads, bias)
    
    def forward(self, input_R, input_S):
        cov_S = self.conv1(input_S)
        norm_R = self.fnorm1(input_R)# input b m c  out: b c m
        cov_S = self.norm1(cov_S)#b c h w

        global_S = self.attn(norm_R, cov_S).permute(0,2,1)#b m c
        
        return global_S
