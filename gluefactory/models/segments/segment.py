#%%
# import yaml, math, os
# with open('config.yaml') as fh:
#     config = yaml.load(fh, Loader=yaml.FullLoader)#指定加载器，将config.yaml文件加载到config中
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from gluefactory.models.base_model import BaseModel
import gc
from .bricks import resize
from .mscan import MSCAN
# from backbone import MSCANet
from .decoder import HamDecoder

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
        
        S, U, V = torch.linalg.svd(cor)
        
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
    
class SegNext(BaseModel):
    default_conf={
        "in_channels" : 3,#?
        "embed_dims"  : [32,64,160,256],  #?
        "ffn_ratios"  : [8,8,4,4], #?
        "depths"       : [3,3,5,2], #?
        "num_stages"  : 4,#?
        "dec_outChannels" : 256, #?
        "dropout"     : 0.1 ,#?
        "drop_path"   : 0.0, #?
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        # 将 norm_cfg 转换为字典
        norm_cfg_dict = dict(type=conf.norm_typ)
        self.backbone = MSCAN(in_chans=conf.in_channels, embed_dims=conf.embed_dims,
                            mlp_ratios=conf.ffn_ratios, depths=conf.depths, num_stages=conf.num_stages,
                            drop_rate=conf.drop_path, drop_path_rate=conf.dropout, norm_cfg=norm_cfg_dict)
        segment_path="/data/zzj/glue-factory-main1/pretrained/segnext_tiny_512x512_ade_160k.pth"
        state_dict = torch.load(segment_path, map_location=torch.device('cpu'))
        backbone_state_dict = {k: v for k, v in state_dict['state_dict'].items() if k.startswith('backbone')}
        # Load the state_dict into your model
        self.load_state_dict(backbone_state_dict, strict=True)

    def _forward(self, data):
        pred = {}
        image = data["image"]
        # 第一阶段
        features = self.backbone(image)
        # dec_out = self.decoder(enc_feats)
        features = features[1:] # drop stage 1 features b/c low level 从 features 中丢弃第一个元素（stage 1 特征），因为它是低级特征
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]#将 features 中的每个特征调整大小以匹配最后一个特征的空间尺寸
        x = torch.cat(features, dim=1)
        #dec_out = zca_whitening(dec_out)
        
        # 第二阶段
        # original_height = image.size()[-2] #获取原始图像的大小,
        # original_weight = image.size()[-1]
        # transform = transforms.Resize((original_height//2, original_weight//2), antialias=True)
        # image = transform(image)
        
        # #image = F.interpolate(image,size=(512,512),mode="bilinear",align_corners=True)

        # # if image.shape[1] == 3:  # RGB
        # #     scale = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        # #     image = (image * scale).sum(1, keepdim=True)#rgb变灰

        # enc_feats = self.encoder(image)
        # dec_out = self.decoder(enc_feats)
        #output = self.cls_conv(dec_out)  # here output will be B x 144 x H/8 x W/8
        
        #-----------------------
        pred = {
            "feature_map": x,#dec_out b*256*h/8*w/8
            #"segment_map": output
        }

        return pred
    
    def loss(self, pred, data):
        raise NotImplementedError

    def metrics(self, pred, data):
        raise NotImplementedError

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def main():
    # 获取当前文件的父目录并将其添加到 sys.path
    current_dir = Path(__file__).resolve().parent
    sys.path.append(str(current_dir))

    # 初始化模型配置
    num_classes = 150  # 例如，ADE20K数据集有150个类别
    conf = {
        "in_channels": 3,
        "embed_dims": [32, 64, 160, 256],
        "ffn_ratios": [8, 8, 4, 4],
        "depths": [3, 3, 5, 2],
        "ham_channels": 256,
        "num_stages": 4,
        # "dec_outChannels": 256,
        "dropout": 0.1,
        "drop_path": 0.0,
        "norm_typ": "SyncBN"  # 假设使用BatchNorm
    }

    # 初始化模型
    model = SegNext(conf=conf)

    # 准备输入图像
    input_image = Image.open('/data/zzj/glue-factory-main2/gluefactory/models/segments/frame_000002.jpg')

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)  # 为批次添加维度

    # 将模型设置为评估模式
    model.eval()

    # 使用模型处理图像
    with torch.no_grad():
        output = model._forward({"image": input_tensor})["feature_map"]

    # 可视化输出
    output_image = output.squeeze(0).cpu().numpy()  # 去掉批次维度
    plt.imshow(output_image, cmap='viridis')  # 用 'viridis' 色图显示
    plt.show()
    
if __name__ == '__main__':
    main()

# class SegNext_original(BaseModel):
#     default_conf={
#         "in_channels" : 3,#?
#         "embed_dims"  : [32,64,160,256],  #?
#         "ffn_ratios"  : [4,4,4,4], #?
#         "depths"       : [3,3,5,2], #?
#         "num_stages"  : 4,#?
#         "dec_outChannels" : 256, #?
#         "dropout"     : 0.1 ,#?
#         "drop_path"   : 0.0, #?
#     }

#     required_data_keys = ["image"]

#     def _init(self, conf):
#         #输入dec_outChannels 输出num_classes
#         #self.cls_conv = nn.Sequential(nn.Dropout2d(p=0.1),
#         #                              nn.Conv2d(conf.dec_outChannels, conf.num_classes, kernel_size=1))#dropout=0.1，在训练过程中随机丢弃输入的一部分元素，以防止过拟合
#         # self.encoder = MSCANet(in_channnels=conf.in_channels, embed_dims=conf.embed_dims,
#         #                        ffn_ratios=conf.ffn_ratios, depths=conf.depths, num_stages=conf.num_stages,
#         #                        drop_path=conf.drop_path, norm_type=conf.norm_typ)
        
#         # 将 norm_cfg 转换为字典
#         norm_cfg_dict = dict(type=conf.norm_typ)
#         self.backbone = MSCAN(in_chans=conf.in_channels, embed_dims=conf.embed_dims,
#                             mlp_ratios=conf.ffn_ratios, depths=conf.depths, num_stages=conf.num_stages,
#                             drop_rate=conf.drop_path, drop_path_rate=conf.dropout, norm_cfg=norm_cfg_dict)
#         # self.decoder = HamDecoder(
#         #     outChannels=conf.dec_outChannels, config=conf, enc_embed_dims=conf.embed_dims)
#         #self.init_weights()
#         segment_path="/data/zzj/glue-factory-main1/pretrained/segnext_tiny_512x512_ade_160k.pth"
#         state_dict = torch.load(segment_path, map_location=torch.device('cpu'))
#         # backbone_state_dict = {k: v for k, v in state_dict['state_dict'].items() if k.startswith('backbone')}
#         # Load the state_dict into your model
#         self.load_state_dict(state_dict, strict=False)

#     def _forward(self, data):
#         pred = {}
#         image = data["image"]
#         # 第一阶段
#         features = self.backbone(image)
#         # dec_out = self.decoder(enc_feats)
#         features = features[1:] # drop stage 1 features b/c low level 从 features 中丢弃第一个元素（stage 1 特征），因为它是低级特征
#         features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]#将 features 中的每个特征调整大小以匹配最后一个特征的空间尺寸
#         x = torch.cat(features, dim=1)
#         #dec_out = zca_whitening(dec_out)
        
#         #-----------------------
#         pred = {
#             "feature_map": x,#dec_out b*256*h/8*w/8
#             #"segment_map": output
#         }

#         return pred
    
#     def loss(self, pred, data):
#         raise NotImplementedError

#     def metrics(self, pred, data):
#         raise NotImplementedError