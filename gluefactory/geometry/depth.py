import kornia
import torch

from .utils import get_image_coords
from .wrappers import Camera
# “这段代码似乎是深度图的配准操作，包含了一些几何变换的计算。以下是这段代码的主要步骤：
#
# sample_fmap 函数： 对于给定的点集 pts 和特征图 fmap，使用双线性插值或最近邻插值，从特征图中采样对应的值。
#
# sample_depth 函数： 对于给定的点集 pts 和深度图 depth_，使用 sample_fmap 函数从深度图中采样深度值，并返回有效性掩码。
#
# sample_normals_from_depth 函数： 对于给定的点集 pts、深度图 depth 和相机内参 K，使用深度图计算法线，并使用 sample_fmap 函数从法线图中采样法向量，并返回有效性掩码。
#
# project 函数： 将点集 kpi 从相机 camera_i 投影到三维空间，然后经过变换 T_itoj 到相机 camera_j 的坐标系中，最后再投影回图像平面。同时，进行一些有效性掩码的计算，以及圆一致性（circle consistency）的检查。
#
# dense_warp_consistency 函数： 给定两个深度图 depthi 和 depthj，以及它们之间的变换 T_itoj，使用 project 函数对 depthi 进行配准操作，返回配准后的坐标和有效性掩码。
#
# 这些函数的目的似乎是进行深度图之间的几何配准，并在此过程中考虑了一些几何变换和有效性的处理”
def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1)[:, None]
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, depth_.new_tensor(float("nan")))
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    validi,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}

    kpi_3d_i = camera_i.image2cam(kpi)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    kpi_j, validj = camera_j.cam2image(kpi_3d_j)
    # di_j = kpi_3d_j[..., -1]
    validi = validi & validj
    if depthj is None or ccth is None:
        return kpi_j, validi & validj
    else:
        # circle consistency
        dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
        kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
        kpi_j_i, validj_i = camera_i.cam2image(T_itoj.inv().transform(kpi_j_3d_j))
        consistent = ((kpi - kpi_j_i) ** 2).sum(-1) < ccth
        visible = validi & consistent & validj_i & validj
        # visible = validi
        return kpi_j, visible


def dense_warp_consistency(
    depthi: torch.Tensor,
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: Camera,
    cameraj: Camera,
    **kwargs,
):
    kpi = get_image_coords(depthi).flatten(-3, -2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir = project(kpi, di, depthj, camerai, cameraj, T_itoj, validi, **kwargs)

    return kpir.unflatten(-2, depthi.shape[-2:]), validir.unflatten(
        -1, (depthj.shape[-2:])
    )
