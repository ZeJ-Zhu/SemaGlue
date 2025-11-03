import torch

from .utils import skew_symmetric, to_homogeneous
from .wrappers import Camera, Pose
# 这段代码主要涉及相机几何学和相对位姿误差计算的一些函数。以下是主要函数的功能：
#
# T_to_E 函数： 将位姿矩阵 T 转换为对应的本质矩阵 E。
#
# T_to_F 函数： 根据两个相机 cam0 和 cam1 以及它们之间的位姿变换 T_0to1，计算基础矩阵 F。
#
# E_to_F 函数： 根据两个相机 cam0 和 cam1 以及本质矩阵 E，计算基础矩阵 F。
#
# F_to_E 函数： 根据两个相机 cam0 和 cam1 以及基础矩阵 F，计算本质矩阵 E。
#
# sym_epipolar_distance 函数： 计算两组点集 p0 和 p1 之间的对称极线距离，使用本质矩阵 E。
#
# sym_epipolar_distance_all 函数： 计算两组点集 p0 和 p1 之间的对称极线距离，使用基础矩阵 F。
#
# generalized_epi_dist 函数： 根据输入的参数，计算两组关键点之间的对称极线距离，可以选择使用本质矩阵 E 或基础矩阵 F。
#
# decompose_essential_matrix 函数： 对本质矩阵 E 进行分解，得到旋转矩阵 R1、R2 和平移向量 T。
#
# angle_error_mat 函数： 计算两个旋转矩阵 R1 和 R2 之间的角度误差。
#
# angle_error_vec 函数： 计算两个向量之间的角度误差。
#
# relative_pose_error 函数： 计算相对位姿误差，包括平移和旋转的误差。
def T_to_E(T: Pose):
    """Convert batched poses (..., 4, 4) to batched essential matrices."""
    return skew_symmetric(T.t) @ T.R


def T_to_F(cam0: Camera, cam1: Camera, T_0to1: Pose):
    return E_to_F(cam0, cam1, T_to_E(T_0to1))


def E_to_F(cam0: Camera, cam1: Camera, E: torch.Tensor):
    assert cam0._data.shape[-1] == 6, "only pinhole cameras supported"
    assert cam1._data.shape[-1] == 6, "only pinhole cameras supported"
    K0 = cam0.calibration_matrix()
    K1 = cam1.calibration_matrix()
    return K1.inverse().transpose(-1, -2) @ E @ K0.inverse()


def F_to_E(cam0: Camera, cam1: Camera, F: torch.Tensor):
    assert cam0._data.shape[-1] == 6, "only pinhole cameras supported"
    assert cam1._data.shape[-1] == 6, "only pinhole cameras supported"
    K0 = cam0.calibration_matrix()
    K1 = cam1.calibration_matrix()
    return K1.transpose(-1, -2) @ F @ K0


def sym_epipolar_distance(p0, p1, E, squared=True):
    """Compute batched symmetric epipolar distances.
    Args:
        p0, p1: batched tensors of N 2D points of size (..., N, 2).
        E: essential matrices from camera 0 to camera 1, size (..., 3, 3).
    Returns:
        The symmetric epipolar distance of each point-pair: (..., N).
    """
    assert p0.shape[-2] == p1.shape[-2]
    if p0.shape[-2] == 0:
        return torch.zeros(p0.shape[:-1]).to(p0)
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum("...ni,...ij,...nj->...n", p1, E, p0)
    E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
    Et_p1 = torch.einsum("...ij,...ni->...nj", E, p1)
    d0 = (E_p0[..., 0] ** 2 + E_p0[..., 1] ** 2).clamp(min=1e-6)
    d1 = (Et_p1[..., 0] ** 2 + Et_p1[..., 1] ** 2).clamp(min=1e-6)
    if squared:
        d = p1_E_p0**2 * (1 / d0 + 1 / d1)
    else:
        d = p1_E_p0.abs() * (1 / d0.sqrt() + 1 / d1.sqrt()) / 2
    return d


def sym_epipolar_distance_all(p0, p1, E, eps=1e-15):
    if p0.shape[-1] != 3:
        p0 = to_homogeneous(p0)
    if p1.shape[-1] != 3:
        p1 = to_homogeneous(p1)
    p1_E_p0 = torch.einsum("...mi,...ij,...nj->...nm", p1, E, p0).abs()
    E_p0 = torch.einsum("...ij,...nj->...ni", E, p0)
    Et_p1 = torch.einsum("...ij,...mi->...mj", E, p1)
    d0 = p1_E_p0 / (E_p0[..., None, 0] ** 2 + E_p0[..., None, 1] ** 2 + eps).sqrt()
    d1 = (
        p1_E_p0
        / (Et_p1[..., None, :, 0] ** 2 + Et_p1[..., None, :, 1] ** 2 + eps).sqrt()
    )
    return (d0 + d1) / 2


def generalized_epi_dist(
    kpts0, kpts1, cam0: Camera, cam1: Camera, T_0to1: Pose, all=True, essential=True
):
    #cam 相机参数
    #如果essential为True，则计算基础矩阵E，调用'sym_epipolar_distance或all'计算广义极线距离
    if essential:
        E = T_to_E(T_0to1)
        p0 = cam0.image2cam(kpts0)
        p1 = cam1.image2cam(kpts1)
        if all:
            return sym_epipolar_distance_all(p0, p1, E, agg="max")
        else:
            return sym_epipolar_distance(p0, p1, E, squared=False)
    #如果为False，则假设相机参数中包含相机内参，计算基础矩阵F
    #并同样调用 sym_epipolar_distance_all 或 sym_epipolar_distance 函数计算广义极线距离
    else:
        assert cam0._data.shape[-1] == 6
        assert cam1._data.shape[-1] == 6
        K0, K1 = cam0.calibration_matrix(), cam1.calibration_matrix()
        F = K1.inverse().transpose(-1, -2) @ T_to_E(T_0to1) @ K0.inverse() #E=K 
        if all:
            return sym_epipolar_distance_all(kpts0, kpts1, F)
        else:
            return sym_epipolar_distance(kpts0, kpts1, F, squared=False)


def decompose_essential_matrix(E):
    # decompose matrix by its singular values
    U, _, V = torch.svd(E)
    Vt = V.transpose(-2, -1)

    mask = torch.ones_like(E)
    mask[..., -1:] *= -1.0  # fill last column with negative values

    maskt = mask.transpose(-2, -1)

    # avoid singularities
    U = torch.where((torch.det(U) < 0.0)[..., None, None], U * mask, U)
    Vt = torch.where((torch.det(Vt) < 0.0)[..., None, None], Vt * maskt, Vt)

    W = skew_symmetric(E.new_tensor([[0, 0, 1]]))
    W[..., 2, 2] += 1.0

    # reconstruct rotations and retrieve translation vector
    U_W_Vt = U @ W @ Vt
    U_Wt_Vt = U @ W.transpose(-2, -1) @ Vt

    # return values
    R1 = U_W_Vt
    R2 = U_Wt_Vt
    T = U[..., -1]
    return R1, R2, T


# pose errors
# TODO: test for batched data
def angle_error_mat(R1, R2):
    cos = (torch.trace(torch.einsum("...ij, ...jk -> ...ik", R1.T, R2)) - 1) / 2
    cos = torch.clip(cos, -1.0, 1.0)  # numerical errors can make it out of bounds
    return torch.rad2deg(torch.abs(torch.arccos(cos)))


def angle_error_vec(v1, v2, eps=1e-10):
    n = torch.clip(v1.norm(dim=-1) * v2.norm(dim=-1), min=eps)
    v1v2 = (v1 * v2).sum(dim=-1)  # dot product in the last dimension
    return torch.rad2deg(torch.arccos(torch.clip(v1v2 / n, -1.0, 1.0)))


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0, eps=1e-10):
    if isinstance(T_0to1, torch.Tensor):
        R_gt, t_gt = T_0to1[:3, :3], T_0to1[:3, 3]
    else:
        R_gt, t_gt = T_0to1.R, T_0to1.t
    R_gt, t_gt = torch.squeeze(R_gt), torch.squeeze(t_gt)

    # angle error between 2 vectors
    t_err = angle_error_vec(t, t_gt, eps)
    t_err = torch.minimum(t_err, 180 - t_err)  # handle E ambiguity
    if t_gt.norm() < ignore_gt_t_thr:  # pure rotation is challenging
        t_err = 0

    # angle error between 2 rotation matrices
    r_err = angle_error_mat(R, R_gt)

    return t_err, r_err
