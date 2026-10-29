#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from plyfile import PlyData, PlyElement
from typing import Tuple

# ===================== 用户参数 =====================

# 输入 / 输出 PLY
in_ply  = r"D:\Codex+Blender/smart.ply"
out_ply = r"D:\Codex+Blender/smart_three.ply"

# 是否保存估计出的变换矩阵
save_T_path = "T_three.npy"

# 尺度模式：
# 多数 3DGS / Gaussian Splatting 实现里 scale_x, scale_y, scale_z 存的是 log-scale
SCALE_MODE = "log"   # "log" 或 "linear"

# 四元数字段顺序：
# rot_0..3 常见可能是 wxyz，也可能是 xyzw
ROT_ORDER = "wxyz"   # "wxyz" 或 "xyzw"

# 是否估计缩放
WITH_SCALE = True

# ----------------- 指定对应点 -----------------
# src_pts[i] 对应 dst_pts[i]
src_pts = np.array([
    [-0.0535, -20.4504,  -0.4706],
    [5.1199, -20.0862, -5.1361],
    [-0.2975, -19.9903, -7.8794]
], dtype=np.float64)

dst_pts = np.array([
    [10.3450, -6.6030, -10.8660],
    [10.278, -6.572, -6.914],
    [13.513, -6.578, -8.092]
], dtype=np.float64)

# ==================================================


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """
    根据对应点估计相似变换:
        dst ≃ s * R * src + t

    参数
    ----
    src, dst : (N,3)
    with_scale : 是否估计统一缩放

    返回
    ----
    T : (4,4)
        齐次变换矩阵，满足：
        [dst,1]^T ≃ T @ [src,1]^T
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    N = src.shape[0]
    if N < 3:
        raise ValueError("至少需要 3 对对应点")

    # 1) 均值
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # 2) 去中心化
    src_centered = src - mu_src
    dst_centered = dst - mu_dst

    # 3) 协方差
    Sigma = (dst_centered.T @ src_centered) / N

    # 4) SVD
    U, D, Vt = np.linalg.svd(Sigma)

    # 5) 防止反射
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    # 6) 旋转
    R = U @ S @ Vt

    # 7) 缩放
    if with_scale:
        var_src = (src_centered ** 2).sum() / N
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0

    # 8) 平移
    t = mu_dst - s * R @ mu_src

    # 9) 齐次矩阵
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def decompose_uniform_RT(T: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    从 4x4 齐次矩阵中提取：
        R(3x3真旋转), s(均匀缩放>0), t(3,)

    要求线性部分近似为 s*R（无剪切、无非均匀缩放）。
    """
    A = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)

    # 列范数估计统一尺度
    sx = np.linalg.norm(A[:, 0])
    sy = np.linalg.norm(A[:, 1])
    sz = np.linalg.norm(A[:, 2])
    s = float((sx + sy + sz) / 3.0)

    rel_var = np.std([sx, sy, sz]) / max(1e-8, s)
    if rel_var > 1e-3:
        print(
            f"Warning: linear part not uniform-scale "
            f"(column norms {sx:.6g}, {sy:.6g}, {sz:.6g}); "
            f"std/mean={rel_var:.2e}. Proceeding with average s={s:.6g}.",
            file=sys.stderr
        )

    if s <= 0:
        raise ValueError("Uniform scale must be > 0.")

    R = A / s

    # 正交化，增强数值稳定性
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R.astype(np.float32), float(s), t.astype(np.float32)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """
    旋转矩阵 -> 四元数 (w,x,y,z)
    """
    m = R.astype(np.float64)
    tr = np.trace(m)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (m[2, 1] - m[1, 2]) / S
        y = (m[0, 2] - m[2, 0]) / S
        z = (m[1, 0] - m[0, 1]) / S
    else:
        if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / S
            x = 0.25 * S
            y = (m[0, 1] + m[1, 0]) / S
            z = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / S
            x = (m[0, 1] + m[1, 0]) / S
            y = 0.25 * S
            z = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / S
            x = (m[0, 2] + m[2, 0]) / S
            y = (m[1, 2] + m[2, 1]) / S
            z = 0.25 * S

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q.astype(np.float32)


def quat_multiply_wxyz(q2: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """
    Hamilton 乘法：
        q = q2 ⊗ q1
    输入格式均为 (w,x,y,z)

    支持 shape=(N,4)
    """
    w2, x2, y2, z2 = q2.T
    w1, x1, y1, z1 = q1.T

    w = w2*w1 - x2*x1 - y2*y1 - z2*z1
    x = w2*x1 + x2*w1 + y2*z1 - z2*y1
    y = w2*y1 - x2*z1 + y2*w1 + z2*x1
    z = w2*z1 + x2*y1 - y2*x1 + z2*w1

    return np.stack([w, x, y, z], axis=1)


def normalize_quat_array(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    return q / n


def main():
    # =====================================================
    # Part 1: 用指定三点/多点估计变换矩阵 T
    # =====================================================
    if src_pts.shape != dst_pts.shape or src_pts.shape[1] != 3:
        raise ValueError("src_pts 和 dst_pts 必须形状一致，且为 (N,3)")
    if src_pts.shape[0] < 3:
        raise ValueError("至少需要 3 对对应点")

    T = umeyama(src_pts, dst_pts, with_scale=WITH_SCALE)

    print("Estimated transform T =")
    print(T)

    # if save_T_path:
    #     np.save(save_T_path, T)
    #     print(f"Saved transform matrix to: {save_T_path}")

    # 简单验证
    src_h = np.hstack([src_pts, np.ones((src_pts.shape[0], 1), dtype=np.float64)])
    dst_est = (T @ src_h.T).T[:, :3]
    err = np.linalg.norm(dst_est - dst_pts, axis=1)

    print("\nVerification:")
    print("Transformed src_pts =")
    print(dst_est)
    print("Target dst_pts =")
    print(dst_pts)
    print("Per-point error =")
    print(err)
    print(f"Mean error = {err.mean():.6g}")

    # =====================================================
    # Part 2: 将 T 应用到 3DGS PLY
    # =====================================================
    R_T, s_uniform, t_T = decompose_uniform_RT(T)
    print(f"\nDecomposed transform:")
    print(f"Uniform scale s = {s_uniform:.6g}")
    print("Rotation R =")
    print(R_T)
    print("Translation t =")
    print(t_T)

    # 读 PLY
    ply = PlyData.read(in_ply)
    v = ply['vertex'].data
    out = v.copy()

    # ---------- 位置更新 ----------
    # p' = s * R * p + t
    P = np.stack([v['x'], v['y'], v['z']], axis=1).astype(np.float32)
    P_tr = (P @ R_T.T) * s_uniform + t_T

    out['x'] = P_tr[:, 0]
    out['y'] = P_tr[:, 1]
    out['z'] = P_tr[:, 2]

    # ---------- 旋转更新 ----------
    # 若整体变换包含旋转，则每个高斯局部旋转也要左乘该旋转
    rot_err = np.linalg.norm(R_T - np.eye(3, dtype=np.float32))
    rot_fields = {'rot_0', 'rot_1', 'rot_2', 'rot_3'}

    if rot_err > 1e-6 and rot_fields.issubset(v.dtype.names):
        q_T = rotmat_to_quat_wxyz(R_T)
        q_T = np.broadcast_to(q_T, (len(v), 4)).astype(np.float32)

        q_in = np.stack(
            [v['rot_0'], v['rot_1'], v['rot_2'], v['rot_3']],
            axis=1
        ).astype(np.float32)

        if ROT_ORDER == "xyzw":
            q_in = q_in[:, [3, 0, 1, 2]]

        q_out = quat_multiply_wxyz(q_T, q_in)
        q_out = normalize_quat_array(q_out)

        if ROT_ORDER == "xyzw":
            q_write = q_out[:, [1, 2, 3, 0]]
        else:
            q_write = q_out

        out['rot_0'] = q_write[:, 0]
        out['rot_1'] = q_write[:, 1]
        out['rot_2'] = q_write[:, 2]
        out['rot_3'] = q_write[:, 3]

        print("Updated quaternion rotations.")
    else:
        print("No rotation update applied (either identity rotation or rot_* fields missing).")

    # ---------- 尺度更新 ----------
    scale_fields = {'scale_0', 'scale_1', 'scale_2'}
    if scale_fields.issubset(v.dtype.names):
        s0 = v['scale_0'].astype(np.float32)
        s1 = v['scale_1'].astype(np.float32)
        s2 = v['scale_2'].astype(np.float32)

        if SCALE_MODE == "log":
            add = np.float32(np.log(max(1e-12, s_uniform)))
            out['scale_0'] = s0 + add
            out['scale_1'] = s1 + add
            out['scale_2'] = s2 + add
        elif SCALE_MODE == "linear":
            mul = np.float32(s_uniform)
            out['scale_0'] = s0 * mul
            out['scale_1'] = s1 * mul
            out['scale_2'] = s2 * mul
        else:
            raise ValueError("SCALE_MODE must be 'log' or 'linear'.")

        print("Updated Gaussian scales.")
    else:
        print("scale_* fields not found, skipped scale update.")

    # 写回
    el = PlyElement.describe(out, 'vertex', comments=ply['vertex'].comments)
    PlyData([el], text=ply.text).write(out_ply)

    print(f"\nSaved transformed 3DGS PLY to: {out_ply}")


if __name__ == "__main__":
    main()