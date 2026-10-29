#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import numpy as np
import laspy
import open3d as o3d


# ===================== 用户参数 =====================

# 输入 / 输出 LAS
input_las = r"E:\ShareData\Process\Project_2026-07-29_16-52-50-MEP-RegionA-Final/Region_A_20260803.las"
output_las = r"E:\ShareData\Process\Project_2026-07-29_16-52-50-MEP-RegionA-Final/Region_A_20260803_standard.las"

# 输出变换矩阵
output_T = "T_align.npy"

# ---------- Step 1: 给定当前场景中的“竖直方向” vz ----------
# 它会被对齐到标准 +Z = [0,0,1]
vz = np.array([-0.00387795,-0.0118074,0.999923], dtype=np.float64)

# 是否强制把 vz 变成朝上
force_z_up = True

# ---------- Step 2: 平面内朝向 ----------
# "x" 表示给 vx，对齐到标准 +X
# "y" 表示给 vy，对齐到标准 +Y
heading_mode = "x"

# 当 heading_mode == "x" 时使用
vx = np.array([0.99604,0.0877049,0.0145705], dtype=np.float64)

# 当 heading_mode == "y" 时使用
vy = np.array([0.0207572, 0.999781, 0.00269602], dtype=np.float64)

# ---------- 旋转中心 ----------
# "centroid" / "origin" / "custom"
center_mode = "custom"
custom_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)

# ---------- 可视化 ----------
visualize = False
vis_max_points = 200000

# ==================================================


# ===================== 基础函数 =====================

def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero-length vector.")
    return v / n


def random_downsample(points, max_points):
    n = len(points)
    if n <= max_points:
        return points
    idx = np.random.choice(n, max_points, replace=False)
    return points[idx]


def skew(v):
    x, y, z = v
    return np.array([
        [0.0, -z,  y],
        [z,   0.0, -x],
        [-y,  x,   0.0]
    ], dtype=np.float64)


def rotation_from_a_to_b(a, b):
    """
    返回 R, 使得 R @ a = b
    """
    a = normalize(a)
    b = normalize(b)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)

    if s < 1e-12:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        else:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(a, tmp)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = normalize(np.cross(a, tmp))
            K = skew(axis)
            return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    K = skew(v)
    R = np.eye(3, dtype=np.float64) + K + (K @ K) * ((1.0 - c) / (s ** 2))
    return R


def rotz(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)


def build_4x4_from_R_center(R, center):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return T


def transform_points(points, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


# ===================== 两步旋转 =====================

def build_two_step_rotation(vz, heading_mode="x", vx=None, vy=None, force_z_up=True):
    z_std = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    x_std = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    y_std = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    vz_use = vz.astype(np.float64).copy()
    if force_z_up and vz_use[2] < 0:
        vz_use = -vz_use

    # Step 1: vz -> +Z
    R1 = rotation_from_a_to_b(vz_use, z_std)

    debug = {
        "vz_input": vz.astype(np.float64),
        "vz_used": vz_use,
        "vz_after_R1": R1 @ normalize(vz_use),
    }

    # Step 2: 水平面内旋转
    if heading_mode == "x":
        if vx is None:
            raise ValueError("heading_mode='x' requires vx.")
        v = normalize(vx.astype(np.float64))
        v_after = R1 @ v

        p = v_after.copy()
        p[2] = 0.0
        n = np.linalg.norm(p)
        if n < 1e-12:
            raise ValueError("After z-alignment, vx projection on XY plane is too small.")

        theta = np.arctan2(p[1], p[0])
        R2 = rotz(-theta)

        debug.update({
            "heading_mode": "x",
            "heading_input": vx.astype(np.float64),
            "heading_after_R1": v_after,
            "heading_xy_after_R1": p / n,
            "theta_rad": theta,
            "theta_deg": np.degrees(theta),
            "heading_after_R2R1": R2 @ v_after,
            "target_axis": x_std,
        })

    elif heading_mode == "y":
        if vy is None:
            raise ValueError("heading_mode='y' requires vy.")
        v = normalize(vy.astype(np.float64))
        v_after = R1 @ v

        p = v_after.copy()
        p[2] = 0.0
        n = np.linalg.norm(p)
        if n < 1e-12:
            raise ValueError("After z-alignment, vy projection on XY plane is too small.")

        theta = np.arctan2(p[1], p[0])
        delta = np.pi / 2.0 - theta
        R2 = rotz(delta)

        debug.update({
            "heading_mode": "y",
            "heading_input": vy.astype(np.float64),
            "heading_after_R1": v_after,
            "heading_xy_after_R1": p / n,
            "theta_rad": theta,
            "theta_deg": np.degrees(theta),
            "delta_rad": delta,
            "delta_deg": np.degrees(delta),
            "heading_after_R2R1": R2 @ v_after,
            "target_axis": y_std,
        })
    else:
        raise ValueError("heading_mode must be 'x' or 'y'.")

    R = R2 @ R1
    debug["R1"] = R1
    debug["R2"] = R2
    debug["R"] = R
    debug["vz_after_R"] = R @ normalize(vz_use)

    return R1, R2, R, debug


# ===================== LAS IO =====================

def load_las_points(path):
    las = laspy.read(path)
    points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    return las, points


def save_las_with_new_points(las, new_points, out_path):
    if new_points.shape[1] != 3:
        raise ValueError("new_points must have shape (N, 3)")
    if len(new_points) != len(las.points):
        raise ValueError("Number of new points must equal number of LAS points")

    new_x = new_points[:, 0]
    new_y = new_points[:, 1]
    new_z = new_points[:, 2]

    # 直接在原 las 对象上改，参考你之前可用的写法
    las.header.offsets = np.array(
        [np.min(new_x), np.min(new_y), np.min(new_z)],
        dtype=np.float64
    )
    las.header.scales = np.array(las.header.scales, dtype=np.float64)

    las.x = new_x
    las.y = new_y
    las.z = new_z

    las.write(out_path)

# ===================== 可视化 =====================

def numpy_to_o3d(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def draw_two_clouds(points_a, points_b, title="Open3D", max_points=200000):
    pa = random_downsample(points_a, max_points)
    pb = random_downsample(points_b, max_points)

    pcd_a = numpy_to_o3d(pa, [1.0, 0.706, 0.0])   # orange
    pcd_b = numpy_to_o3d(pb, [0.0, 0.651, 0.929]) # blue

    o3d.visualization.draw_geometries(
        [pcd_a, pcd_b],
        window_name=title,
        width=1280,
        height=800
    )


# ===================== Main =====================

def main():
    # 读取 LAS
    print("Loading LAS...")
    las, points = load_las_points(input_las)
    print(f"LAS points: {len(points)}")

    if len(points) == 0:
        raise ValueError("Input LAS is empty.")

    # 构造两步旋转
    R1, R2, R, debug = build_two_step_rotation(
        vz=vz,
        heading_mode=heading_mode,
        vx=vx,
        vy=vy,
        force_z_up=force_z_up
    )

    print("\n========== Step 1: Align vz to +Z ==========")
    print("vz input =", debug["vz_input"])
    print("vz used  =", debug["vz_used"])
    print("vz after R1 =", debug["vz_after_R1"])

    print("\nR1 =")
    print(R1)

    print("\n========== Step 2: Planar rotation ==========")
    print("heading mode =", debug["heading_mode"])
    print("heading input =", debug["heading_input"])
    print("heading after R1 =", debug["heading_after_R1"])
    print("heading xy after R1 =", debug["heading_xy_after_R1"])

    if heading_mode == "x":
        print(f"theta (deg) = {debug['theta_deg']:.6f}")
    else:
        print(f"theta (deg) = {debug['theta_deg']:.6f}")
        print(f"delta (deg) = {debug['delta_deg']:.6f}")

    print("heading after R2@R1 =", debug["heading_after_R2R1"])
    print("target axis =", debug["target_axis"])

    print("\n========== Final Rotation ==========")
    print("R =")
    print(R)
    print("vz after R =", debug["vz_after_R"])

    # 旋转中心
    if center_mode == "centroid":
        center = points.mean(axis=0)
    elif center_mode == "origin":
        center = np.zeros(3, dtype=np.float64)
    elif center_mode == "custom":
        center = custom_center.astype(np.float64)
    else:
        raise ValueError("center_mode must be centroid / origin / custom")

    print("\nRotation center =", center)

    # 构造总变换
    T1 = build_4x4_from_R_center(R1, center)
    T = build_4x4_from_R_center(R, center)

    print("\nT =")
    print(T)

    # np.save(output_T, T)
    # print(f"\nSaved transform matrix to: {output_T}")

    # 变换点云
    points_step1 = transform_points(points, T1)
    points_final = transform_points(points, T)

    # 保存
    save_las_with_new_points(las, points_final, output_las)
    print(f"Saved aligned LAS to: {output_las}")

    # 可视化
    if visualize:
        print("\nVisualizing original vs step1...")
        draw_two_clouds(points, points_step1, title="Original (orange) vs Step1-ZAligned (blue)", max_points=vis_max_points)

        print("\nVisualizing step1 vs final...")
        draw_two_clouds(points_step1, points_final, title="Step1-ZAligned (orange) vs Final (blue)", max_points=vis_max_points)

        print("\nVisualizing original vs final...")
        draw_two_clouds(points, points_final, title="Original (orange) vs Final (blue)", max_points=vis_max_points)


if __name__ == "__main__":
    main()