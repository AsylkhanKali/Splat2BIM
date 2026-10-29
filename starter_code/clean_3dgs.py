#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean a point cloud stored in a TXT file (xyzrgb) using DBSCAN clustering.

配置区 (请根据需要修改)：
  - input_txt:        输入点云 TXT，每行 "x y z r g b"
  - output_txt:       输出清理后点云 TXT，同样输出 "x y z r g b"
  - eps:              DBSCAN 邻域半径
  - min_points:       DBSCAN 最小邻居数 (min_samples)
  - min_cluster_size: 丢弃小于此大小的簇；0 表示只丢弃噪声
"""

import numpy as np
import open3d as o3d
from collections import Counter
from plyfile import PlyData, PlyElement

# ========== 配置区 ==========
in_ply = r"D:\MCP\MEP4\3Dscanner.ply"
out_ply = r"D:\MCP\MEP4\3Dscanner_clearn.ply"
eps              = 0.03      # 邻域半径
min_points       = 8        # DBSCAN 最小邻居数
min_cluster_size = 0         # 丢弃小于此大小的簇；0 表示只丢弃噪声
# ============================

if __name__ == "__main__":
    # 1. 从 TXT 加载点云 (每行 "x y z r g b")
    print(f"Loading points from '{in_ply}' ...")

    ply = PlyData.read(in_ply)
    v = ply['vertex'].data  # structured array with all fields
    x = v['x']
    y = v['y']
    z = v['z']
    xyz = np.array([x, y, z]).T
    n_pts = xyz.shape[0]

    # 2. 构造 Open3D 点云并设置颜色 (归一化到 [0,1])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # 如果颜色是 0-255，先除以 255

    # 3. 用 DBSCAN 聚类
    print(f"Clustering with DBSCAN (eps={eps}, min_points={min_points}) ...")
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True),
        dtype=int
    )

    # 4. 统计每个簇大小 (不含噪声 -1)
    counts = Counter(labels[labels >= 0])
    num_clusters = len(counts)
    print(f"  Found {num_clusters} clusters (excluding noise).")

    # 5. 确定要保留的簇标签
    if min_cluster_size > 0:
        keep_labels = {lab for lab, c in counts.items() if c >= min_cluster_size}
        print(f"  Keeping clusters with ≥{min_cluster_size} pts → {len(keep_labels)} clusters.")
    else:
        keep_labels = set(counts.keys())
        print("  Keeping all non-noise clusters.")

    # 6. 选择要保留的点
    mask = np.array([lab in keep_labels for lab in labels])
    kept_idx = np.nonzero(mask)[0]
    print(f"  Keeping {kept_idx.size} points out of {n_pts}.")

    v_cropped = v[mask]
    # Write back preserving the same dtype/props
    el = PlyElement.describe(v_cropped, 'vertex', comments=ply['vertex'].comments)
    ply_out = PlyData([el], text=ply.text)  # keep same ascii/binary mode
    ply_out.write(out_ply)
    print(f"Kept {mask.sum()} / {len(v)} gaussians")