#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
from pathlib import Path
import re
import shutil

import numpy as np
import laspy


# ===================== 用户参数 =====================

# 输入 / 输出 LAS
input_las = r"D:\Codex+Blender\Leica/SMART LAB 261012_0005.las"
output_las = r"D:\Codex+Blender\Leica/SMART LAB 261012_0005_standard.las"

# 输出变换矩阵；相对路径按输出 LAS 所在目录解析，而不是运行工作目录
output_T = "T_align.npy"

# ---------- 360 全景照片 ----------
# 只运行本文件即可同时变换 LAS 和相机位姿，不依赖其他本地脚本。
# JPG 像素不需要旋转；使用原图 + 新 position/orientation 即可对齐新点云。
transform_360_photos = True
panorama_dir = str(Path(input_las).parent)
panorama_pose_glob = "Smart lab*.txt"
output_panorama_dir = str(Path(output_las).with_suffix("")) + "_photos"
copy_panorama_images = True   # 复制 LDR JPG 和存在的 IR JPG，与新 TXT 放在一起
copy_hdr_images = False       # EXR 较大；需要一并复制时改为 True

# ---------- E57 导出的图像及 4x4 poses ----------
transform_e57_images = True
e57_export_dir = str(Path(input_las).parent / "e57_Export")
output_e57_export_dir = str(Path(output_las).with_suffix("")) + "_e57_Export"
copy_e57_images = True        # 复制 images，输出目录可直接配套使用
# 当前 e57_Export/poses 是相机到世界矩阵：C_new = T @ C_old。
# 只有输入实际为世界到相机外参时才改成 "world_to_camera"。
e57_pose_convention = "camera_to_world"

# ---------- Step 1: 给定当前场景中的“竖直方向” vz ----------
# 它会被对齐到标准 +Z = [0,0,1]
vz = np.array([0.00543296,0.00269391,0.999982], dtype=np.float64)

# 是否强制把 vz 变成朝上
force_z_up = True

# ---------- Step 2: 平面内朝向 ----------
# "x" 表示给 vx，对齐到标准 +X
# "y" 表示给 vy，对齐到标准 +Y
heading_mode = "x"

# 当 heading_mode == "x" 时使用
vx = np.array([0.756526,-0.653964,0.000584999], dtype=np.float64)

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
    """X_new = R @ (X_old - center)：将 center 移到原点。"""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return T


def transform_points(points, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return (points @ R.T) + t


def validate_rigid_transform(T, label="Transform"):
    T = np.asarray(T, dtype=np.float64)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-8, rtol=0)):
        raise ValueError(f"{label} must be a rigid 4x4 transform")
    return T


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


# ===================== 360 全景位姿 =====================

PANORAMA_BLOCK = re.compile(
    r"(Ldr Image|Hdr Image|Ir Image):\s*"
    r"position\s*=\s*\[([^\]]+)\];\s*"
    r"orientation\s*=\s*\[([^\]]+)\];"
)


def quaternion_xyzw_to_matrix(q):
    """Leica TXT: [x, y, z, w]，表示相机局部坐标到世界坐标的旋转。"""
    x, y, z, w = normalize(np.asarray(q, dtype=np.float64))
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def matrix_to_quaternion_xyzw(R):
    """稳定处理接近 180° 的旋转，不增加 SciPy 依赖。"""
    trace = np.trace(R)
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        q = np.array([(R[2, 1]-R[1, 2])/s, (R[0, 2]-R[2, 0])/s,
                      (R[1, 0]-R[0, 1])/s, s/4])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k]))
        q = np.zeros(4)
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        q[3] = (R[k, j] - R[j, k]) / s
    q = normalize(q)
    return -q if q[3] < 0 else q


def load_panorama_poses(folder, pattern):
    """在写出 LAS 之前校验全部 TXT，防止漏处理或读入损坏位姿。"""
    records = []
    for path in sorted(Path(folder).glob(pattern)):
        text = path.read_text(encoding="utf-8-sig")
        matches = PANORAMA_BLOCK.findall(text)
        labels = re.findall(r"(Ldr Image|Hdr Image|Ir Image):", text)
        if not matches or len(matches) != len(labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid or duplicate panorama pose blocks: {path}")
        blocks = []
        for label, ptext, qtext in matches:
            p = np.array([float(v.strip()) for v in ptext.split(",")])
            q = np.array([float(v.strip()) for v in qtext.split(",")])
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
                raise ValueError(f"Invalid position/quaternion in {path}: {label}")
            C = np.eye(4)
            C[:3, :3] = quaternion_xyzw_to_matrix(q)
            C[:3, 3] = p
            blocks.append((label, C))
        jpg = path.with_suffix(".jpg")
        if "Ldr Image" in labels and not jpg.is_file():
            raise FileNotFoundError(f"Missing panorama JPG: {jpg}")
        records.append((path, blocks))
    if not records:
        raise FileNotFoundError(f"No panorama TXT matched {Path(folder) / pattern}")
    return records


def export_panorama_poses(records, T, output_folder):
    """使用 LAS 的同一个 T：C_new = T @ C_old，包含旋转和平移。"""
    T = validate_rigid_transform(T)
    out = Path(output_folder).resolve()
    if any(path.resolve() == (out / path.name).resolve() for path, _ in records):
        raise ValueError("Panorama output folder must not overwrite source poses")
    out.mkdir(parents=True, exist_ok=True)
    rows, scans = [], []
    for path, blocks in records:
        text_blocks, images = [], []
        for label, C in blocks:
            D = T @ C
            q = matrix_to_quaternion_xyzw(D[:3, :3])
            text_blocks.append(
                f"{label}:\n"
                f"position = [{', '.join(f'{v:.12f}' for v in D[:3, 3])}];\n"
                f"orientation = [{', '.join(f'{v:.12f}' for v in q)}];\n"
            )
            if label == "Ldr Image":
                # SetupInfo 的列顺序是 W,X,Y,Z，和 TXT 的 X,Y,Z,W 不同。
                rows.append([path.stem, *D[:3, 3], q[3], *q[:3]])
            source_image = (path.with_name(path.stem + "_IR.jpg") if label == "Ir Image"
                            else path.with_suffix(".exr" if label == "Hdr Image" else ".jpg"))
            should_copy = copy_hdr_images if label == "Hdr Image" else copy_panorama_images
            image_path = source_image.resolve()
            copied = False
            if should_copy and source_image.is_file():
                image_path = out / source_image.name
                shutil.copy2(source_image, image_path)
                copied = True
            images.append({
                "label": label, "image_path": str(image_path), "image_copied": copied,
                "image_exists": source_image.is_file(),
                "position_after": D[:3, 3].tolist(), "quaternion_xyzw_after": q.tolist(),
                "camera_to_world_before": C.tolist(), "camera_to_world_after": D.tolist(),
                "world_to_camera_after": np.linalg.inv(D).tolist(),
            })
        (out / path.name).write_text("".join(text_blocks), encoding="utf-8")
        scans.append({"name": path.stem, "source_pose": str(path.resolve()), "images": images})
    with (out / "SetupInfo_transformed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["SetupName", "Position(X)", "Position(Y)", "Position(Z)",
                         "Orientation(W)", "Orientation(X)", "Orientation(Y)", "Orientation(Z)"])
        writer.writerows(rows)
    np.save(out / "T_align.npy", T)
    np.savetxt(out / "T_align.txt", T, fmt="%.15f")
    manifest = {
        "input_las": str(Path(input_las).resolve()), "output_las": str(Path(output_las).resolve()),
        "T_source_to_target": T.tolist(), "pose_convention": "camera_to_world; C_new = T @ C_old",
        "txt_quaternion_order": "xyzw", "csv_quaternion_order": "wxyz",
        "jpg_pixels": "unchanged; use transformed position and orientation", "scans": scans,
    }
    (out / "poses_transformed.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(scans)} transformed panorama poses to: {out}")
    print("JPG pixels are unchanged; use the new TXT/CSV poses with the transformed LAS.")
    return manifest


# ===================== E57 图像 4x4 位姿 =====================

def load_e57_export(folder):
    """按 metadata.csv 加载；优先使用本目录文件，兼容搬迁后的旧绝对路径。"""
    folder = Path(folder).resolve()
    metadata = folder / "metadata.csv"
    with metadata.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if not fields or not {"image_path", "pose_path"}.issubset(fields):
            raise ValueError(f"Missing image_path/pose_path columns in {metadata}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty E57 metadata: {metadata}")

    def local_file(value, subfolder):
        if not value or not value.strip():
            raise ValueError(f"Missing {subfolder} path in {metadata}")
        # 原 metadata 可能记录导出机器的绝对路径，先按文件名在当前目录查找。
        name = Path(value.replace("\\", "/")).name
        local = folder / subfolder / name
        if not local.is_file():
            raise FileNotFoundError(f"Missing E57 {subfolder} file: {local}")
        return local

    records, pose_paths, image_paths = [], set(), set()
    for row in rows:
        if None in row:
            raise ValueError(f"Malformed CSV row in {metadata}")
        pose_path = local_file(row["pose_path"], "poses")
        image_path = local_file(row["image_path"], "images")
        if pose_path in pose_paths or image_path in image_paths:
            raise ValueError(f"Duplicate image/pose entry in {metadata}: {pose_path.name}")
        pose_paths.add(pose_path)
        image_paths.add(image_path)
        C = validate_rigid_transform(np.loadtxt(pose_path), str(pose_path))
        records.append({"metadata": row, "pose_path": pose_path,
                        "image_path": image_path, "pose": C})
    unlisted = set((folder / "poses").glob("*.txt")) - pose_paths
    if unlisted:
        raise ValueError(f"E57 poses missing from metadata.csv: {sorted(p.name for p in unlisted)}")
    return {"folder": folder, "fields": fields, "records": records}


def export_e57_poses(dataset, T, output_folder, convention="camera_to_world", copy_images=True):
    """保持原图像局部坐标轴及内参，更新相机在世界中的位置和朝向。"""
    T = validate_rigid_transform(T)
    if convention not in ("camera_to_world", "world_to_camera"):
        raise ValueError("e57_pose_convention must be camera_to_world or world_to_camera")
    out = Path(output_folder).resolve()
    if out == dataset["folder"]:
        raise ValueError("E57 output must not overwrite the source export")
    pose_dir = out / "poses"
    image_dir = out / "images"
    pose_dir.mkdir(parents=True, exist_ok=True)
    if copy_images:
        image_dir.mkdir(parents=True, exist_ok=True)
    inverse_T = np.linalg.inv(T)
    rows, summaries = [], []
    for item in dataset["records"]:
        C = item["pose"]
        # 直接使用本次 LAS 的 T；包括 centroid/custom_center 引入的平移。
        D = T @ C if convention == "camera_to_world" else C @ inverse_T
        pose_path = pose_dir / item["pose_path"].name
        np.savetxt(pose_path, D, fmt="%.15f")
        image_path = item["image_path"]
        if copy_images:
            image_path = image_dir / image_path.name
            shutil.copy2(item["image_path"], image_path)
        row = dict(item["metadata"])
        row.update(pose_path=str(pose_path), image_path=str(image_path))
        rows.append(row)
        summaries.append({"source_pose": str(item["pose_path"]), "pose_path": str(pose_path),
                          "image_path": str(image_path), "pose_before": C.tolist(), "pose_after": D.tolist()})
    with (out / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dataset["fields"])
        writer.writeheader()
        writer.writerows(rows)
    np.save(out / "T_align.npy", T)
    np.savetxt(out / "T_align.txt", T, fmt="%.15f")
    manifest = {
        "source_export": str(dataset["folder"]), "output_export": str(out),
        "input_las": str(Path(input_las).resolve()), "output_las": str(Path(output_las).resolve()),
        "pose_convention": convention, "camera_local_axes": "unchanged",
        "composition": "C_new = T @ C_old" if convention == "camera_to_world" else "E_new = E_old @ inverse(T)",
        "image_pixels": "unchanged", "images_copied": copy_images,
        "T_source_to_target": T.tolist(), "image_count": len(rows), "images": summaries,
    }
    (out / "poses_transformed.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} transformed E57 image poses to: {out}")
    print(f"Updated image/pose paths in: {out / 'metadata.csv'}")
    return manifest


# ===================== 可视化 =====================

def numpy_to_o3d(points, color=None):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def draw_two_clouds(points_a, points_b, title="Open3D", max_points=200000):
    import open3d as o3d
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
    output_path = Path(output_las).resolve()
    if output_path == Path(input_las).resolve():
        raise ValueError("Output LAS must not overwrite the input LAS")
    matrix_path = Path(output_T)
    if not matrix_path.is_absolute():
        matrix_path = output_path.parent / matrix_path
    if matrix_path.suffix.lower() != ".npy":
        raise ValueError("output_T must use the .npy extension")
    panorama_records = None
    if transform_360_photos:
        if Path(panorama_dir).resolve() == Path(output_panorama_dir).resolve():
            raise ValueError("Use separate input/output folders for panorama poses")
        panorama_records = load_panorama_poses(panorama_dir, panorama_pose_glob)
        print(f"Panorama stations: {len(panorama_records)}")

    e57_dataset = None
    if transform_e57_images:
        if Path(e57_export_dir).resolve() == Path(output_e57_export_dir).resolve():
            raise ValueError("Use separate input/output folders for E57 exports")
        if e57_pose_convention not in ("camera_to_world", "world_to_camera"):
            raise ValueError("e57_pose_convention must be camera_to_world or world_to_camera")
        e57_dataset = load_e57_export(e57_export_dir)
        print(f"E57 image poses: {len(e57_dataset['records'])}")

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

    # 变换点云
    points_step1 = transform_points(points, T1) if visualize else None
    points_final = transform_points(points, T)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_las_with_new_points(las, points_final, output_path)
    print(f"Saved aligned LAS to: {output_path}")
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, T)
    np.savetxt(matrix_path.with_suffix(".txt"), T, fmt="%.15f")
    print(f"Saved transform matrix to: {matrix_path}")

    # 必须传入本次点云使用的 T；不要重算旋转中心或读取旧的矩阵文件。
    if panorama_records is not None:
        export_panorama_poses(panorama_records, T, output_panorama_dir)
    if e57_dataset is not None:
        export_e57_poses(e57_dataset, T, output_e57_export_dir,
                         convention=e57_pose_convention, copy_images=copy_e57_images)

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
