import os
import csv
import glob
import pye57
import numpy as np

L = pye57.libe57


# =========================
# 基础工具
# =========================
def as_structure(node):
    return L.StructureNode(node)


def as_vector(node):
    return L.VectorNode(node)


def as_string(node):
    return L.StringNode(node).value()


def as_int(node):
    return L.IntegerNode(node).value()


def as_float(node):
    return L.FloatNode(node).value()


def get_root_child(root, name):
    for i in range(root.childCount()):
        child = root.get(i)
        if child.elementName() == name:
            return child
    return None


def get_child_by_name(struct_node, name):
    s = as_structure(struct_node)
    for i in range(s.childCount()):
        child = s.get(i)
        if child.elementName() == name:
            return child
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# 四元数转旋转矩阵
# quaternion order: w, x, y, z
# =========================
def quat_to_rotmat(qw, qx, qy, qz):
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        raise ValueError("Zero quaternion")
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n

    R = np.array([
        [1 - 2 * (qy*qy + qz*qz),     2 * (qx*qy - qz*qw),     2 * (qx*qz + qy*qw)],
        [    2 * (qx*qy + qz*qw), 1 - 2 * (qx*qx + qz*qz),     2 * (qy*qz - qx*qw)],
        [    2 * (qx*qz - qy*qw),     2 * (qy*qz + qx*qw), 1 - 2 * (qx*qx + qy*qy)]
    ], dtype=np.float64)
    return R


def build_pose_e57(qw, qx, qy, qz, tx, ty, tz):
    R = quat_to_rotmat(qw, qx, qy, qz)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def convert_pose(pose_e57, offset=np.array([831000.0, 843000.0, 0.0], dtype=np.float64)):
    """
    你指定的转换方式：

    R_wc = pose_e57[:3,:3]
    t = pose_e57[:3,3]
    S = np.diag([1.0, -1.0, -1.0])

    R_c2w = R_wc @ S
    t_new = t - offset
    """
    R_wc = pose_e57[:3, :3]
    t = pose_e57[:3, 3]

    S = np.diag([1.0, -1.0, -1.0])
    R_c2w = R_wc @ S
    t_new = t - offset

    pose_c2w = np.eye(4, dtype=np.float64)
    pose_c2w[:3, :3] = R_c2w
    pose_c2w[:3, 3] = t_new
    return pose_c2w


def save_matrix_txt(path, M):
    np.savetxt(path, M, fmt="%.12f")


# =========================
# 读取 pose
# =========================
def read_pose(image_struct):
    pose = get_child_by_name(image_struct, "pose")
    if pose is None:
        return None

    rotation = get_child_by_name(pose, "rotation")
    translation = get_child_by_name(pose, "translation")
    if rotation is None or translation is None:
        return None

    qw = as_float(get_child_by_name(rotation, "w"))
    qx = as_float(get_child_by_name(rotation, "x"))
    qy = as_float(get_child_by_name(rotation, "y"))
    qz = as_float(get_child_by_name(rotation, "z"))

    tx = as_float(get_child_by_name(translation, "x"))
    ty = as_float(get_child_by_name(translation, "y"))
    tz = as_float(get_child_by_name(translation, "z"))

    return {
        "qw": qw, "qx": qx, "qy": qy, "qz": qz,
        "tx": tx, "ty": ty, "tz": tz
    }


# =========================
# 读取 pinhole / jpeg
# =========================
def read_pinhole(image_struct):
    pinhole = get_child_by_name(image_struct, "pinholeRepresentation")
    if pinhole is None:
        return None

    jpeg_node = get_child_by_name(pinhole, "jpegImage")
    image_height = get_child_by_name(pinhole, "imageHeight")
    image_width = get_child_by_name(pinhole, "imageWidth")

    return {
        "jpeg_node": jpeg_node,
        "imageHeight": as_int(image_height) if image_height else None,
        "imageWidth": as_int(image_width) if image_width else None,
    }


# =========================
# 导出 jpeg blob
# =========================
def export_blob(blob_node, out_path):
    blob = L.BlobNode(blob_node)
    byte_count = blob.byteCount()

    buf = bytearray(byte_count)
    blob.read(buf, 0, byte_count)

    with open(out_path, "wb") as f:
        f.write(buf)


# =========================
# 处理单个 e57
# =========================
def process_one_e57(e57_path, image_dir, pose_dir, global_index_start=0, offset=None, csv_writer=None):
    if offset is None:
        offset = np.array([831000.0, 843000.0, 0.0], dtype=np.float64)

    e57_name = os.path.splitext(os.path.basename(e57_path))[0]
    print(f"\n[INFO] Processing: {e57_name}")

    e57 = pye57.E57(e57_path)
    root = e57.image_file.root()

    images2d_node = get_root_child(root, "images2D")
    if images2d_node is None:
        print(f"[WARN] No images2D found in {e57_path}")
        return global_index_start

    images2d = as_vector(images2d_node)
    print(f"[INFO] images2D childCount = {images2d.childCount()}")

    global_index = global_index_start

    for i in range(images2d.childCount()):
        item = images2d.get(i)

        if item.type() != L.NodeType.E57_STRUCTURE:
            print(f"[WARN] images2D[{i}] is not STRUCTURE, skip")
            continue

        name_node = get_child_by_name(item, "name")
        guid_node = get_child_by_name(item, "guid")
        assoc_guid_node = get_child_by_name(item, "associatedData3DGuid")

        image_name = as_string(name_node) if name_node else f"{e57_name}_{i:03d}"
        guid = as_string(guid_node) if guid_node else ""
        associated_guid = as_string(assoc_guid_node) if assoc_guid_node else ""

        pose_data = read_pose(item)
        pinhole = read_pinhole(item)

        if pose_data is None:
            print(f"[WARN] No pose for {e57_name}[{i}], skip")
            continue
        if pinhole is None or pinhole["jpeg_node"] is None:
            print(f"[WARN] No jpegImage for {e57_name}[{i}], skip")
            continue

        stem = f"{global_index:06d}"

        # 1) 导出图片
        image_out_path = os.path.join(image_dir, stem + ".jpg")
        export_blob(pinhole["jpeg_node"], image_out_path)

        # 2) 构建原始 E57 pose
        pose_e57 = build_pose_e57(
            pose_data["qw"], pose_data["qx"], pose_data["qy"], pose_data["qz"],
            pose_data["tx"], pose_data["ty"], pose_data["tz"]
        )

        # 3) 做你要求的变换
        pose_c2w = convert_pose(pose_e57, offset=offset)

        # 4) 保存 pose
        pose_out_path = os.path.join(pose_dir, stem + ".txt")
        save_matrix_txt(pose_out_path, pose_c2w)

        if csv_writer is not None:
            csv_writer.writerow({
                "global_index": global_index,
                "stem": stem,
                "source_e57": e57_name,
                "local_index": i,
                "name": image_name,
                "guid": guid,
                "associatedData3DGuid": associated_guid,
                "image_path": image_out_path,
                "pose_path": pose_out_path,
                "width": pinhole["imageWidth"] if pinhole else "",
                "height": pinhole["imageHeight"] if pinhole else "",
            })

        print(f"[OK] {e57_name}[{i}] -> {stem}.jpg / {stem}.txt")
        global_index += 1

    return global_index


# =========================
# 批量处理整个文件夹
# =========================
def process_e57_folder(folder_path, out_root, offset=np.array([0, 0, 0.0], dtype=np.float64)):
    ensure_dir(out_root)

    image_dir = os.path.join(out_root, "images")
    pose_dir = os.path.join(out_root, "poses")
    ensure_dir(image_dir)
    ensure_dir(pose_dir)

    meta_csv_path = os.path.join(out_root, "metadata.csv")

    e57_files = sorted(glob.glob(os.path.join(folder_path, "*.e57")))
    if len(e57_files) == 0:
        print("[WARN] No .e57 files found.")
        return

    print(f"[INFO] Found {len(e57_files)} e57 files.")

    global_index = 0

    with open(meta_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "global_index",
            "stem",
            "source_e57",
            "local_index",
            "name",
            "guid",
            "associatedData3DGuid",
            "image_path",
            "pose_path",
            "width",
            "height",
        ])
        writer.writeheader()

        for e57_path in e57_files:
            try:
                global_index = process_one_e57(
                    e57_path=e57_path,
                    image_dir=image_dir,
                    pose_dir=pose_dir,
                    global_index_start=global_index,
                    offset=offset,
                    csv_writer=writer,
                )
            except Exception as e:
                print(f"[ERROR] Failed processing {e57_path}")
                print(e)

    print("\n[Done]")
    print(f"Images saved to: {image_dir}")
    print(f"Poses saved to:  {pose_dir}")
    print(f"Metadata saved to: {meta_csv_path}")


if __name__ == "__main__":
    folder_path = r"D:\Codex+Blender\Leica\e57"
    out_root = r"D:\Codex+Blender\Leica\e57_Export"

    process_e57_folder(folder_path, out_root)