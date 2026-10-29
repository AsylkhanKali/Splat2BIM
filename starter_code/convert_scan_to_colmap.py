"""Export 3D Scanner App raw RGB/ARKit/depth data to COLMAP for Postshot.

Dependencies: numpy, scipy, Pillow. Sources are never changed.
Usage: python convert_scan_to_colmap.py [scan_folder] [--output output_folder]
"""

import argparse
import csv
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial.transform import Rotation


FLIP = np.diag([1.0, -1.0, -1.0])


def normalize_jpeg_orientation(data):
    """Change TIFF Orientation in JPEG APP1 to 1 without re-encoding pixels."""
    result = bytearray(data)
    if result[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG")
    pos = 2
    while pos < len(result):
        if result[pos] != 255:
            raise ValueError("Invalid JPEG marker")
        while result[pos] == 255:
            pos += 1
        marker = result[pos]
        pos += 1
        if marker in (0xDA, 0xD9):
            break
        size = struct.unpack_from(">H", result, pos)[0]
        start = pos + 2
        if marker == 0xE1 and result[start:start + 6] == b"Exif\0\0":
            base = start + 6
            endian = {b"II": "<", b"MM": ">"}[bytes(result[base:base + 2])]
            offset = struct.unpack_from(endian + "I", result, base + 4)[0]
            ifd = base + offset
            count = struct.unpack_from(endian + "H", result, ifd)[0]
            for i in range(count):
                entry = ifd + 2 + i * 12
                tag, typ, n = struct.unpack_from(endian + "HHI", result, entry)
                if tag == 274:
                    if (typ, n) != (3, 1):
                        raise ValueError("Unexpected EXIF orientation encoding")
                    struct.pack_into(endian + "H", result, entry + 8, 1)
        pos += size
    return bytes(result)


def read_pose(meta):
    c2w = np.asarray(meta["cameraPoseARFrame"], dtype=float).reshape(4, 4)
    intrinsics = np.asarray(meta["intrinsics"], dtype=float).reshape(3, 3)
    if not np.isfinite(c2w).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Non-finite camera data")
    if not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Expected row-major camera-to-world matrix")
    raw_r = c2w[:3, :3]
    if np.max(np.abs(raw_r.T @ raw_r - np.eye(3))) > 1e-4:
        raise ValueError("Invalid camera rotation")
    # Remove only floating-point non-orthogonality before quaternion conversion.
    u, _, vt = np.linalg.svd(raw_r)
    r_c2w = u @ vt
    if np.linalg.det(r_c2w) < 0:
        raise ValueError("Camera rotation contains a reflection")
    r = FLIP @ r_c2w.T
    center = c2w[:3, 3]
    t = -r @ center
    q = Rotation.from_matrix(r).as_quat()[[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    return intrinsics, r, t, q, center


def write_models(txt_dir, bin_dir, frames, xyz, colors, owners, uv):
    txt_dir.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    n = len(xyz)
    observations = [np.flatnonzero(owners == i) for i in range(len(frames))]
    point2d_idx = np.empty(n, dtype=np.int32)
    for ids in observations:
        point2d_idx[ids] = np.arange(len(ids))
    with (txt_dir / "cameras.txt").open("w", encoding="utf-8") as f, \
            (bin_dir / "cameras.bin").open("wb") as b:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n")
        b.write(struct.pack("<Q", len(frames)))
        for i, frame in enumerate(frames, 1):
            w, h = frame["size"]
            params = frame["params"]
            f.write(f"{i} PINHOLE {w} {h} " + " ".join(f"{x:.17g}" for x in params) + "\n")
            b.write(struct.pack("<iiQQ4d", i, 1, w, h, *params))
    with (txt_dir / "images.txt").open("w", encoding="utf-8") as f, \
            (bin_dir / "images.bin").open("wb") as b:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# Next line: X Y POINT3D_ID ...\n")
        b.write(struct.pack("<Q", len(frames)))
        for i, (frame, ids) in enumerate(zip(frames, observations), 1):
            values = [*frame["q"], *frame["t"]]
            f.write(str(i) + " " + " ".join(f"{x:.17g}" for x in values) + f" {i} {frame['name']}\n")
            f.write(" ".join(f"{uv[j, 0]:.17g} {uv[j, 1]:.17g} {j + 1}" for j in ids) + "\n")
            b.write(struct.pack("<i4d3di", i, *values, i))
            b.write(frame["name"].encode("utf-8") + b"\0")
            b.write(struct.pack("<Q", len(ids)))
            for j in ids:
                b.write(struct.pack("<ddq", *uv[j], j + 1))
    with (txt_dir / "points3D.txt").open("w", encoding="utf-8") as f, \
            (bin_dir / "points3D.bin").open("wb") as b:
        f.write("# LiDAR samples, not triangulated visual features. One measured RGB-D observation per point.\n")
        f.write("# POINT3D_ID X Y Z R G B ERROR IMAGE_ID POINT2D_IDX\n")
        b.write(struct.pack("<Q", n))
        for j, (p, color) in enumerate(zip(xyz, colors)):
            owner = int(owners[j]) + 1
            idx = int(point2d_idx[j])
            f.write(f"{j + 1} " + " ".join(f"{x:.17g}" for x in p) +
                    " " + " ".join(str(int(x)) for x in color) + f" 0 {owner} {idx}\n")
            b.write(struct.pack("<Q3d3BdQii", j + 1, *p, *color, 0., 1, owner, idx))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", default=r"D:\Codex+Blender\iPhone/3D Scanner App_20261012")
    parser.add_argument("--output", default=r"D:\Codex+Blender\iPhone/postshot_colmap")
    parser.add_argument("--voxel", type=float, default=0.04, help="LiDAR point spacing in meters")
    args = parser.parse_args()
    if args.voxel <= 0:
        parser.error("--voxel must be positive")
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    support = output.with_name(output.name + "_support")
    if output.exists() or support.exists():
        raise FileExistsError("Choose a new output name; existing results will not be overwritten")
    image_paths = sorted(source.glob("frame_*.jpg"))
    if not image_paths:
        raise ValueError("No frame_*.jpg images found")
    image_dir = output / "images"
    image_dir.mkdir(parents=True)
    support.mkdir()
    frames, all_xyz, all_rgb, all_owners, all_uv = [], [], [], [], []
    orientations = Counter()
    integrity = []
    pose_rows = []
    missing_rgb = []
    for path in sorted(source.glob("frame_*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        k, r, t, q, center = read_pose(meta)
        has_rgb = path.with_suffix(".jpg").exists()
        pose_rows.append([meta["frame_index"], path.with_suffix(".jpg").name,
                          int(has_rgb), meta.get("time", ""), *center, *q, *t,
                          k[0, 0], k[1, 1], k[0, 2], k[1, 2], meta.get("motionQuality", "")])
        if not has_rgb:
            missing_rgb.append(meta["frame_index"])
    for i, path in enumerate(image_paths):
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        k, r, t, q, center = read_pose(meta)
        with Image.open(path) as im:
            w, h = im.size
            orientation = im.getexif().get(274, 1)
            rgb = np.array(im)
        orientations[orientation] += 1
        projection = np.array(meta["projectionMatrix"]).reshape(4, 4)
        calibration_size = [2 * k[0, 0] / projection[0, 0], 2 * k[1, 1] / projection[1, 1]]
        if not np.allclose(calibration_size, [w, h], atol=2):
            raise ValueError(f"Intrinsics resolution differs from stored pixels: {path.name}")
        raw = path.read_bytes()
        fixed = normalize_jpeg_orientation(raw)
        target = image_dir / path.name
        target.write_bytes(fixed)
        with Image.open(target) as im:
            if im.getexif().get(274, 1) != 1 or not np.array_equal(rgb, np.asarray(im)):
                raise ValueError("Export image pixels or orientation mismatch")
        integrity.append({"image": path.name, "source_sha256": hashlib.sha256(raw).hexdigest(),
                          "export_sha256": hashlib.sha256(fixed).hexdigest(), "pixels_identical": True,
                          "source_exif_orientation": orientation})
        frames.append({"name": path.name, "size": [w, h],
                       "params": [k[0, 0], k[1, 1], k[0, 2], k[1, 2]],
                       "q": q, "r": r, "t": t, "center": center})
        suffix = path.stem.removeprefix("frame_")
        depth = np.asarray(Image.open(source / f"depth_{suffix}.png"), dtype=float) / 1000.
        confidence = np.asarray(Image.open(source / f"conf_{suffix}.png"))
        dh, dw = depth.shape
        if confidence.shape != depth.shape or abs(w / h - dw / dh) > 1e-6:
            raise ValueError("Depth/RGB aspect ratio mismatch")
        # Keep only high-confidence, finite, in-range samples away from depth discontinuities.
        spread = maximum_filter(depth, size=3) - minimum_filter(depth, size=3)
        yy, xx = np.mgrid[1:dh - 1:2, 1:dw - 1:2]
        z = depth[yy, xx]
        valid = (confidence[yy, xx] == 2) & (z > .15) & (z <= 5.) & (spread[yy, xx] < .05 + .02 * z)
        xx, yy, z = xx[valid], yy[valid], z[valid]
        # Scale intrinsics and pixel coordinates consistently to the native 256x192 depth grid.
        u, v = xx * w / dw, yy * h / dh
        cam = np.column_stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z])
        world = cam @ r + center
        colors = rgb[np.rint(v).astype(int).clip(0, h - 1), np.rint(u).astype(int).clip(0, w - 1)]
        all_xyz.append(world)
        all_rgb.append(colors)
        all_owners.append(np.full(len(world), i, dtype=np.int32))
        all_uv.append(np.column_stack([u, v]))
        if i % 50 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} RGB/depth pairs", flush=True)
    xyz = np.concatenate(all_xyz)
    colors = np.concatenate(all_rgb)
    owners = np.concatenate(all_owners)
    uv = np.concatenate(all_uv)
    candidate_count = len(xyz)
    if not candidate_count:
        raise ValueError("No valid depth points")
    voxels = np.floor(xyz / args.voxel).astype(np.int64)
    # Preserve an actual measured point in each voxel, rather than inventing feature matches.
    _, first = np.unique(voxels, axis=0, return_index=True)
    first.sort()
    xyz, colors, owners, uv = xyz[first], colors[first], owners[first], uv[first]
    max_reprojection_error = 0.
    max_center_error = 0.
    for i, frame in enumerate(frames):
        ids = np.flatnonzero(owners == i)
        camera_xyz = xyz[ids] @ frame["r"].T + frame["t"]
        fx, fy, cx, cy = frame["params"]
        projected = camera_xyz[:, :2] / camera_xyz[:, 2:3] * [fx, fy] + [cx, cy]
        if len(ids):
            max_reprojection_error = max(max_reprojection_error, float(np.max(np.linalg.norm(projected - uv[ids], axis=1))))
            assert np.all(camera_xyz[:, 2] > 0)
        max_center_error = max(max_center_error, float(np.linalg.norm(-frame["r"].T @ frame["t"] - frame["center"])))
    assert max_reprojection_error < 1e-6 and max_center_error < 1e-10
    write_models(support / "colmap_text", output / "sparse" / "0", frames, xyz, colors, owners, uv)
    with (support / "all_camera_poses.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_index", "image_name", "has_rgb", "timestamp", "camera_x", "camera_y", "camera_z",
                         "qw", "qx", "qy", "qz", "tx", "ty", "tz", "fx", "fy", "cx", "cy", "motion_quality"])
        writer.writerows(pose_rows)
    report = {
        "source": str(source), "output": str(output), "metadata_frames": len(pose_rows),
        "exported_images": len(frames), "exported_cameras": len(frames),
        "metadata_without_rgb": len(missing_rgb), "missing_rgb_frame_indices": missing_rgb,
        "image_resolution": sorted(set(tuple(f["size"]) for f in frames)),
        "original_exif_orientations": dict(orientations), "export_exif_orientation": 1,
        "image_pixels_verified_unchanged": True,
        "coordinate_conversion": "R_w2c = diag(1,-1,-1) @ R_arkit_c2w.T; t = -R_w2c @ C; Hamilton quaternion qw,qx,qy,qz",
        "world_coordinates": "Original ARKit world, meters; transformToWorldMap NOT reapplied",
        "rotation_handling": "SVD orthonormalization of floating-point rotation; no pose optimization",
        "camera_model": "PINHOLE; separate per-image intrinsics; distortion coefficients absent in source",
        "point_source": "Paired LiDAR depth PNG (uint16 millimeters) and RGB; one measured observation per point, not SfM tracks",
        "point_filter": {"confidence": 2, "depth_range_m": [.15, 5.], "depth_pixel_step": 2,
                         "max_3x3_depth_range_m": "0.05 + 0.02 * depth", "voxel_m": args.voxel},
        "candidate_depth_points": candidate_count, "exported_points": len(xyz),
        "point_bounds_m": [xyz.min(axis=0).tolist(), xyz.max(axis=0).tolist()],
        "max_arithmetic_reprojection_error_px": max_reprojection_error,
        "max_camera_center_roundtrip_error_m": max_center_error,
        "validation_note": "Arithmetic checks verify conversion, not physical pose/depth accuracy; no bundle adjustment performed",
        "postshot_training_performed": False,
        "sources": ["https://colmap.github.io/format.html",
                    "https://developer.apple.com/documentation/arkit/arcamera/transform",
                    "https://activation.jawset.com/docs/d/Postshot%2BUser%2BGuide/Importing%2BImages"]}
    (support / "export_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (support / "image_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ["exported_images", "exported_cameras", "exported_points",
                                           "max_arithmetic_reprojection_error_px", "output"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
