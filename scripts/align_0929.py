"""Reproduce the 0929 alignment baseline without modifying source assets."""
from pathlib import Path
import itertools
import json
import re

import laspy
import numpy as np
import open3d as o3d
from plyfile import PlyData
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "0929"
OUT = ROOT / "results" / "0929_baseline"


def apply(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def similarity(source, target, with_scale=True):
    """Least-squares proper similarity, column-vector convention."""
    x = source - source.mean(axis=0)
    y = target - target.mean(axis=0)
    u, singular, vt = np.linalg.svd(y.T @ x / len(x))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(u @ vt)
    rotation = (u * sign) @ vt
    scale = (singular @ sign) / np.mean(np.sum(x*x, axis=1)) if with_scale else 1.
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target.mean(axis=0) - matrix[:3, :3] @ source.mean(axis=0)
    return matrix


def downsample(points, voxel):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(pc.voxel_down_sample(voxel).points)


def distances(source, target):
    return cKDTree(target).query(source, workers=4)[0]


def summary(values):
    return dict(count=len(values), median=float(np.median(values)),
                p90=float(np.quantile(values, .9)), p95=float(np.quantile(values, .95)),
                rmse=float(np.sqrt(np.mean(values**2))))


def icp(source, target, initial, thresholds, with_scale=False, iterations=40):
    """Trimmed point-to-point ICP; estimate increments in target coordinates."""
    tree = cKDTree(target)
    matrix = initial.copy()
    for threshold in thresholds:
        for _ in range(iterations):
            current = apply(source, matrix)
            error, index = tree.query(current, workers=4)
            cutoff = min(threshold, np.quantile(error, .75))
            keep = error < cutoff
            if keep.sum() < 30:
                raise ValueError("Too few geometric correspondences")
            delta = similarity(current[keep], target[index[keep]], with_scale)
            matrix = delta @ matrix
            movement = np.sqrt(np.mean(np.sum((apply(current[keep], delta)-current[keep])**2,axis=1)))
            if movement < threshold * 1e-5:
                break
    return matrix


def refine_similarity_planes(source, target, initial):
    """Resolve scale and sliding ambiguity using target surface normals."""
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=.1,max_nn=40))
    normals=np.asarray(cloud.normals)
    tree=cKDTree(target)
    matrix=initial.copy()
    for threshold in (.3,.15,.08,.04):
        for iteration in range(100):
            current=apply(source,matrix)
            error,index=tree.query(current,workers=4)
            keep=error<min(threshold,np.quantile(error,.85))
            p=current[keep];q=target[index[keep]];n=normals[index[keep]]
            residual=np.sum((p-q)*n,axis=1)
            weights=np.minimum(1,.015/np.maximum(np.abs(residual),1e-9))
            jacobian=np.c_[np.cross(p,n),n,np.sum(p*n,axis=1)]
            step=np.linalg.lstsq(jacobian*np.sqrt(weights[:,None]),-residual*np.sqrt(weights),rcond=None)[0]
            delta=np.eye(4)
            delta[:3,:3]=np.exp(step[6])*Rotation.from_rotvec(step[:3]).as_matrix()
            delta[:3,3]=step[3:6]
            matrix=delta@matrix
            if np.linalg.norm(step)<1e-7:break
        print("Surface refinement",threshold,"iterations",iteration+1,
              "scale",float(np.cbrt(np.linalg.det(matrix[:3,:3]))),flush=True)
    return matrix


def camera_pairs():
    source, target, names = [], [], []
    projection = Transformer.from_crs(4326, 2326, always_xy=True)
    with (DATA / "colmap/images.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                continue
            name = fields[9].strip().replace("\\", "/").split("/")[-1]
            photo = DATA / "photos" / name
            if not photo.exists():
                continue
            with photo.open("rb") as image:
                metadata = dict(re.findall(r'drone-dji:(\w+)="([^"]*)"', image.read(65536).decode("latin1")))
            if metadata.get("RtkFlag") != "50":
                continue
            q = np.array(fields[1:5], dtype=float)
            rotation = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            source.append(-rotation.T @ np.array(fields[5:8], dtype=float))
            x,y = projection.transform(float(metadata["GpsLongitude"]), float(metadata["GpsLatitude"]))
            target.append([x,y,float(metadata["AbsoluteAltitude"])])
            names.append(name)
    return np.asarray(source), np.asarray(target), names, projection.description


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    las = laspy.read(DATA / "0929.las")
    reference = np.c_[las.x, las.y, las.z]
    origin = np.r_[np.round(np.median(reference[:,:2],axis=0)/100)*100, 0.]
    reference -= origin
    camera_src, camera_dst, names, projection = camera_pairs()
    camera_dst -= origin
    # Distributed contiguous blocks withheld from the camera fit.
    held_out = (np.arange(len(names)) // 50) % 5 == 0
    camera_T = similarity(camera_src[~held_out], camera_dst[~held_out])
    camera_error = np.linalg.norm(apply(camera_src,camera_T)-camera_dst,axis=1)
    print("Camera fit",summary(camera_error[held_out]),flush=True)
    sparse = np.loadtxt(DATA / "colmap/points3D.txt", usecols=(1,2,3))
    sparse_m = apply(sparse, camera_T)
    print("Sparse -> LAS before refinement",summary(distances(sparse_m[::10], reference)),flush=True)
    target_small = downsample(reference,.8)
    source_small = downsample(sparse_m,1.)
    # Disjoint spatial tiles provide a geometric check, not survey ground truth.
    eval_tiles = (np.floor(source_small[:,0]/20)+np.floor(source_small[:,1]/20)).astype(int)%4 == 0
    xy_error, xy_index = cKDTree(target_small[:,:2]).query(source_small[~eval_tiles,:2],workers=4)
    dz = np.median((target_small[xy_index,2]-source_small[~eval_tiles,2])[xy_error<1.5])
    initial = np.eye(4);initial[2,3] = dz
    las_delta = icp(source_small[~eval_tiles],target_small,initial,[5.,2.,1.],iterations=70)
    sparse_T = las_delta @ camera_T
    print("LAS correction",las_delta,flush=True)
    print("Sparse held-out -> LAS",summary(distances(apply(source_small[eval_tiles],las_delta),reference)),flush=True)
    ply = PlyData.read(DATA / "0929.ply",mmap="r")
    v = ply["vertex"].data[::10]
    xyz = np.c_[v["x"],v["y"],v["z"]].astype(float)
    keep = (v["opacity"]>0) & np.isfinite(xyz).all(axis=1)
    xyz = xyz[keep]
    bounds = np.quantile(xyz,[.005,.995],axis=0)
    xyz = xyz[((xyz>=bounds[0])&(xyz<=bounds[1])).all(axis=1)]
    ply_holdout=np.floor(xyz/.4).sum(axis=1).astype(int)%4==0
    source = downsample(xyz[~ply_holdout],.045)
    target = downsample(sparse,.045)
    # PCA creates orientation hypotheses. Geometry selects and refines them.
    cx,cy = source.mean(axis=0), target.mean(axis=0)
    ex,ux = np.linalg.eigh(np.cov(source.T));ey,uy = np.linalg.eigh(np.cov(target.T))
    scale = np.sqrt(ey.sum()/ex.sum())
    candidates=[]
    for signs in itertools.product([-1,1],repeat=3):
        rotation = uy @ np.diag(signs) @ ux.T
        if np.linalg.det(rotation)<0:continue
        guess=np.eye(4);guess[:3,:3]=scale*rotation;guess[:3,3]=cy-guess[:3,:3]@cx
        fitted=icp(source,target,guess,[.8,.4,.2,.1],with_scale=True,iterations=100)
        error=distances(apply(source,fitted),target)
        candidates.append((float(np.median(error)),fitted))
        print("PLY orientation candidate",signs,summary(error),flush=True)
    candidates.sort(key=lambda item:item[0])
    source_fine=downsample(xyz[~ply_holdout],.025)
    target_fine=downsample(sparse,.025)
    ply_to_sparse=refine_similarity_planes(source_fine,target_fine,candidates[0][1])
    ply_T=sparse_T @ ply_to_sparse
    print("PLY -> sparse best",ply_to_sparse,flush=True)
    source_m=apply(xyz,ply_T)
    final_error=distances(source_m,reference)
    print("PLY -> LAS",summary(final_error),flush=True)
    xy_distance=cKDTree(reference[:,:2]).query(source_m[:,:2],workers=4)[0]
    overlap=xy_distance<1.
    check=ply_holdout & overlap
    print("PLY withheld tiles within LAS horizontal coverage",summary(final_error[check]),flush=True)
    report={"dataset":"0929","status":"diagnostic baseline, not survey validation",
        "origin_global":origin.tolist(),"projection":projection,
        "camera_pairs":len(names),"camera_holdout":summary(camera_error[held_out]),
        "camera_T_local":camera_T.tolist(),"sparse_to_las_local":sparse_T.tolist(),
        "ply_to_sparse":ply_to_sparse.tolist(),"ply_to_las_local":ply_T.tolist(),
        "ply_to_las_distances_m":summary(final_error),
        "ply_withheld_overlap_distances_m":summary(final_error[check]),
        "ply_withheld_overlap_fraction_within_1m":float(np.mean(final_error[check]<1.)),
        "ply_samples":len(xyz),"ply_horizontal_overlap_fraction":float(np.mean(overlap)),
        "sample_filter":"Every tenth Gaussian, opacity logit > 0; exclude coordinate-wise outer 0.5 percent tails",
        "holdout_definition":"PLY source-space 0.4-unit checkerboard tiles, 25 percent nominally withheld from PLY-to-sparse fitting; overlap is XY distance to LAS < 1 m",
        "sparse_holdout_distances_m":summary(distances(apply(source_small[eval_tiles],las_delta),reference)),
        "caveats":["Nearest-neighbor distances are agreement with the supplied LAS, not survey accuracy",
                   "Absolute photo altitude datum is unverified; vertical offset was estimated from geometry",
                   "LAS projection parameters match Hong Kong Grid, but its custom geokeys omit valid ellipsoid values; EPSG:2326 is a documented working assumption supported by BIM",
                   "BIM metadata was inspected; BIM geometry has not been validated",
                   "No independent surveyed control points are available",
                   "The existing cameras and LAS may originate from related photogrammetry; camera holdout is not independent ground truth"],
        "orientation_candidate_medians_source_units":[item[0] for item in candidates]}
    (OUT/"alignment.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(OUT/"diagnostics.npz",reference=target_small,source=source_m[::3],
        sparse=apply(sparse[::10],sparse_T),camera_source=apply(camera_src,camera_T),
        camera_target=camera_dst,camera_error=camera_error,held_out=held_out,
        gaussian_check_error=final_error[check],gaussian_all_error=final_error)


if __name__=="__main__":
    main()
