"""Register each outdoor Gaussian cloud to its own LAS; preserve all source files."""
import argparse
import itertools
import json
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from align_0929 import apply, downsample, icp, summary

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/multidate'
ORIGIN=np.array([825800.,842100.,0.])
DATES=['0331','0504','0609','0708','0723','0831','0929']
# Approximate plan-view initialization for the smaller 0708 Gaussian footprint.
# Blue rectangle, office block and road orientation in 0708_initialization.png:
# PCA XY anchor [-1,.75] -> LAS XY [-28,15], scale ~33, yaw ~40 degrees.
# These are starting estimates, not surveyed control; only training tiles refine them.
VISUAL_SEED_0708=np.array([
    [-27.02711529366277,16.057234541399442,-10.034951807833888,13.188459962668126],
    [18.52844510450093,18.82261298539256,-19.78398247096946,17.25239115246109],
    [-3.9027888634354313,-21.8374554214684,-24.43140969741255,111.3928711339027],
    [0.,0.,0.,1.]])


def samples(date):
    v=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data[::10]
    xyz=np.column_stack([v[k] for k in ('x','y','z')]).astype(float)
    keep=(v['opacity']>0)&np.isfinite(xyz).all(axis=1)
    xyz=xyz[keep]
    rgb=np.clip(.5+.28209479177387814*np.column_stack([v[f'f_dc_{k}'][keep] for k in range(3)]),0,1)
    bounds=np.quantile(xyz,[.005,.995],axis=0)
    keep=((xyz>=bounds[0])&(xyz<=bounds[1])).all(axis=1)
    xyz,rgb=xyz[keep],rgb[keep]
    held=np.floor(xyz/.4).sum(axis=1).astype(int)%4==0
    return xyz,rgb,held


def refine(source,target,initial,thresholds=(18.,9.,4.,2.,1.)):
    pc=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.,max_nn=40))
    normals=np.asarray(pc.normals);tree=cKDTree(target);matrix=initial.copy()
    for threshold in thresholds:
        for iteration in range(60):
            p=apply(source,matrix)
            err,idx=tree.query(p,workers=4,distance_upper_bound=threshold)
            # Correspondences beyond the gate cannot contribute to this step.
            cutoff=np.quantile(np.minimum(err,threshold),.85)
            keep=err<cutoff
            if keep.sum()<100:raise ValueError('Too few correspondences')
            p=p[keep];q=target[idx[keep]];n=normals[idx[keep]]
            residual=np.sum((p-q)*n,axis=1)
            w=np.sqrt(np.minimum(1,.25/np.maximum(np.abs(residual),1e-9)))
            jac=np.c_[np.cross(p,n),n,np.sum(p*n,axis=1)]
            step=np.linalg.lstsq(jac*w[:,None],-residual*w,rcond=None)[0]
            # Limit individual linearization steps, without restricting the final transform.
            step/=max(1,np.linalg.norm(step[:3])/.08,abs(step[6])/.05,np.linalg.norm(step[3:6])/5)
            delta=np.eye(4);delta[:3,:3]=np.exp(step[6])*Rotation.from_rotvec(step[:3]).as_matrix();delta[:3,3]=step[3:6]
            matrix=delta@matrix
            if np.linalg.norm(step)<1e-6:break
        print('plane',threshold,iteration+1,'scale',np.cbrt(np.linalg.det(matrix[:3,:3])),flush=True)
    return matrix


def run(date):
    out=OUT/date;out.mkdir(parents=True,exist_ok=True)
    xyz,rgb,held=samples(date)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las')
    target=np.c_[las.x,las.y,las.z]-ORIGIN
    colors=np.column_stack([getattr(las,k) for k in ('red','green','blue')]).astype(float)
    colors/=65535 if colors.max()>255 else 255
    source=downsample(xyz[~held],.06);small=downsample(target,2.)
    cx,cy=source.mean(0),small.mean(0)
    ex,ux=np.linalg.eigh(np.cov(source.T));ey,uy=np.linalg.eigh(np.cov(small.T))
    scale=np.sqrt(ey.sum()/ex.sum());tree=cKDTree(small)
    candidates=[]
    for signs in itertools.product([-1,1],repeat=3):
        rot=uy@np.diag(signs)@ux.T
        if np.linalg.det(rot)<0:continue
        t=np.eye(4);t[:3,:3]=scale*rot;t[:3,3]=cy-t[:3,:3]@cx
        t=icp(source,small,t,[40,20,10,5],with_scale=True,iterations=80)
        error=tree.query(apply(source,t),workers=4)[0]
        print(date,'candidate',signs,summary(error),flush=True)
        candidates.append((float(np.median(error)),t,'PCA geometry'))
    if date=='0708':candidates.append((0.,VISUAL_SEED_0708,'Approximate visual landmarks, then geometric refinement'))
    # Surface refinement is needed even when a point-to-point candidate looks plausible.
    fine=downsample(target,.6);fsource=downsample(xyz[~held],.025)
    fulltree=cKDTree(target);scores=[]
    for _,t,initialization in sorted(candidates,key=lambda c:c[0]):
        fitted=refine(fsource,fine,t)
        pos=apply(xyz[~held],fitted);error,idx=fulltree.query(pos,workers=4)
        color=np.linalg.norm(rgb[~held]-colors[idx],axis=1)
        score=float(np.median(error))
        scores.append({'median_m':score,'p90_m':float(np.quantile(error,.9)),
                       'median_color_difference':float(np.median(color)), 'matrix':fitted.tolist(),'initialization':initialization})
        print(date,'refined candidate',scores[-1],flush=True)
    scores.sort(key=lambda c:c['median_m'])
    t=np.array(scores[0]['matrix'])
    transformed=apply(xyz,t);error=fulltree.query(transformed,workers=4)[0]
    overlap=cKDTree(target[:,:2]).query(transformed[:,:2],workers=4)[0]<1.
    check=held&overlap
    info={'dataset':date,'origin_global':ORIGIN.tolist(),'ply_to_las_local':t.tolist(),
          'method':'PCA hypotheses (plus a documented visual seed for 0708), trimmed similarity ICP and robust point-to-plane similarity; training tiles only',
          'initialization':scores[0]['initialization'],
          'status':'rejected' if scores[0]['median_m']>.5 or scores[0]['median_color_difference']>.35 else 'diagnostic alignment; not survey validation',
          'scale_m_per_source_unit':float(np.cbrt(np.linalg.det(t[:3,:3]))),
          'ply_samples':len(xyz),'ply_to_las_distances_m':summary(error),
          'ply_withheld_overlap_distances_m':summary(error[check]),
          'ply_withheld_overlap_fraction_within_1m':float(np.mean(error[check]<1)),
          'ply_horizontal_overlap_fraction':float(np.mean(overlap)),
          'candidates':scores}
    (out/'alignment.json').write_text(json.dumps(info,indent=2)+'\n')
    np.savez_compressed(out/'diagnostics.npz',source=transformed,source_rgb=rgb,reference=target[::10],reference_rgb=colors[::10],error=error,held=held,overlap=overlap)
    print(date,'FINAL',info['ply_withheld_overlap_distances_m'],'overlap',info['ply_horizontal_overlap_fraction'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dates',nargs='*',default=DATES[:-1]);args=parser.parse_args()
    for date in args.dates:run(date)
