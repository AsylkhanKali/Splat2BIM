"""Controlled Sim(3) registration components; evaluation never selects a pose."""
import itertools
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def apply(points,T):return points@T[:3,:3].T+T[:3,3]


def similarity(a,b):
    x=a-a.mean(0);y=b-b.mean(0)
    u,s,vt=np.linalg.svd(y.T@x/len(x));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    scale=(s@sign)/np.mean(np.sum(x*x,axis=1));T=np.eye(4)
    T[:3,:3]=scale*(u*sign)@vt;T[:3,3]=b.mean(0)-T[:3,:3]@a.mean(0)
    return T


def pca_hypotheses(source,target):
    a,u=np.linalg.eigh(np.cov(source.T));b,v=np.linalg.eigh(np.cov(target.T))
    scale=np.sqrt(b.sum()/a.sum());out=[]
    for signs in itertools.product([-1,1],repeat=3):
        R=v@np.diag(signs)@u.T
        if np.linalg.det(R)<0:continue
        T=np.eye(4);T[:3,:3]=scale*R;T[:3,3]=target.mean(0)-T[:3,:3]@source.mean(0);out.append(T)
    return out


def voxel(points,size):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    return np.asarray(cloud.voxel_down_sample(size).points)


def sample(points,limit,seed):
    if len(points)<=limit:return points
    return points[np.sort(np.random.default_rng(seed).choice(len(points),limit,replace=False))]


def held_tiles(points):
    tiles=np.floor(points/.4).astype(np.int64)
    return tiles,tiles.sum(1)%4==0


def semantic_classes(source,target,confidence,seed,shuffle=False):
    labels=source['labels'].copy()
    if shuffle:labels=np.random.default_rng(seed).permutation(labels)
    classes=[]
    for group in (1,2,3,4):
        a=source['points'][(labels==group)&(source['confidence']>=confidence)]
        b=target['points'][(target['labels']==group)&(target['confidence']>=confidence)]
        if min(len(a),len(b))>=128:classes.append((group,sample(a,512,seed+group),cKDTree(b)))
    return classes


def scores(T,points,tree,classes,weight):
    geometry=float(np.quantile(tree.query(apply(points,T),workers=2)[0],.75))
    values=[float(np.quantile(np.minimum(t.query(apply(p,T),workers=2)[0],20.),.6)) for _,p,t in classes]
    semantic=float(np.mean(values)) if len(values)>=2 else None
    return {'geometry':geometry,'semantic':semantic,'combined':geometry+weight*semantic if weight and semantic is not None else (geometry if not weight else None)}


def refine(source,target,initial,trim=.75,iterations=500,fine_target=None):
    T=initial.copy();tree=cKDTree(target);trace=[]
    for gate in (40.,20.,10.,5.):
        for iteration in range(iterations):
            moved=apply(source,T);distance,index=tree.query(moved,workers=2)
            keep=(distance<gate)&(distance<=np.quantile(distance,trim))
            if keep.sum()<30:raise ValueError('Insufficient point-to-point correspondences')
            delta=similarity(moved[keep],target[index[keep]]);T=delta@T
            movement=float(np.sqrt(np.mean(np.sum((apply(moved[keep],delta)-moved[keep])**2,axis=1))))
            if movement<gate*1e-5:break
        trace.append({'stage':'point_to_point','gate_m':gate,'iterations':iteration+1,'pairs':int(keep.sum()),
                      'last_update':movement,'tolerance':gate*1e-5,'update_measure':'RMS correspondence displacement, metres',
                      'stop_reason':'tolerance' if movement<gate*1e-5 else 'iteration_limit'})
    coarse=T.copy()
    if fine_target is not None:
        points,normals=fine_target;tree=cKDTree(points)
        for gate in (18.,9.,4.,2.,1.):
            for iteration in range(iterations):
                moved=apply(source,T);distance,index=tree.query(moved,workers=2,distance_upper_bound=gate)
                keep=(distance<gate)&(distance<=np.quantile(np.minimum(distance,gate),trim))
                if keep.sum()<100:raise ValueError('Insufficient point-to-plane correspondences')
                p=moved[keep];q=points[index[keep]];n=normals[index[keep]]
                residual=np.sum((p-q)*n,axis=1);w=np.sqrt(np.minimum(1,.25/np.maximum(abs(residual),1e-9)))
                J=np.c_[np.cross(p,n),n,np.sum(p*n,axis=1)]
                step=np.linalg.lstsq(J*w[:,None],-residual*w,rcond=None)[0]
                step/=max(1,np.linalg.norm(step[:3])/.08,abs(step[6])/.05,np.linalg.norm(step[3:6])/5)
                delta=np.eye(4);delta[:3,:3]=np.exp(step[6])*Rotation.from_rotvec(step[:3]).as_matrix();delta[:3,3]=step[3:6];T=delta@T
                if np.linalg.norm(step)<1e-6:break
            trace.append({'stage':'point_to_plane','gate_m':gate,'iterations':iteration+1,'pairs':int(keep.sum()),
                          'last_update':float(np.linalg.norm(step)),'tolerance':1e-6,'update_measure':'Norm of bounded rotation/translation/log-scale increment',
                          'stop_reason':'tolerance' if np.linalg.norm(step)<1e-6 else 'iteration_limit'})
    return T,coarse,trace


def convergence_summary(trace):
    return {'terminal_converged':trace[-1]['stop_reason']=='tolerance' if trace else None,
            'stages_at_limit':sum(t['stop_reason']=='iteration_limit' for t in trace),
            'total_iterations':sum(t['iterations'] for t in trace)}


def fine_cloud(points,size):
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(voxel(points,size)))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.,max_nn=40))
    return np.asarray(cloud.points),np.asarray(cloud.normals)


def measures(distances):
    return {'n':len(distances),'median_m':float(np.median(distances)),'p90_m':float(np.quantile(distances,.9)),
            'rmse_m':float(np.sqrt(np.mean(distances**2))),
            'within_025m':float(np.mean(distances<=.25)),'within_05m':float(np.mean(distances<=.5)),'within_1m':float(np.mean(distances<=1.))}


def matrix_parts(T):
    scale=float(np.cbrt(np.linalg.det(T[:3,:3])))
    if not np.isfinite(T).all() or not scale>0:raise ValueError('Invalid similarity')
    R=T[:3,:3]/scale
    np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-6)
    return {'scale':scale,'rotation':R.tolist(),'translation_local_m':T[:3,3].tolist()}
