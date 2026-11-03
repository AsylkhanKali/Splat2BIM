"""Calibrated image localization using reference photographs and LAS depths."""
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial import cKDTree,ConvexHull
from scipy.spatial.transform import Rotation

PARAMETERS={'image_width':1200,'sift_features':5000,'reference_stride':12,'query_exclusion_radius':10,
    'reference_depth_pixel_radius':2.5,'reference_depth_spread_m':2.,'retrieval_count':5,
    'ratio_test':.75,'ransac_pixels':3.,'minimum_inliers':25,'minimum_inlier_fraction':.2,
    'minimum_validation_matches':10,'minimum_validation_fraction':.35,'minimum_image_area':.05}


def read_model(folder):
    cameras={};images=[]
    for line in (folder/'cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        fields=line.split();cid=int(fields[0]);assert fields[1]=='FULL_OPENCV'
        p=np.array(fields[4:],float);cameras[cid]={'width':int(fields[2]),'height':int(fields[3]),
            'K':np.array([[p[0],0,p[2]],[0,p[1],p[3]],[0,0,1.]]),'distortion':p[4:]}
    with (folder/'images.txt').open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            p=line.split(maxsplit=9);assert len(p)==10
            q=np.array(p[1:5],float);rotation=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            images.append({'id':int(p[0]),'camera_id':int(p[8]),'name':Path(p[9].strip().replace('\\','/')).name,
                'R':rotation,'t':np.array(p[5:8],float)})
            observations=next(f,'').split()
            if observations:raise ValueError('This pipeline expects the supplied pose-only export; observations need explicit handling.')
    return cameras,images


def local_pose(record,transform):
    scale=np.cbrt(np.linalg.det(transform[:3,:3]));map_rotation=transform[:3,:3]/scale
    center=transform[:3,:3]@(-record['R'].T@record['t'])+transform[:3,3]
    rotation=record['R']@map_rotation.T
    return rotation,-rotation@center


def project(points,rotation,translation,K,distortion):
    camera=points@rotation.T+translation;z=camera[:,2]
    normalized=camera[:,:2]/np.where(abs(z)>1e-9,z,1e-9)[:,None];x,y=normalized.T
    k1,k2,p1,p2,k3,k4,k5,k6=distortion;r2=x*x+y*y
    radial=(1+k1*r2+k2*r2**2+k3*r2**3)/(1+k4*r2+k5*r2**2+k6*r2**3)
    u=x*radial+2*p1*x*y+p2*(r2+2*x*x);v=y*radial+p1*(r2+2*y*y)+2*p2*x*y
    return np.c_[K[0,0]*u+K[0,2],K[1,1]*v+K[1,2]],z


def scaled_camera(camera):
    scale=PARAMETERS['image_width']/camera['width'];K=camera['K'].copy();K[:2]*=scale
    return K,camera['distortion'].copy(),(PARAMETERS['image_width'],round(camera['height']*scale))


def features(path,size):
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if image is None:raise ValueError(f'Cannot read image: {path}')
    image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
    keypoints,descriptor=cv2.SIFT_create(nfeatures=PARAMETERS['sift_features']).detectAndCompute(image,None)
    xy=np.array([k.pt for k in keypoints],np.float32).reshape(-1,2)
    if descriptor is None:return xy,np.empty((0,128),np.float32)
    descriptor=np.sqrt(descriptor/np.maximum(descriptor.sum(1,keepdims=True),1e-12))
    return xy,descriptor


def attach_depth(keypoints,points,rotation,translation,K,distortion,size):
    camera=points@rotation.T+translation
    visible=(camera[:,2]>1)&np.all(abs(camera[:,:2])<camera[:,2,None]*2,axis=1)
    p=points[visible];uv,z=project(p,rotation,translation,K,distortion)
    inside=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<size[0]-.5)&(uv[:,1]>=0)&(uv[:,1]<size[1]-.5)
    uv,z=uv[inside],z[inside]
    pixel=np.rint(uv).astype(int);flat=pixel[:,1]*size[0]+pixel[:,0]
    order=np.lexsort((z,flat));_,first=np.unique(flat[order],return_index=True);keep=order[first]
    uv,z=uv[keep],z[keep]
    if len(uv)<5:return np.zeros(len(keypoints),bool),np.empty((0,3))
    distance,indices=cKDTree(uv).query(keypoints,k=5)
    depths=z[indices];valid=(distance[:,0]<=PARAMETERS['reference_depth_pixel_radius'])
    valid&=(distance[:,-1]<=PARAMETERS['reference_depth_pixel_radius']*2)
    valid&=np.ptp(depths,axis=1)<=PARAMETERS['reference_depth_spread_m']
    if not valid.any():return valid,np.empty((0,3))
    rays=cv2.undistortPoints(keypoints[valid,None,:].astype(float),K,distortion).reshape(-1,2)
    camera_points=np.c_[rays,np.ones(len(rays))]*np.median(depths[valid],axis=1)[:,None]
    return valid,(camera_points-translation)@rotation


def build_matcher(references):
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64))
    matcher.add([r['descriptors'] for r in references]);matcher.train();return matcher


def correspondences(query_xy,query_desc,references,matcher):
    if len(query_desc)<2:return np.empty((0,2)),np.empty((0,3)),[]
    neighbours=matcher.knnMatch(query_desc,k=4);votes=np.zeros(len(references))
    for matches in neighbours:
        seen=set()
        for m in matches:
            if m.imgIdx not in seen and m.distance<.6:votes[m.imgIdx]+=1/(m.distance+.05);seen.add(m.imgIdx)
    retrieved=np.argsort(-votes,kind='stable')[:PARAMETERS['retrieval_count']]
    best={};bf=cv2.BFMatcher(cv2.NORM_L2)
    for index in retrieved:
        reference=references[index];desc=reference['descriptors']
        if len(desc)<2:continue
        reverse=bf.match(desc,query_desc)
        for matches in bf.knnMatch(query_desc,desc,k=2):
            if len(matches)<2:continue
            a,b=matches
            if a.distance>=PARAMETERS['ratio_test']*b.distance or reverse[a.trainIdx].trainIdx!=a.queryIdx:continue
            if a.queryIdx not in best or a.distance<best[a.queryIdx][0]:best[a.queryIdx]=(a.distance,reference['points'][a.trainIdx])
    ids=sorted(best);xy=query_xy[ids];points=np.array([best[i][1] for i in ids]).reshape(-1,3)
    # A single physical location cannot inflate support through many nearby features.
    if len(points):
        _,keep=np.unique(np.floor(points/.2).astype(int),axis=0,return_index=True);xy,points=xy[keep],points[keep]
    ranking=[{'name':references[i]['name'],'votes':float(votes[i])} for i in retrieved]
    return xy,points,ranking


def estimate_pose(xy,points,K,distortion,size):
    held=(np.floor(xy[:,0]/150)+np.floor(xy[:,1]/150)).astype(int)%4==0
    if (~held).sum()<25 or held.sum()<PARAMETERS['minimum_validation_matches']:
        return {'status':'rejected_insufficient_correspondences','correspondences':len(xy)},None
    cv2.setRNGSeed(29)
    ok,rvec,tvec,inliers=cv2.solvePnPRansac(points[~held],xy[~held],K,distortion,iterationsCount=5000,
        reprojectionError=PARAMETERS['ransac_pixels'],confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None:return {'status':'rejected_pnp_failure','correspondences':len(xy)},None
    train=np.flatnonzero(~held);inliers=train[inliers.ravel()]
    rvec,tvec=cv2.solvePnPRefineLM(points[inliers],xy[inliers],K,distortion,rvec,tvec)
    rotation=cv2.Rodrigues(rvec)[0];translation=tvec.ravel();projected,z=project(points,rotation,translation,K,distortion)
    error=np.linalg.norm(projected-xy,axis=1);validation_fraction=float(np.mean((error[held]<=3)&(z[held]>0)))
    area=ConvexHull(xy[inliers]).volume/(size[0]*size[1]) if len(inliers)>=3 and np.linalg.matrix_rank(xy[inliers]-xy[inliers].mean(0))==2 else 0.
    accepted=(len(inliers)>=PARAMETERS['minimum_inliers'] and len(inliers)/len(train)>=PARAMETERS['minimum_inlier_fraction'] and
        validation_fraction>=PARAMETERS['minimum_validation_fraction'] and area>=PARAMETERS['minimum_image_area'] and np.all(z[inliers]>0))
    result={'status':'candidate_for_review' if accepted else 'rejected_quality_gate','correspondences':len(xy),
        'training_matches':len(train),'validation_matches':int(held.sum()),'ransac_inliers':len(inliers),
        'training_inlier_fraction':len(inliers)/len(train),'inlier_image_area_fraction':float(area),
        'validation_fraction_within_3px':validation_fraction,'validation_median_pixels':float(np.median(error[held])),
        'validation_p90_pixels':float(np.quantile(error[held],.9)),'inlier_median_pixels':float(np.median(error[inliers])),
        'world_to_camera_rotation':rotation.tolist(),'world_to_camera_translation':translation.tolist(),
        'camera_center_local_m':(-rotation.T@translation).tolist()}
    return result,{'xy':xy,'points':points,'held':held,'inliers':inliers,'errors':error}
