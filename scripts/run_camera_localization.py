"""Explore single-image pose estimation; query poses are scored only after fitting."""
import json
import time

import cv2
import laspy
import numpy as np
from scipy.spatial.transform import Rotation

from align_dates import ROOT,ORIGIN
from check_las_vertical import file_hash,write_csv
from camera_localization import PARAMETERS,read_model,local_pose,scaled_camera,features,attach_depth,build_matcher,correspondences,estimate_pose

OUT=ROOT/'results/camera_localization';DATA=ROOT/'assets/0929'


def split_images(available):
    query_indices=np.linspace(40,len(available)-40,12,dtype=int)
    ref_indices=[i for i in range(0,len(available),PARAMETERS['reference_stride'])
        if np.min(abs(query_indices-i))>PARAMETERS['query_exclusion_radius']]
    return [available[i] for i in query_indices],[available[i] for i in ref_indices]


def main():
    cv2.setNumThreads(2);cv2.setRNGSeed(29);OUT.mkdir(parents=True,exist_ok=True)
    cameras,records=read_model(DATA/'colmap');by_name={r['name']:r for r in records}
    available=sorted(r['name'] for r in records if (DATA/'photos'/r['name']).exists())
    queries,reference_names=split_images(available)
    reference_indices=[available.index(n) for n in reference_names];query_indices=[available.index(n) for n in queries]
    assert not set(queries)&set(reference_names)
    assert min(abs(a-b) for a in reference_indices for b in query_indices)>PARAMETERS['query_exclusion_radius']
    base_path=ROOT/'results/0929_baseline/alignment.json';base=json.loads(base_path.read_text());transform=np.array(base['sparse_to_las_local'])
    files=[DATA/'0929.las',DATA/'colmap/cameras.txt',DATA/'colmap/images.txt',base_path]+[DATA/'photos'/name for name in queries+reference_names]
    manifest=[{'path':str(p.relative_to(ROOT)),'sha256':file_hash(p)} for p in files]
    protocol={'parameters':PARAMETERS,'query_names':queries,'reference_names':reference_names,
        'reference_count':len(reference_names),'query_count':len(queries),'same_image_in_reference_index':False,
        'minimum_chronological_index_gap':min(abs(a-b) for a in reference_indices for b in query_indices),
        'query_pose_used_for_retrieval_or_estimation':False,'query_intrinsics_given':True,
        'underlying_LAS_and_COLMAP_may_include_query_photos':True,
        'opencv_version':cv2.__version__,'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['camera_localization.py','run_camera_localization.py']}}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');(OUT/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
    las=laspy.read(DATA/'0929.las');points=np.c_[las.x,las.y,las.z]-ORIGIN
    refs=[];cache=OUT/'reference_features';cache.mkdir(exist_ok=True)
    cache_protocol={'parameters':PARAMETERS,'inputs':manifest,'opencv_version':cv2.__version__,
        'source_sha256':protocol['source_sha256']['camera_localization.py']}
    cache_manifest=cache/'protocol.json'
    if cache_manifest.exists() and json.loads(cache_manifest.read_text())!=cache_protocol:
        raise ValueError('Reference cache is stale. Preserve this experiment and use a fresh output directory.')
    cache_manifest.write_text(json.dumps(cache_protocol,indent=2)+'\n')
    for n,name in enumerate(reference_names):
        rec=by_name[name];camera=cameras[rec['camera_id']];K,distortion,size=scaled_camera(camera)
        path=cache/(name+'.npz')
        if path.exists():
            saved=np.load(path);reference={'name':name,'descriptors':saved['descriptors'],'points':saved['points']}
        else:
            xy,descriptor=features(DATA/'photos'/name,size);rotation,translation=local_pose(rec,transform)
            keep,world=attach_depth(xy,points,rotation,translation,K,distortion,size)
            np.savez_compressed(path,descriptors=descriptor[keep],points=world,xy=xy[keep])
            reference={'name':name,'descriptors':descriptor[keep],'points':world}
        refs.append(reference);print('Reference',n+1,'/',len(reference_names),name,len(reference['points']),flush=True)
    matcher=build_matcher(refs);rows=[];poses=[]
    for index,name in enumerate(queries):
        begin=time.monotonic();rec=by_name[name];camera=cameras[rec['camera_id']];K,distortion,size=scaled_camera(camera)
        xy,descriptor=features(DATA/'photos'/name,size)
        pixels,world,retrieved=correspondences(xy,descriptor,refs,matcher)
        result,check=estimate_pose(pixels,world,K,distortion,size)
        result.update(name=name,query_index=index,query_features=len(xy),retrieved=retrieved,
            intrinsics=K.tolist(),distortion=distortion.tolist(),image_size=list(size),seconds=time.monotonic()-begin)
        # Evaluation below cannot influence the fitted pose or quality-gate decision.
        if check is not None:
            truth_R,truth_t=local_pose(rec,transform);truth_center=-truth_R.T@truth_t
            fitted_R=np.array(result['world_to_camera_rotation']);center=np.array(result['camera_center_local_m'])
            result['center_difference_to_colmap_m']=float(np.linalg.norm(center-truth_center))
            result['rotation_difference_to_colmap_degrees']=float(np.rad2deg(Rotation.from_matrix(fitted_R@truth_R.T).magnitude()))
            result['reference_camera_center_local_m']=truth_center.tolist()
            np.savez_compressed(OUT/f'query_{index:02d}_matches.npz',**check)
        poses.append(result);(OUT/f'query_{index:02d}.json').write_text(json.dumps(result,indent=2)+'\n')
        rows.append({k:result.get(k) for k in ['query_index','name','status','correspondences','ransac_inliers',
            'validation_matches','validation_fraction_within_3px','validation_median_pixels','validation_p90_pixels',
            'center_difference_to_colmap_m','rotation_difference_to_colmap_degrees','seconds']})
        print('Query',index,result['status'],'matches',len(pixels),'center',result.get('center_difference_to_colmap_m'),flush=True)
    write_csv(OUT/'comparison.csv',rows);(OUT/'poses.json').write_text(json.dumps(poses,indent=2)+'\n')
    for f in manifest:assert file_hash(ROOT/f['path'])==f['sha256']
    print('Finished',len(poses),'queries; accepted',sum(p['status']=='candidate_for_review' for p in poses),flush=True)


if __name__=='__main__':main()
