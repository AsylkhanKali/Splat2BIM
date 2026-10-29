"""Recompute image-space checks and verify preserved input/registration hashes."""
import json
import re

import numpy as np
from scipy.spatial import ConvexHull

from align_dates import ROOT
from camera_localization import project,PARAMETERS
from check_las_vertical import file_hash
from run_camera_localization import OUT


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['parameters']==PARAMETERS
    assert not set(protocol['query_names'])&set(protocol['reference_names'])
    assert protocol['minimum_chronological_index_gap']>PARAMETERS['query_exclusion_radius']
    for name,digest in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    files={f['path']:f['sha256'] for f in json.loads((OUT/'inputs.json').read_text())}
    previous=json.loads((ROOT/'results/object_recovery/inputs.json').read_text())
    preserved={f['path']:f['sha256'] for value in previous.values() for f in value['files']}
    for path in preserved.keys()&files.keys():assert preserved[path]==files[path],path
    for path,digest in {**preserved,**files}.items():assert file_hash(ROOT/path)==digest,path
    poses=json.loads((OUT/'poses.json').read_text());checks=[]
    assert [p['name'] for p in poses]==protocol['query_names']
    for p in poses:
        if 'world_to_camera_rotation' not in p:
            assert p['status'].startswith('rejected_');continue
        data=np.load(OUT/f"query_{p['query_index']:02d}_matches.npz")
        R=np.array(p['world_to_camera_rotation']);t=np.array(p['world_to_camera_translation'])
        np.testing.assert_allclose(R@R.T,np.eye(3),atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R),1,atol=1e-10)
        np.testing.assert_allclose(-R.T@t,p['camera_center_local_m'],atol=1e-10)
        xy,z=project(data['points'],R,t,np.array(p['intrinsics']),np.array(p['distortion']))
        errors=np.linalg.norm(xy-data['xy'],axis=1);held=data['held'];inliers=data['inliers']
        expected=(np.floor(data['xy'][:,0]/150)+np.floor(data['xy'][:,1]/150)).astype(int)%4==0
        np.testing.assert_array_equal(held,expected);assert not held[inliers].any()
        np.testing.assert_allclose(errors,data['errors'],atol=1e-9)
        fraction=np.mean((errors[held]<=PARAMETERS['ransac_pixels'])&(z[held]>0))
        np.testing.assert_allclose(fraction,p['validation_fraction_within_3px'])
        np.testing.assert_allclose(np.median(errors[held]),p['validation_median_pixels'])
        np.testing.assert_allclose(np.quantile(errors[held],.9),p['validation_p90_pixels'])
        area=ConvexHull(data['xy'][inliers]).volume/np.prod(p['image_size'])
        accepted=(len(inliers)>=PARAMETERS['minimum_inliers'] and len(inliers)/(~held).sum()>=PARAMETERS['minimum_inlier_fraction'] and
            held.sum()>=PARAMETERS['minimum_validation_matches'] and fraction>=PARAMETERS['minimum_validation_fraction'] and
            area>=PARAMETERS['minimum_image_area'] and np.all(z[inliers]>0))
        assert accepted==(p['status']=='candidate_for_review')
        checks.append({'query_index':p['query_index'],'status':p['status'],'training_validation_overlap':0})
    log=(OUT/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':count,'query_checks':checks,
        'camera_input_hashes_matched':len(files),'prior_input_and_registration_hashes_matched':len(preserved),
        'meaning':'Reprojection, split and provenance checks; not independent survey validation.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
