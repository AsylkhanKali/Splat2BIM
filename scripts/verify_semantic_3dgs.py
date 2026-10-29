"""Recompute rendering, label fusion, region counts and strict support evidence."""
import csv
import itertools
import json
import re

import cv2
import laspy
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_objects import dense_training
from align_0929 import apply
from check_las_vertical import file_hash
from check_static_structures import volume_mask
from prepare_semantic_3dgs import OUT,OLD,PREVIOUS,MODEL,SIZE
from gaussian_ortho import project_gaussians,render
from semantic_fusion import fuse
from align_object_recovery import load_fragments
from align_object_consensus import patch_checks
from object_consensus import descriptors,match,spatial_support,PARAMETERS


def main():
    protocol=json.loads((OUT/'segmentation_protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    for name,digest in protocol['model']['files'].items():assert file_hash(MODEL/name)==digest
    assert file_hash(PREVIOUS/'segmentation_protocol.json')==protocol['previous_protocol_sha256']
    for filename in ['protocol.json','segmentation_protocol.json']:
        prior=json.loads((PREVIOUS/filename).read_text())
        for name,digest in prior['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    original_protocol=json.loads((OLD/'segmentation_protocol.json').read_text())
    for name,digest in original_protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    prior=json.loads((ROOT/'results/object_recovery/inputs.json').read_text())
    preserved={f['path']:f['sha256'] for value in prior.values() for f in value['files']}
    for path,digest in preserved.items():assert file_hash(ROOT/path)==digest,path
    print('Preserved',len(preserved),'prior files',flush=True)
    result=json.loads((OUT/'summary.json').read_text());assert not result['registration_rerun']
    regions_path=PREVIOUS/'recognition_regions.json';assert file_hash(regions_path)==result['recognition_regions_sha256']
    regions=json.loads(regions_path.read_text())['regions']
    with (OUT/'recognition.csv').open() as f:recognition=list(csv.DictReader(f))
    checks=[]
    for date in DATES:
        folder=OUT/date;manifest=json.loads((folder/'segmentation.json').read_text());data=dict(np.load(folder/'source_labels.npz'))
        for path,digest in manifest['inputs'].items():assert file_hash(ROOT/path)==digest,path
        assert file_hash(folder/'source_labels.npz')==manifest['clouds']['source']['data_sha256']
        assert file_hash(folder/'render_context.npz')==manifest['render_context_sha256']
        assert file_hash(folder/'source_rgb.png')==manifest['image_sha256']
        drone=dict(np.load(OLD/date/'source_labels.npz'));previous=dict(np.load(PREVIOUS/date/'source_labels.npz'))
        for k in ['points','input_indices','original_ply_indices','probabilities']:np.testing.assert_array_equal(data[k],drone[k])
        probabilities=data['building_expert_probabilities'];assert np.isfinite(probabilities).all() and np.all((probabilities>=0)&(probabilities<=1))
        np.testing.assert_allclose(probabilities.sum(1),1,atol=1e-6)
        labels,confidence,origin=fuse(drone,probabilities)
        for expected,k in [(labels,'labels'),(confidence,'confidence'),(origin,'label_origin')]:np.testing.assert_array_equal(expected,data[k])
        benchmark,_,_=samples(date);training,_,training_ids=dense_training(date,benchmark)
        np.testing.assert_array_equal(data['points'],training[data['input_indices']])
        np.testing.assert_array_equal(data['original_ply_indices'],training_ids[data['input_indices']])
        v=PlyData.read(ROOT/f'assets/{date}/{date}.ply')['vertex'].data
        p=np.c_[v['x'],v['y'],v['z']].astype(float);indices=np.arange(len(v));even=indices%2==0
        allowed=even&(v['opacity']>np.log(1/254))&np.all((p>=training.min(0))&(p<=training.max(0)),axis=1)
        allowed&=np.floor(p/.4).sum(1).astype(int)%4!=0
        ids=np.load(folder/'render_context.npz')['original_ply_indices'];np.testing.assert_array_equal(ids,indices[allowed])
        assert np.isin(data['original_ply_indices'],ids).all() and len(ids)==manifest['render_context_count']
        metadata=manifest['clouds']['source'];old_metadata=json.loads((OLD/date/'segmentation.json').read_text())['clouds']['source']
        np.testing.assert_array_equal(metadata['frame'],old_metadata['frame']);np.testing.assert_array_equal(metadata['center'],old_metadata['center'])
        np.testing.assert_allclose(metadata['origin_xy'],np.array(old_metadata['origin_xy'])-old_metadata['pixel_size_source_units']/4)
        assert metadata['pixel_size_source_units']==old_metadata['pixel_size_source_units']/2
        local=(data['points']-metadata['center'])@np.array(metadata['frame'])
        uv=np.rint((local[:,:2]-metadata['origin_xy'])/metadata['pixel_size_source_units']).astype(int);uv[:,1]=SIZE-1-uv[:,1]
        np.testing.assert_array_equal(uv,data['render_uv']);assert np.all((uv>=0)&(uv<SIZE))
        rgb,alpha=render(project_gaussians(v[ids],metadata,SIZE),SIZE)
        np.testing.assert_array_equal(rgb,cv2.cvtColor(cv2.imread(str(folder/'source_rgb.png')),cv2.COLOR_BGR2RGB))
        np.testing.assert_array_equal(np.uint8(alpha*255),cv2.imread(str(folder/'source_alpha.png'),cv2.IMREAD_GRAYSCALE))
        row=next(r for r in result['dates'] if r['date']==date)
        matrix_path=ROOT/f'results/object_recovery/{date}/selection.json';assert file_hash(matrix_path)==row['evaluation_matrix_sha256']
        world=apply(data['points'],np.array(json.loads(matrix_path.read_text())['matrix_local']))
        masks={r['id']:volume_mask(world,r) for r in regions};masks['all_visible']=np.ones(len(world),bool)
        for method,cloud in [('previous_fusion',previous),('gaussian_render',data)]:
            for region,mask in masks.items():
                record=next(r for r in recognition if r['date']==date and r['method']==method and r['region']==region)
                assert int(record['count'])==int(mask.sum());assert int(record['built_points'])==int(np.sum(cloud['labels'][mask]==1))
                if mask.any():np.testing.assert_allclose(float(record['built_fraction']),np.mean(cloud['labels'][mask]==1),atol=1e-12)
        assert row['roof_count']==masks['roof'].sum() and row['vegetation_count']==masks['vegetation'].sum()
        for key,cloud in [('previous',previous),('new',data)]:
            for region in ['roof','vegetation']:np.testing.assert_allclose(row[f'{key}_{region}_fraction'],np.mean(cloud['labels'][masks[region]]==1),atol=1e-12)
        path=ROOT/f'results/strict_diagnostics/{date}.json';assert file_hash(path)==row['strict_diagnostic_sha256'];diagnostic=json.loads(path.read_text())
        assert diagnostic['parameters']==PARAMETERS and diagnostic['strict_forward_fraction']==.8
        assert file_hash(PREVIOUS/date/'audit.json')==diagnostic['parent_audit_sha256']
        audit=json.loads((PREVIOUS/date/'audit.json').read_text());run=next(r for r in audit['runs'] if r['method']=='semantic_fusion')
        matrix=np.array(run['matrix_local']);np.testing.assert_array_equal(matrix,diagnostic['matrix_local'])
        sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf);assigned=match(sd,td,matrix);assigned_ids={(r['source_id'],r['target_id']) for r in assigned}
        recomputed=patch_checks(run['final_pair_checks'],sf,tf,matrix)
        strict=[p for p in recomputed if p['fraction_within_1m']>=.8 and p['reverse_fraction_within_1m']>=.5 and (p['source_id'],p['target_id']) in assigned_ids]
        assert len(strict)==diagnostic['strict_pair_count']==row['strict_overlap_pairs']
        assert [(p['source_id'],p['target_id']) for p in strict]==[(p['source_id'],p['target_id']) for p in diagnostic['strict_pairs']]
        support=spatial_support(strict,sd,td,matrix);assert len(support)==run['strict_forward_080_spatial_support_count']
        assert row['previous_strict_pass']==(len(support)>=3)
        A=apply(sd['centers'],matrix);B=td['centers'];passed=0
        for triple in itertools.combinations(strict,3):
            a=A[[p['source_index'] for p in triple]];b=B[[p['target_index'] for p in triple]]
            second=[np.linalg.svd(c-c.mean(0),compute_uv=False)[1] for c in [a,b]]
            distance=min(np.linalg.norm(c[i]-c[j]) for c in [a,b] for i,j in [(0,1),(0,2),(1,2)])
            passed+=bool(min(second)>=2 and distance>=10)
        assert passed==diagnostic['admissible_bank_triangles']==row['admissible_triangles']
        if date in ['0708','0723']:
            las=laspy.read(ROOT/f'assets/{date}/{date}.las');tree=cKDTree(np.c_[las.x,las.y,las.z]-ORIGIN)
            P={c['id']:c['points']/sf['normalization_scale']+sf['normalization_center'] for c in sf['candidates']}
            for pair,check in zip(recomputed,diagnostic['patch_diagnostics']):
                for k in ['median_m','p90_m','fraction_within_1m','reverse_fraction_within_1m']:np.testing.assert_allclose(pair[k],check[k],atol=1e-12)
                source=apply(P[pair['source_id']],matrix);fraction=np.mean(tree.query(source)[0]<1)
                np.testing.assert_allclose(fraction,check['source_full_las_within_1m'],atol=1e-12)
        checks.append({'date':date,'render_context_count':len(ids),'render_reproduced_exactly':True,'strict_overlap_pairs':len(strict),'admissible_triangles':passed})
        print('Verified',date,flush=True)
    log=(OUT/'tests.log').read_text();assert log.rstrip().endswith('OK');test_count=int(re.search(r'Ran (\d+) tests',log)[1])
    validation={'all_checks_passed':True,'unit_tests_passed':test_count,'prior_files_preserved':len(preserved),
        'model_hashes_verified':True,'held_out_centers_and_odd_control_records_excluded':True,'region_counts_and_fusion_recomputed':True,
        'strict_overlap_and_all_bank_triangles_recomputed':True,'neural_inference_repeated_in_verifier':False,'date_checks':checks,
        'verification_script_sha256':file_hash(ROOT/'scripts/verify_semantic_3dgs.py'),
        'supporting_code_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['export_3dgs.py','semantic_projection.py','align_objects.py','align_dates.py','diagnose_strict_support.py','report_semantic_3dgs.py']},
        'meaning':'Reproducibility and consistency of development diagnostics; no new alignment result or independent semantic accuracy claim.'}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2)+'\n');print('All checks passed',flush=True)


if __name__=='__main__':main()
