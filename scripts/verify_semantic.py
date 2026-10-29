"""Recompute semantic selection, held-out distances and preserved input hashes."""
import csv
import json
import re

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES,samples
from align_0929 import apply,downsample,summary
from align_objects import dense_training
from align_object_recovery import load_fragments,surviving_pairs,scale_admissible
from object_consensus import descriptors,spatial_support
from check_las_vertical import file_hash
from check_static_structures import volume_mask
from semantic_matching import SemanticMatcher,choose_candidate
from semantic_projection import PARAMETERS,GROUPS
from prepare_semantic_clouds import OUT,MODEL


def main():
    for filename in ['alignment_protocol.json','segmentation_protocol.json']:
        protocol=json.loads((OUT/filename).read_text());assert protocol['parameters']==PARAMETERS
        for name,digest in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    model=json.loads((MODEL/'provenance.json').read_text())
    for name,digest in model['files'].items():assert file_hash(MODEL/name)==digest
    old=json.loads((ROOT/'results/object_recovery/inputs.json').read_text())
    preserved={f['path']:f['sha256'] for value in old.values() for f in value['files']}
    for path,digest in preserved.items():assert file_hash(ROOT/path)==digest,path
    checks=[]
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions']
    for date in DATES:
        folder=OUT/date;audit=json.loads((folder/'audit.json').read_text());seg=json.loads((folder/'segmentation.json').read_text())
        for path,digest in {**audit['candidate_inputs'],**seg['inputs']}.items():assert file_hash(ROOT/path)==digest
        comparison=audit['comparison_input'];assert file_hash(ROOT/comparison['path'])==comparison['sha256']
        source=dict(np.load(folder/'source_labels.npz'));dest=dict(np.load(folder/'target_labels.npz'))
        benchmark,rgb,held=samples(date);train,_,ids=dense_training(date,benchmark)
        np.testing.assert_array_equal(train[source['input_indices']],source['points'])
        np.testing.assert_array_equal(ids[source['input_indices']],source['original_ply_indices'])
        assert np.all(np.floor(source['points']/.4).sum(1).astype(int)%4!=0)
        for side,data in [('source',source),('target',dest)]:
            assert file_hash(folder/f'{side}_labels.npz')==seg['clouds'][side]['data_sha256']
            assert np.all(data['labels'][data['confidence']<PARAMETERS['confidence']]==0)
            np.testing.assert_allclose(data['probabilities'].astype(float).sum(1),1,atol=.001)
            assert len(np.unique(data['input_indices']))==len(data['input_indices'])
            counts=np.bincount(data['labels'],minlength=len(GROUPS))
            assert dict(zip(GROUPS,counts.tolist()))==seg['clouds'][side]['class_counts']
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
        np.testing.assert_array_equal(dest['points'],target[dest['input_indices']])
        tree=cKDTree(target);small=downsample(train,.06);small_tree=cKDTree(downsample(target,2.))
        sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf);matcher=SemanticMatcher(source,dest)
        for c in audit['candidates']:
            matrix=np.array(c['matrix']);value,detail=matcher.score(matrix)
            if value is None:assert c['semantic_score_m'] is None
            else:np.testing.assert_allclose(value,c['semantic_score_m'],atol=1e-9)
            np.testing.assert_allclose(np.median(small_tree.query(apply(small,matrix))[0]),c['geometry_score_m'],atol=1e-9)
            assert len(spatial_support(surviving_pairs(c['initial_pairs'],sd,td,matrix),sd,td,matrix))==c['geometric_support_count']
            assert scale_admissible(matrix,audit['extent_scale_prior'])==c['scale_admissible']
        matrices={'current_automatic':np.array(json.loads((ROOT/comparison['path']).read_text())['matrix_local'])}
        data=np.load(folder/'held_out_distances.npz');previous=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz')
        np.testing.assert_array_equal(data['source_points'],benchmark[held]);np.testing.assert_array_equal(data['fixed_coverage'],previous['fixed_coverage'])
        for run in audit['runs']:
            selected=choose_candidate(audit['candidates'],run['method']=='semantic_rerank')
            if selected is None:assert 'matrix_local' not in run;continue
            assert selected==run['selected_candidate_index'];matrix=np.array(run['matrix_local']);matrices[run['method']]=matrix
            np.testing.assert_allclose(matrix[3],[0,0,0,1],atol=1e-12)
            scale=np.cbrt(np.linalg.det(matrix[:3,:3]));assert scale>0
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3)*scale**2,atol=1e-8)
            error=tree.query(apply(benchmark[~held],matrix))[0]
            for key,value in summary(error).items():np.testing.assert_allclose(value,run['training_final'][key],atol=1e-9)
        with (folder/'static_structures.csv').open() as f:static=list(csv.DictReader(f))
        controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
        for method,matrix in matrices.items():
            error=tree.query(apply(benchmark[held],matrix))[0];np.testing.assert_allclose(error,data[method],atol=1e-9)
            for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',data['fixed_coverage'])]:
                row=next(r for r in audit['comparison'] if r['method']==method and r['scope']==scope)
                for key,value in summary(error[mask]).items():np.testing.assert_allclose(value,row[key],atol=1e-9)
            for region in regions:
                local=cKDTree(target[volume_mask(target,region)])
                for scope in ['unseen_records','spatial_holdout']:
                    points=controls[region['id']+'_'+scope];row=next(r for r in static if r['method']==method and r['region']==region['id'] and r['scope']==scope)
                    assert int(row['count'])==len(points)
                    if len(points):
                        for key,value in summary(local.query(apply(points,matrix))[0]).items():np.testing.assert_allclose(value,float(row[key]),atol=1e-9)
        checks.append({'date':date,'recomputed_matrices':len(matrices),'semantic_training_points':len(source['points']),
            'withheld_points':int(held.sum()),'training_tile_leakage_count':0})
        print('Verified',date,flush=True)
    log=(OUT/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':count,'prior_file_hashes_matched':len(preserved),
        'model_hashes_verified':True,'date_checks':checks,'meaning':'Numerical/provenance checks, not semantic ground truth or survey validation.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed',flush=True)


if __name__=='__main__':main()
