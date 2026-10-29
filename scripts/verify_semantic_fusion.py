"""Verify pre-ICP rankings, candidate identities, acceptance and held-out errors."""
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
from align_object_consensus import patch_checks
from object_consensus import descriptors,match,spatial_support
from check_las_vertical import file_hash
from check_static_structures import volume_mask
from semantic_matching import SemanticMatcher,choose_candidate
from semantic_initial import INITIAL_PARAMETERS,score_pool,choose_starts
from align_semantic_fusion import OUT,LABELS,OLD_LABELS
from semantic_structures import correspondence_bank
from semantic_fusion import fuse


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    parent=json.loads((ROOT/'results/semantic_initial/validation.json').read_text());assert parent['all_checks_passed']
    segmentation=json.loads((OUT/'segmentation_protocol.json').read_text())
    for name,digest in segmentation['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    for name,digest in segmentation['model']['files'].items():assert file_hash(ROOT/'models/loveda_segformer_b2'/name)==digest
    with (OUT/'recognition.csv').open() as f:recognition=list(csv.DictReader(f))
    recognition_regions=json.loads((OUT/'recognition_regions.json').read_text())['regions']
    old=json.loads((ROOT/'results/object_recovery/inputs.json').read_text())
    preserved={f['path']:f['sha256'] for value in old.values() for f in value['files']}
    for path,digest in preserved.items():assert file_hash(ROOT/path)==digest,path
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions'];checks=[]
    for date in DATES:
        folder=OUT/date;audit=json.loads((folder/'audit.json').read_text());saved=np.load(folder/'initial_scores.npz')
        for path,digest in {**audit['input_pool_hashes'],**audit['label_hashes']}.items():assert file_hash(ROOT/path)==digest
        current_path=ROOT/audit['comparison_input']['path'];assert file_hash(current_path)==audit['comparison_input']['sha256']
        source=dict(np.load(LABELS/date/'source_labels.npz'));dest=dict(np.load(LABELS/date/'target_labels.npz'));matcher=SemanticMatcher(source,dest)
        old_source=dict(np.load(OLD_LABELS/date/'source_labels.npz'));old_dest=dict(np.load(OLD_LABELS/date/'target_labels.npz'));old_matcher=SemanticMatcher(old_source,old_dest)
        manifest=json.loads((LABELS/date/'segmentation.json').read_text())
        for path,digest in manifest['inputs'].items():assert file_hash(ROOT/path)==digest
        for side,old_labels,new_labels in [('source',old_source,source),('target',old_dest,dest)]:
            np.testing.assert_array_equal(old_labels['points'],new_labels['points'])
            np.testing.assert_array_equal(old_labels['input_indices'],new_labels['input_indices'])
            assert file_hash(LABELS/date/f'{side}_labels.npz')==manifest['clouds'][side]['data_sha256']
            predicted,_,_=fuse(old_labels,new_labels['building_expert_probabilities'].astype(float))
            mismatch=predicted!=new_labels['labels']
            assert np.all(abs(new_labels['building_expert_probabilities'][mismatch,1].astype(float)-.6)<.0003) # float16 storage boundary
            assert np.all(np.isin(new_labels['labels'],range(7)))
        benchmark,rgb,held=samples(date);train,colors,ids=dense_training(date,benchmark)
        np.testing.assert_array_equal(source['points'],train[source['input_indices']]);assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
        target_rgb=np.c_[las.red,las.green,las.blue].astype(float);target_rgb/=65535 if target_rgb.max()>255 else 255
        tree=cKDTree(downsample(target,2.));full=cKDTree(target);small=downsample(train,.06)
        score_source=downsample(train,.12);rng=np.random.default_rng(29)
        if len(score_source)>INITIAL_PARAMETERS['geometry_score_points']:score_source=score_source[rng.choice(len(score_source),INITIAL_PARAMETERS['geometry_score_points'],replace=False)]
        np.testing.assert_array_equal(score_source,saved['score_source'])
        initial=np.r_[np.load(ROOT/f'results/object_consensus/{date}/initial_matrices.npz')['multi'],np.load(ROOT/f'results/object_recovery/{date}/initial_matrices.npz')['matrices']]
        initial=initial[[scale_admissible(m,audit['extent_scale_prior']) for m in initial]]
        _,unique=np.unique(np.round(initial,8).reshape(len(initial),16),axis=0,return_index=True);initial=initial[np.sort(unique)]
        np.testing.assert_array_equal(initial,saved['matrices'])
        scores=score_pool(score_source,tree,matcher,initial);np.testing.assert_allclose(scores,saved['scores'],atol=1e-9,equal_nan=True)
        anchors=score_source[::max(1,len(score_source)//20)]
        old_scores=score_pool(score_source,tree,old_matcher,initial);np.testing.assert_allclose(old_scores,saved['old_scores'],atol=1e-9,equal_nan=True)
        for method,semantic in [('geometry_bank',False),('old_semantic_bank',True),('semantic_fusion',True)]:
            assert choose_starts(initial,old_scores if method=='old_semantic_bank' else scores,anchors,semantic)==audit['initial_starts'][method]
        sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf)
        for c in audit['coarse_candidates']:
            np.testing.assert_array_equal(c['initial_matrix'],initial[c['pool_index']])
            expected,neighbors=correspondence_bank(initial,c['pool_index'],anchors,sd,td)
            assert expected==c['initial_pairs'];assert neighbors==c['neighbor_pool_indices']
            assert match(sd,td,np.array(c['initial_matrix']))==c['representative_pairs']
            matrix=np.array(c['matrix']);np.testing.assert_allclose(np.median(tree.query(apply(small,matrix))[0]),c['geometry_score_m'],atol=1e-9)
            value,_=matcher.score(matrix)
            if value is None:assert c['semantic_score_m'] is None
            else:np.testing.assert_allclose(value,c['semantic_score_m'],atol=1e-9)
            old_value,_=old_matcher.score(matrix);np.testing.assert_allclose(old_value,c['old_semantic_score_m'],atol=1e-9)
            assert len(spatial_support(surviving_pairs(c['initial_pairs'],sd,td,matrix),sd,td,matrix))==c['geometric_support_count']
            assert scale_admissible(matrix,audit['extent_scale_prior'])==c['scale_admissible']
        matrices={'current_automatic':np.array(json.loads(current_path.read_text())['matrix_local'])}
        for run in audit['runs']:
            indices=run['coarse_indices'];candidates=[audit['coarse_candidates'][i] for i in indices]
            assert [c['pool_index'] for c in candidates]==audit['initial_starts'][run['method']]
            selection_rows=[{**c,'semantic_score_m':c['old_semantic_score_m'] if run['method']=='old_semantic_bank' else c['semantic_score_m']} for c in candidates]
            selected=choose_candidate(selection_rows,run['method']!='geometry_bank')
            if selected is None:assert 'matrix_local' not in run;continue
            assert indices[selected]==run['selected_candidate_index'];c=candidates[selected];matrix=np.array(run['matrix_local']);matrices[run['method']]=matrix
            np.testing.assert_allclose(matrix[3],[0,0,0,1],atol=1e-12);scale=np.cbrt(np.linalg.det(matrix[:3,:3]));assert scale>0
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3)*scale**2,atol=1e-8)
            error,near=full.query(apply(benchmark[~held],matrix));color_error=float(np.median(np.linalg.norm(rgb[~held]-target_rgb[near],axis=1)))
            for key,value in summary(error).items():np.testing.assert_allclose(value,run['training_final'][key],atol=1e-9)
            np.testing.assert_allclose(color_error,run['training_color_difference'],atol=1e-9)
            patches=patch_checks(c['initial_pairs'],sf,tf,matrix);survived={(p['source_id'],p['target_id']):p for p in surviving_pairs(c['initial_pairs'],sd,td,matrix)}
            valid=[{**p,'center_distance_m':survived[(p['source_id'],p['target_id'])]['center_distance_m']} for p in patches
                if (p['source_id'],p['target_id']) in survived and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
            support=spatial_support(valid,sd,td,matrix);strict=spatial_support([p for p in valid if p['fraction_within_1m']>=.8],sd,td,matrix)
            assert [(p['source_id'],p['target_id']) for p in support]==[(p['source_id'],p['target_id']) for p in run['final_spatial_support']]
            assert len(strict)==run['strict_forward_080_spatial_support_count']
            expected='rejected_training_alignment' if np.median(error)>.5 or color_error>.35 or not scale_admissible(matrix,audit['extent_scale_prior']) else ('candidate_for_review' if support else 'rejected_object_correspondences')
            assert run['status']==expected
        held_data=np.load(folder/'held_out_distances.npz');old_data=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz')
        np.testing.assert_array_equal(held_data['source_points'],benchmark[held]);np.testing.assert_array_equal(held_data['fixed_coverage'],old_data['fixed_coverage'])
        with (folder/'static_structures.csv').open() as f:static=list(csv.DictReader(f))
        controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
        for method,matrix in matrices.items():
            error=full.query(apply(benchmark[held],matrix))[0];np.testing.assert_allclose(error,held_data[method],atol=1e-9)
            for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',held_data['fixed_coverage'])]:
                row=next(r for r in audit['comparison'] if r['method']==method and r['scope']==scope)
                for key,value in summary(error[mask]).items():np.testing.assert_allclose(value,row[key],atol=1e-9)
            for region in regions:
                local=cKDTree(target[volume_mask(target,region)])
                for scope in ['unseen_records','spatial_holdout']:
                    points=controls[region['id']+'_'+scope];row=next(r for r in static if r['method']==method and r['region']==region['id'] and r['scope']==scope)
                    assert int(row['count'])==len(points)
                    if len(points):
                        for key,value in summary(local.query(apply(points,matrix))[0]).items():np.testing.assert_allclose(value,float(row[key]),atol=1e-9)
        for side,old_labels,new_labels in [('source',old_source,source),('target',old_dest,dest)]:
            points=apply(source['points'],matrices['current_automatic']) if side=='source' else dest['points']
            for region in recognition_regions:
                mask=volume_mask(points,region)
                for method,data in [('previous',old_labels),('fusion',new_labels)]:
                    row=next(r for r in recognition if r['date']==date and r['side']==side and r['region']==region['id'] and r['method']==method)
                    assert int(row['count'])==int(mask.sum());assert int(row['built_points'])==int(np.sum(data['labels'][mask]==1))
                    if mask.any():np.testing.assert_allclose(float(row['built_fraction']),np.mean(data['labels'][mask]==1),atol=1e-12)
        checks.append({'date':date,'all_initial_scores_recomputed':len(initial),'matrices_recomputed':len(matrices),'withheld_points':int(held.sum())})
        print('Verified',date,flush=True)
    log=(OUT/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':count,'model_hashes_verified':True,'recognition_counts_recomputed':True,'pre_icp_banks_recomputed':True,'prior_file_hashes_matched':len(preserved),'date_checks':checks,
        'meaning':'Recomputed rankings, gates and held-out distances; not independent survey or semantic ground truth.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed',flush=True)


if __name__=='__main__':main()
