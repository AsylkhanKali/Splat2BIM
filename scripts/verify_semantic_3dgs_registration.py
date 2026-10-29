"""Recompute the seven-date semantic registration rankings, gates and metrics."""
import csv
import itertools
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
from object_consensus import descriptors,match,spatial_support,PARAMETERS as OBJECT_PARAMETERS
from check_las_vertical import file_hash
from check_static_structures import volume_mask
from semantic_matching import SemanticMatcher,choose_candidate
from semantic_initial import INITIAL_PARAMETERS,score_pool,choose_starts
from semantic_structures import correspondence_bank,STRUCTURE_PARAMETERS
from semantic_projection import PARAMETERS,GROUPS
from align_semantic_3dgs import OUT,LABELS,PREVIOUS,METHOD


def verify_evidence(record,pairs,sf,tf,matrix):
    sd,td=descriptors(sf),descriptors(tf);patches=patch_checks(pairs,sf,tf,matrix)
    for new,old in zip(patches,record['final_pair_checks']):
        for k in ['median_m','p90_m','fraction_within_1m','reverse_fraction_within_1m']:np.testing.assert_allclose(new[k],old[k],atol=1e-12)
    assert len(patches)==len(record['final_pair_checks'])
    survived={(p['source_id'],p['target_id']):p for p in surviving_pairs(pairs,sd,td,matrix)}
    valid=[{**p,'center_distance_m':survived[(p['source_id'],p['target_id'])]['center_distance_m']} for p in patches
        if (p['source_id'],p['target_id']) in survived and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
    strict_pairs=[p for p in valid if p['fraction_within_1m']>=.8]
    identities=lambda rows:[(p['source_id'],p['target_id']) for p in rows]
    support=spatial_support(valid,sd,td,matrix);strict=spatial_support(strict_pairs,sd,td,matrix)
    assert identities(support)==identities(record['final_spatial_support'])
    assert identities(strict_pairs)==identities(record['strict_overlap_pairs'])
    assert len(strict)==record['strict_forward_080_spatial_support_count']
    A=apply(sd['centers'],matrix);B=td['centers'];triples=list(itertools.combinations(strict_pairs,3))
    assert len(triples)==len(record['strict_triangles']);passes=0
    for triple,saved in zip(triples,record['strict_triangles']):
        assert identities(triple)==[tuple(p) for p in saved['identities']]
        passed=True
        for side,points,key in [('source',A,'source_index'),('target',B,'target_index')]:
            a=points[[p[key] for p in triple]];second=np.linalg.svd(a-a.mean(0),compute_uv=False)[1]
            spacing=min(np.linalg.norm(a[i]-a[j]) for i,j in itertools.combinations(range(3),2))
            np.testing.assert_allclose(second,saved[side+'_second_axis_m'],atol=1e-10)
            np.testing.assert_allclose(spacing,saved[side+'_min_spacing_m'],atol=1e-10)
            passed&=second>=2 and spacing>=10
        assert bool(passed)==saved['passes'];passes+=bool(passed)
    assert passes==record['admissible_strict_triangles']
    return support


def main():
    protocol=json.loads((OUT/'protocol.json').read_text());parent=json.loads((PREVIOUS/'protocol.json').read_text())
    assert protocol['initial_parameters']==parent['initial_parameters']==INITIAL_PARAMETERS
    assert protocol['semantic_parameters']==parent['semantic_parameters']==PARAMETERS
    assert protocol['structure_parameters']==parent['structure_parameters']==STRUCTURE_PARAMETERS
    assert protocol['object_parameters']==OBJECT_PARAMETERS and protocol['strict_forward_fraction']==.8
    for name,digest in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    assert file_hash(PREVIOUS/'protocol.json')==protocol['parent_protocol_sha256']
    assert file_hash(LABELS/'segmentation_protocol.json')==protocol['label_protocol_sha256']
    segmentation=json.loads((LABELS/'segmentation_protocol.json').read_text())
    for name,digest in segmentation['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    for name,digest in segmentation['model']['files'].items():assert file_hash(ROOT/'models/loveda_segformer_b2'/name)==digest
    original=json.loads((ROOT/'results/object_recovery/inputs.json').read_text())
    preserved={f['path']:f['sha256'] for value in original.values() for f in value['files']}
    for path,digest in preserved.items():assert file_hash(ROOT/path)==digest,path
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions'];checks=[]
    report=json.loads((OUT/'summary.json').read_text());diagnostics=json.loads((OUT/'strict_diagnostics.json').read_text())
    assert file_hash(OUT/'strict_diagnostics.json')==report['strict_diagnostic_sha256']
    for n,h in report['source_sha256'].items():assert file_hash(ROOT/'scripts'/n)==h
    with (OUT/'summary.csv').open() as f:report_rows=list(csv.DictReader(f))
    for date in DATES:
        folder=OUT/date;audit=json.loads((folder/'audit.json').read_text());saved=np.load(folder/'initial_scores.npz');run=audit['run']
        assert file_hash(folder/'audit.json')==report['audit_sha256'][date]
        for path,digest in audit['inputs'].items():assert file_hash(ROOT/path)==digest,path
        source=dict(np.load(LABELS/date/'source_labels.npz'));dest=dict(np.load(PREVIOUS/date/'target_labels.npz'));matcher=SemanticMatcher(source,dest)
        for base,side in [(LABELS,'source'),(PREVIOUS,'target')]:
            manifest=json.loads((base/date/'segmentation.json').read_text())
            assert file_hash(base/date/f'{side}_labels.npz')==manifest['clouds'][side]['data_sha256']
        assert audit['shared_classes']==[GROUPS[c[0]] for c in matcher.classes]
        for side,data in [('source',source),('target',dest)]:assert audit[side+'_class_counts']==dict(zip(GROUPS,np.bincount(data['labels'],minlength=7).tolist()))
        benchmark,rgb,held=samples(date);train,_,ids=dense_training(date,benchmark)
        np.testing.assert_array_equal(source['points'],train[source['input_indices']]);np.testing.assert_array_equal(source['original_ply_indices'],ids[source['input_indices']])
        assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0) and np.all(ids%2==0)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
        color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
        tree=cKDTree(downsample(target,2.));full=cKDTree(target);small=downsample(train,.06)
        score_source=downsample(train,.12);rng=np.random.default_rng(29)
        if len(score_source)>INITIAL_PARAMETERS['geometry_score_points']:score_source=score_source[rng.choice(len(score_source),INITIAL_PARAMETERS['geometry_score_points'],replace=False)]
        np.testing.assert_array_equal(score_source,saved['score_source'])
        initial=np.r_[np.load(ROOT/f'results/object_consensus/{date}/initial_matrices.npz')['multi'],np.load(ROOT/f'results/object_recovery/{date}/initial_matrices.npz')['matrices']]
        initial=initial[[scale_admissible(m,audit['extent_scale_prior']) for m in initial]]
        _,unique=np.unique(np.round(initial,8).reshape(len(initial),16),axis=0,return_index=True);initial=initial[np.sort(unique)]
        np.testing.assert_array_equal(initial,saved['matrices']);assert len(initial)==audit['initial_pool_size']
        prior_scores=np.load(PREVIOUS/date/'initial_scores.npz');np.testing.assert_array_equal(initial,prior_scores['matrices'])
        scores=score_pool(score_source,tree,matcher,initial);np.testing.assert_allclose(scores,saved['scores'],atol=1e-9,equal_nan=True)
        np.testing.assert_allclose(scores[:,0],prior_scores['scores'][:,0],atol=1e-9)
        anchors=score_source[::max(1,len(score_source)//20)];assert choose_starts(initial,scores,anchors,True)==audit['initial_starts']
        sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf)
        assert audit['extent_scale_prior']==sf['normalization_scale']/tf['normalization_scale']
        for c in audit['coarse_candidates']:
            np.testing.assert_array_equal(c['initial_matrix'],initial[c['pool_index']]);expected,neighbors=correspondence_bank(initial,c['pool_index'],anchors,sd,td)
            assert expected==c['initial_pairs'] and neighbors==c['neighbor_pool_indices']
            assert match(sd,td,np.array(c['initial_matrix']))==c['representative_pairs']
            matrix=np.array(c['matrix']);np.testing.assert_allclose(np.median(tree.query(apply(small,matrix))[0]),c['geometry_score_m'],atol=1e-9)
            value,_=matcher.score(matrix)
            if value is None:assert c['semantic_score_m'] is None
            else:np.testing.assert_allclose(value,c['semantic_score_m'],atol=1e-9)
            assert len(spatial_support(surviving_pairs(c['initial_pairs'],sd,td,matrix),sd,td,matrix))==c['geometric_support_count']
            assert scale_admissible(matrix,audit['extent_scale_prior'])==c['scale_admissible']
        assert [c['pool_index'] for c in audit['coarse_candidates']]==audit['initial_starts']
        selected=choose_candidate(audit['coarse_candidates'],True);matrices={}
        if selected is None:assert 'matrix_local' not in run and run['status']=='rejected_insufficient_candidate_support'
        else:
            assert selected==run['selected_candidate_index'];c=audit['coarse_candidates'][selected];matrix=np.array(run['matrix_local']);matrices[METHOD]=matrix
            np.testing.assert_array_equal(matrix[3],[0,0,0,1]);scale=np.cbrt(np.linalg.det(matrix[:3,:3]));assert scale>0
            np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3)*scale**2,atol=1e-8)
            error,near=full.query(apply(benchmark[~held],matrix));color_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)))
            for key,value in summary(error).items():np.testing.assert_allclose(value,run['training_final'][key],atol=1e-9)
            np.testing.assert_allclose(color_error,run['training_color_difference'],atol=1e-9)
            support=verify_evidence(run,c['initial_pairs'],sf,tf,matrix)
            expected='rejected_training_alignment' if np.median(error)>.5 or color_error>.35 or not scale_admissible(matrix,audit['extent_scale_prior']) else ('candidate_for_review' if support else 'rejected_object_correspondences')
            assert run['status']==expected
            value,details=matcher.score(matrix);np.testing.assert_allclose(value,run['final_semantic_score_m'],atol=1e-9);assert details==run['final_semantic_class_scores']
        current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text());old=json.loads((PREVIOUS/date/'audit.json').read_text());old_run=next(r for r in old['runs'] if r['method']=='semantic_fusion')
        matrices.update({'current_automatic':np.array(current['matrix_local']),'previous_semantic':np.array(old_run['matrix_local'])})
        assert audit['previous_initial_starts']==old['initial_starts']['semantic_fusion']
        assert audit['previous_selected_candidate']==old_run['selected_candidate_id']
        for name,record in [(METHOD,run),('current_automatic',current),('previous_semantic',old_run)]:
            assert audit['comparison_status'][name]==record['status'];assert audit['comparison_strict_support'][name]==record.get('strict_forward_080_spatial_support_count',0)
            assert (date in report['accepted_dates'][name])==(record['status']=='candidate_for_review')
            assert (date in report['strict_dates'][name])==(record['status']=='candidate_for_review' and record.get('strict_forward_080_spatial_support_count',0)>=3)
        verify_evidence(audit['previous_strict_diagnostics'],old_run['final_pair_checks'],sf,tf,matrices['previous_semantic'])
        if date in diagnostics:
            from diagnose_strict_support import triangle_diagnostics
            diagnostic=diagnostics[date];matrix=matrices[METHOD]
            pairs=patch_checks(match(sd,td,matrix),sf,tf,matrix)
            eligible=[p for p in pairs if p['fraction_within_1m']>=.8 and p['reverse_fraction_within_1m']>=.5]
            assert eligible==diagnostic['postfit_strict_pairs']
            triangles=triangle_diagnostics(eligible,sd,td,matrix)
            assert json.loads(json.dumps(triangles))==diagnostic['postfit_triangles']
            assert diagnostic['postfit_admissible_triangles']==sum(t['passes'] for t in triangles)
        held_data=np.load(folder/'held_out_distances.npz');old_data=np.load(PREVIOUS/date/'held_out_distances.npz')
        np.testing.assert_array_equal(held_data['source_points'],benchmark[held]);np.testing.assert_array_equal(held_data['fixed_coverage'],old_data['fixed_coverage'])
        with (folder/'static_structures.csv').open() as f:static=list(csv.DictReader(f))
        controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
        for method,matrix in matrices.items():
            error=full.query(apply(benchmark[held],matrix))[0];np.testing.assert_allclose(error,held_data[method],atol=1e-9)
            if method!=METHOD:np.testing.assert_allclose(error,old_data['semantic_fusion' if method=='previous_semantic' else method],atol=1e-9)
            for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',held_data['fixed_coverage'])]:
                row=next(r for r in audit['comparison'] if r['method']==method and r['scope']==scope)
                for key,value in summary(error[mask]).items():np.testing.assert_allclose(value,row[key],atol=1e-9)
                np.testing.assert_allclose(row['within_1m'],np.mean(error[mask]<1),atol=1e-12)
                if scope=='fixed_baseline_coverage':
                    report_row=next(r for r in report_rows if r['date']==date and r['method']==method)
                    assert report_row['status']==row['status'] and int(report_row['strict_support_count'])==row['strict_support_count']
                    for key,value in summary(error[mask]).items():np.testing.assert_allclose(float(report_row[key]),value,atol=1e-9)
            for region in regions:
                local=cKDTree(target[volume_mask(target,region)])
                for scope in ['unseen_records','spatial_holdout']:
                    points=controls[region['id']+'_'+scope];row=next(r for r in static if r['method']==method and r['region']==region['id'] and r['scope']==scope)
                    assert int(row['count'])==len(points) and (row['adequate_support']=='True')==(len(points)>=30)
                    if len(points):
                        for key,value in summary(local.query(apply(points,matrix))[0]).items():np.testing.assert_allclose(value,float(row[key]),atol=1e-9)
                    else:assert all(row[k]=='' for k in ['median','p90','p95','rmse'])
        checks.append({'date':date,'all_initial_scores_recomputed':len(initial),'coarse_candidates':len(audit['coarse_candidates']),
            'withheld_points':int(held.sum()),'status':run['status'],'strict_support_count':run.get('strict_forward_080_spatial_support_count',0)})
        print('Verified',date,flush=True)
    log=(OUT/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':count,'prior_file_hashes_matched':len(preserved),
        'model_and_label_hashes_verified':True,'all_initial_scores_and_banks_recomputed':True,
        'strict_gates_and_triangles_recomputed':True,'held_out_and_static_errors_recomputed':True,'date_checks':checks,
        'verification_code_sha256':file_hash(ROOT/'scripts/verify_semantic_3dgs_registration.py'),
        'meaning':'Repeated registration and verified internal comparisons; not independent survey accuracy.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed',flush=True)


if __name__=='__main__':main()
