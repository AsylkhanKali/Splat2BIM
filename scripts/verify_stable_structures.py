"""Recompute surface extraction, persistence, prefit banks, strict gates and held errors."""
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
from align_object_recovery import scale_admissible
from align_object_consensus import patch_checks
from object_consensus import descriptors,PARAMETERS as OBJECT_PARAMETERS
from check_las_vertical import file_hash
from check_static_structures import volume_mask
from semantic_matching import SemanticMatcher,choose_candidate
from semantic_initial import INITIAL_PARAMETERS,choose_starts
from semantic_structures import STRUCTURE_PARAMETERS
from structural_correspondence import gated_pairs,family_support,initial_family_bank
from prepare_stable_structures import load_structures,build
from prepare_temporal_surfaces import TEMPORAL_PARAMETERS
from structural_surfaces import serializable
from semantic_projection import PARAMETERS,GROUPS
from align_stable_structures import OUT,LABELS,PREVIOUS,BASE,METHOD,surviving_bank


def verify_evidence(record,pairs,sf,tf,matrix):
    sd,td=descriptors(sf),descriptors(tf);patches=patch_checks(pairs,sf,tf,matrix)
    for new,old in zip(patches,record['final_pair_checks']):
        for k in ['median_m','p90_m','fraction_within_1m','reverse_fraction_within_1m']:np.testing.assert_allclose(new[k],old[k],atol=1e-12)
    assert len(patches)==len(record['final_pair_checks'])
    survived={(p['source_id'],p['target_id']):p for p in surviving_bank(pairs,sd,td,matrix)}
    valid=[{**p,'center_distance_m':survived[(p['source_id'],p['target_id'])]['center_distance_m']} for p in patches
        if (p['source_id'],p['target_id']) in survived and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
    strict_pairs=[p for p in valid if p['fraction_within_1m']>=.8]
    identities=lambda rows:[(p['source_id'],p['target_id']) for p in rows]
    support=family_support(valid,sf,tf,matrix);strict=family_support(strict_pairs,sf,tf,matrix)
    assert identities(support)==identities(record['final_spatial_support'])
    assert identities(strict_pairs)==identities(record['strict_overlap_pairs'])
    assert len(strict)==record['strict_forward_080_spatial_support_count']
    assert identities(strict)==identities(record['strict_spatial_support'])
    for chosen in [support,strict]:
        assert len({sf['candidates'][p['source_index']]['family'] for p in chosen})==len(chosen)
        assert len({tf['candidates'][p['target_index']]['family'] for p in chosen})==len(chosen)
    A=apply(sd['centers'],matrix);B=td['centers'];triples=list(itertools.combinations(strict_pairs,3))
    assert len(triples)==len(record['strict_triangles']);passes=0
    for triple,saved in zip(triples,record['strict_triangles']):
        assert identities(triple)==[tuple(p) for p in saved['identities']]
        passed=len({sf['candidates'][p['source_index']]['family'] for p in triple})==3 and len({tf['candidates'][p['target_index']]['family'] for p in triple})==3
        assert passed==saved['distinct_families']
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
    protocol=json.loads((OUT/'protocol.json').read_text());parent=json.loads((BASE/'protocol.json').read_text())
    assert protocol['initial_parameters']==parent['initial_parameters']==INITIAL_PARAMETERS
    assert protocol['semantic_parameters']==parent['semantic_parameters']==PARAMETERS
    assert protocol['structure_parameters']==parent['structure_parameters']==STRUCTURE_PARAMETERS
    assert protocol['object_parameters']==OBJECT_PARAMETERS and protocol['strict_forward_fraction']==.8
    for name,digest in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    assert file_hash(BASE/'protocol.json')==protocol['parent_protocol_sha256']
    assert file_hash(OUT/'extraction_protocol.json')==protocol['extraction_protocol_sha256']
    assert file_hash(OUT/'extraction.json')==protocol['extraction_sha256']
    extraction_protocol=json.loads((OUT/'extraction_protocol.json').read_text())
    assert extraction_protocol['temporal_parameters']==TEMPORAL_PARAMETERS
    for name,digest in extraction_protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    extraction=json.loads((OUT/'extraction.json').read_text())
    for path,digest in extraction['inputs'].items():assert file_hash(ROOT/path)==digest,path
    regenerated={}
    for date in DATES:
        regenerated[date],_=build(date)
        for side in ['source','target']:
            found=load_structures(date,side);fresh=regenerated[date][side]
            if side=='target':
                for row in fresh['candidates']:
                    row['temporal_coverage']=extraction['target_coverage_all_alternatives'][date][row['id']]
                    row['supported_dates']=[d for d,v in row['temporal_coverage'].items() if v>=.8]
                retained=[r for r in fresh['candidates'] if date in r['supported_dates'] and len(r['supported_dates'])>=3]
            else:retained=fresh['candidates']
            assert [r['id'] for r in retained]==[r['id'] for r in found['candidates']]
            assert json.loads(json.dumps(serializable({**fresh,'candidates':retained})))==serializable(found)
            for a,b in zip(retained,found['candidates']):np.testing.assert_allclose(a['points'],b['points'],atol=1e-11)
        print('Verified extraction',date,flush=True)
    for reference in DATES:
        las=laspy.read(ROOT/f'assets/{reference}/{reference}.las');tree=cKDTree(np.c_[las.x,las.y,las.z]-ORIGIN)
        for date in DATES:
            found=regenerated[date]['target']
            for row in found['candidates']:
                points=row['points']/found['normalization_scale']+found['normalization_center']
                actual=float(np.mean(tree.query(points)[0]<1.))
                assert actual==extraction['target_coverage_all_alternatives'][date][row['id']][reference]
        print('Verified temporal reference',reference,flush=True)
    del regenerated
    segmentation=json.loads((LABELS/'segmentation_protocol.json').read_text())
    for name,digest in segmentation['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==digest
    for name,digest in segmentation['model']['files'].items():assert file_hash(ROOT/'models/loveda_segformer_b2'/name)==digest
    original=json.loads((ROOT/'results/object_recovery/inputs.json').read_text())
    preserved={f['path']:f['sha256'] for value in original.values() for f in value['files']}
    for path,digest in preserved.items():assert file_hash(ROOT/path)==digest,path
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions'];checks=[]
    for date in DATES:
        folder=OUT/date;audit=json.loads((folder/'audit.json').read_text());saved=np.load(folder/'initial_scores.npz');run=audit['run']
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
        prior_scores=np.load(BASE/date/'initial_scores.npz');np.testing.assert_array_equal(initial,prior_scores['matrices'])
        scores=prior_scores['scores'];np.testing.assert_array_equal(scores,saved['scores'])
        np.testing.assert_allclose(scores[:,0],prior_scores['scores'][:,0],atol=1e-9)
        anchors=score_source[::max(1,len(score_source)//20)];assert choose_starts(initial,scores,anchors,True)==audit['initial_starts']
        sf,tf=load_structures(date,'source'),load_structures(date,'target');sd,td=descriptors(sf),descriptors(tf)
        assert audit['extent_scale_prior']==sf['normalization_scale']/tf['normalization_scale']
        for c in audit['coarse_candidates']:
            np.testing.assert_array_equal(c['initial_matrix'],initial[c['pool_index']]);expected,neighbors=initial_family_bank(initial,c['pool_index'],anchors,sd,td)
            assert expected==c['initial_pairs'] and neighbors==c['neighbor_pool_indices']
            assert gated_pairs(sd,td,np.array(c['initial_matrix']))==c['representative_pairs']
            matrix=np.array(c['matrix']);np.testing.assert_allclose(np.median(tree.query(apply(small,matrix))[0]),c['geometry_score_m'],atol=1e-9)
            value,_=matcher.score(matrix)
            if value is None:assert c['semantic_score_m'] is None
            else:np.testing.assert_allclose(value,c['semantic_score_m'],atol=1e-9)
            assert len(family_support(surviving_bank(c['initial_pairs'],sd,td,matrix),sf,tf,matrix))==c['geometric_support_count']
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
        current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text());old=json.loads((BASE/date/'audit.json').read_text());old_run=old['run']
        matrices.update({'current_automatic':np.array(current['matrix_local']),'previous_semantic':np.array(old_run['matrix_local'])})
        assert audit['previous_initial_starts']==old['initial_starts']
        assert audit['previous_selected_candidate']==old_run['selected_candidate_id']
        for name,record in [(METHOD,run),('current_automatic',current),('previous_semantic',old_run)]:
            assert audit['comparison_status'][name]==record['status'];assert audit['comparison_strict_support'][name]==record.get('strict_forward_080_spatial_support_count',0)
        held_data=np.load(folder/'held_out_distances.npz');old_data=np.load(BASE/date/'held_out_distances.npz')
        np.testing.assert_array_equal(held_data['source_points'],benchmark[held]);np.testing.assert_array_equal(held_data['fixed_coverage'],old_data['fixed_coverage'])
        with (folder/'static_structures.csv').open() as f:static=list(csv.DictReader(f))
        controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
        for method,matrix in matrices.items():
            error=full.query(apply(benchmark[held],matrix))[0];np.testing.assert_allclose(error,held_data[method],atol=1e-9)
            if method!=METHOD:np.testing.assert_allclose(error,old_data['gaussian_semantic' if method=='previous_semantic' else method],atol=1e-9)
            for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',held_data['fixed_coverage'])]:
                row=next(r for r in audit['comparison'] if r['method']==method and r['scope']==scope)
                for key,value in summary(error[mask]).items():np.testing.assert_allclose(value,row[key],atol=1e-9)
                np.testing.assert_allclose(row['within_1m'],np.mean(error[mask]<1),atol=1e-12)
            for region in regions:
                local=cKDTree(target[volume_mask(target,region)])
                for scope in ['unseen_records','spatial_holdout']:
                    points=controls[region['id']+'_'+scope];row=next(r for r in static if r['method']==method and r['region']==region['id'] and r['scope']==scope)
                    assert int(row['count'])==len(points) and (row['adequate_support']=='True')==(len(points)>=30)
                    if len(points):
                        for key,value in summary(local.query(apply(points,matrix))[0]).items():np.testing.assert_allclose(value,float(row[key]),atol=1e-9)
                    else:assert all(row[k]=='' for k in ['median','p90','p95','rmse'])
        checks.append({'date':date,'frozen_initial_scores_verified':len(initial),'coarse_candidates':len(audit['coarse_candidates']),
            'withheld_points':int(held.sum()),'status':run['status'],'strict_support_count':run.get('strict_forward_080_spatial_support_count',0)})
        print('Verified',date,flush=True)
    log=(OUT/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':count,'prior_file_hashes_matched':len(preserved),
        'model_and_label_hashes_verified':True,'frozen_initial_scores_verified':True,'all_banks_recomputed':True,
        'infill_and_temporal_coverage_recomputed':True,'growth_code_and_cache_hashes_verified':True,
        'strict_gates_and_triangles_recomputed':True,'held_out_and_static_errors_recomputed':True,'date_checks':checks,
        'verification_code_sha256':file_hash(ROOT/'scripts/verify_stable_structures.py'),
        'meaning':'Repeated registration and verified internal comparisons; not independent survey accuracy.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed',flush=True)


if __name__=='__main__':main()
