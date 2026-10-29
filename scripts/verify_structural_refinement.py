"""Reproduce local fits and verify selection, strict gates, splits and control metrics."""
import csv
import json
import re

import laspy
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_0929 import apply,summary
from align_objects import dense_training
from align_stable_structures import support_evidence
from prepare_stable_structures import load_structures,OUT as BASE
from fit_structural_refinement import OUT as FIRST,prepare_groups,usable_semantics
from fit_guarded_structural_refinement import OUT as GUARDED
from fit_reviewed_structural_refinement import OUT,prepare_groups as reviewed_groups
from structural_pose_refinement import PARAMETERS,metrics,refine,select_candidate,footprint_mask,extent_anchors
from guarded_structural_pose import context_indices,refine_with_context,fractional_correction,global_guard
from check_static_structures import volume_mask
from check_las_vertical import file_hash


def equal_metrics(actual,saved):
    for key in ['count','median','p90','p95','rmse']:
        if actual[key] is None:assert saved[key] in [None,'']
        else:np.testing.assert_allclose(actual[key],float(saved[key]),atol=1e-10,rtol=1e-10)


def verify_controls(out,date,fit,initial,benchmark,held,target,regions,sf,tf):
    folder=out/date;selected=np.array(fit['matrix_local']);full=cKDTree(target)
    current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text())
    matrices={'current_automatic':np.array(current['matrix_local']),'stable_parent':initial,'structural_refinement':selected}
    saved=np.load(folder/'held_out_distances.npz');previous=np.load(BASE/date/'held_out_distances.npz')
    np.testing.assert_array_equal(saved['source_points'],benchmark[held]);np.testing.assert_array_equal(saved['fixed_coverage'],previous['fixed_coverage'])
    rows=list(csv.DictReader((folder/'comparison.csv').open()))
    for name,matrix in matrices.items():
        error=full.query(apply(benchmark[held],matrix))[0];np.testing.assert_allclose(error,saved[name],atol=1e-10)
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',saved['fixed_coverage'])]:
            row=next(r for r in rows if r['method']==name and r['scope']==scope);equal_metrics(summary(error[mask]),row)
    controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz')
    rows=list(csv.DictReader((folder/'static_structures.csv').open()))
    for region in regions:
        tree=cKDTree(target[volume_mask(target,region)])
        for scope in ['unseen_records','spatial_holdout']:
            points=controls[region['id']+'_'+scope]
            for name,matrix in matrices.items():
                row=next(r for r in rows if r['method']==name and r['region']==region['id'] and r['scope']==scope)
                actual=summary(tree.query(apply(points,matrix))[0]) if len(points) else dict(count=0,median=None,p90=None,p95=None,rmse=None)
                equal_metrics(actual,row);assert (row['adequate_support']=='True')==(len(points)>=30)
    parent=json.loads((BASE/date/'audit.json').read_text());support=parent['run']['strict_spatial_support']
    v=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data[5::10]
    odd=np.c_[v['x'],v['y'],v['z']].astype(float);ids=np.arange(5,5+10*len(v),10)
    valid=(v['opacity']>0)&np.isfinite(odd).all(1);odd,ids=odd[valid],ids[valid]
    labels=dict(np.load(ROOT/f'results/semantic_fusion/{date}/target_labels.npz'));target_valid=usable_semantics(target,labels,tf)
    cache=np.load(folder/'structural_controls.npz');rows=list(csv.DictReader((folder/'structural_controls.csv').open()))
    for number,pair in enumerate(support):
        a=sf['candidates'][pair['source_index']];b=tf['candidates'][pair['target_index']]
        ti=np.flatnonzero(footprint_mask(target,tf,b)&target_valid);np.testing.assert_array_equal(ti,cache[str(number)+'_target_las_ids'])
        tree=cKDTree(target[ti])
        for scope,pool,pool_ids in [('spatial_holdout',benchmark[held],np.flatnonzero(held)),('unseen_records',odd,ids)]:
            mask=footprint_mask(pool,sf,a);points=pool[mask];prefix=str(number)+'_'+scope
            np.testing.assert_array_equal(points,cache[prefix]);np.testing.assert_array_equal(pool_ids[mask],cache[prefix+'_ids'])
            for name,matrix in matrices.items():
                error=tree.query(apply(points,matrix))[0];np.testing.assert_allclose(error,cache[prefix+'_'+name],atol=1e-10)
                row=next(r for r in rows if r['method']==name and r['source_id']==a['id'] and r['scope']==scope)
                equal_metrics(summary(error) if len(error) else dict(count=0,median=None,p90=None,p95=None,rmse=None),row)
    evaluation=json.loads((folder/'evaluation.json').read_text());assert evaluation['fit_sha256']==file_hash(folder/'fit.json')
    fresh=support_evidence(fit['fixed_prefit_bank'],sf,tf,selected)
    assert json.loads(json.dumps(fresh))==evaluation['selected_evidence'];assert fresh['strict_forward_080_spatial_support_count']>=3


def main():
    roots=[FIRST,GUARDED,OUT]
    for out in roots:
        assert all((out/d/'evaluation.json').exists() for d in DATES),'Finish all evaluations first.'
        protocol=json.loads((out/'protocol.json').read_text())
        parent=json.loads((BASE/'protocol.json').read_text());assert protocol['object_parameters']==parent['object_parameters'] and protocol['strict_forward_fraction']==.8
        for n,h in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/n)==h
    original=json.loads((ROOT/'results/object_recovery/inputs.json').read_text());preserved={f['path']:f['sha256'] for v in original.values() for f in v['files']}
    for path,h in preserved.items():assert file_hash(ROOT/path)==h
    regions_path=ROOT/'results/static_structures/regions.json';regions=json.loads(regions_path.read_text())['regions'];checks=[]
    for date in DATES:
        benchmark,rgb,held=samples(date);train,_,ids=dense_training(date,benchmark)
        assert np.all(ids%2==0) and np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN;full=cKDTree(target)
        color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
        parent=json.loads((BASE/date/'audit.json').read_text());initial=np.array(parent['run']['matrix_local'])
        sf,tf=load_structures(date,'source'),load_structures(date,'target')
        auto,auto_meta,auto_cache,_=prepare_groups(date,train,ids,target,initial,sf,tf,parent['run']['strict_spatial_support'])
        auto_cache['anchors']=extent_anchors(train)
        reviewed,reviewed_meta,reviewed_cache=reviewed_groups(train,ids,target,initial,regions)
        for out,groups,metadata,cache in [(FIRST,auto,auto_meta,auto_cache),(OUT,reviewed,reviewed_meta,reviewed_cache)]:
            saved=np.load(out/date/'fit_data.npz');fit=json.loads((out/date/'fit.json').read_text());assert fit['groups']==metadata
            for key,value in cache.items():np.testing.assert_array_equal(value,saved[key])
            for m in metadata:
                a=saved[m['key']+'_fit_ply_ids'];b=saved[m['key']+'_calibration_ply_ids'];assert not len(np.intersect1d(a,b))
                assert np.all(a%2==0) and np.all(b%2==0)
        for out,groups in [(FIRST,auto),(GUARDED,auto),(OUT,reviewed)]:
            fit=json.loads((out/date/'fit.json').read_text());np.testing.assert_array_equal(fit['initial_matrix'],initial)
            for path,h in fit['inputs'].items():assert file_hash(ROOT/path)==h
            anchors=np.load((OUT if out==OUT else FIRST)/date/'fit_data.npz')['anchors'];proposals={}
            if out==GUARDED:
                cache=np.load(out/date/'context.npz');labels=dict(np.load(ROOT/f'results/semantic_3dgs/{date}/source_labels.npz'))
                si=np.flatnonzero(usable_semantics(train,labels,sf));si=si[context_indices(apply(train[si],initial))]
                block=np.floor(apply(train[si],initial)/PARAMETERS['calibration_block_m']).astype(int);si=si[(block@np.array([1,3,7]))%PARAMETERS['calibration_modulus']!=0]
                target_labels=dict(np.load(ROOT/f'results/semantic_fusion/{date}/target_labels.npz'));ti=np.flatnonzero(usable_semantics(target,target_labels,tf))
                np.testing.assert_array_equal(cache['source'],train[si]);np.testing.assert_array_equal(cache['source_ply_ids'],ids[si]);np.testing.assert_array_equal(cache['target_las_ids'],ti)
                context={'fit':train[si],'target':{'points':target[ti],'tree':cKDTree(target[ti])}}
                proposal,diagnostic=refine_with_context(groups,context,initial,anchors)
                np.testing.assert_allclose(proposal,fit['proposal_matrix'],atol=1e-10);assert diagnostic==fit['optimization']
            elif len(groups)>=(2 if out==OUT else 3):
                for weight in PARAMETERS['tangent_weights']:
                    proposals[weight],_=refine(groups,initial,weight,anchors)
            baseline=metrics(groups,initial);assert baseline==fit['baseline_calibration']
            before=summary(full.query(apply(benchmark[~held],initial))[0]);assert before==fit['baseline_training']
            for c in fit['candidates']:
                if out==FIRST:matrix=proposals[c['tangent_weight']]
                else:matrix=fractional_correction(initial,proposal if out==GUARDED else proposals[c['tangent_weight']],c['fraction'],apply(anchors,initial).mean(0))
                np.testing.assert_allclose(matrix,c['matrix_local'],atol=1e-10)
                delta=matrix@np.linalg.inv(initial);np.testing.assert_allclose(delta[:3,:3].T@delta[:3,:3],np.eye(3),atol=1e-10)
                assert np.linalg.norm(apply(anchors,matrix)-apply(anchors,initial),axis=1).max()<=PARAMETERS['maximum_motion_m']+1e-10
                assert metrics(groups,matrix)==c['calibration'] and metrics(groups,matrix,'fit')==c['fit']
                error,near=full.query(apply(benchmark[~held],matrix));training=summary(error);equal_metrics(training,c['training'])
                color_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)));np.testing.assert_allclose(color_error,c['training_rgb'],atol=1e-12)
                evidence=support_evidence(fit['fixed_prefit_bank'],sf,tf,matrix)
                for key,value in evidence.items():assert json.loads(json.dumps(value))==c[key]
                strict=training['median']<=.5 and color_error<=.35 and evidence['strict_forward_080_spatial_support_count']>=3
                assert strict==c['strict_pass']
                guard=global_guard(before,training) if out==GUARDED else training['median']<=before['median']+PARAMETERS['maximum_training_median_regression_m']
                assert bool(guard)==c['global_training_guard']
            chosen=select_candidate(fit['candidates'],baseline);assert chosen==fit['selected_candidate']
            expected=initial if chosen is None else fit['candidates'][chosen]['matrix_local'];np.testing.assert_array_equal(expected,fit['matrix_local'])
            np.testing.assert_allclose(np.loadtxt(out/date/'T_3dgs_to_las_local.txt'),expected,atol=1e-12)
            verify_controls(out,date,fit,initial,benchmark,held,target,regions,sf,tf)
            print('Verified',out.name,date,flush=True)
        checks.append({'date':date,'experiments_verified':3,'all_selected_strict_checks_passed':True})
    log=(GUARDED/'tests.log').read_text();count=int(re.search(r'Ran (\d+) tests',log)[1]);assert log.rstrip().endswith('OK')
    result={'all_checks_passed':True,'unit_tests_passed':count,'preserved_file_hashes':len(preserved),'all_fits_recomputed':True,
        'point_ids_and_spatial_splits_recomputed':True,'strict_gates_unchanged':True,'all_control_errors_recomputed':True,
        'date_checks':checks,'verification_code_sha256':file_hash(ROOT/'scripts/verify_structural_refinement.py'),
        'meaning':'Internal same-site development comparison; reviewed-region experiment uses manual region definitions, never held control records.'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print('All checks passed',flush=True)


if __name__=='__main__':main()
