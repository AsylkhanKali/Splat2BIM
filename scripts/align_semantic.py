"""Controlled semantic reranking of the same eight cached geometric hypotheses."""
import argparse
import json
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES,samples,refine
from align_0929 import apply,downsample,summary
from align_objects import dense_training
from align_object_recovery import load_fragments,surviving_pairs,scale_admissible
from align_object_consensus import patch_checks
from object_consensus import descriptors,spatial_support
from check_las_vertical import file_hash,write_csv
from check_static_structures import volume_mask
from semantic_projection import PARAMETERS
from semantic_matching import SemanticMatcher,choose_candidate
from prepare_semantic_clouds import OUT


def run(date):
    started=time.monotonic();folder=OUT/date
    source_found,target_found=load_fragments(date);sd=descriptors(source_found);td=descriptors(target_found)
    extent=source_found['normalization_scale']/target_found['normalization_scale']
    benchmark,rgb,held=samples(date);train,colors,ids=dense_training(date,benchmark)
    source=dict(np.load(folder/'source_labels.npz'));target_labels=dict(np.load(folder/'target_labels.npz'))
    np.testing.assert_array_equal(source['points'],train[source['input_indices']])
    np.testing.assert_array_equal(source['original_ply_indices'],ids[source['input_indices']])
    assert np.all(np.floor(source['points']/.4).sum(1).astype(int)%4!=0)
    matcher=SemanticMatcher(source,target_labels)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    target_rgb=np.c_[las.red,las.green,las.blue].astype(float);target_rgb/=65535 if target_rgb.max()>255 else 255
    full_tree=cKDTree(target);small=downsample(train,.06);small_tree=cKDTree(downsample(target,2.))
    # Only coarse candidates and their proposed identities are used, not prior final transforms.
    old_paths=[ROOT/f'results/object_consensus/{date}/audit.json',ROOT/f'results/object_recovery/{date}/audit.json']
    previous=json.loads(old_paths[0].read_text());recovery=json.loads(old_paths[1].read_text())
    pools=[next(r for r in previous['runs'] if r['method']=='multi_fragment')['coarse_candidates'],recovery['coarse_candidates']]
    candidates=[]
    for pool_name,pool in zip(['consensus','recovery'],pools):
        for index,c in enumerate(pool):
            matrix=np.array(c['matrix']);pairs=surviving_pairs(c['initial_pairs'],sd,td,matrix)
            support=spatial_support(pairs,sd,td,matrix);semantic_score,details=matcher.score(matrix)
            geometry_score=float(np.median(small_tree.query(apply(small,matrix))[0]))
            np.testing.assert_allclose(geometry_score,c['training_coarse_median_m'],atol=1e-9)
            candidates.append({'id':f'{pool_name}_{index}','matrix':matrix.tolist(),'initial_pairs':c['initial_pairs'],
                'geometric_support_count':len(support),'scale_admissible':scale_admissible(matrix,extent),
                'geometry_score_m':geometry_score,'semantic_score_m':semantic_score,
                'semantic_class_scores':[{k:v for k,v in r.items() if k not in ('source_indices','target_indices')} for r in details]})
    selected={method:choose_candidate(candidates,semantic) for method,semantic in [('pooled_geometry',False),('semantic_rerank',True)]}
    fine_source=downsample(train,.025);fine_target=downsample(target,.6);refined={};runs=[];matrices={}
    for method,index in selected.items():
        if index is None:
            runs.append({'method':method,'status':'rejected_insufficient_candidate_support'});continue
        c=candidates[index]
        if index not in refined:refined[index]=refine(fine_source,fine_target,np.array(c['matrix']))
        matrix=refined[index];matrices[method]=matrix
        error,near=full_tree.query(apply(benchmark[~held],matrix))
        color_error=float(np.median(np.linalg.norm(rgb[~held]-target_rgb[near],axis=1)))
        checks=patch_checks(c['initial_pairs'],source_found,target_found,matrix)
        current={(p['source_id'],p['target_id']):p for p in surviving_pairs(c['initial_pairs'],sd,td,matrix)}
        valid=[{**p,'center_distance_m':current[(p['source_id'],p['target_id'])]['center_distance_m']} for p in checks
            if (p['source_id'],p['target_id']) in current and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
        support=spatial_support(valid,sd,td,matrix);strict=spatial_support([p for p in valid if p['fraction_within_1m']>=.8],sd,td,matrix)
        status='rejected_training_alignment' if np.median(error)>.5 or color_error>.35 or not scale_admissible(matrix,extent) else (
            'candidate_for_review' if support else 'rejected_object_correspondences')
        semantic_score,details=matcher.score(matrix)
        runs.append({'method':method,'status':status,'selected_candidate_index':index,'selected_candidate_id':c['id'],
            'matrix_local':matrix.tolist(),'training_final':summary(error),'training_color_difference':color_error,
            'final_pair_checks':checks,'final_spatial_support':support,'strict_forward_080_spatial_support_count':len(strict),
            'semantic_final_score_m':semantic_score,'semantic_final_correspondences':details})
        print(date,method,c['id'],status,'training median',round(float(np.median(error)),4),flush=True)
    # Evaluation starts only after both new decisions and fits have been fixed.
    current_path=ROOT/f'results/object_recovery/{date}/selection.json';current=json.loads(current_path.read_text())
    matrices['current_automatic']=np.array(current['matrix_local'])
    check=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz')
    np.testing.assert_array_equal(benchmark[held],check['source_points']);fixed=check['fixed_coverage']
    rows=[];distances={}
    for method,matrix in matrices.items():
        error=full_tree.query(apply(benchmark[held],matrix))[0];distances[method]=error
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
            rows.append({'date':date,'method':method,'scope':scope,**summary(error[mask]),'within_1m':float(np.mean(error[mask]<1.))})
    np.savez_compressed(folder/'held_out_distances.npz',source_points=benchmark[held],fixed_coverage=fixed,**distances)
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions']
    controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz');statics=[]
    for region in regions:
        tree=cKDTree(target[volume_mask(target,region)])
        for scope in ['unseen_records','spatial_holdout']:
            control=controls[region['id']+'_'+scope]
            for method,matrix in matrices.items():
                error=tree.query(apply(control,matrix))[0]
                stats=summary(error) if len(error) else {'count':0,'median':None,'p90':None,'p95':None,'rmse':None}
                statics.append({'date':date,'region':region['id'],'scope':scope,'method':method,
                    **stats,'adequate_support':len(error)>=30})
    write_csv(folder/'comparison.csv',rows);write_csv(folder/'static_structures.csv',statics)
    audit={'date':date,'semantic_parameters':PARAMETERS,'extent_scale_prior':extent,'candidates':candidates,
        'shared_candidate_pool':True,'same_geometric_refinement':True,'new_decisions_use_withheld_metrics':False,
        'existing_final_matrices_used_for_new_alignment':False,'training_tile_leakage_count':0,
        'training_records':len(train),'semantic_groups_available':[r[0] for r in matcher.classes],
        'runs':runs,'comparison':rows,'seconds':time.monotonic()-started,
        'candidate_inputs':{str(p.relative_to(ROOT)):file_hash(p) for p in old_paths},
        'comparison_input':{'path':str(current_path.relative_to(ROOT)),'sha256':file_hash(current_path)},
        'current_status':current['status'],'current_strict_support':current['strict_forward_080_spatial_support_count']}
    (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Finished',date,round(audit['seconds'],1),'s',flush=True)


def main(dates):
    protocol={'parameters':PARAMETERS,'candidate_pool':'four consensus and four recovery coarse hypotheses, identical for both new methods',
        'semantic_method':'class-balanced same-class nearest-surface score, geometric score + 0.5 * semantic score',
        'refinement':'unchanged geometric point-to-plane Sim(3), same training tiles and geometric acceptance checks',
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['align_semantic.py','semantic_matching.py','semantic_projection.py']},
        'scope':'development on this known site; no new-site generalization claim'}
    path=OUT/'alignment_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; preserve prior experiment.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    for date in dates:run(date)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
