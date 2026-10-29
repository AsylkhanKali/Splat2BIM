"""Repeat the frozen registration procedure with Gaussian-view source labels."""
import argparse
import json
import time

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES,samples,refine
from align_0929 import apply,downsample,icp,summary
from align_objects import dense_training
from align_object_recovery import load_fragments,surviving_pairs,scale_admissible
from align_object_consensus import patch_checks
from object_consensus import descriptors,spatial_support,match,PARAMETERS as OBJECT_PARAMETERS
from check_las_vertical import file_hash,write_csv
from check_static_structures import volume_mask
from semantic_projection import PARAMETERS,GROUPS
from semantic_matching import SemanticMatcher,choose_candidate
from semantic_initial import INITIAL_PARAMETERS,score_pool,choose_starts
from semantic_structures import STRUCTURE_PARAMETERS,correspondence_bank
from diagnose_strict_support import triangle_diagnostics
from prepare_semantic_3dgs import OUT as LABELS,PREVIOUS

OUT=ROOT/'results/semantic_3dgs_registration'
METHOD='gaussian_semantic'


def support_evidence(pairs,sf,tf,matrix):
    sd,td=descriptors(sf),descriptors(tf);checks=patch_checks(pairs,sf,tf,matrix)
    survived={(p['source_id'],p['target_id']):p for p in surviving_pairs(pairs,sd,td,matrix)}
    valid=[{**p,'center_distance_m':survived[(p['source_id'],p['target_id'])]['center_distance_m']} for p in checks
        if (p['source_id'],p['target_id']) in survived and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
    strict_pairs=[p for p in valid if p['fraction_within_1m']>=.8]
    support=spatial_support(valid,sd,td,matrix);strict=spatial_support(strict_pairs,sd,td,matrix)
    triangles=triangle_diagnostics(strict_pairs,sd,td,matrix)
    return {'final_pair_checks':checks,'final_spatial_support':support,
        'strict_forward_080_spatial_support_count':len(strict),'strict_overlap_pairs':strict_pairs,
        'strict_triangles':triangles,'admissible_strict_triangles':sum(t['passes'] for t in triangles)}


def run(date):
    begin=time.monotonic();folder=OUT/date;folder.mkdir(parents=True,exist_ok=True)
    benchmark,rgb,held=samples(date);train,_,ids=dense_training(date,benchmark)
    source_path=LABELS/date/'source_labels.npz';target_path=PREVIOUS/date/'target_labels.npz'
    source=dict(np.load(source_path));target_labels=dict(np.load(target_path))
    np.testing.assert_array_equal(source['points'],train[source['input_indices']])
    np.testing.assert_array_equal(source['original_ply_indices'],ids[source['input_indices']])
    assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    matcher=SemanticMatcher(source,target_labels)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
    sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf);extent=sf['normalization_scale']/tf['normalization_scale']
    paths=[ROOT/f'results/object_consensus/{date}/initial_matrices.npz',ROOT/f'results/object_recovery/{date}/initial_matrices.npz']
    first=np.load(paths[0])['multi'];second=np.load(paths[1])['matrices'];matrices=np.r_[first,second]
    names=np.array([f'consensus_{i}' for i in range(len(first))]+[f'recovery_{i}' for i in range(len(second))])
    valid=np.array([scale_admissible(m,extent) for m in matrices]);matrices,names=matrices[valid],names[valid]
    _,unique=np.unique(np.round(matrices,8).reshape(len(matrices),16),axis=0,return_index=True);unique=np.sort(unique)
    matrices,names=matrices[unique],names[unique]
    small=downsample(train,.06);target_small=downsample(target,2.);tree=cKDTree(target_small)
    score_source=downsample(train,.12);rng=np.random.default_rng(29)
    if len(score_source)>INITIAL_PARAMETERS['geometry_score_points']:score_source=score_source[rng.choice(len(score_source),INITIAL_PARAMETERS['geometry_score_points'],replace=False)]
    previous_scores=np.load(PREVIOUS/date/'initial_scores.npz')
    for value,key in [(matrices,'matrices'),(names,'names'),(score_source,'score_source')]:np.testing.assert_array_equal(value,previous_scores[key])
    scores=score_pool(score_source,tree,matcher,matrices);anchors=score_source[::max(1,len(score_source)//20)]
    np.testing.assert_allclose(scores[:,0],previous_scores['scores'][:,0],atol=1e-9)
    starts=choose_starts(matrices,scores,anchors,True)
    np.savez_compressed(folder/'initial_scores.npz',matrices=matrices,names=names,scores=scores,score_source=score_source)
    print(date,'initial pool',len(matrices),'starts',starts,'ranking seconds',round(time.monotonic()-begin,1),flush=True)
    # Every identity bank is fixed before any refinement. No final-pose pair additions.
    banks={i:correspondence_bank(matrices,i,anchors,sd,td) for i in starts};coarse=[]
    for index in starts:
        initial=matrices[index];pairs,neighbors=banks[index]
        matrix=icp(small,target_small,initial,INITIAL_PARAMETERS['coarse_gates_m'],with_scale=True,iterations=INITIAL_PARAMETERS['coarse_iterations'])
        support=spatial_support(surviving_pairs(pairs,sd,td,matrix),sd,td,matrix);semantic_score,_=matcher.score(matrix)
        coarse.append({'id':str(names[index]),'pool_index':index,'initial_matrix':initial.tolist(),'matrix':matrix.tolist(),
            'initial_pairs':pairs,'neighbor_pool_indices':neighbors,'representative_pairs':match(sd,td,initial),
            'geometric_support_count':len(support),'scale_admissible':scale_admissible(matrix,extent),
            'geometry_score_m':float(np.median(tree.query(apply(small,matrix))[0])),'semantic_score_m':semantic_score})
    selected=choose_candidate(coarse,True);full_tree=cKDTree(target);results={}
    record={'method':METHOD,'status':'rejected_insufficient_candidate_support','coarse_indices':list(range(len(coarse)))}
    if selected is not None:
        c=coarse[selected];matrix=refine(downsample(train,.025),downsample(target,.6),np.array(c['matrix']));results[METHOD]=matrix
        error,near=full_tree.query(apply(benchmark[~held],matrix));color_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)))
        evidence=support_evidence(c['initial_pairs'],sf,tf,matrix)
        status='rejected_training_alignment' if np.median(error)>.5 or color_error>.35 or not scale_admissible(matrix,extent) else (
            'candidate_for_review' if evidence['final_spatial_support'] else 'rejected_object_correspondences')
        score,details=matcher.score(matrix)
        record.update({'status':status,'selected_candidate_index':selected,'selected_candidate_id':c['id'],
            'matrix_local':matrix.tolist(),'training_final':summary(error),'training_color_difference':color_error,
            'final_semantic_score_m':score,'final_semantic_class_scores':details,**evidence})
        print(date,METHOD,c['id'],status,'train',round(float(np.median(error)),4),'strict support',evidence['strict_forward_080_spatial_support_count'],flush=True)
    # Introduce comparator final matrices only after new fitting and acceptance finish.
    current_path=ROOT/f'results/object_recovery/{date}/selection.json';current=json.loads(current_path.read_text())
    prior_path=PREVIOUS/date/'audit.json';prior=json.loads(prior_path.read_text());old=next(r for r in prior['runs'] if r['method']=='semantic_fusion')
    results.update({'current_automatic':np.array(current['matrix_local']),'previous_semantic':np.array(old['matrix_local'])})
    comparison_status={METHOD:record['status'],'current_automatic':current['status'],'previous_semantic':old['status']}
    comparison_strict={METHOD:record.get('strict_forward_080_spatial_support_count',0),
        'current_automatic':current['strict_forward_080_spatial_support_count'],'previous_semantic':old['strict_forward_080_spatial_support_count']}
    previous=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz');np.testing.assert_array_equal(benchmark[held],previous['source_points']);fixed=previous['fixed_coverage']
    rows=[];distances={}
    for method,matrix in results.items():
        error=full_tree.query(apply(benchmark[held],matrix))[0];distances[method]=error
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
            rows.append({'date':date,'method':method,'status':comparison_status[method],'strict_support_count':comparison_strict[method],
                'scope':scope,**summary(error[mask]),'within_1m':float(np.mean(error[mask]<1.))})
    np.savez_compressed(folder/'held_out_distances.npz',source_points=benchmark[held],fixed_coverage=fixed,**distances)
    controls_path=ROOT/f'results/object_consensus/{date}/static_control_source_points.npz';controls=np.load(controls_path);statics=[]
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions']
    for region in regions:
        local=cKDTree(target[volume_mask(target,region)])
        for scope in ['unseen_records','spatial_holdout']:
            points=controls[region['id']+'_'+scope]
            for method,matrix in results.items():
                error=local.query(apply(points,matrix))[0]
                stats=summary(error) if len(error) else {'count':0,'median':None,'p90':None,'p95':None,'rmse':None}
                statics.append({'date':date,'method':method,'region':region['id'],'scope':scope,**stats,'adequate_support':len(error)>=30})
    write_csv(folder/'comparison.csv',rows);write_csv(folder/'static_structures.csv',statics)
    inputs=paths+[source_path,target_path,PREVIOUS/date/'initial_scores.npz',prior_path,current_path,controls_path]
    inputs+=[ROOT/f'assets/{date}/{date}.{ext}' for ext in ['ply','las']]
    inputs+=[ROOT/f'results/object_consensus/{date}/{side}_{suffix}' for side in ['source','target'] for suffix in ['candidates.json','candidate_points.npz']]
    audit={'date':date,'initial_pool_size':len(matrices),'initial_starts':starts,'coarse_candidates':coarse,'run':record,
        'comparison':rows,'comparison_status':comparison_status,'comparison_strict_support':comparison_strict,
        'extent_scale_prior':extent,'shared_classes':[GROUPS[c[0]] for c in matcher.classes],
        'source_class_counts':dict(zip(GROUPS,np.bincount(source['labels'],minlength=7).tolist())),
        'target_class_counts':dict(zip(GROUPS,np.bincount(target_labels['labels'],minlength=7).tolist())),
        'inputs':{str(p.relative_to(ROOT)):file_hash(p) for p in inputs},
        'previous_initial_starts':prior['initial_starts']['semantic_fusion'],'previous_selected_candidate':old['selected_candidate_id'],
        'previous_strict_diagnostics':support_evidence(old['final_pair_checks'],sf,tf,np.array(old['matrix_local'])),
        'training_tile_leakage_count':0,'final_matrices_used_for_initialization':False,'fresh_icp_and_refinement':True,'seconds':time.monotonic()-begin}
    (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('Finished',date,round(audit['seconds'],1),'s',flush=True)


def main(dates):
    OUT.mkdir(parents=True,exist_ok=True)
    for p in [LABELS/'validation.json',PREVIOUS/'validation.json']:assert json.loads(p.read_text())['all_checks_passed']
    parent=json.loads((PREVIOUS/'protocol.json').read_text())
    for n,h in parent['source_sha256'].items():assert file_hash(ROOT/'scripts'/n)==h
    assert parent['initial_parameters']==INITIAL_PARAMETERS and parent['semantic_parameters']==PARAMETERS and parent['structure_parameters']==STRUCTURE_PARAMETERS
    protocol={'initial_parameters':INITIAL_PARAMETERS,'semantic_parameters':PARAMETERS,'structure_parameters':STRUCTURE_PARAMETERS,
        'object_parameters':OBJECT_PARAMETERS,'strict_forward_fraction':.8,'training_median_limit_m':.5,'training_rgb_limit':.35,
        'same_initial_pool_and_four_start_budget':True,'unchanged_geometry_and_refinement':True,'all_pair_identities_fixed_before_icp':True,
        'only_source_labels_changed':True,'target_labels':'semantic_fusion','fresh_icp_and_refinement':True,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['align_semantic_3dgs.py','semantic_structures.py','semantic_initial.py','semantic_matching.py','semantic_projection.py','object_consensus.py','align_objects.py','align_object_recovery.py','align_object_consensus.py','align_dates.py','align_0929.py','diagnose_strict_support.py']},
        'parent_protocol_sha256':file_hash(PREVIOUS/'protocol.json'),'label_protocol_sha256':file_hash(LABELS/'segmentation_protocol.json'),
        'parameters_changed_using_withheld_errors':False,'same_site_development_experiment':True}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve frozen experiments.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    for date in dates:
        if (OUT/date/'audit.json').exists():print('Cached',date,flush=True);continue
        run(date)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
