"""Compare class-label revision and pre-ICP correspondence banks on seven dates."""
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
from object_consensus import descriptors,match,spatial_support
from check_las_vertical import file_hash,write_csv
from check_static_structures import volume_mask
from semantic_projection import PARAMETERS
from semantic_matching import SemanticMatcher,choose_candidate
from semantic_initial import INITIAL_PARAMETERS,score_pool,choose_starts
from prepare_semantic_structures import OUT as LABELS
from prepare_semantic_clouds import OUT as OLD_LABELS
from semantic_structures import STRUCTURE_PARAMETERS,correspondence_bank

OUT=ROOT/'results/semantic_structures'
METHODS=['geometry_bank','old_semantic_bank','semantic_structures']


def run(date):
    begin=time.monotonic();folder=OUT/date;folder.mkdir(parents=True,exist_ok=True)
    benchmark,rgb,held=samples(date);train,colors,ids=dense_training(date,benchmark)
    source=dict(np.load(LABELS/date/'source_labels.npz'));target_labels=dict(np.load(LABELS/date/'target_labels.npz'))
    np.testing.assert_array_equal(source['points'],train[source['input_indices']])
    assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    matcher=SemanticMatcher(source,target_labels)
    old_matcher=SemanticMatcher(dict(np.load(OLD_LABELS/date/'source_labels.npz')),dict(np.load(OLD_LABELS/date/'target_labels.npz')))
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
    sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf);extent=sf['normalization_scale']/tf['normalization_scale']
    paths=[ROOT/f'results/object_consensus/{date}/initial_matrices.npz',ROOT/f'results/object_recovery/{date}/initial_matrices.npz']
    first=np.load(paths[0])['multi'];second=np.load(paths[1])['matrices'];matrices=np.concatenate([first,second])
    names=np.array([f'consensus_{i}' for i in range(len(first))]+[f'recovery_{i}' for i in range(len(second))])
    valid=np.array([scale_admissible(m,extent) for m in matrices]);matrices,names=matrices[valid],names[valid]
    _,unique=np.unique(np.round(matrices,8).reshape(len(matrices),16),axis=0,return_index=True);unique=np.sort(unique)
    matrices,names=matrices[unique],names[unique]
    small=downsample(train,.06);target_small=downsample(target,2.);tree=cKDTree(target_small)
    score_source=downsample(train,.12);rng=np.random.default_rng(29)
    if len(score_source)>INITIAL_PARAMETERS['geometry_score_points']:
        score_source=score_source[rng.choice(len(score_source),INITIAL_PARAMETERS['geometry_score_points'],replace=False)]
    scores=score_pool(score_source,tree,matcher,matrices);anchors=score_source[::max(1,len(score_source)//20)]
    old_scores=score_pool(score_source,tree,old_matcher,matrices)
    starts={'geometry_bank':choose_starts(matrices,scores,anchors,False),
        'old_semantic_bank':choose_starts(matrices,old_scores,anchors,True),
        'semantic_structures':choose_starts(matrices,scores,anchors,True)}
    np.savez_compressed(folder/'initial_scores.npz',matrices=matrices,names=names,scores=scores,old_scores=old_scores,score_source=score_source)
    print(date,'initial pool',len(matrices),'starts',starts,'ranking seconds',round(time.monotonic()-begin,1),flush=True)
    coarse=[];lookup={}
    union=sorted(set(i for group in starts.values() for i in group))
    banks={i:correspondence_bank(matrices,i,anchors,sd,td) for i in union}
    for index in union:
        initial=matrices[index];pairs,neighbors=banks[index]
        matrix=icp(small,target_small,initial,INITIAL_PARAMETERS['coarse_gates_m'],with_scale=True,iterations=INITIAL_PARAMETERS['coarse_iterations'])
        kept=surviving_pairs(pairs,sd,td,matrix);support=spatial_support(kept,sd,td,matrix)
        semantic_score,_=matcher.score(matrix);old_semantic_score,_=old_matcher.score(matrix)
        lookup[index]=len(coarse)
        coarse.append({'id':str(names[index]),'pool_index':index,'initial_matrix':initial.tolist(),'matrix':matrix.tolist(),
            'initial_pairs':pairs,'neighbor_pool_indices':neighbors,'representative_pairs':match(sd,td,initial),'geometric_support_count':len(support),'scale_admissible':scale_admissible(matrix,extent),
            'geometry_score_m':float(np.median(tree.query(apply(small,matrix))[0])),'semantic_score_m':semantic_score,'old_semantic_score_m':old_semantic_score})
    fine_source=downsample(train,.025);fine_target=downsample(target,.6);full_tree=cKDTree(target)
    refined={};runs=[];results={}
    for method,initial_indices in starts.items():
        indices=[lookup[i] for i in initial_indices]
        selection_rows=[{**coarse[i],'semantic_score_m':coarse[i]['old_semantic_score_m'] if method=='old_semantic_bank' else coarse[i]['semantic_score_m']} for i in indices]
        selected=choose_candidate(selection_rows,method!='geometry_bank')
        if selected is None:
            runs.append({'method':method,'status':'rejected_insufficient_candidate_support','coarse_indices':indices});continue
        index=indices[selected];c=coarse[index]
        if index not in refined:refined[index]=refine(fine_source,fine_target,np.array(c['matrix']))
        matrix=refined[index];results[method]=matrix
        error,near=full_tree.query(apply(benchmark[~held],matrix));color_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)))
        patches=patch_checks(c['initial_pairs'],sf,tf,matrix)
        current={(p['source_id'],p['target_id']):p for p in surviving_pairs(c['initial_pairs'],sd,td,matrix)}
        valid=[{**p,'center_distance_m':current[(p['source_id'],p['target_id'])]['center_distance_m']} for p in patches
            if (p['source_id'],p['target_id']) in current and min(p['fraction_within_1m'],p['reverse_fraction_within_1m'])>=.5]
        support=spatial_support(valid,sd,td,matrix);strict=spatial_support([p for p in valid if p['fraction_within_1m']>=.8],sd,td,matrix)
        status='rejected_training_alignment' if np.median(error)>.5 or color_error>.35 or not scale_admissible(matrix,extent) else (
            'candidate_for_review' if support else 'rejected_object_correspondences')
        runs.append({'method':method,'status':status,'coarse_indices':indices,'selected_candidate_index':index,
            'selected_candidate_id':c['id'],'matrix_local':matrix.tolist(),'training_final':summary(error),
            'training_color_difference':color_error,'final_pair_checks':patches,'final_spatial_support':support,
            'strict_forward_080_spatial_support_count':len(strict)})
        print(date,method,c['id'],status,'train',round(float(np.median(error)),4),'strict pairs',len(strict),flush=True)
    # Current final matrix is introduced only after both new methods have finished.
    current_path=ROOT/f'results/object_recovery/{date}/selection.json';current=json.loads(current_path.read_text())
    results['current_automatic']=np.array(current['matrix_local'])
    previous=np.load(ROOT/f'results/object_consensus/{date}/held_out_distances.npz')
    np.testing.assert_array_equal(benchmark[held],previous['source_points']);fixed=previous['fixed_coverage']
    rows=[];distances={}
    for method,matrix in results.items():
        error=full_tree.query(apply(benchmark[held],matrix))[0];distances[method]=error
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
            rows.append({'date':date,'method':method,'scope':scope,**summary(error[mask]),'within_1m':float(np.mean(error[mask]<1.))})
    np.savez_compressed(folder/'held_out_distances.npz',source_points=benchmark[held],fixed_coverage=fixed,**distances)
    controls=np.load(ROOT/f'results/object_consensus/{date}/static_control_source_points.npz');statics=[]
    regions=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions']
    for region in regions:
        local=cKDTree(target[volume_mask(target,region)])
        for scope in ['unseen_records','spatial_holdout']:
            control=controls[region['id']+'_'+scope]
            for method,matrix in results.items():
                errors=local.query(apply(control,matrix))[0]
                stats=summary(errors) if len(errors) else {'count':0,'median':None,'p90':None,'p95':None,'rmse':None}
                statics.append({'date':date,'method':method,'region':region['id'],'scope':scope,**stats,'adequate_support':len(errors)>=30})
    write_csv(folder/'comparison.csv',rows);write_csv(folder/'static_structures.csv',statics)
    audit={'date':date,'initial_pool_size':len(matrices),'initial_starts':starts,'coarse_candidates':coarse,'runs':runs,
        'comparison':rows,'extent_scale_prior':extent,'current_status':current['status'],
        'current_strict_support':current['strict_forward_080_spatial_support_count'],
        'input_pool_hashes':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        'label_hashes':{str((labels/date/f'{side}_labels.npz').relative_to(ROOT)):file_hash(labels/date/f'{side}_labels.npz') for labels in [LABELS,OLD_LABELS] for side in ['source','target']},
        'comparison_input':{'path':str(current_path.relative_to(ROOT)),'sha256':file_hash(current_path)},
        'training_tile_leakage_count':0,'final_matrices_used_for_initialization':False,'seconds':time.monotonic()-begin}
    (folder/'audit.json').write_text(json.dumps(audit,indent=2)+'\n');print('Finished',date,round(audit['seconds'],1),'s',flush=True)


def main(dates):
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'initial_parameters':INITIAL_PARAMETERS,'semantic_parameters':PARAMETERS,'structure_parameters':STRUCTURE_PARAMETERS,
        'same_initial_pool_and_four_start_budget':True,'semantics_applied_before_icp':True,
        'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['align_semantic_structures.py','semantic_structures.py','semantic_initial.py','semantic_matching.py','semantic_projection.py','object_consensus.py','align_object_recovery.py','align_object_consensus.py','align_dates.py','align_0929.py']},
        'parent_experiment':'semantic_initial: 0723 lost a dispersed seed identity',
        'methods':METHODS,'all_pair_identities_fixed_before_icp':True,
        'old_results_used_to_identify_development_failure':True,
        'parameters_changed_using_withheld_errors':False}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Protocol changed; preserve prior results.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    for date in dates:
        if (OUT/date/'audit.json').exists():print('Cached registration',date,flush=True);continue
        run(date)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dates',nargs='*',default=DATES);main(parser.parse_args().dates)
