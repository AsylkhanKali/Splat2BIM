"""Fit and select local structural corrections before reading test/control errors."""
import argparse
import json

import laspy
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_0929 import apply,summary
from align_objects import dense_training
from align_stable_structures import support_evidence
from prepare_stable_structures import load_structures,OUT as BASE
from structural_surfaces import PARAMETERS as SURFACE_PARAMETERS
from structural_pose_refinement import PARAMETERS,footprint_mask,sample_indices,target_surface,metrics,refine,select_candidate,extent_anchors
from check_las_vertical import file_hash

OUT=ROOT/'results/structural_refinement'


def usable_semantics(points,labels,found):
    distance,nearest=cKDTree(labels['points']).query(points)
    return ~((distance*found['normalization_scale']<=SURFACE_PARAMETERS['semantic_exclusion_distance'])&np.isin(labels['labels'][nearest],SURFACE_PARAMETERS['excluded_classes']))


def prepare_groups(date,train,ids,target,initial,sf,tf,support):
    source_labels=dict(np.load(ROOT/f'results/semantic_3dgs/{date}/source_labels.npz'))
    target_labels=dict(np.load(ROOT/f'results/semantic_fusion/{date}/target_labels.npz'))
    source_valid=usable_semantics(train,source_labels,sf);target_valid=usable_semantics(target,target_labels,tf)
    groups=[];metadata=[];cache={};excluded=[]
    for number,pair in enumerate(support):
        a=sf['candidates'][pair['source_index']];b=tf['candidates'][pair['target_index']]
        si=np.flatnonzero(footprint_mask(train,sf,a)&source_valid);ti=np.flatnonzero(footprint_mask(target,tf,b)&target_valid)
        if len(si)==0 or len(ti)<PARAMETERS['normal_neighbors']:
            excluded.append({'source_id':a['id'],'target_id':b['id'],'reason':'insufficient_raw_samples'});continue
        si=si[sample_indices(apply(train[si],initial))];ti=ti[sample_indices(target[ti])]
        block=np.floor(apply(train[si],initial)/PARAMETERS['calibration_block_m']).astype(int)
        calibration=(block@np.array([1,3,7]))%PARAMETERS['calibration_modulus']==0
        fit_ids=si[~calibration];calibration_ids=si[calibration]
        if len(fit_ids)<PARAMETERS['minimum_fit_points'] or len(calibration_ids)<PARAMETERS['minimum_calibration_points']:
            excluded.append({'source_id':a['id'],'target_id':b['id'],'reason':'insufficient_fit_or_calibration','fit':len(fit_ids),'calibration':len(calibration_ids)});continue
        key=str(number);surface=target_surface(target[ti]);family=a['family']+' → '+b['family']
        groups.append({'family':family,'fit':train[fit_ids],'calibration':train[calibration_ids],'target':surface})
        metadata.append({'key':key,'family':family,'source_id':a['id'],'target_id':b['id'],'source_index':pair['source_index'],'target_index':pair['target_index'],
            'fit_points':len(fit_ids),'calibration_points':len(calibration_ids),'target_points':len(ti),'valid_target_normals':int(surface['valid_normals'].sum())})
        for name,value in [('fit',train[fit_ids]),('calibration',train[calibration_ids]),('target',target[ti]),
            ('fit_ply_ids',ids[fit_ids]),('calibration_ply_ids',ids[calibration_ids]),('target_las_ids',ti)]:cache[key+'_'+name]=value
    return groups,metadata,cache,excluded


def run(date):
    folder=OUT/date;folder.mkdir(exist_ok=True)
    parent=json.loads((BASE/date/'audit.json').read_text());old=parent['run'];initial=np.array(old['matrix_local'])
    assert old['status']=='candidate_for_review' and old['strict_forward_080_spatial_support_count']>=3
    sf,tf=load_structures(date,'source'),load_structures(date,'target');benchmark,rgb,held=samples(date)
    train,_,ids=dense_training(date,benchmark)
    assert np.all(ids%2==0) and np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    color=np.c_[las.red,las.green,las.blue].astype(float);color/=65535 if color.max()>255 else 255
    tree=cKDTree(target);groups,metadata,cache,excluded=prepare_groups(date,train,ids,target,initial,sf,tf,old['strict_spatial_support'])
    anchors=extent_anchors(train);cache['anchors']=anchors;np.savez_compressed(folder/'fit_data.npz',**cache)
    bank=parent['coarse_candidates'][old['selected_candidate_index']]['initial_pairs']
    error=tree.query(apply(benchmark[~held],initial))[0];baseline_training=summary(error)
    baseline=metrics(groups,initial) if groups else None;candidates=[]
    print(date,'families',len(groups),'fit/calibration',[(m['fit_points'],m['calibration_points']) for m in metadata],flush=True)
    if len(groups)>=3:
        for weight in PARAMETERS['tangent_weights']:
            matrix,diagnostic=refine(groups,initial,weight,anchors)
            error,near=tree.query(apply(benchmark[~held],matrix));color_error=float(np.median(np.linalg.norm(rgb[~held]-color[near],axis=1)))
            evidence=support_evidence(bank,sf,tf,matrix);calibration=metrics(groups,matrix)
            strict=bool(np.median(error)<=.5 and color_error<=.35 and evidence['strict_forward_080_spatial_support_count']>=3)
            record={'tangent_weight':weight,'matrix_local':matrix.tolist(),'optimization':diagnostic,'fit':metrics(groups,matrix,'fit'),
                'calibration':calibration,'training':summary(error),'training_rgb':color_error,'strict_pass':strict,
                'global_training_guard':bool(np.median(error)<=baseline_training['median']+PARAMETERS['maximum_training_median_regression_m']),**evidence}
            candidates.append(record)
            print(date,weight,'calibration',round(calibration['loss'],6),'was',round(baseline['loss'],6),'strict',strict,flush=True)
    selected=select_candidate(candidates,baseline) if candidates else None
    matrix=initial if selected is None else np.array(candidates[selected]['matrix_local'])
    np.savetxt(folder/'T_3dgs_to_las_local.txt',matrix,fmt='%.17g');global_matrix=matrix.copy();global_matrix[:3,3]+=ORIGIN
    np.savetxt(folder/'T_3dgs_to_las_global.txt',global_matrix,fmt='%.17g')
    paths=[BASE/date/'audit.json',BASE/'protocol.json',BASE/'extraction_protocol.json',BASE/'validation.json']
    paths+=[BASE/date/f'{side}_{suffix}' for side in ['source','target'] for suffix in ['structures.json','structure_points.npz']]
    paths+=[ROOT/f'assets/{date}/{date}.{ext}' for ext in ['ply','las']]
    paths+=[ROOT/f'results/{kind}/{date}/{side}_labels.npz' for kind,side in [('semantic_3dgs','source'),('semantic_fusion','target')]]
    audit={'date':date,'initial_matrix':initial.tolist(),'matrix_local':matrix.tolist(),'selected_candidate':selected,
        'status':'refined' if selected is not None else 'baseline_retained','baseline_calibration':baseline,'baseline_training':baseline_training,
        'groups':metadata,'excluded_groups':excluded,'candidates':candidates,'fixed_prefit_bank':bank,
        'fitting_data_sha256':file_hash(folder/'fit_data.npz'),'inputs':{str(p.relative_to(ROOT)):file_hash(p) for p in paths},
        'held_or_manual_control_errors_used_for_selection':False,'scale_fixed_to_parent':True}
    (folder/'fit.json').write_text(json.dumps(audit,indent=2)+'\n');print(date,audit['status'],'selected',selected,flush=True)


def main(dates):
    OUT.mkdir(exist_ok=True)
    parent=json.loads((BASE/'protocol.json').read_text());assert json.loads((BASE/'validation.json').read_text())['all_checks_passed']
    for name,h in parent['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==h
    protocol={'refinement_parameters':PARAMETERS,'object_parameters':parent['object_parameters'],
        'strict_forward_fraction':parent['strict_forward_fraction'],'training_median_limit_m':.5,'training_rgb_limit':.35,
        'parent_protocol_sha256':file_hash(BASE/'protocol.json'),'parent_validation_sha256':file_hash(BASE/'validation.json'),
        'initialization':'Previous stable-structure final matrix; this is an explicit local refinement experiment.',
        'same_rules_all_dates':True,'rigid_correction_only':True,'selection_uses_held_out_errors':False,
        'source_sha256':{name:file_hash(ROOT/'scripts'/name) for name in ['fit_structural_refinement.py','structural_pose_refinement.py']}}
    path=OUT/'protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol,'Preserve frozen experiments.'
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    for date in dates:
        if (OUT/date/'fit.json').exists():print('Cached',date,flush=True);continue
        run(date)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dates',nargs='*',default=DATES);main(p.parse_args().dates)
