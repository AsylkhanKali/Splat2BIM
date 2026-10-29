"""Evaluate frozen selected corrections on unchanged and additional fixed controls."""
import json

import laspy
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN,samples
from align_0929 import apply,summary
from align_stable_structures import support_evidence
from fit_structural_refinement import OUT,BASE,usable_semantics
from prepare_stable_structures import load_structures
from structural_pose_refinement import footprint_mask
from check_static_structures import volume_mask
from check_las_vertical import file_hash,write_csv

METHOD='structural_refinement'


def measurements(points,tree,matrix):
    error=tree.query(apply(points,matrix))[0]
    return error,summary(error) if len(error) else {'count':0,'median':None,'p90':None,'p95':None,'rmse':None}


def run(date,out=OUT):
    folder=out/date;fit=json.loads((folder/'fit.json').read_text());parent=json.loads((BASE/date/'audit.json').read_text())
    current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text())
    matrices={'current_automatic':np.array(current['matrix_local']),'stable_parent':np.array(fit['initial_matrix']),METHOD:np.array(fit['matrix_local'])}
    sf,tf=load_structures(date,'source'),load_structures(date,'target')
    evidence=support_evidence(fit['fixed_prefit_bank'],sf,tf,matrices[METHOD])
    assert evidence['strict_forward_080_spatial_support_count']>=3
    benchmark,_,held=samples(date);previous=np.load(BASE/date/'held_out_distances.npz');fixed=previous['fixed_coverage']
    np.testing.assert_array_equal(benchmark[held],previous['source_points'])
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN;tree=cKDTree(target)
    rows=[];errors={'source_points':benchmark[held],'fixed_coverage':fixed}
    for name,matrix in matrices.items():
        distance=tree.query(apply(benchmark[held],matrix))[0];errors[name]=distance
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed)]:
            rows.append({'date':date,'method':name,'scope':scope,**summary(distance[mask]),'within_1m':float(np.mean(distance[mask]<1))})
    np.savez_compressed(folder/'held_out_distances.npz',**errors);write_csv(folder/'comparison.csv',rows)
    controls_path=ROOT/f'results/object_consensus/{date}/static_control_source_points.npz';controls=np.load(controls_path)
    regions_path=ROOT/'results/static_structures/regions.json';regions=json.loads(regions_path.read_text())['regions'];statics=[]
    for region in regions:
        local=cKDTree(target[volume_mask(target,region)])
        for scope in ['unseen_records','spatial_holdout']:
            points=controls[region['id']+'_'+scope]
            for name,matrix in matrices.items():
                _,stats=measurements(points,local,matrix)
                statics.append({'date':date,'method':name,'region':region['id'],'scope':scope,**stats,'adequate_support':len(points)>=30})
    write_csv(folder/'static_structures.csv',statics)
    # New masks use the parent's fixed source footprints; no post-correction recropping.
    v=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data[5::10]
    odd=np.c_[v['x'],v['y'],v['z']].astype(float);ids=np.arange(5,5+10*len(v),10)
    valid=(v['opacity']>0)&np.isfinite(odd).all(1);odd,ids=odd[valid],ids[valid]
    labels=dict(np.load(ROOT/f'results/semantic_fusion/{date}/target_labels.npz'));target_valid=usable_semantics(target,labels,tf)
    structural=[];control_cache={};selected=parent['run']['strict_spatial_support']
    for number,pair in enumerate(selected):
        a=sf['candidates'][pair['source_index']];b=tf['candidates'][pair['target_index']];family=a['family']+' → '+b['family']
        target_ids=np.flatnonzero(footprint_mask(target,tf,b)&target_valid);local=cKDTree(target[target_ids])
        control_cache[str(number)+'_target_las_ids']=target_ids
        for scope,pool,pool_ids in [('spatial_holdout',benchmark[held],np.flatnonzero(held)),('unseen_records',odd,ids)]:
            mask=footprint_mask(pool,sf,a);points=pool[mask]
            control_cache[str(number)+'_'+scope]=points;control_cache[str(number)+'_'+scope+'_ids']=pool_ids[mask]
            for name,matrix in matrices.items():
                distance,stats=measurements(points,local,matrix);control_cache[str(number)+'_'+scope+'_'+name]=distance
                structural.append({'date':date,'method':name,'family':family,'source_id':a['id'],'target_id':b['id'],'scope':scope,
                    **stats,'adequate_support':len(points)>=30})
    np.savez_compressed(folder/'structural_controls.npz',**control_cache)
    write_csv(folder/'structural_controls.csv',structural)
    candidate_rows=[]
    for i,c in enumerate(fit['candidates']):
        error=tree.query(apply(benchmark[held][fixed],np.array(c['matrix_local'])))[0]
        candidate_rows.append({'date':date,'candidate_index':i,'tangent_weight':c['tangent_weight'],'strict_pass':c['strict_pass'],
            'selected':i==fit['selected_candidate'],'calibration_loss':c['calibration']['loss'],**summary(error)})
    if candidate_rows:write_csv(folder/'candidate_comparison.csv',candidate_rows)
    evaluation={'date':date,'fit_sha256':file_hash(folder/'fit.json'),'controls_sha256':file_hash(controls_path),'regions_sha256':file_hash(regions_path),
        'current_selection_sha256':file_hash(ROOT/f'results/object_recovery/{date}/selection.json'),'selected_evidence':evidence,
        'strict_pass':True,'controls_fixed_before_correction':True,'control_errors_change_selection':False}
    (folder/'evaluation.json').write_text(json.dumps(evaluation,indent=2)+'\n');print('Evaluated',date,fit['status'],flush=True)


def main(out=OUT):
    assert all((out/date/'fit.json').exists() for date in DATES),'Freeze all selections before evaluating controls.'
    for date in DATES:run(date,out)


if __name__=='__main__':main()
