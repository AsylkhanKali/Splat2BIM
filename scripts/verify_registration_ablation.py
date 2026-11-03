"""Independently recompute frozen cohort metrics, ranks and transform conventions."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from project_data import ROOT,ORIGIN,file_hash,write_csv
from registration_ablation import apply,voxel,sample,pca_hypotheses,semantic_classes,scores,matrix_parts,measures,convergence_summary,fine_cloud
from run_registration_ablation import load_data


def next_plane_update(source,T,target,trim):
    points,normals=target;distance,index=cKDTree(points).query(apply(source,T),workers=2,distance_upper_bound=1.)
    keep=(distance<1.)&(distance<=np.quantile(np.minimum(distance,1.),trim))
    if keep.sum()<100:return None,None
    p=apply(source,T)[keep];q=points[index[keep]];n=normals[index[keep]]
    residual=np.sum((p-q)*n,axis=1);w=np.sqrt(np.minimum(1,.25/np.maximum(abs(residual),1e-9)))
    J=np.c_[np.cross(p,n),n,np.sum(p*n,axis=1)]
    step=np.linalg.lstsq(J*w[:,None],-residual*w,rcond=None)[0]
    step/=max(1,np.linalg.norm(step[:3])/.08,abs(step[6])/.05,np.linalg.norm(step[3:6])/5)
    delta=np.eye(4);delta[:3,:3]=np.exp(step[6])*Rotation.from_rotvec(step[:3]).as_matrix();delta[:3,3]=step[3:6]
    return float(np.linalg.norm(step)),delta@T


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/ablation');args=parser.parse_args();out=args.out
    protocol=json.loads((out/'protocol.json').read_text());results=json.loads((out/'results.json').read_text())
    for name,h in protocol['input_sha256'].items():assert file_hash(ROOT/name)==h,name
    for name,h in protocol['source_sha256'].items():assert file_hash(ROOT/'scripts'/name)==h,name
    for name,h in protocol['comparison_only_sha256'].items():assert file_hash(ROOT/name)==h,name
    train,evaluation,fit_ids,test_ids,tiles,target,source_labels,target_labels=load_data();saved=np.load(out/'evaluation.npz')
    for name,value in [('source',evaluation),('source_indices',test_ids),('fit_indices',fit_ids),('tiles',tiles)]:np.testing.assert_array_equal(saved[name],value)
    assert not np.intersect1d(fit_ids,test_ids).size
    pca_source=voxel(train,.06);coarse=voxel(target,2.);pool=pca_hypotheses(pca_source,coarse)
    np.testing.assert_allclose(np.load(out/'pca_candidates.npz')['matrices'],pool,atol=1e-10,rtol=0)
    expected={(c['name'],s) for c in protocol['configurations'] for s in protocol['seeds']}
    assert len(results)==len(expected)==len({(r['name'],r['seed']) for r in results})
    assert {(r['name'],r['seed']) for r in results}==expected
    tree=cKDTree(target);ranktree=cKDTree(coarse);checked=0;probes=[];fine={};fit_by_seed={}
    historical=json.loads((ROOT/'results/0929_baseline/alignment.json').read_text());baseline=np.array(historical['ply_to_las_local'])
    baseline_points=apply(evaluation,baseline);baseline_metrics=measures(tree.query(baseline_points,workers=2)[0])
    archived=json.loads((ROOT/'results/ablation_30_iterations/results.json').read_text());comparisons=[];old_controls_verified=0
    for r in results:
        config=r['config'];seed=r['seed'];rank=sample(pca_source,4096,seed);folder=out/r['name']/str(seed)
        classes=semantic_classes(source_labels,target_labels,config['confidence'],seed,config['shuffle']) if config['weight'] else []
        initial=[scores(T,rank,ranktree,classes,config['weight']) for T in pool]
        assert initial==r['initial_scores']
        order=sorted((i for i in range(4) if initial[i]['combined'] is not None),key=lambda i:(initial[i]['combined'],i))[:config['starts']]
        assert order==r['selected_starts']
        if r['metrics'] is None:
            assert not r['candidates'];continue
        for c in r['candidates']:
            s=scores(np.array(c['matrix']),rank,ranktree,classes,config['weight']);assert s==c['training_score']
            for stage in c['trace']:
                assert 1<=stage['iterations']<=config['iterations']
                tolerance=stage['gate_m']*1e-5 if stage['stage']=='point_to_point' else 1e-6
                assert stage['tolerance']==tolerance
                reason='tolerance' if stage['last_update']<tolerance else 'iteration_limit'
                assert stage['stop_reason']==reason
                if reason=='iteration_limit':assert stage['iterations']==config['iterations']
            assert c['convergence']==convergence_summary(c['trace'])
        selected=min(r['candidates'],key=lambda c:(c['training_score']['combined'],c['index']));assert selected['index']==r['selected_index']
        assert r['convergence']==selected['convergence']
        expected_status='evaluated_pca_only' if not config['icp'] else ('evaluated_terminal_converged' if selected['convergence']['terminal_converged'] else 'evaluated_iteration_limited')
        assert r['status']==expected_status
        assert selected['matrix']==r['matrix'];T=np.array(r['matrix']);matrix_parts(T)
        np.testing.assert_allclose(apply(apply(evaluation[:100],T),np.linalg.inv(T)),evaluation[:100],atol=1e-9)
        for stage,metric in r['stages'].items():
            L=np.loadtxt(folder/f'T_{stage}_local.txt');G=np.loadtxt(folder/f'T_{stage}_global.txt')
            expected_matrix={'pca':pool[selected['index']],'after_semantic_labels':pool[selected['index']],
                             'point_to_point_icp':selected['coarse_matrix'],'final':T}[stage]
            np.testing.assert_allclose(L,expected_matrix,atol=1e-10,rtol=0)
            np.testing.assert_allclose(G[:3,3]-L[:3,3],ORIGIN,atol=1e-8,rtol=0)
            d=tree.query(apply(evaluation,L),workers=2)[0];recomputed=measures(d)
            for key in metric:np.testing.assert_allclose(metric[key],recomputed[key],atol=1e-9,rtol=0)
        np.testing.assert_array_equal(np.loadtxt(folder/'T_pca_local.txt'),np.loadtxt(folder/'T_after_semantic_labels_local.txt'))
        d=tree.query(apply(evaluation,T),workers=2)[0]
        np.testing.assert_allclose(d,np.load(folder/'errors.npz')['distance_m'],atol=1e-10,rtol=0)
        for key,v in measures(d).items():np.testing.assert_allclose(v,r['metrics'][key],atol=1e-10,rtol=0)
        if config['icp']:
            size=config['fine_voxel_m']
            if size not in fine:fine[size]=fine_cloud(target,size)
            if seed not in fit_by_seed:fit_by_seed[seed]=sample(voxel(train,.025),10000,seed)
            fit=fit_by_seed[seed];update,next_T=next_plane_update(fit,T,fine[size],config['trim'])
            first_motion=return_motion=None;two_step_return=False
            if next_T is not None:
                _,second_T=next_plane_update(fit,next_T,fine[size],config['trim'])
                first_motion=float(np.max(np.linalg.norm(apply(fit,next_T)-apply(fit,T),axis=1)))
                if second_T is not None:
                    return_motion=float(np.max(np.linalg.norm(apply(fit,second_T)-apply(fit,T),axis=1)))
                    two_step_return=update>=1e-6 and first_motion>1e-8 and return_motion<1e-8
            probes.append({'name':r['name'],'seed':seed,'terminal_converged':r['convergence']['terminal_converged'],
                           'next_update_norm':update,'next_update_below_tolerance':update is not None and update<1e-6,
                           'first_update_max_displacement_m':first_motion,'two_update_return_max_displacement_m':return_motion,
                           'two_step_return_detected':two_step_return})
        if r['name']=='iteration_budget_030':
            old=next(a for a in archived if a['name']=='pca_icp' and a['seed']==seed)
            np.testing.assert_allclose(T,old['matrix'],atol=1e-10,rtol=0)
            for k,v in r['metrics'].items():np.testing.assert_allclose(v,old['metrics'][k],atol=1e-10,rtol=0)
            old_controls_verified+=1
        shift=np.linalg.norm(apply(evaluation,T)-baseline_points,axis=1);relative=T@np.linalg.inv(baseline)
        scale=float(np.cbrt(np.linalg.det(relative[:3,:3])))
        angle=float(np.rad2deg(np.arccos(np.clip((np.trace(relative[:3,:3]/scale)-1)/2,-1,1))))
        comparisons.append({'name':r['name'],'seed':seed,'max_iterations':config['iterations'] if config['icp'] else 0,
                            'terminal_converged':r['convergence']['terminal_converged'],
                            'median_m':r['metrics']['median_m'],'p90_m':r['metrics']['p90_m'],'rmse_m':r['metrics']['rmse_m'],
                            'median_displacement_from_baseline_m':float(np.median(shift)),
                            'relative_scale':scale,'relative_rotation_degrees':angle})
        checked+=1
    assert old_controls_verified==len(protocol['seeds'])
    write_csv(out/'reference_comparison.csv',comparisons)
    (out/'reference_comparison.json').write_text(json.dumps({'baseline_metrics_same_cohort':baseline_metrics,'baseline_matrix':baseline.tolist(),
        'evaluation_sha256':file_hash(out/'evaluation.npz'),'runs':comparisons,
        'note':'Historical RTK/COLMAP-assisted transform is an evaluation comparator only, not independent truth or a fitting input.'},indent=2)+'\n')
    (out/'convergence_checks.json').write_text(json.dumps(probes,indent=2)+'\n')
    receipt={'status':'passed','date':'0929','expected_runs':len(expected),'evaluated_runs_verified':checked,'failed_runs_retained':len(results)-checked,
      'same_evaluation_indices':True,'fit_evaluation_overlap':0,'raw_input_and_runner_hashes_match':True,
      'pca_pool_recomputed':True,'training_only_candidate_ranking_recomputed':True,'all_stage_distances_recomputed':True,
      'semantic_label_stage_matrix_unchanged':True,'local_global_and_inverse_conventions_verified':True,
      'stop_reasons_and_iteration_counts_verified':True,'historical_30_step_controls_reproduced':old_controls_verified,
      'terminal_converged_runs':sum(p['terminal_converged'] for p in probes),'iteration_limited_runs':sum(not p['terminal_converged'] for p in probes),
      'converged_runs_passing_extra_update':sum(p['terminal_converged'] and p['next_update_below_tolerance'] for p in probes),
      'two_step_return_runs':sum(p['two_step_return_detected'] for p in probes),
      'protocol_sha256':file_hash(out/'protocol.json'),'results_sha256':file_hash(out/'results.json'),'verifier_sha256':file_hash(Path(__file__)),
      'limits':'Numerical and reproducibility checks, not independent survey validation. Fits were not rerun by this verifier.'}
    (out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
