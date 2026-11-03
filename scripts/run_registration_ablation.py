"""Freeze and run the 0929 PCA/ICP/semantic ablation without baseline pose inputs."""
import argparse,json,time,platform
from pathlib import Path
from importlib.metadata import version
import laspy
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from project_data import ROOT,ORIGIN,file_hash,write_csv
from registration_ablation import apply,voxel,sample,held_tiles,pca_hypotheses,semantic_classes,scores,refine,fine_cloud,measures,matrix_parts,convergence_summary


def configurations():
    base={'starts':1,'weight':.5,'confidence':.6,'trim':.75,'fine_voxel_m':.6,'shuffle':False,'icp':True,'iterations':500}
    rows=[('pca_only',{'weight':0.,'icp':False}),('pca_icp',{'weight':0.}),('pca_semantic_icp',{}),
          ('geometry_starts_4',{'weight':0.,'starts':4}),('semantic_starts_4',{'starts':4}),
          ('semantic_weight_025',{'weight':.25}),('semantic_weight_100',{'weight':1.}),
          ('confidence_070',{'confidence':.7}),('confidence_080',{'confidence':.8}),
          ('trim_060',{'trim':.6}),('trim_090',{'trim':.9}),
          ('fine_voxel_030',{'fine_voxel_m':.3}),('fine_voxel_120',{'fine_voxel_m':1.2}),
          ('shuffled_source_labels',{'shuffle':True}),
          ('iteration_budget_030',{'weight':0.,'iterations':30}),('iteration_budget_200',{'weight':0.,'iterations':200})]
    return [{'name':name,**base,**changes} for name,changes in rows]


def load_data():
    vertices=PlyData.read(ROOT/'assets/0929/0929.ply',mmap='r')['vertex'].data
    ids=np.arange(0,len(vertices),10);v=vertices[ids]
    xyz=np.c_[v['x'],v['y'],v['z']].astype(float)
    valid=np.isfinite(xyz).all(1)&(v['opacity']>0);xyz,ids=xyz[valid],ids[valid]
    tiles,held=held_tiles(xyz)
    lo,hi=np.quantile(xyz[~held],[.005,.995],axis=0)
    fit=(~held)&np.all((xyz>=lo)&(xyz<=hi),axis=1)
    las=laspy.read(ROOT/'assets/0929/0929.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    source=dict(np.load(ROOT/'results/segmentation/source_labels.npz'))
    labels=dict(np.load(ROOT/'results/segmentation/target_labels.npz'))
    assert not held_tiles(source['points'])[1].any(),'Semantic source overlaps held tiles'
    np.testing.assert_allclose(source['points'],np.c_[vertices['x'],vertices['y'],vertices['z']][source['original_ply_indices']],atol=1e-7)
    return xyz[fit],xyz[held],ids[fit],ids[held],tiles[held],target,source,labels


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=ROOT/'results/ablation');args=parser.parse_args();out=args.out
    if (out/'protocol.json').exists():raise FileExistsError('Use a new output folder; frozen runs are never overwritten.')
    out.mkdir(parents=True,exist_ok=True)
    inputs=['assets/0929/0929.ply','assets/0929/0929.las','results/segmentation/source_labels.npz','results/segmentation/target_labels.npz']
    protocol={'date':'0929','frozen_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'seeds':[17,29,43],
      'configurations':configurations(),'input_sha256':{n:file_hash(ROOT/n) for n in inputs},
      'comparison_only_sha256':{n:file_hash(ROOT/n) for n in ['results/0929_baseline/alignment.json','results/ablation_30_iterations/results.json','results/ablation_30_iterations/protocol.json']},
      'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['registration_ablation.py','run_registration_ablation.py','project_data.py']},
      'runtime':{n:version(n) for n in ['numpy','scipy','open3d','laspy','plyfile']},'platform':platform.platform(),
      'source_record_stride':10,'source_opacity_logit_min':0,'source_tile_width':.4,'heldout_rule':'sum(floor(source_xyz / 0.4)) modulo 4 == 0',
      'fit_only_quantile_crop':[.005,.995],'evaluation_crop':'none; all finite positive-opacity held-out records',
      'coarse_source_voxel':.06,'coarse_target_voxel_m':2.,'fit_points_max':10000,'ranking_points_max':4096,
      'candidate_pool':'Four proper PCA sign hypotheses; no RTK, manual seed, old fitted pose, or correspondence bank',
      'candidate_selection':'75th-percentile training geometry distance + weight * class-balanced 60th-percentile capped semantic distance',
      'semantic_classes':[1,2,3,4],'minimum_points_per_class':128,'minimum_shared_classes':2,'semantic_distance_cap_m':20.,
      'primary_iterations_per_gate':500,'point_to_point_gates_m':[40,20,10,5],'point_to_plane_gates_m':[18,9,4,2,1],
      'normal_radius_m':3.,'normal_max_neighbors':40,'robust_plane_scale_m':.25,
      'stopping_criteria':{'point_to_point':'RMS correspondence displacement < gate * 1e-5 metres','point_to_plane':'Bounded rotation/translation/log-scale increment norm < 1e-6'},
      'quantile_ties':'Include distances equal to the trim quantile, but retain the strict distance gate. This keeps exact zero-error matches; 30-step controls are compared numerically with the archive.',
      'convergence_policy':'Record every stage stop reason. Terminal tolerance means numerical convergence, not pose correctness. Iteration-limited candidates are retained and ranked by the same training score, never silently excluded.',
      'fit_cache':'Within each seed, identical candidate/trim/voxel/iteration fits are reused across ranking-only variants. Every branch is independently rescored and evaluated. This is not a timing benchmark.',
      'revision_reason':'The previous 30-step experiment reached iteration limits. A separate seed-29 diagnostic at 200 steps motivated this follow-up; it is not a blind replication.',
      'semantic_input':'Frozen fused predictions from previous 0929 development; confidence ablations only reject more cached labels, not rerun the networks',
      'semantic_stage_changes_coordinates':False,'bim_used_for_fit_or_evaluation':False,
      'limitations':['One previously explored capture, not an independent new-site test.','Labels/render settings were developed previously; Gaussian footprints can cross held-out boundaries.','LAS is reference geometry, not independent survey truth.','Seeds measure sampling sensitivity, not repeated independent captures.','No model-selection claim is made from the ablation outcomes.'],
      'bootstrap':{'replicates':1000,'seed':2026,'block_width_source_units':1.6,'statistic':'equal-weight mean of occupied block medians; paired differences; not the pooled point median'}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    train,evaluation,fit_ids,test_ids,tiles,target,source_labels,target_labels=load_data()
    assert not np.intersect1d(fit_ids,test_ids).size
    np.savez_compressed(out/'evaluation.npz',source=evaluation,source_indices=test_ids,fit_indices=fit_ids,tiles=tiles)
    coarse=voxel(target,2.);pca_source=voxel(train,.06);pool=pca_hypotheses(pca_source,coarse)
    assert len(pool)==4
    np.savez_compressed(out/'pca_candidates.npz',matrices=np.array(pool))
    fulltree=cKDTree(target);ranktree=cKDTree(coarse);fine={};results=[];rows=[]
    counts={'fit_records':len(train),'heldout_records':len(evaluation),'las_points':len(target),'pca_source_points':len(pca_source),'pca_target_points':len(coarse),'fit_test_overlap':0,'semantic_source_held_tile_overlap':0}
    (out/'sample_counts.json').write_text(json.dumps(counts,indent=2)+'\n')
    print('Samples',counts,flush=True)
    for seed in protocol['seeds']:
        fit=sample(voxel(train,.025),10000,seed);rank=sample(pca_source,4096,seed);fit_cache={}
        for config in protocol['configurations']:
            started=time.perf_counter();name=config['name'];folder=out/name/str(seed);folder.mkdir(parents=True,exist_ok=True)
            classes=semantic_classes(source_labels,target_labels,config['confidence'],seed,config['shuffle']) if config['weight'] else []
            initial=[scores(T,rank,ranktree,classes,config['weight']) for T in pool]
            order=sorted((i for i in range(4) if initial[i]['combined'] is not None),key=lambda i:(initial[i]['combined'],i))[:config['starts']]
            candidates=[];failures=[]
            for i in order:
                try:
                    T=pool[i].copy();coarse_T=T.copy();trace=[];reused=False;fit_seconds=0.
                    if config['icp']:
                        size=config['fine_voxel_m']
                        if size not in fine:fine[size]=fine_cloud(target,size)
                        key=(i,config['trim'],size,config['iterations']);reused=key in fit_cache
                        if not reused:
                            fit_started=time.perf_counter()
                            fitted=refine(fit,coarse,T,config['trim'],config['iterations'],fine[size])
                            fit_cache[key]=(*fitted,time.perf_counter()-fit_started)
                        T,coarse_T,trace,fit_seconds=fit_cache[key]
                    parts=matrix_parts(T);score=scores(T,rank,ranktree,classes,config['weight'])
                    candidates.append({'index':i,'matrix':T.tolist(),'coarse_matrix':coarse_T.tolist(),'parts':parts,'training_score':score,'trace':trace,
                                       'convergence':convergence_summary(trace),'fit_reused':reused,'fit_seconds':fit_seconds})
                except (ValueError,np.linalg.LinAlgError) as error:failures.append({'index':i,'reason':str(error)})
            record={'name':name,'seed':seed,'config':config,'initial_scores':initial,'selected_starts':order,'semantic_class_ids':[c[0] for c in classes],'candidates':candidates,'failures':failures,'registration_seconds':time.perf_counter()-started}
            if not candidates:
                record['status']='rejected_no_candidate';record['metrics']=None
                rows.append({'name':name,'seed':seed,'status':record['status'],'candidate':None,'median_m':None,'p90_m':None,'rmse_m':None,'within_025m':None,'within_05m':None,'within_1m':None,'registration_seconds':record['registration_seconds']})
            else:
                selected=min(candidates,key=lambda c:(c['training_score']['combined'],c['index']));T=np.array(selected['matrix']);record['selected_index']=selected['index']
                record['convergence']=selected['convergence']
                record['status']=('evaluated_pca_only' if not config['icp'] else
                                  ('evaluated_terminal_converged' if record['convergence']['terminal_converged'] else 'evaluated_iteration_limited'))
                error=fulltree.query(apply(evaluation,T),workers=2)[0]
                record['metrics']=measures(error);record['matrix']=T.tolist();record['parts']=selected['parts']
                record['stages']={}
                for stage,matrix in [('pca',pool[selected['index']]),('after_semantic_labels',pool[selected['index']]),('point_to_point_icp',np.array(selected['coarse_matrix'])),('final',T)]:
                    e=fulltree.query(apply(evaluation,matrix),workers=2)[0];record['stages'][stage]=measures(e)
                    np.savetxt(folder/f'T_{stage}_local.txt',matrix,fmt='%.15g')
                    global_T=matrix.copy();global_T[:3,3]+=ORIGIN;np.savetxt(folder/f'T_{stage}_global.txt',global_T,fmt='%.15g')
                np.savez_compressed(folder/'errors.npz',distance_m=error)
                rows.append({'name':name,'seed':seed,'status':record['status'],'candidate':selected['index'],**{k:v for k,v in record['metrics'].items() if k!='n'},'registration_seconds':record['registration_seconds']})
            (folder/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');results.append(record)
            print(name,seed,record['metrics'],flush=True)
    write_csv(out/'summary.csv',rows)
    (out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    for name,digest in protocol['input_sha256'].items():assert file_hash(ROOT/name)==digest
    (out/'completed.json').write_text(json.dumps({'runs':len(results),'input_hashes_unchanged':True,'evaluation_sha256':file_hash(out/'evaluation.npz'),'results_sha256':file_hash(out/'results.json')},indent=2)+'\n')

if __name__=='__main__':main()
