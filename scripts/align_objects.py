"""Compare object-guided and PCA initialization with identical geometric refinement."""
import argparse
import itertools
import json
import time

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,samples,refine
from align_0929 import apply,downsample,icp,summary
from check_las_vertical import file_hash,write_csv
from object_candidates import PARAMETERS,detect,guesses,serializable,normalize,rectangle
from check_static_structures import volume_mask

OUT=ROOT/'results/object_alignment'


def dense_training(date,benchmark_points):
    vertices=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data[::2]
    points=np.c_[vertices['x'],vertices['y'],vertices['z']].astype(float)
    colors=np.clip(.5+.28209479177387814*np.c_[vertices['f_dc_0'],vertices['f_dc_1'],vertices['f_dc_2']],0,1)
    keep=(vertices['opacity']>0)&np.isfinite(points).all(1)
    points,colors=points[keep],colors[keep];ids=np.arange(0,len(vertices)*2,2)[keep]
    keep=np.all((points>=benchmark_points.min(0))&(points<=benchmark_points.max(0)),axis=1)
    keep&=np.floor(points/.4).sum(1).astype(int)%4!=0
    return points[keep],colors[keep],ids[keep]


def pca_guesses(source,target):
    a,ua=np.linalg.eigh(np.cov(source.T));b,ub=np.linalg.eigh(np.cov(target.T));scale=np.sqrt(b.sum()/a.sum());rows=[]
    for signs in itertools.product([-1,1],repeat=3):
        rotation=ub@np.diag(signs)@ua.T
        if np.linalg.det(rotation)<0:continue
        matrix=np.eye(4);matrix[:3,:3]=scale*rotation;matrix[:3,3]=target.mean(0)-matrix[:3,:3]@source.mean(0)
        rows.append({'signs':signs,'matrix':matrix})
    return rows


def reference_roof(points,colors):
    normalized,center,scale=normalize(points)
    region=next(r for r in json.loads((ROOT/'results/las_rigid/regions.json').read_text())['regions'] if r['id']=='roof_shell')
    keep=volume_mask(points,region)&(colors.min(1)>.55)&(np.ptp(colors,axis=1)<.18)
    cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(normalized[keep]));o3d.utility.random.seed(29)
    _,indices=cloud.segment_plane(distance_threshold=.05,ransac_n=3,num_iterations=1500)
    patch=np.asarray(cloud.points)[indices];candidate=rectangle(patch)
    if candidate['frame'][2,2]<0:candidate['frame'][:,1:]*=-1
    candidate.update(id='reference_roof',points=patch,count=len(patch),accepted=True,
        semantic_label='roof',label_source='previously visually reviewed LAS ROI')
    return {'candidates':[candidate],'clusters':[candidate],'normalization_center':center,'normalization_scale':scale,
        'input_count':len(points),'white_count':int(keep.sum()),'reference_region':region,
        'reference_regions_sha256':file_hash(ROOT/'results/las_rigid/regions.json')}


def run(date,known_roof=False):
    out=OUT/(date+'_reference_roof' if known_roof else date);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    base=ROOT/'results/0929_baseline' if date=='0929' else ROOT/f'results/multidate/{date}'
    preserved=[{'path':str((base/name).relative_to(ROOT)),'sha256':file_hash(base/name)} for name in
        ('alignment.json','T_3dgs_to_las_local.txt','T_3dgs_to_las_global.txt')]
    inputs=[{'path':f'assets/{date}/{date}.{extension}','sha256':file_hash(ROOT/f'assets/{date}/{date}.{extension}')} for extension in ('ply','las')]
    benchmark,benchmark_rgb,held=samples(date);train,colors,ids=dense_training(date,benchmark)
    assert np.all(np.floor(train/.4).sum(1).astype(int)%4!=0)
    np.savez_compressed(out/'training_ids.npz',original_ply_indices=ids)
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    target_rgb=np.c_[las.red,las.green,las.blue].astype(float);target_rgb/=65535 if target_rgb.max()>255 else 255
    detection_start=time.monotonic();source_objects=detect(train,colors)
    target_objects=reference_roof(target,target_rgb) if known_roof else detect(target,target_rgb)
    detection_seconds=time.monotonic()-detection_start
    for side,found in [('source',source_objects),('target',target_objects)]:
        (out/f'{side}_candidates.json').write_text(json.dumps(serializable(found),indent=2)+'\n')
        np.savez_compressed(out/f'{side}_candidate_points.npz',**{r['id']:r['points']/found['normalization_scale']+found['normalization_center'] for r in found['candidates']})
    proposals=guesses(source_objects,target_objects)
    if not proposals:
        for item in preserved:assert file_hash(ROOT/item['path'])==item['sha256']
        failure={'date':date,'reference_roof_annotation_used':known_roof,'status':'rejected_no_object_hypotheses',
            'parameters':PARAMETERS,'original_inputs':inputs,'preserved_registration_files':preserved,
            'source_object_count':len(source_objects['candidates']),'target_object_count':len(target_objects['candidates']),
            'training_tile_leakage_count':0,'original_working_clouds_changed':False}
        (out/'audit.json').write_text(json.dumps(failure,indent=2)+'\n');print(failure,flush=True);return
    print(date,'candidates',len(source_objects['candidates']),len(target_objects['candidates']),'hypotheses',len(proposals),flush=True)
    source_coarse=downsample(train,.06);target_coarse=downsample(target,2.)
    score_source=downsample(train,.12)
    rng=np.random.default_rng(29)
    if len(score_source)>2000:score_source=score_source[rng.choice(len(score_source),2000,replace=False)]
    tree=cKDTree(target_coarse);rank_start=time.monotonic()
    for proposal in proposals:
        proposal['initial_score_p75_m']=float(np.quantile(tree.query(apply(score_source,proposal['matrix']))[0],.75))
    proposals.sort(key=lambda row:row['initial_score_p75_m']);ranking_seconds=time.monotonic()-rank_start
    (out/'object_proposals.json').write_text(json.dumps([{**r,'matrix':r['matrix'].tolist()} for r in proposals],indent=2)+'\n')
    fine_source=downsample(train,.025);fine_target=downsample(target,.6)
    outcomes={};method_runs=[];full_tree=cKDTree(target)
    for method,seeds in [('pca',pca_guesses(source_coarse,target_coarse)),('objects',proposals[:4])]:
        begin=time.monotonic();candidates=[]
        for index,seed in enumerate(seeds):
            t=icp(source_coarse,target_coarse,seed['matrix'],[40.,20.,10.,5.],with_scale=True,iterations=40)
            score=float(np.median(tree.query(apply(source_coarse,t))[0]))
            candidates.append({'seed_index':index,'training_coarse_median_m':score,'matrix':t,
                'seed':{k:v for k,v in seed.items() if k!='matrix'}})
        chosen=min(candidates,key=lambda row:row['training_coarse_median_m'])
        matrix=refine(fine_source,fine_target,chosen['matrix'])
        error,near=full_tree.query(apply(benchmark[~held],matrix))
        color_error=float(np.median(np.linalg.norm(benchmark_rgb[~held]-target_rgb[near],axis=1)))
        outcomes[method]=matrix
        method_runs.append({'method':method,'training_final':summary(error),'training_color_difference':color_error,
            'status':'rejected' if np.median(error)>.5 or color_error>.35 else 'candidate_for_review_not_survey_validated',
            'selected_seed':chosen['seed'],
            'matrix_local':matrix.tolist(),'coarse_candidates':[{**r,'matrix':r['matrix'].tolist()} for r in candidates],
            'refinement_seconds':time.monotonic()-begin})
        print(date,method,'selected',chosen['seed'],'training',summary(error),flush=True)
    # The existing transform is opened only after new matrices are fixed.
    baseline=np.array(json.loads((base/'alignment.json').read_text())['ply_to_las_local'])
    outcomes['existing_baseline']=baseline
    benchmark_world=apply(benchmark,baseline)
    fixed_coverage=cKDTree(target[:,:2]).query(benchmark_world[:,:2])[0]<1.
    rows=[];checks={}
    for method,matrix in outcomes.items():
        positions=apply(benchmark[held],matrix);error=full_tree.query(positions)[0]
        checks[method]=error
        for scope,mask in [('all_withheld',np.ones(held.sum(),bool)),('fixed_baseline_coverage',fixed_coverage[held])]:
            rows.append({'date':date,'method':method,'scope':scope,**summary(error[mask]),
                'within_1m':float(np.mean(error[mask]<1.))})
    np.savez_compressed(out/'held_out_distances.npz',benchmark_sample_indices=np.flatnonzero(held),
        source_points=benchmark[held],fixed_coverage=fixed_coverage[held],**checks)
    write_csv(out/'comparison.csv',rows)
    metadata={'date':date,'reference_roof_annotation_used':known_roof,'status':'development experiment; rule-based surface candidates, not semantic segmentation',
        'parameters':PARAMETERS,'original_inputs':inputs,'preserved_registration_files':preserved,
        'detection_source_records':len(train),'benchmark_records':len(benchmark),'withheld_records':int(held.sum()),
        'training_tile_leakage_count':int(np.sum(np.floor(train/.4).sum(1).astype(int)%4==0)),
        'source_object_count':len(source_objects['candidates']),'target_object_count':len(target_objects['candidates']),
        'object_hypotheses':len(proposals),'ranking_source_count':len(score_source),
        'coarse_source_count':len(source_coarse),'coarse_target_count':len(target_coarse),
        'fine_source_count':len(fine_source),'fine_target_count':len(fine_target),
        'detection_seconds':detection_seconds,'ranking_seconds':ranking_seconds,'runs':method_runs,'comparison':rows,
        'total_seconds':time.monotonic()-started,'baseline_used_to_generate_new_transforms':False,
        'original_working_clouds_changed':False}
    for item in preserved:assert file_hash(ROOT/item['path'])==item['sha256']
    (out/'audit.json').write_text(json.dumps(metadata,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    rr=[r for r in rows if r['scope']=='fixed_baseline_coverage']
    labels={'pca':'PCA + ICP','objects':'Object rules + ICP','existing_baseline':'Existing RTK baseline' if date=='0929' else 'Existing baseline'}
    x=np.arange(len(rr));axes[0].bar(x-.17,[r['median'] for r in rr],width=.34,label='Median',color='#168e96')
    axes[0].bar(x+.17,[r['p90'] for r in rr],width=.34,label='P90',color='#b36b2b')
    axes[0].set(xticks=x,xticklabels=[labels[r['method']] for r in rr],ylabel='Distance to LAS, m',title=date+' · identical withheld records');axes[0].tick_params(axis='x',rotation=15);axes[0].legend()
    for method,e in checks.items():
        values=np.sort(e[fixed_coverage[held]]);axes[1].plot(values,np.arange(1,len(values)+1)/len(values),label=labels[method])
    axes[1].set(xlim=(0,2),ylim=(0,1),xlabel='Distance to LAS, m',ylabel='Fraction of withheld records',title='Fixed baseline coverage; no error clipping');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    selected=next(r for r in method_runs if r['method']=='objects')['selected_seed']
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    source_id=selected['source_id'];target_id=selected['target_id']
    src=next(r for r in source_objects['candidates'] if r['id']==source_id)
    dst=next(r for r in target_objects['candidates'] if r['id']==target_id)
    source_patch=apply(src['points']/source_objects['normalization_scale']+source_objects['normalization_center'],outcomes['objects'])
    target_patch=dst['points']/target_objects['normalization_scale']+target_objects['normalization_center']
    center=target_patch.mean(0);limits=np.maximum(np.ptp(target_patch[:,:2],axis=0)/2+8,12)
    nearby=np.all(abs(target[:,:2]-center[:2])<limits,axis=1)
    p=target[nearby];c=target_rgb[nearby];order=np.argsort(p[:,2])
    for ax,patch,label in [(axes[0],source_patch,'Detected source patch, after new alignment'),(axes[1],target_patch,'Detected LAS patch')]:
        ax.scatter(p[order,0],p[order,1],c=c[order],s=2);ax.scatter(patch[:,0],patch[:,1],s=8,facecolors='none',edgecolors='#e03158',linewidths=.6)
        ax.set(aspect='equal',xlim=(center[0]-limits[0],center[0]+limits[0]),ylim=(center[1]-limits[1],center[1]+limits[1]),title=label,xlabel='Local X, m',ylabel='Local Y, m')
    fig.savefig(out/'selected_object.png',dpi=160);plt.close(fig)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('date',default='0929',nargs='?');parser.add_argument('--reference-roof',action='store_true')
    args=parser.parse_args();run(args.date,args.reference_roof)
