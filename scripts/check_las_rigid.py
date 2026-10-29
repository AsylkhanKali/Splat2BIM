"""Trial rigid LAS registration on two fixed regions; check a third, untouched region."""
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from align_dates import ROOT,ORIGIN,DATES
from align_0929 import apply
from check_static_structures import volume_mask
from check_las_vertical import file_hash,voxel_representatives,write_csv

OUT=ROOT/'results/las_rigid'
VOXEL=.25
HUBER=.15
LEVER=100.


def target_surface(points):
    """Local PCA normals: at least 8 of 24 neighbors within 1.25 m."""
    tree=cKDTree(points);dist,idx=tree.query(points,k=24,distance_upper_bound=1.25)
    normals=np.zeros_like(points);valid=np.zeros(len(points),bool)
    for i in range(len(points)):
        local=points[idx[i,np.isfinite(dist[i])]]
        if len(local)<8:continue
        local=local-local.mean(0);values,vectors=np.linalg.eigh(local.T@local/len(local))
        if values[0]/max(values.sum(),1e-12)<.08 and values[1]/max(values[2],1e-12)>.15:
            normals[i]=vectors[:,0];valid[i]=True
    return {'points':points,'normals':normals,'normal_valid':valid,'tree':tree,
        'input_count':len(points),'normal_count':int(valid.sum())}


def linear_system(source,targets,matrix,anchor,gate):
    jacobians=[];residuals=[];base_weights=[];counts={}
    for key,cloud in source.items():
        p=apply(cloud,matrix);target=targets[key]
        dist,idx=target['tree'].query(p);keep=(dist<gate)&target['normal_valid'][idx]
        counts[key]=int(keep.sum())
        if keep.sum()<30:raise ValueError(f'Insufficient fit support: {key}: {keep.sum()}')
        p=p[keep];q=target['points'][idx[keep]];n=target['normals'][idx[keep]]
        residual=np.sum((p-q)*n,axis=1)
        jacobians.append(np.c_[np.cross(p-anchor,n)/LEVER,n]);residuals.append(residual)
        # Equal contribution per region before Huber downweighting.
        base_weights.append(np.full(len(p),1/(len(source)*len(cloud))))
    return np.concatenate(jacobians),np.concatenate(residuals),np.concatenate(base_weights),counts


def training_loss(source,targets,matrix,gate):
    losses=[]
    for key,p in source.items():
        target=targets[key];moved=apply(p,matrix);distance,_=target['tree'].query(moved)
        # Capped Euclidean Huber loss; every original fit point contributes.
        r=np.minimum(distance,gate)
        losses.append(np.mean(np.where(r<=HUBER,.5*r*r,HUBER*(r-.5*HUBER))))
    return float(np.mean(losses))


def rigid_fit(source,targets,initial,anchor):
    """Equal-region robust point-to-point ICP, with scale fixed exactly to one."""
    matrix=initial.copy();iterations=[]
    for gate in (2.,1.,.5):
        status='iteration_limit';loss=training_loss(source,targets,matrix,gate);history=[loss]
        for iteration in range(100):
            sources=[];neighbors=[];weights=[]
            for key,cloud in source.items():
                p=apply(cloud,matrix);target=targets[key];distance,idx=target['tree'].query(p);keep=distance<gate
                if keep.sum()<30:raise ValueError(f'Insufficient fit support: {key}: {keep.sum()}')
                sources.append(p[keep]);neighbors.append(target['points'][idx[keep]])
                weights.append(np.minimum(1.,HUBER/np.maximum(distance[keep],1e-12))/(len(source)*len(cloud)))
            p=np.concatenate(sources);q=np.concatenate(neighbors);w=np.concatenate(weights);w/=w.sum()
            center_p=np.sum(p*w[:,None],axis=0);center_q=np.sum(q*w[:,None],axis=0)
            u,_,vt=np.linalg.svd((p-center_p).T@((q-center_q)*w[:,None]))
            rotation=vt.T@np.diag([1,1,np.linalg.det(vt.T@u.T)])@u.T
            delta=np.eye(4);delta[:3,:3]=rotation;delta[:3,3]=center_q-rotation@center_p
            trial=delta@matrix;trial_loss=training_loss(source,targets,trial,gate)
            if trial_loss>loss+1e-12:status='no_descent';break
            matrix=trial;history.append(trial_loss)
            if abs(loss-trial_loss)<1e-9:status='stable_loss';break
            loss=trial_loss
        iterations.append({'gate_m':gate,'iterations':iteration+1,'status':status,
            'converged':status=='stable_loss','accepted_losses':history})
    jac,_,base,counts=linear_system(source,targets,matrix,anchor,.5)
    singular=np.linalg.svd(jac*np.sqrt(base)[:,None],compute_uv=False)
    return matrix,{'training_loss':training_loss(source,targets,matrix,.5),'fit_counts':counts,'iterations':iterations,
        'singular_values_scaled':singular.tolist(),'scaled_condition':float(singular[0]/max(singular[-1],1e-15)),
        'rank_at_relative_1e_3':int(np.sum(singular>singular[0]*1e-3))}


def evaluate(source,target,matrix):
    moved=apply(source,matrix);a=cKDTree(target).query(moved)[0];b=cKDTree(moved).query(target)[0]
    return {'source_count':len(source),'target_count':len(target),
        'symmetric_median_m':float((np.median(a)+np.median(b))/2),
        'source_p90_m':float(np.quantile(a,.9)),'target_p90_m':float(np.quantile(b,.9)),
        'source_max_m':float(a.max()),'target_max_m':float(b.max())}


def main():
    definition=json.loads((OUT/'regions.json').read_text());regions=definition['regions']
    np.testing.assert_array_equal(definition['origin_global'],ORIGIN)
    fit_keys=[r['id'] for r in regions if r['role']=='fit'];check_keys=[r['id'] for r in regions if r['role']=='check']
    assert fit_keys==['roof_shell','bank_mid'] and check_keys==['bank_south']
    old=json.loads((ROOT/'results/las_vertical/audit.json').read_text())
    preserved=old['preserved_registration_files'];inputs=[];clouds={};support=[]
    figures={r['id']:plt.subplots(2,4,figsize=(16,9),constrained_layout=True) for r in regions}
    for i,date in enumerate(DATES):
        path=ROOT/f'assets/{date}/{date}.las';digest=file_hash(path)
        assert digest==next(r['sha256'] for r in old['headers'] if r['date']==date)
        inputs.append({'date':date,'path':str(path.relative_to(ROOT)),'sha256':digest})
        las=laspy.read(path);p=np.c_[las.x,las.y,las.z]-ORIGIN
        rgb=np.c_[las.red,las.green,las.blue].astype(float);rgb/=65535 if rgb.max()>255 else 255
        clouds[date]={}
        for region in regions:
            key=region['id'];keep=volume_mask(p,region);points=p[keep];colors=rgb[keep]
            clouds[date][key]=voxel_representatives(points,VOXEL)
            support.append({'date':date,'region':key,'raw_count':int(keep.sum()),'voxel_count':len(clouds[date][key])})
            assert len(clouds[date][key])>=100,(date,key)
            np.savez_compressed(OUT/f'{date}_{key}_las.npz',points=clouds[date][key])
            ax=figures[key][1].flat[i];order=np.argsort(points[:,2]);ax.scatter(points[order,0],points[order,1],c=colors[order],s=3)
            polygon=np.array(region['polygon']);polygon=np.r_[polygon,polygon[:1]];ax.plot(polygon[:,0],polygon[:,1],c='red',lw=.8)
            ax.set(aspect='equal',title=f'{date} · {len(points)} points',xlabel='X, m',ylabel='Y, m');ax.grid(alpha=.2)
    for region in regions:
        fig,axes=figures[region['id']];axes.flat[-1].axis('off')
        axes.flat[-1].text(.05,.9,region['label']+'\nOnly selected XYZ points\nZ range: '+str(region['z_range'])+'\nRole: '+region['role'],va='top')
        fig.savefig(OUT/f'{region["id"]}_selection.png',dpi=140);plt.close(fig)
    targets={key:target_surface(clouds['0929'][key]) for key in fit_keys}
    starts=[('center',0,0,0),('x_minus',-.5,0,0),('x_plus',.5,0,0),('y_minus',0,-.5,0),
        ('y_plus',0,.5,0),('yaw_minus',0,0,-.3),('yaw_plus',0,0,.3)]
    all_runs=[];comparison=[];chosen=[]
    for date in DATES[:-1]:
        source={key:clouds[date][key] for key in fit_keys}
        anchor=np.mean([p.mean(0) for p in source.values()],axis=0)
        z=next(r['roof_delta_z_m'] for r in old['summary'] if r['date']==date)
        runs=[]
        for name,dx,dy,yaw in starts:
            initial=np.eye(4);initial[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()
            initial[:3,3]=anchor-initial[:3,:3]@anchor+[dx,dy,z]
            try:
                matrix,fit=rigid_fit(source,targets,initial,anchor)
                np.testing.assert_allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-10)
                assert abs(np.linalg.det(matrix[:3,:3])-1)<1e-10
                runs.append({'date':date,'start':name,'matrix_local':matrix.tolist(),**fit})
            except ValueError as error:
                all_runs.append({'date':date,'start':name,'failed':str(error)})
        assert runs,(date,'No fit succeeded')
        best=min(runs,key=lambda r:r['training_loss']);matrix=np.array(best['matrix_local'])
        # The check cloud is first consulted only after selecting by training loss.
        control=clouds[date]['bank_south'];center=control.mean(0)
        displacements=np.array([apply(center[None],np.array(r['matrix_local']))[0]-center for r in runs])
        for run,displacement in zip(runs,displacements):
            run['check_centroid_displacement_m']=displacement.tolist()
            run['check_metrics']=evaluate(control,clouds['0929']['bank_south'],np.array(run['matrix_local']))
        all_runs.extend(runs)
        chosen.append({'date':date,'chosen_start':best['start'],'training_loss':best['training_loss'],
            'matrix_local':best['matrix_local'],'fit_anchor_local':anchor.tolist(),
            'displacement_at_fit_anchor_m':(apply(anchor[None],matrix)[0]-anchor).tolist(),
            'rotation_vector_degrees':np.rad2deg(Rotation.from_matrix(matrix[:3,:3]).as_rotvec()).tolist(),
            'check_displacement_range_m':np.ptp(displacements,axis=0).tolist(),
            'scaled_condition':best['scaled_condition'],'rank_at_relative_1e_3':best['rank_at_relative_1e_3'],
            'chosen_final_status':best['iterations'][-1]['status'],
            'completed_starts':len(runs),'all_final_stages_converged':all(r['iterations'][-1]['converged'] for r in runs)})
        vertical=np.eye(4);vertical[2,3]=z
        for region in regions:
            key=region['id']
            for method,trial in [('original',np.eye(4)),('roof_z_only',vertical),('rigid_trial',matrix)]:
                comparison.append({'date':date,'region':key,'role':region['role'],'method':method,
                    **evaluate(clouds[date][key],clouds['0929'][key],trial)})
        print(date,json.dumps(chosen[-1]),flush=True)
    for item in preserved:assert file_hash(ROOT/item['path'])==item['sha256']
    audit={'reference_date':'0929','reference_is_survey_truth':False,'definition':definition,
        'regions_sha256':file_hash(OUT/'regions.json'),'source_inputs':inputs,'preserved_registration_files':preserved,
        'method':'equal-region robust point-to-point rigid ICP; normals only for observability diagnostic',
        'voxel_m':VOXEL,'normal_neighbors':24,'normal_radius_m':1.25,'huber_m':HUBER,'rotation_lever_m':LEVER,
        'gates_m':[2,1,.5],'selection_uses_check_region':False,'exported_corrections_applied':False,
        'target_normals':{k:{f:t[f] for f in ['input_count','normal_count']} for k,t in targets.items()},
        'chosen':chosen,'runs':all_runs,'support':support,'comparison':comparison}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    write_csv(OUT/'comparison.csv',comparison);write_csv(OUT/'support.csv',support)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax,key,title in zip(axes,['roof_shell','bank_mid','bank_south'],['Fit: roof and upper faces','Fit: middle riverside','Held-out: south riverside']):
        for method,label,color in [('original','Original','#697885'),('roof_z_only','Roof Z only','#b36b2b'),('rigid_trial','Rigid trial','#168e96')]:
            rr=[r for r in comparison if r['region']==key and r['method']==method]
            ax.plot([r['date'] for r in rr],[r['symmetric_median_m'] for r in rr],'o-',label=label,color=color)
        ax.set(title=title,ylabel='Mean of directional distance medians, m');ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
