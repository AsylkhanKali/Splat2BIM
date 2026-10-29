"""Check unchanged registrations in manually reviewed fixed-structure volumes."""
import csv
import hashlib
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData,PlyElement
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES
from align_0929 import apply,summary
from check_common_areas import COLORS,LIMITS

OUT=ROOT/'results/static_structures'
MIN_COUNT=30


def volume_mask(points,region):
    """Convex XY polygon and fixed height range; no distance-dependent filtering."""
    polygon=np.array(region['polygon'],float)
    edges=np.roll(polygon,-1,axis=0)-polygon
    delta=points[:,None,:2]-polygon[None,:,:]
    cross=edges[None,:,0]*delta[:,:,1]-edges[None,:,1]*delta[:,:,0]
    inside=np.all(cross>=-1e-9,axis=1)|np.all(cross<=1e-9,axis=1)
    return inside&(points[:,2]>=region['z_range'][0])&(points[:,2]<=region['z_range'][1])


def measurements(points,target):
    errors=cKDTree(target).query(points,workers=3)[0]
    return errors,{'count':len(points),'median_m':float(np.median(errors)) if len(points) else None,
        'p90_m':float(np.quantile(errors,.9)) if len(points) else None,
        'max_m':float(errors.max()) if len(points) else None,
        'within_1m':float(np.mean(errors<1)) if len(points) else None}


def roof_plane(points,anchor):
    xy=points[:,:2]-anchor
    design=np.c_[xy,np.ones(len(points))]
    beta=np.linalg.lstsq(design,points[:,2],rcond=None)[0]
    for _ in range(12):
        residual=points[:,2]-design@beta
        weights=np.sqrt(np.minimum(1.,.15/np.maximum(abs(residual),1e-9)))
        beta=np.linalg.lstsq(design*weights[:,None],points[:,2]*weights,rcond=None)[0]
    return beta


def mean_region_score(rows,key='median_m'):
    if any(r['count']<MIN_COUNT for r in rows):return None
    return float(np.mean([r[key] for r in rows]))


def write_ply(path,points,colors):
    vertices=np.empty(len(points),dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate(('x','y','z')):vertices[k]=points[:,i]
    for i,k in enumerate(('red','green','blue')):vertices[k]=colors[:,i]
    PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(path)
    saved=PlyData.read(path,mmap='r')['vertex'].data
    np.testing.assert_allclose(np.column_stack([saved[k] for k in ('x','y','z')]),points,atol=1e-5,rtol=0)
    np.testing.assert_array_equal(np.column_stack([saved[k] for k in ('red','green','blue')]),colors)


def main():
    config=OUT/'regions.json';definition=json.loads(config.read_text());regions=definition['regions']
    np.testing.assert_array_equal(definition['origin_global'],ORIGIN)
    references={};context={};inputs=[];classes={}
    for date in DATES:
        las=laspy.read(ROOT/f'assets/{date}/{date}.las')
        classes[date]=np.unique(las.classification).astype(int).tolist()
        p=np.c_[las.x,las.y,las.z]-ORIGIN
        c=np.c_[las.red,las.green,las.blue].astype(float);c/=65535 if c.max()>255 else 255
        references[date]={};context[date]={}
        for region in regions:
            key=region['id'];mask=volume_mask(p,region)
            references[date][key]=p[mask]
            assert mask.sum()>=MIN_COUNT,(date,key,'LAS support missing')
            np.savez_compressed(OUT/f'{date}_{key}_las.npz',points=p[mask],rgb=c[mask])
            polygon=np.array(region['polygon']);lo=polygon.min(0)-3;hi=polygon.max(0)+3
            nearby=np.all((p[:,:2]>=lo)&(p[:,:2]<=hi),axis=1)
            context[date][key]=(p[nearby],c[nearby])
        base=ROOT/'results/0929_baseline' if date=='0929' else ROOT/f'results/multidate/{date}'
        path=base/'alignment.json';info=json.loads(path.read_text())
        assert info.get('status')!='rejected'
        np.testing.assert_array_equal(info['origin_global'],ORIGIN)
        inputs.append({'date':date,'alignment':str(path.relative_to(ROOT)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'matrix':info['ply_to_las_local']})
    palette=np.array([[int(c[j:j+2],16) for j in (1,3,5)] for c in COLORS],dtype='u1')
    rows=[];by_date={};laser_rows=[];all_checks=[]
    roof_anchor=np.mean(regions[0]['polygon'],axis=0)
    baseline_beta=roof_plane(references['0929']['roof'],roof_anchor)
    for date,source in zip(DATES,inputs):
        vertices=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')['vertex'].data
        ids=np.arange(5,len(vertices),10);v=vertices[5::10]
        xyz=np.column_stack([v[k] for k in ('x','y','z')]).astype(float)
        valid=(v['opacity']>0)&np.isfinite(xyz).all(1)
        xyz,ids=xyz[valid],ids[valid];points=apply(xyz,np.array(source['matrix']))
        spatial=(np.floor(xyz/.4).sum(axis=1).astype(int)%4)==0
        records=[];shown=[];distances=[];saved_ids=[];region_ids=[];roof_stats={}
        for region in regions:
            key=region['id'];keep=volume_mask(points,region);p=points[keep];reference=references[date][key]
            errors,metric=measurements(p,reference)
            _,to_baseline=measurements(p,references['0929'][key])
            z=np.quantile(p[:,2],[.05,.5,.95]) if len(p) else [region['z_range'][0],np.mean(region['z_range']),region['z_range'][1]]
            row={'date':date,'region':key,**metric,'las_count':len(reference),
                'old_spatial_holdout_count':int(spatial[keep].sum()),
                'to_0929_median_m':to_baseline['median_m'],'to_0929_p90_m':to_baseline['p90_m'],
                'z05':float(z[0]),'z_median':float(z[1]),'z95':float(z[2])}
            rows.append(row);records.append(row)
            shown.append(p);distances.append(errors);saved_ids.append(ids[keep]);region_ids.extend([key]*len(p))
            # Equal reference-density check, independent of 3DGS residuals.
            _,indices=np.unique(np.floor(reference/.25).astype(int),axis=0,return_index=True)
            _,laser_metric=measurements(reference[indices],references['0929'][key])
            laser_rows.append({'date':date,'region':key,**laser_metric})
            if key=='roof':
                beta=roof_plane(reference,roof_anchor);norm=np.sqrt(1+np.sum(beta[:2]**2))
                residual=(p[:,2]-np.c_[p[:,:2]-roof_anchor,np.ones(len(p))]@beta)/norm
                plane_error=(reference[:,2]-np.c_[reference[:,:2]-roof_anchor,np.ones(len(reference))]@beta)/norm
                roof_stats={'las_plane_z_at_anchor_m':float(beta[2]),'las_plane_z_delta_vs_0929_m':float(beta[2]-baseline_beta[2]),
                    'las_plane_abs_p90_m':float(np.quantile(abs(plane_error),.9)),
                    'las_fraction_within_015m_of_plane':float(np.mean(abs(plane_error)<.15)),
                    'gs_signed_plane_median_m':float(np.median(residual)) if len(p) else None,
                    'gs_abs_plane_p90_m':float(np.quantile(abs(residual),.9)) if len(p) else None,
                    'normal':(-np.r_[beta[:2],-1]/norm).tolist()}
        p=np.concatenate(shown);e=np.concatenate(distances);source_ids=np.concatenate(saved_ids)
        assert np.all(source_ids%10==5) and len(np.unique(source_ids))==len(source_ids)
        assert all(r['count']>=MIN_COUNT for r in records),(date,'Insufficient control records; do not publish aggregate')
        colors=palette[np.searchsorted(LIMITS,e,side='right')]
        write_ply(OUT/f'{date}_distances.ply',p,colors)
        np.savez_compressed(OUT/f'{date}_samples.npz',points=p,distance_m=e,source_indices=source_ids,region=np.array(region_ids))
        record={'date':date,'samples':len(p),'regions':len(regions),'mean_of_region_medians_m':mean_region_score(records),
            'mean_of_region_p90_m':mean_region_score(records,'p90_m'),'point_weighted_median_m':float(np.median(e)),
            'point_weighted_p90_m':float(np.quantile(e,.9)),'mean_region_fraction_within_1m':float(np.mean([r['within_1m'] for r in records])),
            'mean_region_to_0929_median_m':float(np.mean([r['to_0929_median_m'] for r in records])),
            'old_spatial_holdout_count':sum(r['old_spatial_holdout_count'] for r in records),
            'eligible_record_count':len(points),'outside_positive_masks':len(points)-len(p),
            'points_url':f'/results/static_structures/{date}_distances.ply','roof_plane':roof_stats}
        by_date[date]=record
        all_checks.append({'date':date,'count':len(p),'record_offset':5,'export_reread':True})
        print(date,json.dumps(record),flush=True)
    baseline=by_date['0929']['mean_of_region_medians_m']
    for row in by_date.values():row['delta_mean_median_vs_0929_m']=row['mean_of_region_medians_m']-baseline
    manifest={'kind':'static','origin_global':ORIGIN.tolist(),'region_count':len(regions),
        'sample_protocol':'Every 10th original Gaussian starting at index 5; opacity logit > 0; finite XYZ; no quantile or error clipping. These records were not fitted, but are not an independent spatial holdout.',
        'color_limits_m':LIMITS,'colors':COLORS,'dates':by_date,'regions':[]}
    for region in regions:
        polygon=np.array(region['polygon'])
        manifest['regions'].append({**region,'bounds_xy':np.r_[polygon.min(0),polygon.max(0)].tolist(),
            'dates':{d:next(r for r in rows if r['date']==d and r['region']==region['id']) for d in DATES}})
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    for filename,data in [('regions_comparison.csv',rows),('las_to_0929.csv',laser_rows),
            ('comparison.csv',[{k:v for k,v in row.items() if k not in ('roof_plane','points_url')} for row in by_date.values()]),
            ('roof_plane.csv',[{'date':d,**{k:v for k,v in row['roof_plane'].items() if k!='normal'}} for d,row in by_date.items()])]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    for source in inputs:
        assert hashlib.sha256((ROOT/source['alignment']).read_bytes()).hexdigest()==source['sha256']
    (OUT/'validation.json').write_text(json.dumps({'alignment_inputs':inputs,'regions_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
        'checks':all_checks,'las_classification_values':classes,'distance_outliers_removed':0,'minimum_records_per_region':MIN_COUNT},indent=2)+'\n')
    for region in regions:
        fig,axes=plt.subplots(2,4,figsize=(14,9),constrained_layout=True)
        for date,ax in zip(DATES,axes.flat):
            p,c=context[date][region['id']];order=np.argsort(p[:,2]);ax.scatter(p[order,0],p[order,1],c=c[order],s=3)
            polygon=np.array(region['polygon']);closed=np.r_[polygon,polygon[:1]];ax.plot(closed[:,0],closed[:,1],color='#dc2626',lw=1)
            ax.set(aspect='equal',title=date,xlabel='X, m',ylabel='Y, m');ax.grid(alpha=.25)
        axes.flat[-1].axis('off');axes.flat[-1].text(.05,.9,region['label']+'\nRed: fixed XY mask\nZ band: '+str(region['z_range'])+' m\nSelected without 3DGS residuals',va='top')
        fig.savefig(OUT/f'{region["id"]}_selection_review.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for region,ax in zip(regions,axes):
        rr=[r for r in rows if r['region']==region['id']];x=np.arange(7)
        ax.bar(x-.17,[r['median_m'] for r in rr],width=.34,label='Median',color='#188f91')
        ax.bar(x+.17,[r['p90_m'] for r in rr],width=.34,label='P90',color='#dd873c')
        ax.set(xticks=x,xticklabels=DATES,ylabel='Distance to own masked LAS, m',title=region['label']);ax.tick_params(axis='x',rotation=45);ax.grid(axis='y',alpha=.25);ax.legend()
    axes[2].plot(DATES,[by_date[d]['roof_plane']['las_plane_z_delta_vs_0929_m'] for d in DATES],'o-',color='#734b97');axes[2].axhline(0,color='gray',lw=.8)
    support=by_date['0929']['roof_plane']['las_fraction_within_015m_of_plane']
    axes[2].set(title=f'Dominant LAS roof plane vs 0929\n0929: {support:.0%} within 0.15 m of plane',ylabel='Plane height difference at fixed XY, m');axes[2].tick_params(axis='x',rotation=45);axes[2].grid(alpha=.25)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
