"""Test a vertical-offset hypothesis on a separate candidate static LAS structure."""
import csv
import hashlib
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,ORIGIN,DATES
from check_static_structures import volume_mask

OUT=ROOT/'results/las_vertical'
XY_GATE=.35
BLOCK_SIZE=2.
MIN_PAIRS=3
MIN_BLOCKS=5


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(1024*1024):digest.update(chunk)
    return digest.hexdigest()


def paired_heights(source,target,gate=XY_GATE,block_size=BLOCK_SIZE,min_pairs=MIN_PAIRS):
    """Mutual XY matches and equal-weight block medians, without a Z/error gate."""
    distance,forward=cKDTree(target[:,:2]).query(source[:,:2])
    backward=cKDTree(source[:,:2]).query(target[:,:2])[1]
    ids=np.flatnonzero((distance<=gate)&(backward[forward]==np.arange(len(source))))
    source_pairs,target_pairs=source[ids],target[forward[ids]]
    delta=target_pairs[:,2]-source_pairs[:,2]
    midpoint=(source_pairs[:,:2]+target_pairs[:,:2])/2
    block=np.floor(midpoint/block_size).astype(int)
    unique,inverse=np.unique(block,axis=0,return_inverse=True)
    rows=[]
    for i,key in enumerate(unique):
        selected=inverse==i
        if selected.sum()>=min_pairs:
            rows.append({'ix':int(key[0]),'iy':int(key[1]),'pairs':int(selected.sum()),
                'delta_z_m':float(np.median(delta[selected])),
                'delta_z_iqr_m':float(np.quantile(delta[selected],.75)-np.quantile(delta[selected],.25))})
    values=np.array([r['delta_z_m'] for r in rows])
    fit={'pairs':len(ids),'blocks':len(rows),'source_fraction_matched':len(ids)/len(source),
        'target_fraction_matched':len(ids)/len(target),'delta_z_m':float(np.median(values)) if len(rows)>=MIN_BLOCKS else None,
        'q25_m':float(np.quantile(values,.25)) if len(rows) else None,
        'q75_m':float(np.quantile(values,.75)) if len(rows) else None}
    return fit,rows,{'source':source_pairs,'target':target_pairs,'delta_z_m':delta,'block_xy':block}


def residual_metrics(values,shift=0.):
    residual=np.abs(np.asarray(values)-shift)
    return {'median_m':float(np.median(residual)),'p90_m':float(np.quantile(residual,.9))}


def voxel_representatives(points,size=.25):
    _,indices=np.unique(np.floor(points/size).astype(int),axis=0,return_index=True)
    return points[indices]


def symmetric_distance(source,target,shift=0.):
    # Both samples and masks stay frozen when trying a correction.
    moved=source+np.array([0,0,shift])
    a=cKDTree(target).query(moved)[0];b=cKDTree(moved).query(target)[0]
    return {'mean_directional_medians_m':float((np.median(a)+np.median(b))/2),
        'source_p90_m':float(np.quantile(a,.9)),'target_p90_m':float(np.quantile(b,.9))}


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    region_path=ROOT/'results/static_structures/regions.json'
    regions=json.loads(region_path.read_text())['regions'];keys=[r['id'] for r in regions]
    assert keys==['roof','bank_mid']
    clouds={};headers=[];preserved=[]
    for date in DATES:
        path=ROOT/f'assets/{date}/{date}.las';las=laspy.read(path);h=las.header
        p=np.c_[las.x,las.y,las.z]-ORIGIN
        ids=np.linspace(0,len(p)-1,1000,dtype=int)
        encoded=np.c_[las.X[ids],las.Y[ids],las.Z[ids]]*h.scales+h.offsets-ORIGIN
        np.testing.assert_allclose(encoded,p[ids],atol=1e-9,rtol=0)
        clouds[date]={r['id']:p[volume_mask(p,r)] for r in regions}
        geokeys=[];vlrs=[]
        for vlr in h.vlrs:
            vlrs.append({'user_id':vlr.user_id,'record_id':vlr.record_id,'description':vlr.description,
                'sha256':hashlib.sha256(vlr.record_data_bytes()).hexdigest()})
            if hasattr(vlr,'geo_keys'):
                geokeys=[{'id':k.id,'location':k.tiff_tag_location,'count':k.count,'value_offset':k.value_offset} for k in vlr.geo_keys]
        base=ROOT/'results/0929_baseline' if date=='0929' else ROOT/f'results/multidate/{date}'
        old=json.loads((base/'input_manifest.json').read_text())
        if isinstance(old,dict):old=old.get('files',old.get('inputs',[]))
        old_las=next(item for item in old if item['path'].endswith(f'{date}.las'))
        digest=file_hash(path);assert digest==old_las['sha256'],f'{date} source LAS changed'
        headers.append({'date':date,'version':str(h.version),'point_format':h.point_format.id,'points':len(p),
            'generating_software':h.generating_software,'creation_date':str(h.creation_date),
            'scales':h.scales.tolist(),'offsets':h.offsets.tolist(),'vlrs':vlrs,'geokeys':geokeys,
            'path':str(path.relative_to(ROOT)),'sha256':digest,'integer_decode_checked':True})
        for name in ('alignment.json','T_3dgs_to_las_local.txt','T_3dgs_to_las_global.txt'):
            item=base/name;preserved.append({'path':str(item.relative_to(ROOT)),'sha256':file_hash(item)})
    pair_rows=[];block_rows=[];fits={};baseline_blocks={}
    matrices={key:np.zeros((7,7)) for key in keys}
    for i,date in enumerate(DATES):
        for j in range(i+1,7):
            reference=DATES[j]
            for key in keys:
                fit,blocks,pairs=paired_heights(clouds[date][key],clouds[reference][key])
                assert fit['delta_z_m'] is not None,(date,reference,key,'insufficient XY support')
                pair_rows.append({'source_date':date,'reference_date':reference,'region':key,**fit})
                block_rows.extend({'source_date':date,'reference_date':reference,'region':key,**r} for r in blocks)
                matrices[key][i,j]=fit['delta_z_m'];matrices[key][j,i]=-fit['delta_z_m']
                if reference=='0929':
                    fits[(date,key)]=fit;baseline_blocks[(date,key)]=blocks
                    np.savez_compressed(OUT/f'{date}_to_0929_{key}_pairs.npz',**pairs)
    rows=[]
    for date in DATES[:-1]:
        for fit_key,check_key in [('roof','bank_mid'),('bank_mid','roof')]:
            fit=fits[(date,fit_key)];shift=fit['delta_z_m'];held=baseline_blocks[(date,check_key)]
            values=[r['delta_z_m'] for r in held]
            before=residual_metrics(values);after=residual_metrics(values,shift)
            source=voxel_representatives(clouds[date][check_key]);target=voxel_representatives(clouds['0929'][check_key])
            dist_before=symmetric_distance(source,target);dist_after=symmetric_distance(source,target,shift)
            row={'date':date,'fit_region':fit_key,'check_region':check_key,'trial_delta_z_m':shift,
                'check_blocks':len(held),'check_pairs':fits[(date,check_key)]['pairs'],
                'check_block_median_before_m':before['median_m'],'check_block_median_after_m':after['median_m'],
                'check_block_p90_before_m':before['p90_m'],'check_block_p90_after_m':after['p90_m'],
                'check_symmetric_3d_before_m':dist_before['mean_directional_medians_m'],
                'check_symmetric_3d_after_m':dist_after['mean_directional_medians_m'],
                'check_source_3d_p90_before_m':dist_before['source_p90_m'],'check_source_3d_p90_after_m':dist_after['source_p90_m'],
                'check_target_3d_p90_before_m':dist_before['target_p90_m'],'check_target_3d_p90_after_m':dist_after['target_p90_m']}
            rows.append(row)
    verdict=[]
    for date in DATES[:-1]:
        roof=fits[(date,'roof')];bank=fits[(date,'bank_mid')]
        a=next(r for r in rows if r['date']==date and r['fit_region']=='roof')
        b=next(r for r in rows if r['date']==date and r['fit_region']=='bank_mid')
        verdict.append({'date':date,'roof_delta_z_m':roof['delta_z_m'],'bank_delta_z_m':bank['delta_z_m'],
            'difference_between_estimates_m':roof['delta_z_m']-bank['delta_z_m'],
            'roof_shift_improves_bank_blocks':a['check_block_median_after_m']<a['check_block_median_before_m'],
            'roof_shift_improves_bank_3d':a['check_symmetric_3d_after_m']<a['check_symmetric_3d_before_m'],
            'bank_shift_improves_roof_blocks':b['check_block_median_after_m']<b['check_block_median_before_m'],
            'bank_shift_improves_roof_3d':b['check_symmetric_3d_after_m']<b['check_symmetric_3d_before_m']})
    sensitivity=[]
    for gate in (.2,.35,.5):
        for size in (1.,2.,4.):
            for date in DATES[:-1]:
                trials={key:paired_heights(clouds[date][key],clouds['0929'][key],gate,size)[:2] for key in keys}
                for fit_key,check_key in [('roof','bank_mid'),('bank_mid','roof')]:
                    fit=trials[fit_key][0];check,blocks=trials[check_key]
                    assert fit['delta_z_m'] is not None and check['delta_z_m'] is not None
                    values=[r['delta_z_m'] for r in blocks]
                    before=residual_metrics(values);after=residual_metrics(values,fit['delta_z_m'])
                    sensitivity.append({'date':date,'xy_gate_m':gate,'block_size_m':size,
                        'fit_region':fit_key,'check_region':check_key,'fit_blocks':fit['blocks'],'check_blocks':check['blocks'],
                        'trial_delta_z_m':fit['delta_z_m'],'check_own_delta_z_m':check['delta_z_m'],
                        'check_block_median_before_m':before['median_m'],'check_block_median_after_m':after['median_m'],
                        'change_in_check_median_m':after['median_m']-before['median_m']})
    write_csv(OUT/'pairwise_offsets.csv',pair_rows);write_csv(OUT/'matched_blocks.csv',block_rows)
    write_csv(OUT/'cross_validation.csv',rows);write_csv(OUT/'summary.csv',verdict)
    write_csv(OUT/'sensitivity.csv',sensitivity)
    metadata={'reference':'0929 is a comparison baseline, not surveyed truth','xy_gate_m':XY_GATE,'block_size_m':BLOCK_SIZE,
        'minimum_pairs_per_block':MIN_PAIRS,'minimum_blocks':MIN_BLOCKS,'z_error_gate':None,
        'trial_changes_written_to_clouds':False,'regions_sha256':file_hash(region_path),'preserved_registration_files':preserved,
        'headers':headers,'summary':verdict,'cross_validation':rows,'pairwise':pair_rows,'sensitivity':sensitivity}
    for item in preserved:assert file_hash(ROOT/item['path'])==item['sha256']
    (OUT/'audit.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for key,label,color in [('roof','Roof','#168e96'),('bank_mid','Riverside structure','#b36b2b')]:
        ff=[fits[(d,key)] for d in DATES[:-1]];v=np.array([r['delta_z_m'] for r in ff]);q=np.array([[r['q25_m'],r['q75_m']] for r in ff])
        axes[0].errorbar(np.arange(6)+(0 if key=='roof' else .12),v,yerr=[v-q[:,0],q[:,1]-v],fmt='o',capsize=3,color=color,label=label)
    axes[0].set(xticks=np.arange(6),xticklabels=DATES[:-1],ylabel='Additive Z shift toward 0929, m',title='Estimate on each structure\nBars: IQR of 2 m block offsets');axes[0].legend()
    for ax,fit_key,title in [(axes[1],'roof','Roof-derived shift checked on riverside'),(axes[2],'bank_mid','Riverside-derived shift checked on roof')]:
        rr=[r for r in rows if r['fit_region']==fit_key];x=np.arange(6)
        ax.plot(x,[r['check_block_median_before_m'] for r in rr],'o-',label='Before',color='#697885')
        ax.plot(x,[r['check_block_median_after_m'] for r in rr],'s-',label='Trial shift',color='#b45242')
        ax.set(xticks=x,xticklabels=DATES[:-1],ylabel='Median absolute block height residual, m',title=title);ax.legend()
    for ax in axes:ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2)
    fig.savefig(OUT/'cross_validation.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.8),constrained_layout=True)
    plots=[matrices['roof'],matrices['bank_mid'],matrices['roof']-matrices['bank_mid']]
    limit=max(abs(m).max() for m in plots)
    for ax,m,title in zip(axes,plots,['Roof Z offsets','Riverside Z offsets','Difference: roof minus riverside']):
        im=ax.imshow(m,cmap='RdBu_r',vmin=-limit,vmax=limit)
        for i in range(7):
            for j in range(7):ax.text(j,i,f'{m[i,j]:+.2f}',ha='center',va='center',fontsize=8,color='white' if abs(m[i,j])>.65*limit else 'black')
        ax.set(xticks=range(7),yticks=range(7),xticklabels=DATES,yticklabels=DATES,xlabel='Reference date',ylabel='Source date',title=title);ax.tick_params(axis='x',rotation=45)
    fig.colorbar(im,ax=axes,label='Additive Z shift, m',shrink=.75);fig.savefig(OUT/'pairwise_offsets.png',dpi=170);plt.close(fig)
    print(json.dumps(verdict,indent=2),flush=True)


if __name__=='__main__':main()
