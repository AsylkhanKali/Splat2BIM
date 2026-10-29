"""Compare all dates in a fixed frame and against one unchanged BIM revision."""
import csv
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree

from align_dates import ROOT,OUT,ORIGIN,DATES

BIM=ROOT/'results/0929_bim'
GROUPS={'Deck':'Decks','Canopy':'Canopies','Pier':'Piers','Pile Cap':'Pile caps','Kiosk':'Kiosks','Underground piles':'Underground piles'}


def bim_samples():
    elements=json.loads((BIM/'v3_elements.json').read_text())['elements']
    ply=PlyData.read(BIM/'v3_local.ply',mmap='r',known_list_len={'face':{'vertex_indices':3}})
    vertices=np.column_stack([ply['vertex'][k] for k in ('x','y','z')]).astype(float)
    faces=ply['face']['vertex_indices'];start=0;selected=[];samples=[]
    for e in elements:
        triangles=faces[start:start+e['triangles']];start+=e['triangles']
        group=next((g for g in GROUPS if g.lower() in e['name'].lower()),None)
        if e['type']=='IfcPile':group='Underground piles'
        if group is None:continue
        a,b,c=(vertices[triangles[:,i]] for i in range(3));area=np.linalg.norm(np.cross(b-a,c-a),axis=1)
        rng=np.random.default_rng(int(e['id']));idx=rng.choice(len(triangles),500,p=area/area.sum())
        u=np.sqrt(rng.random(500));v=rng.random(500)
        samples.append((1-u[:,None])*a[idx]+(u*(1-v))[:,None]*b[idx]+(u*v)[:,None]*c[idx])
        selected.append({'id':e['id'],'name':e['name'],'group':group})
    return selected,np.concatenate(samples)


def main():
    OUT.mkdir(exist_ok=True)
    elements,points=bim_samples();rows=[];manifest={'origin_global':ORIGIN.tolist(),'bim_revision':'v3 (inspection default, not date assignment)','dates':[]}
    fig,axes=plt.subplots(2,4,figsize=(16,9),constrained_layout=True)
    for date,ax in zip(DATES,axes.flat):
        base=ROOT/'results/0929_baseline' if date=='0929' else OUT/date
        info=json.loads((base/'alignment.json').read_text())
        if info.get('status')=='rejected':raise ValueError(f'{date}: registration has not passed review')
        gs=base/f'{date}_aligned_local.ply'
        n=len(PlyData.read(gs,mmap='r')['vertex'].data)
        las=laspy.read(ROOT/f'assets/{date}/{date}.las');ref=np.c_[las.x,las.y,las.z]-ORIGIN
        tree=cKDTree(ref);errors=tree.query(points,workers=4)[0].reshape(len(elements),500)
        groupstats={}
        erows=[]
        for e,err in zip(elements,errors):erows.append({**e,'median_m':float(np.median(err)),'fraction_within_1m':float(np.mean(err<1))})
        for group in GROUPS:
            values=[e['median_m'] for e in erows if e['group']==group]
            groupstats[group]={'elements':len(values),'median_of_element_medians_m':float(np.median(values))}
        (OUT/f'{date}_bim.json').write_text(json.dumps({'revision':'v3','groups':groupstats,'elements':erows},indent=2,ensure_ascii=False)+'\n')
        metric=info['ply_withheld_overlap_distances_m']
        row={'date':date,'gaussians':n,'median_m':metric['median'],'p90_m':metric['p90'],'within_1m':info['ply_withheld_overlap_fraction_within_1m'],
             'xy_overlap':info['ply_horizontal_overlap_fraction'],'heldout_samples':metric['count'],'bim_groups':groupstats}
        rows.append(row)
        manifest['dates'].append({**row,'label':date[2:] + '.' + date[:2],
              'landmark_distances':{e['id']:e['median_m'] for e in erows},
              'gs_url':'/'+str(gs.relative_to(ROOT)),
              'las_url':'/results/0929_bim/las_preview.ply' if date=='0929' else f'/results/multidate/{date}/las_preview.ply'})
        data=np.load(base/'diagnostics.npz');source=data['source']
        shown=source[::max(1,len(source)//16000)]
        ax.scatter(ref[::100,0],ref[::100,1],s=.8,c='#6c7c89',alpha=.6,label='LAS')
        ax.scatter(shown[:,0],shown[:,1],s=.5,c='#eb8f3d',alpha=.6,label='3DGS centres')
        ax.set(xlim=(-220,230),ylim=(-250,440),aspect='equal',title=f"{date} · median {metric['median']:.2f} m",xlabel='Local X, m',ylabel='Local Y, m');ax.grid(alpha=.2)
        print(date,row,flush=True)
    ax=axes.flat[-1]
    x=np.arange(7);ax.bar(x,[r['median_m'] for r in rows],label='Median',color='#eb8f3d')
    ax.plot(x,[r['p90_m'] for r in rows],'o-',label='P90',color='#315874');ax.set(xticks=x,xticklabels=DATES,ylabel='Distance to own LAS, m',title='Held-out centres → own LAS');ax.tick_params(axis='x',rotation=45);ax.legend();ax.grid(axis='y',alpha=.2)
    axes.flat[0].legend(loc='upper left',markerscale=4)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    baseline=rows[-1]['median_m']
    for row in rows:row['delta_median_vs_0929_m']=row['median_m']-baseline
    (OUT/'comparison.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='bim_groups'],extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# All seven captures: 3DGS → LAS → BIM','',
           'All dates share one metre coordinate frame: LAS/BIM world = local + `[825800,842100,0]`. Original files are preserved. BIM is not fitted to individual captures; the unchanged v3 is used for comparison. Official BIM-revision assignments to dates remain unconfirmed.','',
           '## 3DGS agreement with same-date LAS','',
           '| Date | Gaussians | Median, m | P90, m | Within 1 m | XY coverage | Median change vs 0929, m |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['date']} | {r['gaussians']:,} | {r['median_m']:.3f} | {r['p90_m']:.3f} | {r['within_1m']:.1%} | {r['xy_overlap']:.1%} | {r['delta_median_vs_0929_m']:+.3f} |")
    lines+=['','Every tenth Gaussian is sampled, with opacity logit > 0 and the outer 0.5% removed on each source axis. About 25% of spatial tiles, each 0.4 source units wide, is excluded from fitting. Median/P90 and fraction within 1 m use held-out centers inside LAS XY coverage (nearest XY point < 1 m). Source-unit size and tile shape differ between dates. This diagnoses agreement with each date-specific LAS; it is neither independent survey accuracy nor a measurement of construction change.',
            '0929 retains the RTK-camera/COLMAP initialization. The other six dates lack camera exports: four geometric orientations are tested, then scale, rotation, and translation are refined. Coverage differences made this insufficient for 0708, so a rough top-view seed from recognizable objects was followed by the same training-tile fit. align_dates.py contains the seed; 0708/landmark_candidate.json and 0708/geometric_rejected.json retain landmarks and rejected hypotheses. These landmarks are not survey control. Color and visual checks help distinguish false local minima.','',
            'Export transforms coordinates, quaternions, log scales, and color spherical harmonics. All original Gaussians are retained, including those excluded from fitting. Local/global matrices, input SHA-256 hashes, and reopened-export checks are saved for every date.','',
            'A diagnostic guard prevents export of clearly poor candidates: training median > 0.5 m or median RGB difference > 0.35 produces rejected status. This is an internal gross-error filter for this dataset, not an agreed accuracy tolerance.','',
            '## One BIM v3 compared with LAS from every date','',
            'Median of element-wise median surface distances to LAS, in metres. Each element uses the same 500 surface samples across dates, and each element has equal weight.','',
            '| Group | '+' | '.join(DATES)+' |','|---|'+'---:|'*7]
    for g,label in GROUPS.items():lines.append('| '+label+' | '+' | '.join(f"{r['bim_groups'][g]['median_of_element_medians_m']:.2f}" for r in rows)+' |')
    lines+=['','Samples include underside, hidden, and underground surfaces. Captures also differ in coverage and construction stage. Large distances do not prove that a structure is missing or registration is wrong. This table cannot yield a completion percentage; changes must be measured separately on commonly observed regions. BIM statistics use the original LAS and are separate from 3DGS registration errors.','',
            '## Viewing and reproduction','',
            'http://127.0.0.1:8765/viewer/ changes 3DGS and its LAS together while preserving the viewpoint and BIM revision. Full clouds load one at a time; the viewer subsamples LAS by 10, while numerical checks use the full LAS.','',
            '```sh','OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/align_dates.py',
            '.venv/bin/python scripts/export_dates.py','.venv/bin/python scripts/compare_dates.py','.venv/bin/python scripts/preview_dates.py','.venv/bin/python scripts/serve_viewer.py','```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
