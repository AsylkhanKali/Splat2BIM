"""Compare persistent surface registration with both frozen reference methods."""
import csv
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES,ORIGIN
from align_0929 import apply
from align_stable_structures import OUT,BASE,METHOD
from prepare_stable_structures import load_structures
from check_las_vertical import file_hash,write_csv

METHODS=['current_automatic','previous_semantic',METHOD]
NAMES=['Current automatic','Previous 3DGS labels','Persistent structures']


def plot_support(date,run,sf,tf):
    pairs=run['strict_spatial_support'];matrix=np.array(run['matrix_local'])
    las=laspy.read(ROOT/f'assets/{date}/{date}.las');xyz=np.c_[las.x,las.y,las.z]-ORIGIN
    rgb=np.c_[las.red,las.green,las.blue].astype(float);rgb/=65535 if rgb.max()>255 else 255
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True);axes=axes.ravel()
    axes[0].scatter(xyz[::20,0],xyz[::20,1],c=rgb[::20],s=.5,rasterized=True)
    centers=[]
    for i,pair in enumerate(pairs):
        a=sf['candidates'][pair['source_index']];b=tf['candidates'][pair['target_index']]
        source=apply(a['points']/sf['normalization_scale']+sf['normalization_center'],matrix)
        target=b['points']/tf['normalization_scale']+tf['normalization_center'];center=target.mean(0);centers.append(center)
        axes[0].scatter(*center[:2],s=120,facecolors='none',edgecolors='#ad193c',linewidths=2)
        axes[0].annotate(str(i+1),center[:2],xytext=(7,6),textcoords='offset points',weight='bold',color='#ad193c')
        if i+1>=len(axes):continue
        ax=axes[i+1];lo=np.minimum(source.min(0),target.min(0))-3;hi=np.maximum(source.max(0),target.max(0))+3
        context=np.all((xyz[:,:2]>=lo[:2])&(xyz[:,:2]<=hi[:2]),axis=1)
        ax.scatter(xyz[context,0],xyz[context,1],c=rgb[context],s=2,alpha=.45,rasterized=True)
        ax.scatter(target[:,0],target[:,1],s=10,facecolors='none',edgecolors='#a33b9c',linewidths=.6,label='LAS surface')
        ax.scatter(source[:,0],source[:,1],s=5,c='#007f78',label='Aligned 3DGS')
        ax.set(xlim=[lo[0],hi[0]],ylim=[lo[1],hi[1]],
            title=f'{i+1}. Forward {pair["fraction_within_1m"]:.1%} / reverse {pair["reverse_fraction_within_1m"]:.1%}\nPersistent in {len(b["supported_dates"])}/7 LAS captures')
        ax.legend(fontsize=8,loc='best')
    centers=np.array(centers)
    if len(centers):
        axes[0].set(xlim=[centers[:,0].min()-30,centers[:,0].max()+30],ylim=[centers[:,1].min()-30,centers[:,1].max()+30])
    axes[0].set_title('Accepted independent surface families')
    for i,ax in enumerate(axes):
        if i>len(pairs):ax.set_visible(False);continue
        ax.set_aspect('equal');ax.set_xlabel('Local X, m');ax.set_ylabel('Local Y, m')
    fig.suptitle(date+' · measured surfaces, unchanged 80% / 50% overlap gates',fontsize=16)
    fig.savefig(OUT/f'{date}_supports.png',dpi=160);plt.close(fig)


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES};rows=[];statics=[];pairs=[];changes=[]
    for date,audit in audits.items():
        run=audit['run'];sf,tf=load_structures(date,'source'),load_structures(date,'target')
        for method in METHODS:
            row=next(r for r in audit['comparison'] if r['method']==method and r['scope']=='fixed_baseline_coverage')
            rows.append({**row,'strict_pass':row['status']=='candidate_for_review' and row['strict_support_count']>=3})
        with (OUT/date/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
        for p in run['strict_spatial_support']:
            a=sf['candidates'][p['source_index']];b=tf['candidates'][p['target_index']]
            pairs.append({'date':date,'source_id':p['source_id'],'target_id':p['target_id'],'source_family':a['family'],'target_family':b['family'],
                'source_extent':a['extent_kind'],'target_extent':b['extent_kind'],
                'forward_fraction':p['fraction_within_1m'],'reverse_fraction':p['reverse_fraction_within_1m'],
                'supported_dates':','.join(b['supported_dates']),'source_points':len(a['points']),'target_points':len(b['points'])})
        held=np.load(OUT/date/'held_out_distances.npz');points=held['source_points'][held['fixed_coverage']]
        moved=apply(points,np.array(run['matrix_local']));change={'date':date}
        previous=json.loads((BASE/date/'audit.json').read_text())['run']
        current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text())
        for name,old in [('current',current),('previous',previous)]:
            distance=np.linalg.norm(moved-apply(points,np.array(old['matrix_local'])),axis=1)
            change[name+'_pose_p95_m']=float(np.quantile(distance,.95))
        changes.append(change)
        if date in ['0708','0723']:plot_support(date,run,sf,tf)
    write_csv(OUT/'summary.csv',rows);write_csv(OUT/'static_structures.csv',statics)
    write_csv(OUT/'support_pairs.csv',pairs);write_csv(OUT/'pose_changes.csv',changes)
    accepted={m:[r['date'] for r in rows if r['method']==m and r['status']=='candidate_for_review'] for m in METHODS}
    strict={m:[r['date'] for r in rows if r['method']==m and r['strict_pass']] for m in METHODS}
    find=lambda d,m:next(r for r in rows if r['date']==d and r['method']==m)
    summary={'accepted_dates':accepted,'strict_dates':strict,'pose_changes':changes,
        'audit_sha256':{d:file_hash(OUT/d/'audit.json') for d in DATES},
        'source_sha256':{'report_stable_structures.py':file_hash(ROOT/'scripts/report_stable_structures.py')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True);x=np.arange(7)
    for offset,method,name,color in zip([-.25,0,.25],METHODS,NAMES,['#738296','#9c73ac','#208473']):
        axes[0].bar(x+offset,[find(d,method)['strict_support_count'] for d in DATES],.23,label=name,color=color)
        axes[1].bar(x+offset,[find(d,method)['median'] for d in DATES],.23,color=color)
    axes[0].axhline(3,color='#555',linestyle='--',linewidth=1);axes[0].legend(fontsize=8)
    axes[0].set(title='Independent supports passing strict gates',ylabel='Support count')
    axes[1].set(title='Held-out median distance to LAS',ylabel='Metres; lower is better')
    for ax in axes:ax.set(xticks=x,xticklabels=DATES);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    table=['| Date | Strict: current / previous / new | New supports | New median / P90, m | Median change vs current, mm |',
        '|---|---|---:|---:|---:|']
    htmlrows=[]
    for d in DATES:
        r=find(d,METHOD);states=['yes' if d in strict[m] else 'no' for m in METHODS]
        values=[d,' / '.join(states),str(r['strict_support_count']),f'{r["median"]:.3f} / {r["p90"]:.3f}',f'{1000*(r["median"]-find(d,"current_automatic")["median"]):+.4f}']
        table.append('| '+' | '.join(values)+' |');htmlrows.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    counts='; '.join(f'{n}: primary {len(accepted[m])}/7, strict {len(strict[m])}/7' for m,n in zip(METHODS,NAMES))
    conclusions=[]
    for d in ['0708','0723']:
        run=audits[d]['run'];chosen={(p['source_id'],p['target_id']) for p in run['strict_spatial_support'][:3]}
        triangle=next((t for t in run['strict_triangles'] if t['passes'] and {tuple(p) for p in t['identities']}==chosen),None)
        text=f'{d}: strict acceptance '+('passed' if d in strict[METHOD] else 'failed')+f', supports {run["strict_forward_080_spatial_support_count"]}.'
        if triangle:text+=f' One accepted triple has 3DGS / LAS second axis: {triangle["source_second_axis_m"]:.3f} / {triangle["target_second_axis_m"]:.3f} m (minimum 2 m); minimum spacing: {triangle["source_min_spacing_m"]:.2f} / {triangle["target_min_spacing_m"]:.2f} m (minimum 10 m).'
        conclusions.append(text)
    maximum=max(c['current_pose_p95_m'] for c in changes)
    explanation=[
        'Surfaces are extracted independently in 3DGS and LAS. Connected planar surfaces of all colors and measured infill within the original footprints supplement the earlier white fragments. No points are synthesized; neighboring planes and confident vegetation, water, and moving-object labels are excluded from growth.',
        'All extents of one physical surface belong to one family. Only one correspondence per family in each scene can count as support. All pairs within the existing geometric gates are retained so that the nearest center cannot displace the correct match. The identity bank is fixed from initial hypotheses before ICP; no new pairs are added after fitting.',
        'A target support must also have at least 80% of its points within 1 m of LAS from at least three dates, including its own. This check uses the supplied shared LAS frame, not the 3DGS transforms. It provides persistence evidence rather than guaranteed classification: stationary equipment could also persist between dates.',
        'All numerical thresholds are unchanged: center distance ≤5 m, normals ≤20°, size ratio 0.5–2; at least three supports, spacing ≥10 m, and second axis ≥2 m in both scenes. Overlap within 1 m must be ≥80% forward and ≥50% reverse. The training-median limit of 0.5 m, color-difference limit of 0.35, and scale limits are preserved.',
        'All seven dates use the same rules, the same initial hypotheses, and four starts. Start scores come from the verified, unchanged Gaussian-label experiment; ICP and final refinement were run again. Previous final matrices enter only the comparison after fitting.',
        f'Maximum P95 pose displacement relative to the current method: {maximum:.9g} m. Independent surfaces provide stronger support for the alignment; these results do not show a meaningful geometric accuracy gain.',
        'Held-out points and fixed coverage are identical for all three methods. The roof and riverside structure are also checked in the previous masks excluding cranes and equipment; static_structures.csv includes both control samples and their counts. These control regions do not select parameters or poses.',
        'This is development and internal evaluation on one site. Strict acceptance is not independent survey certification. BIM and the working viewer exports were not rebuilt: the experiment checks 3DGS → LAS, while the existing LAS → BIM placement is preserved.',
        'Less successful variants were also tested: surface growth alone lost useful original extents, and temporal matching of extracted fragments alone was sensitive to clipping. These diagnostic experiments are retained in results/structural_surfaces. Final numerical thresholds were not tuned on held-out errors.'
    ]
    lines=['# Persistent structures: repeated strict validation','',counts+'.','',*table,'',
        '![Comparison](comparison.png)','',*sum(([t,''] for t in conclusions),[]),'## What changed','',*sum(([t,''] for t in explanation),[]),
        '## Inspected supports','', '![0708](0708_supports.png)','', '![0723](0723_supports.png)','',
        'Exact pairs, point counts, and persistence dates: support_pairs.csv. All alternatives, initial banks, overlaps, and triples: <date>/audit.json. Thresholds and hashes: protocol.json and extraction_protocol.json. Numerical re-verification: validation.json.','',
        '## Reproduction','', '```sh','.venv/bin/python scripts/prepare_stable_structures.py',
        '.venv/bin/python scripts/align_stable_structures.py',".venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/stable_structures/tests.log 2>&1",
        '.venv/bin/python scripts/verify_stable_structures.py','.venv/bin/python scripts/report_stable_structures.py','```','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Persistent structures: seven dates</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:100ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}h1{font-size:clamp(24px,3vw,36px)}a:focus-visible,summary:focus-visible{outline:2px solid #175794}</style>',
        '<a href="/viewer/">← 3D viewer</a><h1>Persistent structures: strict validation</h1><p>'+counts+'.</p>',
        '<p>Thresholds are unchanged. New supports are checked for persistence across LAS dates; each physical surface counts once.</p>',
        '<p><a href="REPORT.md">Method</a> · <a href="summary.csv">Comparison CSV</a> · <a href="support_pairs.csv">Support pairs</a> · <a href="static_structures.csv">Control structures</a> · <a href="validation.json">Numerical verification</a></p>',
        '<div class="table"><table><thead><tr><th>Date</th><th>Strict: current / previous / new</th><th>New supports</th><th>Median / P90, m</th><th>Median change vs current, mm</th></tr></thead><tbody>',*htmlrows,
        '</tbody></table></div><p>Previous refers to the earlier Gaussian-label experiment. Errors use identical held-out points relative to LAS.</p>',
        '<img src="comparison.png" alt="Strict supports and errors across all seven dates">','<h2>0708 and 0723</h2>',*['<p>'+t+'</p>' for t in conclusions],
        '<img src="0708_supports.png" alt="Map and aligned surfaces 0708"><img src="0723_supports.png" alt="Map and aligned surfaces 0723">',
        '<details><summary>Method and limitations</summary>',*['<p>'+t+'</p>' for t in explanation],'</details></html>']
    (OUT/'inspection.html').write_text('\n'.join(page)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
