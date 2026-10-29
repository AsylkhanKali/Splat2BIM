"""Report a controlled source-label substitution without hiding rejected poses."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from align_object_recovery import load_fragments
from align_object_consensus import patch_checks
from object_consensus import match,descriptors
from diagnose_strict_support import triangle_diagnostics
from check_las_vertical import write_csv,file_hash
from align_semantic_3dgs import OUT,METHOD,PREVIOUS

METHODS=['current_automatic','previous_semantic',METHOD]
NAMES=['Current method','Previous semantics','New 3DGS labels']


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES};rows=[];statics=[];changes=[];diagnostics={}
    for date,audit in audits.items():
        for method in METHODS:
            row=next((r for r in audit['comparison'] if r['method']==method and r['scope']=='fixed_baseline_coverage'),None)
            rows.append({'date':date,'method':method,'status':audit['comparison_status'][method],
                'strict_pass':audit['comparison_status'][method]=='candidate_for_review' and audit['comparison_strict_support'][method]>=3,
                'strict_support_count':audit['comparison_strict_support'][method],
                **({k:row[k] for k in ['count','median','p90','p95','rmse','within_1m']} if row else {k:None for k in ['count','median','p90','p95','rmse','within_1m']})})
        with (OUT/date/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
        run=audit['run'];delta={'date':date,'previous_starts':audit['previous_initial_starts'],'new_starts':audit['initial_starts'],
            'previous_selected':audit['previous_selected_candidate'],'new_selected':run.get('selected_candidate_id'),
            'selected_seed_changed':audit['previous_selected_candidate']!=run.get('selected_candidate_id'),
            'shared_classes':audit['shared_classes'],'source_built_count':audit['source_class_counts']['built'],'target_built_count':audit['target_class_counts']['built']}
        if 'matrix_local' in run:
            held=np.load(OUT/date/'held_out_distances.npz');p=held['source_points'][held['fixed_coverage']];new=apply(p,np.array(run['matrix_local']))
            old=json.loads((PREVIOUS/date/'audit.json').read_text());previous=next(r for r in old['runs'] if r['method']=='semantic_fusion')
            current=json.loads((ROOT/f'results/object_recovery/{date}/selection.json').read_text())
            for name,record in [('current',current),('previous',previous)]:
                distance=np.linalg.norm(new-apply(p,np.array(record['matrix_local'])),axis=1)
                delta[name+'_pose_displacement_median_m']=float(np.median(distance));delta[name+'_pose_displacement_p95_m']=float(np.quantile(distance,.95))
            if date in ['0708','0723']:
                matrix=np.array(run['matrix_local']);sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf)
                pairs=patch_checks(match(sd,td,matrix),sf,tf,matrix);strict=[p for p in pairs if p['fraction_within_1m']>=.8 and p['reverse_fraction_within_1m']>=.5]
                triangles=triangle_diagnostics(strict,sd,td,matrix)
                previous_ids={(p['source_id'],p['target_id']) for p in previous['final_pair_checks']};new_ids={(p['source_id'],p['target_id']) for p in run['final_pair_checks']}
                diagnostics[date]={'new_bank_strict_pairs':run['strict_overlap_pairs'],'new_bank_triangles':run['strict_triangles'],
                    'new_bank_admissible_triangles':run['admissible_strict_triangles'],
                    'postfit_all_fragments_diagnostic_only':True,'postfit_strict_pairs':strict,'postfit_triangles':triangles,
                    'postfit_admissible_triangles':sum(t['passes'] for t in triangles),
                    'bank_identities_added':[list(p) for p in sorted(new_ids-previous_ids)],'bank_identities_removed':[list(p) for p in sorted(previous_ids-new_ids)]}
        changes.append(delta)
    write_csv(OUT/'summary.csv',rows);write_csv(OUT/'static_structures.csv',statics);write_csv(OUT/'initialization_changes.csv',changes)
    (OUT/'strict_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    find=lambda d,m:next(r for r in rows if r['date']==d and r['method']==m)
    accepted={m:[r['date'] for r in rows if r['method']==m and r['status']=='candidate_for_review'] for m in METHODS}
    strict={m:[r['date'] for r in rows if r['method']==m and r['strict_pass']] for m in METHODS}
    paired=[d for d in accepted[METHOD] if d in accepted['current_automatic']]
    improved=[d for d in paired if find(d,METHOD)['median']<find(d,'current_automatic')['median']-.01]
    worse=[d for d in paired if find(d,METHOD)['median']>find(d,'current_automatic')['median']+.01]
    summary={'accepted_dates':accepted,'strict_dates':strict,'accepted_median_improvements_over_1cm':improved,
        'accepted_median_regressions_over_1cm':worse,'rejected_dates':[d for d in DATES if d not in accepted[METHOD]],
        'changes':changes,'source_sha256':{n:file_hash(ROOT/'scripts'/n) for n in ['report_semantic_3dgs_registration.py']},
        'audit_sha256':{d:file_hash(OUT/d/'audit.json') for d in DATES},'strict_diagnostic_sha256':file_hash(OUT/'strict_diagnostics.json')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4.2),constrained_layout=True);x=np.arange(7)
    for ax,key,title in zip(axes,['median','p90'],['Median distance to LAS','90th percentile distance to LAS']):
        for shift,method,label,color in zip([-.25,0,.25],METHODS,NAMES,['#6b7d92','#8a77ab','#21876f']):
            values=[find(d,method)[key] for d in DATES];ax.bar(x+shift,[v if v is not None else np.nan for v in values],.23,label=label,color=color)
            for i,d in enumerate(DATES):
                if d not in accepted[method] and values[i] is not None:ax.plot(i+shift,values[i],'x',color='#b32937',markersize=8,markeredgewidth=2)
        ax.set(xticks=x,xticklabels=DATES,ylabel='Metres; lower is better',title=title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Identical held-out points and coverage · × = rejected result')
    fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    gate=lambda d,m:'yes' if d in accepted[m] else 'rejected'
    strict_text=lambda d,m:'yes' if d in strict[m] else 'no'
    table=['| Date | Current median / P90, m | Previous semantic median / P90, m | New-label median / P90, m | New acceptance | Strict: current / previous / new |','|---|---:|---:|---:|---|---|']
    for d in DATES:table.append('| '+d+' | '+' | '.join(fmt(find(d,m)['median'])+' / '+fmt(find(d,m)['p90']) for m in METHODS)+' | '+gate(d,METHOD)+' | '+' / '.join(strict_text(d,m) for m in METHODS)+' |')
    counts='; '.join(f'{name}: primary {len(accepted[m])}/7, strict {len(strict[m])}/7' for m,name in zip(METHODS,NAMES))
    conclusions=[]
    for d in ['0708','0723']:
        r=audits[d]['run'];e=diagnostics.get(d)
        if d in strict[METHOD]:text=f'{d}: strict check passed, spatial supports {r["strict_forward_080_spatial_support_count"]}.'
        elif e:
            text=f'{d}: strict check failed. Pairs with ≥80% forward and ≥50% reverse overlap: {len(r["strict_overlap_pairs"])}; valid triples in the initial bank: {r["admissible_strict_triangles"]}.'
            if r['strict_triangles']:
                t=max(r['strict_triangles'],key=lambda t:min(t['source_second_axis_m'],t['target_second_axis_m']))
                text+=f' For the best triple, second spatial axes in 3DGS / LAS are {t["source_second_axis_m"]:.6f} / {t["target_second_axis_m"]:.6f} m against a minimum of 2 m.'
            text+=f' Diagnostic matching of all fragments at the new final pose gives {len(e["postfit_strict_pairs"])} eligible pairs and {e["postfit_admissible_triangles"]} valid triples; it does not affect acceptance.'
        else:text=f'{d}: no admissible candidate for final refinement.'
        conclusions.append(text)
    lines=['# Registration with new 3DGS labels: seven dates','',counts+'.','',
        f'Among accepted results, median improvements above 1 cm relative to the current method: {len(improved)}; regressions above 1 cm: {len(worse)}. Rejected results: '+(', '.join(summary['rejected_dates']) or 'no')+'.','',
        '## Comparison','',*table,'','![Alignment errors](comparison.png)','',
        'Rejected results retain their measured errors and are not counted as successes. Distances use identical spatially held-out 3DGS centers and the supplied LAS with unchanged coverage. This is internal agreement, not independent survey accuracy. The 1 cm threshold describes difference magnitude, not statistical significance.','',
        'The current method is the previous automatic combined selection. Previous semantics refers to the LoveDA specialist with the neighboring-start pair bank. The third variant replaces only source 3DGS labels with Gaussian-view predictions; LAS labels are unchanged. Historical viewer exports differ from the experimental automatic method and are not replaced.','',
        '## Repeat-experiment conditions','',
        'For every date, the entire previous hypothesis pool is rescored, four spaced starts are selected, and ICP and final refinement are rerun. Classes, weights, random seed, thresholds, and budget are preserved. Neighboring-hypothesis pairs are fixed before ICP. Previous final matrices are not used as starts; they enter only the comparison after the new fit.','',
        'In this implementation, semantics affects start ranking and candidate selection after coarse fitting. Final refinement remains geometric, and strict-check fragments are not re-extracted using the new labels. Better building recognition therefore does not automatically repair support boundaries or spatial distribution.','',
        'Primary acceptance retains the scale constraints, training median ≤0.5 m, color difference ≤0.35, and three spaced pairs with ≥50% bidirectional overlap within 1 m. Strict acceptance additionally requires ≥80% forward overlap, support spacing ≥10 m, and a second singular value of centered support coordinates ≥2 m in both scenes. No threshold changed after inspecting the result.','',
        'The new labels were developed on this site: this is a controlled variant comparison, not an independent transfer test. Roof and riverside control regions do not select the pose or parameters.','',
        '## 0708 and 0723','',*sum(([t,''] for t in conclusions),[]),
        '0708 has many new building labels in 3DGS, but only 23 in LAS against a 128-point shared-class minimum. Buildings therefore do not contribute to semantic distance on this date. A shared structure class appears on 0723; its presence alone does not guarantee three reliable spatial supports.','',
        'All pairs, forward/reverse overlaps, and triples are recorded in `<date>/audit.json` and `strict_diagnostics.json`. All-fragment matching at the final pose is explanatory only and does not add pairs retrospectively.','',
        '## Initialization changes','',
        '| Date | Previous selected start | New selected start | Shared structure class | Pose displacement vs current, P95, m |','|---|---|---|---|---:|']
    for c in changes:lines.append(f'| {c["date"]} | {c["previous_selected"]} | {c["new_selected"] or "—"} | {"yes" if "built" in c["shared_classes"] else "no"} | {c.get("current_pose_displacement_p95_m",float("nan")):.6g} |')
    lines+=['','Pose displacement is the distance between the same held-out points under two transforms. It separates position change from changes in the supporting-pair set.','',
        '## Stable structures','',
        'The roof and riverside structure are evaluated in the previous fixed masks excluding cranes and equipment. `static_structures.csv` includes unused-record and spatial-holdout samples, medians, P90, and counts for all three methods. Samples below 30 points are marked insufficient; missing metrics are not converted to zero. Unused-record samples have nearby training points and are weaker than spatial holdouts.','',
        '## Reproduction','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic_3dgs.py','.venv/bin/python scripts/report_semantic_3dgs_registration.py',
        ".venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_3dgs_registration/tests.log 2>&1",
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic_3dgs_registration.py','```','',
        'The previous labels, geometric pools, and models are prerequisites. No new packages or downloads are required once these are restored. `protocol.json` records conditions and code hashes; `validation.json` records numerical re-verification of hypotheses, acceptance, and errors; `summary.csv` contains results; `initialization_changes.csv` records label effects on start selection. See the [previous recognition experiment](../semantic_3dgs/REPORT.md).','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>New labels: repeated registration across seven dates</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:95ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}a:focus-visible,summary:focus-visible{outline:2px solid #175794}h1{font-size:clamp(24px,3vw,36px)}.rejected{color:#a52635;font-weight:650}</style>',
        '<a href="/viewer/">← 3D viewer</a><h1>New labels: repeated registration across seven dates</h1>',
        '<p>'+counts+'.</p>',
        f'<p>With unchanged thresholds and four initial hypotheses per date, new accepted median improvements above 1 cm: <strong>{len(improved)}</strong>; regressions: <strong>{len(worse)}</strong>. Rejected results are shown in the table and are not counted as successes.</p>',
        '<p><a href="REPORT.md">Method and findings</a> · <a href="summary.csv">All results CSV</a> · <a href="static_structures.csv">Stable structures</a> · <a href="../semantic_3dgs/inspection.html">Recognition before / after</a></p>',
        '<div class="table"><table><thead><tr><th>Date</th><th>Current median / P90, m</th><th>Previous semantics, m</th><th>New labels, m</th><th>New acceptance</th><th>Strict: current / previous / new</th></tr></thead><tbody>']
    for d in DATES:page.append('<tr><td>'+d+'</td>'+''.join('<td>'+fmt(find(d,m)['median'])+' / '+fmt(find(d,m)['p90'])+'</td>' for m in METHODS)+f'<td class="{"rejected" if d not in accepted[METHOD] else ""}">{gate(d,METHOD)}</td><td>'+ ' / '.join(strict_text(d,m) for m in METHODS)+'</td></tr>')
    page+=['</tbody></table></div><p>Identical held-out points; lower distance is better. This measures agreement with the supplied LAS. Full working scenes and BIM are preserved.</p>',
        '<img src="comparison.png" alt="Median and P90 distance to LAS for three variants across seven dates">',
        '<h2>Strict checks on the difficult dates</h2>',*['<p>'+t+'</p>' for t in conclusions],
        '<p>On 0708, a shared building class is unavailable: LAS has only 23 confident points against a minimum of 128. The class appears on 0723. All thresholds are unchanged.</p>']
    for d in ['0708','0723']:
        r=audits[d]['run'];strict_ids={(p['source_id'],p['target_id']) for p in r.get('strict_overlap_pairs',[])}
        page.append(f'<details><summary>{d}: checks for every initial pair</summary><div class="table"><table><thead><tr><th>3DGS → LAS</th><th>Forward overlap</th><th>Reverse overlap</th><th>Strict overlap and assignment</th></tr></thead><tbody>')
        for p in r.get('final_pair_checks',[]):page.append(f'<tr><td>{p["source_id"]} → {p["target_id"]}</td><td>{p["fraction_within_1m"]*100:.1f}%</td><td>{p["reverse_fraction_within_1m"]*100:.1f}%</td><td>{"yes" if (p["source_id"],p["target_id"]) in strict_ids else "no"}</td></tr>')
        page.append('</tbody></table></div><p>Final strict acceptance still requires at least three spatially separated supports.</p></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
