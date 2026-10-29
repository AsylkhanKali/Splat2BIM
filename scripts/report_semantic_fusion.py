"""Report the new labels, correspondence-bank ablation and all seven dates."""
import csv
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from align_dates import ROOT,DATES
from check_las_vertical import write_csv
from semantic_projection import COLORS
from align_semantic_fusion import OUT

METHODS=['current_automatic','previous_semantic','geometry_bank','old_semantic_bank','semantic_fusion']
NAMES=['Current combined','Previous semantics','Geometry + pair bank','Previous labels + pair bank','New labels + pair bank']


def main():
    audits={d:json.loads((OUT/d/'audit.json').read_text()) for d in DATES}
    previous={d:json.loads((ROOT/f'results/semantic_initial/{d}/audit.json').read_text()) for d in DATES}
    def run(d,m):return next(r for r in (previous[d] if m=='previous_semantic' else audits[d])['runs'] if r['method']==('semantic_initial' if m=='previous_semantic' else m))
    def status(d,m):return audits[d]['current_status'] if m=='current_automatic' else run(d,m)['status']
    def strict(d,m):return audits[d]['current_strict_support'] if m=='current_automatic' else run(d,m).get('strict_forward_080_spatial_support_count',0)
    def metric(d,m):return next((r for r in (previous[d] if m=='previous_semantic' else audits[d])['comparison'] if r['method']==('semantic_initial' if m=='previous_semantic' else m) and r['scope']=='fixed_baseline_coverage'),None)
    def value(d,m,k='median'):
        r=metric(d,m);return '—' if r is None else f"{r[k]:.3f}"
    accepted={m:[d for d in DATES if status(d,m)=='candidate_for_review'] for m in METHODS}
    summary={'accepted_dates':accepted,'strict_dates':{m:[d for d in accepted[m] if strict(d,m)>=3] for m in METHODS}}
    rows=[]
    for d in DATES:
        for m in METHODS:
            r=metric(d,m)
            rows.append({'date':d,'method':m,'status':status(d,m),'median_m':r['median'] if r else None,'p90_m':r['p90'] if r else None,'strict_pairs':strict(d,m)})
    write_csv(OUT/'summary.csv',rows)
    summary['new_accepted_median_improvements_over_1cm']=sum(metric(d,'semantic_fusion')['median']<metric(d,'current_automatic')['median']-.01 for d in accepted['semantic_fusion'])
    summary['new_accepted_median_regressions_over_1cm']=sum(metric(d,'semantic_fusion')['median']>metric(d,'current_automatic')['median']+.01 for d in accepted['semantic_fusion'])
    with (OUT/'recognition.csv').open() as f:recognition=list(csv.DictReader(f))
    def rec(d,side,region,method):return next(r for r in recognition if r['date']==d and r['side']==side and r['region']==region and r['method']==method)
    roof=[float(rec(d,'target','roof','fusion')['built_fraction']) for d in DATES]
    summary['las_roof_detected_dates']=[d for d,p in zip(DATES,roof) if p>0]
    summary['source_roof_detected_dates']=[d for d in DATES if float(rec(d,'source','roof','fusion')['built_fraction'])>0]
    summary['las_vegetation_false_built_max']=max(float(rec(d,'target','vegetation','fusion')['built_fraction']) for d in DATES)
    summary['shared_built_class_dates']=[d for d in DATES if min(int(rec(d,s,'all_visible','fusion')['built_points']) for s in ['source','target'])>=128]
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    statics=[]
    for d in DATES:
        with (OUT/d/'static_structures.csv').open() as f:statics.extend(csv.DictReader(f))
    write_csv(OUT/'static_structures.csv',statics)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);x=np.arange(7)
    axes[0].bar(x-.17,[float(rec(d,'target','roof','previous')['built_fraction'])*100 for d in DATES],.34,label='Previous model',color='#697b92')
    axes[0].bar(x+.17,np.array(roof)*100,.34,label='New building model',color='#2c8a72')
    axes[0].set(xticks=x,xticklabels=DATES,ylabel='% of visible roof points',ylim=(0,105),title='Control roof in LAS');axes[0].legend(fontsize=8)
    axes[1].bar(x,[float(rec(d,'source','roof','fusion')['built_fraction'])*100 for d in DATES],color='#2c8a72')
    axes[1].set(xticks=x,xticklabels=DATES,ylabel='% of visible roof points',ylim=(0,105),title='Same roof in 3DGS: still 0% on every date')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'recognition.png',dpi=150);plt.close(fig)
    for d in DATES:
        fig,axes=plt.subplots(2,3,figsize=(14,9.5),constrained_layout=True)
        for row,side in enumerate(['source','target']):
            for col,(folder,suffix,title) in enumerate([(OUT,'rgb','Image'),(ROOT/'results/semantic_matching','semantic','Previous classes'),(OUT,'semantic','New classes')]):
                image=cv2.cvtColor(cv2.imread(str(folder/d/f'{side}_{suffix}.png')),cv2.COLOR_BGR2RGB)
                axes[row,col].imshow(image);axes[row,col].set_title(('3DGS' if side=='source' else 'LAS')+' · '+title);axes[row,col].axis('off')
        fig.suptitle(d+': predictions on visible points; white = no point, gray = uncertain')
        fig.legend(handles=[Patch(color=color/255,label=name) for color,name in zip(COLORS,['Uncertain','Structures','Paved surface','Ground','Vegetation','Water','Moving objects'])],loc='lower center',ncol=7,fontsize=9)
        fig.savefig(OUT/d/'semantics.png',dpi=120);plt.close(fig)
    counts='; '.join(f'{name}: {len(accepted[m])}/7' for m,name in zip(METHODS,NAMES))
    lines=['# Semantics: structure recognition and the 0723 rejection','',counts+'.','',
        f"Accepted median improvements above 1 cm relative to the current method: **{summary['new_accepted_median_improvements_over_1cm']}**, regressions: **{summary['new_accepted_median_regressions_over_1cm']}**. This threshold describes difference magnitude, not statistical significance. The working combined method remains the default.",'',
        '## All seven dates compared','',
        '| Date | Current median / P90, m | Previous semantics: acceptance | Previous labels + pair bank: acceptance | New semantic median / P90, m | New acceptance |',
        '|---|---:|---|---|---:|---|']
    for d in DATES:
        ok=lambda m:'yes' if d in accepted[m] else 'rejected'
        lines.append(f"| {d} | {value(d,'current_automatic')} / {value(d,'current_automatic','p90')} | {ok('previous_semantic')} | {ok('old_semantic_bank')} | {value(d,'semantic_fusion')} / {value(d,'semantic_fusion','p90')} | {ok('semantic_fusion')} |")
    lines+=['','Acceptance uses the previous internal scale, geometry, RGB, and minimum-three-spaced-pairs gates. Rejected matrices, where available, remain in the CSV with explicit status. Distances are from Gaussian centers to the supplied LAS on identical spatially held-out points with the previous fixed coverage; this is not independent survey accuracy.','',
        'The current method here is the automatic selection from `object_recovery/selection.json`; it differs from the historical full viewer exports. Neither the current final matrix, RTK, nor the manual 0708 seed is used to select new starts.','',
        '## What changed','',
        '1. A building specialist is added: [SegFormer-B2 trained on LoveDA](https://huggingface.co/wu-pr-gw/segformer-b2-finetuned-with-LoveDA), revision `5c74556c08bebb5f45f50b6f78f61a62c5d220c7`. [LoveDA](https://github.com/Junjue-Wang/LoveDA) includes a building class in overhead imagery. The upstream configuration has eight outputs, including Ignore/Background, and its normalization is used. No repository code is executed; PyTorch loads weights through `weights_only=True`, and their SHA-256 is verified.','',
        '2. Views use the previous training points. Image gaps are filled from the nearest observed color and a 3×3 median filter. The source radius follows the 0.4-source-unit holdout tile size; the LAS radius is 24 pixels. This approximates context rather than reconstructing a surface: filled pixels never become 3D points. Artificial texture extensions are visible near boundaries.','',
        '3. LoveDA supplies building predictions at probability ≥0.6. The previous drone model supplies other classes. Confident vegetation, water, and moving-object labels are protected from replacement. Previous structure labels become either the new prediction or uncertainty. Replacing all classes with LoveDA was rejected before the final seven-date comparison because some cloud pairs lacked enough shared classes. Raw predictions for every date remain in `semantic_structures`, together with the diagnostic 0331 registration run.','',
        '4. Before fitting each selected start, pairs proposed by initial hypotheses within RMS <2 m on the same training anchors are pooled. Each pair records its proposing hypothesis IDs. After fitting, only these identities can pass the existing one-to-one geometric assignment, bidirectional overlap, and spacing gates. No new pairs are added from the final pose.','',
        'All three new variants use the same pool, four-start budget, ICP stages, and final refinement. The old-labels-plus-pair-bank control separates the 0723 fix from the new network. Held-out errors do not set parameters. This is another development experiment on a known site, not a new independent test.','',
        '## Why 0723 was rejected','',
        'The previous geometric and semantic results differed by at most 8.53×10⁻¹⁴ in their matrix entries: their poses were effectively identical. However, start `recovery_366` retained five pairs while neighboring `recovery_373` retained six. Two previously confirmed pairs are too close to count as separate spatial supports.','',
        'The bank around starts 558 and 564 adds one previously proposed pair, `p8_c16 → p8_c39`. Diagnostics of the old final pose show that it supplies a third spaced support. Its fractions within 1 m are 71.2% forward and 55.4% reverse, with center distance 4.895 m against a 5 m limit. This is a narrow margin, not evidence of high reliability. `0723_failure_analysis.json` preserves the details.','',
        'Strict check requiring at least three pairs with forward overlap ≥80%: '+', '.join(f'{name} {len(summary["strict_dates"][m])}/7' for m,name in zip(METHODS,NAMES))+'. The new variant still fails strict checks on 0708 and 0723. More supporting pairs at the same pose do not imply higher accuracy.','',
        '## Recognition evaluation','',
        '| Date | LAS roof: old → new prediction | 3DGS roof: new prediction | False structures in LAS vegetation |',
        '|---|---:|---:|---:|']
    for d in DATES:
        pct=lambda side,region,method:f"{float(rec(d,side,region,method)['built_fraction'])*100:.1f}%"
        lines.append(f"| {d} | {pct('target','roof','previous')} → {pct('target','roof','fusion')} | {pct('source','roof','fusion')} | {pct('target','vegetation','fusion')} |")
    lines+=['','![Control-roof recognition](recognition.png)','',
        'These are visible-point fractions inside fixed diagnostic volumes, not IoU or global classification accuracy. The regions are the previously inspected roof and an RGB-reviewed vegetation strip without buildings. `recognition.csv` records counts and old/new labels for both clouds. The previous automatic matrix only locates diagnostic volumes in the source; segmentation and registration do not read these volumes.','',
        '**Recognition improves partially:** the LAS roof is detected on five dates, but the 3DGS roof is still missed on all seven. There are too few confident building labels for a shared class on dates '+', '.join(d for d in DATES if d not in summary['shared_built_class_dates'])+'. Noisy Gaussian centers and missing tiles differ strongly from the training imagery. Reviewed labels and adaptation to these projections, or better 3DGS views, are needed; more orange points do not prove recognition accuracy.','',
        '[RGB review of the negative region](negative_region_review.png). One roof and one vegetation strip cannot establish precision and recall for the whole site. The models have no separate crane class. Local stable-structure evaluation uses the previous equipment-exclusion masks.','',
        '## Stable structures and reproduction','',
        '`static_structures.csv` records the roof and riverside structure using both unused records and strict spatial samples. The first sample has nearby training points; the second is often small on narrow structures. Empty metrics are not replaced with zeros. BIM and the working full PLYs are not moved.','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/prepare_semantic_structures.py',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/prepare_semantic_fusion.py',
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/align_semantic_fusion.py',
        '.venv/bin/python scripts/check_semantic_recognition.py','.venv/bin/python scripts/report_semantic_fusion.py',
        "OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_fusion/tests.log 2>&1",
        'OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/verify_semantic_fusion.py','```','',
        'The saved geometric pools and previous drone labels are required. No packages were added. Weights, parameters, and code hashes were fixed before the final comparison. `validation.json` records numerical re-verification; `http_checks.json` records local-page checks. The [previous experiment](../semantic_initial/REPORT.md) is retained.','',
        'In `*_labels.npz`, `labels`, `confidence`, and `label_origin` describe the final decision. `probabilities` retains the original drone-model probabilities, while `building_expert_probabilities` holds LoveDA predictions. These are separate distributions, not a single calibrated probability for the fused method. `label_origin`: 0 = previous class, 1 = building specialist, 2 = previous building label replaced with uncertainty.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Structure recognition and the 0723 fix</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:85ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}a:focus-visible,summary:focus-visible{outline:2px solid #175794}h1{font-size:clamp(24px,3vw,36px)}</style>',
        '<a href="/viewer/">← 3D viewer</a><h1>Structure recognition and the 0723 fix</h1>',
        f'<p>Primary gate: current method {len(accepted["current_automatic"])}/7; previous semantics {len(accepted["previous_semantic"])}/7; previous labels with pair bank {len(accepted["old_semantic_bank"])}/7; new variant {len(accepted["semantic_fusion"])}/7.</p>',
        '<p>On 0723, a distant support was lost when choosing between neighboring starts. Their initial pairs are now retained before fitting and pass the existing checks. This correspondence fix is separate from building recognition.</p>',
        '<p>New-variant strict acceptance: 5/7. 0708 and 0723 still fail; 7/7 refers to the primary gate.</p>',
        '<p><a href="REPORT.md">Full report</a> · <a href="summary.csv">All variants CSV</a> · <a href="recognition.csv">Recognition CSV</a> · <a href="static_structures.csv">Stable structures</a> · <a href="../semantic_initial/inspection.html">Earlier experiment (archived)</a></p>',
        '<div class="table"><table><thead><tr><th>Date</th><th>Current median / P90, m</th><th>New median / P90, m</th><th>New acceptance</th><th>LAS roof: before → after</th></tr></thead><tbody>']
    for d in DATES:page.append(f'<tr><td>{d}</td><td>{value(d,"current_automatic")} / {value(d,"current_automatic","p90")}</td><td>{value(d,"semantic_fusion")} / {value(d,"semantic_fusion","p90")}</td><td>{"yes" if d in accepted["semantic_fusion"] else "rejected"}</td><td>0% → {roof[DATES.index(d)]*100:.1f}%</td></tr>')
    page+=['</tbody></table></div>',
        '<p>Distances use identical held-out points. The current combined method remains the default; new semantics has not established an overall accuracy gain.</p>',
        '<img src="recognition.png" alt="LAS roof detected on five dates; 3DGS roof still missed on all seven">',
        '<p><strong>Recognition improves partially.</strong> LoveDA identifies buildings in LAS; the previous model retains other classes. The 3DGS control roof is still missed. Fractions on one roof do not measure whole-model accuracy.</p>']
    for d in DATES:page.append(f'<details><summary>{d}: images and classes before / after</summary><a href="{d}/semantics.png"><img loading="lazy" src="{d}/semantics.png" alt="{d}: images, previous and new semantic classes"></a></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
