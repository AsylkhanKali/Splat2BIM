"""Compare seven Gaussian recognition runs and explain unchanged strict failures."""
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from check_static_structures import volume_mask
from check_las_vertical import write_csv,file_hash
from prepare_semantic_3dgs import OUT,OLD,PREVIOUS


def main():
    region_path=PREVIOUS/'recognition_regions.json';regions=json.loads(region_path.read_text())['regions']
    rows=[];summary=[];negative_fig,negative_axes=plt.subplots(1,7,figsize=(16,4),constrained_layout=True)
    for date,negative_ax in zip(DATES,negative_axes):
        folder=OUT/date;new=dict(np.load(folder/'source_labels.npz'));old=dict(np.load(PREVIOUS/date/'source_labels.npz'))
        matrix_path=ROOT/f'results/object_recovery/{date}/selection.json';matrix=np.array(json.loads(matrix_path.read_text())['matrix_local'])
        world=apply(new['points'],matrix);masks={r['id']:volume_mask(world,r) for r in regions}
        for method,data in [('previous_fusion',old),('gaussian_render',new)]:
            for region,mask in {**masks,'all_visible':np.ones(len(world),bool)}.items():
                rows.append({'date':date,'method':method,'region':region,'count':int(mask.sum()),
                    'built_points':int(np.sum(data['labels'][mask]==1)),
                    'built_fraction':float(np.mean(data['labels'][mask]==1)) if mask.any() else None})
        previous_audit=json.loads((PREVIOUS/date/'audit.json').read_text());run=next(r for r in previous_audit['runs'] if r['method']=='semantic_fusion')
        diagnostic_path=ROOT/f'results/strict_diagnostics/{date}.json';diagnostic=json.loads(diagnostic_path.read_text())
        value=lambda method,region:next(r['built_fraction'] for r in rows if r['date']==date and r['method']==method and r['region']==region)
        summary.append({'date':date,'roof_count':int(masks['roof'].sum()),'previous_roof_fraction':value('previous_fusion','roof'),
            'new_roof_fraction':value('gaussian_render','roof'),'vegetation_count':int(masks['vegetation'].sum()),
            'previous_vegetation_fraction':value('previous_fusion','vegetation'),'new_vegetation_fraction':value('gaussian_render','vegetation'),
            'old_built_count':int(np.sum(old['labels']==1)),'new_built_count':int(np.sum(new['labels']==1)),
            'strict_overlap_pairs':diagnostic['strict_pair_count'],'admissible_triangles':diagnostic['admissible_bank_triangles'],
            'previous_primary_status':run['status'],'previous_strict_pass':run['strict_forward_080_spatial_support_count']>=3,
            'evaluation_matrix_sha256':file_hash(matrix_path),'strict_diagnostic_sha256':file_hash(diagnostic_path)})
        old_rgb=cv2.cvtColor(cv2.imread(str(PREVIOUS/date/'source_rgb.png')),cv2.COLOR_BGR2RGB)
        new_rgb=cv2.cvtColor(cv2.imread(str(folder/'source_rgb.png')),cv2.COLOR_BGR2RGB)
        overlay=cv2.cvtColor(cv2.imread(str(folder/'source_built.png')),cv2.COLOR_BGR2RGB)
        uv=new['render_uv'];roof=uv[masks['roof']];lo=np.maximum(roof.min(0)-40,0);hi=np.minimum(roof.max(0)+41,2048)
        fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
        for col,(image,title) in enumerate([(old_rgb,'Previous centers and gap fill'),(new_rgb,'Gaussians: shape, opacity, SH'),(overlay,'New structure labels: orange')]):
            axes[0,col].imshow(image);axes[0,col].set_title(title,fontsize=10);axes[0,col].axis('off')
            factor=image.shape[0]/2048;a=np.floor(lo*factor).astype(int);b=np.ceil(hi*factor).astype(int)
            axes[1,col].imshow(image[a[1]:b[1],a[0]:b[0]]);axes[1,col].axis('off')
        fig.suptitle(f"{date} · control roof: {value('previous_fusion','roof')*100:.1f}% → {value('gaussian_render','roof')*100:.1f}%\nBottom row: the same region, selected only for evaluation")
        fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
        neg=uv[masks['vegetation']]
        if len(neg):
            lo=np.maximum(neg.min(0)-15,0);hi=np.minimum(neg.max(0)+16,2048)
            negative_ax.imshow(new_rgb[lo[1]:hi[1],lo[0]:hi[0]])
        negative_ax.set_title(date);negative_ax.axis('off')
    negative_fig.suptitle('3DGS RGB: fixed negative vegetation region, without model labels')
    negative_fig.savefig(OUT/'negative_region_review.png',dpi=150);plt.close(negative_fig)
    write_csv(OUT/'recognition.csv',rows);write_csv(OUT/'summary.csv',summary)
    result={'registration_rerun':False,'recognition_regions_sha256':file_hash(region_path),'dates':summary,
        'roof_detected_dates':[r['date'] for r in summary if r['new_roof_fraction']>0],
        'unchanged_strict_pass_dates':[r['date'] for r in summary if r['previous_strict_pass']],
        'max_vegetation_false_built_fraction':max(r['new_vegetation_fraction'] for r in summary if r['new_vegetation_fraction'] is not None)}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True);x=np.arange(7)
    for ax,region,title in zip(axes,['roof','vegetation'],['3DGS control roof','False structures: 3DGS vegetation']):
        for shift,method,label,color in [(-.18,'previous_fusion','Previous projection','#78879c'),(.18,'gaussian_render','Gaussian projection','#21876f')]:
            values=[next(r['built_fraction'] for r in rows if r['date']==d and r['method']==method and r['region']==region) for d in DATES]
            ax.bar(x+shift,np.array(values)*100,.36,label=label,color=color)
        ax.set(xticks=x,xticklabels=DATES,ylim=(0,105),ylabel='% of visible points',title=title);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(OUT/'recognition.png',dpi=150);plt.close(fig)
    table=['| Date | Roof points | Roof: before → after | False buildings in vegetation | Pairs with sufficient overlap | Strict check of previous pose |','|---|---:|---:|---:|---:|---|']
    for r in summary:table.append(f"| {r['date']} | {r['roof_count']} | {r['previous_roof_fraction']*100:.1f}% → {r['new_roof_fraction']*100:.1f}% | {r['new_vegetation_fraction']*100:.1f}% | {r['strict_overlap_pairs']} | {'passed' if r['previous_strict_pass'] else 'failed'} |")
    failure_0708='0708: three pairs meet the overlap gates, but their centers are almost collinear. The second singular value of centered source coordinates is 1.962274 m against a 2 m minimum; LAS gives 4.353667 m. Minimum spacing is about 59 m, so the issue is transverse spread. The 3.77 cm gap is in this support metric, not object-position error. The threshold is unchanged.'
    failure_0723='0723: only two pairs meet ≥80% forward and ≥50% reverse overlap within 1 m. The third spaced pair p8_c16 → p8_c39 gives 71.15% / 55.38%: 37 of 52 source points are within 1 m of the assigned fragment, while at least 42 are needed. Five confirmed points are missing; this does not justify adding artificial points.'
    note='This experiment improves structure recognition in 3DGS. Strict statuses refer to the previous semantic registration: its transforms and geometric supports are preserved. Improved recognition alone does not prove better alignment.'
    lines=['# 3DGS: structure recognition and strict failures','',note,'',
        f"The control roof received structure labels on {len(result['roof_detected_dates'])}/7 dates instead of 0/7. This is a fraction of selected visible points in one volume, not IoU or whole-model accuracy.",'',*table,'','![Comparison](recognition.png)','',
        '## What changed','',
        'Sparse centers with color-filled gaps are replaced by orthographic Gaussian rendering: covariance is projected from scale and quaternion rotation, opacity is composited back to front, and color includes spherical harmonics up to degree 3. Resolution increases from 1024 to 2048 while preserving image boundaries. This is an orthographic view, not an exact rendering from the original photo camera.','',
        'LoveDA B2, normalization, 512-pixel tiles with four rotations, the 0.6 building threshold, and protection of confident drone vegetation/water/moving-object classes remain unchanged. New labels are assigned to exactly the previous visible 3D points; filled pixels create no geometry. LAS labels come from the preceding experiment.','',
        'Rendering uses only even original PLY records within the previous training bounds and outside spatial holdout tiles of 0.4 source units. Transparent Gaussians with alpha >1/255 are retained for context. Their footprints may cross excluded-tile boundaries, but control-center records are not read by the renderer. Final poses, BIM, and diagnostic volumes do not define the projection or prediction.','',
        '**Evaluation boundary:** resolution was selected after exploratory roof trials on 0723 and 0929; this is development on a known site. The same configuration was then applied to all seven dates. Roof results are not an independent test. Rendering method and resolution changed together, so their separate contributions are not measured. The previous automatic matrix is used only to locate diagnostic volumes.','',
        '[Negative-region RGB across seven dates](negative_region_review.png). The vegetation strip checks one type of false positive. It does not evaluate cranes, equipment, ground, or the whole site; the model has no separate crane class.','',
        '## Why the strict check fails','',failure_0708,'',failure_0723,'',
        'All triples of eligible pairs were checked on both dates; none passes. An additional all-fragment match at the final pose also found none. That check is diagnostic only and does not decide acceptance.','',
        'All points in the third 0723 pair are within 1 m of the full LAS, but not all are near the assigned fragment. This indicates differences in fragment extent or identity: closeness to any LAS neighbor does not establish object correspondence. On 0708, pair p9_c36 → p6_c2 has only 44.52% reverse overlap; 87.2% of reverse misses fall in source holdout tiles. Removing these misses to gain acceptance is not valid.','',
        'The next justified step is to confirm a persistent support away from the near-straight line on 0708 and extract complete corresponding parts of the same object on 0723. Structure identities must be testable and fixed before fitting; pairs were not substituted after seeing results and thresholds were not relaxed.','',
        '![0708](../strict_diagnostics/0708_diagnostic.png)','![0723](../strict_diagnostics/0723_diagnostic.png)','',
        '## Files and reproduction','',
        '`summary.csv` compares seven dates; `recognition.csv` records control-region counts; `<date>/source_labels.npz` stores new source-point labels and separate LoveDA probabilities; `render_context.npz` stores original context indices; `segmentation_protocol.json` records parameters and hashes; `../strict_diagnostics/` stores pairs, triples, and thresholds. probabilities remains the previous drone-model distribution, building_expert_probabilities stores the separate LoveDA prediction, and labels/confidence/label_origin describe fusion.','',
        '```sh','.venv/bin/python scripts/prepare_semantic_3dgs.py','.venv/bin/python scripts/diagnose_strict_support.py','.venv/bin/python scripts/report_semantic_3dgs.py',
        ".venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v > results/semantic_3dgs/tests.log 2>&1",
        '.venv/bin/python scripts/verify_semantic_3dgs.py','```','',
        'The existing packages and weights are used. A clang++ compiler is required for the local renderer. Previous results and raw assets are preserved. validation.json records the checks. See the [previous registration comparison](../semantic_fusion/REPORT.md).','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>3DGS: structures and strict checks</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:90ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}details{border-bottom:1px solid #ccd2d8}a:focus-visible,summary:focus-visible{outline:2px solid #175794}h1{font-size:clamp(24px,3vw,36px)}</style>',
        '<a href="/viewer/">← 3D viewer</a><h1>3DGS: structures and strict checks</h1>',
        f'<p>The control roof is detected on <strong>{len(result["roof_detected_dates"])}/7 dates</strong> instead of 0/7. The new projection accounts for Gaussian shape and opacity. The model and confidence threshold are unchanged.</p>',
        '<p>'+note+'</p>',
        '<p><a href="REPORT.md">Method and limitations</a> · <a href="summary.csv">Comparison CSV</a> · <a href="recognition.csv">Point counts</a> · <a href="../semantic_fusion/inspection.html">Previous registration comparison</a></p>',
        '<div class="table"><table><thead><tr><th>Date</th><th>Roof points</th><th>Detected: before → after</th><th>False buildings in vegetation</th><th>Strict check of previous pose</th></tr></thead><tbody>']
    for r in summary:page.append(f'<tr><td>{r["date"]}</td><td>{r["roof_count"]}</td><td>{r["previous_roof_fraction"]*100:.1f}% → {r["new_roof_fraction"]*100:.1f}%</td><td>{r["new_vegetation_fraction"]*100:.1f}%</td><td>{"Passed" if r["previous_strict_pass"] else "Failed"}</td></tr>')
    page+=['</tbody></table></div><p>Fractions refer to one roof and one vegetation strip. Settings were developed on this site; this is not an independent whole-scene evaluation.</p>',
        '<img src="recognition.png" alt="Control-roof recognition and false structures in vegetation across seven dates">',
        '<h2>Reasons for strict failures</h2><p>'+failure_0708+'</p><p>'+failure_0723+'</p>',
        '<p>All triples were checked; none qualifies. Thresholds are unchanged. Previous-method primary acceptance remains 7/7 and strict acceptance 5/7.</p>']
    for d in ['0708','0723']:page.append(f'<details><summary>{d}: supports, overlap, and full LAS</summary><img loading="lazy" src="../strict_diagnostics/{d}_diagnostic.png" alt="{d}: strict-failure diagnostics"></details>')
    page.append('<h2>Images across all seven dates</h2><p>Top: full scene. Bottom: control roof. Orange points show final structure labels.</p>')
    for d in DATES:page.append(f'<details><summary>{d}: previous projection → Gaussians → detected structures</summary><a href="{d}/comparison.png"><img loading="lazy" src="{d}/comparison.png" alt="{d}: projection and structure-label comparison"></a></details>')
    page.append('<details><summary>Negative vegetation-region check</summary><img loading="lazy" src="negative_region_review.png" alt="Fixed vegetation strip in 3DGS across seven dates"></details>')
    (OUT/'inspection.html').write_text('\n'.join(page)+'</html>\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
