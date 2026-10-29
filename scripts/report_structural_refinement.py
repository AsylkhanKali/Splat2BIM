"""Report actual control accuracy, supervision and tradeoffs for seven dates."""
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from fit_structural_refinement import OUT as FIRST
from fit_guarded_structural_refinement import OUT as GUARDED
from fit_reviewed_structural_refinement import OUT,BASE
from check_las_vertical import write_csv,file_hash

METHODS=['current_automatic','stable_parent','structural_refinement']


def read_csv(path):return list(csv.DictReader(path.open()))


def main():
    statics=[];global_rows=[];auto_controls=[];summaries=[];support=[];trials=[]
    for date in DATES:
        folder=OUT/date;fit=json.loads((folder/'fit.json').read_text());evaluation=json.loads((folder/'evaluation.json').read_text())
        statics+=read_csv(folder/'static_structures.csv');global_rows+=read_csv(folder/'comparison.csv');auto_controls+=read_csv(folder/'structural_controls.csv')
        row={'date':date,'status':fit['status'],'selected_candidate':fit['selected_candidate'],
            'strict_support_count':evaluation['selected_evidence']['strict_forward_080_spatial_support_count']}
        for method in METHODS:
            values=[];tails=[]
            for region in ['roof','bank_mid']:
                metric=next(r for r in statics if r['date']==date and r['method']==method and r['region']==region and r['scope']=='unseen_records')
                assert int(metric['count'])>=30;value=float(metric['median']);tail=float(metric['p90'])
                row[method+'_'+region+'_median_m']=value;row[method+'_'+region+'_p90_m']=tail
                row[region+'_control_records']=int(metric['count']);values.append(value);tails.append(tail)
            row[method+'_mean_region_median_m']=float(np.mean(values));row[method+'_mean_region_p90_m']=float(np.mean(tails))
            metric=next(r for r in global_rows if r['date']==date and r['method']==method and r['scope']=='fixed_baseline_coverage')
            for key in ['median','p90']:row[method+'_global_'+key+'_m']=float(metric[key])
        row['mean_region_median_improvement_m']=row['stable_parent_mean_region_median_m']-row['structural_refinement_mean_region_median_m']
        row['mean_region_median_improvement_percent']=100*row['mean_region_median_improvement_m']/row['stable_parent_mean_region_median_m']
        p=np.load(folder/'held_out_distances.npz')['source_points'];old=apply(p,np.array(fit['initial_matrix']));new=apply(p,np.array(fit['matrix_local']))
        row['pose_motion_p95_m']=float(np.quantile(np.linalg.norm(new-old,axis=1),.95));summaries.append(row)
        for pair in evaluation['selected_evidence']['strict_spatial_support']:
            support.append({'date':date,'source_id':pair['source_id'],'target_id':pair['target_id'],
                'forward_fraction':pair['fraction_within_1m'],'reverse_fraction':pair['reverse_fraction_within_1m']})
        for root,name in [(FIRST,'automatic_structures_only'),(GUARDED,'automatic_context_guarded'),(OUT,'reviewed_regions')]:
            saved=json.loads((root/date/'fit.json').read_text());stats=read_csv(root/date/'static_structures.csv')
            actual=[r for r in stats if r['method']=='structural_refinement' and r['scope']=='unseen_records']
            trials.append({'date':date,'experiment':name,'status':saved['status'],
                'mean_control_region_median_m':float(np.mean([float(r['median']) for r in actual])),
                'strict_support_count':json.loads((root/date/'evaluation.json').read_text())['selected_evidence']['strict_forward_080_spatial_support_count']})
    for filename,rows in [('summary.csv',summaries),('static_structures.csv',statics),('comparison.csv',global_rows),
        ('automatic_structural_controls.csv',auto_controls),('strict_support.csv',support),('experiments.csv',trials)]:write_csv(OUT/filename,rows)
    improved=[r['date'] for r in summaries if r['mean_region_median_improvement_m']>.001]
    regressed=[r['date'] for r in summaries if r['mean_region_median_improvement_m']<-.001]
    unchanged=[r['date'] for r in summaries if abs(r['mean_region_median_improvement_m'])<=.001]
    before=float(np.mean([r['stable_parent_mean_region_median_m'] for r in summaries]));after=float(np.mean([r['structural_refinement_mean_region_median_m'] for r in summaries]))
    tail_regressions=[r['date'] for r in summaries if r['structural_refinement_global_p90_m']>r['stable_parent_global_p90_m']+.01]
    global_median_increases=[r['structural_refinement_global_median_m']-r['stable_parent_global_median_m'] for r in summaries if r['status']=='refined']
    regional_regressions=[{'date':r['date'],'region':k,'increase_m':r['structural_refinement_'+k+'_median_m']-r['stable_parent_'+k+'_median_m']}
        for r in summaries for k in ['roof','bank_mid'] if r['structural_refinement_'+k+'_median_m']>r['stable_parent_'+k+'_median_m']+.001]
    spatial_roof=[]
    for date in DATES:
        before_row=next(r for r in statics if r['date']==date and r['method']=='stable_parent' and r['region']=='roof' and r['scope']=='spatial_holdout')
        after_row=next(r for r in statics if r['date']==date and r['method']=='structural_refinement' and r['region']=='roof' and r['scope']=='spatial_holdout')
        spatial_roof.append({'date':date,'count':int(before_row['count']),'adequate':int(before_row['count'])>=30,
            'before_m':float(before_row['median']),'after_m':float(after_row['median'])})
    summary={'refined_dates':[r['date'] for r in summaries if r['status']=='refined'],'retained_dates':[r['date'] for r in summaries if r['status']=='baseline_retained'],
        'improved_over_1mm':improved,'regressed_over_1mm':regressed,'unchanged_within_1mm':unchanged,
        'mean_of_date_region_medians_before_m':before,'mean_of_date_region_medians_after_m':after,'relative_improvement_percent':100*(before-after)/before,
        'global_p90_regressions_over_1cm':tail_regressions,'individual_region_median_regressions_over_1mm':regional_regressions,
        'global_median_increase_range_m':[min(global_median_increases),max(global_median_increases)],
        'spatial_roof_checks':spatial_roof,
        'strict_pass_dates':[r['date'] for r in summaries if r['strict_support_count']>=3],
        'fit_sha256':{d:file_hash(OUT/d/'fit.json') for d in DATES},'evaluation_sha256':{d:file_hash(OUT/d/'evaluation.json') for d in DATES},
        'source_sha256':{'report_structural_refinement.py':file_hash(ROOT/'scripts/report_structural_refinement.py')}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True);x=np.arange(7)
    for ax,key,title in zip(axes.flat,['roof_median_m','bank_mid_median_m','mean_region_median_m','global_p90_m'],
        ['Roof: median on unused records','Riverside: median on unused records','Mean of two medians: equal region weights','Whole scene: P90 on spatial holdout']):
        ax.bar(x-.18,[r['stable_parent_'+key] for r in summaries],.34,label='Before refinement',color='#748397')
        ax.bar(x+.18,[r['structural_refinement_'+key] for r in summaries],.34,label='After refinement',color='#208473')
        ax.set(xticks=x,xticklabels=DATES,title=title,ylabel='Metres; lower is better');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    headline=f'Mean across seven dates and two regions: {before:.3f} → {after:.3f} m ({100*(before-after)/before:.1f}% reduction).'
    table=['| Date | Roof: before → after, m | Riverside: before → after, m | Mean median: before → after, m | Change | Strict supports |',
        '|---|---:|---:|---:|---|---:|'];htmlrows=[]
    for r in summaries:
        values=[r['date']]+[f'{r["stable_parent_"+k]:.3f} → {r["structural_refinement_"+k]:.3f}' for k in ['roof_median_m','bank_mid_median_m','mean_region_median_m']]
        values+=[f'{r["mean_region_median_improvement_percent"]:+.1f}% reduction' if r['status']=='refined' else 'Previous pose retained',str(r['strict_support_count'])]
        table.append('| '+' | '.join(values)+' |');htmlrows.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
    limits=[
        'This refinement uses the existing manual definitions of two fixed regions: the roof and riverside structure. It is not a new fully automatic structure detector.',
        'Fitting uses only even original 3DGS records outside the previous spatial holdout tiles. A separate block-based calibration sample within the training data selects the correction. The headline errors use the unchanged records 5, 15, 25…; these participate in neither fitting nor selection.',
        'Region definitions now provide training supervision. Evaluation on different records within the same regions is therefore weaker than evaluation on entirely new structures. Reconstructed Gaussian centers are also not independent survey measurements.',
        'The stronger spatial holdout is retained separately. Roof samples are sufficient only on 0331, 0504, 0609, 0723, and 0831; riverside samples contain fewer than 30 points on every date. Insufficient samples do not count as evidence of improvement.',
        'Scale is fixed. Plane, point, and hybrid fits are tested, followed by the same correction fractions. Selection uses only training and internal block-calibration samples. Whole-scene motion is limited to 0.5 m at fixed bounding-volume anchors.',
        'The previous strict conditions are preserved: ≥3 distinct supports; ≥80% forward and ≥50% reverse overlap within 1 m; support spacing ≥10 m, second axis ≥2 m; center distance ≤5 m, normals ≤20°, size ratio 0.5–2. The pair bank is fixed before refinement; no identities are added after fitting.',
        'The first automatic experiment improved selected supports but worsened the riverside region, by about 9 cm on 0504. The scene-context variant with a no-regression guard on training median/P90 retained the previous poses for all seven dates. Both experiments are preserved; experiments.csv compares them with reviewed-region refinement.',
        'The method was developed on this site with knowledge of earlier control results. This is an experimental comparison, not an independent transfer test. The complete working scenes and BIM in the viewer were not overwritten; new local and global matrices are stored in the date folders.'
    ]
    tradeoff=f'Whole-scene median increased on changed dates by {1000*min(global_median_increases):.1f}–{1000*max(global_median_increases):.1f} mm. Dates with a whole-scene P90 increase above 1 cm: '+(', '.join(tail_regressions) or 'no')+'. New matrices are therefore saved separately and do not automatically replace the working exports.'
    regressions='Regions with a median increase above 1 mm: '+('; '.join(f'{r["date"]} {r["region"]}: +{1000*r["increase_m"]:.1f} mm' for r in regional_regressions) or 'no')+'.'
    spatial_text='On the stronger spatial roof holdout, the median decreased on 0331, 0504, 0723, and 0831; it increased by 3.3 mm on 0609. The sample is insufficient on 0708 and 0929. Spatial riverside control samples are insufficient on every date.'
    lines=['# Accuracy refinement with reviewed stable regions','',headline,'',
        f'Strict acceptance: {len(summary["strict_pass_dates"])}/7. Accepted corrections: '+', '.join(summary['refined_dates'])+'. Previous pose: '+(', '.join(summary['retained_dates']) or 'no')+'.','',*table,'',tradeoff,'',regressions,'',
        '![Comparison](comparison.png)','',spatial_text,'', '## Conditions and limitations','',*sum(([p,''] for p in limits),[]),
        '## Whole-scene evaluation','', '| Date | Median before → after, m | P90 before → after, m |','|---|---:|---:|']
    for r in summaries:lines.append('| '+r['date']+' | '+' | '.join(f'{r["stable_parent_global_"+k+"_m"]:.3f} → {r["structural_refinement_global_"+k+"_m"]:.3f}' for k in ['median','p90'])+' |')
    lines+=['','summary.csv compares the original automatic method and the latest persistent-surface method. static_structures.csv contains errors for both control-sample types; automatic_structural_controls.csv covers additional automatic surfaces. All candidates, including rejected ones, are stored in <date>/fit.json and candidate_comparison.csv.','',
        '## Reproduction','', '```sh','.venv/bin/python scripts/fit_structural_refinement.py','.venv/bin/python scripts/fit_guarded_structural_refinement.py',
        '.venv/bin/python scripts/fit_reviewed_structural_refinement.py',
        ".venv/bin/python -c \"import sys;sys.path.insert(0,'scripts');from evaluate_structural_refinement import main;from fit_reviewed_structural_refinement import OUT;main(OUT)\"",
        '.venv/bin/python scripts/verify_structural_refinement.py','.venv/bin/python scripts/report_structural_refinement.py','```','',
        'Numerical re-verification is recorded in validation.json. The first two experiments have separate results/structural_refinement and results/guarded_structural_refinement directories. All original inputs and previous matrices are preserved.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stable-structure alignment accuracy</title>',
        '<style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;max-width:1300px;margin:24px auto;padding:0 20px}p{max-width:105ch}img{width:100%;height:auto}a{color:#175794}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:9px;text-align:left;border-bottom:1px solid #ccd2d8}.table{overflow:auto}summary{cursor:pointer;font-weight:650;padding:14px 0}h1{font-size:clamp(24px,3vw,36px)}a:focus-visible,summary:focus-visible{outline:2px solid #175794}</style>',
        '<a href="/viewer/">← 3D viewer</a><h1>Stable-structure alignment accuracy</h1>',
        '<p><strong>'+headline+'</strong> Strict thresholds are unchanged; acceptance '+str(len(summary['strict_pass_dates']))+'/7.</p>',
        '<p>Refinement uses predefined roof and riverside regions. No control record participates in fitting or correction selection.</p>',
        '<p><a href="REPORT.md">Method</a> · <a href="summary.csv">Seven-date comparison</a> · <a href="static_structures.csv">Control samples</a> · <a href="experiments.csv">All three experiments</a> · <a href="validation.json">Numerical verification</a></p>',
        '<div class="table"><table><thead><tr><th>Date</th><th>Roof, m</th><th>Riverside, m</th><th>Mean median, m</th><th>Result</th><th>Strict supports</th></tr></thead><tbody>',*htmlrows,
        '</tbody></table></div><p>Before → after. Equal weight for both regions and every date; errors use the unchanged unused records.</p>',
        '<p>'+tradeoff+' '+regressions+'</p><p>'+spatial_text+'</p><img src="comparison.png" alt="Changes in roof, riverside, and whole-scene errors across seven dates">',
        '<details><summary>Inputs and evaluation independence</summary>',*['<p>'+p+'</p>' for p in limits],'</details></html>']
    (OUT/'inspection.html').write_text('\n'.join(page)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
