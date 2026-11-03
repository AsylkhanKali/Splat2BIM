"""Publish fixed-cohort results, paired block uncertainty, matrices and two pipelines."""
import json,html,argparse
from pathlib import Path
import numpy as np
import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_data import ROOT,ORIGIN,file_hash,write_csv
from registration_ablation import measures,apply,matrix_parts

MAIN=['pca_only','pca_icp','pca_semantic_icp']
LABELS={'pca_only':'PCA only','pca_icp':'PCA + ICP','pca_semantic_icp':'PCA + semantics + ICP'}

def diagram():
    boxes=[(20,60,330,90,'INPUT: 0929 Gaussian PLY','Centers, covariance, opacity, SH'),(440,60,330,90,'INPUT: 0929 LAS','Reference XYZ in local metres'),
      (20,200,330,90,'Frozen spatial split','Training records / held-out records'),(440,200,330,90,'PCA hypotheses','Four proper rotation/scale starts'),
      (840,60,330,90,'Semantic side branch','Rendered/projection images → SegFormer'),(840,200,330,90,'Labels mapped to 3D points','Class-balanced scoring; no pose change'),
      (440,350,330,100,'Three controlled branches','PCA only / PCA + ICP / semantic + ICP'),(20,510,330,90,'OUTPUT: transformation','Local/global 4×4 matrix, scale, R, t'),
      (440,510,330,90,'OUTPUT: held-out evaluation','Median, P90, RMSE, thresholds, ablations'),(840,510,330,90,'Separate BIM + baseline viewer','Historical export + IFC; no BIM fitting'),
      (20,730,330,90,'INPUT: calibrated query photo','Reference photos + reference poses + LAS'),(440,730,330,90,'SIFT/RootSIFT → 2D–3D pairs','EPnP-RANSAC → LM refinement'),
      (840,730,330,90,'OUTPUT: query camera pose','Held-out reprojection checks / BIM overlay')]
    s=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 860" role="img" aria-labelledby="title"><title id="title">Registration and camera localization inputs and outputs</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#527182"/></marker></defs><rect width="1200" height="860" fill="#f5f7fa"/><g font-family="Arial,sans-serif" fill="#172b39"><text x="20" y="30" font-size="22" font-weight="bold">A. Controlled 0929 registration experiment</text><text x="20" y="690" font-size="22" font-weight="bold">B. Photo-to-camera pose — separate experiment</text>']
    for x,y,w,h,title,body in boxes:
        s.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="white" stroke="#8da5b3"/><text x="{x+14}" y="{y+30}" font-size="17" font-weight="bold">{html.escape(title)}</text><text x="{x+14}" y="{y+60}" font-size="14">{html.escape(body)}</text>')
        if (x,y)==(440,350):s.append('<text x="454" y="433" font-size="13">Up to 500 steps / gate; stop reasons recorded</text>')
    for points in ['185,150 185,200','350,245 440,245','605,150 605,200','350,105 400,105 400,175 1005,175 1005,150','770,105 840,105','1005,150 1005,200','605,290 605,350','1005,290 1005,400 770,400','440,400 185,400 185,510','605,450 605,510','350,775 440,775','770,775 840,775']:
        s.append(f'<polyline points="{points}" fill="none" stroke="#527182" stroke-width="2" marker-end="url(#arrow)"/>')
    s.append('<text x="20" y="635" font-size="15">Held-out records are used only for evaluation. BIM is not an alignment target. Segmentation supplies labels, not a matrix.</text></g></svg>')
    return ''.join(s)


def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'results/ablation');args=parser.parse_args();out=args.out
    protocol=json.loads((out/'protocol.json').read_text());results=json.loads((out/'results.json').read_text());counts=json.loads((out/'sample_counts.json').read_text());evaluation=np.load(out/'evaluation.npz')
    comparison=json.loads((out/'reference_comparison.json').read_text());checks=json.loads((out/'convergence_checks.json').read_text())
    baseline=comparison['baseline_metrics_same_cohort']
    rows=[];aggregate=[]
    for config in protocol['configurations']:
        name=config['name'];runs=[r for r in results if r['name']==name];valid=[r for r in runs if r['metrics'] is not None]
        if not valid:rows.append([name,'0 / 3','Rejected','—','—','—','—','—']);continue
        med=[r['metrics']['median_m'] for r in valid];p90=[r['metrics']['p90_m'] for r in valid];within=[r['metrics']['within_1m'] for r in valid];rmse=[r['metrics']['rmse_m'] for r in valid]
        converged=sum(r['convergence']['terminal_converged'] is True for r in valid)
        limited=sum(r['convergence']['terminal_converged'] is False for r in valid)
        aggregate.append({'name':name,'evaluated':len(valid),'median_mean_m':float(np.mean(med)),'median_min_m':min(med),'median_max_m':max(med),'p90_mean_m':float(np.mean(p90)),
                          'rmse_mean_m':float(np.mean(rmse)),'within_1m_mean':float(np.mean(within)),'terminal_converged':converged,'iteration_limited':limited,'max_iterations':config['iterations'] if config['icp'] else 0})
        stop=f'{converged}/{len(valid)} converged' if config['icp'] else 'Not applicable'
        rows.append([html.escape(LABELS.get(name,name)),f'{len(valid)} / {len(runs)}',f'{np.mean(med):.3f} [{min(med):.3f}, {max(med):.3f}]',f'{np.mean(p90):.3f}',f'{np.mean(rmse):.3f}',f'{np.mean(within):.1%}',stop,', '.join(str(r.get('selected_index','rejected')) for r in runs)])
    write_csv(out/'aggregate.csv',aggregate)
    budget_names=['iteration_budget_030','iteration_budget_200','pca_icp'];budget_rows=[];budget_csv=[]
    for name in budget_names:
        a=next(a for a in aggregate if a['name']==name);rr=[r for r in comparison['runs'] if r['name']==name]
        displacement=float(np.mean([r['median_displacement_from_baseline_m'] for r in rr]))
        budget_rows.append([a['max_iterations'],f'{a["median_mean_m"]:.3f}',f'{a["p90_mean_m"]:.3f}',f'{a["rmse_mean_m"]:.3f}',f'{a["terminal_converged"]}/{a["evaluated"]}',f'{displacement:.3f}'])
        budget_csv.append({**a,'mean_median_displacement_from_baseline_m':displacement})
    write_csv(out/'iteration_comparison.csv',budget_csv)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for seed in protocol['seeds']:
        rr=[next(r for r in comparison['runs'] if r['name']==name and r['seed']==seed) for name in budget_names]
        axes[0].plot([r['max_iterations'] for r in rr],[r['median_m'] for r in rr],'-o',label=f'Seed {seed}')
        axes[1].plot([r['max_iterations'] for r in rr],[r['median_displacement_from_baseline_m'] for r in rr],'-o',label=f'Seed {seed}')
    axes[0].axhline(baseline['median_m'],linestyle='--',color='#555',label='Historical transform, same cohort')
    axes[0].set(ylabel='Median nearest-LAS distance (m)',title='Effect of the ICP iteration limit')
    axes[1].set(ylabel='Median same-center displacement (m)',title='Difference from historical transform')
    for ax in axes:ax.set(xlabel='Maximum iterations per gate',xticks=[30,200,500]);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'iteration_comparison.png',dpi=160);plt.close(fig)
    main_runs={r['name']:r for r in results if r['seed']==29 and r['name'] in MAIN and r['metrics'] is not None}
    error={n:np.load(out/n/'29/errors.npz')['distance_m'] for n in main_runs}
    block=np.floor(evaluation['source']/protocol['bootstrap']['block_width_source_units']).astype(int);_,group=np.unique(block,axis=0,return_inverse=True);groups=[np.flatnonzero(group==g) for g in np.unique(group)]
    block_scores={n:np.array([np.median(e[g]) for g in groups]) for n,e in error.items()}
    draws=np.random.default_rng(2026).integers(0,len(groups),size=(1000,len(groups)));uncertainty=[]
    for a,b in [('pca_only','pca_icp'),('pca_icp','pca_semantic_icp')]:
        if a not in error or b not in error:continue
        delta=block_scores[b]-block_scores[a];lo,hi=np.quantile(delta[draws].mean(1),[.025,.975])
        uncertainty.append({'comparison':b+' minus '+a,'blocks':len(groups),'mean_block_median_change_m':float(delta.mean()),'ci_low_m':float(lo),'ci_high_m':float(hi)})
    write_csv(out/'paired_block_differences.csv',uncertainty)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    x=np.arange(len(main_runs));names=list(main_runs)
    axes[0].bar(x-.18,[main_runs[n]['metrics']['median_m'] for n in names],.36,label='Median',color='#247b93');axes[0].bar(x+.18,[main_runs[n]['metrics']['p90_m'] for n in names],.36,label='P90',color='#c08036')
    axes[0].set(xticks=x,xticklabels=['PCA','PCA + ICP','PCA + semantic + ICP'][:len(x)],ylabel='Distance to LAS (m)',title='Same held-out cohort · seed 29');axes[0].legend()
    for n in names:
        e=np.sort(error[n]);axes[1].plot(np.maximum(e,1e-6),np.arange(1,len(e)+1)/len(e),label=LABELS[n])
    axes[1].set(xscale='log',xlabel='Distance to LAS (m, logarithmic)',ylabel='Cumulative fraction',title='Full distance distributions');axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    las=laspy.read(ROOT/'assets/0929/0929.las');target=np.c_[las.x,las.y,las.z]-ORIGIN
    target=target[::max(1,len(target)//25000)];shown=evaluation['source'][::max(1,len(evaluation['source'])//5000)]
    fig,axes=plt.subplots(1,len(main_runs),figsize=(15,5),constrained_layout=True)
    for ax,(name,r) in zip(np.atleast_1d(axes),main_runs.items()):
        moved=apply(shown,np.array(r['matrix']))
        ax.scatter(target[:,0],target[:,1],s=1,c='#8b9ba4',alpha=.5,label='LAS context')
        ax.scatter(moved[:,0],moved[:,1],s=1,c='#cb6539',alpha=.5,label='Held-out 3DGS centers')
        lo,hi=np.quantile(target[:,:2],[.001,.999],axis=0)
        ax.set(title=LABELS[name],xlabel='Local X, m',ylabel='Local Y, m',xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),aspect='equal');ax.legend(markerscale=5,fontsize=8)
    fig.savefig(out/'alignment_views.png',dpi=150);plt.close(fig)
    (out/'pipeline.svg').write_text(diagram())
    main_ag={a['name']:a for a in aggregate if a['name'] in MAIN}
    conclusion='';a=main_ag.get('pca_only');b=main_ag.get('pca_icp');c=main_ag.get('pca_semantic_icp')
    if a and b:conclusion+=f"Across three sampling seeds, mean held-out median changes from {a['median_mean_m']:.3f} m after PCA to {b['median_mean_m']:.3f} m after ICP. "
    if b and c:conclusion+=f"Adding semantic ranking changes that value by {c['median_mean_m']-b['median_mean_m']:+.6f} m. "
    if all(n in main_runs for n in ['pca_icp','pca_semantic_icp']):
        same=all(np.allclose(a['matrix'],next(b['matrix'] for b in results if b['name']=='pca_semantic_icp' and b['seed']==a['seed']),rtol=0,atol=1e-12) for a in results if a['name']=='pca_icp' and a.get('matrix'))
        if same:conclusion+='The geometric and semantic primary branches selected identical final matrices on all three seeds: semantic ranking did not change the selected alignment in this experiment.'
    old=next(a for a in aggregate if a['name']=='iteration_budget_030')
    correction=f'The earlier 30-step runs were iteration-limited: their mean median was {old["median_mean_m"]:.3f} m. With up to 500 steps per gate and unchanged stopping tolerances, the geometric primary branch reaches {b["median_mean_m"]:.3f} m; {b["terminal_converged"]}/{b["evaluated"]} runs reach the terminal update tolerance. This correction concerns optimization completion, not a relaxed evaluation threshold.'
    if b['iteration_limited']:correction+=f' {b["iteration_limited"]} of {b["evaluated"]} primary runs remain iteration-limited and explicitly flagged. A larger cap alone does not guarantee convergence.'
    headers=['Method / setting','Evaluated','Median mean [range], m','P90, m','RMSE, m','Within 1 m','Terminal stop','PCA IDs']
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Splat2BIM · 0929 controlled ablation</title>',
      '<style>*{box-sizing:border-box}body{margin:32px auto;padding:0 22px;max-width:1160px;font:16px/1.65 system-ui,sans-serif;color:#192d3a;background:#fff}h1{font-size:clamp(28px,4vw,42px);line-height:1.2}h2{margin-top:48px}a{color:#146e8a}p,li{max-width:95ch}.lead{font-size:19px}.note{padding:16px 20px;background:#f2f6f8;border-left:4px solid #247b93}.table{overflow:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;text-align:left;border-bottom:1px solid #d5dfe5;vertical-align:top}th{background:#f2f6f8}img{max-width:100%;height:auto}pre{overflow:auto;padding:16px;background:#f2f6f8;font-size:13px}details{margin:16px 0;border:1px solid #d5dfe5;padding:12px}summary{cursor:pointer;font-weight:650}a:focus-visible,summary:focus-visible{outline:2px solid #146e8a}figure{margin:24px 0}figcaption{font-size:14px;color:#475c69}</style>',
      '<a href="/viewer/">← 3D viewer</a><h1>0929: what each registration step contributes</h1><p class="lead">A controlled comparison of PCA only, PCA + ICP, and PCA + semantic ranking + ICP.</p>',
      '<p><a href="protocol.json">Frozen protocol</a> · <a href="summary.csv">Every run (CSV)</a> · <a href="results.json">Matrices, stages and candidate traces (JSON)</a> · <a href="pipeline.svg">Pipeline diagram</a> · <a href="../camera_localization/inspection.html">Photo-to-camera report</a></p>',
      '<div class="note"><strong>Correction:</strong> '+html.escape(correction)+' <a href="../ablation_30_iterations/inspection.html">Archived 30-step report</a>.</div>',
      f'<p><strong>Scope:</strong> one previously explored capture, {len(protocol["configurations"])} settings × {len(protocol["seeds"])} sampling seeds. This measures source-to-LAS reference agreement, not independent survey accuracy, segmentation IoU, or generalization to a new site. Failed and iteration-limited candidates are retained.</p>',
      '<h2>1. Inputs, outputs and two separate pipelines</h2><img src="pipeline.svg" alt="Registration and photo-to-camera-pose pipeline with inputs and outputs">',
      '<h2>2. Controlled protocol</h2>',
      f'<p>The same {counts["heldout_records"]:,} finite, positive-opacity Gaussian centers are evaluated in every run. Source records are sampled every tenth record; tiles use source coordinates and a width of 0.4 source units. Tiles whose integer-coordinate sum is divisible by four are withheld. The {counts["fit_records"]:,} fitting records are disjoint from these records. Quantile cropping applies only to fitting, not evaluation.</p>',
      '<p>PCA supplies exactly four proper sign hypotheses. The primary branches refine one ranked hypothesis each. The four-start ablations increase the budget for both geometric and semantic methods. No RTK, camera pose, manual seed, previously fitted transformation or historical correspondence bank enters these fits. Final candidate selection uses training scores only. All held-out distances use the entire same-date LAS reference; no candidate-dependent overlap mask or error trimming is applied during evaluation.</p>',
      '<p>Geometric ranking uses the 75th percentile of training distances. Semantic ranking adds 0.5 times the equal-weight mean of per-class 60th-percentile distances, capped at 20 m. Classes are structures, paved surfaces, ground and vegetation; at least two shared classes with 128 points each are required. Unknown, water and movable classes do not contribute. ICP itself remains geometric.</p>',
      '<p>The primary iteration limit is 500 per gate, with early stopping at the original tolerances: point-to-point RMS displacement below gate × 10⁻⁵ metres; point-to-plane bounded parameter-update norm below 10⁻⁶. The latter combines rotation, translation and log-scale increments and is not a metre error. Every stage records its final update, tolerance and stop reason. A converged terminal stage does not imply that every earlier coarse stage converged or that the pose is physically correct. Exact ties at the trimming quantile are retained to handle zero-error matches; the strict distance gates remain unchanged. The 30-step controls reproduce the archived matrices and metrics.</p>',
      '<p><strong>Historical context:</strong> these data and semantic/render settings have been explored before. Cached source label centers exclude held-out tiles, but Gaussian footprints can cross tile boundaries. This is a controlled development ablation, not a newly blinded test. The previous viewer export uses a separate RTK/COLMAP-assisted baseline and remains unchanged. Its old overlap-filtered metrics must not be compared directly with this full-cohort experiment.</p>',
      '<h2>3. Main result</h2><p>'+html.escape(conclusion)+'</p><p>Table values average each per-run metric across three seeds; brackets show the minimum and maximum per-run median. Terminal convergence is counted separately from geometric error.</p>',table(headers,rows[:3]),
      '<p>This experiment tests semantic ranking, not class-constrained ICP. When ranking chooses the same initial hypothesis, both branches supply identical geometric inputs to ICP, so the selected optimization is identical. These results do not establish whether using semantic classes to constrain individual correspondences would help.</p>',
      f'<p><strong>Same-cohort historical comparator:</strong> applying the preserved RTK/COLMAP-assisted transformation to these exact {counts["heldout_records"]:,} centers gives median {baseline["median_m"]:.3f} m, P90 {baseline["p90_m"]:.3f} m and RMSE {baseline["rmse_m"]:.3f} m. This transform is used only for comparison, never as an initialization or selection target. Its original 0.219 m metric used a different overlap-filtered cohort.</p>',
      '<p>RMSE is reported alongside the median: the fixed untrimmed cohort contains very large distances, and a lower median does not guarantee a lower RMSE. All evaluated centers, including the extreme tail, remain in the results.</p>',
      '<details><summary>Full error tail · seed 29</summary>',
      table(['Method','P95, m','P99, m','Maximum, m','Centers beyond 10 m'],[[LABELS[n],f'{np.quantile(e,.95):.3f}',f'{np.quantile(e,.99):.3f}',f'{np.max(e):.3f}',f'{np.sum(e>10):,} / {len(e):,}'] for n,e in error.items()]),'</details>',
      '<img src="comparison.png" alt="Main-method median and P90 comparison and cumulative distance curves">',
      '<figure><img src="alignment_views.png" alt="Top views of held-out 3DGS centers and LAS after each primary method"><figcaption>Seed 29, XY view. Points are subsampled for display; view bounds follow LAS extent. Off-screen points are still included in every numerical metric.</figcaption></figure><h2>4. After PCA, segmentation and ICP</h2><p>Values below use seed 29 and each branch’s selected starting hypothesis. Segmentation changes labels, not coordinates: the matrix and geometric distances immediately before and after label assignment are identical. Semantic scoring can select a different PCA hypothesis; its effect is tested by comparing branches. Point-to-point and point-to-plane ICP checkpoints are saved separately.</p>']
    stage_rows=[]
    for n,r in main_runs.items():
        for stage,m in r['stages'].items():stage_rows.append([LABELS[n],stage,f'{m["median_m"]:.3f}',f'{m["p90_m"]:.3f}',f'{m["rmse_m"]:.3f}',f'{m["within_1m"]:.1%}'])
    page.append(table(['Branch','Checkpoint','Median, m','P90, m','RMSE, m','Within 1 m'],stage_rows))
    page+=['<h2>5. Segmentation evidence</h2><p>The frozen labels combine a drone SegFormer model and a LoveDA building specialist. These images show RGB inputs and predicted labels. The <a href="../segmentation/reproduction.json">independent semantic-cache reproduction check</a> confirms that every regenerated array matches the frozen inputs exactly. There is no independently annotated semantic ground truth, so we do not report segmentation accuracy, IoU or precision/recall. Confidence sweeps filter existing fused labels; they cannot recover classes previously marked unknown.</p>']
    for side in ['source','target']:
        page.append(f'<details><summary>{side.title()}: RGB and semantic labels</summary><img src="../segmentation/{side}_rgb.png" alt="{side} RGB input"><img src="../segmentation/{side}_semantic.png" alt="{side} predicted semantic classes"></details>')
    page+=['<h2>6. Settings ablation and convergence</h2><p>All settings were listed in the frozen protocol before this follow-up run. A prior seed-29 diagnostic at 200 steps motivated the revision; this is not a blinded new experiment. Relative to the semantic primary branch, one factor changes at a time: semantic weight (0.25/0.5/1), minimum confidence (0.6/0.7/0.8), retained fitting-distance quantile (0.6/0.75/0.9), or fine target voxel size (0.3/0.6/1.2 m). Four-start geometry and semantic controls test budget effects. Shuffled source labels are a negative control; their class identities are permuted while geometry stays fixed. They do not remove every statistical property of the labels.</p>',table(headers,rows),
      '<h3>30 versus 200 versus 500 steps</h3><p>These geometric branches change only the iteration cap. Every distance gate, stopping tolerance, seed, fit sample and evaluation point stays fixed. Same-center displacement compares the positions of corresponding Gaussian centers under the new and historical transforms. It exposes sliding that nearest-neighbor distances can hide, but the historical transform is not surveyed ground truth.</p>',
      table(['Step limit','Mean median, m','Mean P90, m','Mean RMSE, m','Terminal convergence','Mean median shift from baseline, m'],budget_rows),
      '<img src="iteration_comparison.png" alt="Iteration budgets compared by nearest-LAS distance and same-center displacement from baseline">',
      '<p>Three seeds measure sensitivity to subsampling, not independent experimental replication. Identical geometric fits are cached within a seed across ranking-only variants; each branch is independently scored and evaluated. Reuse is marked in candidate JSON. Runtime excludes cached work and historical neural inference and is not an end-to-end speed benchmark. Every rejected or iteration-limited attempt remains in the JSON.</p>',
      '<details><summary>Terminal convergence: independent extra-update checks</summary><p>After saving each selected matrix, the verifier computes two additional fine-stage updates on a diagnostic copy using training data. Saved poses and metrics are unchanged. The first update probes numerical stationarity. A non-converged update that moves points and then returns within 10⁻⁸ m after the second update is flagged as a two-step oscillation; this diagnostic threshold does not relax the optimization tolerance. These are not accuracy tests.</p>',
      table(['Setting','Seed','Terminal tolerance reached','Next update norm','Next update below 10⁻⁶','Two-step oscillation'],[[p['name'],p['seed'],str(p['terminal_converged']),f'{p["next_update_norm"]:.3g}' if p['next_update_norm'] is not None else 'Insufficient pairs',str(p['next_update_below_tolerance']),str(p['two_step_return_detected'])] for p in checks]),'</details>',
      '<details><summary>Stage-by-stage stop reasons · primary geometric branch, seed 29</summary>',
      table(['Stage','Gate, m','Iterations','Last update','Tolerance','Stop reason'],[[s['stage'],s['gate_m'],s['iterations'],f'{s["last_update"]:.3g}',f'{s["tolerance"]:.3g}',s['stop_reason']] for s in next(c['trace'] for c in main_runs['pca_icp']['candidates'] if c['index']==main_runs['pca_icp']['selected_index'])]),'</details>',
      '<h2>7. Spatial uncertainty and limitations</h2><p>For seed 29, occupied source-space blocks of width 1.6 units are resampled together in a paired bootstrap (1,000 replicates). The statistic is the equal-weight mean of block medians, not the pooled point median above. Negative differences favor the second method. These approximate conditional intervals describe this scene only; spatial dependence and previous development remain limitations. No multiple-comparison significance or universal superiority claim is made.</p>',
      table(['Comparison (second minus first)','Blocks','Mean block-median change, m','Approx. 95% interval, m'],[[u['comparison'],u['blocks'],f'{u["mean_block_median_change_m"]:+.4f}',f'[{u["ci_low_m"]:+.4f}, {u["ci_high_m"]:+.4f}]'] for u in uncertainty]),
      '<p>Nearest-neighbor distances are directional and density-sensitive. They do not establish correct physical correspondences, absolute pose error, BIM construction tolerance, or repeatability on new data. Low nearest-neighbor error alone is not an acceptance certificate. Large errors and all evaluated records remain in the metrics.</p>',
      '<h2>8. Transformation matrices</h2><p>Column-vector convention: <code>p_local = T_local × [x, y, z, 1]ᵀ</code>. The upper-left block is <code>sR</code>, not just rotation. Local-to-global adds [825800, 842100, 0] once. The saved global matrix already includes that offset. Scale and translation units are explicitly recorded; inverse transformation is <code>inverse(T)</code>.</p>']
    for n,r in main_runs.items():
        T=np.array(r['matrix']);G=T.copy();G[:3,3]+=ORIGIN
        page.append(f'<details><summary>{html.escape(LABELS[n])} · seed 29 · scale {r["parts"]["scale"]:.8f}</summary><p><a href="{n}/29/T_final_local.txt">Local matrix</a> · <a href="{n}/29/T_final_global.txt">Global matrix</a> · <a href="{n}/29/result.json">Complete run and decomposition</a></p><strong>Local</strong><pre>{np.array2string(T,precision=9,suppress_small=True)}</pre><strong>Global</strong><pre>{np.array2string(G,precision=9,suppress_small=True)}</pre></details>')
    page+=['<h2>9. Photo to camera pose</h2><p>This is a separate, previously completed experiment using the historical RTK/COLMAP-assisted reference frame, not the new PCA-ablation transforms: a calibrated query image is matched against reference photographs with known COLMAP-to-LAS poses. SIFT/RootSIFT and mutual ratio-tested matches provide 2D–3D correspondences through LAS depth. EPnP-RANSAC and Levenberg–Marquardt refinement estimate the query pose. Query pose is not an input to estimation; reference poses are inputs. Query images and ten neighboring frames on either side are excluded from the reference index.</p><p>All 12 queries passed the predefined internal gate. Median camera-center difference from COLMAP was 0.167 m. LAS/COLMAP may have used the same underlying photographs, so this is internal consistency, not external camera ground truth. <a href="../camera_localization/inspection.html">Open all photo/LAS/BIM overlays</a>.</p>',
      '<h2>10. Reproduction and audit</h2><p>Restore the original 0929 PLY/LAS and exact semantic caches whose hashes are in the protocol. Use the pinned environment. The runner refuses to overwrite a frozen output directory. Original fits and viewer exports are preserved. Verify before generating the report: verification writes the same-cohort baseline comparison and extra-update probes.</p><pre>python scripts/run_registration_ablation.py --out results/ablation_new\npython scripts/verify_registration_ablation.py --out results/ablation_new\npython scripts/report_registration_ablation.py --out results/ablation_new\npython -m unittest discover -s scripts -p \'test_*.py\' -v</pre><p><a href="verification.json">Numerical verification</a> · <a href="paired_block_differences.csv">Paired block results</a> · <a href="sample_counts.json">Cohort counts</a> · <a href="iteration_comparison.csv">Iteration budgets</a> · <a href="reference_comparison.json">Same-cohort baseline comparison</a> · <a href="convergence_checks.json">Extra-update checks</a></p></html>']
    (out/'inspection.html').write_text('\n'.join(page)+'\n')
    markdown='# 0929 controlled registration ablation — convergence update\n\n'+correction+'\n\n'+conclusion+'\n\n'
    markdown+='| Method | Mean median, m | Mean P90, m | Mean RMSE, m | Terminal convergence |\n|---|---:|---:|---:|---:|\n'
    for n in MAIN:
        a=main_ag[n];stop=f'{a["terminal_converged"]}/{a["evaluated"]}' if n!='pca_only' else 'N/A'
        markdown+=f'| {LABELS[n]} | {a["median_mean_m"]:.3f} | {a["p90_mean_m"]:.3f} | {a["rmse_mean_m"]:.3f} | {stop} |\n'
    markdown+=f'\nThe historical RTK/COLMAP-assisted transform gives median {baseline["median_m"]:.3f} m on the same 48,081 points. It is a comparator, not an input to the new fits or independent ground truth. Large tail distances remain in RMSE.\n\n'
    markdown+='[Full HTML report](inspection.html) · [Every run](summary.csv) · [Protocol](protocol.json) · [Iteration comparison](iteration_comparison.csv) · [Archived 30-step report](../ablation_30_iterations/inspection.html)\n\nOnly one previously explored date is evaluated. These are reference-agreement measurements, not independent survey accuracy.\n'
    (out/'REPORT.md').write_text(markdown)
    (out/'report_summary.json').write_text(json.dumps({'conclusion':conclusion,'correction':correction,'aggregate':aggregate,'iteration_comparison':budget_csv,
        'historical_baseline_same_cohort':baseline,'paired_block_differences':uncertainty,'report_source_sha256':file_hash(Path(__file__))},indent=2)+'\n')
    print(conclusion)

if __name__=='__main__':main()
