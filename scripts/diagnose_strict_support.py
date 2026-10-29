"""Explain unchanged strict gates; diagnostic final-pose matches never change a fit."""
import itertools
import json

import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from align_dates import ROOT,DATES,ORIGIN
from align_0929 import apply
from align_object_recovery import load_fragments
from align_object_consensus import patch_checks
from object_consensus import descriptors,match,spatial_support,PARAMETERS
from check_las_vertical import file_hash,write_csv

OUT=ROOT/'results/strict_diagnostics'


def triangle_diagnostics(pairs,source,target,matrix):
    A=apply(source['centers'],matrix);B=target['centers'];rows=[]
    for chosen in itertools.combinations(pairs,3):
        a=A[[p['source_index'] for p in chosen]];b=B[[p['target_index'] for p in chosen]]
        sa=float(np.linalg.svd(a-a.mean(0),compute_uv=False)[1]);sb=float(np.linalg.svd(b-b.mean(0),compute_uv=False)[1])
        da=min(np.linalg.norm(a[i]-a[j]) for i,j in itertools.combinations(range(3),2));db=min(np.linalg.norm(b[i]-b[j]) for i,j in itertools.combinations(range(3),2))
        rows.append({'identities':[(p['source_id'],p['target_id']) for p in chosen],
            'source_second_axis_m':sa,'target_second_axis_m':sb,'source_min_spacing_m':float(da),'target_min_spacing_m':float(db),
            'passes':bool(min(sa,sb)>=PARAMETERS['min_second_axis_m'] and min(da,db)>=PARAMETERS['min_separation_m'])})
    return rows


def main():
    OUT.mkdir(exist_ok=True);summary=[]
    for date in DATES:
        path=ROOT/f'results/semantic_fusion/{date}/audit.json';audit=json.loads(path.read_text());run=next(r for r in audit['runs'] if r['method']=='semantic_fusion');matrix=np.array(run['matrix_local'])
        sf,tf=load_fragments(date);sd,td=descriptors(sf),descriptors(tf);assigned=match(sd,td,matrix);assigned_ids={(p['source_id'],p['target_id']) for p in assigned}
        allchecks=patch_checks(assigned,sf,tf,matrix)
        eligible=lambda rows:[p for p in rows if p['fraction_within_1m']>=.8 and p['reverse_fraction_within_1m']>=.5 and (p['source_id'],p['target_id']) in assigned_ids]
        strict=eligible(run['final_pair_checks']);posthoc=eligible(allchecks)
        triangles=triangle_diagnostics(strict,sd,td,matrix);posthoc_triangles=triangle_diagnostics(posthoc,sd,td,matrix)
        result={'date':date,'parent_audit_sha256':file_hash(path),'matrix_local':matrix.tolist(),'parameters':PARAMETERS,
            'strict_forward_fraction':.8,'strict_pairs':strict,'strict_pair_count':len(strict),'admissible_bank_triangles':sum(t['passes'] for t in triangles),
            'bank_triangles':triangles,'posthoc_diagnostic_strict_pair_count':len(posthoc),'posthoc_diagnostic_admissible_triangles':sum(t['passes'] for t in posthoc_triangles),
            'posthoc_diagnostic_never_used_for_acceptance':True,'original_strict_support_count':run['strict_forward_080_spatial_support_count']}
        summary.append({k:result[k] for k in ['date','strict_pair_count','admissible_bank_triangles','posthoc_diagnostic_strict_pair_count','posthoc_diagnostic_admissible_triangles','original_strict_support_count']})
        if date in ['0708','0723']:
            las=laspy.read(ROOT/f'assets/{date}/{date}.las');target=np.c_[las.x,las.y,las.z]-ORIGIN;tree=cKDTree(target)
            patches=[];fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
            A=apply(sd['centers'],matrix);B=td['centers'];P={r['id']:r['points']/sf['normalization_scale']+sf['normalization_center'] for r in sf['candidates']};Q={r['id']:r['points']/tf['normalization_scale']+tf['normalization_center'] for r in tf['candidates']}
            for index,pair in enumerate(run['final_pair_checks']):
                a=apply(P[pair['source_id']],matrix);b=Q[pair['target_id']];d=cKDTree(b).query(a)[0];full=tree.query(a)[0];bad=d>=1.
                backward=cKDTree(a).query(b)[0];target_in_source=apply(b,np.linalg.inv(matrix));held=np.floor(target_in_source/.4).sum(1).astype(int)%4==0
                row={**pair,'source_count':len(a),'target_count':len(b),'additional_source_points_needed_for_080':max(0,int(np.ceil(.8*len(a)))-int(np.sum(~bad))),
                    'source_full_las_within_1m':float(np.mean(full<1.)),
                    'source_fraction_far_from_patch_but_near_full_las':float(np.mean(bad&(full<1.))),
                    'source_fraction_far_from_full_las':float(np.mean(full>=1.)),
                    'reverse_fail_fraction_in_source_holdout_tiles':float(np.mean(held[backward>=1.])) if np.any(backward>=1.) else None,
                    'geometric_assignment_survives':(pair['source_id'],pair['target_id']) in assigned_ids}
                patches.append(row)
                if min(pair['fraction_within_1m'],pair['reverse_fraction_within_1m'])>=.5 and row['geometric_assignment_survives']:
                    color='#208b69' if pair['fraction_within_1m']>=.8 else '#c36f28';i=pair['source_index'];j=pair['target_index']
                    axes[0].scatter(*A[i,:2],color=color,s=45);axes[0].plot([A[i,0],B[j,0]],[A[i,1],B[j,1]],color=color)
                    axes[0].annotate(pair['source_id'],A[i,:2],xytext=(3,5),textcoords='offset points',fontsize=8)
            names=[p['source_id'] for p in patches];x=np.arange(len(patches))
            axes[1].bar(x-.18,[p['fraction_within_1m'] for p in patches],.36,label='Source → fragment')
            axes[1].bar(x+.18,[p['reverse_fraction_within_1m'] for p in patches],.36,label='Fragment → source')
            axes[1].axhline(.8,color='#a32f3d',ls='--',label='Forward 80%');axes[1].axhline(.5,color='#777',ls=':',label='Reverse 50%')
            axes[1].set(xticks=x,xticklabels=names,ylim=(0,1.08),title='Overlap within 1 m');axes[1].tick_params(axis='x',rotation=65);axes[1].legend(fontsize=7)
            axes[2].bar(x,[p['source_full_las_within_1m'] for p in patches],color='#208b69',label='Near full LAS')
            axes[2].bar(x,[p['fraction_within_1m'] for p in patches],color='#6785a8',label='Near assigned fragment')
            axes[2].set(xticks=x,xticklabels=names,ylim=(0,1.08),title='Full LAS versus fixed fragment');axes[2].tick_params(axis='x',rotation=65);axes[2].legend(fontsize=7)
            axes[0].set(aspect='equal',xlabel='Local X, m',ylabel='Local Y, m',title='Green: strict pairs; orange: primary only')
            fig.suptitle(date+' · unchanged matrices and thresholds');fig.savefig(OUT/f'{date}_diagnostic.png',dpi=150);plt.close(fig)
            result['patch_diagnostics']=patches;write_csv(OUT/f'{date}_patches.csv',patches)
        (OUT/f'{date}.json').write_text(json.dumps(result,indent=2)+'\n');print(date,summary[-1],flush=True)
    write_csv(OUT/'summary.csv',summary)
    (OUT/'protocol.json').write_text(json.dumps({'code_sha256':file_hash(ROOT/'scripts/diagnose_strict_support.py'),'no_fit_or_threshold_changes':True},indent=2)+'\n')


if __name__=='__main__':main()
