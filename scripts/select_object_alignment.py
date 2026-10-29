"""Use the existing automatic method when valid, otherwise the scale-recovery trial."""
import json

from align_dates import ROOT,DATES
from align_object_recovery import OUT
from check_las_vertical import file_hash,write_csv


def accepted(run):
    return (run.get('status')=='candidate_for_review' and len(run.get('final_spatial_support',[]))>=3 and
        run['training_final']['median']<=.5 and run['training_color_difference']<=.35)


def choose(primary,recovery):
    if accepted(primary):return 'previous_consensus',primary
    if accepted(recovery):return 'pair_preserving',recovery
    return None,None


def main():
    rows=[];routing=[]
    for date in DATES:
        previous=ROOT/f'results/object_consensus/{date}/audit.json';current=OUT/date/'audit.json'
        old=json.loads(previous.read_text());new=json.loads(current.read_text())
        primary=next(r for r in old['runs'] if r['method']=='multi_fragment')
        recovery=next(r for r in new['runs'] if r['method']=='pair_preserving')
        selected,run=choose(primary,recovery)
        audit={'date':date,'rule':'prefer accepted previous automatic consensus; use recovery only on rejection; otherwise reject',
            'selected_method':selected,'status':'candidate_for_review' if run else 'rejected_both_methods',
            'previous_status':primary['status'],'recovery_status':recovery['status'],
            'routing_uses_date_or_withheld_metrics':False,'working_baseline_used_as_candidate':False,
            'input_audits':[{'path':str(p.relative_to(ROOT)),'sha256':file_hash(p)} for p in [previous,current]],
            'selector_sha256':file_hash(ROOT/'scripts/select_object_alignment.py')}
        if run:
            audit.update(matrix_local=run['matrix_local'],final_spatial_support=run['final_spatial_support'],
                strict_forward_080_spatial_support_count=run['strict_forward_080_spatial_support_count'])
            rows.extend({**r,'method':'hybrid','selected_method':selected} for r in new['comparison'] if r['method']==selected)
        (OUT/date/'selection.json').write_text(json.dumps(audit,indent=2)+'\n');routing.append(audit)
    write_csv(OUT/'hybrid_comparison.csv',rows)
    (OUT/'selection_summary.json').write_text(json.dumps(routing,indent=2)+'\n')
    print([(r['date'],r['selected_method'],r['status']) for r in routing])


if __name__=='__main__':main()
