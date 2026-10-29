"""Summarize the unadjusted BIM check and populate viewer inspection points."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/0929_bim'


def main():
    versions={v:json.loads((OUT/f'{v}_elements.json').read_text()) for v in ('v1','v2','v3')}
    manifest=json.loads((OUT/'manifest.json').read_text())
    elements=versions['v3']['elements']
    groups={name:[e for e in elements if name.lower() in e['name'].lower()] for name in ('Deck','Canopy','Pier','Pile Cap','Kiosk')}
    groups['Underground piles']=[e for e in elements if e['type']=='IfcPile']
    summary={}
    for name,items in groups.items():
        errors=np.array([e['median_surface_to_las_m'] for e in items])
        summary[name]={'elements':len(items),'median_of_element_medians_m':float(np.median(errors)),
                       'p90_of_element_medians_m':float(np.quantile(errors,.9))}
    changes={}
    current={e['guid']:e for e in elements}
    for v in ('v1','v2'):
        previous={e['guid']:e for e in versions[v]['elements']}
        common=current.keys()&previous.keys()
        changes[v]={'common':len(common),'same_geometry':sum(current[k]['geometry_sha256']==previous[k]['geometry_sha256'] for k in common),
                    'removed':len(previous.keys()-current.keys()),
                    'added':[current[k] for k in sorted(current.keys()-previous.keys())]}
    choices=[('9525','Deck near FP4/FP5'),('49665','Canopy, north section'),('11021','FP1 support'),
             ('10912','Pile cap: check visibility'),('5828','Kiosk, west section')]
    by_id={e['id']:e for e in elements}
    manifest['landmarks']=[{'id':key,'label':label,'center':by_id[key]['center'],
         'bounds_min':by_id[key]['bounds_min'],'bounds_max':by_id[key]['bounds_max'],
         'note':f"BIM v3 #{key}; median distance to LAS {by_id[key]['median_surface_to_las_m']:.2f} m (500 surface samples)"}
         for key,label in choices]
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    result={'coordinate_adjustment':'Only subtract [825800,842100,0]; no rotation, scale, ICP or fitted translation applied to BIM.',
            'revision_for_inspection':'v3; authoritative revision not confirmed',
            'group_statistics':summary,'revision_comparison':changes,
            'landmarks':[by_id[key] for key,label in choices],
            'limitations':['Distances are 500 area-weighted random BIM surface samples to nearest LAS point per element, not corresponding survey control.',
                           'Entire surfaces include undersides, hidden faces and underground geometry; this is not a construction-completion score.',
                           'Element medians are aggregated with equal element weight, not by area.',
                           'Source-space sample seeds are IFC entity IDs; identical geometry may have slightly different sampled distances between revisions.']}
    (OUT/'bim_check.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    with (OUT/'v3_element_distances.csv').open('w',newline='') as stream:
        fields=['id','guid','type','name','median_surface_to_las_m','fraction_within_1m']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(elements)
    lines=['# 0929 ↔ BIM inspection','',
           '3DGS and all three BIM revisions share one local metre frame. BIM receives only the same origin shift `[825800, 842100, 0]`; no additional BIM fit is applied. The common placement appears consistent, but this is preliminary inspection, not survey acceptance.','',
           '## Revisions and completeness','',
           '| Revision | Elements converted | Not represented by a mesh |','|---|---:|---:|']
    for v,info in versions.items():lines.append(f"| {v} | {info['converted']} | {len(info['unconverted'])} |")
    lines+=['','Only `IFC Project Base Point`, an annotation rather than a construction surface, is excluded in all three files.',
            'v3 retains geometry for all 5579 v2 elements and adds 7: four kiosks and three unnamed elements. It is used for inspection as the more complete of these two revisions; its official status requires confirmation.',
            'v1 and v3 share 5479 elements with identical GUIDs and geometry; 200 v1 elements are absent from v3 and 107 are added. Revisions should not be merged into one model.','',
            '## BIM → LAS distances','',
            'Each element has 500 surface samples drawn with probability proportional to triangle area. The table reports the median of individual element medians, not global registration accuracy.','',
            '| Group | Elements | Median of medians, m | 90th percentile of medians, m |','|---|---:|---:|---:|']
    for name,row in summary.items():lines.append(f"| {name} | {row['elements']} | {row['median_of_element_medians_m']:.3f} | {row['p90_of_element_medians_m']:.3f} |")
    lines+=['','## Visual inspection areas','',
            '| v3 element | Median distance to LAS, m | Fraction of samples within 1 m |','|---|---:|---:|']
    for key,label in choices:
        e=by_id[key];lines.append(f"| {label} · #{key} | {e['median_surface_to_las_m']:.3f} | {e['fraction_within_1m']:.1%} |")
    lines+=['','These areas were selected for inspection, not as independent control samples. Full element-level results are in `v3_element_distances.csv`.',
            'Pile caps and piles include hidden and underground surfaces. Their large distances to aerial observations do not automatically indicate model displacement or missing structures. Decks also include underside and side surfaces that are not visible from every viewpoint.','',
            '## How to open','',
            'Open the running viewer: http://127.0.0.1:8765/viewer/',
            'Toggle 3DGS, LAS, and BIM revisions; the structure list focuses selected elements and outlines them in cyan. The wireframe intentionally includes hidden parts: it is an overlay aid, not physically exact visibility through translucent Gaussians. Disable BIM for an unobstructed capture view.','',
            'To restart from the project root:','',
            '```sh','npm install --prefix viewer --ignore-scripts --no-audit --no-fund',
            '.venv/bin/python scripts/prepare_bim.py','.venv/bin/python scripts/report_bim.py',
            '.venv/bin/python scripts/serve_viewer.py','```','',
            'The viewer uses locally installed Spark 2.4.0 and Three.js 0.180.0. The server binds only to 127.0.0.1 and serves only `viewer/` and `results/`; raw input directories are not served.']
    (OUT/'BIM_CHECK.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
