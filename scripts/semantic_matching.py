"""Class-balanced nearest-surface correspondences for semantic seed selection."""
import numpy as np
from scipy.spatial import cKDTree

from align_0929 import apply
from semantic_projection import PARAMETERS,GROUPS


class SemanticMatcher:
    def __init__(self,source,target):
        self.source=source;self.target=target;self.classes=[]
        rng=np.random.default_rng(29)
        for group in PARAMETERS['static_groups']:
            a=np.flatnonzero(source['labels']==group);b=np.flatnonzero(target['labels']==group)
            if min(len(a),len(b))<PARAMETERS['min_class_points']:continue
            if len(a)>PARAMETERS['max_source_points_per_class']:
                a=np.sort(rng.choice(a,PARAMETERS['max_source_points_per_class'],replace=False))
            self.classes.append((group,a,b,cKDTree(target['points'][b])))

    def score(self,matrix):
        if len(self.classes)<PARAMETERS['min_shared_groups']:return None,[]
        rows=[]
        for group,a,b,tree in self.classes:
            distances,near=tree.query(apply(self.source['points'][a],matrix))
            clipped=np.minimum(distances,PARAMETERS['semantic_distance_cap_m'])
            value=float(np.quantile(clipped,PARAMETERS['semantic_distance_quantile']))
            rows.append({'class_id':group,'class_name':GROUPS[group],'score_m':value,
                'sample_count':len(a),'within_1m':float(np.mean(distances<1.)),
                'median_m':float(np.median(distances)),'source_indices':a.tolist(),'target_indices':b[near].tolist()})
        return float(np.mean([r['score_m'] for r in rows])),rows


def choose_candidate(rows,semantic=False):
    eligible=[i for i,r in enumerate(rows) if r['scale_admissible'] and r['geometric_support_count']>=3]
    if semantic:eligible=[i for i in eligible if rows[i]['semantic_score_m'] is not None]
    if not eligible:return None
    def score(index):
        row=rows[index];base=row['geometry_score_m']
        return base+PARAMETERS['semantic_weight']*row['semantic_score_m'] if semantic else base
    return min(eligible,key=lambda i:(score(i),i))
