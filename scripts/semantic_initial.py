"""Batch scoring before ICP; the same pool and budget for both variants."""
import numpy as np

from align_0929 import apply
from semantic_projection import PARAMETERS

INITIAL_PARAMETERS={'geometry_score_points':512,'batch_size':128,'coarse_starts':4,
    'minimum_start_separation_m':2.,'geometry_quantile':.75,'coarse_iterations':40,
    'coarse_gates_m':[40.,20.,10.,5.]}


def score_pool(source,target_tree,matcher,matrices):
    scores=np.empty((len(matrices),2));p=INITIAL_PARAMETERS
    for start in range(0,len(matrices),p['batch_size']):
        batch=matrices[start:start+p['batch_size']];r=batch[:,:3,:3];t=batch[:,:3,3]
        moved=np.einsum('nj,kij->kni',source,r)+t[:,None,:]
        geometry=np.quantile(target_tree.query(moved)[0],p['geometry_quantile'],axis=1)
        semantic=[]
        for group,a,b,tree in matcher.classes:
            moved=np.einsum('nj,kij->kni',matcher.source['points'][a],r)+t[:,None,:]
            error=np.minimum(tree.query(moved)[0],PARAMETERS['semantic_distance_cap_m'])
            semantic.append(np.quantile(error,PARAMETERS['semantic_distance_quantile'],axis=1))
        scores[start:start+len(batch),0]=geometry
        scores[start:start+len(batch),1]=np.mean(semantic,axis=0) if len(semantic)>=PARAMETERS['min_shared_groups'] else np.nan
    return scores


def choose_starts(matrices,scores,anchors,semantic=False):
    values=scores[:,0]+PARAMETERS['semantic_weight']*scores[:,1] if semantic else scores[:,0]
    chosen=[];positions=[]
    for index in np.argsort(values,kind='stable'):
        if not np.isfinite(values[index]):continue
        moved=apply(anchors,matrices[index])
        if any(np.sqrt(np.mean(np.sum((moved-other)**2,axis=1)))<INITIAL_PARAMETERS['minimum_start_separation_m'] for other in positions):continue
        chosen.append(int(index));positions.append(moved)
        if len(chosen)==INITIAL_PARAMETERS['coarse_starts']:break
    return chosen
