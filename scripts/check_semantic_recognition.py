"""Local positive/negative semantic diagnostics, never used by registration."""
import json

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from align_dates import ROOT,DATES
from align_0929 import apply
from check_static_structures import volume_mask
from check_las_vertical import write_csv
from prepare_semantic_fusion import OUT


def main():
    roof=json.loads((ROOT/'results/static_structures/regions.json').read_text())['regions'][0]
    negative={'id':'vegetation','polygon':[[-60,60],[-52,60],[-22,110],[-30,110]],'z_range':[-5,30]}
    regions={'regions':[roof,negative],'purpose':'Diagnostic positive roof and negative vegetation strip; not training labels or registration input.',
        'negative_selection':'Manually selected from RGB top view 0929; RGB crops reviewed for all seven dates.',
        'source_roi_location':'Previous automatic matrix used only to locate the diagnostic regions.'}
    (OUT/'recognition_regions.json').write_text(json.dumps(regions,indent=2)+'\n')
    rows=[];fig,axes=plt.subplots(1,7,figsize=(16,4),constrained_layout=True)
    for date,ax in zip(DATES,axes):
        current=ROOT/f'results/object_recovery/{date}/selection.json';matrix=np.array(json.loads(current.read_text())['matrix_local'])
        metadata=json.loads((ROOT/f'results/semantic_matching/{date}/segmentation.json').read_text())['clouds']
        for side in ['source','target']:
            old=np.load(ROOT/f'results/semantic_matching/{date}/{side}_labels.npz');new=np.load(OUT/date/f'{side}_labels.npz')
            np.testing.assert_array_equal(old['points'],new['points']);p=old['points'];world=apply(p,matrix) if side=='source' else p
            for region in [roof,negative]:
                mask=volume_mask(world,region)
                for method,data in [('previous',old),('fusion',new)]:
                    rows.append({'date':date,'side':side,'region':region['id'],'method':method,'count':int(mask.sum()),
                        'built_points':int(np.sum(data['labels'][mask]==1)),
                        'built_fraction':float(np.mean(data['labels'][mask]==1)) if mask.any() else None})
            if side=='target':
                m=metadata[side];local=(p-m['center'])@np.array(m['frame']);uv=np.rint((local[:,:2]-m['origin_xy'])/m['pixel_size_source_units']).astype(int);uv[:,1]=1023-uv[:,1]
                rgb=cv2.cvtColor(cv2.imread(str(ROOT/f'results/semantic_matching/{date}/{side}_rgb.png')),cv2.COLOR_BGR2RGB)[uv[:,1],uv[:,0]]/255
                bounds=(p[:,0]>-90)&(p[:,0]<-15)&(p[:,1]>50)&(p[:,1]<120)
                ax.scatter(p[bounds,0],p[bounds,1],c=rgb[bounds],s=1,rasterized=True);ax.add_patch(Polygon(negative['polygon'],fill=False,color='#df2b83',linewidth=1.3))
                ax.set(title=date,xlim=(-90,-15),ylim=(50,120),aspect='equal');ax.tick_params(labelsize=7)
        rows.extend([{'date':date,'side':side,'region':'all_visible','method':method,'count':len(data['labels']),
            'built_points':int(np.sum(data['labels']==1)),'built_fraction':float(np.mean(data['labels']==1))}
            for side in ['source','target'] for method,folder in [('previous',ROOT/'results/semantic_matching'),('fusion',OUT)]
            for data in [np.load(folder/date/f'{side}_labels.npz')]])
    fig.suptitle('RGB review of the negative vegetation region (LAS); no predictions shown')
    fig.savefig(OUT/'negative_region_review.png',dpi=150);plt.close(fig)
    write_csv(OUT/'recognition.csv',rows)
    print(json.dumps([r for r in rows if r['region']=='roof' and r['method']=='fusion'],indent=2))


if __name__=='__main__':main()
