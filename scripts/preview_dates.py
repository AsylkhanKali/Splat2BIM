"""Color plan views for checking orientation and shared physical features."""
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import laspy,numpy as np
from align_dates import ROOT,OUT,DATES,ORIGIN,samples
from align_0929 import apply


def preview(date):
    folder=ROOT/'results/0929_baseline' if date=='0929' else OUT/date
    info=json.loads((folder/'alignment.json').read_text())
    xyz,rgb,_=samples(date);xyz=apply(xyz,np.array(info['ply_to_las_local']))
    las=laspy.read(ROOT/f'assets/{date}/{date}.las')
    ref=np.c_[las.x,las.y,las.z][::10]-ORIGIN
    color=np.column_stack([getattr(las,k)[::10] for k in ('red','green','blue')]).astype(float)
    color/=65535 if color.max()>255 else 255
    fig,axes=plt.subplots(1,2,figsize=(11,9),constrained_layout=True)
    for ax,p,c,title in zip(axes,(xyz,ref),(rgb,color),('Aligned Gaussian centres (DC color)','Reference LAS')):
        order=np.argsort(p[:,2]);ax.set_facecolor('#dce2e5')
        ax.scatter(p[order,0],p[order,1],c=c[order],s=1,rasterized=True)
        ax.set(xlim=(-220,230),ylim=(-250,440),aspect='equal',title=f'{date} · {title}',xlabel='Local X, m',ylabel='Local Y, m')
    path=OUT/f'{date}_plan_check.png';fig.savefig(path,dpi=130);plt.close(fig);print(path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dates',nargs='*',default=DATES);args=parser.parse_args()
    for date in args.dates:preview(date)
