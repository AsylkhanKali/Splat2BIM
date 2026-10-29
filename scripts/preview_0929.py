"""Generate an inspectable static figure from the baseline and source data."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import laspy
import numpy as np
from plyfile import PlyData
from align_0929 import DATA, OUT, apply


def main():
    info=json.loads((OUT/"alignment.json").read_text())
    diagnostics=np.load(OUT/"diagnostics.npz")
    las=laspy.read(DATA/"0929.las")
    ref=np.c_[las.x,las.y,las.z][::8]-np.array(info["origin_global"])
    rgb=np.c_[las.red,las.green,las.blue][::8].astype(float)
    rgb/=65535 if rgb.max()>255 else 255
    vertices=PlyData.read(DATA/"0929.ply",mmap="r")["vertex"].data[::8]
    vertices=vertices[vertices["opacity"]>0]
    points=apply(np.c_[vertices["x"],vertices["y"],vertices["z"]],np.array(info["ply_to_las_local"]))
    colors=np.clip(.5+np.sqrt(1/(4*np.pi))*np.c_[vertices["f_dc_0"],vertices["f_dc_1"],vertices["f_dc_2"]],0,1)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(2,2,figsize=(15,12))
    fig.suptitle("0929 · First outdoor alignment baseline",fontsize=20,x=.07,ha="left",y=.98)
    fig.text(.07,.945,"LAS reference and transformed Gaussian centers in the same local metre coordinates",fontsize=11,color="#555555")
    for ax,xyz,color,title in [(axs[0,0],ref,rgb,"A  Reference LAS"),(axs[0,1],points,colors,"B  Aligned 3DGS centers · DC color preview")]:
        order=np.argsort(xyz[:,2])
        ax.scatter(xyz[order,0],xyz[order,1],c=color[order],s=.4,rasterized=True)
        ax.set(xlim=(-380,280),ylim=(-270,450),xlabel="East (local m)",ylabel="North (local m)",title=title)
        ax.set_aspect("equal")
    ax=axs[1,0]
    ax.scatter(ref[:,0],ref[:,1],c="#215da8",s=.7,label="LAS",alpha=.65)
    ax.scatter(points[:,0],points[:,1],c="#e98a2b",s=.5,label="3DGS centers",alpha=.6)
    ax.set(xlim=(-70,30),ylim=(-30,70),xlabel="East (local m)",ylabel="North (local m)",title="C  Shared-area detail · geometry overlay")
    ax.set_aspect("equal");ax.legend(loc="upper left",markerscale=5)
    ax=axs[1,1]
    errors=np.sort(diagnostics["gaussian_check_error"])
    ax.plot(errors,np.arange(1,len(errors)+1)/len(errors)*100,color="#215da8",lw=2)
    ax.axvline(1,color="#aaaaaa",ls="--",lw=1)
    ax.set(xlim=(0,5),ylim=(0,100),xlabel="Distance to nearest LAS point (m)",ylabel="Cumulative fraction (%)",title="D  Withheld Gaussian samples in shared XY coverage")
    ax.grid(alpha=.2)
    stats=info["ply_withheld_overlap_distances_m"]
    ax.text(.97,.08,f"{stats['count']:,} samples\nMedian {stats['median']:.3f} m\n90th percentile {stats['p90']:.3f} m\nWithin 1 m: {info['ply_withheld_overlap_fraction_within_1m']:.1%}",transform=ax.transAxes,ha="right",va="bottom",fontsize=12)
    fig.text(.07,.03,"Distances measure agreement with the provided LAS, not surveyed accuracy. Gaussian centers are not a surface render.\n25% of source-space tiles were withheld from PLY fitting; shared coverage means nearest LAS XY distance < 1 m.",fontsize=10,color="#555555")
    fig.subplots_adjust(left=.07,right=.96,bottom=.1,top=.90,hspace=.3,wspace=.25)
    fig.savefig(OUT/"alignment_preview.png",dpi=180)
    print(OUT/"alignment_preview.png")


if __name__=="__main__":main()
