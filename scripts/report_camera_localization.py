"""Render estimated-camera overlays without fitting or moving the BIM."""
import json
from html import escape

import cv2
import laspy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData

from align_dates import ROOT,ORIGIN
from camera_localization import project
from run_camera_localization import OUT,DATA


def bim_edges(path):
    mesh=PlyData.read(path);v=mesh['vertex'].data
    vertices=np.c_[v['x'],v['y'],v['z']].astype(float)
    _,keep,inverse=np.unique(np.round(vertices,4),axis=0,return_index=True,return_inverse=True)
    vertices=vertices[keep];faces=inverse[np.stack(mesh['face']['vertex_indices'])]
    normal=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-15)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    face_ids=np.tile(np.arange(len(faces)),3)
    edges,first,inverse,counts=np.unique(edges,axis=0,return_index=True,return_inverse=True,return_counts=True)
    last=np.zeros(len(edges),int);np.maximum.at(last,inverse,np.arange(len(inverse)))
    sharp=abs(np.sum(normal[face_ids[first]]*normal[face_ids[last]],axis=1))<np.cos(np.deg2rad(25))
    selected=((counts==1)|sharp)&(edges[:,0]!=edges[:,1])
    return vertices[edges[selected]]


def overlay_edges(photo,edges,R,t,K,d):
    h,w=photo.shape[:2];uv,z=project(edges.reshape(-1,3),R,t,K,d);uv=uv.reshape(-1,2,2);z=z.reshape(-1,2)
    visible=(z>1).all(1)&np.isfinite(uv).all((1,2))
    visible&=(uv.max(1)>[0,0]).all(1)&(uv.min(1)<[w,h]).all(1)
    selected=edges[visible]
    # Sample edges before rational lens distortion, which bends straight 3D lines.
    along=np.linspace(0,1,9)[None,:,None]
    samples=selected[:,0,None]*(1-along)+selected[:,1,None]*along
    pixels=project(samples.reshape(-1,3),R,t,K,d)[0].reshape(-1,9,2)
    mask=np.zeros((h,w),np.uint8)
    cv2.polylines(mask,np.clip(np.rint(pixels),-100000,100000).astype(np.int32),False,255,1,cv2.LINE_AA)
    alpha=(mask/255*.9)[:,:,None]
    return np.uint8(photo*(1-alpha)+np.array([255,190,55])*alpha)


def overlay_las(photo,points,R,t,K,d):
    h,w=photo.shape[:2];uv,z=project(points,R,t,K,d)
    good=(z>1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<w-.5)&(uv[:,1]>=0)&(uv[:,1]<h-.5)
    uv=np.rint(uv[good]).astype(int);mask=np.zeros((h,w),bool);mask[uv[:,1],uv[:,0]]=True
    result=photo.copy();result[mask]=np.uint8(.4*photo[mask]+.6*np.array([35,235,220]));return result


def main():
    poses=json.loads((OUT/'poses.json').read_text());las=laspy.read(DATA/'0929.las')
    example=poses[0]
    calibration={'width':example['image_size'][0],'height':example['image_size'][1],
        'K':example['intrinsics'],'distortion':example['distortion']}
    (OUT/'example_calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    points=(np.c_[las.x,las.y,las.z]-ORIGIN)[::10]
    edges=bim_edges(ROOT/'results/0929_bim/v3_local.ply');print('BIM crease edges',len(edges),flush=True)
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    for pose in poses:
        photo=cv2.imread(str(DATA/'photos'/pose['name']));photo=cv2.cvtColor(cv2.resize(photo,tuple(pose['image_size'])),cv2.COLOR_BGR2RGB)
        fitted='world_to_camera_rotation' in pose
        fig,axes=plt.subplots(1,3,figsize=(18,5.2),constrained_layout=True)
        panels=[photo,photo,photo];titles=['Original photo','Camera pose unavailable','Camera pose unavailable']
        if fitted:
            R=np.array(pose['world_to_camera_rotation']);t=np.array(pose['world_to_camera_translation'])
            K=np.array(pose['intrinsics']);d=np.array(pose['distortion'])
            panels=[photo,overlay_las(photo,points,R,t,K,d),overlay_edges(photo,edges,R,t,K,d)]
            titles=['Original photo','LAS from the estimated camera','BIM v3 wireframe, including hidden edges']
        for ax,panel,title in zip(axes,panels,titles):ax.imshow(panel);ax.set_title(title);ax.axis('off')
        status='candidate passed the gate' if pose['status']=='candidate_for_review' else 'REJECTED'
        note=f" · difference vs COLMAP {pose['center_difference_to_colmap_m']:.2f} m / {pose['rotation_difference_to_colmap_degrees']:.2f}°" if fitted else ''
        fig.suptitle(f"Image {pose['query_index']+1}/12 · {status}{note}\n{pose['name']}",fontsize=13)
        fig.savefig(figures/f"query_{pose['query_index']:02d}.png",dpi=130);plt.close(fig)
    accepted=[p for p in poses if p['status']=='candidate_for_review'];fitted=[p for p in poses if 'camera_center_local_m' in p]
    summary={'queries':len(poses),'accepted_candidates':len(accepted),'rejected':len(poses)-len(accepted),
        'scope':'0929 calibrated same-model localization; not independent camera ground truth',
        'fitted_queries':len(fitted)}
    if accepted:
        for name in ['center_difference_to_colmap_m','rotation_difference_to_colmap_degrees','validation_fraction_within_3px']:
            values=[p[name] for p in accepted];summary[name]={'median':float(np.median(values)),'max':float(np.max(values))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Camera pose from a single photograph','',
        f"**0929: {len(accepted)} of {len(poses)} images passed the internal quality gate.** The query is matched to a reference image set, then its camera pose is estimated from LAS correspondences. BIM remains in the original shared coordinate frame.",'',
        '## Protocol','',
        'Twelve available images are sampled uniformly by filename. These queries and ±10 neighboring frames are excluded from the reference set; protocol.json records the exact split. Reference cameras are calibrated through the previous COLMAP → LAS transform. Visible LAS depths provide 2D–3D correspondences. The method uses SIFT/RootSIFT, visual retrieval of five reference images, mutual matches with distance ratio <0.75, PnP-RANSAC, and LM refinement. Query-camera coordinates and RTK are not used for retrieval or pose fitting. Known query-lens calibration is used.','',
        'Approximately one quarter of the 150 × 150 pixel image blocks is excluded from PnP and LM. Acceptance requires ≥25 training inliers, an inlier fraction ≥20%, ≥10 control matches, ≥35% of control matches within 3 pixels, an inlier convex hull covering ≥5% of the image, and positive depth. The 3-pixel threshold applies at 1200 × 900, not the original 4032 × 3024 resolution. The full control-error median includes false matches.','',
        'All 12 results are reported, including any rejections. Parameters and gates were fixed before the first run of this series. This is a research experiment on a known site. Withheld photos may have contributed to the original LAS/COLMAP reconstruction; distance to a COLMAP camera measures internal consistency, not independent absolute accuracy. Blocks within one image are not statistically independent either.','',
        '## Results','',
        '| Image | Decision | Matches / inliers | Control ≤3 px | Center difference vs COLMAP, m | Rotation difference, ° |',
        '|---|---|---:|---:|---:|---:|']
    for p in poses:
        decision='candidate' if p['status']=='candidate_for_review' else 'rejected'
        val=lambda key,fmt:format(p[key],fmt) if key in p else '—'
        lines.append(f"| {p['query_index']+1} | {decision} | {p['correspondences']} / {p.get('ransac_inliers','—')} | {val('validation_fraction_within_3px','.0%')} | {val('center_difference_to_colmap_m','.3f')} | {val('rotation_difference_to_colmap_degrees','.3f')} |")
    lines+=['','## Photo overlays','',
        'Cyan points show subsampled LAS without full surface-occlusion handling. Yellow lines show BIM v3 edges from the estimated camera, including hidden structures. Lens distortion is applied to projections; the original photo is not rectified. BIM v3 is illustrative; its validity for 0929 is unconfirmed. BIM agreement is not part of the camera gate. The wireframe is not a mask of completed structures; mismatch may reflect construction stage, design revision, or alignment error.','']
    for p in poses:lines+=[f"### Image {p['query_index']+1}: {p['name']}",'',f"![Photo, LAS and BIM](figures/query_{p['query_index']:02d}.png)",'']
    lines+=['## Files and reproduction','',
        '`poses.json` contains world→camera matrices (X right, Y down, Z forward), centers in local metres, K, and distortion coefficients. The local origin is [825800, 842100, 0]. `query_*_matches.npz` stores control/training pairs, inlier indices, and errors. `inputs.json` and the cache contain input and parameter SHA-256 hashes. The status candidate_for_review means internal acceptance for inspection, not survey approval.','',
        '```sh','OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python scripts/run_camera_localization.py',
        '.venv/bin/python scripts/report_camera_localization.py','```','',
        'The command below processes one image. The example calibration applies only to the first control photo; a different lens/image needs its own calibration. The output file must be new. Images already in the reference index are rejected to avoid presenting self-retrieval as localization.','',
        '```sh',f'.venv/bin/python scripts/localize_photo.py --image assets/0929/photos/{example["name"]} --calibration results/camera_localization/example_calibration.json --output results/camera_localization/my_pose.json','```','',
        'The method assumes known lens calibration, overlap with reference photos, and already aligned reference cameras. Equivalent data is unavailable for the other six dates. Transfer to a new site, an unknown lens, or a substantially different viewpoint is untested.','',
        'Primary references: [COLMAP text format](https://colmap.github.io/format.html), [OpenCV PnP and LM](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html), [SIFT](https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html).','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    gallery=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        '<title>0929: photos, LAS and BIM</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#17202a;background:#fff;margin:24px auto;padding:0 20px;max-width:1400px}p{max-width:78ch}img{width:100%;height:auto}figure{margin:36px 0;border-top:1px solid #ccd2d8;padding-top:16px}a{color:#175794}a:focus-visible{outline:2px solid #175794}figcaption{overflow-wrap:anywhere}h1{font-size:clamp(24px,3vw,36px)}</style>',
        '<a href="/viewer/">← 3D viewer</a><h1>0929: camera pose from a photograph</h1>',
        f'<p>{len(accepted)} of {len(poses)} images passed the internal gate. All overlays use the estimated camera. BIM placement is fixed.</p>',
        '<p>Cyan points: LAS. Yellow lines: BIM v3, including hidden structures. Difference from COLMAP measures consistency with the original project. Independent survey accuracy and BIM validity are unconfirmed.</p>',
        '<p><a href="REPORT.md">Method and limitations</a> · <a href="comparison.csv">All numerical results (CSV)</a> · <a href="poses.json">Camera matrices (JSON)</a></p>']
    for p in poses:
        decision='Passed the internal gate' if p['status']=='candidate_for_review' else 'Rejected'
        gallery.append(f'<figure><figcaption><strong>Image {p["query_index"]+1}. {decision}</strong><br>{escape(p["name"])}</figcaption><a href="figures/query_{p["query_index"]:02d}.png"><img loading="lazy" width="2340" height="676" src="figures/query_{p["query_index"]:02d}.png" alt="Image {p["query_index"]+1}: original photo, LAS overlay and BIM wireframe"></a></figure>')
    (OUT/'inspection.html').write_text('\n'.join(gallery)+'</html>\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
