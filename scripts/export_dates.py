"""Export the six registered 3DGS clouds with covariance and SH rotation."""
import argparse
import hashlib
import json

import laspy
import numpy as np
from plyfile import PlyData,PlyElement

from align_dates import ROOT,OUT,ORIGIN,DATES
from align_0929 import apply
from export_3dgs import transform_vertices


def export(date):
    out=OUT/date
    info=json.loads((out/'alignment.json').read_text());t=np.array(info['ply_to_las_local'])
    if info.get('status')=='rejected':raise ValueError(f'{date}: registration has not passed review')
    original=PlyData.read(ROOT/f'assets/{date}/{date}.ply',mmap='r')
    source=original['vertex'].data
    path=out/f'{date}_aligned_local.ply'
    vertices=transform_vertices(source,t)
    PlyData([PlyElement.describe(vertices,'vertex')],text=False,comments=list(original.comments)+[
        'Local metres; add [825800,842100,0] for LAS / BIM world coordinates']).write(path)
    del vertices
    saved=PlyData.read(path,mmap='r')['vertex'].data
    ix=np.arange(0,len(source),1000)
    expected=apply(np.column_stack([source[k][ix] for k in ('x','y','z')]).astype(float),t)
    actual=np.column_stack([saved[k][ix] for k in ('x','y','z')])
    checks={'count_preserved':len(saved)==len(source),'all_fields_preserved':saved.dtype==source.dtype,
            'all_output_values_finite':all(np.isfinite(saved[k]).all() for k in saved.dtype.names),
            'opacity_preserved':bool(np.array_equal(saved['opacity'],source['opacity'])),
            'dc_color_preserved':all(np.array_equal(saved[f'f_dc_{k}'],source[f'f_dc_{k}']) for k in range(3)),
            'max_sampled_position_rounding_m':float(np.max(np.abs(expected-actual)))}
    norms=np.sqrt(sum(saved[f'rot_{k}'].astype(float)**2 for k in range(4)))
    checks['max_quaternion_norm_error']=float(np.max(np.abs(norms-1)))
    assert all(checks[k] for k in ('count_preserved','all_fields_preserved','all_output_values_finite','opacity_preserved','dc_color_preserved'))
    assert checks['max_sampled_position_rounding_m']<.001 and checks['max_quaternion_norm_error']<1e-6
    (out/'export_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    np.savetxt(out/'T_3dgs_to_las_local.txt',t,fmt='%.12f')
    global_t=t.copy();global_t[:3,3]+=ORIGIN
    np.savetxt(out/'T_3dgs_to_las_global.txt',global_t,fmt='%.12f')
    las=laspy.read(ROOT/f'assets/{date}/{date}.las')
    points=np.c_[las.x,las.y,las.z][::10]-ORIGIN
    preview=np.empty(len(points),dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate(('x','y','z')):preview[k]=points[:,i]
    for k in ('red','green','blue'):
        values=np.asarray(getattr(las,k))[::10]
        preview[k]=np.rint(values/257 if values.max()>255 else values).astype('u1')
    PlyData([PlyElement.describe(preview,'vertex')],text=False).write(out/'las_preview.ply')
    inputs=[]
    for ext in ('ply','las'):
        file=ROOT/f'assets/{date}/{date}.{ext}'
        with file.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        inputs.append({'path':str(file.relative_to(ROOT)),'bytes':file.stat().st_size,'sha256':digest})
    (out/'input_manifest.json').write_text(json.dumps(inputs,indent=2)+'\n')
    print(date,'export verified',len(saved),checks,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dates',nargs='*',default=DATES[:-1]);args=parser.parse_args()
    for date in args.dates:export(date)
