"""Tessellate IFC revisions in the existing LAS-local frame; do not fit BIM."""
from pathlib import Path
from collections import Counter
import hashlib
import json

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.unit
import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/0929_bim'
BASE=json.loads((ROOT/'results/0929_baseline/alignment.json').read_text())
ORIGIN=np.array(BASE['origin_global'])


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    las=laspy.read(ROOT/'assets/0929/0929.las')
    reference=np.c_[las.x,las.y,las.z]-ORIGIN
    tree=cKDTree(reference)
    manifest={'origin_global':ORIGIN.tolist(),'bim_adjustment':'None. IFC world geometry minus the same origin as the aligned 3DGS.',
              'versions':{},'landmarks':[]}
    for file in sorted((ROOT/'assets/BIM').glob('*.ifc')):
        version=file.stem.split()[-1]
        model=ifcopenshell.open(file)
        products=[e for e in model.by_type('IfcProduct') if e.Representation]
        settings=ifcopenshell.geom.settings();settings.set('use-world-coords',True)
        iterator=ifcopenshell.geom.iterator(settings,model,4,include=products)
        verts=[];faces=[];elements=[];count=0;converted=set()
        print(version,'represented',len(products),'types',dict(Counter(e.is_a() for e in products)),flush=True)
        if iterator.initialize():
            while True:
                shape=iterator.get();entity=model.by_id(shape.id)
                points=np.array(shape.geometry.verts,dtype=np.float64).reshape(-1,3)-ORIGIN
                triangles=np.array(shape.geometry.faces,dtype=np.int32).reshape(-1,3)
                if len(points) and len(triangles):
                    rng=np.random.default_rng(shape.id)
                    a,b,c=points[triangles[:,0]],points[triangles[:,1]],points[triangles[:,2]]
                    area=np.linalg.norm(np.cross(b-a,c-a),axis=1)
                    if area.sum()>0:
                        chosen=rng.choice(len(triangles),size=500,p=area/area.sum())
                        u=rng.random(500);v=rng.random(500);u=np.sqrt(u)
                        samples=(1-u[:,None])*a[chosen]+(u*(1-v))[:,None]*b[chosen]+(u*v)[:,None]*c[chosen]
                        error=tree.query(samples,workers=2)[0]
                    else:
                        samples=points[:1];error=tree.query(samples)[0]
                    item={'id':str(shape.id),'guid':entity.GlobalId,'type':entity.is_a(),'name':entity.Name or '',
                          'bounds_min':points.min(axis=0).tolist(),'bounds_max':points.max(axis=0).tolist(),
                          'center':((points.min(axis=0)+points.max(axis=0))/2).tolist(),
                          'triangles':len(triangles),'vertices':len(points),
                          'median_surface_to_las_m':float(np.median(error)),'fraction_within_1m':float(np.mean(error<1)),
                          'geometry_sha256':hashlib.sha256(np.round(points,4).tobytes()+triangles.tobytes()).hexdigest()}
                    elements.append(item);converted.add(shape.id)
                    verts.append(points.astype(np.float32));faces.append(triangles+count);count+=len(points)
                if not iterator.next():break
        points=np.concatenate(verts);triangles=np.concatenate(faces)
        vv=np.empty(len(points),dtype=[('x','f4'),('y','f4'),('z','f4')])
        for i,k in enumerate(('x','y','z')):vv[k]=points[:,i]
        ff=np.empty(len(triangles),dtype=[('vertex_indices','i4',(3,))]);ff['vertex_indices']=triangles
        PlyData([PlyElement.describe(vv,'vertex'),PlyElement.describe(ff,'face')],text=False).write(OUT/f'{version}_local.ply')
        missing=[{'id':e.id(),'type':e.is_a(),'name':e.Name} for e in products if e.id() not in converted]
        info={'source':str(file.relative_to(ROOT)),'schema':model.schema,'length_unit_scale':ifcopenshell.util.unit.calculate_unit_scale(model),
              'represented':len(products),'converted':len(converted),'unconverted':missing,'vertices':len(points),'triangles':len(triangles),
              'bounds_min':points.min(axis=0).tolist(),'bounds_max':points.max(axis=0).tolist(),'elements':elements}
        (OUT/f'{version}_elements.json').write_text(json.dumps(info,indent=2))
        manifest['versions'][version]={k:v for k,v in info.items() if k!='elements'}
        print(version,'converted',len(converted),'missing',len(missing),'bounds',info['bounds_min'],info['bounds_max'],flush=True)
        # Save incrementally so the viewer can load completed revisions.
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    preview=np.empty(len(reference[::10]),dtype=[(k,'f4') for k in ('x','y','z')]+[(k,'u1') for k in ('red','green','blue')])
    for i,k in enumerate(('x','y','z')):preview[k]=reference[::10,i]
    for k in ('red','green','blue'):
        values=np.asarray(getattr(las,k))[::10];preview[k]=np.rint(values/257 if values.max()>255 else values).astype('u1')
    PlyData([PlyElement.describe(preview,'vertex')],text=False).write(OUT/'las_preview.ply')
    # A real source camera pose provides a useful initial inspection viewpoint.
    with (ROOT/'assets/0929/colmap/images.txt').open() as stream:
        line=next(line for line in stream if line.strip() and not line.startswith('#'))
    fields=line.split(maxsplit=9);q=np.array(fields[1:5],float)
    r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();center=-r.T@np.array(fields[5:8],float)
    T=np.array(BASE['sparse_to_las_local']);position=T[:3,:3]@center+T[:3,3]
    forward=T[:3,:3]@r.T@np.array([0,0,1]);forward/=np.linalg.norm(forward)
    manifest['photo_view']={'position':position.tolist(),'target':(position+120*forward).tolist(),'image':fields[9].strip()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
