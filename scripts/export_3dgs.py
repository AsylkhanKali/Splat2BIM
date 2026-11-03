"""Apply a proper similarity to Gaussian geometry and view-dependent color."""
import json
import hashlib

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.special import sph_harm_y
from scipy.spatial.transform import Rotation

from align_0929 import DATA, OUT, apply


def sh_basis(directions, degree):
    """Real SH in the standard 3DGS order, m=-l..l (Condon-Shortley phase)."""
    theta = np.arccos(np.clip(directions[:,2], -1, 1))
    phi = np.arctan2(directions[:,1],directions[:,0])
    columns=[]
    for m in range(-degree,degree+1):
        value=sph_harm_y(degree,abs(m),theta,phi)
        columns.append(value.real if m==0 else np.sqrt(2)*(value.imag if m<0 else value.real))
    return np.stack(columns,axis=1)


def sh_rotation(rotation):
    """Compute degree-preserving SH rotation by overdetermined basis fitting."""
    directions=np.random.default_rng(42).normal(size=(128,3))
    directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    matrix=np.zeros((15,15))
    start=0
    for degree in (1,2,3):
        end=start+2*degree+1
        matrix[start:end,start:end]=np.linalg.lstsq(
            sh_basis(directions,degree), sh_basis(directions @ rotation,degree),rcond=None)[0]
        start=end
    return matrix


def transform_vertices(vertices, matrix):
    """Standard PLY: wxyz quaternion, log scales, channel-major degree-3 SH."""
    scale=np.cbrt(np.linalg.det(matrix[:3,:3]))
    rotation=matrix[:3,:3]/scale
    if scale<=0 or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-7):
        raise ValueError("Expected a positive uniform scale and a proper rotation")
    sh=sh_rotation(rotation)
    out=vertices.copy()
    rest=[f"f_rest_{i}" for i in range(45)]
    if not set(rest).issubset(vertices.dtype.names):
        raise ValueError("This exporter expects the supplied degree-3 Gaussian PLY format")
    global_rotation=Rotation.from_matrix(rotation)
    for start in range(0,len(out),100000):
        part=vertices[start:start+100000]
        dst=out[start:start+100000]
        xyz=np.column_stack([part[k] for k in ("x","y","z")])
        xyz=apply(xyz.astype(float),matrix)
        for j,k in enumerate(("x","y","z")):dst[k]=xyz[:,j]
        quat=np.column_stack([part[f"rot_{i}"] for i in range(4)])
        quat=(global_rotation*Rotation.from_quat(quat,scalar_first=True)).as_quat(scalar_first=True)
        for i in range(4):dst[f"rot_{i}"]=quat[:,i]
        for i in range(3):dst[f"scale_{i}"]=part[f"scale_{i}"]+np.log(scale)
        coefficients=np.column_stack([part[k] for k in rest]).reshape(-1,3,15)
        coefficients=(coefficients@sh.T).reshape(-1,45)
        for i,k in enumerate(rest):dst[k]=coefficients[:,i]
    return out


def main():
    info=json.loads((OUT/"alignment.json").read_text())
    matrix=np.array(info["ply_to_las_local"])
    origin=np.array(info["origin_global"])
    ply=PlyData.read(DATA/"0929.ply",mmap="r")
    vertices=transform_vertices(ply["vertex"].data,matrix)
    output=OUT/"0929_aligned_local.ply"
    PlyData([PlyElement.describe(vertices,"vertex")],text=False,
            comments=list(ply.comments)+["Local metres; add origin_global from alignment.json for LAS coordinates"]).write(output)
    global_matrix=matrix.copy();global_matrix[:3,3]+=origin
    np.savetxt(OUT/"T_3dgs_to_las_local.txt",matrix,fmt="%.12f")
    np.savetxt(OUT/"T_3dgs_to_las_global.txt",global_matrix,fmt="%.12f")
    las=laspy.read(DATA/"0929.las")
    reference=np.empty(len(las.points),dtype=[(k,"<f4") for k in ("x","y","z")]+[(k,"u1") for k in ("red","green","blue")])
    for i,k in enumerate(("x","y","z")):reference[k]=np.asarray(getattr(las,k))-origin[i]
    for k in ("red","green","blue"):
        values=np.asarray(getattr(las,k))
        reference[k]=np.rint(values/257 if values.max()>255 else values).astype(np.uint8)
    PlyData([PlyElement.describe(reference,"vertex")],text=False,
            comments=["Reference LAS in the same local coordinate frame as 0929_aligned_local.ply"]).write(OUT/"0929_reference_local.ply")
    # Re-open the actual export and verify positions and invariant attributes.
    reread=PlyData.read(output,mmap="r")["vertex"].data
    sample=np.arange(0,len(reread),1000)
    original=ply["vertex"].data[sample]
    expected=apply(np.column_stack([original[k] for k in ("x","y","z")]),matrix)
    actual=np.column_stack([reread[k][sample] for k in ("x","y","z")])
    assert len(reread)==len(ply["vertex"].data)
    assert np.max(np.abs(expected-actual))<.001
    assert np.array_equal(original["opacity"],reread["opacity"][sample])
    checks={"count_preserved":len(reread)==len(ply["vertex"].data),
            "all_fields_preserved":reread.dtype==ply["vertex"].data.dtype,
            "opacity_preserved":bool(np.array_equal(reread["opacity"],ply["vertex"].data["opacity"])),
            "dc_color_preserved":all(np.array_equal(reread[f"f_dc_{k}"],ply["vertex"].data[f"f_dc_{k}"]) for k in range(3)),
            "all_output_values_finite":all(np.isfinite(reread[k]).all() for k in reread.dtype.names),
            "max_sampled_position_rounding_m":float(np.max(np.abs(expected-actual))),
            "reference_count":len(reference)}
    quaternion=np.column_stack([reread[f"rot_{i}"] for i in range(4)])
    checks["max_quaternion_norm_error"]=float(np.max(np.abs(np.linalg.norm(quaternion,axis=1)-1)))
    assert checks["all_output_values_finite"] and checks["all_fields_preserved"] and checks["dc_color_preserved"] and checks["opacity_preserved"]
    assert checks["max_quaternion_norm_error"]<1e-6
    (OUT/"export_checks.json").write_text(json.dumps(checks,indent=2)+"\n")
    manifest=[]
    for file in [DATA/"0929.las", DATA/"0929.ply", DATA/"colmap/images.txt", DATA/"colmap/points3D.txt"]:
        with file.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        manifest.append({"path":str(file.relative_to(DATA.parents[1])),"bytes":file.stat().st_size,"sha256":digest})
    (OUT/"input_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Exported and re-read {len(reread):,} Gaussians; maximum sampled coordinate rounding {np.max(np.abs(expected-actual)):.8f} m")


if __name__=="__main__":main()
