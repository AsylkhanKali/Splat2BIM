"""Numerical checks for frame conventions, Gaussian shape, and SH appearance."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from align_0929 import apply, similarity, icp, refine_similarity_planes
from export_3dgs import transform_vertices, sh_basis, sh_rotation
from align_dates import refine


class RegistrationTests(unittest.TestCase):
    def test_similarity_recovers_known_transform(self):
        source=np.random.default_rng(1).normal(size=(100,3))
        expected=np.eye(4)
        expected[:3,:3]=63.4*Rotation.from_euler("xyz",[37,-21,109],degrees=True).as_matrix()
        expected[:3,3]=[825800,842200,12]
        actual=similarity(source,apply(source,expected))
        np.testing.assert_allclose(actual,expected,atol=1e-9)

    def test_icp_refines_known_small_misalignment(self):
        source=np.random.default_rng(2).normal(size=(1500,3))
        expected=np.eye(4)
        expected[:3,:3]=Rotation.from_euler("z",2,degrees=True).as_matrix()
        expected[:3,3]=[.03,-.02,.04]
        actual=icp(source,apply(source,expected),np.eye(4),[.3,.1])
        np.testing.assert_allclose(actual,expected,atol=1e-5)

    def test_sh_degree_one_convention(self):
        directions=np.eye(3)
        expected=np.array([[0,0,-1],[-1,0,0],[0,1,0]])*np.sqrt(3/(4*np.pi))
        np.testing.assert_allclose(sh_basis(directions,1),expected,atol=1e-12)

    def test_surface_refinement_recovers_scale(self):
        rng=np.random.default_rng(9)
        source=rng.normal(size=(4000,3))
        source/=np.linalg.norm(source,axis=1,keepdims=True)
        source*=np.array([2.,1.,.5])
        expected=np.eye(4)
        expected[:3,:3]=1.08*Rotation.from_euler("xyz",[.02,-.03,.04]).as_matrix()
        expected[:3,3]=[.04,-.03,.02]
        actual=refine_similarity_planes(source,apply(source,expected),np.eye(4))
        np.testing.assert_allclose(actual,expected,atol=1e-3)

    def test_sh_color_preserved_when_view_direction_rotates(self):
        rng=np.random.default_rng(3)
        directions=rng.normal(size=(80,3));directions/=np.linalg.norm(directions,axis=1,keepdims=True)
        rotation=Rotation.from_euler("xyz",[.3,-.7,1.2]).as_matrix()
        coefficients=rng.normal(size=(15,3))
        old_basis=np.concatenate([sh_basis(directions,d) for d in (1,2,3)],axis=1)
        new_basis=np.concatenate([sh_basis(directions@rotation.T,d) for d in (1,2,3)],axis=1)
        np.testing.assert_allclose(new_basis@sh_rotation(rotation)@coefficients,old_basis@coefficients,atol=1e-12)

    def test_date_refinement_recovers_metre_scale_transform(self):
        rng=np.random.default_rng(10)
        source=rng.normal(size=(5000,3))
        source/=np.linalg.norm(source,axis=1,keepdims=True)
        source*=np.array([2.,1.,.5])
        expected=np.eye(4)
        expected[:3,:3]=25.6*Rotation.from_euler('xyz',[.4,-.2,1.1]).as_matrix()
        expected[:3,3]=[10,35,90]
        perturb=np.eye(4);perturb[:3,:3]=.98*Rotation.from_euler('z',.015).as_matrix()
        perturb[:3,3]=[.5,-.6,.8]
        actual=refine(source,apply(source,expected),perturb@expected)
        np.testing.assert_allclose(actual,expected,atol=1e-3)

    def test_gaussian_covariance_and_roundtrip(self):
        names=["x","y","z"]+[f"rot_{i}" for i in range(4)]+[f"scale_{i}" for i in range(3)]+["opacity"]+[f"f_dc_{i}" for i in range(3)]+[f"f_rest_{i}" for i in range(45)]
        data=np.zeros(10,dtype=[(k,"f4") for k in names]);rng=np.random.default_rng(4)
        for k in names:data[k]=rng.normal(size=10)
        quaternion=Rotation.random(10,random_state=rng).as_quat(scalar_first=True)
        for i in range(4):data[f"rot_{i}"]=quaternion[:,i]
        matrix=np.eye(4);R=Rotation.from_euler("xyz",[.4,.9,-.1]).as_matrix()
        matrix[:3,:3]=2.5*R;matrix[:3,3]=[4,5,6]
        out=transform_vertices(data,matrix)
        def covariance(vertex):
            Q=Rotation.from_quat([vertex[f"rot_{i}"] for i in range(4)],scalar_first=True).as_matrix()
            return Q@np.diag(np.exp(2*np.array([vertex[f"scale_{i}"] for i in range(3)])))@Q.T
        np.testing.assert_allclose(covariance(out[0]),2.5**2*R@covariance(data[0])@R.T,rtol=2e-6)
        inverse=transform_vertices(out,np.linalg.inv(matrix))
        for k in ["x","y","z","opacity"]+[f"scale_{i}" for i in range(3)]+[f"f_rest_{i}" for i in range(45)]:
            np.testing.assert_allclose(inverse[k],data[k],atol=2e-6)


if __name__=="__main__":unittest.main()
