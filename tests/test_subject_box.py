import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.clip_box import fit_subject_box
from src.gltf_gsplat import write_gsplat_glb, read_gsplat_glb

class SubjectBoxTests(unittest.TestCase):
    def test_centers_main_subject_not_background_or_small_dense_cluster(self):
        rng=np.random.default_rng(12)
        subject=rng.normal([8,3,-2],[.3,.15,.1],(1500,3))
        background=rng.uniform(-30,30,(600,3))
        distraction=rng.normal([-10,0,0],.01,(50,3))
        points=np.vstack([subject,background,distraction])
        original=points.copy()
        bounds,axes=fit_subject_box(points)
        np.testing.assert_allclose(bounds.mean(0)@axes.T,[8,3,-2],atol=.06)
        np.testing.assert_allclose(bounds[1]-bounds[0],np.repeat(bounds[1,0]-bounds[0,0],3),rtol=1e-5)
        self.assertLess((bounds[1]-bounds[0]).max(),2)
        np.testing.assert_array_equal(points,original)

    def test_final_box_metadata_is_preserved(self):
        bounds,axes=fit_subject_box(np.random.default_rng(3).normal(size=(100,3)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.glb'
            write_gsplat_glb(path,np.zeros((1,3)),np.ones((1,3)),np.array([[1,0,0,0]]),np.ones(1),np.ones((1,3)),clip_bounds=bounds,clip_axes=axes,clip_final=True)
            data=read_gsplat_glb(path)
            self.assertTrue(data['clip_final'])
            np.testing.assert_array_equal(data['clip_bounds'],bounds)
            np.testing.assert_array_equal(data['clip_axes'],axes)

    def test_duplicates_do_not_move_subject(self):
        points=np.random.default_rng(5).normal(size=(100,3))
        bounds,axes=fit_subject_box(points)
        repeated=np.vstack([points,np.repeat(points[:1],1000,axis=0)])
        b,a=fit_subject_box(repeated)
        np.testing.assert_allclose(bounds,b)
        np.testing.assert_allclose(axes,a)
