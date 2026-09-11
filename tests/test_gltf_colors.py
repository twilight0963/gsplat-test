import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.gltf_gsplat import read_gsplat_glb, write_gsplat_glb


class GlbColorTests(unittest.TestCase):
    def roundtrip(self, color_space):
        colors = np.array([
            [0., 0.5, 1.], [0.01, 0.04045, 0.04046],
            [0.2, 0.7, 0.9], [0.003, 0.05, 0.8],
        ], dtype=np.float32)
        means = np.arange(12, dtype=np.float32).reshape(4, 3)
        scales = np.linspace(0.01, 1., 12, dtype=np.float32).reshape(4, 3)
        quats = np.eye(4, dtype=np.float32)
        opacities = np.array([0., 0.01, 0.5, 1.], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.glb'
            write_gsplat_glb(path, means, scales, quats, opacities, colors, color_space)
            loaded = read_gsplat_glb(path)
        np.testing.assert_allclose(loaded['colors'], colors, atol=1e-6, rtol=1e-5)
        for name, expected in [('means', means), ('scales', scales),
                               ('quats', quats), ('opacities', opacities)]:
            np.testing.assert_array_equal(loaded[name], expected)
        return colors, loaded['colors'].numpy(), opacities

    def test_srgb_roundtrip_and_alpha_compositing(self):
        colors, loaded, opacity = self.roundtrip('srgb_rec709_display')
        # Conversion must happen per splat, before blending, not on the image.
        weights = opacity * np.cumprod(np.r_[1., 1. - opacity[:-1]])
        np.testing.assert_allclose(weights @ loaded, weights @ colors, atol=1e-6)

    def test_linear_roundtrip_does_not_apply_srgb_conversion(self):
        self.roundtrip('lin_rec709_display')


if __name__ == '__main__':
    unittest.main()
