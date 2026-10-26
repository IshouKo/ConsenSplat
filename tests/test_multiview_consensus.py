import math
import unittest

import torch

from physics_modular_3dgs.multiview_consensus import (
    invert_degradation,
    reproject_consensus,
)
from utils.graphics_utils import getProjectionMatrix


class _Camera:
    FoVx = math.pi / 2
    FoVy = math.pi / 2
    world_view_transform = torch.eye(4)
    full_proj_transform = getProjectionMatrix(0.01, 100.0, FoVx, FoVy).transpose(0, 1)


class MultiViewConsensusTest(unittest.TestCase):
    def test_hard_physics_inverse(self):
        clean = torch.full((1, 3, 4, 4), 0.4)
        e = torch.full((1, 1, 4, 4), 0.5)
        t = torch.full((1, 1, 4, 4), 0.8)
        p = torch.full((1, 3, 4, 4), 0.1)
        observed = clean * e * t + p
        torch.testing.assert_close(invert_degradation(observed, e, t, p), clean)

    def test_identity_reprojection_preserves_source(self):
        camera = _Camera()
        image = torch.linspace(0, 1, 16).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)
        depth = torch.ones(1, 1, 4, 4)
        consensus, confidence = reproject_consensus(
            camera, depth, [(camera, image, depth), (camera, image, depth)], agreement_scale=0.1
        )
        torch.testing.assert_close(consensus, image, atol=1e-5, rtol=1e-5)
        self.assertGreater(float(confidence.mean()), 0.99)


    def test_depth_inconsistent_sources_are_rejected(self):
        camera = _Camera()
        image = torch.ones(1, 3, 4, 4)
        target_depth = torch.ones(1, 1, 4, 4)
        wrong_source_depth = torch.full_like(target_depth, 3.0)
        _, confidence = reproject_consensus(
            camera, target_depth,
            [(camera, image, wrong_source_depth), (camera, image, wrong_source_depth)],
            min_views=2,
        )
        self.assertEqual(float(confidence.max()), 0.0)


if __name__ == "__main__":
    unittest.main()
