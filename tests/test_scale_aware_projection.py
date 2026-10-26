import unittest

import torch

from physics_modular_3dgs.integration import project_reliability_to_gaussians


class _IdentityCamera:
    full_proj_transform = torch.eye(4)


class ScaleAwareProjectionTest(unittest.TestCase):
    def test_projected_radius_selects_a_wider_average(self):
        trust = torch.ones(1, 1, 17, 17)
        trust[:, :, 8, 8] = 0.0
        xyz = torch.tensor([[0.0, 0.0, 0.0]]).repeat(3, 1)
        radii = torch.tensor([1.0, 3.0, 8.0])

        sampled = project_reliability_to_gaussians(
            _IdentityCamera(), xyz, trust, radii
        )

        self.assertLess(float(sampled[0]), float(sampled[1]))
        self.assertLess(float(sampled[1]), float(sampled[2]))
        self.assertFalse(sampled.requires_grad)


if __name__ == "__main__":
    unittest.main()
