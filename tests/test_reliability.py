import unittest

import torch

from physics_modular_3dgs.reliability import (
    physics_aware_densification_weights,
    pixel_reliability,
)


class PixelReliabilityTest(unittest.TestCase):
    def test_low_light_and_transmission_keep_a_floor(self):
        effective_attenuation = torch.zeros(1, 1, 1, 1)
        transmission = torch.zeros(1, 1, 1, 1)
        particle_mask = torch.zeros(1, 1, 1, 1)

        trust = pixel_reliability(effective_attenuation, transmission, particle_mask)

        self.assertAlmostEqual(float(trust.item()), 0.09, places=6)

    def test_particle_mask_can_suppress_trust(self):
        effective_attenuation = torch.ones(1, 1, 1, 1)
        transmission = torch.ones(1, 1, 1, 1)
        particle_mask = torch.ones(1, 1, 1, 1)

        trust = pixel_reliability(effective_attenuation, transmission, particle_mask)

        self.assertEqual(float(trust.item()), 0.0)


class DensificationWeightTest(unittest.TestCase):
    def test_geometry_need_is_mean_normalized_without_physics(self):
        weights = physics_aware_densification_weights(
            torch.tensor([1.0, 2.0]), None
        )

        torch.testing.assert_close(weights, torch.tensor([2.0 / 3.0, 4.0 / 3.0]))

    def test_physics_trust_redistributes_but_does_not_zero_weights(self):
        weights = physics_aware_densification_weights(
            torch.ones(2),
            torch.tensor([1.0, 0.0]),
            reliability_floor=0.3,
            min_weight=0.5,
            max_weight=3.0,
        )

        self.assertGreater(float(weights[0]), 1.0)
        self.assertEqual(float(weights[1]), 0.5)
        self.assertFalse(weights.requires_grad)

    def test_invalid_values_are_finite_and_clamped(self):
        weights = physics_aware_densification_weights(
            torch.tensor([float("nan"), float("inf"), -1.0]),
            torch.ones(3),
        )

        self.assertTrue(bool(torch.isfinite(weights).all()))
        self.assertTrue(bool((weights >= 0.5).all()))
        self.assertTrue(bool((weights <= 3.0).all()))

    def test_hard_gate_removes_unreliable_gaussians(self):
        weights = physics_aware_densification_weights(
            torch.ones(3), torch.tensor([0.2, 0.5, 0.9]),
            reliability_floor=0.0, reliability_threshold=0.35,
            reliability_gamma=2.0,
        )
        self.assertEqual(float(weights[0]), 0.0)
        self.assertGreater(float(weights[1]), 0.0)
        self.assertGreater(float(weights[2]), float(weights[1]))


if __name__ == "__main__":
    unittest.main()
