import unittest

import torch

from physics_modular_3dgs.particle_supervision import (
    ParticlePseudoLabelCache,
    make_particle_pseudo_label,
)
from physics_modular_3dgs.training_schedule import physics_prediction_weight, particle_strength, training_stage, trust_weights


class ParticlePseudoLabelTest(unittest.TestCase):
    def test_positive_residual_is_detached_and_nonnegative(self):
        observed = torch.tensor([[[[0.8]], [[0.2]], [[0.5]]]], requires_grad=True)
        continuous = torch.tensor([[[[0.3]], [[0.4]], [[0.5]]]], requires_grad=True)

        label = make_particle_pseudo_label(observed, continuous)

        self.assertFalse(label.mask.requires_grad)
        self.assertFalse(label.residual.requires_grad)
        self.assertTrue(bool((label.residual >= 0.0).all()))
        self.assertEqual(float(label.residual[0, 1, 0, 0]), 0.0)

    def test_cache_stays_fixed_until_refresh_interval(self):
        cache = ParticlePseudoLabelCache(refresh_interval=100)
        observed = torch.ones(1, 3, 2, 2)
        first = cache.get(3, 10, observed, torch.zeros_like(observed))
        cached = cache.get(3, 50, observed, torch.ones_like(observed))
        refreshed = cache.get(3, 110, observed, torch.ones_like(observed))

        torch.testing.assert_close(first.residual, cached.residual)
        self.assertGreater(float(cached.residual.mean()), 0.0)
        self.assertEqual(float(refreshed.residual.max()), 0.0)
        self.assertEqual(len(cache), 1)

    def test_cache_round_trip_preserves_named_view_buffers(self):
        cache = ParticlePseudoLabelCache(refresh_interval=100)
        observed = torch.ones(1, 3, 2, 2)
        expected = cache.get("frame_001", 25, observed, torch.zeros_like(observed))
        restored = ParticlePseudoLabelCache(refresh_interval=100)

        restored.load_state_dict(cache.state_dict())
        actual = restored.cached("frame_001", torch.device("cpu"), torch.float32)

        self.assertIsNotNone(actual)
        torch.testing.assert_close(actual.residual, expected.residual)
        self.assertEqual(actual.iteration, 25)


class TrainingScheduleTest(unittest.TestCase):
    def test_three_stage_boundaries_and_particle_ramp(self):
        self.assertEqual(training_stage(4000, 4000, 2000), "geometry_medium_init")
        self.assertEqual(training_stage(4001, 4000, 2000), "particle_init")
        self.assertEqual(training_stage(6000, 4000, 2000), "particle_init")
        self.assertEqual(training_stage(6001, 4000, 2000), "joint")
        self.assertEqual(particle_strength(4000, 4000, 2000, "geometry_medium_init"), 0.0)
        self.assertGreater(particle_strength(4001, 4000, 2000, "particle_init"), 0.0)
        self.assertLess(particle_strength(4001, 4000, 2000, "particle_init"), 0.01)
        self.assertEqual(particle_strength(6000, 4000, 2000, "particle_init"), 1.0)
        self.assertEqual(particle_strength(6001, 4000, 2000, "joint"), 1.0)
        self.assertEqual(physics_prediction_weight(0, 10000), 0.0)
        self.assertEqual(physics_prediction_weight(5000, 10000), 0.5)
        self.assertEqual(physics_prediction_weight(10000, 10000), 1.0)

    def test_physics_trust_is_introduced_in_stages(self):
        self.assertEqual(trust_weights(4000, 4000, 2000), (0.0, 0.0, 0.0))
        self.assertEqual(trust_weights(5000, 4000, 2000), (0.0, 0.0, 1.0))
        eta_l, eta_t, eta_m = trust_weights(6500, 4000, 2000)
        self.assertEqual(eta_l, 0.0)
        self.assertGreater(eta_t, 0.0)
        self.assertEqual(eta_m, 1.0)
        self.assertEqual(trust_weights(11000, 4000, 2000), (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
